"""
SO-ARM101 押印シミュレーション（MuJoCo）の共通部品

お手本の記録（Colab ノートブック）と、学習したAIの評価（eval_act_sim.py）の
両方で使う処理をまとめたファイルです。

  - SO-101 のモデルにハンコを持たせる
  - 逆運動学（印面を目標の位置へ、ハンコを真下へ）
  - 紙・丸枠・印影の描き足し
  - 押した位置の成功判定
"""
import io
import os
import sys
import zipfile
import urllib.request
from pathlib import Path

import numpy as np

# Linux（Colab など）で画面がないときは、GPU か CPU で描く。Windows は何もしなくてよい
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402

import paper_gen  # noqa: E402

# ==================================================
# 設定（お手本を記録したときと同じ値）
# ==================================================
STAMP_BODY_DIAMETER = 0.0105    # ハンコの胴体の直径 [m]
STAMP_LENGTH = 0.062            # ハンコの全長 [m]
STAMP_PROTRUDE = 0.025          # 指先の目印から印面までの距離 [m]

STAMP_DIAMETER_MM = 10.0        # 印影の直径
FRAME_INNER_DIAMETER_MM = 20.0  # 丸枠の内径
MAX_OFFSET_MM = (FRAME_INNER_DIAMETER_MM - STAMP_DIAMETER_MM) / 2  # 許容ずれ 5mm

PAPER_Z = 0.0                   # 紙の表面の高さ [m]
APPROACH_HEIGHT = 0.04          # 枠の真上で一度止まる高さ [m]
PRESS_DEPTH = 0.001             # 紙の面より少し下を狙って押し付ける [m]
DESCENT_STEP_MM = 1.0           # まっすぐ下りる経路を何mmごとに計算するか
REACH_RANGE = (0.16, 0.26)      # 紙を置く距離の範囲 [m]
ANGLE_RANGE_DEG = 25            # 紙を置く向きの範囲（正面から左右）[度]

FPS = 30
IMG_H, IMG_W = 480, 640
TASK = "Press the stamp inside the circle on the paper."

CAM_LOOKAT = [0.25, 0.0, 0.0]
CAM_DISTANCE = 0.45
CAM_AZIMUTH = 180
CAM_ELEVATION = -55

WRIST_CAM_BACK = 0.035          # 手首カメラ：印面から手首側へ下がる距離 [m]
WRIST_CAM_SIDE = 0.040          # 手首カメラ：ハンコの軸から横へ離す距離 [m]
WRIST_CAM_FOVY = 75             # 手首カメラの画角 [度]
PAPER_YAW_DEG = 8               # 伝票風の紙を置くときの傾き（±）[度]

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

SO_ARM_ZIP_URL = "https://github.com/TheRobotStudio/SO-ARM100/archive/refs/heads/main.zip"


def find_or_download_scene(base_dir):
    """SO-101 のモデル（scene.xml）を探し、無ければ GitHub からダウンロードする"""
    base_dir = Path(base_dir)
    candidates = [
        base_dir / "SO-ARM100" / "Simulation" / "SO101" / "scene.xml",
        base_dir / "SO-ARM100-main" / "Simulation" / "SO101" / "scene.xml",
        Path("/content/SO-ARM100/Simulation/SO101/scene.xml"),
    ]
    for c in candidates:
        if c.exists():
            return c
    print("SO-101 のモデルが見つからないので、GitHub からダウンロードします（数分かかります）")
    with urllib.request.urlopen(SO_ARM_ZIP_URL) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if "/Simulation/SO101/" in name:
                z.extract(name, base_dir)
    scene = base_dir / "SO-ARM100-main" / "Simulation" / "SO101" / "scene.xml"
    if not scene.exists():
        raise FileNotFoundError("scene.xml が見つかりません。SO-ARM100 を手動でダウンロードしてください")
    return scene


class StampSim:
    """ハンコを持った SO-101 のシミュレーション"""

    def __init__(self, xml_path, seed=0, paper="simple", wrist=False):
        """
        paper : "simple"（白い紙に赤い丸）、"form"（伝票風の紙・毎回ちがう模様）、
                "fixed"（伝票風の紙・いつも同じ1種類。置く位置と向きだけ変わる）
        wrist : True なら手首カメラも付ける
        """
        assert paper in ("simple", "form", "fixed")
        self.rng = np.random.default_rng(seed)
        self.paper_mode = paper
        self.wrist = wrist
        self.frame_center = np.zeros(3)
        self.paper_pos = np.zeros(3)
        self.paper_quat = np.array([1.0, 0, 0, 0])
        self._paper_version = 0
        self._uploaded = {}
        self._build_model(str(xml_path))
        self._setup()

    @property
    def camera_names(self):
        return ["front", "wrist"] if self.wrist else ["front"]

    # ---------- モデルを作る ----------
    def _build_model(self, xml_path):
        base_model = mujoco.MjModel.from_xml_path(xml_path)
        base_data = mujoco.MjData(base_model)
        if base_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(base_model, base_data, 0)
        mujoco.mj_forward(base_model, base_data)

        frame_id = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        holder_id = int(base_model.site_bodyid[frame_id])
        holder_name = base_model.body(holder_id).name
        parent_id = int(base_model.body_parentid[holder_id])

        # ハンコの軸：手首から指先へ向かう向き（グリッパーのボディ座標）
        v_world = base_data.site_xpos[frame_id] - base_data.xpos[parent_id]
        R_holder = base_data.xmat[holder_id].reshape(3, 3)
        axis_body = R_holder.T @ v_world
        axis_body /= np.linalg.norm(axis_body)

        frame_pos = base_model.site_pos[frame_id].copy()
        face_pos = frame_pos + axis_body * STAMP_PROTRUDE
        center_pos = face_pos - axis_body * STAMP_LENGTH / 2
        stamp_quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(stamp_quat, axis_body)

        spec = mujoco.MjSpec.from_file(xml_path)
        holder = spec.body(holder_name) if hasattr(spec, "body") else spec.find_body(holder_name)

        stamp = holder.add_geom()
        stamp.name = "stamp"
        stamp.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        stamp.size = [STAMP_BODY_DIAMETER / 2, STAMP_LENGTH / 2, 0]
        stamp.pos = center_pos
        stamp.quat = stamp_quat
        stamp.mass = 0.01
        stamp.rgba = [0.1, 0.1, 0.1, 1]

        face = holder.add_site()
        face.name = "stamp_face"
        face.pos = face_pos
        face.quat = stamp_quat
        face.size = [0.002, 0, 0]

        if self.wrist:
            # 手首カメラ：ハンコの斜め後ろから、印面の少し先を見る
            up_world = np.array([0.0, 0.0, 1.0])
            up_body = R_holder.T @ up_world
            side = up_body - axis_body * (up_body @ axis_body)
            side /= np.linalg.norm(side)
            cam_pos = face_pos - axis_body * WRIST_CAM_BACK + side * WRIST_CAM_SIDE
            target = face_pos + axis_body * 0.02
            zc = cam_pos - target
            zc /= np.linalg.norm(zc)            # カメラは -z の向きを見る
            xc = np.cross(side, zc)
            xc /= np.linalg.norm(xc)
            yc = np.cross(zc, xc)
            cam_quat = np.zeros(4)
            mujoco.mju_mat2Quat(cam_quat, np.column_stack([xc, yc, zc]).flatten())
            cam = holder.add_camera()
            cam.name = "wrist_cam"
            cam.pos = cam_pos
            cam.quat = cam_quat
            cam.fovy = WRIST_CAM_FOVY

        if self.paper_mode in ("form", "fixed"):
            # 伝票風の紙：見た目だけの板（ぶつからない）。画像は毎回入れ替える
            W = int(round(paper_gen.PAPER_W_MM * paper_gen.PX_PER_MM))
            H = int(round(paper_gen.PAPER_H_MM * paper_gen.PX_PER_MM))
            tex = spec.add_texture()
            tex.name = "paper_tex"
            tex.type = mujoco.mjtTexture.mjTEXTURE_2D
            tex.width, tex.height, tex.nchannel = W, H, 3
            tex.data = bytes(np.full(W * H * 3, 255, np.uint8))
            mat = spec.add_material()
            mat.name = "paper_mat"
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "paper_tex"
            mat.texuniform = False
            mat.texrepeat = [1, 1]
            mat.specular = 0.0       # てかりを消す（白く飛ばないように）
            mat.shininess = 0.0
            mat.reflectance = 0.0
            mat.rgba = [0.8, 0.8, 0.8, 1]
            body = spec.worldbody.add_body()
            body.name = "paper"
            body.mocap = True
            g = body.add_geom()
            g.name = "paper_geom"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [paper_gen.PAPER_W_MM / 2000, paper_gen.PAPER_H_MM / 2000, 0.0001]
            g.pos = [0, 0, 0.0001]
            g.material = "paper_mat"
            g.contype = 0
            g.conaffinity = 0

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)

    # ---------- 関節・モーター・カメラの準備 ----------
    def _setup(self):
        m = self.model
        self.tip_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "stamp_face")
        self.tip_body_id = int(m.site_bodyid[self.tip_site_id])
        self.tool_axis = np.array([0.0, 0.0, 1.0])  # 印面の目印の z 軸 = ハンコの軸
        self.down = np.array([0.0, 0.0, -1.0])

        self.arm = []
        for name in ARM_JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            acts = [i for i in range(m.nu) if int(m.actuator_trnid[i, 0]) == jid]
            self.arm.append({"name": name, "jid": jid,
                             "qadr": int(m.jnt_qposadr[jid]),
                             "dadr": int(m.jnt_dofadr[jid]),
                             "act": acts[0] if acts else None})
        self.qadr = np.array([j["qadr"] for j in self.arm])
        self.dadr = np.array([j["dadr"] for j in self.arm])

        # 記録・AI に使う全モーター（腕5つ + グリッパー）
        self.motor_names, motor_qadr = [], []
        for i in range(m.nu):
            jid = int(m.actuator_trnid[i, 0])
            self.motor_names.append(f"{m.joint(jid).name}.pos")
            motor_qadr.append(int(m.jnt_qposadr[jid]))
        self.motor_qadr = np.array(motor_qadr)

        self.reset_pose()
        tip_home, _ = self.tip_pose(self.data)
        base_id = int(m.jnt_bodyid[self.arm[0]["jid"]])
        self.base_xy = self.data.xpos[base_id][:2].copy()
        d = tip_home[:2] - self.base_xy
        self.forward = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 0.02 else 0.0

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = CAM_LOOKAT
        self.cam.distance = CAM_DISTANCE
        self.cam.azimuth = CAM_AZIMUTH
        self.cam.elevation = CAM_ELEVATION

        self.steps_per_frame = max(1, int(round(1 / (FPS * m.opt.timestep))))

    # ---------- 基本の道具 ----------
    def reset_pose(self):
        """初期姿勢に戻し、全モーターの目標を今の角度に合わせる"""
        m, d = self.model, self.data
        if m.nkey > 0:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        else:
            mujoco.mj_resetData(m, d)
        if self.paper_mode in ("form", "fixed"):
            d.mocap_pos[0] = self.paper_pos
            d.mocap_quat[0] = self.paper_quat
        mujoco.mj_forward(m, d)
        for i in range(m.nu):
            jid = int(m.actuator_trnid[i, 0])
            d.ctrl[i] = d.qpos[m.jnt_qposadr[jid]]

    def tip_pose(self, data):
        """印面の位置と向き"""
        return (data.site_xpos[self.tip_site_id].copy(),
                data.site_xmat[self.tip_site_id].reshape(3, 3).copy())

    def motor_angles_deg(self):
        """今の全モーターの角度 [度]（AI に渡す「状態」）"""
        return np.degrees(self.data.qpos[self.motor_qadr]).astype(np.float32)

    def set_all_ctrl_deg(self, target_deg):
        """全モーターに目標角度 [度] を送る（AI の「行動」をそのまま使う）"""
        m = self.model
        target = np.radians(np.asarray(target_deg, dtype=float))
        for i in range(m.nu):
            c = target[i]
            if m.actuator_ctrllimited[i]:
                c = np.clip(c, *m.actuator_ctrlrange[i])
            self.data.ctrl[i] = c

    def set_arm_ctrl(self, q_cmd):
        """腕のモーターに目標角度 [rad] を送る"""
        m = self.model
        for i, j in enumerate(self.arm):
            if j["act"] is None:
                continue
            c = q_cmd[i]
            if m.actuator_ctrllimited[j["act"]]:
                c = np.clip(c, *m.actuator_ctrlrange[j["act"]])
            self.data.ctrl[j["act"]] = c

    # ---------- 逆運動学 ----------
    @staticmethod
    def _skew(a):
        return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    def solve_ik(self, target, q_start, iters=300, axis_weight=0.05, damping=1e-4):
        m, ik = self.model, self.ik_data
        ik.qpos[:] = q_start
        for _ in range(iters):
            mujoco.mj_kinematics(m, ik)
            mujoco.mj_comPos(m, ik)
            p, R = self.tip_pose(ik)
            axis = R @ self.tool_axis
            if np.linalg.norm(p - target) < 2e-4 and np.linalg.norm(axis - self.down) < 0.02:
                break
            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, ik, jacp, jacr, p, self.tip_body_id)
            J = np.vstack([jacp[:, self.dadr],
                           axis_weight * (-self._skew(axis) @ jacr[:, self.dadr])])
            r = np.concatenate([p - target, axis_weight * (axis - self.down)])
            dq = -J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), r)
            ik.qpos[self.qadr] += np.clip(dq, -0.1, 0.1)
            for j in self.arm:
                if m.jnt_limited[j["jid"]]:
                    lo, hi = m.jnt_range[j["jid"]]
                    ik.qpos[j["qadr"]] = np.clip(ik.qpos[j["qadr"]], lo, hi)
        mujoco.mj_kinematics(m, ik)
        p, R = self.tip_pose(ik)
        tilt = np.degrees(np.arccos(np.clip(np.dot(R @ self.tool_axis, self.down), -1, 1)))
        return ik.qpos[self.qadr].copy(), float(np.linalg.norm(p - target)), float(tilt)

    # ---------- 紙の位置 ----------
    def sample_frame_center(self):
        """アームが届く範囲に、紙（丸枠）の位置をランダムに決める"""
        self.reset_pose()
        q_start = self.data.qpos.copy()
        for _ in range(50):
            r = self.rng.uniform(*REACH_RANGE)
            a = self.forward + np.radians(self.rng.uniform(-ANGLE_RANGE_DEG, ANGLE_RANGE_DEG))
            c = np.array([self.base_xy[0] + r * np.cos(a),
                          self.base_xy[1] + r * np.sin(a), PAPER_Z])
            _, err, _ = self.solve_ik(c + [0, 0, -PRESS_DEPTH], q_start)
            if err < 1e-3:
                return c
        raise RuntimeError("届く紙の位置が見つかりません")

    def new_scene(self):
        """紙の置き場所（と、伝票風なら紙の模様）を新しく決める。戻り値は押印欄の中心 [m]"""
        center = self.sample_frame_center()
        self.frame_center = center
        if self.paper_mode in ("form", "fixed"):
            if self.paper_mode == "form":
                img, field_mm = paper_gen.make_form(self.rng)
            else:
                img, field_mm = paper_gen.fixed_form()
            yaw = np.radians(self.rng.uniform(-PAPER_YAW_DEG, PAPER_YAW_DEG))
            c, s_ = np.cos(yaw), np.sin(yaw)
            offset = np.array([c * field_mm[0] - s_ * field_mm[1],
                               s_ * field_mm[0] + c * field_mm[1]]) / 1000
            self.paper_pos = np.array([center[0] - offset[0], center[1] - offset[1], PAPER_Z])
            self.paper_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "paper_tex")
            adr = self.model.tex_adr[tid]
            if self.paper_mode == "form" or self._paper_version == 0:
                self.model.tex_data[adr:adr + img.size] = img.reshape(-1)
                self._paper_version += 1
        self.reset_pose()
        return center

    # ---------- 描画 ----------
    @staticmethod
    def _add_shape(scene, gtype, size, pos, rgba):
        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype,
                            np.asarray(size, float), np.asarray(pos, float),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
        scene.ngeom += 1

    def draw_overlay(self, scene, frame_center, stamp_xy):
        """画像に、白い紙・赤い丸枠・押した後の印影を描き足す"""
        x, y = frame_center[:2]
        CYL, BOX = mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_BOX
        self._add_shape(scene, BOX, [0.074, 0.0525, 0.0002], [x, y, PAPER_Z + 0.0002], [1, 1, 1, 1])
        self._add_shape(scene, CYL, [0.011, 0.0002, 0], [x, y, PAPER_Z + 0.0006], [0.85, 0.1, 0.1, 1])
        self._add_shape(scene, CYL, [0.010, 0.0002, 0], [x, y, PAPER_Z + 0.0008], [1, 1, 1, 1])
        if stamp_xy is not None:
            self._add_shape(scene, CYL, [STAMP_DIAMETER_MM / 2000, 0.0002, 0],
                            [stamp_xy[0], stamp_xy[1], PAPER_Z + 0.0010], [0.8, 0.0, 0.0, 0.8])

    def render(self, renderer, stamp_xy=None, camera="front", frame_center=None):
        """カメラの画像を1枚描く（camera は "front" か "wrist"）"""
        if frame_center is None:
            frame_center = self.frame_center
        if self.paper_mode in ("form", "fixed") and self._uploaded.get(id(renderer)) != self._paper_version:
            tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "paper_tex")
            mujoco.mjr_uploadTexture(self.model, renderer._mjr_context, tid)
            self._uploaded[id(renderer)] = self._paper_version
        renderer.update_scene(self.data, camera=self.cam if camera == "front" else "wrist_cam")
        if self.paper_mode == "simple":
            self.draw_overlay(renderer.scene, frame_center, stamp_xy)
        elif stamp_xy is not None:
            self._add_shape(renderer.scene, mujoco.mjtGeom.mjGEOM_CYLINDER,
                            [STAMP_DIAMETER_MM / 2000, 0.0002, 0],
                            [stamp_xy[0], stamp_xy[1], PAPER_Z + 0.0010], [0.8, 0.0, 0.0, 0.8])
        return renderer.render().copy()


def judge_stamp(stamp_center_xy_mm, frame_center_xy_mm):
    """印影の中心が丸枠の中心から MAX_OFFSET_MM 以内なら成功"""
    offset = float(np.linalg.norm(np.asarray(stamp_center_xy_mm) - np.asarray(frame_center_xy_mm)))
    return {"success": offset <= MAX_OFFSET_MM, "offset_mm": offset}
