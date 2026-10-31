"""
お手本（先生役の押印）を記録して、LeRobot のデータセットにする（G-Tune 用）

Colab のノートブックと同じ動き・同じ記録方法です。
  1. 紙の位置を毎回変えて、先生役（逆運動学）に押させる
  2. 1回ごとに、映像（mp4）と関節の角度（npz）を保存する
     → 途中で止めても、もう一度実行すれば続きから記録する
  3. 全部そろったら、LeRobot のデータセットにまとめる

使い方（Anaconda Prompt で、conda activate lerobot のあと、C:\\so101 で）:
    python record_demos.py --episodes 200                         # 白い紙・全体カメラだけ
    python record_demos.py --episodes 200 --wrist --paper form \\
        --raw C:/so101/raw_form_wrist --dataset C:/so101/so101_stamp_form_wrist   # 伝票風の紙＋手首カメラ
"""
import argparse
import shutil
import time
from pathlib import Path

import stamp_sim as ss  # 描画の設定があるので、mujoco より先に読み込む

import numpy as np
import mujoco
import mediapy as media

if shutil.which("ffmpeg") is None:  # ffmpeg が見つからないときは、pip で入る版を使う
    try:
        import imageio_ffmpeg
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="お手本を記録して LeRobot のデータセットにする")
    p.add_argument("--episodes", type=int, default=200, help="お手本の回数")
    p.add_argument("--workdir", default="C:/so101", help="作業フォルダ")
    p.add_argument("--raw", default="C:/so101/raw_episodes", help="1回ずつのお手本の保存先")
    p.add_argument("--dataset", default="C:/so101/so101_stamp_sim200", help="作るデータセットの場所")
    p.add_argument("--repo-id", default="local/so101_stamp_sim200")
    p.add_argument("--seed", type=int, default=0, help="紙の位置を決める乱数の種（評価は 1000 を使う）")
    p.add_argument("--paper", choices=["simple", "form", "fixed"], default="simple",
                   help="simple: 白い紙に赤い丸 / form: 伝票風の紙（毎回ちがう模様） / fixed: いつも同じ1種類の伝票")
    p.add_argument("--wrist", action="store_true", help="手首カメラの映像も記録する")
    return p.parse_args()


# ==================================================
# 先生役の動き（Colab ノートブックの「6.」と同じ）
# ==================================================
def smoothstep(x):
    return 3 * x**2 - 2 * x**3


def vertical_path(sim, aim, z_top, z_bottom, q_start_full):
    """枠の真上 z_top から z_bottom まで、真下へ一直線に下りる関節角度の列"""
    n = max(2, int(abs(z_top - z_bottom) * 1000 / ss.DESCENT_STEP_MM) + 1)
    q_full = q_start_full.copy()
    path, worst = [], 0.0
    for z in np.linspace(z_top, z_bottom, n):
        q_arm, err, _ = sim.solve_ik(np.array([aim[0], aim[1], z]), q_full, iters=100)
        q_full[sim.qadr] = q_arm
        path.append(q_arm)
        worst = max(worst, err)
    return np.array(path), worst


def pose_on_path(path, s):
    if len(path) == 1:
        return path[0]
    x = smoothstep(float(np.clip(s, 0, 1))) * (len(path) - 1)
    i = min(int(x), len(path) - 2)
    return path[i] + (path[i + 1] - path[i]) * (x - i)


def plan_episode(sim, aim):
    """動きの予定表：待つ → 真上へ → まっすぐ押す → 押したまま → 持ち上げる → 戻る → 待つ"""
    sim.reset_pose()
    q_home = sim.data.qpos[sim.qadr].copy()
    q_above, e1, _ = sim.solve_ik(aim + [0, 0, ss.APPROACH_HEIGHT], sim.data.qpos)
    start = sim.data.qpos.copy()
    start[sim.qadr] = q_above
    down, e2 = vertical_path(sim, aim, ss.PAPER_Z + ss.APPROACH_HEIGHT,
                             ss.PAPER_Z - ss.PRESS_DEPTH, start)
    if max(e1, e2) > 1e-3:
        return None
    q_press = down[-1]
    sim.reset_pose()
    return [
        (np.array([q_home]), 0.5, "wait"),
        (np.array([q_home, q_above]), 1.2, "move"),
        (down, 0.8, "press"),
        (np.array([q_press]), 0.4, "press"),
        (down[::-1], 0.6, "lift"),
        (np.array([q_above, q_home]), 1.0, "move"),
        (np.array([q_home]), 0.3, "wait"),
    ]


def command_at(plan, t):
    start = 0.0
    for path, duration, phase in plan:
        if t < start + duration:
            return pose_on_path(path, (t - start) / duration), phase
        start += duration
    path, _, phase = plan[-1]
    return path[-1], phase


def record_episode(sim, renderer, frame_center):
    """お手本を1回分実行し、画像（カメラごと）・状態・行動を 1/30 秒ごとに集める"""
    plan = plan_episode(sim, frame_center)
    if plan is None:
        return None
    n_frames = int(round(sum(d for _, d, _ in plan) * ss.FPS))
    images = {cam: [] for cam in sim.camera_names}
    states, actions = [], []
    touch_xy, lowest_xy, min_z = None, None, np.inf
    for f in range(n_frames):
        for cam in sim.camera_names:
            images[cam].append(sim.render(renderer, touch_xy, camera=cam))
        states.append(sim.motor_angles_deg())
        q_next, phase = command_at(plan, (f + 1) / ss.FPS)
        sim.set_arm_ctrl(q_next)
        actions.append(np.degrees(sim.data.ctrl[:sim.model.nu]).astype(np.float32))
        for _ in range(sim.steps_per_frame):
            mujoco.mj_step(sim.model, sim.data)
            p, _ = sim.tip_pose(sim.data)
            if touch_xy is None and p[2] <= ss.PAPER_Z + 0.001:
                touch_xy = p[:2].copy()
            if phase == "press" and p[2] < min_z:
                min_z, lowest_xy = p[2], p[:2].copy()
    if touch_xy is None or min_z > ss.PAPER_Z + 0.002:
        return None
    judged = ss.judge_stamp(lowest_xy * 1000, frame_center[:2] * 1000)
    if not judged["success"]:
        return None
    return {"images": images, "state": np.array(states), "action": np.array(actions),
            "offset_mm": judged["offset_mm"]}


# ==================================================
# 記録とデータセット作り
# ==================================================
def saved_episodes(raw_dir):
    return sorted(p for p in raw_dir.glob("ep_*.npz") if p.with_suffix(".mp4").exists())


def record_all(args):
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)
    scene = ss.find_or_download_scene(args.workdir)
    sim = ss.StampSim(scene, paper=args.paper, wrist=args.wrist)
    print(f"紙: {args.paper} / カメラ: {', '.join(sim.camera_names)}")

    done = len(saved_episodes(raw))
    print(f"保存済み: {done} 回 → {done + 1} 回目から記録します（目標 {args.episodes} 回）")
    t0 = time.time()
    with mujoco.Renderer(sim.model, height=ss.IMG_H, width=ss.IMG_W) as renderer:
        i = done
        while i < args.episodes:
            for attempt in range(20):
                # 回の番号ごとに決まった乱数にする（途中から再開しても同じ紙の位置になる）
                sim.rng = np.random.default_rng([args.seed, i, attempt])
                center = sim.new_scene()
                ep = record_episode(sim, renderer, center)
                if ep is not None:
                    break
            else:
                raise RuntimeError("お手本を作れませんでした")
            name = raw / f"ep_{i:03d}"
            for cam in sim.camera_names:
                suffix = "" if cam == "front" else f"_{cam}"
                media.write_video(raw / f"ep_{i:03d}{suffix}.mp4", ep["images"][cam], fps=ss.FPS)
            np.savez(name.with_suffix(".npz"), state=ep["state"], action=ep["action"],
                     frame_center_mm=center * 1000, offset_mm=ep["offset_mm"])  # npz を最後に＝保存完了の印
            i += 1
            print(f"お手本 {i}/{args.episodes} を保存（ずれ {ep['offset_mm']:.2f} mm、経過 {time.time() - t0:.0f} 秒）")
    return sim.motor_names, sim.camera_names


def build_dataset(args, motor_names, camera_names):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    files = saved_episodes(Path(args.raw))[:args.episodes]
    root = Path(args.dataset)
    if root.exists():
        shutil.rmtree(root)  # 前回の作りかけが残っていたら消す
    n = len(motor_names)
    features = {
        "observation.state": {"dtype": "float32", "shape": (n,), "names": motor_names},
        "action": {"dtype": "float32", "shape": (n,), "names": motor_names},
    }
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {"dtype": "video", "shape": (ss.IMG_H, ss.IMG_W, 3),
                                                 "names": ["height", "width", "channels"]}
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=ss.FPS, features=features, root=root,
                               robot_type="so101_follower", use_videos=True)
    for k, f in enumerate(files):
        d = np.load(f)
        videos = {cam: media.read_video(f.with_name(f.stem + ("" if cam == "front" else f"_{cam}") + ".mp4"))
                  for cam in camera_names}
        n = min([len(v) for v in videos.values()] + [len(d["state"])])
        for j in range(n):
            frame = {"observation.state": d["state"][j].astype(np.float32),
                     "action": d["action"][j].astype(np.float32),
                     "task": ss.TASK}
            for cam, v in videos.items():
                frame[f"observation.images.{cam}"] = np.asarray(v[j], dtype=np.uint8)
            ds.add_frame(frame)
        try:
            ds.save_episode()
        except TypeError:
            ds.save_episode(task=ss.TASK)
        print(f"データセットに追加 {k + 1}/{len(files)}")
    ds.finalize()
    print(f"\nデータセットを作りました: {root}（お手本 {len(files)} 回）")


def main():
    args = parse_args()
    motor_names, camera_names = record_all(args)
    build_dataset(args, motor_names, camera_names)


if __name__ == "__main__":
    main()
