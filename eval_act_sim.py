"""
学習した AI（ACT）に、シミュレーションの中でハンコを押させて評価する

  - 紙（丸枠）の位置を毎回ランダムに変える（学習のときとは別の位置）
  - AI はカメラ画像と関節の角度だけを見て、1/30 秒ごとに次の目標角度を出す
  - 押した位置が丸枠の中心から 5mm 以内なら成功
  - 成功率・ずれ・動画・散布図を出力する

使い方（Anaconda Prompt で、conda activate lerobot のあと）:
    python eval_act_sim.py
"""
import argparse
import csv
import time
from pathlib import Path

import stamp_sim as ss  # 描画の設定があるので、mujoco より先に読み込む

import numpy as np
import torch
import mujoco
import mediapy as media
import shutil
if shutil.which("ffmpeg") is None:  # ffmpeg が見つからないときは、pip で入る版を使う
    try:
        import imageio_ffmpeg
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
except ImportError as e:
    raise SystemExit("LeRobot が見つかりません。conda activate lerobot を実行してから動かしてください") from e


def parse_args():
    p = argparse.ArgumentParser(description="学習した ACT をシミュレーションで評価する")
    p.add_argument("--policy", default="C:/so101/outputs/act_stamp/checkpoints/020000/pretrained_model",
                   help="学習したモデルのフォルダ（pretrained_model）")
    p.add_argument("--workdir", default="C:/so101", help="SO-101 のモデルを置く・探す場所")
    p.add_argument("--out", default="C:/so101/eval_act", help="結果の保存先")
    p.add_argument("--trials", type=int, default=20, help="試す回数")
    p.add_argument("--video-trials", type=int, default=5, help="動画にする回数")
    p.add_argument("--seconds", type=float, default=6.0, help="1回あたりの時間 [秒]")
    p.add_argument("--seed", type=int, default=1000, help="紙の位置を決める乱数の種（学習時は 0）")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--paper", choices=["simple", "form", "fixed"], default="simple",
                   help="simple: 白い紙に赤い丸 / form: 伝票風の紙（毎回ちがう模様） / fixed: いつも同じ1種類の伝票")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="AI の予定を何コマ実行したら、もう一度カメラを見て予定を立て直すか（学習時は 100）")
    p.add_argument("--temporal-ensemble", action="store_true",
                   help="毎コマ予定を立て直し、過去の予定と平均して滑らかにする（ACT の論文の方法）")
    return p.parse_args()


def load_policy(path, device, n_action_steps=None, temporal_ensemble=False):
    """学習したモデルと、入出力の変換（正規化など）を読み込む"""
    config = PreTrainedConfig.from_pretrained(path)
    if temporal_ensemble:
        config.n_action_steps = 1
        config.temporal_ensemble_coeff = 0.01
    elif n_action_steps is not None:
        config.n_action_steps = n_action_steps
    policy = ACTPolicy.from_pretrained(path, config=config)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocess, postprocess


def label(image, lines, color):
    """動画の左上に文字を書く"""
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, 360, 26 * len(lines) + 8], fill=(0, 0, 0))
    for i, text in enumerate(lines):
        draw.text((8, 4 + 26 * i), text, fill=color if i == len(lines) - 1 else (255, 255, 255), font=font)
    return np.asarray(img)


def side_by_side(images):
    """カメラが2台なら、横に並べて1枚にする（動画用）"""
    return np.concatenate([images[c] for c in ("front", "wrist") if c in images], axis=1)


@torch.no_grad()
def run_trial(sim, policy, preprocess, postprocess, renderer, frame_center, n_frames, make_video, trial):
    """AI に1回押させる"""
    sim.reset_pose()
    policy.reset()  # 前の回の「行動の予定」を消す

    frames = []
    touch_xy, lowest_xy, min_z = None, None, np.inf
    for f in range(n_frames):
        # (1) AI が見るもの：カメラ画像と、今の関節の角度
        images = {cam: sim.render(renderer, touch_xy, camera=cam) for cam in sim.camera_names}
        state = sim.motor_angles_deg()
        obs = {"observation.state": torch.from_numpy(state)}
        for cam, img in images.items():
            obs[f"observation.images.{cam}"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # (2) AI が次の目標角度を決める
        action = postprocess(policy.select_action(preprocess(obs)))
        action = action.squeeze(0).cpu().numpy()

        # (3) モーターに送り、1/30 秒分の物理計算を進める
        sim.set_all_ctrl_deg(action)
        for _ in range(sim.steps_per_frame):
            mujoco.mj_step(sim.model, sim.data)
            p, _ = sim.tip_pose(sim.data)
            if p[2] <= ss.PAPER_Z + 0.001 and touch_xy is None:
                touch_xy = p[:2].copy()
            if p[2] <= ss.PAPER_Z + 0.002 and p[2] < min_z:
                min_z, lowest_xy = p[2], p[:2].copy()

        if make_video:
            frames.append(label(side_by_side(images), [f"AI (ACT) policy  trial {trial + 1}",
                                        f"t = {f / ss.FPS:4.1f} s"], (255, 255, 255)))

    if lowest_xy is None:
        result = {"result": "紙に触れない", "success": False, "offset_mm": np.nan,
                  "dx_mm": np.nan, "dy_mm": np.nan}
    else:
        j = ss.judge_stamp(lowest_xy * 1000, frame_center[:2] * 1000)
        result = {"result": "成功" if j["success"] else "枠外", "success": j["success"],
                  "offset_mm": j["offset_mm"],
                  "dx_mm": (lowest_xy[0] - frame_center[0]) * 1000,
                  "dy_mm": (lowest_xy[1] - frame_center[1]) * 1000}

    if make_video:
        # 最後に結果を1秒表示する
        final = side_by_side({cam: sim.render(renderer, touch_xy, camera=cam) for cam in sim.camera_names})
        if result["success"]:
            text, color = f"SUCCESS  offset {result['offset_mm']:.1f} mm", (80, 255, 80)
        elif np.isnan(result["offset_mm"]):
            text, color = "FAIL  (did not touch)", (255, 80, 80)
        else:
            text, color = f"FAIL  offset {result['offset_mm']:.1f} mm", (255, 80, 80)
        frames += [label(final, [f"AI (ACT) policy  trial {trial + 1}", text], color)] * ss.FPS
    return result, frames


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"モデル: {args.policy}")
    print(f"計算に使う装置: {args.device}")
    policy, preprocess, postprocess = load_policy(args.policy, args.device,
                                                  args.n_action_steps, args.temporal_ensemble)
    mode = ("temporal ensemble" if args.temporal_ensemble
            else f"replan every {policy.config.n_action_steps} steps")
    print(f"動かし方: {mode}")

    scene = ss.find_or_download_scene(args.workdir)
    cams = [k.split(".")[-1] for k in policy.config.input_features if k.startswith("observation.images.")]
    sim = ss.StampSim(scene, seed=args.seed, paper=args.paper, wrist="wrist" in cams)
    print(f"紙: {args.paper} / AI が見るカメラ: {', '.join(cams)}")
    n_frames = int(args.seconds * ss.FPS)

    results, video = [], []
    t0 = time.time()
    with mujoco.Renderer(sim.model, height=ss.IMG_H, width=ss.IMG_W) as renderer:
        for trial in range(args.trials):
            center = sim.new_scene()
            res, frames = run_trial(sim, policy, preprocess, postprocess, renderer,
                                    center, n_frames, trial < args.video_trials, trial)
            res.update({"trial": trial, "frame_x_mm": center[0] * 1000, "frame_y_mm": center[1] * 1000})
            results.append(res)
            video += frames
            print(f"#{trial + 1:02d} {res['result']:<6} ずれ {res['offset_mm']:5.2f} mm"
                  f"  （経過 {time.time() - t0:5.0f} 秒）")

    # ---- 集計 ----
    n_ok = sum(r["success"] for r in results)
    offsets = np.array([r["offset_mm"] for r in results if not np.isnan(r["offset_mm"])])
    summary = [f"紙: {args.paper} / カメラ: {', '.join(cams)}", f"動かし方: {mode}", f"成功率 {n_ok}/{len(results)} ({100 * n_ok / len(results):.0f}%)"]
    if len(offsets):
        summary.append(f"ずれ 平均 {offsets.mean():.2f} mm / 最大 {offsets.max():.2f} mm"
                       f"（許容 {ss.MAX_OFFSET_MM:.1f} mm）")
    print("\n" + "\n".join(summary))
    (out / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = ["trial", "result", "offset_mm", "dx_mm", "dy_mm", "frame_x_mm", "frame_y_mm"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})

    # ---- 散布図（赤い点線の円の内側なら成功）----
    pts = [(r["dx_mm"], r["dy_mm"], r["success"]) for r in results if not np.isnan(r["dx_mm"])]
    fig, ax = plt.subplots(figsize=(5, 5))
    for dx, dy, ok in pts:
        ax.plot(dx, dy, "o", color="tab:blue" if ok else "tab:red")
    ax.add_patch(plt.Circle((0, 0), ss.MAX_OFFSET_MM, fill=False, color="red", ls="--"))
    ax.plot(0, 0, "+", color="black", markersize=12)
    lim = max([ss.MAX_OFFSET_MM * 1.5] + [abs(v) for p in pts for v in p[:2]])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("dx [mm]")
    ax.set_ylabel("dy [mm]")
    ax.grid(alpha=0.3)
    ax.set_title(f"AI (ACT): success {n_ok}/{len(results)}")
    fig.savefig(out / "offsets.png", dpi=120)

    if video:
        try:
            media.write_video(out / "act_rollouts.mp4", video, fps=ss.FPS)
        except RuntimeError as e:
            print(f"動画を保存できませんでした（{e}）。pip install imageio-ffmpeg を実行してください")
    print(f"\n保存先: {out.resolve()}")
    print("  summary.txt（成功率）、results.csv（1回ごとの結果）、offsets.png（散布図）、act_rollouts.mp4（動画）")


if __name__ == "__main__":
    main()
