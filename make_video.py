"""
提出用の動画を作る（1280×720、30fps、約80秒）

  1. タイトル
  2. タスクと成功条件
  3. AI が学習していく様子（学習の誤差が下がるグラフ）
  4. 学習した AI（ACT）がシミュレーションで押す映像（eval_act_sim.py が作った動画）
  5. 押した位置の散布図（1回ずつ点が増えていく）
  6. 条件ごとの成功率の棒グラフ（棒が伸びていく）
  7. わかったこと
  8. おわり

使い方（Anaconda Prompt で）:
    conda activate lerobot
    cd C:\\so101
    python make_video.py

できあがり: C:\\so101\\PAI最終課題_Doppelar.mp4
"""
import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
import mediapy as media

if shutil.which("ffmpeg") is None:
    try:
        import imageio_ffmpeg
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        sys.exit("ffmpeg が見つかりません。pip install imageio-ffmpeg を実行してください")

W, H, FPS = 1280, 720, 30
BG = (250, 250, 247)
INK = (30, 32, 38)
SUB = (95, 100, 110)
ACCENT = (200, 45, 55)     # 朱肉の赤
GOOD = (40, 120, 90)

# ---------- 日本語フォント ----------
FONT_FILES = ["C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/msgothic.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]
FONT_PATH = next((f for f in FONT_FILES if Path(f).exists()), None)
if FONT_PATH is None:
    sys.exit("日本語のフォントが見つかりません")
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
_fonts = {}


def font(size):
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(FONT_PATH, size)
    return _fonts[size]


# ---------- 結果（実験でわかった数字。summary.txt があればそちらを使う）----------
RESULTS = [
    # (表示名, 成功率[%], 評価フォルダ, 強調)
    ("先生役\n（正解の位置を知っている）", 100, None, False),
    ("全体カメラ\nお手本50回", 44, "eval_demo50", False),
    ("全体カメラ\nお手本200回", 34, "eval_demo200", False),
    ("全体カメラ\n200回・学習8万回", 32, "eval_demo200_80k", False),
    ("全体＋手首カメラ\nお手本200回", 62, "eval_simple_wrist", True),
    ("伝票風の紙\n（毎回ちがう伝票）", 0, "eval_form_wrist", False),
    ("伝票風の紙\n（同じ伝票1種類）", None, "eval_fixed_wrist", False),
]

# 学習の誤差（お手本50回のAI、最初の5000回。学習の画面に出た loss の値）
LOSS = [(200, 6.428), (400, 2.862), (600, 2.403), (800, 1.977), (1000, 1.738), (1200, 1.568),
        (1400, 1.410), (1600, 1.259), (1800, 1.131), (2000, 1.023), (2200, 0.925), (2400, 0.835),
        (2600, 0.743), (2800, 0.665), (3000, 0.603), (3200, 0.552), (3400, 0.496), (3600, 0.443),
        (3800, 0.404), (4000, 0.372), (4200, 0.342), (4400, 0.318), (4600, 0.292), (4800, 0.276),
        (5000, 0.260)]


def read_rate(folder):
    f = folder / "summary.txt"
    if not f.exists():
        return None
    m = re.search(r"成功率\s*(\d+)/(\d+)", f.read_text(encoding="utf-8"))
    return round(100 * int(m.group(1)) / int(m.group(2))) if m else None


# ---------- 絵を描く道具 ----------
def card(lines, footer=None):
    """文字だけの画面。lines = [(文字, 大きさ, 色), ...]"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 14, H], fill=ACCENT)
    total = sum(s * 1.6 for _, s, _ in lines)
    y = (H - total) / 2
    for text, size, color in lines:
        d.text((110, y), text, font=font(size), fill=color)
        y += size * 1.6
    if footer:
        d.text((110, H - 70), footer, font=font(24), fill=SUB)
    return np.asarray(img)


def fig_to_array(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def fade(frames, n=12):
    """最初と最後を白からフェード"""
    out = []
    bg = np.array(BG, np.float32)
    for i, f in enumerate(frames):
        a = min(1.0, (i + 1) / n, (len(frames) - i) / n)
        out.append((f.astype(np.float32) * a + bg * (1 - a)).astype(np.uint8))
    return out


def hold(img, seconds):
    return fade([img] * int(seconds * FPS))


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


# ---------- 各場面 ----------
def scene_title():
    return hold(card([
        ("SO-101 押印ロボット", 64, INK),
        ("AIが、紙の丸枠の中にハンコを押す", 40, ACCENT),
        ("", 20, INK),
        ("シミュレーション（MuJoCo）× 模倣学習（LeRobot ACT）", 28, SUB),
    ], footer="PAI最終課題　Doppelar　／　等倍再生"), 4)


def scene_task():
    return hold(card([
        ("やりたいこと", 30, ACCENT),
        ("置かれた紙の丸枠を、AIがカメラで見つけてハンコを押す", 34, INK),
        ("", 16, INK),
        ("成功の条件", 30, ACCENT),
        ("印影 10mm ／ 丸枠 20mm → 印影の中心が枠の中心から 5mm 以内", 30, INK),
        ("", 16, INK),
        ("流れ", 30, ACCENT),
        ("先生役（逆運動学）が押す → 映像と関節角度を記録 → ACT で学習 → AIだけで押す", 28, INK),
    ]), 7)


def scene_loss():
    frames = []
    steps = np.array([s for s, _ in LOSS])
    vals = np.array([v for _, v in LOSS])
    n = int(6 * FPS)
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(np.array(BG) / 255)
    for i in range(n):
        k = max(2, int(round(ease(i / (n * 0.75)) * len(steps))))
        ax.clear()
        ax.set_facecolor(np.array(BG) / 255)
        ax.plot(steps[:k], vals[:k], color=np.array(ACCENT) / 255, lw=4)
        ax.scatter(steps[k - 1], vals[k - 1], s=90, color=np.array(ACCENT) / 255, zorder=3)
        ax.set_xlim(0, 5200)
        ax.set_ylim(0, 7)
        ax.set_xlabel("学習の回数", fontsize=20)
        ax.set_ylabel("お手本とのちがい（loss）", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.grid(alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"AIがお手本を学習していく様子　loss {vals[k - 1]:.2f}", fontsize=28, pad=20)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.14)
        frames.append(fig_to_array(fig))
    plt.close(fig)
    return fade(frames)


def scene_rollouts(clip_path):
    """動画を1枚ずつ読みながら、上下に説明を付ける（メモリを節約するため1枚ずつ渡す）"""
    top = 90
    with media.VideoReader(clip_path) as reader:
      for f in reader:
        img = Image.new("RGB", (W, H), (20, 22, 26))
        src = Image.fromarray(f)
        scale = min(W / src.width, (H - top - 60) / src.height)
        src = src.resize((int(src.width * scale), int(src.height * scale)))
        img.paste(src, ((W - src.width) // 2, top + (H - top - 60 - src.height) // 2))
        d = ImageDraw.Draw(img)
        d.text((40, 18), "学習したAI（ACT）がシミュレーションでハンコを押す", font=font(34), fill=(255, 255, 255))
        d.text((40, H - 48),
               "AIが見るのはカメラ画像と関節の角度だけ。丸枠の位置は教えていない" +
               ("　（左：全体カメラ ／ 右：手首カメラ）" if src.width > src.height * 1.5 else ""),
               font=font(22), fill=(200, 205, 215))
        yield np.asarray(img)


def scene_scatter(csv_path, title):
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["dx_mm"]), float(r["dy_mm"]), r["result"] == "成功"))
            except (ValueError, KeyError):
                rows.append(None)  # 紙に触れなかった回
    hits = [r for r in rows if r is not None]
    lim = max([7.5] + [abs(v) + 1 for r in hits for v in r[:2]])
    lim = min(lim, 20)
    n = int(8 * FPS)
    frames = []
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(np.array(BG) / 255)
    for i in range(n):
        k = int(round(ease(i / (n * 0.7)) * len(rows)))
        shown = [r for r in rows[:k] if r is not None]
        ok = sum(r[2] for r in shown)
        ax.clear()
        ax.set_facecolor(np.array(BG) / 255)
        ax.add_patch(plt.Circle((0, 0), 5, fill=False, color=np.array(ACCENT) / 255, ls="--", lw=2.5))
        ax.add_patch(plt.Circle((0, 0), 10, fill=False, color=np.array(ACCENT) / 255, lw=1.5, alpha=0.5))
        ax.plot(0, 0, "+", color="black", ms=16, mew=2)
        for dx, dy, good in shown:
            ax.plot(dx, dy, "o", ms=11, color=np.array(GOOD if good else ACCENT) / 255, alpha=0.85)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("ずれ 左右 [mm]", fontsize=18)
        ax.set_ylabel("ずれ 前後 [mm]", fontsize=18)
        ax.tick_params(labelsize=14)
        ax.grid(alpha=0.3)
        ax.set_title(f"{title}　成功 {ok} / {k} 回", fontsize=26, pad=16)
        fig.text(0.70, 0.55, "● 成功（5mm以内）", color=np.array(GOOD) / 255, fontsize=20)
        fig.text(0.70, 0.48, "● 失敗", color=np.array(ACCENT) / 255, fontsize=20)
        fig.text(0.70, 0.41, "点線：5mm\n薄い線：丸枠（20mm）", color=np.array(SUB) / 255, fontsize=16)
        fig.subplots_adjust(left=0.05, right=0.65, top=0.88, bottom=0.12)
        frames.append(fig_to_array(fig))
        fig.texts.clear()
    plt.close(fig)
    return fade(frames)


def scene_bars(results):
    labels = [r[0] for r in results]
    vals = [r[1] for r in results]
    hi = [r[3] for r in results]
    n = int(10 * FPS)
    frames = []
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(np.array(BG) / 255)
    for i in range(n):
        ax.clear()
        ax.set_facecolor(np.array(BG) / 255)
        for j, (v, h) in enumerate(zip(vals, hi)):
            t = ease((i - j * 12) / 40)
            color = np.array(ACCENT if h else (150, 155, 165)) / 255
            if j == 0:
                color = np.array((90, 95, 105)) / 255
            ax.bar(j, v * t, color=color, width=0.65)
            if t > 0.9:
                ax.text(j, v * t + 2, f"{v}%", ha="center", fontsize=22,
                        color=np.array(ACCENT if h else INK) / 255, weight="bold" if h else None)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_ylim(0, 112)
        ax.set_ylabel("成功率 [%]", fontsize=18)
        ax.tick_params(axis="y", labelsize=14)
        ax.grid(axis="y", alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title("条件ごとの成功率：お手本・学習を増やすより、手首カメラが効いた", fontsize=26, pad=20)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.2)
        frames.append(fig_to_array(fig))
    plt.close(fig)
    return fade(frames)


def scene_findings(results):
    fixed = next((r[1] for r in results if "1種類" in r[0]), None)
    lines = [
        ("わかったこと", 40, ACCENT),
        ("", 10, INK),
        ("① お手本を4倍、学習を2.7倍にしても成功率は 30〜45% のまま", 30, INK),
        ("② 原因は「量」ではなく「目」：遠くのカメラでは丸枠が約20画素しかない", 30, INK),
        ("③ 手首カメラを足すと、同じお手本の量で 34% → 62%", 30, INK),
        ("④ 毎回ちがう伝票では 0%：押印欄を探す力は、200回では覚えられない", 30, INK),
    ]
    if fixed is not None:
        lines.append((f"⑤ 伝票を1種類に決めると {fixed}%：実際の使い方に合わせた設計が大事", 30, INK))
    return hold(card(lines), 9)


def scene_end():
    return hold(card([
        ("次は実機へ", 52, INK),
        ("同じ手順（お手本の記録 → 学習 → AIで押す）を、SO-ARM101 の実機で", 30, SUB),
        ("", 20, INK),
        ("コード：GitHub（Doppelar）", 26, SUB),
    ], footer="PAI最終課題　Doppelar"), 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="C:/so101", help="評価の結果が入っているフォルダ")
    p.add_argument("--clip", default=None, help="AIが押す動画（指定しなければ自動で探す）")
    p.add_argument("--out", default=None, help="できあがりのファイル名")
    args = p.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "PAI最終課題_Doppelar.mp4"

    # 結果の数字（summary.txt があれば上書き）
    results = []
    for name, rate, folder, h in RESULTS:
        if folder:
            r = read_rate(root / folder)
            rate = r if r is not None else rate
        if rate is not None:
            results.append((name, rate, folder, h))

    # AIの動画と散布図の元データ（手首カメラのAIを優先）
    order = ["eval_fixed_wrist", "eval_simple_wrist", "eval_act_ensemble", "eval_demo50"]
    clip = Path(args.clip) if args.clip else next(
        (root / f / "act_rollouts.mp4" for f in order[1:] if (root / f / "act_rollouts.mp4").exists()), None)
    if clip is None or not clip.exists():
        sys.exit("AIが押す動画が見つかりません。--clip で act_rollouts.mp4 の場所を指定してください")
    csv_dir = clip.parent
    names = {"eval_simple_wrist": "全体＋手首カメラのAI", "eval_fixed_wrist": "伝票1種類のAI",
             "eval_demo50": "全体カメラのAI", "eval_act_ensemble": "全体カメラのAI"}
    title = names.get(csv_dir.name, "学習したAI")
    print(f"AIの動画: {clip}")

    scenes = [
        ("タイトル", scene_title),
        ("タスク", scene_task),
        ("学習のグラフ", scene_loss),
        ("AIが押す映像", lambda: scene_rollouts(clip)),
        ("散布図", lambda: scene_scatter(csv_dir / "results.csv", title)),
        ("成功率のグラフ", lambda: scene_bars(results)),
        ("わかったこと", lambda: scene_findings(results)),
        ("おわり", scene_end),
    ]
    tmp = out.with_name(out.stem + "_作成中.mp4")
    total = 0
    with media.VideoWriter(tmp, shape=(H, W), fps=FPS) as writer:
        for name, make in scenes:
            print(f"  {name} を作っています...")
            for f in make():
                writer.add_image(f)
                total += 1
    tmp.replace(out)
    print(f"\nできました: {out}（{total / FPS:.0f} 秒）")


if __name__ == "__main__":
    main()
