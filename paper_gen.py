"""
伝票風の紙の画像を、毎回ちがう見た目で作る

  - 表の枠線、項目名、細かい文字、バーコード風の模様を描く
  - 押印欄は「点線の丸」（内径 20mm）で、中に「印」「荷受印」などの文字が入る
  - 押印欄の位置、印刷の色、紙の色、表の割り方は毎回ランダム
  - まぎらわしい物（実線の丸、楕円、角丸の枠）もわざと入れる

座標は紙の中心が原点、右が +x、上が +y [mm]。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAPER_W_MM = 148.0   # 紙の横幅（A6 横向き）
PAPER_H_MM = 105.0   # 紙の縦幅
PX_PER_MM = 4        # 1mm あたりの画素数
STAMP_FIELD_R_MM = 10.0  # 押印欄の半径（内径 20mm）

LABELS = ["お届け先", "ご依頼主", "品名", "電話番号", "郵便番号", "ご住所", "お名前", "様",
          "お届け予定日", "時間帯", "午前中", "備考", "個数", "重量", "受付日", "伝票番号",
          "取扱店", "配達店", "冷蔵", "冷凍", "ワレモノ", "天地無用", "精密機器", "なまもの",
          "受領日", "担当", "数量", "金額", "合計", "摘要", "発行日", "承認"]
FIELD_TEXTS = ["印", "荷受印", "受領印", "受取印", "認印", "検印"]
KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん"

INKS = [(200, 40, 50), (190, 60, 40), (30, 30, 30), (40, 60, 150), (30, 110, 70), (120, 70, 40)]
PAPERS = [(250, 250, 248), (252, 248, 235), (252, 240, 240), (240, 246, 252), (245, 245, 238)]

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/msgothic.ttc", "C:/Windows/Fonts/YuGothM.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
]
_font_path = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)
_font_cache = {}


def _font(size_px):
    size_px = max(6, int(size_px))
    if size_px not in _font_cache:
        if _font_path:
            _font_cache[size_px] = ImageFont.truetype(_font_path, size_px)
        else:
            print("日本語のフォントが見つからないので、標準のフォントを使います", file=sys.stderr)
            _font_cache[size_px] = ImageFont.load_default()
    return _font_cache[size_px]


def _mm(v):
    return int(round(v * PX_PER_MM))


def _dashed_circle(draw, cx, cy, r, color, width, n_dash):
    """点線の丸"""
    for k in range(n_dash):
        a0 = 360.0 * k / n_dash
        draw.arc([cx - r, cy - r, cx + r, cy + r], a0, a0 + 360.0 / n_dash * 0.55, fill=color, width=width)


def make_form(rng):
    """
    伝票風の紙を1枚作る。
    戻り値: (画像 H×W×3 uint8, 押印欄の中心 (x, y) [mm]（紙の中心が原点、右・上が +）)
    """
    W, H = _mm(PAPER_W_MM), _mm(PAPER_H_MM)
    paper = PAPERS[rng.integers(len(PAPERS))]
    ink = INKS[rng.integers(len(INKS))]
    light = tuple(int(c + (255 - c) * 0.75) for c in ink)  # 薄い色（塗りつぶし用）
    img = Image.new("RGB", (W, H), paper)
    d = ImageDraw.Draw(img)
    lw = max(2, _mm(rng.uniform(0.5, 0.8)))  # カメラで見えるよう、実物より少し太め

    # ---- 外枠と表 ----
    m = _mm(rng.uniform(3, 6))
    d.rectangle([m, m, W - m, H - m], outline=ink, width=lw + 1)
    n_rows = int(rng.integers(3, 7))
    ys = np.sort(rng.uniform(m, H - m, n_rows - 1)).astype(int)
    rows = [m] + list(ys) + [H - m]
    for r0, r1 in zip(rows[:-1], rows[1:]):
        d.line([m, r0, W - m, r0], fill=ink, width=lw)
        n_cols = int(rng.integers(1, 5))
        xs = np.sort(rng.uniform(m, W - m, n_cols - 1)).astype(int)
        cols = [m] + list(xs) + [W - m]
        for c0, c1 in zip(cols[:-1], cols[1:]):
            d.line([c0, r0, c0, r1], fill=ink, width=lw)
            if c1 - c0 > _mm(14) and r1 - r0 > _mm(6):
                # 項目名（ときどき色付きの見出し）
                fs = _mm(rng.uniform(2.2, 3.5))
                text = LABELS[rng.integers(len(LABELS))]
                if rng.random() < 0.25:
                    d.rectangle([c0, r0, c0 + _mm(16), r0 + fs + _mm(2)], fill=ink)
                    d.text((c0 + _mm(1), r0 + _mm(1)), text, fill=paper, font=_font(fs))
                else:
                    d.text((c0 + _mm(1), r0 + _mm(1)), text, fill=ink, font=_font(fs))
                # 細かい文字の並び
                if rng.random() < 0.4:
                    fs2 = _mm(rng.uniform(1.5, 2.0))
                    for k in range(int(rng.integers(1, 4))):
                        y = r0 + fs + _mm(2) + k * (fs2 + 2)
                        if y + fs2 < r1:
                            n = max(1, int((c1 - c0) / (fs2 + 1)) - 2)
                            s = "".join(rng.choice(list(KANA), n))
                            d.text((c0 + _mm(1), y), s, fill=ink, font=_font(fs2))
                # 薄い色の塗りつぶし
                if rng.random() < 0.15:
                    d.rectangle([c0 + lw, r0 + lw, c1 - lw, r1 - lw], fill=light)

    # ---- バーコード風の模様 ----
    if rng.random() < 0.7:
        bx, by = _mm(rng.uniform(8, 100)), _mm(rng.uniform(8, 80))
        x = bx
        for _ in range(int(rng.integers(25, 45))):
            w = int(rng.integers(1, 4))
            d.rectangle([x, by, x + w - 1, by + _mm(10)], fill=(20, 20, 20))
            x += w + int(rng.integers(1, 4))

    # ---- まぎらわしい物：実線の丸、楕円、角丸の枠 ----
    for _ in range(int(rng.integers(2, 6))):
        cx, cy = _mm(rng.uniform(10, PAPER_W_MM - 10)), _mm(rng.uniform(10, PAPER_H_MM - 10))
        kind = rng.integers(3)
        if kind == 0:  # 大きさのちがう実線の丸
            r = _mm(rng.choice([rng.uniform(4, 7), rng.uniform(13, 16)]))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ink, width=lw)
        elif kind == 1:  # 楕円（時間帯の欄のような）
            a, b = _mm(rng.uniform(7, 11)), _mm(rng.uniform(3, 5))
            d.ellipse([cx - a, cy - b, cx + a, cy + b], outline=ink, width=lw)
            d.text((cx - a + _mm(2), cy - _mm(1.5)), f"{rng.integers(8, 21)}時", fill=ink, font=_font(_mm(2.5)))
        else:  # 角丸の枠
            a, b = _mm(rng.uniform(6, 12)), _mm(rng.uniform(4, 7))
            d.rounded_rectangle([cx - a, cy - b, cx + a, cy + b], radius=_mm(2), outline=ink, width=lw)

    # ---- 押印欄（点線の丸 + 中の文字）----
    r_mm = STAMP_FIELD_R_MM
    edge = r_mm + 4
    fx = rng.uniform(-PAPER_W_MM / 2 + edge, PAPER_W_MM / 2 - edge)
    fy = rng.uniform(-PAPER_H_MM / 2 + edge, PAPER_H_MM / 2 - edge)
    cx, cy = _mm(PAPER_W_MM / 2 + fx), _mm(PAPER_H_MM / 2 - fy)  # 画像は下向きが +
    r = _mm(r_mm)
    clear = r + _mm(1.5)
    d.ellipse([cx - clear, cy - clear, cx + clear, cy + clear], fill=paper)  # 欄の中を空ける
    _dashed_circle(d, cx, cy, r, ink, lw + 1, int(rng.integers(20, 32)))
    text = FIELD_TEXTS[rng.integers(len(FIELD_TEXTS))]
    fs = _mm(rng.uniform(2.5, 3.5))
    if rng.random() < 0.5:  # 上の方に小さく
        tw = d.textlength(text, font=_font(fs))
        d.text((cx - tw / 2, cy - r + _mm(2)), text, fill=ink, font=_font(fs))
    else:  # 縦書きで右寄り
        for k, ch in enumerate(text):
            d.text((cx + r * 0.35, cy - r * 0.6 + k * (fs + 1)), ch, fill=ink, font=_font(fs))

    return np.asarray(img, dtype=np.uint8), np.array([fx, fy])


FIXED_FORM_SEED = 3   # 「いつも同じ1種類の伝票」を作るときの乱数の種
_fixed = None


def fixed_form():
    """いつも同じ1種類の伝票（make_form を決まった種で1回だけ作る）"""
    global _fixed
    if _fixed is None:
        _fixed = make_form(np.random.default_rng(FIXED_FORM_SEED))
    return _fixed


if __name__ == "__main__":
    # お試し：6枚並べて保存
    rng = np.random.default_rng(0)
    tiles = [make_form(rng)[0] for _ in range(6)]
    grid = np.concatenate([np.concatenate(tiles[i:i + 3], 1) for i in (0, 3)], 0)
    Image.fromarray(grid).save("paper_samples.png")
    print("paper_samples.png を保存しました")
