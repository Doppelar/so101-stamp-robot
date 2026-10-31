"""
実機の状態を「読むだけ」で調べる（モーターには何も書き込まない＝動かない）

  - ソフトのバージョン
  - 各アームの6つのモーターが、何回呼んで何回返事をしたか（通信の安定度）
  - 電圧・温度・負荷・エラー状態
  - モーターに保存されている設定（力の上限、過負荷の設定、動ける範囲）

使い方（Anaconda Prompt で）:
    conda activate lerobot
    python C:\\so101\\real\\check_arm.py

ポート番号が変わったときは:
    python C:\\so101\\real\\check_arm.py --follower COM4 --leader COM3
"""
import argparse
import platform
import time

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Status（エラー状態）の各ビットの意味（STS3215 の説明書より）
STATUS_BITS = {0: "電圧", 1: "角度センサー", 2: "温度", 3: "電流", 5: "過負荷"}

LIVE = ["Present_Position", "Present_Voltage", "Present_Temperature", "Present_Load", "Status"]
SETTINGS = ["Torque_Enable", "Max_Torque_Limit", "Protection_Current", "Overload_Torque",
            "Min_Position_Limit", "Max_Position_Limit"]


def versions():
    import lerobot
    import torch
    print("=== バージョン ===")
    print(f"Python   : {platform.python_version()}")
    print(f"LeRobot  : {getattr(lerobot, '__version__', '不明')}")
    print(f"PyTorch  : {torch.__version__}  GPU: {torch.cuda.is_available()}")
    try:
        from importlib.metadata import version
        print(f"Feetech  : {version('feetech-servo-sdk')}")
    except Exception:
        print("Feetech  : 不明")
    print()


def status_text(value):
    if not value:
        return "なし"
    return "・".join(name for bit, name in STATUS_BITS.items() if value & (1 << bit)) or f"不明({value})"


def check_port(label, port, rounds):
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    print(f"=== {label}（{port}）===")
    bus = FeetechMotorsBus(
        port=port,
        motors={name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, name in enumerate(MOTORS)},
    )
    try:
        bus.connect(handshake=False)  # モーターの数を確かめずに、ポートだけ開く
    except Exception as e:
        print(f"  ポートを開けません：{e}")
        print("  → 番号が違うか、別のプログラム（teleoperate など）が使用中です\n")
        return

    stats = {m: {"ok": 0, "volt": [], "temp": [], "load": [], "status": set()} for m in MOTORS}
    try:
        print(f"  {rounds} 回呼びかけています（約 {rounds * 0.15:.0f} 秒）...")
        for _ in range(rounds):
            for m in MOTORS:
                if bus.ping(m) is None:  # 返事がない
                    continue
                stats[m]["ok"] += 1
                try:
                    v = {r: bus.read(r, m, normalize=False) for r in LIVE}
                except Exception:
                    continue
                stats[m]["volt"].append(v["Present_Voltage"] / 10)
                stats[m]["temp"].append(v["Present_Temperature"])
                stats[m]["load"].append(abs(v["Present_Load"]) / 10)
                if v["Status"]:
                    stats[m]["status"].add(int(v["Status"]))
            time.sleep(0.1)

        print("  モーター        返事   電圧[V]      温度[℃]  負荷[%]  エラー")
        for i, m in enumerate(MOTORS):
            s = stats[m]
            rate = f"{s['ok']}/{rounds}"
            if s["volt"]:
                volt = f"{min(s['volt']):4.1f}〜{max(s['volt']):4.1f}"
                temp = f"{max(s['temp']):3d}"
                load = f"{max(s['load']):5.1f}"
            else:
                volt, temp, load = "   ―    ", " ―", "  ―"
            errs = "、".join(status_text(x) for x in sorted(s["status"])) or "なし"
            print(f"  {i + 1} {m:<13} {rate:>6}   {volt}   {temp}     {load}   {errs}")

        print("\n  モーターに保存されている設定")
        print("  モーター        力入り  力の上限  保護電流  過負荷時の力  範囲（下限〜上限）")
        for i, m in enumerate(MOTORS):
            if stats[m]["ok"] == 0:
                print(f"  {i + 1} {m:<13} 読めません")
                continue
            try:
                c = {r: bus.read(r, m, normalize=False) for r in SETTINGS}
                print(f"  {i + 1} {m:<13} {c['Torque_Enable']:>4}   {c['Max_Torque_Limit']:>7}  "
                      f"{c['Protection_Current']:>7}  {c['Overload_Torque']:>9}     "
                      f"{c['Min_Position_Limit']}〜{c['Max_Position_Limit']}")
            except Exception as e:
                print(f"  {i + 1} {m:<13} 読み取り失敗（{e}）")
    finally:
        bus.disconnect(disable_torque=False)  # 何も書き込まずに閉じる
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--follower", default="COM4", help="白いアーム（フォロワー）のポート")
    p.add_argument("--leader", default="COM3", help="黒いアーム（リーダー）のポート")
    p.add_argument("--rounds", type=int, default=30, help="呼びかける回数")
    args = p.parse_args()

    versions()
    check_port("白：フォロワー", args.follower, args.rounds)
    check_port("黒：リーダー", args.leader, args.rounds)
    print("終わりました。この画面の内容をそのまま貼ってください。")


if __name__ == "__main__":
    main()
