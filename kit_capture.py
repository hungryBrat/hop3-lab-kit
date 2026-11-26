#!/usr/bin/env python3
"""Fallback data-capture tool. Needs only scservo_sdk. Run from kit root:
  python kit_capture.py probe
  python kit_capture.py torque off|on
  python kit_capture.py read 30 handguide1        # log encoders 30 s (~50 Hz)
  python kit_capture.py pose touch_F1              # average 20 reads -> logs/poses.csv
  python kit_capture.py goto 90 180 180            # servo degrees J1 J2 J3 (slow, limit-checked)
  python kit_capture.py play data/cmd_letterB.csv        # wait-mode playback (every 10th row)
Options: --port /dev/ttyACM0
Only IDs 1-3 are ever read or written. The gripper (ID 4) is DEAD: never written; probe gives it one read-only
ping and reports it as ignored. --gripper is refused.
"""
import argparse, csv, glob, os, sys, time
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS, SMS_STS_PRESENT_POSITION_L, SMS_STS_TORQUE_ENABLE, SMS_STS_OFS_L, SMS_STS_PRESENT_VOLTAGE, SMS_STS_PRESENT_TEMPERATURE

LIM = {1: (0, 180), 2: (135, 270), 3: (90, 315)}
MARGIN = 5
IDS = (1, 2, 3)
GRIPPER_ID = 4
os.makedirs("logs", exist_ok=True)

def c2d(c): return c * 360.0 / 4096
def d2c(d): return int(round(d * 4096 / 360.0))
def ts(): return time.strftime("%Y%m%d_%H%M%S")

def open_bus(port):
    ph = PortHandler(port)
    if not ph.openPort() or not ph.setBaudRate(1000000):
        sys.exit(f"cannot open {port} @1M. Try: ls /dev/ttyACM* /dev/ttyUSB* ; --port ...")
    return ph, sms_sts(ph)

def read_pos(pk, i):
    p, r, e = pk.read2ByteTxRx(i, SMS_STS_PRESENT_POSITION_L)
    return p if r == COMM_SUCCESS else None

def check(i, deg, margin):
    lo, hi = LIM[i]
    if not (lo + margin <= deg <= hi - margin):
        sys.exit(f"REFUSED: J{i}={deg:.1f} outside [{lo + margin},{hi - margin}]")

def wait_settle(pk, goals, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(read_pos(pk, i) is not None and abs(read_pos(pk, i) - g) <= 3 for i, g in goals.items()):
            return True
        time.sleep(0.05)
    print("WARN: not settled within timeout")
    return False

def torque(pk, on, ids=IDS):
    for i in ids:
        assert i in IDS, f"refusing to write id {i}"
        if on:
            p = read_pos(pk, i)
            if p is not None:
                pk.WritePosEx(i, p, 200, 20)
        pk.write1ByteTxRx(i, SMS_STS_TORQUE_ENABLE, 1 if on else 0)

def log_row(w, t0, pk, label, extra=()):
    row = [f"{time.perf_counter() - t0:.4f}", label]
    for i in IDS:
        p = read_pos(pk, i)
        row += [p if p is not None else "", f"{c2d(p):.3f}" if p is not None else ""]
    w.writerow(row + list(extra))

HDR = ["t_s", "label", "c1", "deg1", "c2", "deg2", "c3", "deg3"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("args", nargs="*")
    ap.add_argument("--port", default=(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*") + ["/dev/ttyACM0"])[0])
    ap.add_argument("--gripper", type=float, default=None, help="REFUSED: gripper (ID 4) is dead")
    ap.add_argument("--speed", type=int, default=300)
    a = ap.parse_args()
    if a.gripper is not None:
        sys.exit("REFUSED: --gripper. The gripper servo (ID 4) is dead and must never be commanded.")
    if a.speed <= 0:
        sys.exit("REFUSED: --speed must be > 0 (0 = max speed on ST3215)")
    ph, pk = open_bus(a.port)
    print("port", a.port)
    try:
        if a.cmd == "probe":
            with open(f"logs/probe_{ts()}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["id", "ping_model", "pos_counts", "pos_deg", "offset_reg31", "voltage_0.1V", "temp_C"])
                for i in IDS:
                    m, r, e = pk.ping(i)
                    p = read_pos(pk, i)
                    o, _, _ = pk.read2ByteTxRx(i, SMS_STS_OFS_L)
                    v, _, _ = pk.read1ByteTxRx(i, SMS_STS_PRESENT_VOLTAGE)
                    t, _, _ = pk.read1ByteTxRx(i, SMS_STS_PRESENT_TEMPERATURE)
                    ok = r == COMM_SUCCESS
                    print(f"ID{i}: {'OK' if ok else 'NO REPLY'} pos={p} ({c2d(p):.2f} deg) ofs={o} V={v/10:.1f} T={t}" if ok else f"ID{i}: NO REPLY")
                    w.writerow([i, m if ok else "", p, f"{c2d(p):.3f}" if p is not None else "", o, v, t])
                m, r, e = pk.ping(GRIPPER_ID)
                state = "dead/ignored" if r != COMM_SUCCESS else f"answered ping (model {m}) but ignored"
                print(f"ID{GRIPPER_ID} (gripper): {state} - never written, not required")
                w.writerow([GRIPPER_ID, m if r == COMM_SUCCESS else "", "", "", "ignored", "", ""])
        elif a.cmd == "torque":
            torque(pk, a.args[0] == "on"); print("torque", a.args[0], "(SUPPORT THE ARM when off)")
        elif a.cmd == "read":
            secs = float(a.args[0]) if a.args else 20; label = a.args[1] if len(a.args) > 1 else "read"
            fn = f"logs/read_{label}_{ts()}.csv"
            with open(fn, "w", newline="") as f:
                w = csv.writer(f); w.writerow(HDR); t0 = time.perf_counter(); n = 0
                while time.perf_counter() - t0 < secs:
                    log_row(w, t0, pk, label); n += 1
                    time.sleep(0.005)
            print(f"{n} rows -> {fn}")
        elif a.cmd == "pose":
            label = a.args[0] if a.args else "pose"
            acc = {i: [] for i in IDS}
            for _ in range(20):
                for i in acc:
                    p = read_pos(pk, i)
                    if p is not None: acc[i].append(p)
                time.sleep(0.02)
            new = not os.path.exists("logs/poses.csv")
            with open("logs/poses.csv", "a", newline="") as f:
                w = csv.writer(f)
                if new: w.writerow(["time", "label", "c1", "c2", "c3", "deg1", "deg2", "deg3", "n1", "n2", "n3"])
                m = {i: (sum(v) / len(v) if v else float("nan")) for i, v in acc.items()}
                w.writerow([ts(), label] + [f"{m[i]:.2f}" for i in IDS] + [f"{c2d(m[i]):.3f}" for i in IDS] + [len(acc[i]) for i in IDS])
            print(label, " ".join(f"J{i}={c2d(m[i]):.2f}" for i in IDS), "-> logs/poses.csv")
        elif a.cmd == "goto":
            s = [float(x) for x in a.args[:3]]
            for i, d in zip(IDS, s): check(i, d, MARGIN)
            goals = {i: d2c(d) for i, d in zip(IDS, s)}
            fn = f"logs/goto_{ts()}.csv"
            with open(fn, "w", newline="") as f:
                w = csv.writer(f); w.writerow(HDR + ["goal1", "goal2", "goal3"]); t0 = time.perf_counter()
                for i, g in goals.items(): pk.WritePosEx(i, g, a.speed, 20)
                while time.perf_counter() - t0 < 6:
                    log_row(w, t0, pk, "goto", [goals[1], goals[2], goals[3]]); time.sleep(0.01)
            wait_settle(pk, goals)
            print("at", " ".join(f"J{i}={c2d(read_pos(pk, i)):.2f}(cmd {d:.2f})" for i, d in zip(IDS, s)), "->", fn)
        elif a.cmd == "play":
            rows = list(csv.DictReader(open(a.args[0])))
            step = int(a.args[1]) if len(a.args) > 1 else 10
            pts = rows[::step] + [rows[-1]]
            for r in pts:
                for i in IDS: check(i, float(r[f"s{i}_deg"]), MARGIN)
            name = os.path.splitext(os.path.basename(a.args[0]))[0]
            fn = f"logs/play_{name}_{ts()}.csv"
            input(f"{len(pts)} waypoints from {a.args[0]}. Arm clear? Enter to start (Ctrl-C abort)")
            with open(fn, "w", newline="") as f:
                w = csv.writer(f); w.writerow(HDR + ["k", "goal1", "goal2", "goal3"]); t0 = time.perf_counter()
                for r in pts:
                    goals = {i: int(r[f"c{i}"]) for i in IDS}
                    for i, g in goals.items(): pk.WritePosEx(i, g, a.speed, 20)
                    tw = time.perf_counter()
                    while time.perf_counter() - tw < 3:
                        log_row(w, t0, pk, "play", [r["k"], goals[1], goals[2], goals[3]])
                        if all(abs((read_pos(pk, i) or -99) - g) <= 3 for i, g in goals.items()): break
            print("done ->", fn)
        else:
            print(__doc__)
    except KeyboardInterrupt:
        torque(pk, True, IDS); print("ABORT: holding present position")
    finally:
        ph.closePort()

if __name__ == "__main__":
    main()
