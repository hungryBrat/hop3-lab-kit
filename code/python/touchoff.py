"""Touch-off: torque off on the arm joints, hand-guide the tip onto each fiducial, log encoders 3x each.

python touchoff.py [--mock] [--port P] [--reps 3]
Writes logs/touchoff_<ts>.csv (point_id,rep,pos_c1,pos_c2,pos_c3) and session_data.json touchoff_log.
"""
import argparse
import csv
import json
import sys

import numpy as np

import hop3cfg as hc
from bus import make_bus


def fiducial_ids():
    p = hc.ROOT / "data" / "fiducials.json"
    if p.exists():
        tp = json.loads(p.read_text()).get("touch_points")
        if tp:
            return list(tp.keys()), tp
    return ["F1", "F2", "F3", "F4"], {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--port")
    ap.add_argument("--reps", type=int, default=3)
    hc.add_prompt_args(ap)
    args = ap.parse_args()
    ask = hc.prompter_from_args(args)
    cfg = hc.load_config()
    points, nominal = fiducial_ids()

    bus = make_bus(cfg, mock=args.mock)
    port = "mock" if args.mock else (args.port or hc.load_session(cfg).values["port"] or cfg["port"])
    try:
        bus.open(port, cfg["stream"]["baud"])
    except IOError as e:
        hc.die(str(e))
    missing = [i for i, m in bus.ping_all(bus.ids).items() if not m]
    if missing:
        hc.die(f"no ping from {missing}")

    print(f"points: {points} x {args.reps} reps. Approach every point from the same direction.")
    print("!!! Torque goes OFF on IDs " + ",".join(map(str, bus.ids)) + ": the arm WILL sag. SUPPORT IT BY HAND.")
    if not ask.confirm("Holding the arm. Switch torque off?"):
        bus.close()
        sys.exit(2)

    rows = []
    ts = hc.timestamp()
    out = hc.logs_dir(args.mock) / f"touchoff_{ts}.csv"
    try:
        for i in bus.ids:
            bus.set_torque(i, False)
        for pid in points:
            where = f" (sheet {nominal[pid]} mm)" if pid in nominal else ""
            for rep in range(1, args.reps + 1):
                ask.enter(f"Guide tip onto {pid}{where}, rep {rep}/{args.reps}, then")
                rd, ok = None, False
                for _ in range(10):
                    rd, ok = bus.sync_read()
                    if ok:
                        break
                if not ok:
                    print("  read failed 10x; press Enter again to retry this rep")
                    continue
                c = [rd[i]["pos"] for i in bus.ids]
                rows.append([pid, rep] + c)
                p = hc.fk_servo(hc.counts_to_deg(c, cfg), cfg)
                print(f"  {pid} rep {rep}: counts {c}  DH tip {np.round(p, 1)} mm")
        ask.enter("Done. Move the arm by hand back near upright (neutral) and keep holding; to re-enable torque")
    finally:
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hc.TOUCHOFF_HEADER)
            w.writerows(rows)
        try:
            bus.torque_on_safely()
            print("torque ON at the present pose (goal = present first, no jump)")
        except Exception as e:
            print(f"!!! could not re-enable torque cleanly: {e}. Keep supporting the arm.")
        bus.close()

    sp = hc.update_session({"touchoff_log": hc.rel(out)}, mock=args.mock)
    print(f"wrote {hc.rel(out)} ({len(rows)} rows); touchoff_log -> {hc.rel(sp)}")
    print("next (MATLAB): touchoff_fit -> plane z, sheet->robot transform; then pick z_c")


if __name__ == "__main__":
    main()
