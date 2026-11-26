"""Guided B6.2 convention test: neutral, s2->200 (+X0), s1->110 (sigma1), s3->200 (sigma3).

python convention_test.py [--mock] [--port P] [--torque-limit 300]
Every move is announced and needs y before it happens. Hand near the power cut.
"""
import argparse
import sys

import numpy as np

import hop3cfg as hc
import safety
from bus import make_bus, move_slow

DEG_S = 5.0
SPEED = 300
ACC = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--port")
    ap.add_argument("--torque-limit", type=int, default=300, help="RAM torque limit for the test (default 300 = 30%%)")
    hc.add_prompt_args(ap)
    args = ap.parse_args()
    ask = hc.prompter_from_args(args)
    cfg = hc.load_config()
    neutral = np.array(hc.NEUTRAL_S)

    bus = make_bus(cfg, mock=args.mock)
    port = "mock" if args.mock else (args.port or hc.load_session(cfg).values["port"] or cfg["port"])
    try:
        bus.open(port, cfg["stream"]["baud"])
    except IOError as e:
        hc.die(str(e))
    missing = [i for i, m in bus.ping_all(bus.ids).items() if not m]
    if missing:
        hc.die(f"no ping from {missing}")

    def move(desc, s):
        print(f"\nNEXT MOVE: {desc}\n  target servo deg (s1,s2,s3) = {tuple(s)}  at {DEG_S} deg/s, "
              f"torque limit {args.torque_limit}")
        if not ask.confirm("Arm clear, hand on the power cut. Move?"):
            print("not confirmed: stopping (torque stays on, arm holds)")
            bus.close()
            sys.exit(2)
        if not move_slow(bus, s, DEG_S, SPEED, ACC):
            print("  WARNING: did not fully settle")
        print(f"  present: {np.round(bus.present_deg(), 2)} deg")

    try:
        for i in bus.ids:
            bus.set_torque_limit(i, args.torque_limit)
        bus.torque_on_safely()
        move("neutral (90,180,180): arm straight up", neutral)
        nominal = float(hc.fk_servo(neutral, cfg)[2])
        h = ask.number(f"Measured tip height above base bottom [mm] (DH says {nominal:.1f})")
        if h is not None:
            print(f"  DH - measured = {nominal - h:+.1f} mm")

        move("s2 -> 200: arm leans ~20 deg; THAT side is +X0", neutral + [0, 20, 0])
        ask.enter("Tape a +X0 arrow on the table in the lean direction, then")
        move("back to neutral", neutral)

        move("s1 -> 110: base turns 20 deg", neutral + [20, 0, 0])
        sigma1 = 1 if ask.yes_no("Viewed from ABOVE, did the arm turn counter-clockwise (CCW)?") else -1
        move("back to neutral", neutral)

        move("s3 -> 200: link 3 tilts 20 deg", neutral + [0, 0, 20])
        sigma3 = -1 if ask.yes_no("Did link 3 tilt TOWARD +X0 (the taped arrow)?") else 1
        move("back to neutral", neutral)
    except (KeyboardInterrupt, safety.SafetyError, IOError) as e:
        print(f"\nABORT ({type(e).__name__}: {e}): holding present position, torque on")
        try:
            bus.hold_present()
        except Exception as e2:
            print(f"  hold_present failed: {e2}")
        bus.close()
        sys.exit(2)
    bus.close()

    sp = hc.update_session({"sigma1": sigma1, "sigma3": sigma3, "neutral_tip_height_mm": h}, mock=args.mock)
    print(f"\nrecorded sigma1={sigma1:+d} sigma3={sigma3:+d} neutral_tip_height_mm={h} -> {hc.rel(sp)}")

    cur1, cur3 = cfg["map"]["theta1"]["sign"], cfg["map"]["theta3"]["sign"]
    if (cur1, cur3) != (sigma1, sigma3):
        print(f"config.json map signs are theta1={cur1:+d} theta3={cur3:+d}; measured theta1={sigma1:+d} "
              f"theta3={sigma3:+d}")
        if ask.confirm("Update config.json map signs to the measured values?"):
            target = hc.config_path()
            if args.mock:
                target = hc.mock_dir() / "config.json"
                target.write_text(hc.config_path().read_text())
                print("  (mock: writing logs/mock/config.json; real config.json untouched)")
            hc.set_config_signs(sigma1, sigma3, target)
            print(f"  updated {hc.rel(target)}. NOW: .venv/bin/python code/python/retarget_cmds.py data/cmd_*.csv "
                  "(re-runs safety on every file)")
        else:
            print("  config NOT changed: command files still use the old map; do not run them until resolved")
    else:
        print("config.json map signs already match")

    if sigma3 == -1:
        print("\nDECISION: sigma3 = -1 (slide 12) -> run the PRIMARY trajectory family")
    else:
        print("\nDECISION: sigma3 = +1 -> run the BACKUP trajectory (data/cmd_backup.csv); note the reason in the report")


if __name__ == "__main__":
    main()
