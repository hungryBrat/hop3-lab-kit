"""Execute a command CSV open-loop and log encoders.

python run_traj.py data/cmd_<name>.csv --mode stream|wait [--mock] [--dashboard [--offscreen]]
                   [--port P] [--torque-limit N] [--dry]
ESC or Ctrl-C: hold present position, torque stays on, log closed cleanly (no return to neutral).
Only IDs 1-3 are pinged/commanded; the gripper (ID 4) is dead and is never written (config gripper_enabled=false).
If the command meta has `segments`, progress is printed at each segment start.
"""
import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

import hop3cfg as hc
import safety
from bus import make_bus, move_slow, plan_slow_path, check_path, wait_settled

APPROACH_DEG_S = 10.0
WAIT_POINTS = 90
WAIT_TIMEOUT_S = 3.0
MAX_CONSEC_FAIL = 25


class KeyWatch:
    """Non-blocking ESC detection on a tty stdin; inert otherwise."""

    def __enter__(self):
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        if self.fd is not None:
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def esc(self):
        if self.fd is None:
            return False
        while select.select([self.fd], [], [], 0)[0]:
            if os.read(self.fd, 1) == b"\x1b":
                return True
        return False

    def __exit__(self, *a):
        if self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


class Abort(Exception):
    pass


def row_fields(rd, ids):
    pos, spd, load = [], [], []
    for i in ids:
        r = rd.get(i)
        pos.append("NaN" if r is None else r["pos"])
        spd.append("NaN" if r is None else r["speed"])
        load.append("NaN" if r is None else r["load"])
    return pos, spd, load


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd")
    ap.add_argument("--mode", choices=["stream", "wait"], default="stream")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dashboard", action="store_true")
    ap.add_argument("--offscreen", action="store_true", help="dashboard on Agg, saves figures/dashboard_snapshot.png")
    ap.add_argument("--port")
    ap.add_argument("--torque-limit", type=int, help="RAM torque limit 1..1000 on IDs 1-3 for this run")
    ap.add_argument("--dry", action="store_true", help="validate + read-only bus checks, no writes")
    ap.add_argument("--seed", type=int, help="mock RNG seed")
    args = ap.parse_args()

    cfg = hc.load_config()
    cmd_path = Path(args.cmd)
    errors, _ = safety.validate_file(cmd_path, cfg)
    if errors:
        print(f"SAFETY FAIL {cmd_path}: aborting before any motion")
        for e in errors:
            print("   ", e)
        sys.exit(1)
    header, arr = hc.read_cmd(cmd_path)
    goals_c = np.column_stack([hc.col(header, arr, n) for n in ("c1", "c2", "c3")]).astype(int)
    goals_s = hc.counts_to_deg(goals_c, cfg)
    n = len(arr)
    name = hc.cmd_name(cmd_path)
    st = cfg["stream"]
    rate = st["rate_hz"]
    print(f"safety PASS: {cmd_path} ({n} rows, {n / rate:.1f} s at {rate} Hz)")
    cmd_meta = hc.load_cmd_meta(cmd_path)
    if cmd_meta.get("placeholder"):
        print("NOTE: command meta says placeholder=true")
    segs = hc.segments(cmd_meta, n)
    if segs:
        print("segments: " + "  ".join(f"{nm} k{a}-{b} ({(b - a + 1) / rate:.1f} s)" for nm, a, b in segs))
    seg_at = {a: (j, nm, b) for j, (nm, a, b) in enumerate(segs)}

    bus = make_bus(cfg, mock=args.mock, seed=args.seed)
    port = "mock" if args.mock else (args.port or hc.load_session(cfg).values["port"] or cfg["port"])
    try:
        bus.open(port, st["baud"])
    except IOError as e:
        hc.die(str(e))
    pings = bus.ping_all()
    missing = [i for i in bus.ids if pings[i] is None]
    print(f"port {port}  ping: " + "  ".join(f"id{i}={'ok' if m else 'NO REPLY'}" for i, m in pings.items())
          + ("" if bus.gripper_enabled else f"  (id{bus.gid} gripper: dead/ignored, never written)"))
    if missing:
        hc.die(f"joint servo(s) {missing} did not answer ping: check power/cables/port (cheatsheet)")
    offsets0 = bus.read_offsets()
    print(f"offsets reg31: {offsets0}")
    present = bus.present_deg()
    approach = plan_slow_path(present, goals_s[0], APPROACH_DEG_S, rate)
    try:
        check_path(bus, approach)
    except safety.SafetyError as e:
        hc.die(f"approach path from present {np.round(present, 1)} to row 0 is unsafe: {e}\n"
               "  torque off and hand-guide the arm nearer neutral (90,180,180), then retry")
    ret_path = plan_slow_path(goals_s[-1], hc.NEUTRAL_S, APPROACH_DEG_S, rate)
    check_path(bus, ret_path)
    print(f"present {np.round(present, 1)} deg -> row 0 {np.round(goals_s[0], 1)} deg "
          f"({len(approach) / rate:.1f} s approach), return to neutral checked")
    if args.dry:
        print("DRY: no writes made")
        bus.close()
        return

    ts = hc.timestamp()
    out_dir = hc.logs_dir(args.mock)
    log_path = out_dir / f"run_{name}_{ts}.csv"
    if args.mode == "wait":
        idx = list(range(0, n, max(1, round(n / WAIT_POINTS))))
        if idx[-1] != n - 1:
            idx.append(n - 1)
    dash = None
    if args.dashboard:
        from dashboard import DashLink
        t_max = n / rate * 1.1 if args.mode == "stream" else len(idx) * 0.4
        dash = DashLink(cmd_path, cfg, args.offscreen, hc.ROOT / "figures" / "dashboard_snapshot.png", t_max)

    meta = {"cmd_file": hc.rel(cmd_path), "mode": args.mode, "mock": args.mock, "synthetic": False,
            "config_snapshot": cfg, "offsets_reg31": offsets0, "port": port, "started": ts,
            "torque_limit": args.torque_limit, "rate_hz_cmd": rate, "speed": st["speed"], "acc": st["acc"],
            "n_rows_cmd": n, "cmd_meta": cmd_meta}
    stats = {"n_logged": 0, "comm_fail_reads": 0, "aborted": False, "abort_reason": None}
    f = open(log_path, "w")
    f.write(",".join(hc.RUN_HEADER) + "\n")
    ids = bus.ids

    def log(t, k, goal, rd, ok):
        pos, spd, load = row_fields(rd, ids)
        f.write(f"{t:.5f},{k},{goal[0]},{goal[1]},{goal[2]},{pos[0]},{pos[1]},{pos[2]},"
                f"{spd[0]},{spd[1]},{spd[2]},{load[0]},{load[1]},{load[2]},{int(ok)}\n")
        stats["n_logged"] += 1
        if not ok:
            stats["comm_fail_reads"] += 1

    def meas_deg(rd, ok):
        return hc.counts_to_deg([rd[i]["pos"] for i in ids], cfg) if ok else None

    t_stream = [None, None]

    def seg_progress(k, t):
        if k in seg_at:
            j, nm, b = seg_at[k]
            print(f"  segment {j + 1}/{len(segs)} {nm}: k {k}-{b}  t={t:6.2f}s  (cmd t {arr[k, 1]:.2f}s)", flush=True)
    try:
        with KeyWatch() as kw:
            if args.torque_limit:
                for i in ids:
                    bus.set_torque_limit(i, args.torque_limit)
            bus.torque_on_safely()
            print("approaching row 0 ...")
            if not move_slow(bus, goals_s[0], APPROACH_DEG_S, abort=kw.esc):
                print("WARNING: approach did not fully settle (MOVING!=0 or >3 counts off); continuing")
            print(f"running {args.mode} ... (ESC/Ctrl-C = hold)")
            t0 = time.perf_counter()
            if args.mode == "stream":
                period = 1.0 / rate
                consec = 0
                anchor = t0
                for k in range(n):
                    target = anchor + k * period
                    now = time.perf_counter()
                    if now < target:
                        time.sleep(target - now)
                    elif now - target > 3 * period:
                        anchor += now - target
                    if kw.esc():
                        raise Abort("ESC")
                    gc = goals_c[k]
                    tw = time.perf_counter() - t0
                    if segs:
                        seg_progress(k, tw)
                    bus.sync_write(dict(zip(ids, gc.tolist())), st["speed"], st["acc"], cfg["max_step_deg"])
                    rd, ok = bus.sync_read()
                    log(tw, k, gc, rd, ok)
                    if t_stream[0] is None:
                        t_stream[0] = tw
                    t_stream[1] = tw
                    consec = 0 if ok else consec + 1
                    if consec >= MAX_CONSEC_FAIL:
                        raise Abort(f"{consec} consecutive failed sync reads (bus dead?)")
                    if dash and k % 3 == 0:
                        dash.send(tw, k, goals_s[k], meas_deg(rd, ok))
            else:
                last_dash = [0.0]
                n_timeouts = 0
                seg_starts = sorted(seg_at)
                for k in idx:
                    if kw.esc():
                        raise Abort("ESC")
                    while segs and seg_starts and seg_starts[0] <= k:
                        seg_progress(seg_starts.pop(0), time.perf_counter() - t0)
                    gc = goals_c[k]
                    bus.sync_write(dict(zip(ids, gc.tolist())), st["speed"], st["acc"])

                    def wlog(rd, ok, k=k, gc=gc):
                        t = time.perf_counter() - t0
                        log(t, k, gc, rd, ok)
                        if dash and t - last_dash[0] >= 0.05:
                            last_dash[0] = t
                            dash.send(t, k, goals_s[k], meas_deg(rd, ok))

                    if not wait_settled(bus, WAIT_TIMEOUT_S, log=wlog):
                        n_timeouts += 1
                stats["wait_timeouts"] = n_timeouts
                t_stream = [0.0, time.perf_counter() - t0]
            print("done; returning to neutral slowly ...")
            move_slow(bus, hc.NEUTRAL_S, APPROACH_DEG_S, abort=kw.esc)
    except (KeyboardInterrupt, Abort, safety.SafetyError, IOError) as e:
        stats["aborted"] = True
        stats["abort_reason"] = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        print(f"\nABORT ({stats['abort_reason']}): holding present position, torque stays ON")
        try:
            bus.hold_present()
        except Exception as e2:
            print(f"  hold_present failed ({e2}); servos keep their last goal")
    finally:
        f.close()
        offsets1 = bus.read_offsets()
        unchanged = offsets1 == offsets0
        if args.mode == "stream" and stats["n_logged"] > 1 and t_stream[1] > t_stream[0]:
            achieved = (stats["n_logged"] - 1) / (t_stream[1] - t_stream[0])
        else:
            achieved = None
        meta.update(stats, offsets_reg31_end=offsets1, offsets_unchanged=unchanged, achieved_rate_hz=achieved,
                    dropped_replies=stats["comm_fail_reads"], mock_dropped=getattr(bus, "dropped", None),
                    dashboard_dropped_frames=dash.dropped if dash else None, log_csv=hc.rel(log_path),
                    written_ids=sorted(bus.written_ids), gripper_enabled=bus.gripper_enabled,
                    segments=cmd_meta.get("segments"))
        log_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        bus.close()
        if dash:
            dash.close()
    rate_s = f"{achieved:.1f} Hz" if achieved else "n/a (wait mode)"
    print(f"log: {hc.rel(log_path)}  rows={stats['n_logged']}  achieved rate={rate_s}  "
          f"dropped replies={stats['comm_fail_reads']}")
    if not unchanged:
        print(f"!!! OFFSETS (reg 31) CHANGED during run: {offsets0} -> {offsets1}. Calibration is invalid; "
              "stop and tell the team.")
        sys.exit(3)
    if stats["aborted"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
