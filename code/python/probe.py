"""Session minute 0: find the port, ping IDs 1-3, read positions + offsets (reg 31), record them.

The gripper (ID 4) is dead: with gripper_enabled=false it gets one read-only ping, reported "dead/ignored",
and never fails the probe. Nothing is written to any servo.

python probe.py [--mock] [--port P]
Writes session_data.json {port, servo_offsets_reg31} (other fields preserved) and logs/probe_<ts>.csv.
"""
import argparse
import csv
import os
import shutil
import sys

import numpy as np

import hop3cfg as hc
from bus import make_bus

KNOWN_VID = {0x1A86: "WCH (CH340/CH343)", 0x10C4: "SiLabs CP210x", 0x0403: "FTDI"}


def list_ports():
    from serial.tools import list_ports as lp
    ports = sorted(lp.comports(), key=lambda p: p.device)
    for p in ports:
        vid = f"{p.vid:04x}:{p.pid:04x}" if p.vid is not None else "-"
        print(f"  {p.device:16s} {vid:10s} {KNOWN_VID.get(p.vid, ''):20s} {p.description}")
    cands = [p for p in ports if p.device.startswith(("/dev/ttyACM", "/dev/ttyUSB"))]
    cands.sort(key=lambda p: (p.vid not in KNOWN_VID, not p.device.startswith("/dev/ttyACM")))
    return ports, (cands[0].device if cands else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--port")
    hc.add_prompt_args(ap)
    args = ap.parse_args()
    cfg = hc.load_config()

    print("serial ports:")
    ports, suggested = list_ports()
    if not ports:
        print("  (none)")
    print(f"suggested port: {suggested}")
    if not args.mock and not suggested and not args.port:
        if shutil.which("brltty"):
            print("brltty is installed and may be grabbing a CH340 (shows in lsusb, never in /dev):\n"
                  "  sudo systemctl mask brltty-udev.service brltty.service && sudo systemctl stop brltty-udev "
                  "brltty ; then replug")
        hc.die("no /dev/ttyACM* or /dev/ttyUSB* found: check cable/power, `lsusb`, `dmesg | tail`")
    port = "mock" if args.mock else (args.port or suggested)
    if not args.mock and not os.access(port, os.R_OK | os.W_OK):
        hc.die(f"no read/write permission on {port}: `groups` must show dialout (re-login after usermod), "
               f"or quick fix `sudo chmod 666 {port}`")

    bus = make_bus(cfg, mock=args.mock)
    try:
        bus.open(port, cfg["stream"]["baud"])
    except IOError as e:
        hc.die(str(e))
    all_ids = bus.ids + ([bus.gid] if bus.gripper_enabled else [])
    pings = bus.ping_all(all_ids)
    offsets = bus.read_offsets(all_ids)
    grip_ping = None if bus.gripper_enabled else bus.ping_all([bus.gid])[bus.gid]
    rd, ok = {}, False
    for _ in range(10):
        rd, ok = bus.sync_read(all_ids)
        if ok:
            break
    bus.close()

    rows = []
    print(f"\nport {port} @ {cfg['stream']['baud']} baud")
    print(" id  ping   counts    s_deg   theta_deg   reg31")
    s_joint = []
    for i in all_ids:
        r = rd.get(i)
        c = r["pos"] if r else None
        s = float(hc.counts_to_deg(c, cfg)) if c is not None else None
        th = None
        if i in bus.ids and s is not None:
            j = bus.ids.index(i)
            th = float(hc.s_to_theta(np.array([s if k == j else hc.NEUTRAL_S[k] for k in range(3)]), cfg)[j])
            s_joint.append(s)
        print(f" {i:2d}  {'ok ' if pings[i] else 'NO '}  {c if c is not None else '-':>7}  "
              f"{'%8.2f' % s if s is not None else '       -'}  {'%9.2f' % th if th is not None else '        -'}"
              f"   {offsets[i] if offsets[i] is not None else '-'}")
        rows.append([i, pings[i] or "", "" if c is None else c, "" if s is None else f"{s:.3f}",
                     "" if th is None else f"{th:.3f}", "" if offsets[i] is None else offsets[i]])
    if not bus.gripper_enabled:
        state = "dead/ignored" if grip_ping is None else f"answered ping (model {grip_ping}) but ignored"
        print(f" {bus.gid:2d}  gripper: {state} (gripper_enabled=false, never written, not required)")
        rows.append([bus.gid, grip_ping or "", "", "", "", "ignored"])
    if len(s_joint) == 3:
        print(f"DH tip at present pose: {np.round(hc.fk_servo(s_joint, cfg), 1)} mm (config map)")

    ts = hc.timestamp()
    out = hc.logs_dir(args.mock) / f"probe_{ts}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "model", "pos_counts", "s_deg", "theta_deg", "offset_reg31"])
        w.writerows(rows)
    sp = hc.update_session({"port": port, "servo_offsets_reg31": {str(i): offsets[i] for i in all_ids}},
                           mock=args.mock)
    print(f"\nwrote {hc.rel(out)} and {hc.rel(sp)} (port, servo_offsets_reg31)")
    missing = [i for i in all_ids if not pings[i]]
    if missing or not ok:
        print(f"PROBLEM: no ping from {missing}" if missing else "PROBLEM: sync read incomplete")
        print("  -> check servo power (PSU on, voltage matches servos), daisy-chain cables, adapter jumper "
              "in position A, baud 1M; see SESSION_CHEATSHEET.md")
        sys.exit(1)
    if not args.mock and cfg.get("port") != port:
        print(f"NOTE: config.json port is {cfg.get('port')}; pass --port {port} or update config (coordinator)")
    print("PROBE OK")


if __name__ == "__main__":
    main()
