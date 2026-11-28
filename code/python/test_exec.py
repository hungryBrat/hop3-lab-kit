"""Automated tests for the execution stack (mock bus only). python test_exec.py [--quick]

Generates data/test/cmd_test_letterB.csv (analytic IK, front plane x=260) and data/test/cmd_test_circle.csv, then
runs every tool in a temp sandbox root (HOP3_ROOT) so the real config.json / session_data.json / logs are never
touched. Every mock write is audited: the dead gripper (ID 4) must never be written.
The offscreen dashboard snapshot of the B run is copied to figures/dashboard_snapshot.png.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hop3cfg as hc  # noqa: E402
import safety  # noqa: E402

PY = sys.executable
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def gen_test_cmd(root, cfg, name="test_circle", centre=(240.0, 0.0, 40.0), R=50.0):
    rate, T = cfg["stream"]["rate_hz"], 20.0
    n = int(T * rate) + 1
    t = np.arange(n) / rate
    ph = 2 * np.pi * t / T
    p = np.column_stack([centre[0] + R * np.cos(ph), centre[1] + R * np.sin(ph), np.full(n, centre[2])])
    arr = safety.build_cmd(t, p, cfg)
    d = root / "data" / "test"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"cmd_{name}.csv"
    hc.write_cmd(path, arr)
    errors, checks = safety.validate_arrays(hc.CMD_HEADER, arr, cfg)
    meta = {"name": name, "centre": list(centre), "R": R, "T_s": T, "rate_hz": rate, "n_rows": n,
            "map_used": cfg["map"], "model_source": "DH_STANDIN_TEST (x_nn = DH FK)",
            "checks": {k: checks[k] for k in ("limits", "step", "zfloor", "finite")}, "placeholder": True}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def min_jerk(n_steps):
    tau = np.arange(1, n_steps + 1) / n_steps
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def letter_b_path(cfg, x_plane=260.0, zc=150.0, H=100.0, v_max=15.0, dwell=0.4, yc=0.0):
    """-> t, p (N,3), segments. Spine up at y=yc-H/8, top then bottom semicircle (radius H/4) toward +y.
    Per-segment min-jerk (peak speed 1.875 L/T = v_max, T rounded up to the 50 Hz grid), dwell after each."""
    rate = cfg["stream"]["rate_hz"]
    r, ys = H / 4, yc - H / 8
    z0, z1 = zc - H / 2, zc + H / 2

    def spine(u):
        return np.column_stack([np.full_like(u, ys), z0 + (z1 - z0) * u])

    def bump(zcen):
        return lambda u: np.column_stack([ys + r * np.cos(np.pi / 2 - np.pi * u), zcen + r * np.sin(np.pi / 2 - np.pi * u)])

    strokes = [("spine_up", H, spine), ("top_bump", np.pi * r, bump(z1 - r)), ("bottom_bump", np.pi * r, bump(z0 + r))]
    yz = [spine(np.zeros(1))]
    segs, k = [], 0
    n_dwell = int(round(dwell * rate))
    for nm, L, f in strokes:
        n = int(np.ceil(1.875 * L / v_max * rate - 1e-9))
        pts = f(min_jerk(n))
        segs.append({"name": nm, "k_start": k, "k_end": k + n})
        yz += [pts, np.repeat(pts[-1:], n_dwell, axis=0)]
        k += n + n_dwell
    yz = np.vstack(yz)
    t = np.arange(len(yz)) / rate
    p = np.column_stack([np.full(len(yz), x_plane), yz])
    return t, p, segs


def gen_letter_b_cmd(root, cfg, name="test_letterB", **kw):
    t, p, segs = letter_b_path(cfg, **kw)
    arr = safety.build_cmd(t, p, cfg)
    d = root / "data" / "test"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"cmd_{name}.csv"
    hc.write_cmd(path, arr)
    errors, checks = safety.validate_arrays(hc.CMD_HEADER, arr, cfg)
    lb = {"x_plane": 260.0, "zc": 150.0, "H": 100.0, "yc": 0.0, **kw}
    meta = {"name": name, "plane": "front", "x_plane_mm": lb["x_plane"], "centre_yz": [lb["yc"], lb["zc"]],
            "H": lb["H"], "centre": [lb["x_plane"], lb["yc"], lb["zc"]], "R": lb["H"] / 4, "T_s": float(t[-1]),
            "rate_hz": cfg["stream"]["rate_hz"], "n_rows": len(arr), "map_used": cfg["map"],
            "model_source": "DH_STANDIN_TEST (x_nn = DH FK)",
            "checks": {k: checks[k] for k in ("limits", "step", "zfloor", "finite")}, "placeholder": True,
            "segments": segs, "test_file": True}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path, errors


def run(args, env, timeout=300):
    r = subprocess.run([PY, *args], cwd=HERE, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def latest(d, pattern):
    files = sorted(d.glob(pattern))
    return files[-1] if files else None


def read_log(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], np.array(rows[1:], dtype=float)


def test_bus_guards(cfg):
    from mock_bus import MockBus
    from bus import assert_safe_register
    b = MockBus(cfg, seed=0).open()
    s0 = np.array(hc.NEUTRAL_S)

    def raises(fn, exc):
        try:
            fn()
        except exc:
            return True
        return False

    c0 = dict(zip(b.ids, hc.deg_to_counts(s0, cfg).tolist()))
    check("bus: speed 0 refused", raises(lambda: b.sync_write(c0, 0, 50), AssertionError))
    check("bus: acc 0 refused", raises(lambda: b.sync_write(c0, 1000, 0), AssertionError))
    check("bus: reg40=128 refused", raises(lambda: assert_safe_register(40, 128), AssertionError))
    check("bus: _write1(40,128) refused", raises(lambda: b._write1(1, 40, 128), AssertionError))
    check("bus: EEPROM reg 31 write refused", raises(lambda: b._write2(1, 31, 0), AssertionError))
    check("bus: set_torque needs bool", raises(lambda: b.set_torque(1, 128), AssertionError))
    g4 = {**c0, b.gid: 2048}
    check("bus: gripper in sync_write refused", raises(lambda: b.sync_write(g4, 1000, 50), safety.SafetyError))
    bad = {**c0, b.ids[2]: int(hc.deg_to_counts(320.0, cfg))}
    check("bus: out-of-limit goal refused", raises(lambda: b.sync_write(bad, 1000, 50), safety.SafetyError))
    b.sync_write(c0, 1000, 50)
    jump = {**c0, b.ids[0]: int(hc.deg_to_counts(100.0, cfg))}
    check("bus: 10 deg jump refused when streaming",
          raises(lambda: b.sync_write(jump, 1000, 50, cfg["max_step_deg"]), safety.SafetyError))
    b.set_torque(1, False)
    b.set_torque(1, True)
    check("bus: only 0/1 ever written to reg 40", all(v in (0, 1) for _, a, v in b.writes if a == 40))
    gid = b.gid
    check("bus: gripper disabled in config", not hc.gripper_enabled(cfg) and b.ids == [1, 2, 3])
    check("bus: ping_all/read_offsets default to ids 1-3", sorted(b.ping_all()) == [1, 2, 3]
          and sorted(b.read_offsets()) == [1, 2, 3])
    check("mock: dead gripper does not answer ping", b.ping_all([gid])[gid] is None)
    check("bus: hold_gripper refused", raises(lambda: b.hold_gripper(), safety.SafetyError))
    check("bus: torque write to gripper refused", raises(lambda: b.set_torque(gid, True), safety.SafetyError))
    check("bus: torque-limit write to gripper refused", raises(lambda: b.set_torque_limit(gid, 300), safety.SafetyError))
    check("bus: raw goal write to gripper refused", raises(lambda: b._write_goals({gid: 2048}, 200, 20), safety.SafetyError))
    check("bus: raw reg-40 write to gripper refused", raises(lambda: b._write1(gid, 40, 0), safety.SafetyError))
    b.torque_on_safely()
    check("mock: no write ever recorded for the gripper", b.writes and all(w[0] != gid for w in b.writes),
          f"{len(b.writes)} writes, ids {sorted({w[0] for w in b.writes})}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="skip the two 20-60 s mock runs")
    args = ap.parse_args()
    cfg = hc.load_config()
    print("== safety self-test")
    rc, out = run(["safety.py", "--selftest"], dict(os.environ))
    check("safety.py --selftest", rc == 0 and "SELFTEST PASS" in out, out.strip().splitlines()[-1])

    print("== bus guards")
    test_bus_guards(cfg)

    print("== test command files")
    cmd = gen_test_cmd(hc.ROOT, cfg)
    rc, out = run(["safety.py", str(cmd)], dict(os.environ))
    check(f"generated {hc.rel(cmd)} passes safety", rc == 0, out.strip().splitlines()[0])
    bcmd, berr = gen_letter_b_cmd(hc.ROOT, cfg)
    rc, out = run(["safety.py", str(bcmd)], dict(os.environ))
    check(f"generated {hc.rel(bcmd)} passes safety", rc == 0 and not berr, out.strip().splitlines()[0])
    bh, ba = hc.read_cmd(bcmd)
    bmeta = json.loads(bcmd.with_suffix(".json").read_text())
    p_dh = hc.fk_servo(hc.counts_to_deg(ba[:, 5:8], cfg), cfg)
    p_des = ba[:, 11:14]
    check("letter B: header per CONTRACTS, 3 segments in meta", bh == hc.CMD_HEADER
          and [sg["name"] for sg in bmeta["segments"]] == ["spine_up", "top_bump", "bottom_bump"])
    check("letter B: plane x=260, y in [-12.5, 12.5], z in [100, 200]",
          np.allclose(p_des[:, 0], 260) and np.allclose([p_des[:, 1].min(), p_des[:, 1].max(), p_des[:, 2].min(),
                                                  p_des[:, 2].max()], [-12.5, 12.5, 100, 200], atol=0.01))
    spd = np.linalg.norm(np.diff(p_des, axis=0), axis=1) * cfg["stream"]["rate_hz"]
    check("letter B: peak speed <= 15 mm/s, closed path", spd.max() <= 15.0 + 1e-6
          and np.linalg.norm(p_des[-1] - p_des[0]) < 1e-9, f"v_max {spd.max():.2f} mm/s")
    check("letter B: DH(counts) within 1 mm of desired (IK + quantisation)",
          np.abs(p_dh - p_des).max() < 1.0, f"max {np.abs(p_dh - p_des).max():.3f} mm")

    tmp = Path(tempfile.mkdtemp(prefix="hop3_test_"))
    for rel in ("config.json", "session/session_data.json", "data/fiducials.json",
                "data/test/cmd_test_circle.csv", "data/test/cmd_test_circle.json",
                "data/test/cmd_test_letterB.csv", "data/test/cmd_test_letterB.json"):
        src = hc.ROOT / rel
        if src.exists():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tmp / rel)
    (tmp / "figures").mkdir()
    writelog = tmp / "mock_writes.jsonl"
    env = dict(os.environ, HOP3_ROOT=str(tmp), HOP3_MOCK_WRITELOG=str(writelog))
    ccmd = tmp / "data" / "test" / "cmd_test_circle.csv"
    tcmd = tmp / "data" / "test" / "cmd_test_letterB.csv"
    n_rows = len(hc.read_cmd(tcmd)[1])
    mlogs = tmp / "logs" / "mock"
    real_cfg_text = (tmp / "config.json").read_text()
    real_sess_text = (tmp / "session" / "session_data.json").read_text()
    print(f"   sandbox {tmp}")

    if not args.quick:
        print("== mock stream run (letter B)")
        rc, out = run(["run_traj.py", str(tcmd), "--mode", "stream", "--mock", "--seed", "1"], env)
        log = latest(mlogs, "run_test_letterB_*.csv")
        check("stream run exit 0", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
        check("segment progress printed for all 3 segments",
              all(f"segment {j}/3 {nm}" in out for j, nm in ((1, "spine_up"), (2, "top_bump"), (3, "bottom_bump"))))
        check("run_traj reports gripper dead/ignored", "gripper: dead/ignored" in out)
        if log:
            hdr, a = read_log(log)
            meta = json.loads(log.with_suffix(".json").read_text())
            check("log header per CONTRACTS", hdr == hc.RUN_HEADER)
            check("all rows logged", len(a) == n_rows and np.array_equal(a[:, 1], np.arange(n_rows)),
                  f"{len(a)}/{n_rows}")
            rate = meta["achieved_rate_hz"] or 0
            check("achieved rate >= 45 Hz", rate >= 45, f"{rate:.2f} Hz, dropped {meta['dropped_replies']}")
            check("offsets unchanged + mock flag", meta["offsets_unchanged"] and meta["mock"] is True)
            check("run meta: written_ids == [1,2,3], segments carried", meta.get("written_ids") == [1, 2, 3]
                  and len(meta.get("segments") or []) == 3, f"written_ids {meta.get('written_ids')}")
            check("logged goals == cmd counts",
                  np.array_equal(a[:, 2:5], hc.read_cmd(tcmd)[1][:, 5:8]))
            ok = a[:, 14] == 1
            err = np.abs(a[ok, 5:8] - a[ok, 2:5]).max()
            check("mock tracking within 40 counts", err < 40, f"max |pos-goal| {err:.0f} counts")
            print(f"   stream: {out.strip().splitlines()[-1]}")
        print("== mock wait run")
        rc, out = run(["run_traj.py", str(tcmd), "--mode", "wait", "--mock", "--seed", "2"], env)
        logs = sorted(mlogs.glob("run_test_letterB_*.csv"))
        check("wait run exit 0", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
        check("wait mode prints segment progress", "segment 3/3 bottom_bump" in out)
        if len(logs) >= 2:
            hdr, a = read_log(logs[-1])
            meta = json.loads(logs[-1].with_suffix(".json").read_text())
            ks = sorted(set(a[:, 1].astype(int)))
            check("wait mode visits ~90 decimated targets incl. last", 85 <= len(ks) <= 100 and ks[-1] == n_rows - 1,
                  f"{len(ks)} targets")
            check("wait mode offsets unchanged, no settle timeouts",
                  meta["offsets_unchanged"] and meta.get("wait_timeouts") == 0, f"timeouts {meta.get('wait_timeouts')}")

        print("== mock stream run with offscreen dashboard (front view)")
        snap = tmp / "figures" / "dashboard_snapshot.png"
        rc, out = run(["run_traj.py", str(tcmd), "--mode", "stream", "--mock", "--seed", "3", "--dashboard",
                       "--offscreen"], env)
        check("dashboard run exit 0 + snapshot written", rc == 0 and snap.exists(),
              out.strip().splitlines()[-1] if out.strip() else "")
        if snap.exists():
            dst = hc.ROOT / "figures" / "dashboard_snapshot.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snap, dst)
            print(f"   snapshot -> {hc.rel(dst)}")

    print("== dry run + unsafe file refused")
    rc, out = run(["run_traj.py", str(tcmd), "--mock", "--dry"], env)
    check("--dry exits 0 with no log", rc == 0 and "DRY" in out)
    h, arr = hc.read_cmd(tcmd)
    arr[500:, 2] += 10
    arr[:, 5] = hc.deg_to_counts(arr[:, 2], cfg)
    arr[:, 8:11] = hc.s_to_theta(arr[:, 2:5], cfg)
    badp = tmp / "data" / "test" / "cmd_bad_jump.csv"
    hc.write_cmd(badp, arr)
    rc, out = run(["run_traj.py", str(badp), "--mock"], env)
    check("run_traj refuses a file with a 10 deg jump", rc == 1 and "SAFETY FAIL" in out)

    print("== retarget round trip (letter B, valid under both J3 signs)")
    rt, _ = gen_letter_b_cmd(tmp, cfg, "rt")
    orig = hc.read_cmd(rt)[1]
    s1b = cfg["map"]["theta1"]["sign"]
    hc.set_config_signs(s1b, 1, tmp / "config.json")
    before = ccmd.read_bytes()
    rc, out = run(["retarget_cmds.py", str(ccmd)], env)
    check("low circle under sigma3=+1 refused, file untouched",
          rc == 1 and "[limits]" in out and ccmd.read_bytes() == before)
    rc, out = run(["safety.py", str(rt)], env)
    check("after sign flip, stale file is rejected ([map])", rc == 1 and "[map]" in out)
    rc, out = run(["retarget_cmds.py", str(rt)], env)
    flipped = hc.read_cmd(rt)[1]
    check("retarget to sigma3=+1 ok + .bak written", rc == 0 and rt.with_name(rt.name + ".bak").exists(),
          out.strip().splitlines()[-1])
    check("s3 recomputed as 180 + th3", np.allclose(flipped[:, 4], 180 + orig[:, 10], atol=1e-5))
    check("meta map_used updated", json.loads(rt.with_suffix(".json").read_text())["map_used"]["theta3"]["sign"] == 1)
    rc, _ = run(["safety.py", str(rt)], env)
    check("retargeted file passes safety", rc == 0)
    hc.set_config_signs(s1b, -1, tmp / "config.json")
    rc, _ = run(["retarget_cmds.py", str(rt)], env)
    back = hc.read_cmd(rt)[1]
    check("round trip restores original s/c", rc == 0 and np.allclose(back[:, 2:8], orig[:, 2:8], atol=1e-5))
    check("set_config_signs restores the file byte-for-byte", (tmp / "config.json").read_text() == real_cfg_text)

    print("== probe --mock")
    rc, out = run(["probe.py", "--mock", "--yes"], env)
    ms = mlogs / "session_data.json"
    check("probe exit 0", rc == 0 and "PROBE OK" in out, out.strip().splitlines()[-1])
    check("probe reports ID 4 dead/ignored without failing", "gripper: dead/ignored" in out)
    sess = json.loads(ms.read_text()) if ms.exists() else {}
    check("probe wrote port + offsets (ids 1-3 only) to mock session",
          sess.get("port") == "mock" and sess.get("servo_offsets_reg31", {}).get("2") == -7
          and sorted(sess.get("servo_offsets_reg31", {})) == ["1", "2", "3"])
    check("probe csv written", latest(mlogs, "probe_*.csv") is not None)

    print("== convention_test --mock (scripted: sigma1=+1, sigma3=+1)")
    answers = "y,395,y,,y,y,y,y,y,n,y,y"
    rc, out = run(["convention_test.py", "--mock", "--answers", answers], env)
    check("convention exit 0", rc == 0, out.strip().splitlines()[-1])
    sess = json.loads(ms.read_text())
    check("sigma1/sigma3/height recorded", sess.get("sigma1") == 1 and sess.get("sigma3") == 1
          and sess.get("neutral_tip_height_mm") == 395.0)
    check("probe fields preserved", sess.get("port") == "mock")
    mc = json.loads((mlogs / "config.json").read_text()) if (mlogs / "config.json").exists() else {}
    check("confirmed sign update lands in logs/mock/config.json", mc.get("map", {}).get("theta3", {}).get("sign") == 1)
    check("real (sandbox) config.json untouched", (tmp / "config.json").read_text() == real_cfg_text)
    check("decision for sigma3=+1 = retarget + letter B (no backup)",
          "DECISION: sigma3 = +1" in out and "cmd_letterB" in out and "no backup" in out)
    rc, out = run(["convention_test.py", "--mock", "--yes"], env)
    check("convention --yes -> sigma3=-1 -> PRIMARY letter B", rc == 0 and "PRIMARY data/cmd_letterB.csv" in out
          and ("already match" in out or s1b != 1))
    rc, out = run(["convention_test.py", "--mock", "--answers", "n"], env)
    check("declining a move stops without motion", rc == 2 and "not confirmed" in out)

    print("== touchoff --mock")
    rc, out = run(["touchoff.py", "--mock", "--yes"], env)
    check("touchoff exit 0", rc == 0, out.strip().splitlines()[-1])
    to = latest(mlogs, "touchoff_*.csv")
    if to:
        with open(to) as f:
            rows = list(csv.reader(f))
        check("touchoff csv header + 4x3 rows", rows[0] == hc.TOUCHOFF_HEADER and len(rows) == 13,
              f"{len(rows) - 1} rows")
        check("touchoff ids from fiducials.json", {r[0] for r in rows[1:]} == {"F1", "F2", "F3", "F4"})
    sess = json.loads(ms.read_text())
    check("touchoff_log recorded in mock session", str(sess.get("touchoff_log", "")).startswith("logs/mock/touchoff_"))
    check("real (sandbox) session_data.json untouched",
          (tmp / "session" / "session_data.json").read_text() == real_sess_text)

    print("== write audit (every mock bus in every subprocess)")
    wl = [json.loads(x) for x in writelog.read_text().splitlines()] if writelog.exists() else []
    ids_w = sorted({w[0] for w in wl})
    check("mock write log non-empty and only ids 1-3 ever written (gripper never)", wl and ids_w == [1, 2, 3],
          f"{len(wl)} writes, ids {ids_w}")
    check("reg 40 only 0/1, no EEPROM register written", all(w[1] in (40, 42, 48) for w in wl)
          and all(w[2] in (0, 1) for w in wl if w[1] == 40))

    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    if n_fail == 0:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"sandbox kept for inspection: {tmp}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
