"""Automated tests for the execution stack (mock bus only). python test_exec.py [--quick]

Generates data/test/cmd_test_circle.csv, then runs every tool in a temp sandbox root (HOP3_ROOT) so the real
config.json / session_data.json / logs are never touched.
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

    print("== test command file")
    cmd = gen_test_cmd(hc.ROOT, cfg)
    rc, out = run(["safety.py", str(cmd)], dict(os.environ))
    check(f"generated {hc.rel(cmd)} passes safety", rc == 0, out.strip().splitlines()[0])

    tmp = Path(tempfile.mkdtemp(prefix="hop3_test_"))
    for rel in ("config.json", "session/session_data.json", "data/fiducials.json",
                "data/test/cmd_test_circle.csv", "data/test/cmd_test_circle.json"):
        src = hc.ROOT / rel
        if src.exists():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tmp / rel)
    (tmp / "figures").mkdir()
    env = dict(os.environ, HOP3_ROOT=str(tmp))
    tcmd = tmp / "data" / "test" / "cmd_test_circle.csv"
    n_rows = len(hc.read_cmd(tcmd)[1])
    mlogs = tmp / "logs" / "mock"
    real_cfg_text = (tmp / "config.json").read_text()
    real_sess_text = (tmp / "session" / "session_data.json").read_text()
    print(f"   sandbox {tmp}")

    if not args.quick:
        print("== mock stream run")
        rc, out = run(["run_traj.py", str(tcmd), "--mode", "stream", "--mock", "--seed", "1"], env)
        log = latest(mlogs, "run_test_circle_*.csv")
        check("stream run exit 0", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
        if log:
            hdr, a = read_log(log)
            meta = json.loads(log.with_suffix(".json").read_text())
            check("log header per CONTRACTS", hdr == hc.RUN_HEADER)
            check("all rows logged", len(a) == n_rows and np.array_equal(a[:, 1], np.arange(n_rows)),
                  f"{len(a)}/{n_rows}")
            rate = meta["achieved_rate_hz"] or 0
            check("achieved rate >= 45 Hz", rate >= 45, f"{rate:.2f} Hz, dropped {meta['dropped_replies']}")
            check("offsets unchanged + mock flag", meta["offsets_unchanged"] and meta["mock"] is True)
            check("logged goals == cmd counts",
                  np.array_equal(a[:, 2:5], hc.read_cmd(tcmd)[1][:, 5:8]))
            ok = a[:, 14] == 1
            err = np.abs(a[ok, 5:8] - a[ok, 2:5]).max()
            check("mock tracking within 40 counts", err < 40, f"max |pos-goal| {err:.0f} counts")
            print(f"   stream: {out.strip().splitlines()[-1]}")
        print("== mock wait run")
        rc, out = run(["run_traj.py", str(tcmd), "--mode", "wait", "--mock", "--seed", "2"], env)
        logs = sorted(mlogs.glob("run_test_circle_*.csv"))
        check("wait run exit 0", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
        if len(logs) >= 2:
            hdr, a = read_log(logs[-1])
            meta = json.loads(logs[-1].with_suffix(".json").read_text())
            ks = sorted(set(a[:, 1].astype(int)))
            check("wait mode visits ~90 decimated targets incl. last", 85 <= len(ks) <= 100 and ks[-1] == n_rows - 1,
                  f"{len(ks)} targets")
            check("wait mode offsets unchanged, no settle timeouts",
                  meta["offsets_unchanged"] and meta.get("wait_timeouts") == 0, f"timeouts {meta.get('wait_timeouts')}")

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

    print("== retarget round trip (backup-centre circle, valid under both J3 signs)")
    rt = gen_test_cmd(tmp, cfg, "rt", (270.0, 0.0, 170.0), 40.0)
    orig = hc.read_cmd(rt)[1]
    hc.set_config_signs(1, 1, tmp / "config.json")
    before = tcmd.read_bytes()
    rc, out = run(["retarget_cmds.py", str(tcmd)], env)
    check("primary circle under sigma3=+1 refused, file untouched",
          rc == 1 and "[limits]" in out and tcmd.read_bytes() == before)
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
    hc.set_config_signs(1, -1, tmp / "config.json")
    rc, _ = run(["retarget_cmds.py", str(rt)], env)
    back = hc.read_cmd(rt)[1]
    check("round trip restores original s/c", rc == 0 and np.allclose(back[:, 2:8], orig[:, 2:8], atol=1e-5))
    check("set_config_signs restores the file byte-for-byte", (tmp / "config.json").read_text() == real_cfg_text)

    print("== probe --mock")
    rc, out = run(["probe.py", "--mock", "--yes"], env)
    ms = mlogs / "session_data.json"
    check("probe exit 0", rc == 0 and "PROBE OK" in out, out.strip().splitlines()[-1])
    sess = json.loads(ms.read_text()) if ms.exists() else {}
    check("probe wrote port + offsets to mock session",
          sess.get("port") == "mock" and sess.get("servo_offsets_reg31", {}).get("2") == -7)
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
    check("decision = BACKUP for sigma3=+1", "DECISION: sigma3 = +1" in out and "BACKUP" in out)
    rc, out = run(["convention_test.py", "--mock", "--yes"], env)
    check("convention --yes -> sigma3=-1 -> PRIMARY", rc == 0 and "PRIMARY" in out and "already match" in out)
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

    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    if n_fail == 0:
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"sandbox kept for inspection: {tmp}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
