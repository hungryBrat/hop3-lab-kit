"""Pre-motion validation of command CSVs and the per-write goal check.

python safety.py data/cmd_x.csv [...] [--z-floor Z]
python safety.py --selftest
"""
import argparse
import sys

import numpy as np

import hop3cfg as hc


class SafetyError(Exception):
    pass


def bounds(cfg, margin=True):
    m = cfg["safety_margin_deg"] if margin else 0.0
    lim = cfg["servo_limits_deg"]
    lo = np.array([lim[j][0] + m for j in hc.JOINTS], float)
    hi = np.array([lim[j][1] - m for j in hc.JOINTS], float)
    return lo, hi


def count_tol(cfg):
    return 360.0 / hc.counts_per_rev(cfg)


def check_goal(s_deg, cfg, prev_s=None, max_step=None, z_floor=None):
    """Raise SafetyError unless the 3-joint servo-degree goal is safe to send."""
    s = np.asarray(s_deg, float)
    if s.shape != (3,):
        raise SafetyError(f"goal must have exactly 3 joints, got shape {s.shape}")
    if not np.all(np.isfinite(s)):
        raise SafetyError(f"non-finite goal {s}")
    lo, hi = bounds(cfg)
    bad = (s < lo - 1e-9) | (s > hi + 1e-9)
    if bad.any():
        j = int(np.argmax(bad))
        raise SafetyError(f"j{j + 1} goal {s[j]:.2f} deg outside [{lo[j]:.1f}, {hi[j]:.1f}] (limits - margin)")
    zf = cfg["z_floor_mm"] if z_floor is None else z_floor
    z = float(hc.fk_servo(s, cfg)[2])
    if z < zf:
        raise SafetyError(f"tip z {z:.1f} mm below floor {zf:.1f} mm for goal {np.round(s, 2)}")
    if prev_s is not None and max_step is not None:
        step = np.max(np.abs(s - np.asarray(prev_s, float)))
        if step > max_step + count_tol(cfg) + 1e-9:
            raise SafetyError(f"joint step {step:.2f} deg exceeds max_step {max_step} deg")


def validate_arrays(header, arr, cfg, z_floor=None, rate_hz=None):
    """Return (errors, checks). checks mirrors the cmd meta json keys plus format/map/rate."""
    errors = []
    checks = {"format": True, "finite": True, "limits": True, "step": True, "zfloor": True,
              "map": True, "rate": True}

    def fail(key, msg):
        checks[key] = False
        errors.append(f"[{key}] {msg}")

    grip = [h for h in header if "4" in h or "grip" in h.lower()]
    if grip:
        fail("format", f"gripper column(s) {grip} present: the stream must never command the gripper")
    if header != hc.CMD_HEADER:
        fail("format", f"header != CONTRACTS header\n  got:  {','.join(header)}\n  want: {','.join(hc.CMD_HEADER)}")
        return errors, checks
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != len(hc.CMD_HEADER):
        fail("format", f"bad shape {arr.shape}")
        return errors, checks

    bad_rows = np.where(~np.all(np.isfinite(arr), axis=1))[0]
    if bad_rows.size:
        fail("finite", f"{bad_rows.size} row(s) with NaN/Inf, first at row k={bad_rows[0]}")
        return errors, checks

    c = lambda n: hc.col(header, arr, n)
    k = c("k")
    if not np.all(np.diff(k) == 1):
        fail("format", "k is not consecutive")
    rate = rate_hz or cfg["stream"]["rate_hz"]
    dt = np.diff(c("t_s"))
    if np.any(np.abs(dt - 1.0 / rate) > 0.02 / rate + 1e-4):
        fail("rate", f"t_s spacing {np.median(dt):.4f}s (min {dt.min():.4f}, max {dt.max():.4f}) "
                     f"does not match stream rate {rate} Hz: rows would play at the wrong speed")

    s = np.column_stack([c("s1_deg"), c("s2_deg"), c("s3_deg")])
    cnt = np.column_stack([c("c1"), c("c2"), c("c3")])
    th = np.column_stack([c("th1"), c("th2"), c("th3")])
    s_sent = hc.counts_to_deg(cnt, cfg)

    if np.any(np.abs(cnt - np.round(s * hc.counts_per_rev(cfg) / 360.0)) > 1):
        fail("map", "count columns disagree with s columns (c != round(s*4096/360))")
    th_err = np.abs(hc.s_to_theta(s, cfg) - th)
    if np.max(th_err) > 0.05:
        r = int(np.argmax(np.max(th_err, axis=1)))
        fail("map", f"th columns inconsistent with s under the CURRENT config map (row {r}, "
                    f"err {np.max(th_err):.2f} deg): the file was made with another map -> run retarget_cmds.py")

    lo, hi = bounds(cfg)
    for name, v in (("s", s), ("counts", s_sent)):
        bad = (v < lo - 1e-9) | (v > hi + 1e-9)
        if bad.any():
            r, j = np.argwhere(bad)[0]
            fail("limits", f"{bad.any(axis=1).sum()} row(s) outside limits-margin ({name}); first row {r} "
                           f"j{j + 1}={v[r, j]:.2f} not in [{lo[j]:.1f}, {hi[j]:.1f}]")
            break

    max_step = cfg["max_step_deg"]
    for name, v, tol in (("s", s, 1e-9), ("counts", s_sent, count_tol(cfg) + 1e-9)):
        step = np.max(np.abs(np.diff(v, axis=0)), axis=1)
        if np.any(step > max_step + tol):
            r = int(np.argmax(step))
            fail("step", f"joint step {step[r]:.2f} deg between rows {r}->{r + 1} exceeds {max_step} deg ({name})")
            break

    zf = cfg["z_floor_mm"] if z_floor is None else z_floor
    z = hc.fk_servo(s_sent, cfg)[:, 2]
    if np.any(z < zf):
        r = int(np.argmin(z))
        fail("zfloor", f"{(z < zf).sum()} row(s) with DH tip z below {zf} mm; min {z[r]:.1f} mm at row {r}")
    return errors, checks


def validate_file(path, cfg, z_floor=None):
    try:
        header, arr = hc.read_cmd(path)
    except Exception as e:
        return [f"[format] cannot read {path}: {e}"], {"format": False}
    return validate_arrays(header, arr, cfg, z_floor)


def _circle_cmd(cfg, centre=(240.0, 0.0, 40.0), R=50.0, T=4.0, rate=50):
    n = int(round(T * rate)) + 1
    t = np.arange(n) / rate
    ph = 2 * np.pi * t / T
    p = np.column_stack([centre[0] + R * np.cos(ph), centre[1] + R * np.sin(ph), np.full(n, centre[2])])
    return build_cmd(t, p, cfg)


def build_cmd(t, p_des, cfg):
    th = hc.ik_elbow_up(p_des, cfg)
    s = hc.theta_to_s(th, cfg)
    cnt = hc.deg_to_counts(s, cfg)
    p_nn = hc.fk_dh(th, cfg)
    return np.column_stack([np.arange(len(t)), t, s, cnt, th, p_des, p_nn])


def _set_s(cfg, arr, rows, j, value=None, add=None):
    a = arr.copy()
    sc = 2 + j
    if value is not None:
        a[rows, sc] = value
    if add is not None:
        a[rows, sc] += add
    a[:, 5 + j] = hc.deg_to_counts(a[:, sc], cfg)
    a[:, 8:11] = hc.s_to_theta(a[:, 2:5], cfg)
    return a


def selftest(cfg):
    ok = True

    def expect(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok &= bool(cond)

    good = _circle_cmd(cfg)
    e, ch = validate_arrays(hc.CMD_HEADER, good, cfg)
    expect("clean circle accepted", not e)

    e, ch = validate_arrays(hc.CMD_HEADER, _set_s(cfg, good, [10], 2, value=320.0), cfg)
    expect("out-of-limit row rejected (limits)", not ch["limits"])
    e, ch = validate_arrays(hc.CMD_HEADER, _set_s(cfg, good, slice(20, None), 0, add=10.0), cfg)
    expect("10 deg jump rejected (step)", not ch["step"])
    low = _circle_cmd(cfg, centre=(240.0, 0.0, cfg["z_floor_mm"] - 15.0))
    e, ch = validate_arrays(hc.CMD_HEADER, low, cfg)
    expect("z < floor circle rejected (zfloor)", not ch["zfloor"] and ch["limits"] and ch["step"])
    nan = good.copy()
    nan[3, 3] = np.nan
    e, ch = validate_arrays(hc.CMD_HEADER, nan, cfg)
    expect("NaN row rejected (finite)", not ch["finite"])
    e, ch = validate_arrays(hc.CMD_HEADER + ["s4_deg"], np.column_stack([good, np.full(len(good), 180)]), cfg)
    expect("gripper column rejected (format)", not ch["format"])
    flipped = dict(cfg, map=dict(cfg["map"], theta3={"sign": -cfg["map"]["theta3"]["sign"],
                                                     "offset": cfg["map"]["theta3"]["offset"]}))
    e, ch = validate_arrays(hc.CMD_HEADER, good, flipped)
    expect("file made under another map rejected (map)", not ch["map"])
    fast = good.copy()
    fast[:, 1] *= 0.5
    e, ch = validate_arrays(hc.CMD_HEADER, fast, cfg)
    expect("wrong t_s spacing rejected (rate)", not ch["rate"])

    def raises(fn):
        try:
            fn()
        except SafetyError:
            return True
        return False

    s0 = good[0, 2:5]
    expect("check_goal accepts clean row", not raises(lambda: check_goal(s0, cfg, s0, cfg["max_step_deg"])))
    expect("check_goal rejects out-of-limit", raises(lambda: check_goal([s0[0], s0[1], 320.0], cfg)))
    expect("check_goal rejects 10 deg jump",
           raises(lambda: check_goal(s0 + [10, 0, 0], cfg, s0, cfg["max_step_deg"])))
    expect("check_goal rejects z < floor", raises(lambda: check_goal(low[0, 2:5], cfg)))
    expect("check_goal rejects NaN", raises(lambda: check_goal([np.nan, 180, 180], cfg)))
    expect("check_goal rejects 4-joint goal (gripper)", raises(lambda: check_goal([90, 180, 180, 180], cfg)))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--z-floor", type=float, help="override config z_floor_mm")
    args = ap.parse_args()
    cfg = hc.load_config()
    if args.selftest:
        print("safety self-test")
        ok = selftest(cfg)
        print("SELFTEST", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    if not args.files:
        ap.error("give command file(s) or --selftest")
    all_ok = True
    zf = cfg["z_floor_mm"] if args.z_floor is None else args.z_floor
    for f in args.files:
        errors, _ = validate_file(f, cfg, args.z_floor)
        print(f"{'PASS' if not errors else 'FAIL'}  {f}  (z_floor {zf} mm, margin {cfg['safety_margin_deg']} deg, "
              f"max_step {cfg['max_step_deg']} deg)")
        for e in errors:
            print("   ", e)
        all_ok &= not errors
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
