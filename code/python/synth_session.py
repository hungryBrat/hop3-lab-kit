"""SYNTHETIC session data for testing the analysis pipeline. Not robot data.

Writes data/synthetic/{cmd_synth_z040.*, sheet_synth_z040.jpg, session_data_synth.json, truth.json}
and logs/synthetic/{run_synth_z040_<ts>.*, touchoff_<ts>.csv}.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fiducial_sheet as sheet  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "config.json").read_text())
DH = CFG["dh"]
LT = DH["L3"] + DH["f"]
CPR = CFG["encoder"]["counts_per_rev"]
MAP = [CFG["map"][f"theta{i}"] for i in (1, 2, 3)]

NAME = "synth_z040"
CENTRE = np.array([240.0, 0.0, 40.0])
R = 50.0
T_S = 20.0
RATE = CFG["stream"]["rate_hz"]
LAG_S = 0.080
HYST_DEG = 0.5
DROP_FRAC = 0.01
MECH_RADIAL_MM = 2.0
YAW_DEG = 3.0
T_SHEET = np.array([240.0, 2.0, 40.0])
PRINT_SCALE = 0.97
TS_RUN = "20261126_120000"
TS_TOUCH = "20261126_115500"
rng = np.random.default_rng(7)


def fk(th):
    th = np.radians(th)
    r = DH["L2"] * np.cos(th[1]) + LT * np.cos(th[1] + th[2])
    return np.array([np.cos(th[0]) * r, np.sin(th[0]) * r,
                     DH["d1"] + DH["L2"] * np.sin(th[1]) + LT * np.sin(th[1] + th[2])])


def ik_elbow_up(p):
    x, y, z = p
    t1 = np.arctan2(y, x)
    r = np.hypot(x, y)
    zz = z - DH["d1"]
    D = (r ** 2 + zz ** 2 - DH["L2"] ** 2 - LT ** 2) / (2 * DH["L2"] * LT)
    t3 = -np.arccos(np.clip(D, -1, 1))
    t2 = np.arctan2(zz, r) - np.arctan2(LT * np.sin(t3), DH["L2"] + LT * np.cos(t3))
    return np.degrees(np.array([t1, t2, t3]))


def th_to_servo(th):
    return np.array([m["offset"] + m["sign"] * th[i] for i, m in enumerate(MAP)])


def servo_to_th(s):
    return np.array([m["sign"] * (s[i] - m["offset"]) for i, m in enumerate(MAP)])


def to_counts(s):
    return np.round(s * CPR / 360.0).astype(int)


def rotz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def write_csv(path, header, rows, fmts):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fm % v for fm, v in zip(fmts, r)) + "\n")


def make_cmd(dd):
    n = int(T_S * RATE) + 1
    t = np.arange(n) / RATE
    phi = 2 * np.pi * t / T_S
    p_des = CENTRE[:, None] + np.vstack([R * np.cos(phi), R * np.sin(phi), np.zeros(n)])
    th = np.column_stack([ik_elbow_up(p_des[:, k]) for k in range(n)])
    s = th_to_servo(th)
    c = to_counts(s)
    x_dh = fk(th)
    fake_nn = 0.2 * np.vstack([np.sin(phi + 0.3), np.cos(2 * phi), 0.5 * np.sin(3 * phi)])
    x_nn = x_dh + fake_nn
    rows = [[k, t[k], *s[:, k], *c[:, k], *th[:, k], *p_des[:, k], *x_nn[:, k]] for k in range(n)]
    hdr = "k,t_s,s1_deg,s2_deg,s3_deg,c1,c2,c3,th1,th2,th3,x_des,y_des,z_des,x_nn,y_nn,z_nn".split(",")
    write_csv(dd / f"cmd_{NAME}.csv", hdr, rows, ["%d", "%.4f"] + ["%.6f"] * 3 + ["%d"] * 3 + ["%.6f"] * 3 + ["%.4f"] * 6)
    meta = {"name": NAME, "centre": CENTRE.tolist(), "R": R, "T_s": T_S, "rate_hz": RATE, "n_rows": n,
            "map_used": CFG["map"], "model_source": "SYNTHETIC",
            "checks": {"limits": True, "step": True, "zfloor": True, "finite": True},
            "placeholder": False, "synthetic": True, "zc": float(CENTRE[2])}
    (dd / f"cmd_{NAME}.json").write_text(json.dumps(meta, indent=2))
    return t, th, c


def play(u, w):
    y = np.empty_like(u)
    y[0] = u[0]
    for k in range(1, len(u)):
        y[k] = min(max(y[k - 1], u[k] - w), u[k] + w)
    return y


def make_run(ld, t_cmd, th_cmd, c_cmd):
    n = len(t_cmd)
    t_log = t_cmd + rng.normal(0, 0.001, n)
    t_log[0] = 0.0
    t_log = np.maximum.accumulate(t_log)
    th_true = np.vstack([play(np.interp(t_log - LAG_S, t_cmd, th_cmd[j]), HYST_DEG) for j in range(3)])
    pos = to_counts(th_to_servo(th_true))
    ok = np.ones(n, int)
    drop = rng.random(n) < DROP_FRAC
    drop[0] = False
    ok[drop] = 0
    for k in np.flatnonzero(drop):
        pos[:, k] = pos[:, k - 1]
    spd = np.hstack([np.zeros((3, 1)), np.diff(pos, axis=1) * RATE]).astype(int)
    load = np.round(rng.normal(80, 10, (3, n))).astype(int)
    rows = [[t_log[k], k, *c_cmd[:, k], *pos[:, k], *spd[:, k], *load[:, k], ok[k]] for k in range(n)]
    hdr = "t_s,k,goal_c1,goal_c2,goal_c3,pos_c1,pos_c2,pos_c3,spd1,spd2,spd3,load1,load2,load3,comm_ok".split(",")
    stem = f"run_{NAME}_{TS_RUN}"
    write_csv(ld / f"{stem}.csv", hdr, rows, ["%.4f"] + ["%d"] * 14)
    meta = {"cmd_file": f"data/synthetic/cmd_{NAME}.csv", "mode": "stream", "mock": False,
            "config_snapshot": CFG, "offsets_reg31": {"1": 0, "2": 0, "3": 0},
            "synthetic": True, "synthetic_truth": {"lag_s": LAG_S, "hysteresis_halfwidth_deg": HYST_DEG,
                                                  "drop_frac": DROP_FRAC}}
    (ld / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return th_true, stem


def make_touchoff(ld, geo, Rm):
    rows = []
    for pid, (sx, sy) in geo["touch_points"].items():
        p = Rm @ (PRINT_SCALE * np.array([sx, sy, 0.0])) + T_SHEET
        th = ik_elbow_up(p)
        for rep in range(1, 4):
            s = th_to_servo(th + rng.normal(0, 0.05, 3))
            rows.append([pid, rep, *to_counts(s)])
    write_csv(ld / f"touchoff_{TS_TOUCH}.csv", ["point_id", "rep", "pos_c1", "pos_c2", "pos_c3"], rows,
              ["%s", "%d", "%d", "%d", "%d"])
    return f"logs/synthetic/touchoff_{TS_TOUCH}.csv"


def kasa(xy):
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = -(xy ** 2).sum(1)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    c = np.array([-D / 2, -E / 2])
    return c, float(np.sqrt(c @ c - F))


def make_photo(dd, trace_sheet):
    img, (ppx, ppy) = sheet.render_rgb(dpi=300)
    pts = trace_sheet / PRINT_SCALE
    px = np.column_stack([(pts[:, 0] + sheet.W_MM / 2) * ppx, (sheet.H_MM / 2 - pts[:, 1]) * ppy])
    canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.polylines(canvas, [np.round(px * 16).astype(np.int32)], False, (170, 70, 25),
                  thickness=max(1, int(round(0.7 * ppx))), lineType=cv2.LINE_AA, shift=4)
    hw, hh = sheet.W_MM / 2 * PRINT_SCALE, sheet.H_MM / 2 * PRINT_SCALE
    corners_mm = np.float32([[-hw, hh], [hw, hh], [hw, -hh], [-hw, -hh]])
    Wc, Hc = 3000, 4000
    corners_px = np.float32([[330, 260], [2700, 330], [2790, 3720], [250, 3640]])
    H_mm2cam = cv2.getPerspectiveTransform(corners_mm, corners_px)
    A_mm2png = np.array([[ppx / PRINT_SCALE, 0, sheet.W_MM / 2 * ppx],
                         [0, -ppy / PRINT_SCALE, sheet.H_MM / 2 * ppy], [0, 0, 1]])
    H_png2cam = H_mm2cam @ np.linalg.inv(A_mm2png)
    table = np.full((Hc, Wc, 3), (95, 120, 150), np.float32) + rng.normal(0, 6, (Hc, Wc, 1))
    warped = cv2.warpPerspective(canvas, H_png2cam, (Wc, Hc), flags=cv2.INTER_LINEAR).astype(np.float32)
    mask = cv2.warpPerspective(np.ones(canvas.shape[:2], np.float32), H_png2cam, (Wc, Hc))[..., None]
    out = mask * warped + (1 - mask) * table
    yy, xx = np.mgrid[0:Hc, 0:Wc]
    out *= (0.82 + 0.18 * (1 - ((xx - 1800) ** 2 + (yy - 1500) ** 2) / 3.2e7))[..., None]
    out = cv2.GaussianBlur(out, (0, 0), 1.2) + rng.normal(0, 4, out.shape)
    path = dd / f"sheet_{NAME}.jpg"
    cv2.imwrite(str(path), np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path, H_mm2cam


def main():
    dd = ROOT / "data" / "synthetic"
    ld = ROOT / "logs" / "synthetic"
    dd.mkdir(parents=True, exist_ok=True)
    ld.mkdir(parents=True, exist_ok=True)
    geo = sheet.geometry()
    Rm = rotz(YAW_DEG)

    t_cmd, th_cmd, c_cmd = make_cmd(dd)
    th_true, run_stem = make_run(ld, t_cmd, th_cmd, c_cmd)
    touch_rel = make_touchoff(ld, geo, Rm)

    p_meas = fk(th_true)
    d = p_meas[:2] - CENTRE[:2, None]
    pen_robot = p_meas[:2] + MECH_RADIAL_MM * d / np.linalg.norm(d, axis=0)
    pen3 = np.vstack([pen_robot, np.full(pen_robot.shape[1], T_SHEET[2])])
    trace_sheet = (Rm.T @ (pen3 - T_SHEET[:, None]))[:2].T
    photo, H_mm2cam = make_photo(dd, trace_sheet)

    c_true, R_true = kasa(trace_sheet)
    cr_true, Rr_true = kasa(pen_robot.T)
    cm_true, Rm_true = kasa(p_meas[:2].T)
    session = json.loads((ROOT / "session" / "session_data.json").read_text())
    session.update({
        "_comment": "SYNTHETIC session values for test_analysis.m. Not robot data.",
        "date": "2026-11-26 (SYNTHETIC)", "sigma1": MAP[0]["sign"], "sigma3": MAP[2]["sign"],
        "neutral_tip_height_mm": 398.6, "touchoff_log": touch_rel, "fiducials_mm": "data/fiducials.json",
        "paper_plane_z_mm": float(T_SHEET[2]), "pen_offset_mm": 0.0,
        "printed_scale_check_mm": 100.0 * PRINT_SCALE, "chosen_trajectory": NAME, "z_c_mm": float(CENTRE[2]),
        "run_logs": [f"logs/synthetic/{run_stem}.csv"], "sheet_photos": [str(photo.relative_to(ROOT))],
        "notes": "SYNTHETIC", "synthetic": True})
    (dd / "session_data_synth.json").write_text(json.dumps(session, indent=2))
    truth = {
        "synthetic": True, "lag_ms": LAG_S * 1000, "hysteresis_halfwidth_deg": HYST_DEG,
        "mechanics_radial_mm": MECH_RADIAL_MM, "print_scale": PRINT_SCALE,
        "sheet_to_robot": {"yaw_deg": YAW_DEG, "R": Rm.tolist(), "t_mm": T_SHEET.tolist()},
        "plane_z_mm": float(T_SHEET[2]),
        "trace_sheet_fit": {"centre_mm": c_true.tolist(), "R_mm": R_true},
        "pen_robot_fit": {"centre_mm": cr_true.tolist(), "R_mm": Rr_true},
        "dh_meas_fit": {"centre_mm": cm_true.tolist(), "R_mm": Rm_true},
        "H_sheetmm_to_photo": H_mm2cam.tolist(),
        "files": {"cmd": f"data/synthetic/cmd_{NAME}.csv", "run_log": f"logs/synthetic/{run_stem}.csv",
                  "touchoff": touch_rel, "photo": str(photo.relative_to(ROOT)),
                  "session": "data/synthetic/session_data_synth.json"},
    }
    (dd / "truth.json").write_text(json.dumps(truth, indent=2))
    print(f"SYNTHETIC data written: {dd} and {ld}")


if __name__ == "__main__":
    main()
