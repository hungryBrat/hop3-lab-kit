"""SYNTHETIC session data for testing the analysis pipeline. Not robot data.

usage: synth_session.py [letterB|circle|all] [--own-cmd]   (default all; --own-cmd ignores data/cmd_letterB.csv)
letterB: data/synthetic/{cmd_letterB.* (unless data/cmd_letterB.csv exists), session_data_synth.json,
         truth_letterB.json, video_letterB.mp4, video_truth_letterB.csv}, logs/synthetic/run_letterB_<ts>.* x3.
circle (legacy pen path): data/synthetic/{cmd_synth_z040.*, sheet_synth_z040.jpg, session_data_synth_circle.json,
         truth_circle.json}, logs/synthetic/{run_synth_z040_<ts>.*, touchoff_<ts>.csv}.
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


def main_circle():
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
    session["_comment"] = "SYNTHETIC circle/pen session for test_analysis.m (legacy pen path). Not robot data."
    (dd / "session_data_synth_circle.json").write_text(json.dumps(session, indent=2))
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
                  "session": "data/synthetic/session_data_synth_circle.json"},
    }
    (dd / "truth_circle.json").write_text(json.dumps(truth, indent=2))
    print(f"SYNTHETIC data written: {dd} and {ld}")


# ---------------------------------------------------------------- letter B
LB = CFG["trajectories"]["letterB"]
B_NAME = "letterB"
B_TS = ["20261127_100000", "20261127_100300", "20261127_100600"]
B_SEEDS = [101, 102, 103]
B_MECH_YZ = np.array([0.6, -1.375])
B_SYNC_OFFSET_S = -1.733
FPS = 30.0
VID_W, VID_H = 480, 640
CAM_F = 700.0
CAM_C = np.array([float(LB["x_plane_mm"]) + 460.0, 25.0, 175.0])
CAM_PP = np.array([VID_W / 2 + 3.0, VID_H / 2 - 4.0])
DOT_R_MM = 4.0


def b_geometry():
    H = float(LB["H"])
    yc, zc = (float(v) for v in LB["centre_yz"])
    return float(LB["x_plane_mm"]), yc - H / 8, zc, H, H / 4


def b_point(seg, u):
    x, ys, zc, H, R = b_geometry()
    if seg == "spine":
        return np.array([x, ys, zc - H / 2 + H * u])
    cz = zc + H / 4 if seg == "top" else zc - H / 4
    ph = np.pi / 2 - np.pi * u
    return np.array([x, ys + R * np.cos(ph), cz + R * np.sin(ph)])


def nn_forward(th):
    wf = ROOT / "models" / "fk_weights_standin.json"
    if (ROOT / "models" / "fk_weights.json").exists():
        wf = ROOT / "models" / "fk_weights.json"
    if not wf.exists():
        return None, "none"
    W = json.loads(wf.read_text())
    im, om = W["in_map"], W["out_map"]
    a = (th - np.array(im["xoffset"])[:, None]) * np.array(im["gain"])[:, None] + im["ymin"]
    for L in W["layers"]:
        a = np.array(L["W"]) @ a + np.array(L["b"])[:, None]
        if L["act"] == "tansig":
            a = np.tanh(a)
    return (a - om["ymin"]) / np.array(om["gain"])[:, None] + np.array(om["xoffset"])[:, None], W["meta"]["source"]


def make_cmd_b(dd):
    ext = ROOT / "data" / f"cmd_{B_NAME}.csv"
    if ext.exists() and "--own-cmd" not in sys.argv:
        d = np.genfromtxt(ext, delimiter=",", names=True)
        meta = json.loads(ext.with_suffix(".json").read_text())
        th = np.vstack([d["th1"], d["th2"], d["th3"]])
        c = np.vstack([d["c1"], d["c2"], d["c3"]]).astype(int)
        return "data/cmd_letterB.csv", d["t_s"], th, c, meta
    _, _, _, H, R = b_geometry()
    v, nd = float(LB["v_max_mm_s"]), int(round(LB["corner_dwell_s"] * RATE))
    pts, segs = [b_point("spine", 0.0)] * nd, []
    for seg, L in (("spine", H), ("top", np.pi * R), ("bottom", np.pi * R)):
        n = int(np.ceil(1.875 * L / v * RATE))
        tau = np.arange(1, n + 1) / n
        u = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        k0 = 0 if not segs else segs[-1]["k_end"] + 1
        pts += [b_point(seg, ui) for ui in u] + [b_point(seg, 1.0)] * nd
        segs.append({"name": seg, "k_start": k0, "k_end": len(pts) - 1 - (0 if seg == "bottom" else nd)})
    segs[1]["k_start"] = segs[0]["k_end"] + 1
    segs[2]["k_start"] = segs[1]["k_end"] + 1
    p_des = np.array(pts).T
    n = p_des.shape[1]
    t = np.arange(n) / RATE
    th = np.column_stack([ik_elbow_up(p_des[:, k]) for k in range(n)])
    s = th_to_servo(th)
    c = to_counts(s)
    x_nn, src = nn_forward(th)
    if x_nn is None:
        x_nn = fk(th) + 0.2
    rows = [[k, t[k], *s[:, k], *c[:, k], *th[:, k], *p_des[:, k], *x_nn[:, k]] for k in range(n)]
    hdr = "k,t_s,s1_deg,s2_deg,s3_deg,c1,c2,c3,th1,th2,th3,x_des,y_des,z_des,x_nn,y_nn,z_nn".split(",")
    write_csv(dd / f"cmd_{B_NAME}.csv", hdr, rows,
              ["%d", "%.4f"] + ["%.6f"] * 3 + ["%d"] * 3 + ["%.6f"] * 3 + ["%.4f"] * 6)
    ds = np.abs(np.diff(s, axis=1)).max()
    lim = CFG["servo_limits_deg"]
    inlim = all(lim[f"j{j + 1}"][0] + CFG["safety_margin_deg"] <= s[j].min() and
                s[j].max() <= lim[f"j{j + 1}"][1] - CFG["safety_margin_deg"] for j in range(3))
    meta = {"name": B_NAME, "centre": [float(LB["x_plane_mm"]), *map(float, LB["centre_yz"])], "R": R, "H": H,
            "plane": "front", "x_plane_mm": float(LB["x_plane_mm"]), "centre_yz": list(map(float, LB["centre_yz"])),
            "T_s": float(t[-1]), "rate_hz": RATE, "n_rows": n, "map_used": CFG["map"],
            "model_source": f"SYNTHETIC (analytic IK; x_nn = {src} net)", "segments": segs,
            "checks": {"limits": bool(inlim), "step": bool(ds <= CFG["max_step_deg"]),
                       "zfloor": bool(p_des[2].min() >= CFG["z_floor_mm"]), "finite": bool(np.isfinite(th).all())},
            "placeholder": False, "synthetic": True}
    (dd / f"cmd_{B_NAME}.json").write_text(json.dumps(meta, indent=2))
    return f"data/synthetic/cmd_{B_NAME}.csv", t, th, c, meta


def true_joints(t_cmd, th_cmd, t_query):
    tf = np.arange(t_cmd[0] - 1.0, t_cmd[-1] + 3.0, 0.001)
    u = [np.interp(tf - LAG_S, t_cmd, th_cmd[j]) for j in range(3)]
    y = [play(u[j], HYST_DEG) for j in range(3)]
    return np.vstack([np.interp(t_query, tf, y[j]) for j in range(3)])


def make_run_b(ld, cmd_rel, t_cmd, th_cmd, c_cmd, seed, ts):
    g = np.random.default_rng(seed)
    n = len(t_cmd)
    t_log = t_cmd + g.normal(0, 0.001, n)
    t_log[0] = 0.0
    t_log = np.maximum.accumulate(t_log)
    th_true = true_joints(t_cmd, th_cmd, t_log)
    pos = to_counts(th_to_servo(th_true))
    ok = np.ones(n, int)
    drop = g.random(n) < DROP_FRAC
    drop[0] = False
    ok[drop] = 0
    for k in np.flatnonzero(drop):
        pos[:, k] = pos[:, k - 1]
    spd = np.hstack([np.zeros((3, 1)), np.diff(pos, axis=1) * RATE]).astype(int)
    load = np.round(g.normal(80, 10, (3, n))).astype(int)
    rows = [[t_log[k], k, *c_cmd[:, k], *pos[:, k], *spd[:, k], *load[:, k], ok[k]] for k in range(n)]
    hdr = "t_s,k,goal_c1,goal_c2,goal_c3,pos_c1,pos_c2,pos_c3,spd1,spd2,spd3,load1,load2,load3,comm_ok".split(",")
    stem = f"run_{B_NAME}_{ts}"
    write_csv(ld / f"{stem}.csv", hdr, rows, ["%.4f"] + ["%d"] * 14)
    meta = {"cmd_file": cmd_rel, "mode": "stream", "mock": False, "config_snapshot": CFG,
            "offsets_reg31": {"1": 2, "2": -5, "3": 3}, "synthetic": True,
            "synthetic_truth": {"lag_s": LAG_S, "backlash_halfwidth_deg": HYST_DEG, "drop_frac": DROP_FRAC,
                                "seed": seed, "n_dropped": int(drop.sum())}}
    (ld / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return f"logs/synthetic/{stem}.csv"


def camera():
    target = np.array([float(LB["x_plane_mm"]), float(LB["centre_yz"][0]), float(LB["centre_yz"][1])])
    fz = (target - CAM_C) / np.linalg.norm(target - CAM_C)
    fx = np.cross(fz, [0, 0, 1.0])
    fx /= np.linalg.norm(fx)
    Rc = np.vstack([fx, np.cross(fz, fx), fz])
    K = np.array([[CAM_F, 0, CAM_PP[0]], [0, CAM_F, CAM_PP[1]], [0, 0, 1]])
    return K, Rc


def project(K, Rc, P):
    pc = Rc @ (np.atleast_2d(P).T.reshape(3, -1) - CAM_C[:, None])
    uv = K @ pc
    return (uv[:2] / uv[2]).T, pc[2]


def elbow(th):
    t = np.radians(th)
    r = DH["L2"] * np.cos(t[1])
    return np.array([np.cos(t[0]) * r, np.sin(t[0]) * r, DH["d1"] + DH["L2"] * np.sin(t[1])])


def make_video_b(dd, t_cmd, th_cmd):
    import make_video_board as vb
    g = np.random.default_rng(11)
    K, Rc = camera()
    geo = vb.geometry()
    xb = geo["placement"]["plane_x_mm"] + geo["placement"]["board_offset_x_mm"]
    yb, zb = geo["placement"]["centre_yz_mm"]
    raster, (ppx, ppy) = vb.render_rgb(dpi=100)
    raster = cv2.GaussianBlur(raster, (0, 0), 0.8)
    A_px2uv = np.array([[1 / ppx, 0, 0.5 / ppx - vb.W_MM / 2], [0, -1 / ppy, vb.H_MM / 2 - 0.5 / ppy], [0, 0, 1]])
    M_uv2world = np.array([[0, 0, xb], [1, 0, yb], [0, 1, zb]])
    H_px2img = K @ Rc @ (M_uv2world - np.outer(CAM_C, [0, 0, 1])) @ A_px2uv
    H_px2img /= H_px2img[2, 2]
    alpha = np.ones(raster.shape[:2], np.float32)
    w = geo["window_mm"]
    c0, c1 = int((w["u"][0] + vb.W_MM / 2) * ppx), int((w["u"][1] + vb.W_MM / 2) * ppx)
    r0, r1 = int((vb.H_MM / 2 - w["v"][1]) * ppy), int((vb.H_MM / 2 - w["v"][0]) * ppy)
    alpha[r0:r1, c0:c1] = 0
    bw = cv2.warpPerspective(raster, H_px2img, (VID_W, VID_H)).astype(np.float32)
    aw = cv2.warpPerspective(alpha, H_px2img, (VID_W, VID_H))[..., None]
    yy, xx = np.mgrid[0:VID_H, 0:VID_W]
    bg = np.dstack([70 + 25 * yy / VID_H, 75 + 20 * xx / VID_W, 85 + 0 * xx]).astype(np.float32)

    t_end = t_cmd[-1] + 1.0
    n_fr = int(np.ceil((t_end - B_SYNC_OFFSET_S) * FPS))
    t_vid = np.arange(n_fr) / FPS
    t_run = t_vid + B_SYNC_OFFSET_S
    th = true_joints(t_cmd, th_cmd, t_run)
    tip = fk(th)
    dot = tip + np.vstack([np.zeros(n_fr), np.tile(B_MECH_YZ[:, None], (1, n_fr))])
    path = dd / f"phone_{B_NAME}.mp4"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VID_W, VID_H))
    for i in range(n_fr):
        fr = bg + g.normal(0, 2.0, (VID_H, VID_W, 1))
        (pt, _), (pe, _) = project(K, Rc, tip[:, i]), project(K, Rc, elbow(th[:, i]))
        cv2.line(fr, tuple(np.round(pt[0]).astype(int)), tuple(np.round(pe[0]).astype(int)),
                 (55, 55, 60), 14, cv2.LINE_AA)
        fr = aw * bw + (1 - aw) * fr
        pd, depth = project(K, Rc, dot[:, i])
        r = CAM_F * DOT_R_MM / depth[0]
        cv2.circle(fr, tuple(np.round(pd[0] * 16).astype(int)), int(round(r * 16)), (20, 120, 245), -1,
                   cv2.LINE_AA, shift=4)
        fr = fr * (1 + 0.03 * np.sin(0.7 * t_vid[i]))
        vw.write(np.clip(cv2.GaussianBlur(fr, (0, 0), 0.6), 0, 255).astype(np.uint8))
    vw.release()
    write_csv(dd / f"video_truth_{B_NAME}.csv", ["frame", "t_video_s", "t_run_s", "x_mm", "y_mm", "z_mm"],
              [[i, t_vid[i], t_run[i], *dot[:, i]] for i in range(n_fr)], ["%d", "%.5f", "%.5f"] + ["%.4f"] * 3)
    return path, K, Rc, n_fr


def main_letterb():
    dd = ROOT / "data" / "synthetic"
    ld = ROOT / "logs" / "synthetic"
    dd.mkdir(parents=True, exist_ok=True)
    ld.mkdir(parents=True, exist_ok=True)
    cmd_rel, t_cmd, th_cmd, c_cmd, cmeta = make_cmd_b(dd)
    logs = [make_run_b(ld, cmd_rel, t_cmd, th_cmd, c_cmd, sd, ts) for sd, ts in zip(B_SEEDS, B_TS)]
    vpath, K, Rc, n_fr = make_video_b(dd, t_cmd, th_cmd)
    vrel = str(vpath.relative_to(ROOT))
    _, ys, zc, H, R = b_geometry()
    session = json.loads((ROOT / "session" / "session_data.json").read_text())
    session.update({
        "_comment": "SYNTHETIC session values (letter B). Treated as ground truth until the real session exists.",
        "date": "2026-11-27 (SYNTHETIC)", "port": "/dev/ttyACM0", "usb_chip_lsusb": "1a86:55d3 QinHeng CH343 (SYNTHETIC)",
        "psu_voltage_V": 7.4, "servo_offsets_reg31": {"1": 2, "2": -5, "3": 3, "4": None},
        "sigma1": 1, "sigma3": -1, "neutral_tip_height_mm": 396.8,
        "chosen_trajectory": B_NAME, "z_c_mm": zc, "run_logs": logs, "sheet_photos": [],
        "video": vrel, "video_run": logs[0], "video_board": "data/video_board.json", "video_colour": "orange",
        "use_pen": False, "notes": "SYNTHETIC", "synthetic": True})
    (dd / "session_data_synth.json").write_text(json.dumps(session, indent=2))
    truth = {
        "synthetic": True, "lag_ms": LAG_S * 1000, "backlash_halfwidth_deg": HYST_DEG, "drop_frac": DROP_FRAC,
        "quantisation_counts": 1, "seeds": B_SEEDS, "mechanics_offset_yz_mm": B_MECH_YZ.tolist(),
        "mechanics_mm": float(np.linalg.norm(B_MECH_YZ)), "sync_offset_s": B_SYNC_OFFSET_S,
        "sync_convention": "t_run = t_video + sync_offset_s", "fps": FPS, "n_frames": n_fr,
        "camera": {"K": K.tolist(), "R_world_to_cam": Rc.tolist(), "centre_mm": CAM_C.tolist()},
        "B": {"x_plane_mm": b_geometry()[0], "spine_y_mm": ys, "z_c_mm": zc, "H_mm": H, "R_mm": R,
              "spine_length_mm": H},
        "files": {"cmd": cmd_rel, "run_logs": logs, "video": vrel, "video_truth": f"data/synthetic/video_truth_{B_NAME}.csv",
                  "session": "data/synthetic/session_data_synth.json", "board": "data/video_board.json"},
    }
    (dd / "truth_letterB.json").write_text(json.dumps(truth, indent=2))
    print(f"SYNTHETIC letter-B data written: {cmd_rel}, {len(logs)} run logs, {vrel} ({n_fr} frames)")


def main():
    mode = next((a for a in sys.argv[1:] if not a.startswith("--")), "all")
    if mode in ("circle", "all"):
        main_circle()
    if mode in ("letterB", "all"):
        main_letterb()


if __name__ == "__main__":
    main()
