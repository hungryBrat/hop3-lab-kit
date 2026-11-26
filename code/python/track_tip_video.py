"""Track the coloured tip marker in a phone video against the ArUco board in the letter plane.

usage: track_tip_video.py video.mp4 [--board-json data/video_board.json] [--colour orange|green|pink]
       [--run-log logs/run_<name>_<ts>.csv] [--session session/session_data.json] [--board-offset MM]
       [--fpx PX] [--per-frame] [--debug]
Writes results[/synthetic]/video_<stem>.json {t_s[] (run time), y_mm[], z_mm[], reproj_rms_mm, sync_offset_s, ...}
and figures[/synthetic]/video_<stem>_overlay.png. Sync convention: t_run = t_video + sync_offset_s.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hop3cfg  # noqa: E402

ROOT = hop3cfg.ROOT
HSV = {"orange": [((6, 110, 110), (24, 255, 255))],
       "green": [((40, 80, 70), (85, 255, 255))],
       "pink": [((150, 70, 110), (175, 255, 255))]}
N_STATIC = 30


def absr(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def frame_times(path, n, fps):
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)],
                                 capture_output=True, text=True, check=True).stdout.split()
            t = np.array([float(x.strip(",")) for x in out if x.strip(",")])
            if len(t) >= n:
                return t[:n] - t[0], "ffprobe"
        except (subprocess.CalledProcessError, ValueError):
            pass
    return np.arange(n) / fps, "fps"


def board_points(board):
    ids, obj = [], []
    for k, m in board["markers"].items():
        ids.append(int(k))
        obj.append(np.array(m["corners_mm"], float))
    return dict(zip(ids, obj))


def detect(det, gray, objs):
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None, None
    img, obj = [], []
    for c, i in zip(corners, ids.ravel()):
        if int(i) in objs:
            img.append(c.reshape(4, 2))
            obj.append(objs[int(i)])
    if len(img) < 2:
        return None, None
    return np.vstack(obj), np.vstack(img)


def blob(frame, colour, roi):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], np.uint8)
    for lo, hi in HSV[colour]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask &= roi
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None, mask
    i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    if st[i, cv2.CC_STAT_AREA] < 12:
        return None, mask
    m = cv2.moments((lab == i).astype(np.uint8), binaryImage=True)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]), mask


def img_to_board(H, pts):
    p = cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), np.linalg.inv(H))
    return p.reshape(-1, 2)


def parallax_fix(px, obj, img, K, offset):
    """Intersect the pixel ray with the tip plane, which lies `offset` mm behind the board face (w = -offset)."""
    obj3 = np.column_stack([obj, np.zeros(len(obj))])
    ok, rv, tv = cv2.solvePnP(obj3, img, K, None, flags=cv2.SOLVEPNP_IPPE)
    R = cv2.Rodrigues(rv)[0]
    C = -R.T @ tv.ravel()
    out = np.full((len(px), 2), np.nan)
    for i, p in enumerate(px):
        d = R.T @ np.linalg.solve(K, [p[0], p[1], 1.0])
        lam = (-offset - C[2]) / d[2]
        out[i] = (C + lam * d)[:2]
    return out


def meas_z(run_log, cfg):
    import csv
    with open(run_log) as f:
        rows = [r for r in csv.DictReader(f) if int(r["comm_ok"]) == 1]
    t = np.array([float(r["t_s"]) for r in rows])
    c = np.array([[float(r[f"pos_c{j}"]) for j in (1, 2, 3)] for r in rows])
    p = hop3cfg.fk_dh(hop3cfg.s_to_theta(hop3cfg.counts_to_deg(c, cfg), cfg), cfg)
    return t, p


def sync(tv, zv, tm, zm, dt=0.01):
    g = np.isfinite(zv)
    tv, zv = tv[g], zv[g]
    tgv = np.arange(tv[0], tv[-1], dt)
    a = np.interp(tgv, tv, zv)
    tgm = np.arange(tm[0], tm[-1], dt)
    b = np.interp(tgm, tm, zm)
    a0, b0 = a - a.mean(), b - b.mean()
    c = np.correlate(np.pad(b0, (len(a0), len(a0))), a0, "valid")
    coarse = tgm[0] - len(a0) * dt + np.argmax(c) * dt - tgv[0]

    def cost(o):
        t = tv + o
        m = (t >= tm[0]) & (t <= tm[-1])
        if m.sum() < 0.5 * len(tv):
            return 1e9
        d = zv[m] - np.interp(t[m], tm, zm)
        return np.mean((d - d.mean()) ** 2)
    r = minimize_scalar(cost, bounds=(coarse - 0.1, coarse + 0.1), method="bounded", options={"xatol": 1e-4})
    return float(r.x), float(np.sqrt(r.fun))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--board-json", default="data/video_board.json")
    ap.add_argument("--colour", default="orange", choices=sorted(HSV))
    ap.add_argument("--run-log", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--board-offset", type=float, default=None, help="board face x minus tip-plane x [mm]")
    ap.add_argument("--fpx", type=float, default=None, help="focal length [px] for --board-offset != 0")
    ap.add_argument("--per-frame", action="store_true", help="re-detect the board every frame (moving camera)")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    cfg = hop3cfg.load_config()
    vpath = absr(a.video)
    synthetic = "synthetic" in str(vpath)
    board = json.loads(absr(a.board_json).read_text())
    objs = board_points(board)
    pl = board["placement"]
    off = pl.get("board_offset_x_mm", 0.0) if a.board_offset is None else a.board_offset
    yc, zc = pl["centre_yz_mm"]
    run_log = a.run_log
    if run_log is None:
        sf = absr(a.session) if a.session else (ROOT / "data" / "synthetic" / "session_data_synth.json"
                                                 if synthetic else hop3cfg.session_path())
        if Path(sf).exists():
            s = json.loads(Path(sf).read_text())
            run_log = s.get("video_run") or (s.get("run_logs") or [None])[0]
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board["dict"])),
                                  cv2.aruco.DetectorParameters())

    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    n = len(frames)
    if n == 0:
        sys.exit(f"cannot read {vpath}")
    h, w = frames[0].shape[:2]
    tvid, tsrc = frame_times(vpath, n, fps)
    K = np.array([[a.fpx or 0.77 * max(w, h), 0, w / 2], [0, a.fpx or 0.77 * max(w, h), h / 2], [0, 0, 1]])

    first = [detect(det, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), objs) for f in frames[:N_STATIC]]
    first = [x for x in first if x[0] is not None and len(x[0]) == 4 * len(objs)]
    static = (not a.per_frame and len(first) >= 5 and
              float(np.max(np.std(np.stack([x[1] for x in first]), axis=0))) < 0.5)
    if static:
        obj0, img0 = first[0][0], np.median(np.stack([x[1] for x in first]), axis=0)
    win = board.get("window_mm")
    uv, px_all, reproj, nb = np.full((n, 2), np.nan), np.full((n, 2), np.nan), [], 0
    H = None
    dbg_mask = None
    for i, fr in enumerate(frames):
        if static:
            obj, img = obj0, img0
        else:
            obj, img = detect(det, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), objs)
            if obj is None:
                if H is None:
                    continue
                obj, img = last
            last = (obj, img)
        if H is None or not static:
            H, _ = cv2.findHomography(obj, img, 0)
            reproj.append(np.sqrt(np.mean(np.sum((img_to_board(H, img) - obj) ** 2, 1))))
            nb += 1
            roi = np.full((h, w), 255, np.uint8)
            if win:
                q = np.array([[win["u"][0] - 10, win["v"][0] - 10], [win["u"][1] + 10, win["v"][0] - 10],
                              [win["u"][1] + 10, win["v"][1] + 10], [win["u"][0] - 10, win["v"][1] + 10]])
                qi = cv2.perspectiveTransform(q.reshape(-1, 1, 2), H).reshape(-1, 2)
                roi[:] = 0
                cv2.fillPoly(roi, [np.round(qi).astype(np.int32)], 255)
        c, mask = blob(fr, a.colour, roi)
        if i == n // 2:
            dbg_mask = mask
        if c is None:
            continue
        px_all[i] = c
        uv[i] = img_to_board(H, c)[0] if off == 0 else parallax_fix([c], obj, img, K, off)[0]
    y, z = yc + uv[:, 0], zc + uv[:, 1]
    good = np.isfinite(y)

    sync_off, sync_res, tm, pm = None, None, None, None
    if run_log and absr(run_log).exists():
        tm, pm = meas_z(absr(run_log), cfg)
        sync_off, sync_res = sync(tvid[good], z[good], tm, pm[:, 2])

    stem = vpath.stem
    sub = "synthetic" if synthetic else ""
    rd, fd = ROOT / "results" / sub, ROOT / "figures" / sub
    rd.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)
    trun = tvid + (sync_off or 0.0)
    out = {
        "video": hop3cfg.rel(vpath), "synthetic": synthetic, "placeholder": False,
        "t_s": [round(float(v), 5) for v in trun[good]], "y_mm": [round(float(v), 4) for v in y[good]],
        "z_mm": [round(float(v), 4) for v in z[good]], "t_video_s": [round(float(v), 5) for v in tvid[good]],
        "frame": np.flatnonzero(good).tolist(),
        "reproj_rms_mm": float(np.mean(reproj)) if reproj else None,
        "sync_offset_s": sync_off, "sync_convention": "t_run = t_video + sync_offset_s",
        "sync_residual_rms_mm": sync_res, "sync_run_log": run_log, "time_source": tsrc, "fps": fps,
        "n_frames": n, "n_tracked": int(good.sum()), "board_json": a.board_json, "board_static": bool(static),
        "n_board_fits": nb, "board_offset_x_mm": off, "plane_x_mm": pl["plane_x_mm"], "colour": a.colour,
        "parallax_model": None if off == 0 else f"solvePnP with fpx={K[0, 0]:.0f}",
    }
    jf = rd / f"video_{stem}.json"
    jf.write_text(json.dumps(out, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    k = np.flatnonzero(good)
    snap = frames[k[len(k) // 2]].copy() if len(k) else frames[0].copy()
    pts = px_all[good]
    cv2.polylines(snap, [np.round(pts).astype(np.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
    if H is not None:
        for m in objs.values():
            cv2.polylines(snap, [np.round(cv2.perspectiveTransform(m.reshape(-1, 1, 2), H)).astype(np.int32)],
                          True, (0, 255, 0), 1)
    ax[0].imshow(cv2.cvtColor(snap, cv2.COLOR_BGR2RGB))
    ax[0].set_title("frame + tracked trail (magenta), board fit (green)")
    ax[0].axis("off")
    if pm is not None:
        ax[1].plot(pm[:, 1], pm[:, 2], color=(0.85, 0.1, 0.1), lw=1, label="DH(q_meas) run log")
    ax[1].plot(y[good], z[good], color=(0.8, 0.4, 0), lw=1, label="video tracked")
    ax[1].set_aspect("equal")
    ax[1].grid(True)
    ax[1].set_xlabel("Y0 [mm] (right, seen from +X0)")
    ax[1].set_ylabel("Z0 [mm]")
    ax[1].legend()
    ax[1].set_title("front view")
    if pm is not None:
        ax[2].plot(tm, pm[:, 2], color=(0.85, 0.1, 0.1), lw=1, label="DH(q_meas) z")
    ax[2].plot(trun[good], z[good], color=(0.8, 0.4, 0), lw=1, ls="--", label="video z (synced)")
    ax[2].set_xlabel("run time t [s]")
    ax[2].set_ylabel("Z0 [mm]")
    ax[2].grid(True)
    ax[2].legend()
    ttl = f"sync offset {sync_off:+.3f} s, residual {sync_res:.2f} mm" if sync_off is not None else "no run log: unsynced"
    ax[2].set_title(ttl)
    fig.suptitle(("SYNTHETIC - " if synthetic else "") + f"video tracking {stem}: {int(good.sum())}/{n} frames, "
                 f"board reproj {out['reproj_rms_mm']:.3f} mm")
    fig.tight_layout()
    fig.savefig(fd / f"video_{stem}_overlay.png", dpi=200)
    plt.close(fig)
    if a.debug and dbg_mask is not None:
        cv2.imwrite(str(fd / f"video_{stem}_mask.png"), dbg_mask)
    print(f"{jf}: {int(good.sum())}/{n} frames tracked, reproj {out['reproj_rms_mm']:.3f} mm, sync "
          f"{'n/a' if sync_off is None else f'{sync_off:+.4f} s'}")


if __name__ == "__main__":
    main()
