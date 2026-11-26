"""Digitise a pen trace from a photo of the fiducial sheet.

usage: digitize_trace.py photo.jpg [--debug] [--session FILE] [--scale-check-mm L]
Pen must be BLUE or BLACK: ink is found as pixels much darker than the local paper
in the RED channel, which hides the red printing and the light-cyan grid.
Outputs results/trace_<stem>.json and figures/trace_<stem>.png
(results/synthetic, figures/synthetic when the photo lives under a 'synthetic' dir).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def kasa(xy):
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = -(xy ** 2).sum(1)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    c = np.array([-D / 2, -E / 2])
    return c, float(np.sqrt(c @ c - F))


def geometric_refine(xy, c, r, iters=10):
    for _ in range(iters):
        d = xy - c
        rho = np.linalg.norm(d, axis=1)
        J = np.column_stack([-d[:, 0] / rho, -d[:, 1] / rho, -np.ones(len(xy))])
        step = np.linalg.lstsq(J, -(rho - r), rcond=None)[0]
        c, r = c + step[:2], r + step[2]
    return c, float(r)


def directed_hausdorff(a, b):
    from scipy.spatial import cKDTree
    return float(cKDTree(b).query(a)[0].max())


def centreline(xy, c, bin_deg=0.5):
    d = xy - c
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
    rad = np.linalg.norm(d, axis=1)
    idx = (ang / bin_deg).astype(int)
    out = []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() < 3:
            continue
        a = np.radians(np.median(ang[sel]))
        r = np.median(rad[sel])
        out.append(c + r * np.array([np.cos(a), np.sin(a)]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--session", default=str(ROOT / "session" / "session_data.json"))
    ap.add_argument("--scale-check-mm", type=float, default=None)
    ap.add_argument("--fiducials", default=str(ROOT / "data" / "fiducials.json"))
    a = ap.parse_args()

    photo = Path(a.photo).resolve()
    synthetic = "synthetic" in photo.parts
    sub = "synthetic" if synthetic else ""
    res_dir, fig_dir = ROOT / "results" / sub, ROOT / "figures" / sub
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fid = json.loads(Path(a.fiducials).read_text())
    sess = json.loads(Path(a.session).read_text()) if Path(a.session).exists() else {}
    scale_mm = a.scale_check_mm if a.scale_check_mm is not None else sess.get("printed_scale_check_mm")
    placeholder = scale_mm is None
    scale = (scale_mm if scale_mm is not None else 100.0) / 100.0

    img = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {photo}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, fid["dict"])), params)
    corners, ids, _ = det.detectMarkers(gray)
    found = {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    found = {i: c for i, c in found.items() if str(i) in fid["markers"]}
    if len(found) < 3:
        raise SystemExit(f"only {len(found)} fiducial markers found (need >= 3, ideally 4): {sorted(found)}")
    img_pts = np.vstack([found[i] for i in sorted(found)]).astype(np.float64)
    mm_pts = scale * np.vstack([np.array(fid["markers"][str(i)]["corners_mm"]) for i in sorted(found)])
    H, _ = cv2.findHomography(img_pts, mm_pts, 0)
    reproj = cv2.perspectiveTransform(img_pts[None], H)[0] - mm_pts
    reproj_rms = float(np.sqrt((reproj ** 2).sum(1).mean()))

    Hinv = np.linalg.inv(H)
    c_img = cv2.perspectiveTransform(np.array([[[0.0, 0.0]]]), Hinv)[0, 0]
    probe = cv2.perspectiveTransform(np.array([[[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]]), Hinv)[0]
    ppm = float(np.sqrt(abs(np.cross(probe[1] - probe[0], probe[2] - probe[0]))) / 10.0)

    red = img[:, :, 2].astype(np.float32)
    k = int(2 * round(2.0 * ppm) + 1)
    bg = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    ink = ((bg - red) > np.maximum(45.0, 0.40 * bg)).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    ys, xs = np.nonzero(ink)
    pix = np.column_stack([xs, ys]).astype(np.float64)
    mm = cv2.perspectiveTransform(pix[None], H)[0]
    hw, hh = scale * fid["sheet_mm"][0] / 2 - 4, scale * fid["sheet_mm"][1] / 2 - 4
    keep = (np.abs(mm[:, 0]) < hw) & (np.abs(mm[:, 1]) < hh)
    q = scale * (fid["marker_size_mm"] / 2 + 4)
    for m in fid["markers"].values():
        cx, cy = scale * np.array(m["centre_mm"])
        keep &= ~((np.abs(mm[:, 0] - cx) < q) & (np.abs(mm[:, 1] - cy) < q))
    mask = np.zeros_like(ink)
    mask[ys[keep], xs[keep]] = 1
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = 3.0 * ppm * ppm
    good = np.zeros(n_lab, bool)
    good[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    sel = good[lab[ys, xs]] & keep
    xy = mm[sel]
    if len(xy) < 50:
        raise SystemExit(f"too few ink pixels ({len(xy)}): pen colour must be blue/black")

    c0, r0 = kasa(xy)
    c, r = geometric_refine(xy, c0, r0)
    cl = centreline(xy, c)
    nom_c = scale * np.array(fid["circle"]["centre"], float)
    nom_r = scale * fid["circle"]["R"]
    radial_fit = np.linalg.norm(cl - c, axis=1) - r
    radial_nom = np.linalg.norm(cl - nom_c, axis=1) - nom_r
    tt = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    ideal = nom_c + nom_r * np.column_stack([np.cos(tt), np.sin(tt)])
    haus = max(directed_hausdorff(cl, ideal), directed_hausdorff(ideal, cl))

    out = {
        "photo": str(photo), "synthetic": synthetic, "placeholder": placeholder,
        "pen_colour_required": "blue or black",
        "printed_scale_check_mm": scale_mm, "scale_applied": scale,
        "markers_found": sorted(found), "homography_img_to_mm": H.tolist(),
        "reproj_rms_mm": reproj_rms, "px_per_mm": ppm, "n_ink_px": int(len(xy)),
        "frame": "sheet mm (origin = sheet centre, +x = robot +X0); scaled by printed scale check",
        "fit": {"centre_mm": c.tolist(), "R_mm": r, "kasa_centre_mm": c0.tolist(), "kasa_R_mm": r0},
        "nominal": {"centre_mm": nom_c.tolist(), "R_mm": nom_r},
        "metrics": {
            "R_mm": r, "R_err_mm": r - nom_r, "centre_offset_mm": float(np.linalg.norm(c - nom_c)),
            "radial_rms_vs_fit_mm": float(np.sqrt(np.mean(radial_fit ** 2))),
            "radial_max_vs_fit_mm": float(np.abs(radial_fit).max()),
            "radial_rms_vs_nominal_mm": float(np.sqrt(np.mean(radial_nom ** 2))),
            "radial_max_vs_nominal_mm": float(np.abs(radial_nom).max()),
            "hausdorff_vs_nominal_mm": haus},
        "points_mm": cl.round(4).tolist(),
    }
    stem = photo.stem
    (res_dir / f"trace_{stem}.json").write_text(json.dumps(out, indent=1))

    ov = img.copy()
    ov[ys[sel], xs[sel]] = (0, 200, 0)
    for poly, col in [(c + r * np.column_stack([np.cos(tt), np.sin(tt)]), (255, 0, 255)), (ideal, (0, 200, 255))]:
        p = cv2.perspectiveTransform(poly[None], Hinv)[0]
        cv2.polylines(ov, [np.round(p * 16).astype(np.int32)], True, col, max(2, int(ppm / 3)), cv2.LINE_AA, 4)
    cv2.aruco.drawDetectedMarkers(ov, [found[i].reshape(1, 4, 2).astype(np.float32) for i in sorted(found)],
                                  np.array(sorted(found)).reshape(-1, 1))
    cv2.drawMarker(ov, tuple(int(v) for v in c_img), (0, 0, 255), cv2.MARKER_CROSS, int(5 * ppm), 3)
    s = 2000.0 / max(ov.shape[:2])
    ov = cv2.resize(ov, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lines = [("SYNTHETIC  " if synthetic else "") + f"trace {stem}",
             f"fit R={r:.2f} mm (nominal {nom_r:.2f}), centre ({c[0]:+.2f},{c[1]:+.2f}) mm",
             f"Hausdorff vs nominal {haus:.2f} mm; reproj {reproj_rms:.3f} mm; scale {scale:.4f}",
             "green=ink  magenta=fit  orange=nominal"]
    for i, t in enumerate(lines):
        cv2.putText(ov, t, (20, 45 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(ov, t, (20, 45 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if i == 0 and synthetic
                    else (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(fig_dir / f"trace_{stem}.png"), ov)
    if a.debug:
        cv2.imwrite(str(fig_dir / f"trace_{stem}_inkmask.png"), (mask * 255).astype(np.uint8))
        print(json.dumps({k: v for k, v in out.items() if k != "points_mm"}, indent=1))
    print(f"R={r:.3f} mm centre=({c[0]:.3f},{c[1]:.3f}) n_ink={len(xy)} -> {res_dir / f'trace_{stem}.json'}")


if __name__ == "__main__":
    main()
