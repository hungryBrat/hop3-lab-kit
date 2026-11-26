"""A4 ArUco board for phone-video tip tracking in the front plane x = 260 mm.

Board frame (u, v) mm: origin = board centre, +u = robot +Y0 (viewer's right, camera at +X0), +v = +Z0 up.
Place the board face in the plane of the letter (or at a measured x offset), centre at (y, z) = centre_yz.
Cut out the dashed window so the forearm can pass through; the tip moves inside the window.
Writes data/video_board.pdf, data/video_board.png, data/video_board.json.
"""
import io
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "config.json").read_text())
LB = CFG["trajectories"]["letterB"]
W_MM, H_MM = 210.0, 297.0
DICT_NAME = "DICT_4X4_50"
MARKER_MM = 30.0
MARKER_CENTRES = {10: (-80.0, 120.0), 11: (80.0, 120.0), 12: (80.0, -120.0), 13: (-80.0, -120.0)}
WINDOW = (-55.0, 55.0, -95.0, 95.0)
RED = (0.85, 0.05, 0.05)


def marker_corners(c):
    cx, cy = c
    h = MARKER_MM / 2
    return [[cx - h, cy + h], [cx + h, cy + h], [cx + h, cy - h], [cx - h, cy - h]]


def geometry():
    return {
        "dict": DICT_NAME, "marker_size_mm": MARKER_MM, "sheet_mm": [W_MM, H_MM],
        "frame": "origin=board centre; +u=robot +Y0 (right, seen from +X0); +v=robot +Z0 (up)",
        "markers": {str(i): {"centre_mm": list(c), "corners_mm": marker_corners(c)}
                    for i, c in MARKER_CENTRES.items()},
        "window_mm": {"u": [WINDOW[0], WINDOW[1]], "v": [WINDOW[2], WINDOW[3]]},
        "placement": {"plane_x_mm": float(LB["x_plane_mm"]), "centre_yz_mm": [float(v) for v in LB["centre_yz"]],
                      "board_offset_x_mm": 0.0,
                      "note": "board face x = plane_x_mm + board_offset_x_mm (negative = behind the tip plane)"},
    }


def mm_to_pt(mm):
    return mm * 72.0 / 25.4


def draw(pdf=None, png=None, dpi=300):
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-W_MM / 2, W_MM / 2)
    ax.set_ylim(-H_MM / 2, H_MM / 2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    cell = MARKER_MM / 6
    for mid, (cx, cy) in MARKER_CENTRES.items():
        bits = cv2.aruco.generateImageMarker(dic, mid, 6, borderBits=1)
        ax.add_patch(Rectangle((cx - MARKER_MM / 2, cy - MARKER_MM / 2), MARKER_MM, MARKER_MM,
                               fc="black", ec="none", zorder=3))
        for r in range(6):
            for c in range(6):
                if bits[r, c] > 127:
                    x0 = cx - MARKER_MM / 2 + c * cell
                    y0 = cy + MARKER_MM / 2 - (r + 1) * cell
                    ax.add_patch(Rectangle((x0 - 0.02, y0 - 0.02), cell + 0.04, cell + 0.04,
                                           fc="white", ec="none", zorder=4))
        ax.text(cx, cy + (MARKER_MM / 2 + 4) * (1 if cy > 0 else -1), f"id {mid}  ({cx:+.0f}, {cy:+.0f})",
                color=RED, ha="center", va="center", fontsize=6, zorder=5)
    u0, u1, v0, v1 = WINDOW
    ax.plot([u0, u1, u1, u0, u0], [v0, v0, v1, v1, v0], color=RED, lw=mm_to_pt(0.3), ls="--", zorder=5)
    for s in (-1, 1):
        ax.plot([s * (u1 + 1), s * (u1 + 6)], [0, 0], color=RED, lw=mm_to_pt(0.3), zorder=5)
        ax.plot([0, 0], [s * (v1 + 1), s * (v1 + 6)], color=RED, lw=mm_to_pt(0.3), zorder=5)
    zc = LB["centre_yz"][1]
    ax.text(0, v1 + 9, f"CUT OUT window.  Ticks: board centre = (y 0, z {zc:.0f}) mm, x = {LB['x_plane_mm']} mm",
            color=RED, fontsize=6.5, ha="center", zorder=5)
    ax.plot([-50, 50], [-141, -141], color=RED, lw=mm_to_pt(0.4), zorder=5)
    ax.text(0, -137, "100 mm (check)   print at 100% / Actual size;  face toward the camera (+X0)",
            color=RED, fontsize=6.5, ha="center", zorder=5)
    ax.text(0, 143, "HOP-3 video board  -  +u = robot +Y0 (right)  +v = +Z0 (up)", color=RED, fontsize=7.5,
            ha="center", va="center", zorder=5)
    if pdf:
        fig.savefig(pdf, dpi=dpi)
    if png:
        fig.savefig(png, dpi=dpi)
    plt.close(fig)


def render_rgb(dpi=200):
    buf = io.BytesIO()
    draw(png=buf, dpi=dpi)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return img, (img.shape[1] / W_MM, img.shape[0] / H_MM)


def main():
    out = ROOT / "data"
    draw(out / "video_board.pdf", out / "video_board.png")
    (out / "video_board.json").write_text(json.dumps(geometry(), indent=2))
    print(f"wrote {out / 'video_board.pdf'}, video_board.png, video_board.json")


if __name__ == "__main__":
    sys.exit(main())
