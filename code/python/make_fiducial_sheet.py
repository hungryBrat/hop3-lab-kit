"""A4 fiducial sheet for touch-off and pen-trace digitisation.

Sheet frame: origin = sheet centre, +x along robot +X0 (away from the base),
+y = robot +Y0, z up. Portrait A4, x along the 210 mm side.
Printed colours: markers black, everything else red or light cyan so a
BLUE or BLACK pen trace can be separated (red channel is dark only for ink).
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W_MM, H_MM = 210.0, 297.0
DICT_NAME = "DICT_4X4_50"
MARKER_MM = 30.0
# |x|=70, |y|=95: corners must stay reachable at z~40 with sheet centre at (240,0):
# far corner (310,95) -> theta3 ~ -25 deg, near corner (170,95) -> theta3 ~ -110 deg.
MARKER_CENTRES = {0: (-70.0, -95.0), 1: (70.0, -95.0), 2: (70.0, 95.0), 3: (-70.0, 95.0)}
CIRCLE_R = 50.0
GRID_MM = 10.0
SCALE_BAR = ((-50.0, 128.0), (50.0, 128.0))
RED = (0.85, 0.05, 0.05)
CYAN = (0.72, 0.90, 0.96)


def marker_corners(c):
    cx, cy = c
    h = MARKER_MM / 2
    # ArUco corner order TL, TR, BR, BL of the marker image; image top = sheet +y.
    return [[cx - h, cy + h], [cx + h, cy + h], [cx + h, cy - h], [cx - h, cy - h]]


def geometry():
    return {
        "dict": DICT_NAME,
        "marker_size_mm": MARKER_MM,
        "sheet_mm": [W_MM, H_MM],
        "frame": "origin=sheet centre; +x along robot +X0 (away from base); +y=robot +Y0; z up",
        "markers": {str(i): {"centre_mm": list(c), "corners_mm": marker_corners(c)}
                    for i, c in MARKER_CENTRES.items()},
        "touch_points": {f"F{i + 1}": list(MARKER_CENTRES[i]) for i in range(4)},
        "circle": {"centre": [0.0, 0.0], "R": CIRCLE_R},
        "scale_bar_mm": [list(SCALE_BAR[0]), list(SCALE_BAR[1])],
        "placement_nominal_robot_mm": [240.0, 0.0],
        "pen": "blue or black; never red",
    }


def mm_to_pt(mm):
    return mm * 72.0 / 25.4


def draw(fig_path_pdf=None, png_path=None, dpi=300):
    fig = plt.figure(figsize=(W_MM / 25.4, H_MM / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-W_MM / 2, W_MM / 2)
    ax.set_ylim(-H_MM / 2, H_MM / 2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    lw_grid = mm_to_pt(0.2)
    for g in np.arange(-100, 101, GRID_MM):
        ax.plot([g, g], [-140, 140], color=CYAN, lw=lw_grid, zorder=1)
    for g in np.arange(-140, 141, GRID_MM):
        ax.plot([-100, 100], [g, g], color=CYAN, lw=lw_grid, zorder=1)

    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    cell = MARKER_MM / 6
    for mid, (cx, cy) in MARKER_CENTRES.items():
        q = MARKER_MM / 2 + 5
        ax.add_patch(Rectangle((cx - q, cy - q), 2 * q, 2 * q, fc="white", ec="none", zorder=2))
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
        lw = mm_to_pt(0.2)
        h = MARKER_MM / 2
        ax.plot([cx - h, cx + h], [cy, cy], color=RED, lw=lw, zorder=5)
        ax.plot([cx, cx], [cy - h, cy + h], color=RED, lw=lw, zorder=5)
        for sx, sy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ax.plot([cx + sx * (h + 1), cx + sx * (h + 5)], [cy + sy * (h + 1), cy + sy * (h + 5)],
                    color=RED, lw=mm_to_pt(0.3), zorder=5)
        ax.text(cx, cy + (h + 6.5) * (1 if cy > 0 else -1), f"F{mid + 1} (id {mid})  ({cx:+.0f}, {cy:+.0f})",
                color=RED, ha="center", va="center", fontsize=7, zorder=5)

    ax.add_patch(Circle((0, 0), CIRCLE_R, fill=False, ec=RED, lw=mm_to_pt(0.3), zorder=5))
    ax.plot([-3, 3], [0, 0], color=RED, lw=mm_to_pt(0.2), zorder=5)
    ax.plot([0, 0], [-3, 3], color=RED, lw=mm_to_pt(0.2), zorder=5)
    ax.add_patch(FancyArrow(0, 0, 30, 0, width=0.4, head_width=3, head_length=4, color=RED, zorder=5))
    ax.add_patch(FancyArrow(0, 0, 0, 30, width=0.4, head_width=3, head_length=4, color=RED, zorder=5))
    ax.text(36, 0, "+X0\n(away from\nrobot base)", color=RED, fontsize=7, va="center", zorder=5)
    ax.text(0, 37, "+Y0", color=RED, fontsize=7, ha="center", zorder=5)
    ax.text(-4, -4, "O = robot (240, 0)", color=RED, fontsize=6, ha="right", va="top", zorder=5)
    ax.text(0, -CIRCLE_R - 4, "nominal circle R = 50 mm", color=RED, fontsize=6, ha="center", va="top", zorder=5)

    (x0, y0), (x1, _) = SCALE_BAR
    ax.plot([x0, x1], [y0, y0], color=RED, lw=mm_to_pt(0.4), zorder=5)
    for xe in (x0, x1):
        ax.plot([xe, xe], [y0 - 3, y0 + 3], color=RED, lw=mm_to_pt(0.3), zorder=5)
    ax.text(0, y0 + 4.5, "100 mm  -  measure me  (record as printed_scale_check_mm)",
            color=RED, fontsize=7, ha="center", zorder=5)
    ax.text(0, 141, "HOP-3 fiducial sheet  -  print at 100% / Actual size (NO fit-to-page)",
            color=RED, fontsize=8, ha="center", va="center", zorder=5)
    ax.text(0, -128, "Pen: BLUE or BLACK only (never red).  Touch-off: pen tip on red cross centres F1-F4.",
            color=RED, fontsize=6.5, ha="center", zorder=5)
    ax.text(0, -136, "Place sheet centre O at robot (240, 0) with the +X0 arrow pointing away from the base.",
            color=RED, fontsize=6.5, ha="center", zorder=5)
    ax.text(-101, 0, "robot base this side  (-X0)", color=RED, fontsize=7, rotation=90,
            ha="center", va="center", zorder=5)

    if fig_path_pdf:
        fig.savefig(fig_path_pdf, dpi=dpi)
    if png_path:
        fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def render_rgb(dpi=300):
    """Sheet raster (RGB uint8) and its px/mm, rendered through the same drawing path as the PDF."""
    import io
    buf = io.BytesIO()
    draw(png_path=buf, dpi=dpi)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, (img.shape[1] / W_MM, img.shape[0] / H_MM)


def main():
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    draw(out / "fiducial_sheet.pdf", out / "fiducial_sheet.png", dpi=300)
    with open(out / "fiducials.json", "w") as f:
        json.dump(geometry(), f, indent=2)
    print(f"wrote {out / 'fiducial_sheet.pdf'}, fiducial_sheet.png, fiducials.json")


if __name__ == "__main__":
    sys.exit(main())
