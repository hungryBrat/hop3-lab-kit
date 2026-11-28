"""Jitter study: 50 Hz stream variants vs matched-speed stream -> results/jitter_study.json, report table, figure."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).parent))
import hop3cfg as hc

RUNS = [("logs/experiments/run_letterB_20261127_190651.csv", "50 Hz stream, full torque", 15),
        ("logs/experiments/run_letterB_20261127_190748.csv", "50 Hz, speed cap 120, acc 254", 15),
        ("logs/experiments/run_letterB_v30_20261127_190944.csv", "50 Hz, B at 30 mm/s", 30),
        ("logs/experiments/run_letterB_v45_20261127_191019.csv", "50 Hz, B at 45 mm/s", 45),
        ("logs/experiments/run_letterB_v30_20261127_193024.csv", "matched speed, B at 30 mm/s", 30),
        ("logs/run_letterB_20261127_192850.csv", "matched speed, B at 15 mm/s (final)", 15),
        ("logs/run_letterB_20261127_193116.csv", "matched speed, repeat 2", 15),
        ("logs/run_letterB_20261127_193218.csv", "matched speed, repeat 3", 15)]


def tip(d, m, kind, cfg):
    s = np.stack([d[f"{kind}_c{j}"][m] for j in (1, 2, 3)], 1) * 360 / cfg["encoder"]["counts_per_rev"]
    return hc.fk_servo(s, cfg)


def contour(pp, path):
    seg_a, seg_b = path[:-1], path[1:]
    ab = seg_b - seg_a
    L2 = np.maximum(np.sum(ab ** 2, 1), 1e-12)
    out = np.empty(len(pp))
    for i, p in enumerate(pp):
        t = np.clip(np.sum((p - seg_a) * ab, 1) / L2, 0, 1)
        out[i] = np.sqrt(np.min(np.sum((seg_a + t[:, None] * ab - p) ** 2, 1)))
    return out


def main():
    cfg = hc.load_config()
    rows, curves = [], []
    for f, label, v in RUNS:
        p = hc.ROOT / f
        d = np.genfromtxt(p, delimiter=",", names=True)
        meta = json.loads(p.with_suffix(".json").read_text())
        cfg_r = meta.get("config_snapshot", cfg)
        m = (d["k"] >= 50) & (d["k"] <= d["k"].max() - 50) & (d["comm_ok"] == 1)
        pg, pp = tip(d, m, "goal", cfg_r), tip(d, m, "pos", cfg_r)
        dense = tip(d, d["k"] >= 0, "goal", cfg_r)
        c = contour(pp, dense)
        e = np.stack([d[f"pos_c{j}"][m] - d[f"goal_c{j}"][m] for j in (1, 2, 3)], 1) * 360 / 4096
        rows.append({"log": f, "label": label, "v_mm_s": v, "decim": meta.get("decim", 1),
                     "torque_limit": meta.get("torque_limit"), "speed": meta.get("speed"), "acc": meta.get("acc"),
                     "contour_rms_mm": float(np.sqrt(np.mean(c ** 2))), "contour_max_mm": float(c.max()),
                     "joint_err_sd_deg": [float(x) for x in e.std(0)],
                     "roughness_mm": float(np.sqrt(np.mean(np.sum(np.diff(pp, 2, axis=0) ** 2, 1))))})
        curves.append((label, pp, dense))
        print(f"{label:40s} contour rms {rows[-1]['contour_rms_mm']:.2f} max {rows[-1]['contour_max_mm']:.2f}  "
              f"joint sd {np.round(e.std(0), 2)} deg")
    (hc.ROOT / "results" / "jitter_study.json").write_text(json.dumps({"runs": rows}, indent=2))
    lines = [r"\begin{tabular}{@{}lrrrr@{}}\toprule",
             r"Run & contour RMS & contour max & joint-error s.d.\ J1/J2/J3 & B speed\\",
             r" & (mm) & (mm) & (deg) & (mm/s)\\\midrule"]
    for r in rows:
        sd = "/".join(f"{x:.2f}" for x in r["joint_err_sd_deg"])
        lines.append(f"{r['label']} & {r['contour_rms_mm']:.2f} & {r['contour_max_mm']:.2f} & {sd} & {r['v_mm_s']}\\\\")
    lines.append(r"\bottomrule\end{tabular}")
    (hc.ROOT / "report" / "generated" / "tab_jitter.tex").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    for a, i in zip(ax, (0, 5)):
        lab, pp, dense = curves[i]
        a.plot(dense[:, 1], dense[:, 2], "k-", lw=1, label="commanded DH(q_cmd)")
        a.plot(pp[:, 1], pp[:, 2], "-", color="tab:blue", lw=1, label="traced DH(q_meas)")
        a.set_title(lab, fontsize=10); a.set_aspect("equal"); a.grid(alpha=.3)
        a.set_xlabel("y0 (mm)")
    ax[0].set_ylabel("z0 (mm, J2 axis = 0)"); ax[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(hc.ROOT / "figures" / "jitter_study.png", dpi=300)
    print("wrote results/jitter_study.json, report/generated/tab_jitter.tex, figures/jitter_study.png")


if __name__ == "__main__":
    main()
