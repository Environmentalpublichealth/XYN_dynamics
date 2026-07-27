"""
RMSF for all 18 simulations plotted individually — no smoothing, no shading.
Layout: 2 rows (Apo / Holo) × 3 cols (Sim1 / Sim2 / Sim3).
All three variants overlaid per panel. Units: Å.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS   = {"WT": "#2ca02c", "E7": "#17becf", "Q4": "#e317cf"}
VARIANTS = ["WT", "E7", "Q4"]
SIM_LABELS = ["Sim1", "Sim2", "Sim3"]

FILES = {
    ("WT", "apo"):  ["rmsf_WT.xvg",     "rmsf_WT2.xvg",     "rmsf_WT3.xvg"],
    ("E7", "apo"):  ["rmsf_E7.xvg",     "rmsf_E72.xvg",     "rmsf_E73.xvg"],
    ("Q4", "apo"):  ["rmsf_Q4.xvg",     "rmsf_Q42.xvg",     "rmsf_Q43.xvg"],
    ("WT", "holo"): ["rmsf_WTholo.xvg", "rmsf_WTholo2.xvg", "rmsf_WTholo3.xvg"],
    ("E7", "holo"): ["rmsf_E7holo.xvg", "rmsf_E7holo2.xvg", "rmsf_E7holo3.xvg"],
    ("Q4", "holo"): ["rmsf_Q4holo.xvg", "rmsf_Q4holo2.xvg", "rmsf_Q4holo3.xvg"],
}


def read_xvg(path):
    x, y = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            x.append(float(parts[0]))
            y.append(float(parts[1]))
    return np.array(x), np.array(y)


def style_ax(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(direction="in", length=4, width=1.1, labelsize=13)


fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True, sharex=True)
fig.patch.set_facecolor("white")

for row, cond in enumerate(["apo", "holo"]):
    for col, sim_idx in enumerate([0, 1, 2]):
        ax = axes[row][col]
        for var in VARIANTS:
            res, rmsf = read_xvg(os.path.join(OUT_DIR, FILES[(var, cond)][sim_idx]))
            ax.plot(res, rmsf * 10, color=COLORS[var], linewidth=1.0, alpha=0.85)
        style_ax(ax)
        ax.set_xlim(res.min(), res.max())
        ax.set_ylim(bottom=0)
        cond_label = "Apo" if cond == "apo" else "Holo"
        ax.set_title(f"{cond_label} — {SIM_LABELS[sim_idx]}", fontsize=16, fontweight="bold")
        if col == 0:
            ax.set_ylabel("RMSF (Å)", fontsize=14)
        if row == 1:
            ax.set_xlabel("Residue Number", fontsize=14)

leg = [mlines.Line2D([], [], color=COLORS[v], linewidth=2.5, label=v)
       for v in VARIANTS]
fig.legend(handles=leg, loc="lower center", ncol=3,
           fontsize=15, frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(OUT_DIR, "rmsf_all_sims.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
