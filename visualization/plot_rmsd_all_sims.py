"""
RMSD for all 18 simulations plotted individually — no smoothing, no shading.
Layout: 2 rows (Apo / Holo) × 3 cols (WT / E7 / Q4).
Line style encodes replicate: Sim1=solid, Sim2=dashed, Sim3=dash-dot.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS = {"WT": "#2ca02c", "E7": "#17becf", "Q4": "#e317cf"}
VARIANTS = ["WT", "E7", "Q4"]
SIM_LS     = ["-", "--", "-."]
SIM_LABELS = ["Sim1", "Sim2", "Sim3"]

FILES = {
    ("WT", "apo"):  ["rmsd_WT.xvg",     "rmsd_WT2.xvg",     "rmsd_WT3.xvg"],
    ("E7", "apo"):  ["rmsd_E7.xvg",     "rmsd_E72.xvg",     "rmsd_E73.xvg"],
    ("Q4", "apo"):  ["rmsd_Q4.xvg",     "rmsd_Q42.xvg",     "rmsd_Q43.xvg"],
    ("WT", "holo"): ["rmsd_WTholo.xvg", "rmsd_WTholo2.xvg", "rmsd_WTholo3.xvg"],
    ("E7", "holo"): ["rmsd_E7holo.xvg", "rmsd_E7holo2.xvg", "rmsd_E7holo3.xvg"],
    ("Q4", "holo"): ["rmsd_Q4holo.xvg", "rmsd_Q4holo2.xvg", "rmsd_Q4holo3.xvg"],
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
    ax.tick_params(direction="in", length=4, width=1.1, labelsize=11)


fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True, sharex=True)
fig.patch.set_facecolor("white")

for row, cond in enumerate(["apo", "holo"]):
    for col, sim_idx in enumerate([0, 1, 2]):
        ax = axes[row][col]
        for var in VARIANTS:
            t, rmsd = read_xvg(os.path.join(OUT_DIR, FILES[(var, cond)][sim_idx]))
            ax.plot(t, rmsd * 10, color=COLORS[var], linewidth=0.8, alpha=0.85)
        style_ax(ax)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        cond_label = "Apo" if cond == "apo" else "Holo"
        ax.set_title(f"{cond_label} — {SIM_LABELS[sim_idx]}", fontsize=16, fontweight="bold")
        ax.tick_params(labelsize=13)
        if col == 0:
            ax.set_ylabel("RMSD (Å)", fontsize=14)
        if row == 1:
            ax.set_xlabel("Time (ns)", fontsize=14)

# shared legend for variants
leg = [mlines.Line2D([], [], color=COLORS[v], linewidth=2.5, label=v)
       for v in VARIANTS]
fig.legend(handles=leg, loc="lower center", ncol=3,
           fontsize=15, frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
out = os.path.join(OUT_DIR, "rmsd_all_sims.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
