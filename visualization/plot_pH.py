import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

def smooth_pchip(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x_new = np.linspace(x.min(), x.max(), 300)
    return x_new, PchipInterpolator(x, y)(x_new)

OUT_DIR = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/Activity"

colors  = {"WT": "#2E8B57",   # sea green  (muted green)
           "D2": "#1A9BA6",   # teal       (muted cyan)
           "Q4": "#B5367A"}   # rose       (muted magenta)
markers = {"WT": "o", "D2": "s", "Q4": "^"}
FIGSIZE = (6, 4.5)
LW = 3
MS = 7

def styled_ax(ax, xlabel, ylabel, title, show_legend=False, **legend_kw):
    ax.set_xlabel(xlabel, fontsize=20, color="black")
    ax.set_ylabel(ylabel, fontsize=20, color="black")
    ax.tick_params(labelsize=17, colors="black", width=2, length=5)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(2)
    if show_legend:
        ax.legend(fontsize=16, handlelength=1.2, handletextpad=0.5,
                  labelspacing=0.3, **legend_kw)
    else:
        ax.legend_.remove() if ax.legend_ else None
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── pH — raw replicates, mean ± SD computed here ─────────────────────────────
df = pd.read_csv(f"{OUT_DIR}/pH.csv", index_col=0, na_values=["/"])
x = [int(c) for c in df.columns]

means, stds, ns = {}, {}, {}
for v in ["WT", "D2", "Q4"]:
    sub = df.loc[[v]].astype(float)   # force DataFrame even if only 1 row
    means[v] = sub.mean(axis=0, skipna=True).values
    stds[v]  = sub.std(axis=0, ddof=1, skipna=True).values
    ns[v]    = sub.count(axis=0).values
    print(f"{v}: n per pH = {ns[v].tolist()}")

fig, ax = plt.subplots(figsize=FIGSIZE)
for v in ["WT", "D2", "Q4"]:
    xs, ys = smooth_pchip(x, means[v])
    ax.plot(xs, ys, color=colors[v], linewidth=LW)
    ax.errorbar(x, means[v], yerr=stds[v], label=v,
                color=colors[v], marker=markers[v], linestyle="none",
                markersize=MS, capsize=4)
styled_ax(ax, "pH", "Specific Activity (U/mg)", "Effect of pH on Activity")
plt.tight_layout()
outfile = f"{OUT_DIR}/activity_pH_updated.png"
plt.savefig(outfile, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {outfile}")
