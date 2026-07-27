import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline, PchipInterpolator
from scipy.optimize import curve_fit

def smooth(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x_new = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    return x_new, spl(x_new)

def smooth_pchip(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x_new = np.linspace(x.min(), x.max(), 300)
    return x_new, PchipInterpolator(x, y)(x_new)

def exp_decay_fixed(x, b):
    """Exponential decay forced through 100% at t=0."""
    return 100 * np.exp(-b * x)

def smooth_decay(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    popt, _ = curve_fit(exp_decay_fixed, x, y, p0=[0.01], maxfev=5000)
    x_new = np.linspace(x.min(), x.max(), 300)
    return x_new, exp_decay_fixed(x_new, *popt)

OUT_DIR = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/Activity"

colors  = {"WT":    "#2E8B57",   # sea green  (muted green)
           "D2":    "#1A9BA6",   # teal       (muted cyan)
           "Q4":    "#B5367A",   # rose       (muted magenta)
           "N111D": "#D4690A",   # amber      (muted orange)
           "E112Q": "#7B3FA0"}   # violet     (muted purple)
markers = {"WT": "o",       "D2": "s",        "Q4": "^",
           "N111D": "D",      "E112Q": "v"}
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

# ── 1. pH ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(f"{OUT_DIR}/pH.csv", index_col=0)
x = [int(c) for c in df.columns]
means = {"WT": df.loc["WT"].values,
         "D2": df.loc["D2"].values,
         "Q4": df.loc["Q4"].values}
stds  = {"WT": df.iloc[3].values,
         "D2": df.iloc[4].values,
         "Q4": df.iloc[5].values}

fig, ax = plt.subplots(figsize=FIGSIZE)
for v in ["WT", "D2", "Q4"]:
    xs, ys = smooth_pchip(x, means[v])
    ax.plot(xs, ys, color=colors[v], linewidth=LW)
    ax.errorbar(x, means[v], yerr=stds[v], label=v,
                color=colors[v], marker=markers[v], linestyle="none",
                markersize=MS, capsize=4)
styled_ax(ax, "pH", "Specific Activity (U/mg)", "Effect of pH on Activity")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/activity_pH.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved activity_pH.png")

# ── 2. Temperature ────────────────────────────────────────────────────────────
df = pd.read_csv(f"{OUT_DIR}/temperature.csv")
means_df = df[df["Temp"] != "std"].copy()
stds_df  = df[df["Temp"] == "std"].copy()
x = means_df["Temp"].astype(int).values
means = {"WT":    means_df["WT"].values,
         "D2":    means_df["D2"].values,
         "Q4":    means_df["Q4"].values,
         "N111D": means_df["N111D"].values,
         "E112Q": means_df["E112Q"].values}
stds  = {"WT":    stds_df["WT"].values,
         "D2":    stds_df["D2"].values,
         "Q4":    stds_df["Q4"].values,
         "N111D": stds_df["N111D"].values,
         "E112Q": stds_df["E112Q"].values}

fig, ax = plt.subplots(figsize=FIGSIZE)
for v in ["WT", "D2", "Q4", "N111D", "E112Q"]:
    xs, ys = smooth_pchip(x, means[v])
    ax.plot(xs, ys, color=colors[v], linewidth=LW)
    ax.errorbar(x, means[v], yerr=stds[v], label=v,
                color=colors[v], marker=markers[v], linestyle="none",
                markersize=MS, capsize=4)
styled_ax(ax, "Temperature (°C)", "Specific Activity (U/mg)", "Effect of Temperature on Activity",
          show_legend=True,
          loc="upper right", framealpha=0.0, edgecolor="none")
ax.set_xticks(x)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/activity_temperature.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved activity_temperature.png")

# ── 3. Substrate ──────────────────────────────────────────────────────────────
df = pd.read_excel(f"{OUT_DIR}/substrate.xlsx", index_col=0)
x = [int(c) for c in df.columns]
means = {"WT": df.loc["WT"].values,
         "D2": df.loc["D2"].values,
         "Q4": df.loc["Q4"].values}
stds  = {"WT": df.iloc[3].values,
         "D2": df.iloc[4].values,
         "Q4": df.iloc[5].values}

fig, ax = plt.subplots(figsize=FIGSIZE)
for v in ["WT", "D2", "Q4"]:
    xs, ys = smooth(x, means[v])
    ax.plot(xs, ys, color=colors[v], linewidth=LW)
    ax.errorbar(x, means[v], yerr=stds[v], label=v,
                color=colors[v], marker=markers[v], linestyle="none",
                markersize=MS, capsize=4)
styled_ax(ax, "Substrate (g/L)", "Specific Activity (U/mg)", "Effect of Substrate Concentration on Activity")
ax.set_xticks(x)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/activity_substrate.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved activity_substrate.png")

# ── 4. Half-life ──────────────────────────────────────────────────────────────
df = pd.read_csv(f"{OUT_DIR}/halflife.csv", index_col=0)
x_all = [int(c) for c in df.columns]

# WT has NaN for 200 and 300 min — drop those points
wt_mean = df.loc["WT"].values.astype(float)
wt_std  = df.iloc[3].values.astype(float)
wt_mask = ~np.isnan(wt_mean)

means = {"WT": (np.array(x_all)[wt_mask], wt_mean[wt_mask], wt_std[wt_mask]),
         "D2": (x_all, df.loc["D2"].values.astype(float), df.iloc[4].values.astype(float)),
         "Q4": (x_all, df.loc["Q4"].values.astype(float), df.iloc[5].values.astype(float))}

fig, ax = plt.subplots(figsize=FIGSIZE)
for v in ["WT", "D2", "Q4"]:
    xv, yv, ev = means[v]
    xs, ys = smooth_decay(xv, yv)
    ax.plot(xs, ys, color=colors[v], linewidth=LW)
    ax.errorbar(xv, yv, yerr=ev, label=v,
                color=colors[v], marker=markers[v], linestyle="none",
                markersize=MS, capsize=4)
ax.axhline(50, color="gray", linestyle="--", linewidth=1.5)
styled_ax(ax, "Time (min)", "Residual Activity (%)", "Thermostability")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/activity_halflife.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved activity_halflife.png")

print("\nAll activity plots saved to Activity/")
