import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import openpyxl
import warnings

OUT_DIR = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/Activity"
E_conc  = 0.0555   # mg enzyme

# ── Read substrate.xlsx (3 raw replicates per variant, no pre-computed avg) ──
wb = openpyxl.load_workbook(f"{OUT_DIR}/substrate.xlsx")
ws = wb.active
raw = [row for row in ws.iter_rows(values_only=True) if any(c is not None for c in row)]

header = [c for c in raw[0][1:] if c is not None]
S = np.array(header, dtype=float)   # [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]

reps_raw = {}
for row in raw[1:]:
    label = row[0]
    if label is None:
        continue
    vals = np.array([v for v in row[1:] if v is not None], dtype=float)
    reps_raw.setdefault(label, []).append(vals)

colors  = {"WT": "#2E8B57", "Q4": "#B5367A"}
markers = {"WT": "o",       "Q4": "^"}

def mm_eq(S, Vmax, Km):
    return Vmax * S / (Km + S)

results = {}

# ── Fit each variant (WT and Q4) with MM ─────────────────────────────────────
for var in ["WT", "Q4"]:
    reps = np.array(reps_raw[var])        # shape (n_reps, n_points)
    avg  = reps.mean(axis=0)
    std  = reps.std(axis=0, ddof=1)

    # Guard against zero std (identical replicates)
    min_std = std[std > 0].min() if (std > 0).any() else 1.0
    std_fit = np.where(std == 0, min_std, std)

    # V/S diagnostic
    vs = avg / S
    print(f"\n{var} V/S ratios:")
    for s, v, r in zip(S, avg, vs):
        print(f"  S={s:.1f}  V={v:.1f}  V/S={r:.1f}")

    # MM fit with multiple starting points
    best_popt, best_pcov, best_res = None, None, np.inf
    for km0 in [S.mean(), S.max(), S.max()*2, S.max()*5, 0.5, 1.0, 2.0]:
        for vm0 in [avg.max()*1.2, avg.max()*2, avg.max()*5]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, pcov = curve_fit(
                        mm_eq, S, avg, p0=[vm0, km0],
                        sigma=std_fit, absolute_sigma=True,
                        bounds=([0, 0], [np.inf, np.inf]),
                        maxfev=20000, method="trf")
                res = np.sum(((mm_eq(S, *popt) - avg) / std_fit) ** 2)
                if res < best_res:
                    best_res, best_popt, best_pcov = res, popt, pcov
            except RuntimeError:
                pass

    Vmax, Km   = best_popt
    SE_Vmax, SE_Km = np.sqrt(np.diag(best_pcov))

    Kcat       = Vmax / E_conc
    SE_Kcat    = SE_Vmax / E_conc
    Kcat_Km    = Kcat / Km
    SE_Kcat_Km = Kcat_Km * np.sqrt((SE_Kcat/Kcat)**2 + (SE_Km/Km)**2)

    results[var] = dict(reps=reps, avg=avg, std=std,
                        Vmax=Vmax, Km=Km, Kcat=Kcat, Kcat_Km=Kcat_Km,
                        SE_Vmax=SE_Vmax, SE_Km=SE_Km,
                        SE_Kcat=SE_Kcat, SE_Kcat_Km=SE_Kcat_Km)

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'':6} {'Km (g/L)':>18} {'Vmax (U/mg)':>22} {'Kcat':>18} {'Kcat/Km':>22}")
print("-" * 90)
for var in ["WT", "Q4"]:
    r = results[var]
    print(f"{var:<6} "
          f"{r['Km']:.3f} ± {r['SE_Km']:.3f}{'':>8} "
          f"{r['Vmax']:.1f} ± {r['SE_Vmax']:.1f}{'':>8} "
          f"{r['Kcat']:.1f} ± {r['SE_Kcat']:.1f}{'':>4} "
          f"{r['Kcat_Km']:.1f} ± {r['SE_Kcat_Km']:.1f}")

# ── Individual plots ──────────────────────────────────────────────────────────
LW, MS = 2, 8

for var in ["WT", "Q4"]:
    r    = results[var]
    c, m = colors[var], markers[var]

    fig, ax = plt.subplots(figsize=(5, 4))

    # Individual replicates (faint)
    for rep in r["reps"]:
        ax.scatter(S, rep, color=c, marker=m, s=25, alpha=0.35, zorder=3)

    # Mean ± std
    ax.errorbar(S, r["avg"], yerr=r["std"],
                color=c, marker=m, markersize=MS,
                linestyle="none", capsize=4, capthick=1.5,
                elinewidth=1.5, zorder=5)

    S_curve = np.linspace(0, S.max() * 1.15, 300)
    ax.plot(S_curve, mm_eq(S_curve, r["Vmax"], r["Km"]),
            color=c, linewidth=LW)

    ann = (f"$K_m$ = {r['Km']:.3f} ± {r['SE_Km']:.3f} g/L\n"
           f"$V_{{max}}$ = {r['Vmax']:.1f} ± {r['SE_Vmax']:.1f} U/mg\n"
           f"$k_{{cat}}$ = {r['Kcat']:.0f} ± {r['SE_Kcat']:.0f}\n"
           f"$k_{{cat}}/K_m$ = {r['Kcat_Km']:.0f} ± {r['SE_Kcat_Km']:.0f}")

    ax.text(0.97, 0.05, ann, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="lightgrey", alpha=0.9))

    ax.set_xlabel("[Xylan] (g/L)", fontsize=14)
    ax.set_ylabel("Specific activity (U/mg)", fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=12, width=1.5, length=4)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/mm_{var}.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved mm_{var}.png")
