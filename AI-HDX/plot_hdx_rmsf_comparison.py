"""
Separate plots per variant:
  plot_WT.png  : WT HDX bars (thin, at value) + RMSF line
  plot_E7.png  : E7 ΔpredHDX bars + ΔRMSF line, mutations T70R Q116E
  plot_Q4.png  : Q4 ΔpredHDX bars + ΔRMSF line, mutations T70R Q116E S97C N141C
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PRED_DIR = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/predictions"
MD_DIR   = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/MDsimulation"

COLORS = {"WT": "#2ca02c", "E7": "#17becf", "Q4": "#e317cf"}

# ── helpers ────────────────────────────────────────────────────────────────────
def load_xvg(path):
    res, vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("@") or not line:
                continue
            parts = line.split()
            res.append(int(float(parts[0])))
            vals.append(float(parts[1]))
    return np.array(res), np.array(vals)

def load_lnpf(path):
    """Load ResID and ln(Pf) from a 3-column .dat file (# comment header)."""
    res, vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            res.append(int(float(parts[0])))
            vals.append(float(parts[1]))
    return np.array(res), np.array(vals)

def load_dfraction(path):
    """Load deuterated fractions at multiple time points from a .dat file.
    Returns: res (array), times (list of floats), fracs (2D array: n_res x n_times).
    Columns are interleaved value/SE pairs; SE is skipped.
    """
    times = None
    res, fracs = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "Times / min:" in line:
                    times = [float(t) for t in line.split("Times / min:")[1].split()]
                continue
            parts = line.split()
            res.append(int(float(parts[0])))
            fracs.append([float(parts[i]) for i in range(1, len(parts), 2)])
    return np.array(res), times, np.array(fracs)

MODEL_COLS = ["m1","m2","m3","m4","m5","m6","m7","m8","m9","m10"]

def load_pred(path):
    df = pd.read_csv(path, header=0)
    df.columns = ["start","end","sequence","hdx_exp"] + MODEL_COLS + ["average","SD","CI"]
    df["start"]   = df["start"].astype(int)
    df["end"]     = df["end"].astype(int)
    df["average"] = df["average"].astype(float) * 100
    for m in MODEL_COLS:
        df[m] = df[m].astype(float) * 100
    return df

def peptide_significance(df_var, df_wt, alpha=0.05):
    """Return merged df with per-peptide delta and FDR-corrected significant flag.
    Uses paired differences across 10 models + Benjamini-Hochberg correction."""
    cols = ["start", "end"] + MODEL_COLS + ["average"]
    var_uniq = df_var[cols].drop_duplicates(subset=["start", "end"])
    wt_uniq  = df_wt [cols].drop_duplicates(subset=["start", "end"])
    merged = pd.merge(var_uniq, wt_uniq, on=["start","end"], suffixes=("_var","_wt"))
    pvals, deltas = [], []
    for _, row in merged.iterrows():
        d = np.array([row[f"{m}_var"] - row[f"{m}_wt"] for m in MODEL_COLS])
        _, p = stats.ttest_1samp(d, 0)
        pvals.append(p)
        deltas.append(d.mean())
    _, qvals, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    merged["delta"]       = deltas
    merged["pval"]        = pvals
    merged["qval"]        = qvals
    merged["significant"] = qvals < alpha
    return merged

def load_truth(path):
    df = pd.read_csv(path, header=0)
    df.columns = ["start","end","sequence","hdx"]
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    df["hdx"]   = df["hdx"].astype(float)
    return df

def draw_thin_bars(ax, df, val_col, bar_h, pos_color, neg_color, alpha=0.75, sig_col=None):
    """Draw thin horizontal bars centered at y=val, spanning [start, end].
    Non-significant bars (sig_col=False) are drawn in gray."""
    for _, row in df.iterrows():
        val = row[val_col]
        if sig_col and not row[sig_col]:
            color = "lightgray"
            a     = 0.6
        else:
            color = pos_color if val >= 0 else neg_color
            a     = alpha
        rect = Rectangle((row["start"] - 0.5, val - bar_h / 2),
                          row["end"] - row["start"] + 1, bar_h,
                          facecolor=color, edgecolor="none", alpha=a, zorder=3)
        ax.add_patch(rect)
    ax.autoscale_view()

def mark_mutations(ax, mutations, ymin, ymax):
    for res, label in mutations:
        ax.axvline(res, color="black", linestyle="--", linewidth=1.2, zorder=5)
        ax.text(res + 0.8, ymax * 0.80, label, fontsize=14, color="black",
                rotation=90, va="top", ha="left")

def style_axes(ax, ax_r, ylabel_l, ylabel_r, lcolor, title=None):
    ax.set_ylabel(ylabel_l, fontsize=16, color=lcolor)
    ax_r.set_ylabel(ylabel_r, fontsize=16, color="black")
    ax.tick_params(axis="y", labelcolor=lcolor, labelsize=15)
    ax_r.tick_params(axis="y", labelcolor="black", labelsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_xlabel("Residue Number", fontsize=16)
    ax.set_xlim(1, 178)
    ax.set_xticks(range(25, 179, 25))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.grid(True, linestyle="--", alpha=0.3)

def style_single_ax(ax, ylabel, ycolor="#d73027"):
    ax.set_ylabel(ylabel, fontsize=16, color=ycolor)
    ax.tick_params(axis="y", labelcolor=ycolor, labelsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_xlabel("Residue Number", fontsize=16)
    ax.set_xlim(1, 178)
    ax.set_xticks(range(25, 179, 25))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.grid(True, linestyle="--", alpha=0.3)

def smooth_arr(res, vals, n_res=178, window=5):
    """Place vals at residue indices, then apply rolling mean."""
    arr = np.full(n_res, np.nan)
    for r, v in zip(res, vals):
        if 1 <= r <= n_res:
            arr[r - 1] = v
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().values

def load_carbon_rmsf_reps(variant, mode="apo"):
    """Load backbone carbon RMSF replicates; return (common_res, rep_dict)."""
    if mode == "holo":
        fnames = {
            "WT": ["rmsf_WTholo.xvg",  "rmsf_WTholo2.xvg"],
            "E7": ["rmsf_E7holo.xvg",  "rmsf_E7holo2.xvg"],
            "Q4": ["rmsf_Q4holo.xvg",  "rmsf_Q4holo2.xvg"],
        }
    else:
        fnames = {
            "WT": ["rmsf_WT.xvg",  "rmsf_WT2.xvg",  "rmsf_WT3.xvg"],
            "E7": ["rmsf_E7.xvg",  "rmsf_E72.xvg",  "rmsf_E73.xvg"],
            "Q4": ["rmsf_Q4.xvg",  "rmsf_Q42.xvg",  "rmsf_Q43.xvg"],
        }
    loaded = [load_xvg(f"{MD_DIR}/{n}") for n in fnames[variant]]
    common = loaded[0][0]
    for res, _ in loaded[1:]:
        common = np.intersect1d(common, res)
    rep_dict = {r: np.array([float(v[res == r][0]) for res, v in loaded]) for r in common}
    return common, rep_dict

def align_zeros(lo_l, hi_l, lo_r, hi_r):
    """Return axis limits (lo_l', hi_l', lo_r', hi_r') with zero at the same
    relative position in both axes while keeping all data in view."""
    lo_l, hi_l = min(lo_l, 0), max(hi_l, 0)
    lo_r, hi_r = min(lo_r, 0), max(hi_r, 0)
    if hi_l == lo_l: lo_l, hi_l = -0.1, 0.1
    if hi_r == lo_r: lo_r, hi_r = -0.1, 0.1
    f = max(-lo_l / (hi_l - lo_l), -lo_r / (hi_r - lo_r))
    f = min(max(f, 0.01), 0.99)
    r_l = max(hi_l / (1 - f), -lo_l / f)
    r_r = max(hi_r / (1 - f), -lo_r / f)
    return -f * r_l, (1 - f) * r_l, -f * r_r, (1 - f) * r_r

# ── amideH RMSF helpers ───────────────────────────────────────────────────────
def load_amideH_reps(variant, n_reps=3, mode="apo"):
    """Load n_reps amideH RMSF xvg files; return (common_res, per_res_mean, rep_dict)."""
    if mode == "holo":
        fnames = [f"{MD_DIR}/rmsf_holo{variant}amideH{i}.xvg" for i in range(1, n_reps + 1)]
    else:
        fnames = [f"{MD_DIR}/rmsf_{variant}amideH{i}.xvg" for i in range(1, n_reps + 1)]
    loaded = [load_xvg(fn) for fn in fnames]
    common = loaded[0][0]
    for res, _ in loaded[1:]:
        common = np.intersect1d(common, res)
    rep_dict = {r: np.array([float(v[res == r][0]) for res, v in loaded]) for r in common}
    mean_vals = np.array([rep_dict[r].mean() for r in common])
    return common, mean_vals, rep_dict

def peptide_ah_mean(start, end, rep_dict):
    """Mean amideH RMSF for [start, end]: per-replicate average over residues, then mean."""
    residues = [r for r in range(start, end + 1) if r in rep_dict]
    if not residues:
        return np.nan
    n_reps = len(next(iter(rep_dict.values())))
    return float(np.mean([np.mean([rep_dict[r][i] for r in residues]) for i in range(n_reps)]))

def build_ah_line(rep_dict, selected, n_res=178):
    """Full-length amideH RMSF: peptide-avg for covered, gap-avg for uncovered gaps."""
    line   = np.full(n_res, np.nan)
    is_cov = np.zeros(n_res, dtype=bool)
    for _, row in selected.iterrows():
        s, e = int(row["start"]), int(row["end"])
        m = peptide_ah_mean(s, e, rep_dict)
        for r in range(s, e + 1):
            if 0 <= r - 1 < n_res:
                line[r - 1] = m
                is_cov[r - 1] = True
    gap_start = None
    for i in range(n_res):
        r = i + 1
        if not is_cov[i] and r in rep_dict:
            if gap_start is None:
                gap_start = r
        else:
            if gap_start is not None:
                gap_end = r - 1 if is_cov[i] else r
                gm = peptide_ah_mean(gap_start, gap_end, rep_dict)
                for gr in range(gap_start, gap_end + 1):
                    if gr - 1 < n_res:
                        line[gr - 1] = gm
                gap_start = None
    if gap_start is not None:
        gm = peptide_ah_mean(gap_start, n_res, rep_dict)
        for gr in range(gap_start, n_res + 1):
            if gr - 1 < n_res:
                line[gr - 1] = gm
    return line, is_cov

def delta_ah_res_stats(rep_dict_var, rep_dict_ref):
    """Per-residue Δ amideH RMSF: mean and SD of paired per-replicate differences."""
    shared = np.array(sorted(set(rep_dict_var.keys()) & set(rep_dict_ref.keys())))
    deltas = np.array([rep_dict_var[r] - rep_dict_ref[r] for r in shared])  # (n_res, n_reps)
    return shared, deltas.mean(axis=1), deltas.std(axis=1)

def peptide_ah_delta_stats(start, end, rep_dict_var, rep_dict_ref):
    """Peptide Δ amideH RMSF: mean/SD of paired per-replicate differences."""
    shared_res = [r for r in range(start, end + 1)
                  if r in rep_dict_var and r in rep_dict_ref]
    if not shared_res:
        return np.nan, np.nan
    n_reps = len(next(iter(rep_dict_var.values())))
    var_reps = np.array([np.mean([rep_dict_var[r][i] for r in shared_res]) for i in range(n_reps)])
    ref_reps = np.array([np.mean([rep_dict_ref[r][i] for r in shared_res]) for i in range(n_reps)])
    d = var_reps - ref_reps
    return d.mean(), d.std()

def build_ah_delta_line(rep_dict_var, rep_dict_ref, selected, n_res=178):
    """Full-length Δ amideH RMSF with SD: peptide-avg covered, gap-avg uncovered."""
    line_mean = np.full(n_res, np.nan)
    line_sd   = np.full(n_res, np.nan)
    is_cov    = np.zeros(n_res, dtype=bool)
    for _, row in selected.iterrows():
        s, e = int(row["start"]), int(row["end"])
        m, sd = peptide_ah_delta_stats(s, e, rep_dict_var, rep_dict_ref)
        for r in range(s, e + 1):
            if 0 <= r - 1 < n_res:
                line_mean[r - 1] = m
                line_sd[r - 1]   = sd
                is_cov[r - 1]    = True
    shared_set = set(rep_dict_var.keys()) & set(rep_dict_ref.keys())
    gap_start = None
    for i in range(n_res):
        r = i + 1
        if not is_cov[i] and r in shared_set:
            if gap_start is None:
                gap_start = r
        else:
            if gap_start is not None:
                gap_end = r - 1 if is_cov[i] else r
                gm, gsd = peptide_ah_delta_stats(gap_start, gap_end, rep_dict_var, rep_dict_ref)
                for gr in range(gap_start, gap_end + 1):
                    if gr - 1 < n_res:
                        line_mean[gr - 1] = gm
                        line_sd[gr - 1]   = gsd
                gap_start = None
    if gap_start is not None:
        gm, gsd = peptide_ah_delta_stats(gap_start, n_res, rep_dict_var, rep_dict_ref)
        for gr in range(gap_start, n_res + 1):
            if gr - 1 < n_res:
                line_mean[gr - 1] = gm
                line_sd[gr - 1]   = gsd
    return line_mean, line_sd, is_cov

# ── Domain bar helpers ────────────────────────────────────────────────────────
_DOM_SEGS = [
    (1,   57,  "#FF7043", "Fingers"),
    (58,  73,  "#26A69A", "Palm"),
    (74,  79,  "#FF7043", "Fingers"),
    (80,  83,  "#26A69A", "Palm"),
    (84,  92,  "#D4E157", "Cord"),
    (93,  110, "#26A69A", "Palm"),
    (111, 121, "#7E57C2", "Thumb"),
    (122, 160, "#26A69A", "Palm"),
    (161, 178, "#FF7043", "Fingers"),
]
_DOM_COLORS = {"Fingers": "#FF7043", "Palm": "#26A69A", "Cord": "#D4E157", "Thumb": "#7E57C2"}
_BAR_H = 1.0    # domain bar subplot (upper half = legend, lower half = bar)
_LEG_H = 0.45   # dedicated data legend row

def draw_domain_bar(ax):
    for start, end, color, _ in _DOM_SEGS:
        ax.add_patch(Rectangle((start - 0.5, 0.02), end - start + 1, 0.35,
                               facecolor=color, edgecolor="white", linewidth=0.5))
    ax.set_xlim(1, 178)
    ax.set_ylim(0, 1)
    ax.axis("off")

def domain_legend_handles():
    import matplotlib.patches as _mp
    return [_mp.Patch(color=_DOM_COLORS[n], alpha=0.85, label=n)
            for n in ["Fingers", "Palm", "Cord", "Thumb"]]

def draw_mutation_highlights(ax):
    for pos in [70, 116]:                          # shared E7 + Q4
        ax.axvspan(pos - 0.5, pos,       color="cyan",    alpha=0.25, zorder=1)
        ax.axvspan(pos,       pos + 0.5, color="magenta", alpha=0.25, zorder=1)
    for pos in [97, 141]:                          # Q4 only
        ax.axvspan(pos - 0.5, pos + 0.5, color="magenta", alpha=0.25, zorder=1)

def make_fig(w, h):
    """2-row: domain bar + data plot. Domain legend in bar."""
    fig, (ax_bar, ax) = plt.subplots(
        2, 1, figsize=(w, h + _BAR_H),
        gridspec_kw={"height_ratios": [_BAR_H, h], "hspace": 0.04})
    draw_domain_bar(ax_bar)
    ax_bar.legend(handles=domain_legend_handles(), loc="upper center", ncol=4,
                  fontsize=16, frameon=False, handlelength=1.4, handleheight=1.2,
                  columnspacing=1.2)
    return fig, ax_bar, ax

def make_fig3(w, h):
    """3-row: domain bar + dedicated data legend row + data plot."""
    fig, (ax_bar, ax_leg, ax) = plt.subplots(
        3, 1, figsize=(w, h + _BAR_H + _LEG_H),
        gridspec_kw={"height_ratios": [_BAR_H, _LEG_H, h], "hspace": 0.04})
    draw_domain_bar(ax_bar)
    ax_bar.legend(handles=domain_legend_handles(), loc="upper center", ncol=4,
                  fontsize=16, frameon=False, handlelength=1.4, handleheight=1.2,
                  columnspacing=1.2)
    ax_leg.axis("off")
    return fig, ax_bar, ax_leg, ax

# ── load data ──────────────────────────────────────────────────────────────────
truth_wt = load_truth(f"{PRED_DIR}/1xyn_apo_truth.csv")
pred_wt  = load_pred(f"{PRED_DIR}/1xyn_apo_pred.csv")
pred_e7  = load_pred(f"{PRED_DIR}/E7_apo_pred.csv")
pred_q4  = load_pred(f"{PRED_DIR}/Q4_apo_pred.csv")

dfrac_res_wt, dfrac_times, dfrac_wt = load_dfraction(f"{MD_DIR}/WT1_Dfraction.dat")
dfrac_early = dfrac_wt[:, dfrac_times.index(0.167)]
lnpf_res_wt, lnpf_wt = load_lnpf(f"{MD_DIR}/WT1_lnPf.dat")
lnpf_res_e7, lnpf_e7 = load_lnpf(f"{MD_DIR}/E7_lnPf.dat")
lnpf_res_q4, lnpf_q4 = load_lnpf(f"{MD_DIR}/Q4_lnPf.dat")

def delta_lnpf(res_var, vals_var, res_wt, vals_wt):
    """Δln(P) = variant − WT for shared residue IDs."""
    shared = np.intersect1d(res_var, res_wt)
    idx_var = np.isin(res_var, shared)
    idx_wt  = np.isin(res_wt,  shared)
    return shared, vals_var[idx_var] - vals_wt[idx_wt]

dlnpf_res_e7, dlnpf_e7 = delta_lnpf(lnpf_res_e7, lnpf_e7, lnpf_res_wt, lnpf_wt)
dlnpf_res_q4, dlnpf_q4 = delta_lnpf(lnpf_res_q4, lnpf_q4, lnpf_res_wt, lnpf_wt)
dlnpf_res_q4e7, dlnpf_q4e7 = delta_lnpf(lnpf_res_q4, lnpf_q4, lnpf_res_e7, lnpf_e7)

diff_q4e7 = peptide_significance(pred_q4, pred_e7)

rmsf_res_wt, rmsf_wt = load_xvg(f"{MD_DIR}/rmsf_WT.xvg")
rmsf_res_e7, rmsf_e7 = load_xvg(f"{MD_DIR}/rmsf_E7.xvg")
rmsf_res_q4, rmsf_q4 = load_xvg(f"{MD_DIR}/rmsf_Q4.xvg")

delta_rmsf_e7 = rmsf_e7 - rmsf_wt
delta_rmsf_q4 = rmsf_q4 - rmsf_wt

diff_e7 = peptide_significance(pred_e7, pred_wt)
diff_q4 = peptide_significance(pred_q4, pred_wt)

# ── load amideH RMSF replicates ────────────────────────────────────────────────
ah_res_wt, _, rep_ah_wt = load_amideH_reps("WT")
ah_res_e7, _, rep_ah_e7 = load_amideH_reps("E7")
ah_res_q4, _, rep_ah_q4 = load_amideH_reps("Q4")

# ── load carbon RMSF replicates ────────────────────────────────────────────────
cr_res_wt, rep_cr_wt = load_carbon_rmsf_reps("WT")
cr_res_e7, rep_cr_e7 = load_carbon_rmsf_reps("E7")
cr_res_q4, rep_cr_q4 = load_carbon_rmsf_reps("Q4")

res_all = np.arange(1, 179)

# ── Plot 1: WT HDX bars + RMSF (3 replicates mean ± SD) ──────────────────────
fig, _, ax = make_fig(12, 5)
ax_r = ax.twinx()

bar_h_wt = 2.5

draw_thin_bars(ax, truth_wt, "hdx", bar_h_wt, "#d73027", "#4575b4", alpha=0.65)

# 3 individual replicates from rep_cr_wt (convert nm → Å)
cr_res  = np.array(sorted(rep_cr_wt.keys()))
cr_mat  = np.array([rep_cr_wt[r] for r in cr_res]) * 10.0   # nm → Å
rep_colors = ["black", "#444444", "#888888"]
for i in range(cr_mat.shape[1]):
    ax_r.plot(cr_res, cr_mat[:, i], color=rep_colors[i],
              linewidth=1.2, alpha=0.75, zorder=5)

ax.set_ylim(0, 105)
ax_r.set_ylim(0, None)
style_axes(ax, ax_r, "HDX Rate (%)", "RMSF (Å)", "#d73027",
           "WT Apo — Experimental HDX Rate & RMSF")

plt.tight_layout()
plt.savefig(f"{PRED_DIR}/plot_WT_hdx_rmsf.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_WT_hdx_rmsf.png")

# ── Plot 1b: WT (100−HDX) bars + lnPf scatter ────────────────────────────────
fig, _, ax = make_fig(12, 5)
ax_r = ax.twinx()

_truth_wt_prot = truth_wt.copy()
_truth_wt_prot["hdx"] = 100 - _truth_wt_prot["hdx"]

draw_thin_bars(ax, _truth_wt_prot, "hdx", bar_h_wt, "#d73027", "#4575b4", alpha=0.65)
ax_r.scatter(lnpf_res_wt, lnpf_wt, color="black", s=30, alpha=0.7, zorder=5, linewidths=0)

ax.set_ylim(0, 105)
ax_r.set_ylim(0, max(lnpf_wt) * 1.1)
style_axes(ax, ax_r, "(100 − HDX) (%)", "ln(P)", "#d73027",
           "WT Apo — Experimental HDX Rate & ln(P)")

plt.tight_layout()
plt.savefig(f"{PRED_DIR}/plot_WT_hdx_lnpf.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_WT_hdx_lnpf.png")

# ── Plot 1c: WT HDX bars + −lnPf scatter ─────────────────────────────────────
fig, _, ax = make_fig(8, 5)
ax_r = ax.twinx()

neg_lnpf = -lnpf_wt

draw_mutation_highlights(ax)
draw_thin_bars(ax, truth_wt, "hdx", bar_h_wt, "#d73027", "#4575b4", alpha=0.65)
ax_r.scatter(lnpf_res_wt, neg_lnpf, color="black", s=30, alpha=0.7, zorder=5, linewidths=0)

ax.set_ylim(0, 105)
ax_r.set_ylim(min(neg_lnpf) * 1.1, 0)
style_axes(ax, ax_r, "HDX Rate (%)", "−ln(P)", "#d73027",
           "WT Apo — Experimental HDX Rate & −ln(P)")

plt.tight_layout()
plt.savefig(f"{PRED_DIR}/plot_WT_hdx_neg_lnpf.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_WT_hdx_neg_lnpf.png")

BAR_H_FRAC = 0.044   # bar height as fraction of y-range, matched to Q4 appearance

# ── Plot 2a: E7 ΔpredHDX bars ─────────────────────────────────────────────────
fig, _, ax = make_fig(8, 5)
_d2 = diff_e7["delta"]; _p2 = (_d2.max() - _d2.min()) * 0.15
_ylo2, _yhi2 = min(_d2.min() - _p2, 0), max(_d2.max() + _p2, 0)
draw_thin_bars(ax, diff_e7.rename(columns={"delta":"hdx"}), "hdx",
               (_yhi2 - _ylo2) * BAR_H_FRAC, "#d73027", "#4575b4", alpha=0.75, sig_col="significant")
ax.set_ylim(_ylo2, _yhi2)
style_single_ax(ax, "ΔpredHDX E7−WT (%)")
mark_mutations(ax, [(70, "T70R"), (116, "Q116E")], *ax.get_ylim())
plt.subplots_adjust(hspace=0.04)
plt.savefig(f"{PRED_DIR}/plot_E7_diff_hdx.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_E7_diff_hdx.png")


# ── Plot 3a: Q4 ΔpredHDX bars ─────────────────────────────────────────────────
fig, _, ax = make_fig(8, 5)
_d3 = diff_q4["delta"]; _p3 = (_d3.max() - _d3.min()) * 0.15
_ylo3, _yhi3 = min(_d3.min() - _p3, 0), max(_d3.max() + _p3, 0)
draw_thin_bars(ax, diff_q4.rename(columns={"delta":"hdx"}), "hdx",
               (_yhi3 - _ylo3) * BAR_H_FRAC, "#d73027", "#4575b4", alpha=0.75, sig_col="significant")
ax.set_ylim(_ylo3, _yhi3)
style_single_ax(ax, "ΔpredHDX Q4−WT (%)")
mark_mutations(ax, [(70, "T70R"), (97, "S97C"), (116, "Q116E"), (141, "N141C")], *ax.get_ylim())
plt.subplots_adjust(hspace=0.04)
plt.savefig(f"{PRED_DIR}/plot_Q4_diff_hdx.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_Q4_diff_hdx.png")


# ── Plot: Q4 vs E7 ΔpredHDX bars ─────────────────────────────────────────────
fig, _, ax = make_fig(8, 5)
_dqe = diff_q4e7["delta"]; _pqe = (_dqe.max() - _dqe.min()) * 0.15
_yloqe, _yhiqe = min(_dqe.min() - _pqe, 0), max(_dqe.max() + _pqe, 0)
draw_thin_bars(ax, diff_q4e7.rename(columns={"delta": "hdx"}), "hdx",
               (_yhiqe - _yloqe) * BAR_H_FRAC, "#d73027", "#4575b4", alpha=0.75, sig_col="significant")
ax.set_ylim(_yloqe, _yhiqe)
style_single_ax(ax, "ΔpredHDX Q4−E7 (%)")
mark_mutations(ax, [(97, "S97C"), (141, "N141C")], *ax.get_ylim())
plt.tight_layout()
plt.savefig(f"{PRED_DIR}/plot_Q4_vs_E7_diff_hdx.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_Q4_vs_E7_diff_hdx.png")

# ── Plot: amideH RMSF comparison WT / E7 / Q4 ────────────────────────────────
fig, _, ax_leg, ax = make_fig3(8, 5)
draw_mutation_highlights(ax)
for res_arr, rep_d, color, label in [
        (ah_res_wt, rep_ah_wt, COLORS["WT"], "WT"),
        (ah_res_e7, rep_ah_e7, COLORS["E7"], "E7"),
        (ah_res_q4, rep_ah_q4, COLORS["Q4"], "Q4"),
]:
    mean_v = np.array([rep_d[r].mean() for r in res_arr])
    sd_v   = np.array([rep_d[r].std()  for r in res_arr])
    sm_m   = smooth_arr(res_arr, mean_v)
    sm_sd  = smooth_arr(res_arr, sd_v)
    ax.fill_between(res_all, sm_m - sm_sd, sm_m + sm_sd, color=color, alpha=0.20)
    ax.plot(res_all, sm_m, color=color, linewidth=1.8, alpha=0.85, label=label)
ax.set_xlim(1, 178); ax.set_xticks(range(25, 179, 25))
ax.set_xlabel("Residue Number", fontsize=16)
ax.set_ylabel("Amide H RMSF (nm)", fontsize=16)
ax.tick_params(axis="both", labelsize=15)
ax.set_ylim(0, None); ax.grid(True, linestyle="--", alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
ax_leg.legend(handles, labels, loc="center", ncol=3, fontsize=15, framealpha=0.9)
plt.subplots_adjust(hspace=0.04)
plt.savefig(f"{PRED_DIR}/plot_amideH_rmsf_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_amideH_rmsf_comparison.png")

# ── Plot: carbon RMSF comparison WT / E7 / Q4 ────────────────────────────────
fig, _, ax_leg, ax = make_fig3(8, 5)
draw_mutation_highlights(ax)
for res_arr, rep_d, color, label in [
        (cr_res_wt, rep_cr_wt, COLORS["WT"], "WT"),
        (cr_res_e7, rep_cr_e7, COLORS["E7"], "E7"),
        (cr_res_q4, rep_cr_q4, COLORS["Q4"], "Q4"),
]:
    mean_v = np.array([rep_d[r].mean() for r in res_arr])
    sd_v   = np.array([rep_d[r].std()  for r in res_arr])
    sm_m   = smooth_arr(res_arr, mean_v)
    sm_sd  = smooth_arr(res_arr, sd_v)
    ax.fill_between(res_all, sm_m - sm_sd, sm_m + sm_sd, color=color, alpha=0.20)
    ax.plot(res_all, sm_m, color=color, linewidth=1.8, alpha=0.85, label=label)
ax.set_xlim(1, 178); ax.set_xticks(range(25, 179, 25))
ax.set_xlabel("Residue Number", fontsize=16)
ax.set_ylabel("Carbon RMSF (nm)", fontsize=16)
ax.tick_params(axis="both", labelsize=15)
ax.set_ylim(0, None); ax.grid(True, linestyle="--", alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
ax_leg.legend(handles, labels, loc="center", ncol=3, fontsize=15, framealpha=0.9)
plt.subplots_adjust(hspace=0.04)
plt.savefig(f"{PRED_DIR}/plot_carbon_rmsf_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_carbon_rmsf_comparison.png")

# ── Plot: lnPf smoothed line comparison WT / E7 / Q4 ─────────────────────────
sm_wt = smooth_arr(lnpf_res_wt, lnpf_wt)
sm_e7 = smooth_arr(lnpf_res_e7, lnpf_e7)
sm_q4 = smooth_arr(lnpf_res_q4, lnpf_q4)

fig, _, ax_leg, ax = make_fig3(8, 5)
draw_mutation_highlights(ax)

ax.plot(res_all, sm_wt, color=COLORS["WT"], linewidth=1.8, alpha=0.85, label="WT")
ax.plot(res_all, sm_e7, color=COLORS["E7"], linewidth=1.8, alpha=0.85, label="E7")
ax.plot(res_all, sm_q4, color=COLORS["Q4"], linewidth=1.8, alpha=0.85, label="Q4")

ax.set_xlim(1, 178)
ax.set_xticks(range(25, 179, 25))
ax.set_xlabel("Residue Number", fontsize=16)
ax.set_ylabel("ln(P)", fontsize=16)
ax.tick_params(axis="both", labelsize=15)
ax.set_ylim(0, max(np.nanmax(sm_wt), np.nanmax(sm_e7), np.nanmax(sm_q4)) * 1.1)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
ax.grid(True, linestyle="--", alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
ax_leg.legend(handles, labels, loc="center", ncol=3, fontsize=15, framealpha=0.9)

plt.subplots_adjust(hspace=0.04)
plt.savefig(f"{PRED_DIR}/plot_lnpf_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_lnpf_comparison.png")

# ── load holo RMSF replicates (2 reps each) ───────────────────────────────────
ah_res_wt_h, _, rep_ah_wt_h = load_amideH_reps("WT", n_reps=2, mode="holo")
ah_res_e7_h, _, rep_ah_e7_h = load_amideH_reps("E7", n_reps=2, mode="holo")
ah_res_q4_h, _, rep_ah_q4_h = load_amideH_reps("Q4", n_reps=2, mode="holo")

cr_res_wt_h, rep_cr_wt_h = load_carbon_rmsf_reps("WT", mode="holo")
cr_res_e7_h, rep_cr_e7_h = load_carbon_rmsf_reps("E7", mode="holo")
cr_res_q4_h, rep_cr_q4_h = load_carbon_rmsf_reps("Q4", mode="holo")

# ── Plot: amideH RMSF comparison holo WT / E7 / Q4 ───────────────────────────
fig, _, ax_leg, ax = make_fig3(8, 5)
draw_mutation_highlights(ax)
for res_arr, rep_d, color, label in [
        (ah_res_wt_h, rep_ah_wt_h, COLORS["WT"], "WT"),
        (ah_res_e7_h, rep_ah_e7_h, COLORS["E7"], "E7"),
        (ah_res_q4_h, rep_ah_q4_h, COLORS["Q4"], "Q4"),
]:
    mean_v = np.array([rep_d[r].mean() for r in res_arr])
    sd_v   = np.array([rep_d[r].std()  for r in res_arr])
    sm_m   = smooth_arr(res_arr, mean_v)
    sm_sd  = smooth_arr(res_arr, sd_v)
    ax.fill_between(res_all, sm_m - sm_sd, sm_m + sm_sd, color=color, alpha=0.20)
    ax.plot(res_all, sm_m, color=color, linewidth=1.8, alpha=0.85, label=label)
ax.set_xlim(1, 178); ax.set_xticks(range(25, 179, 25))
ax.set_xlabel("Residue Number", fontsize=16)
ax.set_ylabel("Amide H RMSF (nm)", fontsize=16)
ax.tick_params(axis="both", labelsize=15)
ax.set_ylim(0, None); ax.grid(True, linestyle="--", alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
ax_leg.legend(handles, labels, loc="center", ncol=3, fontsize=15, framealpha=0.9)
plt.subplots_adjust(hspace=0.04)
plt.savefig(f"{PRED_DIR}/plot_amideH_rmsf_comparison_holo.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_amideH_rmsf_comparison_holo.png")

# ── Plot: carbon RMSF comparison holo WT / E7 / Q4 ───────────────────────────
fig, _, ax_leg, ax = make_fig3(8, 5)
draw_mutation_highlights(ax)
for res_arr, rep_d, color, label in [
        (cr_res_wt_h, rep_cr_wt_h, COLORS["WT"], "WT"),
        (cr_res_e7_h, rep_cr_e7_h, COLORS["E7"], "E7"),
        (cr_res_q4_h, rep_cr_q4_h, COLORS["Q4"], "Q4"),
]:
    mean_v = np.array([rep_d[r].mean() for r in res_arr])
    sd_v   = np.array([rep_d[r].std()  for r in res_arr])
    sm_m   = smooth_arr(res_arr, mean_v)
    sm_sd  = smooth_arr(res_arr, sd_v)
    ax.fill_between(res_all, sm_m - sm_sd, sm_m + sm_sd, color=color, alpha=0.20)
    ax.plot(res_all, sm_m, color=color, linewidth=1.8, alpha=0.85, label=label)
ax.set_xlim(1, 178); ax.set_xticks(range(25, 179, 25))
ax.set_xlabel("Residue Number", fontsize=16)
ax.set_ylabel("Carbon RMSF (nm)", fontsize=16)
ax.tick_params(axis="both", labelsize=15)
ax.set_ylim(0, None); ax.grid(True, linestyle="--", alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
ax_leg.legend(handles, labels, loc="center", ncol=3, fontsize=15, framealpha=0.9)
plt.subplots_adjust(hspace=0.04)
plt.savefig(f"{PRED_DIR}/plot_carbon_rmsf_comparison_holo.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot_carbon_rmsf_comparison_holo.png")
