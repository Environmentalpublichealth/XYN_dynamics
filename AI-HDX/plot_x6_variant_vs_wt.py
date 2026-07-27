"""
ΔHDX plots comparing X6-bound variant vs X6-bound WT (tuned predictions).
  D2 X6 vs WT X6
  Q4 X6 vs WT X6

Significance: one-sample t-test across 10 models, FDR-corrected (Benjamini-Hochberg).
Stars mark mutation sites.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PRED_DIR  = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/predictions"
MD_DIR    = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/MDsimulation"
PDB_BASE  = f"{PRED_DIR}/HDX_PDB/1xyn_apo_HDX.pdb"
N_RES     = 178
ALPHA     = 0.05
HDX_LIM   = 15.0
NM_TO_ANG = 10.0
RMSF_COLOR = "#D4930A"   # amber for amideH RMSF

MODEL_COLS = ["m1","m2","m3","m4","m5","m6","m7","m8","m9","m10"]

# ── domain bar ────────────────────────────────────────────────────────────────
DOM_SEGS = [
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
DOM_COLORS = {"Fingers": "#FF7043", "Palm": "#26A69A", "Cord": "#D4E157", "Thumb": "#7E57C2"}
BAR_H = 1.5

def draw_domain_bar(ax):
    for start, end, color, _ in DOM_SEGS:
        ax.add_patch(plt.Rectangle((start - 0.5, 0.02), end - start + 1, 0.35,
                                   facecolor=color, edgecolor="white", linewidth=0.5))
    ax.set_xlim(1, 178)
    ax.set_ylim(0, 1)
    ax.axis("off")

def domain_legend_handles():
    import matplotlib.patches as _mp
    return [_mp.Patch(color=DOM_COLORS[n], alpha=0.85, label=n)
            for n in ["Fingers", "Palm", "Cord", "Thumb"]]

def make_fig_with_bar(w=8, h=5, title="", legend_row=False):
    _LEG_H = 0.45
    if legend_row:
        fig, (ax_bar, ax_leg, ax) = plt.subplots(
            3, 1, figsize=(w, h + BAR_H + _LEG_H),
            gridspec_kw={"height_ratios": [BAR_H, _LEG_H, h], "hspace": 0.04})
        ax_leg.axis("off")
    else:
        fig, (ax_bar, ax) = plt.subplots(
            2, 1, figsize=(w, h + BAR_H),
            gridspec_kw={"height_ratios": [BAR_H, h], "hspace": 0.04})
        ax_leg = None
    fig.patch.set_facecolor("white")
    draw_domain_bar(ax_bar)
    ax_bar.legend(handles=domain_legend_handles(), loc="upper center", ncol=4,
                  fontsize=22, frameon=False, handlelength=1.4, handleheight=1.2,
                  columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    if title:
        fig.suptitle(title, fontsize=17, fontweight="bold", y=1.01)
    return fig, ax_leg, ax

# ── amideH RMSF helpers ───────────────────────────────────────────────────────
def load_xvg(path):
    res, vals = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line.startswith("@") or not line:
                continue
            parts = line.split()
            res.append(int(float(parts[0])))
            vals.append(float(parts[1]) * NM_TO_ANG)
    return np.array(res), np.array(vals)

def load_rmsf_mean(paths):
    """Average replicates; return {residue: mean_Å}."""
    store = {}
    for p in paths:
        r, v = load_xvg(p)
        for ri, vi in zip(r, v):
            store.setdefault(ri, []).append(vi)
    return {ri: np.mean(vs) for ri, vs in store.items()}

def peptide_rmsf_diff(start, end, rmsf_var, rmsf_ref):
    """Mean RMSF difference (var − ref) over a peptide window."""
    residues = [r for r in range(int(start), int(end)+1)
                if r in rmsf_var and r in rmsf_ref]
    if not residues:
        return np.nan
    return np.mean([rmsf_var[r] - rmsf_ref[r] for r in residues])

# ── loaders ───────────────────────────────────────────────────────────────────
def load_pred(path):
    df = pd.read_csv(path, header=0)
    df.columns = ["start","end","sequence","hdx_exp"] + MODEL_COLS + ["average","SD","CI"]
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    for m in MODEL_COLS:
        df[m] = df[m].astype(float) * 100
    return df

# ── significance with FDR correction ─────────────────────────────────────────
def peptide_significance_fdr(df_var, df_ref, alpha=ALPHA):
    merged = pd.merge(
        df_var[["start","end"] + MODEL_COLS].drop_duplicates(["start","end"]),
        df_ref[["start","end"] + MODEL_COLS].drop_duplicates(["start","end"]),
        on=["start","end"], suffixes=("_var","_ref")
    )
    deltas, pvals = [], []
    for _, row in merged.iterrows():
        d = np.array([row[f"{m}_var"] - row[f"{m}_ref"] for m in MODEL_COLS])
        _, p = stats.ttest_1samp(d, 0)
        deltas.append(d.mean())
        pvals.append(p)

    # Benjamini-Hochberg FDR correction
    _, qvals, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    merged["delta"]       = deltas
    merged["pval"]        = pvals
    merged["qval"]        = qvals
    merged["significant"] = qvals < alpha
    n_sig = merged["significant"].sum()
    print(f"  {n_sig}/{len(merged)} peptides significant (BH-FDR q<{alpha})")
    return merged

# ── residue-level delta + significance ───────────────────────────────────────
def residue_delta_sig(df_merged):
    delta_sum = np.zeros(N_RES + 1)
    delta_cnt = np.zeros(N_RES + 1)
    sig_res   = np.zeros(N_RES + 1, dtype=bool)
    for _, row in df_merged.iterrows():
        for r in range(int(row["start"]), int(row["end"]) + 1):
            if 1 <= r <= N_RES:
                delta_sum[r] += row["delta"]
                delta_cnt[r] += 1
                if row["significant"]:
                    sig_res[r] = True
    delta_res = np.full(N_RES + 1, np.nan)
    for r in range(1, N_RES + 1):
        if delta_cnt[r] > 0:
            delta_res[r] = delta_sum[r] / delta_cnt[r]
    return delta_res, sig_res

# ── PyMOL script writer ───────────────────────────────────────────────────────
def write_pml(variant, obj, delta_res, sig_res, out_path, hdx_lim=HDX_LIM):
    lines = []
    lines.append(f"load {PDB_BASE}, {obj}")
    lines.append("hide everything")
    lines.append(f"show cartoon, {obj}")
    lines.append("")
    for r in range(1, N_RES + 1):
        val = delta_res[r] if not np.isnan(delta_res[r]) else 0.0
        lines.append(f"alter ({obj} and resi {r}), b={val:.4f}")
    lines.append(f"sort {obj}")
    lines.append(f"spectrum b, blue_white_red, {obj}, minimum={-hdx_lim:.1f}, maximum={hdx_lim:.1f}")
    lines.append("")
    lines.append("bg_color white")
    lines.append("set cartoon_fancy_helices, 1")
    lines.append("set ray_shadows, 0")
    lines.append(f"zoom {obj}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved {out_path}")

# ── bar plot helper ───────────────────────────────────────────────────────────
def draw_thin_bars(ax, df, bar_h, sig_only=False, threshold=None):
    """Draw peptide delta bars.
    sig_only: skip non-significant peptides entirely.
    threshold: if set, only color significant bars with |delta| > threshold; rest are grey.
    """
    for _, row in df.iterrows():
        val = row["delta"]
        if not row["significant"]:
            if sig_only:
                continue
            color, a = "lightgray", 0.6
        else:
            if threshold is not None and abs(val) <= threshold:
                color, a = "lightgray", 0.6
            else:
                color, a = ("#d73027" if val >= 0 else "#4575b4"), 0.75
        rect = Rectangle((row["start"] - 0.5, val - bar_h / 2),
                          row["end"] - row["start"] + 1, bar_h,
                          facecolor=color, edgecolor="none", alpha=a, zorder=3)
        ax.add_patch(rect)
    ax.autoscale_view()

def style_axes(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=26, color="#d73027")
    ax.tick_params(axis="y", labelcolor="#d73027", labelsize=22)
    ax.tick_params(axis="x", labelsize=22)
    ax.set_xlabel("Residue Number", fontsize=26)
    ax.set_xlim(1, 178)
    ax.set_xticks(range(25, 179, 25))
    ax.axhline(0,  color="gray", linewidth=0.8)
    ax.axhline( 5, color="gray", linewidth=0.8, linestyle="--")
    ax.axhline(-5, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, linestyle="--", alpha=0.3)

from matplotlib.transforms import blended_transform_factory

MUT_COLORS = {"D2": "#17becf", "Q4": "#e317cf"}

MUT_SITES = {
    "D2": [(70, "T70R"), (116, "Q116E")],
    "Q4": [(70, "T70R"), (97, "S97C"), (116, "Q116E"), (141, "N141C")],
}


def run_comparisons(comparisons, out_suffix, ylabel_fmt, scale_only=None,
                    sig_only=False, threshold=None, rmsf_dict=None,
                    ylim_override=None):
    """Compute diffs, auto-scale y, and save one figure per variant.
    rmsf_dict: {variant: (rmsf_var_mean, rmsf_ref_mean)} for amideH RMSF overlay.
    """
    import matplotlib.patches as _mp
    import matplotlib.lines as _ml

    all_diffs = {}
    global_max = 0.0
    for variant, (df_var, df_ref, title) in comparisons.items():
        print(f"{variant}:")
        diff = peptide_significance_fdr(df_var, df_ref)
        all_diffs[variant] = diff
        local_max = max(abs(diff["delta"].max()), abs(diff["delta"].min()))
        global_max = max(global_max, local_max)

    for _name, (_df_var, _df_ref) in (scale_only or {}).items():
        _merged = pd.merge(
            _df_var[["start","end"] + MODEL_COLS],
            _df_ref[["start","end"] + MODEL_COLS],
            on=["start","end"], suffixes=("_var","_ref")
        )
        _deltas = [np.mean([row[f"{m}_var"] - row[f"{m}_ref"] for m in MODEL_COLS])
                   for _, row in _merged.iterrows()]
        global_max = max(global_max, max(abs(d) for d in _deltas))

    global_lim = 15.0
    bar_h      = global_lim * 0.10
    use_legend = rmsf_dict is not None

    for variant, (df_var, df_ref, title) in comparisons.items():
        diff = all_diffs[variant]
        fig, ax_leg, ax = make_fig_with_bar(w=8, h=5, title=title,
                                            legend_row=use_legend)
        draw_thin_bars(ax, diff, bar_h, sig_only=sig_only, threshold=threshold)
        ylim = (ylim_override or {}).get(variant, global_lim)
        ax.set_ylim(-ylim, ylim)
        style_axes(ax, ylabel_fmt.format(variant=variant))

        # ── amideH RMSF overlay (per-residue line, avg of 2 holo reps) ──────
        if rmsf_dict and variant in rmsf_dict:
            rmsf_var, rmsf_ref = rmsf_dict[variant]
            ax_r = ax.twinx()
            res_all = np.arange(1, N_RES + 1)
            rmsf_diff_arr = np.array([
                rmsf_var.get(r, np.nan) - rmsf_ref.get(r, np.nan)
                for r in res_all
            ])
            ax_r.plot(res_all, rmsf_diff_arr, color=RMSF_COLOR,
                      linewidth=1.5, alpha=0.9, zorder=4)
            ax_r.fill_between(res_all, rmsf_diff_arr, 0,
                              where=(~np.isnan(rmsf_diff_arr)),
                              color=RMSF_COLOR, alpha=0.20, zorder=3)
            ax_r.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
            rmsf_abs = np.nanmax(np.abs(rmsf_diff_arr))
            ax_r.set_ylim(-rmsf_abs * 1.5, rmsf_abs * 1.5)
            ax_r.set_xlim(1, 178)
            ax_r.set_ylabel("ΔAmide H RMSF (Å)", color=RMSF_COLOR, fontsize=14)
            ax_r.tick_params(axis="y", labelcolor=RMSF_COLOR, labelsize=13)
            ax_r.set_autoscale_on(False)
            ax.set_ylim(-ylim, ylim)

        # ── mutation markers (bigger) ────────────────────────────────────────
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        mc = MUT_COLORS[variant]
        for res, label in MUT_SITES[variant]:
            ax.text(res, 0.97, "★", fontsize=36, color=mc,
                    ha="center", va="top", transform=trans, zorder=6)
            ax.text(res, 0.88, label, fontsize=24, color=mc,
                    ha="center", va="top", rotation=90, transform=trans, zorder=6)

        # ── legend ───────────────────────────────────────────────────────────
        if use_legend:
            hdx_pos  = _mp.Patch(facecolor="#d73027", alpha=0.75, label="ΔpredHDX > +5%")
            hdx_neg  = _mp.Patch(facecolor="#4575b4", alpha=0.75, label="ΔpredHDX < −5%")
            hdx_grey = _mp.Patch(facecolor="lightgray", alpha=0.6, label="ΔpredHDX ±5%")
            rmsf_h   = _ml.Line2D([], [], color=RMSF_COLOR, linewidth=2,
                                   label="ΔAmide H RMSF per residue (right)")
            ax_leg.legend(handles=[hdx_pos, hdx_neg, hdx_grey, rmsf_h],
                          loc="center", ncol=4, fontsize=18, framealpha=0.9)

        plt.subplots_adjust(hspace=0.04)
        out = f"{PRED_DIR}/plot_{variant}_{out_suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")

    return all_diffs


# ── X6 comparisons (untuned predictions) ─────────────────────────────────────
pred_wt_x6  = load_pred(f"{PRED_DIR}/1xyn_xylohexaose_pred.csv")
pred_d2_x6  = load_pred(f"{PRED_DIR}/E7_xylohexaose_pred.csv")
pred_q4_x6  = load_pred(f"{PRED_DIR}/Q4_xylohexaose_pred.csv")

# ── holo amideH RMSF (2 reps each) ───────────────────────────────────────────
rmsf_wt_holo = load_rmsf_mean([f"{MD_DIR}/rmsf_holoWTamideH{i}.xvg" for i in range(1, 3)])
rmsf_e7_holo = load_rmsf_mean([f"{MD_DIR}/rmsf_holoE7amideH{i}.xvg" for i in range(1, 3)])
rmsf_q4_holo = load_rmsf_mean([f"{MD_DIR}/rmsf_holoQ4amideH{i}.xvg" for i in range(1, 3)])

run_comparisons(
    {
        "D2": (pred_d2_x6, pred_wt_x6, "D2 X6 vs WT X6 — Δ Predicted HDX"),
        "Q4": (pred_q4_x6, pred_wt_x6, "Q4 X6 vs WT X6 — Δ Predicted HDX"),
    },
    out_suffix="x6_vs_wt_x6",
    ylabel_fmt="ΔpredHDX {variant} X6 − WT X6 (%)",
    sig_only=True,
    threshold=5.0,
    ylim_override={"Q4": 17.0},
)
