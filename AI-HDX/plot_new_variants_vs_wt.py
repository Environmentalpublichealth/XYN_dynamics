"""
ΔHDX bar plots and PyMOL scripts for new XYN variants vs WT apo.
Variants: E112C, E112I, E112Q, N111D
Comparison: variant apo − WT apo
Significance: one-sample t-test across 10 models with BH-FDR correction.
PyMOL colorscale: global symmetric range across all 6 apo variants.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

PRED_DIR = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/predictions"
AF3_BASE = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/AlphaFold3_outputs"
N_RES    = 178
ALPHA    = 0.05

# Per-variant PDB paths (used to generate HDX PDBs and PyMOL scripts)
VARIANT_PDB = {
    "E7":    f"{PRED_DIR}/HDX_PDB/E7_apo_HDX.pdb",
    "Q4":    f"{PRED_DIR}/HDX_PDB/Q4_apo_HDX.pdb",
    "E112Q": f"{PRED_DIR}/HDX_PDB/E112Q_apo_HDX.pdb",
    "N111D": f"{PRED_DIR}/HDX_PDB/N111D_apo_HDX.pdb",
}
VARIANT_AF3_PDB = {
    "E7":    f"{AF3_BASE}/AlphaFold3_outputs_apo/e7_apo/e7_apo_model.pdb",
    "Q4":    f"{AF3_BASE}/AlphaFold3_outputs_apo/q4_apo/q4_apo_model.pdb",
    "E112Q": f"{AF3_BASE}/folds_2026_1XYN/e112q/E112Q.pdb",
    "N111D": f"{AF3_BASE}/folds_2026_1XYN/n111d/N111D.pdb",
}

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
BAR_H = 1.0

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

def make_fig_with_bar(w=8, h=5, title=""):
    _LEG_H = 0.45
    fig, (ax_bar, ax_leg, ax) = plt.subplots(
        3, 1, figsize=(w, h + BAR_H + _LEG_H),
        gridspec_kw={"height_ratios": [BAR_H, _LEG_H, h], "hspace": 0.04})
    fig.patch.set_facecolor("white")
    draw_domain_bar(ax_bar)
    ax_bar.legend(handles=domain_legend_handles(), loc="upper center", ncol=4,
                  fontsize=16, frameon=False, handlelength=1.4, handleheight=1.2,
                  columnspacing=1.2)
    ax_leg.axis("off")
    if title:
        fig.suptitle(title, fontsize=17, fontweight="bold", y=1.01)
    return fig, ax_leg, ax

# ── loaders ───────────────────────────────────────────────────────────────────
def load_pred(path):
    df = pd.read_csv(path, header=0)
    df.columns = ["start","end","sequence","hdx_exp"] + MODEL_COLS + ["average","SD","CI"]
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    for m in MODEL_COLS:
        df[m] = df[m].astype(float) * 100
    return df

# ── significance with BH-FDR correction ──────────────────────────────────────
def peptide_significance_fdr(df_var, df_ref, alpha=ALPHA, verbose=True):
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
    _, qvals, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    merged["delta"]       = deltas
    merged["qval"]        = qvals
    merged["fdr_pass"]    = qvals < alpha                                   # q < 0.05
    merged["significant"] = (qvals < alpha) & (np.abs(deltas) >= 5.0)      # q < 0.05 AND |Δ| ≥ 5%
    if verbose:
        n_fdr = merged["fdr_pass"].sum()
        n_sig = merged["significant"].sum()
        print(f"  {n_fdr}/{len(merged)} FDR-pass, {n_sig} also |Δ|≥5%")
    return merged

# ── residue-level delta + significance ────────────────────────────────────────
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

def residue_delta_sig_highres(df_merged):
    """For each residue use only the shortest covering peptide (highest spatial resolution)."""
    delta_res = np.full(N_RES + 1, np.nan)
    sig_res   = np.zeros(N_RES + 1, dtype=bool)
    pep_len   = np.full(N_RES + 1, np.inf)
    for _, row in df_merged.iterrows():
        length = int(row["end"]) - int(row["start"]) + 1
        for r in range(int(row["start"]), int(row["end"]) + 1):
            if 1 <= r <= N_RES and length < pep_len[r]:
                pep_len[r]   = length
                delta_res[r] = row["delta"]
                sig_res[r]   = row["significant"]
    return delta_res, sig_res

# ── bar plot ──────────────────────────────────────────────────────────────────
def draw_thin_bars(ax, df, bar_h):
    for _, row in df.iterrows():
        if not row["fdr_pass"]:          # q > 0.05 → skip entirely
            continue
        val = row["delta"]
        if not row["significant"]:       # q < 0.05 but |Δ| < 5% → grey
            color, a = "lightgray", 0.6
        else:                            # q < 0.05 AND |Δ| ≥ 5% → colored
            color, a = ("#d73027" if val >= 0 else "#4575b4"), 0.75
        rect = Rectangle((row["start"] - 0.5, val - bar_h / 2),
                          row["end"] - row["start"] + 1, bar_h,
                          facecolor=color, edgecolor="none", alpha=a, zorder=3)
        ax.add_patch(rect)
    ax.autoscale_view()

def style_axes(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=16, color="#d73027")
    ax.tick_params(axis="y", labelcolor="#d73027", labelsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.set_xlabel("Residue Number", fontsize=16)
    ax.set_xlim(1, 178)
    ax.set_xticks(range(25, 179, 25))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.3)

# ── PyMOL residue significance: any covering significant peptide ──────────────
def residue_sig_for_pml(df_merged):
    """A residue is significant if ANY peptide covering it passes both criteria.
    Delta value is averaged over all significant peptides covering the residue."""
    delta_sum = np.zeros(N_RES + 1)
    delta_cnt = np.zeros(N_RES + 1)
    sig_res   = np.zeros(N_RES + 1, dtype=bool)
    for _, row in df_merged.iterrows():
        if not row["significant"]:
            continue
        for r in range(int(row["start"]), int(row["end"]) + 1):
            if 1 <= r <= N_RES:
                delta_sum[r] += row["delta"]
                delta_cnt[r] += 1
                sig_res[r] = True
    delta_res = np.full(N_RES + 1, np.nan)
    for r in range(1, N_RES + 1):
        if delta_cnt[r] > 0:
            delta_res[r] = delta_sum[r] / delta_cnt[r]
    return delta_res, sig_res

# ── PyMOL script ──────────────────────────────────────────────────────────────
def write_pml(variant, obj, delta_res, sig_res, out_path, hdx_lim, pdb_path=None):
    if pdb_path is None:
        pdb_path = VARIANT_PDB.get(variant, f"{PRED_DIR}/HDX_PDB/1xyn_apo_HDX.pdb")
    sig_list = [r for r in range(1, N_RES + 1) if sig_res[r]]
    lines = []
    lines.append(f"load {pdb_path}, {obj}")
    lines.append("hide everything")
    lines.append(f"show cartoon, {obj}")
    lines.append("")
    # all grey by default
    lines.append(f"color gray80, {obj}")
    lines.append(f"alter {obj}, b=0.0")
    # set b-factors only for significant residues
    for r in sig_list:
        val = delta_res[r] if not np.isnan(delta_res[r]) else 0.0
        lines.append(f"alter ({obj} and resi {r}), b={val:.4f}")
    lines.append(f"sort {obj}")
    # apply spectrum only to significant residues
    if sig_list:
        sel = "+".join(str(r) for r in sig_list)
        lines.append(f"spectrum b, blue_white_red, {obj} and resi {sel}, "
                     f"minimum={-hdx_lim:.1f}, maximum={hdx_lim:.1f}")
    lines.append("")
    lines.append("bg_color white")
    lines.append("set cartoon_fancy_helices, 1")
    lines.append("set ray_shadows, 0")
    lines.append(f"zoom {obj}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

# ── variant definitions ───────────────────────────────────────────────────────
# pred_file: untuned prediction; wt_file: matching WT reference
VARIANTS = {
    "E7": {
        "label":     "E7",
        "pred_file": f"{PRED_DIR}/E7_apo_pred.csv",
        "wt_file":   f"{PRED_DIR}/1xyn_apo_pred.csv",
        "mut_sites": [(70, "T70R"), (116, "Q116E")],
        "color":     "#17becf",
    },
    "Q4": {
        "label":     "Q4",
        "pred_file": f"{PRED_DIR}/Q4_apo_pred.csv",
        "wt_file":   f"{PRED_DIR}/1xyn_apo_pred.csv",
        "mut_sites": [(70, "T70R"), (97, "S97C"), (116, "Q116E"), (141, "N141C")],
        "color":     "#e317cf",
    },
    "E112Q": {
        "label":     "E112Q",
        "pred_file": f"{PRED_DIR}/E112Q_pred_102fix.csv",
        "wt_file":   f"{PRED_DIR}/1xyn_apo_pred_102fix.csv",
        "mut_sites": [(112, "E112Q")],
        "color":     "#2ca02c",
    },
    "N111D": {
        "label":     "N111D",
        "pred_file": f"{PRED_DIR}/N111D_pred_102fix.csv",
        "wt_file":   f"{PRED_DIR}/1xyn_apo_pred_102fix.csv",
        "mut_sites": [(111, "N111D")],
        "color":     "#9467bd",
    },
}

hdx_lim = 6.0   # shared PyMOL colorscale: ±6%
print(f"Global PyMOL colorscale: ±{hdx_lim:.0f}%\n")

# ── Generate per-variant HDX PDB files ───────────────────────────────────────
def build_residue_hdx(csv_path):
    df = pd.read_csv(csv_path, header=0)
    df.columns = ["start","end","sequence","hdx_exp"] + MODEL_COLS + ["average","SD","CI"]
    residue_vals = {}
    for _, row in df.iterrows():
        for res in range(int(row["start"]), int(row["end"]) + 1):
            residue_vals.setdefault(res, []).append(float(row["average"]))
    return {res: np.mean(vals) * 100 for res, vals in residue_vals.items()}

def write_bfactor_pdb(pdb_src, res_hdx, out_path):
    with open(pdb_src) as f:
        lines = f.readlines()
    out_lines = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            res_num = int(line[22:26].strip())
            bfactor = res_hdx.get(res_num, -1.0)
            line = line[:60] + f"{bfactor:6.2f}" + line[66:]
        out_lines.append(line)
    with open(out_path, "w") as fout:
        fout.writelines(out_lines)

print("Generating per-variant HDX PDB files...")
for vname, vcfg in VARIANTS.items():
    out_hdx_pdb = VARIANT_PDB[vname]
    af3_src = VARIANT_AF3_PDB[vname]
    res_hdx = build_residue_hdx(vcfg["pred_file"])
    write_bfactor_pdb(af3_src, res_hdx, out_hdx_pdb)
    print(f"  {vname}: {len(res_hdx)} residues -> {out_hdx_pdb}")
print()

# ── Pass 1: compute all diffs, find global y-limit ────────────────────────────
all_diffs = {}
global_max = 0.0
for name, cfg in VARIANTS.items():
    print(f"{name}:")
    df_var = load_pred(cfg["pred_file"])
    df_wt  = load_pred(cfg["wt_file"])
    diff   = peptide_significance_fdr(df_var, df_wt)
    all_diffs[name] = diff
    local_max = max(abs(diff["delta"].max()), abs(diff["delta"].min()))
    global_max = max(global_max, local_max)

global_lim = 15.0
bar_h      = global_lim * 0.10

# ── Pass 2: bar plots + PyMOL scripts ────────────────────────────────────────
for name, cfg in VARIANTS.items():
    diff = all_diffs[name]
    delta_res, sig_res = residue_delta_sig_highres(diff)
    n_sig = sig_res[1:].sum()
    print(f"{name}: {n_sig}/{N_RES} significant residues")

    title = f"{cfg['label']} Apo vs WT Apo — Δ Predicted HDX"
    fig, _, ax = make_fig_with_bar(w=8, h=5, title=title)
    draw_thin_bars(ax, diff, bar_h)
    ax.set_ylim(-global_lim, global_lim)
    style_axes(ax, f"ΔpredHDX {cfg['label']} − WT Apo (%)")

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    mc = cfg["color"]
    for res, label in cfg["mut_sites"]:
        ax.text(res, 0.96, "★", fontsize=28, color=mc,
                ha="center", va="top", transform=trans, zorder=6)
        ax.text(res, 0.88, label, fontsize=16, color=mc,
                ha="center", va="top", rotation=90, transform=trans, zorder=6)

    plt.tight_layout()
    out_png = f"{PRED_DIR}/plot_{name}_apo_vs_wt.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_png}")

    # PyMOL script — use "any covering significant peptide" for residue coloring
    pml_delta, pml_sig = residue_sig_for_pml(diff)
    out_pml = f"{PRED_DIR}/color_hdx_{name}_vs_wt.pml"
    write_pml(name, name.lower(), pml_delta, pml_sig, out_pml, hdx_lim)
    print(f"  Saved {out_pml}\n")

    # significance CSV
    rows = [{"Residue": r,
             "delta_HDX": round(delta_res[r], 3) if not np.isnan(delta_res[r]) else np.nan,
             "significant": bool(sig_res[r])}
            for r in range(1, N_RES + 1)]
    pd.DataFrame(rows).to_csv(f"{PRED_DIR}/hdx_significance_{name}.csv", index=False)
    print(f"  Saved hdx_significance_{name}.csv")

print("\nAll done.")
