"""
Visualization of MD simulation RMSD and RMSF for WT, E7, Q4.
Colors match the HDX prediction variant colors from visualize_predictions.py.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Same variant colors as apo_colors in visualize_predictions.py
COLORS = {
    "WT": "#2ca02c",   # forest green
    "E7": "#17becf",   # teal cyan
    "Q4": "#e317cf",   # magenta
}
VARIANTS = ["WT", "E7", "Q4"]


def read_xvg(filepath):
    """Parse a GROMACS .xvg file, skipping comment/metadata lines."""
    x, y = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            parts = line.split()
            x.append(float(parts[0]))
            y.append(float(parts[1]))
    return np.array(x), np.array(y)


def style_ax(ax):
    """Apply publication-style formatting matching the reference figure."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(direction="in", length=4, width=1.2, labelsize=12)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


# ── Replicate file lists ──────────────────────────────────────────────────────
WINDOW = 500  # rolling mean window (frames)

APO_RMSD_REPS = {
    "WT": ["rmsd_WT.xvg",  "rmsd_WT2.xvg",  "rmsd_WT3.xvg"],
    "E7": ["rmsd_E7.xvg",  "rmsd_E72.xvg",  "rmsd_E73.xvg"],
    "Q4": ["rmsd_Q4.xvg",  "rmsd_Q42.xvg",  "rmsd_Q43.xvg"],
}
APO_RMSF_REPS = {
    "WT": ["rmsf_WT.xvg",  "rmsf_WT2.xvg",  "rmsf_WT3.xvg"],
    "E7": ["rmsf_E7.xvg",  "rmsf_E72.xvg",  "rmsf_E73.xvg"],
    "Q4": ["rmsf_Q4.xvg",  "rmsf_Q42.xvg",  "rmsf_Q43.xvg"],
}
REP_LS = ["-", "--", "-."]   # line styles for rep 1, 2, 3


def load_reps_rmsd(file_list):
    """Load RMSD replicates; interpolate to common time grid. Returns (time, matrix n_reps×n_t)."""
    from scipy.interpolate import interp1d
    all_t, all_v = [], []
    for f in file_list:
        t, v = read_xvg(os.path.join(OUT_DIR, f))
        all_t.append(t); all_v.append(v)
    t_common = all_t[0]
    aligned = []
    for t, v in zip(all_t, all_v):
        if len(t) == len(t_common) and np.allclose(t, t_common):
            aligned.append(v)
        else:
            aligned.append(interp1d(t, v, bounds_error=False,
                                    fill_value=(v[0], v[-1]))(t_common))
    return t_common, np.stack(aligned)


def load_reps_rmsf(file_list):
    """Load RMSF replicates; return (common_res, matrix n_reps×n_res)."""
    all_r, all_v = [], []
    for f in file_list:
        r, v = read_xvg(os.path.join(OUT_DIR, f))
        all_r.append(r); all_v.append(v)
    common = all_r[0]
    for r in all_r[1:]:
        common = np.intersect1d(common, r)
    aligned = np.stack([v[np.isin(r, common)] for r, v in zip(all_r, all_v)])
    return common, aligned


# ── Replicate / condition constants (used throughout) ────────────────────────
CONDITIONS  = ["apo", "holo"]
COND_LABELS = {"apo": "Apo", "holo": "Holo"}
APO_REPS    = [1, 2, 3]
HOLO_REPS   = [1, 2]
REP_STYLES  = {1: ("-", "Rep 1"), 2: ("--", "Rep 2"), 3: ("-.", "Rep 3")}


def _rmsd_path(var, cond, rep):
    suf = {1: "", 2: "2", 3: "3"}[rep]
    fname = f"rmsd_{var}{suf}.xvg" if cond == "apo" else f"rmsd_{var}holo{suf}.xvg"
    return os.path.join(OUT_DIR, fname)


def _rmsf_path(var, cond, rep):
    suf = {1: "", 2: "2", 3: "3"}[rep]
    fname = f"rmsf_{var}{suf}.xvg" if cond == "apo" else f"rmsf_{var}holo{suf}.xvg"
    return os.path.join(OUT_DIR, fname)


# ── Plot 1: RMSD comparison — mean ± SD across 3 apo reps ────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
fig.patch.set_facecolor("white")

for var in VARIANTS:
    t, mat = load_reps_rmsd(APO_RMSD_REPS[var])
    smooth = np.stack([pd.Series(mat[i]).rolling(WINDOW, center=True, min_periods=1).mean().values
                       for i in range(mat.shape[0])])
    mean = smooth.mean(axis=0); std = smooth.std(axis=0)
    ax.fill_between(t, mean - std, mean + std, color=COLORS[var], alpha=0.20)
    ax.plot(t, mean, linewidth=2.0, color=COLORS[var], label=var)

style_ax(ax)
ax.set_xlim(left=0); ax.set_ylim(bottom=0)
ax.set_xlabel("Time (ns)", fontsize=13)
ax.set_ylabel("RMSD (nm)", fontsize=13)
ax.legend(fontsize=12, frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmsd_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved rmsd_comparison.png")


# ── Plot 2: RMSF comparison — mean ± SD across 3 apo reps ────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_facecolor("white")

for var in VARIANTS:
    res, mat = load_reps_rmsf(APO_RMSF_REPS[var])
    mean = mat.mean(axis=0); std = mat.std(axis=0)
    ax.fill_between(res, mean - std, mean + std, color=COLORS[var], alpha=0.20)
    ax.plot(res, mean, linewidth=1.8, color=COLORS[var], label=var)

style_ax(ax)
ax.set_xlim(left=res.min()); ax.set_ylim(bottom=0)
ax.set_xlabel("Residue Number", fontsize=13)
ax.set_ylabel("RMSF (nm)", fontsize=13)
ax.legend(fontsize=12, frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmsf_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved rmsf_comparison.png")

# ── Plot 3: Apo (top) + Holo (bottom) RMSD, variants compared within each ────
fig, (ax_apo, ax_holo) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
fig.patch.set_facecolor("white")

for var in VARIANTS:
    for ax, fname, title in [
        (ax_apo,  f"rmsd_{var}.xvg",     "Apo"),
        (ax_holo, f"rmsd_{var}holo.xvg", "Holo"),
    ]:
        path = os.path.join(OUT_DIR, fname)
        time, rmsd = read_xvg(path)
        smooth = pd.Series(rmsd).rolling(WINDOW, center=True, min_periods=1).mean().values
        ax.plot(time, rmsd, linewidth=0.5, color=COLORS[var], alpha=0.2)
        ax.plot(time, smooth, linewidth=2.0, color=COLORS[var], label=var)
        ax.set_title(title, fontsize=13, fontweight="bold")

for ax in (ax_apo, ax_holo):
    style_ax(ax)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("RMSD (nm)", fontsize=13)
    ax.legend(fontsize=12, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

ax_holo.set_xlabel("Time (ns)", fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmsd_apo_holo_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved rmsd_apo_holo_comparison.png")

# ── Plot 4: RMSD replicates 1 / 2 / 3 (side by side, shared y) ───────────────
REP_FNAMES = [
    ("Replicate 1", {"WT": "rmsd_WT.xvg",  "E7": "rmsd_E7.xvg",  "Q4": "rmsd_Q4.xvg"}),
    ("Replicate 2", {"WT": "rmsd_WT2.xvg", "E7": "rmsd_E72.xvg", "Q4": "rmsd_Q42.xvg"}),
    ("Replicate 3", {"WT": "rmsd_WT3.xvg", "E7": "rmsd_E73.xvg", "Q4": "rmsd_Q43.xvg"}),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=True)
fig.patch.set_facecolor("white")

for ax, (title, fmap) in zip(axes, REP_FNAMES):
    for var, fname in fmap.items():
        time, rmsd = read_xvg(os.path.join(OUT_DIR, fname))
        smooth = pd.Series(rmsd).rolling(WINDOW, center=True, min_periods=1).mean().values
        ax.plot(time, rmsd, linewidth=0.5, color=COLORS[var], alpha=0.2)
        ax.plot(time, smooth, linewidth=2.0, color=COLORS[var], label=var)
    style_ax(ax)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (ns)", fontsize=13)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=12, frameon=False)

axes[0].set_ylabel("RMSD (nm)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmsd_replicates.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved rmsd_replicates.png")

# ── Plot 4: ΔlnPf (holo − apo) per residue, all three variants ───────────────
LNPF_FILES = {
    "WT": ("WT1_lnPf.dat",    "WT1holo_lnPf.dat"),
    "E7": ("E7_lnPf.dat",     "E7holo1_lnPf.dat"),
    "Q4": ("Q4_lnPf.dat",     "Q4holo1_lnPf.dat"),
}

def read_lnpf(filepath):
    """Parse lnPf .dat file; returns DataFrame with ResID, lnPf, Std."""
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return pd.DataFrame(rows, columns=["ResID", "lnPf", "Std"])

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor("white")

for var, (apo_file, holo_file) in LNPF_FILES.items():
    apo  = read_lnpf(os.path.join(OUT_DIR, apo_file)).set_index("ResID")
    holo = read_lnpf(os.path.join(OUT_DIR, holo_file)).set_index("ResID")
    common = apo.index.intersection(holo.index)
    apo, holo = apo.loc[common], holo.loc[common]

    delta     = holo["lnPf"] - apo["lnPf"]
    delta_err = np.sqrt(apo["Std"]**2 + holo["Std"]**2)  # propagated error

    res = common.to_numpy()
    ax.plot(res, delta.values, linewidth=1.5, color=COLORS[var], label=var)
    ax.fill_between(res,
                    (delta - delta_err).values,
                    (delta + delta_err).values,
                    color=COLORS[var], alpha=0.15)

style_ax(ax)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
for mut_res in [70, 97, 116, 141]:
    ax.axvline(mut_res, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
ax.set_xlabel("Residue Number", fontsize=13)
ax.set_ylabel("ΔlnPf (Holo − Apo)", fontsize=13)
ax.legend(fontsize=12, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "delta_lnPf.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved delta_lnPf.png")

# ── Domain bar & mutation helpers ─────────────────────────────────────────────
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
MUT_RES = [70, 97, 116, 141]
BAR_H   = 0.55  # inches for domain bar panel

def draw_domain_bar(ax):
    for start, end, color, label in DOM_SEGS:
        ax.add_patch(plt.Rectangle((start - 0.5, 0.05), end - start + 1, 0.9,
                                   facecolor=color, edgecolor="white", linewidth=0.5))
        if label:
            tc = "white" if color == "#7b2d8b" else "#5a3000"
            ax.text((start + end) / 2, 0.5, label, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=tc)
    ax.set_xlim(1, 178)
    ax.set_ylim(0, 1)
    ax.axis("off")

def make_fig_with_bar(w, h):
    fig, (ax_bar, ax) = plt.subplots(
        2, 1, figsize=(w, h + BAR_H),
        gridspec_kw={"height_ratios": [BAR_H, h], "hspace": 0.04})
    fig.patch.set_facecolor("white")
    draw_domain_bar(ax_bar)
    return fig, ax


# ── Plots 5 & 6: ΔlnPf variant vs WT — apo and holo ─────────────────────────
LNPF_VARIANT_FILES = {
    "apo":  {"WT": "WT1_lnPf.dat",    "E7": "E7_lnPf.dat",      "Q4": "Q4_lnPf.dat"},
    "holo": {"WT": "WT1holo_lnPf.dat","E7": "E7holo1_lnPf.dat", "Q4": "Q4holo1_lnPf.dat"},
}

for cond, files in LNPF_VARIANT_FILES.items():
    wt = read_lnpf(os.path.join(OUT_DIR, files["WT"])).set_index("ResID")

    fig, ax = make_fig_with_bar(12, 5)

    for var in ["E7", "Q4"]:
        df = read_lnpf(os.path.join(OUT_DIR, files[var])).set_index("ResID")
        common = wt.index.intersection(df.index)
        delta     = df.loc[common, "lnPf"] - wt.loc[common, "lnPf"]
        delta_err = np.sqrt(df.loc[common, "Std"]**2 + wt.loc[common, "Std"]**2)
        res = common.to_numpy()
        ax.plot(res, delta.values, linewidth=1.5, color=COLORS[var], label=f"{var} − WT")
        ax.fill_between(res, (delta - delta_err).values, (delta + delta_err).values,
                        color=COLORS[var], alpha=0.15)

    for m in MUT_RES:
        ax.axvline(m, color="gray", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    style_ax(ax)
    ax.set_xlim(1, 178)
    ax.set_xlabel("Residue Number", fontsize=13)
    ax.set_ylabel("ΔlnPf (Variant − WT)", fontsize=13)
    ax.set_title(f"{cond.capitalize()}", fontsize=13, fontweight="bold", pad=4)
    ax.legend(fontsize=12, frameon=False)

    fname = f"delta_lnPf_vs_WT_{cond}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")

# ── Plot 7: RMSD replicates overlaid — one panel per variant, 3 reps ─────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
fig.patch.set_facecolor("white")

for ax, var in zip(axes, VARIANTS):
    color = COLORS[var]
    for rep in APO_REPS:
        ls, label = REP_STYLES[rep]
        time, rmsd = read_xvg(_rmsd_path(var, "apo", rep))
        smooth = pd.Series(rmsd).rolling(WINDOW, center=True, min_periods=1).mean().values
        ax.plot(time, rmsd, linewidth=0.5, color=color, alpha=0.2, linestyle=ls)
        ax.plot(time, smooth, linewidth=1.8, color=color, linestyle=ls, label=label)
    style_ax(ax)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_title(var, fontsize=12, fontweight="bold", color=color)
    ax.legend(fontsize=10, frameon=False)

axes[0].set_ylabel("RMSD (nm)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmsd_replicates_overlay.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved rmsd_replicates_overlay.png")

# ── Plot A: RMSD — 2 rows (apo/holo) × 3 cols (rep1/2/3, holo uses rep1/2) ──
fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=True, sharex=True)
fig.patch.set_facecolor("white")

for row, cond in enumerate(CONDITIONS):
    reps = APO_REPS if cond == "apo" else HOLO_REPS
    for col, rep in enumerate([1, 2, 3]):
        ax = axes[row][col]
        if rep not in reps:
            ax.set_visible(False)
            continue
        for var in VARIANTS:
            time, rmsd = read_xvg(_rmsd_path(var, cond, rep))
            smooth = pd.Series(rmsd).rolling(WINDOW, center=True, min_periods=1).mean().values
            ax.plot(time, rmsd, linewidth=0.5, color=COLORS[var], alpha=0.2)
            ax.plot(time, smooth, linewidth=1.8, color=COLORS[var], label=var)
        style_ax(ax)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.set_title(f"{COND_LABELS[cond]} — Replicate {rep}", fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("RMSD (nm)", fontsize=12)
        if row == 1:
            ax.set_xlabel("Time (ns)", fontsize=12)
        ax.legend(fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmsd_variants_per_rep_apo_holo.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved rmsd_variants_per_rep_apo_holo.png")


# ── Plot B: RMSD — 2×3 grid (apo/holo rows × WT/E7/Q4 cols), all reps each ──
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
fig.patch.set_facecolor("white")

for row, cond in enumerate(CONDITIONS):
    reps = APO_REPS if cond == "apo" else HOLO_REPS
    for col, var in enumerate(VARIANTS):
        ax = axes[row][col]
        color = COLORS[var]
        for rep in reps:
            ls, rep_label = REP_STYLES[rep]
            time, rmsd = read_xvg(_rmsd_path(var, cond, rep))
            smooth = pd.Series(rmsd).rolling(WINDOW, center=True, min_periods=1).mean().values
            ax.plot(time, rmsd, linewidth=0.5, color=color, alpha=0.2, linestyle=ls)
            ax.plot(time, smooth, linewidth=1.8, color=color, linestyle=ls, label=rep_label)
        style_ax(ax)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.set_title(f"{COND_LABELS[cond]} — {var}", fontsize=12, fontweight="bold", color=color)
        if col == 0:
            ax.set_ylabel("RMSD (nm)", fontsize=12)
        if row == 1:
            ax.set_xlabel("Time (ns)", fontsize=12)
        ax.legend(fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmsd_reps_per_variant_apo_holo.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved rmsd_reps_per_variant_apo_holo.png")


# ── Plot C: RMSF — 2 rows (apo/holo) × 3 cols (rep1/2/3) ────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=True, sharex=True)
fig.patch.set_facecolor("white")

for row, cond in enumerate(CONDITIONS):
    reps = APO_REPS if cond == "apo" else HOLO_REPS
    for col, rep in enumerate([1, 2, 3]):
        ax = axes[row][col]
        if rep not in reps:
            ax.set_visible(False)
            continue
        for var in VARIANTS:
            residue, rmsf = read_xvg(_rmsf_path(var, cond, rep))
            ax.plot(residue, rmsf, linewidth=0.9, color=COLORS[var], label=var)
        style_ax(ax)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{COND_LABELS[cond]} — Replicate {rep}", fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("RMSF (nm)", fontsize=12)
        if row == 1:
            ax.set_xlabel("Residue Number", fontsize=12)
        ax.legend(fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmsf_variants_per_rep_apo_holo.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved rmsf_variants_per_rep_apo_holo.png")


# ── Plot D: RMSF — 2×3 grid (apo/holo rows × WT/E7/Q4 cols), all reps each ──
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
fig.patch.set_facecolor("white")

for row, cond in enumerate(CONDITIONS):
    reps = APO_REPS if cond == "apo" else HOLO_REPS
    for col, var in enumerate(VARIANTS):
        ax = axes[row][col]
        color = COLORS[var]
        for rep in reps:
            ls, rep_label = REP_STYLES[rep]
            residue, rmsf = read_xvg(_rmsf_path(var, cond, rep))
            ax.plot(residue, rmsf, linewidth=0.9, color=color, linestyle=ls, label=rep_label)
        style_ax(ax)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{COND_LABELS[cond]} — {var}", fontsize=12, fontweight="bold", color=color)
        if col == 0:
            ax.set_ylabel("RMSF (nm)", fontsize=12)
        if row == 1:
            ax.set_xlabel("Residue Number", fontsize=12)
        ax.legend(fontsize=10, frameon=False)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmsf_reps_per_variant_apo_holo.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved rmsf_reps_per_variant_apo_holo.png")


# ── Plot E: two-panel RMSD comparison — mean ± SD across replicates ──────────
HOLO_RMSD_REPS = {
    "WT": ["rmsd_WTholo.xvg",  "rmsd_WTholo2.xvg"],
    "E7": ["rmsd_E7holo.xvg",  "rmsd_E7holo2.xvg"],
    "Q4": ["rmsd_Q4holo.xvg",  "rmsd_Q4holo2.xvg"],
}

fig, (ax_apo, ax_holo) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
fig.patch.set_facecolor("white")

for ax, rep_dict, label in [
        (ax_apo,  APO_RMSD_REPS,  "Apo"),
        (ax_holo, HOLO_RMSD_REPS, "Holo"),
]:
    for var in VARIANTS:
        t, mat = load_reps_rmsd(rep_dict[var])
        smooth = np.stack([
            pd.Series(mat[i]).rolling(WINDOW, center=True, min_periods=1).mean().values
            for i in range(mat.shape[0])
        ])
        mean = smooth.mean(axis=0)
        std  = smooth.std(axis=0)
        ax.fill_between(t, mean - std, mean + std, color=COLORS[var], alpha=0.20)
        ax.plot(t, mean, linewidth=2.0, color=COLORS[var], label=var)
    style_ax(ax)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.set_ylabel("RMSD (nm)", fontsize=13)
    ax.text(0.02, 0.95, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top")

ax_holo.legend(fontsize=12, frameon=False, loc="lower right")
ax_holo.set_xlabel("Time (ns)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "rmsd_apo_holo_mean_sd.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved rmsd_apo_holo_mean_sd.png")


print("\nDone.")
