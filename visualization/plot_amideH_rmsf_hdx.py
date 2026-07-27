"""
Single dual-axis plot: HDX bars (red, left axis) + amideH RMSF lines per
simulation (right axis).  Three simulations shown individually with different
line styles (Sim1=solid, Sim2=dashed, Sim3=dash-dot).

Covered regions  : flat line at per-peptide average RMSF per sim, drawn with
                   variant linestyle in solid black (with matching linestyle).
Uncovered regions: per-residue RMSF per sim, drawn grey with matching linestyle.

Units: Å (converted from nm × 10)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

MD_DIR   = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/MDsimulation"
PRED_DIR = "/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/predictions"

NM_TO_ANG  = 10.0
SIM_LS     = ["-", "--", "-."]
SIM_LABELS = ["Sim1", "Sim2", "Sim3"]

# ── loaders ───────────────────────────────────────────────────────────────────
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


def load_truth(path):
    df = pd.read_csv(path, header=0)
    df.columns = ["start", "end", "sequence", "hdx"]
    df["start"] = df["start"].astype(int)
    df["end"]   = df["end"].astype(int)
    df["hdx"]   = df["hdx"].astype(float)
    return df


# ── load 3 amideH replicates ──────────────────────────────────────────────────
res1, v1 = load_xvg(f"{MD_DIR}/rmsf_WTamideH1.xvg")
res2, v2 = load_xvg(f"{MD_DIR}/rmsf_WTamideH2.xvg")
res3, v3 = load_xvg(f"{MD_DIR}/rmsf_WTamideH3.xvg")

all_res   = [res1, res2, res3]
all_vals  = [v1, v2, v3]
common_res = np.intersect1d(np.intersect1d(res1, res2), res3)

# per-residue RMSF for each replicate on common residues
rep_arrays = {
    r: np.array([v1[res1 == r][0], v2[res2 == r][0], v3[res3 == r][0]])
    for r in common_res
}

# ── HDX truth ─────────────────────────────────────────────────────────────────
truth = load_truth(f"{PRED_DIR}/1xyn_apo_truth.csv")

# ── select non-overlapping peptides (greedy, longest-first) ──────────────────
unique_peps = (truth.groupby(["start", "end"], as_index=False)
               .agg(hdx=("hdx", "mean")))
unique_peps["length"] = unique_peps["end"] - unique_peps["start"]
unique_peps = unique_peps.sort_values("length", ascending=False).reset_index(drop=True)

covered_mask = np.zeros(300, dtype=bool)
selected_rows = []
for _, row in unique_peps.iterrows():
    s, e = int(row["start"]), int(row["end"])
    if not covered_mask[s - 1 : e].any():
        covered_mask[s - 1 : e] = True
        selected_rows.append(row)

selected = (pd.DataFrame(selected_rows)
            .sort_values("start")
            .reset_index(drop=True))


# ── per-sim per-peptide RMSF ──────────────────────────────────────────────────
def peptide_rmsf_sim(start, end, sim_idx):
    """Mean RMSF for a peptide/region in a single simulation."""
    residues = [r for r in range(start, end + 1) if r in rep_arrays]
    if not residues:
        return np.nan
    return np.mean([rep_arrays[r][sim_idx] for r in residues])


N_RES   = 178
res_all = np.arange(1, N_RES + 1)

# Build one line array per simulation
sim_line_vals = []   # list of N_RES arrays
sim_is_cov    = []   # list of N_RES bool arrays

for sim_idx in range(3):
    line_v = np.full(N_RES, np.nan)
    is_cov = np.zeros(N_RES, dtype=bool)

    # covered segments: flat peptide-level RMSF
    for _, row in selected.iterrows():
        s, e = int(row["start"]), int(row["end"])
        v = peptide_rmsf_sim(s, e, sim_idx)
        for r in range(s, e + 1):
            idx = r - 1
            if 0 <= idx < N_RES:
                line_v[idx] = v
                is_cov[idx] = True

    # gap regions: contiguous uncovered stretches → region-average RMSF
    gap_start = None
    for i in range(N_RES):
        r = i + 1
        if not is_cov[i] and r in rep_arrays:
            if gap_start is None:
                gap_start = r
        else:
            if gap_start is not None:
                gap_end = r - 1 if is_cov[i] else r
                g_v = peptide_rmsf_sim(gap_start, gap_end, sim_idx)
                for gr in range(gap_start, gap_end + 1):
                    if gr - 1 < N_RES:
                        line_v[gr - 1] = g_v
                gap_start = None
    if gap_start is not None:
        g_v = peptide_rmsf_sim(gap_start, N_RES, sim_idx)
        for gr in range(gap_start, N_RES + 1):
            if gr - 1 < N_RES:
                line_v[gr - 1] = g_v

    sim_line_vals.append(line_v)
    sim_is_cov.append(is_cov)

# extend covered segments by 1 residue at each boundary for clean connections
def extend_cov(is_cov):
    ext = is_cov.copy()
    for i in range(1, N_RES - 1):
        if is_cov[i] and not is_cov[i - 1]:
            ext[i - 1] = True
        if is_cov[i] and not is_cov[i + 1]:
            ext[i + 1] = True
    return ext


# ── domain / highlight helpers ────────────────────────────────────────────────
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
_DOM_H = 1.0


def draw_domain_bar(ax):
    for start, end, color, _ in _DOM_SEGS:
        ax.add_patch(Rectangle((start - 0.5, 0.02), end - start + 1, 0.35,
                               facecolor=color, edgecolor="white", linewidth=0.5))
    ax.set_xlim(1, 178)
    ax.set_ylim(0, 1)
    ax.axis("off")

def domain_legend_handles():
    return [mpatches.Patch(color=_DOM_COLORS[n], alpha=0.85, label=n)
            for n in ["Fingers", "Palm", "Cord", "Thumb"]]


def draw_mutation_highlights(ax):
    for pos in [70, 116]:
        ax.axvspan(pos - 0.5, pos,       color="cyan",    alpha=0.25, zorder=1)
        ax.axvspan(pos,       pos + 0.5, color="magenta", alpha=0.25, zorder=1)
    for pos in [97, 141]:
        ax.axvspan(pos - 0.5, pos + 0.5, color="magenta", alpha=0.25, zorder=1)


# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_dom, ax_leg, ax) = plt.subplots(
    3, 1, figsize=(12, 6.5),
    gridspec_kw={"height_ratios": [_DOM_H, 0.45, 5], "hspace": 0.04})

draw_domain_bar(ax_dom)
ax_leg.axis('off')
ax_r = ax.twinx()

# ── HDX bars (red, left axis) ─────────────────────────────────────────────────
bar_h_hdx = 2.5
for _, row in truth.iterrows():
    rect = Rectangle((row["start"] - 0.5, row["hdx"] - bar_h_hdx / 2),
                     row["end"] - row["start"] + 1, bar_h_hdx,
                     facecolor="#d73027", edgecolor="none", alpha=0.65, zorder=3)
    ax.add_patch(rect)
ax.set_ylim(0, 105)

# ── RMSF per sim (right axis) ─────────────────────────────────────────────────
all_valid = np.concatenate([v[~np.isnan(v)] for v in sim_line_vals])
rlo, rhi  = all_valid.min(), all_valid.max()
pad_r     = (rhi - rlo) * 0.15
ax_r.set_ylim(rlo - pad_r, rhi + pad_r)

for sim_idx in range(3):
    ax_r.plot(res_all, sim_line_vals[sim_idx], color="black", linewidth=1.8,
              linestyle=SIM_LS[sim_idx], zorder=5)

# ── axes styling ──────────────────────────────────────────────────────────────
ax.set_ylabel("HDX Rate (%)", fontsize=16, color="#d73027")
ax_r.set_ylabel("Amide H RMSF (Å)", fontsize=16, color="black")
ax.tick_params(axis="y", labelcolor="#d73027", labelsize=14)
ax_r.tick_params(axis="y", labelcolor="black",   labelsize=14)
ax.tick_params(axis="x", labelsize=14)
ax.set_xlabel("Residue Number", fontsize=16)
ax.set_xlim(1, 178)
ax.set_xticks(range(25, 179, 25))
ax.axhline(0, color="gray", linewidth=0.8)
ax.grid(True, linestyle="--", alpha=0.3)

# ── legend ────────────────────────────────────────────────────────────────────
leg_hdx  = mpatches.Patch(facecolor="#d73027", alpha=0.65, label="HDX Rate (%)")
leg_sims = [mlines.Line2D([], [], color="black", linewidth=1.8,
                           linestyle=SIM_LS[i], label=SIM_LABELS[i])
            for i in range(3)]
# data legend — dedicated row between domain bar and line plot
ax_leg.legend(handles=[leg_hdx] + leg_sims,
              loc='center', ncol=4, fontsize=15, framealpha=0.9)

# domain legend — inside ax_dom, upper half (above the bar strip)
ax_dom.legend(handles=domain_legend_handles(),
              loc='upper center', ncol=4, fontsize=16, frameon=False,
              handlelength=1.4, handleheight=1.2, columnspacing=1.2)

plt.subplots_adjust(hspace=0.04)
out = f"{PRED_DIR}/plot_WT_hdx_amideH_rmsf.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
