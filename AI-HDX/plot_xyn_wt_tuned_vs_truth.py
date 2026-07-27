"""
XYN WT apo: predicted − truth residual (single panel, uniform color).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

PRED = '/Users/jiali/Desktop/Jiali/TAMU/Dynamics/XYNmodels/predictions'

# ── domain bar ────────────────────────────────────────────────────────────────
DOMAINS = [
    (1,   57,  'Fingers', '#FF7043'),
    (58,  73,  'Palm',    '#26A69A'),
    (74,  79,  'Fingers', '#FF7043'),
    (80,  83,  'Palm',    '#26A69A'),
    (84,  92,  'Cord',    '#D4E157'),
    (93,  110, 'Palm',    '#26A69A'),
    (111, 121, 'Thumb',   '#7E57C2'),
    (122, 160, 'Palm',    '#26A69A'),
    (161, 178, 'Fingers', '#FF7043'),
]
DOMAIN_COLORS = {'Fingers': '#FF7043', 'Palm': '#26A69A', 'Cord': '#D4E157', 'Thumb': '#7E57C2'}

def draw_domain_bar(ax):
    for start, end, _, color in DOMAINS:
        ax.add_patch(Rectangle((start - 0.5, 0.02), end - start + 1, 0.35,
                               facecolor=color, edgecolor='white', linewidth=0.5))
    ax.set_xlim(0.5, 179.5)
    ax.set_ylim(0, 1)
    ax.axis('off')

def domain_legend_handles():
    return [mpatches.Patch(color=DOMAIN_COLORS[n], alpha=0.85, label=n)
            for n in ['Fingers', 'Palm', 'Cord', 'Thumb']]

# ── loaders ───────────────────────────────────────────────────────────────────
def load_pred(path):
    df = pd.read_csv(path)
    df.rename(columns={'0': 'start', '1': 'end', '2': 'seq', '3': 'hdx_exp'}, inplace=True)
    return df.groupby(['start', 'end'])['average'].mean().reset_index()

def load_truth_abs(path):
    df = pd.read_csv(path)
    df.columns = ['start', 'end', 'seq', 'hdx']
    df['hdx'] = df['hdx'].astype(float)
    return df.groupby(['start', 'end'])['hdx'].mean().reset_index()

def load_truth_diff(path):
    df = pd.read_csv(path)
    df['change'] = df['change'].astype(str).str.strip().astype(float)
    return df.groupby(['start', 'end'])['change'].mean().reset_index()

# ── load data ─────────────────────────────────────────────────────────────────
pred_apo = load_pred(f'{PRED}/1xyn_apo_pred_tuned.csv')
pred_apo['pred_pct'] = pred_apo['average'] * 100

truth_abs = load_truth_abs(f'{PRED}/1xyn_apo_truth.csv')

# ── compute residuals (predicted − truth) ─────────────────────────────────────
resid_abs = pd.merge(pred_apo[['start', 'end', 'pred_pct']],
                     truth_abs[['start', 'end', 'hdx']],
                     on=['start', 'end'])
resid_abs['resid'] = resid_abs['pred_pct'] - resid_abs['hdx']

print(f"Absolute residual peptides : {len(resid_abs)}")
print(f"Abs residual range   : {resid_abs['resid'].min():.1f} to {resid_abs['resid'].max():.1f} %")

# ── figure layout: domain bar / legend spacer / data ─────────────────────────
BAR_H  = 2.5
BAR_COLOR = '#00838F'   # uniform teal

fig, (ax_bar, ax_leg, ax_abs) = plt.subplots(
    3, 1, figsize=(18, 4),
    gridspec_kw={'height_ratios': [1.0, 0.45, 4.5], 'hspace': 0.04})
fig.patch.set_facecolor('white')

draw_domain_bar(ax_bar)
ax_leg.legend(handles=domain_legend_handles(), loc='center', ncol=4,
              fontsize=16, frameon=False, handlelength=1.4, handleheight=1.2,
              columnspacing=1.2)
ax_leg.axis('off')

# ── data panel ────────────────────────────────────────────────────────────────
for _, row in resid_abs.iterrows():
    ax_abs.add_patch(Rectangle(
        (row['start'] - 0.5, row['resid'] - BAR_H / 2),
        row['end'] - row['start'] + 1, BAR_H,
        facecolor=BAR_COLOR, alpha=0.75, edgecolor='none', zorder=3))

abs_lim = max(abs(resid_abs['resid'].min()), abs(resid_abs['resid'].max())) * 1.3
ax_abs.set_xlim(0.5, 179.5)
ax_abs.set_ylim(-abs_lim, abs_lim)
ax_abs.axhline( 5, color='gray', linewidth=1.0, linestyle='--', zorder=1)
ax_abs.axhline(-5, color='gray', linewidth=1.0, linestyle='--', zorder=1)
ax_abs.set_ylabel('Predicted − Experimental (%)', fontsize=15)
ax_abs.set_xlabel('Residue Number', fontsize=15)
ax_abs.tick_params(labelsize=13)
ax_abs.set_xticks(range(25, 179, 25))
ax_abs.grid(True, linestyle='--', alpha=0.25)

plt.tight_layout()
out = f'{PRED}/xyn_wt_tuned_vs_truth.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
