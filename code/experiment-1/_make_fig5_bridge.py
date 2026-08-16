"""
Rebuild Figure 5 (process->product bridge) for the rewritten chapter.

Three panels on n = 87 paired participants with complete originality + stance data:
  (a) Δ user_certainty  → Δ orig_same   (distinctiveness lever)
  (b) Δ user_certainty  → Δ orig_all
  (c) Δ user_proposing  → Δ n_ideas     (fluency lever)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT, 'figures', 'fig16_process_product_bridge.png')

# Per-participant originality (Persona - GPT deltas already computed)
o = pd.read_csv(os.path.join(ROOT, 'analysis_out', 'production', '_wide_originality.csv'))
# Per-participant stance (need to compute deltas from wide table)
s = pd.read_csv(os.path.join(ROOT, 'analysis_out', 'user_creativity', 'wide_user_assistant_paired.csv'))
s['d_u_cer']  = s['u_cer__Persona']  - s['u_cer__GPT']
s['d_u_prop'] = s['u_prop__Persona'] - s['u_prop__GPT']
s_small = s[['User_id', 'd_u_cer', 'd_u_prop']].rename(columns={'User_id': 'user'})

df = o.merge(s_small, on='user', how='inner').dropna(subset=['d_u_cer','d_u_prop','d_orig_same','d_orig_all','d_n_ideas'])
n = len(df)
print(f'paired rows for bridge: n = {n}')

# Aesthetic
plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 220, 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
})

PURPLE = '#5E4FA2'   # divergent/persona accent
TEAL   = '#1B7C7B'   # convergent/fluency accent
GREY   = '#7F7F7F'

def panel(ax, x, y, color, xlabel, ylabel, title):
    rho, p = stats.spearmanr(x, y)
    ax.scatter(x, y, s=26, c=color, alpha=0.70, edgecolor='white', linewidth=0.6)
    # OLS line for visual reference
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m*xs + b, color=color, lw=1.6, alpha=0.85)
    ax.axhline(0, color=GREY, lw=0.6, ls='--', alpha=0.6)
    ax.axvline(0, color=GREY, lw=0.6, ls='--', alpha=0.6)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
    ax.text(0.04, 0.96,
            f'ρ = {rho:+.2f}{star}\np = {p:.3f}\nn = {len(x)}',
            transform=ax.transAxes, va='top', ha='left',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GREY, alpha=0.9, lw=0.6))

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

panel(axes[0],
      df['d_u_cer'].values, df['d_orig_same'].values, PURPLE,
      'Δ user certainty (Persona − Standard LLM)',
      'Δ orig_same (distinctiveness vs peers)',
      'Certainty → same-condition distinctiveness')

panel(axes[1],
      df['d_u_cer'].values, df['d_orig_all'].values, PURPLE,
      'Δ user certainty (Persona − Standard LLM)',
      'Δ orig_all (distinctiveness vs all)',
      'Certainty → overall distinctiveness')

panel(axes[2],
      df['d_u_prop'].values, df['d_n_ideas'].values, TEAL,
      'Δ user proposing (Persona − Standard LLM)',
      'Δ n_ideas (canonical ideas per round)',
      'Proposing → fluency')

fig.suptitle('Within-subject process → product bridge (n = %d)' % n, fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT, bbox_inches='tight')
print('wrote', OUT)
