"""
Question-mark frequency per conversation quarter per persona type — Experiment 1.

Mirrors the Experiment-2 analysis: for each conversation, split user turns into
4 equal quarters (Q1–Q4) and compute the fraction of user messages that contain
at least one '?'. Plot lines per persona type and run mixed/paired statistical
tests to flag differences vs the GPT baseline in Q2–Q4.

Output:
  figures/fig_question_rate_by_quarter.png
  analysis_out/question_rate_by_quarter.csv
  (prints a stats table to stdout)
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings('ignore')
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from scipy.stats import mannwhitneyu

# ── paths ────────────────────────────────────────────────────────────────────
ROOT   = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
DATA   = os.path.join(ROOT, 'Experiment1_logs.csv')
FIGDIR = os.path.join(ROOT, 'figures')
OUTDIR = os.path.join(ROOT, 'analysis_out')
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# ── load & clean ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
df['message'] = df['message'].fillna('').astype(str)

# Normalise persona labels
PERSONA_MAP = {
    'GPT':                 'GPT (Control)',
    'Divergent':           'Divergent',
    'Convergent':          'Convergent',
    'strictly rational':   'Rational',
    'bounded rationality': 'BoundedRational',
}
df['persona_label'] = df['Persona_type'].map(PERSONA_MAP).fillna(df['Persona_type'])

# Keep user messages only; sort within each conversation by message_id
user = (df[df['message_src'] == 'user']
          .sort_values(['conversation_id', 'message_id'])
          .copy())

user['has_q'] = user['message'].str.contains(r'\?').astype(int)

# ── assign quarter labels per conversation ───────────────────────────────────
def _quarters_for_group(series_idx, n):
    """Return quarter labels (Q1..Q4) for n items by equal split."""
    bins  = np.linspace(0, n, 5)
    bucket = np.digitize(np.arange(n), bins[1:])
    return ['Q' + str(min(b + 1, 4)) for b in bucket]

# Assign quarters without losing columns in groupby.apply
quarters_list = []
for conv_id, grp in user.groupby('conversation_id'):
    n = len(grp)
    qs = _quarters_for_group(grp.index, n)
    for idx, q in zip(grp.index, qs):
        quarters_list.append((idx, q))

q_map = pd.DataFrame(quarters_list, columns=['idx', 'quarter']).set_index('idx')
user = user.copy()
user['quarter'] = q_map['quarter']

# ── conversation-level quarter rates ─────────────────────────────────────────
# rate = fraction of user messages containing '?' per (conv, quarter)
conv_qrate = (user.groupby(['conversation_id', 'persona_label', 'quarter'])
                  .agg(n_msg=('has_q', 'count'), n_q=('has_q', 'sum'))
                  .reset_index())
conv_qrate['q_rate'] = conv_qrate['n_q'] / conv_qrate['n_msg']

# ── group means (for the line chart) ─────────────────────────────────────────
summary = (conv_qrate.groupby(['persona_label', 'quarter'])
                     .agg(mean_rate=('q_rate', 'mean'),
                          se_rate=('q_rate', lambda x: x.sem()))
                     .reset_index())

# save flat table
conv_qrate.to_csv(os.path.join(OUTDIR, 'question_rate_by_quarter.csv'), index=False)

# ── figure ────────────────────────────────────────────────────────────────────
QUARTERS = ['Q1', 'Q2', 'Q3', 'Q4']
PERSONA_ORDER = ['GPT (Control)', 'Divergent', 'Convergent', 'Rational', 'BoundedRational']
COLORS = {
    'GPT (Control)':  '#888888',
    'Divergent':      '#E84855',
    'Convergent':     '#3A86FF',
    'Rational':       '#FF9F1C',
    'BoundedRational':'#06D6A0',
}
MARKERS = {
    'GPT (Control)':  'o',
    'Divergent':      's',
    'Convergent':     '^',
    'Rational':       'D',
    'BoundedRational':'v',
}
LINESTYLES = {
    'GPT (Control)':  '--',
    'Divergent':      '-',
    'Convergent':     '-',
    'Rational':       '-',
    'BoundedRational':'-',
}

fig, ax = plt.subplots(figsize=(7, 4.2))

for persona in PERSONA_ORDER:
    sub = summary[summary['persona_label'] == persona].set_index('quarter')
    if len(sub) == 0:
        continue
    ys = [sub.loc[q, 'mean_rate'] if q in sub.index else np.nan for q in QUARTERS]
    es = [sub.loc[q, 'se_rate']   if q in sub.index else np.nan for q in QUARTERS]
    xs = np.arange(1, 5)
    ax.errorbar(xs, ys, yerr=es,
                label=persona,
                color=COLORS[persona],
                marker=MARKERS[persona],
                linestyle=LINESTYLES[persona],
                linewidth=2, markersize=7, capsize=3)

ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
ax.set_xlabel('Conversation Quarter', fontsize=11)
ax.set_ylabel('Mean Question-mark Rate\n(fraction of user messages with ≥1 "?")', fontsize=10)
ax.set_title('User Question-mark Frequency by Conversation Quarter\n(Experiment 1)', fontsize=11)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_ylim(0, None)
ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out_fig = os.path.join(FIGDIR, 'fig_question_rate_by_quarter.png')
fig.savefig(out_fig, dpi=150)
plt.close()
print(f'figure → {out_fig}')

# ── statistical tests ─────────────────────────────────────────────────────────
# For each persona × quarter: Mann-Whitney U vs GPT (Control) in same quarter.
# Also: within-persona Friedman test across Q1-Q4 for engagement trajectory.
print('\n━━━  Between-condition tests (each persona × quarter vs GPT baseline)  ━━━')
print(f"{'Persona':<20} {'Quarter':<8} {'GPT mean':>9} {'Prs mean':>9} "
      f"{'MWU p':>9} {'sig':>5}")
print('─' * 68)

gpт_wide = (conv_qrate[conv_qrate['persona_label'] == 'GPT (Control)']
              .pivot(index='conversation_id', columns='quarter', values='q_rate'))

results_rows = []
for persona in [p for p in PERSONA_ORDER if p != 'GPT (Control)']:
    prs_wide = (conv_qrate[conv_qrate['persona_label'] == persona]
                  .pivot(index='conversation_id', columns='quarter', values='q_rate'))
    for q in QUARTERS:
        gvals = gpт_wide[q].dropna().values if q in gpт_wide.columns else np.array([])
        pvals = prs_wide[q].dropna().values if q in prs_wide.columns else np.array([])
        if len(gvals) < 3 or len(pvals) < 3:
            continue
        stat_u, p_val = mannwhitneyu(pvals, gvals, alternative='two-sided')
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else
              ('*' if p_val < 0.05 else ('†' if p_val < 0.10 else '')))
        print(f"{persona:<20} {q:<8} {gvals.mean():>9.3f} {pvals.mean():>9.3f} "
              f"{p_val:>9.4f} {sig:>5}")
        results_rows.append(dict(persona=persona, quarter=q,
                                 gpt_mean=gvals.mean(), persona_mean=pvals.mean(),
                                 mwu_p=p_val, sig=sig))

# Within-persona: Friedman test for trend Q1→Q4
print('\n━━━  Within-persona Friedman test (Q1 vs Q2 vs Q3 vs Q4)  ━━━')
print(f"{'Persona':<20} {'χ²':>8} {'p':>10} {'sig':>5}")
print('─' * 48)
for persona in PERSONA_ORDER:
    sub = conv_qrate[conv_qrate['persona_label'] == persona]
    wide = (sub.pivot_table(index='conversation_id', columns='quarter',
                            values='q_rate', aggfunc='mean')
               .dropna(subset=QUARTERS))
    if len(wide) < 5:
        print(f"{persona:<20} {'n/a':>8}")
        continue
    chi2, p_frd = stats.friedmanchisquare(*[wide[q].values for q in QUARTERS])
    sig = '***' if p_frd < 0.001 else ('**' if p_frd < 0.01 else
          ('*' if p_frd < 0.05 else ('†' if p_frd < 0.10 else '')))
    print(f"{persona:<20} {chi2:>8.2f} {p_frd:>10.4f} {sig:>5}")

# Pairwise within-persona post-hoc: Q1 vs Q4 (Wilcoxon)
print('\n━━━  Within-persona Q1 vs Q4 shift (Wilcoxon signed-rank)  ━━━')
print(f"{'Persona':<20} {'Q1 mean':>9} {'Q4 mean':>9} {'Δ':>7} {'p':>10} {'sig':>5}")
print('─' * 65)
for persona in PERSONA_ORDER:
    sub = conv_qrate[conv_qrate['persona_label'] == persona]
    wide = (sub.pivot_table(index='conversation_id', columns='quarter',
                            values='q_rate', aggfunc='mean')
               .dropna(subset=['Q1', 'Q4']))
    if len(wide) < 5:
        continue
    stat_w, p_w = stats.wilcoxon(wide['Q4'].values, wide['Q1'].values,
                                  alternative='two-sided')
    sig = '***' if p_w < 0.001 else ('**' if p_w < 0.01 else
          ('*' if p_w < 0.05 else ('†' if p_w < 0.10 else '')))
    delta = wide['Q4'].mean() - wide['Q1'].mean()
    print(f"{persona:<20} {wide['Q1'].mean():>9.3f} {wide['Q4'].mean():>9.3f} "
          f"{delta:>+7.3f} {p_w:>10.4f} {sig:>5}")


# ── Paired within-subject analysis (each user: GPT quarter rate vs Persona quarter rate) ──
print('\n━━━  Paired within-subject: Persona vs GPT per quarter (Wilcoxon)  ━━━')
print('(each participant contributes one GPT rate and one Persona rate per quarter)')
print(f"{'Persona':<20} {'Quarter':<8} {'GPT mean':>9} {'Prs mean':>9} "
      f"{'Δ':>7} {'W p':>9} {'sig':>5}")
print('─' * 72)

# Build per-user per-quarter rates for GPT and each persona
user_ids_in_data = df['User_id'].unique()
for persona in [p for p in PERSONA_ORDER if p != 'GPT (Control)']:
    prs_rates = (conv_qrate[conv_qrate['persona_label'] == persona]
                   .merge(df[['conversation_id','User_id']].drop_duplicates(),
                          on='conversation_id')
                   .rename(columns={'q_rate': 'prs_rate'}))
    gpt_rates = (conv_qrate[conv_qrate['persona_label'] == 'GPT (Control)']
                   .merge(df[['conversation_id','User_id']].drop_duplicates(),
                          on='conversation_id')
                   .rename(columns={'q_rate': 'gpt_rate'}))
    merged = prs_rates.merge(gpt_rates[['User_id','quarter','gpt_rate']],
                              on=['User_id','quarter'])
    for q in QUARTERS:
        sub = merged[merged['quarter'] == q].dropna(subset=['prs_rate','gpt_rate'])
        if len(sub) < 5:
            continue
        stat_w, p_w = stats.wilcoxon(sub['prs_rate'].values, sub['gpt_rate'].values,
                                      alternative='two-sided')
        sig = '***' if p_w < 0.001 else ('**' if p_w < 0.01 else
              ('*' if p_w < 0.05 else ('†' if p_w < 0.10 else '')))
        delta = sub['prs_rate'].mean() - sub['gpt_rate'].mean()
        print(f"{persona:<20} {q:<8} {sub['gpt_rate'].mean():>9.3f} "
              f"{sub['prs_rate'].mean():>9.3f} {delta:>+7.3f} {p_w:>9.4f} {sig:>5}")

print(f'\nFigure saved to {out_fig}')
