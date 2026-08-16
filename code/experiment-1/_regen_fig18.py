"""
Regenerate fig18_partner_footing_coupling.png from message embeddings,
with the condition label 'GPT' replaced by 'Standard LLM'.

For each conversation, compute the mean consecutive cosine distance
between adjacent messages, separated by transition type:
  - assistant->user (au): distance from an assistant turn to the next user turn
  - user->assistant (ua): distance from a user turn to the next assistant turn
Then compare conditions with a two-sample Welch t-test.
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

emb = np.load(os.path.join(ROOT, 'analysis_out', 'msg_embeddings.npy'))      # (3412, 384)
meta = pd.read_csv(os.path.join(ROOT, 'analysis_out', 'full_stance_predictions_condition_blind.csv'))
# `message_src` is the speaker; standardise role label
meta = meta.copy()
meta['role'] = meta['message_src'].astype(str).str.lower()
# Detect what user vs assistant values look like
print(meta['role'].value_counts().head(10))

# Normalise role labels
role_map = {}
for v in meta['role'].unique():
    if v in ('user','human','participant'): role_map[v] = 'user'
    elif v in ('assistant','gpt','model','llm','ai'): role_map[v] = 'assistant'
    else: role_map[v] = v
meta['role2'] = meta['role'].map(role_map)
print('mapped roles:', meta['role2'].value_counts().head(10))

# embeddings rows correspond to rows in meta
assert len(meta) == emb.shape[0], f'len mismatch: meta={len(meta)} emb={emb.shape[0]}'

# L2-normalise for cosine sim via dot
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

# Compute per-conversation consecutive cosine distances by transition type.
def conv_metrics(group):
    g = group.sort_values('turn_frac').reset_index(drop=True)
    idx = g.index.tolist()
    if len(idx) < 2: return pd.Series({'au_cos_mean': np.nan, 'ua_cos_mean': np.nan})
    au, ua = [], []
    for i in range(len(idx) - 1):
        r1 = g.loc[idx[i], 'role2']; r2 = g.loc[idx[i+1], 'role2']
        e1 = emb[g.loc[idx[i], '_row']]; e2 = emb[g.loc[idx[i+1], '_row']]
        d = 1.0 - float(np.dot(e1, e2))
        if r1 == 'assistant' and r2 == 'user': au.append(d)
        elif r1 == 'user' and r2 == 'assistant': ua.append(d)
    return pd.Series({'au_cos_mean': np.mean(au) if au else np.nan,
                      'ua_cos_mean': np.mean(ua) if ua else np.nan})

meta = meta.reset_index().rename(columns={'index': '_row'})
conv = meta.groupby(['conversation_id', 'condition']).apply(conv_metrics).reset_index()
print(conv.head())
print('cond counts:', conv['condition'].value_counts())

# Rename condition label for plot
conv['condition_disp'] = conv['condition'].replace({'GPT': 'Standard LLM'})

# Two-sample tests
a = conv[conv['condition'] == 'GPT']['au_cos_mean'].dropna()
b = conv[conv['condition'] == 'Persona']['au_cos_mean'].dropna()
_, p_au = stats.ttest_ind(a, b, equal_var=False)
a = conv[conv['condition'] == 'GPT']['ua_cos_mean'].dropna()
b = conv[conv['condition'] == 'Persona']['ua_cos_mean'].dropna()
_, p_ua = stats.ttest_ind(a, b, equal_var=False)

def stars(p):
    if np.isnan(p): return ''
    return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'

# Plot
plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 220, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})
fig, ax = plt.subplots(figsize=(8.5, 5.6))
width = 0.38
x = np.arange(2)
conds = ['Standard LLM', 'Persona']
colors = {'Standard LLM': '#bfbfbf', 'Persona': '#5482AB'}

for i, cond in enumerate(conds):
    sub = conv[conv['condition_disp'] == cond]
    means = [sub['au_cos_mean'].mean(), sub['ua_cos_mean'].mean()]
    sems  = [sub['au_cos_mean'].sem(),  sub['ua_cos_mean'].sem()]
    ax.bar(x + (i - 0.5) * width, means, width, yerr=sems,
           color=colors[cond], label=cond, capsize=4,
           edgecolor='black', linewidth=0.4)

ax.text(0, 0.74, f'p = {p_au:.1e} {stars(p_au)}', ha='center', fontsize=10)
ax.text(1, 0.62, f'p = {p_ua:.1e} {stars(p_ua)}', ha='center', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(['assistant → user transition\n(user accepts assistant turn)',
                    'user → assistant transition\n(assistant accepts user turn)'])
ax.set_ylabel('Mean consecutive cosine distance')
ax.set_title('Partner-footing coupling asymmetry by transition type')
ax.set_ylim(0, 0.80)
ax.legend(loc='upper right', framealpha=0.95)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()

out = os.path.join(ROOT, 'figures', 'fig18_partner_footing_coupling.png')
fig.savefig(out, bbox_inches='tight')
print('wrote', out, '| p_au=%.2e, p_ua=%.2e' % (p_au, p_ua))
