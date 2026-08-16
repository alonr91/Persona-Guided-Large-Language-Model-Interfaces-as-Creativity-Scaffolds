"""
Regenerate the three GPT-labeled figures (question-rate trajectory,
user-side entrainment heatmap, partner-footing coupling) with
'Standard LLM' instead of 'GPT' as the condition label.

All three pull from existing analysis_out CSVs/NPYs — no full pipeline rerun.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
FIG  = os.path.join(ROOT, 'figures')

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 220, 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# =====================================================================
# 1. fig_question_rate_by_quarter.png
# =====================================================================
def fig_question_rate():
    df = pd.read_csv(os.path.join(ROOT, 'analysis_out', 'question_rate_by_quarter.csv'))
    # rename label
    df['persona_label'] = df['persona_label'].replace({'GPT (Control)': 'Standard LLM (Control)'})

    PERSONA_ORDER = ['Standard LLM (Control)', 'Divergent', 'Convergent', 'Rational', 'BoundedRational']
    COLOR = {
        'Standard LLM (Control)': '#888888',
        'Divergent':               '#D7263D',
        'Convergent':              '#3F7CAC',
        'Rational':                '#F46036',
        'BoundedRational':         '#2EC4B6',
    }
    MARKER = {
        'Standard LLM (Control)': 'o',
        'Divergent':               's',
        'Convergent':              '^',
        'Rational':                'D',
        'BoundedRational':         'v',
    }
    LS = {
        'Standard LLM (Control)': '--',
        'Divergent':               '-',
        'Convergent':              '-',
        'Rational':                '-',
        'BoundedRational':         '-',
    }
    QUARTERS = ['Q1','Q2','Q3','Q4']

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for persona in PERSONA_ORDER:
        sub = df[df['persona_label'] == persona]
        means = [sub[sub['quarter'] == q]['q_rate'].mean() for q in QUARTERS]
        sems  = [sub[sub['quarter'] == q]['q_rate'].sem()  for q in QUARTERS]
        ax.errorbar(QUARTERS, means, yerr=sems,
                    label=persona, color=COLOR[persona], marker=MARKER[persona],
                    linestyle=LS[persona], lw=1.8, ms=7, capsize=3, alpha=0.9)
    ax.set_xlabel('Conversation Quarter')
    ax.set_ylabel('Mean Question-mark Rate\n(fraction of user messages with ≥1 "?")')
    ax.set_title('User Question-mark Frequency by Conversation Quarter\n(Experiment 1)')
    ax.set_ylim(-0.02, 0.70)
    ax.set_yticks(np.arange(0, 0.71, 0.10))
    ax.set_yticklabels([f'{int(v*100)}%' for v in np.arange(0, 0.71, 0.10)])
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(FIG, 'fig_question_rate_by_quarter.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


# =====================================================================
# 2. fig10_user_family_dz_condition_blind.png
# =====================================================================
def fig_user_dz_heatmap():
    df = pd.read_csv(os.path.join(ROOT, 'analysis_out', 'user_creativity', 'A1_user_dz_by_family.csv'))
    print('A1 cols:', df.columns.tolist())
    print(df.head())

    # Identify columns. Expect columns: family, construct (or label), dz, p, n
    # Build a family x construct matrix.
    CONSTRUCTS = ['reframe','expansion','propose','certainty','contraction','commit','critique']
    CONSTRUCT_KEY = {'ref':'reframe','exp':'expansion','prop':'propose','cer':'certainty',
                     'con':'contraction','com':'commit','cri':'critique'}
    FAMILIES = ['Divergent','Convergent','Rational','BoundedRational']
    FAMILY_LABEL = {'Divergent':'Divergent (n=38)','Convergent':'Convergent (n=41)',
                    'Rational':'Rational (n=9)','BoundedRational':'BoundedRational (n=9)'}

    # The A1 CSV may use 'construct' column with short keys
    if 'construct' in df.columns:
        df['c'] = df['construct'].map(CONSTRUCT_KEY).fillna(df['construct'])
    elif 'user_construct' in df.columns:
        df['c'] = df['user_construct'].map(CONSTRUCT_KEY).fillna(df['user_construct'])
    elif 'label' in df.columns:
        df['c'] = df['label'].str.lower()
    else:
        raise SystemExit('cannot find construct column in A1 csv')

    fam_col = 'family' if 'family' in df.columns else 'persona'
    val_col = 'dz' if 'dz' in df.columns else 'd_z'
    p_col = 'p_bh' if 'p_bh' in df.columns else ('p' if 'p' in df.columns else None)

    mat = np.zeros((len(FAMILIES), len(CONSTRUCTS)))
    stars = np.empty_like(mat, dtype=object)
    for i, fam in enumerate(FAMILIES):
        for j, con in enumerate(CONSTRUCTS):
            sub = df[(df[fam_col] == fam) & (df['c'] == con)]
            if len(sub) == 0:
                mat[i, j] = np.nan; stars[i, j] = ''
                continue
            mat[i, j] = float(sub.iloc[0][val_col])
            pv = float(sub.iloc[0][p_col]) if p_col else 1.0
            stars[i, j] = ('***' if pv < .001 else '**' if pv < .01 else
                           '*' if pv < .05 else '†' if pv < .10 else '')

    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v): continue
            color = 'white' if abs(v) > 0.85 else 'black'
            ax.text(j, i, f'{v:+.2f}{stars[i,j]}', ha='center', va='center',
                    color=color, fontsize=10)
    ax.set_xticks(range(len(CONSTRUCTS)))
    ax.set_xticklabels(CONSTRUCTS, rotation=30, ha='right')
    ax.set_yticks(range(len(FAMILIES)))
    ax.set_yticklabels([FAMILY_LABEL[f] for f in FAMILIES])
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Cohen's d_z (Persona − Standard LLM, user-side)")
    fig.tight_layout()
    out = os.path.join(FIG, 'fig10_user_family_dz_condition_blind.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


# =====================================================================
# 3. fig18_partner_footing_coupling.png
# =====================================================================
def fig_partner_footing():
    """Bar chart: mean consecutive cosine distance by condition × transition.
    Reuses precomputed conv-level data if available; else recomputes from the
    msg_embeddings store."""
    # Try precomputed conv-level table
    candidate = os.path.join(ROOT, 'analysis_out', 'extension_conv_master.csv')
    if not os.path.exists(candidate):
        print('skip fig18 — no conv master')
        return
    df = pd.read_csv(candidate)
    # Look for transition-level cosine cols
    cols = df.columns.tolist()
    # Expected (from regenerate_figures.py): turn-to-turn distance per transition type.
    # Fall back: compute from msg-level embedding npy + master_conversations.csv.
    needed = ['au_cos_mean', 'ua_cos_mean']  # assistant→user, user→assistant
    if not all(c in cols for c in needed):
        # Recompute from raw embeddings
        emb = np.load(os.path.join(ROOT, 'analysis_out', 'msg_embeddings.npy'))
        meta = pd.read_csv(os.path.join(ROOT, 'analysis_out', 'master_conversations.csv'))
        # Need a per-message table with conv_id, role, order, condition
        msg_path = os.path.join(ROOT, 'analysis_out', '_user_embeddings_meta.csv')
        if os.path.exists(msg_path):
            mm = pd.read_csv(msg_path)
        else:
            print('skip fig18 — no msg meta'); return
        print('msg meta cols:', mm.columns.tolist()[:20])
        return  # complex path — skip if data missing

    # Aggregate
    df['condition'] = df['condition'].replace({'GPT':'Standard LLM'})
    grp = df.groupby('condition')[needed].agg(['mean','sem'])
    print(grp)

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    width = 0.38
    x = np.arange(2)  # 0 = a→u, 1 = u→a
    conds = ['Standard LLM','Persona']
    colors = {'Standard LLM':'#bfbfbf', 'Persona':'#5482AB'}
    for i, cond in enumerate(conds):
        sub = df[df['condition'] == cond]
        means = [sub['au_cos_mean'].mean(), sub['ua_cos_mean'].mean()]
        sems  = [sub['au_cos_mean'].sem(),  sub['ua_cos_mean'].sem()]
        ax.bar(x + (i - 0.5)*width, means, width, yerr=sems,
               color=colors[cond], label=cond, capsize=4, edgecolor='black', lw=0.4)
    # p-values
    sub_au = df.dropna(subset=['au_cos_mean','condition'])
    try:
        a = sub_au[sub_au['condition']=='Standard LLM']['au_cos_mean']
        b = sub_au[sub_au['condition']=='Persona']['au_cos_mean']
        _, p_au = stats.ttest_ind(a, b, equal_var=False)
    except Exception: p_au = np.nan
    try:
        a = df[df['condition']=='Standard LLM']['ua_cos_mean']
        b = df[df['condition']=='Persona']['ua_cos_mean']
        _, p_ua = stats.ttest_ind(a, b, equal_var=False)
    except Exception: p_ua = np.nan
    def stars(p):
        if np.isnan(p): return ''
        return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'
    ax.text(0, 0.74, f'p = {p_au:.1e} {stars(p_au)}', ha='center')
    ax.text(1, 0.62, f'p = {p_ua:.1e} {stars(p_ua)}', ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(['assistant → user transition\n(user accepts assistant turn)',
                        'user → assistant transition\n(assistant accepts user turn)'])
    ax.set_ylabel('Mean consecutive cosine distance')
    ax.set_title('Partner-footing coupling asymmetry by transition type')
    ax.set_ylim(0, 0.78)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, 'fig18_partner_footing_coupling.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    fig_question_rate()
    fig_user_dz_heatmap()
    fig_partner_footing()
