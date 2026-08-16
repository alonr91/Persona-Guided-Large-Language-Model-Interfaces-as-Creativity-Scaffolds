"""Regenerate the four problematic figures cleanly for the v14 chapter.

Targets (chapter figure number -> image file in v13 docx -> new output):
  Fig 8  ->  image9.png   (assistant paired stance shifts; was Fig M3)
  Fig 12 ->  image22.png  (Big-5 personality moderation; dense heatmap)
  Fig 16 ->  image23.png  (process-product bridge; was tagged "A5")
  Fig 18 ->  image16.png  (partner-footing coupling; was "Yes-And asymmetry")

All four are saved to figures/ with new names (`fig8_*`, `fig12_*`, `fig16_*`,
`fig18_*`) and then copied over the corresponding images in the v13 unpacked
docx so the chapter renders the clean versions.
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT, 'analysis_out')
FIG  = os.path.join(ROOT, 'figures')
DOCX_MEDIA = os.path.join(ROOT, '_v13_unpacked', 'word', 'media')

# Common data loads
master = pd.read_csv(os.path.join(OUT, 'master_conversations.csv'))
users  = pd.read_csv(os.path.join(OUT, 'master_users.csv'))
# Use the condition-blind predictions as the primary source for the v14 chapter.
preds  = pd.read_csv(os.path.join(OUT, 'full_stance_predictions_condition_blind.csv'))

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.25, linewidth=0.6)

# -------------------------------------------------------------
# Figure 8 — Paired assistant-side stance shifts (no "Fig M3" title)
# Uses the condition-blind expansion and contraction predictions, aggregated
# to per-user means.
# -------------------------------------------------------------
def fig8():
    cid_to_user = dict(zip(master['conversation_id'], master['user']))
    ast = preds[preds['message_src'] == 'assistant'].copy()
    ast['user'] = ast['conversation_id'].map(cid_to_user)
    ast = ast.dropna(subset=['user'])
    agg = ast.groupby(['user', 'condition'])[['exp', 'con']].mean().reset_index()
    wide = (agg.pivot(index='user', columns='condition', values=['exp', 'con']))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), dpi=160)
    pairs = [('exp', 'Assistant expansion'), ('con', 'Assistant contraction')]
    for ax, (c, ti) in zip(axes, pairs):
        df = wide[c][['GPT', 'Persona']].dropna()
        for _, r in df.iterrows():
            ax.plot([0, 1], [r['GPT'], r['Persona']],
                    color='gray', alpha=0.30, lw=0.6)
        bp = ax.boxplot([df['GPT'], df['Persona']],
                         positions=[0, 1], widths=0.35,
                         patch_artist=True, showfliers=False)
        for patch, fc in zip(bp['boxes'], ['#d8d8d8', '#9fbfdf']):
            patch.set_facecolor(fc); patch.set_alpha(0.85)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['GPT', 'Persona'])
        ax.set_title(ti, fontsize=11)
        ax.set_ylabel('Mean stance (0–3)', fontsize=10)
        t, p = stats.ttest_rel(df['Persona'], df['GPT'])
        ymax = float(df.max().max())
        ax.text(0.5, ymax * 1.04, f'paired t = {t:+.2f}, p = {p:.1e}',
                ha='center', fontsize=9)
        style_axes(ax)
    plt.tight_layout()
    path = os.path.join(FIG, 'fig8_assistant_stance_paired.png')
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print('saved', path)
    return path

# -------------------------------------------------------------
# Figure 12 — Big-5 × Δ user stance heatmap, redrawn in landscape with no overlap
# Rows = Big-5 traits, columns = the 7 Taxonomy-2 user constructs, by family.
# We compute Spearman ρ within each (family x trait x construct) cell.
# -------------------------------------------------------------
def fig12():
    cid_to_user = dict(zip(master['conversation_id'], master['user']))
    us = preds[preds['message_src'] == 'user'].copy()
    us['user'] = us['conversation_id'].map(cid_to_user)
    us = us.dropna(subset=['user'])
    CONSTRUCTS = ['exp', 'con', 'cri', 'cer', 'com', 'ref', 'prop']
    LABELS = {'exp':'expansion','con':'contraction','cri':'critique',
              'cer':'certainty','com':'commit','ref':'reframe','prop':'propose'}
    TRAITS = ['Extraversion', 'Agreeableness', 'Conscientiousness',
              'Negative Emotionality', 'Open-Mindedness']

    # Per-user paired delta (Persona - GPT) for each construct
    by_uc = us.groupby(['user', 'condition'])[CONSTRUCTS].mean().reset_index()
    p_df = by_uc[by_uc['condition'] == 'Persona'].set_index('user')[CONSTRUCTS]
    g_df = by_uc[by_uc['condition'] == 'GPT'].set_index('user')[CONSTRUCTS]
    delta = (p_df - g_df).reset_index()

    # Attach personality + family
    fam_map = master[master['condition'] == 'Persona'].set_index('user')['family']
    delta = delta.join(fam_map.rename('family'), on='user')
    pers = users.set_index('id')[TRAITS]
    delta = delta.join(pers, on='user')

    families = ['Divergent', 'Convergent', 'Rational', 'BoundedRational']
    fig, axes = plt.subplots(1, len(families), figsize=(15, 4.5), dpi=160,
                              sharey=True)
    cmap = plt.cm.RdBu_r
    for ax, fam in zip(axes, families):
        sub = delta[delta['family'] == fam]
        n = len(sub)
        mat = np.full((len(TRAITS), len(CONSTRUCTS)), np.nan)
        pm  = np.full_like(mat, np.nan)
        for i, t in enumerate(TRAITS):
            for j, c in enumerate(CONSTRUCTS):
                x = sub[t].astype(float)
                y = sub[c].astype(float)
                m = (~x.isna()) & (~y.isna())
                if m.sum() < 4:
                    continue
                rho, p = stats.spearmanr(x[m], y[m])
                mat[i, j] = rho; pm[i, j] = p
        im = ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=0.5, aspect='auto')
        ax.set_xticks(range(len(CONSTRUCTS)))
        ax.set_xticklabels([LABELS[c] for c in CONSTRUCTS],
                            rotation=40, ha='right', fontsize=8)
        ax.set_yticks(range(len(TRAITS)))
        if fam == families[0]:
            ax.set_yticklabels(TRAITS, fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_title(f'{fam} (n={n})', fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                p = pm[i, j]
                if np.isnan(v):
                    continue
                star = ''
                if p < 0.01: star = '**'
                elif p < 0.05: star = '*'
                elif p < 0.10: star = '†'
                colour = 'black' if abs(v) < 0.3 else 'white'
                ax.text(j, i, f'{v:+.2f}{star}',
                         ha='center', va='center',
                         fontsize=7.5, color=colour)
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8,
                         label='Spearman ρ (Big-5 trait × Δ user construct)',
                         orientation='vertical', pad=0.02)
    plt.suptitle('', y=1.02)  # no internal title; chapter caption carries text
    path = os.path.join(FIG, 'fig12_personality_moderation_landscape.png')
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print('saved', path)
    return path

# -------------------------------------------------------------
# Figure 16 — Process-product bridge (no "A5" title)
# Three small panels: Δ user certainty vs Δ orig_same / orig_all; and
# Δ user proposing vs Δ n_ideas. Uses the extension master data where
# available.
# -------------------------------------------------------------
def fig16():
    ext_paired = pd.read_csv(os.path.join(OUT, 'extension_paired.csv'))
    # extension_paired has per-user deltas for orig_same/all/cross and n_ideas
    # plus user-side stance deltas if available
    # Compute Δ user certainty and Δ user proposing from condition-blind preds.
    cid_to_user = dict(zip(master['conversation_id'], master['user']))
    us = preds[preds['message_src'] == 'user'].copy()
    us['user'] = us['conversation_id'].map(cid_to_user)
    us = us.dropna(subset=['user'])
    pers_g = us[us['condition'] == 'Persona'].groupby('user')[['cer', 'prop']].mean()
    gpt_g  = us[us['condition'] == 'GPT'].groupby('user')[['cer', 'prop']].mean()
    user_delta = (pers_g - gpt_g).reset_index()
    user_delta.columns = ['user', 'd_user_certainty', 'd_user_proposing']

    # Map onto extension table by 'user' if present, else by user id
    if 'user' in ext_paired.columns:
        merged = ext_paired.merge(user_delta, on='user', how='inner')
    else:
        # fallback: assume order matches
        merged = ext_paired.copy().reset_index(drop=True)
        merged['d_user_certainty'] = user_delta['d_user_certainty'].values[:len(merged)]
        merged['d_user_proposing'] = user_delta['d_user_proposing'].values[:len(merged)]

    panels = [
        ('d_user_certainty', 'd_orig_same',
         'Δ user certainty (Persona − GPT)',
         'Δ same-condition originality',
         '#5b8dbe'),
        ('d_user_certainty', 'd_orig_all',
         'Δ user certainty (Persona − GPT)',
         'Δ all-participant originality',
         '#5b8dbe'),
        ('d_user_proposing', 'd_n_ideas',
         'Δ user proposing (Persona − GPT)',
         'Δ canonical user-ideas extracted',
         '#d98c5b'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), dpi=160)
    for ax, (xc, yc, xl, yl, colour) in zip(axes, panels):
        if xc not in merged.columns or yc not in merged.columns:
            ax.set_title(f'{yc} unavailable'); continue
        x = merged[xc].astype(float); y = merged[yc].astype(float)
        m = (~x.isna()) & (~y.isna())
        x = x[m]; y = y[m]
        ax.scatter(x, y, alpha=0.55, s=22, color=colour, edgecolor='white', lw=0.5)
        if len(x) >= 3:
            rho, p = stats.spearmanr(x, y)
            xx = np.linspace(x.min(), x.max(), 50)
            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(xx, slope * xx + intercept, color='black', lw=1.0, alpha=0.7)
            ax.text(0.04, 0.95, f'ρ = {rho:+.3f}\np = {p:.3f}\nn = {len(x)}',
                    transform=ax.transAxes, va='top', ha='left', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='lightgray',
                              boxstyle='round,pad=0.25', alpha=0.9))
        ax.axhline(0, color='gray', lw=0.5, alpha=0.7)
        ax.axvline(0, color='gray', lw=0.5, alpha=0.7)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        style_axes(ax)
    plt.tight_layout()
    path = os.path.join(FIG, 'fig16_process_product_bridge.png')
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print('saved', path)
    return path

# -------------------------------------------------------------
# Figure 18 — Partner-footing coupling (renamed from Yes-And)
# Bar chart of mean consecutive cosine distance for each transition type,
# under GPT vs Persona.
# -------------------------------------------------------------
def fig18():
    # Compute per-conversation transition means from message embeddings.
    cid_to_user = dict(zip(master['conversation_id'], master['user']))
    E = np.load(os.path.join(OUT, 'msg_embeddings.npy'))
    cb = preds.copy()
    # Order is preserved across both files.
    cb['emb_idx'] = np.arange(len(cb))
    # Per conversation, compute consecutive distances by transition type.
    rows = []
    for cid, sub in cb.groupby('conversation_id'):
        sub = sub.sort_values('emb_idx')
        idxs = sub['emb_idx'].values
        srcs = sub['message_src'].values
        # Normalise embeddings before cosine distance.
        embs = E[idxs]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        a2u, u2a = [], []
        for i in range(1, len(sub)):
            d = 1.0 - float(embs[i].dot(embs[i-1]))
            if srcs[i-1] == 'assistant' and srcs[i] == 'user':
                a2u.append(d)
            elif srcs[i-1] == 'user' and srcs[i] == 'assistant':
                u2a.append(d)
        rows.append({
            'conversation_id': cid,
            'user': cid_to_user.get(cid),
            'condition': sub['condition'].iloc[0],
            'a2u': np.mean(a2u) if a2u else np.nan,
            'u2a': np.mean(u2a) if u2a else np.nan,
        })
    df = pd.DataFrame(rows)
    pivot = df.set_index(['user', 'condition'])[['a2u', 'u2a']].unstack('condition')

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    width = 0.36
    x = np.arange(2)
    metrics = [
        ('a2u', 'assistant → user transition\n(user accepts assistant turn)'),
        ('u2a', 'user → assistant transition\n(assistant accepts user turn)'),
    ]
    means_gpt   = [pivot[(m, 'GPT')].mean()     for m, _ in metrics]
    means_pers  = [pivot[(m, 'Persona')].mean() for m, _ in metrics]
    sems_gpt    = [pivot[(m, 'GPT')].sem()      for m, _ in metrics]
    sems_pers   = [pivot[(m, 'Persona')].sem()  for m, _ in metrics]
    ax.bar(x - width/2, means_gpt,  width, yerr=sems_gpt,  capsize=4,
           color='#bfbfbf', alpha=0.9, label='GPT')
    ax.bar(x + width/2, means_pers, width, yerr=sems_pers, capsize=4,
           color='#5b8dbe', alpha=0.9, label='Persona')
    for i, (m, _) in enumerate(metrics):
        a = pivot[(m, 'Persona')].astype(float)
        b = pivot[(m, 'GPT')].astype(float)
        mask = (~a.isna()) & (~b.isna())
        t, p = stats.ttest_rel(a[mask], b[mask])
        if p < 0.001:  star = '***'
        elif p < 0.01: star = '**'
        elif p < 0.05: star = '*'
        else:          star = 'n.s.'
        ymax = max(means_gpt[i], means_pers[i])
        ax.text(i, ymax * 1.07, f'p = {p:.1e} {star}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics], fontsize=9)
    ax.set_ylabel('Mean consecutive cosine distance', fontsize=10)
    ax.set_title('Partner-footing coupling asymmetry by transition type',
                  fontsize=11)
    ax.legend(loc='upper right', frameon=False)
    style_axes(ax)
    plt.tight_layout()
    path = os.path.join(FIG, 'fig18_partner_footing_coupling.png')
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print('saved', path)
    return path

paths = {
    'fig8':  fig8(),
    'fig12': fig12(),
    'fig16': fig16(),
    'fig18': fig18(),
}

# Copy regenerated images over the v13 media files used by the chapter:
#   fig8  -> image9.png
#   fig12 -> image22.png
#   fig16 -> image23.png
#   fig18 -> image16.png
import shutil
mapping = {
    'fig8':  'image9.png',
    'fig12': 'image22.png',
    'fig16': 'image23.png',
    'fig18': 'image16.png',
}
for tag, fname in mapping.items():
    src = paths[tag]
    dst = os.path.join(DOCX_MEDIA, fname)
    shutil.copyfile(src, dst)
    print(f'copied {tag} -> {dst}')

print('\nRegeneration complete.')
