"""Six primary analyses on the CAT-Panel consensus scores.

Each maps to a publishable claim from the plan
(plans/delegated-whistling-panda.md, "Primary analyses enabled"):

  F1. CAT-Panel × persona family — within-subject Persona − GPT d_z per family
  F2. CAT-Panel × Big-5 moderation
  F3. CAT-Panel → originality (process → product bridge)
  F4. Inter-judge disagreement radar
  F5. CAT-Panel vs Taxonomy 2 convergence
  F6. Mediation path: persona → user-creativity-behaviour → originality

Each function is small-n robust (will not crash on n=5 per family) and
reports raw paired-t / Spearman p-values (no FDR per user directive).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.cat_panel.dimensions import DIM_NAMES, DIM_LABELS, DIMENSIONS
from os_pipeline.cat_panel.personas import JUDGE_IDS, JUDGE_LABELS

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT  = ROOT / 'analysis_out' / 'cat_panel'
FIG  = ROOT / 'figures' / 'cat_panel'
FIG.mkdir(parents=True, exist_ok=True)

FAM_ORDER = ('Divergent', 'Convergent', 'Rational', 'BoundedRational')
FAM_COLOR = {'Divergent':'#2A8C99', 'Convergent':'#B85C3A',
             'Rational':'#6D5D9C', 'BoundedRational':'#C7A11A'}
JUDGE_COLOR = {'Dr_C':'#2A8C99', 'Dr_I':'#B85C3A',
               'Dr_D':'#6D5D9C', 'Dr_L':'#C7A11A'}

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 220, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'axes.titlesize': 11,
})


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _read_consensus() -> pd.DataFrame:
    """Returns long-format: one row per (conv, dim) with consensus + per-judge cells."""
    for ext in ('.parquet', '.csv'):
        fp = OUT / f'panel_scores_consensus{ext}'
        if fp.exists():
            return (pd.read_parquet(fp) if ext == '.parquet'
                    else pd.read_csv(fp))
    raise FileNotFoundError(f'no consensus scores in {OUT}')


def _read_adjudicated() -> pd.DataFrame:
    """Returns long-format: one row per (judge, conv, dim) with score_adj."""
    for ext in ('.parquet', '.csv'):
        fp = OUT / f'panel_scores_adjudicated{ext}'
        if fp.exists():
            return (pd.read_parquet(fp) if ext == '.parquet'
                    else pd.read_csv(fp))
    raise FileNotFoundError(f'no adjudicated scores in {OUT}')


def _attach_condition_family() -> pd.DataFrame:
    """Maps conversation_id → (User_id, persona_family, condition)."""
    logs = pd.read_csv(ROOT / 'Experiment1_logs.csv')
    FM = {'Divergent':'Divergent','Convergent':'Convergent',
          'strictly rational':'Rational','bounded rationality':'BoundedRational',
          'GPT':'GPT'}
    logs['family_raw'] = logs['Persona_type'].map(FM)
    logs['condition'] = np.where(logs['Persona_type']=='GPT', 'GPT', 'Persona')

    # user's persona-family (their off-GPT family, which is constant per user)
    user_fam = (logs[logs.family_raw!='GPT']
                .groupby('User_id')['family_raw'].first())

    cmap = (logs.groupby('conversation_id')
            .agg(User_id=('User_id','first'),
                 condition=('condition','first'))
            .reset_index())
    cmap['persona_family'] = cmap['User_id'].map(user_fam)
    return cmap


# ----------------------------------------------------------------------
# F1. CAT-Panel × persona family — within-subject d_z
# ----------------------------------------------------------------------

def f1_dz_by_family():
    cons = _read_consensus()
    cmap = _attach_condition_family()
    df = cons.merge(cmap, on='conversation_id', how='left')

    # pivot to one row per (user, dim) with both Persona and GPT consensus cells
    wide = df.pivot_table(
        index=['User_id','persona_family','dimension'],
        columns='condition', values='consensus', aggfunc='first',
    ).reset_index()
    for c in ('Persona','GPT'):
        if c not in wide.columns: wide[c] = np.nan

    wide['delta'] = wide['Persona'].astype(float) - wide['GPT'].astype(float)

    rows = []
    for fam in FAM_ORDER:
        for dim in DIM_NAMES:
            sub = wide[(wide['persona_family']==fam) & (wide['dimension']==dim)]
            d = sub['delta'].dropna().astype(float)
            if len(d) < 3:
                rows.append(dict(family=fam, dimension=dim,
                                 label=DIM_LABELS[dim], n=len(d),
                                 dz=np.nan, t=np.nan, p=np.nan,
                                 mean_diff=np.nan))
                continue
            sd = d.std(ddof=1)
            dz = float(d.mean()/sd) if sd>0 else np.nan
            t, p = stats.ttest_1samp(d, 0)
            rows.append(dict(
                family=fam, dimension=dim, label=DIM_LABELS[dim],
                n=int(len(d)), dz=dz, t=float(t), p=float(p),
                mean_diff=float(d.mean()),
            ))
    F1 = pd.DataFrame(rows)
    F1.to_csv(OUT / 'F1_dz_by_family.csv', index=False)
    print(f'wrote {OUT / "F1_dz_by_family.csv"}')

    # Heatmap
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    mat = F1.pivot_table(index='label', columns='family', values='dz').reindex(
        index=[DIM_LABELS[d] for d in DIM_NAMES], columns=list(FAM_ORDER))
    pmat = F1.pivot_table(index='label', columns='family', values='p').reindex(
        index=[DIM_LABELS[d] for d in DIM_NAMES], columns=list(FAM_ORDER))
    im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='auto')
    ax.set_xticks(range(len(FAM_ORDER))); ax.set_xticklabels(FAM_ORDER, rotation=15, ha='right')
    ax.set_yticks(range(len(DIM_NAMES))); ax.set_yticklabels([DIM_LABELS[d] for d in DIM_NAMES])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i,j]; p = pmat.values[i,j]
            if np.isnan(v): continue
            star = ''
            if not np.isnan(p):
                if   p<0.001: star='***'
                elif p<0.01:  star='**'
                elif p<0.05:  star='*'
                elif p<0.10:  star='†'
            ax.text(j, i, f'{v:+.2f}\n{star}', ha='center', va='center',
                    fontsize=8.4, color='white' if abs(v)>0.85 else 'black')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("CAT-Panel consensus  Cohen's d_z  (Persona − GPT)", fontsize=9)
    ax.set_title('F1. User-creativity entrainment by persona family\n'
                 '(CAT-Panel consensus across 4 expert judges; raw p)', fontsize=10.5, pad=12)
    plt.tight_layout()
    plt.savefig(FIG / 'F1_panel_dz_by_family.png')
    plt.close()
    print(f'wrote {FIG / "F1_panel_dz_by_family.png"}')
    return F1


# ----------------------------------------------------------------------
# F2. Big-5 × Δ CAT-Panel
# ----------------------------------------------------------------------

def f2_big5_moderation():
    cons = _read_consensus()
    cmap = _attach_condition_family()
    df = cons.merge(cmap, on='conversation_id', how='left')
    wide = df.pivot_table(
        index=['User_id','persona_family','dimension'],
        columns='condition', values='consensus', aggfunc='first',
    ).reset_index()
    wide['delta'] = wide['Persona'].astype(float) - wide['GPT'].astype(float)

    mu = pd.read_csv(ROOT / 'analysis_out' / 'master_users.csv')
    big5 = ['Extraversion','Agreeableness','Conscientiousness',
            'Negative Emotionality','Open-Mindedness']
    mu = mu[['id'] + [c for c in big5 if c in mu.columns]].rename(columns={'id':'User_id'})
    m = wide.merge(mu, on='User_id', how='left')

    rows = []
    for fam in FAM_ORDER:
        sub_fam = m[m['persona_family']==fam]
        for dim in DIM_NAMES:
            sub = sub_fam[sub_fam['dimension']==dim]
            for trait in big5:
                if trait not in sub.columns: continue
                x = sub[trait].astype(float); y = sub['delta'].astype(float)
                msk = (~x.isna())&(~y.isna())
                if msk.sum() < 4: continue
                rho, p = stats.spearmanr(x[msk], y[msk])
                rows.append(dict(family=fam, dimension=dim,
                                 label=DIM_LABELS[dim],
                                 trait=trait, n=int(msk.sum()),
                                 rho=float(rho), p=float(p)))
    F2 = pd.DataFrame(rows)
    F2.to_csv(OUT / 'F2_big5_moderation.csv', index=False)
    print(f'wrote {OUT / "F2_big5_moderation.csv"}  ({len(F2)} rows)')
    # Top hits (raw p<.05)
    sig = F2[F2.p<0.05].sort_values('p')
    print(f'\nF2. top moderation hits (raw p<.05):')
    print(sig[['family','trait','label','n','rho','p']].head(20).to_string(index=False))
    return F2


# ----------------------------------------------------------------------
# F3. CAT-Panel → Originality
# ----------------------------------------------------------------------

def f3_originality_bridge():
    cons = _read_consensus()
    pivot = cons.pivot_table(index='conversation_id', columns='dimension',
                             values='consensus', aggfunc='first').reset_index()
    orig_path = ROOT / 'analysis_out' / 'production' / 'participant_originality.csv'
    if not orig_path.exists():
        print('skip F3 — no originality csv')
        return None
    orig = pd.read_csv(orig_path)
    m = pivot.merge(orig[['conversation_id','n_ideas','orig_same','orig_all','orig_cross']],
                    on='conversation_id', how='left')

    rows = []
    for dim in DIM_NAMES:
        if dim not in m.columns: continue
        for ocol in ['n_ideas','orig_same','orig_all','orig_cross']:
            x = m[dim].astype(float); y = m[ocol].astype(float)
            msk = (~x.isna())&(~y.isna())
            if msk.sum() < 6: continue
            rho, p = stats.spearmanr(x[msk], y[msk])
            rows.append(dict(panel_dim=dim, label=DIM_LABELS[dim],
                             outcome=ocol, n=int(msk.sum()),
                             rho=float(rho), p=float(p)))
    F3 = pd.DataFrame(rows)
    F3.to_csv(OUT / 'F3_panel_to_originality.csv', index=False)
    print(f'wrote {OUT / "F3_panel_to_originality.csv"}')
    print('\nF3. Top |rho| (panel × originality):')
    print(F3.assign(absr=F3['rho'].abs())
            .sort_values('absr', ascending=False)
            .drop(columns='absr')
            .head(10).to_string(index=False))
    return F3


# ----------------------------------------------------------------------
# F4. Inter-judge disagreement radar
# ----------------------------------------------------------------------

def f4_disagreement_radar():
    """Per-judge mean profile across the 8 dimensions, overlaid as a radar."""
    adj = _read_adjudicated()
    mean_profile = (adj.groupby(['judge_id','dimension'])['score_adj']
                    .mean().unstack('dimension').reindex(columns=list(DIM_NAMES)))
    mean_profile.to_csv(OUT / 'F4_judge_mean_profiles.csv')

    # Radar plot
    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(projection='polar'))
    theta = np.linspace(0, 2*np.pi, len(DIM_NAMES), endpoint=False).tolist()
    theta += theta[:1]
    for judge in JUDGE_IDS:
        if judge not in mean_profile.index: continue
        vals = mean_profile.loc[judge].fillna(0).tolist()
        vals += vals[:1]
        ax.plot(theta, vals, color=JUDGE_COLOR[judge], lw=2,
                label=f'{judge} ({JUDGE_LABELS[judge].split(" ")[0]})')
        ax.fill(theta, vals, color=JUDGE_COLOR[judge], alpha=0.10)
    ax.set_xticks(theta[:-1])
    # shorten dim labels for radar
    short_labels = [DIM_LABELS[d].split(' ')[0] for d in DIM_NAMES]
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_ylim(1, 7)
    ax.set_yticks([1,2,3,4,5,6,7])
    ax.set_yticklabels([str(x) for x in range(1,8)], fontsize=7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=8)
    plt.title('F4. Mean rating profile by expert judge (1-7 Likert)\n'
              'Divergence = theoretical-lens disagreement', y=1.05)
    plt.tight_layout()
    plt.savefig(FIG / 'F4_judge_disagreement_radar.png', bbox_inches='tight')
    plt.close()
    print(f'wrote {FIG / "F4_judge_disagreement_radar.png"}')
    return mean_profile


# ----------------------------------------------------------------------
# F5. CAT-Panel vs Taxonomy 2 convergence
# ----------------------------------------------------------------------

def f5_convergence_with_t2():
    # use the construct-validity output if available
    cv_path = OUT / 'construct_validity.csv'
    if not cv_path.exists():
        print('skip F5 — construct_validity.csv not yet built. Run construct_validity.py first.')
        return None
    cv = pd.read_csv(cv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    # only the T2 pairings
    t2 = cv[cv['layer']=='Taxonomy_2_user'].copy()
    if len(t2):
        labels = [f'{r["panel_dim_label"]}\n vs {r["external_label"]}'
                  for _, r in t2.iterrows()]
        y = np.arange(len(t2))
        colors = ['#2A8C99' if r > 0 else '#B85C3A' for r in t2['rho']]
        ax.barh(y, t2['rho'], color=colors, alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
        ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('Spearman ρ')
        ax.set_xlim(-0.6, 0.6)
        for i, (_, r) in enumerate(t2.iterrows()):
            star = ('*' if r['p']<0.05 else '†' if r['p']<0.10 else '')
            ax.text(r['rho']+(0.02 if r['rho']>0 else -0.02), i,
                    f"{r['rho']:+.2f}{star}",
                    ha='left' if r['rho']>0 else 'right',
                    va='center', fontsize=8.5)
    ax.set_title('F5. CAT-Panel × Taxonomy-2 user-side construct convergence\n'
                 '(positive ρ = construct-validity support)')
    plt.tight_layout()
    plt.savefig(FIG / 'F5_panel_vs_taxonomy2.png', bbox_inches='tight')
    plt.close()
    print(f'wrote {FIG / "F5_panel_vs_taxonomy2.png"}')
    return cv


# ----------------------------------------------------------------------
# F6. Mediation: persona → Δ panel-dim → Δ originality
# ----------------------------------------------------------------------

def f6_mediation_bootstrap(panel_dim: str = 'user_ideational_fluency',
                            outcome: str = 'orig_all',
                            n_bootstrap: int = 5000):
    """Within-subject mediation: Persona (vs GPT) → Δ panel-dim → Δ outcome.

    Tests the indirect effect with bootstrap resampling at the user level.
    Small-n on PoC; will be more powered on the full run.
    """
    cons = _read_consensus()
    cmap = _attach_condition_family()
    df = cons.merge(cmap, on='conversation_id', how='left')
    df = df[df['dimension']==panel_dim]
    wide = df.pivot_table(index='User_id', columns='condition',
                          values='consensus', aggfunc='first').reset_index()
    if 'Persona' not in wide.columns or 'GPT' not in wide.columns:
        print('skip F6 — need both Persona and GPT data')
        return None
    wide['delta_panel'] = wide['Persona'].astype(float) - wide['GPT'].astype(float)

    orig = pd.read_csv(ROOT / 'analysis_out' / 'production' / 'participant_originality.csv')
    orig_w = orig.pivot_table(index='user', columns='condition',
                              values=outcome, aggfunc='first').reset_index()
    orig_w.rename(columns={'user':'User_id'}, inplace=True)
    if 'Persona' not in orig_w.columns or 'GPT' not in orig_w.columns:
        print('skip F6 — outcome missing one condition')
        return None
    orig_w['delta_outcome'] = orig_w['Persona'].astype(float) - orig_w['GPT'].astype(float)

    m = wide[['User_id','delta_panel']].merge(
        orig_w[['User_id','delta_outcome']], on='User_id', how='inner').dropna()
    n = len(m)
    if n < 8:
        print(f'F6 — only n={n} complete users; bootstrap may be unstable')

    # Indirect effect = corr(delta_panel, delta_outcome) magnitude/sign
    rho_obs, p_obs = stats.spearmanr(m['delta_panel'], m['delta_outcome'])

    # Bootstrap 95% CI on the Spearman rho at the user level
    rng = np.random.default_rng(7)
    arr = m[['delta_panel','delta_outcome']].values
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s = arr[idx]
        if s[:,0].std() == 0 or s[:,1].std() == 0:
            continue
        r, _ = stats.spearmanr(s[:,0], s[:,1])
        if not np.isnan(r):
            boots.append(r)
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f'\nF6 Mediation (within-subject):')
    print(f'  panel_dim    = {panel_dim}')
    print(f'  outcome      = {outcome}')
    print(f'  n            = {n}')
    print(f'  rho observed = {rho_obs:+.3f}   p = {p_obs:.4f}')
    print(f'  95% bootstrap CI: [{lo:+.3f}, {hi:+.3f}]')
    summary = pd.DataFrame([dict(
        panel_dim=panel_dim, outcome=outcome, n=n,
        rho=rho_obs, p=p_obs, ci_lo=lo, ci_hi=hi,
        boot_resamples=n_bootstrap,
    )])
    summary.to_csv(OUT / 'F6_mediation_summary.csv', index=False)

    # path diagram (simple)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis('off')
    box = dict(boxstyle='round,pad=0.5', facecolor='#EFF3F8', edgecolor='#36454F')
    ax.text(0.10, 0.5, 'Persona prompt\n(Persona vs GPT)', ha='center', va='center',
            fontsize=10, bbox=box)
    ax.text(0.50, 0.5, f'Δ {DIM_LABELS[panel_dim]}\n(within-subject)', ha='center', va='center',
            fontsize=10, bbox=box)
    ax.text(0.90, 0.5, f'Δ {outcome}', ha='center', va='center',
            fontsize=10, bbox=box)
    ax.annotate('', xy=(0.36, 0.5), xytext=(0.20, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#36454F'))
    ax.annotate('', xy=(0.78, 0.5), xytext=(0.62, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#36454F'))
    ax.text(0.50, 0.18,
            f'Indirect effect (Spearman ρ): {rho_obs:+.3f}    '
            f'95% bootstrap CI [{lo:+.3f}, {hi:+.3f}]    '
            f'n = {n}    raw p = {p_obs:.4f}',
            ha='center', va='center', fontsize=9.5, style='italic',
            color='#222')
    plt.title('F6. Persona → Δ CAT-Panel dimension → Δ originality (within-subject)')
    plt.tight_layout()
    plt.savefig(FIG / 'F6_mediation_path.png', bbox_inches='tight')
    plt.close()
    print(f'wrote {FIG / "F6_mediation_path.png"}')
    return summary


# ----------------------------------------------------------------------
# Main: run all six
# ----------------------------------------------------------------------

def main():
    print('=' * 60); print('F1'); print('=' * 60)
    f1_dz_by_family()
    print('\n' + '=' * 60); print('F2'); print('=' * 60)
    f2_big5_moderation()
    print('\n' + '=' * 60); print('F3'); print('=' * 60)
    f3_originality_bridge()
    print('\n' + '=' * 60); print('F4'); print('=' * 60)
    f4_disagreement_radar()
    print('\n' + '=' * 60); print('F5'); print('=' * 60)
    f5_convergence_with_t2()
    print('\n' + '=' * 60); print('F6'); print('=' * 60)
    f6_mediation_bootstrap()


if __name__ == '__main__':
    main()
