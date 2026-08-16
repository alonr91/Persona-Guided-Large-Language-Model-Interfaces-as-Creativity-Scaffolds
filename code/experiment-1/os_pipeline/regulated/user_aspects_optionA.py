"""Option A — Per-persona-family user-only analysis using existing measures.

Within-subject design (each participant has both a Persona round and a GPT
round). For each user-only measure we compute the paired diff
(Persona_value − GPT_value) per participant, group by family, and report:

  Within-family one-sample test of paired diff against 0
    - mean_diff, sd_diff, dz, t, p, q_BH, sig_05

  Pairwise between-family contrasts of those paired diffs
    - Welch t on the (paired-diff) vectors, Hedges' g, q_BH, sig_05

Outputs:
  regulated_llm_reanalysis/17_user_optionA_paired_by_family.csv / .md
  regulated_llm_reanalysis/17_user_optionA_pairwise_diffs.csv / .md
  regulated_llm_reanalysis/figures/fig_user_optionA_paired_by_family.png

The user-only measures used are intentionally a curated subset — process-
behaviour measures plus Pipeline 1 originality/fluency. We avoid joint
measures (assistant features, full-conversation aggregates).
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

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
ANA = ROOT / 'analysis_out'
OUT = ROOT / 'regulated_llm_reanalysis'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

FAMILY_ORDER = ('Divergent', 'Convergent', 'Rational', 'BoundedRational')
COLORS = {
    'Divergent': '#2a9d8f', 'Convergent': '#e76f51',
    'Rational': '#6f62b6', 'BoundedRational': '#e9c46a',
}

# Curated USER-ONLY measures.
# Process-behaviour set comes from master_conversations (per-conv per-user).
# Originality set comes from production/participant_originality (per-round).
USER_PROCESS_MEASURES = {
    'n_user_msg': 'Number of user messages',
    'user_words': 'Total user words',
    'user_mean_len': 'Mean user message length (words)',
    'user_q_rate': 'User question rate (Qs / user msgs)',
    'user_word_share': 'User share of total conversation words',
    'u_propose': 'User proposal rate',
    'u_question': 'User question rate (taxonomy)',
    'u_reframe': 'User reframing rate',
    'u_critique': 'User critique rate',
    'u_clarify': 'User clarification rate',
    'u_commit': 'User commitment rate',
}
USER_PRODUCT_MEASURES = {
    'n_ideas': 'User idea fluency (ideas / round)',
    'orig_same': 'User idea originality (vs same-condition peers)',
    'orig_all':  'User idea originality (vs all peers)',
    'orig_cross': 'User idea originality (vs cross-condition peers)',
}


# ----------------------------------------------------------------------- #
# Stats helpers
# ----------------------------------------------------------------------- #

def bh_fdr(p: list[float]) -> list[float]:
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0: return []
    order = np.argsort(p); ranked = p[order]
    out = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        r = i + 1
        out[i] = min(prev, ranked[i] * n / r); prev = out[i]
    res = np.empty(n)
    for i, pos in enumerate(order):
        res[pos] = out[i]
    return [float(x) for x in res]


def one_sample_paired(diffs: np.ndarray) -> dict:
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 5:
        return dict(n=n, mean_diff=float('nan'), sd_diff=float('nan'),
                    dz=float('nan'), t=float('nan'), p=float('nan'))
    sd = diffs.std(ddof=1)
    dz = diffs.mean() / sd if sd > 0 else float('nan')
    try:
        t, p = stats.ttest_1samp(diffs, 0.0)
    except Exception:
        t, p = float('nan'), float('nan')
    return dict(
        n=n, mean_diff=float(diffs.mean()), sd_diff=float(sd),
        dz=float(dz) if dz == dz else float('nan'),
        t=float(t) if t == t else float('nan'),
        p=float(p) if p == p else float('nan'),
    )


def hedges_g(a: np.ndarray, b: np.ndarray) -> dict:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return dict(n_a=na, n_b=nb, g=float('nan'),
                    welch_t=float('nan'), p=float('nan'))
    va = a.var(ddof=1); vb = b.var(ddof=1)
    sp2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    sp = np.sqrt(sp2) if sp2 > 0 else float('nan')
    d = (a.mean() - b.mean()) / sp if sp and sp > 0 else float('nan')
    J = 1 - 3 / (4 * (na + nb) - 9)
    g = d * J if not np.isnan(d) else float('nan')
    try:
        t, p = stats.ttest_ind(a, b, equal_var=False)
    except Exception:
        t, p = float('nan'), float('nan')
    return dict(
        n_a=na, n_b=nb, g=float(g) if not np.isnan(g) else float('nan'),
        welch_t=float(t) if t == t else float('nan'),
        p=float(p) if p == p else float('nan'),
    )


# ----------------------------------------------------------------------- #
# Data loaders
# ----------------------------------------------------------------------- #

def load_user_paired_long() -> pd.DataFrame:
    """Return long-format paired-diff table:
       [user, family, measure, persona_val, gpt_val, paired_diff]."""
    mc = pd.read_csv(ANA / 'master_conversations.csv')
    po = pd.read_csv(ANA / 'production' / 'participant_originality.csv')

    # Process measures from master_conversations (per-conv -> pivot to per-user)
    keep = ['user', 'family', 'condition'] + list(USER_PROCESS_MEASURES)
    proc = mc[keep].copy()
    proc_wide = proc.pivot_table(index=['user', 'family'],
                                  columns='condition',
                                  values=list(USER_PROCESS_MEASURES),
                                  aggfunc='mean')
    proc_long = []
    for measure in USER_PROCESS_MEASURES:
        try:
            sub = proc_wide[measure].reset_index()
        except KeyError:
            continue
        sub = sub.rename(columns={'GPT': 'gpt_val', 'Persona': 'persona_val'})
        sub['measure'] = measure
        proc_long.append(sub)
    proc_long = pd.concat(proc_long, ignore_index=True)

    # Originality measures from participant_originality (already per-round)
    # Originality file has 'condition' = GPT/Persona and 'user'
    # Pipeline-1 family info needs to come from master_conversations
    fam_map = mc[['user', 'family']].drop_duplicates(subset=['user'])
    po2 = po.merge(fam_map, on='user', how='left')
    po_keep = ['user', 'family', 'condition'] + list(USER_PRODUCT_MEASURES)
    po_wide = po2[po_keep].pivot_table(index=['user', 'family'],
                                        columns='condition',
                                        values=list(USER_PRODUCT_MEASURES),
                                        aggfunc='mean')
    po_long = []
    for measure in USER_PRODUCT_MEASURES:
        try:
            sub = po_wide[measure].reset_index()
        except KeyError:
            continue
        sub = sub.rename(columns={'GPT': 'gpt_val', 'Persona': 'persona_val'})
        sub['measure'] = measure
        po_long.append(sub)
    if po_long:
        po_long = pd.concat(po_long, ignore_index=True)
    else:
        po_long = pd.DataFrame()

    full = pd.concat([proc_long, po_long], ignore_index=True)
    full['paired_diff'] = full['persona_val'] - full['gpt_val']
    full = full[full['family'].isin(FAMILY_ORDER)].copy()
    return full[['user', 'family', 'measure', 'persona_val', 'gpt_val',
                 'paired_diff']]


# ----------------------------------------------------------------------- #
# Tests
# ----------------------------------------------------------------------- #

def within_family_paired(long: pd.DataFrame) -> pd.DataFrame:
    """Within each family, test if paired diff (Persona − GPT) ≠ 0.
    Significance is raw p < 0.05 (no multiple-comparison correction)."""
    rows = []
    for measure in long['measure'].unique():
        for fam in FAMILY_ORDER:
            sub = long[(long['measure'] == measure) & (long['family'] == fam)]
            diffs = sub['paired_diff'].dropna().to_numpy()
            stat = one_sample_paired(diffs)
            stat.update(dict(measure=measure, family=fam,
                             persona_mean=float(sub.persona_val.mean()),
                             gpt_mean=float(sub.gpt_val.mean())))
            rows.append(stat)
    df = pd.DataFrame(rows)
    df['sig_05'] = df['p'].fillna(1.0) < 0.05
    cols = ['measure', 'family', 'n', 'persona_mean', 'gpt_mean',
            'mean_diff', 'sd_diff', 'dz', 't', 'p', 'sig_05']
    return df[cols]


def pairwise_between_family_diffs(long: pd.DataFrame) -> pd.DataFrame:
    """Compare paired-diff vectors between persona families (Welch + g)."""
    pairs = []
    for i, fa in enumerate(FAMILY_ORDER):
        for fb in FAMILY_ORDER[i + 1:]:
            pairs.append((fa, fb))
    rows = []
    for measure in long['measure'].unique():
        for fa, fb in pairs:
            a = long[(long['measure'] == measure) & (long['family'] == fa)]['paired_diff'].dropna().to_numpy()
            b = long[(long['measure'] == measure) & (long['family'] == fb)]['paired_diff'].dropna().to_numpy()
            stat = hedges_g(a, b)
            stat.update(dict(measure=measure, family_a=fa, family_b=fb,
                             contrast=f'{fa} vs {fb}',
                             mean_a=float(np.mean(a)) if len(a) else float('nan'),
                             mean_b=float(np.mean(b)) if len(b) else float('nan')))
            rows.append(stat)
    df = pd.DataFrame(rows)
    df['sig_05'] = df['p'].fillna(1.0) < 0.05
    cols = ['measure', 'contrast', 'family_a', 'family_b',
            'mean_a', 'mean_b', 'n_a', 'n_b', 'g',
            'welch_t', 'p', 'sig_05']
    return df[cols]


# ----------------------------------------------------------------------- #
# Markdown
# ----------------------------------------------------------------------- #

LABEL = {**USER_PROCESS_MEASURES, **USER_PRODUCT_MEASURES}


def md_paired(df: pd.DataFrame, path: Path) -> None:
    lines = [
        '# Option A — Per-family within-subject paired analysis (Persona − GPT)',
        '',
        'Each participant has a Persona round and a GPT round (within-subject '
        'design). For each user-only measure, we take the per-participant '
        'paired difference (Persona_value − GPT_value), group by the persona '
        'family the participant was assigned, and test whether the mean '
        'paired difference is non-zero. **Positive** mean_diff means the user '
        'expressed *more* of the measure in the Persona round than the GPT '
        'round. Significance is raw `p < 0.05` (no multiple-comparison '
        'correction).',
        '',
    ]
    for fam in FAMILY_ORDER:
        sub = df[df['family'] == fam].copy()
        if len(sub) == 0:
            continue
        lines += [f'## Family: {fam}', '', '| measure | n | persona̅ | gpt̅ | mean_diff | dz | p | sig |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for _, r in sub.iterrows():
            lines.append(
                f"| {LABEL.get(r['measure'], r['measure'])} | {int(r['n'])} | "
                f"{r['persona_mean']:.3f} | {r['gpt_mean']:.3f} | "
                f"{r['mean_diff']:+.3f} | {r['dz']:+.2f} | "
                f"{r['p']:.3g} | "
                f"{'✓' if r['sig_05'] else '·'} |"
            )
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def md_pairwise(df: pd.DataFrame, path: Path) -> None:
    lines = [
        '# Option A — Pairwise between-family contrasts of paired diffs',
        '',
        'For each user-only measure, we compare the per-participant '
        'paired-diff vectors (Persona − GPT) across pairs of persona '
        'families. Welch t with Hedges\' g. Significance is raw `p < 0.05` '
        '(no multiple-comparison correction).',
        '',
    ]
    for contrast in df['contrast'].drop_duplicates():
        sub = df[df['contrast'] == contrast]
        lines += [f'## {contrast}', '',
                  '| measure | mean_a | mean_b | n_a | n_b | g | p | sig |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for _, r in sub.iterrows():
            lines.append(
                f"| {LABEL.get(r['measure'], r['measure'])} | "
                f"{r['mean_a']:+.3f} | {r['mean_b']:+.3f} | "
                f"{int(r['n_a'])} | {int(r['n_b'])} | "
                f"{r['g']:+.2f} | {r['p']:.3g} | "
                f"{'✓' if r['sig_05'] else '·'} |"
            )
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


# ----------------------------------------------------------------------- #
# Figure
# ----------------------------------------------------------------------- #

def fig_paired_by_family(df: pd.DataFrame, path: Path) -> None:
    """One panel per measure; each panel shows mean paired diff per family
    with 95% CI, sorted by Divergent's diff."""
    measures = list(LABEL.keys())
    measures = [m for m in measures if m in df['measure'].unique()]
    n = len(measures)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.8 * rows),
                             sharey=False)
    axes = axes.ravel() if n > 1 else [axes]
    for i, measure in enumerate(measures):
        ax = axes[i]
        ms = df[df['measure'] == measure]
        means = []; cis = []; fams_present = []
        for fam in FAMILY_ORDER:
            r = ms[ms['family'] == fam]
            if len(r) == 0 or pd.isna(r['mean_diff'].iloc[0]):
                continue
            r = r.iloc[0]
            n_ = int(r['n']); m_ = r['mean_diff']; sd_ = r['sd_diff']
            se = sd_ / np.sqrt(max(n_, 1))
            ci = 1.96 * se
            means.append(m_); cis.append(ci); fams_present.append(fam)
        x = np.arange(len(fams_present))
        bar_colors = [COLORS[f] for f in fams_present]
        ax.bar(x, means, yerr=cis, capsize=4, color=bar_colors,
               edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(fams_present, rotation=20, ha='right', fontsize=8)
        ax.set_title(LABEL[measure], fontsize=9)
        ax.grid(alpha=0.25, axis='y')
    for j in range(len(measures), len(axes)):
        axes[j].axis('off')
    fig.suptitle("User-only measures — paired diff (Persona − GPT) by family, "
                 "with 95% CI", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'wrote {path}')


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #

def main():
    long = load_user_paired_long()
    print('paired-diff long table:', long.shape)
    print(long.groupby('family').size())

    paired = within_family_paired(long)
    paired.to_csv(OUT / '17_user_optionA_paired_by_family.csv', index=False)
    md_paired(paired, OUT / '17_user_optionA_paired_by_family.md')
    print('wrote 17_user_optionA_paired_by_family.{csv,md}')

    pair = pairwise_between_family_diffs(long)
    pair.to_csv(OUT / '17_user_optionA_pairwise_diffs.csv', index=False)
    md_pairwise(pair, OUT / '17_user_optionA_pairwise_diffs.md')
    print('wrote 17_user_optionA_pairwise_diffs.{csv,md}')

    fig_paired_by_family(paired, FIG / 'fig_user_optionA_paired_by_family.png')


if __name__ == '__main__':
    main()
