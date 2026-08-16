"""Option B — User-rubric analysis. Mirrors family_analysis.py but on the
6 user-side criteria scored by Gemini in 18_user_rubric_raw_scorerC.csv.

Outputs:
  18_user_rubric_between_group.csv / .md     each family vs GPT
  18_user_rubric_within_arm.csv / .md        Persona-arm pairwise
  figures/fig_user_rubric_radar.png          per-family radar (0-4)
  figures/fig_user_rubric_g_by_family.png    bar plot of g per criterion×family
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
OUT = ROOT / 'regulated_llm_reanalysis'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

CRITERIA = (
    'user_initiative', 'user_question_richness', 'user_proposal_specificity',
    'user_acceptance_yes_and', 'user_reframing', 'user_engagement_depth',
)
FAMILY_ORDER = ('Divergent', 'Convergent', 'Rational', 'BoundedRational')
COLORS = {
    'GPT': '#6c6c6c',
    'Divergent': '#2a9d8f', 'Convergent': '#e76f51',
    'Rational': '#6f62b6', 'BoundedRational': '#e9c46a',
}


# -------- stats helpers --------

def hedges_g_with_ci(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    if n_a < 3 or n_b < 3:
        return dict(n_a=n_a, n_b=n_b, mean_a=float('nan'), mean_b=float('nan'),
                    g=float('nan'), ci_low=float('nan'), ci_high=float('nan'),
                    welch_t=float('nan'), p_raw=float('nan'))
    va = a.var(ddof=1); vb = b.var(ddof=1)
    sp2 = ((n_a-1)*va + (n_b-1)*vb) / (n_a+n_b-2)
    sp = np.sqrt(sp2) if sp2 > 0 else float('nan')
    d = (a.mean()-b.mean())/sp if sp and sp > 0 else float('nan')
    J = 1 - 3/(4*(n_a+n_b)-9)
    g = d*J if not np.isnan(d) else float('nan')
    if not np.isnan(g):
        se = np.sqrt((n_a+n_b)/(n_a*n_b) + g**2 / (2*(n_a+n_b)))
        lo, hi = g - 1.96*se, g + 1.96*se
    else:
        lo = hi = float('nan')
    try:
        t, p = stats.ttest_ind(a, b, equal_var=False)
    except Exception:
        t, p = float('nan'), float('nan')
    return dict(
        n_a=n_a, n_b=n_b, mean_a=float(a.mean()), mean_b=float(b.mean()),
        g=float(g) if not np.isnan(g) else float('nan'),
        ci_low=float(lo) if not np.isnan(lo) else float('nan'),
        ci_high=float(hi) if not np.isnan(hi) else float('nan'),
        welch_t=float(t) if t == t else float('nan'),
        p_raw=float(p) if p == p else float('nan'),
    )


def bh_fdr(p):
    p = np.asarray([x if x == x else 1.0 for x in p], dtype=float)
    n = len(p)
    if n == 0: return []
    order = np.argsort(p); ranked = p[order]
    out = np.empty(n); prev = 1.0
    for i in range(n-1, -1, -1):
        out[i] = min(prev, ranked[i] * n / (i+1)); prev = out[i]
    res = np.empty(n)
    for i, pos in enumerate(order):
        res[pos] = out[i]
    return [float(x) for x in res]


# -------- load --------

def load_long():
    df = pd.read_csv(OUT / '18_user_rubric_raw_scorerC.csv')
    df['score'] = pd.to_numeric(df['score_0_4'], errors='coerce')
    df = df.rename(columns={'condition_original_hidden': 'condition',
                             'persona_family_original_hidden': 'persona_family'})
    return df[['conversation_id', 'episode_id', 'criterion', 'score',
               'condition', 'persona_family']]


# -------- between/within --------

def between(df) -> pd.DataFrame:
    """Significance is raw p < 0.05 (no multiple-comparison correction)."""
    rows = []
    for fam in FAMILY_ORDER:
        for crit in CRITERIA:
            a = df[(df.criterion == crit) & (df.persona_family == fam) & df.score.notna()]['score'].to_numpy()
            b = df[(df.criterion == crit) & (df.condition == 'GPT') & df.score.notna()]['score'].to_numpy()
            stat = hedges_g_with_ci(a, b)
            stat.update(dict(family=fam, contrast=f'{fam} vs GPT', criterion=crit))
            p = stat['p_raw']
            stat['sig_05'] = bool(p < 0.05) if p == p else False
            rows.append(stat)
    cols = ['family', 'contrast', 'criterion', 'n_a', 'n_b',
            'mean_a', 'mean_b', 'g', 'ci_low', 'ci_high',
            'welch_t', 'p_raw', 'sig_05']
    return pd.DataFrame(rows)[cols]


def within(df) -> pd.DataFrame:
    """Significance is raw p < 0.05 (no multiple-comparison correction)."""
    pairs = []
    for i, fa in enumerate(FAMILY_ORDER):
        for fb in FAMILY_ORDER[i+1:]:
            pairs.append((fa, fb))
    rows = []
    for fa, fb in pairs:
        for crit in CRITERIA:
            a = df[(df.criterion == crit) & (df.persona_family == fa) & df.score.notna()]['score'].to_numpy()
            b = df[(df.criterion == crit) & (df.persona_family == fb) & df.score.notna()]['score'].to_numpy()
            stat = hedges_g_with_ci(a, b)
            stat.update(dict(family_a=fa, family_b=fb,
                             contrast=f'{fa} vs {fb}', criterion=crit))
            p = stat['p_raw']
            stat['sig_05'] = bool(p < 0.05) if p == p else False
            rows.append(stat)
    cols = ['contrast', 'family_a', 'family_b', 'criterion', 'n_a', 'n_b',
            'mean_a', 'mean_b', 'g', 'ci_low', 'ci_high',
            'welch_t', 'p_raw', 'sig_05']
    return pd.DataFrame(rows)[cols]


# -------- markdown --------

def md_between(df, path):
    lines = ['# User-rubric — between-group (each persona family vs GPT)', '',
             'Each persona family compared to the GPT control arm on the 6 '
             "user-behaviour criteria. Positive g = persona family scored "
             "higher on user-side measure. Significance is raw `p < 0.05` "
             "(no multiple-comparison correction).", '']
    for fam in FAMILY_ORDER:
        sub = df[df.family == fam].copy().sort_values('criterion')
        if len(sub) == 0: continue
        lines += [f'## {fam} vs GPT', '',
                  '| criterion | n_fam | n_gpt | mean_fam | mean_gpt | g | 95% CI | p | sig |',
                  '|---|---:|---:|---:|---:|---:|:---:|---:|---:|']
        for _, r in sub.iterrows():
            ci = f'[{r.ci_low:+.2f}, {r.ci_high:+.2f}]' if not np.isnan(r.ci_low) else '—'
            lines.append(
                f"| {r.criterion} | {int(r.n_a)} | {int(r.n_b)} | "
                f"{r.mean_a:.2f} | {r.mean_b:.2f} | {r.g:+.2f} | {ci} | "
                f"{r.p_raw:.3g} | "
                f"{'✓' if r.sig_05 else '·'} |"
            )
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def md_within(df, path):
    lines = ['# User-rubric — within-Persona-arm pairwise contrasts', '',
             'All pairwise comparisons among the four persona families on '
             'the 6 user-behaviour criteria. Significance is raw `p < 0.05` '
             '(no multiple-comparison correction).', '']
    for c in df['contrast'].drop_duplicates():
        sub = df[df.contrast == c].copy().sort_values('criterion')
        lines += [f'## {c}', '',
                  '| criterion | n_a | n_b | mean_a | mean_b | g | 95% CI | p | sig |',
                  '|---|---:|---:|---:|---:|---:|:---:|---:|---:|']
        for _, r in sub.iterrows():
            ci = f'[{r.ci_low:+.2f}, {r.ci_high:+.2f}]' if not np.isnan(r.ci_low) else '—'
            lines.append(
                f"| {r.criterion} | {int(r.n_a)} | {int(r.n_b)} | "
                f"{r.mean_a:.2f} | {r.mean_b:.2f} | {r.g:+.2f} | {ci} | "
                f"{r.p_raw:.3g} | "
                f"{'✓' if r.sig_05 else '·'} |"
            )
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


# -------- figures --------

def fig_radar(df, path):
    fams = ['GPT'] + list(FAMILY_ORDER)
    means = {fam: {} for fam in fams}
    for fam in fams:
        for crit in CRITERIA:
            sub = df[(df.criterion == crit) &
                     ((df.persona_family == fam)
                      if fam != 'GPT' else (df.condition == 'GPT'))
                     & df.score.notna()]['score']
            means[fam][crit] = float(sub.mean()) if len(sub) else float('nan')

    angles = np.linspace(0, 2*np.pi, len(CRITERIA), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, polar=True)
    for fam in fams:
        vals = [means[fam][c] for c in CRITERIA]
        vals += vals[:1]
        ax.plot(angles, vals, marker='o', label=fam,
                color=COLORS.get(fam, 'black'), linewidth=1.5)
        ax.fill(angles, vals, alpha=0.08, color=COLORS.get(fam, 'black'))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('_', '\n') for c in CRITERIA], fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_title('User-behaviour rubric — profile per persona family '
                 '(Gemini Scorer C, 0–4)', y=1.08)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, -0.05), fontsize=8)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'wrote {path}')


def fig_g_by_family(between_df, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    crits = list(CRITERIA)
    fams = list(FAMILY_ORDER)
    width = 0.18
    x = np.arange(len(crits))
    for i, fam in enumerate(fams):
        gs = []; cis = []
        for crit in crits:
            r = between_df[(between_df.family == fam) & (between_df.criterion == crit)]
            r = r.iloc[0] if len(r) else None
            if r is None or pd.isna(r['g']):
                gs.append(0); cis.append(0)
            else:
                gs.append(r['g'])
                cis.append((r['ci_high'] - r['ci_low']) / 2 if not np.isnan(r['ci_high']) else 0)
        offs = (i - (len(fams)-1)/2) * width
        ax.bar(x + offs, gs, width, yerr=cis, label=fam,
               color=COLORS[fam], capsize=2, edgecolor='black', linewidth=0.4)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in crits], fontsize=9)
    ax.set_ylabel("Hedges' g (family − GPT, episode-level)")
    ax.set_title("User-rubric: per-family vs GPT effect sizes (95% CI)")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.25, axis='y')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'wrote {path}')


def main():
    df = load_long()
    print(f'rows: {len(df)}')
    print(df.groupby(['persona_family']).size())

    b = between(df)
    b.to_csv(OUT / '18_user_rubric_between_group.csv', index=False)
    md_between(b, OUT / '18_user_rubric_between_group.md')
    print('wrote 18_user_rubric_between_group.{csv,md}')

    w = within(df)
    w.to_csv(OUT / '18_user_rubric_within_arm.csv', index=False)
    md_within(w, OUT / '18_user_rubric_within_arm.md')
    print('wrote 18_user_rubric_within_arm.{csv,md}')

    fig_radar(df, FIG / 'fig_user_rubric_radar.png')
    fig_g_by_family(b, FIG / 'fig_user_rubric_g_by_family.png')


if __name__ == '__main__':
    main()
