"""Per-family vs GPT (between-group) and within-Persona-arm (within-group)
statistical analysis of rubric scores, with side-by-side Qwen-A vs Gemini-C.

Produces three deliverables in regulated_llm_reanalysis/:

  14_between_group_family_vs_gpt.csv     — each persona family vs GPT
  14_between_group_family_vs_gpt.md      — readable summary
  15_within_arm_pairwise.csv             — Divergent vs Convergent etc.
  15_within_arm_pairwise.md
  16_qwen_vs_gemini_sidebyside.csv       — sign_match table per (criterion × contrast)
  16_qwen_vs_gemini_sidebyside.md

Stats per contrast × criterion × scorer:
  n_a, n_b, mean_a, mean_b, mean_diff, hedges_g, ci_low, ci_high,
  welch_t, welch_df, p_raw, sig_05 (= p_raw < 0.05).

Significance threshold: raw p < 0.05 (no multiple-comparison correction).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass


ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT = ROOT / 'regulated_llm_reanalysis'

CRITERIA_ALL = (
    'exploration_opening', 'reframing_quality', 'evaluative_discipline',
    'agency_preservation', 'anchor_management', 'coregulation_uptake',
    'timing_fit', 'implementation_grounding', 'cognitive_load_clarity',
    'stance_integrity', 'premature_convergence_risk', 'runaway_divergence_risk',
)
PERSONA_FAMILIES = ('Divergent', 'Convergent', 'Rational', 'BoundedRational')


def hedges_g_with_ci(a: np.ndarray, b: np.ndarray) -> dict:
    """Welch test + Hedges' g + 95% CI for the standardised mean difference.

    The CI for g uses the standard large-sample SE:
      SE(g) ≈ sqrt((n_a + n_b)/(n_a*n_b) + g²/(2(n_a+n_b)))
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    if n_a < 3 or n_b < 3:
        return dict(n_a=n_a, n_b=n_b, mean_a=float('nan'), mean_b=float('nan'),
                    mean_diff=float('nan'), hedges_g=float('nan'),
                    ci_low=float('nan'), ci_high=float('nan'),
                    welch_t=float('nan'), welch_df=float('nan'),
                    p_raw=float('nan'))
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    sp2 = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    sp = np.sqrt(sp2) if sp2 > 0 else float('nan')
    d = (a.mean() - b.mean()) / sp if sp and sp > 0 else float('nan')
    J = 1 - 3 / (4 * (n_a + n_b) - 9) if (n_a + n_b) > 2 else 1.0
    g = d * J if not np.isnan(d) else float('nan')

    # 95% CI on g using normal approximation
    if not np.isnan(g):
        se_g = np.sqrt((n_a + n_b) / (n_a * n_b) + g ** 2 / (2 * (n_a + n_b)))
        ci_low, ci_high = g - 1.96 * se_g, g + 1.96 * se_g
    else:
        ci_low = ci_high = float('nan')

    # Welch t-test
    try:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        # Welch–Satterthwaite df
        df = ((var_a / n_a + var_b / n_b) ** 2) / (
            (var_a / n_a) ** 2 / max(n_a - 1, 1)
            + (var_b / n_b) ** 2 / max(n_b - 1, 1)
        )
    except Exception:
        t, p, df = float('nan'), float('nan'), float('nan')

    return dict(
        n_a=n_a, n_b=n_b,
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        mean_diff=float(a.mean() - b.mean()),
        hedges_g=float(g) if not np.isnan(g) else float('nan'),
        ci_low=float(ci_low) if not np.isnan(ci_low) else float('nan'),
        ci_high=float(ci_high) if not np.isnan(ci_high) else float('nan'),
        welch_t=float(t) if t == t else float('nan'),
        welch_df=float(df) if df == df else float('nan'),
        p_raw=float(p) if p == p else float('nan'),
    )


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini–Hochberg adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * n / rank)
        adj[i] = val
        prev = val
    out = np.empty(n)
    for i, pos in enumerate(order):
        out[pos] = adj[i]
    return [float(x) for x in out]


def load_long_scores() -> pd.DataFrame:
    """Return long-format scores with columns:
       episode_id, criterion, scorer, score, condition, persona_family
    Combines Qwen Scorer A (from 04_episode_rubric_scores_raw.csv) with
    Gemini Scorer C (from 04_episode_rubric_scores_raw_scorerC.csv).
    """
    a_path = OUT / '04_episode_rubric_scores_raw.csv'
    c_path = OUT / '04_episode_rubric_scores_raw_scorerC.csv'

    a = pd.read_csv(a_path)
    c = pd.read_csv(c_path)

    a = a[a['scorer'] == 'A'].copy()
    c = c[c['scorer'] == 'C'].copy()

    # Use the hidden-condition columns the scorer attached.
    cols = ['conversation_id', 'episode_id', 'criterion', 'scorer',
            'score_0_4', 'condition_original_hidden',
            'persona_family_original_hidden']
    a = a[cols]
    c = c[cols]
    df = pd.concat([a, c], ignore_index=True)
    df = df.rename(columns={
        'score_0_4': 'score',
        'condition_original_hidden': 'condition',
        'persona_family_original_hidden': 'persona_family',
    })
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    return df


# ----------------------------- Between-group ----------------------------- #

def between_group_family_vs_gpt(df: pd.DataFrame) -> pd.DataFrame:
    """Each persona family vs GPT control, per criterion, per scorer.
    Significance is raw p < 0.05 (no multiple-comparison correction)."""
    rows = []
    for scorer in ('A', 'C'):
        d = df[df['scorer'] == scorer]
        for fam in PERSONA_FAMILIES:
            for crit in CRITERIA_ALL:
                a_vals = d[(d.criterion == crit)
                           & (d.persona_family == fam)
                           & d.score.notna()]['score'].to_numpy()
                b_vals = d[(d.criterion == crit)
                           & (d.condition == 'GPT')
                           & d.score.notna()]['score'].to_numpy()
                stat = hedges_g_with_ci(a_vals, b_vals)
                stat.update(dict(scorer=scorer, contrast=f'{fam} vs GPT',
                                 family=fam, criterion=crit))
                p = stat['p_raw']
                stat['sig_05'] = bool(p < 0.05) if p == p else False
                rows.append(stat)
    cols = ['scorer', 'contrast', 'family', 'criterion',
            'n_a', 'n_b', 'mean_a', 'mean_b', 'mean_diff',
            'hedges_g', 'ci_low', 'ci_high',
            'welch_t', 'welch_df', 'p_raw', 'sig_05']
    return pd.DataFrame(rows)[cols]


# ----------------------------- Within-arm ------------------------------- #

def within_arm_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """All pairwise comparisons among Persona families, per criterion, per scorer.
    GPT is excluded (this is *within* the Persona arm).
    Significance is raw p < 0.05 (no multiple-comparison correction)."""
    pairs = []
    for i, fa in enumerate(PERSONA_FAMILIES):
        for fb in PERSONA_FAMILIES[i + 1:]:
            pairs.append((fa, fb))

    rows = []
    for scorer in ('A', 'C'):
        d = df[df['scorer'] == scorer]
        for fa, fb in pairs:
            for crit in CRITERIA_ALL:
                a_vals = d[(d.criterion == crit)
                           & (d.persona_family == fa)
                           & d.score.notna()]['score'].to_numpy()
                b_vals = d[(d.criterion == crit)
                           & (d.persona_family == fb)
                           & d.score.notna()]['score'].to_numpy()
                stat = hedges_g_with_ci(a_vals, b_vals)
                stat.update(dict(scorer=scorer, contrast=f'{fa} vs {fb}',
                                 family_a=fa, family_b=fb, criterion=crit))
                p = stat['p_raw']
                stat['sig_05'] = bool(p < 0.05) if p == p else False
                rows.append(stat)
    cols = ['scorer', 'contrast', 'family_a', 'family_b', 'criterion',
            'n_a', 'n_b', 'mean_a', 'mean_b', 'mean_diff',
            'hedges_g', 'ci_low', 'ci_high',
            'welch_t', 'welch_df', 'p_raw', 'sig_05']
    return pd.DataFrame(rows)[cols]


# --------------------------- Side-by-side ------------------------------- #

def sidebyside(between_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the between-group table to put Qwen-A and Gemini-C side by side."""
    a = between_df[between_df.scorer == 'A'][
        ['contrast', 'criterion', 'n_a', 'n_b',
         'hedges_g', 'p_raw', 'sig_05']
    ].rename(columns={
        'n_a': 'n_fam_A', 'n_b': 'n_gpt_A',
        'hedges_g': 'g_Qwen', 'p_raw': 'p_Qwen', 'sig_05': 'sig_Qwen'})
    c = between_df[between_df.scorer == 'C'][
        ['contrast', 'criterion',
         'hedges_g', 'p_raw', 'sig_05']
    ].rename(columns={
        'hedges_g': 'g_Gemini', 'p_raw': 'p_Gemini', 'sig_05': 'sig_Gemini'})
    m = a.merge(c, on=['contrast', 'criterion'])

    def sign(x):
        if pd.isna(x):
            return None
        return 1 if x > 0 else (-1 if x < 0 else 0)

    m['sign_match'] = m.apply(
        lambda r: (sign(r.g_Qwen) == sign(r.g_Gemini))
                   if (pd.notna(r.g_Qwen) and pd.notna(r.g_Gemini)) else False,
        axis=1,
    )
    m['both_sig'] = m['sig_Qwen'] & m['sig_Gemini']
    return m


# ----------------------------- Markdown rendering ----------------------- #

def _md_table(df: pd.DataFrame, fmt_cols: dict[str, str]) -> str:
    """Render a small markdown table from a DataFrame with selected columns."""
    cols = list(fmt_cols.keys())
    head = '| ' + ' | '.join(cols) + ' |'
    sep = '|' + '|'.join(['---:' if fmt_cols[c] != 'str' else '---'
                           for c in cols]) + '|'
    rows = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmt_cols[c]
            if f == 'str' or isinstance(v, str):
                cells.append(str(v))
            elif pd.isna(v):
                cells.append('—')
            elif f == 'int':
                cells.append(f'{int(v)}')
            elif f.startswith('f'):
                cells.append(format(float(v), f))
            elif f == 'bool':
                cells.append('✓' if bool(v) else '·')
            else:
                cells.append(str(v))
        rows.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def write_between_md(between_df: pd.DataFrame, path: Path) -> None:
    lines = [
        '# Between-group: each persona family vs GPT',
        '',
        'Each persona family compared to the GPT control arm on every '
        'rubric criterion. Scores are 0–4 episode-level rubric scores. '
        '`g` = Hedges\' g (positive = family higher than GPT). '
        'Significance is raw `p < 0.05` (no multiple-comparison correction). '
        'For the two RISK criteria (premature_convergence_risk, '
        'runaway_divergence_risk), positive g means MORE risk, not better.',
        '',
    ]
    for fam in PERSONA_FAMILIES:
        for scorer, label in (('A', 'Qwen Scorer A'), ('C', 'Gemini Scorer C')):
            sub = between_df[(between_df.family == fam)
                              & (between_df.scorer == scorer)].copy()
            sub = sub.sort_values('criterion')
            lines += [
                f'## {fam} vs GPT — {label}',
                '',
                _md_table(sub, {
                    'criterion': 'str', 'n_a': 'int', 'n_b': 'int',
                    'mean_a': '.2f', 'mean_b': '.2f',
                    'hedges_g': '+.2f', 'ci_low': '+.2f', 'ci_high': '+.2f',
                    'p_raw': '.3g', 'sig_05': 'bool',
                }),
                '',
            ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_within_md(within_df: pd.DataFrame, path: Path) -> None:
    lines = [
        '# Within Persona arm: pairwise family comparisons',
        '',
        'All pairwise comparisons among the four persona families, '
        'excluding the GPT arm. `g` is Hedges\' g (positive = '
        'family_a higher than family_b). Significance is raw `p < 0.05` '
        '(no multiple-comparison correction).',
        '',
    ]
    for contrast in within_df['contrast'].drop_duplicates():
        for scorer, label in (('A', 'Qwen Scorer A'), ('C', 'Gemini Scorer C')):
            sub = within_df[(within_df.contrast == contrast)
                             & (within_df.scorer == scorer)].sort_values('criterion')
            if len(sub) == 0:
                continue
            lines += [
                f'## {contrast} — {label}',
                '',
                _md_table(sub, {
                    'criterion': 'str', 'n_a': 'int', 'n_b': 'int',
                    'mean_a': '.2f', 'mean_b': '.2f',
                    'hedges_g': '+.2f', 'ci_low': '+.2f', 'ci_high': '+.2f',
                    'p_raw': '.3g', 'sig_05': 'bool',
                }),
                '',
            ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_sidebyside_md(side_df: pd.DataFrame, path: Path) -> None:
    lines = [
        '# Side-by-side: Qwen-A vs Gemini-C, per (family vs GPT) × criterion',
        '',
        'Each row is one (family vs GPT) × criterion contrast. `sign_match` is '
        '✓ when the two models agree on the direction of the effect '
        '(both positive or both negative). `both_sig` is ✓ when both reach '
        'raw `p < 0.05` (no multiple-comparison correction).',
        '',
        '## Summary',
        '',
    ]
    sm = side_df.copy()
    sm['sign_match'] = sm['sign_match'].astype(bool)
    summary = sm.groupby('contrast').agg(
        n=('criterion', 'count'),
        sign_match_rate=('sign_match', 'mean'),
        both_sig=('both_sig', 'sum'),
        sig_qwen=('sig_Qwen', 'sum'),
        sig_gemini=('sig_Gemini', 'sum'),
    ).reset_index()
    lines += [
        _md_table(summary, {
            'contrast': 'str', 'n': 'int', 'sign_match_rate': '.0%',
            'both_sig': 'int', 'sig_qwen': 'int', 'sig_gemini': 'int',
        }),
        '',
        '## Per-contrast detail',
        '',
    ]
    for contrast in side_df['contrast'].drop_duplicates():
        sub = side_df[side_df.contrast == contrast].sort_values('criterion')
        lines += [
            f'### {contrast}',
            '',
            _md_table(sub, {
                'criterion': 'str',
                'n_fam_A': 'int', 'n_gpt_A': 'int',
                'g_Qwen': '+.2f', 'p_Qwen': '.3g', 'sig_Qwen': 'bool',
                'g_Gemini': '+.2f', 'p_Gemini': '.3g', 'sig_Gemini': 'bool',
                'sign_match': 'bool', 'both_sig': 'bool',
            }),
            '',
        ]
    path.write_text('\n'.join(lines), encoding='utf-8')


# ----------------------------------- Main ------------------------------- #

def main() -> None:
    df = load_long_scores()

    # Quick summary of cell sizes
    counts = df.groupby(['scorer', 'persona_family']).size().unstack(fill_value=0)
    print('--- score-row counts (criterion × episode rows per cell) ---')
    print(counts)

    between = between_group_family_vs_gpt(df)
    between.to_csv(OUT / '14_between_group_family_vs_gpt.csv', index=False)
    write_between_md(between, OUT / '14_between_group_family_vs_gpt.md')
    print(f'wrote 14_between_group_family_vs_gpt.{{csv,md}}')

    within = within_arm_pairwise(df)
    within.to_csv(OUT / '15_within_arm_pairwise.csv', index=False)
    write_within_md(within, OUT / '15_within_arm_pairwise.md')
    print(f'wrote 15_within_arm_pairwise.{{csv,md}}')

    side = sidebyside(between)
    side.to_csv(OUT / '16_qwen_vs_gemini_sidebyside.csv', index=False)
    write_sidebyside_md(side, OUT / '16_qwen_vs_gemini_sidebyside.md')
    print(f'wrote 16_qwen_vs_gemini_sidebyside.{{csv,md}}')


if __name__ == '__main__':
    main()
