"""Cross-model agreement: Qwen Scorer A vs Gemini Scorer C.

Reads the two rubric-score files, joins on (episode_id, criterion), and
computes per-criterion agreement metrics:

  - n           : number of dual-scored rows where both scorers gave a non-null score
  - mean_abs_d  : mean |score_A - score_C|  (lower = better agreement)
  - pct_d_ge2   : fraction of rows with |score_A - score_C| >= 2 (high disagreement)
  - kappa_qw    : quadratic-weighted Cohen's kappa on the 0-4 ordinal
  - spearman_r  : Spearman rank correlation
  - spearman_p  : Spearman p-value
  - mean_A, mean_C : marginal means

Outputs:
  - regulated_llm_reanalysis/13_cross_model_agreement.csv
  - regulated_llm_reanalysis/13_cross_model_agreement.md

NOTE: Scorer C is informational. The adjudicator and published numbers are
unchanged; this report is appended as a robustness audit.
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

SCORES_AB_PATH = OUT / '04_episode_rubric_scores_raw.csv'
SCORES_C_PATH = OUT / '04_episode_rubric_scores_raw_scorerC.csv'
OUT_CSV = OUT / '13_cross_model_agreement.csv'
OUT_MD = OUT / '13_cross_model_agreement.md'


def quadratic_weighted_kappa(y1: np.ndarray, y2: np.ndarray, n_classes: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal ratings on integers 0..n_classes-1.
    Standard formula; returns NaN if either rater has zero variance."""
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    if len(y1) == 0:
        return float('nan')

    # Confusion matrix
    O = np.zeros((n_classes, n_classes), dtype=float)
    for a, b in zip(y1, y2):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1

    # Quadratic weights
    W = np.zeros((n_classes, n_classes), dtype=float)
    denom = (n_classes - 1) ** 2
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / denom

    # Expected matrix from marginals
    h1 = O.sum(axis=1)
    h2 = O.sum(axis=0)
    n = O.sum()
    if n == 0:
        return float('nan')
    E = np.outer(h1, h2) / n

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return float('nan')
    return float(1.0 - num / den)


def main():
    if not SCORES_AB_PATH.exists():
        raise FileNotFoundError(f'missing {SCORES_AB_PATH}')
    if not SCORES_C_PATH.exists():
        raise FileNotFoundError(
            f'missing {SCORES_C_PATH} -- run scorer with --scorer-c first'
        )

    ab = pd.read_csv(SCORES_AB_PATH)
    c = pd.read_csv(SCORES_C_PATH)

    a = ab[ab['scorer'] == 'A'].copy()
    c = c[c['scorer'] == 'C'].copy()

    # Inner-join on (episode_id, criterion); only rows with both A and C are usable.
    merged = a.merge(
        c, on=['episode_id', 'criterion'],
        suffixes=('_A', '_C'), how='inner',
    )
    # Keep only rows where both scores are non-null and within 0..4
    m = merged.dropna(subset=['score_0_4_A', 'score_0_4_C']).copy()
    m['score_0_4_A'] = m['score_0_4_A'].astype(float)
    m['score_0_4_C'] = m['score_0_4_C'].astype(float)
    m = m[(m['score_0_4_A'].between(0, 4)) & (m['score_0_4_C'].between(0, 4))]

    rows = []
    for crit, sub in m.groupby('criterion'):
        if len(sub) == 0:
            continue
        a_vals = sub['score_0_4_A'].to_numpy()
        c_vals = sub['score_0_4_C'].to_numpy()
        diffs = np.abs(a_vals - c_vals)
        try:
            sr, sp = stats.spearmanr(a_vals, c_vals)
        except Exception:
            sr, sp = float('nan'), float('nan')
        kqw = quadratic_weighted_kappa(np.round(a_vals).astype(int),
                                       np.round(c_vals).astype(int))
        rows.append({
            'criterion': crit,
            'n': int(len(sub)),
            'mean_A': float(a_vals.mean()),
            'mean_C': float(c_vals.mean()),
            'mean_abs_d': float(diffs.mean()),
            'pct_d_ge2': float((diffs >= 2).mean()),
            'kappa_qw': kqw,
            'spearman_r': float(sr) if sr == sr else float('nan'),
            'spearman_p': float(sp) if sp == sp else float('nan'),
        })

    res = pd.DataFrame(rows).sort_values('criterion').reset_index(drop=True)

    # Overall summary across all criteria pooled
    if len(m):
        all_a = m['score_0_4_A'].to_numpy()
        all_c = m['score_0_4_C'].to_numpy()
        all_d = np.abs(all_a - all_c)
        sr, sp = stats.spearmanr(all_a, all_c)
        kqw = quadratic_weighted_kappa(np.round(all_a).astype(int),
                                       np.round(all_c).astype(int))
        overall = {
            'criterion': '__pooled__',
            'n': int(len(m)),
            'mean_A': float(all_a.mean()),
            'mean_C': float(all_c.mean()),
            'mean_abs_d': float(all_d.mean()),
            'pct_d_ge2': float((all_d >= 2).mean()),
            'kappa_qw': kqw,
            'spearman_r': float(sr) if sr == sr else float('nan'),
            'spearman_p': float(sp) if sp == sp else float('nan'),
        }
        res = pd.concat([res, pd.DataFrame([overall])], ignore_index=True)

    res.to_csv(OUT_CSV, index=False)
    print(f'[cross_model_agreement] wrote {OUT_CSV}')

    # Markdown summary
    lines = [
        '# Cross-model agreement: Qwen Scorer A vs Gemini Scorer C',
        '',
        'Same prompt (SYSTEM_A), different model. Computed on the dual-scored '
        'subset (episodes also scored by Scorer B in the original run).',
        '',
        'Reading: `kappa_qw` = quadratic-weighted Cohen\'s kappa on the 0-4 '
        'ordinal (>=0.6 substantial; >=0.8 strong). `mean_abs_d` is the mean '
        'absolute score difference per criterion (lower is better). '
        '`pct_d_ge2` is the fraction of dual-scored rows where the two models '
        'disagree by 2 points or more.',
        '',
        '| criterion | n | mean_A | mean_C | mean\\|Δ\\| | %\\|Δ\\|≥2 | κ (qw) | ρ | p |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for _, r in res.iterrows():
        lines.append(
            f"| {r['criterion']} | {int(r['n'])} | {r['mean_A']:.2f} | "
            f"{r['mean_C']:.2f} | {r['mean_abs_d']:.2f} | "
            f"{r['pct_d_ge2']*100:.1f}% | {r['kappa_qw']:.2f} | "
            f"{r['spearman_r']:.2f} | {r['spearman_p']:.3g} |"
        )
    lines += [
        '',
        'Scorer C is **informational**: it is not used by the adjudicator and '
        'does not affect the final scores in '
        '`05_episode_rubric_scores_adjudicated.csv` or any downstream '
        'statistics. Its purpose is to convert the original prompt-paraphrase '
        'robustness check (Scorer A vs Scorer B, same model) into a true '
        'cross-model agreement signal.',
        '',
    ]
    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[cross_model_agreement] wrote {OUT_MD}')


if __name__ == '__main__':
    main()
