"""Stage 8 — Validation and bias audits → 08_validation_and_bias_audit.csv."""
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


def _icc2(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(2,1)-like estimate: Pearson correlation as a cheap proxy."""
    a = np.asarray(a, dtype='float64')
    b = np.asarray(b, dtype='float64')
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3: return float('nan')
    r = np.corrcoef(a[m], b[m])[0,1]
    return float(r) if np.isfinite(r) else float('nan')


def main() -> None:
    adj = pd.read_csv(OUT / '05_episode_rubric_scores_adjudicated.csv')
    raw = pd.read_csv(OUT / '04_episode_rubric_scores_raw.csv')

    rows: list[dict] = []

    # G1: Multi-agent agreement per criterion
    for crit, g in adj.groupby('criterion'):
        g2 = g.dropna(subset=['score_0_4_A','score_0_4_B'])
        n = len(g2)
        if n == 0:
            rows.append(dict(audit='multi_agent_agreement', criterion=crit,
                             n=0, mean_a=np.nan, mean_b=np.nan,
                             mae_ab=np.nan, pct_abs_ge_2=np.nan, icc=np.nan,
                             flag='no_B_data'))
            continue
        mae = float((g2['score_0_4_A'] - g2['score_0_4_B']).abs().mean())
        pct_ge2 = float(((g2['score_0_4_A'] - g2['score_0_4_B']).abs() >= 2).mean())
        icc = _icc2(g2['score_0_4_A'].values, g2['score_0_4_B'].values)
        rows.append(dict(audit='multi_agent_agreement', criterion=crit,
                         n=n, mean_a=g2['score_0_4_A'].mean(),
                         mean_b=g2['score_0_4_B'].mean(),
                         mae_ab=mae, pct_abs_ge_2=pct_ge2, icc=icc,
                         flag='high_disagreement' if mae >= 1.0 else ''))

    # G4: length bias - score_final ~ word_count + condition_hidden
    for crit, g in adj.groupby('criterion'):
        g2 = g.dropna(subset=['final_score','episode_word_count','condition_original_hidden'])
        if len(g2) < 20: continue
        import statsmodels.api as sm
        try:
            X = pd.get_dummies(g2['condition_original_hidden'], drop_first=True).astype(float)
            X['word_count'] = g2['episode_word_count'].astype(float)
            X = sm.add_constant(X)
            y = g2['final_score'].astype(float)
            m = sm.OLS(y, X, missing='drop').fit()
            cond_cols = [c for c in X.columns if c.startswith('Persona') or c == 'Persona']
            cond_beta = float(m.params.get(cond_cols[0], np.nan)) if cond_cols else np.nan
            word_beta = float(m.params.get('word_count', np.nan))
            cond_p = float(m.pvalues.get(cond_cols[0], np.nan)) if cond_cols else np.nan
            word_p = float(m.pvalues.get('word_count', np.nan))
            flag = 'length_dominates' if (abs(word_beta * g2['episode_word_count'].std())
                                          > abs(cond_beta)) else ''
            rows.append(dict(audit='length_bias', criterion=crit, n=len(g2),
                             cond_beta=cond_beta, cond_p=cond_p,
                             word_beta=word_beta, word_p=word_p, flag=flag))
        except Exception as e:
            rows.append(dict(audit='length_bias', criterion=crit, n=len(g2),
                             flag=f'fit_failed:{type(e).__name__}'))

    # G6: positive control — Divergent vs Rational on exploration_opening;
    # Rational vs Divergent on evaluative_discipline
    for crit, fam_hi, fam_lo in [('exploration_opening','Divergent','Rational'),
                                  ('evaluative_discipline','Rational','Divergent')]:
        g = adj[(adj.criterion==crit) & (adj.final_score.notna())]
        a = g.loc[g.persona_family_original_hidden==fam_hi, 'final_score']
        b = g.loc[g.persona_family_original_hidden==fam_lo, 'final_score']
        if len(a)<5 or len(b)<5:
            rows.append(dict(audit='positive_control', criterion=crit,
                             comparison=f'{fam_hi}>{fam_lo}', n_a=len(a), n_b=len(b),
                             mean_a=a.mean(), mean_b=b.mean(), flag='too_few'))
            continue
        t,p = stats.ttest_ind(a, b, equal_var=False)
        ok = float(a.mean()) > float(b.mean())
        rows.append(dict(audit='positive_control', criterion=crit,
                         comparison=f'{fam_hi}>{fam_lo}', n_a=len(a), n_b=len(b),
                         mean_a=float(a.mean()), mean_b=float(b.mean()),
                         t=float(t), p=float(p),
                         flag=('' if ok else 'failed_positive_control')))

    # G7: negative control — correlation of each criterion's final_score with
    # episode_word_count
    for crit, g in adj.groupby('criterion'):
        g2 = g.dropna(subset=['final_score','episode_word_count'])
        if len(g2) < 20: continue
        r = float(g2['final_score'].corr(g2['episode_word_count']))
        rows.append(dict(audit='negative_control_length', criterion=crit, n=len(g2),
                         corr_len=r,
                         flag='high_length_correlation' if abs(r) > 0.5 else ''))

    # Scorer-A invalid-JSON rate (overall validity metric)
    total_calls = raw['episode_id'].nunique() * raw['scorer'].nunique()
    invalid = (raw['criterion']=='ALL').sum()
    rows.append(dict(audit='scorer_json_validity', criterion='all',
                     n=total_calls,
                     n_invalid=int(invalid),
                     flag=f'invalid_rate={invalid/max(1,total_calls):.3f}'))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / '08_validation_and_bias_audit.csv', index=False)
    print(f'wrote {OUT / "08_validation_and_bias_audit.csv"} ({len(df)} audit rows)')


if __name__ == '__main__':
    main()
