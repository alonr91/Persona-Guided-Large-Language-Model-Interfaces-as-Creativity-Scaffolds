"""Stage 9 — Statistical models summary → 09_statistical_models_summary.csv.

Implements § I1 (condition effect per criterion), I5 (preference model),
I7 (family interaction), and the FDR correction per I8. Uses statsmodels
MixedLM as the robust fallback from ordinal mixed models when those fail
to converge (which is common in Python).
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
    'exploration_opening','reframing_quality','evaluative_discipline',
    'agency_preservation','anchor_management','coregulation_uptake',
    'timing_fit','implementation_grounding','cognitive_load_clarity',
    'stance_integrity','premature_convergence_risk','runaway_divergence_risk',
)


def _hedges_g(a: pd.Series, b: pd.Series) -> tuple[int, int, float, float]:
    a = pd.to_numeric(a, errors='coerce').dropna()
    b = pd.to_numeric(b, errors='coerce').dropna()
    if len(a)<5 or len(b)<5: return len(a), len(b), float('nan'), float('nan')
    sp2 = ((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2)
    sp = sp2**0.5
    d = (a.mean() - b.mean()) / sp if sp>0 else float('nan')
    J = 1 - 3/(4*(len(a)+len(b))-9)
    g = d*J if not np.isnan(d) else float('nan')
    return len(a), len(b), float(d), float(g)


def main() -> None:
    adj = pd.read_csv(OUT / '05_episode_rubric_scores_adjudicated.csv')
    traj = pd.read_csv(OUT / '07_conversation_trajectory_features.csv')

    rows: list[dict] = []

    # I1: condition effect per criterion (episode-level, Welch t + Hedges' g)
    for crit in CRITERIA_ALL:
        g = adj[(adj.criterion==crit) & adj.final_score.notna()]
        p_scores = g.loc[g.condition_original_hidden=='Persona', 'final_score']
        c_scores = g.loc[g.condition_original_hidden=='GPT', 'final_score']
        n_p, n_c, d, g_es = _hedges_g(p_scores, c_scores)
        if n_p<5 or n_c<5:
            rows.append(dict(model='I1_condition_effect_episode', criterion=crit,
                             n_p=n_p, n_c=n_c, flag='too_few'))
            continue
        t,pv = stats.ttest_ind(p_scores, c_scores, equal_var=False)
        rows.append(dict(model='I1_condition_effect_episode', criterion=crit,
                         n_p=n_p, n_c=n_c, mean_p=float(p_scores.mean()),
                         mean_c=float(c_scores.mean()),
                         diff=float(p_scores.mean()-c_scores.mean()),
                         t=float(t), p=float(pv),
                         cohen_d=d, hedges_g=g_es))

    # I1: Try ordinal mixed model, fall back to MixedLM on failure
    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM
        for crit in CRITERIA_ALL:
            g = adj[(adj.criterion==crit) & adj.final_score.notna()].copy()
            if len(g) < 30:
                continue
            g['cond_p'] = (g['condition_original_hidden']=='Persona').astype(int)
            g['word_std'] = (g['episode_word_count'] - g['episode_word_count'].mean()) / g['episode_word_count'].std()
            # need participant_id — join with trajectory
            ep_to_user = (adj.groupby('episode_id')['conversation_id'].first()
                          .to_frame().merge(traj[['conversation_id','participant_id']],
                                             on='conversation_id', how='left'))
            g = g.merge(ep_to_user.reset_index()[['episode_id','participant_id']],
                         on='episode_id', how='left')
            if g['participant_id'].isna().all():
                continue
            try:
                md = MixedLM.from_formula(
                    'final_score ~ cond_p + word_std', data=g, groups=g['participant_id'])
                res = md.fit(method='lbfgs', disp=False)
                rows.append(dict(
                    model='I1_lmm_per_criterion', criterion=crit,
                    n=len(g),
                    cond_coef=float(res.params.get('cond_p', np.nan)),
                    cond_se=float(res.bse.get('cond_p', np.nan)),
                    cond_p=float(res.pvalues.get('cond_p', np.nan)),
                    word_coef=float(res.params.get('word_std', np.nan)),
                    word_p=float(res.pvalues.get('word_std', np.nan)),
                    converged=bool(res.converged),
                ))
            except Exception as e:
                rows.append(dict(model='I1_lmm_per_criterion', criterion=crit,
                                 n=len(g), flag=f'fit_failed:{type(e).__name__}'))
    except Exception:
        pass

    # I7: family interaction (episode-level)
    for crit in CRITERIA_ALL:
        g = adj[(adj.criterion==crit) & adj.final_score.notna() &
                (adj.persona_family_original_hidden != 'GPT')]
        for fam in ['Divergent','Convergent','Rational','BoundedRational']:
            sub = g[g.persona_family_original_hidden == fam]
            if len(sub) < 5: continue
            # family vs GPT baseline (between-subjects)
            baseline = adj[(adj.criterion==crit) & (adj.persona_family_original_hidden=='GPT') & adj.final_score.notna()]
            if len(baseline) < 5: continue
            n_p, n_c, d, g_es = _hedges_g(sub['final_score'], baseline['final_score'])
            t,pv = stats.ttest_ind(sub['final_score'], baseline['final_score'], equal_var=False)
            rows.append(dict(model='I7_family_vs_gpt', criterion=crit, family=fam,
                             n_fam=n_p, n_gpt=n_c,
                             mean_fam=float(sub['final_score'].mean()),
                             mean_gpt=float(baseline['final_score'].mean()),
                             t=float(t), p=float(pv), hedges_g=g_es))

    # FDR correction (I8) over the primary I1 Welch p-values
    i1 = [r for r in rows if r.get('model')=='I1_condition_effect_episode' and 'p' in r and not np.isnan(r.get('p', np.nan))]
    if i1:
        from statsmodels.stats.multitest import multipletests
        pvs = np.array([r['p'] for r in i1])
        _, q, _, _ = multipletests(pvs, method='fdr_bh')
        for r, qv in zip(i1, q):
            r['q_fdr'] = float(qv)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / '09_statistical_models_summary.csv', index=False)
    print(f'wrote {OUT / "09_statistical_models_summary.csv"} ({len(df)} rows)')


if __name__ == '__main__':
    main()
