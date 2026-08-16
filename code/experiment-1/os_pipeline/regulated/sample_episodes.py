"""Stratified sampler for the rubric-scoring stage.

Selects ~N episodes to maximize coverage across (condition × family ×
episode_type × round). Target sample size ~200 for the Scorer-A pass and a
subset of ~50 for the Scorer-B dual-scoring pass.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT = ROOT / 'regulated_llm_reanalysis'

TARGET_SCORER_A = 200
TARGET_SCORER_B = 50
SEED = 7


def pick_stratified(df: pd.DataFrame, target: int, seed: int = SEED) -> list[str]:
    rng = np.random.default_rng(seed)
    pool = df[df['usable_for_scoring'] == True].copy()
    # prioritize episodes that carry more signal
    priority = {
        'ideation_burst': 5, 'reframe_event': 5, 'commitment_event': 5,
        'critique_event': 4, 'anchor_return': 4,
        'implementation_grounding_event': 4, 'user_agency_event': 4,
        'repair_event': 3, 'summary_or_consolidation': 3,
        'opening_frame': 2, 'other': 1,
    }
    pool['_pri'] = pool['episode_type'].map(priority).fillna(1)
    # build strata: (condition × family × round)
    strata = pool.groupby(['condition_original','persona_family_original','round'])
    n_strata = strata.ngroups
    per_stratum = max(2, target // max(1, n_strata))
    picked = []
    for keys, g in strata:
        g2 = g.sort_values('_pri', ascending=False)
        k = min(per_stratum, len(g2))
        idx = rng.choice(g2.index.to_numpy(), size=k, replace=False)
        picked.extend(idx.tolist())
    # if under target, top up from remaining pool (priority-weighted)
    remaining = pool.index.difference(picked)
    if len(picked) < target and len(remaining):
        more = pool.loc[remaining].sort_values('_pri', ascending=False).head(target - len(picked))
        picked.extend(more.index.tolist())
    if len(picked) > target:
        picked = list(rng.choice(np.asarray(picked), size=target, replace=False))
    return df.loc[picked, 'episode_id'].tolist()


def main() -> None:
    eps = pd.read_csv(OUT / '02_episode_table.csv')
    sel_a = pick_stratified(eps, TARGET_SCORER_A, seed=SEED)
    sub_b_source = eps[eps.episode_id.isin(sel_a)]
    sel_b = pick_stratified(sub_b_source, TARGET_SCORER_B, seed=SEED + 1)

    sample_a = eps[eps.episode_id.isin(sel_a)].copy()
    sample_a['scorer_A'] = True
    sample_a['scorer_B'] = sample_a['episode_id'].isin(sel_b)
    sample_a.to_csv(OUT / '_scoring_sample.csv', index=False)

    n_a = int(sample_a['scorer_A'].sum())
    n_b = int(sample_a['scorer_B'].sum())
    by_cond = sample_a.groupby(['condition_original','persona_family_original']).size().reset_index(name='n')
    print(f'wrote {OUT / "_scoring_sample.csv"}')
    print(f'  Scorer A: {n_a} episodes')
    print(f'  Scorer B: {n_b} episodes (subset)')
    print()
    print('coverage (condition × family):')
    print(by_cond.to_string(index=False))
    print()
    print('coverage by episode_type:')
    print(sample_a['episode_type'].value_counts().to_string())


if __name__ == '__main__':
    main()
