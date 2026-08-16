"""Stage 7 — Conversation trajectory features.

Aggregates adjudicated episode scores into the fields specified in § H of
the instructions. Splits each conversation into early/mid/late phases by
normalized episode order.
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


def _zsc(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    if s.std(ddof=0) == 0 or s.isna().all():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def main() -> None:
    adj = pd.read_csv(OUT / '05_episode_rubric_scores_adjudicated.csv')
    eps = pd.read_csv(OUT / '02_episode_table.csv')
    # attach episode order within conversation
    eps = eps.sort_values(['conversation_id','start_turn']).reset_index(drop=True)
    eps['ep_order'] = eps.groupby('conversation_id').cumcount()
    eps['ep_total'] = eps.groupby('conversation_id')['ep_order'].transform('max') + 1
    eps['phase'] = pd.cut(eps['ep_order'] / eps['ep_total'],
                           bins=[-0.01, 0.33, 0.67, 1.0],
                           labels=['early','mid','late'])
    adj = adj.merge(eps[['episode_id','ep_order','phase','round']],
                    on='episode_id', how='left')

    wanted = ['exploration_opening','reframing_quality','evaluative_discipline',
              'agency_preservation','anchor_management','coregulation_uptake',
              'timing_fit','implementation_grounding','cognitive_load_clarity',
              'stance_integrity','premature_convergence_risk','runaway_divergence_risk']
    # conversation-level aggregates
    gb = adj.groupby(['conversation_id','criterion'])
    means = gb['final_score'].mean().unstack('criterion')
    # phase-stratified for 3 headline criteria
    phase_means: dict[str, pd.DataFrame] = {}
    for crit in ['exploration_opening','reframing_quality','evaluative_discipline']:
        s = adj[adj.criterion == crit].groupby(['conversation_id','phase'], observed=True)['final_score'].mean().unstack('phase')
        s.columns = [f'{crit}_{c}' for c in s.columns]
        phase_means[crit] = s

    conv_meta = (eps.groupby('conversation_id')
                     .agg(participant_id=('participant_id','first'),
                          condition_original=('condition_original','first'),
                          persona_family_original=('persona_family_original','first'),
                          round=('round','first'),
                          challenge=('challenge','first'),
                          n_episodes=('episode_id','count'))
                     .reset_index())

    out = conv_meta.copy()
    for crit in wanted:
        col = f'{crit}_mean'
        out[col] = out['conversation_id'].map(means[crit]) if crit in means.columns else np.nan
    # user_agency_proxy_mean — approximation via agency_preservation & coregulation
    out['user_agency_proxy_mean'] = (
        pd.to_numeric(out.get('agency_preservation_mean'), errors='coerce').fillna(np.nan)
        + pd.to_numeric(out.get('coregulation_uptake_mean'), errors='coerce').fillna(np.nan)
    ) / 2.0
    for crit, df in phase_means.items():
        for c in df.columns:
            out[c] = out['conversation_id'].map(df[c])

    # regulation_balance_index per § H2
    need = ['exploration_opening_mean','reframing_quality_mean','evaluative_discipline_mean',
            'timing_fit_mean','agency_preservation_mean',
            'premature_convergence_risk_mean','runaway_divergence_risk_mean']
    for c in need:
        if c not in out.columns: out[c] = np.nan
    z = {c: _zsc(out[c]) for c in need}
    out['regulation_balance_index'] = (
        z['exploration_opening_mean'] + z['reframing_quality_mean']
        + z['evaluative_discipline_mean'] + z['timing_fit_mean']
        + z['agency_preservation_mean']
        - z['premature_convergence_risk_mean'] - z['runaway_divergence_risk_mean']
    )

    # trajectory_type — simple rule-based labels (skip clustering until we know
    # silhouette quality; per the instructions we should not overclaim clusters)
    def _ttype(r):
        if pd.notna(r.get('premature_convergence_risk_mean')) and r['premature_convergence_risk_mean'] >= 2.5:
            return 'premature_convergence'
        if pd.notna(r.get('runaway_divergence_risk_mean')) and r['runaway_divergence_risk_mean'] >= 2.5:
            return 'runaway_divergence'
        if pd.notna(r.get('agency_preservation_mean')) and r['agency_preservation_mean'] >= 3.5 \
                and r.get('coregulation_uptake_mean', 0) >= 3.0:
            return 'user_led_elaboration'
        if pd.notna(r.get('reframing_quality_mean')) and r['reframing_quality_mean'] >= 3.0:
            return 'reframe_and_expand'
        if pd.notna(r.get('evaluative_discipline_mean')) and r['evaluative_discipline_mean'] >= 3.0:
            return 'critique_to_commitment'
        if pd.notna(r.get('agency_preservation_mean')) and r['agency_preservation_mean'] < 2.0:
            return 'assistant_led_solution_delivery'
        return 'mixed_or_unclear'
    out['trajectory_type'] = out.apply(_ttype, axis=1)

    out.to_csv(OUT / '07_conversation_trajectory_features.csv', index=False)
    print(f'wrote {OUT / "07_conversation_trajectory_features.csv"} ({len(out)} conv rows)')
    print(f'  trajectory_type: {out["trajectory_type"].value_counts().to_dict()}')


if __name__ == '__main__':
    main()
