"""V7 — Construct-validity audit.

Correlate the CAT-Panel consensus scores with existing layers:

  (a) Taxonomy-2 user-side discourse stance (from full_stance_predictions.csv)
      Theoretical pairings:
        - panel ideational fluency       ~ T2 user `prop` (propose-new-idea)
        - panel cognitive flexibility    ~ T2 user `ref`  (reframing)
        - panel epistemic stance reg     ~ T2 user `cer`  (certainty, neg.)
        - panel implementation progress  ~ T2 user `com`  (commitment)

  (b) Extracted-idea originality (from participant_originality.csv)
        - panel ideational fluency        ~ n_ideas    (positive)
        - panel implementation progress   ~ orig_all   (positive)
        - panel cognitive flexibility     ~ orig_cross (positive)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.cat_panel.dimensions import DIM_NAMES, DIM_LABELS

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT  = ROOT / 'analysis_out' / 'cat_panel'


def _spearman(a, b):
    a = pd.to_numeric(a, errors='coerce')
    b = pd.to_numeric(b, errors='coerce')
    m = (~a.isna()) & (~b.isna())
    if m.sum() < 6:
        return dict(n=int(m.sum()), rho=np.nan, p=np.nan)
    rho, p = stats.spearmanr(a[m], b[m])
    return dict(n=int(m.sum()), rho=float(rho), p=float(p))


def main():
    for ext in ('.parquet', '.csv'):
        fp = OUT / f'panel_scores_consensus{ext}'
        if fp.exists():
            consensus_path = fp; break
    else:
        raise FileNotFoundError(f'consensus scores not found in {OUT}')

    cons = (pd.read_parquet(consensus_path) if consensus_path.suffix == '.parquet'
            else pd.read_csv(consensus_path))
    # pivot to one row per conversation with all 8 dimension columns
    pivot = cons.pivot_table(index='conversation_id', columns='dimension',
                             values='consensus', aggfunc='first').reset_index()

    rows = []

    # --- (a) Taxonomy-2 user-side stance --------------------------------
    t2_path = ROOT / 'analysis_out' / 'full_stance_predictions.csv'
    if t2_path.exists():
        t2 = pd.read_csv(t2_path)
        # per-conversation user-side means
        t2u = (t2[t2['message_src']=='user']
               .groupby('conversation_id')[['exp','con','cri','cer','com','ref','prop']]
               .mean().reset_index())
        m = pivot.merge(t2u, on='conversation_id', how='left')

        pairings_a = [
            ('user_ideational_fluency',       'prop', 'T2 propose-new-idea'),
            ('user_cognitive_flexibility',    'ref',  'T2 reframing'),
            ('user_epistemic_stance_regulation','cer','T2 certainty (expect negative)'),
            ('user_implementation_relevant_progress','com','T2 commitment'),
            ('user_problem_frame_development','ref',  'T2 reframing'),
            ('user_constraint_integration',   'con',  'T2 contraction'),
        ]
        for dim, t2col, lbl in pairings_a:
            if dim not in m.columns or t2col not in m.columns:
                continue
            r = _spearman(m[dim], m[t2col])
            rows.append(dict(layer='Taxonomy_2_user',
                             panel_dim=dim,
                             external=t2col, external_label=lbl,
                             **r))

    # --- (b) Extracted-idea originality ---------------------------------
    orig_path = ROOT / 'analysis_out' / 'production' / 'participant_originality.csv'
    if orig_path.exists():
        orig = pd.read_csv(orig_path)
        m = pivot.merge(orig[['conversation_id','n_ideas','orig_same',
                              'orig_all','orig_cross']],
                        on='conversation_id', how='left')

        pairings_b = [
            ('user_ideational_fluency',          'n_ideas',    'n canonical ideas'),
            ('user_implementation_relevant_progress','orig_all','orig_all distinctiveness'),
            ('user_cognitive_flexibility',       'orig_cross', 'orig_cross distinctiveness'),
            ('user_problem_frame_development',   'orig_all',   'orig_all distinctiveness'),
        ]
        for dim, ocol, lbl in pairings_b:
            if dim not in m.columns or ocol not in m.columns:
                continue
            r = _spearman(m[dim], m[ocol])
            rows.append(dict(layer='os_pipeline_originality',
                             panel_dim=dim,
                             external=ocol, external_label=lbl,
                             **r))

    df = pd.DataFrame(rows)
    df['panel_dim_label'] = df['panel_dim'].map(DIM_LABELS)
    df = df[['layer','panel_dim','panel_dim_label','external','external_label','n','rho','p']]
    df.to_csv(OUT / 'construct_validity.csv', index=False)

    print(f'wrote {OUT / "construct_validity.csv"} ({len(df)} pairings)')
    print('\n--- construct-validity pairings sorted by |rho| ---')
    print(df.assign(absrho=df['rho'].abs())
            .sort_values('absrho', ascending=False)
            .drop(columns='absrho')
            .to_string(index=False))


if __name__ == '__main__':
    main()
