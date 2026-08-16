"""Stage 5 — Rule-based adjudication → 05_episode_rubric_scores_adjudicated.csv.

For each (episode, criterion):
  - If Scorer B didn't score: use Scorer A directly, record as `single_scorer`.
  - If both scored and |ΔA,B| ≤ 1: keep mean, `adjudication_decision = keep_mean`.
  - If both scored and |ΔA,B| ≥ 2: use the LOWER score (conservative default),
    `use_lower_score`. Flag with `high_disagreement`.
  - If the criterion is `null` in both: `exclude_not_applicable`.
  - If `usable_for_inference=false` in Scorer A: exclude.
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


def main() -> None:
    df = pd.read_csv(OUT / '04_episode_rubric_scores_raw.csv')
    # drop ALL-invalid-json placeholder rows; they're flagged separately
    df = df[df['criterion'] != 'ALL'].copy()

    # pivot to (episode × criterion) with Scorer A and B columns
    wide = df.pivot_table(
        index=['conversation_id','episode_id','criterion'],
        columns='scorer',
        values=['score_0_4','confidence_0_1','usable_for_inference',
                 'evidence_quotes','possible_biases'],
        aggfunc='first',
    )
    wide.columns = [f'{m}_{c}' for m, c in wide.columns]
    wide = wide.reset_index()
    # pull in hidden condition columns from raw
    meta = (df.groupby(['episode_id'])
              [['condition_original_hidden','persona_family_original_hidden',
                 'episode_type','episode_word_count']]
              .first().reset_index())
    wide = wide.merge(meta, on='episode_id', how='left')

    def _adjudicate(row):
        a = row.get('score_0_4_A')
        b = row.get('score_0_4_B')
        ua = row.get('usable_for_inference_A', True)

        if pd.isna(a) and pd.isna(b):
            return pd.Series(dict(final_score=np.nan, decision='exclude_not_applicable',
                                   disagreement=np.nan, high_disagreement=False,
                                   final_confidence=np.nan))
        if pd.isna(ua) is False and bool(ua) is False and pd.isna(b):
            return pd.Series(dict(final_score=np.nan,
                                   decision='exclude_insufficient_evidence',
                                   disagreement=np.nan, high_disagreement=False,
                                   final_confidence=0.0))

        if pd.isna(b):
            # Scorer A only
            return pd.Series(dict(final_score=float(a) if not pd.isna(a) else np.nan,
                                   decision='single_scorer',
                                   disagreement=np.nan, high_disagreement=False,
                                   final_confidence=float(row.get('confidence_0_1_A') or 0.5)))

        if pd.isna(a):
            return pd.Series(dict(final_score=float(b),
                                   decision='scorer_b_only',
                                   disagreement=np.nan, high_disagreement=False,
                                   final_confidence=float(row.get('confidence_0_1_B') or 0.5)))

        d = abs(float(a) - float(b))
        if d <= 1.0:
            return pd.Series(dict(final_score=(float(a) + float(b)) / 2.0,
                                   decision='keep_mean', disagreement=d,
                                   high_disagreement=False,
                                   final_confidence=(float(row.get('confidence_0_1_A') or 0.5) +
                                                    float(row.get('confidence_0_1_B') or 0.5)) / 2.0))
        else:
            # high disagreement -> conservative (lower score)
            return pd.Series(dict(final_score=float(min(a, b)),
                                   decision='use_lower_score', disagreement=d,
                                   high_disagreement=True,
                                   final_confidence=min(float(row.get('confidence_0_1_A') or 0.5),
                                                       float(row.get('confidence_0_1_B') or 0.5))))

    adj_cols = wide.apply(_adjudicate, axis=1)
    out = pd.concat([wide, adj_cols], axis=1)
    cols = ['conversation_id','episode_id','criterion',
            'episode_type','condition_original_hidden','persona_family_original_hidden',
            'episode_word_count',
            'score_0_4_A','score_0_4_B','disagreement','high_disagreement',
            'final_score','decision','final_confidence',
            'usable_for_inference_A','usable_for_inference_B',
            'evidence_quotes_A','evidence_quotes_B',
            'possible_biases_A','possible_biases_B']
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]
    out.to_csv(OUT / '05_episode_rubric_scores_adjudicated.csv', index=False)

    kept = int((~out['final_score'].isna()).sum())
    total = len(out)
    hd = int(out['high_disagreement'].sum())
    dec = out['decision'].value_counts().to_dict()
    print(f'wrote {OUT / "05_episode_rubric_scores_adjudicated.csv"} '
          f'({total} rows, {kept} with final_score)')
    print(f'  decisions: {dec}')
    print(f'  high-disagreement rows (|Δ|≥2): {hd}')


if __name__ == '__main__':
    main()
