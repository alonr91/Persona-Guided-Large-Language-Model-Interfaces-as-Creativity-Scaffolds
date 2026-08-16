"""V4 — Length-bias audit.

For each (judge, dimension), regress the score on conversation word
count and turn count. Report the standardised beta and t-statistic.
Headline effects later should be re-stated as residuals after
partialling out word count.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.cat_panel.scorer import _load_conversations
from os_pipeline.cat_panel.dimensions import DIM_NAMES, DIM_LABELS
from os_pipeline.cat_panel.personas import JUDGE_IDS

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT  = ROOT / 'analysis_out' / 'cat_panel'


def main():
    # accept either parquet or csv
    for ext in ('.parquet', '.csv'):
        fp = OUT / f'panel_scores_adjudicated{ext}'
        if fp.exists():
            fused_path = fp; break
    else:
        raise FileNotFoundError(f'adjudicated scores not found in {OUT}')

    f = (pd.read_parquet(fused_path) if fused_path.suffix == '.parquet'
         else pd.read_csv(fused_path))
    f = f.rename(columns={'score_adj': 'score'})
    # word counts per conversation
    convs = _load_conversations()
    convs['n_words'] = convs['transcript_masked'].str.split().str.len()
    m = f.merge(convs[['conversation_id','n_words','n_turns']],
                on='conversation_id', how='left')

    rows = []
    for judge in JUDGE_IDS:
        for dim in DIM_NAMES:
            sub = m[(m.judge_id == judge) & (m.dimension == dim)]
            v = sub.dropna(subset=['score','n_words'])
            if len(v) < 10:
                continue
            # standardised beta of score on log(n_words)
            x = np.log1p(v['n_words'].astype(float))
            y = v['score'].astype(float)
            slope, intercept, r, p, _ = stats.linregress(x, y)
            rows.append(dict(
                judge_id=judge, dimension=dim, label=DIM_LABELS[dim],
                n=len(v), r_log_words=r, p=p,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'audit_length_bias.csv', index=False)
    print(f'wrote {OUT / "audit_length_bias.csv"}  ({len(df)} rows)')
    print('\nTop length-bias correlations (|r| > 0.30, p < 0.05):')
    sig = df[(df['r_log_words'].abs() > 0.30) & (df['p'] < 0.05)]
    print(sig.sort_values('r_log_words', key=lambda s: s.abs(), ascending=False)
          .to_string(index=False))


if __name__ == '__main__':
    main()
