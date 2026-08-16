"""V6 — Halo-bias audit.

For each judge, compute the Pearson r matrix across the 8 dimensions
(across conversations). If any pair of conceptually distinct dimensions
has |r| > 0.85, that's evidence the judge anchors on one dimension and
lets it bleed into others (halo). Such pairs are flagged for either
prompt-rewrite or construct-merge before primary inference.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.cat_panel.dimensions import DIM_NAMES, DIM_LABELS
from os_pipeline.cat_panel.personas import JUDGE_IDS

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT  = ROOT / 'analysis_out' / 'cat_panel'


def main():
    for ext in ('.parquet', '.csv'):
        fp = OUT / f'panel_scores_adjudicated{ext}'
        if fp.exists():
            fused_path = fp; break
    else:
        raise FileNotFoundError(f'adjudicated scores not found in {OUT}')

    f = (pd.read_parquet(fused_path) if fused_path.suffix == '.parquet'
         else pd.read_csv(fused_path))
    rows = []
    for judge in JUDGE_IDS:
        sub = f[f.judge_id == judge]
        wide = sub.pivot_table(index='conversation_id', columns='dimension',
                               values='score_adj', aggfunc='first')
        if len(wide) < 5: continue
        corr = wide.corr(method='pearson')
        for i, d1 in enumerate(DIM_NAMES):
            for d2 in DIM_NAMES[i+1:]:
                if d1 in corr.columns and d2 in corr.columns:
                    r = corr.loc[d1, d2]
                    rows.append(dict(judge_id=judge,
                                     dim_a=d1, dim_b=d2,
                                     label_a=DIM_LABELS[d1], label_b=DIM_LABELS[d2],
                                     pearson_r=float(r) if not np.isnan(r) else np.nan,
                                     halo_flag=bool(abs(r) > 0.85) if not np.isnan(r) else False))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'audit_halo.csv', index=False)
    print(f'wrote {OUT / "audit_halo.csv"}  ({len(df)} rows)')
    flagged = df[df.halo_flag]
    if len(flagged):
        print(f'\n!!! {len(flagged)} dimension pairs flagged for halo bias '
              '(|r| > 0.85):')
        print(flagged[['judge_id','label_a','label_b','pearson_r']]
              .to_string(index=False))
    else:
        print('\nno halo-flagged pairs.')


if __name__ == '__main__':
    main()
