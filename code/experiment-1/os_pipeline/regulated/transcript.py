"""Stage 1 — Transcript reconstruction with masking.

Builds a single turn-level parquet with both original and masked text.
Also produces the unmasked lookup keyed by (conversation_id, turn_index)
for post-hoc statistics.
"""
from __future__ import annotations
import os, sys, re
from pathlib import Path
import pandas as pd
import numpy as np

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.regulated.masking import mask

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT = ROOT / 'regulated_llm_reanalysis'
OUT.mkdir(parents=True, exist_ok=True)

FM = {'Divergent':'Divergent','Convergent':'Convergent',
      'strictly rational':'Rational','bounded rationality':'BoundedRational',
      'GPT':'GPT'}


def _count_words(s: str) -> int:
    if not s: return 0
    return len(re.findall(r'\w+', s))


def main() -> Path:
    logs = pd.read_csv(ROOT / 'Experiment1_logs.csv')
    logs = logs.sort_values(['conversation_id', 'message_id']).reset_index(drop=True)

    # conv-level fields
    logs['persona_family_original'] = logs['Persona_type'].map(FM).fillna('Unknown')
    logs['condition_original'] = np.where(logs['Persona_type'] == 'GPT', 'GPT', 'Persona')
    logs['condition_masked']   = 'Assistant_?'   # not revealed to scorer

    # round: round 1 = earlier timestamp per user, round 2 = later
    if 'timestamp' in logs.columns:
        conv_t0 = (logs.groupby('conversation_id')['timestamp'].min()
                     .reset_index(name='t0'))
        conv_t0['user'] = logs.groupby('conversation_id')['User_id'].first().values
        conv_t0 = conv_t0.sort_values(['user', 't0'])
        conv_t0['round'] = conv_t0.groupby('user').cumcount() + 1
        round_map = conv_t0.set_index('conversation_id')['round'].to_dict()
    else:
        round_map = {}

    rows = []
    for cid, g in logs.groupby('conversation_id', sort=False):
        g = g.sort_values('message_id').reset_index(drop=True)
        for ti, r in g.iterrows():
            msg = str(r['message']) if pd.notna(r.get('message')) else ''
            masked_msg = mask(msg)
            rows.append(dict(
                participant_id=int(r['User_id']),
                conversation_id=int(cid),
                condition_original=r['condition_original'],
                condition_masked='Assistant_?',
                persona_family_original=r['persona_family_original'],
                round=round_map.get(int(cid), np.nan),
                challenge=str(r.get('Corrected Challenge type', 'unknown')),
                turn_index=int(ti),
                speaker_original=str(r['message_src']).lower(),
                speaker_masked=('assistant' if str(r['message_src']).lower() == 'assistant' else 'user'),
                message_text_original=msg,
                message_text_masked=masked_msg,
                timestamp=r.get('timestamp') if 'timestamp' in r.index else None,
                timestamp_inferred=('timestamp' not in r.index),
                message_word_count=_count_words(msg),
                message_char_count=len(msg),
                contains_question_mark=('?' in msg),
                language='unknown',  # placeholder; logs aren't labeled per-message
            ))
    turns = pd.DataFrame(rows)
    # parquet might require pyarrow; fall back to CSV if not available
    try:
        import pyarrow  # noqa
        out = OUT / 'turn_table.parquet'
        turns.to_parquet(out, index=False)
    except Exception:
        out = OUT / 'turn_table.csv'
        turns.to_csv(out, index=False)

    # unmasked lookup for post-hoc statistics
    lookup = turns[['conversation_id','turn_index','condition_original',
                     'persona_family_original','round','challenge']].copy()
    lookup.to_csv(OUT / '_unmasked_lookup.csv', index=False)

    print(f'wrote {out} ({len(turns)} turns)')
    print(f'wrote {OUT / "_unmasked_lookup.csv"}')
    return out


if __name__ == '__main__':
    main()
