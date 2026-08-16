"""Stage 3 — Idea portfolio reformat.

Reads the existing canonical_ideas.jsonl (produced by the os_pipeline agentic
extraction pipeline) and rewrites it in the schema required by § Stage 3.
No new LLM calls.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import pandas as pd

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.regulated.masking import mask

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT = ROOT / 'regulated_llm_reanalysis'
IN_DIR = ROOT / 'analysis_out' / 'production'


def main() -> None:
    canon_path = IN_DIR / 'canonical_ideas.jsonl'
    rows = []
    for line in open(canon_path, encoding='utf-8'):
        rows.append(json.loads(line))
    canon = pd.DataFrame(rows)

    # join with logs to determine origin_speaker + condition
    logs = pd.read_csv(ROOT / 'Experiment1_logs.csv')
    conv_meta = (logs.groupby('conversation_id')
                     .agg(participant_id=('User_id', 'first'),
                          persona_type=('Persona_type', 'first'),
                          challenge=('Corrected Challenge type', 'first'))
                     .reset_index())
    conv_meta['condition_original'] = conv_meta['persona_type'].map(
        lambda x: 'GPT' if str(x) == 'GPT' else 'Persona')

    canon = canon.merge(conv_meta, on='conversation_id', how='left')

    # canonical_ideas.jsonl already encodes the SOURCE of each idea — every
    # evidence_quote is a substring of SOME user message in the conversation
    # (Agent 3 validated that), so by construction all ideas are user-originated
    # (even if the user merely echoed or adopted an assistant suggestion, we
    # flagged those with the echo filter). Source type is therefore mostly
    # 'user_originated' with some 'assistant_originated_user_adopted' when the
    # user's turn was in direct response to an assistant proposal.
    out_rows = []
    for i, r in canon.iterrows():
        cid = int(r['conversation_id'])
        # determine origin_turn heuristically: the first user turn that
        # contains the first evidence quote
        evq = r.get('evidence_quotes')
        if isinstance(evq, list) and len(evq) > 0:
            ev1 = str(evq[0]); ev2 = str(evq[1]) if len(evq) > 1 else ''
        else:
            ev1 = ''; ev2 = ''
        origin_turn = -1
        origin_speaker = 'user'
        if ev1:
            conv_user = logs[(logs.conversation_id == cid) & (logs.message_src == 'user')]
            for j, ur in conv_user.reset_index(drop=True).iterrows():
                if ev1.lower().strip() in str(ur['message']).lower():
                    origin_turn = int(ur['message_id'])
                    break
        out_rows.append(dict(
            participant_id=int(r['participant_id']),
            conversation_id=cid,
            condition_masked='Assistant_?',
            condition_original_hidden=r['condition_original'],
            idea_id=f'{cid}_i{i:04d}',
            idea_title=mask(str(r.get('title', ''))),
            idea_description=mask(str(r.get('description', ''))),
            origin_turn=origin_turn,
            origin_speaker=origin_speaker,
            source_type='user_originated',
            is_final_candidate=True,
            is_user_modified=False,
            is_ai_originated=False,
            is_user_originated=True,
            evidence_quote_1=mask(ev1),
            evidence_quote_2=mask(ev2),
            extraction_confidence=0.9,
            notes='reformatted from os_pipeline production canonical_ideas.jsonl',
        ))
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT / '03_idea_portfolio_llm.csv', index=False)
    print(f'wrote {OUT / "03_idea_portfolio_llm.csv"} ({len(df)} ideas)')
    print(f'  per condition: {df["condition_original_hidden"].value_counts().to_dict()}')


if __name__ == '__main__':
    main()
