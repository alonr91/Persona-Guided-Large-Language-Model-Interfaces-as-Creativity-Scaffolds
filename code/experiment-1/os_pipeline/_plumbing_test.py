"""Quick plumbing test: load LLM + embedder, run Agent 1 on a fabricated
user message and verify the JSON-schema pathway works end-to-end.
Runs in ~60 s once model weights are on disk.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from os_pipeline.llm_client import LLMClient
from os_pipeline.agent1_extractor import extract_for_message
from os_pipeline.agent2_consolidator import consolidate
from os_pipeline.agent3_validator import validate

FAKE_USER = (
    "We could add a cafe inside the library with discount coffee for students. "
    "Also, what about hosting monthly open-mic poetry nights for young adults? "
    "I think making the space more social would help."
)
FAKE_ASST = "What features might make libraries more appealing to younger visitors?"

print('[plumbing] loading model...')
LLMClient.load()
print('[plumbing] running Agent 1...')
rows, dbg = extract_for_message(
    conversation_id=999, message_id=1, user_turn_idx=0,
    prev_assistant=FAKE_ASST, user_msg=FAKE_USER,
    challenge='Library', persona_label='Divergent',
)
print(f'valid_json={dbg.get("valid_json")} n_ideas={len(rows)} n_out_tokens={dbg.get("n_output_tokens")}')
print('--- raw output (first 1500 chars) ---')
print((dbg.get('raw_output') or '')[:1500])
print('--- parse error: ', dbg.get('parse_error'))
for r in rows:
    print(f'  • {r.title}  —  {r.description}')
    print(f'    evidence: "{r.evidence_span}"  (conf={r.confidence:.2f})')

if rows:
    print('[plumbing] running Agent 2 + 3...')
    canon = consolidate(rows, conversation_id=999)
    print(f'  canonical: {len(canon)}')
    kept, report = validate(canon, [FAKE_USER])
    for v in report:
        print(f'  {v.title}  [{v.status}]  fuzzy={v.best_fuzzy_score:.0f}')
    print(f'  kept {len(kept)} of {len(canon)}')
print('[plumbing] done')
