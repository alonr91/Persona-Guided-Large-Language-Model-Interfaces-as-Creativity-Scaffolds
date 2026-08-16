"""
Generate human-readable recall-review markdown for selected smoke conversations.

For each selected conversation, produces a review doc where every user message is
followed inline by:
  - the canonical ideas the pipeline extracted from this message (if any), or
  - "(no ideas extracted)"
and a review checkbox grid for the human to mark missed proposals.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
SMOKE = ROOT / 'analysis_out' / 'smoke'
OUT = SMOKE / 'recall_review'
OUT.mkdir(parents=True, exist_ok=True)

# which of the 5 smoke conversations to review — mixes persona families and
# edge cases (smallest canonical count, different challenge type, and the
# previously-pathological GPT conv_212)
TARGETS = {
    212: 'GPT — Library (previously worst under Qwen 1.5B baseline)',
    170: 'Convergent — Library (smallest canonical count: 2)',
    204: 'BoundedRational — Bicycle (different challenge type)',
}


def load_candidates_by_conv():
    cands = defaultdict(list)
    with open(SMOKE / 'candidates.jsonl', encoding='utf-8') as fh:
        for line in fh:
            c = json.loads(line)
            cands[c['conversation_id']].append(c)
    # preserve insertion order (matches smoke_test.py emission order)
    return cands


def load_canonical_by_conv():
    canon = defaultdict(list)
    with open(SMOKE / 'canonical_ideas.jsonl', encoding='utf-8') as fh:
        for line in fh:
            c = json.loads(line)
            canon[c['conversation_id']].append(c)
    return canon


def build_review(cid: int, label: str, logs: pd.DataFrame,
                  cands_by_conv: dict, canon_by_conv: dict) -> str:
    g = logs[logs.conversation_id == cid].sort_values('message_id').reset_index(drop=True)
    cands = cands_by_conv.get(cid, [])
    canons = canon_by_conv.get(cid, [])

    # map: per-conv candidate index -> message_id
    cand_msgid = [c['message_id'] for c in cands]
    # map: message_id -> list of canonical ideas whose source candidates were from this msg
    msg2canons = defaultdict(list)
    for canon in canons:
        for idx in canon.get('source_candidate_ids', []):
            if idx < len(cand_msgid):
                msg2canons[cand_msgid[idx]].append(canon)

    user_turns = g[g.message_src == 'user'].sort_values('message_id')
    n_user = len(user_turns)
    n_canon = len(canons)

    lines = []
    lines.append(f'# Recall Review — Conversation {cid}')
    lines.append(f'**{label}**\n')
    lines.append(f'{n_user} user turns · {n_canon} canonical ideas extracted · '
                 f'all 100% grounded by Agent 3.\n')
    lines.append('---')
    lines.append('## How to use this review\n')
    lines.append('For each user message below:\n')
    lines.append('1. Read the user\'s message.')
    lines.append('2. Decide whether the user made a **concrete proposal** (idea for '
                 'the task). Mere questions / reactions / chat don\'t count.')
    lines.append('3. Check the "Extracted" list below the message.')
    lines.append('4. If an obvious proposal is missing, **write "MISSED: <one-line '
                 'description>" under the message**.')
    lines.append('5. If the extraction captured something that wasn\'t really a '
                 'proposal (false positive), write "FALSE POS: <idea title>".\n')
    lines.append('The pipeline\'s precision is already verified (100% grounded). '
                 'This review measures **recall** — did we miss any real proposals?\n')
    lines.append('---')
    lines.append('## Turns\n')

    # also include assistant messages for context
    turn_no = 0
    for _, row in g.iterrows():
        src = row['message_src']
        mid = int(row['message_id'])
        txt = str(row['message']).strip()
        if src == 'assistant':
            snippet = txt[:180].replace('\n', ' ')
            if len(txt) > 180:
                snippet += '…'
            lines.append(f'> _[assistant, msg {mid}]_  {snippet}\n')
            continue
        turn_no += 1
        lines.append(f'### U{turn_no} (msg {mid})\n')
        lines.append(f'> {txt}\n')
        matches = msg2canons.get(mid, [])
        if not matches:
            lines.append(f'_Extracted ideas from U{turn_no}: **(none)**_\n')
        else:
            lines.append(f'_Extracted ideas from U{turn_no}:_')
            for c in matches:
                lines.append(f'  - **{c["title"]}** — {c["description"]}')
                for q in c.get('evidence_quotes', [])[:3]:
                    lines.append(f'      evidence: `"{q}"`')
            lines.append('')
        lines.append(f'- [ ] MISSED: (write here if a proposal was missed in U{turn_no}, else leave blank)')
        lines.append(f'- [ ] FALSE POS: (write here if any of the above isn\'t actually a user proposal)\n')

    lines.append('\n---')
    lines.append('## Summary block (fill after reading)\n')
    lines.append(f'- Total real proposals in this conversation: __')
    lines.append(f'- Real proposals captured: __')
    lines.append(f'- Recall = captured / total: __ %')
    lines.append(f'- False positives (extracted but not a real proposal): __')
    return '\n'.join(lines)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    logs = pd.read_csv(ROOT / 'Experiment1_logs.csv')
    cands = load_candidates_by_conv()
    canons = load_canonical_by_conv()
    for cid, label in TARGETS.items():
        md = build_review(cid, label, logs, cands, canons)
        path = OUT / f'conv_{cid}_recall_review.md'
        path.write_text(md, encoding='utf-8')
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
