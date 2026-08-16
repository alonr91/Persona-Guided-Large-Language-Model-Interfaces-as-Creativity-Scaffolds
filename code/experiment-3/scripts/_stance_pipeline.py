"""Zero-shot stance classification on English (or English-translated) messages.

Per assistant turn we score:
  Divergent stance markers  (Taylor's intended stance contract):
    - "the speaker proposes broadening the search by introducing alternatives or new angles"
    - "the speaker asks an open exploratory 'what if' question"
    - "the speaker reframes the problem or invites multiple interpretations"
    - "the speaker uses analogy or metaphor to expand thinking"
    - "the speaker keeps options open and avoids ranking or selecting"

  Convergent stance markers (Alex's intended stance contract):
    - "the speaker articulates explicit criteria or constraints for evaluation"
    - "the speaker compares options and recommends a specific one"
    - "the speaker prioritizes or ranks ideas"
    - "the speaker offers structured stepwise planning"
    - "the speaker critiques or identifies weaknesses in proposed ideas"

We use multi-label zero-shot NLI: each label is independent, score in [0,1].
For analysis we average the 5 divergent labels and the 5 convergent labels per turn,
yielding a (D, C) score pair per turn. The 'manipulation check' compares D and C
distributions across (group × persona) cells.

Model: MoritzLaurer/deberta-v3-large-zeroshot-v2.0 (English, strong NLI)
Falls back to multilingual mDeBERTa for any residual Hebrew. Since translation will
have run upstream, the English model is appropriate for all messages.
"""
import json, sys, time
import torch
from transformers import pipeline

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'

# Load translated set
with open(f'{base}/data/experiment3_messages_en.json','r',encoding='utf-8') as f:
    msgs = json.load(f)

DIVERGENT_HYPOTHESES = [
    "the speaker proposes broadening the search by introducing alternatives or new angles",
    "the speaker asks an open exploratory 'what if' question",
    "the speaker reframes the problem or invites multiple interpretations",
    "the speaker uses analogy or metaphor to expand thinking",
    "the speaker keeps options open and avoids ranking or selecting",
]
CONVERGENT_HYPOTHESES = [
    "the speaker articulates explicit criteria or constraints for evaluation",
    "the speaker compares options and recommends a specific one",
    "the speaker prioritizes or ranks ideas",
    "the speaker offers structured stepwise planning",
    "the speaker critiques or identifies weaknesses in proposed ideas",
]
ALL_HYPS = DIVERGENT_HYPOTHESES + CONVERGENT_HYPOTHESES

print('Loading zero-shot classifier (deberta-v3-base-zeroshot-v2.0)...')
clf = pipeline('zero-shot-classification',
               model='MoritzLaurer/deberta-v3-base-zeroshot-v2.0',
               device=-1)  # CPU
print('Model loaded.')

# We classify ALL messages (user and assistant) — both relevant per analysis A and B
def truncate(text, max_chars=1500):
    return text[:max_chars]

t0 = time.time()
out_rows = []
N = len(msgs)
for i, m in enumerate(msgs):
    text = truncate(m['message_en'])
    if not text.strip():
        continue
    res = clf(text, candidate_labels=ALL_HYPS, multi_label=True)
    score_map = dict(zip(res['labels'], res['scores']))
    div = sum(score_map[h] for h in DIVERGENT_HYPOTHESES) / len(DIVERGENT_HYPOTHESES)
    con = sum(score_map[h] for h in CONVERGENT_HYPOTHESES) / len(CONVERGENT_HYPOTHESES)
    row = {
        'message_id': m['message_id'],
        'conversation_id': m['conversation_id'],
        'message_src': m['message_src'],
        'persona_id': m.get('persona_id'),
        'group': m['group'],
        'agent_role': m['agent_role'],
        'timestamp': m['timestamp'],
        'original_lang': m.get('original_lang','en'),
        'len_chars': len(m['message_en']),
        'divergent_score': round(div, 4),
        'convergent_score': round(con, 4),
        'd_minus_c': round(div - con, 4),
    }
    # Per-hypothesis scores
    for h in ALL_HYPS:
        row[f'h_{h[:30].replace(" ","_")}'] = round(score_map[h], 4)
    out_rows.append(row)
    if (i+1) % 25 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i+1) * (N - i - 1)
        print(f'  {i+1}/{N} elapsed={elapsed:.0f}s eta={eta:.0f}s', flush=True)

# Save
import csv
with open(f'{base}/outputs/stance_per_message.csv','w',encoding='utf-8',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
with open(f'{base}/outputs/stance_per_message.json','w',encoding='utf-8') as f:
    json.dump(out_rows, f, ensure_ascii=False, indent=2)
print(f'\nWrote {len(out_rows)} rows. total time {(time.time()-t0)/60:.1f} min')
