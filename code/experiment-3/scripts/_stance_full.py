# -*- coding: utf-8 -*-
"""Zero-shot stance classification on the NEW Google-translated full-tier corpus.

Same NLI method as _stance_pipeline.py, re-pointed at data/experiment3_full_en.json
(Google-translated) with the new schema (conv_group, persona). Scores every turn on
5 divergent + 5 convergent stance hypotheses; outputs stance_per_message_full.json/csv.
"""
import json, csv, time
from transformers import pipeline

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
with open(f'{base}/data/experiment3_full_en.json', 'r', encoding='utf-8') as f:
    msgs = json.load(f)

DIVERGENT = [
    "the speaker proposes broadening the search by introducing alternatives or new angles",
    "the speaker asks an open exploratory 'what if' question",
    "the speaker reframes the problem or invites multiple interpretations",
    "the speaker uses analogy or metaphor to expand thinking",
    "the speaker keeps options open and avoids ranking or selecting",
]
CONVERGENT = [
    "the speaker articulates explicit criteria or constraints for evaluation",
    "the speaker compares options and recommends a specific one",
    "the speaker prioritizes or ranks ideas",
    "the speaker offers structured stepwise planning",
    "the speaker critiques or identifies weaknesses in proposed ideas",
]
ALL = DIVERGENT + CONVERGENT

print('Loading deberta-v3-base-zeroshot-v2.0 ...', flush=True)
clf = pipeline('zero-shot-classification',
               model='MoritzLaurer/deberta-v3-base-zeroshot-v2.0', device=-1)
print('Model loaded.', flush=True)

t0 = time.time(); rows = []; N = len(msgs)
for i, m in enumerate(msgs):
    text = (m.get('message_en') or m['message'])[:1500]
    if not text.strip():
        continue
    res = clf(text, candidate_labels=ALL, multi_label=True)
    sm = dict(zip(res['labels'], res['scores']))
    div = sum(sm[h] for h in DIVERGENT) / 5
    con = sum(sm[h] for h in CONVERGENT) / 5
    row = {'message_id': m['message_id'], 'conversation_id': m['conversation_id'],
           'message_src': m['message_src'], 'persona_id': m.get('persona_id'),
           'group': m['conv_group'], 'agent_role': m['persona'],
           'timestamp': m['timestamp'], 'original_lang': m.get('original_lang', 'en'),
           'len_chars': len(m.get('message_en') or ''),
           'divergent_score': round(div, 4), 'convergent_score': round(con, 4),
           'd_minus_c': round(div - con, 4)}
    for h in ALL:
        row[f'h_{h[:30].replace(" ", "_")}'] = round(sm[h], 4)
    rows.append(row)
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print(f'  {i+1}/{N} elapsed={el:.0f}s eta={el/(i+1)*(N-i-1):.0f}s', flush=True)

with open(f'{base}/outputs/stance_per_message_full.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(f'{base}/outputs/stance_per_message_full.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print(f'\nWrote {len(rows)} rows. total {(time.time()-t0)/60:.1f} min', flush=True)
