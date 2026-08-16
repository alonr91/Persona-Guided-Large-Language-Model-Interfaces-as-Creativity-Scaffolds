# -*- coding: utf-8 -*-
"""Idea-portfolio extraction for Experiment 3 using EXPERIMENT 1's method.

Runs Experiment 1's os_pipeline agents on the Exp-3 full-tier conversations:
  Agent 1 (local Qwen3-4B OV-INT4): per user message, extract user-originated ideas
  Agent 2: per-conversation consolidation (bge-large-en-v1.5 + agglomerative + LLM canonicalize)
  Agent 4: corpus-level categorization (HDBSCAN + LLM labels)
  Agent 5: participant-centroid originality (same-condition / all / cross-NN)

Per-conversation outputs (creativity measures):
  fluency           = number of canonical ideas               (QUANTITATIVE)
  flexibility       = number of distinct categories spanned    (QUANTITATIVE)
  orig_same/all/cross = idea-centroid originality              (SEMANTIC)
  within_idea_diversity = mean pairwise distance among own ideas (SEMANTIC)

Caches Agent-1 candidates for resume. Outputs outputs/idea_portfolio_exp1.csv
and outputs/ideas_canonical_exp1.csv.
"""
import sys, json, csv, time
from collections import defaultdict
import numpy as np

EXP1 = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
sys.path.insert(0, EXP1)

from os_pipeline import config as cfg
from os_pipeline.llm_client import LLMClient
from os_pipeline.agent1_extractor import extract_for_message, CandidateRow
from os_pipeline.agent2_consolidator import consolidate, _get_embedder
from os_pipeline.agent4_categorizer import categorize
from os_pipeline.agent5_originality import compute_centroids, compute_originality

CHAL = {
 'galilee_upper':'Upper Galilee: connect older and younger populations to the kibbutz physical and social space.',
 'eshkol_nir_yitzhak':'Eshkol southern kibbutzim cooperation to grow shared community and economic capital.',
 'natal_trauma_language':'NATAL: build a language reflecting mental-distress states in Israel since Oct 7.',
 'sderot_wellbeing':'Sderot: improve the mood and sense of meaning of returning residents.',
 'polyron_sleep':'Polyron: new therapeutic sleep products, the next-generation mattress, for trauma and rehab.',
 'ichilov_rehab_future':'Ichilov: the rehabilitation hospital of the future for rehab patients, amputees, combat-injured.',
 'joint_rikma_jewish_arab':'Joint Rikma: Jewish-Arab organizations, empathy and inclusive culture in mixed workplaces.',
 'ta_south_community':'Tel Aviv South: make diverse populations feel represented; shared public meeting points for neighbors.',
 'ta_east_reut_yad_eliyahu':'Tel Aviv East: Reut rehabilitation hospital and the Yad Eliyahu neighborhood, mutual community resilience.',
 'ta_youth_disability_clothing':'Tel Aviv Youth: young people with disabilities and the challenge of clothing and dressing.',
 'other_unclear':'An open creative design problem.',
}

msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
labels = {int(r['conversation_id']): r['challenge_id'] for r in csv.DictReader(open(f'{BASE}/outputs/challenge_labels.csv', encoding='utf-8'))}

by = defaultdict(list)
for m in msgs:
    by[m['conversation_id']].append(m)
for c in by:
    by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))
groups = {c: by[c][0]['conv_group'] for c in by}

print('Loading local Qwen model (Agent 1/2/4 backend)...', flush=True)
LLMClient.load()
print('Model ready.', flush=True)

# ---------- Agent 1: per-message extraction (cached) ----------
cache_path = f'{BASE}/outputs/_agent1_candidates_cache.json'
try:
    cache = json.load(open(cache_path, encoding='utf-8'))
except FileNotFoundError:
    cache = {}

cands_by_conv = defaultdict(list)
t0 = time.time(); n_calls = 0
for ci, c in enumerate(sorted(by), 1):
    ck = str(c)
    chal_txt = CHAL.get(labels.get(c, 'other_unclear'), CHAL['other_unclear'])
    if groups[c] == 'treatment':
        plabel_for = lambda p: f"{p} ({'divergent' if p == 'Taylor' else 'convergent'})"
    else:
        plabel_for = lambda p: 'standard assistant'
    prev_assistant = None
    uturn = 0
    if ck in cache:
        cands_by_conv[c] = [CandidateRow(**d) for d in cache[ck]]
        # still need to advance prev_assistant? no—cached fully
        print(f'[{ci}/{len(by)}] conv {c} cached ({len(cache[ck])} cands)', flush=True)
        continue
    conv_rows = []
    for m in by[c]:
        if m['message_src'] == 'assistant':
            prev_assistant = m.get('message_en') or m['message']
            continue
        umsg = (m.get('message_en') or m['message'] or '').strip()
        if not umsg:
            uturn += 1; continue
        rows, _ = extract_for_message(c, m['message_id'], uturn, prev_assistant, umsg,
                                      chal_txt, plabel_for(m['persona']))
        conv_rows.extend(rows); n_calls += 1; uturn += 1
    cands_by_conv[c] = conv_rows
    cache[ck] = [vars(r) for r in conv_rows]
    json.dump(cache, open(cache_path, 'w', encoding='utf-8'), ensure_ascii=False)
    el = time.time() - t0
    print(f'[{ci}/{len(by)}] conv {c} [{groups[c][:4]}] -> {len(conv_rows)} candidates '
          f'(calls={n_calls}, {el:.0f}s)', flush=True)

# ---------- Agent 2: consolidate per conversation ----------
print('Consolidating (Agent 2)...', flush=True)
canon_by_conv = {}
for c in sorted(by):
    canon_by_conv[c] = consolidate(cands_by_conv[c], c)

all_canon = [cr for c in sorted(by) for cr in canon_by_conv[c]]
print(f'Total canonical ideas: {len(all_canon)}', flush=True)

# ---------- Agent 4: categorize corpus ----------
print('Categorizing (Agent 4)...', flush=True)
cats, _V, csum = categorize(all_canon, min_cluster_size=4)
cat_of = {(ci.conversation_id, ci.canonical_id): ci.category_id for ci in cats}

# ---------- embed canonical ideas (bge) for Agent 5 ----------
embedder = _get_embedder()
idea_vecs_by_conv = {}
for c in sorted(by):
    crs = canon_by_conv[c]
    if not crs:
        continue
    texts = [f'{cr.title}: {cr.description}' for cr in crs]
    V = np.asarray(embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype='float32')
    idea_vecs_by_conv[c] = V

# ---------- Agent 5: originality ----------
meta = {c: {'user': c, 'condition': groups[c], 'persona_label': groups[c],
            'challenge': labels.get(c, 'other_unclear'), 'n_ideas': len(canon_by_conv[c])}
        for c in idea_vecs_by_conv}
centroids, aligned = compute_centroids(sorted(idea_vecs_by_conv), idea_vecs_by_conv)
orig = {o.conversation_id: o for o in compute_originality(centroids, aligned, meta)}

def within_div(V):
    if V is None or len(V) < 2: return float('nan')
    ds = [1 - float(np.dot(V[i], V[j])) for i in range(len(V)) for j in range(i+1, len(V))]
    return float(np.mean(ds))

# ---------- write per-conversation portfolio ----------
rows = []
for c in sorted(by):
    crs = canon_by_conv[c]
    cat_ids = set(cat_of.get((c, cr.canonical_id), -1) for cr in crs)
    flexibility = len([x for x in cat_ids if x >= 0]) + sum(1 for cr in crs if cat_of.get((c, cr.canonical_id), -1) == -1)  # non-noise clusters + each unclustered idea
    o = orig.get(c)
    rows.append({'conversation_id': c, 'group': groups[c], 'challenge_id': labels.get(c, 'other_unclear'),
                 'fluency': len(crs),
                 'flexibility': len(cat_ids) if crs else 0,
                 'orig_same': round(o.orig_same, 4) if o else '',
                 'orig_all': round(o.orig_all, 4) if o else '',
                 'orig_cross': round(o.orig_cross, 4) if o else '',
                 'within_idea_diversity': round(within_div(idea_vecs_by_conv.get(c)), 4) if c in idea_vecs_by_conv else ''})

with open(f'{BASE}/outputs/idea_portfolio_exp1.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(f'{BASE}/outputs/ideas_canonical_exp1.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f); w.writerow(['conversation_id', 'group', 'canonical_id', 'title', 'description', 'category_id'])
    for c in sorted(by):
        for cr in canon_by_conv[c]:
            w.writerow([c, groups[c], cr.canonical_id, cr.title, cr.description, cat_of.get((c, cr.canonical_id), -1)])

import statistics as _s
for grp in ('treatment', 'control'):
    fl = [r['fluency'] for r in rows if r['group'] == grp]
    print(f'{grp}: n={len(fl)} mean fluency={_s.mean(fl):.2f}', flush=True)
print(f'Total time {(time.time()-t0)/60:.1f} min', flush=True)
print('Wrote: outputs/idea_portfolio_exp1.csv and outputs/ideas_canonical_exp1.csv', flush=True)
