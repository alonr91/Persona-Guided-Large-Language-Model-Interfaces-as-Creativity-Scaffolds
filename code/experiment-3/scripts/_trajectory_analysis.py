"""Stance trajectory analysis — replicates paper's Sec 5.1 quartile-based finding.

Paper claim: users engage divergent persona more in early quarters, convergent
later (the divergent-early, convergent-late macro pattern).

Method (using stance_per_message scores already computed):
  - Within each conversation, sort turns by timestamp and split into Q1..Q4.
  - For each quartile compute:
      * mean divergent_score  (assistant turns)
      * mean convergent_score (assistant turns)
      * D-C balance           (assistant turns)
      * proportion of user turns sent to Taylor (treatment only)
      * mean user-side D-C balance (test of co-regulation)
  - Plot trajectories per group.
"""
import json, csv, math, statistics
from collections import defaultdict
import numpy as np

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'

with open(f'{base}/outputs/stance_per_message.json','r',encoding='utf-8') as f:
    stance = json.load(f)

# Group by conv
by_conv = defaultdict(list)
for r in stance:
    by_conv[r['conversation_id']].append(r)
for cid in by_conv:
    by_conv[cid].sort(key=lambda x: x['timestamp'])

def quartiles(items):
    n = len(items)
    if n < 4: return None
    q = [n*i//4 for i in range(5)]
    return [items[q[i]:q[i+1]] for i in range(4)]

# Per-conv quartile aggregates
rows = []
for cid, items in by_conv.items():
    qs = quartiles(items)
    if qs is None: continue
    grp = items[0]['group']
    for qi, q in enumerate(qs, 1):
        a = [r for r in q if r['message_src']=='assistant']
        u = [r for r in q if r['message_src']=='user']
        if not a and not u: continue
        # how many user msgs to Taylor (1 or 3) vs Alex (2 or 4)?
        n_taylor = sum(1 for r in u if str(r['persona_id']) in ('1','3'))
        n_alex   = sum(1 for r in u if str(r['persona_id']) in ('2','4'))
        rows.append({
            'conv_id': cid,
            'group': grp,
            'quartile': qi,
            'n_assistant_turns': len(a),
            'n_user_turns': len(u),
            'mean_assistant_div':  round(statistics.mean([r['divergent_score']  for r in a]),4) if a else '',
            'mean_assistant_con':  round(statistics.mean([r['convergent_score'] for r in a]),4) if a else '',
            'mean_assistant_dmc':  round(statistics.mean([r['d_minus_c']        for r in a]),4) if a else '',
            'mean_user_div':       round(statistics.mean([r['divergent_score']  for r in u]),4) if u else '',
            'mean_user_con':       round(statistics.mean([r['convergent_score'] for r in u]),4) if u else '',
            'mean_user_dmc':       round(statistics.mean([r['d_minus_c']        for r in u]),4) if u else '',
            'pct_user_to_taylor':  round(n_taylor/(n_taylor+n_alex)*100,2) if (n_taylor+n_alex)>0 else '',
        })

with open(f'{base}/outputs/trajectory_per_conv_quartile.csv','w',encoding='utf-8',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# Group-level summary by quartile
print('='*78)
print('STANCE TRAJECTORY BY GROUP × QUARTILE')
print('='*78)
for grp in ['treatment','control']:
    print(f'\n--- {grp} ---')
    print(f'{"Q":<3}{"n_convs":<10}{"asst_D":<10}{"asst_C":<10}{"asst_D-C":<12}{"user_D":<10}{"user_C":<10}{"user_D-C":<12}{"%user→Taylor":<14}')
    for qi in range(1,5):
        cell = [r for r in rows if r['group']==grp and r['quartile']==qi]
        if not cell: continue
        def m(k):
            vals = [r[k] for r in cell if isinstance(r[k],(int,float))]
            return f'{statistics.mean(vals):.3f}' if vals else 'n/a'
        print(f'{qi:<3}{len(cell):<10}{m("mean_assistant_div"):<10}{m("mean_assistant_con"):<10}{m("mean_assistant_dmc"):<12}'
              f'{m("mean_user_div"):<10}{m("mean_user_con"):<10}{m("mean_user_dmc"):<12}{m("pct_user_to_taylor"):<14}')

print(f'\nWrote: {base}/outputs/trajectory_per_conv_quartile.csv')
