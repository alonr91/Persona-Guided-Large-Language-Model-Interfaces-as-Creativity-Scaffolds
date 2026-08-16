# -*- coding: utf-8 -*-
"""Exploration: persona-switching & orchestration dynamics (full-tier).

The field dataset's advantage over Experiment 2's 20-min lab sessions is LONG,
NATURAL conversations. This script quantifies how users orchestrate the two
personas across a conversation:
  - switch count / switch rate between addressed personas (Taylor<->Alex)
  - share of messages to the divergent persona, first half vs second half
  - cross-persona brokerage moves (user names/queries the OTHER persona)
Treatment vs control (in control the two buttons are the same model, so any
structured switching there is a baseline/null).
"""
import json, csv, re, math, statistics
from collections import defaultdict

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
msgs = json.load(open(f'{base}/data/experiment3_full_en.json', encoding='utf-8'))
by = defaultdict(list)
for m in msgs:
    by[m['conversation_id']].append(m)
for c in by:
    by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))

def welch(a, b):
    a=[float(x) for x in a]; b=[float(x) for x in b]
    na,nb=len(a),len(b)
    if na<2 or nb<2: return None
    ma,mb=statistics.mean(a),statistics.mean(b); va,vb=statistics.variance(a),statistics.variance(b)
    se=math.sqrt(va/na+vb/nb)
    if se==0: return None
    t=(ma-mb)/se; df=(va/na+vb/nb)**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)); d=(ma-mb)/sp if sp else 0
    return dict(ma=ma,mb=mb,na=na,nb=nb,t=t,df=df,g=d*(1-3/(4*(na+nb)-9)))

BROKER = re.compile(r'\b(taylor|alex)\b', re.I)

rows = []
for c, ms in by.items():
    grp = ms[0]['conv_group']
    users = [m for m in ms if m['message_src'] == 'user']
    seq = [m['persona'] for m in users]          # addressed persona per user turn
    n = len(seq)
    switches = sum(1 for i in range(1, n) if seq[i] != seq[i-1])
    half = n // 2
    div_first = seq[:half].count('Taylor') / max(1, len(seq[:half]))
    div_second = seq[half:].count('Taylor') / max(1, len(seq[half:]))
    # brokerage: user message naming a persona (treats system as a team)
    broker = sum(1 for m in users if BROKER.search(m.get('message_en') or ''))
    rows.append({'conversation_id': c, 'group': grp, 'n_user': n,
                 'switches': switches, 'switch_rate': round(switches/max(1, n-1), 3),
                 'div_share_first_half': round(div_first, 3),
                 'div_share_second_half': round(div_second, 3),
                 'broker_moves': broker})

with open(f'{base}/outputs/switching_per_conv.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

T = lambda k: [r[k] for r in rows if r['group'] == 'treatment']
C = lambda k: [r[k] for r in rows if r['group'] == 'control']

def show(name, k):
    w = welch(T(k), C(k))
    if w: print(f"  {name:34} T={w['ma']:.3f}(n{w['na']}) C={w['mb']:.3f}(n{w['nb']}) t={w['t']:.2f} g={w['g']:.2f}")

print('=== Persona-switching & orchestration (treatment vs control, full-tier) ===')
show('switches per conversation', 'switches')
show('switch rate (per adjacent pair)', 'switch_rate')
show('divergent share, first half', 'div_share_first_half')
show('divergent share, second half', 'div_share_second_half')
show('brokerage moves (names a persona)', 'broker_moves')

# within-treatment: first vs second half divergent share (paired-ish)
tr = [r for r in rows if r['group'] == 'treatment']
d1 = statistics.mean(r['div_share_first_half'] for r in tr)
d2 = statistics.mean(r['div_share_second_half'] for r in tr)
print(f'\nTreatment divergent-share shift: first half {d1:.3f} -> second half {d2:.3f} (drop = move toward convergent)')
co = [r for r in rows if r['group'] == 'control']
print(f'Control  divergent-share shift: first half {statistics.mean(r["div_share_first_half"] for r in co):.3f} '
      f'-> second half {statistics.mean(r["div_share_second_half"] for r in co):.3f}')
# how many treatment convs used BOTH personas
both = sum(1 for r in tr if 0 < (next(rr["switches"] for rr in tr if rr is r) ) or r['switches']>0)
used_both = sum(1 for r in tr if r['switches']>0 or (r['div_share_first_half'] not in (0.0,1.0)))
print(f'\nTreatment convs with >=1 switch: {sum(1 for r in tr if r["switches"]>0)}/{len(tr)}')
print(f'Control   convs with >=1 switch: {sum(1 for r in co if r["switches"]>0)}/{len(co)}')
print('Wrote: outputs/switching_per_conv.csv')
