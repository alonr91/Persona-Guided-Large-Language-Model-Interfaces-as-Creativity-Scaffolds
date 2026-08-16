# -*- coding: utf-8 -*-
"""Quarter-resolved persona-choice (which persona the USER addresses), long sample.

_analyze_long.py already reports a first-half vs second-half divergent-share split
and a quartile trend for the ASSISTANT stance score (d_minus_c). This script adds
the missing piece: the divergent SHARE OF USER MESSAGES (Taylor vs Alex), split into
four quarters instead of two halves, to more finely justify the macro
divergence-to-convergence trajectory claim in Sec 3.2.
"""
import json, csv, math, statistics
from collections import defaultdict
from scipy import stats as sstats

BASE = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
LONG = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv', encoding='utf-8'))}
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))

by = defaultdict(list)
for m in msgs:
    if m['conversation_id'] in LONG:
        by[m['conversation_id']].append(m)
for c in by:
    by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))

def welch(a, b):
    a = [float(x) for x in a if x is not None]; b = [float(x) for x in b if x is not None]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return None
    ma, mb = statistics.mean(a), statistics.mean(b); va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va/na + vb/nb)
    if se == 0: return None
    t = (ma-mb)/se; df = (va/na+vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    sp = math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)); g = (ma-mb)/sp*(1-3/(4*(na+nb)-9)) if sp else 0
    p = 2*(1-sstats.t.cdf(abs(t), df))
    return dict(ma=round(ma,3), mb=round(mb,3), na=na, nb=nb, t=round(t,2), df=round(df,1), g=round(g,2), p=round(p,3))

rows = []
for c, ms in by.items():
    grp = ms[0]['conv_group']
    users = [m for m in ms if m['message_src'] == 'user']
    seq = [m['persona'] for m in users]
    n = len(seq)
    if n < 4:
        continue
    q = [seq[i*n//4:(i+1)*n//4] for i in range(4)]
    shares = [ (seg.count('Taylor')/len(seg)) if seg else None for seg in q]
    rows.append({'conversation_id': c, 'group': grp, 'n_user': n,
                 'q1': shares[0], 'q2': shares[1], 'q3': shares[2], 'q4': shares[3]})

with open(f'{BASE}/outputs/choreography_quartiles.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

T = [r for r in rows if r['group'] == 'treatment']
C = [r for r in rows if r['group'] == 'control']
print(f'n = {len(rows)} ({len(T)} treatment, {len(C)} control)\n')
print('Mean divergent (Taylor) share of user messages, by quarter:')
for qi, key in enumerate(['q1','q2','q3','q4'], start=1):
    tvals = [r[key] for r in T if r[key] is not None]
    cvals = [r[key] for r in C if r[key] is not None]
    w = welch(tvals, cvals)
    tm = round(statistics.mean(tvals), 3) if tvals else None
    cm = round(statistics.mean(cvals), 3) if cvals else None
    print(f'  Q{qi}: treatment={tm} (n={len(tvals)})  control={cm} (n={len(cvals)})  ->  {w}')

print('\nTreatment monotonic trend Q1->Q4:', [round(statistics.mean([r[k] for r in T if r[k] is not None]),3) for k in ['q1','q2','q3','q4']])
print('Control   monotonic trend Q1->Q4:', [round(statistics.mean([r[k] for r in C if r[k] is not None]),3) for k in ['q1','q2','q3','q4']])
print('\nWrote outputs/choreography_quartiles.csv')
