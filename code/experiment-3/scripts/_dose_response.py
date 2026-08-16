# -*- coding: utf-8 -*-
"""Within-TREATMENT dose-response: does the way users split attention between the
convergent (Alex) and divergent (Taylor) personas relate to their creativity?

Control group is excluded (its two buttons are the same model, so persona split is
meaningless there). Predictors are interaction-style metrics; outcomes are
topic-controlled SEMANTIC creativity measures (embeddings). Quantitative measures
(fluency/flexibility) are merged in later from the Gemini extraction.

Simple statistics: Pearson r with two-sided p (t-test on r), no correction.
"""
import json, csv, math, statistics
from collections import defaultdict

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
summ = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/experiment3_conversations_summary.csv', encoding='utf-8'))}
orig = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/originality_topic_controlled.csv', encoding='utf-8'))}
sw   = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/switching_per_conv.csv', encoding='utf-8'))}
# Experiment-1 idea-portfolio measures (fluency, flexibility, idea-centroid originality)
import os as _os
PORT = {}
_pp = f'{BASE}/outputs/idea_portfolio_exp1.csv'
if _os.path.exists(_pp):
    PORT = {int(r['conversation_id']): r for r in csv.DictReader(open(_pp, encoding='utf-8'))}

by = defaultdict(list)
for m in msgs:
    by[m['conversation_id']].append(m)
for c in by:
    by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))

def pearson(xs, ys):
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)
           if x not in ('', 'nan', None) and y not in ('', 'nan', None)
           and not (isinstance(x, float) and math.isnan(x)) and not (isinstance(y, float) and math.isnan(y))]
    n = len(pts)
    if n < 4: return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx == 0 or sy == 0: return None
    r = sum((x-mx)*(y-my) for x, y in pts) / (n*sx*sy)
    r = max(-0.999999, min(0.999999, r))
    t = r*math.sqrt((n-2)/(1-r*r)); df = n-2
    # two-sided p via incomplete beta
    x = df/(df+t*t); a, b = df/2, 0.5
    def betacf(x,a,b):
        FPMIN=1e-300; qab=a+b; qap=a+1; qam=a-1; c=1; d=1-qab*x/qap
        d=1/(d if abs(d)>FPMIN else FPMIN); h=d
        for m in range(1,200):
            m2=2*m; aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d; d=1/(d if abs(d)>FPMIN else FPMIN)
            c=1+aa/c; c=c if abs(c)>FPMIN else FPMIN; h*=d*c
            aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d; d=1/(d if abs(d)>FPMIN else FPMIN)
            c=1+aa/c; c=c if abs(c)>FPMIN else FPMIN; de=d*c; h*=de
            if abs(de-1)<3e-12: break
        return h
    lb=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    ib=(math.exp(a*math.log(x)+b*math.log(1-x)-lb)*betacf(x,a,b)/a) if x<(a+1)/(a+b+2) else \
       (1-math.exp(a*math.log(x)+b*math.log(1-x)-lb)*betacf(1-x,b,a)/b)
    return dict(r=r, n=n, p=max(0.0, min(1.0, ib)))

# ---- build per-treatment-conversation predictors ----
rows = []
for c, ms in by.items():
    if summ[c]['group'] != 'treatment' or summ[c]['quality_tier'] != 'full':
        continue
    users = [m for m in ms if m['message_src'] == 'user']
    n_tay = sum(1 for m in users if m['persona'] == 'Taylor')
    n_alex = sum(1 for m in users if m['persona'] == 'Alex')
    tot = n_tay + n_alex
    alex_share = n_alex / tot if tot else 0
    # timing: mean normalized position (0..1) of Alex-addressed user turns; high => convergent used late
    pos = [i/(len(users)-1) for i, m in enumerate(users) if m['persona'] == 'Alex'] if len(users) > 1 else []
    alex_timing = statistics.mean(pos) if pos else float('nan')
    used_both = 1 if (n_tay > 0 and n_alex > 0) else 0
    o = orig.get(c, {})
    p = PORT.get(c, {})
    rows.append({
        'conversation_id': c, 'n_user': len(users), 'n_taylor': n_tay, 'n_alex': n_alex,
        'alex_share': round(alex_share, 3), 'alex_timing': round(alex_timing, 3) if pos else '',
        'switches': sw.get(c, {}).get('switches', ''), 'used_both': used_both,
        # Experiment-1 idea-portfolio creativity measures
        'fluency': p.get('fluency', ''),
        'flexibility': p.get('flexibility', ''),
        'orig_same': p.get('orig_same', ''),
        'orig_all': p.get('orig_all', ''),
        'within_idea_diversity': p.get('within_idea_diversity', ''),
        # topic-controlled embedding cross-check
        'dist_to_challenge_centroid': o.get('dist_to_challenge_centroid', ''),
        'resid_same_cond_originality': o.get('resid_same_cond_originality', ''),
        'within_conv_diversity': o.get('within_conv_diversity', ''),
    })

with open(f'{BASE}/outputs/dose_response_treatment.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

PRED = ['alex_share', 'n_alex', 'n_taylor', 'alex_timing', 'switches', 'n_user']
OUTC = [('fluency', 'fluency = # distinct ideas (Exp1, QUANTITATIVE)'),
        ('flexibility', 'flexibility = # categories (Exp1, QUANTITATIVE)'),
        ('orig_same', 'idea-portfolio originality, same-condition (Exp1, SEMANTIC)'),
        ('orig_all', 'idea-portfolio originality, all (Exp1, SEMANTIC)'),
        ('within_idea_diversity', 'within-portfolio idea diversity (Exp1, SEMANTIC)'),
        ('dist_to_challenge_centroid', 'originality vs challenge norm (embedding cross-check)'),
        ('resid_same_cond_originality', 'between-user distinctiveness (embedding cross-check)')]
print(f'Within-treatment dose-response (n={len(rows)} full-tier treatment conversations)')
print('Pearson r (two-sided p, no correction). Positive r with a convergent predictor = more convergent engagement -> higher creativity.\n')
for okey, oname in OUTC:
    print(f'OUTCOME: {oname}')
    for pk in PRED:
        res = pearson([r[pk] for r in rows], [r[okey] for r in rows])
        if res:
            star = '*' if res['p'] < .05 else ('.' if res['p'] < .10 else ' ')
            print(f"   {pk:14} r={res['r']:+.3f}  p={res['p']:.3f}  (n={res['n']}) {star}")
    print()
print('Wrote: outputs/dose_response_treatment.csv')
