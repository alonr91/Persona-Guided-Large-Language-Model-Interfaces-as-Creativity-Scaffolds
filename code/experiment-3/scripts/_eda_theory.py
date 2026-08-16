# -*- coding: utf-8 -*-
"""Theory-driven EDA on Exp-3 per-turn stance (full-tier).

Two lit-review-grounded analyses the long, natural conversations make uniquely testable:

(A) PERSONA FIDELITY / DRIFT (Lit review 2.8): does each persona hold its stance
    contract as context accumulates over long conversations, or drift toward the
    centre / counter-persona? Tests whether the divergent-vs-convergent separation
    PERSISTS late in conversations.

(B) MESO CO-REGULATION: ALIGNMENT vs COMPLEMENTARITY (2.3.3; Fusaroli & Tylen 2016):
    do users adopt the stance of the persona they address (alignment), or stay
    generative and delegate evaluation to the convergent persona (complementarity)?
"""
import json, math, statistics
from collections import defaultdict

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
st = json.load(open(f'{BASE}/outputs/stance_per_message_full.json', encoding='utf-8'))

def welch(a, b):
    a=[x for x in a]; b=[x for x in b]; na,nb=len(a),len(b)
    if na<2 or nb<2: return None
    ma,mb=statistics.mean(a),statistics.mean(b); va,vb=statistics.variance(a),statistics.variance(b)
    se=math.sqrt(va/na+vb/nb)
    if se==0: return None
    t=(ma-mb)/se; df=(va/na+vb/nb)**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)); g=(ma-mb)/sp*(1-3/(4*(na+nb)-9)) if sp else 0
    # two-sided p
    x=df/(df+t*t); A,B=df/2,.5
    def bcf(x,a,b):
        f=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1;d=1-qab*x/qap;d=1/(d if abs(d)>f else f);h=d
        for k in range(1,200):
            k2=2*k;aa=k*(b-k)*x/((qam+k2)*(a+k2));d=1+aa*d;d=1/(d if abs(d)>f else f);c=1+aa/c;c=c if abs(c)>f else f;h*=d*c
            aa=-(a+k)*(qab+k)*x/((a+k2)*(qap+k2));d=1+aa*d;d=1/(d if abs(d)>f else f);c=1+aa/c;c=c if abs(c)>f else f;de=d*c;h*=de
            if abs(de-1)<3e-12:break
        return h
    lb=math.lgamma(A)+math.lgamma(B)-math.lgamma(A+B)
    p=(math.exp(A*math.log(x)+B*math.log(1-x)-lb)*bcf(x,A,B)/A) if x<(A+1)/(A+B+2) else (1-math.exp(A*math.log(x)+B*math.log(1-x)-lb)*bcf(1-x,B,A)/B)
    return round(ma,3),round(mb,3),round(t,2),round(g,2),round(max(0,min(1,p)),4),na,nb

def pearson(pairs):
    pts=[(a,b) for a,b in pairs]; n=len(pts)
    if n<4: return None
    X=[p[0] for p in pts];Y=[p[1] for p in pts];mx,my=statistics.mean(X),statistics.mean(Y)
    sx,sy=statistics.pstdev(X),statistics.pstdev(Y)
    if sx==0 or sy==0: return None
    r=max(-.999,min(.999,sum((a-mx)*(b-my) for a,b in pts)/(n*sx*sy)))
    t=r*math.sqrt((n-2)/(1-r*r));df=n-2;x=df/(df+t*t)
    def bcf(x,a,b):
        f=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1;d=1-qab*x/qap;d=1/(d if abs(d)>f else f);h=d
        for k in range(1,200):
            k2=2*k;aa=k*(b-k)*x/((qam+k2)*(a+k2));d=1+aa*d;d=1/(d if abs(d)>f else f);c=1+aa/c;c=c if abs(c)>f else f;h*=d*c
            aa=-(a+k)*(qab+k)*x/((a+k2)*(qap+k2));d=1+aa*d;d=1/(d if abs(d)>f else f);c=1+aa/c;c=c if abs(c)>f else f;de=d*c;h*=de
            if abs(de-1)<3e-12:break
        return h
    A,B=df/2,.5;lb=math.lgamma(A)+math.lgamma(B)-math.lgamma(A+B)
    p=(math.exp(A*math.log(x)+B*math.log(1-x)-lb)*bcf(x,A,B)/A) if x<(A+1)/(A+B+2) else (1-math.exp(A*math.log(x)+B*math.log(1-x)-lb)*bcf(1-x,B,A)/B)
    return round(r,3), n, round(max(0,min(1,p)),4)

by = defaultdict(list)
for r in st:
    by[r['conversation_id']].append(r)
for c in by:
    by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))
grp = {c: by[c][0]['group'] for c in by}

# ============ (A) PERSONA FIDELITY / DRIFT ============
print('='*74)
print('(A) PERSONA FIDELITY / DRIFT over long conversations (2.8)')
print('='*74)
# within-persona normalized position vs stance, pooled across treatment convs
for persona, metric, lbl in [('Taylor','d_minus_c','Taylor D-C'), ('Taylor','divergent_score','Taylor divergent'),
                              ('Alex','d_minus_c','Alex D-C'), ('Alex','convergent_score','Alex convergent')]:
    pairs=[]
    for c in by:
        if grp[c] != 'treatment': continue
        turns=[r for r in by[c] if r['message_src']=='assistant' and r['agent_role']==persona]
        if len(turns) < 3: continue
        for i,r in enumerate(turns):
            pairs.append((i/(len(turns)-1), r[metric]))
    res=pearson(pairs)
    print(f'  {lbl:18} vs within-persona position: r={res[0]:+.3f} p={res[2]:.4f} (n_turns={res[1]})' if res else f'  {lbl}: n/a')

# manipulation persistence: Taylor vs Alex divergent gap, early vs late half of conversation
def half_turns(half):
    T,A=[],[]
    for c in by:
        if grp[c]!='treatment': continue
        ast=[r for r in by[c] if r['message_src']=='assistant']
        n=len(ast)
        seg = ast[:n//2] if half=='early' else ast[n//2:]
        T += [r['divergent_score'] for r in seg if r['agent_role']=='Taylor']
        A += [r['divergent_score'] for r in seg if r['agent_role']=='Alex']
    return T,A
for half in ('early','late'):
    T,A=half_turns(half)
    w=welch(T,A)
    if w: print(f'  [{half} half] Taylor vs Alex divergent: {w[0]} vs {w[1]}  g={w[3]} p={w[4]} (n {w[5]}/{w[6]})')

# ============ (B) CO-REGULATION: ALIGNMENT vs COMPLEMENTARITY ============
print('\n' + '='*74)
print('(B) CO-REGULATION: alignment vs complementarity (2.3.3)')
print('='*74)
# user stance when addressing Taylor vs Alex (treatment)
for g in ('treatment','control'):
    uT=[r['d_minus_c'] for r in st if r['message_src']=='user' and r['group']==g and r['agent_role']=='Taylor']
    uA=[r['d_minus_c'] for r in st if r['message_src']=='user' and r['group']==g and r['agent_role']=='Alex']
    w=welch(uT,uA)
    if w: print(f'  [{g}] USER D-C addressing Taylor vs Alex: {w[0]} vs {w[1]}  g={w[3]} p={w[4]} (n {w[5]}/{w[6]})')
print('  (alignment => users more convergent (lower D-C) to Alex; complementarity => no difference)')

# turn-level entrainment: user turn D-C vs immediately preceding assistant turn D-C
print('\n  Turn-level entrainment (user D-C vs preceding assistant D-C):')
for g in ('treatment','control'):
    pairs=[]
    for c in by:
        if grp[c]!=g: continue
        seq=by[c]; prev_a=None
        for r in seq:
            if r['message_src']=='assistant': prev_a=r['d_minus_c']
            elif r['message_src']=='user' and prev_a is not None:
                pairs.append((prev_a, r['d_minus_c']))
    res=pearson(pairs)
    if res: print(f'    [{g}] r={res[0]:+.3f} p={res[2]:.4f} (n_pairs={res[1]})')
