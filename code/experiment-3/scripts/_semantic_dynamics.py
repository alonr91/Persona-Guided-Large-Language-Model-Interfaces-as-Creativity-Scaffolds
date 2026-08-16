# -*- coding: utf-8 -*-
"""Semantic-trajectory analysis on the LONG sample (treatment-focused).

The literature review frames divergent/convergent as EXPANSION vs CONTRACTION of the
semantic search space (White 2003; 2.3.3), unfolding as exploration->exploitation over
time (2.3.4), with FIXATION/anchoring to early ideas as the failure mode (2.4.2). We
test these as trajectories in embedding space, using per-turn bge-large embeddings.

Measures (per conversation, turns in time order):
  novelty-to-history(turn i) = 1 - max cosine similarity to all PRIOR turns of the
    relevant stream (assistant turns vs all prior; user turns vs prior USER turns).
  drift-from-first(user i)   = 1 - cosine(user_i, user_0).
Analyses:
  (1) Persona content signature: assistant novelty-to-history, Taylor vs Alex (treatment).
  (2) Exploration->exploitation: user novelty-to-history vs normalized position (T vs C).
  (3) Persona-steered breadth: user novelty-to-history by addressed persona (treatment).
  (4) Fixation: drift-from-first slope over position (T vs C); within-treatment vs Taylor share.
"""
import json, csv, math, statistics
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
LONG = {int(r['conversation_id']): r['group'] for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv', encoding='utf-8'))}
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
port = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/long_portfolio.csv', encoding='utf-8'))}

def welch(a,b):
    a=[x for x in a if x is not None]; b=[x for x in b if x is not None]; na,nb=len(a),len(b)
    if na<2 or nb<2: return None
    ma,mb=statistics.mean(a),statistics.mean(b); va,vb=statistics.variance(a),statistics.variance(b)
    se=math.sqrt(va/na+vb/nb)
    if se==0: return None
    t=(ma-mb)/se; df=(va/na+vb/nb)**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)); g=(ma-mb)/sp*(1-3/(4*(na+nb)-9)) if sp else 0
    x=df/(df+t*t);A,B=df/2,.5
    def bcf(x,a,b):
        f=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1;d=1-qab*x/qap;d=1/(d if abs(d)>f else f);h=d
        for k in range(1,200):
            k2=2*k;aa=k*(b-k)*x/((qam+k2)*(a+k2));d=1+aa*d;d=1/(d if abs(d)>f else f);c=1+aa/c;c=c if abs(c)>f else f;h*=d*c
            aa=-(a+k)*(qab+k)*x/((a+k2)*(qap+k2));d=1+aa*d;d=1/(d if abs(d)>f else f);c=1+aa/c;c=c if abs(c)>f else f;de=d*c;h*=de
            if abs(de-1)<3e-12:break
        return h
    lb=math.lgamma(A)+math.lgamma(B)-math.lgamma(A+B)
    p=(math.exp(A*math.log(x)+B*math.log(1-x)-lb)*bcf(x,A,B)/A) if x<(A+1)/(A+B+2) else (1-math.exp(A*math.log(x)+B*math.log(1-x)-lb)*bcf(1-x,B,A)/B)
    return round(ma,3),round(mb,3),round(g,2),round(max(0,min(1,p)),4),na,nb
def pear(pairs):
    pts=[(a,b) for a,b in pairs if a is not None and b is not None]; n=len(pts)
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
    return round(r,3),n,round(max(0,min(1,p)),4)
def fmt(w): return f"T={w[0]} C={w[1]} g={w[2]:+.2f} p={w[3]} (n {w[4]}/{w[5]})" if w else "n/a"

by=defaultdict(list)
for m in msgs:
    if m['conversation_id'] in LONG: by[m['conversation_id']].append(m)
for c in by: by[c].sort(key=lambda x:(x['timestamp'],x['message_id']))

print('embedding all turns (bge)...', flush=True)
bge=SentenceTransformer('BAAI/bge-large-en-v1.5')
emb={}
for c in by:
    txts=[(m.get('message_en') or m['message']).strip()[:1500] for m in by[c]]
    V=np.asarray(bge.encode(txts, normalize_embeddings=True), dtype='float32')
    for m,v in zip(by[c],V): emb[m['message_id']]=v

def nn_novelty(vec, hist):
    if not hist: return None
    return 1 - max(float(np.dot(vec, h)) for h in hist)

# per-turn novelty streams
asst_nov=defaultdict(list)   # (conv, persona) -> [novelty]
user_nov_by_persona=defaultdict(list)
user_traj=[]                 # (group, normpos, novelty)
drift_slopes={}              # conv -> slope of drift-from-first over user index
for c in by:
    g=LONG[c]; seq=by[c]
    hist_all=[]; hist_user=[]; first_user=None; drifts=[]; upos=[]
    users_idx=0; n_users=sum(1 for m in seq if m['message_src']=='user')
    for m in seq:
        v=emb[m['message_id']]
        if m['message_src']=='assistant':
            nv=nn_novelty(v, hist_all)
            if nv is not None: asst_nov[(g,m['persona'])].append(nv)
        else:
            nv=nn_novelty(v, hist_user)
            if nv is not None:
                user_nov_by_persona[(g,m['persona'])].append(nv)
                pos=users_idx/(n_users-1) if n_users>1 else 0
                user_traj.append((g,pos,nv))
            if first_user is None: first_user=v
            else: drifts.append((users_idx, 1-float(np.dot(v,first_user))))
            hist_user.append(v); users_idx+=1
        hist_all.append(v)
    # drift slope (regress drift on user index)
    if len(drifts)>=4:
        xs=[d[0] for d in drifts]; ys=[d[1] for d in drifts]
        mx=statistics.mean(xs);my=statistics.mean(ys);sx=statistics.pstdev(xs)
        if sx>0: drift_slopes[c]=sum((x-mx)*(y-my) for x,y in drifts)/(len(drifts)*sx*sx)

print('\n'+'='*74)
print('(1) PERSONA CONTENT SIGNATURE — assistant turn novelty-to-history')
print('='*74)
print('  treatment Taylor vs Alex:', fmt(welch(asst_nov[('treatment','Taylor')], asst_nov[('treatment','Alex')])))
print('  control  Taylor vs Alex:', fmt(welch(asst_nov[('control','Taylor')], asst_nov[('control','Alex')])))
print('  (expansion: Taylor injects content farther from history; contraction: Alex stays near)')

print('\n'+'='*74)
print('(2) EXPLORATION->EXPLOITATION — user novelty-to-history vs position')
print('='*74)
for g in ('treatment','control'):
    pairs=[(p,n) for (gg,p,n) in user_traj if gg==g]
    print(f'  [{g}] novelty ~ position:', pear(pairs))
# quartile means
for g in ('treatment','control'):
    q=[[],[],[],[]]
    for gg,p,n in user_traj:
        if gg==g: q[min(3,int(p*4))].append(n)
    print(f'  [{g}] novelty quartile means:', [round(statistics.mean(x),3) if x else None for x in q])

print('\n'+'='*74)
print('(3) PERSONA-STEERED BREADTH — user novelty-to-history by addressed persona (treatment)')
print('='*74)
print('  Taylor vs Alex:', fmt(welch(user_nov_by_persona[('treatment','Taylor')], user_nov_by_persona[('treatment','Alex')])))

print('\n'+'='*74)
print('(4) FIXATION — drift from first idea')
print('='*74)
for g in ('treatment','control'):
    sl=[drift_slopes[c] for c in drift_slopes if LONG[c]==g]
    print(f'  [{g}] mean drift slope (per user turn): {statistics.mean(sl):+.4f} (n={len(sl)} convs); >0 = escaping the anchor')
# within treatment: drift slope vs Taylor share
tr=[c for c in drift_slopes if LONG[c]=='treatment']
def tshare(c):
    t=float(port[c]['msgs_to_Taylor']); a=float(port[c]['msgs_to_Alex']); return t/(t+a) if (t+a)>0 else None
print('  [treatment] drift slope ~ Taylor share:', pear([(tshare(c), drift_slopes[c]) for c in tr]))
print('\ndone.')
