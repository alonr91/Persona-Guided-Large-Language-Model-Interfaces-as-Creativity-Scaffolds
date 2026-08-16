# -*- coding: utf-8 -*-
"""Semantic co-regulation on the LONG sample (treatment-focused).

Extends the (stance-level) complementarity finding into EMBEDDING space.
(A) ACCOMMODATION: cosine(user turn, preceding assistant turn) — do users' CONTENT
    move toward the persona's just-offered content (alignment) or stay distinct
    (complementarity)? Trajectory over position + by persona + vs control.
(B) COMPLEMENTARITY INDEX: is a user turn closer to their OWN prior content than to
    the persona's preceding turn? (own-history sim - persona sim) > 0 => stays distinct.
(C) REFRAMING (2.4.1): user jump = 1 - cos(user_i, user_{i-1}); is the jump larger
    right after engaging Taylor (reframe) than Alex?
Per-turn AND per-conversation summaries (conversation-level avoids non-independence).
"""
import json, csv, math, statistics
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
LONG = {int(r['conversation_id']): r['group'] for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv', encoding='utf-8'))}
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))

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

print('embedding turns (bge)...', flush=True)
bge=SentenceTransformer('BAAI/bge-large-en-v1.5')
emb={}
for c in by:
    V=np.asarray(bge.encode([(m.get('message_en') or m['message']).strip()[:1500] for m in by[c]], normalize_embeddings=True),dtype='float32')
    for m,v in zip(by[c],V): emb[m['message_id']]=v
def cos(a,b): return float(np.dot(a,b))

# accommodation + complementarity index + reframing
acc_by_g=defaultdict(list); acc_by_persona=defaultdict(list)
acc_traj=[]; compl_index=defaultdict(list)
reframe_by_persona=defaultdict(list)
acc_conv=defaultdict(list)  # conv -> [acc] for conv-level mean
for c in by:
    g=LONG[c]; seq=by[c]; prev_a=None; user_hist=[]; prev_user=None; n_users=sum(1 for m in seq if m['message_src']=='user'); ui=0
    for m in seq:
        v=emb[m['message_id']]
        if m['message_src']=='assistant':
            prev_a=(v,m['persona'])
        else:
            if prev_a is not None:
                a=cos(v,prev_a[0]); acc_by_g[g].append(a); acc_by_persona[(g,prev_a[1])].append(a)
                acc_conv[c].append(a)
                pos=ui/(n_users-1) if n_users>1 else 0; acc_traj.append((g,pos,a))
                if user_hist:
                    own=max(cos(v,h) for h in user_hist)
                    compl_index[g].append(own - a)   # >0: closer to own history than to persona
            if prev_user is not None:
                jump=1-cos(v,prev_user)
                # attribute jump to the persona the user is currently addressing
                reframe_by_persona[(g,m['persona'])].append(jump)
            user_hist.append(v); prev_user=v; ui+=1

print('\n'+'='*74); print('(A) SEMANTIC ACCOMMODATION — user turn vs preceding persona turn'); print('='*74)
print('  treatment vs control (mean accommodation):', fmt(welch(acc_by_g['treatment'], acc_by_g['control'])))
print('  treatment Taylor vs Alex:', fmt(welch(acc_by_persona[('treatment','Taylor')], acc_by_persona[('treatment','Alex')])))
for g in ('treatment','control'):
    print(f'  [{g}] accommodation ~ position:', pear([(p,a) for (gg,p,a) in acc_traj if gg==g]))
# conversation-level (independence-safe)
convT=[statistics.mean(acc_conv[c]) for c in acc_conv if LONG[c]=='treatment' and acc_conv[c]]
convC=[statistics.mean(acc_conv[c]) for c in acc_conv if LONG[c]=='control' and acc_conv[c]]
print('  conversation-level mean accommodation T vs C:', fmt(welch(convT,convC)))

print('\n'+'='*74); print('(B) COMPLEMENTARITY INDEX — (own-history sim) - (persona sim); >0 = stays distinct'); print('='*74)
for g in ('treatment','control'):
    xs=compl_index[g]
    print(f'  [{g}] mean = {statistics.mean(xs):+.3f} (n_turns={len(xs)}); positive => user content closer to own history than to persona')

print('\n'+'='*74); print('(C) REFRAMING — user semantic jump from own previous turn, by persona addressed'); print('='*74)
print('  treatment Taylor vs Alex:', fmt(welch(reframe_by_persona[('treatment','Taylor')], reframe_by_persona[('treatment','Alex')])))
print('  (larger jump when addressing Taylor => divergent persona associated with reframing)')
print('\ndone.')
