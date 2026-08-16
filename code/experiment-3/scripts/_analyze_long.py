# -*- coding: utf-8 -*-
"""Master analysis on the LONG-conversation sample (>=10 user turns; multi-day kept).

Recomputes ALL headline measures on the long sample, with peer-relative originality
RE-COMPUTED against the long-sample peer set (not filtered), and length normalization
throughout. Writes outputs/long_portfolio.csv and prints every contrast for the report.
"""
import json, csv, math, statistics
from collections import defaultdict
import numpy as np

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
LONG = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv', encoding='utf-8'))}
summ = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/experiment3_conversations_summary.csv', encoding='utf-8'))}
sw   = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/switching_per_conv.csv', encoding='utf-8'))}
port = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/idea_portfolio_exp1.csv', encoding='utf-8'))}
lab  = {int(r['conversation_id']): r['challenge_id'] for r in csv.DictReader(open(f'{BASE}/outputs/challenge_labels.csv', encoding='utf-8'))}
st   = json.load(open(f'{BASE}/outputs/stance_per_message_full.json', encoding='utf-8'))
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
canon = list(csv.DictReader(open(f'{BASE}/outputs/ideas_canonical_exp1.csv', encoding='utf-8')))

ids = sorted(LONG); grp = {c: LONG[c]['group'] for c in ids}
T = [c for c in ids if grp[c]=='treatment']; C = [c for c in ids if grp[c]=='control']

def fnum(v):
    try: x=float(v); return None if (math.isnan(x) or math.isinf(x)) else x
    except: return None
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
def pearson(pairs):
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
def l2(x): return x/ (np.linalg.norm(x)+1e-12)

# ===== RE-EMBED originality on the LONG peer set =====
from sentence_transformers import SentenceTransformer
# (A) idea-centroid originality (bge), the Experiment-1 metric
print('embedding canonical ideas (bge)...', flush=True)
bge = SentenceTransformer('BAAI/bge-large-en-v1.5')
ideas_by = defaultdict(list)
for r in canon:
    cid=int(r['conversation_id'])
    if cid in LONG: ideas_by[cid].append(f"{r['title']}: {r['description']}")
idea_vecs={c: np.asarray(bge.encode(ideas_by[c], normalize_embeddings=True), dtype='float32') for c in ideas_by if ideas_by[c]}
cent={c: l2(idea_vecs[c].mean(axis=0)) for c in idea_vecs}
def same_cond_orig(c, table):
    peers=[x for x in table if x!=c and grp[x]==grp[c] and x in cent]
    if not peers: return None
    return float(np.mean([1-float(np.dot(cent[c],cent[x])) for x in peers]))
orig_idea={c: same_cond_orig(c, ids) for c in cent}
within_div={}
for c in idea_vecs:
    V=idea_vecs[c]
    if len(V)>=2:
        within_div[c]=float(np.mean([1-float(np.dot(V[i],V[j])) for i in range(len(V)) for j in range(i+1,len(V))]))

# (B) topic-residualized originality (e5) on long subset
print('embedding conversation text (e5)...', flush=True)
e5 = SentenceTransformer('intfloat/multilingual-e5-large')
by_msg=defaultdict(list)
for m in msgs: by_msg[m['conversation_id']].append(m)
conv_vec={}
for c in ids:
    ut=[('passage: '+(m.get('message_en') or m['message']).strip()) for m in by_msg[c] if m['message_src']=='user' and (m.get('message_en') or m['message']).strip()]
    if ut: conv_vec[c]=l2(np.asarray(e5.encode(ut, normalize_embeddings=True),dtype='float32').mean(axis=0))
chal_mem=defaultdict(list)
for c in conv_vec: chal_mem[lab.get(c,'other_unclear')].append(c)
chal_cent={k: np.vstack([conv_vec[x] for x in v]).mean(axis=0) for k,v in chal_mem.items()}
resid={c: l2(conv_vec[c]-chal_cent[lab.get(c,'other_unclear')]) for c in conv_vec}
def resid_orig(c):
    peers=[x for x in resid if x!=c and grp[x]==grp[c]]
    if not peers: return None
    return float(np.mean([1-float(np.dot(resid[c],resid[x])) for x in peers]))
orig_resid={c: resid_orig(c) for c in resid}

# ===== per-conv long table =====
rows=[]
for c in ids:
    nu=fnum(summ[c]['n_user']); fl=fnum(port.get(c,{}).get('fluency'))
    rows.append({'conversation_id':c,'group':grp[c],'challenge_id':lab.get(c,'other_unclear'),
        'n_user':nu,'fluency':fl,'flexibility':fnum(port.get(c,{}).get('flexibility')),
        'ideas_per_user_turn': (fl/nu) if (fl is not None and nu) else None,
        'msgs_to_Taylor':fnum(summ[c]['msgs_to_Taylor']),'msgs_to_Alex':fnum(summ[c]['msgs_to_Alex']),
        'alex_share': (fnum(summ[c]['msgs_to_Alex'])/(fnum(summ[c]['msgs_to_Taylor'])+fnum(summ[c]['msgs_to_Alex']))) if (fnum(summ[c]['msgs_to_Taylor'])+fnum(summ[c]['msgs_to_Alex']))>0 else None,
        'div_share_first_half':fnum(sw[c]['div_share_first_half']),'div_share_second_half':fnum(sw[c]['div_share_second_half']),
        'switches':fnum(sw[c]['switches']),
        'orig_idea_same':orig_idea.get(c),'orig_resid_same':orig_resid.get(c),'within_idea_diversity':within_div.get(c)})
with open(f'{BASE}/outputs/long_portfolio.csv','w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
R={r['conversation_id']:r for r in rows}

print('\n'+'='*78); print(f'LONG SAMPLE ANALYSIS  (n={len(ids)}: {len(T)} treatment, {len(C)} control)'); print('='*78)
# stance helpers
def acell(g,per): return [r['divergent_score'] for r in st if r['message_src']=='assistant' and r['group']==g and r['agent_role']==per and r['conversation_id'] in LONG]
print('\n[Manipulation] Taylor vs Alex divergent (treatment):', fmt(welch(acell('treatment','Taylor'),acell('treatment','Alex'))))
print('[Manipulation] Taylor vs Alex divergent (control):  ', fmt(welch(acell('control','Taylor'),acell('control','Alex'))))
print('\n[Preference] msgs Taylor vs Alex (treatment):', fmt(welch([R[c]['msgs_to_Taylor'] for c in T],[R[c]['msgs_to_Alex'] for c in T])))
print('[Preference] msgs Taylor vs Alex (control):  ', fmt(welch([R[c]['msgs_to_Taylor'] for c in C],[R[c]['msgs_to_Alex'] for c in C])))
print('\n[Choreography] divergent share FIRST half (T vs C):', fmt(welch([R[c]['div_share_first_half'] for c in T],[R[c]['div_share_first_half'] for c in C])))
# trajectory quartiles (assistant D-C)
bya=defaultdict(list)
for r in st:
    if r['message_src']=='assistant' and r['conversation_id'] in LONG: bya[r['conversation_id']].append(r)
for c in bya: bya[c].sort(key=lambda x:(x['timestamp'],x['message_id']))
def quart(group):
    Q=[[],[],[],[]]
    for c in bya:
        if grp[c]!=group: continue
        ms=bya[c]; n=len(ms)
        if n<4: continue
        for qi in range(4):
            seg=ms[qi*n//4:(qi+1)*n//4]
            if seg: Q[qi].append(statistics.mean(x['d_minus_c'] for x in seg))
    return [round(statistics.mean(q),3) if q else None for q in Q]
print('  trajectory assistant D-C quartiles  treatment:',quart('treatment'),' control:',quart('control'))
print('\n[Fidelity] Taylor vs Alex divergent — early vs late half (treatment):')
for h in ('early','late'):
    Tt,Aa=[],[]
    for c in T:
        ast=bya[c]; n=len(ast); seg=ast[:n//2] if h=='early' else ast[n//2:]
        Tt+=[r['divergent_score'] for r in seg if r['agent_role']=='Taylor']; Aa+=[r['divergent_score'] for r in seg if r['agent_role']=='Alex']
    print(f'   [{h}]:',fmt(welch(Tt,Aa)))
print('\n[Co-regulation] USER D-C addressing Taylor vs Alex (treatment):',
      fmt(welch([r['d_minus_c'] for r in st if r['message_src']=='user' and r['group']=='treatment' and r['agent_role']=='Taylor' and r['conversation_id'] in LONG],
                [r['d_minus_c'] for r in st if r['message_src']=='user' and r['group']=='treatment' and r['agent_role']=='Alex' and r['conversation_id'] in LONG])))

print('\n[PRODUCT] treatment vs control (length-normalized):')
print('   idea rate /turn   :', fmt(welch([R[c]['ideas_per_user_turn'] for c in T],[R[c]['ideas_per_user_turn'] for c in C])))
print('   fluency (raw)     :', fmt(welch([R[c]['fluency'] for c in T],[R[c]['fluency'] for c in C])))
print('   orig idea-centroid:', fmt(welch([R[c]['orig_idea_same'] for c in T],[R[c]['orig_idea_same'] for c in C])))
print('   orig topic-resid  :', fmt(welch([R[c]['orig_resid_same'] for c in T],[R[c]['orig_resid_same'] for c in C])))
print('   within-idea div   :', fmt(welch([R[c]['within_idea_diversity'] for c in T],[R[c]['within_idea_diversity'] for c in C])))

print('\n[DOSE-RESPONSE] convergent share -> creativity (within long treatment, n=%d):'%len(T))
print('   alex_share -> idea rate   :', pearson([(R[c]['alex_share'],R[c]['ideas_per_user_turn']) for c in T]))
print('   alex_share -> orig idea   :', pearson([(R[c]['alex_share'],R[c]['orig_idea_same']) for c in T]))
print('   alex_share -> orig resid  :', pearson([(R[c]['alex_share'],R[c]['orig_resid_same']) for c in T]))
print('   alex_share -> within-div  :', pearson([(R[c]['alex_share'],R[c]['within_idea_diversity']) for c in T]))
print('\nWrote: outputs/long_portfolio.csv')
