# -*- coding: utf-8 -*-
"""Creativity in embedding space, TREATMENT-ONLY, conditioned on the persona engaged.

Lit-grounded measures on user IDEAS (Agent-1 candidates, bge-large embeddings):
  - FORWARD FLOW (Gray et al. 2019): mean cosine distance of idea_i to all PRIOR
    ideas in the conversation -> how far thought moves into new territory.
  - SEMANTIC-DISTANCE ORIGINALITY (Beaty & Johnson 2021): distance from the
    conversation's CHALLENGE brief to the idea -> remoteness from the problem.
  - DSI (Johnson et al. 2022): mean pairwise distance among a conversation's ideas.
Each idea is tagged by the persona the user addressed in the turn that produced it
(Taylor=divergent / Alex=convergent). We ask: are divergent-engaged ideas more remote
and higher forward flow than convergent-engaged ones?
"""
import json, csv, math, statistics
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
cache = json.load(open(f'{BASE}/outputs/_agent1_candidates_cache.json', encoding='utf-8'))
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
labels = {int(r['conversation_id']): r['challenge_id'] for r in csv.DictReader(open(f'{BASE}/outputs/challenge_labels.csv', encoding='utf-8'))}
LONG = {int(r['conversation_id']) for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv', encoding='utf-8'))}
summ = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/experiment3_conversations_summary.csv', encoding='utf-8'))}
CHAL = {
 'galilee_upper':'connect older and younger populations to the kibbutz physical and social space',
 'eshkol_nir_yitzhak':'cooperation among southern kibbutzim to grow shared community and economic capital',
 'natal_trauma_language':'a language reflecting mental-distress states in Israel since October 7',
 'sderot_wellbeing':'improve the mood and sense of meaning of returning Sderot residents',
 'polyron_sleep':'new therapeutic sleep products and the next-generation mattress for trauma and rehabilitation',
 'ichilov_rehab_future':'the rehabilitation hospital of the future for rehab patients, amputees and combat-injured',
 'joint_rikma_jewish_arab':'Jewish-Arab organizations keeping empathy and an inclusive culture in mixed workplaces',
 'ta_south_community':'make diverse Tel Aviv South populations feel represented with shared public meeting points',
 'ta_east_reut_yad_eliyahu':'Reut rehabilitation hospital and the Yad Eliyahu neighborhood, mutual community resilience',
 'ta_youth_disability_clothing':'young people with disabilities and the challenge of clothing and dressing',
 'other_unclear':'an open creative design problem',
}
msg_persona = {m['message_id']: m['persona'] for m in msgs}
msg_ts = {m['message_id']: (m['timestamp'], m['message_id']) for m in msgs}
conv_group = {}
for m in msgs: conv_group[m['conversation_id']] = m['conv_group']

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
def paired_t(diffs):
    diffs=[d for d in diffs if d is not None]; n=len(diffs)
    if n<3: return None
    m=statistics.mean(diffs); sd=statistics.stdev(diffs); se=sd/math.sqrt(n)
    if se==0: return None
    t=m/se; dz=m/sd
    # two-sided p via t
    df=n-1; x=df/(df+t*t);A,B=df/2,.5
    def bcf(x,a,b):
        f=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1;d=1-qab*x/qap;d=1/(d if abs(d)>f else f);h=d
        for k in range(1,200):
            k2=2*k;aa=k*(b-k)*x/((qam+k2)*(a+k2));d=1+aa*d;d=1/(d if abs(d)>f else f);c=1+aa/c;c=c if abs(c)>f else f;h*=d*c
            aa=-(a+k)*(qab+k)*x/((a+k2)*(qap+k2));d=1+aa*d;d=1/(d if abs(d)>f else f);c=1+aa/c;c=c if abs(c)>f else f;de=d*c;h*=de
            if abs(de-1)<3e-12:break
        return h
    lb=math.lgamma(A)+math.lgamma(B)-math.lgamma(A+B)
    p=(math.exp(A*math.log(x)+B*math.log(1-x)-lb)*bcf(x,A,B)/A) if x<(A+1)/(A+B+2) else (1-math.exp(A*math.log(x)+B*math.log(1-x)-lb)*bcf(1-x,B,A)/B)
    return round(m,4),round(dz,2),round(max(0,min(1,p)),4),n
def pear(pairs):
    pts=[(a,b) for a,b in pairs if a is not None and b is not None]; n=len(pts)
    if n<4: return None
    X=[p[0] for p in pts];Y=[p[1] for p in pts];mx,my=statistics.mean(X),statistics.mean(Y);sx,sy=statistics.pstdev(X),statistics.pstdev(Y)
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

# gather treatment ideas
bge = SentenceTransformer('BAAI/bge-large-en-v1.5')
chal_emb = {k: np.asarray(bge.encode('passage: '+v, normalize_embeddings=True),dtype='float32') for k,v in CHAL.items()}

def run(conv_set, label):
    ideas=defaultdict(list)  # conv -> list of (ts, persona, text)
    for ck, cands in cache.items():
        c=int(ck)
        if c not in conv_set or conv_group.get(c)!='treatment': continue
        for d in cands:
            mid=int(d['message_id']); per=msg_persona.get(mid)
            ideas[c].append((msg_ts.get(mid,('',mid)), per, f"{d['title']}: {d['description']}"))
    # embed
    alltext=[]; idx=[]
    for c in ideas:
        ideas[c].sort(key=lambda x:x[0])
        for i,(ts,per,txt) in enumerate(ideas[c]): alltext.append('passage: '+txt); idx.append((c,i))
    if not alltext: print('no ideas'); return
    V=np.asarray(bge.encode(alltext, normalize_embeddings=True),dtype='float32')
    vec={idx[j]:V[j] for j in range(len(idx))}
    # per-idea forward flow + challenge distance, tagged by persona
    ff_by_p=defaultdict(list); cd_by_p=defaultdict(list)
    ff_conv_p=defaultdict(lambda: defaultdict(list)); cd_conv_p=defaultdict(lambda: defaultdict(list))
    dsi=[]; tshare=[]
    for c in ideas:
        seq=ideas[c]; cb=chal_emb[labels.get(c,'other_unclear')]
        embs=[vec[(c,i)] for i in range(len(seq))]
        for i,(ts,per,txt) in enumerate(seq):
            cd=1-float(np.dot(embs[i],cb)); cd_by_p[per].append(cd); cd_conv_p[c][per].append(cd)
            if i>0:
                ff=float(np.mean([1-float(np.dot(embs[i],embs[j])) for j in range(i)]))
                ff_by_p[per].append(ff); ff_conv_p[c][per].append(ff)
        if len(embs)>=2:
            dsi.append(float(np.mean([1-float(np.dot(embs[a],embs[b])) for a in range(len(embs)) for b in range(a+1,len(embs))])))
            t=float(summ[c]['msgs_to_Taylor']);a=float(summ[c]['msgs_to_Alex']); tshare.append(t/(t+a) if (t+a)>0 else None)
    print(f'\n############ {label}  (convs={len(ideas)}, ideas={len(idx)}) ############')
    print('--- idea-level (Taylor-engaged vs Alex-engaged) ---')
    print('  Forward flow:        ', welch(ff_by_p['Taylor'], ff_by_p['Alex']), '(g>0 => Taylor ideas move farther from prior)')
    print('  Challenge distance:  ', welch(cd_by_p['Taylor'], cd_by_p['Alex']), '(g>0 => Taylor ideas more remote from problem)')
    print('--- conversation-level paired (within-conv Taylor-mean minus Alex-mean) ---')
    ff_diffs=[statistics.mean(ff_conv_p[c]['Taylor'])-statistics.mean(ff_conv_p[c]['Alex']) for c in ff_conv_p if ff_conv_p[c]['Taylor'] and ff_conv_p[c]['Alex']]
    cd_diffs=[statistics.mean(cd_conv_p[c]['Taylor'])-statistics.mean(cd_conv_p[c]['Alex']) for c in cd_conv_p if cd_conv_p[c]['Taylor'] and cd_conv_p[c]['Alex']]
    print('  Forward flow (paired Δ Taylor-Alex):', paired_t(ff_diffs))
    print('  Challenge dist (paired Δ Taylor-Alex):', paired_t(cd_diffs))
    print('--- DSI (portfolio integration) vs Taylor share, per conversation ---')
    print('  DSI ~ Taylor share:', pear(list(zip(tshare,dsi))))
    print(f'  mean DSI={statistics.mean(dsi):.3f}')

run(set(summ), 'ALL TREATMENT (full-tier)')
run(LONG, 'LONG TREATMENT')
