# -*- coding: utf-8 -*-
"""Semantic search SHAPE (treatment, long sample): does the idea space expand then plateau?

For each conversation, take user IDEAS (Agent-1 candidates) in time order, embed (bge),
and trace how the explored idea space grows:
  cumulative spread S(k) = mean pairwise cosine distance among the first k ideas
  incremental expansion inc(k) = distance of idea k to the centroid of ideas 1..k-1
A concave S-curve (rises early, flattens late) + a declining inc-curve = the semantic
analogue of explore-then-consolidate (Double Diamond). Treatment is the focus; control
shown as a faint reference. Also reports: fraction of final spread reached by the
conversation midpoint, and an area-under-curve concavity index (AUC of normalised
S vs progress; 0.5 = linear, >0.5 = front-loaded expansion).
"""
import json, csv, math, statistics
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'; FIG=f'{BASE}/report/figures'
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
TREAT,CTRL='#2c6fbb','#9aa0a6'
cache=json.load(open(f'{BASE}/outputs/_agent1_candidates_cache.json',encoding='utf-8'))
msgs=json.load(open(f'{BASE}/data/experiment3_full_en.json',encoding='utf-8'))
LONG={int(r['conversation_id']):r['group'] for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv',encoding='utf-8'))}
msg_ts={m['message_id']:(m['timestamp'],m['message_id']) for m in msgs}

bge=SentenceTransformer('BAAI/bge-large-en-v1.5')
ideas=defaultdict(list)
for ck,cands in cache.items():
    c=int(ck)
    if c not in LONG: continue
    for d in cands: ideas[c].append((msg_ts.get(int(d['message_id']),('',0)), f"{d['title']}: {d['description']}"))
alltext=[]; idx=[]
for c in ideas:
    ideas[c].sort(key=lambda x:x[0])
    for i,(ts,txt) in enumerate(ideas[c]): alltext.append('passage: '+txt); idx.append((c,i))
V=np.asarray(bge.encode(alltext,normalize_embeddings=True),dtype='float32')
vec={idx[j]:V[j] for j in range(len(idx))}

GRID=np.linspace(0,1,11)
def curves(group):
    cum=[]; inc=[]; frac_mid=[]; auc=[]
    for c in ideas:
        if LONG[c]!=group: continue
        E=[vec[(c,i)] for i in range(len(ideas[c]))]; N=len(E)
        if N<4: continue
        S=[]; I=[]
        for k in range(2,N+1):
            sub=E[:k]
            S.append(float(np.mean([1-float(np.dot(sub[a],sub[b])) for a in range(k) for b in range(a+1,k)])))
        cen=E[0].copy()
        for k in range(1,N):
            c0=np.mean(E[:k],axis=0); c0=c0/(np.linalg.norm(c0)+1e-12)
            I.append(1-float(np.dot(E[k],c0)))
        # normalise S to final, progress over k=2..N
        Sf=S[-1] if S[-1]>0 else 1
        prog=np.linspace(0,1,len(S)); Sn=np.array(S)/Sf
        cum.append(np.interp(GRID,prog,Sn))
        progi=np.linspace(0,1,len(I)); inc.append(np.interp(GRID,progi,np.array(I)))
        # fraction of final spread reached by midpoint
        frac_mid.append(float(np.interp(0.5,prog,Sn)))
        auc.append(float(np.sum((Sn[1:]+Sn[:-1])/2*np.diff(prog))))  # trapezoid; >0.5 => concave/front-loaded
    return np.array(cum),np.array(inc),frac_mid,auc

cumT,incT,fmT,aucT=curves('treatment')
cumC,incC,fmC,aucC=curves('control')
def mean_ci(M):
    m=M.mean(axis=0); e=1.96*M.std(axis=0,ddof=1)/math.sqrt(M.shape[0]); return m,e
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.3))
m,e=mean_ci(cumT); ax1.plot(GRID,m,color=TREAT,lw=2.5,marker='o',label=f'Treatment (n={cumT.shape[0]})'); ax1.fill_between(GRID,m-e,m+e,color=TREAT,alpha=.15)
if len(cumC): mc,ec=mean_ci(cumC); ax1.plot(GRID,mc,color=CTRL,lw=1.5,ls='--',marker='s',ms=4,label=f'Control (n={cumC.shape[0]})')
ax1.plot([0,1],[0,1],color='k',lw=0.7,ls=':',label='linear (no plateau)')
ax1.set_title('Cumulative semantic coverage of ideas\n(expand → plateau)'); ax1.set_xlabel('Conversation progress'); ax1.set_ylabel('Idea-space spread (× final)'); ax1.legend(frameon=False,fontsize=9)
m,e=mean_ci(incT); ax2.plot(GRID,m,color=TREAT,lw=2.5,marker='o',label='Treatment'); ax2.fill_between(GRID,m-e,m+e,color=TREAT,alpha=.15)
if len(incC): mc,ec=mean_ci(incC); ax2.plot(GRID,mc,color=CTRL,lw=1.5,ls='--',marker='s',ms=4,label='Control')
ax2.set_title('Incremental expansion (new territory per idea)\n(exploration → exploitation)'); ax2.set_xlabel('Conversation progress'); ax2.set_ylabel('Distance of new idea to prior centroid'); ax2.legend(frameon=False,fontsize=9)
fig.suptitle('Semantic search shape over the conversation — treatment group',y=1.02)
fig.tight_layout(); fig.savefig(f'{FIG}/fig5_searchshape.png',bbox_inches='tight'); plt.close(fig)
print('=== SEARCH SHAPE (treatment, long) ===')
print(f'  conversations: treatment={cumT.shape[0]}, control={len(cumC)}')
print(f'  fraction of final idea-space spread reached by MIDPOINT: treatment mean={statistics.mean(fmT):.2f} (n={len(fmT)})'+ (f', control={statistics.mean(fmC):.2f}' if fmC else ''))
print(f'  concavity AUC (0.5=linear, >0.5=front-loaded expansion): treatment mean={statistics.mean(aucT):.3f}')
# one-sample test AUC>0.5
import statistics as S
m=S.mean(aucT); sd=S.stdev(aucT); t=(m-0.5)/(sd/math.sqrt(len(aucT)))
print(f'    AUC vs 0.5: t={t:.2f}, n={len(aucT)} (positive => concave/front-loaded)')
print('  fig5_searchshape.png written')
