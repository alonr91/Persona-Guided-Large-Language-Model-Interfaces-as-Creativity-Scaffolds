# -*- coding: utf-8 -*-
"""Figure 5 — semantic co-regulation trajectories (long sample).
Left:  user idea novelty-to-own-history by quartile (exploitation: ideas converge).
Right: accommodation to the persona (user<->preceding-persona similarity) by quartile
       (treatment individuates from the AI's content over the session; control flat).
"""
import json, csv, math, statistics
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

BASE='C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'; FIG=f'{BASE}/report/figures'
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
TREAT,CTRL='#2c6fbb','#9aa0a6'
LONG={int(r['conversation_id']):r['group'] for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv',encoding='utf-8'))}
msgs=json.load(open(f'{BASE}/data/experiment3_full_en.json',encoding='utf-8'))
by=defaultdict(list)
for m in msgs:
    if m['conversation_id'] in LONG: by[m['conversation_id']].append(m)
for c in by: by[c].sort(key=lambda x:(x['timestamp'],x['message_id']))
bge=SentenceTransformer('BAAI/bge-large-en-v1.5')
emb={}
for c in by:
    V=np.asarray(bge.encode([(m.get('message_en') or m['message']).strip()[:1500] for m in by[c]],normalize_embeddings=True),dtype='float32')
    for m,v in zip(by[c],V): emb[m['message_id']]=v
def cos(a,b): return float(np.dot(a,b))
nov=defaultdict(lambda:[[],[],[],[]]); acc=defaultdict(lambda:[[],[],[],[]])
for c in by:
    g=LONG[c]; seq=by[c]; uh=[]; prev_a=None; n_u=sum(1 for m in seq if m['message_src']=='user'); ui=0
    for m in seq:
        v=emb[m['message_id']]
        if m['message_src']=='assistant': prev_a=v
        else:
            q=min(3,int((ui/(n_u-1) if n_u>1 else 0)*4))
            if uh: nov[g][q].append(1-max(cos(v,h) for h in uh))
            if prev_a is not None: acc[g][q].append(cos(v,prev_a))
            uh.append(v); ui+=1
def mc(bins): return [statistics.mean(b) if b else float('nan') for b in bins],[1.96*statistics.stdev(b)/math.sqrt(len(b)) if len(b)>1 else 0 for b in bins]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.3)); qx=[1,2,3,4]
for g,col,mk,lbl in [('treatment',TREAT,'o','Treatment'),('control',CTRL,'s','Control')]:
    m,e=mc(nov[g]); ax1.errorbar(qx,m,yerr=e,marker=mk,color=col,capsize=3,lw=2,label=lbl)
    m,e=mc(acc[g]); ax2.errorbar(qx,m,yerr=e,marker=mk,color=col,capsize=3,lw=2,label=lbl)
ax1.set_title('Idea novelty vs own history\n(semantic exploitation)'); ax1.set_xlabel('Conversation quartile'); ax1.set_ylabel('Novelty-to-history (1 − max cos)'); ax1.set_xticks(qx); ax1.legend(frameon=False)
ax2.set_title('Accommodation to the persona\n(content-level individuation)'); ax2.set_xlabel('Conversation quartile'); ax2.set_ylabel('User ↔ preceding-persona similarity'); ax2.set_xticks(qx); ax2.legend(frameon=False)
fig.suptitle('Semantic co-regulation over the conversation — long sample',y=1.02)
fig.tight_layout(); fig.savefig(f'{FIG}/fig5_semantic.png',bbox_inches='tight'); print('fig5 OK')
