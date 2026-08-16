# -*- coding: utf-8 -*-
"""Regenerate all report figures on the LONG-conversation sample (>=10 user turns)."""
import json, csv, math, statistics, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
FIG = f'{base}/report/figures'; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
TREAT,CTRL='#2c6fbb','#9aa0a6'; DIV,CON='#22A06B','#7A6CF5'  # divergent green, convergent purple (consistent palette)

LONG={int(r['conversation_id']):r['group'] for r in csv.DictReader(open(f'{base}/outputs/long_sample_ids.csv',encoding='utf-8'))}
st=json.load(open(f'{base}/outputs/stance_per_message_full.json',encoding='utf-8'))
port={int(r['conversation_id']):r for r in csv.DictReader(open(f'{base}/outputs/long_portfolio.csv',encoding='utf-8'))}
def ci95(xs):
    xs=[float(x) for x in xs if x not in ('','nan',None)]
    if len(xs)<2: return (statistics.mean(xs) if xs else 0),0
    return statistics.mean(xs),1.96*statistics.stdev(xs)/math.sqrt(len(xs))
def fnum(v):
    try: x=float(v); return None if (math.isnan(x) or math.isinf(x)) else x
    except: return None

# FIG1 manipulation (long)
cells=[('treatment','Taylor','Treatment\nTaylor (div.)'),('treatment','Alex','Treatment\nAlex (conv.)'),
       ('control','Taylor','Control\nTaylor'),('control','Alex','Control\nAlex')]
D,De,Cc,Ce,labs=[],[],[],[],[]
for g,p,lb in cells:
    sub=[r for r in st if r['message_src']=='assistant' and r['group']==g and r['agent_role']==p and r['conversation_id'] in LONG]
    dm,de=ci95([r['divergent_score'] for r in sub]); cm,ce=ci95([r['convergent_score'] for r in sub])
    D.append(dm);De.append(de);Cc.append(cm);Ce.append(ce);labs.append(lb)
x=np.arange(4);w=0.38
fig,ax=plt.subplots(figsize=(8,4.2))
ax.bar(x-w/2,D,w,yerr=De,capsize=3,color=DIV,label='Divergent stance')
ax.bar(x+w/2,Cc,w,yerr=Ce,capsize=3,color=CON,label='Convergent stance')
ax.set_xticks(x);ax.set_xticklabels(labs);ax.set_ylabel('Mean stance score (0–1)')
ax.set_title('Manipulation check — long conversations (assistant turns)')
ax.legend(frameon=False,loc='upper right');fig.tight_layout();fig.savefig(f'{FIG}/fig1_manipulation_check.png');plt.close(fig)

# FIG2 persona engagement (long)
fig,(a1,a2)=plt.subplots(1,2,figsize=(9.5,4.2))
for ax,grp,title in [(a1,'treatment','Treatment'),(a2,'control','Control')]:
    rs=[port[c] for c in LONG if LONG[c]==grp]
    tm,te=ci95([r['msgs_to_Taylor'] for r in rs]); am,ae=ci95([r['msgs_to_Alex'] for r in rs])
    ax.bar([0,1],[tm,am],yerr=[te,ae],capsize=4,color=[DIV,CON],width=0.6)
    ax.set_xticks([0,1]);ax.set_xticklabels(['Divergent\n(Taylor)','Convergent\n(Alex)'])
    ax.set_title(f'{title} (n={len(rs)})')
    if ax is a1: ax.set_ylabel('Mean messages addressed (±95% CI)')
fig.suptitle('Persona engagement — long conversations',y=1.0);fig.tight_layout();fig.savefig(f'{FIG}/fig2_persona_engagement.png');plt.close(fig)

# FIG3 trajectory (long)
bya=defaultdict(list)
for r in st:
    if r['message_src']=='assistant' and r['conversation_id'] in LONG: bya[r['conversation_id']].append(r)
for c in bya: bya[c].sort(key=lambda x:(x['timestamp'],x['message_id']))
def qm(group):
    Q=[[],[],[],[]]
    for c,ms in bya.items():
        if LONG[c]!=group or len(ms)<4: continue
        n=len(ms)
        for qi in range(4):
            seg=ms[qi*n//4:(qi+1)*n//4]
            if seg: Q[qi].append(statistics.mean(r['d_minus_c'] for r in seg))
    return [statistics.mean(q) if q else float('nan') for q in Q],[ci95(q)[1] if len(q)>1 else 0 for q in Q]
tm,te=qm('treatment');cm,ce=qm('control')
fig,ax=plt.subplots(figsize=(7.5,4.3));qx=[1,2,3,4]
ax.errorbar(qx,tm,yerr=te,marker='o',color=TREAT,capsize=3,label='Treatment',lw=2)
ax.errorbar(qx,cm,yerr=ce,marker='s',color=CTRL,capsize=3,label='Control',lw=2)
ax.axhline(0,color='k',lw=0.6,ls=':');ax.set_xticks(qx);ax.set_xlabel('Conversation quartile')
ax.set_ylabel('Assistant divergent–convergent balance (D−C)')
ax.set_title('Divergence-to-convergence trajectory — long conversations')
ax.legend(frameon=False);fig.tight_layout();fig.savefig(f'{FIG}/fig3_trajectory.png');plt.close(fig)

# FIG3b choreography (long)
fig,ax=plt.subplots(figsize=(7,4.3))
for grp,color,mark in [('treatment',TREAT,'o'),('control',CTRL,'s')]:
    rs=[port[c] for c in LONG if LONG[c]==grp]
    m1,e1=ci95([r['div_share_first_half'] for r in rs]); m2,e2=ci95([r['div_share_second_half'] for r in rs])
    ax.errorbar([0,1],[m1,m2],yerr=[e1,e2],marker=mark,color=color,capsize=4,lw=2,label=f'{grp.capitalize()} (n={len(rs)})')
ax.set_xticks([0,1]);ax.set_xticklabels(['First half','Second half']);ax.set_ylim(0.1,1.0)
ax.set_ylabel('Share of messages to divergent persona');ax.axhline(0.5,color='k',lw=0.6,ls=':')
ax.set_title('Front-loaded divergent engagement — long conversations')
ax.legend(frameon=False);fig.tight_layout();fig.savefig(f'{FIG}/fig3b_choreography.png');plt.close(fig)

# FIG4 product effect sizes (long, normalized)
def g_from(key):
    Tt=[fnum(port[c][key]) for c in LONG if LONG[c]=='treatment']; Cc=[fnum(port[c][key]) for c in LONG if LONG[c]=='control']
    Tt=[x for x in Tt if x is not None]; Cc=[x for x in Cc if x is not None]
    if len(Tt)<2 or len(Cc)<2: return None
    na,nb=len(Tt),len(Cc);ma,mb=statistics.mean(Tt),statistics.mean(Cc);va,vb=statistics.variance(Tt),statistics.variance(Cc)
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2));return (ma-mb)/sp*(1-3/(4*(na+nb)-9)) if sp else 0
bars=[('Idea rate\n(ideas / user turn)','ideas_per_user_turn'),('Fluency\n(idea count)','fluency'),
      ('Within-portfolio\nidea diversity','within_idea_diversity'),
      ('Originality\n(topic-residualized)','orig_resid_same'),('Originality\n(idea-centroid)','orig_idea_same')]
B=[(lb,g_from(k)) for lb,k in bars]; B=[(lb,g) for lb,g in B if g is not None]
fig,ax=plt.subplots(figsize=(8,0.7*len(B)+1.5))
gs=[b[1] for b in B];y=np.arange(len(B));colors=[TREAT if v>=0 else CON for v in gs]
ax.barh(y,gs,color=colors,height=0.6);ax.axvline(0,color='k',lw=0.8)
for thr in (0.2,0.5,0.8): ax.axvline(thr,color='gray',lw=0.5,ls=':');ax.axvline(-thr,color='gray',lw=0.5,ls=':')
ax.set_yticks(y);ax.set_yticklabels([b[0] for b in B]);ax.invert_yaxis()
ax.set_xlabel("Hedges' g (treatment − control), length-normalised");ax.set_xlim(-1.9,1.1)
ax.set_title('Idea-portfolio creativity — long conversations')
for yi,v in zip(y,gs): ax.text(v+(0.02 if v>=0 else -0.02),yi,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
fig.tight_layout();fig.savefig(f'{FIG}/fig4_originality.png');plt.close(fig)
print('Long-sample figures written to',FIG)
