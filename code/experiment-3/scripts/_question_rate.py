# -*- coding: utf-8 -*-
"""Question-rate over the conversation (Exp-2 Fig 6 analogue), long sample.

% of USER messages containing '?' by conversation quarter (Q1-Q4), split by the
persona addressed (divergent=Taylor / convergent=Alex) in treatment, plus control
overall. Tests the divergent-early / convergent-late questioning cadence.
Colours: divergent #22A06B, convergent #7A6CF5, control blue dashed.
"""
import json, csv, math, statistics
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'; FIG=f'{BASE}/report/figures'
plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
DIV,CON,CTRL='#22A06B','#7A6CF5','#2C6FBB'
LONG={int(r['conversation_id']):r['group'] for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv',encoding='utf-8'))}
msgs=json.load(open(f'{BASE}/data/experiment3_full_en.json',encoding='utf-8'))
by=defaultdict(list)
for m in msgs:
    if m['conversation_id'] in LONG: by[m['conversation_id']].append(m)
for c in by: by[c].sort(key=lambda x:(x['timestamp'],x['message_id']))

def hasq(m): return '?' in (m.get('message_en') or m.get('message') or '')

# collect: (quarter, group, persona) -> list of 0/1 ; control pooled across personas
cell=defaultdict(list)
for c,ms in by.items():
    g=LONG[c]; users=[m for m in ms if m['message_src']=='user']
    n=len(users)
    if n<4: continue
    for i,m in enumerate(users):
        q=min(3,int(i/n*4))
        cell[(q,g,'all')].append(1 if hasq(m) else 0)
        if g=='treatment':
            cell[(q,g,m['persona'])].append(1 if hasq(m) else 0)

def pct(key):
    v=cell.get(key,[]); return (100*statistics.mean(v) if v else float('nan')), len(v)
print('Question-rate (% user msgs with ?) by quarter — long sample')
print('Q   Treat-Div(Taylor)   Treat-Conv(Alex)   Control-overall')
div=[];con=[];ctl=[]
for q in range(4):
    d,_=pct((q,'treatment','Taylor')); c2,_=pct((q,'treatment','Alex')); ct,_=pct((q,'control','all'))
    div.append(d);con.append(c2);ctl.append(ct)
    print(f'Q{q+1}  {d:5.1f}              {c2:5.1f}             {ct:5.1f}')

# Welch tests over Q2-Q4 (exclude Q1 familiarisation, as in Exp 2): treatment persona vs control
def welch(a,b):
    a=[x for x in a];b=[x for x in b];na,nb=len(a),len(b)
    if na<2 or nb<2:return None
    ma,mb=statistics.mean(a),statistics.mean(b);va,vb=statistics.variance(a),statistics.variance(b)
    se=math.sqrt(va/na+vb/nb)
    if se==0:return None
    t=(ma-mb)/se;sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2));g=(ma-mb)/sp*(1-3/(4*(na+nb)-9)) if sp else 0
    return round(100*ma,1),round(100*mb,1),round(t,2),round(g,2),na,nb
q234=lambda grp,per: [x for q in (1,2,3) for x in cell.get((q,grp,per),[])]
print('\nQ2-Q4 contrasts (% with ?):')
print('  Treat divergent vs Control:', welch(q234('treatment','Taylor'), q234('control','all')))
print('  Treat convergent vs Control:', welch(q234('treatment','Alex'), q234('control','all')))

fig,ax=plt.subplots(figsize=(9,5))
qx=[1,2,3,4]
ax.plot(qx,div,marker='o',color=DIV,lw=2.5,ms=8,label='Treatment — Divergent persona (Taylor)')
ax.plot(qx,con,marker='o',color=CON,lw=2.5,ms=8,label='Treatment — Convergent persona (Alex)')
ax.plot(qx,ctl,marker='o',color=CTRL,lw=2.5,ms=7,ls='--',label='Control — overall')
ax.set_xticks(qx); ax.set_xticklabels(['Q1','Q2','Q3','Q4'])
ax.set_xlabel('Conversation quarter'); ax.set_ylabel('% of user messages with “?”')
ax.set_ylim(0,max(40,max(div+con+ctl)+5))
ax.set_title('Question-asking by conversation quarter and persona')
ax.legend(frameon=False,fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/fig4_question_rate.png',bbox_inches='tight'); print('\nfig4_question_rate.png written')
