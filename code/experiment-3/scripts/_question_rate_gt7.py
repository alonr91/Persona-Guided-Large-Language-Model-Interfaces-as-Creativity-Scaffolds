# -*- coding: utf-8 -*-
"""Question-rate by quarter x persona for conversations with > 7 USER messages.

Same metric/figure as fig4 but sample = single-condition full-tier convs with
n_user > 7 (i.e., >=8). Reports the matrix + Q2-Q4 contrasts and saves the figure.
"""
import json, csv, math, statistics
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'; FIG=f'{BASE}/report/figures'
plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
DIV,CON,CTRL='#22A06B','#7A6CF5','#2C6FBB'
summ={int(r['conversation_id']):r for r in csv.DictReader(open(f'{BASE}/outputs/experiment3_conversations_summary.csv',encoding='utf-8'))}
msgs=json.load(open(f'{BASE}/data/experiment3_full_en.json',encoding='utf-8'))
# sample: single-condition (full-tier), n_user > 7
SAMP={c:summ[c]['group'] for c in summ if summ[c]['quality_tier']=='full' and int(summ[c]['n_user'])>7}
from collections import Counter
print('Sample (n_user>7):', len(SAMP), dict(Counter(SAMP.values())))
by=defaultdict(list)
for m in msgs:
    if m['conversation_id'] in SAMP: by[m['conversation_id']].append(m)
for c in by: by[c].sort(key=lambda x:(x['timestamp'],x['message_id']))
def hasq(m): return '?' in (m.get('message_en') or m.get('message') or '')
cell=defaultdict(list)
for c,ms in by.items():
    g=SAMP[c]; users=[m for m in ms if m['message_src']=='user']; n=len(users)
    if n<4: continue
    for i,m in enumerate(users):
        q=min(3,int(i/n*4))
        if g=='treatment': cell[(q,m['persona'])].append(hasq(m))
        if g=='control': cell[(q,'control')].append(hasq(m))
def p(k): v=cell.get(k,[]);return 100*statistics.mean(v) if v else float('nan')
div=[p((q,'Taylor')) for q in range(4)]; con=[p((q,'Alex')) for q in range(4)]; ctl=[p((q,'control')) for q in range(4)]
print('   Q1   Q2   Q3   Q4'); print('Div ',' '.join(f'{x:4.0f}' for x in div)); print('Conv',' '.join(f'{x:4.0f}' for x in con)); print('Ctrl',' '.join(f'{x:4.0f}' for x in ctl))
def welch(a,b):
    a=[x for x in a];b=[x for x in b];na,nb=len(a),len(b)
    if na<2 or nb<2:return None
    ma,mb=statistics.mean(a),statistics.mean(b);va,vb=statistics.variance(a),statistics.variance(b)
    se=math.sqrt(va/na+vb/nb)
    if se==0:return None
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2));g=(ma-mb)/sp*(1-3/(4*(na+nb)-9)) if sp else 0;t=(ma-mb)/se
    return round(100*ma,1),round(100*mb,1),round(t,2),round(g,2),na,nb
q234=lambda per:[x for q in (1,2,3) for x in cell.get((q,per),[])]
print('\nQ2-Q4: Treat divergent vs Control:', welch(q234('Taylor'),q234('control')))
print('Q2-Q4: Treat convergent vs Control:', welch(q234('Alex'),q234('control')))
fig,ax=plt.subplots(figsize=(9,5)); qx=[1,2,3,4]
ax.plot(qx,div,marker='o',color=DIV,lw=2.5,ms=8,label='Treatment — Divergent persona (Taylor)')
ax.plot(qx,con,marker='o',color=CON,lw=2.5,ms=8,label='Treatment — Convergent persona (Alex)')
ax.plot(qx,ctl,marker='o',color=CTRL,lw=2.5,ms=7,ls='--',label='Control — overall')
ax.set_xticks(qx); ax.set_xticklabels(['Q1','Q2','Q3','Q4'])
ax.set_xlabel('Conversation quarter'); ax.set_ylabel('% of user messages with “?”'); ax.set_ylim(0,max(40,max(div+con+ctl)+5))
ax.set_title('Question-asking by quarter and persona (>7 user messages; 14 T / 7 C)')
ax.legend(frameon=False,fontsize=10)
fig.tight_layout(); fig.savefig(f'{FIG}/fig_qrate_gt7.png',bbox_inches='tight'); print('\nfig_qrate_gt7.png written')
