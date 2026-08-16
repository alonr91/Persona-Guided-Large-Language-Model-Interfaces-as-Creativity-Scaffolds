# -*- coding: utf-8 -*-
"""Adjudicate + audit + contrast the USER-only rubric scores (Exp 3 clean sample).

Same machinery as the dialogic-rubric analyzer (adjudicator + bias_audit + Welch),
applied to Experiment 1's six user-behaviour criteria. No reverse-scored criteria.
Directional expectations are exploratory (the user's own behaviour need not shift
as much as the dialogue), so directions are reported, not gated.

Writes: user_rubric_adjudicated.csv, user_rubric_audit.csv, user_rubric_contrast.csv
"""
import csv, math, statistics
from collections import defaultdict

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
RAW = f'{BASE}/outputs/user_rubric_raw.csv'
ADJ = f'{BASE}/outputs/user_rubric_adjudicated.csv'
AUDIT = f'{BASE}/outputs/user_rubric_audit.csv'
CONTRAST = f'{BASE}/outputs/user_rubric_contrast.csv'

CRITERIA = ['user_initiative','user_question_richness','user_proposal_specificity',
            'user_acceptance_yes_and','user_reframing','user_engagement_depth']
# exploratory expected direction (treatment - control); generative-behaviour criteria
EXPECT = {'user_initiative':+1,'user_proposal_specificity':+1,'user_reframing':+1,'user_engagement_depth':+1}

def _betacf(a,b,x):
    qab=a+b;qap=a+1.0;qam=a-1.0;c=1.0;d=1.0-qab*x/qap
    if abs(d)<1e-30:d=1e-30
    d=1.0/d;h=d
    for mm in range(1,300):
        m2=2*mm;aa=mm*(b-mm)*x/((qam+m2)*(a+m2))
        d=1.0+aa*d;d=1e-30 if abs(d)<1e-30 else d;c=1.0+aa/c;c=1e-30 if abs(c)<1e-30 else c
        d=1.0/d;h*=d*c
        aa=-(a+mm)*(qab+mm)*x/((a+m2)*(qap+m2))
        d=1.0+aa*d;d=1e-30 if abs(d)<1e-30 else d;c=1.0+aa/c;c=1e-30 if abs(c)<1e-30 else c
        d=1.0/d;de=d*c;h*=de
        if abs(de-1.0)<1e-12:break
    return h
def _ibeta(x,a,b):
    if x<=0:return 0.0
    if x>=1:return 1.0
    lb=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    fr=math.exp(a*math.log(x)+b*math.log(1-x)-lb)
    return fr*_betacf(a,b,x)/a if x<(a+1.0)/(a+b+2.0) else 1.0-fr*_betacf(b,a,1-x)/b
def welch(A,B):
    A=[float(x) for x in A]; B=[float(x) for x in B]; na,nb=len(A),len(B)
    if na<2 or nb<2: return None
    ma,mb=statistics.mean(A),statistics.mean(B); va,vb=statistics.variance(A),statistics.variance(B)
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)); g=(ma-mb)/sp*(1-3/(4*(na+nb)-9)) if sp>0 else 0.0
    se=math.sqrt(va/na+vb/nb)
    if se==0: return dict(ma=ma,mb=mb,na=na,nb=nb,t=0.0,df=na+nb-2,g=g,p=1.0)
    t=(ma-mb)/se; df=(va/na+vb/nb)**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    return dict(ma=ma,mb=mb,na=na,nb=nb,t=t,df=df,g=g,p=_ibeta(df/(df+t*t),df/2.0,0.5))
def pearson(xs,ys):
    n=len(xs)
    if n<3: return None
    mx=sum(xs)/n; my=sum(ys)/n
    sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); sxx=sum((x-mx)**2 for x in xs); syy=sum((y-my)**2 for y in ys)
    if sxx==0 or syy==0: return 0.0
    return sxy/math.sqrt(sxx*syy)

rows = list(csv.DictReader(open(RAW, encoding='utf-8')))
def fnum(v):
    try:
        return None if v in ('','None',None) else float(v)
    except: return None
cell = defaultdict(dict); meta = {}
for r in rows:
    c=int(r['conversation_id']); crit=r['criterion']; s=r['scorer']
    usable=(str(r['usable_for_inference']).lower()=='true')
    cell[(c,crit)][s]=(fnum(r['score_0_4']) if usable else None)
    cell[(c,crit)][s+'_usable']=usable
    meta[c]=(r['group'], int(float(r['conv_word_count'])))

adj = {}
for (c,crit),d in cell.items():
    a=d.get('A'); b=d.get('B')
    if a is None and b is None: adj[(c,crit)]=dict(final=None,decision='exclude',dis=None,high=False); continue
    if b is None: adj[(c,crit)]=dict(final=a,decision='single_A',dis=None,high=False); continue
    if a is None: adj[(c,crit)]=dict(final=b,decision='single_B',dis=None,high=False); continue
    dd=abs(a-b)
    adj[(c,crit)]=(dict(final=(a+b)/2.0,decision='keep_mean',dis=dd,high=False) if dd<=1.0
                   else dict(final=min(a,b),decision='use_lower',dis=dd,high=True))

with open(ADJ,'w',encoding='utf-8',newline='') as f:
    w=csv.writer(f); w.writerow(['conversation_id','group','criterion','score_A','score_B','final_score','decision','disagreement','high_disagreement','conv_word_count'])
    for (c,crit) in sorted(adj):
        d=adj[(c,crit)]; cd=cell[(c,crit)]
        w.writerow([c,meta[c][0],crit,cd.get('A'),cd.get('B'),'' if d['final'] is None else round(d['final'],2),
                    d['decision'],'' if d['dis'] is None else d['dis'],d['high'],meta[c][1]])

n_hd=sum(1 for d in adj.values() if d['high']); n_sc=sum(1 for d in adj.values() if d['final'] is not None)
usable_rate=sum(1 for d in cell.values() if d.get('A_usable') or d.get('B_usable'))/max(1,len(cell))
print(f"scored cells: {n_sc}/{len(adj)} | high-disagreement (|d|>=2): {n_hd} | usable-evidence rate: {usable_rate:.0%}")
print(f"\n{'criterion':28} {'A-B MAE':>7} {'%|d|>=2':>7} {'A-B r':>6} {'len-bias R2':>11}")
arows=[]
for crit in CRITERIA:
    A=[]; B=[]; sc=[]; wc=[]
    for c in meta:
        cd=cell.get((c,crit),{})
        if cd.get('A') is not None and cd.get('B') is not None: A.append(cd['A']); B.append(cd['B'])
        fin=adj.get((c,crit),{}).get('final')
        if fin is not None: sc.append(fin); wc.append(meta[c][1])
    mae=sum(abs(a-b) for a,b in zip(A,B))/len(A) if A else float('nan')
    pcd=100*sum(1 for a,b in zip(A,B) if abs(a-b)>=2)/len(A) if A else float('nan')
    rab=pearson(A,B) if len(A)>=3 else None
    rl=pearson(sc,wc) if len(sc)>=3 else None; r2=(rl*rl) if rl is not None else None
    arows.append([crit,round(mae,2),round(pcd,0),'' if rab is None else round(rab,2),'' if r2 is None else round(r2,2)])
    print(f"{crit:28} {mae:7.2f} {pcd:6.0f}% {('' if rab is None else f'{rab:+.2f}'):>6} {('' if r2 is None else f'{r2:.2f}'+('*' if r2>0.30 else '')):>11}")
with open(AUDIT,'w',encoding='utf-8',newline='') as f:
    w=csv.writer(f); w.writerow(['criterion','A_B_MAE','pct_disagree_ge2','A_B_pearson_r','length_bias_R2']); w.writerows(arows)

print(f"\n=== USER-only rubric: treatment vs control (conversation-level; Welch, Hedges g; no FDR) ===")
print(f"{'criterion':28} {'T':>6} {'C':>6} {'g':>7} {'p':>6}  exp-dir")
crows=[]
for crit in CRITERIA:
    T=[adj[(c,crit)]['final'] for c in meta if meta[c][0]=='treatment' and adj.get((c,crit),{}).get('final') is not None]
    C=[adj[(c,crit)]['final'] for c in meta if meta[c][0]=='control' and adj.get((c,crit),{}).get('final') is not None]
    w=welch(T,C)
    if not w: crows.append([crit,'','','','','']); continue
    exp=EXPECT.get(crit); d=''
    if exp is not None: d='OK' if ((w['g']>0)==(exp>0)) else 'x'
    crows.append([crit,round(w['ma'],2),round(w['mb'],2),round(w['g'],2),round(w['p'],3),d])
    print(f"{crit:28} {w['ma']:6.2f} {w['mb']:6.2f} {w['g']:+7.2f} {w['p']:6.3f}  {d}")
with open(CONTRAST,'w',encoding='utf-8',newline='') as f:
    w=csv.writer(f); w.writerow(['criterion','treatment_mean','control_mean','hedges_g','p_value','exploratory_direction']); w.writerows(crows)
print(f"\nWrote: {ADJ}\n       {AUDIT}\n       {CONTRAST}")
