# -*- coding: utf-8 -*-
"""Sensitivity analysis: LONG & STABLE conversations only.

LONG  = >= 10 user turns (above the natural 7->10 gap in the length distribution).
STABLE= single calendar day (a coherent working session, not an intermittent multi-day
        return; this also removes the >1000-minute idle-gap outliers).
Subset: 12 treatment, 5 control. Re-runs the headline contrasts and compares to the
full-tier (31/9) values. Simple Welch/Pearson, no correction. Small n -> exploratory.
"""
import json, csv, math, statistics
from collections import defaultdict

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
summ = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/experiment3_conversations_summary.csv', encoding='utf-8'))}
st   = json.load(open(f'{BASE}/outputs/stance_per_message_full.json', encoding='utf-8'))
sw   = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/switching_per_conv.csv', encoding='utf-8'))}
port = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/idea_portfolio_exp1.csv', encoding='utf-8'))}
nrm  = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/dose_response_normalized.csv', encoding='utf-8'))}
emb  = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/originality_topic_controlled.csv', encoding='utf-8'))}

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

# ---- define subset ----
def is_ls(c):
    r=summ.get(c)
    return r and r['quality_tier']=='full' and int(r['n_user'])>=10 and int(r['n_days'])==1
LS=set(c for c in summ if is_ls(c))
gT=[c for c in LS if summ[c]['group']=='treatment']; gC=[c for c in LS if summ[c]['group']=='control']
print(f"LONG & STABLE subset: {len(LS)} convs ({len(gT)} treatment, {len(gC)} control)")
print(f"  treatment ids: {sorted(gT)}\n  control ids:   {sorted(gC)}\n")

def fmt(w):
    return f"T={w[0]} C={w[1]} g={w[2]:+.2f} p={w[3]} (n {w[4]}/{w[5]})" if w else "n/a"

# 1) Manipulation: Taylor vs Alex divergent (assistant turns) in treatment subset
T=[r['divergent_score'] for r in st if r['message_src']=='assistant' and r['group']=='treatment' and r['agent_role']=='Taylor' and r['conversation_id'] in LS]
A=[r['divergent_score'] for r in st if r['message_src']=='assistant' and r['group']=='treatment' and r['agent_role']=='Alex' and r['conversation_id'] in LS]
print("1) Manipulation (Taylor vs Alex divergent, treatment):", fmt(welch(T,A)), " [full-tier g=1.47]")

# 2) Persona preference: treatment msgs to Taylor vs Alex
mt=[fnum(summ[c]['msgs_to_Taylor']) for c in gT]; ma=[fnum(summ[c]['msgs_to_Alex']) for c in gT]
print("2) Persona preference (msgs Taylor vs Alex, treatment):", fmt(welch(mt,ma)), " [full-tier g=0.75]")

# 3) Choreography: divergent-share first half, treatment vs control
t1=[fnum(sw[c]['div_share_first_half']) for c in gT]; c1=[fnum(sw[c]['div_share_first_half']) for c in gC]
print("3) Choreography (divergent share, FIRST half, T vs C):", fmt(welch(t1,c1)), " [full-tier g=1.07]")
# within-treatment first vs second half
f1=[fnum(sw[c]['div_share_first_half']) for c in gT]; f2=[fnum(sw[c]['div_share_second_half']) for c in gT]
print(f"   treatment divergent share first->second half: {statistics.mean([x for x in f1 if x is not None]):.3f} -> {statistics.mean([x for x in f2 if x is not None]):.3f}")

# 4) Product (length-normalized), treatment vs control
print("\n4) PRODUCT (treatment vs control), long & stable:")
print("   idea rate /turn :", fmt(welch([fnum(nrm[c]['ideas_per_user_turn']) for c in gT],[fnum(nrm[c]['ideas_per_user_turn']) for c in gC])), " [full g=-0.02]")
print("   originality(same):", fmt(welch([fnum(port[c]['orig_same']) for c in gT],[fnum(port[c]['orig_same']) for c in gC])), " [full g=-0.17]")
print("   within-idea div  :", fmt(welch([fnum(port[c]['within_idea_diversity']) for c in gT],[fnum(port[c]['within_idea_diversity']) for c in gC])), " [full g=0.16]")
print("   resid originality:", fmt(welch([fnum(emb[c]['resid_same_cond_originality']) for c in gT if c in emb],[fnum(emb[c]['resid_same_cond_originality']) for c in gC if c in emb])), " [full g=0.80]")
print("   fluency (raw)    :", fmt(welch([fnum(port[c]['fluency']) for c in gT],[fnum(port[c]['fluency']) for c in gC])))

# 5) Dose-response within treatment subset
print("\n5) DOSE-RESPONSE within long&stable treatment (convergent share -> creativity):")
ash=lambda c: fnum(nrm[c]['alex_share'])
print("   alex_share -> idea rate   :", pearson([(ash(c), fnum(nrm[c]['ideas_per_user_turn'])) for c in gT]))
print("   alex_share -> originality :", pearson([(ash(c), fnum(port[c]['orig_same'])) for c in gT]))
print("   alex_share -> within-div  :", pearson([(ash(c), fnum(port[c]['within_idea_diversity'])) for c in gT]))

# 6) Fidelity/drift on long&stable treatment (early vs late half gap)
def half(half_):
    Tt,Aa=[],[]
    for c in gT:
        ast=[r for r in st if r['conversation_id']==c and r['message_src']=='assistant']
        ast.sort(key=lambda x:(x['timestamp'],x['message_id'])); n=len(ast)
        seg=ast[:n//2] if half_=='early' else ast[n//2:]
        Tt+=[r['divergent_score'] for r in seg if r['agent_role']=='Taylor']
        Aa+=[r['divergent_score'] for r in seg if r['agent_role']=='Alex']
    return Tt,Aa
print("\n6) Fidelity (Taylor vs Alex divergent, treatment):")
for h in ('early','late'):
    Tt,Aa=half(h); print(f"   [{h} half]:", fmt(welch(Tt,Aa)))
