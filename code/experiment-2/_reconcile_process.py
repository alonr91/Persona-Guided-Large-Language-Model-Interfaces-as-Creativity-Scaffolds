# Process-layer reconciliation: mode switching, ending persona, question rate, conscientiousness.
import pandas as pd, numpy as np
from scipy import stats
DATA=r"Data for analysis and scripts runs"

msg=pd.read_excel(f"{DATA}/messages1307_quartered.xlsx", sheet_name='messages1307_quartered')
# keep clean rows
msg=msg[msg['persona_id'].isin([1,2,3,4]) & msg['message_src'].isin(['user','assistant'])].copy()
msg['persona_id']=msg['persona_id'].astype(int)
msg['grp']=msg['group typer'].astype(str).str.lower().apply(lambda s:'T' if 'exp' in s else ('C' if 'control' in s else np.nan))
msg=msg.dropna(subset=['grp'])
print("clean messages:",len(msg),"| convs:",msg['conversation_id'].nunique())
print("by grp convs:", msg.groupby('grp')['conversation_id'].nunique().to_dict())

# USER messages only for engagement direction
um=msg[msg['message_src']=='user'].copy()
um=um.sort_values(['conversation_id','pos'])

# convergent (persona 2) vs divergent (persona 1) in treatment
def conv_count(g): return (g['persona_id']==2).sum()
def div_count(g): return (g['persona_id']==1).sum()
def longest_streak(seq,val):
    best=cur=0
    for x in seq:
        cur=cur+1 if x==val else 0
        best=max(best,cur)
    return best
rows=[]
for cid,g in um.groupby('conversation_id'):
    grp=g['grp'].iloc[0]
    seq=g['persona_id'].tolist()
    last=seq[-1]
    rows.append(dict(conversation_id=cid,grp=grp,
        conv_msgs=conv_count(g),div_msgs=div_count(g),
        conv_streak=longest_streak(seq,2 if grp=='T' else 4),
        n_user=len(seq),
        end_conv=(last in (2,4)),  # ended with convergent-side button
        end_persona=last))
P=pd.DataFrame(rows)

# merge traits
q=pd.read_excel(f"{DATA}/Responses 13 07 (1).xlsx")
q.columns=[c.strip() for c in q.columns]
traits=q[['conversation_id','Conscientiousness','Negative Emotionality','Extraversion','Open-Mindedness','Agreeableness']].copy()
M=P.merge(traits,on='conversation_id',how='left')
T=M[M['grp']=='T']; C=M[M['grp']=='C']

print("\n-- Conscientiousness x convergent engagement (Treatment) --")
for col,lab in [('conv_msgs','convergent message count'),('conv_streak','convergent streak length')]:
    sub=T[['Conscientiousness',col]].dropna()
    r,p=stats.pearsonr(sub['Conscientiousness'],sub[col])
    print(f"   Conscientiousness x {lab:26s} r={r:+.3f} p={p:.4g} n={len(sub)}")
print("   (Control:)")
for col,lab in [('conv_msgs','convergent message count'),('conv_streak','convergent streak length')]:
    sub=C[['Conscientiousness',col]].dropna()
    r,p=stats.pearsonr(sub['Conscientiousness'],sub[col])
    print(f"   Conscientiousness x {lab:26s} r={r:+.3f} p={p:.4g} n={len(sub)}")

print("\n-- Ending persona by trait quartiles (Treatment) --")
def quartile_chi(df,trait):
    d=df[['end_conv',trait]].dropna()
    lo=d[d[trait]<=d[trait].quantile(0.25)]
    hi=d[d[trait]>=d[trait].quantile(0.75)]
    # contingency: rows=lo/hi, cols=end_div/end_conv
    tab=np.array([[ (~lo['end_conv']).sum(), lo['end_conv'].sum()],
                  [ (~hi['end_conv']).sum(), hi['end_conv'].sum()]])
    chi,p,dof,_=stats.chi2_contingency(tab,correction=False)
    return tab,chi,p,len(lo),len(hi)
for trait in ['Conscientiousness','Negative Emotionality','Extraversion']:
    tab,chi,p,nlo,nhi=quartile_chi(T,trait)
    print(f"   {trait:22s} low(n={nlo}) [endDiv,endConv]={tab[0].tolist()} | high(n={nhi})={tab[1].tolist()} chi2={chi:.3f} p={p:.4g}")
print("   (Control:)")
for trait in ['Conscientiousness','Negative Emotionality','Extraversion']:
    tab,chi,p,nlo,nhi=quartile_chi(C,trait)
    print(f"   {trait:22s} chi2={chi:.3f} p={p:.4g}")

print("\n-- Question rate by quarter & persona (treatment vs control) --")
# question marks per user message
um['nq']=um['message'].astype(str).str.count(r'\?')
um['hasq']=(um['nq']>0).astype(int)
um['side']=um['persona_id'].map({1:'div',2:'conv',3:'div',4:'conv'})  # control 3/4 arbitrary; treat as pooled
# treatment by persona
for side in ['div','conv']:
    print(f"  [{side}] mean '?' per user msg by quarter:")
    for grp in ['T','C']:
        sub=um[(um['grp']==grp)&(um['side']==side)]
        byq=sub.groupby('quarter')['nq'].mean()
        print(f"     {grp}:", {k:round(v,3) for k,v in byq.items()})
# Q2-Q4 divergent treatment vs control overall
for side in ['div','conv']:
    a=um[(um['grp']=='T')&(um['side']==side)&(um['quarter'].isin(['Q2','Q3','Q4']))].groupby('conversation_id')['nq'].mean()
    b=um[(um['grp']=='C')&(um['side']==side)&(um['quarter'].isin(['Q2','Q3','Q4']))].groupby('conversation_id')['nq'].mean()
    t,p=stats.ttest_ind(a,b,equal_var=False)
    print(f"   Q2-4 {side}: T mean={a.mean():.3f}(n={len(a)}) C={b.mean():.3f}(n={len(b)}) t={t:.2f} p={p:.4g}")
