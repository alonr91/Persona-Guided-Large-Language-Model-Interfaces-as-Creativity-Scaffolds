"""Divergent vs Convergent exploratory analysis (HCI lens)."""
import os, re, numpy as np, pandas as pd
from scipy import stats
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)

OUT = 'analysis_out'
logs = pd.read_csv('Experiment1_logs.csv').sort_values(['conversation_id','message_id']).reset_index(drop=True)
logs['turn_idx'] = logs.groupby('conversation_id').cumcount()
logs['turn_frac'] = logs['turn_idx'] / logs.groupby('conversation_id')['turn_idx'].transform('max').replace(0,1)

up = (logs.groupby('User_id')['Persona_type']
      .apply(lambda s: [x for x in s.unique() if x!='GPT'][0])
      .reset_index().rename(columns={'Persona_type':'persona_cond'}))
fm = {'Divergent':'Divergent','Convergent':'Convergent','strictly rational':'Rational','bounded rationality':'BoundedRational'}
up['family'] = up['persona_cond'].map(fm)

RX = {
 'propose'  : re.compile(r"\b(what if|how about|could|maybe|suggest|propose|idea(s)?|imagine|consider|another option|alternatively)\b", re.I),
 'critique' : re.compile(r"\b(but|however|issue|problem|concern|doesn't|won'?t work|too (expensive|complex|hard)|drawback|risk|downside|not sure|disagree)\b", re.I),
 'compare'  : re.compile(r"\b(vs\.?|versus|compare|compared|trade ?off|rather than|better than|worse than)\b", re.I),
 'commit'   : re.compile(r"\b(let\'?s go with|decide|final|choose|pick|commit|we will|settle on|go with)\b", re.I),
 'reframe'  : re.compile(r"\b(actually|reframe|different angle|step back|bigger picture|instead think|what if the problem|really about)\b", re.I),
 'question' : re.compile(r"\?"),
}
for k,rx in RX.items():
    logs[f'tag_{k}'] = logs['message'].fillna('').str.contains(rx)

def wc(x): return 0 if pd.isna(x) else len(re.findall(r"\w+", str(x)))
logs['n_words'] = logs['message'].apply(wc)

conv = logs.groupby('conversation_id').agg(user=('User_id','first'), persona_type=('Persona_type','first')).reset_index()
conv['condition'] = np.where(conv['persona_type']=='GPT','GPT','Persona')
conv = conv.merge(up[['User_id','family']], left_on='user', right_on='User_id').drop(columns=['User_id'])

def stance_agg(g):
    out={}
    for side, sub in [('u', g[g.message_src=='user']), ('a', g[g.message_src=='assistant'])]:
        n = max(1, len(sub))
        for k in RX: out[f'{side}_{k}'] = sub[f'tag_{k}'].sum()/n
    return pd.Series(out)
st = logs.groupby('conversation_id').apply(stance_agg, include_groups=False).reset_index()
conv = conv.merge(st, on='conversation_id')

E = np.load(os.path.join(OUT,'msg_embeddings.npy'))
prev = logs['conversation_id'].shift(1); same = (logs['conversation_id']==prev).values
sim = np.full(len(E), np.nan); sim[1:] = (E[1:]*E[:-1]).sum(axis=1)
trans = logs.assign(prev_speaker=logs['message_src'].shift(1), dist=1-sim)[same].copy().rename(columns={'message_src':'speaker'})
trans['transition_type'] = trans['prev_speaker'].astype(str)+'->'+trans['speaker'].astype(str)
conv = conv.merge(trans.groupby('conversation_id')['dist'].mean().rename('novelty_all'), on='conversation_id')
conv = conv.merge(trans.groupby(['conversation_id','speaker'])['dist'].mean().unstack('speaker')
                  .rename(columns={'user':'novelty_user','assistant':'novelty_ast'}), on='conversation_id')

S = np.load(os.path.join(OUT,'msg_surprise_per_tok.npy'))
msurp = logs.assign(surprise=S).dropna(subset=['surprise']).rename(columns={'message_src':'speaker'})
conv = conv.merge(msurp.groupby('conversation_id')['surprise'].mean().rename('surp_all'), on='conversation_id', how='left')
conv = conv.merge(msurp.groupby(['conversation_id','speaker'])['surprise'].mean().unstack('speaker')
                  .rename(columns={'user':'surp_user','assistant':'surp_ast'}), on='conversation_id', how='left')

vals = ['novelty_all','novelty_user','novelty_ast','surp_all','surp_user','surp_ast',
        'u_propose','u_critique','u_compare','u_commit','u_reframe','u_question',
        'a_propose','a_critique','a_compare','a_commit','a_reframe']
wide = conv.pivot_table(index='user', columns='condition', values=vals, aggfunc='first')
wide.columns = [f'{a}__{b}' for a,b in wide.columns]
wide = wide.reset_index().merge(up[['User_id','family']], left_on='user', right_on='User_id').drop(columns=['User_id'])
for v in vals:
    wide[f'd_{v}'] = wide[f'{v}__Persona'].astype(float) - wide[f'{v}__GPT'].astype(float)

DC = wide[wide['family'].isin(['Divergent','Convergent'])].copy()
div = DC[DC.family=='Divergent']; cvg = DC[DC.family=='Convergent']
print(f'Divergent users: {len(div)}   Convergent users: {len(cvg)}')

print('\n' + '='*25 + ' J1. MANIPULATION CHECK (Persona condition) ' + '='*25)
print('Assistant stance rates, Divergent vs Convergent personas (Welch t, between-subjects):')
for k in ['propose','critique','compare','commit','reframe']:
    c = f'a_{k}__Persona'
    a = div[c].astype(float).dropna(); b = cvg[c].astype(float).dropna()
    t,p = stats.ttest_ind(a,b, equal_var=False)
    star = ' *' if p<0.05 else ''
    print(f'  a_{k:10s}  D={a.mean():.3f}  C={b.mean():.3f}  D-C={(a.mean()-b.mean()):+.3f}  t={t:+.2f}  p={p:.4g}{star}')
print('\nUser stance rates, Divergent users vs Convergent users (Persona condition):')
for k in ['propose','critique','compare','commit','reframe','question']:
    c = f'u_{k}__Persona'
    a = div[c].astype(float).dropna(); b = cvg[c].astype(float).dropna()
    t,p = stats.ttest_ind(a,b, equal_var=False)
    star = ' *' if p<0.05 else ''
    print(f'  u_{k:10s}  D={a.mean():.3f}  C={b.mean():.3f}  D-C={(a.mean()-b.mean()):+.3f}  t={t:+.2f}  p={p:.4g}{star}')

print('\n' + '='*25 + ' J2. DELTA-OF-DELTA (Persona-GPT shift, D vs C) ' + '='*25)
metrics = ['novelty_all','novelty_user','novelty_ast','surp_all','surp_user','surp_ast',
           'u_propose','u_critique','u_commit','u_reframe','a_propose','a_critique','a_commit','a_reframe']
for m in metrics:
    col = f'd_{m}'
    a = div[col].astype(float).dropna(); b = cvg[col].astype(float).dropna()
    t,p = stats.ttest_ind(a,b, equal_var=False)
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2)) if len(a)+len(b)>2 else np.nan
    d = (a.mean()-b.mean())/sp if sp and sp>0 else np.nan
    star = ' *' if p<0.05 else ('.' if p<0.10 else '')
    print(f'  {col:20s}  D_delta={a.mean():+.3f}  C_delta={b.mean():+.3f}  (D-C)_delta={(a.mean()-b.mean()):+.3f}  d={d:+.2f}  t={t:+.2f}  p={p:.4g}{star}')

print('\n' + '='*25 + ' J3. SIGN ALIGNMENT with design intent ' + '='*25)
def frac(df, col, sgn):
    v = df[col].astype(float).dropna()
    if len(v)==0: return (0,0,np.nan)
    al = (np.sign(v)==sgn).sum()
    # binomial p
    pp = stats.binomtest(al, len(v), 0.5).pvalue
    return (al, len(v), al/len(v), pp)
intent_D = [('d_novelty_all',+1),('d_novelty_user',+1),('d_u_propose',+1),
            ('d_u_reframe',+1),('d_u_commit',-1),('d_surp_user',+1)]
intent_C = [('d_novelty_all',-1),('d_novelty_user',-1),('d_u_propose',-1),
            ('d_u_commit',+1),('d_u_critique',+1),('d_u_compare',+1)]
print('Divergent users (expected direction shown):')
for col,sgn in intent_D:
    al,n,f,pp = frac(div, col, sgn)
    star = ' *' if pp<0.05 else ''
    print(f'  {col:20s}  exp sign={sgn:+d}   {al:3d}/{n:3d} = {f:.2%}   binom p={pp:.3g}{star}')
print('Convergent users (expected direction shown):')
for col,sgn in intent_C:
    al,n,f,pp = frac(cvg, col, sgn)
    star = ' *' if pp<0.05 else ''
    print(f'  {col:20s}  exp sign={sgn:+d}   {al:3d}/{n:3d} = {f:.2%}   binom p={pp:.3g}{star}')

print('\n' + '='*25 + ' J4. Novelty shift explained by stance shifts? ' + '='*25)
for name, sub in [('Divergent', div), ('Convergent', cvg)]:
    print(f'\n  -- {name} (n={len(sub)}) --')
    for predictor in ['d_u_propose','d_u_reframe','d_u_commit','d_u_critique','d_a_propose','d_a_reframe']:
        x = sub[predictor].astype(float); y = sub['d_novelty_all'].astype(float)
        m = (~x.isna())&(~y.isna())
        if m.sum()<10: continue
        rho,pv = stats.spearmanr(x[m], y[m])
        star = ' *' if pv<0.05 else ''
        print(f'    d_novelty_all ~ {predictor:18s}   rho={rho:+.3f}  p={pv:.4g}{star}')

print('\n' + '='*25 + ' J5. Perception by family ' + '='*25)
users = pd.read_excel(os.path.join(OUT,'users_translated.xlsx'), sheet_name='corrected_users')
users.columns = [c.strip() for c in users.columns]
def gpt_round(row):
    r1 = str(row.get('Persona round 1','')).lower(); r2 = str(row.get('Persona round 2','')).lower()
    if 'gpt' in r1: return 1
    if 'gpt' in r2: return 2
    return np.nan
users['gpt_round'] = users.apply(gpt_round, axis=1)
def mk(df, a, b):
    g = np.where(df['gpt_round']==1, df[a], np.where(df['gpt_round']==2, df[b], np.nan))
    p = np.where(df['gpt_round']==1, df[b], np.where(df['gpt_round']==2, df[a], np.nan))
    return pd.Series(g, index=df.index), pd.Series(p, index=df.index)
users['cr_gpt'], users['cr_per'] = mk(users,'Creativity assistant #1','Creativity assistant #2')
users['ow_gpt'], users['ow_per'] = mk(users,'Ownership #1','Ownership #2')
users['cr_diff'] = users['cr_per'].astype(float) - users['cr_gpt'].astype(float)
users['ow_diff'] = users['ow_per'].astype(float) - users['ow_gpt'].astype(float)
users = users.merge(up[['User_id','family']], left_on='id', right_on='User_id').drop(columns=['User_id'])
udc = users[users.family.isin(['Divergent','Convergent'])]
for col,lbl in [('cr_diff','Creativity'),('ow_diff','Ownership')]:
    for fam in ['Divergent','Convergent']:
        v = udc[udc.family==fam][col].astype(float).dropna()
        t,p = stats.ttest_1samp(v, 0)
        star = ' *' if p<0.05 else ''
        print(f'  {lbl:11s} Delta  [{fam:10s}]  n={len(v)}  mean={v.mean():+.3f}  t(vs 0)={t:+.2f}  p={p:.4g}{star}')
    a = udc[udc.family=='Divergent'][col].astype(float).dropna()
    b = udc[udc.family=='Convergent'][col].astype(float).dropna()
    t,p = stats.ttest_ind(a, b, equal_var=False)
    star = ' *' if p<0.05 else ''
    print(f'  -> D vs C (between-subjects) {lbl} Delta: t={t:+.2f}  p={p:.4g}{star}')

print('\n' + '='*25 + ' J6. Big-5 moderation within each family ' + '='*25)
pers_cols = ['Extraversion','Agreeableness','Conscientiousness','Negative Emotionality','Open-Mindedness']
for src in ['cr_diff','ow_diff'] + pers_cols:
    if src in users.columns:
        wide[src] = wide['user'].map(users.set_index('id')[src])
DC = wide[wide.family.isin(['Divergent','Convergent'])]
for m in ['d_novelty_all','d_surp_all','d_surp_user','cr_diff','ow_diff']:
    print(f'\n  -- {m} vs Big-5, split by family --')
    for fam in ['Divergent','Convergent']:
        sub = DC[DC.family==fam]
        for t in pers_cols:
            x = sub[m].astype(float); y = sub[t].astype(float)
            mk2 = (~x.isna())&(~y.isna())
            if mk2.sum()<15: continue
            rho,pv = stats.spearmanr(x[mk2], y[mk2])
            if pv<0.15:
                star = ' *' if pv<0.05 else '.'
                print(f'    [{fam:10s}]  {t:22s}  rho={rho:+.3f}  p={pv:.4g}{star}  n={mk2.sum()}')

print('\n' + '='*25 + ' J7. Challenge-type moderation within family ' + '='*25)
conv['challenge'] = logs.groupby('conversation_id')['Corrected Challenge type'].first().reindex(conv['conversation_id']).values
dc_conv = conv[conv.family.isin(['Divergent','Convergent'])]
print('Cell counts (family x challenge x condition):')
print(dc_conv.groupby(['family','challenge','condition']).size().unstack().fillna(0).astype(int))
for fam in ['Divergent','Convergent']:
    print(f'\n  -- {fam}: novelty_all by challenge x condition --')
    sub = dc_conv[dc_conv.family==fam]
    g = sub.groupby(['challenge','condition'])['novelty_all'].agg(['count','mean','std']).round(3)
    print(g)
