"""Standalone LAYER L runner — does not need cleaned logs, only
analysis_out/production/participant_originality.csv + users_translated.xlsx."""
import os, sys, numpy as np, pandas as pd
from scipy import stats
try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT,'analysis_out')

orig = pd.read_csv(os.path.join(OUT,'production','participant_originality.csv'))
print(f'loaded {len(orig)} participant-round originality rows')

# rebuild user_persona from raw logs (same pattern as analyze.py)
logs = pd.read_csv(os.path.join(ROOT,'Experiment1_logs.csv'))
up = (logs.groupby('User_id')['Persona_type']
      .apply(lambda s: [x for x in s.unique() if x!='GPT'][0])
      .reset_index().rename(columns={'Persona_type':'persona_cond'}))
fm = {'Divergent':'Divergent','Convergent':'Convergent','strictly rational':'Rational','bounded rationality':'BoundedRational'}
up['family'] = up['persona_cond'].map(fm)

# users + Big-5 + perception deltas
users = pd.read_excel(os.path.join(OUT,'users_translated.xlsx'), sheet_name='corrected_users')
users.columns = [c.strip() for c in users.columns]
def gpt_round(row):
    r1 = str(row.get('Persona round 1','')).lower(); r2 = str(row.get('Persona round 2','')).lower()
    if 'gpt' in r1: return 1
    if 'gpt' in r2: return 2
    return np.nan
users['gpt_round'] = users.apply(gpt_round, axis=1)
def mk(df,a,b):
    g = np.where(df['gpt_round']==1,df[a],np.where(df['gpt_round']==2,df[b],np.nan))
    p = np.where(df['gpt_round']==1,df[b],np.where(df['gpt_round']==2,df[a],np.nan))
    return pd.Series(g,index=df.index), pd.Series(p,index=df.index)
users['cr_gpt'], users['cr_per'] = mk(users,'Creativity assistant #1','Creativity assistant #2')
users['ow_gpt'], users['ow_per'] = mk(users,'Ownership #1','Ownership #2')
users['cr_diff'] = users['cr_per'].astype(float)-users['cr_gpt'].astype(float)
users['ow_diff'] = users['ow_per'].astype(float)-users['ow_gpt'].astype(float)
pers_cols = ['Extraversion','Agreeableness','Conscientiousness','Negative Emotionality','Open-Mindedness']

print('\n'+'='*20,'DESCRIPTIVE by condition','='*20)
for col in ['n_ideas','orig_same','orig_all','orig_cross']:
    g = orig.groupby('condition')[col].agg(['count','mean','std']).round(4)
    print(f'\n{col}:')
    print(g.to_string())

print('\n'+'='*20,'BETWEEN-SUBJECTS Welch t (Exp 2 methodology)','='*20)
def welch_g(a, b):
    a = pd.to_numeric(a, errors='coerce').dropna()
    b = pd.to_numeric(b, errors='coerce').dropna()
    if len(a)<5 or len(b)<5: return None
    t,p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    d = (a.mean()-b.mean())/sp if sp>0 else np.nan
    J = 1 - 3/(4*(len(a)+len(b))-9)
    g_es = d*J if not np.isnan(d) else np.nan
    return dict(n_P=len(a), n_G=len(b), mean_P=a.mean(), mean_G=b.mean(),
                diff=a.mean()-b.mean(), t=t, p=p, cohen_d=d, hedges_g=g_es)

for col in ['n_ideas','orig_same','orig_all','orig_cross']:
    r = welch_g(orig.loc[orig.condition=='Persona', col],
                orig.loc[orig.condition=='GPT', col])
    if r:
        star = ' *' if r['p']<0.05 else ('.' if r['p']<0.10 else '')
        print(f"  {col:10s}  P={r['mean_P']:.4f}  G={r['mean_G']:.4f}  diff={r['diff']:+.4f}  t={r['t']:+.2f}  p={r['p']:.4g}  g={r['hedges_g']:+.3f}{star}")

print('\n'+'='*20,'WITHIN-SUBJECT paired (Persona vs GPT per user)','='*20)
wide = orig.pivot_table(index='user', columns='condition',
                        values=['n_ideas','orig_same','orig_all','orig_cross'],
                        aggfunc='first')
wide.columns = [f'{m}__{c}' for m,c in wide.columns]
wide = wide.reset_index().merge(up[['User_id','family']], left_on='user', right_on='User_id', how='left').drop(columns=['User_id'])
print(f'  participants with both rounds extracted: {wide[["n_ideas__GPT","n_ideas__Persona"]].notna().all(axis=1).sum()}')

for col in ['n_ideas','orig_same','orig_all','orig_cross']:
    cp, cg = f'{col}__Persona', f'{col}__GPT'
    a = wide[cp].astype(float); b = wide[cg].astype(float)
    m = (~a.isna())&(~b.isna())
    if m.sum()<5: continue
    a, b = a[m], b[m]
    d = a - b
    t,p = stats.ttest_rel(a,b)
    try: w,pw = stats.wilcoxon(a,b)
    except: w,pw = np.nan,np.nan
    dz = d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else np.nan
    star = ' *' if p<0.05 else ('.' if p<0.10 else '')
    print(f"  {col:10s}  n={int(m.sum()):3d}  P={a.mean():.4f}  G={b.mean():.4f}  diff={d.mean():+.4f}  t={t:+.2f}  p={p:.4g}  dz={dz:+.3f}  W p={pw:.4g}{star}")

print('\n'+'='*20,'BY PERSONA FAMILY (within-subject d)','='*20)
for col in ['n_ideas','orig_same','orig_all','orig_cross']:
    cp, cg = f'{col}__Persona', f'{col}__GPT'
    if cp not in wide.columns: continue
    wide[f'd_{col}'] = wide[cp].astype(float)-wide[cg].astype(float)
    print(f'\n  {col}: one-sample t-test of (Persona-GPT) against 0, by family')
    for fam in ['Divergent','Convergent','Rational','BoundedRational']:
        v = wide[wide.family==fam][f'd_{col}'].dropna()
        if len(v)<5:
            print(f'    {fam:16s}  n={len(v):2d}  (too few)')
            continue
        t,p = stats.ttest_1samp(v, 0)
        star = ' *' if p<0.05 else ('.' if p<0.10 else '')
        print(f'    {fam:16s}  n={len(v):3d}  mean_diff={v.mean():+.4f}  t={t:+.2f}  p={p:.4g}{star}')

print('\n'+'='*20,'Big-5 × doriginality (Spearman, p<.10 shown)','='*20)
users_slim = users.set_index('id')
for p in pers_cols:
    if p in users_slim.columns:
        wide[p] = wide['user'].map(users_slim[p])
rows = []
for col in ['n_ideas','orig_same','orig_all','orig_cross']:
    dcol = f'd_{col}'
    if dcol not in wide.columns: continue
    for t in pers_cols:
        if t not in wide.columns: continue
        x = wide[dcol].astype(float); y = wide[t].astype(float)
        m = (~x.isna())&(~y.isna())
        if m.sum()<15: continue
        rho,pv = stats.spearmanr(x[m], y[m])
        if pv < 0.10:
            rows.append((dcol, t, int(m.sum()), rho, pv))
for row in sorted(rows, key=lambda r: r[4]):
    star = ' *' if row[4]<0.05 else '.'
    print(f"  {row[0]:16s}  vs  {row[1]:22s}  n={row[2]:3d}  rho={row[3]:+.3f}  p={row[4]:.4g}{star}")

print('\n'+'='*20,'Perception bridge: doriginality × dcreativity/downership','='*20)
for src in ['cr_diff','ow_diff']:
    if src in users_slim.columns:
        wide[src] = wide['user'].map(users_slim[src])
for col in ['n_ideas','orig_same','orig_all','orig_cross']:
    dcol = f'd_{col}'
    if dcol not in wide.columns: continue
    for tgt, lbl in [('cr_diff','d creativity'),('ow_diff','d ownership')]:
        if tgt not in wide.columns: continue
        x = wide[dcol].astype(float); y = wide[tgt].astype(float)
        m = (~x.isna())&(~y.isna())
        if m.sum()<15: continue
        rho,pv = stats.spearmanr(x[m], y[m])
        star = ' *' if pv<0.05 else ('.' if pv<0.10 else '')
        print(f"  {dcol:14s}  ×  {lbl:18s}  n={int(m.sum()):3d}  rho={rho:+.3f}  p={pv:.4g}{star}")

# save wide table for downstream figure creation
wide.to_csv(os.path.join(OUT,'production','_wide_originality.csv'), index=False)
print(f'\nsaved wide table: analysis_out/production/_wide_originality.csv')
