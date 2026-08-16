"""
Speaker-separated semantic-trajectory exploration.
K1: user-only consecutive distance (skipping persona/assistant turns), binned by user-turn position.
K2: persona/assistant-only consecutive distance (skipping user turns).
K3: within-user cumulative drift from each speaker's first message.
K4: within-speaker pairwise-distance spread (ideation breadth).
K5: cross-speaker alignment (distance from user reply to the persona turn it follows).
K6: centroid separation between persona-type user-message clouds.
"""
import os, re, numpy as np, pandas as pd
from scipy import stats
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)

ROOT = '.'
OUT  = 'analysis_out'

logs = pd.read_csv('Experiment1_logs.csv').sort_values(['conversation_id','message_id']).reset_index(drop=True)
E = np.load(os.path.join(OUT,'msg_embeddings.npy'))
assert len(E) == len(logs)

# sanity spot check: verify label alignment by printing a few
print('=== SPOT-CHECK: 5 user and 5 assistant messages ===')
for src in ['user','assistant']:
    idx = logs[logs.message_src==src].index[:3]
    for i in idx:
        txt = str(logs.iloc[i]['message'])[:80].replace('\n',' ')
        norm = float(np.linalg.norm(E[i]))
        print(f'  row {i:5d}  src={src:9s}  ||E||={norm:.3f}  msg="{txt}"')

# families
up = (logs.groupby('User_id')['Persona_type']
      .apply(lambda s: [x for x in s.unique() if x!='GPT'][0])
      .reset_index().rename(columns={'Persona_type':'persona_cond'}))
fm = {'Divergent':'Divergent','Convergent':'Convergent','strictly rational':'Rational','bounded rationality':'BoundedRational'}
up['family'] = up['persona_cond'].map(fm)

conv = logs.groupby('conversation_id').agg(user=('User_id','first'), persona_type=('Persona_type','first')).reset_index()
conv['condition'] = np.where(conv['persona_type']=='GPT','GPT','Persona')
conv = conv.merge(up[['User_id','family']], left_on='user', right_on='User_id').drop(columns=['User_id'])
# mixed persona label: "GPT" if GPT condition else family
conv['persona_label'] = np.where(conv['condition']=='GPT','GPT', conv['family'])
logs = logs.merge(conv[['conversation_id','condition','family','persona_label']], on='conversation_id')

# =====================================================================
# K1. USER-ONLY consecutive distance (skip persona turns)
# =====================================================================
print('\n=== K1. User-only consecutive distance (within conversation, user->user skip) ===')
rows = []
for cid, g in logs.groupby('conversation_id', sort=False):
    u = g[g.message_src=='user'].sort_values('message_id')
    idx = u.index.to_numpy()
    for k in range(1, len(idx)):
        d = 1.0 - float(E[idx[k]] @ E[idx[k-1]])
        rows.append(dict(conversation_id=cid, user_turn_k=k,
                         persona_label=u.iloc[k]['persona_label'],
                         condition=u.iloc[k]['condition'],
                         family=u.iloc[k]['family'],
                         user_turn_frac=(k)/max(1,len(idx)-1),
                         dist=d))
u_tr = pd.DataFrame(rows)
print(f'user-only transitions: {len(u_tr)}  (expected ~n_user - n_conv = {(logs.message_src=="user").sum()-logs["conversation_id"].nunique()})')

# overall means by persona_label
print('\n  mean user-user consec distance by persona_label:')
print(u_tr.groupby('persona_label')['dist'].agg(['count','mean','std','sem']).round(4))

# 5 bins (quintiles of user_turn_frac) by persona_label
u_tr['bin'] = pd.cut(u_tr['user_turn_frac'], bins=np.linspace(0,1,6), labels=False, include_lowest=True)
print('\n  mean user-user distance by bin x persona_label:')
print(u_tr.groupby(['bin','persona_label'])['dist'].mean().unstack('persona_label').round(3))

# slope of distance vs turn fraction (per conversation) -- is the user opening up over time?
print('\n  per-conversation slope of user-user distance across turn_frac:')
slope_rows = []
for cid, g in u_tr.groupby('conversation_id'):
    if len(g)<3: continue
    s, *_ = stats.linregress(g['user_turn_frac'].values, g['dist'].values)
    slope_rows.append(dict(conversation_id=cid, slope=s,
                           persona_label=g['persona_label'].iloc[0], condition=g['condition'].iloc[0]))
sl = pd.DataFrame(slope_rows).merge(conv[['conversation_id','user']], on='conversation_id')
print(sl.groupby('persona_label')['slope'].agg(['count','mean','sem']).round(4))
# within-user paired: does Persona vs GPT give different slopes?
w = sl.pivot_table(index='user', columns='condition', values='slope', aggfunc='first').dropna()
t,p = stats.ttest_rel(w['Persona'], w['GPT'])
print(f'  paired slope Persona vs GPT: n={len(w)}  mean_P={w["Persona"].mean():+.4f}  mean_G={w["GPT"].mean():+.4f}  diff={w["Persona"].mean()-w["GPT"].mean():+.4f}  t={t:+.2f}  p={p:.4g}')

# between-persona slope comparison (within Persona condition)
for fam in ['Divergent','Convergent','Rational','BoundedRational']:
    a = sl[sl['persona_label']==fam]['slope']
    b = sl[sl['persona_label']=='GPT']['slope']
    t,p = stats.ttest_ind(a, b, equal_var=False)
    print(f'  {fam:16s}  n={len(a):3d}  slope={a.mean():+.4f}  vs GPT(n={len(b)}) slope={b.mean():+.4f}  t={t:+.2f}  p={p:.4g}')

# =====================================================================
# K2. PERSONA/ASSISTANT-ONLY consecutive distance
# =====================================================================
print('\n=== K2. Assistant-only consecutive distance ===')
rows = []
for cid, g in logs.groupby('conversation_id', sort=False):
    a = g[g.message_src=='assistant'].sort_values('message_id')
    idx = a.index.to_numpy()
    for k in range(1, len(idx)):
        d = 1.0 - float(E[idx[k]] @ E[idx[k-1]])
        rows.append(dict(conversation_id=cid, asst_turn_k=k,
                         persona_label=a.iloc[k]['persona_label'],
                         condition=a.iloc[k]['condition'],
                         family=a.iloc[k]['family'],
                         asst_turn_frac=(k)/max(1,len(idx)-1),
                         dist=d))
a_tr = pd.DataFrame(rows)
print(f'asst-only transitions: {len(a_tr)}')
print('\n  mean asst-asst consec distance by persona_label:')
print(a_tr.groupby('persona_label')['dist'].agg(['count','mean','std','sem']).round(4))

a_tr['bin'] = pd.cut(a_tr['asst_turn_frac'], bins=np.linspace(0,1,6), labels=False, include_lowest=True)
print('\n  mean asst-asst distance by bin x persona_label:')
print(a_tr.groupby(['bin','persona_label'])['dist'].mean().unstack('persona_label').round(3))

# =====================================================================
# K3. Within-user cumulative DRIFT from each speaker's first message
# =====================================================================
print('\n=== K3. Drift from first same-speaker message ===')
rows = []
for cid, g in logs.groupby('conversation_id', sort=False):
    for src in ['user','assistant']:
        sub = g[g.message_src==src].sort_values('message_id')
        idx = sub.index.to_numpy()
        if len(idx) < 2: continue
        anchor = E[idx[0]]
        for k in range(1, len(idx)):
            d = 1.0 - float(E[idx[k]] @ anchor)
            rows.append(dict(conversation_id=cid, src=src,
                             persona_label=sub.iloc[k]['persona_label'],
                             condition=sub.iloc[k]['condition'],
                             frac=k/max(1,len(idx)-1), drift=d))
dr = pd.DataFrame(rows)
print('\n  mean drift from first same-speaker message, by src x persona_label:')
print(dr.groupby(['src','persona_label'])['drift'].mean().unstack('persona_label').round(3))

# drift slope per conversation
print('\n  per-conversation slope of drift vs frac (user-side only):')
slope_rows = []
for cid, g in dr[dr.src=='user'].groupby('conversation_id'):
    if len(g)<3: continue
    s,*_ = stats.linregress(g['frac'].values, g['drift'].values)
    slope_rows.append(dict(conversation_id=cid, slope=s,
                           persona_label=g['persona_label'].iloc[0], condition=g['condition'].iloc[0]))
sl2 = pd.DataFrame(slope_rows).merge(conv[['conversation_id','user']], on='conversation_id')
print(sl2.groupby('persona_label')['slope'].agg(['count','mean','sem']).round(4))
w2 = sl2.pivot_table(index='user', columns='condition', values='slope', aggfunc='first').dropna()
t,p = stats.ttest_rel(w2['Persona'], w2['GPT'])
print(f'  paired drift-slope Persona vs GPT: n={len(w2)}  P={w2["Persona"].mean():+.4f}  G={w2["GPT"].mean():+.4f}  t={t:+.2f}  p={p:.4g}')

# =====================================================================
# K4. Pairwise ideation breadth per user per conversation
# =====================================================================
print('\n=== K4. User-message ideation breadth (mean pairwise cosine distance within conv) ===')
rows = []
for cid, g in logs.groupby('conversation_id', sort=False):
    for src in ['user','assistant']:
        sub = g[g.message_src==src].sort_values('message_id')
        idx = sub.index.to_numpy()
        if len(idx) < 3: continue
        V = E[idx]
        sim = V @ V.T
        upper = sim[np.triu_indices(len(sim), k=1)]
        mean_pair_dist = 1 - float(upper.mean())
        rows.append(dict(conversation_id=cid, src=src, breadth=mean_pair_dist))
br = pd.DataFrame(rows).merge(conv[['conversation_id','user','condition','persona_label','family']], on='conversation_id')
print('\n  mean ideation breadth by src x persona_label:')
print(br.groupby(['src','persona_label'])['breadth'].agg(['count','mean','sem']).round(4))
# paired Persona vs GPT by src
for src in ['user','assistant']:
    w = br[br.src==src].pivot_table(index='user', columns='condition', values='breadth', aggfunc='first').dropna()
    t,p = stats.ttest_rel(w['Persona'], w['GPT'])
    print(f'  paired breadth Persona vs GPT [{src:9s}]: n={len(w)}  P={w["Persona"].mean():.3f}  G={w["GPT"].mean():.3f}  diff={(w["Persona"]-w["GPT"]).mean():+.3f}  t={t:+.2f}  p={p:.4g}')
# between-persona (Persona condition only) — user side
pmask = br[(br.src=='user')&(br.condition=='Persona')]
print('\n  breadth across persona families (user-only, Persona condition, ANOVA):')
groups = [pmask[pmask['persona_label']==f]['breadth'].values for f in ['Divergent','Convergent','Rational','BoundedRational']]
F, p = stats.f_oneway(*groups)
print(f'    ANOVA F={F:.2f}  p={p:.4g}')
for f in ['Divergent','Convergent','Rational','BoundedRational']:
    v = pmask[pmask['persona_label']==f]['breadth']
    print(f'    {f:16s}  n={len(v):3d}  mean={v.mean():.3f}  sem={v.sem():.3f}')

# =====================================================================
# K5. Cross-speaker alignment (how close is user's reply to preceding persona turn?)
# =====================================================================
print('\n=== K5. User-reply alignment to preceding assistant turn (user reading the partner) ===')
rows = []
for cid, g in logs.groupby('conversation_id', sort=False):
    g = g.sort_values('message_id').reset_index(drop=True)
    for i in range(1, len(g)):
        if g.iloc[i]['message_src']=='user' and g.iloc[i-1]['message_src']=='assistant':
            # indices into E
            gidx_this = g.index[i]  # re-index now local; need original
    # redo using original index
    gg = logs[logs.conversation_id==cid].sort_values('message_id')
    ids = gg.index.to_numpy()
    srcs = gg['message_src'].to_numpy()
    for k in range(1, len(ids)):
        if srcs[k]=='user' and srcs[k-1]=='assistant':
            d = 1.0 - float(E[ids[k]] @ E[ids[k-1]])
            rows.append(dict(conversation_id=cid, user_turn_frac=k/max(1,len(ids)-1),
                             persona_label=gg.iloc[k]['persona_label'],
                             condition=gg.iloc[k]['condition'],
                             family=gg.iloc[k]['family'],
                             dist=d))
xs = pd.DataFrame(rows)
print('\n  mean user-to-preceding-asst distance by persona_label:')
print(xs.groupby('persona_label')['dist'].agg(['count','mean','sem']).round(4))

# =====================================================================
# K6. Centroid separation between user-message clouds across persona types
# =====================================================================
print('\n=== K6. Centroid separation between user-message clouds (by persona_label) ===')
user_mask = logs.message_src=='user'
for lbl, sub in logs[user_mask].groupby('persona_label'):
    V = E[sub.index.to_numpy()]
    c = V.mean(axis=0); c = c/np.linalg.norm(c)
    # store
    globals().setdefault('_centroids',{})[lbl] = c
cents = globals()['_centroids']
labels_order = ['GPT','Divergent','Convergent','Rational','BoundedRational']
present = [l for l in labels_order if l in cents]
print('  pairwise cosine distance between centroids:')
print('         ' + ' '.join(f'{l:>10s}' for l in present))
for l1 in present:
    row = f'{l1:10s}'
    for l2 in present:
        d = 1 - float(cents[l1] @ cents[l2])
        row += f' {d:10.4f}'
    print(row)

# silhouette-lite: for each user message, nearest-centroid identity rate
import numpy.linalg as la
um = logs[user_mask]
V = E[um.index.to_numpy()]
true = um['persona_label'].to_numpy()
cent_M = np.stack([cents[l] for l in present], axis=0)
sims = V @ cent_M.T
pred = np.array(present)[np.argmax(sims, axis=1)]
acc = (pred==true).mean()
print(f'\n  naive nearest-centroid accuracy for persona_label: {acc:.3f}  (chance={1/len(present):.3f})')
print('  confusion (row=true, col=pred):')
print(pd.crosstab(pd.Series(true,name='true'), pd.Series(pred,name='pred')).loc[present, present])

# save for downstream UMAP
np.save(os.path.join(OUT,'_user_embeddings.npy'), V)
um[['message_id','conversation_id','User_id','persona_label','condition','family']].to_csv(os.path.join(OUT,'_user_embeddings_meta.csv'), index=False)
print('\nsaved analysis_out/_user_embeddings.npy and _user_embeddings_meta.csv for UMAP step.')
