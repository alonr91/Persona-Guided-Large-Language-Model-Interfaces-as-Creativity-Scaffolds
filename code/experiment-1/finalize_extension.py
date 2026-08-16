"""
Final extension analyses using SBERT embeddings + Claude-propagated stance scores.

- Portfolio-level distinctiveness (SBERT, replacing TF-IDF)
- Fixation / drift index from first AI proposal
- Conversation-level stance aggregates (from propagated predictions)
- Persona fidelity manipulation check
- Paired tests on LLM-coded metrics
- Figures for the extension
"""
import os, sys, warnings, json
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT,'analysis_out')
FIG  = os.path.join(ROOT,'figures')

logs = pd.read_csv(os.path.join(ROOT,'Experiment1_logs_cleaned_keepable_paired_translated.csv'))
logs = logs.sort_values(['conversation_id','message_id']).reset_index(drop=True)
logs['turn_idx'] = logs.groupby('conversation_id').cumcount()
logs['turn_frac'] = logs['turn_idx'] / logs.groupby('conversation_id')['turn_idx'].transform('max').replace(0,1)
fam_map = {'Divergent':'Divergent','Convergent':'Convergent',
           'strictly rational':'Rational','bounded rationality':'BoundedRational',
           'GPT':'GPT'}
logs['family'] = logs['Persona_type'].map(fam_map)
logs['condition'] = np.where(logs['Persona_type']=='GPT','GPT','Persona')

users = pd.read_excel(os.path.join(OUT,'users_translated.xlsx'),
                      sheet_name='corrected_users')
users.columns = [c.strip() for c in users.columns]

E = np.load(os.path.join(OUT,'msg_embeddings.npy'))
stance = pd.read_csv(os.path.join(OUT,'full_stance_predictions.csv'))

# conversation meta
conv_meta = (logs.groupby('conversation_id')
             .agg(user=('User_id','first'),
                  persona_type=('Persona_type','first'),
                  challenge=('Corrected Challenge type','first'),
                  family=('family','first'),
                  condition=('condition','first'))
             .reset_index())

# ===================================================================
# 1. SBERT portfolio distinctiveness (conv-level, replacing TF-IDF)
# ===================================================================
print('=== SBERT portfolio distinctiveness ===')
# centroid per conversation on USER messages
user_mask = logs['message_src']=='user'
conv_ids = conv_meta['conversation_id'].values
centroids = []
for cid in conv_ids:
    idx = np.where((logs['conversation_id']==cid) & user_mask)[0]
    if len(idx)==0:
        centroids.append(np.zeros(E.shape[1]))
    else:
        v = E[idx].mean(0); v = v/(np.linalg.norm(v)+1e-9)
        centroids.append(v)
C = np.vstack(centroids)

# within-challenge distinctiveness (1 - mean cosine sim to same-challenge peers)
conv_meta['sbert_distinctiveness'] = np.nan
for ch, sub in conv_meta.groupby('challenge'):
    ids = sub['conversation_id'].tolist()
    pos = [list(conv_ids).index(i) for i in ids]
    Csub = C[pos]
    S = cosine_similarity(Csub); np.fill_diagonal(S,np.nan)
    d = 1 - np.nanmean(S, axis=1)
    for i,cid in enumerate(ids):
        conv_meta.loc[conv_meta.conversation_id==cid,'sbert_distinctiveness'] = d[i]

# within-conv user-message breadth
conv_meta['sbert_breadth'] = np.nan
conv_meta['sbert_redundancy'] = np.nan
for cid in conv_ids:
    idx = np.where((logs['conversation_id']==cid) & user_mask)[0]
    if len(idx) < 2: continue
    S = cosine_similarity(E[idx]); np.fill_diagonal(S,np.nan)
    conv_meta.loc[conv_meta.conversation_id==cid,'sbert_breadth'] = 1 - np.nanmean(S)
    conv_meta.loc[conv_meta.conversation_id==cid,'sbert_redundancy'] = np.nanmax(S)

# ===================================================================
# 2. Fixation index: similarity to first AI proposal turn
# ===================================================================
print('=== Fixation index ===')
fix_rows=[]
for cid in conv_ids:
    g = logs[logs.conversation_id==cid].sort_values('message_id')
    a = g[g.message_src=='assistant']
    u = g[g.message_src=='user']
    if len(u) < 3 or len(a) < 1: continue
    # anchor = first assistant turn with propose_new_idea >= 1 from stance predictions
    a_mids = a['message_id'].tolist()
    a_stance = stance[stance.message_id.isin(a_mids)].sort_values('message_id')
    cand = a_stance[a_stance.prop >= 1.0]
    if len(cand)==0:
        anchor_mid = a_mids[0]
    else:
        anchor_mid = cand.iloc[0]['message_id']
    anchor_row = int(np.where(logs['message_id']==anchor_mid)[0][0])
    user_rows = np.where((logs['conversation_id']==cid) & user_mask)[0]
    av = E[anchor_row:anchor_row+1]
    sims = cosine_similarity(av, E[user_rows]).ravel()
    early = sims[:max(1,len(sims)//3)].mean()
    late  = sims[-max(1,len(sims)//3):].mean()
    fix_rows.append(dict(conversation_id=cid,
        fixation_index=sims.mean(),
        anchor_sim_early=early, anchor_sim_late=late,
        drift_trajectory=early-late,  # >0 = user moved away from anchor
        anchor_mid=anchor_mid))
fx = pd.DataFrame(fix_rows)
conv_meta = conv_meta.merge(fx, on='conversation_id', how='left')

# ===================================================================
# 3. Conversation-level stance aggregates from propagated predictions
# ===================================================================
print('=== Stance aggregates ===')
stance_u = stance[stance.message_src=='user']
stance_a = stance[stance.message_src=='assistant']
agg_u = stance_u.groupby('conversation_id')[['exp','con','cri','cer','com','ref','prop']].mean().add_prefix('u_').reset_index()
agg_a = stance_a.groupby('conversation_id')[['exp','con','cri','cer','com','ref','prop']].mean().add_prefix('a_').reset_index()
# tone distribution (assistant side)
tone_a = pd.crosstab(stance_a['conversation_id'], stance_a['tone_pred'], normalize='index').add_prefix('a_tone_')
tone_a = tone_a.reset_index()
qtype_u = pd.crosstab(stance_u['conversation_id'], stance_u['qtype_pred'], normalize='index').add_prefix('u_qtype_')
qtype_u = qtype_u.reset_index()

conv_meta = conv_meta.merge(agg_u, on='conversation_id', how='left')
conv_meta = conv_meta.merge(agg_a, on='conversation_id', how='left')
conv_meta = conv_meta.merge(tone_a, on='conversation_id', how='left')
conv_meta = conv_meta.merge(qtype_u, on='conversation_id', how='left')
conv_meta.to_csv(os.path.join(OUT,'extension_conv_master.csv'), index=False)

# ===================================================================
# 4. Manipulation check by family (persona fidelity; §2.8)
# ===================================================================
print('\n=== MANIPULATION CHECK ===')
fam_cols = ['a_exp','a_con','a_cri','a_cer','a_com','a_ref','a_prop',
            'u_exp','u_con','u_cri','u_cer','u_com','u_ref','u_prop']
mc = conv_meta[conv_meta.condition=='Persona'].groupby('family')[fam_cols].mean()
mc.loc['GPT_baseline'] = conv_meta[conv_meta.condition=='GPT'][fam_cols].mean()
mc.to_csv(os.path.join(OUT,'manipulation_check.csv'))
print(mc.round(3).to_string())

# tone and qtype by family
tone_cols = [c for c in conv_meta.columns if c.startswith('a_tone_')]
qtype_cols = [c for c in conv_meta.columns if c.startswith('u_qtype_')]
mc_tone = conv_meta[conv_meta.condition=='Persona'].groupby('family')[tone_cols].mean()
mc_tone.loc['GPT_baseline'] = conv_meta[conv_meta.condition=='GPT'][tone_cols].mean()
print('\n--- Assistant critique tone by family ---')
print(mc_tone.round(3).to_string())
mc_qt = conv_meta[conv_meta.condition=='Persona'].groupby('family')[qtype_cols].mean()
mc_qt.loc['GPT_baseline'] = conv_meta[conv_meta.condition=='GPT'][qtype_cols].mean()
print('\n--- User question type by family ---')
print(mc_qt.round(3).to_string())

# ===================================================================
# 5. Paired tests — Persona vs GPT on LLM-coded + SBERT metrics
# ===================================================================
print('\n=== PAIRED TESTS: Extension metrics ===')
def paired(a,b,nm):
    m=(~pd.isna(a))&(~pd.isna(b)); a,b=a[m],b[m]
    if len(a)<5: return None
    d=a-b; t,p=stats.ttest_rel(a,b)
    dz = d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else np.nan
    try:
        w, pw = stats.wilcoxon(a,b) if (d!=0).any() else (np.nan,np.nan)
    except Exception:
        w,pw=np.nan,np.nan
    return dict(metric=nm,n=len(a),mean_P=a.mean(),mean_G=b.mean(),
                diff=d.mean(),t=t,p=p,dz=dz,p_wilcoxon=pw)

test_cols = ['sbert_distinctiveness','sbert_breadth','sbert_redundancy',
             'fixation_index','anchor_sim_early','anchor_sim_late','drift_trajectory'
             ] + fam_cols + tone_cols + qtype_cols
wide = conv_meta.pivot_table(index='user', columns='condition', values=test_cols, aggfunc='first')
wide.columns = [f'{a}__{b}' for a,b in wide.columns]
rows=[]
for c in test_cols:
    cg=f'{c}__GPT'; cp=f'{c}__Persona'
    if cg in wide.columns and cp in wide.columns:
        r = paired(wide[cp].astype(float), wide[cg].astype(float), c)
        if r: rows.append(r)
pair_df = pd.DataFrame(rows).sort_values('p')
pair_df.to_csv(os.path.join(OUT,'extension_paired.csv'), index=False)
print(pair_df.to_string(index=False))

# ===================================================================
# 6. Cross-layer: LLM-coded behavior vs subjective ratings (Story 3)
# ===================================================================
print('\n=== Perception-behavior: behavioral Δ vs subjective Δ ===')
# reconstruct subjective diffs
def round_of_gpt(row):
    r1 = str(row.get('Persona round 1','')).strip().lower()
    r2 = str(row.get('Persona round 2','')).strip().lower()
    if 'gpt' in r1: return 1
    if 'gpt' in r2: return 2
    return np.nan
users['gpt_round'] = users.apply(round_of_gpt, axis=1)
def make_gp(df, r1c, r2c):
    gpt = np.where(df['gpt_round']==1, df[r1c], np.where(df['gpt_round']==2, df[r2c], np.nan))
    per = np.where(df['gpt_round']==1, df[r2c], np.where(df['gpt_round']==2, df[r1c], np.nan))
    return pd.Series(gpt, index=df.index), pd.Series(per, index=df.index)
users['creativity_gpt'], users['creativity_persona'] = make_gp(users,'Creativity assistant #1','Creativity assistant #2')
users['ownership_gpt'],  users['ownership_persona']  = make_gp(users,'Ownership #1','Ownership #2')
users['cr_diff'] = users['creativity_persona'].astype(float)-users['creativity_gpt'].astype(float)
users['ow_diff'] = users['ownership_persona'].astype(float)-users['ownership_gpt'].astype(float)

u_idx = users.set_index('id')
# behavioral Δs
corr_rows=[]
for c in ['a_exp','a_con','a_cri','a_cer','a_com','a_ref','a_prop',
          'u_exp','u_con','u_cer','u_ref',
          'sbert_distinctiveness','sbert_breadth',
          'fixation_index','drift_trajectory']:
    cg=f'{c}__GPT'; cp=f'{c}__Persona'
    if cg not in wide.columns: continue
    w = wide[[cg,cp]].dropna()
    beh = (w[cp]-w[cg])
    for subj, subj_name in [('cr_diff','creativity'),('ow_diff','ownership')]:
        y = w.index.to_series().map(u_idx[subj].astype(float))
        mask = ~y.isna()
        if mask.sum()<20: continue
        r,p = stats.spearmanr(beh[mask], y[mask])
        corr_rows.append(dict(behavioral=c, subjective=subj_name, n=mask.sum(), rho=r, p=p))
corr_df = pd.DataFrame(corr_rows).sort_values('p')
corr_df.to_csv(os.path.join(OUT,'extension_perception_corr.csv'), index=False)
print(corr_df.to_string(index=False))

# ===================================================================
# 7. Fixation × creativity — sanity check
# ===================================================================
print('\n=== Fixation vs creativity diff ===')
for cond in ['GPT','Persona']:
    x = wide[f'fixation_index__{cond}'].astype(float)
    for subj in ['cr_diff','ow_diff']:
        y = wide.index.to_series().map(u_idx[subj].astype(float))
        m = (~x.isna())&(~y.isna())
        if m.sum()<20: continue
        r,p = stats.spearmanr(x[m], y[m])
        print(f'  {cond} fixation vs {subj}: ρ={r:.3f} p={p:.3f}')

# ===================================================================
# 8. FIGURES
# ===================================================================
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':160,'font.size':10})

# Fig M1: manipulation check heatmap
import seaborn as sns
fig,ax = plt.subplots(figsize=(11,4))
plot_df = mc.loc[['GPT_baseline','Divergent','Convergent','Rational','BoundedRational']]
sns.heatmap(plot_df, annot=True, fmt='.2f', cmap='vlag', center=0.5, ax=ax,
            cbar_kws={'label':'mean stance score (0-3)'})
ax.set_title('Fig M1. Persona fidelity — Claude-coded stance density by family\n(a_* = assistant, u_* = user)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'figM1_manipulation.png')); plt.close()

# Fig M2: tone + qtype by family
fig, axes = plt.subplots(1, 2, figsize=(13,4))
sns.heatmap(mc_tone.loc[['GPT_baseline','Divergent','Convergent','Rational','BoundedRational']],
            annot=True, fmt='.2f', cmap='Blues', ax=axes[0])
axes[0].set_title('Assistant critique tone')
sns.heatmap(mc_qt.loc[['GPT_baseline','Divergent','Convergent','Rational','BoundedRational']],
            annot=True, fmt='.2f', cmap='Greens', ax=axes[1])
axes[1].set_title('User question type')
plt.suptitle('Fig M2. Tone and question type composition by family')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'figM2_tone_qtype.png')); plt.close()

# Fig M3: paired dotplots for expansion and contraction (assistant)
fig, axes = plt.subplots(1,2, figsize=(9,4))
for ax, c, ti in zip(axes, ['a_exp','a_con'], ['Assistant expansion','Assistant contraction']):
    cg=f'{c}__GPT'; cp=f'{c}__Persona'
    uu = wide[[cg,cp]].dropna()
    for _,r in uu.iterrows():
        ax.plot([0,1],[r[cg],r[cp]], color='gray', alpha=0.3, lw=0.5)
    ax.boxplot([uu[cg], uu[cp]], positions=[0,1], widths=0.35, showfliers=False)
    ax.set_xticks([0,1]); ax.set_xticklabels(['GPT','Persona'])
    ax.set_title(ti); ax.set_ylabel('mean stance (0-3)')
plt.suptitle('Fig M3. Paired LLM-coded stance (Persona vs GPT)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'figM3_stance_paired.png')); plt.close()

# Fig M4: fixation / drift paired + by family
fig, axes = plt.subplots(1,3, figsize=(13,4))
# paired fixation
uu = wide[['fixation_index__GPT','fixation_index__Persona']].dropna()
for _,r in uu.iterrows():
    axes[0].plot([0,1],[r['fixation_index__GPT'],r['fixation_index__Persona']], color='gray', alpha=0.3, lw=0.5)
axes[0].boxplot([uu.iloc[:,0], uu.iloc[:,1]], positions=[0,1], widths=0.35, showfliers=False)
axes[0].set_xticks([0,1]); axes[0].set_xticklabels(['GPT','Persona'])
axes[0].set_title('Fixation index (paired)')

# by family (persona only)
for ax, m, ti in zip(axes[1:], ['fixation_index','drift_trajectory'], ['Fixation by family','Drift by family']):
    data = []
    labels = ['GPT']+['Divergent','Convergent','Rational','BoundedRational']
    data.append(conv_meta[conv_meta.condition=='GPT'][m].dropna().values)
    for f in ['Divergent','Convergent','Rational','BoundedRational']:
        data.append(conv_meta[(conv_meta.condition=='Persona') & (conv_meta.family==f)][m].dropna().values)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(ti); ax.tick_params(axis='x', rotation=30)
plt.suptitle('Fig M4. Fixation and drift')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'figM4_fixation.png')); plt.close()

# Fig M5: SBERT distinctiveness paired (replaces fig6)
fig, ax = plt.subplots(figsize=(5,4))
uu = wide[['sbert_distinctiveness__GPT','sbert_distinctiveness__Persona']].dropna()
for _,r in uu.iterrows():
    ax.plot([0,1],[r['sbert_distinctiveness__GPT'],r['sbert_distinctiveness__Persona']], color='gray', alpha=0.3, lw=0.5)
ax.boxplot([uu.iloc[:,0], uu.iloc[:,1]], positions=[0,1], widths=0.35, showfliers=False)
ax.set_xticks([0,1]); ax.set_xticklabels(['GPT','Persona'])
ax.set_title('Fig M5. SBERT portfolio distinctiveness (paired)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'figM5_sbert_distinct.png')); plt.close()

# Fig M6: UMAP of conversation centroids
try:
    import umap
    red = umap.UMAP(n_components=2, random_state=0, metric='cosine').fit_transform(C)
except Exception:
    from sklearn.decomposition import PCA
    red = PCA(n_components=2).fit_transform(C)
conv_meta['u0']=red[:,0]; conv_meta['u1']=red[:,1]
fig, axes = plt.subplots(1,2, figsize=(12,5))
for ax, ch in zip(axes, ['Bicycle','Library']):
    sub = conv_meta[conv_meta.challenge==ch]
    for cond, marker, col in [('GPT','o','tab:blue'),('Persona','^','tab:red')]:
        s2 = sub[sub.condition==cond]
        ax.scatter(s2['u0'], s2['u1'], s=25, alpha=0.6, label=cond, marker=marker, color=col)
    ax.set_title(ch); ax.legend()
plt.suptitle('Fig M6. Conversation-level idea-space map (SBERT UMAP of user-message centroids)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'figM6_umap.png')); plt.close()

print('\nDONE — artifacts in', OUT)
