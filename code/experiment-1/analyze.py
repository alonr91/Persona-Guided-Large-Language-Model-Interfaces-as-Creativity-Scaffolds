"""
HCI analysis of persona-guided LLM co-creative sessions.
Single-pass script. Outputs tables to analysis_out/ and figures to figures/.
"""
import os, re, json, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Windows stdout unicode
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT = os.path.join(ROOT, 'analysis_out')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

logs = pd.read_csv(os.path.join(ROOT, 'Experiment1_logs_cleaned_keepable_paired_translated.csv'))
users = pd.read_excel(os.path.join(OUT, 'users_translated.xlsx'),
                      sheet_name='corrected_users')

# --- clean whitespace in column names ---
users.columns = [c.strip() for c in users.columns]

# ===================================================================
# A. DATA AUDIT / MASTER TABLES
# ===================================================================
print("="*70); print("A. DATA AUDIT"); print("="*70)
print(f"logs: {len(logs)} rows, {logs['User_id'].nunique()} users, "
      f"{logs['conversation_id'].nunique()} conversations")
print(f"users: {len(users)} rows, {users['id'].nunique()} unique ids")
print(f"id overlap: {logs['User_id'].isin(users['id']).mean():.3f}")

# user-persona condition label (the non-GPT one each user got)
user_persona = (logs.groupby('User_id')['Persona_type']
                .apply(lambda s: [x for x in s.unique() if x != 'GPT'][0])
                .reset_index().rename(columns={'Persona_type':'persona_cond'}))
print("\npersona_cond distribution:", user_persona['persona_cond'].value_counts().to_dict())

# persona family grouping (divergent/convergent = "creative dialectic"; rational/bounded = "decision style")
family_map = {'Divergent':'Divergent', 'Convergent':'Convergent',
              'strictly rational':'Rational', 'bounded rationality':'BoundedRational'}
user_persona['family'] = user_persona['persona_cond'].map(family_map)

# ---- conversation level master ----
conv = (logs.sort_values(['conversation_id','message_id'])
            .groupby('conversation_id')
            .agg(user=('User_id','first'),
                 persona_type=('Persona_type','first'),
                 challenge=('Corrected Challenge type','first'),
                 role=('Desginer or Engineer','first'),
                 n_msg=('message_id','count'),
                 t0=('timestamp','min'),
                 t1=('timestamp','max'))
            .reset_index())
conv['duration_min'] = (conv['t1']-conv['t0'])*24*60  # excel-day fractions
conv['condition'] = np.where(conv['persona_type']=='GPT','GPT','Persona')
conv = conv.merge(user_persona[['User_id','persona_cond','family']],
                  left_on='user', right_on='User_id', how='left').drop(columns=['User_id'])

# round order: round 1 = earlier timestamp per user, round 2 = later
conv = conv.sort_values(['user','t0'])
conv['round'] = conv.groupby('user').cumcount()+1

# ---- per-message features ----
def wc(x):
    if pd.isna(x): return 0
    return len(re.findall(r"\w+", str(x)))
logs['n_words'] = logs['message'].apply(wc)
logs['has_q']   = logs['message'].fillna('').str.contains(r'\?')
logs['msg_len'] = logs['message'].fillna('').str.len()
# order within conversation
logs = logs.sort_values(['conversation_id','message_id'])
logs['turn_idx'] = logs.groupby('conversation_id').cumcount()
logs['turn_frac'] = logs['turn_idx'] / logs.groupby('conversation_id')['turn_idx'].transform('max').replace(0,1)

# ===================================================================
# B. CONVERSATION PROCESS METRICS
# ===================================================================
print("\n"+"="*70); print("B. PROCESS METRICS"); print("="*70)

def conv_features(g):
    u = g[g.message_src=='user']; a = g[g.message_src=='assistant']
    feats = dict(
        n_user_msg = len(u),
        n_ast_msg  = len(a),
        user_words = u['n_words'].sum(),
        ast_words  = a['n_words'].sum(),
        user_mean_len = u['n_words'].mean() if len(u) else 0,
        ast_mean_len  = a['n_words'].mean() if len(a) else 0,
        user_q_rate = u['has_q'].mean() if len(u) else np.nan,
        ast_q_rate  = a['has_q'].mean() if len(a) else np.nan,
        user_word_share = u['n_words'].sum()/max(1, u['n_words'].sum()+a['n_words'].sum()),
    )
    # turn-length trend: slope of user msg word count over turn index
    if len(u) >= 3:
        slope, *_ = stats.linregress(np.arange(len(u)), u['n_words'].values)
        feats['user_len_slope'] = slope
    else:
        feats['user_len_slope'] = np.nan
    # early vs late user engagement
    if len(u) >= 4:
        half = len(u)//2
        feats['user_words_early'] = u['n_words'].iloc[:half].mean()
        feats['user_words_late']  = u['n_words'].iloc[half:].mean()
    else:
        feats['user_words_early']=feats['user_words_late']=np.nan
    return pd.Series(feats)

procf = logs.groupby('conversation_id').apply(conv_features).reset_index()
conv  = conv.merge(procf, on='conversation_id')

# ===================================================================
# C. TRAJECTORY / TURN CODING
# ===================================================================
# Lightweight rule-based stance tagging. Treat as EXPLORATORY signals.
RX = {
 'propose'   : re.compile(r'\b(what if|how about|could|maybe|suggest|propose|idea(s)?|imagine|consider|another option|alternatively)\b', re.I),
 'critique'  : re.compile(r"\b(but|however|issue|problem|concern|doesn't|won'?t work|too (expensive|complex|hard)|drawback|risk|downside|not sure|disagree)\b", re.I),
 'compare'   : re.compile(r'\b(vs\.?|versus|compare|compared|trade ?off|rather than|better than|worse than)\b', re.I),
 'clarify'   : re.compile(r'\b(what do you mean|can you (explain|clarify)|I don\'?t understand|could you elaborate|more detail)\b', re.I),
 'commit'    : re.compile(r'\b(let\'?s go with|decide|final|choose|pick|commit|we will|settle on|go with)\b', re.I),
 'reframe'   : re.compile(r"\b(actually|reframe|different angle|step back|bigger picture|instead think|what if the problem|really about)\b", re.I),
 'question'  : re.compile(r'\?'),
}
for k, rx in RX.items():
    logs[f'tag_{k}'] = logs['message'].fillna('').str.contains(rx)

# conversation-level stance aggregates (user side and assistant side)
def stance_agg(g):
    out={}
    for side, sub in [('u', g[g.message_src=='user']), ('a', g[g.message_src=='assistant'])]:
        n = max(1, len(sub))
        for k in RX:
            out[f'{side}_{k}'] = sub[f'tag_{k}'].sum()/n
    # positional: first third = early, middle = mid, last third = late (user msgs only)
    u = g[g.message_src=='user'].sort_values('message_id')
    if len(u):
        tf = u['turn_frac'].values
        for seg, mask in [('early', tf<=0.33),('mid',(tf>0.33)&(tf<0.67)),('late',tf>=0.67)]:
            seg_sub = u[mask]
            for k in ['propose','critique','compare','commit','reframe','question']:
                out[f'u_{seg}_{k}'] = seg_sub[f'tag_{k}'].mean() if len(seg_sub) else np.nan
    # time to first commit and first critique (user side)
    for k in ['commit','critique','reframe']:
        hits = u[u[f'tag_{k}']==True]
        out[f'u_first_{k}_frac'] = hits['turn_frac'].min() if len(hits) else np.nan
    return pd.Series(out)

stf = logs.groupby('conversation_id').apply(stance_agg).reset_index()
conv = conv.merge(stf, on='conversation_id')

# ===================================================================
# Derived HCI metrics
# ===================================================================
# Exploration Breadth Ratio: propose density before first commit
conv['exploration_breadth'] = conv['u_propose'].fillna(0)
# Convergence Timing Index: first commit fraction (lower = earlier closer)
conv['convergence_timing'] = conv['u_first_commit_frac']
# Recovery After Critique: proportion of user-propose turns AFTER first critique
def recovery(g):
    u = g[g.message_src=='user'].sort_values('message_id')
    if u['tag_critique'].sum()==0 or len(u)<4: return np.nan
    first_cri = u[u['tag_critique']].iloc[0]['turn_frac']
    post = u[u['turn_frac']>first_cri]
    if len(post)==0: return np.nan
    return post['tag_propose'].mean()
conv['recovery_after_critique'] = logs.groupby('conversation_id').apply(recovery).values

# AI dependence proxy: assistant word share
conv['ai_word_share'] = 1 - conv['user_word_share']
# Reframe rate already available as u_reframe

# ===================================================================
# Build paired (within-user) table
# ===================================================================
print("CONV columns pre-pivot:", list(conv.columns))
print("CONV dups:", conv.columns[conv.columns.duplicated()].tolist())
# drop duplicate columns if any
conv = conv.loc[:, ~conv.columns.duplicated()]
_exclude = {'conversation_id','user','t0','t1','persona_id','round',
            'condition','persona_type','challenge','role','persona_cond','family'}
_num = [c for c in conv.columns if c not in _exclude and pd.api.types.is_numeric_dtype(conv[c])]
wide = conv.pivot_table(index='user', columns='condition', values=_num, aggfunc='first')
wide.columns = [f'{m}__{c}' for m,c in wide.columns]
wide = wide.reset_index().merge(user_persona, left_on='user', right_on='User_id', how='left').drop(columns=['User_id'])
wide = wide.merge(users, left_on='user', right_on='id', how='left')

# Persona preference binary: did user prefer persona interface?
# Column: 'More effective interface for creative solution'
pref_col = 'More effective interface for creative solution'
if pref_col in wide.columns:
    wide['pref_persona'] = wide[pref_col].astype(str).str.lower().str.contains('dedicated|character', regex=True)
# better: need to read actual values
print("\nInterface pref values:", users[pref_col].value_counts(dropna=False).to_dict())

# ===================================================================
# LAYER A: QUESTIONNAIRE PAIRED ANALYSES
# ===================================================================
print("\n"+"="*70); print("LAYER A — QUESTIONNAIRE PAIRED TESTS"); print("="*70)

def paired_test(a, b, name):
    mask = (~pd.isna(a))&(~pd.isna(b))
    a,b = a[mask], b[mask]
    if len(a)<5: return None
    d = a-b
    t,p = stats.ttest_rel(a,b)
    w,pw = stats.wilcoxon(a,b, zero_method='wilcox') if (d!=0).any() else (np.nan,np.nan)
    dz = d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else np.nan
    return dict(name=name, n=len(a),
                mean_1=a.mean(), mean_2=b.mean(),
                mean_diff=d.mean(), sd_diff=d.std(ddof=1),
                t=t, p_t=p, W=w, p_w=pw, dz=dz)

rows=[]
for a,b,nm in [('Creativity assistant #1','Creativity assistant #2','Creativity R1 vs R2'),
               ('Ownership #1','Ownership #2','Ownership R1 vs R2')]:
    r = paired_test(users[a], users[b], nm); rows.append(r)

# Round mapping: what was round 1 vs round 2 persona_type per user?
# We need: GPT vs Persona paired across creativity/ownership.
# Map using 'Persona round 1' / 'Persona round 2' fields to identify which round was GPT.
def round_of_gpt(row):
    r1 = str(row.get('Persona round 1','')).strip().lower()
    r2 = str(row.get('Persona round 2','')).strip().lower()
    if 'gpt' in r1 or r1=='' or r1=='nan': pass
    if 'gpt' in r1: return 1
    if 'gpt' in r2: return 2
    return np.nan
users['gpt_round'] = users.apply(round_of_gpt, axis=1)
print("gpt_round counts:", users['gpt_round'].value_counts(dropna=False).to_dict())

# Construct GPT and Persona scores per user
def make_gp(df, r1c, r2c):
    gpt = np.where(df['gpt_round']==1, df[r1c], np.where(df['gpt_round']==2, df[r2c], np.nan))
    per = np.where(df['gpt_round']==1, df[r2c], np.where(df['gpt_round']==2, df[r1c], np.nan))
    return pd.Series(gpt, index=df.index), pd.Series(per, index=df.index)

users['creativity_gpt'], users['creativity_persona'] = make_gp(users,'Creativity assistant #1','Creativity assistant #2')
users['ownership_gpt'],  users['ownership_persona']  = make_gp(users,'Ownership #1','Ownership #2')
for a,b,nm in [('creativity_gpt','creativity_persona','Creativity GPT vs Persona'),
               ('ownership_gpt','ownership_persona','Ownership GPT vs Persona')]:
    rows.append(paired_test(users[a].astype(float), users[b].astype(float), nm))

qres = pd.DataFrame([r for r in rows if r is not None])
print(qres.to_string(index=False))
qres.to_csv(os.path.join(OUT,'A_questionnaire_paired.csv'), index=False)

# by persona family
users = users.merge(user_persona[['User_id','persona_cond','family']],
                    left_on='id', right_on='User_id', how='left')
print("\n— Creativity (Persona-GPT) by persona family —")
users['cr_diff'] = users['creativity_persona'].astype(float)-users['creativity_gpt'].astype(float)
users['ow_diff'] = users['ownership_persona'].astype(float)-users['ownership_gpt'].astype(float)
fam = users.groupby('family')[['cr_diff','ow_diff']].agg(['mean','std','count'])
print(fam)
fam.to_csv(os.path.join(OUT,'A_by_family.csv'))

# personality correlations with diff
pers_cols = ['Extraversion','Agreeableness','Conscientiousness','Negative Emotionality','Open-Mindedness']
corr_rows=[]
for p in pers_cols:
    for d,lbl in [('cr_diff','Persona-GPT Creativity diff'),('ow_diff','Persona-GPT Ownership diff')]:
        x = users[p].astype(float); y = users[d].astype(float)
        m = (~x.isna())&(~y.isna())
        if m.sum()<10: continue
        r,pv = stats.spearmanr(x[m],y[m])
        corr_rows.append(dict(trait=p, diff=lbl, n=m.sum(), rho=r, p=pv))
pers_res = pd.DataFrame(corr_rows)
print("\n— Personality x diff correlations —"); print(pers_res.to_string(index=False))
pers_res.to_csv(os.path.join(OUT,'A_personality_corr.csv'), index=False)

# Interface preference
pref = users[pref_col].astype(str).str.strip()
print("\nInterface pref raw values:", pref.value_counts(dropna=False).to_dict())

# ===================================================================
# LAYER B/C paired process tests (GPT vs Persona within user)
# ===================================================================
print("\n"+"="*70); print("LAYER B/C — PROCESS/TRAJECTORY PAIRED"); print("="*70)
proc_metrics = ['n_user_msg','user_words','user_mean_len','user_q_rate','user_word_share',
                'user_len_slope','duration_min',
                'u_propose','u_critique','u_compare','u_clarify','u_commit','u_reframe','u_question',
                'a_propose','a_critique','a_compare','a_commit','a_reframe',
                'u_early_propose','u_late_commit','u_first_commit_frac','u_first_critique_frac',
                'u_first_reframe_frac','recovery_after_critique','ai_word_share']
proc_rows=[]
for m in proc_metrics:
    colg = f'{m}__GPT'; colp=f'{m}__Persona'
    if colg not in wide.columns or colp not in wide.columns: continue
    r = paired_test(wide[colp].astype(float), wide[colg].astype(float),
                    f'{m} Persona vs GPT')
    if r: proc_rows.append(r)
proc_df = pd.DataFrame(proc_rows).sort_values('p_t')
proc_df.to_csv(os.path.join(OUT,'B_process_paired.csv'), index=False)
print(proc_df.to_string(index=False))

# By family (unpaired comparison vs GPT baseline within-user diff)
print("\n— process diffs by persona family (mean Persona-GPT) —")
fam_rows=[]
for m in proc_metrics:
    colg = f'{m}__GPT'; colp=f'{m}__Persona'
    if colg not in wide.columns: continue
    w = wide.copy()
    w['d'] = w[colp].astype(float)-w[colg].astype(float)
    for f, sub in w.groupby('family'):
        if sub['d'].notna().sum()<5: continue
        fam_rows.append(dict(metric=m, family=f, n=sub['d'].notna().sum(),
                             mean_diff=sub['d'].mean(),
                             p=stats.wilcoxon(sub['d'].dropna()).pvalue if (sub['d'].dropna()!=0).any() else np.nan))
fam_proc = pd.DataFrame(fam_rows)
fam_proc.to_csv(os.path.join(OUT,'B_process_by_family.csv'), index=False)

# ===================================================================
# Perception–behavior dissociation (Story 3)
# ===================================================================
print("\n"+"="*70); print("STORY 3 — PERCEPTION vs BEHAVIOR"); print("="*70)
# behavioral authorship proxy = user word share (higher = more user authorship)
# reported ownership = ownership_persona / ownership_gpt
# compute "ownership gap" = z(reported) - z(behavioral)
for cond,colg,colp,lbl in [('GPT','user_word_share__GPT','ownership_gpt','GPT'),
                           ('Persona','user_word_share__Persona','ownership_persona','Persona')]:
    b = wide[colg].astype(float)
    # ownership from users-merged
    r = wide[colp].astype(float) if colp in wide.columns else pd.Series([np.nan]*len(wide))
    # fallback fetch from users
    if r.isna().all():
        r = wide['user'].map(users.set_index('id')[colp].astype(float))
    zb = (b-b.mean())/b.std()
    zr = (r-r.mean())/r.std()
    wide[f'ownership_gap__{lbl}'] = zr - zb
    rho, p = stats.spearmanr(b, r, nan_policy='omit')
    print(f'{lbl}: ownership(reported) vs user_word_share(behavioral)  rho={rho:.3f} p={p:.3g}')

# Cross-condition: does creativity score track exploration breadth?
for cond,colb,cols,lbl in [('GPT','u_propose__GPT','creativity_gpt','GPT'),
                           ('Persona','u_propose__Persona','creativity_persona','Persona')]:
    if colb not in wide.columns: continue
    b = wide[colb].astype(float)
    if cols not in wide.columns:
        s = wide['user'].map(users.set_index('id')[cols].astype(float))
    else:
        s = wide[cols].astype(float)
    r,p = stats.spearmanr(b, s, nan_policy='omit')
    print(f'{lbl}: creativity(reported) vs u_propose(behavioral)  rho={r:.3f} p={p:.3g}')

# ===================================================================
# Interaction archetypes via k-means on process features
# ===================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

feat_cols = ['user_mean_len','user_q_rate','user_word_share',
             'u_propose','u_critique','u_compare','u_commit','u_reframe',
             'u_first_commit_frac','recovery_after_critique','u_late_commit','u_early_propose']
X = conv[feat_cols].copy()
X = X.fillna(X.median())
Xs = StandardScaler().fit_transform(X)
# Pick k by silhouette briefly
from sklearn.metrics import silhouette_score
best=(None,-1)
for k in range(3,7):
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
    s = silhouette_score(Xs, km.labels_)
    if s>best[1]: best=(k,s); best_km=km
print(f"\nArchetypes: best k={best[0]}, silhouette={best[1]:.3f}")
conv['archetype'] = best_km.labels_
# characterize
cent = pd.DataFrame(best_km.cluster_centers_, columns=feat_cols)
cent.to_csv(os.path.join(OUT,'C_archetype_centroids.csv'))
print("Archetype centroids (standardized):")
print(cent.round(2))
# cross-tab vs condition / family
print("\nArchetype x condition:\n", pd.crosstab(conv['archetype'], conv['condition']))
print("\nArchetype x family:\n", pd.crosstab(conv['archetype'], conv['family']))
conv[['conversation_id','user','condition','family','challenge','archetype']].to_csv(
    os.path.join(OUT,'C_archetypes.csv'), index=False)

# ===================================================================
# Idea portfolio — local embeddings if available, else TF-IDF fallback
# ===================================================================
print("\n"+"="*70); print("LAYER D — IDEA PORTFOLIO (TF-IDF fallback)"); print("="*70)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Concatenate user messages per conversation
user_text = (logs[logs.message_src=='user']
             .groupby('conversation_id')['message']
             .apply(lambda s: ' '.join(str(x) for x in s if isinstance(x,str)))
             .reindex(conv['conversation_id']).values)
tfidf = TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=(1,2), min_df=2)
Xt = tfidf.fit_transform(user_text)
# distinctiveness vs same-challenge peers
conv['distinctiveness'] = np.nan
for ch in conv['challenge'].unique():
    idx = np.where(conv['challenge'].values==ch)[0]
    sub = Xt[idx]
    sim = cosine_similarity(sub)
    np.fill_diagonal(sim,np.nan)
    mean_sim = np.nanmean(sim, axis=1)
    conv.loc[conv.index[idx], 'distinctiveness'] = 1-mean_sim
# within-user diversity: cosine distance between user's 2 convs
def user_div(u):
    rows = conv[conv.user==u].sort_values('round').index
    if len(rows)!=2: return np.nan
    v = Xt[rows]
    return 1-cosine_similarity(v)[0,1]
users['within_user_div'] = users['id'].map(lambda u: user_div(u))
# between condition
print("distinctiveness by condition:\n", conv.groupby('condition')['distinctiveness'].agg(['mean','std','count']))
# paired test distinctiveness Persona vs GPT
w = conv.pivot_table(index='user', columns='condition', values='distinctiveness', aggfunc='first')
if 'Persona' in w.columns and 'GPT' in w.columns:
    r = paired_test(w['Persona'].astype(float), w['GPT'].astype(float), 'distinctiveness Persona vs GPT')
    print('distinctiveness paired:', r)

# ===================================================================
# LAYER E — CONSECUTIVE-MESSAGE NOVELTY (SBERT)
# ===================================================================
# Cosine distance between successive message embeddings within each
# conversation, adapted from yes_and_novelty_surprise.py.
# Each message is one embedding (multi-sentence messages are not split),
# matching how msg_embeddings.npy was produced.
print("\n"+"="*70); print("LAYER E — CONSECUTIVE-MESSAGE NOVELTY (SBERT)"); print("="*70)

trans = None  # exposed for LAYERs H / I
emb_path = os.path.join(OUT, 'msg_embeddings.npy')
if not os.path.exists(emb_path):
    print(f"  [skip] {emb_path} not found. Run embed_messages.py first.")
else:
    E = np.load(emb_path)
    logs_aligned = logs.sort_values(['conversation_id','message_id']).reset_index(drop=True)
    assert len(logs_aligned) == len(E), f"embedding/logs length mismatch: {len(E)} vs {len(logs_aligned)}"

    # E2: per-transition distances
    prev_cid = logs_aligned['conversation_id'].shift(1)
    same_conv = (logs_aligned['conversation_id'] == prev_cid).values
    sim_prev = np.full(len(E), np.nan)
    sim_prev[1:] = (E[1:] * E[:-1]).sum(axis=1)
    dist = 1.0 - sim_prev
    trans = logs_aligned.assign(
        prev_speaker=logs_aligned['message_src'].shift(1),
        dist=dist,
    )
    trans = trans[same_conv].copy()
    trans = trans.rename(columns={'message_src':'speaker'})
    trans['transition_type'] = trans['prev_speaker'].astype(str) + '->' + trans['speaker'].astype(str)
    trans = trans.merge(conv[['conversation_id','condition','family']], on='conversation_id', how='left')
    trans = trans[['conversation_id','User_id','condition','family',
                   'Corrected Challenge type','speaker','prev_speaker',
                   'turn_idx','turn_frac','dist','transition_type']]

    # E3: aggregate to conversation means
    conv_all = trans.groupby('conversation_id')['dist'].mean().rename('mean_dist_all')
    conv_sp  = (trans.groupby(['conversation_id','speaker'])['dist']
                .mean().unstack('speaker')
                .rename(columns={'user':'mean_dist_user','assistant':'mean_dist_assistant'}))
    for col in ('mean_dist_user','mean_dist_assistant'):
        if col not in conv_sp.columns: conv_sp[col] = np.nan
    conv = conv.merge(conv_all, on='conversation_id', how='left').merge(
        conv_sp[['mean_dist_user','mean_dist_assistant']], on='conversation_id', how='left')

    E_vals = ['mean_dist_all','mean_dist_user','mean_dist_assistant']
    E_wide = conv.pivot_table(index='user', columns='condition', values=E_vals, aggfunc='first')
    E_wide.columns = [f'{m}__{c}' for m,c in E_wide.columns]
    E_wide = E_wide.reset_index()
    E_wide = E_wide.merge(user_persona[['User_id','family']],
                          left_on='user', right_on='User_id', how='left').drop(columns=['User_id'])
    # merge new cols into master wide (avoid duplicating 'family')
    new_cols = [c for c in E_wide.columns if c not in wide.columns and c != 'family']
    wide = wide.merge(E_wide[['user'] + new_cols], on='user', how='left')

    # E4: paired tests — three schemes
    e_rows = []
    def _row(r, scheme, family, speaker):
        if r is None: return None
        out = {'scheme':scheme, 'family':family, 'speaker':speaker}
        out.update(r)
        return out

    def _pair(df, col, fam_label, sp_label, scheme):
        cg, cp = f'{col}__GPT', f'{col}__Persona'
        if cg not in df.columns or cp not in df.columns: return None
        r = paired_test(df[cp].astype(float), df[cg].astype(float),
                        f'{col} Persona vs GPT' + (f' [{fam_label}]' if fam_label!='all' else ''))
        return _row(r, scheme, fam_label, sp_label)

    for r in [_pair(E_wide,'mean_dist_all','all','all','condition')]:
        if r: e_rows.append(r)
    for sp, col in [('user','mean_dist_user'), ('assistant','mean_dist_assistant')]:
        r = _pair(E_wide, col, 'all', sp, 'condition_x_speaker')
        if r: e_rows.append(r)
    for fam, sub in E_wide.groupby('family'):
        for sp, col in [('user','mean_dist_user'), ('assistant','mean_dist_assistant')]:
            r = _pair(sub, col, fam, sp, 'condition_x_speaker_x_family')
            if r: e_rows.append(r)

    e_paired = pd.DataFrame(e_rows)
    e_paired.to_csv(os.path.join(OUT,'E_consec_novelty_paired.csv'), index=False)
    print(e_paired.to_string(index=False) if len(e_paired) else '(no paired rows)')

    # E5: descriptive by (condition, speaker, family) over transitions
    e_desc = trans.groupby(['condition','speaker','family'])['dist'].agg(['count','mean','std']).rename(
        columns={'count':'n_transitions'})
    e_desc['sem'] = e_desc['std'] / np.sqrt(e_desc['n_transitions'].clip(lower=1))
    e_desc = e_desc.reset_index()
    e_desc.to_csv(os.path.join(OUT,'E_consec_novelty_by_group.csv'), index=False)
    print("\n-- descriptive by (condition, speaker, family) --")
    print(e_desc.to_string(index=False))

# ===================================================================
# LAYER F — MESSAGE-LEVEL SURPRISE (causal-LM NLL)
# ===================================================================
# Negative log-likelihood of each message given prior in-conversation
# context, computed offline by compute_surprise.py (GPT-2 small).
# Use per-token surprise as the headline metric (invariant to message length).
print("\n"+"="*70); print("LAYER F — MESSAGE-LEVEL SURPRISE (GPT-2)"); print("="*70)

msurp = None  # exposed for LAYERs G / I
surp_path = os.path.join(OUT, 'msg_surprise_per_tok.npy')
if not os.path.exists(surp_path):
    print(f"  [skip] {surp_path} not found. Run compute_surprise.py first.")
else:
    S = np.load(surp_path)
    logs_aligned_f = logs.sort_values(['conversation_id','message_id']).reset_index(drop=True)
    assert len(logs_aligned_f) == len(S), f"surprise/logs length mismatch: {len(S)} vs {len(logs_aligned_f)}"

    msurp = logs_aligned_f.assign(surprise=S).dropna(subset=['surprise']).copy()
    msurp = msurp.rename(columns={'message_src':'speaker'})
    msurp = msurp.merge(conv[['conversation_id','condition','family']], on='conversation_id', how='left')
    print(f"valid surprise scores: {len(msurp)} / {len(logs_aligned_f)}")

    # conversation means
    cs_all = msurp.groupby('conversation_id')['surprise'].mean().rename('mean_surp_all')
    cs_sp  = (msurp.groupby(['conversation_id','speaker'])['surprise']
              .mean().unstack('speaker')
              .rename(columns={'user':'mean_surp_user','assistant':'mean_surp_assistant'}))
    for col in ('mean_surp_user','mean_surp_assistant'):
        if col not in cs_sp.columns: cs_sp[col] = np.nan
    conv = conv.merge(cs_all, on='conversation_id', how='left').merge(
        cs_sp[['mean_surp_user','mean_surp_assistant']], on='conversation_id', how='left')

    F_vals = ['mean_surp_all','mean_surp_user','mean_surp_assistant']
    F_wide = conv.pivot_table(index='user', columns='condition', values=F_vals, aggfunc='first')
    F_wide.columns = [f'{m}__{c}' for m,c in F_wide.columns]
    F_wide = F_wide.reset_index()
    F_wide = F_wide.merge(user_persona[['User_id','family']],
                          left_on='user', right_on='User_id', how='left').drop(columns=['User_id'])
    new_cols_f = [c for c in F_wide.columns if c not in wide.columns and c != 'family']
    wide = wide.merge(F_wide[['user'] + new_cols_f], on='user', how='left')

    f_rows = []
    def _frow(r, scheme, family, speaker):
        if r is None: return None
        out = {'scheme':scheme, 'family':family, 'speaker':speaker}
        out.update(r)
        return out

    def _fpair(df, col, fam_label, sp_label, scheme):
        cg, cp = f'{col}__GPT', f'{col}__Persona'
        if cg not in df.columns or cp not in df.columns: return None
        r = paired_test(df[cp].astype(float), df[cg].astype(float),
                        f'{col} Persona vs GPT' + (f' [{fam_label}]' if fam_label!='all' else ''))
        return _frow(r, scheme, fam_label, sp_label)

    for r in [_fpair(F_wide,'mean_surp_all','all','all','condition')]:
        if r: f_rows.append(r)
    for sp, col in [('user','mean_surp_user'), ('assistant','mean_surp_assistant')]:
        r = _fpair(F_wide, col, 'all', sp, 'condition_x_speaker')
        if r: f_rows.append(r)
    for fam, sub in F_wide.groupby('family'):
        for sp, col in [('user','mean_surp_user'), ('assistant','mean_surp_assistant')]:
            r = _fpair(sub, col, fam, sp, 'condition_x_speaker_x_family')
            if r: f_rows.append(r)

    f_paired = pd.DataFrame(f_rows)
    f_paired.to_csv(os.path.join(OUT,'F_surprise_paired.csv'), index=False)
    print(f_paired.to_string(index=False) if len(f_paired) else '(no paired rows)')

    f_desc = msurp.groupby(['condition','speaker','family'])['surprise'].agg(['count','mean','std']).rename(
        columns={'count':'n_messages'})
    f_desc['sem'] = f_desc['std'] / np.sqrt(f_desc['n_messages'].clip(lower=1))
    f_desc = f_desc.reset_index()
    f_desc.to_csv(os.path.join(OUT,'F_surprise_by_group.csv'), index=False)
    print("\n-- descriptive by (condition, speaker, family) --")
    print(f_desc.to_string(index=False))

# ===================================================================
# LAYER G — BEHAVIOR <-> PERCEPTION <-> PERSONALITY
# ===================================================================
# Within-user deltas on behavioral axes (novelty, surprise) correlated with
# (a) perception deltas (creativity, ownership) and (b) Big-5 personality
# traits. Also reports each behavioral axis by interface-preference group.
print("\n"+"="*70); print("LAYER G — BEHAVIOR x PERCEPTION x PERSONALITY"); print("="*70)

# Build per-user deltas (Persona - GPT) on behavioral columns now in `wide`
behav_bases = [c for c in ['mean_dist_all','mean_dist_user','mean_dist_assistant',
                            'mean_surp_all','mean_surp_user','mean_surp_assistant']
               if f'{c}__GPT' in wide.columns and f'{c}__Persona' in wide.columns]
for c in behav_bases:
    wide[f'd_{c}'] = wide[f'{c}__Persona'].astype(float) - wide[f'{c}__GPT'].astype(float)

# Get perception deltas & Big-5 on the `wide` frame (merge from users)
users_slim = users.set_index('id')
for src in ['cr_diff','ow_diff','creativity_persona','creativity_gpt',
            'ownership_persona','ownership_gpt'] + pers_cols:
    if src in users_slim.columns:
        wide[src] = wide['user'].map(users_slim[src])

# --- G1: behavior delta vs perception delta ---
corr_rows = []
def _sp(x, y):
    m = (~pd.isna(x))&(~pd.isna(y))
    if m.sum()<10: return None
    rho, p = stats.spearmanr(x[m], y[m])
    return dict(n=int(m.sum()), rho=rho, p=p)

for b in behav_bases:
    for t,lbl in [('cr_diff','Creativity diff (P-G)'),
                  ('ow_diff','Ownership diff (P-G)')]:
        r = _sp(wide[f'd_{b}'].astype(float), wide[t].astype(float))
        if r: corr_rows.append({'behavior_delta':f'd_{b}', 'target':lbl, **r})

# --- G2: behavior delta vs Big-5 ---
for b in behav_bases:
    for p in pers_cols:
        if p not in wide.columns: continue
        r = _sp(wide[f'd_{b}'].astype(float), wide[p].astype(float))
        if r: corr_rows.append({'behavior_delta':f'd_{b}', 'target':p, **r})

# --- G3: absolute behavior vs Big-5 (GPT baseline, Persona, per-condition) ---
for b in behav_bases:
    for cond in ['GPT','Persona']:
        col = f'{b}__{cond}'
        for p in pers_cols:
            if p not in wide.columns or col not in wide.columns: continue
            r = _sp(wide[col].astype(float), wide[p].astype(float))
            if r: corr_rows.append({'behavior_delta':col, 'target':p, **r})

G_corr = pd.DataFrame(corr_rows).sort_values('p')
G_corr.to_csv(os.path.join(OUT,'G_behavior_perception_personality_corr.csv'), index=False)
print("Top 15 by |rho| (significant at p<.05):")
sig = G_corr[G_corr['p']<0.05].copy()
sig['abs_rho'] = sig['rho'].abs()
print(sig.sort_values('abs_rho', ascending=False).head(15).to_string(index=False))

# --- G4: does personality moderate the behavior-delta direction? ---
# For the headline two behavior axes, split each Big-5 at median and test delta differs
g4_rows = []
for b in behav_bases:
    d = wide[f'd_{b}'].astype(float)
    for p in pers_cols:
        if p not in wide.columns: continue
        v = wide[p].astype(float)
        m = (~d.isna())&(~v.isna())
        if m.sum()<20: continue
        med = v[m].median()
        hi, lo = d[m & (v>med)], d[m & (v<=med)]
        if len(hi)<5 or len(lo)<5: continue
        t, pv = stats.ttest_ind(hi, lo, equal_var=False)
        g4_rows.append(dict(behavior_delta=f'd_{b}', trait=p, n_hi=len(hi), n_lo=len(lo),
                            mean_hi=hi.mean(), mean_lo=lo.mean(), t=t, p=pv))
G_mod = pd.DataFrame(g4_rows).sort_values('p')
G_mod.to_csv(os.path.join(OUT,'G_behavior_delta_by_trait_splits.csv'), index=False)

# ===================================================================
# LAYER H — YES-AND: TRANSITION-TYPE DECOMPOSITION
# ===================================================================
# Split LAYER E distances by transition type: who speaks at i given who
# spoke at i-1. "assistant->user" = user's response to partner (acceptance
# / building); "user->assistant" = assistant's response to user.
print("\n"+"="*70); print("LAYER H — YES-AND TRANSITION DECOMPOSITION"); print("="*70)

if trans is None:
    print("  [skip] trans not available (LAYER E did not run).")
else:
    # transition-type frequencies
    print("transition-type frequencies:")
    print(trans['transition_type'].value_counts().to_string())

    # conv-level means per transition type
    h_conv = (trans.groupby(['conversation_id','transition_type'])['dist']
              .mean().unstack('transition_type'))
    # rename columns to safe tokens
    h_cols = {c: f"d_{c.replace('->','_to_')}" for c in h_conv.columns}
    h_conv = h_conv.rename(columns=h_cols).reset_index()
    conv_h = conv[['conversation_id','user','condition','family']].merge(h_conv, on='conversation_id', how='left')

    value_cols = list(h_cols.values())
    H_wide = conv_h.pivot_table(index='user', columns='condition', values=value_cols, aggfunc='first')
    H_wide.columns = [f'{m}__{c}' for m,c in H_wide.columns]
    H_wide = H_wide.reset_index().merge(user_persona[['User_id','family']],
                                        left_on='user', right_on='User_id', how='left').drop(columns=['User_id'])

    h_rows = []
    for col in value_cols:
        cg, cp = f'{col}__GPT', f'{col}__Persona'
        if cg not in H_wide.columns or cp not in H_wide.columns: continue
        r = paired_test(H_wide[cp].astype(float), H_wide[cg].astype(float),
                        f'{col} Persona vs GPT')
        if r:
            row = {'transition_type': col.replace('d_','').replace('_to_','->'),
                   'scheme':'condition','family':'all'}
            row.update(r); h_rows.append(row)
        # by family
        for fam, sub in H_wide.groupby('family'):
            mask = sub[[cg,cp]].notna().all(axis=1)
            if mask.sum()<5: continue
            r = paired_test(sub.loc[mask,cp].astype(float), sub.loc[mask,cg].astype(float),
                            f'{col} Persona vs GPT [{fam}]')
            if r:
                row = {'transition_type': col.replace('d_','').replace('_to_','->'),
                       'scheme':'condition_x_family','family':fam}
                row.update(r); h_rows.append(row)

    H_paired = pd.DataFrame(h_rows).sort_values('p_t')
    H_paired.to_csv(os.path.join(OUT,'H_transition_type_paired.csv'), index=False)
    print("\n-- transition-type paired tests (all families) --")
    print(H_paired[H_paired.family=='all'].to_string(index=False) if len(H_paired) else '(none)')

# ===================================================================
# LAYER I — TRAJECTORY (novelty & surprise vs turn_frac)
# ===================================================================
# Bin messages by normalized conversation position. Report mean novelty
# (LAYER E) and surprise (LAYER F) per bin x condition x speaker.
print("\n"+"="*70); print("LAYER I — TRAJECTORY"); print("="*70)

i_rows = []
BINS = np.linspace(0,1,11)
BIN_MIDS = (BINS[:-1]+BINS[1:])/2

if trans is not None:
    tn = trans.copy()
    tn['bin'] = pd.cut(tn['turn_frac'], bins=BINS, labels=False, include_lowest=True)
    agg = tn.groupby(['bin','condition','speaker','family'])['dist'].agg(['count','mean','std']).reset_index()
    agg['axis']='novelty'; agg = agg.rename(columns={'count':'n','mean':'value'})
    i_rows.append(agg)

if msurp is not None:
    sn = msurp.copy()
    sn['bin'] = pd.cut(sn['turn_frac'], bins=BINS, labels=False, include_lowest=True)
    agg = sn.groupby(['bin','condition','speaker','family'])['surprise'].agg(['count','mean','std']).reset_index()
    agg['axis']='surprise'; agg = agg.rename(columns={'count':'n','mean':'value'})
    i_rows.append(agg)

if i_rows:
    I_traj = pd.concat(i_rows, ignore_index=True)
    I_traj['bin_mid'] = I_traj['bin'].map(dict(enumerate(BIN_MIDS)))
    I_traj.to_csv(os.path.join(OUT,'I_trajectory.csv'), index=False)
    print(f"saved I_trajectory.csv  ({len(I_traj)} rows over {len(BIN_MIDS)} bins)")
else:
    I_traj = None
    print("  [skip] no trans or msurp available")

# ===================================================================
# LAYER L — EXTRACTED-IDEA ORIGINALITY (Experiment 2 parity)
# ===================================================================
# Uses the per-participant originality scores from the open-source agentic
# idea-extraction pipeline (os_pipeline.production_run on all 194 convs).
# Three originality measures per participant-round, following Experiment 2
# (Rosenbaum et al., UIST 2026):
#   orig_same  — mean cosine distance to same-condition peer centroids
#   orig_all   — mean cosine distance to all other participant centroids
#   orig_cross — minimum distance to the opposite-condition nearest neighbor
# Here we ALSO run within-subject paired tests (Persona vs GPT) which
# Experiment 2 could not because it was between-subjects.
print("\n"+"="*70); print("LAYER L — EXTRACTED-IDEA ORIGINALITY"); print("="*70)

_orig_path = os.path.join(OUT, 'production', 'participant_originality.csv')
_cat_path  = os.path.join(OUT, 'production', 'categorized_ideas.csv')
if not os.path.exists(_orig_path):
    print(f"  [skip] {_orig_path} not found. Run os_pipeline.production_run --all first.")
else:
    orig = pd.read_csv(_orig_path)
    print(f"loaded {len(orig)} participant-round originality rows")

    # fluency per participant-round = n_ideas after validation
    print("\n-- descriptive by condition (mean ± sd) --")
    for col in ['n_ideas','orig_same','orig_all','orig_cross']:
        g = orig.groupby('condition')[col].agg(['count','mean','std']).round(3)
        print(f'\n  {col}:')
        print(g.to_string())

    # ---- Between-subjects test: Persona vs GPT (Welch t, matches Experiment 2) ----
    print("\n-- between-subjects Welch t (matches Exp 2 paper methodology) --")
    bs_rows = []
    def _welch(a, b, name):
        a = pd.to_numeric(a, errors='coerce').dropna()
        b = pd.to_numeric(b, errors='coerce').dropna()
        if len(a)<5 or len(b)<5: return None
        t,p = stats.ttest_ind(a, b, equal_var=False)
        # Hedges' g (small-sample corrected Cohen's d)
        sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
        d = (a.mean()-b.mean())/sp if sp>0 else np.nan
        # Hedges correction
        J = 1 - 3/(4*(len(a)+len(b))-9)
        g = d*J if not np.isnan(d) else np.nan
        return dict(name=name, n_persona=len(a), n_gpt=len(b),
                    mean_persona=a.mean(), mean_gpt=b.mean(),
                    diff=a.mean()-b.mean(), t=t, p=p, cohen_d=d, hedges_g=g)
    for col in ['n_ideas','orig_same','orig_all','orig_cross']:
        r = _welch(orig.loc[orig.condition=='Persona', col],
                   orig.loc[orig.condition=='GPT', col],
                   f'{col} Persona vs GPT')
        if r: bs_rows.append(r)
    bs_df = pd.DataFrame(bs_rows)
    bs_df.to_csv(os.path.join(OUT,'L_originality_between_subjects.csv'), index=False)
    print(bs_df.to_string(index=False))

    # ---- Within-subject paired test (our advantage over Exp 2) ----
    print("\n-- within-subject paired tests (Persona - GPT per user) --")
    wide_L = orig.pivot_table(index='user', columns='condition',
                              values=['n_ideas','orig_same','orig_all','orig_cross'],
                              aggfunc='first')
    wide_L.columns = [f'{m}__{c}' for m,c in wide_L.columns]
    wide_L = wide_L.reset_index()
    # family for each user (already available via user_persona)
    wide_L = wide_L.merge(user_persona[['User_id','family']], left_on='user', right_on='User_id', how='left').drop(columns=['User_id'])

    ws_rows = []
    for col in ['n_ideas','orig_same','orig_all','orig_cross']:
        cp, cg = f'{col}__Persona', f'{col}__GPT'
        if cp not in wide_L.columns or cg not in wide_L.columns: continue
        r = paired_test(wide_L[cp].astype(float), wide_L[cg].astype(float),
                        f'{col} Persona vs GPT (paired)')
        if r: ws_rows.append(r)
    ws_df = pd.DataFrame(ws_rows)
    ws_df.to_csv(os.path.join(OUT,'L_originality_paired.csv'), index=False)
    print(ws_df.to_string(index=False))

    # ---- By persona family (Divergent / Convergent / Rational / BoundedRational) ----
    print("\n-- by persona family (within-subject Δ = Persona - GPT) --")
    fam_rows = []
    for col in ['n_ideas','orig_same','orig_all','orig_cross']:
        cp, cg = f'{col}__Persona', f'{col}__GPT'
        if cp not in wide_L.columns or cg not in wide_L.columns: continue
        wide_L[f'd_{col}'] = wide_L[cp].astype(float) - wide_L[cg].astype(float)
        for fam, sub in wide_L.groupby('family'):
            v = sub[f'd_{col}'].dropna()
            if len(v)<5: continue
            t,p = stats.ttest_1samp(v, 0)
            fam_rows.append(dict(metric=col, family=fam, n=len(v),
                                 mean_diff=v.mean(), std=v.std(ddof=1), t=t, p=p))
    fam_df = pd.DataFrame(fam_rows).sort_values(['metric','family'])
    fam_df.to_csv(os.path.join(OUT,'L_originality_by_family.csv'), index=False)
    print(fam_df.to_string(index=False))

    # ---- Big-5 correlations with within-subject originality deltas ----
    print("\n-- Big-5 × Δoriginality correlations (Spearman) --")
    users_slim = users.set_index('id')
    for p in pers_cols:
        if p in users_slim.columns:
            wide_L[p] = wide_L['user'].map(users_slim[p])
    b5_rows = []
    for col in ['orig_same','orig_all','orig_cross','n_ideas']:
        dcol = f'd_{col}'
        if dcol not in wide_L.columns: continue
        for trait in pers_cols:
            if trait not in wide_L.columns: continue
            x = wide_L[dcol].astype(float); y = wide_L[trait].astype(float)
            m = (~x.isna())&(~y.isna())
            if m.sum()<15: continue
            rho,pv = stats.spearmanr(x[m], y[m])
            if pv < 0.10:
                b5_rows.append(dict(metric=dcol, trait=trait, n=int(m.sum()), rho=rho, p=pv))
    b5_df = pd.DataFrame(b5_rows).sort_values('p')
    b5_df.to_csv(os.path.join(OUT,'L_originality_big5.csv'), index=False)
    print(b5_df.to_string(index=False) if len(b5_df) else '  (no correlations at p<.10)')

    # ---- Perception bridge: does Δ_originality predict Δ_creativity/Δ_ownership? ----
    print("\n-- perception bridge: Δ_originality × Δ_creativity / Δ_ownership --")
    for src in ['cr_diff','ow_diff']:
        if src in users_slim.columns:
            wide_L[src] = wide_L['user'].map(users_slim[src])
    pb_rows = []
    for col in ['orig_same','orig_all','orig_cross','n_ideas']:
        dcol = f'd_{col}'
        if dcol not in wide_L.columns: continue
        for tgt,lbl in [('cr_diff','Δ creativity'),('ow_diff','Δ ownership')]:
            if tgt not in wide_L.columns: continue
            x = wide_L[dcol].astype(float); y = wide_L[tgt].astype(float)
            m = (~x.isna())&(~y.isna())
            if m.sum()<15: continue
            rho,pv = stats.spearmanr(x[m], y[m])
            pb_rows.append(dict(metric=dcol, target=lbl, n=int(m.sum()), rho=rho, p=pv))
    pb_df = pd.DataFrame(pb_rows).sort_values('p')
    pb_df.to_csv(os.path.join(OUT,'L_originality_perception_bridge.csv'), index=False)
    print(pb_df.to_string(index=False) if len(pb_df) else '  (no rows)')

    # ---- Agent 4 category summary ----
    if os.path.exists(_cat_path):
        cats = pd.read_csv(_cat_path)
        n_clusters = int(cats['category_id'].nunique() - (1 if -1 in cats['category_id'].values else 0))
        print(f"\n-- Agent 4 categorization: {len(cats)} ideas in {n_clusters} clusters "
              f"+ {(cats.category_id==-1).sum()} unclustered --")
        top_clusters = (cats[cats.category_id>=0]
                        .groupby(['category_id','category_name']).size()
                        .reset_index(name='n').sort_values('n', ascending=False).head(10))
        print(top_clusters.to_string(index=False))

# ===================================================================
# FIGURES
# ===================================================================
plt.rcParams.update({'figure.dpi':120, 'savefig.dpi':160, 'font.size':10})

# Fig 1: paired questionnaire
fig, axes = plt.subplots(1,2, figsize=(9,4))
for ax, (a,b,ti) in zip(axes, [('creativity_gpt','creativity_persona','Creativity support'),
                                ('ownership_gpt','ownership_persona','Ownership')]):
    uu = users[[a,b]].dropna().astype(float)
    for _, r in uu.iterrows():
        ax.plot([0,1],[r[a],r[b]], color='gray', alpha=0.3, lw=0.6)
    ax.boxplot([uu[a], uu[b]], positions=[0,1], widths=0.35, showfliers=False)
    ax.set_xticks([0,1]); ax.set_xticklabels(['GPT','Persona'])
    ax.set_title(ti)
plt.suptitle('Fig 1. Paired subjective ratings — GPT vs Persona')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig1_paired_questionnaire.png')); plt.close()

# Fig 2: trajectory of user propose/commit by condition
bins = np.linspace(0,1,11)
logs['bin']=pd.cut(logs['turn_frac'], bins=bins, labels=False, include_lowest=True)
logs_m = logs.merge(conv[['conversation_id','condition','family']], on='conversation_id')
tr = (logs_m[logs_m.message_src=='user']
      .groupby(['condition','bin'])[['tag_propose','tag_commit','tag_critique','tag_reframe']]
      .mean().reset_index())
fig, axes = plt.subplots(2,2, figsize=(9,6), sharex=True)
for ax, k in zip(axes.ravel(), ['tag_propose','tag_commit','tag_critique','tag_reframe']):
    for c, sub in tr.groupby('condition'):
        ax.plot(sub['bin']/10, sub[k], marker='o', label=c)
    ax.set_title(k); ax.set_ylim(0,None)
axes[0,0].legend()
plt.suptitle('Fig 2. User-turn stance across normalized conversation progress')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig2_trajectory.png')); plt.close()

# Fig 3: by persona family
tr2 = (logs_m[logs_m.message_src=='user']
       .groupby(['family','bin'])[['tag_propose','tag_commit','tag_critique','tag_reframe']]
       .mean().reset_index())
fig, axes = plt.subplots(2,2, figsize=(9,6), sharex=True)
for ax, k in zip(axes.ravel(), ['tag_propose','tag_commit','tag_critique','tag_reframe']):
    for f, sub in tr2.groupby('family'):
        ax.plot(sub['bin']/10, sub[k], marker='o', label=f)
    ax.set_title(k)
axes[0,0].legend(fontsize=8)
plt.suptitle('Fig 3. User stance trajectory by persona family')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig3_family_trajectory.png')); plt.close()

# Fig 4: ownership vs behavioral authorship scatter
fig, axes = plt.subplots(1,2, figsize=(9,4))
for ax, lbl, cond in zip(axes, ['GPT','Persona'], ['GPT','Persona']):
    x = wide[f'user_word_share__{cond}'].astype(float)
    if cond=='GPT':
        y = wide['user'].map(users.set_index('id')['ownership_gpt'].astype(float))
    else:
        y = wide['user'].map(users.set_index('id')['ownership_persona'].astype(float))
    ax.scatter(x, y, alpha=0.6)
    ax.set_xlabel('User word share (behavioral authorship)')
    ax.set_ylabel('Reported ownership')
    ax.set_title(lbl)
plt.suptitle('Fig 4. Ownership (reported) vs authorship (behavioral)')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig4_ownership_gap.png')); plt.close()

# Fig 5: archetype distribution by condition
fig, ax = plt.subplots(figsize=(7,4))
ct = pd.crosstab(conv['archetype'], conv['condition'], normalize='columns')
ct.plot.bar(ax=ax)
ax.set_ylabel('Share of conversations')
ax.set_title('Fig 5. Interaction archetype share by condition')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig5_archetypes.png')); plt.close()

# Fig 6: distinctiveness paired
fig, ax = plt.subplots(figsize=(5,4))
if 'Persona' in w.columns and 'GPT' in w.columns:
    for _, r in w.dropna().iterrows():
        ax.plot([0,1],[r['GPT'],r['Persona']], color='gray', alpha=0.3, lw=0.6)
    ax.boxplot([w['GPT'].dropna(), w['Persona'].dropna()], positions=[0,1], widths=0.35, showfliers=False)
    ax.set_xticks([0,1]); ax.set_xticklabels(['GPT','Persona'])
    ax.set_title('Fig 6. Portfolio distinctiveness (TF-IDF) — paired')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig6_distinctiveness.png')); plt.close()

# Fig 7: personality x creativity diff
fig, axes = plt.subplots(1,5, figsize=(14,3.2), sharey=True)
for ax, p in zip(axes, pers_cols):
    x=users[p].astype(float); y=users['cr_diff'].astype(float)
    m=(~x.isna())&(~y.isna())
    ax.scatter(x[m],y[m], alpha=0.6)
    if m.sum()>=10:
        r,_ = stats.spearmanr(x[m],y[m])
        ax.set_title(f'{p}\nρ={r:.2f}')
    ax.axhline(0, color='r', lw=0.5)
    ax.set_xlabel(p)
axes[0].set_ylabel('Creativity diff (Persona-GPT)')
plt.suptitle('Fig 7. Personality moderation of persona benefit on creativity support')
plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig7_personality.png')); plt.close()

# Fig 8: consecutive-message novelty — paired (GPT vs Persona) across speakers
if 'mean_dist_all__GPT' in wide.columns and 'mean_dist_all__Persona' in wide.columns:
    fig, axes = plt.subplots(1,3, figsize=(11,4))
    panels = [('mean_dist_all','All messages'),
              ('mean_dist_user','User messages'),
              ('mean_dist_assistant','Assistant messages')]
    for ax, (col, ti) in zip(axes, panels):
        cg, cp = f'{col}__GPT', f'{col}__Persona'
        if cg not in wide.columns or cp not in wide.columns: continue
        uu = wide[[cg,cp]].dropna().astype(float)
        for _, r in uu.iterrows():
            ax.plot([0,1],[r[cg],r[cp]], color='gray', alpha=0.3, lw=0.6)
        ax.boxplot([uu[cg], uu[cp]], positions=[0,1], widths=0.35, showfliers=False)
        ax.set_xticks([0,1]); ax.set_xticklabels(['GPT','Persona'])
        ax.set_title(ti)
        ax.set_ylabel('Mean consec. cosine distance')
    plt.suptitle('Fig 8. Consecutive-message novelty (SBERT) — paired GPT vs Persona')
    plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig8_consec_novelty_paired.png')); plt.close()

# Fig 9: grouped bars — family x condition, faceted by speaker
_desc_path = os.path.join(OUT,'E_consec_novelty_by_group.csv')
if os.path.exists(_desc_path):
    desc = pd.read_csv(_desc_path)
    fams_order = [f for f in ['Divergent','Convergent','Rational','BoundedRational']
                  if f in desc['family'].dropna().unique()]
    fig, axes = plt.subplots(1,2, figsize=(11,4.5), sharey=True)
    for ax, sp in zip(axes, ['user','assistant']):
        sub = desc[desc['speaker']==sp]
        x = np.arange(len(fams_order)); width=0.38
        for i, cond in enumerate(['GPT','Persona']):
            means, sems = [], []
            for f in fams_order:
                cell = sub[(sub.family==f)&(sub.condition==cond)]
                means.append(cell['mean'].values[0] if len(cell) else np.nan)
                sems.append(cell['sem'].values[0] if len(cell) else np.nan)
            ax.bar(x + (i-0.5)*width, means, width, yerr=sems, capsize=4, label=cond, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(fams_order, rotation=15)
        ax.set_title(f'{sp} messages')
        ax.set_ylabel('Mean consec. cosine distance')
    axes[0].legend()
    plt.suptitle('Fig 9. Consecutive-message novelty by persona family x condition')
    plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig9_consec_novelty_by_family.png')); plt.close()

# Fig 10: surprise — paired (GPT vs Persona) across speakers
if 'mean_surp_all__GPT' in wide.columns and 'mean_surp_all__Persona' in wide.columns:
    fig, axes = plt.subplots(1,3, figsize=(11,4))
    panels = [('mean_surp_all','All messages'),
              ('mean_surp_user','User messages'),
              ('mean_surp_assistant','Assistant messages')]
    for ax, (col, ti) in zip(axes, panels):
        cg, cp = f'{col}__GPT', f'{col}__Persona'
        if cg not in wide.columns or cp not in wide.columns: continue
        uu = wide[[cg,cp]].dropna().astype(float)
        for _, r in uu.iterrows():
            ax.plot([0,1],[r[cg],r[cp]], color='gray', alpha=0.3, lw=0.6)
        ax.boxplot([uu[cg], uu[cp]], positions=[0,1], widths=0.35, showfliers=False)
        ax.set_xticks([0,1]); ax.set_xticklabels(['GPT','Persona'])
        ax.set_title(ti)
        ax.set_ylabel('Mean per-token NLL (GPT-2)')
    plt.suptitle('Fig 10. Message surprise (GPT-2) — paired GPT vs Persona')
    plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig10_surprise_paired.png')); plt.close()

# Fig 11: surprise grouped bars — family x condition, faceted by speaker
_sdesc_path = os.path.join(OUT,'F_surprise_by_group.csv')
if os.path.exists(_sdesc_path):
    sdesc = pd.read_csv(_sdesc_path)
    fams_order = [f for f in ['Divergent','Convergent','Rational','BoundedRational']
                  if f in sdesc['family'].dropna().unique()]
    fig, axes = plt.subplots(1,2, figsize=(11,4.5), sharey=True)
    for ax, sp in zip(axes, ['user','assistant']):
        sub = sdesc[sdesc['speaker']==sp]
        x = np.arange(len(fams_order)); width=0.38
        for i, cond in enumerate(['GPT','Persona']):
            means, sems = [], []
            for f in fams_order:
                cell = sub[(sub.family==f)&(sub.condition==cond)]
                means.append(cell['mean'].values[0] if len(cell) else np.nan)
                sems.append(cell['sem'].values[0] if len(cell) else np.nan)
            ax.bar(x + (i-0.5)*width, means, width, yerr=sems, capsize=4, label=cond, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(fams_order, rotation=15)
        ax.set_title(f'{sp} messages')
        ax.set_ylabel('Mean per-token NLL (GPT-2)')
    axes[0].legend()
    plt.suptitle('Fig 11. Surprise by persona family x condition')
    plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig11_surprise_by_family.png')); plt.close()

# Fig 12: behavior-delta x Big-5 heatmap (only for behavior bases that exist)
if len(behav_bases) and any(p in wide.columns for p in pers_cols):
    import numpy.ma as ma
    mat = np.full((len(behav_bases), len(pers_cols)), np.nan)
    pmat = np.full_like(mat, np.nan)
    for i,b in enumerate(behav_bases):
        for j,p in enumerate(pers_cols):
            if p not in wide.columns: continue
            x = wide[f'd_{b}'].astype(float); y = wide[p].astype(float)
            m = (~x.isna())&(~y.isna())
            if m.sum()<10: continue
            r,pv = stats.spearmanr(x[m], y[m])
            mat[i,j]=r; pmat[i,j]=pv
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    ax.set_xticks(range(len(pers_cols))); ax.set_xticklabels(pers_cols, rotation=25, ha='right')
    ax.set_yticks(range(len(behav_bases))); ax.set_yticklabels([f'd_{b}' for b in behav_bases])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i,j]):
                star = '*' if pmat[i,j]<0.05 else ''
                ax.text(j, i, f'{mat[i,j]:+.2f}{star}', ha='center', va='center',
                        fontsize=8, color='k')
    plt.colorbar(im, ax=ax, label='Spearman rho')
    ax.set_title('Fig 12. Within-user behavior delta (Persona-GPT) x Big-5')
    plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig12_big5_x_behavior.png')); plt.close()

# Fig 13: behavior delta vs perception delta scatter grid
if 'cr_diff' in wide.columns and len(behav_bases):
    cols = [b for b in behav_bases if b in ['mean_dist_all','mean_surp_all']]
    if cols:
        fig, axes = plt.subplots(len(cols), 2, figsize=(8, 3.2*len(cols)), squeeze=False)
        for i,b in enumerate(cols):
            for j,(t,lbl) in enumerate([('cr_diff','Creativity diff'),('ow_diff','Ownership diff')]):
                ax = axes[i,j]
                x = wide[f'd_{b}'].astype(float); y = wide[t].astype(float)
                m = (~x.isna())&(~y.isna())
                ax.scatter(x[m], y[m], alpha=0.6)
                if m.sum()>=10:
                    rho,pv = stats.spearmanr(x[m],y[m])
                    ax.set_title(f'd_{b}  vs  {lbl}\nrho={rho:+.2f} p={pv:.3g} n={m.sum()}')
                ax.axhline(0,color='r',lw=0.5); ax.axvline(0,color='r',lw=0.5)
                ax.set_xlabel(f'd_{b} (Persona-GPT)')
                ax.set_ylabel(lbl)
        plt.suptitle('Fig 13. Behavior delta vs perception delta')
        plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig13_behavior_vs_perception.png')); plt.close()

# Fig 14: transition-type paired (LAYER H)
_hpath = os.path.join(OUT,'H_transition_type_paired.csv')
if os.path.exists(_hpath):
    Hd = pd.read_csv(_hpath)
    Hd = Hd[Hd['family']=='all'].copy()
    if len(Hd):
        fig, ax = plt.subplots(figsize=(8,4.5))
        tts = Hd['transition_type'].tolist()
        x = np.arange(len(tts)); width=0.38
        gpt_means = Hd['mean_2'].values; per_means = Hd['mean_1'].values
        gpt_se = Hd['sd_diff'].values/np.sqrt(Hd['n'].values)
        ax.bar(x-width/2, gpt_means, width, label='GPT', alpha=0.85)
        ax.bar(x+width/2, per_means, width, label='Persona', alpha=0.85)
        for i,row in Hd.reset_index(drop=True).iterrows():
            star = '*' if row['p_t']<0.05 else ''
            ymax = max(row['mean_1'], row['mean_2'])
            ax.text(i, ymax*1.02, f"p={row['p_t']:.2g}{star}", ha='center', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(tts, rotation=15)
        ax.set_ylabel('Mean cosine distance')
        ax.set_title('Fig 14. Consecutive-message distance by transition type - paired')
        ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig14_transition_types.png')); plt.close()

# Fig 15: trajectory (novelty & surprise vs turn_frac)
_ipath = os.path.join(OUT,'I_trajectory.csv')
if os.path.exists(_ipath):
    Id = pd.read_csv(_ipath)
    fig, axes = plt.subplots(2,2, figsize=(11,7), sharex=True)
    for row_i, axis_name in enumerate(['novelty','surprise']):
        for col_j, sp in enumerate(['user','assistant']):
            ax = axes[row_i, col_j]
            sub = Id[(Id['axis']==axis_name)&(Id['speaker']==sp)]
            if len(sub)==0: continue
            agg = sub.groupby(['bin_mid','condition'])['value'].mean().reset_index()
            for cond, g in agg.groupby('condition'):
                ax.plot(g['bin_mid'], g['value'], marker='o', label=cond)
            ax.set_title(f'{axis_name} - {sp} messages')
            ax.set_xlabel('turn_frac (normalized position)')
            ax.set_ylabel('mean '+axis_name)
    axes[0,0].legend()
    plt.suptitle('Fig 15. Novelty & surprise trajectories across conversation')
    plt.tight_layout(); plt.savefig(os.path.join(FIG,'fig15_trajectory.png')); plt.close()

# Save master conv + users
conv.to_csv(os.path.join(OUT,'master_conversations.csv'), index=False)
users.to_csv(os.path.join(OUT,'master_users.csv'), index=False)
wide.to_csv(os.path.join(OUT,'master_wide.csv'), index=False)
print("\nDONE. Outputs in", OUT, "and", FIG)
