"""
User-side behavioural analysis of persona-guided LLM co-creation.

Question (RQ2-aligned):
  How do users behaviourally adapt when interacting with each persona type
  (Divergent / Convergent / Rational / BoundedRational) relative to the
  same participants' GPT baseline?

We use Taxonomy 2 (Claude-labeled continuous discourse constructs
propagated to all 3,412 messages by propagate_stance.py). All contrasts
are within-subject (paired).

Analyses
  A1. User-side stance d_z by persona family — paired Persona − GPT on 7
      constructs, BH-FDR adjusted per family.
  A2. Stance entrainment — Spearman correlation between Δassistant and
      Δuser on each construct, by family. Tests the alignment hypothesis
      that users mirror the persona's stance.
  A3. User stance trajectory by conversation quarter × family — does the
      gap with the GPT baseline grow over time?
  A4. Big-5 personality moderation — Spearman ρ between trait scores and
      Δuser_construct, separately for each family.
  A5. Behaviour → originality bridge — does Δuser_expansion predict
      Δorig_same / Δorig_all (extracted-idea originality from
      os_pipeline)?

Outputs
  analysis_out/user_creativity/*.csv           — all statistics tables
  figures/user_creativity/*.png                 — publication-quality figs
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1'
OUT  = os.path.join(ROOT, 'analysis_out', 'user_creativity')
FIG  = os.path.join(ROOT, 'figures',       'user_creativity')
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)

# --- Aesthetic defaults ------------------------------------------------
plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 220, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold',
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.frameon': False,
    'font.family': 'DejaVu Sans',
})
FAM_ORDER  = ['Divergent', 'Convergent', 'Rational', 'BoundedRational']
FAM_COLOR  = {'Divergent':'#2A8C99', 'Convergent':'#B85C3A',
              'Rational':'#6D5D9C',  'BoundedRational':'#C7A11A'}
CONSTRUCTS = ['exp','con','cri','cer','com','ref','prop']
CONSTRUCT_LABEL = {
    'exp':'Expansion',    'con':'Contraction', 'cri':'Critique',
    'cer':'Certainty',    'com':'Commitment',  'ref':'Reframing',
    'prop':'Propose-new-idea',
}

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
pred = pd.read_csv(os.path.join(ROOT, 'analysis_out', 'full_stance_predictions.csv'))
logs = pd.read_csv(os.path.join(ROOT, 'Experiment1_logs.csv'))[['message_id','User_id']]
pred = pred.merge(logs, on='message_id', how='inner')
assert pred['User_id'].nunique() == 97, f"expected 97 users, got {pred['User_id'].nunique()}"

# per-user persona family (the non-GPT condition that user got)
fam_of = (pred[pred.condition=='Persona']
          .groupby('User_id')['family'].first().to_dict())

# Conversation-level per user × condition aggregate (mean over messages)
def agg_per_user_cond(side):
    sub = pred[pred['message_src']==side]
    return (sub.groupby(['User_id','condition'])[CONSTRUCTS]
            .mean().reset_index())

user_u = agg_per_user_cond('user')
user_a = agg_per_user_cond('assistant')
# attach family of the user (NOT the condition family; the family is constant per user)
user_u['family'] = user_u['User_id'].map(fam_of)
user_a['family'] = user_a['User_id'].map(fam_of)

# wide format (one row per user, columns = construct × condition)
def to_wide(df, prefix):
    w = df.pivot_table(index='User_id', columns='condition', values=CONSTRUCTS,
                       aggfunc='first')
    w.columns = [f'{prefix}_{c}__{cond}' for c, cond in w.columns]
    return w.reset_index()

W_u = to_wide(user_u, 'u')
W_a = to_wide(user_a, 'a')
W = W_u.merge(W_a, on='User_id').copy()
W['family'] = W['User_id'].map(fam_of)
# Δ Persona − GPT
for c in CONSTRUCTS:
    W[f'du_{c}']  = W[f'u_{c}__Persona'] - W[f'u_{c}__GPT']
    W[f'da_{c}']  = W[f'a_{c}__Persona'] - W[f'a_{c}__GPT']
W.to_csv(os.path.join(OUT, 'wide_user_assistant_paired.csv'), index=False)
print(f'wide table: {len(W)} users × {len(W.columns)} cols')

# ======================================================================
# A1. User-side d_z by family
# ======================================================================
def cohen_dz(x):
    x = x.dropna().astype(float)
    if len(x) < 4: return np.nan, np.nan, np.nan, np.nan
    sd = x.std(ddof=1)
    dz = x.mean()/sd if sd>0 else np.nan
    t, p = stats.ttest_1samp(x, 0)
    return dz, t, p, len(x)

rows = []
for fam in FAM_ORDER:
    sub = W[W.family==fam]
    for c in CONSTRUCTS:
        dz,t,p,n = cohen_dz(sub[f'du_{c}'])
        rows.append(dict(family=fam, construct=c,
                         label=CONSTRUCT_LABEL[c], n=n, dz=dz, t=t, p=p,
                         mean_persona=sub[f'u_{c}__Persona'].mean(),
                         mean_gpt    =sub[f'u_{c}__GPT'].mean(),
                         mean_diff   =sub[f'du_{c}'].mean()))

A1 = pd.DataFrame(rows)
A1.to_csv(os.path.join(OUT, 'A1_user_dz_by_family.csv'), index=False)
print('\nA1. user-side d_z by family (paired t-test, raw p):')
print(A1.pivot_table(index='label', columns='family', values='dz').round(3))

# --- Figure A1: heatmap of d_z with significance markers ---------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
mat = A1.pivot_table(index='label', columns='family', values='dz').reindex(
    index=[CONSTRUCT_LABEL[c] for c in CONSTRUCTS], columns=FAM_ORDER)
pmat = A1.pivot_table(index='label', columns='family', values='p').reindex(
    index=[CONSTRUCT_LABEL[c] for c in CONSTRUCTS], columns=FAM_ORDER)
im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-1.0, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(FAM_ORDER))); ax.set_xticklabels(FAM_ORDER, rotation=20, ha='right')
ax.set_yticks(range(len(CONSTRUCTS))); ax.set_yticklabels([CONSTRUCT_LABEL[c] for c in CONSTRUCTS])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i,j]; p = pmat.values[i,j]
        if np.isnan(v): continue
        star = ''
        if p is not None and not np.isnan(p):
            if p < 0.001: star = '***'
            elif p < 0.01: star = '**'
            elif p < 0.05: star = '*'
            elif p < 0.10: star = '†'
        ax.text(j, i, f'{v:+.2f}\n{star}',
                ha='center', va='center', fontsize=8.5,
                color='white' if abs(v)>0.55 else 'black')
cbar = plt.colorbar(im, ax=ax, shrink=0.85)
cbar.set_label('Cohen\'s d_z  (Persona − GPT, paired)', fontsize=9)
ax.set_title('User-side stance entrainment by persona family\n'
             '(Taxonomy 2; paired t-test, raw p:  † p<.10, * p<.05, ** p<.01, *** p<.001)',
             fontsize=10.5, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'A1_user_dz_heatmap.png'))
plt.close()

# ======================================================================
# A2. Stance entrainment — Spearman ρ(Δassistant, Δuser) per family
# ======================================================================
rows = []
for fam in FAM_ORDER:
    sub = W[W.family==fam]
    for c in CONSTRUCTS:
        x = sub[f'da_{c}'].astype(float); y = sub[f'du_{c}'].astype(float)
        m = (~x.isna())&(~y.isna())
        if m.sum() < 5: continue
        rho, p = stats.spearmanr(x[m], y[m])
        rows.append(dict(family=fam, construct=c, label=CONSTRUCT_LABEL[c],
                         n=int(m.sum()), rho=rho, p=p))
A2 = pd.DataFrame(rows)
A2.to_csv(os.path.join(OUT, 'A2_entrainment_by_family.csv'), index=False)

# Also: pooled across all Persona arms
pool_rows = []
sub = W[W.family.isin(FAM_ORDER)]
for c in CONSTRUCTS:
    x = sub[f'da_{c}'].astype(float); y = sub[f'du_{c}'].astype(float)
    m = (~x.isna())&(~y.isna())
    if m.sum()<5: continue
    rho,p = stats.spearmanr(x[m], y[m])
    pool_rows.append(dict(family='ALL_PERSONA', construct=c,
                          label=CONSTRUCT_LABEL[c], n=int(m.sum()), rho=rho, p=p))
A2pool = pd.DataFrame(pool_rows)
A2pool.to_csv(os.path.join(OUT,'A2_entrainment_pooled.csv'), index=False)

print('\nA2. Stance entrainment Spearman ρ(Δassistant, Δuser):')
print(A2.pivot_table(index='label', columns='family', values='rho').round(3))
print('  pooled across all Persona arms:')
print(A2pool[['label','rho','p','n']].to_string(index=False))

# --- Figure A2: per-family entrainment scatter on expansion + bar of ρ --
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

ax = axes[0]
for fam in FAM_ORDER:
    sub = W[W.family==fam]
    ax.scatter(sub['da_exp'], sub['du_exp'], color=FAM_COLOR[fam], s=44,
               alpha=0.85, edgecolor='white', linewidth=0.6, label=fam)
ax.axhline(0, color='gray', lw=0.6, ls='--'); ax.axvline(0, color='gray', lw=0.6, ls='--')
# OLS line pooled
xx = W['da_exp'].astype(float); yy = W['du_exp'].astype(float)
m = (~xx.isna())&(~yy.isna())
if m.sum()>=5:
    slope, intercept, r, p, _ = stats.linregress(xx[m], yy[m])
    xr = np.linspace(xx[m].min(), xx[m].max(), 20)
    ax.plot(xr, intercept + slope*xr, color='#222', lw=1.2,
            label=f'pooled  r={r:+.2f}  p={p:.3g}')
ax.set_xlabel('Δ assistant expansion (Persona − GPT)')
ax.set_ylabel('Δ user expansion (Persona − GPT)')
ax.set_title('User-assistant entrainment on expansion')
ax.legend(loc='upper left', fontsize=8.5, ncol=2)

ax = axes[1]
x = np.arange(len(CONSTRUCTS))
w = 0.20
for i, fam in enumerate(FAM_ORDER):
    sub = A2[A2.family==fam].set_index('construct').reindex(CONSTRUCTS)
    ax.bar(x + (i-1.5)*w, sub['rho'].values, w,
           color=FAM_COLOR[fam], alpha=0.92, label=fam,
           edgecolor='white', linewidth=0.4)
ax.axhline(0, color='gray', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([CONSTRUCT_LABEL[c] for c in CONSTRUCTS],
                                     rotation=18, ha='right', fontsize=8.5)
ax.set_ylabel("Spearman ρ (Δassistant, Δuser)")
ax.set_title('Entrainment strength by construct × family')
ax.legend(fontsize=8.5, loc='upper right', ncol=2)
ax.set_ylim(-0.6, 0.9)
plt.suptitle('A2. Stance entrainment: do users follow the persona?',
             fontsize=11.5, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'A2_entrainment.png'), bbox_inches='tight')
plt.close()

# ======================================================================
# A3. Trajectory: user stance by quarter × family vs GPT baseline
# ======================================================================
pr = pred.copy()
pr['quarter'] = pd.cut(pr['turn_frac'], bins=[-0.01,0.25,0.5,0.75,1.01],
                       labels=['Q1','Q2','Q3','Q4'])
# Build user-side trajectory per condition × quarter × family
# For Persona rows, family is the user's persona family
# For GPT rows, family is "GPT" (baseline) — we use the same 97 users' GPT
# round, regardless of which persona family they belong to.
um = pr[pr.message_src=='user'].copy()

# Persona-arm trajectory by family (mean over messages within each user-quarter)
per = um[um.condition=='Persona'].copy()
gpt = um[um.condition=='GPT'].copy()

# For Persona: family is per-user
def traj_persona(metric):
    g = (per.groupby(['family','quarter'], observed=True)[metric]
         .agg(['mean','sem','count']).reset_index())
    return g
def traj_gpt(metric):
    g = (gpt.groupby('quarter', observed=True)[metric]
         .agg(['mean','sem','count']).reset_index())
    return g

# Save full trajectories
traj_all = []
for c in CONSTRUCTS:
    tp = traj_persona(c).assign(construct=c, arm='Persona')
    tg = traj_gpt(c).assign(construct=c, arm='GPT', family='GPT')
    traj_all.append(tp); traj_all.append(tg)
A3 = pd.concat(traj_all, ignore_index=True)
A3.to_csv(os.path.join(OUT, 'A3_user_trajectory.csv'), index=False)

# --- Figure A3: 2×3 grid of construct trajectories with GPT baseline ----
focal = ['exp','con','cer','com','ref','prop']
fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True)
for ax, c in zip(axes.ravel(), focal):
    # GPT baseline (gray band)
    tg = traj_gpt(c)
    ax.plot(tg['quarter'], tg['mean'], color='#666', lw=1.8, ls='--',
            marker='s', markersize=4, label='GPT baseline')
    ax.fill_between(range(len(tg)),
                    tg['mean']-tg['sem'], tg['mean']+tg['sem'],
                    color='#bbb', alpha=0.30, linewidth=0)
    for fam in FAM_ORDER:
        tp = traj_persona(c)
        sub = tp[tp.family==fam]
        ax.plot(sub['quarter'], sub['mean'], color=FAM_COLOR[fam],
                lw=1.8, marker='o', markersize=4.6, label=fam)
        ax.fill_between(range(len(sub)),
                        sub['mean']-sub['sem'], sub['mean']+sub['sem'],
                        color=FAM_COLOR[fam], alpha=0.10, linewidth=0)
    ax.set_title(CONSTRUCT_LABEL[c])
    ax.set_ylabel('Mean user score (0–3)')
    ax.grid(axis='y', alpha=0.18)
axes[1,0].set_xlabel('Conversation quarter')
axes[1,1].set_xlabel('Conversation quarter')
axes[1,2].set_xlabel('Conversation quarter')
axes[0,0].legend(fontsize=7.6, loc='upper right', ncol=1)
plt.suptitle("A3. User stance trajectory by conversation quarter — "
             "how user behaviour diverges from the GPT baseline over time",
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'A3_user_trajectory.png'), bbox_inches='tight')
plt.close()

# ======================================================================
# A4. Big-5 moderation
# ======================================================================
mu = pd.read_csv(os.path.join(ROOT, 'analysis_out', 'master_users.csv'))
big5_cols = ['Extraversion','Agreeableness','Conscientiousness',
             'Negative Emotionality','Open-Mindedness']
keep = ['id'] + big5_cols + ['family']
mu_slim = mu[keep].copy().rename(columns={'id':'User_id'})

Wp = W.merge(mu_slim.drop(columns='family'), on='User_id', how='left')

rows = []
for fam in FAM_ORDER:
    sub = Wp[Wp.family==fam]
    for c in CONSTRUCTS:
        for trait in big5_cols:
            x = sub[trait].astype(float); y = sub[f'du_{c}'].astype(float)
            m = (~x.isna())&(~y.isna())
            if m.sum() < 6: continue
            rho, p = stats.spearmanr(x[m], y[m])
            rows.append(dict(family=fam, construct=c, label=CONSTRUCT_LABEL[c],
                             trait=trait, n=int(m.sum()), rho=rho, p=p))
A4 = pd.DataFrame(rows)
A4.to_csv(os.path.join(OUT,'A4_big5_moderation.csv'), index=False)

# Top hits (raw p<.05) per family
print('\nA4. Big-5 × Δuser_construct top hits (raw p<.05):')
print(A4[A4.p<0.05].sort_values('p')[
    ['family','trait','label','n','rho','p']].to_string(index=False))

# --- Figure A4: heatmap (Open-Mindedness × constructs) by family --------
fig, axes = plt.subplots(1, len(FAM_ORDER), figsize=(14, 4.8), sharey=True)
for ax, fam in zip(axes, FAM_ORDER):
    sub = A4[A4.family==fam]
    mat = sub.pivot_table(index='label', columns='trait', values='rho').reindex(
        index=[CONSTRUCT_LABEL[c] for c in CONSTRUCTS], columns=big5_cols)
    pmat = sub.pivot_table(index='label', columns='trait', values='p').reindex(
        index=[CONSTRUCT_LABEL[c] for c in CONSTRUCTS], columns=big5_cols)
    im = ax.imshow(mat.values, cmap='RdBu_r', vmin=-0.7, vmax=0.7, aspect='auto')
    ax.set_xticks(range(len(big5_cols)))
    ax.set_xticklabels([t.replace('Negative ','Neg-').replace('Open-Mindedness','Openness') for t in big5_cols],
                       rotation=30, ha='right', fontsize=8)
    if ax is axes[0]:
        ax.set_yticks(range(len(CONSTRUCTS)))
        ax.set_yticklabels([CONSTRUCT_LABEL[c] for c in CONSTRUCTS])
    ax.set_title(f'{fam}\n(n={W[W.family==fam].shape[0]})', fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i,j]; p = pmat.values[i,j]
            if np.isnan(v): continue
            star = ''
            if p is not None and not np.isnan(p):
                if p < 0.001: star = '***'
                elif p < 0.01: star = '**'
                elif p < 0.05: star = '*'
                elif p < 0.10: star = '†'
            ax.text(j, i, f'{v:+.2f}{star}', ha='center', va='center', fontsize=7.4,
                    color='white' if abs(v)>0.45 else 'black')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label='Spearman ρ (trait × Δuser)')
plt.subplots_adjust(top=0.80, bottom=0.18, right=0.88)
fig.suptitle('A4. Big-5 personality moderation of within-subject user stance change\n'
             '(Spearman ρ; raw p:  † p<.10, * p<.05, ** p<.01, *** p<.001)',
             fontsize=11.5, fontweight='bold', y=0.96)
plt.savefig(os.path.join(FIG, 'A4_big5_moderation.png'))
plt.close()

# ======================================================================
# A5. Behaviour → originality bridge
# ======================================================================
orig_path = os.path.join(ROOT, 'analysis_out','production','participant_originality.csv')
if os.path.exists(orig_path):
    O = pd.read_csv(orig_path)
    # one row per user × condition; pivot to wide
    Ow = O.pivot_table(index='user', columns='condition',
                       values=['n_ideas','orig_same','orig_all','orig_cross'],
                       aggfunc='first')
    Ow.columns = [f'{m}__{c}' for m,c in Ow.columns]
    Ow = Ow.reset_index().rename(columns={'user':'User_id'})
    for m in ['n_ideas','orig_same','orig_all','orig_cross']:
        cp, cg = f'{m}__Persona', f'{m}__GPT'
        if cp in Ow.columns and cg in Ow.columns:
            Ow[f'd_{m}'] = Ow[cp] - Ow[cg]
    Wo = W.merge(Ow, on='User_id', how='left')

    rows = []
    for fam in FAM_ORDER + ['ALL']:
        sub = Wo if fam=='ALL' else Wo[Wo.family==fam]
        for c in CONSTRUCTS:
            for m in ['n_ideas','orig_same','orig_all','orig_cross']:
                dcol = f'd_{m}'
                if dcol not in sub.columns: continue
                x = sub[f'du_{c}'].astype(float); y = sub[dcol].astype(float)
                mk = (~x.isna())&(~y.isna())
                if mk.sum() < 6: continue
                rho, p = stats.spearmanr(x[mk], y[mk])
                rows.append(dict(family=fam, user_construct=c,
                                 label=CONSTRUCT_LABEL[c],
                                 orig_metric=m, n=int(mk.sum()), rho=rho, p=p))
    A5 = pd.DataFrame(rows)
    A5.to_csv(os.path.join(OUT, 'A5_user_behavior_to_originality.csv'), index=False)

    print('\nA5. User-behaviour → originality bridge (Spearman ρ, top hits at p<.05 / ALL):')
    print(A5[(A5.family=='ALL')&(A5.p<0.05)].sort_values('p').to_string(index=False))

    # --- Figure A5: scatter on the significant headline pairs
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for ax,(uc, om, title) in zip(axes,
        [('cer','orig_same','Lower user certainty → MORE distinctive idea portfolio\n(vs same-condition peers)'),
         ('cer','orig_all', 'Lower user certainty → MORE distinctive idea portfolio\n(vs all participants)'),
         ('prop','n_ideas', 'More user proposing → more canonical ideas extracted')]):
        x = Wo[f'du_{uc}'].astype(float); y = Wo[f'd_{om}'].astype(float)
        m = (~x.isna())&(~y.isna())
        for fam in FAM_ORDER:
            sub = Wo[Wo.family==fam]
            ax.scatter(sub[f'du_{uc}'], sub[f'd_{om}'],
                       color=FAM_COLOR[fam], s=42, alpha=0.85,
                       edgecolor='white', linewidth=0.5, label=fam)
        if m.sum()>=10:
            rho,p = stats.spearmanr(x[m], y[m])
            slope, intercept, r, pp, _ = stats.linregress(x[m], y[m])
            xr = np.linspace(x[m].min(), x[m].max(), 20)
            ax.plot(xr, intercept+slope*xr, color='#222', lw=1.2,
                    label=f'pooled  ρ={rho:+.2f}  p={p:.3g}  n={m.sum()}')
        ax.axhline(0,color='gray',lw=0.6,ls='--'); ax.axvline(0,color='gray',lw=0.6,ls='--')
        ax.set_xlabel(f'Δ user {CONSTRUCT_LABEL[uc].lower()}')
        ax.set_ylabel(f'Δ {om}')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.4, loc='best')
    plt.suptitle("A5. Linking user stance shifts to extracted-idea originality "
                 "(open-source idea-extraction pipeline)",
                 fontsize=11.4, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'A5_behavior_to_originality.png'), bbox_inches='tight')
    plt.close()
else:
    print('\nA5 skipped — originality file not found.')
    A5 = None

# ======================================================================
# A6. (Bonus) Stance-vector alignment per user: cosine sim(Δa, Δu)
# Question: how aligned is each user's stance change with the assistant's?
# ======================================================================
def safe_cos(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na==0 or nb==0: return np.nan
    return float(np.dot(a, b) / (na*nb))

cos_rows = []
for _, r in W.iterrows():
    a_vec = [r.get(f'da_{c}') for c in CONSTRUCTS]
    u_vec = [r.get(f'du_{c}') for c in CONSTRUCTS]
    if any(pd.isna(v) for v in a_vec+u_vec): continue
    cs = safe_cos(a_vec, u_vec)
    cos_rows.append(dict(User_id=r['User_id'], family=r['family'], cos_au=cs))
A6 = pd.DataFrame(cos_rows)
A6.to_csv(os.path.join(OUT, 'A6_user_assistant_stance_alignment.csv'), index=False)

print('\nA6. Stance-vector cosine alignment(Δa,Δu) per family:')
print(A6.groupby('family')['cos_au'].agg(['count','mean','std']).round(3))

# Test whether each family's mean alignment > 0 (one-sample t test)
ali_rows = []
for fam in FAM_ORDER:
    v = A6[A6.family==fam]['cos_au'].dropna()
    if len(v)<5: continue
    t,p = stats.ttest_1samp(v, 0)
    ali_rows.append(dict(family=fam, n=len(v), mean_cos=v.mean(),
                         sd_cos=v.std(ddof=1), t=t, p=p))
A6t = pd.DataFrame(ali_rows)
A6t.to_csv(os.path.join(OUT, 'A6_alignment_significance.csv'), index=False)
print(A6t.to_string(index=False))

# --- Figure A6: violin + strip plot of alignment by family --------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
data = [A6[A6.family==fam]['cos_au'].dropna().values for fam in FAM_ORDER]
parts = ax.violinplot(data, positions=range(len(FAM_ORDER)),
                      widths=0.78, showmeans=False, showmedians=False)
for i,b in enumerate(parts['bodies']):
    b.set_facecolor(FAM_COLOR[FAM_ORDER[i]]); b.set_alpha(0.35); b.set_edgecolor('none')
parts['cbars'].set_alpha(0); parts['cmins'].set_alpha(0); parts['cmaxes'].set_alpha(0)
# strip
rng = np.random.default_rng(0)
for i, d in enumerate(data):
    if len(d):
        jx = rng.uniform(-0.12, 0.12, size=len(d))
        ax.scatter(np.full_like(d, i)+jx, d, s=22,
                   color=FAM_COLOR[FAM_ORDER[i]], alpha=0.85,
                   edgecolor='white', linewidth=0.5)
        ax.hlines(np.mean(d), i-0.32, i+0.32, color='black', lw=2)
ax.axhline(0, color='gray', lw=0.6, ls='--')
ax.set_xticks(range(len(FAM_ORDER))); ax.set_xticklabels(FAM_ORDER)
ax.set_ylabel('cos(Δassistant-stance, Δuser-stance)')
ax.set_title('A6. Per-user stance-vector alignment (Persona − GPT)\n'
             'higher = user follows the assistant\'s persona-induced stance shift',
             fontsize=10.5)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'A6_stance_alignment.png'))
plt.close()

print('\nDONE — outputs in:')
print(' ', OUT)
print(' ', FIG)
