"""UMAP of user messages (and assistant messages overlaid as context) by persona type."""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT='analysis_out'; FIG='figures'
os.makedirs(FIG, exist_ok=True)

logs = pd.read_csv('Experiment1_logs.csv').sort_values(['conversation_id','message_id']).reset_index(drop=True)
E = np.load(os.path.join(OUT,'msg_embeddings.npy'))

up = (logs.groupby('User_id')['Persona_type']
      .apply(lambda s: [x for x in s.unique() if x!='GPT'][0])
      .reset_index().rename(columns={'Persona_type':'persona_cond'}))
fm = {'Divergent':'Divergent','Convergent':'Convergent','strictly rational':'Rational','bounded rationality':'BoundedRational'}
up['family'] = up['persona_cond'].map(fm)
conv = logs.groupby('conversation_id').agg(user=('User_id','first'), persona_type=('Persona_type','first')).reset_index()
conv['condition'] = np.where(conv['persona_type']=='GPT','GPT','Persona')
conv = conv.merge(up[['User_id','family']], left_on='user', right_on='User_id').drop(columns=['User_id'])
conv['persona_label'] = np.where(conv['condition']=='GPT','GPT', conv['family'])
logs = logs.merge(conv[['conversation_id','condition','family','persona_label']], on='conversation_id')

# subsample challenge-balanced (to avoid challenge confound in the map)
# take all messages but annotate challenge
logs['challenge'] = logs['Corrected Challenge type'].fillna('Unknown')

from umap import UMAP
print('fitting UMAP on all 3412 messages...')
reducer = UMAP(n_neighbors=25, min_dist=0.15, metric='cosine', random_state=0)
X2 = reducer.fit_transform(E)
logs['x'] = X2[:,0]; logs['y'] = X2[:,1]

# 4-panel figure:
# (A) user messages colored by persona_label
# (B) assistant messages colored by persona_label
# (C) both sides, faceted by persona_label, showing user vs asst
# (D) per-family user-only maps faceted
COLORS = {'GPT':'#888888','Divergent':'#2a9d8f','Convergent':'#e76f51',
          'Rational':'#6f62b6','BoundedRational':'#e9c46a'}
labels_order = ['GPT','Divergent','Convergent','Rational','BoundedRational']

fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.15)

# A: user messages
ax = fig.add_subplot(gs[0,0])
u = logs[logs.message_src=='user']
for lbl in labels_order:
    sub = u[u.persona_label==lbl]
    ax.scatter(sub['x'], sub['y'], s=5, alpha=0.35, color=COLORS[lbl], label=f'{lbl} (n={len(sub)})')
ax.set_title('A. User messages — colored by persona type')
ax.legend(markerscale=2, fontsize=8, loc='best'); ax.set_xticks([]); ax.set_yticks([])

# B: assistant messages
ax = fig.add_subplot(gs[0,1])
a = logs[logs.message_src=='assistant']
for lbl in labels_order:
    sub = a[a.persona_label==lbl]
    ax.scatter(sub['x'], sub['y'], s=5, alpha=0.35, color=COLORS[lbl], label=f'{lbl} (n={len(sub)})')
ax.set_title('B. Assistant messages — colored by persona type')
ax.legend(markerscale=2, fontsize=8, loc='best'); ax.set_xticks([]); ax.set_yticks([])

# C: user vs assistant, both sides, single map, colored by speaker
ax = fig.add_subplot(gs[1,0])
for src, c, a_v in [('user','#1f4e79',0.35),('assistant','#b04a33',0.35)]:
    sub = logs[logs.message_src==src]
    ax.scatter(sub['x'], sub['y'], s=4, alpha=a_v, color=c, label=f'{src} (n={len(sub)})')
ax.set_title('C. User vs assistant — speakers occupy distinct regions')
ax.legend(markerscale=2, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

# D: trajectory overlay — connect consecutive user messages within each conversation (sample)
ax = fig.add_subplot(gs[1,1])
np.random.seed(1)
persona_sample = []
for fam_v in ['Divergent','Convergent','Rational','BoundedRational']:
    pool = conv[(conv.condition=='Persona') & (conv.family==fam_v)]
    if len(pool)==0: continue
    persona_sample.append(pool.sample(min(3,len(pool)), random_state=1))
sample_convs = pd.concat(persona_sample, ignore_index=True) if persona_sample else pd.DataFrame()
gpt_sample = conv[conv.condition=='GPT'].sample(3, random_state=1)
for cid in gpt_sample['conversation_id']:
    g = logs[(logs.conversation_id==cid)&(logs.message_src=='user')].sort_values('message_id')
    ax.plot(g['x'], g['y'], color='#888888', alpha=0.5, lw=1)
    ax.scatter(g['x'], g['y'], s=10, color='#888888', alpha=0.7)
seen_fams = set()
for _, row in sample_convs.iterrows():
    cid = row['conversation_id']; fam_v = row['family']
    g = logs[(logs.conversation_id==cid)&(logs.message_src=='user')].sort_values('message_id')
    lbl = fam_v if fam_v not in seen_fams else None
    seen_fams.add(fam_v)
    ax.plot(g['x'], g['y'], color=COLORS[fam_v], alpha=0.7, lw=1.2, label=lbl)
    ax.scatter(g['x'], g['y'], s=12, color=COLORS[fam_v], alpha=0.9)
ax.set_title('D. User-message trajectories (3 convs/family, 3 GPT)')
ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])

# E: per-family small multiples (user-only)
for i, lbl in enumerate(labels_order):
    ax = fig.add_subplot(gs[2, i%2]) if i<2 else None
# cleaner: faceted by persona
fig2, axes = plt.subplots(1, len(labels_order), figsize=(16, 3.5), sharex=True, sharey=True)
u_bg = logs[logs.message_src=='user']
xlim = (u_bg['x'].quantile(0.01), u_bg['x'].quantile(0.99))
ylim = (u_bg['y'].quantile(0.01), u_bg['y'].quantile(0.99))
for ax, lbl in zip(axes, labels_order):
    ax.scatter(u_bg['x'], u_bg['y'], s=3, alpha=0.10, color='#cccccc')
    sub = u_bg[u_bg.persona_label==lbl]
    ax.scatter(sub['x'], sub['y'], s=8, alpha=0.55, color=COLORS[lbl])
    # draw centroid
    cx, cy = sub['x'].mean(), sub['y'].mean()
    ax.plot(cx, cy, marker='X', markersize=12, color='black', markeredgecolor='white', mew=1.5)
    ax.set_title(f'{lbl} (n={len(sub)})')
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_xticks([]); ax.set_yticks([])
fig2.suptitle('User-message density by persona type (black X = persona-cloud centroid)', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG,'figR_umap_user_by_persona.png'), dpi=170, bbox_inches='tight')
plt.close(fig2)

fig.suptitle('UMAP of 384-dim SBERT message embeddings', y=0.995)
plt.figure(fig.number)
plt.savefig(os.path.join(FIG,'figR_umap_overview.png'), dpi=170, bbox_inches='tight')
plt.close(fig)
print('saved figR_umap_overview.png and figR_umap_user_by_persona.png')
