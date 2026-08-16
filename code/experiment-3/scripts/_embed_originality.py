"""Compute conversation-level originality and diversity using multilingual embeddings.

Replicates RQ3 metrics from the Scaffolding Creativity paper, adapted for
conversation as the unit (we have no participant IDs in Exp 3).

Inputs:  experiment3_messages_en.json  (English text via translation)
Models:  intfloat/multilingual-e5-large   (≈2.2 GB, top-tier multilingual)
         OR sentence-transformers/all-mpnet-base-v2 if model isn't cached and we
         want to stay small (English-only post-translation, 420MB).

Strategy:
  1. For each conversation, concatenate user+assistant English text into one
     "idea portfolio" representation.
     [Alternative: also extract per-turn embeddings and aggregate, but for the
      paper's originality metric the per-participant aggregate suffices.]
  2. Compute three originality scores per conversation (paper's measures):
       (1) same_condition_originality: mean cosine distance to peers in same group
       (2) all_originality          : mean cosine distance to ALL conversations
       (3) cross_condition_nn       : nearest-neighbor distance to opposite group
  3. Compute within-conv diversity: mean pairwise cosine distance among that
     conv's user turns.
  4. Welch's t test treatment vs control on each measure.
"""
import json, math, statistics, csv, os
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'

with open(f'{base}/data/experiment3_messages_en.json','r',encoding='utf-8') as f:
    msgs = json.load(f)

# Group messages by conversation
by_conv = defaultdict(list)
for m in msgs:
    by_conv[m['conversation_id']].append(m)
for cid in by_conv:
    by_conv[cid].sort(key=lambda x:(x['timestamp'], x['message_id']))

# Use English text. Pre-translation, 'message_en' = English version (or untouched English)
def conv_text(ms, only_user=False):
    if only_user:
        ms = [m for m in ms if m['message_src']=='user']
    return ' \n '.join(m['message_en'] for m in ms)

# --- model ---
MODEL_NAME = 'intfloat/multilingual-e5-large'  # change to all-mpnet-base-v2 if too slow/big
print(f'Loading {MODEL_NAME}...')
model = SentenceTransformer(MODEL_NAME)
print('Model loaded.')

# --- conversation-level embeddings ---
# multilingual-e5-large requires "passage: " prefix for documents and "query: " for queries
# For symmetric similarity, prefix all with "passage: "
def emb_passages(texts, batch_size=4):
    if MODEL_NAME.startswith('intfloat/multilingual-e5'):
        texts = [f'passage: {t}' for t in texts]
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)

cids = sorted(by_conv.keys())
groups = [by_conv[c][0]['group'] for c in cids]
texts = [conv_text(by_conv[c]) for c in cids]
print(f'Embedding {len(texts)} conversations (full-text)...')
E = emb_passages(texts)  # shape: (n_conv, d)
print(f'Conv embedding shape: {E.shape}')

# Cosine distance matrix (since normalized: dist = 1 - dot)
sim = E @ E.T            # (n,n)
dist = 1.0 - sim
n = len(cids)

def mean_dist(idxs, exclude=None):
    vals = []
    for i in idxs:
        for j in idxs:
            if i == j: continue
            if exclude is not None and j == exclude: continue
            vals.append(dist[i, j])
    return float(np.mean(vals)) if vals else float('nan')

results = []
trt_idx = [i for i,g in enumerate(groups) if g=='treatment']
ctl_idx = [i for i,g in enumerate(groups) if g=='control']

for i, cid in enumerate(cids):
    g = groups[i]
    # same-condition originality: mean dist to other convs in same group
    same_idx = trt_idx if g=='treatment' else ctl_idx
    same_orig = mean_dist([i] + [j for j in same_idx if j!=i], exclude=None) if False else \
                float(np.mean([dist[i,j] for j in same_idx if j!=i])) if len(same_idx)>1 else float('nan')
    # all-participants originality
    all_orig = float(np.mean([dist[i,j] for j in range(n) if j!=i]))
    # cross-condition nearest-neighbor (paper)
    cross_idx = ctl_idx if g=='treatment' else trt_idx
    cross_nn = float(np.min([dist[i,j] for j in cross_idx])) if cross_idx else float('nan')
    results.append({
        'conv_id': cid, 'group': g,
        'same_cond_originality': round(same_orig, 4),
        'all_originality':      round(all_orig, 4),
        'cross_cond_nn':        round(cross_nn, 4),
    })

# --- within-conversation diversity (user turns) ---
# Per the paper: mean pairwise distance among one participant's turns
print('Embedding user turns for within-conv diversity...')
diversity = {}
for cid in cids:
    user_texts = [m['message_en'] for m in by_conv[cid] if m['message_src']=='user' and m['message_en'].strip()]
    if len(user_texts) < 2:
        diversity[cid] = float('nan')
        continue
    Ue = emb_passages(user_texts, batch_size=8)
    Usim = Ue @ Ue.T
    Udist = 1.0 - Usim
    iu, ju = np.triu_indices(len(user_texts), k=1)
    diversity[cid] = float(np.mean(Udist[iu, ju]))

for r in results:
    r['within_conv_diversity'] = round(diversity[r['conv_id']], 4) if not math.isnan(diversity[r['conv_id']]) else ''

# Save
with open(f'{base}/outputs/originality_per_conv.csv','w',encoding='utf-8',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)

# --- group comparisons (Welch + Hedges' g) ---
def welch(a, b):
    a = [x for x in a if isinstance(x,(int,float)) and not (isinstance(x,float) and math.isnan(x))]
    b = [x for x in b if isinstance(x,(int,float)) and not (isinstance(x,float) and math.isnan(x))]
    if len(a)<2 or len(b)<2: return None
    ma,mb = statistics.mean(a), statistics.mean(b)
    va,vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va/len(a)+vb/len(b))
    if se==0: return None
    t = (ma-mb)/se
    df = (va/len(a)+vb/len(b))**2 / ((va/len(a))**2/(len(a)-1) + (vb/len(b))**2/(len(b)-1))
    sp = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb)/(len(a)+len(b)-2))
    d = (ma-mb)/sp if sp else 0
    J = 1 - 3/(4*(len(a)+len(b))-9)
    g = d*J
    # two-sided p via SciPy-free betai (re-using prior implementation)
    from math import lgamma, exp, log
    def betai(a_,b_,x):
        if x<0 or x>1: return float('nan')
        if x==0 or x==1: return 0 if x==0 else 1
        bt = exp(lgamma(a_+b_)-lgamma(a_)-lgamma(b_) + a_*log(x) + b_*log(1-x))
        def cf(a,b,x,maxit=200,eps=3e-7):
            qab=a+b; qap=a+1; qam=a-1
            c=1; d=1-qab*x/qap
            if abs(d)<1e-30: d=1e-30
            d=1/d; h=d
            for m in range(1,maxit+1):
                m2=2*m
                aa=m*(b-m)*x/((qam+m2)*(a+m2))
                d=1+aa*d
                if abs(d)<1e-30: d=1e-30
                c=1+aa/c
                if abs(c)<1e-30: c=1e-30
                d=1/d; h*=d*c
                aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
                d=1+aa*d
                if abs(d)<1e-30: d=1e-30
                c=1+aa/c
                if abs(c)<1e-30: c=1e-30
                d=1/d; delta=d*c; h*=delta
                if abs(delta-1)<eps: break
            return h
        if x < (a_+1)/(a_+b_+2): return bt*cf(a_,b_,x)/a_
        return 1 - bt*cf(b_,a_,1-x)/b_
    p = betai(df/2, 0.5, df/(df + t*t))
    return {'mean_a':ma,'mean_b':mb,'sd_a':math.sqrt(va),'sd_b':math.sqrt(vb),
            'n_a':len(a),'n_b':len(b),'t':t,'df':df,'p':p,'hedges_g':g}

trt_rows = [r for r in results if r['group']=='treatment']
ctl_rows = [r for r in results if r['group']=='control']

print('\n='*0)
print('='*78)
print('ORIGINALITY & DIVERSITY — Treatment vs Control')
print('='*78)
for key in ['same_cond_originality','all_originality','cross_cond_nn','within_conv_diversity']:
    a = [r[key] for r in trt_rows if isinstance(r[key],(int,float))]
    b = [r[key] for r in ctl_rows if isinstance(r[key],(int,float))]
    res = welch(a,b)
    if res:
        print(f'\n{key}:')
        print(f'  Treatment M={res["mean_a"]:.4f}  SD={res["sd_a"]:.4f}  n={res["n_a"]}')
        print(f'  Control   M={res["mean_b"]:.4f}  SD={res["sd_b"]:.4f}  n={res["n_b"]}')
        print(f'  Welch t={res["t"]:.2f}  df={res["df"]:.1f}  p={res["p"]:.4f}  Hedges g={res["hedges_g"]:.2f}')
    else:
        print(f'\n{key}: insufficient data')

print(f'\nWrote: {base}/outputs/originality_per_conv.csv')
