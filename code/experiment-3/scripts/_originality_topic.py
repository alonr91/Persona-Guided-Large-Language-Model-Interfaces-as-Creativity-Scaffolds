# -*- coding: utf-8 -*-
"""Topic-controlled originality for the full-tier set (challenge-aware embeddings).

Problem: with heterogeneous challenges, raw cross-conversation distance is dominated
by topic. Fix: condition on the challenge label.

Embeddings: multilingual-E5-large over each conversation's USER-only English text
(participant's own contribution, as in Experiment 2), mean-pooled per conversation.

Topic-controlled measures (treatment vs control, Welch t + Hedges g):
  1. orig_vs_challenge_centroid  (HEADLINE) — cosine distance from a conversation to
     the leave-one-out centroid of its OWN challenge. High = explores further from the
     typical solution for that challenge. Topic is removed because the reference is the
     same-challenge mean. Pooled across challenges (each score is challenge-relative).
  2. resid_same_cond / resid_all / resid_cross_nn — classic originality measures, but
     computed on challenge-RESIDUALIZED embeddings (x - own-challenge centroid).
  3. within_conv_diversity — mean pairwise distance among a conversation's own user
     turns (already topic-free); treatment vs control.
Also reports the ichilov_rehab_future cell alone (the only well-powered cell:
11 treatment vs 5 control).
"""
import json, csv, math, statistics
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
labels = {int(r['conversation_id']): r['challenge_id']
          for r in csv.DictReader(open(f'{BASE}/outputs/challenge_labels.csv', encoding='utf-8'))}

by = defaultdict(list)
for m in msgs:
    by[m['conversation_id']].append(m)
for c in by:
    by[c].sort(key=lambda x: (x['timestamp'], x['message_id']))

def welch(a, b):
    a = [float(x) for x in a]; b = [float(x) for x in b]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va/na + vb/nb)
    if se == 0: return None
    t = (ma-mb)/se
    df = (va/na+vb/nb)**2 / ((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    sp = math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2))
    d = (ma-mb)/sp if sp else 0
    J = 1 - 3/(4*(na+nb)-9)
    # two-sided p via t survival (normal approx fallback)
    try:
        from statistics import NormalDist
        # Welch-Satterthwaite df; use t-dist approx via normal for simplicity flagged below
    except Exception:
        pass
    return dict(ma=ma, mb=mb, na=na, nb=nb, t=t, df=df, g=d*J)

def pval(t, df):
    # two-sided p-value from t using a numeric incomplete-beta (no scipy dependency)
    x = df/(df+t*t)
    # regularized incomplete beta I_x(df/2, 1/2) via continued fraction
    a, b = df/2, 0.5
    def betacf(x,a,b):
        MAXIT=200; EPS=3e-12; FPMIN=1e-300
        qab=a+b; qap=a+1; qam=a-1; c=1; d=1-qab*x/qap
        if abs(d)<FPMIN: d=FPMIN
        d=1/d; h=d
        for m in range(1,MAXIT):
            m2=2*m
            aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1+aa*d
            if abs(d)<FPMIN: d=FPMIN
            c=1+aa/c
            if abs(c)<FPMIN: c=FPMIN
            d=1/d; h*=d*c
            aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1+aa*d
            if abs(d)<FPMIN: d=FPMIN
            c=1+aa/c
            if abs(c)<FPMIN: c=FPMIN
            d=1/d; de=d*c; h*=de
            if abs(de-1)<EPS: break
        return h
    lbeta=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    if x<(a+1)/(a+b+2):
        ib=math.exp(a*math.log(x)+b*math.log(1-x)-lbeta)*betacf(x,a,b)/a
    else:
        ib=1-math.exp(a*math.log(x)+b*math.log(1-x)-lbeta)*betacf(1-x,b,a)/b
    return max(0.0, min(1.0, ib))  # this is the two-sided p for t

def report(name, res):
    if not res: print(f'{name}: insufficient n'); return
    p = pval(res['t'], res['df'])
    print(f'{name}:')
    print(f"  Treatment M={res['ma']:.4f} n={res['na']}   Control M={res['mb']:.4f} n={res['nb']}"
          f"   t={res['t']:.2f} df={res['df']:.1f} p={p:.4f} g={res['g']:.2f}")

# ---- embed user-only text per conversation (mean-pooled e5) ----
print('Loading multilingual-e5-large...')
model = SentenceTransformer('intfloat/multilingual-e5-large')
cids = [c for c in sorted(by)]
emb = {}
for c in cids:
    utext = [('passage: ' + (m.get('message_en') or m['message']).strip())
             for m in by[c] if m['message_src'] == 'user' and (m.get('message_en') or m['message']).strip()]
    if not utext:
        continue
    v = model.encode(utext, normalize_embeddings=True)
    mp = np.asarray(v).mean(axis=0)
    mp = mp / (np.linalg.norm(mp) + 1e-12)
    emb[c] = mp

groups = {c: by[c][0]['conv_group'] for c in emb}
chal = {c: labels.get(c, 'other_unclear') for c in emb}

# challenge centroids (full) and counts
chal_members = defaultdict(list)
for c in emb:
    chal_members[chal[c]].append(c)

def loo_centroid(c):
    peers = [x for x in chal_members[chal[c]] if x != c]
    if not peers: return None
    M = np.vstack([emb[x] for x in peers]).mean(axis=0)
    return M / (np.linalg.norm(M) + 1e-12)

def cos(a, b): return float(np.dot(a, b))

# ---- measure 1: distance to own-challenge LOO centroid ----
score1 = {}
for c in emb:
    cen = loo_centroid(c)
    if cen is None: continue
    score1[c] = 1 - cos(emb[c], cen)

# ---- residualized embeddings (x - own-challenge full centroid) ----
chal_centroid = {k: np.vstack([emb[x] for x in v]).mean(axis=0) for k, v in chal_members.items()}
resid = {}
for c in emb:
    r = emb[c] - chal_centroid[chal[c]]
    n = np.linalg.norm(r)
    resid[c] = r / n if n > 1e-9 else r

def resid_originality(c, scope):
    others = [x for x in resid if x != c and (scope is None or groups[x] == scope)]
    if not others: return None
    return statistics.mean(1 - cos(resid[c], resid[x]) for x in others)

# ---- within-conversation diversity (user-turn pairwise distance) ----
div = {}
for c in emb:
    uts = [(m.get('message_en') or m['message']).strip()
           for m in by[c] if m['message_src'] == 'user' and (m.get('message_en') or m['message']).strip()]
    if len(uts) < 2: continue
    V = np.asarray(model.encode(['passage: ' + t for t in uts], normalize_embeddings=True))
    ds = [1 - float(np.dot(V[i], V[j])) for i in range(len(V)) for j in range(i+1, len(V))]
    div[c] = statistics.mean(ds)

T = lambda d: [d[c] for c in d if groups[c] == 'treatment']
C = lambda d: [d[c] for c in d if groups[c] == 'control']

print('\n' + '='*78)
print('TOPIC-CONTROLLED ORIGINALITY — full-tier set (treatment vs control)')
print('='*78)
print(f'(convs embedded: {len(emb)}; with a same-challenge peer: {len(score1)})\n')
report('1. distance to own-challenge centroid (HEADLINE)', welch(T(score1), C(score1)))
report('2a. residualized same-condition originality', welch([resid_originality(c,'treatment') for c in resid if groups[c]=='treatment'],
                                                            [resid_originality(c,'control') for c in resid if groups[c]=='control']))
report('2b. residualized all-conversation originality', welch([resid_originality(c,None) for c in resid if groups[c]=='treatment'],
                                                              [resid_originality(c,None) for c in resid if groups[c]=='control']))
report('3. within-conversation diversity', welch(T(div), C(div)))

# ichilov-only cell
ich = [c for c in emb if chal[c] == 'ichilov_rehab_future']
ich_score = {c: score1[c] for c in ich if c in score1}
print('\n--- Ichilov rehab cell only (best-powered: 11 treatment vs 5 control) ---')
report('   distance to challenge centroid (ichilov)', welch(T(ich_score), C(ich_score)))

# write per-conv scores
with open(f'{BASE}/outputs/originality_topic_controlled.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f); w.writerow(['conversation_id','group','challenge_id','dist_to_challenge_centroid',
                                   'resid_same_cond_originality','within_conv_diversity'])
    for c in sorted(emb):
        rs = resid_originality(c, groups[c])
        w.writerow([c, groups[c], chal[c], round(score1.get(c, float('nan')),4),
                    round(rs,4) if rs is not None else 'nan', round(div.get(c, float('nan')),4)])
print('\nWrote: outputs/originality_topic_controlled.csv')
print('NOTE: p-values use a t-distribution; small/unequal n (esp. control) → interpret cautiously.')
