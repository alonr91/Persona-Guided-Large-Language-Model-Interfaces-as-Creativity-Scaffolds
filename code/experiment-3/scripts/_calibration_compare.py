"""Compare Claude's hand-ratings (1-5 divergent↔convergent) with the NLI
classifier's d_minus_c score, on the 60-turn calibration sample.

My rating scale:
  1 = strongly divergent (broadens, open questions, analogy, multiple alternatives)
  2 = mostly divergent
  3 = neutral / balanced / clarification
  4 = mostly convergent (concrete recommendations, criteria)
  5 = strongly convergent (structured stepwise plans, decisive)

Mapping to bipolar D-C in [-1,+1]:  1→+1, 2→+0.5, 3→0, 4→-0.5, 5→-1
Then correlate with NLI's d_minus_c.
"""
import json, statistics, math
import numpy as np

# Hand ratings keyed by rate_id
RATINGS = {
    1: 4, 2: 4, 3: 4, 4: 5, 5: 2, 6: 1, 7: 2, 8: 4, 9: 2, 10: 3,
    11: 3, 12: 5, 13: 4, 14: 3, 15: 3, 16: 2, 17: 3, 18: 4, 19: 4, 20: 1,
    21: 3, 22: 2, 23: 2, 24: 1, 25: 4, 26: 2, 27: 3, 28: 3, 29: 5, 30: 4,
    31: 5, 32: 5, 33: 3, 34: 3, 35: 3, 36: 2, 37: 4, 38: 5, 39: 3, 40: 4,
    41: 4, 42: 1, 43: 1, 44: 2, 45: 3, 46: 4, 47: 3, 48: 3, 49: 2, 50: 5,
    51: 4, 52: 2, 53: 3, 54: 3, 55: 4, 56: 5, 57: 2, 58: 4, 59: 1, 60: 3,
}

def to_bipolar(r):  # 1→+1, 5→-1
    return (3 - r) / 2.0

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
sample = json.load(open(f'{base}/outputs/calibration_sample.json','r',encoding='utf-8'))

claude_bipolar = []
nli_bipolar = []
for s in sample:
    r = RATINGS.get(s['rate_id'])
    if r is None: continue
    claude_bipolar.append(to_bipolar(r))
    nli_bipolar.append(s['nli_d_minus_c'])

cb = np.array(claude_bipolar); nb = np.array(nli_bipolar)
n = len(cb)

# Pearson correlation
pearson = np.corrcoef(cb, nb)[0,1]

# Spearman (rank correlation) — easier to interpret with ordinal ratings
def rank(arr):
    arr = np.asarray(arr)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(arr))
    # average ranks for ties
    return ranks
spearman = np.corrcoef(rank(cb), rank(nb))[0,1]

# Mean absolute error
mae = float(np.mean(np.abs(cb - nb)))

# Direction agreement (sign of d-c agrees, ignoring 0)
agree = sum(1 for c,n_ in zip(cb,nb) if (c>0 and n_>0) or (c<0 and n_<0) or (c==0 and abs(n_)<0.05))
print('='*88)
print('CALIBRATION — Claude (hand-rated) vs NLI classifier on 60 turns')
print('='*88)
print(f'\nN = {n}')
print(f'Pearson r  = {pearson:.3f}')
print(f'Spearman ρ = {spearman:.3f}')
print(f'MAE on bipolar [-1,+1] = {mae:.3f}')
print(f'Sign agreement (D vs C direction): {agree}/{n} ({agree/n*100:.1f}%)')

# By group/persona cell
print('\nBy stratum (Claude mean vs NLI mean):')
strata = {}
for s in sample:
    r = RATINGS[s['rate_id']]
    key = (s['group'], s['agent_role'])
    strata.setdefault(key, []).append((to_bipolar(r), s['nli_d_minus_c']))

for k, vals in sorted(strata.items()):
    cs = [v[0] for v in vals]; ns = [v[1] for v in vals]
    print(f'  {str(k):35s}  n={len(vals):3d}  Claude_mean={np.mean(cs):+.3f}  NLI_mean={np.mean(ns):+.3f}  diff={np.mean(cs)-np.mean(ns):+.3f}')

# Print biggest disagreements for inspection
print('\nLargest disagreements (|Claude - NLI| > 0.7):')
for s in sample:
    r = RATINGS[s['rate_id']]
    cb1 = to_bipolar(r); nb1 = s['nli_d_minus_c']
    if abs(cb1-nb1) > 0.7:
        print(f'  rate_id={s["rate_id"]}  Claude={cb1:+.2f}  NLI={nb1:+.2f}  group={s["group"]}/{s["agent_role"]}')
        print(f'    text: {s["text"][:150]}')

# Save the rated sample
for s in sample:
    s['claude_rating_1to5'] = RATINGS[s['rate_id']]
    s['claude_bipolar']     = round(to_bipolar(RATINGS[s['rate_id']]), 3)
with open(f'{base}/outputs/calibration_sample_rated.json','w',encoding='utf-8') as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)
print(f'\nWrote calibration_sample_rated.json')
