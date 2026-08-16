"""Tier 1 replication of Scaffolding Creativity Sec 5.1 on Exp 3 logs.

Replicates:
  (a) Mean number of '?' per user message
  (b) Percentage of threads (assistant turns) containing at least one '?'
       — focus on quarters 2-4 (paper excludes Q1 familiarization)
  (c) Welch's t-test treatment vs control
  (d) Within-treatment: Taylor (divergent) vs Alex (convergent) — messages per persona, ending persona

Unit of analysis: conversation_id (no participant IDs available in Exp 3).
"""
import json, csv, statistics, math
from collections import defaultdict, Counter

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
with open(f'{base}/data/experiment3_messages_clean.json','r',encoding='utf-8') as f:
    msgs = json.load(f)

by_conv = defaultdict(list)
for m in msgs:
    by_conv[m['conversation_id']].append(m)
for cid in by_conv:
    by_conv[cid].sort(key=lambda x:(x['timestamp'], x['message_id']))

def conv_group(ms):
    return ms[0]['group']  # already validated single-group in clean set

def quarters(items):
    """Split items into 4 roughly equal quarters by index order."""
    n = len(items)
    if n == 0: return [[],[],[],[]]
    q = [n*i//4 for i in range(5)]
    return [items[q[i]:q[i+1]] for i in range(4)]

# Question-mark detection: support ASCII '?' and Hebrew/Arabic FF1F (rare); ASCII covers EN+HE
def count_q(text):
    return text.count('?')

# Per-conversation metrics
rows = []
for cid, ms in by_conv.items():
    g = conv_group(ms)
    user_msgs = [m for m in ms if m['message_src']=='user']
    assistant_msgs = [m for m in ms if m['message_src']=='assistant']
    if len(user_msgs) < 4:
        # paper uses quarters 2-4 — need at least ~4 user turns to have meaningful Q2-Q4
        few = True
    else:
        few = False

    # All-conv metrics
    n_user = len(user_msgs)
    qmarks_per_user_all = sum(count_q(m['message']) for m in user_msgs) / max(n_user,1)
    pct_user_with_q_all = sum(1 for m in user_msgs if '?' in m['message']) / max(n_user,1) * 100

    # Q2-Q4 metrics (excluding familiarization Q1)
    qs_user = quarters(user_msgs)
    q234_user = qs_user[1] + qs_user[2] + qs_user[3]
    n_q234 = len(q234_user)
    if n_q234 > 0:
        qmarks_per_user_q234 = sum(count_q(m['message']) for m in q234_user) / n_q234
        pct_user_with_q_q234 = sum(1 for m in q234_user if '?' in m['message']) / n_q234 * 100
    else:
        qmarks_per_user_q234 = float('nan')
        pct_user_with_q_q234 = float('nan')

    # Per-persona within treatment: msgs sent to each role + ending persona
    msgs_to_taylor = sum(1 for m in user_msgs if str(m.get('persona_id')) in ('1','3'))
    msgs_to_alex   = sum(1 for m in user_msgs if str(m.get('persona_id')) in ('2','4'))
    last_assistant = next((m for m in reversed(ms) if m['message_src']=='assistant'), None)
    ending_persona = None
    if last_assistant:
        pid = str(last_assistant.get('persona_id'))
        ending_persona = 'Taylor' if pid in ('1','3') else ('Alex' if pid in ('2','4') else None)

    rows.append({
        'conv_id': cid,
        'group': g,
        'n_user': n_user,
        'n_assistant': len(assistant_msgs),
        'n_user_q234': n_q234,
        'qmarks_per_user_all': round(qmarks_per_user_all,3),
        'pct_user_with_q_all': round(pct_user_with_q_all,2),
        'qmarks_per_user_q234': round(qmarks_per_user_q234,3) if not math.isnan(qmarks_per_user_q234) else '',
        'pct_user_with_q_q234': round(pct_user_with_q_q234,2) if not math.isnan(pct_user_with_q_q234) else '',
        'msgs_to_taylor': msgs_to_taylor,
        'msgs_to_alex': msgs_to_alex,
        'ending_persona': ending_persona,
        'few_user_msgs': few,
    })

# Write per-conversation CSV
with open(f'{base}/outputs/tier1_per_conversation.csv','w',encoding='utf-8',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# Welch's t-test (no scipy — implement)
def welch_t(a, b):
    a = [x for x in a if x is not None and not (isinstance(x,float) and math.isnan(x))]
    b = [x for x in b if x is not None and not (isinstance(x,float) and math.isnan(x))]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va/na + vb/nb)
    if se == 0: return None
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    df = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    # Two-sided p via t-distribution survival function (approximation using stdlib)
    # Use the Student's t CDF approximation
    from math import lgamma, log, sqrt, pi, exp
    # Use scipy-free approximation: A&S 26.7.8 series — but simpler, use regularized incomplete beta
    # Implement two-sided p via incomplete beta function
    def betacf(a, b, x, max_it=200, eps=3e-7):
        qab = a+b; qap = a+1; qam = a-1
        c = 1.0
        d = 1.0 - qab*x/qap
        if abs(d) < 1e-30: d = 1e-30
        d = 1.0/d
        h = d
        for m in range(1, max_it+1):
            m2 = 2*m
            aa = m*(b-m)*x / ((qam+m2)*(a+m2))
            d = 1.0 + aa*d
            if abs(d) < 1e-30: d = 1e-30
            c = 1.0 + aa/c
            if abs(c) < 1e-30: c = 1e-30
            d = 1.0/d
            h *= d*c
            aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
            d = 1.0 + aa*d
            if abs(d) < 1e-30: d = 1e-30
            c = 1.0 + aa/c
            if abs(c) < 1e-30: c = 1e-30
            d = 1.0/d
            delta = d*c
            h *= delta
            if abs(delta-1.0) < eps: break
        return h
    def betai(a,b,x):
        if x<0 or x>1: return float('nan')
        if x==0 or x==1: bt = 0
        else:
            bt = exp(lgamma(a+b) - lgamma(a) - lgamma(b) + a*log(x) + b*log(1-x))
        if x < (a+1)/(a+b+2):
            return bt*betacf(a,b,x)/a
        else:
            return 1 - bt*betacf(b,a,1-x)/b
    p = betai(df/2, 0.5, df/(df + t*t))
    # Hedges' g (small-sample-corrected Cohen's d)
    sp = math.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    if sp == 0: g = 0
    else:
        d = (ma-mb)/sp
        J = 1 - 3/(4*(na+nb)-9)
        g = d*J
    return {'mean_a':ma,'mean_b':mb,'sd_a':math.sqrt(va),'sd_b':math.sqrt(vb),
            'n_a':na,'n_b':nb,'t':t,'df':df,'p':p,'hedges_g':g}

# Group comparisons
def get(rows, group, key):
    out = []
    for r in rows:
        if r['group']!=group: continue
        v = r[key]
        if v == '' or v is None: continue
        out.append(float(v))
    return out

print('='*78)
print('TIER 1 — ENGAGEMENT ANALYSIS (replicating Sec 5.1 of paper)')
print('='*78)
print(f"\nN conversations: treatment={sum(1 for r in rows if r['group']=='treatment')}, control={sum(1 for r in rows if r['group']=='control')}")
print(f"Few user msgs (<4) excluded from Q2-Q4 analysis: {sum(1 for r in rows if r['few_user_msgs'])}")

print('\n--- (a) Mean question marks per user message ---')
for window in ['all','q234']:
    key = f'qmarks_per_user_{window}'
    a = get(rows, 'treatment', key)
    b = get(rows, 'control', key)
    res = welch_t(a, b)
    if res is None:
        print(f'  [{window}] insufficient data')
        continue
    print(f'  [{window}] Treatment M={res["mean_a"]:.3f} SD={res["sd_a"]:.3f} (n={res["n_a"]})  '
          f'Control M={res["mean_b"]:.3f} SD={res["sd_b"]:.3f} (n={res["n_b"]})  '
          f't={res["t"]:.2f}, df={res["df"]:.1f}, p={res["p"]:.4f}, g={res["hedges_g"]:.2f}')

print('\n--- (b) Percent of user messages containing ? ---')
for window in ['all','q234']:
    key = f'pct_user_with_q_{window}'
    a = get(rows, 'treatment', key)
    b = get(rows, 'control', key)
    res = welch_t(a, b)
    if res is None:
        print(f'  [{window}] insufficient data')
        continue
    print(f'  [{window}] Treatment M={res["mean_a"]:.2f}% SD={res["sd_a"]:.2f} (n={res["n_a"]})  '
          f'Control M={res["mean_b"]:.2f}% SD={res["sd_b"]:.2f} (n={res["n_b"]})  '
          f't={res["t"]:.2f}, df={res["df"]:.1f}, p={res["p"]:.4f}, g={res["hedges_g"]:.2f}')

# Per-persona within treatment
print('\n--- (c) Within treatment: messages addressed to each persona ---')
trt = [r for r in rows if r['group']=='treatment']
to_t = [r['msgs_to_taylor'] for r in trt]
to_a = [r['msgs_to_alex']   for r in trt]
res = welch_t(to_t, to_a)
print(f'  to Taylor (divergent): M={statistics.mean(to_t):.2f} SD={statistics.stdev(to_t):.2f}')
print(f'  to Alex   (convergent): M={statistics.mean(to_a):.2f} SD={statistics.stdev(to_a):.2f}')
if res:
    print(f'  paired-style (between-conv) Welch: t={res["t"]:.2f}, df={res["df"]:.1f}, p={res["p"]:.4f}, g={res["hedges_g"]:.2f}')

# Ending persona distribution within treatment
end_counter = Counter(r['ending_persona'] for r in trt if r['ending_persona'])
total = sum(end_counter.values())
print(f'\n--- (d) Within treatment: ending persona ---')
for k,v in end_counter.most_common():
    print(f'  ended with {k}: {v}/{total} ({v/total*100:.1f}%)')

# Same for control (Taylor 3 vs Alex 4)
ctrl = [r for r in rows if r['group']=='control']
ct_t = [r['msgs_to_taylor'] for r in ctrl]
ct_a = [r['msgs_to_alex']   for r in ctrl]
print(f'\n--- (c2) Within control: messages addressed to each persona (baseline LLM, both) ---')
print(f'  to Taylor (control-LLM): M={statistics.mean(ct_t):.2f} SD={statistics.stdev(ct_t) if len(ct_t)>1 else 0:.2f}')
print(f'  to Alex   (control-LLM): M={statistics.mean(ct_a):.2f} SD={statistics.stdev(ct_a) if len(ct_a)>1 else 0:.2f}')

end_counter_c = Counter(r['ending_persona'] for r in ctrl if r['ending_persona'])
total_c = sum(end_counter_c.values())
print(f'\n--- (d2) Within control: ending persona ---')
for k,v in end_counter_c.most_common():
    print(f'  ended with {k}: {v}/{total_c} ({v/total_c*100:.1f}%)')

print(f'\nWritten: {base}/outputs/tier1_per_conversation.csv')
