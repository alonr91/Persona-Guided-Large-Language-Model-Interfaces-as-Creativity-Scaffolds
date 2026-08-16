"""Manipulation check: did Taylor (divergent) emit more divergent stance than
Alex (convergent), in treatment but NOT in control?

Inputs: stance_per_message.json (raw NLI scores per turn)
Outputs: manipcheck_summary.txt with means / Welch t-tests / Hedges' g
"""
import json, statistics, math, csv
from collections import defaultdict

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'

with open(f'{base}/outputs/stance_per_message.json','r',encoding='utf-8') as f:
    rows = json.load(f)

# Helpers
def welch(a,b):
    a = [x for x in a if isinstance(x,(int,float)) and not math.isnan(x)]
    b = [x for x in b if isinstance(x,(int,float)) and not math.isnan(x)]
    na, nb = len(a), len(b)
    if na<2 or nb<2: return None
    ma,mb = statistics.mean(a), statistics.mean(b)
    va,vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va/na+vb/nb)
    if se==0: return None
    t = (ma-mb)/se
    df = (va/na+vb/nb)**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1))
    sp = math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2))
    d = (ma-mb)/sp if sp else 0
    J = 1 - 3/(4*(na+nb)-9)
    g = d*J
    from math import lgamma, exp, log
    def betai(a_,b_,x):
        if x<=0: return 0
        if x>=1: return 1
        bt = exp(lgamma(a_+b_)-lgamma(a_)-lgamma(b_) + a_*log(x) + b_*log(1-x))
        def cf(a,b,x,maxit=200,eps=3e-7):
            qab=a+b; qap=a+1; qam=a-1
            c=1.0; d=1-qab*x/qap
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
    return ma, statistics.stdev(a), na, mb, statistics.stdev(b), nb, t, df, p, g

def fmt(name, res):
    if res is None: return f'  {name}: insufficient data'
    ma,sa,na,mb,sb,nb,t,df,p,g = res
    return (f'  {name:55s}  M_A={ma:.3f} (SD={sa:.3f}, n={na})  M_B={mb:.3f} (SD={sb:.3f}, n={nb})  '
            f't({df:.1f})={t:.2f}, p={p:.4g}, g={g:.2f}')

# Filter to assistant turns only for the manipulation check
asst = [r for r in rows if r['message_src']=='assistant']
user = [r for r in rows if r['message_src']=='user']

def extract(rows, group, agent_role, score_key):
    return [r[score_key] for r in rows if r['group']==group and r['agent_role']==agent_role]

print('='*88)
print('MANIPULATION CHECK — Did Taylor and Alex emit different stances within treatment?')
print('='*88)

# 1. WITHIN TREATMENT: Taylor vs Alex on each score
trt_taylor_div = extract(asst, 'treatment', 'Taylor', 'divergent_score')
trt_alex_div   = extract(asst, 'treatment', 'Alex',   'divergent_score')
trt_taylor_con = extract(asst, 'treatment', 'Taylor', 'convergent_score')
trt_alex_con   = extract(asst, 'treatment', 'Alex',   'convergent_score')
trt_taylor_dmc = extract(asst, 'treatment', 'Taylor', 'd_minus_c')
trt_alex_dmc   = extract(asst, 'treatment', 'Alex',   'd_minus_c')

print('\n[A] Within TREATMENT — Taylor (divergent persona) vs Alex (convergent persona):')
print(fmt('Taylor vs Alex on divergent_score',  welch(trt_taylor_div, trt_alex_div)))
print(fmt('Taylor vs Alex on convergent_score', welch(trt_taylor_con, trt_alex_con)))
print(fmt('Taylor vs Alex on d_minus_c balance',welch(trt_taylor_dmc, trt_alex_dmc)))

# 2. WITHIN CONTROL: should NOT differ (both regular LLMs)
ctl_taylor_div = extract(asst, 'control', 'Taylor', 'divergent_score')
ctl_alex_div   = extract(asst, 'control', 'Alex',   'divergent_score')
ctl_taylor_con = extract(asst, 'control', 'Taylor', 'convergent_score')
ctl_alex_con   = extract(asst, 'control', 'Alex',   'convergent_score')
ctl_taylor_dmc = extract(asst, 'control', 'Taylor', 'd_minus_c')
ctl_alex_dmc   = extract(asst, 'control', 'Alex',   'd_minus_c')

print('\n[B] Within CONTROL — Taylor (regular LLM) vs Alex (regular LLM)  [should be ~null]:')
print(fmt('Taylor vs Alex on divergent_score',  welch(ctl_taylor_div, ctl_alex_div)))
print(fmt('Taylor vs Alex on convergent_score', welch(ctl_taylor_con, ctl_alex_con)))
print(fmt('Taylor vs Alex on d_minus_c balance',welch(ctl_taylor_dmc, ctl_alex_dmc)))

# 3. BETWEEN GROUPS: treatment-Taylor vs control-Taylor; treatment-Alex vs control-Alex
print('\n[C] Between groups — does treatment Taylor differ from control Taylor?')
print(fmt('Taylor: treatment vs control on divergent_score',  welch(trt_taylor_div, ctl_taylor_div)))
print(fmt('Taylor: treatment vs control on d_minus_c',         welch(trt_taylor_dmc, ctl_taylor_dmc)))
print(fmt('Alex: treatment vs control on convergent_score',    welch(trt_alex_con, ctl_alex_con)))
print(fmt('Alex: treatment vs control on d_minus_c',           welch(trt_alex_dmc, ctl_alex_dmc)))

# 4. USER-SIDE CO-REGULATION (analysis B from strategy)
print('\n[D] USER stance uptake — does the user mirror the persona they address?')
u_to_taylor_trt = [r['d_minus_c'] for r in user if r['group']=='treatment' and r['agent_role']=='Taylor']
u_to_alex_trt   = [r['d_minus_c'] for r in user if r['group']=='treatment' and r['agent_role']=='Alex']
u_to_taylor_ctl = [r['d_minus_c'] for r in user if r['group']=='control'   and r['agent_role']=='Taylor']
u_to_alex_ctl   = [r['d_minus_c'] for r in user if r['group']=='control'   and r['agent_role']=='Alex']
print('  TREATMENT — user d_minus_c when addressing Taylor vs Alex:')
print('    ', fmt('Taylor vs Alex (user side)', welch(u_to_taylor_trt, u_to_alex_trt)))
print('  CONTROL — user d_minus_c when addressing Taylor vs Alex (regular LLM):')
print('    ', fmt('Taylor vs Alex (user side)', welch(u_to_taylor_ctl, u_to_alex_ctl)))

# 5. PER-HYPOTHESIS DRILL-DOWN within treatment (which div/con facets differ most?)
print('\n[E] Per-hypothesis Taylor vs Alex (treatment only, assistant turns):')
hyps_keys = [k for k in rows[0].keys() if k.startswith('h_')]
for k in hyps_keys:
    a_ = [r[k] for r in asst if r['group']=='treatment' and r['agent_role']=='Taylor']
    b_ = [r[k] for r in asst if r['group']=='treatment' and r['agent_role']=='Alex']
    res = welch(a_, b_)
    if res:
        ma,sa,na,mb,sb,nb,t,df,p,g = res
        print(f'  {k[2:]:40s}  M(T)={ma:.3f}  M(A)={mb:.3f}  diff={ma-mb:+.3f}  g={g:.2f}  p={p:.4g}')

# Per-conversation aggregate: D-C per persona per conv (for plotting later)
out_rows = []
by_conv = defaultdict(list)
for r in asst:
    by_conv[r['conversation_id']].append(r)
for cid, rs in by_conv.items():
    grp = rs[0]['group']
    for role in ['Taylor','Alex']:
        ts = [r for r in rs if r['agent_role']==role]
        if not ts: continue
        out_rows.append({
            'conv_id': cid, 'group': grp, 'agent_role': role,
            'n_turns': len(ts),
            'mean_div': round(statistics.mean(r['divergent_score'] for r in ts), 4),
            'mean_con': round(statistics.mean(r['convergent_score'] for r in ts), 4),
            'mean_dmc': round(statistics.mean(r['d_minus_c'] for r in ts), 4),
        })

with open(f'{base}/outputs/manipcheck_per_conv_persona.csv','w',encoding='utf-8',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)

print(f'\nWrote: {base}/outputs/manipcheck_per_conv_persona.csv')
