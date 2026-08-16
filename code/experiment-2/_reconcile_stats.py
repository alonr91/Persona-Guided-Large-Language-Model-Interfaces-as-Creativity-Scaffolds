# Throwaway reconciliation script: recompute headline stats from raw data.
import numpy as np, pandas as pd
from scipy import stats
pd.set_option('display.width', 200)

DATA = r"Data for analysis and scripts runs"

def hedges_g(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((na-1)*sa**2 + (nb-1)*sb**2)/(na+nb-2))
    d = (a.mean()-b.mean())/sp
    J = 1 - 3/(4*(na+nb)-9)
    return d*J

def welch(a, b):
    a = pd.Series(a).dropna(); b = pd.Series(b).dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return dict(Ma=a.mean(), SDa=a.std(ddof=1), na=len(a),
                Mb=b.mean(), SDb=b.std(ddof=1), nb=len(b),
                t=t, p=p, g=hedges_g(a, b), d=a.mean()-b.mean())

def line(label, r):
    print(f"{label:45s} T M={r['Ma']:.3f} SD={r['SDa']:.2f} (n={r['na']}) | "
          f"C M={r['Mb']:.3f} SD={r['SDb']:.2f} (n={r['nb']}) | "
          f"D={r['d']:+.3f} t={r['t']:.2f} p={r['p']:.4g} g={r['g']:+.2f}")

print("="*100)
print("QUESTIONNAIRE SAMPLE  (Responses 13 07)")
print("="*100)
q = pd.read_excel(f"{DATA}/Responses 13 07 (1).xlsx", sheet_name=0)
q.columns = [c.strip() for c in q.columns]
q['grp'] = q['Group type'].astype(str).str.lower().apply(lambda s: 'T' if 'exp' in s else ('C' if 'control' in s else np.nan))
print("Raw N by group:\n", q['grp'].value_counts(dropna=False))
# persona_helped valid 1-4
qh = q[q['persona_helped'].isin([1,2,3,4])]
print("\nValid persona_helped N by group:\n", qh['grp'].value_counts())
T = lambda col, d=q: d[d['grp']=='T'][col]
C = lambda col, d=q: d[d['grp']=='C'][col]

print("\n-- RQ1 questionnaire items --")
line("persona_helped (1=div..4=conv)", welch(T('persona_helped',qh), C('persona_helped',qh)))
line("creativity_increase_taylor(div)", welch(T('creativity_increase_taylor'), C('creativity_increase_taylor')))
line("creativity_increase_alex(conv)", welch(T('creativity_increase_alex'), C('creativity_increase_alex')))
line("taylor_interface_rating(div)", welch(T('taylor_interface_rating'), C('taylor_interface_rating')))
line("alex_interface_rating(conv)", welch(T('alex_interface_rating'), C('alex_interface_rating')))
line("solution_ownership", welch(T('solution_ownership'), C('solution_ownership')))
line("interface_comparison(vs std LLM)", welch(T('interface_comparison'), C('interface_comparison')))
line("ai_tool_mastery", welch(T('ai_tool_mastery'), C('ai_tool_mastery')))
# age
line("age", welch(T('age'), C('age')))

# one-sample vs midpoint 2.5 for persona_helped
for g in ['T','C']:
    x = qh[qh['grp']==g]['persona_helped'].dropna()
    t,p = stats.ttest_1samp(x, 2.5)
    print(f"   persona_helped one-sample vs 2.5 [{g}]: M={x.mean():.2f} t={t:.2f} p={p:.4g} n={len(x)}")
# distribution
print("\npersona_helped distribution (proportion):")
print(pd.crosstab(qh['grp'], qh['persona_helped'], normalize='index').round(3))

print("\n-- RQ2 personality correlations (Treatment only) --")
traits = {'Agreeableness':'Agreeableness','Extraversion':'Extraversion',
          'Conscientiousness':'Conscientiousness','Openness':'Open-Mindedness',
          'Neuroticism':'Negative Emotionality'}
outcomes = {'div_help':'taylor_interface_rating','conv_help':'alex_interface_rating',
            'div_creat':'creativity_increase_taylor','conv_creat':'creativity_increase_alex',
            'ownership':'solution_ownership','iface_vs_std':'interface_comparison',
            'persona_helped':'persona_helped'}
for grp in ['T','C']:
    d = q[q['grp']==grp]
    print(f"\n[{grp}] significant Pearson r (p<.05):")
    for tn,tc in traits.items():
        for on,oc in outcomes.items():
            sub = d[[tc,oc]].dropna()
            if len(sub) < 5: continue
            r,p = stats.pearsonr(sub[tc], sub[oc])
            if p < .05:
                print(f"   {tn:16s} x {on:14s} r={r:+.2f} p={p:.4g} n={len(sub)}")

print("\n"+"="*100)
print("ORIGINALITY SAMPLE  (participants_metrics)")
print("="*100)
m = pd.read_excel(f"{DATA}/participants_metrics.xlsx", sheet_name=0)
m.columns=[c.strip() for c in m.columns]
m['grp']=m['group'].astype(str).str.lower().apply(lambda s:'T' if ('exp' in s or s=='1') else ('C' if ('control' in s or s=='0') else s))
print("N by group:\n", m['grp'].value_counts())
Tm=lambda col:m[m['grp']=='T'][col]; Cm=lambda col:m[m['grp']=='C'][col]
print("\n-- RQ3 originality / fluency --")
line("n_ideas (fluency)", welch(Tm('n_ideas'), Cm('n_ideas')))
line("same-condition orig", welch(Tm('inter_mean_to_samegroup'), Cm('inter_mean_to_samegroup')))
line("all-participants orig", welch(Tm('inter_mean_to_all'), Cm('inter_mean_to_all')))
line("cross-cond nearest neighbor", welch(Tm('inter_min_to_othergroup'), Cm('inter_min_to_othergroup')))
line("within-participant diversity", welch(Tm('within_mean_d'), Cm('within_mean_d')))

# quantity-distinctiveness tradeoff (treatment+control pooled spearman)
print("\n-- exploratory Spearman (all participants) --")
for col,lab in [('inter_mean_to_samegroup','orig_same vs n_ideas'),('within_mean_d','within_div vs n_ideas')]:
    sub=m[[ 'n_ideas',col]].dropna()
    rho,p=stats.spearmanr(sub['n_ideas'],sub[col])
    print(f"   {lab:28s} rho={rho:+.3f} p={p:.4g} n={len(sub)}")
