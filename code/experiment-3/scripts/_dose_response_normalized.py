# -*- coding: utf-8 -*-
"""Length-NORMALIZED dose-response (treatment-only) + group rate comparison.

Conversations are asymmetric (4-34 user turns; minutes to multi-day), so raw
fluency/originality are dominated by length. This re-analysis normalizes:
  QUANTITATIVE (rate):   ideas_per_user_turn = fluency / n_user
                         ideas_per_assistant_turn = fluency / n_assistant
                         flex_per_idea = flexibility / fluency
  SEMANTIC (intensive):  orig_same/all are centroid distances (already intensive),
                         but confounded by fluency -> report partial controlling fluency;
                         within_idea_diversity is a mean (already normalized).
Interaction STYLE predictors are length-independent: alex_share, alex_timing,
switch_rate. Volume predictors (n_user) are reported only to show the confound.

Simple stats: Pearson r, two-sided p, no correction. (duration_min reported with a
caveat: multi-day idle gaps make wall-clock time unreliable; turn-count is the
robust normalizer.)
"""
import csv, math, statistics

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
summ = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/experiment3_conversations_summary.csv', encoding='utf-8'))}
port = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/idea_portfolio_exp1.csv', encoding='utf-8'))}
dose = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/dose_response_treatment.csv', encoding='utf-8'))}

def fnum(v):
    try:
        x = float(v); return x if not (math.isnan(x) or math.isinf(x)) else None
    except Exception:
        return None

def pear(pairs):
    pts = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pts)
    if n < 4: return None
    X = [p[0] for p in pts]; Y = [p[1] for p in pts]
    mx, my = statistics.mean(X), statistics.mean(Y); sx, sy = statistics.pstdev(X), statistics.pstdev(Y)
    if sx == 0 or sy == 0: return None
    r = max(-.999, min(.999, sum((a-mx)*(b-my) for a, b in pts)/(n*sx*sy)))
    t = r*math.sqrt((n-2)/(1-r*r)); df = n-2; x = df/(df+t*t)
    def bcf(x, a, b):
        f=1e-300; qab=a+b; qap=a+1; qam=a-1; c=1; d=1-qab*x/qap; d=1/(d if abs(d)>f else f); h=d
        for k in range(1,200):
            k2=2*k; aa=k*(b-k)*x/((qam+k2)*(a+k2)); d=1+aa*d; d=1/(d if abs(d)>f else f); c=1+aa/c; c=c if abs(c)>f else f; h*=d*c
            aa=-(a+k)*(qab+k)*x/((a+k2)*(qap+k2)); d=1+aa*d; d=1/(d if abs(d)>f else f); c=1+aa/c; c=c if abs(c)>f else f; de=d*c; h*=de
            if abs(de-1)<3e-12: break
        return h
    a,b=df/2,.5; lb=math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
    ib=(math.exp(a*math.log(x)+b*math.log(1-x)-lb)*bcf(x,a,b)/a) if x<(a+1)/(a+b+2) else (1-math.exp(a*math.log(x)+b*math.log(1-x)-lb)*bcf(1-x,b,a)/b)
    return round(r,3), n, round(max(0,min(1,ib)),4)

def welch(A, B):
    A=[x for x in A if x is not None]; B=[x for x in B if x is not None]
    na,nb=len(A),len(B)
    if na<2 or nb<2: return None
    ma,mb=statistics.mean(A),statistics.mean(B); va,vb=statistics.variance(A),statistics.variance(B)
    se=math.sqrt(va/na+vb/nb)
    if se==0: return None
    t=(ma-mb)/se; sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)); g=(ma-mb)/sp*(1-3/(4*(na+nb)-9)) if sp else 0
    return round(ma,3),round(mb,3),round(t,2),round(g,2),na,nb

# ---- build normalized per-conversation table (all full-tier) ----
rows = []
for c, s in summ.items():
    if s['quality_tier'] != 'full': continue
    p = port.get(c, {})
    fl = fnum(p.get('fluency')); flex = fnum(p.get('flexibility'))
    nu = fnum(s['n_user']); na = fnum(s['n_assistant']); dur = fnum(s['duration_minutes'])
    rec = {'conversation_id': c, 'group': s['group'],
           'fluency': fl, 'flexibility': flex, 'n_user': nu,
           'ideas_per_user_turn': (fl/nu) if (fl is not None and nu) else None,
           'ideas_per_assistant_turn': (fl/na) if (fl is not None and na) else None,
           'ideas_per_min': (fl/dur) if (fl is not None and dur and dur > 0) else None,
           'flex_per_idea': (flex/fl) if (flex is not None and fl) else None,
           'orig_same': fnum(p.get('orig_same')), 'orig_all': fnum(p.get('orig_all')),
           'within_idea_diversity': fnum(p.get('within_idea_diversity'))}
    d = dose.get(c, {})
    rec['alex_share'] = fnum(d.get('alex_share')); rec['alex_timing'] = fnum(d.get('alex_timing'))
    sw = fnum(d.get('switches'))
    rec['switch_rate'] = (sw/(nu-1)) if (sw is not None and nu and nu > 1) else None
    rows.append(rec)

tr = [r for r in rows if r['group'] == 'treatment']
co = [r for r in rows if r['group'] == 'control']

print('='*74)
print('1) GROUP COMPARISON of length-normalized creativity (treatment vs control)')
print('='*74)
for k in ['fluency', 'ideas_per_user_turn', 'ideas_per_assistant_turn', 'flex_per_idea',
          'orig_same', 'within_idea_diversity']:
    w = welch([r[k] for r in tr], [r[k] for r in co])
    if w: print(f"   {k:26} T={w[0]} C={w[1]}  t={w[2]} g={w[3]}  (n {w[4]}/{w[5]})")

print('\n' + '='*74)
print('2) WITHIN-TREATMENT dose-response on NORMALIZED outcomes (n=31)')
print('   STYLE predictors are length-independent; n_user shown to expose the confound.')
print('='*74)
PRED = ['alex_share', 'alex_timing', 'switch_rate', 'n_user']
OUTC = [('ideas_per_user_turn', 'idea RATE (ideas / user turn) [QUANT, normalized]'),
        ('ideas_per_min', 'idea rate / minute [QUANT, caveat: idle gaps]'),
        ('flex_per_idea', 'category breadth per idea [QUANT, normalized]'),
        ('orig_same', 'idea-portfolio originality (SEMANTIC, intensive)'),
        ('within_idea_diversity', 'within-portfolio diversity (SEMANTIC, normalized)')]
for okey, oname in OUTC:
    print(f'\nOUTCOME: {oname}')
    for pk in PRED:
        res = pear([(r[pk], r[okey]) for r in tr])
        if res:
            r_, n_, p_ = res
            star = '*' if p_ < .05 else ('.' if p_ < .10 else ' ')
            print(f"   {pk:13} r={r_:+.3f}  p={p_:.3f}  (n={n_}) {star}")

# partial: alex_share vs orig_same controlling fluency (length)
def partial(xk, yk, zk, data):
    P = [(r[xk], r[yk], r[zk]) for r in data if r[xk] is not None and r[yk] is not None and r[zk] is not None]
    n = len(P)
    if n < 5: return None
    def rr(i, j):
        mi=statistics.mean(p[i] for p in P); mj=statistics.mean(p[j] for p in P)
        si=statistics.pstdev([p[i] for p in P]); sj=statistics.pstdev([p[j] for p in P])
        return sum((p[i]-mi)*(p[j]-mj) for p in P)/(n*si*sj)
    rxy,rxz,ryz=rr(0,1),rr(0,2),rr(1,2)
    pr=(rxy-rxz*ryz)/math.sqrt((1-rxz**2)*(1-ryz**2)); t=pr*math.sqrt((n-3)/(1-pr*pr))
    return round(pr,3), n, round(t,2)
print('\n' + '='*74)
print('3) Is convergent engagement related to originality once length is controlled?')
print('='*74)
print('   alex_share <-> orig_same | fluency :', partial('alex_share','orig_same','fluency', tr))
print('   alex_share <-> ideas_per_user_turn :', pear([(r['alex_share'], r['ideas_per_user_turn']) for r in tr]))

with open(f'{BASE}/outputs/dose_response_normalized.csv', 'w', encoding='utf-8', newline='') as f:
    keys = ['conversation_id','group','fluency','ideas_per_user_turn','ideas_per_assistant_turn',
            'ideas_per_min','flex_per_idea','orig_same','within_idea_diversity','alex_share','alex_timing','switch_rate','n_user']
    w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
print('\nWrote: outputs/dose_response_normalized.csv')
