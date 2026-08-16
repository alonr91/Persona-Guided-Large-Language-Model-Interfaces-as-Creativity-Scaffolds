# -*- coding: utf-8 -*-
"""Validate that the LONG conversations carry sufficient on-task creative content.

Length (>=10 user turns) does not guarantee substance. For each long conversation we
report group, inferred challenge, length, extracted-idea count (fluency) and idea
density (ideas per user turn), and the opening user messages, then flag off-task /
thin conversations (challenge='other_unclear' or very low idea density). Finally we
re-run the headline contrasts excluding flagged conversations to check robustness.
"""
import json, csv, math, statistics
from collections import defaultdict

BASE = r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
LONG = {int(r['conversation_id']): r['group'] for r in csv.DictReader(open(f'{BASE}/outputs/long_sample_ids.csv', encoding='utf-8'))}
summ = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/experiment3_conversations_summary.csv', encoding='utf-8'))}
port = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/long_portfolio.csv', encoding='utf-8'))}
lab  = {int(r['conversation_id']): r for r in csv.DictReader(open(f'{BASE}/outputs/challenge_labels.csv', encoding='utf-8'))}
msgs = json.load(open(f'{BASE}/data/experiment3_full_en.json', encoding='utf-8'))
st   = json.load(open(f'{BASE}/outputs/stance_per_message_full.json', encoding='utf-8'))

byc = defaultdict(list)
for m in msgs:
    byc[m['conversation_id']].append(m)
for c in byc: byc[c].sort(key=lambda x: (x['timestamp'], x['message_id']))

def first_user(c, k=2):
    us = [(m.get('message_en') or m['message']).strip().replace('\n',' ') for m in byc[c] if m['message_src']=='user']
    return ' ⏐ '.join(u[:70] for u in us[:k])

print('LONG conversation content audit (n=%d)\n' % len(LONG))
print(f"{'conv':4} {'grp':4} {'challenge':22} {'nU':>3} {'flu':>3} {'i/turn':>6} {'conf':>4}  opening user messages")
print('-'*135)
rows=[]
for c in sorted(LONG, key=lambda c:(LONG[c], lab[c]['challenge_id'])):
    p=port[c]; nu=int(summ[c]['n_user']); flu=p['fluency']; ipt=p['ideas_per_user_turn'] if 'ideas_per_user_turn' in p else ''
    try: ipt=float(p.get('ideas_per_user_turn') or (float(flu)/nu));
    except: ipt=float(flu)/nu if flu else 0
    ch=lab[c]['challenge_id']; conf=lab[c]['confidence']
    rows.append((c,LONG[c],ch,nu,int(float(flu)),ipt,conf))
    print(f"{c:4} {LONG[c][:4]:4} {ch:22} {nu:3} {int(float(flu)):3} {ipt:6.2f} {conf:>4}  {first_user(c)}")

# flag off-task / thin
flagged=set()
for c,g,ch,nu,flu,ipt,conf in rows:
    if ch=='other_unclear' or ipt<0.25 or flu<3:
        flagged.add(c)
print('\nFLAGGED as off-task / thin:', sorted(flagged))
for c in sorted(flagged):
    print(f'  conv {c} [{LONG[c]}] {lab[c]["challenge_id"]} nU={summ[c]["n_user"]} fluency={port[c]["fluency"]} :: {first_user(c,3)}')

# idea-density distribution
ipts=[r[5] for r in rows]
print(f'\nIdea density (ideas/user turn): median={statistics.median(ipts):.2f}, min={min(ipts):.2f}, max={max(ipts):.2f}')
print('On-task substantive (not flagged):', len(LONG)-len(flagged), 'of', len(LONG))

# ---- robustness: re-run headline contrasts excluding flagged ----
def welch(a,b):
    a=[x for x in a if x is not None];b=[x for x in b if x is not None];na,nb=len(a),len(b)
    if na<2 or nb<2:return None
    ma,mb=statistics.mean(a),statistics.mean(b);va,vb=statistics.variance(a),statistics.variance(b)
    se=math.sqrt(va/na+vb/nb)
    if se==0:return None
    t=(ma-mb)/se;sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2));g=(ma-mb)/sp*(1-3/(4*(na+nb)-9)) if sp else 0
    df=(va/na+vb/nb)**2/((va/na)**2/(na-1)+(vb/nb)**2/(nb-1)); x=df/(df+t*t);A,B=df/2,.5
    def bcf(x,a,b):
        f=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1;d=1-qab*x/qap;d=1/(d if abs(d)>f else f);h=d
        for k in range(1,200):
            k2=2*k;aa=k*(b-k)*x/((qam+k2)*(a+k2));d=1+aa*d;d=1/(d if abs(d)>f else f);c=1+aa/c;c=c if abs(c)>f else f;h*=d*c
            aa=-(a+k)*(qab+k)*x/((a+k2)*(qap+k2));d=1+aa*d;d=1/(d if abs(d)>f else f);c=1+aa/c;c=c if abs(c)>f else f;de=d*c;h*=de
            if abs(de-1)<3e-12:break
        return h
    lb=math.lgamma(A)+math.lgamma(B)-math.lgamma(A+B)
    pp=(math.exp(A*math.log(x)+B*math.log(1-x)-lb)*bcf(x,A,B)/A) if x<(A+1)/(A+B+2) else (1-math.exp(A*math.log(x)+B*math.log(1-x)-lb)*bcf(1-x,B,A)/B)
    return round(ma,3),round(mb,3),round(g,2),round(max(0,min(1,pp)),4),na,nb
keep=set(LONG)-flagged
print('\n=== Robustness: headline contrasts on cleaned long sample (n=%d) ===' % len(keep))
# manipulation
T=[r['divergent_score'] for r in st if r['message_src']=='assistant' and r['group']=='treatment' and r['agent_role']=='Taylor' and r['conversation_id'] in keep]
A=[r['divergent_score'] for r in st if r['message_src']=='assistant' and r['group']=='treatment' and r['agent_role']=='Alex' and r['conversation_id'] in keep]
w=welch(T,A); print('Manipulation Taylor vs Alex divergent:', f'T={w[0]} C={w[1]} g={w[2]:+.2f} p={w[3]} (n {w[4]}/{w[5]})')
# preference
tids=[c for c in keep if LONG[c]=='treatment']
mt=[float(summ[c]['msgs_to_Taylor']) for c in tids]; ma=[float(summ[c]['msgs_to_Alex']) for c in tids]
w=welch(mt,ma); print('Preference msgs Taylor vs Alex (treatment):', f'T={w[0]} C={w[1]} g={w[2]:+.2f} p={w[3]} (n {w[4]}/{w[5]})')
print('  (full long sample was: manipulation g=1.31; preference g=1.15)')
