# -*- coding: utf-8 -*-
"""Generate Experiment-2-style figures for the Experiment 3 report.

Reads the output CSV/JSON artifacts and writes PNGs to report/figures/.
Robust to missing inputs (skips a figure if its data isn't ready yet).
"""
import json, csv, math, statistics, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = 'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 3'
FIG = f'{base}/report/figures'
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 150})
TREAT, CTRL = '#2c6fbb', '#9aa0a6'          # treatment blue, control gray
DIV, CON = '#1b9e77', '#d95f02'             # divergent teal, convergent orange

def ci95(xs):
    xs = [float(x) for x in xs]
    if len(xs) < 2: return (statistics.mean(xs) if xs else 0), 0
    return statistics.mean(xs), 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))

def load_json(p):
    try: return json.load(open(p, encoding='utf-8'))
    except FileNotFoundError: return None
def load_csv(p):
    try: return list(csv.DictReader(open(p, encoding='utf-8')))
    except FileNotFoundError: return None

# ---------- FIG 1: manipulation check (stance by group x persona) ----------
stance = load_json(f'{base}/outputs/stance_per_message_full.json')
if stance:
    cells = [('treatment','Taylor','Treatment\nTaylor (div.)'), ('treatment','Alex','Treatment\nAlex (conv.)'),
             ('control','Taylor','Control\nTaylor'), ('control','Alex','Control\nAlex')]
    D, Derr, C, Cerr, labels = [], [], [], [], []
    for g, p, lab in cells:
        sub = [r for r in stance if r['message_src']=='assistant' and r['group']==g and r['agent_role']==p]
        dm, de = ci95([r['divergent_score'] for r in sub]); cm, ce = ci95([r['convergent_score'] for r in sub])
        D.append(dm); Derr.append(de); C.append(cm); Cerr.append(ce); labels.append(lab)
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x-w/2, D, w, yerr=Derr, capsize=3, color=DIV, label='Divergent stance')
    ax.bar(x+w/2, C, w, yerr=Cerr, capsize=3, color=CON, label='Convergent stance')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Mean stance score (0–1)')
    ax.set_title('Manipulation check: persona stance by condition (assistant turns)')
    ax.legend(frameon=False, loc='upper right')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig1_manipulation_check.png'); plt.close(fig)
    print('fig1 OK')

# ---------- FIG 2: persona engagement (messages to each persona) ----------
summ = load_csv(f'{base}/outputs/experiment3_conversations_summary.csv')
if summ:
    full = [r for r in summ if r['quality_tier']=='full']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, grp, title in [(ax1,'treatment','Treatment'), (ax2,'control','Control')]:
        rs=[r for r in full if r['group']==grp]
        tm,te=ci95([r['msgs_to_Taylor'] for r in rs]); am,ae=ci95([r['msgs_to_Alex'] for r in rs])
        ax.bar([0,1],[tm,am],yerr=[te,ae],capsize=4,color=[DIV,CON],width=0.6)
        ax.set_xticks([0,1]); ax.set_xticklabels(['Divergent\n(Taylor)','Convergent\n(Alex)'])
        ax.set_title(f'{title} (n={len(rs)})'); ax.set_ylim(0, 9)
        if ax is ax1: ax.set_ylabel('Mean messages addressed (±95% CI)')
    fig.suptitle('Persona engagement: messages addressed to each persona', y=1.0)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig2_persona_engagement.png'); plt.close(fig)
    print('fig2 OK')

# ---------- FIG 3: divergence->convergence trajectory (quartile) ----------
if stance:
    by=defaultdict(list)
    for r in stance:
        if r['message_src']=='assistant': by[r['conversation_id']].append(r)
    for c in by: by[c].sort(key=lambda x:(x['timestamp'],x['message_id']))
    grp_of={c:by[c][0]['group'] for c in by}
    def quart_means(group):
        Q=[[],[],[],[]]
        for c,ms in by.items():
            if grp_of[c]!=group or len(ms)<4: continue
            n=len(ms)
            for qi in range(4):
                seg=ms[qi*n//4:(qi+1)*n//4]
                if seg: Q[qi].append(statistics.mean(r['d_minus_c'] for r in seg))
        return [statistics.mean(q) if q else float('nan') for q in Q], [ci95(q)[1] if len(q)>1 else 0 for q in Q]
    tm,te=quart_means('treatment'); cm,ce=quart_means('control')
    fig,ax=plt.subplots(figsize=(7.5,4.3))
    qx=[1,2,3,4]
    ax.errorbar(qx,tm,yerr=te,marker='o',color=TREAT,capsize=3,label='Treatment',lw=2)
    ax.errorbar(qx,cm,yerr=ce,marker='s',color=CTRL,capsize=3,label='Control',lw=2)
    ax.axhline(0,color='k',lw=0.6,ls=':')
    ax.set_xticks(qx); ax.set_xlabel('Conversation quartile'); ax.set_ylabel('Assistant divergent–convergent balance (D−C)')
    ax.set_title('Divergence-to-convergence trajectory across the conversation')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig3_trajectory.png'); plt.close(fig)
    print('fig3 OK')

# ---------- FIG 3b: divergent-share first vs second half (choreography) ----------
sw = load_csv(f'{base}/outputs/switching_per_conv.csv')
if sw:
    fig,ax=plt.subplots(figsize=(7,4.3))
    for grp,color,mark in [('treatment',TREAT,'o'),('control',CTRL,'s')]:
        rs=[r for r in sw if r['group']==grp]
        f1=[float(r['div_share_first_half']) for r in rs]; f2=[float(r['div_share_second_half']) for r in rs]
        m1,e1=ci95(f1); m2,e2=ci95(f2)
        ax.errorbar([0,1],[m1,m2],yerr=[e1,e2],marker=mark,color=color,capsize=4,lw=2,
                    label=f'{grp.capitalize()} (n={len(rs)})')
    ax.set_xticks([0,1]); ax.set_xticklabels(['First half','Second half'])
    ax.set_ylabel('Share of messages to divergent persona'); ax.set_ylim(0.2,1.0)
    ax.axhline(0.5,color='k',lw=0.6,ls=':')
    ax.set_title('Front-loaded divergent engagement, drifting toward convergent')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig3b_choreography.png'); plt.close(fig)
    print('fig3b OK')

# ---------- FIG 4: topic-controlled originality (effect sizes) ----------
def g_from(rows, key, fil=lambda r: True):
    T=[float(r[key]) for r in rows if r['group']=='treatment' and fil(r) and r[key] not in ('','nan')]
    C=[float(r[key]) for r in rows if r['group']=='control' and fil(r) and r[key] not in ('','nan')]
    if len(T)<2 or len(C)<2: return None
    na,nb=len(T),len(C); ma,mb=statistics.mean(T),statistics.mean(C)
    va,vb=statistics.variance(T),statistics.variance(C)
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2))
    if sp==0: return 0
    return (ma-mb)/sp*(1-3/(4*(na+nb)-9))

bars=[]
nrm=load_csv(f'{base}/outputs/dose_response_normalized.csv')
if nrm:
    for k,lab in [('ideas_per_user_turn','Idea rate\n(ideas / user turn)'),
                  ('flex_per_idea','Categories\nper idea'),
                  ('within_idea_diversity','Within-portfolio\nidea diversity'),
                  ('orig_same','Idea-portfolio\noriginality')]:
        g=g_from(nrm,k)
        if g is not None: bars.append((lab,g))
emb=load_csv(f'{base}/outputs/originality_topic_controlled.csv')
if emb and 'resid_same_cond_originality' in emb[0]:
    g=g_from(emb,'resid_same_cond_originality')
    if g is not None: bars.append(('Embedding cross-check:\nresidualized originality', g))
jud=load_csv(f'{base}/outputs/judge_originality_gemini.csv')
if jud and len([r for r in jud if r.get('n_judges') not in ('','0')]) >= 30:
    g=g_from(jud,'orig_vs_challenge', fil=lambda r: r['challenge_id']!='other_unclear')
    if g is not None: bars.append(('Judge: originality\nvs challenge', g))
if bars:
    fig,ax=plt.subplots(figsize=(8,0.7*len(bars)+1.5))
    labs=[b[0] for b in bars]; gs=[b[1] for b in bars]
    colors=[TREAT if v>=0 else CON for v in gs]
    y=np.arange(len(bars))
    ax.barh(y,gs,color=colors,height=0.6)
    ax.axvline(0,color='k',lw=0.8)
    for thr in (0.2,0.5,0.8):
        ax.axvline(thr,color='gray',lw=0.5,ls=':'); ax.axvline(-thr,color='gray',lw=0.5,ls=':')
    ax.set_yticks(y); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel("Hedges' g (treatment − control), length-normalised")
    ax.set_title('Idea-portfolio creativity: effect sizes (all n.s.)')
    ax.set_xlim(-0.95, 1.05)
    for yi,v in zip(y,gs): ax.text(v+(0.02 if v>=0 else -0.02), yi, f'{v:+.2f}',
                                   va='center', ha='left' if v>=0 else 'right', fontsize=9)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig4_originality.png'); plt.close(fig)
    print('fig4 OK')

print('Figures written to', FIG)
