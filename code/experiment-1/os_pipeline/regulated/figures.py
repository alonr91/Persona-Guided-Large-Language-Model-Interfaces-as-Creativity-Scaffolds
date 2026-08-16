"""Stage 10 — Figures for the regulated reanalysis.

Produces the 7 figures specified in § Required Output Files. Each figure is
saved under regulated_llm_reanalysis/figures/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT = ROOT / 'regulated_llm_reanalysis'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'figure.dpi':120, 'savefig.dpi':160, 'font.size':10})

CRITERIA_ORDER = (
    'exploration_opening','reframing_quality','evaluative_discipline',
    'agency_preservation','anchor_management','coregulation_uptake',
    'timing_fit','implementation_grounding','cognitive_load_clarity',
    'stance_integrity','premature_convergence_risk','runaway_divergence_risk',
)
FAMILY_ORDER = ('Divergent','Convergent','Rational','BoundedRational')
COLORS = {'GPT':'#6c6c6c','Persona':'#1a7a8b',
          'Divergent':'#2a9d8f','Convergent':'#e76f51',
          'Rational':'#6f62b6','BoundedRational':'#e9c46a'}


def _hedges_g(a, b):
    a = pd.to_numeric(a, errors='coerce').dropna()
    b = pd.to_numeric(b, errors='coerce').dropna()
    if len(a)<5 or len(b)<5: return float('nan'), float('nan'), float('nan')
    sp = (((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))**0.5
    d = (a.mean()-b.mean())/sp if sp>0 else float('nan')
    J = 1 - 3/(4*(len(a)+len(b))-9)
    return float(d), float(d*J), float(a.mean()-b.mean())


def fig_rubric_condition_effects(adj: pd.DataFrame):
    rows = []
    for crit in CRITERIA_ORDER:
        g = adj[(adj.criterion == crit) & adj.final_score.notna()]
        a = g.loc[g.condition_original_hidden=='Persona', 'final_score']
        b = g.loc[g.condition_original_hidden=='GPT', 'final_score']
        _, g_es, diff = _hedges_g(a, b)
        rows.append(dict(criterion=crit, hedges_g=g_es, diff=diff,
                         n_p=len(a), n_c=len(b)))
    d = pd.DataFrame(rows).sort_values('hedges_g')
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2a9d8f' if v>=0 else '#e76f51' for v in d['hedges_g']]
    ax.barh(d['criterion'], d['hedges_g'], color=colors, alpha=0.85)
    ax.axvline(0, color='black', lw=0.7)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r['hedges_g'] + (0.02 if r['hedges_g']>=0 else -0.02), i,
                f"{r['hedges_g']:+.2f}  (Δ={r['diff']:+.2f})",
                va='center', ha='left' if r['hedges_g']>=0 else 'right', fontsize=9)
    ax.set_xlabel("Hedges' g  (Persona − GPT)")
    ax.set_title('Rubric condition effects, episode-level (positive = Persona higher)')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_rubric_condition_effects.png')
    plt.close()


def fig_regulation_trajectory_by_condition(adj: pd.DataFrame, eps: pd.DataFrame):
    eps = eps.sort_values(['conversation_id','start_turn']).copy()
    eps['ep_order'] = eps.groupby('conversation_id').cumcount()
    eps['ep_total'] = eps.groupby('conversation_id')['ep_order'].transform('max') + 1
    eps['phase'] = pd.cut(eps['ep_order'] / eps['ep_total'],
                           bins=[-0.01,0.33,0.67,1.0],
                           labels=['early','mid','late'])
    m = adj.merge(eps[['episode_id','phase']], on='episode_id', how='left')
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharey=True)
    for ax, crit in zip(axes.ravel(), CRITERIA_ORDER):
        gb = (m[(m.criterion==crit) & m.final_score.notna()]
               .groupby(['condition_original_hidden','phase'], observed=True)
               ['final_score'].mean().reset_index())
        for cond in ['GPT','Persona']:
            sub = gb[gb.condition_original_hidden==cond]
            ax.plot(sub['phase'].astype(str), sub['final_score'], marker='o',
                    color=COLORS[cond], label=cond)
        ax.set_title(crit, fontsize=9)
        ax.set_ylim(0, 4)
        ax.grid(alpha=0.25)
    axes[0,0].legend(fontsize=8)
    plt.suptitle('Figure — rubric means across conversation phase, by condition', y=1.01)
    plt.tight_layout()
    plt.savefig(FIG / 'fig_regulation_trajectory_by_condition.png', bbox_inches='tight')
    plt.close()


def fig_assistant_to_user_uptake(tt: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in ['GPT','Persona']:
        sub = tt[tt.condition == cond].dropna(subset=['a_agency_preservation','u_coregulation_uptake'])
        if len(sub)<10: continue
        bins = pd.cut(sub['a_agency_preservation'], bins=[-0.5,0.5,1.5,2.5,3.5,4.5],
                      labels=[0,1,2,3,4])
        gb = sub.groupby(bins, observed=True)['u_coregulation_uptake'].agg(['mean','sem']).reset_index()
        gb['a_agency_preservation'] = gb['a_agency_preservation'].astype(float)
        ax.errorbar(gb['a_agency_preservation'], gb['mean'], yerr=gb['sem'],
                    marker='o', capsize=4, color=COLORS[cond], label=cond)
    ax.set_xlabel('Assistant prior-turn agency_preservation (0-4)')
    ax.set_ylabel('User next-episode coregulation_uptake (0-4)')
    ax.set_title('Figure — assistant regulation → user uptake, by condition')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / 'fig_assistant_to_user_uptake.png')
    plt.close()


def fig_anchor_management_by_condition(traj: pd.DataFrame, adj: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    g = adj[(adj.criterion=='anchor_management') & adj.final_score.notna()]
    gpt = g.loc[g.condition_original_hidden=='GPT','final_score'].values
    per = g.loc[g.condition_original_hidden=='Persona','final_score'].values
    ax.boxplot([gpt, per], labels=['GPT','Persona'],
               patch_artist=True, widths=0.45, showfliers=False)
    ax.scatter(np.random.normal(1, 0.04, len(gpt)), gpt, alpha=0.3, color=COLORS['GPT'])
    ax.scatter(np.random.normal(2, 0.04, len(per)), per, alpha=0.3, color=COLORS['Persona'])
    t, p = stats.ttest_ind(gpt, per, equal_var=False) if len(gpt)>=5 and len(per)>=5 else (np.nan, np.nan)
    ax.set_title(f'Figure — anchor_management rubric (episode-level)\n'
                 f'GPT mean={np.mean(gpt):.2f}, Persona mean={np.mean(per):.2f}, '
                 f'Welch p={p:.3g}' if not np.isnan(p) else 'Figure — anchor_management rubric')
    ax.set_ylabel('anchor_management score (0-4)')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_anchor_management_by_condition.png')
    plt.close()


def fig_persona_family_rubric_profiles(adj: pd.DataFrame):
    import matplotlib.pyplot as plt
    fams = ['GPT'] + list(FAMILY_ORDER)
    means = {}
    for fam in fams:
        means[fam] = {}
        for crit in CRITERIA_ORDER:
            sub = adj[(adj.criterion==crit) & (adj.persona_family_original_hidden==fam) & adj.final_score.notna()]
            means[fam][crit] = float(sub['final_score'].mean()) if len(sub) else np.nan
    # polar radar
    angles = np.linspace(0, 2*np.pi, len(CRITERIA_ORDER), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, polar=True)
    for fam in fams:
        vals = [means[fam][c] for c in CRITERIA_ORDER] + [means[fam][CRITERIA_ORDER[0]]]
        ax.plot(angles, vals, marker='o', label=fam,
                color=COLORS.get(fam, 'black'), linewidth=1.5)
        ax.fill(angles, vals, alpha=0.08, color=COLORS.get(fam, 'black'))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace('_','\n') for c in CRITERIA_ORDER], fontsize=8)
    ax.set_ylim(0, 4)
    ax.set_title('Figure — rubric profiles by persona family (0-4)', y=1.08)
    ax.legend(loc='lower right', bbox_to_anchor=(1.2, -0.05), fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / 'fig_persona_family_rubric_profiles.png', bbox_inches='tight')
    plt.close()


def fig_validation_disagreement_heatmap(adj: pd.DataFrame):
    # disagreement rate per (criterion × condition)
    d = adj.dropna(subset=['score_0_4_A','score_0_4_B']).copy()
    if len(d) == 0:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.text(0.5, 0.5, 'no dual-scorer rows', ha='center', va='center')
        plt.savefig(FIG / 'fig_validation_disagreement_heatmap.png'); plt.close()
        return
    d['abs_diff'] = (d['score_0_4_A'] - d['score_0_4_B']).abs()
    mat = d.pivot_table(index='criterion', columns='condition_original_hidden',
                         values='abs_diff', aggfunc='mean')
    mat = mat.reindex(index=CRITERIA_ORDER)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mat.values, cmap='Reds', vmin=0, vmax=2, aspect='auto')
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i,j]
            if np.isnan(v): continue
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=9,
                     color='black' if v<1.2 else 'white')
    plt.colorbar(im, ax=ax, label='mean |Scorer A − Scorer B|')
    ax.set_title('Figure — Scorer A vs Scorer B mean absolute disagreement')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_validation_disagreement_heatmap.png')
    plt.close()


def fig_preference_model_effects(stats_df: pd.DataFrame):
    # placeholder: show condition effect per criterion (Hedges' g) plus FDR-adjusted significance
    rel = stats_df[stats_df.model == 'I1_condition_effect_episode'].copy()
    if len(rel) == 0:
        return
    rel = rel.sort_values('hedges_g')
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e76f51' if q<0.05 else '#888' for q in rel.get('q_fdr', rel['p'])]
    ax.barh(rel['criterion'], rel['hedges_g'], color=colors, alpha=0.85)
    ax.axvline(0, color='black', lw=0.7)
    for i, (_, r) in enumerate(rel.iterrows()):
        q = r.get('q_fdr', r['p'])
        ax.text(r['hedges_g'], i, f"  q={q:.3f}", va='center', fontsize=8)
    ax.set_xlabel("Hedges' g (Persona − GPT)")
    ax.set_title('Figure — preference-model-style summary of condition effects\n'
                 '(red bars significant at q<0.05 after FDR correction)')
    plt.tight_layout()
    plt.savefig(FIG / 'fig_preference_model_effects.png')
    plt.close()


def main():
    adj = pd.read_csv(OUT / '05_episode_rubric_scores_adjudicated.csv')
    eps = pd.read_csv(OUT / '02_episode_table.csv')
    traj = pd.read_csv(OUT / '07_conversation_trajectory_features.csv')
    tt_path = OUT / '06_turn_transition_table.csv'
    tt = pd.read_csv(tt_path) if tt_path.exists() else pd.DataFrame()
    st_path = OUT / '09_statistical_models_summary.csv'
    stats_df = pd.read_csv(st_path) if st_path.exists() else pd.DataFrame()

    fig_rubric_condition_effects(adj)
    fig_regulation_trajectory_by_condition(adj, eps)
    if len(tt): fig_assistant_to_user_uptake(tt)
    fig_anchor_management_by_condition(traj, adj)
    fig_persona_family_rubric_profiles(adj)
    fig_validation_disagreement_heatmap(adj)
    if len(stats_df): fig_preference_model_effects(stats_df)
    print(f'figures saved to {FIG}')


if __name__ == '__main__':
    main()
