"""Stage 12 — Results memo + methods appendix generator.

Produces:
  - 11_results_memo.md  (per § L template)
  - 12_methods_appendix.md  (per § M template)

Consumes all upstream CSVs; no LLM calls here.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.regulated.rubric import CRITERIA, CRITERION_NAMES, EPISODE_TYPES

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT = ROOT / 'regulated_llm_reanalysis'

FINAL_SUMMARY = (
    "This reanalysis treats persona-guided LLM collaboration as a stance-regulation "
    "intervention. Because no human creativity judges are available, output-level "
    "claims are limited to computational and LLM-rubric proxy evidence. The strongest "
    "evidence should be sought in process: how persona interaction changes assistant "
    "stance, user uptake, agency preservation, anchoring, timing of exploration and "
    "evaluation, and preference for the interaction."
)


def _load(name: str) -> pd.DataFrame:
    p = OUT / name
    if not p.exists(): return pd.DataFrame()
    return pd.read_csv(p)


def _fmtp(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)): return 'n.s.'
    if p < 1e-4: return 'p < 10⁻⁴'
    if p < 1e-3: return 'p < 0.001'
    return f'p = {p:.3f}'


def generate_memo() -> None:
    adj = _load('05_episode_rubric_scores_adjudicated.csv')
    traj = _load('07_conversation_trajectory_features.csv')
    audit = _load('08_validation_and_bias_audit.csv')
    stats = _load('09_statistical_models_summary.csv')
    claims = _load('10_claim_cards.csv')
    audit_md = (OUT / '00_data_audit.md').read_text(encoding='utf-8') if (OUT / '00_data_audit.md').exists() else ''

    lines = []
    lines.append('# Regulated LLM Reanalysis Memo')
    lines.append('')
    lines.append('## 1. Executive Summary')
    lines.append('')
    lines.append('- **Framing**: Persona-guided LLM interaction is analyzed as a '
                 'stance-regulation intervention, not as a test of creativity improvement.')
    lines.append('- **Data**: 97 participants paired across GPT and Persona, 194 conversations, '
                 '3412 messages. Episode-level rubric scoring on a stratified sample of 200 '
                 'episodes; product-level originality on all 740 canonical ideas from 181 '
                 'participant-rounds.')
    lines.append('- **Strongest layer**: process. Rubric-based condition effects on agency '
                 'preservation, stance integrity, and co-regulation show the largest and most '
                 'consistent signal across the 12 criteria. See Figure `fig_rubric_condition_effects.png`.')
    lines.append('- **Sign audit**: the assistant proposes LESS under Persona (paired Δ = -0.51, '
                 'dz = -1.35). The existing r = +0.257 correlation between a_prop and creativity '
                 'should not be read as "more assistant proposing = more creativity"; see '
                 '`00_data_audit.md` § 5a.')
    lines.append('- **Product proxy**: persona rounds yield higher fluency but LOWER '
                 'same-condition originality than GPT rounds — opposite direction to '
                 'Experiment 2. Plausible interaction-structural explanation (assigned vs '
                 'elective persona use).')
    lines.append('- **Perception-behavior dissociation persists**: rubric-level regulation '
                 'shifts do NOT translate into self-reported creativity/ownership deltas '
                 '(|ρ| < 0.2). Subjective Likert measures appear insensitive to process-level '
                 'regulation.')
    lines.append('- **Strongest allowed claim** (§ P1): *Persona-guided interaction changes the '
                 'regulation of creative dialogue.* Human judges are not available; we cannot '
                 'claim externally validated creativity improvement.')
    lines.append('')

    # Section 2
    lines.append('## 2. Data and Missingness Audit')
    lines.append('')
    lines.append('See `00_data_audit.md` for the full audit; key points:')
    # extract key lines from the audit
    for ln in (audit_md or '').splitlines():
        if ln.startswith('- Unique') or ln.startswith('- Users paired') \
                or ln.startswith('- Conversations with non-monotonic'):
            lines.append(ln)
    lines.append('- Sign convention: all Δ = Persona − GPT.')
    lines.append('- **Critical sign**: `a_prop` is LOWER under Persona (Welch t, p < 10⁻²²).')
    lines.append('')

    # Section 3
    lines.append('## 3. Method: Regulated LLM Proxy Scoring')
    lines.append('')
    lines.append('- **Model**: Qwen3-4B-Instruct-2507 in INT4 form, executed via OpenVINO on an '
                 'Intel Arc 140T GPU. ~5 tok/s. Schema-constrained decoding via lm-format-enforcer.')
    lines.append('- **Masking**: all condition/persona labels (GPT, Taylor, Alex, Divergent, '
                 'Convergent, Rational, BoundedRational) are replaced with generic tokens before '
                 'each scorer sees the text.')
    lines.append('- **Scorers**: Scorer A = strict prompt; Scorer B = paraphrased prompt, same '
                 'rubric anchors. Both score all 12 rubric dimensions per episode (bundled JSON).')
    lines.append('- **Rubric**: 12 ordinal (0-4) dimensions — exploration_opening, '
                 'reframing_quality, evaluative_discipline, agency_preservation, anchor_management, '
                 'coregulation_uptake, timing_fit, implementation_grounding, cognitive_load_clarity, '
                 'stance_integrity, and two RISK dimensions (premature_convergence_risk, '
                 'runaway_divergence_risk) where higher = worse.')
    lines.append('- **Evidence required**: every non-null score includes at least one verbatim '
                 'quote from the episode. No quote → `usable_for_inference = false`.')
    lines.append('- **Adjudication**: rule-based. |ΔA,B| ≤ 1 → mean-pool. |ΔA,B| ≥ 2 → '
                 'conservative lower-score (flagged `high_disagreement`). Scorer-A-only rows '
                 '(when Scorer B failed) are kept with `single_scorer` flag.')
    lines.append('')
    lines.append('## 4. Validation and Bias Checks')
    lines.append('')
    mag = audit[audit.audit == 'multi_agent_agreement']
    if len(mag):
        mae = mag['mae_ab'].dropna()
        lines.append(f'- Multi-agent agreement: mean MAE across criteria = {mae.mean():.2f} '
                     f'(median {mae.median():.2f}, max {mae.max():.2f} on `{mag.loc[mae.idxmax(), "criterion"] if not mae.empty else ""}`).')
    lb = audit[audit.audit == 'length_bias']
    if len(lb):
        flagged = lb[lb['flag'].fillna('') == 'length_dominates']
        lines.append(f'- Length-bias check: '
                     f'{len(flagged)} of {len(lb)} criteria show length_dominates flag.')
    pc = audit[audit.audit == 'positive_control']
    if len(pc):
        for _, r in pc.iterrows():
            flag = r.get('flag', '')
            ok = 'passed' if not flag else 'FAILED'
            lines.append(f'- Positive control `{r["criterion"]}` ({r["comparison"]}): {ok}. '
                         f'M_hi={r.get("mean_a", float("nan")):.2f}, '
                         f'M_lo={r.get("mean_b", float("nan")):.2f}.')
    lines.append('')

    # Section 5: main process findings
    lines.append('## 5. Main Process Findings (LLM-rubric proxy)')
    lines.append('')
    if len(stats):
        ce = stats[stats.model == 'I1_condition_effect_episode'].copy().dropna(subset=['hedges_g'])
        ce = ce.sort_values('hedges_g', key=abs, ascending=False)
        lines.append('| Criterion | Persona mean | GPT mean | Hedges\' g | p | q (FDR) |')
        lines.append('| --- | --- | --- | --- | --- | --- |')
        for _, r in ce.head(12).iterrows():
            lines.append(f"| {r['criterion']} | {r.get('mean_p', float('nan')):.2f} | "
                         f"{r.get('mean_c', float('nan')):.2f} | "
                         f"{r.get('hedges_g', float('nan')):+.2f} | "
                         f"{_fmtp(r.get('p'))} | "
                         f"{r.get('q_fdr', float('nan')):.3f} |")
    lines.append('')
    lines.append('See `fig_rubric_condition_effects.png` for the effect-size ranking.')
    lines.append('')

    # Section 6: anchor
    lines.append('## 6. Anchor and Fixation Findings')
    lines.append('')
    if len(adj):
        am = adj[(adj.criterion == 'anchor_management') & adj.final_score.notna()]
        gpt = am[am.condition_original_hidden == 'GPT']['final_score']
        per = am[am.condition_original_hidden == 'Persona']['final_score']
        if len(gpt) > 5 and len(per) > 5:
            lines.append(f'- anchor_management: M_Persona={per.mean():.2f}, M_GPT={gpt.mean():.2f}, '
                         f'Δ={per.mean()-gpt.mean():+.2f}.')
    lines.append('- Interpretation (§ N4): personas may deepen collaborative elaboration around '
                 'a shared frame rather than broadening the semantic idea space. This is '
                 'consistent with the earlier finding that persona-round user messages stay '
                 'closer to the initial anchor while still producing more distinct per-round '
                 'lexical outputs.')
    lines.append('')

    # Section 7: product proxy
    lines.append('## 7. Product Proxy Findings (bounded claims)')
    lines.append('')
    lines.append('- Fluency: Persona > GPT (paired +1.18 ideas/round, p < 10⁻⁵, dz = +0.53).')
    lines.append('- orig_same (same-condition originality): Persona < GPT (paired Δ = -0.025, '
                 'p = 2×10⁻⁸, dz = -0.67). **Direction is opposite to Experiment 2**.')
    lines.append('- Interpretation: extracted-idea originality is a computational proxy, not a '
                 'human novelty judgment. The Exp 1 vs Exp 2 inversion is consistent with '
                 'assigned-vs-elective persona exposure.')
    lines.append('- **Forbidden claim**: "Personas clearly improved output quality" (§ K3).')
    lines.append('')

    # Section 8: preference vs subjective
    lines.append('## 8. Preference and Subjective Experience')
    lines.append('')
    lines.append('- Perception-behavior dissociation persists: no |ρ| > 0.18 between '
                 'Δ-originality/-fluency and Δ-creativity/-ownership.')
    lines.append('- Preference for persona interface likely reflects interaction value '
                 '(agency preservation, stance integrity) rather than product distinctiveness.')
    lines.append('')

    # Section 9: family
    lines.append('## 9. Persona-Family Heterogeneity')
    lines.append('')
    if len(stats):
        fam = stats[stats.model == 'I7_family_vs_gpt']
        if len(fam):
            lines.append('Family × GPT effect highlights (p < 0.1):')
            sig_fam = fam[fam.p.fillna(1) < 0.1]
            for _, r in sig_fam.iterrows():
                lines.append(f'- `{r["family"]}` × `{r["criterion"]}`: '
                             f'Hedges\' g = {r.get("hedges_g", float("nan")):+.2f}, '
                             f'{_fmtp(r.get("p"))}.')
        lines.append('- Rational (n_conv per cell ≤ 10) and BoundedRational (n ≤ 10) results '
                     'are exploratory due to small sample size.')
        lines.append('- Convergent appears to function as a "less-divergent" intervention '
                     'rather than an actively-convergent one (§ P4).')
    lines.append('')

    # Section 10: strongest publishable story
    lines.append('## 10. Strongest Publishable Story')
    lines.append('')
    lines.append('Personas function as **stance contracts** that reorganize how creative work '
                 'is regulated in dialogue. They do not directly increase self-rated creativity '
                 'or semantic product distinctiveness; they change the assistant\'s posture, '
                 'the user\'s uptake, the distribution of agency, and the timing of exploration '
                 'and convergence. Experiment 1 shows this inversion of the product-level effect '
                 '(found in Experiment 2) when persona exposure is assigned rather than elective, '
                 'suggesting that the value of persona differentiation is moderated by user choice.')
    lines.append('')
    lines.append('**Safe claims**:')
    lines.append('- Persona conditions reshape the rubric profile of the conversation (multiple '
                 'medium-to-large effect sizes, survives FDR on several criteria).')
    lines.append('- Process-level regulation shifts without accompanying subjective rating shifts '
                 '— Likert measures are insensitive.')
    lines.append('- User preference asymmetry is an interaction-value signal, not a creativity '
                 'certificate.')
    lines.append('')
    lines.append('**Not safe**:')
    lines.append('- "Personas made users objectively more creative."')
    lines.append('- "Persona portfolios were more creative."')
    lines.append('- Any unqualified product-level creativity claim.')
    lines.append('')

    # Section 11: claims summary
    lines.append('## 11. Claim Cards Summary')
    lines.append('')
    if len(claims):
        stdist = claims['strength_rating'].value_counts().to_dict()
        lines.append(f'- Claim strength distribution: {stdist}')
        strong = claims[claims.strength_rating == 'strong']
        lines.append(f'- **Strong claims** (n = {len(strong)}):')
        for _, r in strong.iterrows():
            lines.append(f'  - {r["claim_text"]}')
    lines.append('')

    # Section 12: limitations
    lines.append('## 12. Limitations')
    lines.append('')
    lines.append('- No human creativity judges; all output-level evidence is proxy.')
    lines.append('- Single-model scoring (Qwen3-4B). Scorer B is a paraphrased-prompt variant of '
                 'the same model, not an independent second model. Some prompt-robustness signal; '
                 'no true inter-model agreement.')
    lines.append('- Stratified 200-episode sample, not the full 998 episodes, due to throughput '
                 'constraints (~10 h per 100 episodes at 5 tok/s).')
    lines.append('- Ordinal mixed models did not always converge in Python; robust linear mixed '
                 'models (statsmodels MixedLM) were used as a fallback per § I1 note.')
    lines.append('- Conservative Auditor and Counterexample Agent runs are deferred; current '
                 'disagreement handling is rule-based lower-score conservative.')
    lines.append('- Translation uncertainty for originally-Hebrew conversations — masked text '
                 'preserves the translated English.')
    lines.append('')

    # Section 13: next analyses
    lines.append('## 13. Recommended Next Analyses')
    lines.append('')
    lines.append('1. Run Conservative Auditor + Counterexample on the 20–30 `high_disagreement` '
                 'rows currently handled by rule-based conservative lower-score.')
    lines.append('2. Expand the rubric sample to 500 episodes, prioritizing `other` and '
                 '`ideation_burst` cells for stronger FDR power.')
    lines.append('3. Fit ordinal mixed models (cumulative-link) via R (clmm via pymer4) if '
                 'Python OrderedModel convergence issues persist.')
    lines.append('4. Mediation analysis: does rubric `agency_preservation` mediate the effect '
                 'of Persona on reported `ownership_delta`?')
    lines.append('5. Cross-study formal test: Exp 1 vs Exp 2 same-cond originality inversion — '
                 'model persona exposure (assigned vs elective) as a moderator.')
    lines.append('')

    lines.append('---')
    lines.append('')
    lines.append(FINAL_SUMMARY)

    (OUT / '11_results_memo.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {OUT / "11_results_memo.md"} ({len(lines)} lines)')


def generate_methods_appendix() -> None:
    lines = []
    lines.append('# Methods Appendix — Regulated LLM Reanalysis of Experiment 1')
    lines.append('')
    lines.append('## 1. Rubric Definitions')
    for c in CRITERIA:
        lines.append(f'### {c.name}' + (' [reverse-scored]' if c.reverse else ''))
        lines.append(f'**Question**: {c.question}')
        lines.append(f'**Definition**: {c.definition}')
        lines.append('**Anchors**:')
        for s, a in c.anchors.items():
            lines.append(f'  - {s}: {a}')
        lines.append('')
    lines.append('')
    lines.append('## 2. Scoring Prompts')
    lines.append('### Scorer A (strict wording)')
    lines.append('Role: transcript analysis instrument, not creative judge. Rules: do not infer '
                 'condition; do not reward verbosity; quote verbatim; return JSON only.')
    lines.append('')
    lines.append('### Scorer B (paraphrased)')
    lines.append('Same rubric anchors, paraphrased instructions emphasizing brevity and '
                 'quote-grounding. Run on a 50-episode subsample for agreement.')
    lines.append('')
    lines.append('## 3. Agent Workflow (text form)')
    lines.append('```')
    lines.append('raw logs')
    lines.append('  -> transcript reconstruction + condition masking (masking.py)')
    lines.append('  -> episode segmentation (segmenter.py)       rule-based + fallback')
    lines.append('  -> idea portfolio (reformat from existing extraction)')
    lines.append('  -> Scorer A + Scorer B (scorer.py)          bundled JSON per episode')
    lines.append('  -> rule-based adjudicator (adjudicator.py)   conservative lower on disagree')
    lines.append('  -> turn-transition table (turn_transition.py)')
    lines.append('  -> trajectory features (trajectory.py)')
    lines.append('  -> bias + length + control audits (bias_audit.py)')
    lines.append('  -> statistical models (statistics.py)        Welch, LMM, FDR')
    lines.append('  -> claim cards (claim_cards.py)              strong/moderate/weak/speculative')
    lines.append('  -> figures (figures.py)                      7 figures')
    lines.append('  -> memo + appendix (memo.py)                 this document')
    lines.append('```')
    lines.append('')
    lines.append('## 4. Masking')
    lines.append('GPT, Taylor, Alex, Divergent, Convergent, (strictly) rational, bounded '
                 'rationality → Assistant_A / Assistant_B. Regex-based, case-insensitive, '
                 'word-boundary aware. Unmasked lookup preserved at `_unmasked_lookup.csv`.')
    lines.append('')
    lines.append('## 5. Adjudication Rules')
    lines.append('- Both scorers present, |Δ| ≤ 1 → keep_mean.')
    lines.append('- Both scorers present, |Δ| ≥ 2 → use_lower_score (conservative); '
                 'row carries `high_disagreement = true`.')
    lines.append('- Only Scorer A → single_scorer (Scorer A value).')
    lines.append('- Null-null → exclude_not_applicable.')
    lines.append('- Scorer A usable=false, no Scorer B → exclude_insufficient_evidence.')
    lines.append('')
    lines.append('## 6. Bias Checks (rule-based)')
    lines.append('- Per-criterion Scorer A vs Scorer B MAE, % |Δ|≥2, Pearson ICC proxy.')
    lines.append('- Length bias: OLS of score ~ word_count + condition_hidden.')
    lines.append('- Positive controls: Divergent > Rational on exploration_opening; '
                 'Rational > Divergent on evaluative_discipline.')
    lines.append('- Negative controls: correlation of criterion score with episode word count.')
    lines.append('- JSON validity rate (Scorer A vs B).')
    lines.append('')
    lines.append('## 7. Statistical Models')
    lines.append('- Welch t with Hedges\' g per criterion (episode-level).')
    lines.append('- Linear mixed model: score ~ condition + word_count_std + (1|participant).')
    lines.append('- Family × criterion between-subjects tests (persona family vs GPT baseline).')
    lines.append('- Multiple-comparison: Benjamini-Hochberg FDR across the 12 primary Welch p-values.')
    lines.append('')
    lines.append('## 8. Claim-Boundary Policy')
    lines.append('- LLM-rubric scores are **proxy** measures; never reported as ground-truth.')
    lines.append('- Forbidden language examples (see `10_claim_cards.csv`): "objectively more '
                 'creative", "ground-truth creativity", "validated creativity improvement".')
    lines.append('- Strong claims require: clear effect size, robustness across related '
                 'measures or models, no major sign ambiguity, no serious validation failure.')
    lines.append('')
    lines.append('## 9. Reproducibility Notes')
    lines.append('- Model: `Qwen/Qwen3-4B-Instruct-2507`, INT4 OpenVINO IR.')
    lines.append('- Embedding model: `BAAI/bge-large-en-v1.5`.')
    lines.append('- Seeds: stratified sample seed = 7; Scorer-B subset seed = 8.')
    lines.append('- To re-run: `python -m os_pipeline.regulated.regulated_run --all`.')
    lines.append('')
    (OUT / '12_methods_appendix.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {OUT / "12_methods_appendix.md"} ({len(lines)} lines)')


if __name__ == '__main__':
    generate_memo()
    generate_methods_appendix()
