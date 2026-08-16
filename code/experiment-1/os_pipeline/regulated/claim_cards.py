"""Stage 11 — Claim cards (§ K) → 10_claim_cards.csv.

Builds claim rows from the adjudicated rubric, statistical models, and
existing process/originality findings. Each claim is assigned a strength
rating per § K2 and labeled with allowed/forbidden language per § K3.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT = ROOT / 'regulated_llm_reanalysis'


FORBIDDEN_LANG = (
    '"Persona interaction increased creativity."',
    '"Users were objectively more creative."',
    '"Personas clearly improved output quality."',
    '"ground-truth creativity"',
    '"validated creativity improvement"',
    '"true creativity"',
)

ALLOWED_LANG_TEMPLATE = (
    'Persona-guided interaction changed regulated interaction processes as '
    'measured by LLM-rubric proxy scores. Output-level claims are bounded '
    'to computational and proxy evidence; no human judges available.'
)


def _rate_strength(hedges_g, p, q=None) -> str:
    if pd.isna(hedges_g) or pd.isna(p):
        return 'speculative'
    sig = p < 0.05 or (q is not None and not pd.isna(q) and q < 0.05)
    abs_g = abs(hedges_g)
    if abs_g >= 0.5 and sig:
        return 'strong'
    if abs_g >= 0.3 and sig:
        return 'moderate'
    if abs_g >= 0.2 or sig:
        return 'weak'
    return 'speculative'


def main() -> None:
    rows: list[dict] = []
    stats_path = OUT / '09_statistical_models_summary.csv'
    if not stats_path.exists():
        print('[claim_cards] 09_statistical_models_summary.csv missing; skipping')
        return
    st = pd.read_csv(stats_path)

    # Claims from I1 Welch/condition effects
    rel = st[st.model == 'I1_condition_effect_episode']
    for i, r in rel.iterrows():
        strength = _rate_strength(r.get('hedges_g'), r.get('p'), r.get('q_fdr'))
        sign = 'positive' if (r.get('hedges_g', 0) or 0) >= 0 else 'negative'
        rows.append(dict(
            claim_id=f'C{i+1:03d}',
            claim_text=(
                f"LLM-rubric proxy '{r['criterion']}' was "
                f"{sign}ly associated with the Persona condition "
                f"(Hedges' g = {r.get('hedges_g', float('nan')):+.2f}, "
                f"p = {r.get('p', float('nan')):.3g}, "
                f"q_fdr = {r.get('q_fdr', float('nan')):.3g})."),
            claim_type='dyadic_process' if 'coregulation' in r['criterion'] else 'assistant_process',
            analysis_layer='episode-level rubric (LLM proxy)',
            evidence_files='05_episode_rubric_scores_adjudicated.csv; 09_statistical_models_summary.csv',
            primary_variables=r['criterion'],
            model_or_test='Welch t + Hedges g + BH-FDR',
            effect_direction=sign,
            effect_size_or_interval=f"g = {r.get('hedges_g', float('nan')):+.2f}",
            strength_rating=strength,
            confirmatory_or_exploratory='exploratory',
            limitations='Proxy judgment; no human judges; bundled scoring.',
            counterevidence='see 08_validation_and_bias_audit.csv',
            allowed_language=ALLOWED_LANG_TEMPLATE,
            forbidden_language='; '.join(FORBIDDEN_LANG),
        ))

    # Claims from family effects (I7)
    fam = st[st.model == 'I7_family_vs_gpt']
    for i, r in fam.iterrows():
        strength = _rate_strength(r.get('hedges_g'), r.get('p'))
        sign = 'positive' if (r.get('hedges_g', 0) or 0) >= 0 else 'negative'
        rows.append(dict(
            claim_id=f'F{len(rows)+1:03d}',
            claim_text=(
                f"Within persona family '{r['family']}', '{r['criterion']}' "
                f"differed from GPT with a {sign} between-subjects effect "
                f"(Hedges' g = {r.get('hedges_g', float('nan')):+.2f}, "
                f"p = {r.get('p', float('nan')):.3g})."),
            claim_type='persona_family',
            analysis_layer='episode-level rubric (LLM proxy)',
            evidence_files='05_episode_rubric_scores_adjudicated.csv; 09_statistical_models_summary.csv',
            primary_variables=f'{r["criterion"]} (family={r["family"]})',
            model_or_test='Welch t + Hedges g (between-subjects)',
            effect_direction=sign,
            effect_size_or_interval=f"g = {r.get('hedges_g', float('nan')):+.2f}",
            strength_rating=strength,
            confirmatory_or_exploratory='exploratory',
            limitations=('Proxy judgment; no human judges. For Rational and '
                         'BoundedRational families, small n — treat as exploratory.'),
            counterevidence='see 08_validation_and_bias_audit.csv',
            allowed_language=ALLOWED_LANG_TEMPLATE,
            forbidden_language='; '.join(FORBIDDEN_LANG),
        ))

    # Cross-references to existing findings (not re-computed here, just cited)
    cross_refs = [
        dict(claim_id='X001',
             claim_text=('Assistant proposes LESS under Persona than under GPT '
                         '(a_prop paired Δ = −0.51, t = −13.3, p < 10⁻²³, '
                         'dz = −1.35). The earlier r=+0.257 correlation between '
                         'a_prop and creativity must be read with this sign in mind.'),
             claim_type='assistant_process',
             analysis_layer='existing process layer',
             evidence_files='extension_paired.csv; 00_data_audit.md §5a',
             primary_variables='a_prop',
             model_or_test='paired t + Wilcoxon',
             effect_direction='negative',
             effect_size_or_interval='dz = −1.35',
             strength_rating='strong',
             confirmatory_or_exploratory='confirmatory (pre-existing)',
             limitations='Surface-move label; see rubric-layer for regulation quality.',
             counterevidence='',
             allowed_language=ALLOWED_LANG_TEMPLATE,
             forbidden_language='; '.join(FORBIDDEN_LANG)),
        dict(claim_id='X002',
             claim_text=('Extracted-idea portfolio fluency is HIGHER under '
                         'Persona than GPT (paired Δ = +1.18 ideas/round, '
                         'p < 10⁻⁵, dz = +0.53). Same-condition originality '
                         'is LOWER under Persona (paired Δ = −0.025, '
                         'p = 2×10⁻⁸, dz = −0.67) — the direction is opposite '
                         'to Experiment 2, plausibly because Experiment 1 '
                         'assigns persona while Experiment 2 allows elective use.'),
             claim_type='product_proxy',
             analysis_layer='agentic idea extraction (product proxy)',
             evidence_files='analysis_out/production/participant_originality.csv',
             primary_variables='n_ideas, orig_same, orig_all, orig_cross',
             model_or_test='paired t + Welch t + Hedges g',
             effect_direction='mixed',
             effect_size_or_interval='dz_fluency = +0.53; dz_orig_same = −0.67',
             strength_rating='moderate',
             confirmatory_or_exploratory='exploratory',
             limitations=('Originality is a COMPUTATIONAL proxy, not a human '
                          'novelty judgment. No human judges available.'),
             counterevidence='',
             allowed_language=ALLOWED_LANG_TEMPLATE,
             forbidden_language='; '.join(FORBIDDEN_LANG)),
    ]
    rows.extend(cross_refs)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / '10_claim_cards.csv', index=False)
    print(f'wrote {OUT / "10_claim_cards.csv"} ({len(df)} claims)')
    print(f'  strength distribution: {df["strength_rating"].value_counts().to_dict()}')


if __name__ == '__main__':
    main()
