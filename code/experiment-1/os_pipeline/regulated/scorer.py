"""Stage 4 — Rubric scoring with bundled criteria.

Scorer A and Scorer B both run Qwen3-4B-Instruct on the same masked episode,
but with paraphrased prompt templates for prompt-robustness. Each returns a
JSON list with one entry per criterion; schema-constrained via
lm-format-enforcer so malformed JSON is impossible.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from os_pipeline.llm_client import LLMClient
from os_pipeline.regulated.rubric import (
    CRITERIA, CRITERION_NAMES, EPISODE_TYPE_TO_CRITERIA,
    BundledEpisodeScore, CriterionScore, BIAS_FLAGS_ALLOWED,
)

# Scorer C (Gemini) is imported lazily so the local-only path doesn't pay
# the import cost or require google-genai to be installed.
def _get_gemini_client():
    from os_pipeline.gemini_client import GeminiClient
    return GeminiClient

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass


# -------- Prompt templates --------

_RUBRIC_TEXT_BLOCK = '\n\n'.join(
    f"Criterion: {c.name}\n"
    f"Question: {c.question}\n"
    f"Definition: {c.definition}\n"
    "Anchors:\n" +
    '\n'.join(f"  {s} = {a}" for s, a in c.anchors.items()) +
    (f"\n  [NOTE: higher = WORSE for this criterion]" if c.reverse else '')
    for c in CRITERIA
)


SYSTEM_A = (
    "You are a transcript analysis instrument, not a creative judge. "
    "You score a MASKED episode from a human-AI creative collaboration.\n\n"
    "RULES:\n"
    "- Never infer the experimental condition.\n"
    "- Never reward length, fluency, politeness, or confidence.\n"
    "- Use only evidence present in the episode.\n"
    "- Score each applicable criterion exactly as defined.\n"
    "- If evidence is insufficient for a criterion, return score_0_4 = null and "
    "usable_for_inference = false.\n"
    "- Every non-null score must include at least one exact evidence quote "
    "copied verbatim from the episode (same words, same punctuation).\n"
    "- Include counterevidence if present.\n"
    "- Return JSON only. No prose.\n"
    "- Output MUST be a single object {conversation_id, episode_id, scores} "
    "where scores is a list with one entry per criterion in the RUBRIC BLOCK.\n"
    "- For the two RISK criteria (premature_convergence_risk, "
    "runaway_divergence_risk) higher scores mean MORE RISK.\n"
    "\nRUBRIC BLOCK:\n" + _RUBRIC_TEXT_BLOCK
)

# Paraphrased prompt for Scorer B — same anchors, different wording.
SYSTEM_B = (
    "You are an evidence-auditor that assigns 0-4 ordinal ratings to a masked "
    "transcript episode from a human-AI creative problem-solving session.\n\n"
    "REQUIREMENTS (do not deviate):\n"
    "1. Never guess the study condition, persona, or treatment group.\n"
    "2. Ignore surface polish, politeness, and verbosity.\n"
    "3. Support every rating with an exact short quote from the episode.\n"
    "4. If the episode contains no evidence for a given criterion, set "
    "score_0_4 to null and usable_for_inference to false for that row.\n"
    "5. Mark bias flags honestly.\n"
    "6. Output must be valid JSON matching the schema; no commentary around it.\n"
    "7. For risk criteria, a higher rating means the risk is MORE pronounced.\n"
    "\nYou will evaluate 12 criteria. The definitions and anchor points are:\n"
    + _RUBRIC_TEXT_BLOCK
)


def build_user_prompt(episode_row) -> str:
    return (
        f"conversation_id: {episode_row['conversation_id']}\n"
        f"episode_id: {episode_row['episode_id']}\n"
        f"episode_type: {episode_row['episode_type']}\n"
        f"turn_range: {episode_row['start_turn']}-{episode_row['end_turn']}\n"
        f"num_turns: {episode_row['num_turns']}\n"
        "\nEPISODE TEXT:\n"
        f"{episode_row['episode_text_masked']}\n"
        "\nReturn one BundledEpisodeScore JSON object with a `scores` list "
        "containing one CriterionScore per criterion from the RUBRIC BLOCK. "
        "Use null score_0_4 for inapplicable criteria."
    )


def _score_episode(episode_row, scorer: str) -> tuple[BundledEpisodeScore | None, dict]:
    """Run one scorer on one episode.

    Scorer A: Qwen3-4B + SYSTEM_A             (primary, all sampled episodes)
    Scorer B: Qwen3-4B + SYSTEM_B             (paraphrased prompt; subset)
    Scorer C: Gemini    + SYSTEM_A            (cross-MODEL agreement; subset)

    A vs B isolates prompt variance; A vs C isolates model variance.
    """
    prompt = build_user_prompt(episode_row)

    if scorer == 'A':
        client, system = LLMClient, SYSTEM_A
    elif scorer == 'B':
        client, system = LLMClient, SYSTEM_B
    elif scorer == 'C':
        # Same prompt as Scorer A; only the model changes. This keeps the
        # A-vs-C comparison interpretable as model variance (not prompt).
        client, system = _get_gemini_client(), SYSTEM_A
    else:
        raise ValueError(f'unknown scorer: {scorer!r}')

    obj, dbg = client.generate_json(
        BundledEpisodeScore, system, prompt,
        temperature=0.15,
        max_new_tokens=2200,   # 12 criteria × ~180 tokens each headroom
    )
    return obj, dbg


def _row_from_criterion(conv_id, episode_id, scorer: str, cs: CriterionScore,
                         episode_row) -> dict:
    return dict(
        conversation_id=int(conv_id),
        episode_id=str(episode_id),
        scorer=scorer,
        criterion=cs.criterion,
        score_0_4=cs.score_0_4,
        confidence_0_1=float(cs.confidence_0_1),
        evidence_quotes='|'.join((cs.evidence_quotes or [])),
        reason_short=str(cs.reason_short or '')[:500],
        counterevidence=str(cs.counterevidence or '')[:500],
        possible_biases='|'.join(cs.possible_biases or ['none']),
        usable_for_inference=bool(cs.usable_for_inference),
        episode_word_count=int(len(str(episode_row['episode_text_masked']).split())),
        episode_type=episode_row['episode_type'],
        condition_original_hidden=episode_row['condition_original'],
        persona_family_original_hidden=episode_row['persona_family_original'],
    )


def score_episodes(episodes_df: pd.DataFrame, out_csv: Path,
                   checkpoint_every: int = 20,
                   scorers: tuple[str, ...] = ('A', 'B'),
                   limit: int | None = None,
                   ) -> pd.DataFrame:
    """Run scorers on each usable episode, appending rows to out_csv."""
    # Only load the local Qwen if at least one scorer needs it.
    if any(s in ('A', 'B') for s in scorers):
        LLMClient.load()
    if 'C' in scorers:
        _get_gemini_client().load()
    OUT = out_csv.parent
    OUT.mkdir(parents=True, exist_ok=True)

    # resume: if out_csv exists, skip episode_ids already scored
    done: set[tuple[str, str]] = set()   # (episode_id, scorer)
    existing_rows: list[dict] = []
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        for _, r in prev.iterrows():
            done.add((str(r['episode_id']), str(r['scorer'])))
        existing_rows = prev.to_dict(orient='records')
        print(f'[scorer] resuming: {len(done)} (episode, scorer) pairs already scored')

    usable = episodes_df[episodes_df['usable_for_scoring'] == True].copy()
    if limit is not None:
        usable = usable.head(limit)
    total = len(usable) * len(scorers)
    rows = list(existing_rows)
    n_done = 0
    n_crashed = 0
    import time
    t0 = time.time()
    for i, ep in usable.iterrows():
        for scorer in scorers:
            if (ep['episode_id'], scorer) in done:
                continue
            t1 = time.time()
            try:
                obj, dbg = _score_episode(ep, scorer)
            except Exception as e:
                print(f'  !! episode {ep["episode_id"]} scorer {scorer} failed: {type(e).__name__}: {e}', flush=True)
                n_crashed += 1
                continue
            dt = time.time() - t1
            if obj is None:
                # record a single placeholder row so we don't re-score
                rows.append(dict(
                    conversation_id=int(ep['conversation_id']),
                    episode_id=ep['episode_id'], scorer=scorer,
                    criterion='ALL', score_0_4=None, confidence_0_1=0.0,
                    evidence_quotes='', reason_short='invalid json',
                    counterevidence='', possible_biases='insufficient_context',
                    usable_for_inference=False,
                    episode_word_count=int(len(str(ep['episode_text_masked']).split())),
                    episode_type=ep['episode_type'],
                    condition_original_hidden=ep['condition_original'],
                    persona_family_original_hidden=ep['persona_family_original'],
                ))
                n_done += 1
                print(f'  [{n_done}/{total}] ep {ep["episode_id"]} S{scorer} INVALID  ({dt:.1f}s)', flush=True)
                continue
            # normalize criterion count: clamp to rubric names, drop unknown
            allowed = set(CRITERION_NAMES)
            for cs in obj.scores:
                if cs.criterion not in allowed:
                    continue
                rows.append(_row_from_criterion(ep['conversation_id'], ep['episode_id'],
                                                 scorer, cs, ep))
            n_done += 1
            if n_done % 10 == 0 or n_done == total:
                rate = n_done / max(1, time.time() - t0)
                eta = (total - n_done) / max(rate, 1e-9)
                print(f'  [{n_done}/{total}] ep {ep["episode_id"]} S{scorer} ok '
                      f'({dt:.1f}s; overall {rate*60:.1f} ep/min; eta {eta/60:.0f} min)', flush=True)
            # periodic flush
            if n_done % checkpoint_every == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'[scorer] done. {n_done} scored, {n_crashed} crashed. saved {out_csv}')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=5, help='limit to first N usable episodes')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--sample', action='store_true',
                    help='use _scoring_sample.csv (scorer A on all sampled, B on flagged subset)')
    ap.add_argument('--scorer-c', action='store_true',
                    help='run Gemini Scorer C on the same 50-episode B subset for cross-model agreement')
    ap.add_argument('--scorer-c-only', action='store_true',
                    help='run only Scorer C (skips A and B; requires existing _scoring_sample.csv)')
    ap.add_argument('--scorer-c-full', action='store_true',
                    help='run Scorer C on ALL 200 stratified-sample episodes (not just the B subset). '
                         'Resume logic skips episodes already scored.')
    args = ap.parse_args()

    ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
    OUT = ROOT / 'regulated_llm_reanalysis'

    if args.sample or args.scorer_c_only or args.scorer_c_full:
        sample = pd.read_csv(OUT / '_scoring_sample.csv')

        if not (args.scorer_c_only or args.scorer_c_full):
            # first pass: Scorer A on all 200
            score_episodes(sample, OUT / '04_episode_rubric_scores_raw.csv',
                            scorers=('A',))
            # second pass: Scorer B on the 50-episode subset
            sub_b = sample[sample.scorer_B == True].copy()
            if len(sub_b):
                score_episodes(sub_b, OUT / '04_episode_rubric_scores_raw.csv',
                                scorers=('B',))

        # third pass (optional): Scorer C (Gemini).
        # Output goes to a SEPARATE file so the published adjudicated scores
        # in 05_episode_rubric_scores_adjudicated.csv are not perturbed.
        # --scorer-c / --scorer-c-only : 50-episode B subset (cross-model agreement check)
        # --scorer-c-full              : all 200 stratified-sample episodes (full reanalysis)
        if args.scorer_c_full:
            sub_c = sample.copy()
        elif args.scorer_c or args.scorer_c_only:
            sub_c = sample[sample.scorer_B == True].copy()
        else:
            sub_c = pd.DataFrame()
        if len(sub_c):
            score_episodes(sub_c,
                           OUT / '04_episode_rubric_scores_raw_scorerC.csv',
                           scorers=('C',))
    else:
        eps = pd.read_csv(OUT / '02_episode_table.csv')
        limit = None if args.all else args.n
        score_episodes(eps, OUT / '04_episode_rubric_scores_raw.csv', limit=limit)
