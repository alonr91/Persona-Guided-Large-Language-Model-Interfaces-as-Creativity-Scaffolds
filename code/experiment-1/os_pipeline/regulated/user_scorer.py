"""User-rubric scorer (Gemini Scorer C-style, applied to the 6 user-side criteria).

Same architecture as os_pipeline.regulated.scorer.score_episodes — bundled
JSON output, schema-constrained via response_schema, resume-on-restart.
Uses GeminiClient because (a) cross-model rubric work is already on Gemini
and (b) keeping local Qwen for the dialogic rubric and Gemini for the
user rubric is the cleanest separation.

Output:
  regulated_llm_reanalysis/18_user_rubric_raw_scorerC.csv
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.gemini_client import GeminiClient
from os_pipeline.regulated.user_rubric import (
    USER_CRITERIA, USER_CRITERION_NAMES,
    BundledUserEpisodeScore, UserCriterionScore,
)


# -------- Prompt --------

_USER_RUBRIC_BLOCK = '\n\n'.join(
    f"Criterion: {c.name}\n"
    f"Question: {c.question}\n"
    f"Definition: {c.definition}\n"
    "Anchors:\n" +
    '\n'.join(f"  {s} = {a}" for s, a in c.anchors.items())
    for c in USER_CRITERIA
)

SYSTEM_USER = (
    "You are a transcript analysis instrument scoring USER behaviour in a "
    "human-AI creative collaboration. The transcript contains both [USER] "
    "and [ASSISTANT] turns; you read the full context but rate ONLY the "
    "USER's behaviour.\n\n"
    "RULES:\n"
    "- Score the user, not the assistant. The assistant's output is context, "
    "not a target of scoring.\n"
    "- Never reward length, fluency, politeness, or confidence in itself.\n"
    "- Use only evidence present in the episode.\n"
    "- Every non-null score must include at least one exact verbatim quote "
    "copied from a USER turn (not an assistant turn). If you cannot quote "
    "from a user turn, return score_0_4 = null and "
    "usable_for_inference = false.\n"
    "- Echoes of the assistant in the user's turns do NOT count as user "
    "initiative or proposals; they may count for user_acceptance_yes_and.\n"
    "- Score on the 0-4 anchors exactly as defined.\n"
    "- Return JSON only. No prose.\n"
    "- Output MUST be a single object {conversation_id, episode_id, scores} "
    "where scores is a list with one entry per criterion in the RUBRIC BLOCK.\n"
    "\nRUBRIC BLOCK:\n" + _USER_RUBRIC_BLOCK
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
        "\nReturn one BundledUserEpisodeScore JSON object with a `scores` list "
        "containing one UserCriterionScore per criterion from the RUBRIC BLOCK. "
        "Use null score_0_4 only when no user evidence is present for that criterion."
    )


def _row(conv_id, episode_id, cs: UserCriterionScore, episode_row) -> dict:
    return dict(
        conversation_id=int(conv_id),
        episode_id=str(episode_id),
        scorer='C_user',
        criterion=cs.criterion,
        score_0_4=cs.score_0_4,
        confidence_0_1=float(cs.confidence_0_1),
        evidence_quotes='|'.join((cs.evidence_quotes or [])),
        reason_short=str(cs.reason_short or '')[:500],
        counterevidence=str(cs.counterevidence or '')[:500],
        usable_for_inference=bool(cs.usable_for_inference),
        episode_word_count=int(len(str(episode_row['episode_text_masked']).split())),
        episode_type=episode_row['episode_type'],
        condition_original_hidden=episode_row['condition_original'],
        persona_family_original_hidden=episode_row['persona_family_original'],
    )


def score_episodes(episodes_df: pd.DataFrame, out_csv: Path,
                   checkpoint_every: int = 20, limit: int | None = None,
                   ) -> pd.DataFrame:
    GeminiClient.load()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    rows: list[dict] = []
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(str(eid) for eid in prev['episode_id'].unique())
        rows = prev.to_dict(orient='records')
        print(f'[user_scorer] resuming: {len(done)} episodes already scored')

    usable = episodes_df[episodes_df['usable_for_scoring'] == True].copy()
    if limit is not None:
        usable = usable.head(limit)

    total = len(usable)
    n_done = 0; n_crash = 0
    t0 = time.time()
    for _, ep in usable.iterrows():
        if str(ep['episode_id']) in done:
            continue
        t1 = time.time()
        try:
            obj, dbg = GeminiClient.generate_json(
                BundledUserEpisodeScore, SYSTEM_USER, build_user_prompt(ep),
                temperature=0.15, max_new_tokens=1600,
            )
        except Exception as e:
            print(f'  !! ep {ep["episode_id"]} failed: {type(e).__name__}: {e}', flush=True)
            n_crash += 1
            continue
        dt = time.time() - t1
        if obj is None:
            rows.append(_row(ep['conversation_id'], ep['episode_id'],
                             UserCriterionScore(criterion='ALL', score_0_4=None,
                                                 confidence_0_1=0.0,
                                                 evidence_quotes=[],
                                                 reason_short='invalid json',
                                                 counterevidence='',
                                                 usable_for_inference=False), ep))
            n_done += 1
            print(f'  [{n_done}/{total}] ep {ep["episode_id"]} INVALID ({dt:.1f}s)', flush=True)
            continue
        allowed = set(USER_CRITERION_NAMES)
        for cs in obj.scores:
            if cs.criterion not in allowed:
                continue
            rows.append(_row(ep['conversation_id'], ep['episode_id'], cs, ep))
        n_done += 1
        if n_done % 10 == 0 or n_done == total:
            rate = n_done / max(1, time.time() - t0)
            eta = (total - n_done) / max(rate, 1e-9)
            print(f'  [{n_done}/{total}] ep {ep["episode_id"]} ok '
                  f'({dt:.1f}s; {rate*60:.1f} ep/min; eta {eta/60:.0f} min)', flush=True)
        if n_done % checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'[user_scorer] done. {n_done} scored, {n_crash} crashed. saved {out_csv}')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--full', action='store_true',
                    help='use the existing 200-episode stratified sample')
    args = ap.parse_args()
    ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
    OUT = ROOT / 'regulated_llm_reanalysis'
    if args.full:
        sample = pd.read_csv(OUT / '_scoring_sample.csv')
        score_episodes(sample, OUT / '18_user_rubric_raw_scorerC.csv')
    else:
        eps = pd.read_csv(OUT / '02_episode_table.csv')
        score_episodes(eps, OUT / '18_user_rubric_raw_scorerC.csv', limit=args.n)
