"""Stage B — CAT-Panel scoring loop.

For each (judge x conversation x dimension x paraphrase) we issue one
Gemini call. The schema is enforced via Gemini's response_schema, so
malformed JSON is impossible.

One dimension per call (Chen et al. 2024 finding: avoids halo bias).

Output: one PanelScore row per call, saved incrementally to parquet so
that long runs are crash-resumable.
"""
from __future__ import annotations
import os, sys, json, time, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.gemini_client import GeminiClient
from os_pipeline.cat_panel.dimensions import DIMENSIONS, DIM_NAMES, DIM_BY_NAME
from os_pipeline.cat_panel.personas import (
    JUDGE_IDS, PARAPHRASE_IDS, build_system_prompt, prompt_hash,
    JUDGE_LABELS,
)
from os_pipeline.regulated.masking import mask
from os_pipeline import config as _global_config
from os_pipeline.cat_panel import config as cat_config


ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT  = ROOT / 'analysis_out' / 'cat_panel'
OUT.mkdir(parents=True, exist_ok=True)

# Use parquet if pyarrow is available, otherwise CSV (mirrors existing
# os_pipeline/regulated/transcript.py pattern).
try:
    import pyarrow  # noqa: F401
    _USE_PARQUET = True
    RAW_PATH = OUT / 'panel_scores_raw.parquet'
except Exception:
    _USE_PARQUET = False
    RAW_PATH = OUT / 'panel_scores_raw.csv'

LOG_PATH  = OUT / 'scoring_log.csv'
REG_PATH  = OUT / 'prompt_registry.json'


def _read_raw() -> pd.DataFrame:
    if _USE_PARQUET:
        return pd.read_parquet(RAW_PATH)
    return pd.read_csv(RAW_PATH)


def _write_raw(df: pd.DataFrame) -> None:
    if _USE_PARQUET:
        df.to_parquet(RAW_PATH, index=False)
    else:
        df.to_csv(RAW_PATH, index=False)

# When the conversation transcript is very long, truncate to this many
# characters (we do not want to exceed the context window of the model).
MAX_TRANSCRIPT_CHARS = 16000


# ----------------------------------------------------------------------
# Pydantic schema for one (judge x conv x dim x paraphrase) score
# ----------------------------------------------------------------------

class PanelScore(BaseModel):
    score_1_7: int | None = Field(
        description='Likert 1-7 score for the user on this dimension, '
                    'or null if evidence is insufficient. 1 = not at all '
                    'evident, 4 = moderately evident, 7 = strongly evident.'
    )
    confidence_0_1: float = Field(
        description='How confident the judge is in this score, 0.0-1.0'
    )
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description='1-3 verbatim quotes from USER turns supporting the '
                    'score. Must match the transcript word-for-word.'
    )
    rationale_short: str = Field(
        description='1-2 sentence rationale for the score.'
    )
    counterevidence: str = Field(
        description='What would have raised or lowered this score if '
                    'present, or empty string if not applicable.'
    )
    possible_biases: list[str] = Field(
        default_factory=list,
        description='Self-reported biases that might distort this score '
                    '(e.g. length-effect, fluency-as-quality, halo).'
    )
    usable_for_inference: bool = Field(
        description='False if score_1_7 is null OR the judge has serious '
                    'doubts. True otherwise.'
    )


# ----------------------------------------------------------------------
# Conversation reconstruction (re-uses masking from existing pipeline)
# ----------------------------------------------------------------------

def _load_conversations() -> pd.DataFrame:
    """Return one row per conversation with full masked transcript.

    Each transcript is a string of turns separated by blank lines,
    formatted as 'USER turn N: ...' / 'ASSISTANT turn N: ...' with
    persona-name masking applied via os_pipeline.regulated.masking.mask.
    """
    logs = pd.read_csv(ROOT / 'Experiment1_logs.csv')
    logs = logs.sort_values(['conversation_id', 'message_id']).reset_index(drop=True)

    FM = {'Divergent': 'Divergent', 'Convergent': 'Convergent',
          'strictly rational': 'Rational',
          'bounded rationality': 'BoundedRational',
          'GPT': 'GPT'}
    logs['persona_family_original'] = logs['Persona_type'].map(FM).fillna('Unknown')
    logs['condition_original'] = np.where(
        logs['Persona_type'] == 'GPT', 'GPT', 'Persona'
    )

    rows = []
    for cid, g in logs.groupby('conversation_id', sort=False):
        g = g.sort_values('message_id').reset_index(drop=True)
        lines = []
        for ti, r in g.iterrows():
            msg = str(r['message']) if pd.notna(r.get('message')) else ''
            masked_msg = mask(msg)
            speaker = 'USER' if str(r['message_src']).lower() == 'user' else 'ASSISTANT'
            lines.append(f'{speaker} turn {ti+1}: {masked_msg}')
        transcript = '\n\n'.join(lines)
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = transcript[:MAX_TRANSCRIPT_CHARS] + '\n\n[...transcript truncated...]'
        rows.append(dict(
            conversation_id=int(cid),
            participant_id=int(g['User_id'].iloc[0]),
            persona_family_original=g['persona_family_original'].iloc[0],
            condition_original=g['condition_original'].iloc[0],
            challenge=str(g.get('Corrected Challenge type', pd.Series(['unknown'])).iloc[0]),
            n_turns=len(g),
            transcript_masked=transcript,
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------

def _build_user_prompt(conv_row: dict, dim_name: str) -> str:
    return (
        f'conversation_id: {conv_row["conversation_id"]}  '
        f'(condition_masked: Assistant_?, challenge: {conv_row["challenge"]})\n'
        f'number of turns: {conv_row["n_turns"]}\n'
        '\n----- MASKED TRANSCRIPT -----\n'
        f'{conv_row["transcript_masked"]}\n'
        '\n----- END TRANSCRIPT -----\n'
        f'\nScore the USER on the dimension `{dim_name}` defined above. '
        'Return one PanelScore JSON object only.'
    )


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def write_prompt_registry(extra: dict | None = None) -> None:
    """Save the model/config/prompt fingerprint for reproducibility."""
    payload = {
        'model_preferences': list(cat_config.GEMINI_MODEL_PREFERENCES),
        'model_id_primary': cat_config.GEMINI_MODEL_ID,
        'temperature': cat_config.GEMINI_TEMPERATURE,
        'max_output_tokens': cat_config.GEMINI_MAX_OUTPUT_TOKENS,
        'max_retries': _global_config.GEMINI_MAX_RETRIES,
        'dim_names': list(DIM_NAMES),
        'judges': list(JUDGE_IDS),
        'paraphrases': list(PARAPHRASE_IDS),
        'prompt_sha256': prompt_hash(),
        'wrote_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    if extra:
        payload.update(extra)
    REG_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _existing_keys() -> set[tuple]:
    if not RAW_PATH.exists():
        return set()
    df = _read_raw()
    keys = set(zip(df['conversation_id'].tolist(),
                   df['judge_id'].tolist(),
                   df['paraphrase'].tolist(),
                   df['dimension'].tolist()))
    return keys


_append_lock = threading.Lock()

# Canonical column order for append-mode writes. Must match the schema
# the rest of the code expects. If a column is missing from `row`, it
# will be written as empty.
_COLUMNS = [
    'conversation_id', 'judge_id', 'paraphrase', 'dimension',
    'score_1_7', 'confidence_0_1', 'evidence_quotes',
    'rationale_short', 'counterevidence', 'possible_biases',
    'usable_for_inference', 'parse_error', 'n_attempts', 'wrote_at',
    'model_used',
]


def _append_row(row: dict) -> None:
    """Fast thread-safe append using append-mode CSV writes.

    No full-file read or rewrite — each call appends exactly one line.
    Duplicate prevention is handled upstream by the work-list (each
    (conv, judge, paraphrase, dim) tuple is submitted to the executor
    exactly once). A post-run dedupe step is available via
    `dedupe_raw_csv()` if needed.

    Performance: the prior read-modify-write approach was O(N) per
    append because pandas had to re-read the entire CSV. At N=700 rows
    on a typical SSD this was ~0.5 s/append; with 8 workers, the global
    lock serialised throughput to ~13 rows/min. Append-only is O(1)
    per row regardless of file size.
    """
    df = pd.DataFrame([{c: row.get(c, '') for c in _COLUMNS}])
    with _append_lock:
        header_needed = not RAW_PATH.exists() or RAW_PATH.stat().st_size == 0
        df.to_csv(RAW_PATH, mode='a', header=header_needed,
                  index=False, encoding='utf-8')


def dedupe_raw_csv() -> int:
    """Post-run dedupe utility. Returns number of rows removed.

    Strategy:
      - For each (conv, judge, paraphrase, dim) key, keep:
          - the success row if any exists (with the highest confidence)
          - otherwise the first failure row
    """
    if not RAW_PATH.exists():
        return 0
    df = _read_raw()
    n_before = len(df)
    key_cols = ['conversation_id', 'judge_id', 'paraphrase', 'dimension']
    # sort so that valid (non-null score_1_7) come first within each key
    df = df.sort_values(['score_1_7', 'confidence_0_1'],
                        ascending=[True, False],
                        na_position='last')
    df = df.drop_duplicates(subset=key_cols, keep='first')
    _write_raw(df)
    return n_before - len(df)


def run_scoring(
    conversation_ids: list[int] | None = None,
    judges: list[str] | None = None,
    paraphrases: list[str] | None = None,
    dimensions: list[str] | None = None,
    max_calls: int | None = None,
    sleep_between_calls_s: float = 0.0,
    retry_failed: bool = False,
    n_workers: int = 1,
    verbose: bool = True,
) -> dict:
    """Run the CAT-Panel scoring loop.

    Crash-resumable: skips any (conv, judge, paraphrase, dim) tuple that
    is already present in panel_scores_raw.parquet.

    Returns a small summary dict (n_calls, n_failed, wall_seconds).
    """
    judges = judges or list(JUDGE_IDS)
    paraphrases = paraphrases or list(PARAPHRASE_IDS)
    dimensions = dimensions or list(DIM_NAMES)

    convs = _load_conversations()
    if conversation_ids is not None:
        convs = convs[convs['conversation_id'].isin(conversation_ids)].copy()
    if convs.empty:
        raise RuntimeError('no conversations selected')

    write_prompt_registry({'run_started_at': time.strftime('%Y-%m-%d %H:%M:%S')})

    done = _existing_keys()
    # If retry_failed, drop the rows that had no score so we re-attempt them
    if retry_failed and RAW_PATH.exists():
        existing = _read_raw()
        # 'score_1_7' is null on failed rows
        failed_mask = existing['score_1_7'].isna()
        if failed_mask.any():
            n_failed = int(failed_mask.sum())
            kept = existing[~failed_mask].copy()
            _write_raw(kept)
            done = _existing_keys()  # re-read after pruning
            if verbose:
                print(f'[cat_panel] retry_failed: pruned {n_failed} previously-failed rows')
    # Build the full work list (skipping already-done cells)
    convs_by_id = {int(r['conversation_id']): r.to_dict() for _, r in convs.iterrows()}
    work: list[tuple] = []
    for cid in convs_by_id:
        for judge in judges:
            for paraphrase in paraphrases:
                for dim_name in dimensions:
                    key = (cid, judge, paraphrase, dim_name)
                    if key in done:
                        continue
                    work.append(key)
    if max_calls is not None:
        work = work[:max_calls]

    if verbose:
        prefs_str = ' > '.join(cat_config.GEMINI_MODEL_PREFERENCES)
        print(f'[cat_panel] models (preference): {prefs_str}  temp={cat_config.GEMINI_TEMPERATURE}')
        print(f'[cat_panel] {len(convs)} convs x {len(judges)} judges '
              f'x {len(paraphrases)} paraphrases x {len(dimensions)} dims')
        print(f'[cat_panel] already done: {len(done)} cells; '
              f'remaining: {len(work)} cells; workers: {n_workers}')
        if sleep_between_calls_s > 0:
            print(f'[cat_panel] per-worker throttle: {sleep_between_calls_s}s')

    n_calls_total = [0]; n_failed_total = [0]
    counter_lock = threading.Lock()
    t0 = time.time()

    def _score_one(key: tuple) -> None:
        cid, judge, paraphrase, dim_name = key
        conv_dict = convs_by_id[cid]
        sys_prompt = build_system_prompt(judge, paraphrase, dim_name)
        usr_prompt = _build_user_prompt(conv_dict, dim_name)
        parsed, debug = GeminiClient.generate_json(
            schema_cls=PanelScore,
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            temperature=cat_config.GEMINI_TEMPERATURE,
            max_new_tokens=cat_config.GEMINI_MAX_OUTPUT_TOKENS,
            model_preferences=cat_config.GEMINI_MODEL_PREFERENCES,
        )
        model_used = debug.get('model_used', cat_config.GEMINI_MODEL_PREFERENCES[0])
        if parsed is None:
            row = dict(
                conversation_id=cid,
                judge_id=judge,
                paraphrase=paraphrase,
                dimension=dim_name,
                score_1_7=None,
                confidence_0_1=np.nan,
                evidence_quotes='',
                rationale_short='',
                counterevidence='',
                possible_biases='',
                usable_for_inference=False,
                parse_error=str(debug.get('parse_error',''))[:300],
                n_attempts=debug.get('n_attempts', 0),
                wrote_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                model_used=model_used,
            )
            with counter_lock:
                n_failed_total[0] += 1
        else:
            row = dict(
                conversation_id=cid,
                judge_id=judge,
                paraphrase=paraphrase,
                dimension=dim_name,
                score_1_7=(int(parsed.score_1_7)
                           if parsed.score_1_7 is not None else None),
                confidence_0_1=float(parsed.confidence_0_1),
                evidence_quotes='||'.join(parsed.evidence_quotes or []),
                rationale_short=parsed.rationale_short,
                counterevidence=parsed.counterevidence,
                possible_biases='||'.join(parsed.possible_biases or []),
                usable_for_inference=bool(parsed.usable_for_inference),
                parse_error='',
                n_attempts=debug.get('n_attempts', 1),
                wrote_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                model_used=model_used,
            )
        _append_row(row)
        with counter_lock:
            n_calls_total[0] += 1
            n_done = n_calls_total[0]
        if sleep_between_calls_s > 0:
            time.sleep(sleep_between_calls_s)
        if verbose and n_done % max(10, n_workers) == 0:
            rate = n_done / max(1e-6, time.time()-t0)
            print(f'[cat_panel] {n_done}/{len(work)} calls '
                  f'({n_failed_total[0]} failed) at {rate:.2f} calls/s '
                  f'(pool: {GeminiClient.pool_summary()})')

    if n_workers <= 1:
        # sequential — preserves backwards-compatible single-thread behaviour
        for key in work:
            _score_one(key)
    else:
        # Pre-warm the pool (so the load print happens once, not once per worker)
        GeminiClient.load()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_score_one, k) for k in work]
            for _ in as_completed(futures):
                pass  # _score_one already appends + counts

    summary = dict(n_calls=n_calls_total[0], n_failed=n_failed_total[0],
                   wall_seconds=time.time()-t0)
    if verbose:
        print(f'[cat_panel] done. {summary["n_calls"]} calls, '
              f'{summary["n_failed"]} failed, '
              f'{summary["wall_seconds"]:.1f} s '
              f'({summary["n_calls"]/max(1e-6, summary["wall_seconds"]):.2f} calls/s)')
    return summary


if __name__ == '__main__':
    # Tiny smoke check (do not run the full pipeline by importing this module).
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--max-calls', type=int, default=8)
    p.add_argument('--conv-ids', type=int, nargs='*', default=None)
    p.add_argument('--judges', type=str, nargs='*', default=['Dr_C'])
    p.add_argument('--paraphrases', type=str, nargs='*', default=['A'])
    p.add_argument('--dimensions', type=str, nargs='*', default=None)
    args = p.parse_args()
    run_scoring(
        conversation_ids=args.conv_ids,
        judges=args.judges,
        paraphrases=args.paraphrases,
        dimensions=args.dimensions,
        max_calls=args.max_calls,
    )
