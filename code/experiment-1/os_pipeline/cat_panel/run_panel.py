"""One-command orchestrator for the CAT-Panel layer.

Stage A — scoring (parallel: GeminiClient handles I/O concurrency internally)
Stage B — adjudication (paraphrase fusion + cross-judge consensus)
Stage C — audits (mask leak, length bias, halo, construct validity)

Usage:
  Smoke test (5 convs x Dr_C only x both paraphrases x 8 dims = 80 calls):
      python -m os_pipeline.cat_panel.run_panel --convs 5 --judges Dr_C

  Full run (all 194 convs x 4 judges x 2 paraphrases x 8 dims = 12,416 calls):
      python -m os_pipeline.cat_panel.run_panel --all

  Adjudication-only (after scoring already complete):
      python -m os_pipeline.cat_panel.run_panel --skip-scoring

The scorer is crash-resumable: existing rows in panel_scores_raw.parquet
are detected and skipped on a second invocation.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.cat_panel.scorer import run_scoring, _load_conversations
from os_pipeline.cat_panel.adjudicator import adjudicate
from os_pipeline.cat_panel.personas import JUDGE_IDS, PARAPHRASE_IDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--all', action='store_true',
                   help='full run: all 194 convs x 4 judges x 2 paraphrases x 8 dims')
    p.add_argument('--poc', action='store_true',
                   help='proof-of-concept: 40 stratified convs (5 users x 4 families x 2 rounds)')
    p.add_argument('--convs', type=int, default=None,
                   help='cap to first N conversations (smoke test)')
    p.add_argument('--conv-ids', type=int, nargs='*', default=None,
                   help='explicit list of conversation_ids')
    p.add_argument('--judges', type=str, nargs='*', default=None,
                   help='subset of judges (default: all 4)')
    p.add_argument('--paraphrases', type=str, nargs='*', default=None,
                   help='subset of paraphrases (default: both A and B)')
    p.add_argument('--dimensions', type=str, nargs='*', default=None,
                   help='subset of dimensions (default: all 8)')
    p.add_argument('--max-calls', type=int, default=None,
                   help='hard cap on total API calls (safety)')
    p.add_argument('--sleep', type=float, default=0.0,
                   help='seconds to sleep between Gemini calls (per-worker; for free-tier rate limits)')
    p.add_argument('--workers', type=int, default=1,
                   help='concurrent worker threads (use 8 to saturate 8-key pool)')
    p.add_argument('--retry-failed', action='store_true',
                   help='drop previously-failed rows and re-attempt them')
    p.add_argument('--skip-scoring', action='store_true',
                   help='only run adjudication on already-stored raw scores')
    p.add_argument('--skip-adjudication', action='store_true',
                   help='only run scoring, defer adjudication')
    args = p.parse_args()

    # ---- selection ----
    conv_ids = None
    if args.conv_ids:
        conv_ids = args.conv_ids
    elif args.poc:
        from os_pipeline.cat_panel.sample_poc import sample as _poc_sample
        conv_ids = _poc_sample(verbose=False)
        print(f'[run_panel] PoC sample: {len(conv_ids)} conversations '
              f'(5 users x 4 families x 2 rounds)')
    elif args.convs:
        convs_df = _load_conversations()
        conv_ids = sorted(convs_df['conversation_id'].tolist())[:args.convs]

    # ---- scoring stage ----
    if not args.skip_scoring:
        print('=' * 60)
        print('STAGE A — CAT-Panel scoring')
        print('=' * 60)
        summary = run_scoring(
            conversation_ids=conv_ids,
            judges=args.judges,
            paraphrases=args.paraphrases,
            dimensions=args.dimensions,
            max_calls=args.max_calls,
            sleep_between_calls_s=args.sleep,
            retry_failed=args.retry_failed,
            n_workers=args.workers,
        )
        print(f'\nstage A summary: {summary}')

    # ---- adjudication stage ----
    if not args.skip_adjudication:
        print()
        print('=' * 60)
        print('STAGE B — Adjudication')
        print('=' * 60)
        summary = adjudicate()
        print(f'\nstage B summary: {summary}')


if __name__ == '__main__':
    main()
