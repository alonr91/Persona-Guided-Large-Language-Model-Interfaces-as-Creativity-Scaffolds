"""End-to-end downstream pipeline runner.

Runs AFTER scoring is complete. Executes in order:
  1. adjudicator         — paraphrase A/B fusion + cross-judge consensus + ICC
  2. audit_mask_leak     — V5 mask-leak audit (issues ~50 extra Gemini calls)
  3. run_audits          — V4 length-bias, V6 halo, V7 construct validity
  4. analyses            — F1–F6 figures and tables
  5. generate_poc_report — single consolidated markdown report

Each stage is skipped gracefully if its prerequisite outputs are missing.

Usage:
    python -m os_pipeline.cat_panel.complete_pipeline
    python -m os_pipeline.cat_panel.complete_pipeline --skip-mask-leak
"""
from __future__ import annotations
import sys, argparse, traceback
from pathlib import Path

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass


def _stage(label: str, fn):
    print('\n' + '=' * 70)
    print(f'STAGE: {label}')
    print('=' * 70)
    try:
        fn()
        print(f'[stage {label}] OK')
    except Exception as e:
        print(f'[stage {label}] FAILED: {type(e).__name__}: {e}')
        traceback.print_exc()
        print('continuing to next stage...')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-mask-leak', action='store_true',
                   help='skip the mask-leak audit (saves ~50 Gemini calls)')
    args = p.parse_args()

    from os_pipeline.cat_panel.adjudicator import adjudicate
    _stage('1. Adjudication (paraphrase fusion + cross-judge consensus)',
           lambda: adjudicate())

    if not args.skip_mask_leak:
        from os_pipeline.cat_panel import audit_mask_leak
        _stage('2. V5 — Mask-leak audit (50 Gemini calls)',
               audit_mask_leak.main)
    else:
        print('\n[skip] V5 mask-leak audit (--skip-mask-leak)')

    from os_pipeline.cat_panel import audit_length_bias, audit_halo, construct_validity
    _stage('3a. V4 — Length-bias audit', audit_length_bias.main)
    _stage('3b. V6 — Halo audit',        audit_halo.main)
    _stage('3c. V7 — Construct validity', construct_validity.main)

    from os_pipeline.cat_panel.analyses import main as analyses_main
    _stage('4. Analyses F1–F6', analyses_main)

    from os_pipeline.cat_panel.generate_poc_report import main as report_main
    _stage('5. PoC report', report_main)

    print('\n' + '=' * 70)
    print('DONE — see CAT_Panel_PoC_Report.md')
    print('=' * 70)


if __name__ == '__main__':
    main()
