"""Regulated reanalysis orchestrator. Runs every downstream stage that
doesn't require additional LLM calls (adjudication, turn-transition,
trajectory, bias audit, statistics, figures, claim cards, memo)."""
from __future__ import annotations
import sys

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

from os_pipeline.regulated import adjudicator, turn_transition, trajectory
from os_pipeline.regulated import bias_audit, statistics, figures
from os_pipeline.regulated import claim_cards, memo


def main():
    print('=== Stage 5: adjudication ===')
    adjudicator.main()
    print('\n=== Stage 6: turn transitions ===')
    turn_transition.main()
    print('\n=== Stage 7: trajectory features ===')
    trajectory.main()
    print('\n=== Stage 8: bias + validation audits ===')
    bias_audit.main()
    print('\n=== Stage 9: statistical models ===')
    statistics.main()
    print('\n=== Stage 10: figures ===')
    figures.main()
    print('\n=== Stage 11: claim cards ===')
    claim_cards.main()
    print('\n=== Stage 12: memo + methods appendix ===')
    memo.generate_memo()
    memo.generate_methods_appendix()
    print('\nAll post-scorer stages complete.')


if __name__ == '__main__':
    main()
