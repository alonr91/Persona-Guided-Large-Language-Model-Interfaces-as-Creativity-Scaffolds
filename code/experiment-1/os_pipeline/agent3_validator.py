"""Agent 3 — grounding validator.

For each canonical idea, check whether every evidence_quote is a
verbatim substring of some message in the source conversation.
Exact substring (case/whitespace normalized) first; rapidfuzz
partial_ratio >= 90 as a fuzzy fallback.

Pure rule-based; no LLM call. Ideas that fail both checks are dropped
from the canonical list and logged.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, replace
from rapidfuzz import fuzz

from .config import FUZZY_PARTIAL_RATIO_THRESHOLD
from .agent2_consolidator import CanonicalRow
from . import filters as _filters


_WS = re.compile(r'\s+')
def _norm(s: str) -> str:
    return _WS.sub(' ', s.lower().strip())


@dataclass
class ValidationRow:
    conversation_id: int
    canonical_id: int
    title: str
    status: str  # 'grounded' | 'grounded_fuzzy' | 'ungrounded' | 'title_hallucination'
    best_fuzzy_score: float
    n_quotes: int
    consistency_missing_words: str = ''  # pipe-separated, if title_hallucination


def validate(canon: list[CanonicalRow], user_messages: list[str]
             ) -> tuple[list[CanonicalRow], list[ValidationRow]]:
    """Returns (kept canonical ideas, one validation row per original canonical)."""
    norm_msgs = [_norm(m) for m in user_messages]
    kept: list[CanonicalRow] = []
    report: list[ValidationRow] = []
    for c in canon:
        if not c.evidence_quotes:
            report.append(ValidationRow(c.conversation_id, c.canonical_id, c.title,
                                        'ungrounded', 0.0, 0))
            continue
        exact_all = True
        best_fuzzy = 0.0
        for q in c.evidence_quotes:
            qn = _norm(q)
            exact = any(qn in m for m in norm_msgs)
            if not exact:
                exact_all = False
                # best partial_ratio across all user msgs
                fz = max((fuzz.partial_ratio(qn, m) for m in norm_msgs), default=0.0)
                best_fuzzy = max(best_fuzzy, fz)
        if exact_all:
            status = 'grounded'
            best_fuzzy = 100.0
        else:
            # compute min best-fuzzy across quotes that failed exact
            per_quote_best: list[float] = []
            for q in c.evidence_quotes:
                qn = _norm(q)
                if any(qn in m for m in norm_msgs):
                    per_quote_best.append(100.0)
                else:
                    per_quote_best.append(
                        max((fuzz.partial_ratio(qn, m) for m in norm_msgs), default=0.0)
                    )
            worst = min(per_quote_best) if per_quote_best else 0.0
            if worst >= FUZZY_PARTIAL_RATIO_THRESHOLD:
                status = 'grounded_fuzzy'
            else:
                status = 'ungrounded'
            best_fuzzy = worst

        # ---- Filter 3: title-evidence consistency ----
        # Even if all evidence is grounded in user text, the title may
        # introduce a concept (e.g. 'rewards') that isn't in the evidence.
        missing_words: list[str] = []
        if status in ('grounded', 'grounded_fuzzy'):
            ok, missing_words = _filters.title_evidence_consistent(
                c.title, c.description, c.evidence_quotes)
            if not ok:
                status = 'title_hallucination'

        if status in ('grounded', 'grounded_fuzzy'):
            kept.append(c)
        report.append(ValidationRow(
            c.conversation_id, c.canonical_id, c.title,
            status, float(best_fuzzy), len(c.evidence_quotes),
            '|'.join(missing_words),
        ))
    return kept, report
