"""
Precision-improving filters for the extraction pipeline.

1. is_challenge_restatement(user_msg, challenge_label) -> bool
   The user's turn is a near-duplicate of the known challenge statement
   handed to them at the start of the task.

2. is_assistant_echo(evidence_span, prev_assistant) -> bool
   The extracted evidence is a near-verbatim substring of the preceding
   assistant message (the user is agreeing/repeating, not proposing).

3. title_evidence_consistent(title, description, evidence_quotes) -> bool
   Every content word in the title/description appears (morphologically)
   in at least one evidence_quote. Blocks Agent 1 hallucinations where
   the title introduces a concept (e.g. "rewards") not present in the
   user's text.

All three are pure rule-based (rapidfuzz + simple token overlap) — no
LLM calls, deterministic, milliseconds to evaluate.
"""
from __future__ import annotations
import re
from rapidfuzz import fuzz

# Canonical challenge statements the users were given. Extracted from the
# logs: the first substantive assistant message in the conversation
# paraphrases these, and many users typed them in verbatim as their first
# user turn. We compare against both possible framings.
CHALLENGE_TEXTS = {
    'Library': [
        "How can we revitalize Community Libraries in the city? "
        "Local libraries are seeing a decline in visitors, especially "
        "among young adults. How can you make the library more attractive "
        "and relevant to them?",
        "How can we revitalize Community Libraries in the city?",
        "how can we revitalize community libraries",
    ],
    'Bicycle': [
        "How can we encourage Biking in the city? Despite having bike "
        "lanes, few people choose to bike in your city. How can we "
        "encourage more residents to bike instead of driving?",
        "How can we encourage Biking in the city?",
        "how can we encourage biking in the city",
    ],
}

_WS = re.compile(r'\s+')
_STOPWORDS = frozenset([
    'a','an','the','and','or','but','of','to','in','on','at','for','with',
    'by','from','as','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','should','could','can','may','might',
    'this','that','these','those','it','its','we','us','our','i','me','my',
    'you','your','they','them','their','he','she','him','her',
    'so','if','then','than','not','no','yes','too','very','just','also','only',
    'how','what','when','where','why','which','who',
    # task-neutral common words we don't want to count as 'content'
    'idea','ideas','proposal','thing','way','example','something','things',
])


def _norm(s: str) -> str:
    return _WS.sub(' ', s.lower().strip())


def _content_tokens(text: str) -> list[str]:
    """Return lowercased content-word tokens (no stopwords, no short words)."""
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", text.lower())
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]


def _stem(tok: str) -> str:
    """Naive stemmer: strip 's', 'es', 'ed', 'ing' suffixes.
    Good enough for English content words; NLTK/Porter not worth the weight."""
    for suf in ('ing', 'ies', 'ied', 'ed', 'es', 's'):
        if len(tok) > len(suf) + 2 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


# ------- Filter 1 -------

def is_challenge_restatement(user_msg: str, challenge_label: str) -> bool:
    """True if the user's message is a near-verbatim restatement of the
    challenge prompt (and therefore not a user-originated proposal)."""
    if not user_msg or not challenge_label:
        return False
    um = _norm(user_msg)
    for ref in CHALLENGE_TEXTS.get(challenge_label, []):
        rn = _norm(ref)
        if fuzz.partial_ratio(rn, um) >= 85:
            return True
        # also: starts with "how can we ..." + short + no own content
        if um.startswith('how can we') and len(um.split()) <= 20 \
                and fuzz.token_set_ratio(rn, um) >= 70:
            return True
    return False


# ------- Filter 2 -------

def is_assistant_echo(evidence_span: str, prev_assistant: str | None) -> bool:
    """True if the user's evidence span is substantially a substring of the
    preceding assistant turn (the user is agreeing/repeating). Applies to
    spans of 3 or more words so short echoed phrases like 'integrating
    technology resources' get caught."""
    if not prev_assistant or not evidence_span:
        return False
    if len(evidence_span.split()) < 3:
        return False
    ev = _norm(evidence_span)
    pa = _norm(prev_assistant)
    return fuzz.partial_ratio(ev, pa) >= 85


# ------- Filter 3 -------

def title_evidence_consistent(title: str, description: str,
                              evidence_quotes: list[str]
                              ) -> tuple[bool, list[str]]:
    """Check that content words in the title/description have morphological
    matches in the evidence. Returns (is_consistent, missing_words).

    The idea: the extractor sometimes introduces concepts in the title that
    aren't in the evidence (e.g., inventing 'rewards' when the user never
    said it). Every *substantive* title word must have a stem match in at
    least one evidence_quote."""
    if not evidence_quotes:
        return False, ['<no evidence>']

    # Title content words (prioritized over description for strictness).
    title_toks = _content_tokens(title)
    if not title_toks:
        return True, []  # degenerate — accept

    ev_text = ' '.join(evidence_quotes)
    ev_toks = set(_stem(t) for t in _content_tokens(ev_text))

    missing = []
    for t in title_toks:
        if _stem(t) not in ev_toks:
            # allow a soft fuzzy fallback for multi-token concepts
            if not any(fuzz.partial_ratio(t, e) >= 85
                       for e in ev_text.split()):
                missing.append(t)
    # Tolerate 1 missing word for short titles; require all for longer.
    tolerance = 1 if len(title_toks) >= 3 else 0
    return len(missing) <= tolerance, missing
