"""Agent 1 — per-user-message idea candidate extractor.

Given one user message + the immediately preceding assistant message as
context, extracts 0-3 concrete proposals that THE USER made. Ignores
assistant-originated ideas. Empty list is an acceptable answer.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .config import (Extraction, AGENT1_TEMPERATURE, AGENT1_MAX_NEW_TOKENS,
                     AGENT1_MAX_CONTEXT_CHARS, AGENT1_MIN_EVIDENCE_WORDS)
from .llm_client import LLMClient
from . import filters as _filters

SYSTEM_PROMPT = (
    "You extract concrete ideas THE USER ORIGINATED in the current user message. "
    "Output JSON matching the schema; no prose.\n\n"
    "ROLE BOUNDARY (most important):\n"
    "The prompt contains two blocks. The [ASSISTANT] block is context ONLY. "
    "NEVER extract ideas from it. NEVER use its text as an evidence_span. "
    "Only the content of the [USER] block is an acceptable source.\n\n"
    "EVIDENCE RULE:\n"
    "Every `evidence_span` MUST be a verbatim substring copied letter-for-letter "
    "from the [USER] block (same words, same spelling, same typos, same "
    "punctuation). Preserve typos AS-IS. Short quotes (3-20 words) are better "
    "than long ones. If you cannot find a clean verbatim span for a candidate "
    "idea, drop that idea.\n\n"
    "TITLE-EVIDENCE CONSISTENCY:\n"
    "Every substantive word in the title must also appear (with allowances for "
    "plural / tense) in the evidence_span. Do NOT introduce concepts in the "
    "title that aren't in the user's actual words. Example: if the user says "
    "'bike lanes', do not title the idea 'Encourage Biking with Rewards' "
    "because the user didn't say 'rewards'.\n\n"
    "DECOMPOSITION RULE:\n"
    "Multiple distinct user proposals -> multiple ideas, one per proposal. "
    "Example: 'add a cafe AND host poetry nights' -> TWO ideas.\n\n"
    "WHAT IS NOT A PROPOSAL (return {\"ideas\": []} in ALL these cases):\n"
    "(a) CHALLENGE RESTATEMENT. If the user is merely restating the task "
    "    question they were given (e.g. 'How can we revitalize Community "
    "    Libraries in the city? Local libraries are seeing a decline in "
    "    visitors, especially among young adults.'), they did not propose "
    "    anything. Return [].\n"
    "(b) ECHO / AGREEMENT. If the user is agreeing with or repeating the "
    "    assistant's phrasing (e.g. assistant said 'enhance the library's "
    "    digital presence'; user then says 'yes, enhance the library's "
    "    digital presence would help'), that is not a user proposal. "
    "    Return [].\n"
    "(c) META-QUESTION / CLARIFICATION. If the user is asking a clarifying "
    "    question about constraints (e.g. 'how can we do it on a low "
    "    budget?'), they did not propose anything. Return [].\n"
    "(d) CRITIQUE / OBJECTION. If the user is pointing out a problem with "
    "    something without offering a replacement (e.g. 'but if people "
    "    won\u2019t volunteer it won\u2019t work'), return [].\n"
    "(e) PURE REACTION. 'Yes', 'Thanks', 'Ok', 'Hi', 'I see', 'do you have "
    "    more ideas?' -> return [].\n\n"
    "A QUESTION-PHRASED IDEA IS STILL AN IDEA:\n"
    "'do you think adding a cafe would help?' -> ONE idea ('adding a cafe'). "
    "The user introduced a specific concrete option.\n\n"
    "OTHER RULES:\n"
    "- Title: short noun phrase, <=10 words, using the user's own vocabulary.\n"
    "- Description: 1 sentence summarizing this specific idea.\n"
    "- confidence: decimal between 0.0 and 1.0.\n"
    "- At most 3 ideas per message."
)


def _truncate(txt: str, limit: int) -> str:
    if len(txt) <= limit:
        return txt
    return txt[: limit - 1] + '…'


def build_user_prompt(prev_assistant: Optional[str], user_msg: str,
                      challenge: str, persona_label: str) -> str:
    cap = AGENT1_MAX_CONTEXT_CHARS // 2
    prev_block = (
        f'[ASSISTANT (context, do not extract from this)]:\n{_truncate(prev_assistant, cap)}\n\n'
        if prev_assistant else ''
    )
    return (
        f'Challenge: {challenge}\n'
        f'Persona the user is talking to: {persona_label}\n\n'
        f'{prev_block}'
        f'[USER (extract ideas from this message only)]:\n{_truncate(user_msg, cap)}\n\n'
        f'Produce JSON matching the Extraction schema.'
    )


@dataclass
class CandidateRow:
    conversation_id: int
    message_id: int
    user_turn_idx: int
    title: str
    description: str
    evidence_span: str
    confidence: float
    raw_user_message: str
    extraction_valid: bool


def extract_for_message(conversation_id: int, message_id: int, user_turn_idx: int,
                         prev_assistant: Optional[str], user_msg: str,
                         challenge: str, persona_label: str) -> tuple[list[CandidateRow], dict]:
    # ---- Filter 1 (pre-LLM): challenge restatement ----
    # If the user's whole turn is just a restatement of the task question,
    # skip the LLM call entirely (cheaper + forces empty list).
    if _filters.is_challenge_restatement(user_msg, challenge):
        return [], {'conversation_id': conversation_id, 'message_id': message_id,
                    'user_turn_idx': user_turn_idx, 'valid_json': True,
                    'skipped_reason': 'challenge_restatement', 'n_ideas': 0}

    prompt = build_user_prompt(prev_assistant, user_msg, challenge, persona_label)
    obj, dbg = LLMClient.generate_json(
        Extraction,
        SYSTEM_PROMPT,
        prompt,
        temperature=AGENT1_TEMPERATURE,
        max_new_tokens=AGENT1_MAX_NEW_TOKENS,
    )
    rows: list[CandidateRow] = []
    if obj is None:
        return rows, {**dbg, 'conversation_id': conversation_id, 'message_id': message_id,
                      'user_turn_idx': user_turn_idx}
    # track filter drops for diagnostics
    dropped = {'short_evidence': 0, 'assistant_echo': 0}
    for ic in obj.ideas[:3]:  # hard cap at 3 per message
        ev = ic.evidence_span.strip()
        # Drop trivial evidence spans (less than N words)
        if len(ev.split()) < AGENT1_MIN_EVIDENCE_WORDS:
            dropped['short_evidence'] += 1
            continue
        # ---- Filter 2 (post-LLM, per candidate): assistant echo ----
        if _filters.is_assistant_echo(ev, prev_assistant):
            dropped['assistant_echo'] += 1
            continue
        # Normalize confidence: models sometimes emit 0-100 instead of 0-1.
        conf = float(ic.confidence)
        if conf > 1.0:
            conf = min(conf / 100.0, 1.0)
        elif conf < 0.0:
            conf = 0.0
        rows.append(CandidateRow(
            conversation_id=conversation_id,
            message_id=message_id,
            user_turn_idx=user_turn_idx,
            title=ic.title.strip(),
            description=ic.description.strip(),
            evidence_span=ev,
            confidence=conf,
            raw_user_message=user_msg,
            extraction_valid=True,
        ))
    return rows, {**dbg, 'conversation_id': conversation_id, 'message_id': message_id,
                  'user_turn_idx': user_turn_idx, 'n_ideas': len(rows),
                  'dropped': dropped}
