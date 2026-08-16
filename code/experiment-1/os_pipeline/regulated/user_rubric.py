"""Option B — User-behaviour rubric for Gemini Scorer C.

Six 0-4 ordinal criteria targeting USER behaviour specifically. The scorer
sees the FULL masked episode (assistant turns included, for context) but is
instructed to score only the user's contribution. Each rating must include
at least one verbatim quote from a USER turn.

This is a separate rubric from the 12 dialogic criteria scored elsewhere;
it is informational and does not modify the published adjudicated rubric
or any downstream stats.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class UserCriterion:
    name: str
    question: str
    definition: str
    anchors: dict[int, str]
    reverse: bool = False


USER_CRITERIA: tuple[UserCriterion, ...] = (
    UserCriterion(
        name='user_initiative',
        question='How proactively does the USER introduce new directions, '
                 'problems, or ideas without being prompted?',
        definition=('User-side proactivity. Score what the USER originates '
                    '(not what they accept from the assistant). Echoes of the '
                    'assistant do NOT count.'),
        anchors={
            0: 'User is fully passive; one-word answers, never introduces new content.',
            1: 'User responds substantively but does not introduce new directions.',
            2: 'User occasionally introduces a new direction or substantive follow-up.',
            3: 'User regularly initiates new directions or proposes new content.',
            4: 'User drives the agenda; introduces multiple distinct directions and connects them.',
        },
    ),
    UserCriterion(
        name='user_question_richness',
        question='When the USER asks questions, how substantive and probing '
                 'are they versus surface clarifications?',
        definition=('Quality of user questioning. Mere "what?" or '
                    '"can you repeat?" is surface; questions that test ideas, '
                    'open new angles, or probe assumptions are rich.'),
        anchors={
            0: 'User asks no questions.',
            1: 'User asks only surface clarifications.',
            2: 'User asks specific clarifying questions about details.',
            3: 'User asks probing questions that test ideas or open new angles.',
            4: 'User asks deeply probing or reframing questions that reshape the conversation.',
        },
    ),
    UserCriterion(
        name='user_proposal_specificity',
        question='When the USER proposes ideas, how concrete and specified '
                 'are they?',
        definition=('Concreteness of user-originated proposals. Vague slogans '
                    'low; mechanisms / examples / actors / venues high.'),
        anchors={
            0: 'User proposes nothing.',
            1: 'Proposals are vague slogans or restatements.',
            2: 'Proposals have one concrete element.',
            3: 'Proposals are concrete with at least one specific mechanism or example.',
            4: 'Proposals are highly concrete and multi-element (mechanisms, examples, actors, or venues).',
        },
    ),
    UserCriterion(
        name='user_acceptance_yes_and',
        question='Does the USER build on the assistant\'s contributions '
                 '(yes-and) versus ignore or block them?',
        definition=('Yes-and uptake on the user side. Acknowledgement without '
                    'building is low; explicit extension/integration is high.'),
        anchors={
            0: 'User ignores the assistant entirely.',
            1: 'User acknowledges but does not build (e.g. "ok").',
            2: 'User accepts and adds tangential content.',
            3: 'User accepts and explicitly extends the assistant\'s contribution.',
            4: 'User strongly yes-ands; integrates and builds on assistant content into a unified thread.',
        },
    ),
    UserCriterion(
        name='user_reframing',
        question='Does the USER reframe the problem, take a new angle, or '
                 'challenge assumptions?',
        definition=('User-initiated reframing. Staying inside the assistant\'s '
                    'framing low; substantively challenging or reformulating high.'),
        anchors={
            0: 'User stays strictly within the assistant\'s framing.',
            1: 'User makes minor adjustments to the framing.',
            2: 'User offers an alternative angle once.',
            3: 'User offers multiple reframings or significant alternative angles.',
            4: 'User actively challenges or substantially reframes the problem.',
        },
    ),
    UserCriterion(
        name='user_engagement_depth',
        question='How substantive versus surface is the USER\'s engagement '
                 'across the episode?',
        definition=('Depth of user reasoning. Single-word and yes/no '
                    'reactions low; multi-step reasoning and integration high.'),
        anchors={
            0: 'User is purely surface (yes/no, single words).',
            1: 'User responds but stays at surface level.',
            2: 'User offers some substance with elaboration.',
            3: 'User shows substantive engagement; builds reasoning chains.',
            4: 'User shows deep engagement; integrates multiple considerations and reasons through them.',
        },
    ),
)

USER_CRITERION_NAMES = tuple(c.name for c in USER_CRITERIA)


# -------- Pydantic schemas --------

class UserCriterionScore(BaseModel):
    criterion: str = Field(description='one of the 6 user-rubric criterion names')
    score_0_4: Optional[int] = Field(
        default=None,
        description='Integer 0-4. Use null when no user evidence is present.'
    )
    confidence_0_1: float = Field(
        default=0.0,
        description='Score confidence 0-1.'
    )
    evidence_quotes: List[str] = Field(
        default_factory=list,
        description='Verbatim quotes copied from USER turns only (not assistant).'
    )
    reason_short: str = Field(default='')
    counterevidence: str = Field(default='')
    usable_for_inference: bool = Field(default=True)


class BundledUserEpisodeScore(BaseModel):
    conversation_id: str
    episode_id: str
    scores: List[UserCriterionScore]
