"""Rubric definitions and scoring schemas for the regulated LLM reanalysis.

Implements Part C of llm_agent_regulated_creativity_analysis_instructions.md:
12 ordinal (0-4) criteria, each with a question, definition, and five
score anchors. Two of the 12 criteria (premature_convergence_risk,
runaway_divergence_risk) are reverse-scored — higher = worse.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from pydantic import BaseModel, Field


# -------- 12 rubric criteria --------

@dataclass(frozen=True)
class Criterion:
    name: str
    question: str
    definition: str
    anchors: dict[int, str]
    reverse: bool = False  # True when higher score = worse (risk criteria)
    applicable_to: tuple[str, ...] = ('assistant', 'user', 'dyad')


CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        name='exploration_opening',
        question='Does the episode expand the possibility space in a meaningful way?',
        definition=('Rate how much the episode opens the conversation to new options, '
                    'frames, stakeholders, or analogies beyond the current focus.'),
        anchors={
            0: 'No expansion; repeats or narrows immediately',
            1: 'Adds one obvious variant',
            2: 'Adds several relevant but conventional options',
            3: 'Opens a meaningfully different direction',
            4: 'Opens a new frame, analogy, stakeholder, or design space',
        },
    ),
    Criterion(
        name='reframing_quality',
        question='Does the episode change how the problem is understood?',
        definition=('Rate whether the episode genuinely shifts assumptions, target users, '
                    'constraints, or success criteria, vs merely rewording the same idea.'),
        anchors={
            0: 'No reframe',
            1: 'Surface rewording only',
            2: 'Shifts emphasis but not assumptions',
            3: 'Changes assumptions, target user, constraint, or success criterion',
            4: 'Produces a generative new problem frame that guides later turns',
        },
    ),
    Criterion(
        name='evaluative_discipline',
        question='Does the episode help narrow responsibly using criteria, tradeoffs, or constraints?',
        definition=('Rate whether the episode introduces criteria, weighs options, or makes '
                    'tradeoffs explicit, vs offering vague preferences or ungrounded judgments.'),
        anchors={
            0: 'No evaluation or ungrounded judgment',
            1: 'Vague preference or unsupported ranking',
            2: 'Some criteria but weak comparison',
            3: 'Clear tradeoffs and criteria',
            4: 'Strong feasibility or usefulness critique while preserving alternatives',
        },
    ),
    Criterion(
        name='agency_preservation',
        question='Does the assistant preserve the user\u2019s authorship and control?',
        definition=('Rate how much the user\'s goals, ideas, and judgments shape the '
                    'conversation, vs the assistant taking over the solution.'),
        anchors={
            0: 'Assistant takes over; user becomes accepter',
            1: 'Assistant mostly directs the solution',
            2: 'Mixed control',
            3: 'User goals and ideas shape the response',
            4: 'Assistant explicitly scaffolds user decision-making and ownership',
        },
    ),
    Criterion(
        name='anchor_management',
        question='Does the episode reduce fixation on early assistant suggestions or manage anchors transparently?',
        definition=('Rate whether the episode acknowledges, contrasts, or helps the user '
                    'escape an initial anchor (first strong suggestion), vs reinforcing it.'),
        anchors={
            0: 'Reinforces the first anchor without alternatives',
            1: 'Minor variation around the anchor',
            2: 'Some alternatives but anchor remains dominant',
            3: 'Actively contrasts anchor with alternatives',
            4: 'Helps user escape, reinterpret, or deliberately choose the anchor',
        },
    ),
    Criterion(
        name='coregulation_uptake',
        question='Does one party\u2019s stance productively shape the next party\u2019s response?',
        definition=('Rate the quality of uptake: does the user build on the assistant\'s '
                    'move, or does the assistant build on the user\'s intent, to transform '
                    'the idea jointly?'),
        anchors={
            0: 'No uptake or breakdown',
            1: 'Superficial acknowledgment',
            2: 'User follows the topic but not the reasoning',
            3: 'User builds on the assistant\'s stance, or assistant builds on user intent',
            4: 'Clear collaborative transformation of the idea',
        },
    ),
    Criterion(
        name='timing_fit',
        question='Is the episode\u2019s regulatory move appropriate for the current stage of the conversation?',
        definition=('Rate whether the episode\'s move (opening, narrowing, committing, '
                    'reframing) is well-timed for the conversation\'s current phase.'),
        anchors={
            0: 'Clearly mistimed; harms process',
            1: 'Weak timing',
            2: 'Acceptable but generic',
            3: 'Well matched to current stage',
            4: 'Precisely regulates transition from exploration to evaluation or vice versa',
        },
    ),
    Criterion(
        name='implementation_grounding',
        question='Does the episode move from abstract idea toward actionable, constraint-aware solution development?',
        definition=('Rate whether the episode moves from abstract toward actionable: '
                    'constraints, resources, stakeholders, next steps.'),
        anchors={
            0: 'Purely abstract or decorative',
            1: 'Vague implementation language',
            2: 'Some practical constraints',
            3: 'Clear next steps, stakeholders, or resources',
            4: 'Strong implementation pathway with constraints and tradeoffs',
        },
    ),
    Criterion(
        name='cognitive_load_clarity',
        question='Does the episode reduce the user\u2019s burden and make the next action clearer?',
        definition=('Rate whether the episode is structured, concise, and makes the '
                    'next usable action obvious, vs overwhelming or vague.'),
        anchors={
            0: 'Overwhelming, vague, or confusing',
            1: 'Too many options without structure',
            2: 'Some structure',
            3: 'Clear, usable, appropriately concise',
            4: 'Strong scaffolding that makes the next action obvious',
        },
    ),
    Criterion(
        name='stance_integrity',
        question='Does the assistant remain faithful to its apparent role or stance without collapsing into generic answer-giving?',
        definition=('Rate how stably the assistant maintains a distinct stance (open, '
                    'evaluative, structured, bounded, etc.) vs drifting into generic '
                    'answer-giving.'),
        anchors={
            0: 'Stance collapses into generic behavior',
            1: 'Weak or inconsistent stance',
            2: 'Some stance markers',
            3: 'Clear stance behavior',
            4: 'Strong, stable stance without oversteering',
        },
    ),
    Criterion(
        name='premature_convergence_risk',
        question='Does the episode narrow too early, over-commit, or make alternatives socially/cognitively unavailable?',
        definition=('Rate the risk of premature closure: early commitment, pressure to '
                    'agree, or suppression of alternatives. HIGHER = WORSE.'),
        anchors={
            0: 'No premature convergence risk',
            1: 'Slight narrowing',
            2: 'Moderate narrowing but recoverable',
            3: 'Strong premature closure',
            4: 'Severe closure that dominates later interaction',
        },
        reverse=True,
    ),
    Criterion(
        name='runaway_divergence_risk',
        question='Does the episode expand without helping selection, criteria, or progress?',
        definition=('Rate the risk of uncontrolled branching without a commitment path. '
                    'HIGHER = WORSE.'),
        anchors={
            0: 'No runaway divergence risk',
            1: 'Slight expansion without structure',
            2: 'Many options but some organization',
            3: 'Expansion creates overload or postpones progress',
            4: 'Severe uncontrolled branching without commitment path',
        },
        reverse=True,
    ),
)

CRITERION_NAMES: tuple[str, ...] = tuple(c.name for c in CRITERIA)


# -------- Episode type taxonomy (per Stage 2) --------
EPISODE_TYPES = (
    'opening_frame',
    'ideation_burst',
    'reframe_event',
    'critique_event',
    'commitment_event',
    'repair_event',
    'anchor_return',
    'user_agency_event',
    'implementation_grounding_event',
    'summary_or_consolidation',
    'other',
)

# Which criteria to emphasize for each episode type (applicability hints for
# the scorer). Returning null for inapplicable criteria is always allowed.
EPISODE_TYPE_TO_CRITERIA: dict[str, tuple[str, ...]] = {
    'opening_frame': ('exploration_opening', 'stance_integrity', 'timing_fit', 'cognitive_load_clarity'),
    'ideation_burst': ('exploration_opening', 'runaway_divergence_risk', 'agency_preservation', 'coregulation_uptake'),
    'reframe_event': ('reframing_quality', 'exploration_opening', 'timing_fit', 'stance_integrity'),
    'critique_event': ('evaluative_discipline', 'agency_preservation', 'premature_convergence_risk', 'timing_fit'),
    'commitment_event': ('evaluative_discipline', 'implementation_grounding', 'timing_fit', 'premature_convergence_risk'),
    'repair_event': ('coregulation_uptake', 'cognitive_load_clarity', 'agency_preservation'),
    'anchor_return': ('anchor_management', 'premature_convergence_risk'),
    'user_agency_event': ('agency_preservation', 'coregulation_uptake'),
    'implementation_grounding_event': ('implementation_grounding', 'cognitive_load_clarity', 'evaluative_discipline'),
    'summary_or_consolidation': ('cognitive_load_clarity', 'premature_convergence_risk', 'timing_fit'),
    'other': CRITERION_NAMES,   # score everything applicable
}


# -------- Bias-flag enum (per § E1) --------
BIAS_FLAGS_ALLOWED = (
    'none',
    'length_bias',
    'fluency_bias',
    'condition_leakage',
    'persona_label_leakage',
    'order_bias',
    'insufficient_context',
    'translation_uncertainty',
    'ambiguous_speaker_role',
    'unsupported_inference',
)


# -------- JSON schemas (bundled scorer returns one object per episode) --------

class CriterionScore(BaseModel):
    criterion: str = Field(description='one of the 12 criterion names')
    score_0_4: int | None = Field(description='integer 0-4, or null if inapplicable')
    confidence_0_1: float = Field(description='confidence in the score, 0.0-1.0')
    evidence_quotes: list[str] = Field(default_factory=list, description='verbatim quotes from the episode supporting the score')
    reason_short: str = Field(description='one or two sentence rationale')
    counterevidence: str = Field(default='', description='quote or note contradicting the score, or empty')
    possible_biases: list[str] = Field(default_factory=lambda: ['none'])
    usable_for_inference: bool = Field(default=True)


class BundledEpisodeScore(BaseModel):
    """One call per episode returns this; the scorer decides which criteria apply."""
    conversation_id: str
    episode_id: str
    scores: list[CriterionScore] = Field(description='one entry per applicable criterion; null score_0_4 for inapplicable')


class AuditDecision(BaseModel):
    """Conservative Auditor output (per § F2)."""
    conversation_id: str
    episode_id: str
    criterion: str
    original_score: int | None
    audit_decision: Literal['keep', 'lower', 'exclude', 'flag']
    recommended_score: int | None
    audit_reason: str
    counterevidence: str = ''
    bias_flags: list[str] = Field(default_factory=list)


class Counterexample(BaseModel):
    """Counterexample Agent output (per § F3)."""
    conversation_id: str
    episode_id: str
    criterion: str
    counterexample_found: bool
    counterexample_quote: str = ''
    why_it_matters: str = ''
    recommended_action: Literal['keep', 'lower', 'exclude', 'flag']


# -------- Episode-segmenter schema --------

class EpisodeBoundary(BaseModel):
    start_turn: int
    end_turn: int
    episode_type: str
    segmentation_reason: str
    confidence_0_1: float


class EpisodeSegmentation(BaseModel):
    conversation_id: str
    episodes: list[EpisodeBoundary]
