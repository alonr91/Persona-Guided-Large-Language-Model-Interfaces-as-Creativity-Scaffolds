"""Eight user-creativity dimensions for the CAT-Panel layer.

Each dimension is scored on a 1-7 Likert scale (CHI/CSCW convention).
Anchors are written at 1 / 4 / 7 with brief operational definitions
between them. Citations point back to the lit-review constructs.

Scope: every dimension scores the USER's behaviour in a conversation
between a participant and a masked AI assistant. Holistic creativity
is NOT judged here — only bounded user-behaviour constructs whose
evidence must be quoted verbatim from user turns.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Dimension:
    name: str            # snake_case identifier
    short_label: str     # human-readable label for figures
    question: str        # the rating question shown to the judge
    construct: str       # one-line operational definition
    anchors: dict[int, str]   # 1, 4, 7 anchored
    citations: str       # canonical lit-review pointers


DIMENSIONS: Sequence[Dimension] = (
    Dimension(
        name='user_ideational_fluency',
        short_label='Ideational fluency',
        question=(
            'To what extent does the USER generate a productive volume of '
            'distinct candidate directions, options, or proposals across '
            'the conversation?'
        ),
        construct=(
            'Quantity-of-distinct-ideas in the user turns. Not the same as '
            'word count: a user who writes long but elaborates a single '
            'direction scores low; a user who offers several different '
            'directions scores high.'
        ),
        anchors={
            1: 'The user offers no candidate directions, or only one, or '
               'merely accepts what the assistant proposes.',
            4: 'The user offers a moderate number of distinct directions '
               '(roughly 2-4), with some elaboration but limited range.',
            7: 'The user generates many distinct directions throughout the '
               'conversation, visibly seeking breadth rather than depth on '
               'any single proposal.',
        },
        citations='Guilford 1967; Amabile 1983; Acar et al. 2017',
    ),
    Dimension(
        name='user_cognitive_flexibility',
        short_label='Cognitive flexibility',
        question=(
            'To what extent does the USER move across categories or '
            'representational frames rather than elaborating one direction?'
        ),
        construct=(
            'Category-jumping vs. within-category elaboration. A user who '
            'reframes the search across different domains (e.g., physical '
            'design to social system to economic mechanism) scores high; '
            'a user who deepens one frame throughout scores low.'
        ),
        anchors={
            1: 'The user remains within a single representational frame for '
               'the entire conversation; no category jumps.',
            4: 'The user makes one or two category jumps, with most turns '
               'staying within a single frame.',
            7: 'The user repeatedly jumps across categories or '
               'representations, deliberately probing different parts of '
               'the solution space.',
        },
        citations='Pinkow 2023; Nijstad et al. 2010 dual-pathway',
    ),
    Dimension(
        name='user_problem_frame_development',
        short_label='Problem-frame development',
        question=(
            'To what extent does the USER\'s framing of the problem itself '
            'evolve through the conversation (problem-solution co-evolution)?'
        ),
        construct=(
            'Does the user merely apply a fixed framing of the task across '
            'turns, or does the way the user understands "what the problem '
            'is" actually shift as the dialogue unfolds? Pure elaboration '
            'of solutions under a frozen framing is NOT problem-frame '
            'development.'
        ),
        anchors={
            1: 'The user keeps the original problem framing untouched; only '
               'solutions evolve, the problem definition does not.',
            4: 'The user revisits the problem framing once or twice in '
               'response to constraints or new information, with modest '
               'reframing.',
            7: 'The user repeatedly reformulates what the problem actually '
               'is — surfacing new constraints, stakeholders, or success '
               'criteria — and the later turns use a meaningfully different '
               'framing than the opening turns.',
        },
        citations='Dorst & Cross 2001 co-evolution; Schön & Wiggins 1992',
    ),
    Dimension(
        name='user_reflective_engagement',
        short_label='Reflective engagement',
        question=(
            'To what extent does the USER display in-action reflection on '
            'what they are doing and why — articulating assumptions, '
            'questioning their own framing, or surfacing rationale?'
        ),
        construct=(
            'Reflection-in-action in the Schön (1983) sense. Look for '
            'turns where the user steps back and comments on the search '
            'itself, the assumptions in play, or why a direction is or is '
            'not promising. Surface "I think X" without rationale is not '
            'reflection.'
        ),
        anchors={
            1: 'The user proceeds transactionally; no visible reflection on '
               'their own search, assumptions, or rationale.',
            4: 'The user articulates rationale on a few key moves but does '
               'not sustain reflection across the conversation.',
            7: 'The user repeatedly surfaces assumptions, questions their '
               'own framing, and reasons aloud about why they are or are '
               'not pursuing a direction.',
        },
        citations='Schön 1983; Schön & Wiggins 1992',
    ),
    Dimension(
        name='user_constraint_integration',
        short_label='Constraint integration',
        question=(
            'To what extent does the USER acknowledge and work with '
            'feasibility, resource, context, or stakeholder constraints '
            'rather than ignoring them?'
        ),
        construct=(
            'Constraint-aware ideation in the Acar et al. (2019) sense. '
            'Input, process, and output constraints all count. A user who '
            'imagines anything without grounding scores low; a user who '
            'visibly integrates constraints into their proposals scores '
            'high. Note: blanket dismissal of constraints (everything is '
            'impossible) is also low.'
        ),
        anchors={
            1: 'The user proposes ideas with no acknowledgement of '
               'feasibility, cost, time, stakeholder, or context constraints.',
            4: 'The user references constraints occasionally but does not '
               'visibly let them shape proposal generation.',
            7: 'The user integrates feasibility, resource, and context '
               'constraints into proposals throughout, treating them as '
               'design inputs rather than blockers.',
        },
        citations='Acar et al. 2019; Anderson et al. 2014',
    ),
    Dimension(
        name='user_epistemic_stance_regulation',
        short_label='Epistemic stance regulation',
        question=(
            'To what extent does the USER modulate between exploratory '
            'openness and evaluative narrowing across the conversation — '
            'rather than staying in one mode throughout?'
        ),
        construct=(
            'The user\'s meta-control over divergent vs convergent '
            'thinking. Look for hedging-to-asserting transitions, '
            '"what-if" exploration phases followed by criteria-setting '
            'phases, and evidence that the user is choosing when to widen '
            'vs narrow. Staying purely divergent or purely convergent the '
            'whole conversation scores low on regulation (even though it '
            'may score high on other dimensions).'
        ),
        anchors={
            1: 'The user remains in a single mode (purely exploratory OR '
               'purely evaluative) throughout the conversation, with no '
               'visible mode switching.',
            4: 'The user makes one or two clear mode switches but does not '
               'sustain a regulated alternation between openness and '
               'narrowing.',
            7: 'The user repeatedly and visibly regulates between '
               'exploratory openness and evaluative narrowing, with timing '
               'that fits the local task demands.',
        },
        citations='White 2003 expansion/contraction; Kiesling 2022; '
                  'Sowden et al. 2015; Zhang et al. 2020 metacontrol',
    ),
    Dimension(
        name='user_authorship_direction_setting',
        short_label='Authorship & direction-setting',
        question=(
            'To what extent does the USER own the direction of the '
            'collaboration — the assistant follows the user rather than '
            'the user following the assistant?'
        ),
        construct=(
            'Who is driving. A user who issues instructions, sets '
            'direction, and uses the assistant as a tool scores high. A '
            'user who waits for the assistant to propose then says "yes, '
            'do that" scores low. Felt-authorship in the Draxler / Hwang '
            'sense, but read from interaction-level behaviour.'
        ),
        anchors={
            1: 'The assistant leads; the user reacts. Most turns are user '
               'agreement, acceptance, or minor requests for elaboration.',
            4: 'Mixed leadership: the user sets direction on some turns '
               'and reacts on others.',
            7: 'The user clearly leads throughout: setting the agenda, '
               'redirecting when needed, and treating the assistant as a '
               'tool rather than a co-author.',
        },
        citations='Draxler et al. 2024; Hwang et al. 2025; '
                  'Pickering & Garrod 2004 alignment',
    ),
    Dimension(
        name='user_implementation_relevant_progress',
        short_label='Implementation-relevant progress',
        question=(
            'To what extent does the USER move the dialogue toward an '
            'implementable, constraint-sensitive solution — rather than '
            'collecting ideas without progressing toward action?'
        ),
        construct=(
            'Innovation-relevant rather than purely creativity-relevant '
            '(per the Anderson et al. 2014 / Amabile & Pratt 2016 '
            'distinction). Has the user, by the end of the conversation, '
            'moved meaningfully closer to something a stakeholder could '
            'act on? Pure ideation without selection or refinement scores '
            'low even if it is creative.'
        ),
        anchors={
            1: 'The conversation ends with a pile of ideas and no '
               'visible progress toward an implementable direction.',
            4: 'The conversation reaches a partially-specified candidate '
               'direction by the end, with some but limited implementation '
               'grounding.',
            7: 'The conversation reaches a meaningfully implementable, '
               'constraint-sensitive direction with clear next-step '
               'specifics by the end.',
        },
        citations='Anderson et al. 2014; Amabile & Pratt 2016; '
                  'Bharadwaj & Menon 2000',
    ),
)

DIM_NAMES = tuple(d.name for d in DIMENSIONS)
DIM_LABELS = {d.name: d.short_label for d in DIMENSIONS}
DIM_BY_NAME = {d.name: d for d in DIMENSIONS}


def render_dimension_block(d: Dimension) -> str:
    """Render one dimension as the rubric text the judge sees."""
    anchors = '\n'.join(f'  {k} = {v}' for k, v in d.anchors.items())
    return (
        f'Dimension: {d.name}\n'
        f'Question: {d.question}\n'
        f'Construct: {d.construct}\n'
        f'Citations: {d.citations}\n'
        f'Anchors on the 1-7 Likert (1 = not at all evident, '
        f'4 = moderately evident, 7 = strongly evident):\n{anchors}'
    )


def render_all_dimensions() -> str:
    """For audit purposes — render the full dimension block as one string."""
    return '\n\n---\n\n'.join(render_dimension_block(d) for d in DIMENSIONS)


if __name__ == '__main__':
    # Self-check: print all dimensions
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
    print(f'CAT-Panel has {len(DIMENSIONS)} dimensions:\n')
    for d in DIMENSIONS:
        print(f'  - {d.name}  ({d.short_label})')
    print('\n--- full dimension block ---\n')
    print(render_all_dimensions())
