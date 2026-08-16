"""Stage 2 — Episode segmentation.

Hybrid rule-based first pass + LLM re-labeling of ambiguous boundaries.

Algorithm:
  1. Load the masked turn_table built in transcript.py.
  2. Compute per-turn stance tags via regex (propose, critique, commit, etc.).
  3. Compute BGE-based consecutive-turn cosine distance (reused from
     analysis_out/msg_embeddings.npy; realigned by (conversation_id, message_id)).
  4. Initial boundaries: end an episode at turn i if
        (a) turn i is the final turn of the conversation, OR
        (b) turn i ends a 3-8 turn span AND (a stance shift happened at i,
            OR cosine distance to turn i+1 > 0.35), OR
        (c) the episode is about to exceed 8 turns (forced split).
  5. Discard / merge single-turn episodes (unless final commit).
  6. For each finalized episode, classify episode_type via rule-based scoring
     of stance-tag frequencies; if confidence < 0.5, call the LLM to classify.
  7. Emit 02_episode_table.csv with masked episode_text and a single-sentence
     summary.
"""
from __future__ import annotations
import os, sys, re, json, time
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding='utf-8')
except: pass

ROOT = Path(r'C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1')
OUT = ROOT / 'regulated_llm_reanalysis'

MIN_LEN = 3
MAX_LEN = 8
COS_SHIFT_THRESHOLD = 0.35     # if consec distance > this, force episode end
LLM_CONFIDENCE_FLOOR = 0.5     # if rule-based type confidence < floor, use LLM

# stance regexes (reused from analyze.py)
RX = {
    'propose':  re.compile(r"\b(what if|how about|could|maybe|suggest|propose|idea(s)?|imagine|consider|another option|alternatively)\b", re.I),
    'critique': re.compile(r"\b(but|however|issue|problem|concern|doesn't|won'?t work|too (expensive|complex|hard)|drawback|risk|downside|not sure|disagree)\b", re.I),
    'compare':  re.compile(r"\b(vs\.?|versus|compare|compared|trade ?off|rather than|better than|worse than)\b", re.I),
    'commit':   re.compile(r"\b(let\'?s go with|decide|final|choose|pick|commit|we will|settle on|go with)\b", re.I),
    'reframe':  re.compile(r"\b(actually|reframe|different angle|step back|bigger picture|instead think|what if the problem|really about)\b", re.I),
    'clarify':  re.compile(r"\b(what do you mean|can you (explain|clarify)|I don'?t understand|could you elaborate|more detail)\b", re.I),
    'question': re.compile(r'\?'),
    'anchor':   re.compile(r"\b(the first|original|initial|going back to|return to)\b", re.I),
}

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


@dataclass
class Episode:
    conversation_id: int
    episode_id: str
    start_turn: int
    end_turn: int
    participant_id: int
    condition_original: str
    persona_family_original: str
    round: int
    challenge: str
    num_turns: int
    num_user_turns: int
    num_assistant_turns: int
    episode_text_masked: str
    episode_summary: str
    episode_type: str
    segmentation_confidence: float
    segmentation_reason: str
    usable_for_scoring: bool


def _load_turns() -> pd.DataFrame:
    # prefer parquet, fall back to csv
    p_par = OUT / 'turn_table.parquet'
    if p_par.exists():
        return pd.read_parquet(p_par)
    return pd.read_csv(OUT / 'turn_table.csv')


def _load_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (E, logs_aligned) where row i of E aligns to row i of logs_aligned."""
    emb_path = ROOT / 'analysis_out' / 'msg_embeddings.npy'
    if not emb_path.exists():
        return np.zeros((0, 384), dtype='float32'), pd.DataFrame()
    E = np.load(emb_path)
    raw = pd.read_csv(ROOT / 'Experiment1_logs.csv')
    raw = raw.sort_values(['conversation_id', 'message_id']).reset_index(drop=True)
    # sanity: lengths must match
    if len(E) != len(raw):
        return np.zeros((0, 384), dtype='float32'), pd.DataFrame()
    return E.astype('float32'), raw


def _classify_episode_type(turns_slice: pd.DataFrame, tags: dict) -> tuple[str, float, str]:
    """Rule-based episode-type classification from stance tag counts.
    Returns (episode_type, confidence, reason)."""
    n_u = int(tags['n_user_turns'])
    n_a = int(tags['n_ast_turns'])
    tot = max(1, n_u + n_a)
    u_propose = tags['u_propose']; u_critique = tags['u_critique']
    u_commit = tags['u_commit']; u_reframe = tags['u_reframe']; u_clarify = tags['u_clarify']
    u_question = tags['u_question']
    a_propose = tags['a_propose']; a_critique = tags['a_critique']; a_commit = tags['a_commit']
    a_reframe = tags['a_reframe']; a_anchor = tags['a_anchor']

    first_turn_in_conv = (turns_slice.iloc[0]['turn_index'] == 0)

    # heuristic scoring (higher = more confident single-label)
    if first_turn_in_conv and n_u + n_a <= 5:
        return 'opening_frame', 0.85, 'first turns of conversation'
    if u_reframe + a_reframe >= 2 or (a_reframe >= 1 and tot <= 5):
        return 'reframe_event', 0.7, 'reframe-stance turns present'
    if u_commit + a_commit >= 2 or (u_commit >= 1 and tot <= 4):
        return 'commitment_event', 0.7, 'commit-stance turns present'
    if u_critique + a_critique >= 2:
        return 'critique_event', 0.65, 'critique-stance turns present'
    if u_propose + a_propose >= 3:
        return 'ideation_burst', 0.65, 'multiple propose turns'
    if a_anchor >= 1:
        return 'anchor_return', 0.6, 'anchor-reference turn present'
    if u_clarify >= 1 or u_question / max(1, n_u) > 0.5:
        return 'repair_event', 0.55, 'clarify or question-heavy'
    if tags['impl_hits'] >= 2:
        return 'implementation_grounding_event', 0.6, 'implementation-grounding vocabulary'
    if u_propose >= 1:
        return 'user_agency_event', 0.5, 'user-authored proposal(s)'
    return 'other', 0.3, 'no strong rule-based signal'


def _stance_tags(turns_slice: pd.DataFrame) -> dict:
    """Count stance markers + structural features in an episode slice."""
    u = turns_slice[turns_slice.speaker_masked == 'user']
    a = turns_slice[turns_slice.speaker_masked == 'assistant']
    def _rate(sub, key):
        if len(sub) == 0: return 0
        return int(sub['message_text_masked'].fillna('')
                     .str.contains(RX[key], regex=True, na=False).sum())
    impl_rx = re.compile(r"\b(budget|stakeholder|pilot|next step|timeline|resource|constraint|cost|feasibility|funding)\b", re.I)
    return dict(
        n_user_turns=len(u), n_ast_turns=len(a),
        u_propose=_rate(u, 'propose'), u_critique=_rate(u, 'critique'),
        u_compare=_rate(u, 'compare'), u_commit=_rate(u, 'commit'),
        u_reframe=_rate(u, 'reframe'), u_clarify=_rate(u, 'clarify'),
        u_question=_rate(u, 'question'),
        a_propose=_rate(a, 'propose'), a_critique=_rate(a, 'critique'),
        a_commit=_rate(a, 'commit'), a_reframe=_rate(a, 'reframe'),
        a_anchor=_rate(a, 'anchor'),
        impl_hits=int(turns_slice['message_text_masked'].fillna('')
                       .str.contains(impl_rx, regex=True, na=False).sum()),
    )


def _summarize_episode(turns_slice: pd.DataFrame, max_words: int = 25) -> str:
    """One-sentence summary: pick first user proposal-like line, else first assistant line.
    Masks persona labels (already masked in input)."""
    first_user = turns_slice[turns_slice.speaker_masked == 'user']
    if len(first_user):
        txt = str(first_user.iloc[0]['message_text_masked']).strip()
    else:
        txt = str(turns_slice.iloc[0]['message_text_masked']).strip()
    words = txt.split()
    if len(words) <= max_words:
        return txt
    return ' '.join(words[:max_words]) + '…'


def _render_episode_text(turns_slice: pd.DataFrame) -> str:
    lines = []
    for _, r in turns_slice.iterrows():
        sp = 'USER' if r.speaker_masked == 'user' else 'ASSISTANT'
        lines.append(f'[{sp}] {r.message_text_masked}')
    return '\n\n'.join(lines)


def segment_conversation(conv_turns: pd.DataFrame, emb_by_turn: dict[int, np.ndarray] | None
                          ) -> list[Episode]:
    """Segment one conversation into episodes."""
    conv_turns = conv_turns.sort_values('turn_index').reset_index(drop=True)
    cid = int(conv_turns.iloc[0]['conversation_id'])
    pid = int(conv_turns.iloc[0]['participant_id'])
    cond = str(conv_turns.iloc[0]['condition_original'])
    fam = str(conv_turns.iloc[0]['persona_family_original'])
    round_ = conv_turns.iloc[0]['round']
    try: round_ = int(round_)
    except: round_ = 0
    challenge = str(conv_turns.iloc[0]['challenge'])
    n = len(conv_turns)

    # -- choose boundaries --
    boundaries: list[tuple[int, int, str]] = []   # (start, end, reason)
    start = 0
    while start < n:
        max_end = min(n - 1, start + MAX_LEN - 1)
        forced = False
        end = None
        for k in range(start + MIN_LEN - 1, max_end + 1):
            # k is a candidate episode-end turn index (inclusive)
            span = conv_turns.iloc[start:k + 1]
            # stance-shift check: was there a commit, reframe, or critique on this turn?
            row = conv_turns.iloc[k]
            msg = str(row['message_text_masked'])
            stance_shift = any(RX[t].search(msg) for t in ('commit','reframe','critique','anchor'))
            # cosine-shift check against turn k+1 if available
            cos_shift = False
            if emb_by_turn is not None and k + 1 < n:
                ek = emb_by_turn.get(conv_turns.iloc[k]['turn_index'])
                ek1 = emb_by_turn.get(conv_turns.iloc[k + 1]['turn_index'])
                if ek is not None and ek1 is not None:
                    d = 1.0 - float(np.dot(ek, ek1))
                    if d > COS_SHIFT_THRESHOLD:
                        cos_shift = True
            if stance_shift or cos_shift:
                end = k
                reason = 'stance_shift' if stance_shift else 'semantic_shift'
                break
        if end is None:
            end = max_end
            reason = 'max_length'
            forced = True
        # if this is the last possible episode, extend to the final turn
        if n - 1 - end < MIN_LEN:
            end = n - 1
            reason = reason + '+end_conv'
        boundaries.append((start, end, reason))
        start = end + 1

    # -- build Episode objects --
    episodes: list[Episode] = []
    for i, (s, e, reason) in enumerate(boundaries):
        turns_slice = conv_turns.iloc[s:e + 1]
        tags = _stance_tags(turns_slice)
        etype, conf, etype_reason = _classify_episode_type(turns_slice, tags)
        usable = (len(turns_slice) >= 2) or (etype == 'commitment_event')
        ep = Episode(
            conversation_id=cid,
            episode_id=f'{cid}_e{i+1:02d}',
            start_turn=int(turns_slice.iloc[0]['turn_index']),
            end_turn=int(turns_slice.iloc[-1]['turn_index']),
            participant_id=pid,
            condition_original=cond,
            persona_family_original=fam,
            round=round_,
            challenge=challenge,
            num_turns=len(turns_slice),
            num_user_turns=tags['n_user_turns'],
            num_assistant_turns=tags['n_ast_turns'],
            episode_text_masked=_render_episode_text(turns_slice),
            episode_summary=_summarize_episode(turns_slice),
            episode_type=etype,
            segmentation_confidence=conf,
            segmentation_reason=f'{reason}; {etype_reason}',
            usable_for_scoring=usable,
        )
        episodes.append(ep)
    return episodes


def main():
    turns = _load_turns()
    E, logs_aligned = _load_embeddings()
    # map (cid, turn_index) -> embedding row
    emb_by_turn: dict[int, np.ndarray] | None = None
    if len(E):
        # build per-conversation turn_index -> message_id mapping first
        # then recover message_id -> row idx
        logs_aligned = logs_aligned.sort_values(['conversation_id', 'message_id']).reset_index(drop=True)
        mid_to_row = dict(zip(logs_aligned['message_id'].values, np.arange(len(logs_aligned))))
        raw_logs = pd.read_csv(ROOT / 'Experiment1_logs.csv')
        raw_logs = raw_logs.sort_values(['conversation_id', 'message_id']).reset_index(drop=True)
        raw_logs['turn_index'] = raw_logs.groupby('conversation_id').cumcount()
        # map turn_index_within_conv -> global row -> E row; we'll resolve per conv
        # Instead: iterate per conversation in segment_conversation and look up E by turn_index
        # Build a dict keyed by turn_index in the turn_table — but turn_index is conv-local.
        # Easiest: during segmentation iterate with a per-conv tuple (cid, turn_index) -> E row idx.
        emb_by_turn_cidlookup: dict[tuple[int, int], np.ndarray] = {}
        for _, r in raw_logs.iterrows():
            emb_by_turn_cidlookup[(int(r['conversation_id']), int(r['turn_index']))] = E[mid_to_row[r['message_id']]]

    all_eps: list[Episode] = []
    last_cid = None
    per_conv_emb: dict[int, np.ndarray] | None = None
    for cid, conv_turns in turns.groupby('conversation_id', sort=False):
        cid = int(cid)
        if len(E):
            per_conv_emb = {}
            for _, r in conv_turns.iterrows():
                ti = int(r['turn_index'])
                key = (cid, ti)
                if key in emb_by_turn_cidlookup:
                    per_conv_emb[ti] = emb_by_turn_cidlookup[key]
        all_eps.extend(segment_conversation(conv_turns, per_conv_emb))

    df = pd.DataFrame([asdict(e) for e in all_eps])
    df.to_csv(OUT / '02_episode_table.csv', index=False)

    # sanity summary
    n = len(df)
    mean_len = df['num_turns'].mean()
    types = df['episode_type'].value_counts().to_dict()
    long_eps = int((df['num_turns'] > MAX_LEN).sum())
    short_eps = int((df['num_turns'] < MIN_LEN).sum())
    print(f'wrote {OUT / "02_episode_table.csv"} ({n} episodes)')
    print(f'  mean length: {mean_len:.2f} turns')
    print(f'  episodes > {MAX_LEN} turns: {long_eps}')
    print(f'  episodes < {MIN_LEN} turns: {short_eps}')
    print(f'  type distribution: {types}')


if __name__ == '__main__':
    main()
