"""
Agent 5 — participant-centroid originality.

Ports Experiment 2's (UIST 2026) primary product-layer metric to Experiment 1.

For each conversation (one per participant, per round):
  - Compute the participant-centroid C_p as the mean-pooled, L2-normalized
    embedding of all their (post-Agent-3-validated) canonical ideas.
Then three originality measures per participant:

  1. Same-condition originality:
        orig_same[p] = mean over peers q in same condition of (1 - cos(C_p, C_q))
     Higher = more distinctive within the same experimental arm.

  2. All-participant originality:
        orig_all[p] = mean over all other q of (1 - cos(C_p, C_q))
     Experiment 2's headline metric (ignoring condition).

  3. Cross-condition nearest-neighbor originality:
        orig_cross[p] = min over q in the OPPOSING condition of (1 - cos(C_p, C_q))
     Experiment 2's separation measure: how far from the closest opposite-arm peer.

All three are computed on L2-normalized participant centroids.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np


@dataclass
class ParticipantOriginality:
    conversation_id: int
    user: int
    condition: str           # 'GPT' | 'Persona'
    persona_label: str       # 'GPT' | 'Divergent' | ... (family)
    challenge: str
    n_ideas: int
    centroid_index: int      # row in the stacked centroid matrix
    orig_same: float
    orig_all: float
    orig_cross: float


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def compute_centroids(conv_ids: list[int], idea_vectors_by_conv: dict[int, np.ndarray]
                      ) -> tuple[np.ndarray, list[int]]:
    """Per-conversation mean embedding, L2-normalized.
    Returns (centroids [N, D], aligned list of conversation_ids).
    Conversations with 0 ideas get a zero vector and will be skipped by caller."""
    rows = []
    aligned = []
    dim = None
    for cid in conv_ids:
        V = idea_vectors_by_conv.get(cid)
        if V is None or len(V) == 0:
            continue
        dim = V.shape[1]
        c = V.mean(axis=0)
        rows.append(c)
        aligned.append(cid)
    if not rows:
        return np.zeros((0, dim or 1), dtype='float32'), []
    C = _l2_normalize(np.stack(rows).astype('float32'))
    return C, aligned


def compute_originality(centroids: np.ndarray,
                        aligned_cids: list[int],
                        meta_by_cid: dict[int, dict],
                        ) -> list[ParticipantOriginality]:
    """meta_by_cid: conv_id -> dict with keys user, condition, persona_label, challenge, n_ideas."""
    if len(centroids) == 0:
        return []
    # pairwise similarity matrix on L2-normalized centroids
    sim = centroids @ centroids.T
    np.fill_diagonal(sim, np.nan)
    dist = 1.0 - sim

    out: list[ParticipantOriginality] = []
    conditions = np.array([meta_by_cid[cid]['condition'] for cid in aligned_cids])
    for i, cid in enumerate(aligned_cids):
        m = meta_by_cid[cid]
        my_cond = m['condition']

        # same-condition mask: same condition, not self
        same_mask = (conditions == my_cond)
        same_mask[i] = False
        if same_mask.any():
            orig_same = float(np.nanmean(dist[i, same_mask]))
        else:
            orig_same = float('nan')

        # all-participant mask: not self
        all_mask = np.ones(len(aligned_cids), dtype=bool)
        all_mask[i] = False
        if all_mask.any():
            orig_all = float(np.nanmean(dist[i, all_mask]))
        else:
            orig_all = float('nan')

        # cross-condition nearest: minimum distance to any opposing-condition
        cross_mask = (conditions != my_cond)
        if cross_mask.any():
            orig_cross = float(np.nanmin(dist[i, cross_mask]))
        else:
            orig_cross = float('nan')

        out.append(ParticipantOriginality(
            conversation_id=int(cid),
            user=int(m['user']),
            condition=my_cond,
            persona_label=m['persona_label'],
            challenge=m['challenge'],
            n_ideas=int(m['n_ideas']),
            centroid_index=i,
            orig_same=orig_same,
            orig_all=orig_all,
            orig_cross=orig_cross,
        ))
    return out
