"""Agent 2 — per-conversation consolidator.

Takes Agent-1 candidates from a single conversation, embeds them with
BAAI/bge-large-en-v1.5, clusters at cosine distance <= 0.20 (sim >= 0.80)
via agglomerative clustering, then asks the LLM to summarize each non-
trivial cluster into one canonical idea. Size-1 clusters pass through.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

from .config import (Canonicalization, CONSOLIDATION_SIM_THRESHOLD,
                     EMBED_MODEL_NAME)
from .llm_client import LLMClient
from .agent1_extractor import CandidateRow


_embed_model: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f'[agent2] loading embed model: {EMBED_MODEL_NAME}')
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _embed_candidates(cands: list[CandidateRow]) -> np.ndarray:
    model = _get_embedder()
    texts = [f'{c.title}: {c.description}' for c in cands]
    V = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(V, dtype='float32')


SYSTEM_CANONICALIZE = (
    "You merge multiple candidate descriptions of the SAME IDEA into one "
    "canonical entry. Respond only with JSON matching the schema.\n\n"
    "VOCABULARY RULE (strict):\n"
    "The title MUST use words that appear in at least one of the candidate "
    "titles or descriptions. Do NOT introduce new synonyms or domain "
    "vocabulary that isn't already in the candidates. For example, if the "
    "candidates mention 'cycling clothes that have become trends', the "
    "title may say 'Cycling clothes as trends' but NOT 'Fashionable "
    "Cycling Apparel' (apparel/fashionable are synonyms the user didn't use).\n\n"
    "If the candidates actually describe DIFFERENT ideas (e.g. 'cycling "
    "clothes' vs 'helmets'), DO NOT merge them into one composite title. "
    "Instead pick the strongest single-idea title from the candidates; the "
    "pipeline will handle multi-idea cases separately.\n\n"
    "Title: <=10 words, using candidate vocabulary.\n"
    "Description: one tight sentence that captures the shared proposal "
    "without speculation or new content."
)


def _canonicalize_cluster(members: list[CandidateRow]) -> tuple[str, str]:
    if len(members) == 1:
        return members[0].title, members[0].description
    body = '\n\n'.join(
        f'Candidate {i+1}:\n  title: {m.title}\n  description: {m.description}'
        for i, m in enumerate(members)
    )
    user_prompt = (
        f'The following {len(members)} candidates describe the same underlying '
        f'idea. Produce one canonical title + description.\n\n{body}'
    )
    obj, _ = LLMClient.generate_json(
        Canonicalization, SYSTEM_CANONICALIZE, user_prompt,
        temperature=0.1, max_new_tokens=200,
    )
    if obj is None:
        # fallback: pick the longest description as canonical
        best = max(members, key=lambda m: len(m.description))
        return best.title, best.description
    return obj.title.strip(), obj.description.strip()


@dataclass
class CanonicalRow:
    conversation_id: int
    canonical_id: int
    title: str
    description: str
    evidence_quotes: list[str]
    source_candidate_ids: list[int]  # indices into the per-conv candidate list


def consolidate(candidates: list[CandidateRow], conversation_id: int) -> list[CanonicalRow]:
    """Cluster + summarize candidates from one conversation."""
    if len(candidates) == 0:
        return []
    if len(candidates) == 1:
        c = candidates[0]
        return [CanonicalRow(conversation_id=conversation_id, canonical_id=0,
                             title=c.title, description=c.description,
                             evidence_quotes=[c.evidence_span],
                             source_candidate_ids=[0])]

    V = _embed_candidates(candidates)
    # cosine distance = 1 - cos sim on L2-normalized
    dist_thresh = 1.0 - CONSOLIDATION_SIM_THRESHOLD
    clf = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dist_thresh,
        metric='cosine', linkage='average'
    )
    labels = clf.fit_predict(V)

    rows: list[CanonicalRow] = []
    for cid in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == cid]
        members = [candidates[i] for i in idx]
        title, desc = _canonicalize_cluster(members)
        rows.append(CanonicalRow(
            conversation_id=conversation_id,
            canonical_id=int(cid),
            title=title,
            description=desc,
            evidence_quotes=[m.evidence_span for m in members],
            source_candidate_ids=idx,
        ))
    return rows
