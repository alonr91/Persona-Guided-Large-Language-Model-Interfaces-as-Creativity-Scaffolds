"""
Agent 4 — cross-participant idea categorization.

Input: all canonical ideas from Agent 2/3 across all conversations.
Step 1: embed "title + ': ' + description" with BAAI/bge-large-en-v1.5 (L2-normed).
Step 2: cluster via sklearn HDBSCAN in the full 1024-d space (or optionally
        UMAP-reduced) with min_cluster_size=4.
Step 3: for each non-noise cluster, call the LLM once with 3-5 exemplar ideas
        to produce a short category name (<=5 words).
Step 4: ideas flagged as noise by HDBSCAN get category_id=-1 and
        category_name='unclustered'.

Returns list[CategorizedIdea] and cluster_summary dict.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable
import numpy as np
from sklearn.cluster import HDBSCAN

from pydantic import BaseModel, Field

from .agent2_consolidator import CanonicalRow, _get_embedder
from .llm_client import LLMClient


class CategoryLabel(BaseModel):
    name: str = Field(description='category name, 2-5 words, title case')


SYSTEM_LABEL = (
    "You assign a short category name to a cluster of related ideas. "
    "Respond only with JSON matching the schema.\n\n"
    "Rules:\n"
    "- Category name: 2-5 words, title case.\n"
    "- Use vocabulary drawn from the ideas themselves; no new synonyms.\n"
    "- Prefer concrete nouns over abstractions when possible.\n"
    "- Describe the common theme, not one particular idea.\n"
    "Example: [{'title':'Cafe in library'},{'title':'Morning cafe operation'},"
    "{'title':'Library Cafe with Lectures'}] -> 'Library Cafe'."
)


def _label_cluster(members: list[CanonicalRow]) -> str:
    # pick up to 5 exemplars by description length (mid-sized are most
    # representative; avoid shortest and longest)
    if len(members) <= 5:
        exemplars = members
    else:
        sorted_by_len = sorted(members, key=lambda m: len(m.description))
        mid = len(sorted_by_len) // 2
        exemplars = sorted_by_len[max(0, mid - 2): mid + 3]
    body = '\n'.join(
        f'  - {{"title": "{m.title}", "description": "{m.description[:160]}"}}'
        for m in exemplars
    )
    user = (
        f'Assign a short category name to this cluster of {len(members)} '
        f'related ideas (showing {len(exemplars)} exemplars):\n{body}'
    )
    obj, _ = LLMClient.generate_json(
        CategoryLabel, SYSTEM_LABEL, user,
        temperature=0.15, max_new_tokens=60,
    )
    if obj is None:
        # fallback: use the first member's title as the label
        return members[0].title[:40]
    return obj.name.strip()


@dataclass
class CategorizedIdea:
    conversation_id: int
    canonical_id: int
    title: str
    description: str
    category_id: int
    category_name: str
    embedding_index: int      # row in the stacked embedding matrix


def categorize(canon_all: list[CanonicalRow],
               min_cluster_size: int = 4,
               return_embeddings: bool = True,
               ) -> tuple[list[CategorizedIdea], np.ndarray | None, dict]:
    """Cluster + label all canonical ideas across the corpus.

    Returns (categorized list, embedding matrix or None, summary dict).
    """
    if len(canon_all) == 0:
        return [], None, {'n_ideas': 0, 'n_clusters': 0}

    model = _get_embedder()
    texts = [f'{c.title}: {c.description}' for c in canon_all]
    V = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    V = np.asarray(V, dtype='float32')

    n = len(V)
    if n < min_cluster_size:
        print(f'[agent4] only {n} ideas, skipping clustering (below min_cluster_size={min_cluster_size})')
        labels = np.full(n, -1, dtype=int)
    else:
        # cosine distance via 1 - dot product on L2-normalized vectors
        clf = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='cosine',
            cluster_selection_method='eom',
            allow_single_cluster=False,
        )
        labels = clf.fit_predict(V)

    # per-cluster labeling
    unique = sorted(set(int(l) for l in labels if l >= 0))
    cluster_names = {-1: 'unclustered'}
    for cid in unique:
        members = [canon_all[i] for i in range(n) if labels[i] == cid]
        cluster_names[cid] = _label_cluster(members)

    cats: list[CategorizedIdea] = []
    for i, c in enumerate(canon_all):
        lb = int(labels[i])
        cats.append(CategorizedIdea(
            conversation_id=c.conversation_id,
            canonical_id=c.canonical_id,
            title=c.title,
            description=c.description,
            category_id=lb,
            category_name=cluster_names[lb],
            embedding_index=i,
        ))

    summary = {
        'n_ideas': n,
        'n_clusters': len(unique),
        'n_unclustered': int(np.sum(labels < 0)),
        'cluster_sizes': {cluster_names[c]: int(np.sum(labels == c)) for c in unique},
    }
    return cats, (V if return_embeddings else None), summary
