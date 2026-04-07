from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .embeddings import SentenceTransformerEmbedder
from .types import Embedder
from .utils import cosine_similarity_matrix, normalize_text


@dataclass(slots=True)
class DeduplicationResult:
    original_texts: List[str]
    unique_texts: List[str]
    groups: List[List[str]]
    selected_indices: List[int]



def _select_representative(group_indices: List[int], texts: Sequence[str], similarity: np.ndarray) -> int:
    if len(group_indices) == 1:
        return group_indices[0]
    submatrix = similarity[np.ix_(group_indices, group_indices)]
    centrality = submatrix.mean(axis=1)
    best_score = None
    best_index = None
    for local_idx, global_idx in enumerate(group_indices):
        score = float(centrality[local_idx])
        length_bonus = len(texts[global_idx]) * 1e-6
        composite = score + length_bonus
        if best_score is None or composite > best_score:
            best_score = composite
            best_index = global_idx
    assert best_index is not None
    return best_index



def deduplicate_texts(
    texts: Sequence[str],
    *,
    threshold: float = 0.9,
    embedder: Embedder | None = None,
) -> DeduplicationResult:
    """Deduplicate words, phrases, sentences, or paragraphs.

    The function first merges exact normalized duplicates, then performs semantic
    grouping using Sentence Transformer embeddings and a cosine similarity
    threshold.
    """

    cleaned = [text.strip() for text in texts if text and text.strip()]
    if not cleaned:
        return DeduplicationResult([], [], [], [])

    normalized_groups: dict[str, List[int]] = {}
    for idx, text in enumerate(cleaned):
        normalized_groups.setdefault(normalize_text(text), []).append(idx)

    exact_representatives = [indices[0] for indices in normalized_groups.values()]
    reduced_texts = [cleaned[idx] for idx in exact_representatives]

    if len(reduced_texts) == 1:
        return DeduplicationResult(cleaned, [reduced_texts[0]], [[cleaned[i] for i in normalized_groups[next(iter(normalized_groups))]]], [0])

    embedder = embedder or SentenceTransformerEmbedder()
    embeddings = embedder.encode(reduced_texts)
    similarity = cosine_similarity_matrix(embeddings)

    visited = set()
    semantic_groups: List[List[int]] = []
    for idx in range(len(reduced_texts)):
        if idx in visited:
            continue
        current_group = [idx]
        visited.add(idx)
        for other_idx in range(idx + 1, len(reduced_texts)):
            if other_idx in visited:
                continue
            if similarity[idx, other_idx] >= threshold:
                current_group.append(other_idx)
                visited.add(other_idx)
        semantic_groups.append(current_group)

    selected_reduced_indices = [
        _select_representative(group_indices, reduced_texts, similarity)
        for group_indices in semantic_groups
    ]

    selected_indices = [exact_representatives[i] for i in selected_reduced_indices]
    unique_texts = [cleaned[i] for i in selected_indices]

    groups: List[List[str]] = []
    for group in semantic_groups:
        texts_in_group: List[str] = []
        for reduced_idx in group:
            normalized = normalize_text(reduced_texts[reduced_idx])
            for original_idx in normalized_groups[normalized]:
                texts_in_group.append(cleaned[original_idx])
        groups.append(texts_in_group)

    return DeduplicationResult(cleaned, unique_texts, groups, selected_indices)
