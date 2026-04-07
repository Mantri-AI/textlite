from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .deduplication import deduplicate_texts
from .embeddings import SentenceTransformerEmbedder, TokenCounter
from .summarization import summarize_text
from .types import Embedder
from .utils import cosine_similarity_matrix, cosine_similarity_to_vector, flatten_paragraphs



def _prepare_candidates(
    paragraphs: Sequence[str],
    *,
    paragraph_summary_max_sentences: int,
    long_paragraph_token_threshold: int,
    embedder: Embedder | None,
    token_counter: TokenCounter,
) -> List[str]:
    candidates: List[str] = []
    for paragraph in flatten_paragraphs(paragraphs):
        tokens = token_counter.count_tokens(paragraph)
        if tokens > long_paragraph_token_threshold:
            candidates.append(
                summarize_text(
                    paragraph,
                    max_sentences=paragraph_summary_max_sentences,
                    max_tokens=max(long_paragraph_token_threshold // 2, 1),
                    embedder=embedder,
                    token_counter=token_counter,
                )
            )
        else:
            candidates.append(paragraph)
    return [candidate for candidate in candidates if candidate]



def compress_contexts(
    paragraphs: Sequence[str],
    num_tokens: int,
    *,
    dedup_threshold: float = 0.9,
    similarity_budget_threshold: float = 0.8,
    paragraph_summary_max_sentences: int = 2,
    long_paragraph_token_threshold: int = 120,
    embedder: Embedder | None = None,
    token_counter: TokenCounter | None = None,
) -> List[str]:
    """Compress many paragraphs into a representative subset under a token budget.

    Strategy:
    1. Clean and optionally summarize very long paragraphs.
    2. Deduplicate semantically redundant candidates.
    3. Score by similarity to corpus centroid.
    4. Select with an MMR-like coverage/diversity tradeoff under the token budget.
    """

    if num_tokens <= 0:
        return []

    token_counter = token_counter or TokenCounter()
    embedder = embedder or SentenceTransformerEmbedder()

    candidates = _prepare_candidates(
        paragraphs,
        paragraph_summary_max_sentences=paragraph_summary_max_sentences,
        long_paragraph_token_threshold=long_paragraph_token_threshold,
        embedder=embedder,
        token_counter=token_counter,
    )
    if not candidates:
        return []

    deduped = deduplicate_texts(candidates, threshold=dedup_threshold, embedder=embedder)
    unique_candidates = deduped.unique_texts
    if not unique_candidates:
        return []

    embeddings = embedder.encode(unique_candidates)
    pairwise_similarity = cosine_similarity_matrix(embeddings)
    centroid = embeddings.mean(axis=0)
    importance = cosine_similarity_to_vector(embeddings, centroid)

    selected: List[int] = []
    remaining = set(range(len(unique_candidates)))
    token_total = 0

    while remaining:
        best_idx = None
        best_score = None
        for idx in remaining:
            candidate = unique_candidates[idx]
            candidate_tokens = token_counter.count_tokens(candidate)
            if selected and token_total + candidate_tokens > num_tokens:
                continue
            redundancy = max((pairwise_similarity[idx, chosen] for chosen in selected), default=0.0)
            score = 0.7 * importance[idx] - 0.3 * redundancy
            if redundancy >= similarity_budget_threshold:
                score -= 0.1
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        candidate = unique_candidates[best_idx]
        candidate_tokens = token_counter.count_tokens(candidate)
        if token_total + candidate_tokens > num_tokens and not selected:
            summary = summarize_text(
                candidate,
                max_sentences=1,
                max_tokens=num_tokens,
                embedder=embedder,
                token_counter=token_counter,
            )
            if summary and token_counter.count_tokens(summary) <= num_tokens:
                return [summary]
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        token_total += candidate_tokens

        if token_total >= num_tokens:
            break

    return [unique_candidates[idx] for idx in selected]
