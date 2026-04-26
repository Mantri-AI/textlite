from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Iterable, List, Sequence

import numpy as np

from .embeddings import SentenceTransformerEmbedder
from .types import Embedder
from .utils import cosine_similarity_matrix, cosine_similarity_to_vector


_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")

# Keep this lightweight and dependency-free.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "will", "with", "into", "about", "across", "over",
    "under", "between", "while", "than", "then", "also", "can", "could", "should",
}


@dataclass(slots=True)
class Keyphrase:
    text: str
    score: float
    occurrences: int


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text)]


def _valid_token(token: str) -> bool:
    if len(token) < 3:
        return False
    if token in _STOPWORDS:
        return False
    return True


def _candidate_ngrams(tokens: Sequence[str], min_ngram: int, max_ngram: int) -> Iterable[str]:
    size = len(tokens)
    for n in range(min_ngram, max_ngram + 1):
        if n <= 0 or n > size:
            continue
        for idx in range(0, size - n + 1):
            phrase_tokens = tokens[idx: idx + n]
            if not all(_valid_token(t) for t in phrase_tokens):
                continue
            phrase = " ".join(phrase_tokens)
            yield phrase


def _collect_candidates(
    texts: Sequence[str],
    *,
    min_ngram: int,
    max_ngram: int,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = _tokenize(text)
        for phrase in _candidate_ngrams(tokens, min_ngram=min_ngram, max_ngram=max_ngram):
            counter[phrase] += 1
    return counter


def _mmr_select(
    base_scores: np.ndarray,
    pairwise_similarity: np.ndarray,
    *,
    top_k: int,
    diversity: float,
) -> List[int]:
    remaining = set(range(len(base_scores)))
    selected: List[int] = []

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = None
        for idx in remaining:
            if not selected:
                score = float(base_scores[idx])
            else:
                redundancy = max(float(pairwise_similarity[idx, s]) for s in selected)
                score = (1.0 - diversity) * float(base_scores[idx]) - diversity * redundancy
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        assert best_idx is not None
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def extract_keyphrases(
    texts: Sequence[str] | str,
    *,
    top_k: int = 20,
    min_occurrences: int = 1,
    min_ngram: int = 1,
    max_ngram: int = 4,
    diversity: float = 0.15,
    embedder: Embedder | None = None,
) -> List[Keyphrase]:
    """Extract embedding-ranked keyphrases from one or more texts."""

    if isinstance(texts, str):
        corpus = [texts]
    else:
        corpus = [t.strip() for t in texts if t and t.strip()]

    if not corpus or top_k <= 0:
        return []

    candidate_counts = _collect_candidates(corpus, min_ngram=min_ngram, max_ngram=max_ngram)
    if not candidate_counts:
        return []

    candidates = [
        phrase for phrase, count in candidate_counts.items()
        if count >= min_occurrences
    ]
    if not candidates:
        return []

    embedder = embedder or SentenceTransformerEmbedder()
    candidate_embeddings = embedder.encode(candidates)
    text_embeddings = embedder.encode(corpus)
    centroid = text_embeddings.mean(axis=0)

    semantic_scores = cosine_similarity_to_vector(candidate_embeddings, centroid)

    max_count = max(candidate_counts[p] for p in candidates)
    freq_scores = np.asarray(
        [math.log1p(candidate_counts[p]) / math.log1p(max_count) for p in candidates],
        dtype=float,
    )
    ngram_bonus = np.asarray(
        [min(max(len(phrase.split()) - 1, 0), 2) * 0.08 for phrase in candidates],
        dtype=float,
    )
    base_scores = 0.72 * semantic_scores + 0.18 * freq_scores + 0.10 * ngram_bonus

    pairwise = cosine_similarity_matrix(candidate_embeddings)
    selected = _mmr_select(
        base_scores,
        pairwise,
        top_k=min(top_k, len(candidates)),
        diversity=max(0.0, min(1.0, diversity)),
    )

    ranked = [
        Keyphrase(
            text=candidates[idx],
            score=float(base_scores[idx]),
            occurrences=int(candidate_counts[candidates[idx]]),
        )
        for idx in selected
    ]
    ranked.sort(key=lambda kp: kp.score, reverse=True)
    return ranked


def keyphrases_to_tags(keyphrases: Sequence[Keyphrase], source: str = "textlite_keyphrases") -> List[dict]:
    """Convert keyphrases to annotation-style tags."""
    tags: List[dict] = []
    for item in keyphrases:
        tags.append(
            {
                "text": item.text,
                "type": "KEYPHRASE",
                "source": source,
                "metadata": {
                    "score": round(float(item.score), 4),
                    "occurrences": int(item.occurrences),
                },
            }
        )
    return tags