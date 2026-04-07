from __future__ import annotations

import re
from typing import Iterable, List, Sequence

import numpy as np

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s*([,.;:!?])\s*")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if parts:
        return parts
    return [text]


def flatten_paragraphs(paragraphs: Sequence[str]) -> List[str]:
    return [paragraph.strip() for paragraph in paragraphs if paragraph and paragraph.strip()]


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros((0, 0), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normalized = vectors / safe_norms
    matrix = normalized @ normalized.T
    return np.clip(matrix, -1.0, 1.0)


def cosine_similarity_to_vector(vectors: np.ndarray, vector: np.ndarray) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros((0,), dtype=float)
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0:
        return np.zeros((len(vectors),), dtype=float)
    row_norms = np.linalg.norm(vectors, axis=1)
    row_norms = np.where(row_norms == 0, 1.0, row_norms)
    return np.clip((vectors @ vector) / (row_norms * vector_norm), -1.0, 1.0)


def cheap_token_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text))


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output
