from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from textlite.embeddings import TokenCounter


class FakeEmbedder:
    def __init__(self) -> None:
        self.vectors = {
            "apple": np.array([1.0, 0.0, 0.0]),
            "apple inc": np.array([0.97, 0.03, 0.0]),
            "apple incorporated": np.array([0.96, 0.04, 0.0]),
            "microsoft": np.array([0.0, 1.0, 0.0]),
            "microsoft corporation": np.array([0.03, 0.97, 0.0]),
            "neural search": np.array([0.0, 0.0, 1.0]),
            "semantic retrieval": np.array([0.0, 0.06, 0.94]),
            "llm": np.array([0.7, 0.7, 0.0]),
            "large language model": np.array([0.68, 0.72, 0.0]),
            "apple releases earnings tomorrow.": np.array([1.0, 0.0, 0.0]),
            "apple releases earnings tomorrow!": np.array([0.99, 0.01, 0.0]),
            "microsoft expands azure ai offerings.": np.array([0.0, 1.0, 0.0]),
            "sentence embeddings help compare meaning across sentences.": np.array([1.0, 0.0, 0.0]),
            "they are useful for semantic search and summarization.": np.array([0.8, 0.2, 0.0]),
            "the weather is sunny today.": np.array([0.0, 1.0, 0.0]),
            "sentence embeddings help compare meaning across sentences. they are useful for semantic search and summarization.": np.array([0.95, 0.05, 0.0]),
            "retrieval augmented generation combines search with generation.": np.array([1.0, 0.0, 0.0]),
            "rag systems use retrieval to ground answers with documents.": np.array([0.92, 0.08, 0.0]),
            "vector databases speed up semantic retrieval over embeddings.": np.array([0.0, 1.0, 0.0]),
            "gardening tools are on sale this weekend.": np.array([0.0, 0.0, 1.0]),
            "retrieval augmented generation combines search with generation. rag systems use retrieval to ground answers with documents.": np.array([0.96, 0.04, 0.0]),
        }

    def encode(self, texts):
        encoded = []
        for text in texts:
            key = text.strip().lower()
            if key not in self.vectors:
                seed = sum(ord(ch) for ch in key) % 997
                vec = np.array([
                    (seed % 17) / 17.0,
                    (seed % 31) / 31.0,
                    (seed % 43) / 43.0,
                ])
                self.vectors[key] = vec
            encoded.append(self.vectors[key])
        return np.vstack(encoded)


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def token_counter() -> TokenCounter:
    return TokenCounter()
