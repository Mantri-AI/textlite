from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    """Protocol for embedding providers used throughout the package."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a 2D array shaped (n_texts, embedding_dim)."""
