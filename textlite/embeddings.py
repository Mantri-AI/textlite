from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .utils import cheap_token_count


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """Lazy SentenceTransformer wrapper to keep imports lightweight."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    _model: object | None = None

    def _load_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required. Install dependencies from "
                    "textlite/requirements.txt"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_model()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(vectors, dtype=float)


@dataclass(slots=True)
class BertTokenizerCounter:
    """Accurate token counting when transformers is available."""

    model_name: str = "bert-base-uncased"
    _tokenizer: object | None = None

    def _load_tokenizer(self) -> object:
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required. Install dependencies from "
                    "textlite/requirements.txt"
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        tokenizer = self._load_tokenizer()
        return len(tokenizer.encode(text, add_special_tokens=False))


@dataclass(slots=True)
class TokenCounter:
    """Best-effort token counting with optional BERT tokenizer."""

    tokenizer_counter: Optional[BertTokenizerCounter] = None

    def count_tokens(self, text: str) -> int:
        if self.tokenizer_counter is None:
            return cheap_token_count(text)
        try:
            return self.tokenizer_counter.count_tokens(text)
        except Exception:
            return cheap_token_count(text)
