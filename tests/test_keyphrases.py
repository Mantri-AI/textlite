from __future__ import annotations

import numpy as np

from textlite import extract_keyphrases


class DeterministicEmbedder:
    def __init__(self) -> None:
        self.vectors = {
            "quantum": np.array([0.95, 0.05, 0.0]),
            "entanglement": np.array([0.94, 0.06, 0.0]),
            "research": np.array([0.8, 0.2, 0.0]),
            "quantum entanglement": np.array([0.99, 0.01, 0.0]),
            "entanglement research": np.array([0.96, 0.04, 0.0]),
            "quantum entanglement research": np.array([1.0, 0.0, 0.0]),
            "battery": np.array([0.05, 0.95, 0.0]),
            "chemistry": np.array([0.08, 0.92, 0.0]),
        }

    def encode(self, texts):
        out = []
        for text in texts:
            key = text.strip().lower()
            out.append(self.vectors.get(key, np.array([0.0, 0.0, 0.0])))
        return np.vstack(out)


def test_extract_keyphrases_prefers_semantic_core_phrase():
    text = (
        "Quantum entanglement research is accelerating. "
        "Leading labs share quantum entanglement research findings globally. "
        "Battery chemistry costs are also discussed."
    )

    keyphrases = extract_keyphrases(
        text,
        top_k=6,
        min_ngram=1,
        max_ngram=3,
        embedder=DeterministicEmbedder(),
    )

    phrases = [item.text for item in keyphrases]
    assert "quantum entanglement research" in phrases
    assert keyphrases[0].score >= keyphrases[-1].score