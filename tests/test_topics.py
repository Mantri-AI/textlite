from __future__ import annotations

import numpy as np

from textlite import assign_entities_to_topics, extract_topics


class TopicEmbedder:
    def __init__(self) -> None:
        self.vectors = {
            "solar": np.array([1.0, 0.0, 0.0]),
            "energy": np.array([0.95, 0.05, 0.0]),
            "solar energy": np.array([0.99, 0.01, 0.0]),
            "panel": np.array([0.92, 0.08, 0.0]),
            "solar panel": np.array([0.97, 0.03, 0.0]),
            "ai": np.array([0.0, 1.0, 0.0]),
            "model": np.array([0.05, 0.95, 0.0]),
            "ai model": np.array([0.02, 0.98, 0.0]),
            "language": np.array([0.1, 0.9, 0.0]),
            "language model": np.array([0.04, 0.96, 0.0]),
        }

    def encode(self, texts):
        vectors = []
        for text in texts:
            key = text.strip().lower()
            vectors.append(self.vectors.get(key, np.array([0.0, 0.0, 0.0])))
        return np.vstack(vectors)


def test_extract_topics_clusters_semantic_phrases():
    texts = [
        "Solar energy projects rely on efficient solar panel design.",
        "AI model quality depends on language model alignment.",
    ]

    topics = extract_topics(
        texts,
        top_k_keyphrases=16,
        max_topics=5,
        min_topic_size=1,
        similarity_threshold=0.8,
        embedder=TopicEmbedder(),
    )

    assert topics
    labels = [topic.label for topic in topics]
    assert any("solar" in label for label in labels)
    assert any("model" in label or "ai" in label for label in labels)


def test_assign_entities_to_topics_matches_by_phrase_overlap():
    texts = [
        "Solar energy projects rely on efficient solar panel design.",
        "AI model quality depends on language model alignment.",
    ]
    topics = extract_topics(
        texts,
        top_k_keyphrases=16,
        max_topics=5,
        min_topic_size=1,
        similarity_threshold=0.8,
        embedder=TopicEmbedder(),
    )
    entities = [
        {"id": "e1", "name": "Solar Energy", "weight": 8, "importance_score": 0.9},
        {"id": "e2", "name": "Language Model", "weight": 7, "importance_score": 0.85},
    ]

    enriched = assign_entities_to_topics(topics, entities)
    assert enriched
    assert any(item.get("entities") for item in enriched)