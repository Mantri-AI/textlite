from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .keyphrases import Keyphrase, extract_keyphrases
from .types import Embedder
from .utils import cosine_similarity_matrix


@dataclass(slots=True)
class Topic:
    id: str
    label: str
    score: float
    size: int
    keyphrases: List[str]


def _to_phrase_weights(keyphrases: Sequence[Keyphrase]) -> dict[str, float]:
    return {item.text: float(item.score) for item in keyphrases}


def extract_topics(
    texts: Sequence[str] | str,
    *,
    top_k_keyphrases: int = 80,
    max_topics: int = 20,
    min_topic_size: int = 2,
    similarity_threshold: float = 0.72,
    embedder: Embedder | None = None,
) -> List[Topic]:
    """Extract high-level topics by clustering semantically similar keyphrases."""

    keyphrases = extract_keyphrases(
        texts,
        top_k=top_k_keyphrases,
        min_occurrences=1,
        min_ngram=2,
        max_ngram=4,
        diversity=0.2,
        embedder=embedder,
    )
    if not keyphrases:
        return []

    phrase_texts = [item.text for item in keyphrases]
    phrase_weights = _to_phrase_weights(keyphrases)
    if len(phrase_texts) == 1:
        only = phrase_texts[0]
        return [Topic(id="topic_1", label=only, score=phrase_weights[only], size=1, keyphrases=[only])]

    assert embedder is not None or len(phrase_texts) > 0
    # Reuse the embedder instance used by extract_keyphrases when provided.
    working_embedder = embedder
    if working_embedder is None:
        # extract_keyphrases already created a default embedder internally; we need one here.
        from .embeddings import SentenceTransformerEmbedder
        working_embedder = SentenceTransformerEmbedder()

    embeddings = working_embedder.encode(phrase_texts)
    similarity = cosine_similarity_matrix(embeddings)
    distance = 1.0 - similarity

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - similarity_threshold,
        n_clusters=None,
    )
    labels = clustering.fit_predict(distance)

    grouped_indices: dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        grouped_indices.setdefault(int(label), []).append(idx)

    topics: List[Topic] = []
    topic_counter = 1
    for _, indices in sorted(grouped_indices.items(), key=lambda kv: -len(kv[1])):
        if len(indices) < min_topic_size and len(grouped_indices) > 1:
            continue

        members = [phrase_texts[i] for i in indices]
        members.sort(key=lambda phrase: phrase_weights.get(phrase, 0.0), reverse=True)
        label = members[0]
        score = float(np.mean([phrase_weights.get(phrase, 0.0) for phrase in members]))

        topics.append(
            Topic(
                id=f"topic_{topic_counter}",
                label=label,
                score=score,
                size=len(members),
                keyphrases=members[:12],
            )
        )
        topic_counter += 1

    topics.sort(key=lambda t: (t.score, t.size), reverse=True)
    return topics[:max_topics]


def assign_entities_to_topics(
    topics: Sequence[Topic],
    entities: Sequence[Mapping[str, object]],
    *,
    max_entities_per_topic: int = 12,
) -> List[dict]:
    """Attach relevant entities to each topic using lightweight lexical matching."""
    enriched: List[dict] = []

    normalized_entities = []
    for entity in entities:
        name = str(entity.get("name") or "").strip()
        if not name:
            continue
        normalized_entities.append(
            {
                "id": entity.get("id"),
                "name": name,
                "weight": entity.get("weight", 0),
                "importance_score": entity.get("importance_score", 0),
                "lower": name.lower(),
            }
        )

    for topic in topics:
        topic_terms = {topic.label.lower(), *[kp.lower() for kp in topic.keyphrases]}
        matches = []
        for entity in normalized_entities:
            lower_name = str(entity["lower"])
            if any(term in lower_name or lower_name in term for term in topic_terms):
                matches.append(entity)

        matches.sort(
            key=lambda item: (
                float(item.get("importance_score") or 0.0),
                float(item.get("weight") or 0.0),
            ),
            reverse=True,
        )

        enriched.append(
            {
                "id": topic.id,
                "label": topic.label,
                "score": round(float(topic.score), 4),
                "size": int(topic.size),
                "keyphrases": list(topic.keyphrases),
                "entities": [
                    {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "weight": item.get("weight", 0),
                        "importance_score": item.get("importance_score", 0),
                    }
                    for item in matches[:max_entities_per_topic]
                ],
            }
        )

    return enriched