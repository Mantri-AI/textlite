from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .deduplication import deduplicate_texts
from .embeddings import SentenceTransformerEmbedder
from .types import Embedder
from .utils import cosine_similarity_matrix


@dataclass(slots=True)
class EntityCluster:
    label: str
    members: List[str]
    representative: str
    cohesion: float



def _cluster_label(members: Sequence[str], representative: str) -> str:
    if representative:
        return representative
    longest = max(members, key=len)
    return longest



def cluster_entities(
    entities: Sequence[str],
    *,
    similarity_threshold: float = 0.72,
    dedup_threshold: float = 0.92,
    embedder: Embedder | None = None,
) -> List[EntityCluster]:
    """Cluster semantically similar entities into high-quality groups."""

    deduped = deduplicate_texts(entities, threshold=dedup_threshold, embedder=embedder)
    unique_entities = deduped.unique_texts
    dedup_groups = deduped.groups
    if not unique_entities:
        return []
    if len(unique_entities) == 1:
        entity = unique_entities[0]
        return [EntityCluster(label=entity, members=[entity], representative=entity, cohesion=1.0)]

    embedder = embedder or SentenceTransformerEmbedder()
    embeddings = embedder.encode(unique_entities)
    similarity = cosine_similarity_matrix(embeddings)
    distance = 1.0 - similarity

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=1.0 - similarity_threshold,
        n_clusters=None,
    )
    labels = clustering.fit_predict(distance)

    clusters: List[EntityCluster] = []
    for label_id in sorted(set(labels)):
        member_indices = [idx for idx, value in enumerate(labels) if value == label_id]
        collapsed_members = [unique_entities[idx] for idx in member_indices]
        members = []
        for idx in member_indices:
            members.extend(dedup_groups[idx])
        submatrix = similarity[np.ix_(member_indices, member_indices)]
        centrality = submatrix.mean(axis=1)
        best_local_idx = int(np.argmax(centrality))
        representative = collapsed_members[best_local_idx]
        cohesion = float(submatrix.mean()) if len(members) > 1 else 1.0
        clusters.append(
            EntityCluster(
                label=_cluster_label(members, representative),
                members=sorted(members),
                representative=representative,
                cohesion=cohesion,
            )
        )

    clusters.sort(key=lambda cluster: (-len(cluster.members), -cluster.cohesion, cluster.label.lower()))
    return clusters
