"""Lightweight semantic text utilities."""

from .clustering import EntityCluster, cluster_entities
from .compression import compress_contexts
from .deduplication import DeduplicationResult, deduplicate_texts
from .summarization import summarize_paragraphs, summarize_text

__all__ = [
    "DeduplicationResult",
    "EntityCluster",
    "cluster_entities",
    "compress_contexts",
    "deduplicate_texts",
    "summarize_paragraphs",
    "summarize_text",
]
