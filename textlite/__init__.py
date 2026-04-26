"""Lightweight semantic text utilities."""

from .clustering import EntityCluster, cluster_entities
from .compression import compress_contexts
from .deduplication import DeduplicationResult, deduplicate_texts
from .keyphrases import Keyphrase, extract_keyphrases, keyphrases_to_tags
from .summarization import summarize_paragraphs, summarize_text
from .topics import Topic, assign_entities_to_topics, extract_topics

__all__ = [
    "DeduplicationResult",
    "EntityCluster",
    "Keyphrase",
    "Topic",
    "assign_entities_to_topics",
    "cluster_entities",
    "compress_contexts",
    "deduplicate_texts",
    "extract_keyphrases",
    "extract_topics",
    "keyphrases_to_tags",
    "summarize_paragraphs",
    "summarize_text",
]
