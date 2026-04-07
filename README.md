# textlite

`textlite` is a lightweight package for semantic clustering, deduplication,
extractive summarization, and context compression using BERT/Sentence Transformer
style models.

## Capabilities

- Entity clustering based on semantic similarity
- Sentence-BERT style extractive summarization
- Deduplication of words, phrases, sentences, or paragraphs
- Context compression under a token budget

## Install

```bash
pip install -r textlite/requirements.txt
pip install -e .
```

## Example

```python
from textlite import cluster_entities, deduplicate_texts, summarize_text, compress_contexts

entities = [
    "large language model",
    "LLM",
    "neural search",
    "semantic retrieval",
]
clusters = cluster_entities(entities)

unique = deduplicate_texts([
    "Apple releases earnings tomorrow.",
    "Apple releases earnings tomorrow!",
    "Microsoft expands Azure AI offerings.",
])

summary = summarize_text(
    "Sentence embeddings help compare meaning across sentences. "
    "They are useful for semantic search and summarization."
)

compressed = compress_contexts([
    "Paragraph 1 ...",
    "Paragraph 2 ...",
], num_tokens=80)
```
