from textlite import compress_contexts
from textlite.embeddings import TokenCounter


def test_compress_contexts_stays_within_budget_and_keeps_relevant_content(fake_embedder):
    paragraphs = [
        "Retrieval augmented generation combines search with generation.",
        "RAG systems use retrieval to ground answers with documents.",
        "Vector databases speed up semantic retrieval over embeddings.",
        "Gardening tools are on sale this weekend.",
    ]
    token_counter = TokenCounter()

    compressed = compress_contexts(
        paragraphs,
        num_tokens=20,
        dedup_threshold=0.9,
        embedder=fake_embedder,
        token_counter=token_counter,
    )

    assert compressed
    assert sum(token_counter.count_tokens(item) for item in compressed) <= 20
    joined = " ".join(compressed)
    assert "retrieval" in joined.lower() or "generation" in joined.lower()
