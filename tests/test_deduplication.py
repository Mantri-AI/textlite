from textlite import deduplicate_texts


def test_deduplicate_texts_merges_exact_and_semantic_duplicates(fake_embedder):
    result = deduplicate_texts(
        [
            "Apple releases earnings tomorrow.",
            "Apple releases earnings tomorrow!",
            "apple releases earnings tomorrow.",
            "Microsoft expands Azure AI offerings.",
        ],
        threshold=0.95,
        embedder=fake_embedder,
    )

    assert len(result.unique_texts) == 2
    assert "Microsoft expands Azure AI offerings." in result.unique_texts
    assert any(len(group) >= 3 for group in result.groups)
