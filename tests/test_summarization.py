from textlite import summarize_text


def test_summarize_text_prefers_central_sentences(fake_embedder, token_counter):
    text = (
        "Sentence embeddings help compare meaning across sentences. "
        "They are useful for semantic search and summarization. "
        "The weather is sunny today."
    )

    summary = summarize_text(
        text,
        max_sentences=2,
        embedder=fake_embedder,
        token_counter=token_counter,
    )

    assert "Sentence embeddings help compare meaning across sentences." in summary
    assert "They are useful for semantic search and summarization." in summary
    assert "The weather is sunny today." not in summary
