from textlite import cluster_entities


def test_cluster_entities_groups_semantic_neighbors(fake_embedder):
    clusters = cluster_entities(
        [
            "Apple",
            "Apple Inc",
            "Apple Incorporated",
            "Microsoft",
            "Microsoft Corporation",
            "Neural Search",
            "Semantic Retrieval",
        ],
        similarity_threshold=0.85,
        dedup_threshold=0.95,
        embedder=fake_embedder,
    )

    member_sets = [set(cluster.members) for cluster in clusters]
    assert {"Apple", "Apple Inc", "Apple Incorporated"} in member_sets
    assert {"Microsoft", "Microsoft Corporation"} in member_sets
    assert {"Neural Search", "Semantic Retrieval"} in member_sets
