import torch

from llm4rec.semantic_ids.global_retriever import (
    RetrieverConfig,
    SemanticCatalog,
    SemanticSequenceRetriever,
    ranking_metrics,
    semantic_id_to_codes,
)


def test_semantic_catalog_preserves_collision_bucket_and_prefixes():
    mapping = {"A": "S000102", "B": "S000102XABCDEF01", "C": "S010203"}
    catalog = SemanticCatalog(mapping, num_levels=3, codebook_size=4)
    assert catalog.size == 3
    assert catalog.unique_paths == 2
    assert catalog.allowed(()) == (0, 1)
    assert catalog.allowed((0,)) == (1,)
    assert set(catalog.expand((0, 1, 2))) == {"S000102", "S000102XABCDEF01"}
    assert semantic_id_to_codes("S000102XABCDEF01", 3) == (0, 1, 2)


def test_random_retriever_only_returns_valid_full_catalog_ids():
    torch.manual_seed(7)
    cfg = RetrieverConfig(
        num_levels=2,
        codebook_size=4,
        embedding_dim=8,
        hidden_dim=12,
        max_history_items=4,
        beam_width=8,
        top_k=3,
    )
    model = SemanticSequenceRetriever(cfg)
    catalog = SemanticCatalog({"A": "S0001", "B": "S0102", "C": "S0303"}, 2, 4)
    histories = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)
    rankings = model.retrieve(histories, torch.tensor([2]), catalog, beam_width=8, top_k=3)
    assert len(rankings[0]) == 3
    assert set(rankings[0]) == set(catalog.sid_to_item)
    assert ranking_metrics(rankings, [rankings[0][0]])["HR@1"] == 1.0
