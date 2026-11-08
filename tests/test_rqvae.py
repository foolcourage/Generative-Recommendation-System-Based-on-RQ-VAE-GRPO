import numpy as np

from llm4rec.semantic_ids.rqvae import (
    RQConfig,
    build_hashed_text_features,
    encode_semantic_ids,
    resolve_base_code_collisions,
    train_rqvae,
)


def test_rqvae_cpu_codes_are_unique_and_deterministic():
    metadata = {
        "A": {"title": "red face cream", "category": "Beauty"},
        "B": {"title": "blue face cream", "category": "Beauty"},
        "C": {"title": "hair brush", "category": "Hair"},
    }
    ids, features = build_hashed_text_features(metadata, feature_dim=32)
    cfg = RQConfig(
        feature_dim=32,
        hidden_dim=16,
        latent_dim=8,
        num_levels=2,
        codebook_size=4,
        epochs=2,
        batch_size=2,
        device="cpu",
    )
    model, stats = train_rqvae(features, cfg)
    item_to_sid, sid_to_item = encode_semantic_ids(model, ids, features)
    assert len(item_to_sid) == 3
    assert len(sid_to_item) == 3
    assert set(item_to_sid.values()) == set(sid_to_item)
    assert all(value.startswith("S") for value in item_to_sid.values())
    assert stats["loss"] >= 0.0
    assert len(stats["codebook_utilization_per_level"]) == cfg.num_levels
    assert stats["collision_refresh_attempts_requested"] == cfg.collision_refresh_attempts
    assert 1 <= stats["collision_refresh_attempts_executed"] <= cfg.collision_refresh_attempts


def test_raw_code_collision_uses_stable_item_hash_suffix():
    first, reverse = resolve_base_code_collisions({"A": "0000", "B": "0000"})
    second, _ = resolve_base_code_collisions({"B": "0000", "A": "0000", "C": "0101"})
    assert len(set(first.values())) == 2
    assert all("X" in sid for sid in first.values())
    assert first["A"] == second["A"]
    assert first["B"] == second["B"]
    assert reverse[first["A"]] == "A"
