"""Train the small CPU-safe RQ-VAE and export an item Semantic ID mapping."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from llm4rec.data.preprocess import enforce_k_core, load_metadata, load_reviews
from llm4rec.semantic_ids.rqvae import (
    RQConfig,
    build_hashed_text_features,
    encode_semantic_ids,
    save_rqvae_checkpoint,
    train_rqvae,
)
from llm4rec.utils import load_yaml, project_root, resolve_project_path, seed_everything


def train(config_path: Path) -> dict:
    cfg = load_yaml(config_path)
    root = project_root()
    seed_everything(int(cfg.get("seed", 42)))
    raw_dir = resolve_project_path(cfg["raw_dir"], root)
    output_dir = resolve_project_path(cfg["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    category = str(cfg["category"])
    reviews = enforce_k_core(
        load_reviews(raw_dir / f"{category}_5.json.gz"),
        int(cfg.get("min_user_interactions", 5)),
        int(cfg.get("min_item_interactions", 5)),
    )
    item_meta = load_metadata(
        raw_dir / f"meta_{category}.json.gz",
        set(reviews["item_id"]),
        int(cfg.get("title_max_chars", 64)),
    )
    item_ids, features = build_hashed_text_features(
        item_meta,
        feature_dim=int(cfg.get("feature_dim", 256)),
        seed=int(cfg.get("feature_seed", 17)),
    )
    max_items = cfg.get("max_items")
    if max_items:
        item_ids = item_ids[: int(max_items)]
        features = features[: int(max_items)]
        item_meta = {item_id: item_meta[item_id] for item_id in item_ids}
    rq_cfg = RQConfig(
        feature_dim=int(cfg.get("feature_dim", 256)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        latent_dim=int(cfg.get("latent_dim", 32)),
        num_levels=int(cfg.get("num_levels", 3)),
        codebook_size=int(cfg.get("codebook_size", 16)),
        epochs=int(cfg.get("epochs", 80)),
        batch_size=min(128, int(cfg.get("batch_size", 32))),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        commitment_weight=float(cfg.get("commitment_weight", 0.25)),
        codebook_weight=float(cfg.get("codebook_weight", 1.0)),
        reconstruction_weight=float(cfg.get("reconstruction_weight", 1.0)),
        collision_refresh_attempts=int(cfg.get("collision_refresh_attempts", 4)),
        collision_suffix_bytes=int(cfg.get("collision_suffix_bytes", 4)),
        seed=int(cfg.get("seed", 42)),
        device=str(cfg.get("device", "cpu")),
        max_items=int(max_items) if max_items else None,
    )
    model, stats = train_rqvae(features, rq_cfg)
    item_to_sid, sid_to_item = encode_semantic_ids(model, item_ids, features)
    base_codes = [semantic_id.split("X", 1)[0] for semantic_id in item_to_sid.values()]
    collision_items = sum(count - 1 for count in Counter(base_codes).values())
    collision_groups = sum(count > 1 for count in Counter(base_codes).values())
    mapping = {
        "category": category,
        "identifier_mode": "semantic_id",
        "feature_backend": "hashing_tfidf",
        "num_levels": rq_cfg.num_levels,
        "codebook_size": rq_cfg.codebook_size,
        "item_to_sid": item_to_sid,
        "sid_to_item": sid_to_item,
        "stats": {
            **stats,
            "num_items": len(item_ids),
            "unique_base_codes": len(set(base_codes)),
            "collision_items": collision_items,
            "collision_groups": collision_groups,
            "collision_resolution": "stable_item_hash_suffix",
            "resolved_unique_semantic_ids": len(set(item_to_sid.values())),
        },
    }
    mapping_path = resolve_project_path(cfg.get("mapping_file", str(output_dir / "semantic_ids.json")), root)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    save_rqvae_checkpoint(output_dir / "rqvae.pt", model, rq_cfg, item_to_sid, sid_to_item, mapping["stats"])
    (output_dir / "rqvae_stats.json").write_text(json.dumps(mapping["stats"], indent=2), encoding="utf-8")
    print(json.dumps({"mapping_file": str(mapping_path), **mapping["stats"]}, indent=2))
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rqvae.yaml")
    args = parser.parse_args()
    train(resolve_project_path(args.config, project_root()))


if __name__ == "__main__":
    main()
