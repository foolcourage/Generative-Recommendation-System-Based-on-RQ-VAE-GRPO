"""Train and evaluate a full-catalog Semantic-ID retriever."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from llm4rec.semantic_ids.global_retriever import (
    RetrieverConfig,
    SemanticCatalog,
    SemanticSequenceRetriever,
    SequenceRecordDataset,
    collate_sequences,
    evaluate_retriever,
    popularity_prior,
    popularity_ranking_metrics,
    read_jsonl,
)
from llm4rec.semantic_ids.rqvae import load_semantic_mapping
from llm4rec.utils import load_yaml, project_root, resolve_project_path, seed_everything


def _loader(records: list[dict], cfg: RetrieverConfig, shuffle: bool) -> DataLoader:
    dataset = SequenceRecordDataset(records, cfg.num_levels, cfg.max_history_items)
    return DataLoader(
        dataset,
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_sequences,
    )


def train(config_path: Path) -> dict:
    cfg_raw = load_yaml(config_path)
    root = project_root()
    seed = int(cfg_raw.get("seed", 42))
    seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = str(cfg_raw.get("device", "cpu")).lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"

    mapping_path = resolve_project_path(cfg_raw["mapping_file"], root)
    mapping = load_semantic_mapping(mapping_path)
    num_levels = int(cfg_raw.get("num_levels", mapping.get("num_levels", 3)))
    codebook_size = int(cfg_raw.get("codebook_size", mapping.get("codebook_size", 16)))
    cfg = RetrieverConfig(
        num_levels=num_levels,
        codebook_size=codebook_size,
        embedding_dim=int(cfg_raw.get("embedding_dim", 48)),
        hidden_dim=int(cfg_raw.get("hidden_dim", 96)),
        max_history_items=int(cfg_raw.get("max_history_items", 20)),
        epochs=int(cfg_raw.get("epochs", 120)),
        batch_size=int(cfg_raw.get("batch_size", 32)),
        learning_rate=float(cfg_raw.get("learning_rate", 2e-3)),
        weight_decay=float(cfg_raw.get("weight_decay", 1e-4)),
        label_smoothing=float(cfg_raw.get("label_smoothing", 0.05)),
        beam_width=int(cfg_raw.get("beam_width", 32)),
        top_k=int(cfg_raw.get("top_k", 20)),
        eval_every=int(cfg_raw.get("eval_every", 5)),
        patience=int(cfg_raw.get("patience", 8)),
        seed=seed,
        device=device_name,
    )
    data_dir = resolve_project_path(cfg_raw["data_dir"], root)
    output_dir = resolve_project_path(cfg_raw["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_records = read_jsonl(data_dir / "train.jsonl")
    val_records = read_jsonl(data_dir / "val.jsonl")
    test_records = read_jsonl(data_dir / "test.jsonl")
    train_loader = _loader(train_records, cfg, shuffle=True)
    val_loader = _loader(val_records, cfg, shuffle=False)
    test_loader = _loader(test_records, cfg, shuffle=False)
    catalog = SemanticCatalog(mapping["item_to_sid"], num_levels, codebook_size)
    prior_counter = popularity_prior(train_records)
    prior = {sid: float(prior_counter.get(sid, 0)) for sid in catalog.sid_to_item}

    device = torch.device(device_name)
    model = SemanticSequenceRetriever(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, max(1, cfg.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["histories"].to(device),
                batch["lengths"].to(device),
                batch["targets"].to(device),
            )
            outputs["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = len(batch["target_sids"])
            total_count += size
            total_loss += float(outputs["loss"].detach().cpu()) * size
        epoch_record = {"epoch": epoch, "train_loss": total_loss / max(1, total_count)}
        if epoch == 1 or epoch % max(1, cfg.eval_every) == 0 or epoch == cfg.epochs:
            val_metrics = evaluate_retriever(model, val_loader, catalog, cfg, prior)
            epoch_record["val_HR@10"] = val_metrics["HR@10"]
            epoch_record["val_NDCG@10"] = val_metrics["NDCG@10"]
            score = val_metrics["NDCG@10"]
            if score > best_score + 1e-9:
                best_score = score
                best_epoch = epoch
                stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= max(1, cfg.patience):
                    history.append(epoch_record)
                    break
        history.append(epoch_record)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    val_metrics = evaluate_retriever(model, val_loader, catalog, cfg, prior)
    test_metrics = evaluate_retriever(model, test_loader, catalog, cfg, prior)
    test_targets = [sample[2] for sample in SequenceRecordDataset(test_records, cfg.num_levels, cfg.max_history_items).samples]
    popularity_metrics = popularity_ranking_metrics(catalog, prior_counter, test_targets)
    checkpoint = {
        "config": cfg_raw | {"resolved": cfg.__dict__},
        "state_dict": model.state_dict(),
        "item_to_sid": mapping["item_to_sid"],
        "catalog_size": catalog.size,
        "unique_code_paths": catalog.unique_paths,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, output_dir / "semantic_retriever.pt")
    report = {
        "model": "hierarchical_semantic_id_retriever",
        "identifier_mode": "semantic_id",
        "candidate_mode": "full_catalog",
        "prefiltered_candidate_size": None,
        "catalog_size": catalog.size,
        "unique_code_paths": catalog.unique_paths,
        "num_levels": num_levels,
        "codebook_size": codebook_size,
        "beam_width": cfg.beam_width,
        "top_k": cfg.top_k,
        "train_queries": len(train_loader.dataset),
        "val_queries": len(val_loader.dataset),
        "test_queries": len(test_loader.dataset),
        "best_epoch": best_epoch,
        "val": val_metrics,
        "test": test_metrics,
        "popularity_full_catalog_test": popularity_metrics,
        "history": history,
        "note": "Full-catalog prototype on the retained All_Beauty catalog; no 16-item candidate list is supplied to retrieval.",
    }
    (output_dir / "semantic_retriever_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/global_retriever.yaml")
    args = parser.parse_args()
    train(resolve_project_path(args.config, project_root()))


if __name__ == "__main__":
    main()
