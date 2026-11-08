"""Full-catalog hierarchical retrieval over RQ-VAE Semantic-ID paths.

The retriever deliberately separates global retrieval from the Qwen reranker.
It encodes a user's Semantic-ID history once, autoregressively predicts residual
code levels, and uses a catalog trie to prevent generation of nonexistent paths.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


@dataclass
class RetrieverConfig:
    num_levels: int = 3
    codebook_size: int = 16
    embedding_dim: int = 48
    hidden_dim: int = 96
    max_history_items: int = 20
    epochs: int = 120
    batch_size: int = 32
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    beam_width: int = 32
    top_k: int = 20
    eval_every: int = 5
    patience: int = 8
    seed: int = 42
    device: str = "cpu"


def semantic_id_to_codes(semantic_id: str, num_levels: int) -> tuple[int, ...]:
    """Parse the residual path and ignore a collision-resolution suffix."""

    text = str(semantic_id).strip()
    if not text.startswith("S"):
        raise ValueError(f"Semantic ID must start with S: {semantic_id}")
    base = text[1:].split("X", 1)[0]
    expected = int(num_levels) * 2
    if len(base) != expected or not base.isdigit():
        raise ValueError(f"Expected {expected} base-code digits in {semantic_id}")
    return tuple(int(base[offset : offset + 2]) for offset in range(0, expected, 2))


class SemanticCatalog:
    """Valid residual-code paths and their collision buckets."""

    def __init__(self, item_to_sid: dict[str, str], num_levels: int, codebook_size: int):
        self.num_levels = int(num_levels)
        self.codebook_size = int(codebook_size)
        self.path_to_sids: dict[tuple[int, ...], list[str]] = {}
        self.sid_to_item = {str(sid): str(item) for item, sid in item_to_sid.items()}
        for item_id, semantic_id in sorted(item_to_sid.items()):
            path = semantic_id_to_codes(semantic_id, self.num_levels)
            if any(code < 0 or code >= self.codebook_size for code in path):
                raise ValueError(f"Code outside configured codebook: {semantic_id}")
            self.path_to_sids.setdefault(path, []).append(str(semantic_id))
        self.prefix_children: dict[tuple[int, ...], tuple[int, ...]] = {}
        mutable: dict[tuple[int, ...], set[int]] = {}
        for path in self.path_to_sids:
            for level, code in enumerate(path):
                mutable.setdefault(path[:level], set()).add(code)
        self.prefix_children = {prefix: tuple(sorted(children)) for prefix, children in mutable.items()}

    @property
    def size(self) -> int:
        return sum(len(value) for value in self.path_to_sids.values())

    @property
    def unique_paths(self) -> int:
        return len(self.path_to_sids)

    def allowed(self, prefix: Sequence[int]) -> tuple[int, ...]:
        return self.prefix_children.get(tuple(int(value) for value in prefix), ())

    def expand(self, path: Sequence[int]) -> list[str]:
        return list(self.path_to_sids.get(tuple(int(value) for value in path), ()))


class SequenceRecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], num_levels: int, max_history_items: int):
        self.samples: list[tuple[list[tuple[int, ...]], tuple[int, ...], str]] = []
        for record in records:
            history_ids = [str(item["item_id"]) for item in record.get("history", [])]
            history_ids = history_ids[-max(1, int(max_history_items)) :]
            if not history_ids:
                continue
            target_sid = str(record.get("target_recommendation_id", record.get("target_item_id", "")))
            history = [semantic_id_to_codes(value, num_levels) for value in history_ids]
            target = semantic_id_to_codes(target_sid, num_levels)
            self.samples.append((history, target, target_sid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[list[tuple[int, ...]], tuple[int, ...], str]:
        return self.samples[index]


def collate_sequences(
    samples: list[tuple[list[tuple[int, ...]], tuple[int, ...], str]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("empty sequence batch")
    levels = len(samples[0][1])
    lengths = torch.tensor([len(history) for history, _, _ in samples], dtype=torch.long)
    max_length = int(lengths.max().item())
    histories = torch.zeros((len(samples), max_length, levels), dtype=torch.long)
    targets = torch.zeros((len(samples), levels), dtype=torch.long)
    target_sids: list[str] = []
    for row, (history, target, target_sid) in enumerate(samples):
        histories[row, : len(history)] = torch.tensor(history, dtype=torch.long)
        targets[row] = torch.tensor(target, dtype=torch.long)
        target_sids.append(target_sid)
    return {"histories": histories, "lengths": lengths, "targets": targets, "target_sids": target_sids}


class SemanticSequenceRetriever(nn.Module):
    """GRU history encoder with an autoregressive residual-code decoder."""

    def __init__(self, cfg: RetrieverConfig):
        super().__init__()
        self.cfg = cfg
        self.code_embeddings = nn.ModuleList(
            nn.Embedding(cfg.codebook_size, cfg.embedding_dim) for _ in range(cfg.num_levels)
        )
        self.position_embedding = nn.Embedding(cfg.max_history_items, cfg.embedding_dim)
        self.history_norm = nn.LayerNorm(cfg.embedding_dim)
        self.encoder = nn.GRU(cfg.embedding_dim, cfg.hidden_dim, batch_first=True)
        self.level_embedding = nn.Embedding(cfg.num_levels, cfg.embedding_dim)
        self.decoder = nn.GRUCell(cfg.embedding_dim, cfg.hidden_dim)
        self.output_heads = nn.ModuleList(nn.Linear(cfg.hidden_dim, cfg.codebook_size) for _ in range(cfg.num_levels))

    def embed_items(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = torch.zeros((*codes.shape[:-1], self.cfg.embedding_dim), device=codes.device)
        for level, table in enumerate(self.code_embeddings):
            embedding = embedding + table(codes[..., level])
        return embedding / math.sqrt(max(1, self.cfg.num_levels))

    def encode_history(self, histories: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(histories.shape[1], device=histories.device)
        embedded = self.embed_items(histories) + self.position_embedding(positions)[None, :, :]
        embedded = self.history_norm(embedded)
        packed = pack_padded_sequence(embedded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        return hidden[-1]

    def forward(self, histories: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
        state = self.encode_history(histories, lengths)
        logits_by_level: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        for level in range(self.cfg.num_levels):
            logits = self.output_heads[level](state)
            logits_by_level.append(logits)
            losses.append(
                F.cross_entropy(logits, targets[:, level], label_smoothing=float(self.cfg.label_smoothing))
            )
            decoder_input = self.code_embeddings[level](targets[:, level]) + self.level_embedding.weight[level]
            state = self.decoder(decoder_input, state)
        return {"loss": torch.stack(losses).mean(), "level_losses": losses, "logits": logits_by_level}

    @torch.no_grad()
    def retrieve(
        self,
        histories: torch.Tensor,
        lengths: torch.Tensor,
        catalog: SemanticCatalog,
        beam_width: int,
        top_k: int,
        collision_prior: dict[str, float] | None = None,
    ) -> list[list[str]]:
        self.eval()
        initial_states = self.encode_history(histories, lengths)
        rankings: list[list[str]] = []
        prior = collision_prior or {}
        for row in range(histories.shape[0]):
            beams: list[tuple[tuple[int, ...], float, torch.Tensor]] = [
                ((), 0.0, initial_states[row])
            ]
            for level in range(self.cfg.num_levels):
                expanded: list[tuple[tuple[int, ...], float, torch.Tensor]] = []
                for prefix, score, state in beams:
                    allowed = catalog.allowed(prefix)
                    if not allowed:
                        continue
                    log_probabilities = F.log_softmax(self.output_heads[level](state), dim=-1)
                    for code in allowed:
                        code_tensor = torch.tensor(code, device=state.device, dtype=torch.long)
                        decoder_input = self.code_embeddings[level](code_tensor) + self.level_embedding.weight[level]
                        next_state = self.decoder(decoder_input.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
                        expanded.append((prefix + (code,), score + float(log_probabilities[code]), next_state))
                expanded.sort(key=lambda value: (-value[1], value[0]))
                beams = expanded[: max(1, int(beam_width))]
            scored_sids: list[tuple[float, str]] = []
            for path, score, _ in beams:
                for semantic_id in catalog.expand(path):
                    # Only breaks ties inside a raw-code collision bucket.
                    collision_score = 1e-6 * math.log1p(max(0.0, float(prior.get(semantic_id, 0.0))))
                    scored_sids.append((score + collision_score, semantic_id))
            scored_sids.sort(key=lambda value: (-value[0], value[1]))
            rankings.append([semantic_id for _, semantic_id in scored_sids[: max(1, int(top_k))]])
        return rankings


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def popularity_prior(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(str(item["item_id"]) for item in record.get("history", []))
        target = record.get("target_recommendation_id")
        if target:
            counts[str(target)] += 1
    return counts


def ranking_metrics(rankings: list[list[str]], targets: list[str], cutoffs: Sequence[int] = (1, 5, 10, 20)) -> dict[str, float]:
    if len(rankings) != len(targets):
        raise ValueError("ranking and target counts differ")
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        hits = 0.0
        ndcg = 0.0
        reciprocal = 0.0
        for ranking, target in zip(rankings, targets):
            try:
                rank = ranking[: int(cutoff)].index(target) + 1
            except ValueError:
                continue
            hits += 1.0
            ndcg += 1.0 / math.log2(rank + 1)
            reciprocal += 1.0 / rank
        denominator = max(1, len(targets))
        metrics[f"HR@{cutoff}"] = hits / denominator
        metrics[f"Recall@{cutoff}"] = hits / denominator
        metrics[f"NDCG@{cutoff}"] = ndcg / denominator
        metrics[f"MRR@{cutoff}"] = reciprocal / denominator
    return metrics


@torch.no_grad()
def evaluate_retriever(
    model: SemanticSequenceRetriever,
    loader: DataLoader,
    catalog: SemanticCatalog,
    cfg: RetrieverConfig,
    collision_prior: dict[str, float] | None = None,
) -> dict[str, float]:
    device = next(model.parameters()).device
    rankings: list[list[str]] = []
    targets: list[str] = []
    started = time.perf_counter()
    for batch in loader:
        batch_rankings = model.retrieve(
            batch["histories"].to(device),
            batch["lengths"].to(device),
            catalog,
            beam_width=cfg.beam_width,
            top_k=max(cfg.top_k, 20),
            collision_prior=collision_prior,
        )
        rankings.extend(batch_rankings)
        targets.extend(batch["target_sids"])
    elapsed = time.perf_counter() - started
    metrics = ranking_metrics(rankings, targets)
    metrics["queries"] = float(len(targets))
    metrics["mean_retrieval_ms_per_query"] = 1000.0 * elapsed / max(1, len(targets))
    metrics["catalog_size"] = float(catalog.size)
    metrics["unique_code_paths"] = float(catalog.unique_paths)
    metrics["beam_width"] = float(cfg.beam_width)
    return metrics


def popularity_ranking_metrics(
    catalog: SemanticCatalog,
    counts: Counter[str],
    targets: list[str],
) -> dict[str, float]:
    ranking = sorted(catalog.sid_to_item, key=lambda sid: (-counts.get(sid, 0), sid))
    return ranking_metrics([ranking for _ in targets], targets)


def config_as_dict(cfg: RetrieverConfig) -> dict[str, Any]:
    return asdict(cfg)
