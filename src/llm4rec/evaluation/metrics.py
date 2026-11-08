"""Ranking metrics for one-next-item leave-one-out evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def recall_at_k(ranked_item_ids: Iterable[str], ground_truth: str | Iterable[str], k: int = 10) -> float:
    relevant = {str(ground_truth)} if isinstance(ground_truth, str) else {str(item) for item in ground_truth}
    if not relevant:
        return 0.0
    retrieved = set(list(ranked_item_ids)[:k])
    return len(relevant & retrieved) / len(relevant)


def hit_rate_at_k(ranked_item_ids: Iterable[str], ground_truth: str | Iterable[str], k: int = 10) -> float:
    return float(recall_at_k(ranked_item_ids, ground_truth, k) > 0.0)


def ndcg_at_k(ranked_item_ids: Iterable[str], ground_truth: str | Iterable[str], k: int = 10) -> float:
    relevant = {str(ground_truth)} if isinstance(ground_truth, str) else {str(item) for item in ground_truth}
    if not relevant:
        return 0.0
    ranked = [str(item) for item in list(ranked_item_ids)[:k]]
    seen: set[str] = set()
    dcg = 0.0
    for rank, item in enumerate(ranked):
        if item in relevant and item not in seen:
            dcg += 1.0 / math.log2(rank + 2)
        seen.add(item)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


class MetricAccumulator:
    def __init__(self, k: int = 10):
        self.k = k
        self.totals: dict[str, float] = defaultdict(float)
        self.count = 0

    def add(self, ranked: list[str], target: str, valid_format: bool, latency_seconds: float = 0.0) -> dict[str, float]:
        values = {
            f"recall@{self.k}": recall_at_k(ranked, target, self.k),
            f"hr@{self.k}": hit_rate_at_k(ranked, target, self.k),
            f"ndcg@{self.k}": ndcg_at_k(ranked, target, self.k),
            "format_valid": float(valid_format),
            "duplicate_free": float(len(ranked) == len(set(ranked))),
            "latency_seconds": latency_seconds,
        }
        for key, value in values.items():
            self.totals[key] += float(value)
        self.count += 1
        return values

    def compute(self) -> dict[str, Any]:
        if self.count == 0:
            return {"num_samples": 0}
        return {"num_samples": self.count, **{key: value / self.count for key, value in self.totals.items()}}
