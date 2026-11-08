"""Continuous rule reward used by VERL GRPO.

The public `compute_score` signature follows VERL's custom reward interface.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from llm4rec.schema import normalize_item_id, parse_recommendation_json


def _ground_truth_dict(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    return dict(ground_truth)


def _semantic_prefix_similarity(candidate: str, target: str) -> float:
    """Return the fraction of matching RQ-VAE levels from the left."""
    def codes(value: str) -> list[str]:
        verbose = re.findall(r"L\d+_C(\d+)", value)
        if verbose:
            return verbose
        compact = str(value).split("X", 1)[0]
        if compact.startswith("S"):
            compact = compact[1:]
        if "-" in compact:
            return compact.split("-")
        return re.findall(r"\d{2}", compact)

    candidate_codes = codes(str(candidate))
    target_codes = codes(str(target))
    if not candidate_codes or not target_codes:
        return 0.0
    matched = 0
    for candidate_code, target_code in zip(candidate_codes, target_codes):
        if candidate_code != target_code:
            break
        matched += 1
    return matched / max(1, len(target_codes))


def compute_score(
    solution_str: str,
    ground_truth: str | dict[str, Any],
    data_source: str | None = None,
    extra_info: dict[str, Any] | None = None,
    hr_weight: float = 0.45,
    ndcg_weight: float = 0.45,
    format_bonus: float = 0.10,
    invalid_json_penalty: float = -1.0,
    duplicate_penalty: float = 0.15,
    unknown_item_penalty: float = 0.10,
    semantic_prefix_weight: float = 0.0,
) -> dict[str, float]:
    """Score ranking quality, exact JSON format, duplicates and invented IDs."""
    del data_source, extra_info
    truth = _ground_truth_dict(ground_truth)
    unknown_item_penalty = float(truth.get("unknown_item_penalty", unknown_item_penalty))
    semantic_prefix_weight = float(truth.get("semantic_prefix_weight", semantic_prefix_weight))
    target = normalize_item_id(truth["target_item_id"])
    candidate_ids = [normalize_item_id(item_id) for item_id in truth["candidate_ids"]]
    top_k = int(truth.get("top_k", 10))
    parsed = parse_recommendation_json(
        solution_str,
        expected_k=top_k,
        allowed_item_ids=candidate_ids,
        strict_exact_json=True,
    )

    # Completely unparsable output receives a decisive negative signal.
    if parsed.payload is None:
        return {
            "score": float(invalid_json_penalty),
            "hr": 0.0,
            "ndcg": 0.0,
            "format_reward": float(invalid_json_penalty),
            "duplicate_cost": 0.0,
            "unknown_item_cost": 0.0,
        }

    ranked = parsed.item_ids[:top_k]
    rank = ranked.index(target) + 1 if target in ranked else None
    hr = 1.0 if rank is not None else 0.0
    ndcg = 1.0 / math.log2(rank + 1) if rank is not None else 0.0
    duplicate_count = max(0, len(ranked) - len(set(ranked)))
    unknown_count = sum(item_id not in set(candidate_ids) for item_id in ranked)
    format_reward = format_bonus if parsed.valid else -0.25
    length_cost = min(0.25, 0.05 * abs(len(ranked) - top_k))
    duplicate_cost = min(0.45, duplicate_penalty * duplicate_count)
    unknown_cost = min(0.40, unknown_item_penalty * unknown_count)
    prefix_reward = 0.0
    if semantic_prefix_weight > 0.0 and (str(target).startswith("SID_") or str(target).startswith("S")):
        prefix_reward = max(
            (_semantic_prefix_similarity(item_id, target) / math.log2(rank + 1) for rank, item_id in enumerate(ranked, 1)),
            default=0.0,
        ) * semantic_prefix_weight
    score = (
        hr_weight * hr
        + ndcg_weight * ndcg
        + prefix_reward
        + format_reward
        - length_cost
        - duplicate_cost
        - unknown_cost
    )
    return {
        "score": float(max(-1.0, min(1.0, score))),
        "hr": hr,
        "ndcg": ndcg,
        "format_reward": float(format_reward),
        "duplicate_cost": float(duplicate_cost),
        "unknown_item_cost": float(unknown_cost),
        "semantic_prefix_reward": float(prefix_reward),
    }
