"""Build leave-one-out sequential recommendation data from Amazon 5-core reviews."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from datasets import Dataset

from llm4rec.prompts import build_prompt_messages, build_sft_messages, build_target_json
from llm4rec.semantic_ids.rqvae import load_semantic_mapping
from llm4rec.utils import load_yaml, project_root, resolve_project_path, write_jsonl


def iter_gzip_json(path: Path) -> Iterator[dict[str, Any]]:
    """Read both strict JSON-lines and legacy Python-dict Amazon files."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def load_reviews(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in iter_gzip_json(path):
        user_id = row.get("reviewerID", row.get("user_id"))
        item_id = row.get("asin", row.get("parent_asin"))
        timestamp = row.get("unixReviewTime", row.get("timestamp", 0))
        if user_id and item_id:
            rows.append({"user_id": str(user_id), "item_id": str(item_id), "timestamp": int(timestamp or 0)})
    if not rows:
        raise RuntimeError(f"No review interactions parsed from {path}")
    return pd.DataFrame(rows).drop_duplicates(["user_id", "item_id", "timestamp"])


def enforce_k_core(frame: pd.DataFrame, user_k: int, item_k: int) -> pd.DataFrame:
    """Iteratively enforce k-core even when the source filename claims 5-core."""
    current = frame
    while True:
        before = len(current)
        user_counts = current["user_id"].value_counts()
        current = current[current["user_id"].isin(user_counts[user_counts >= user_k].index)]
        item_counts = current["item_id"].value_counts()
        current = current[current["item_id"].isin(item_counts[item_counts >= item_k].index)]
        if len(current) == before:
            return current.reset_index(drop=True)


def _category_text(value: Any) -> str:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            value = value[0]
        return " > ".join(str(part) for part in value if part)
    return str(value or "Unknown")


def load_metadata(path: Path, wanted_items: set[str], title_max_chars: int) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for row in iter_gzip_json(path):
        item_id = str(row.get("asin", row.get("parent_asin", "")))
        if not item_id or item_id not in wanted_items:
            continue
        title = " ".join(str(row.get("title") or "Unknown title").split())[:title_max_chars]
        category = _category_text(row.get("category", row.get("categories")))[:64]
        metadata[item_id] = {"item_id": item_id, "title": title, "category": category}
    for item_id in wanted_items:
        metadata.setdefault(item_id, {"item_id": item_id, "title": "Unknown title", "category": "Unknown"})
    return metadata


class CandidateSampler:
    """Deterministic mixed hard/popularity negative sampler."""

    def __init__(self, item_meta: dict[str, dict[str, str]], popularity: Counter[str], seed: int):
        self.item_meta = item_meta
        self.items = sorted(item_meta)
        self.pop_items = [item_id for item_id, _ in popularity.most_common()]
        self.pop_weights = [max(1.0, float(popularity[item_id]) ** 0.75) for item_id in self.pop_items]
        self.by_category: dict[str, list[str]] = defaultdict(list)
        for item_id, meta in item_meta.items():
            self.by_category[meta["category"]].append(item_id)
        self.seed = seed

    def sample(self, target: str, seen: set[str], size: int, sample_key: str) -> list[str]:
        rng = random.Random(f"{self.seed}:{sample_key}")
        selected = [target]
        forbidden = set(seen) | {target}
        same_category = list(self.by_category.get(self.item_meta[target]["category"], []))
        rng.shuffle(same_category)

        # Half of negatives are category-aware hard negatives.
        hard_budget = max(0, (size - 1) // 2)
        for item_id in same_category:
            if item_id not in forbidden and item_id not in selected:
                selected.append(item_id)
            if len(selected) >= hard_budget + 1:
                break

        attempts = 0
        while len(selected) < size and attempts < size * 100:
            item_id = rng.choices(self.pop_items, weights=self.pop_weights, k=1)[0]
            attempts += 1
            if item_id not in forbidden and item_id not in selected:
                selected.append(item_id)
        if len(selected) < size:
            fallback = self.items[:]
            rng.shuffle(fallback)
            for item_id in fallback:
                if item_id not in forbidden and item_id not in selected:
                    selected.append(item_id)
                if len(selected) == size:
                    break
        rng.shuffle(selected)
        return selected


def make_sample(
    user_id: str,
    sequence: list[str],
    target_index: int,
    split: str,
    cfg: dict[str, Any],
    item_meta: dict[str, dict[str, str]],
    popularity: Counter[str],
    sampler: CandidateSampler,
    semantic_mapping: dict[str, Any] | None = None,
    semantic_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = sequence[target_index]
    history_ids = sequence[max(0, target_index - int(cfg["max_history_items"])) : target_index]
    sample_id = f"{split}:{user_id}:{target_index}"
    candidate_ids = sampler.sample(target, set(sequence), int(cfg["candidate_size"]), sample_id)
    semantic_mapping = semantic_mapping or {}
    semantic_cfg = semantic_cfg or {}
    item_to_sid = semantic_mapping.get("item_to_sid", {})
    semantic_enabled = bool(item_to_sid)

    def render(raw_item_id: str) -> dict[str, str]:
        rendered = dict(item_meta[raw_item_id])
        rendered["asin"] = raw_item_id
        rendered["raw_item_id"] = raw_item_id
        if semantic_enabled:
            rendered["item_id"] = str(item_to_sid[raw_item_id])
        return rendered

    candidates = [render(item_id) for item_id in candidate_ids]
    history = [render(item_id) for item_id in history_ids]

    # SID target has the true next item at rank 1; remaining slots use popularity as a stable tie-break.
    distractors = sorted((item_id for item_id in candidate_ids if item_id != target), key=popularity.get, reverse=True)
    supervised_ranking = [target, *distractors][: int(cfg["top_k"])]
    target_recommendation_ids = [str(item_to_sid[item_id]) for item_id in supervised_ranking] if semantic_enabled else supervised_ranking
    target_json = build_target_json(target_recommendation_ids, top_k=int(cfg["top_k"]))
    prompt = build_prompt_messages(
        history,
        candidates,
        top_k=int(cfg["top_k"]),
        title_max_chars=int(cfg["title_max_chars"]),
        identifier_name="semantic IDs" if semantic_enabled else "item IDs",
    )
    target_recommendation_id = str(item_to_sid[target]) if semantic_enabled else target
    return {
        "sample_id": sample_id,
        "split": split,
        "user_id": user_id,
        "history": history,
        "candidates": candidates,
        "target_item_id": target,
        "target_recommendation_id": target_recommendation_id,
        "identifier_mode": "semantic_id" if semantic_enabled else "asin",
        "semantic_prefix_weight": float(semantic_cfg.get("prefix_reward_weight", 0.0)) if semantic_enabled else 0.0,
        "unknown_item_penalty": float(semantic_cfg.get("unknown_item_penalty", 0.10)),
        "prompt": prompt,
        "target_json": target_json,
        "messages": build_sft_messages(prompt, target_json),
    }


def to_verl_record(sample: dict[str, Any], top_k: int) -> dict[str, Any]:
    candidate_ids = [item["item_id"] for item in sample["candidates"]]
    ground_truth = json.dumps(
        {
            "target_item_id": sample.get("target_recommendation_id", sample["target_item_id"]),
            "target_asin": sample["target_item_id"],
            "candidate_ids": candidate_ids,
            "top_k": top_k,
            "identifier_mode": sample.get("identifier_mode", "asin"),
            "semantic_prefix_weight": float(sample.get("semantic_prefix_weight", 0.0)),
            "unknown_item_penalty": float(sample.get("unknown_item_penalty", 0.10)),
        },
        separators=(",", ":"),
    )
    return {
        "data_source": "amazon_llm4rec",
        "prompt": sample["prompt"],
        "ability": "sequential_recommendation",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "sample_id": sample["sample_id"],
            "split": sample["split"],
            "target_item_id": sample["target_item_id"],
            "target_recommendation_id": sample.get("target_recommendation_id", sample["target_item_id"]),
            "identifier_mode": sample.get("identifier_mode", "asin"),
            "candidate_ids": candidate_ids,
        },
    }


def preprocess(config_path: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    root = project_root()
    raw_dir = resolve_project_path(cfg["raw_dir"], root)
    output_dir = resolve_project_path(cfg["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    category = cfg["category"]
    review_path = raw_dir / f"{category}_5.json.gz"
    metadata_path = raw_dir / f"meta_{category}.json.gz"

    reviews = enforce_k_core(
        load_reviews(review_path), int(cfg["min_user_interactions"]), int(cfg["min_item_interactions"])
    )
    reviews = reviews.sort_values(["user_id", "timestamp", "item_id"], kind="stable")
    popularity: Counter[str] = Counter(reviews["item_id"].tolist())
    item_meta = load_metadata(metadata_path, set(popularity), int(cfg["title_max_chars"]))
    sampler = CandidateSampler(item_meta, popularity, int(cfg["seed"]))
    semantic_cfg = dict(cfg.get("semantic_id", {}))
    semantic_mapping: dict[str, Any] = {}
    if bool(semantic_cfg.get("enabled", False)):
        mapping_path = resolve_project_path(
            semantic_cfg.get("mapping_file", str(output_dir / "semantic_ids.json")), root
        )
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"Semantic ID mapping not found at {mapping_path}. Run "
                "python -m llm4rec.semantic_ids.train_rqvae --config configs/rqvae.yaml first."
            )
        semantic_mapping = load_semantic_mapping(mapping_path)
        missing = set(item_meta) - set(semantic_mapping.get("item_to_sid", {}))
        if missing:
            raise ValueError(f"Semantic ID mapping misses {len(missing)} items, e.g. {sorted(missing)[:3]}")

    samples: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    grouped = list(reviews.groupby("user_id", sort=True))
    random.Random(int(cfg["seed"])).shuffle(grouped)
    for user_id, group in grouped:
        sequence = group["item_id"].tolist()
        if len(sequence) < max(5, int(cfg["min_history_items"]) + 3):
            continue
        train_positions = list(range(int(cfg["min_history_items"]), len(sequence) - 2))
        train_positions = train_positions[-int(cfg.get("max_train_positions_per_user", 3)) :]
        for target_index in train_positions:
            if len(samples["train"]) >= int(cfg["max_train_samples"]):
                break
            samples["train"].append(
                make_sample(
                    user_id, sequence, target_index, "train", cfg, item_meta, popularity, sampler,
                    semantic_mapping, semantic_cfg,
                )
            )
        if len(samples["val"]) < int(cfg["max_val_samples"]):
            samples["val"].append(
                make_sample(
                    user_id, sequence, len(sequence) - 2, "val", cfg, item_meta, popularity, sampler,
                    semantic_mapping, semantic_cfg,
                )
            )
        if len(samples["test"]) < int(cfg["max_test_samples"]):
            samples["test"].append(
                make_sample(
                    user_id, sequence, len(sequence) - 1, "test", cfg, item_meta, popularity, sampler,
                    semantic_mapping, semantic_cfg,
                )
            )
        if all(
            len(samples[name]) >= int(cfg[f"max_{name}_samples"])
            for name in ("train", "val", "test")
        ):
            break

    for split, split_samples in samples.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_samples)
        if split in {"train", "val"}:
            sft_records = ({"sample_id": sample["sample_id"], "messages": sample["messages"]} for sample in split_samples)
            write_jsonl(output_dir / f"sft_{split}.jsonl", sft_records)
            verl_records = [to_verl_record(sample, int(cfg["top_k"])) for sample in split_samples]
            Dataset.from_list(verl_records).to_parquet(str(output_dir / f"verl_{split}.parquet"))

    stats = {
        "category": category,
        "interactions": len(reviews),
        "users": int(reviews["user_id"].nunique()),
        "items": int(reviews["item_id"].nunique()),
        "samples": {name: len(value) for name, value in samples.items()},
        "candidate_size": int(cfg["candidate_size"]),
        "top_k": int(cfg["top_k"]),
        "identifier_mode": "semantic_id" if semantic_mapping else "asin",
        "semantic_id_mapping": str(
            resolve_project_path(semantic_cfg.get("mapping_file", str(output_dir / "semantic_ids.json")), root)
        ) if semantic_mapping else None,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    preprocess(resolve_project_path(args.config, project_root()))


if __name__ == "__main__":
    main()
