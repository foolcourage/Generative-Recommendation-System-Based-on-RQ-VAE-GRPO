"""Run one structured recommendation example from the processed test set."""

from __future__ import annotations

import argparse
import json

from llm4rec.evaluation.evaluate import generate_one, load_policy
from llm4rec.schema import parse_recommendation_json
from llm4rec.utils import load_yaml, project_root, read_jsonl, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    root = project_root()
    cfg = load_yaml(resolve_project_path(args.config, root))
    if args.model_name_or_path:
        cfg["model_name_or_path"] = args.model_name_or_path
    if args.adapter_path:
        cfg["adapter_path"] = args.adapter_path
    records = list(read_jsonl(resolve_project_path(cfg["test_file"], root), limit=args.sample_index + 1))
    if args.sample_index >= len(records):
        raise IndexError(f"sample-index {args.sample_index} exceeds test data")
    record = records[args.sample_index]
    adapter = resolve_project_path(cfg["adapter_path"], root) if cfg.get("adapter_path") else None
    model_path_candidate = resolve_project_path(cfg["model_name_or_path"], root)
    model_path = str(model_path_candidate) if model_path_candidate.exists() else cfg["model_name_or_path"]
    model, tokenizer = load_policy(model_path, adapter, bool(cfg.get("load_in_4bit", True)))
    response, latency = generate_one(model, tokenizer, record["prompt"], cfg)
    candidate_ids = [item["item_id"] for item in record["candidates"]]
    parsed = parse_recommendation_json(response, int(cfg["top_k"]), candidate_ids)
    semantic_to_asin = {
        str(item.get("item_id")): str(item.get("asin", item.get("raw_item_id", item.get("item_id"))))
        for item in record.get("candidates", [])
    }
    print(json.dumps({
        "sample_id": record["sample_id"],
        "target_item_id": record["target_item_id"],
        "target_recommendation_id": record.get("target_recommendation_id", record["target_item_id"]),
        "identifier_mode": record.get("identifier_mode", "asin"),
        "raw_output": response,
        "parsed_item_ids": parsed.item_ids,
        "decoded_asins": [semantic_to_asin[item_id] for item_id in parsed.item_ids if item_id in semantic_to_asin],
        "valid": parsed.valid,
        "errors": parsed.errors,
        "latency_seconds": latency,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
