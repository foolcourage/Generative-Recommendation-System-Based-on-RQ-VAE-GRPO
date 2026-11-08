"""Recompute metrics from cached generations after parser/metric changes."""

from __future__ import annotations

import argparse
import json

from llm4rec.evaluation.metrics import MetricAccumulator
from llm4rec.schema import parse_recommendation_json
from llm4rec.utils import project_root, read_jsonl, resolve_project_path, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    root = project_root()
    test_records = {
        record["sample_id"]: record for record in read_jsonl(resolve_project_path(args.test_file, root))
    }
    accumulator = MetricAccumulator(args.top_k)
    rescored: list[dict] = []

    for prediction in read_jsonl(resolve_project_path(args.predictions, root)):
        record = test_records[prediction["sample_id"]]
        allowed_ids = [item["item_id"] for item in record["candidates"]]
        parsed = parse_recommendation_json(
            prediction.get("prediction", ""),
            expected_k=args.top_k,
            allowed_item_ids=allowed_ids,
            strict_exact_json=True,
        )
        latency = float(prediction.get("metrics", {}).get("latency_seconds", 0.0))
        target_identifier = record.get("target_recommendation_id", record["target_item_id"])
        sample_metrics = accumulator.add(parsed.item_ids, target_identifier, parsed.valid, latency)
        rescored.append(
            {
                **prediction,
                "parsed_item_ids": parsed.item_ids,
                "target_recommendation_id": target_identifier,
                "parse_errors": parsed.errors,
                "metrics": sample_metrics,
            }
        )

    metrics = accumulator.compute()
    output_file = resolve_project_path(args.output_file, root)
    metrics_file = resolve_project_path(args.metrics_file, root)
    write_jsonl(output_file, rescored)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
