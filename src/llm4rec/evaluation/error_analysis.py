"""Summarize parse errors, duplicates, hits and target-rank behavior."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from llm4rec.utils import read_jsonl


def analyze(path: str) -> dict:
    errors: Counter[str] = Counter()
    ranks: list[int] = []
    duplicate_samples = 0
    total = 0
    for row in read_jsonl(Path(path)):
        total += 1
        errors.update(row.get("parse_errors", []))
        ranked = row.get("parsed_item_ids", [])
        if len(ranked) != len(set(ranked)):
            duplicate_samples += 1
        target = row.get("target_recommendation_id", row.get("target_item_id"))
        if target in ranked[:10]:
            ranks.append(ranked.index(target) + 1)
    rank_histogram = Counter(ranks)
    return {
        "num_samples": total,
        "parse_error_counts": dict(errors.most_common()),
        "duplicate_samples": duplicate_samples,
        "hit_samples": len(ranks),
        "mean_target_rank_when_hit": sum(ranks) / len(ranks) if ranks else None,
        "target_rank_histogram": {str(rank): rank_histogram[rank] for rank in sorted(rank_histogram)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
