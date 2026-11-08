"""Create an SFT-only versus SFT+GRPO metrics comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--grpo", required=True)
    parser.add_argument("--output", default="outputs/evaluation/comparison.md")
    args = parser.parse_args()
    sft, grpo = load(args.sft), load(args.grpo)
    keys = ["recall@10", "hr@10", "ndcg@10", "format_valid", "duplicate_free", "latency_seconds", "peak_allocated_gb"]
    lines = ["| Metric | SFT-only | SFT+GRPO | Delta |", "|---|---:|---:|---:|"]
    for key in keys:
        if key not in sft and key not in grpo:
            continue
        left, right = float(sft.get(key, 0.0)), float(grpo.get(key, 0.0))
        lines.append(f"| {key} | {left:.6f} | {right:.6f} | {right - left:+.6f} |")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

