"""Inspect chat-template token lengths before launching VERL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from llm4rec.utils import project_root, read_jsonl, resolve_project_path


def inspect(path: Path, model_name_or_path: str, limit: int | None = None) -> dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    lengths: list[int] = []
    first_roles: list[str] = []
    for record in read_jsonl(path, limit=limit):
        messages = record["prompt"]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
        if not first_roles:
            first_roles = [str(message.get("role")) for message in messages]
    if not lengths:
        raise ValueError(f"No records found in {path}")
    lengths.sort()
    result = {
        "num_samples": len(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "p95_tokens": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))],
        "first_prompt_roles": first_roles,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    inspect(resolve_project_path(args.file, project_root()), args.model, args.limit)


if __name__ == "__main__":
    main()
