"""Print tokenized feature types for diagnosing tokenizer-version changes."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from llm4rec.data.dataset import RecommendationSFTDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--file", default="data/processed/all_beauty/sft_train.jsonl")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    ds = RecommendationSFTDataset(args.file, tokenizer, max_length=768, max_samples=1)
    raw_prompt = tokenizer.apply_chat_template(ds.records[0]["messages"][:-1], tokenize=True, add_generation_prompt=True)
    raw_target = tokenizer(ds.records[0]["messages"][-1]["content"], add_special_tokens=False)["input_ids"]
    print("raw_types", type(raw_prompt).__name__, type(raw_target).__name__)
    print("raw_lengths", len(raw_prompt["input_ids"]), len(raw_target))
    feature = ds[0]
    print({key: (type(value).__name__, type(value[0]).__name__ if value else None, len(value)) for key, value in feature.items()})


if __name__ == "__main__":
    main()
