"""Offline generation and Recall@10/NDCG@10 evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llm4rec.evaluation.metrics import MetricAccumulator
from llm4rec.schema import parse_recommendation_json
from llm4rec.utils import load_yaml, project_root, read_jsonl, resolve_project_path, seed_everything, write_jsonl


def load_policy(
    model_name_or_path: str,
    adapter_path: str | Path | None,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    tokenizer_source = str(adapter_path) if adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists() else model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model_kwargs: dict[str, Any] = {"device_map": {"": 0}, "dtype": torch.bfloat16}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if adapter_path:
        adapter = Path(adapter_path)
        if (adapter / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(adapter))
        else:
            raise FileNotFoundError(f"adapter_config.json not found in {adapter}")
    model.eval()
    return model, tokenizer


def _encode_prompt(tokenizer: Any, messages: list[dict[str, str]], max_length: int) -> dict[str, torch.Tensor]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if isinstance(ids, str) or (ids and isinstance(ids[0], str)):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if len(ids) > max_length:
        head = min(192, max_length // 3)
        ids = [*ids[:head], *ids[-(max_length - head) :]]
    input_ids = torch.tensor([ids], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


@torch.inference_mode()
def generate_one(model: Any, tokenizer: Any, messages: list[dict[str, str]], cfg: dict[str, Any]) -> tuple[str, float]:
    inputs = {key: value.to(model.device) for key, value in _encode_prompt(tokenizer, messages, int(cfg["max_input_length"])).items()}
    temperature = float(cfg.get("temperature", 0.0))
    started = time.perf_counter()
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "do_sample": temperature > 0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if temperature > 0:
        generation_kwargs.update({"temperature": temperature, "top_p": 0.95})
    output = model.generate(**inputs, **generation_kwargs)
    elapsed = time.perf_counter() - started
    response_ids = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip(), elapsed


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    root = project_root()
    seed_everything(int(cfg["seed"]))
    adapter_path = resolve_project_path(cfg["adapter_path"], root) if cfg.get("adapter_path") else None
    model_path = str(resolve_project_path(cfg["model_name_or_path"], root)) if resolve_project_path(cfg["model_name_or_path"], root).exists() else cfg["model_name_or_path"]
    model, tokenizer = load_policy(model_path, adapter_path, bool(cfg.get("load_in_4bit", True)))
    test_path = resolve_project_path(cfg["test_file"], root)
    accumulator = MetricAccumulator(int(cfg["top_k"]))
    predictions: list[dict[str, Any]] = []

    for record in read_jsonl(test_path, limit=cfg.get("max_samples")):
        response, latency = generate_one(model, tokenizer, record["prompt"], cfg)
        candidates = [item["item_id"] for item in record["candidates"]]
        parsed = parse_recommendation_json(
            response,
            expected_k=int(cfg["top_k"]),
            allowed_item_ids=candidates,
            strict_exact_json=True,
        )
        target_identifier = record.get("target_recommendation_id", record["target_item_id"])
        sample_metrics = accumulator.add(parsed.item_ids, target_identifier, parsed.valid, latency)
        semantic_to_asin = {
            str(item.get("item_id")): str(item.get("asin", item.get("raw_item_id", item.get("item_id"))))
            for item in record.get("candidates", [])
        }
        decoded_asins = [semantic_to_asin[item_id] for item_id in parsed.item_ids if item_id in semantic_to_asin]
        predictions.append(
            {
                "sample_id": record["sample_id"],
                "target_item_id": record["target_item_id"],
                "target_recommendation_id": target_identifier,
                "identifier_mode": record.get("identifier_mode", "asin"),
                "prediction": response,
                "parsed_item_ids": parsed.item_ids,
                "decoded_asins": decoded_asins,
                "parse_errors": parsed.errors,
                "metrics": sample_metrics,
            }
        )

    metrics = accumulator.compute()
    if torch.cuda.is_available():
        metrics["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        metrics["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
    output_file = resolve_project_path(cfg["output_file"], root)
    metrics_file = resolve_project_path(cfg["metrics_file"], root)
    write_jsonl(output_file, predictions)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--test-file")
    parser.add_argument("--output-file")
    parser.add_argument("--metrics-file")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    cfg = load_yaml(resolve_project_path(args.config, project_root()))
    for key in ("model_name_or_path", "adapter_path", "test_file", "output_file", "metrics_file", "max_samples"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    evaluate(cfg)


if __name__ == "__main__":
    main()
