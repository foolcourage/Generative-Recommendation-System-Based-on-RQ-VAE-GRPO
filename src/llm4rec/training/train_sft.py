"""Stage 1: completion-only Sequence Instruction Tuning with 4-bit QLoRA."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from llm4rec.data.dataset import CompletionOnlyCollator, RecommendationSFTDataset
from llm4rec.utils import load_yaml, project_root, resolve_project_path, seed_everything


def _dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def build_training_arguments(cfg: dict[str, Any], output_dir: Path, has_eval: bool) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(cfg["num_train_epochs"]),
        "max_steps": int(cfg.get("max_steps", -1)),
        "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(cfg["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["learning_rate"]),
        "warmup_ratio": float(cfg["warmup_ratio"]),
        "weight_decay": float(cfg["weight_decay"]),
        "lr_scheduler_type": "cosine",
        "logging_steps": int(cfg["logging_steps"]),
        "save_strategy": "steps",
        "save_steps": int(cfg["save_steps"]),
        "save_total_limit": int(cfg["save_total_limit"]),
        "bf16": bool(cfg["bf16"]),
        "tf32": bool(cfg["tf32"]),
        "gradient_checkpointing": bool(cfg["gradient_checkpointing"]),
        "optim": str(cfg["optim"]),
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": 2,
        "seed": int(cfg["seed"]),
        "data_seed": int(cfg["seed"]),
    }
    if "group_by_length" in inspect.signature(TrainingArguments).parameters:
        kwargs["group_by_length"] = True
    eval_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    kwargs[eval_key] = "steps" if has_eval else "no"
    if has_eval:
        kwargs["eval_steps"] = int(cfg["eval_steps"])
    return TrainingArguments(**kwargs)


def train(config_path: Path) -> Path:
    cfg = load_yaml(config_path)
    root = project_root()
    seed_everything(int(cfg["seed"]))
    output_dir = resolve_project_path(cfg["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_cfg = cfg["quantization"]
    compute_dtype = _dtype(str(quant_cfg["bnb_4bit_compute_dtype"]))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(quant_cfg["load_in_4bit"]),
        bnb_4bit_quant_type=str(quant_cfg["bnb_4bit_quant_type"]),
        bnb_4bit_use_double_quant=bool(quant_cfg["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "device_map": {"": 0},
        "torch_dtype": compute_dtype,
    }
    if cfg.get("flash_attention_2"):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], **model_kwargs)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(cfg["gradient_checkpointing"])
    )
    lora_config = LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg["lora_target_modules"]),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = RecommendationSFTDataset(
        resolve_project_path(cfg["train_file"], root),
        tokenizer,
        int(cfg["max_length"]),
        cfg.get("max_train_samples"),
    )
    eval_path = resolve_project_path(cfg["eval_file"], root)
    eval_dataset = (
        RecommendationSFTDataset(eval_path, tokenizer, int(cfg["max_length"]), cfg.get("max_eval_samples"))
        if eval_path.exists() and cfg.get("max_eval_samples", 0) != 0
        else None
    )
    collator = CompletionOnlyCollator(tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=build_training_arguments(cfg, output_dir, eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = dict(result.metrics)
    if torch.cuda.is_available():
        metrics["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        metrics["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
    (output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft_qlora.yaml")
    args = parser.parse_args()
    train(resolve_project_path(args.config, project_root()))


if __name__ == "__main__":
    main()
