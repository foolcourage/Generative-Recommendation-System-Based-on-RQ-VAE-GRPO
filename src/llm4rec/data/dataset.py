"""PyTorch dataset and completion-only collator for QLoRA SID training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from llm4rec.utils import read_jsonl


def _middle_truncate(tokens: list[int], budget: int, keep_head: int = 192) -> list[int]:
    """Keep system instructions and the newest history/candidates when over length."""
    if len(tokens) <= budget:
        return tokens
    head = min(keep_head, budget // 3)
    return [*tokens[:head], *tokens[-(budget - head) :]]


class RecommendationSFTDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: Any, max_length: int, max_samples: int | None = None):
        self.records = list(read_jsonl(path, limit=max_samples))
        self.tokenizer = tokenizer
        self.max_length = max_length
        if not self.records:
            raise ValueError(f"No samples found in {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        messages = self.records[index]["messages"]
        prompt_messages = messages[:-1]
        target_text = messages[-1]["content"]
        prompt_ids = self.tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
        # Transformers 5 may return a BatchEncoding when chat-template tokenization
        # is requested; normalize both that form and the historical list[int] form.
        if hasattr(prompt_ids, "keys") and "input_ids" in prompt_ids:
            prompt_ids = prompt_ids["input_ids"]
        if prompt_ids and isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
        if isinstance(prompt_ids, str) or (prompt_ids and isinstance(prompt_ids[0], str)):
            rendered_prompt = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        if target_ids and isinstance(target_ids[0], list):
            target_ids = target_ids[0]
        if isinstance(target_ids, str) or (target_ids and isinstance(target_ids[0], str)):
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
            target_ids.append(eos_id)

        prompt_budget = max(1, self.max_length - len(target_ids))
        prompt_ids = _middle_truncate(prompt_ids, prompt_budget)
        input_ids = [*prompt_ids, *target_ids[: self.max_length - len(prompt_ids)]]
        labels = [-100] * len(prompt_ids) + target_ids[: self.max_length - len(prompt_ids)]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class CompletionOnlyCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
