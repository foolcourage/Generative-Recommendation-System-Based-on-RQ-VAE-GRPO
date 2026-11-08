"""Shared IO, configuration and reproducibility helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def resolve_project_path(path: str | os.PathLike[str], root: Path | None = None) -> Path:
    candidate = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if candidate.is_absolute():
        return candidate
    return (root or project_root()) / candidate


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_jsonl(path: str | os.PathLike[str], limit: int | None = None) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike[str], records: Iterable[dict[str, Any]]) -> int:
    target = ensure_parent(path)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def estimate_vram_gb(model_params_billion: float, mode: str) -> dict[str, float]:
    """Conservative 24GB planning estimates, not allocator-exact measurements."""
    if mode == "sft_qlora":
        weights = model_params_billion * 0.65
        activations = 4.0 if model_params_billion >= 3 else 2.5
        overhead = 3.0
        return {"quantized_weights": weights, "activations": activations, "overhead": overhead, "total": weights + activations + overhead}
    if mode == "grpo_lora_vllm":
        actor = model_params_billion * 2.1
        rollout = model_params_billion * 2.1
        kv_and_activations = 5.0
        overhead = 2.5
        return {"actor": actor, "rollout": rollout, "kv_and_activations": kv_and_activations, "overhead": overhead, "total": actor + rollout + kv_and_activations + overhead}
    raise ValueError(f"Unknown mode: {mode}")

