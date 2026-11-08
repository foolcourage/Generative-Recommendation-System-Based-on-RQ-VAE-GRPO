"""Export a VERL LoRA-only FSDP checkpoint as a standard PEFT adapter."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", required=True)
    parser.add_argument("--adapter-config-source", required=True)
    parser.add_argument("--target-dir", required=True)
    args = parser.parse_args()

    actor_dir = Path(args.actor_dir)
    config_source = Path(args.adapter_config_source)
    target_dir = Path(args.target_dir)
    # VERL/FSDP writes one LoRA-only file per rank even when the tensors are
    # fully materialized by ``summon_full_params``.  The rank files are
    # identical for this save path, so rank 0 is sufficient for a PEFT export.
    checkpoint = actor_dir / "model_world_size_1_rank_0.pt"
    if not checkpoint.exists():
        candidates = sorted(actor_dir.glob("model_world_size_*_rank_0.pt"))
        if candidates:
            checkpoint = candidates[-1]
    if not checkpoint.exists():
        raise FileNotFoundError(f"LoRA actor checkpoint not found under: {actor_dir}")
    if not (config_source / "adapter_config.json").exists():
        raise FileNotFoundError(f"PEFT adapter config not found: {config_source}")

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapter_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if "lora_" not in key and ".adapter_" not in key:
            continue
        # PEFT save_pretrained removes the runtime adapter name ``default``.
        export_key = key.replace(".default.", ".")
        adapter_state[export_key] = value.detach().contiguous().cpu()
    if not adapter_state:
        raise ValueError(f"No LoRA tensors found in {checkpoint}")

    target_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((config_source / "adapter_config.json").read_text(encoding="utf-8"))
    config["inference_mode"] = True
    (target_dir / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    save_file(adapter_state, target_dir / "adapter_model.safetensors", metadata={"format": "pt"})

    tokenizer_source = actor_dir / "huggingface"
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        source = tokenizer_source / name
        if source.exists():
            shutil.copy2(source, target_dir / name)
    print(json.dumps({"target_dir": str(target_dir), "tensors": len(adapter_state)}, indent=2))


if __name__ == "__main__":
    main()
