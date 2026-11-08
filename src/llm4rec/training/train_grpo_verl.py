"""Stage 2: launch VERL GRPO with a custom recommendation reward."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from llm4rec.utils import load_yaml, project_root, resolve_project_path


def _path_or_hf_id(value: str, root: Path) -> str:
    candidate = resolve_project_path(value, root)
    return str(candidate) if candidate.exists() else value


def build_overrides(cfg: dict[str, Any], root: Path) -> list[str]:
    memory = cfg["memory"]
    n_gpus = int(cfg.get("n_gpus_per_node", 1))
    nnodes = int(cfg.get("nnodes", 1))
    tensor_parallel_size = int(cfg.get("tensor_model_parallel_size", 1))
    if n_gpus < 1 or tensor_parallel_size < 1 or tensor_parallel_size > n_gpus:
        raise ValueError("tensor_model_parallel_size must be between 1 and n_gpus_per_node")
    if n_gpus % tensor_parallel_size != 0:
        raise ValueError("n_gpus_per_node must be divisible by tensor_model_parallel_size")
    # VERL expresses ppo_mini_batch_size in source prompts and multiplies it by
    # rollout.n internally before splitting the trajectory batch.
    if int(cfg["train_batch_size"]) % int(cfg["ppo_mini_batch_size"]) != 0:
        raise ValueError("ppo_mini_batch_size must divide train_batch_size (before group expansion)")
    adapter = resolve_project_path(cfg["sft_adapter_path"], root)
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"SFT LoRA adapter not found: {adapter}")
    reward_path = resolve_project_path(cfg["reward_function"], root)
    train_file = resolve_project_path(cfg["train_file"], root)
    val_file = resolve_project_path(cfg["val_file"], root)
    output_dir = resolve_project_path(cfg["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={train_file}",
        f"data.val_files={val_file}",
        f"data.train_batch_size={cfg['train_batch_size']}",
        f"data.max_prompt_length={cfg['max_prompt_length']}",
        f"data.max_response_length={cfg['max_response_length']}",
        "data.filter_overlong_prompts=True",
        "data.filter_overlong_prompts_workers=1",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={_path_or_hf_id(cfg['model_name_or_path'], root)}",
        f"actor_rollout_ref.model.lora_adapter_path={adapter}",
        f"actor_rollout_ref.model.lora_rank={cfg['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={cfg['lora_alpha']}",
        f"actor_rollout_ref.model.use_remove_padding={str(memory['remove_padding']).lower()}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={str(memory['gradient_checkpointing']).lower()}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"actor_rollout_ref.actor.optim.lr={cfg['actor_learning_rate']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg['ppo_micro_batch_size_per_gpu']}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg['max_token_len_per_gpu']}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={cfg['kl_loss_coef']}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.actor.entropy_coeff={cfg['entropy_coeff']}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={str(memory['actor_param_offload']).lower()}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={str(memory['actor_optimizer_offload']).lower()}",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        f"+actor_rollout_ref.actor.checkpoint.save_lora_only={str(memory.get('save_lora_only', True)).lower()}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tensor_parallel_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg['rollout_gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.n={cfg['group_size']}",
        f"actor_rollout_ref.rollout.max_num_seqs={cfg['rollout_max_num_seqs']}",
        f"actor_rollout_ref.rollout.max_model_len={int(cfg['max_prompt_length']) + int(cfg['max_response_length'])}",
        f"actor_rollout_ref.rollout.free_cache_engine={str(memory['free_cache_engine']).lower()}",
        f"actor_rollout_ref.rollout.enforce_eager={str(memory['enforce_eager']).lower()}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={str(memory['enable_chunked_prefill']).lower()}",
        "actor_rollout_ref.rollout.enable_prefix_caching=False",
        # VERL 0.9's layered LoRA summon falls back to
        # summon_full_params(offload_to_cpu=True) when FSDP1 uses NO_SHARD on
        # one GPU. PyTorch explicitly rejects that combination. The regular
        # summon path is safe here because a single rank already owns the full
        # LoRA tensors and does not request CPU offload.
        f"actor_rollout_ref.rollout.layered_summon={str(memory.get('layered_summon', False)).lower()}",
        "actor_rollout_ref.rollout.load_format=safetensors",
        f"actor_rollout_ref.rollout.temperature={cfg['temperature']}",
        f"actor_rollout_ref.rollout.top_p={cfg['top_p']}",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        f"actor_rollout_ref.ref.log_prob_max_token_len_per_gpu={cfg['max_token_len_per_gpu']}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={str(memory['ref_param_offload']).lower()}",
        f"reward.custom_reward_function.path={reward_path}",
        "reward.custom_reward_function.name=compute_score",
        "reward.reward_manager.name=naive",
        f"trainer.n_gpus_per_node={n_gpus}",
        f"trainer.nnodes={nnodes}",
        # The inspected VERL source checkout omits the experimental transfer_queue
        # package required by V1. Legacy PPO remains supported in this release and
        # uses the same GRPO, FSDP, vLLM and LoRA worker implementations.
        "trainer.use_v1=false",
        f"trainer.total_epochs={cfg['total_epochs']}",
        f"trainer.save_freq={cfg['save_freq']}",
        f"trainer.test_freq={cfg['test_freq']}",
        "trainer.val_before_train=False",
        "trainer.balance_batch=True",
        "trainer.logger=[\"console\"]",
        "trainer.project_name=llm4rec_amazon",
        f"trainer.experiment_name={Path(cfg['output_dir']).name}",
        f"trainer.default_local_dir={output_dir}",
        "trainer.resume_mode=disable",
    ]
    if cfg.get("total_training_steps") is not None:
        overrides.append(f"trainer.total_training_steps={cfg['total_training_steps']}")
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo_verl.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra overrides")
    args = parser.parse_args()
    root = project_root()
    cfg = load_yaml(resolve_project_path(args.config, root))
    if cfg.get("reference_mode") != "shared_frozen_base":
        raise ValueError("Current VERL LoRA path supports reference_mode=shared_frozen_base only")
    command = [sys.executable, "-m", "verl.trainer.main_ppo", *build_overrides(cfg, root), *args.overrides]
    print(json.dumps(command, indent=2))
    if args.dry_run:
        return
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NCCL_DEBUG"] = env.get("NCCL_DEBUG", "WARN")
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
