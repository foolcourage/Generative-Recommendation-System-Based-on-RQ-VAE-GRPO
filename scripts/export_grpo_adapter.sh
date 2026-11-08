#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
RUN_DIR=${1:-outputs/grpo/qwen2.5-3b-all-beauty}
STEP_DIR=${2:-}
SFT_ADAPTER=${3:-outputs/sft/qwen2.5-3b-all-beauty}
if [[ -z "$STEP_DIR" ]]; then
  STEP_DIR=$(find "$RUN_DIR" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sort -V | tail -1)
fi
if [[ -z "$STEP_DIR" || ! -d "$RUN_DIR/$STEP_DIR/actor" ]]; then
  echo "No VERL actor checkpoint found under $RUN_DIR" >&2
  exit 1
fi
TARGET="$RUN_DIR/$STEP_DIR/export"
ACTOR_DIR="$RUN_DIR/$STEP_DIR/actor"
if [[ -f "$ACTOR_DIR/lora_train_meta.json" ]]; then
  python -m llm4rec.training.export_lora_checkpoint \
    --actor-dir "$ACTOR_DIR" \
    --adapter-config-source "$SFT_ADAPTER" \
    --target-dir "$TARGET/lora_adapter"
else
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET"
fi
echo "LoRA adapter: $TARGET/lora_adapter"
