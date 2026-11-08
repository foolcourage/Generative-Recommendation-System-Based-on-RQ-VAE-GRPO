#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
CONFIG=${1:-configs/grpo_verl.yaml}
shift || true
mkdir -p outputs/grpo
python -m llm4rec.training.train_grpo_verl --config "$CONFIG" "$@" 2>&1 | tee "outputs/grpo/train_$(date +%Y%m%d_%H%M%S).log"
