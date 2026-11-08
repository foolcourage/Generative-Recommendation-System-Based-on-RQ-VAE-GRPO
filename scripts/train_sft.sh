#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
CONFIG=${1:-configs/sft_qlora.yaml}
mkdir -p outputs/sft
python -m llm4rec.training.train_sft --config "$CONFIG" 2>&1 | tee "outputs/sft/train_$(date +%Y%m%d_%H%M%S).log"
