#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
bash scripts/prepare_data.sh configs/data.yaml
bash scripts/train_sft.sh configs/sft_smoke.yaml
python -m llm4rec.training.train_grpo_verl --config configs/grpo_smoke.yaml --dry-run
pytest -q

