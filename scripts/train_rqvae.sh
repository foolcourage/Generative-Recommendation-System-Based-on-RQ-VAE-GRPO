#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
CONFIG=${1:-configs/rqvae.yaml}
python -m llm4rec.semantic_ids.train_rqvae --config "$CONFIG" 2>&1 | tee "outputs/rqvae_train_$(date +%Y%m%d_%H%M%S).log"
