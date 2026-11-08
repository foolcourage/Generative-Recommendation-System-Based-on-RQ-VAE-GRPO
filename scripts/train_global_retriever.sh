#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
CONFIG=${1:-configs/global_retriever.yaml}
python -m llm4rec.semantic_ids.train_global_retriever --config "$CONFIG" 2>&1 | tee "outputs/global_retriever_$(date +%Y%m%d_%H%M%S).log"
