#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
CONFIG=${1:-configs/eval.yaml}
shift || true
python -m llm4rec.evaluation.evaluate --config "$CONFIG" "$@"

