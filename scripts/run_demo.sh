#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
python -m llm4rec.inference.demo "$@"

