#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX=${LLM4REC_ENV_PREFIX:-/root/autodl-tmp/envs/llm4rec}

source /root/miniconda3/etc/profile.d/conda.sh
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "Environment missing: $ENV_PREFIX. Run scripts/setup_env.sh first." >&2
  exit 1
fi
conda activate "$ENV_PREFIX"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/root/autodl-tmp/huggingface}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/root/autodl-tmp/vllm-cache}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"
cd "$PROJECT_ROOT"
