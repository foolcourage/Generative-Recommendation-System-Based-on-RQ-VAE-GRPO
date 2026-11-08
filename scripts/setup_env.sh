#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_PREFIX=${LLM4REC_ENV_PREFIX:-/root/autodl-tmp/envs/llm4rec}
VERL_SRC=${VERL_SRC:-/root/autodl-tmp/verl-src}
VERL_COMMIT=${VERL_COMMIT:-d908005de87b930e40b24dce307971f6b4580650}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}

source /root/miniconda3/etc/profile.d/conda.sh
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.12 pip
fi
conda activate "$ENV_PREFIX"
export PIP_CACHE_DIR CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=${MAX_JOBS:-4}

python -m pip install --upgrade pip setuptools wheel
# Install vLLM first because its wheel pins the compatible PyTorch/CUDA runtime.
python -m pip install "vllm==0.24.0"
python -m pip install -r "$PROJECT_ROOT/requirements.txt"
python -m pip install \
  "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata "ray[default]" codetiming hydra-core \
  pylatexenc wandb dill pybind11 liger-kernel mathruler qwen-vl-utils \
  "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" \
  "pydantic>=2.9" "grpcio>=1.62.1"

if [[ ! -d "$VERL_SRC/.git" ]]; then
  git clone https://github.com/volcengine/verl.git "$VERL_SRC"
fi
if [[ "$(git -C "$VERL_SRC" rev-parse HEAD)" != "$VERL_COMMIT" ]]; then
  git -C "$VERL_SRC" fetch --depth 1 origin "$VERL_COMMIT"
  git -C "$VERL_SRC" checkout --detach "$VERL_COMMIT"
fi
python -m pip install --no-deps -e "$VERL_SRC"

# VERL can run without remove-padding kernels. Build FlashAttention only when
# the local CUDA toolkit exactly matches the CUDA runtime used by PyTorch.
if ! python -c "import flash_attn" >/dev/null 2>&1; then
  TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or '')")
  NVCC_CUDA=$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)
  if [[ -n "$TORCH_CUDA" && "$TORCH_CUDA" == "$NVCC_CUDA" ]]; then
    python -m pip install --no-build-isolation "flash_attn==2.8.3"
  else
    echo "Skipping FlashAttention: torch CUDA=$TORCH_CUDA, nvcc CUDA=$NVCC_CUDA"
    echo "Keep memory.remove_padding=false in GRPO configs for this environment."
  fi
fi

# Legacy VERL unconditionally imports FlashAttention padding helpers while
# computing old/reference log-probabilities. When FlashAttention cannot be
# built safely, install a small source compatibility patch that uses the
# equivalent PyTorch helpers bundled with Transformers.
if ! python -c "import flash_attn" >/dev/null 2>&1; then
  ATTENTION_UTILS="$VERL_SRC/verl/utils/attention_utils.py"
  if ! grep -q "from transformers.modeling_flash_attention_utils import" "$ATTENTION_UTILS"; then
    patch -d "$VERL_SRC" -p1 < "$PROJECT_ROOT/patches/verl_no_flash_attn.patch"
  fi
fi

# VERL's LoRA-only FSDP checkpoint path still materializes the full base-model
# state dict before filtering, which OOMs with a colocated 3B vLLM server.
CHECKPOINT_MANAGER="$VERL_SRC/verl/utils/checkpoint/fsdp_checkpoint_manager.py"
if ! grep -q "Do not build the full FSDP state dict before filtering" "$CHECKPOINT_MANAGER"; then
  patch -d "$VERL_SRC" -p1 < "$PROJECT_ROOT/patches/verl_lora_only_checkpoint.patch"
fi
python -m pip install -e "$PROJECT_ROOT"

python - <<'PY'
import torch, transformers, peft, datasets, vllm, verl
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("datasets", datasets.__version__)
print("vllm", vllm.__version__)
print("verl", getattr(verl, "__version__", "installed"))
PY
