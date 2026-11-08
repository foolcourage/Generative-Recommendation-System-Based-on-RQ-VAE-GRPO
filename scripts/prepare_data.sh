#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common_env.sh"
CONFIG=${1:-configs/data.yaml}
CATEGORY=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['category'])")
python -m llm4rec.data.download_amazon --category "$CATEGORY" --output-dir data/raw
SEMANTIC=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(str(c.get('semantic_id', {}).get('enabled', False)).lower())")
if [[ "$SEMANTIC" == "true" ]]; then
  RQ_CONFIG=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('semantic_id', {}).get('rqvae_config', 'configs/rqvae.yaml'))")
  python -m llm4rec.semantic_ids.train_rqvae --config "$RQ_CONFIG"
fi
python -m llm4rec.data.preprocess --config "$CONFIG"
