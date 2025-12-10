 #!/usr/bin/env bash
# Helper to launch distributed training with automatic LR scaling
# Usage:
#   ./run_distributed.sh <num_gpus> <per_gpu_batch> [<base_lr>] [<epochs>]
# Example:
#   ./run_distributed.sh 7 32 0.0005 150

set -euo pipefail

NUM_GPUS=${1:-7}
PER_GPU_BATCH=${2:-32}
BASE_LR=${3:-0.0005}
EPOCHS=${4:-150}
WARMUP=${5:-10}
TRAIN_DIR=${6:-/data/imagenet/train}
VAL_DIR=${7:-/data/imagenet/val}
SAVE_DIR=${8:-output}

GLOBAL_BATCH=$(( NUM_GPUS * PER_GPU_BATCH ))
# LR scaling relative to global batch 256
SCALED_LR=$(python - <<PY
from math import isfinite
base_lr = ${BASE_LR}
global_batch = ${GLOBAL_BATCH}
scaled = base_lr * (global_batch / 256.0)
print(f"{scaled:.6g}")
PY
)

# Quick dependency check to fail fast with a helpful message
python - <<PY
try:
  import timm
except Exception:
  print("Missing Python package 'timm'. Install with: pip install timm or pip install -r requirements.txt")
  raise SystemExit(1)
try:
  import einops
except Exception:
  print("Missing Python package 'einops'. Install with: pip install einops or pip install -r requirements.txt")
  raise SystemExit(1)
PY

echo "Launching training on ${NUM_GPUS} GPUs"
echo "Per-GPU batch: ${PER_GPU_BATCH}, Global batch: ${GLOBAL_BATCH}, LR: ${SCALED_LR}, Epochs: ${EPOCHS}"

torchrun --nproc_per_node=${NUM_GPUS} train.py \
  --train-dir ${TRAIN_DIR} --val-dir ${VAL_DIR} \
  --batch-size ${PER_GPU_BATCH} --epochs ${EPOCHS} --lr ${SCALED_LR} --warmup-epochs ${WARMUP} \
  --save-dir ${SAVE_DIR} --tensorboard

