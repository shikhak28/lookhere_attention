#!/usr/bin/env bash
# quick_manual_tests.sh
# Abort any running batch_sweep/torchrun jobs, then run one-epoch tests for a set of per-GPU batch sizes.
# Usage: ./quick_manual_tests.sh /data/imagenet

DATA_DIR=${1:-/data/imagenet}
# mode: "simple" enables --simple-test (short run); default is full 1-epoch runs
MODE=${2:-full}
# optional override for num workers (third arg)
NUM_GPUS=7
NUM_WORKERS=${3:-8}
# optional simple-iters (fourth arg). Default to 200 for stable short runs
SIMPLE_ITERS=${4:-200}
SAVE_BASE=output/debug_throughput_manual_bs
# sweep inclusive 24..36 (13 sizes)
BATCH_SIZES=($(seq 24 36))

echo "Stopping any existing sweep/torchrun processes (may require privileges)..."
pkill -f batch_sweep.sh || true
pkill -f torchrun || true
sleep 1

for b in "${BATCH_SIZES[@]}"; do
  outdir="$SAVE_BASE$b"
  mkdir -p "$outdir"
  echo "\n=== Trying per-GPU batch-size $b ==="
  if [ "$MODE" = "simple" ] || [ "$MODE" = "--simple" ]; then
    SIMPLE_FLAG="--simple-test"
    SIMPLE_ITERS_FLAG="--simple-iters $SIMPLE_ITERS"
    echo "Mode: simple (short run) iters=$SIMPLE_ITERS"
  else
    SIMPLE_FLAG=""
    SIMPLE_ITERS_FLAG=""
  fi

  echo "Command: torchrun --nproc_per_node=$NUM_GPUS -- $(pwd)/train.py --data-dir $DATA_DIR --preset paper --augment 3augment --batch-size $b --epochs 1 --num-workers $NUM_WORKERS $SIMPLE_FLAG $SIMPLE_ITERS_FLAG --save-dir $outdir"

  # Run and capture stdout+stderr to files. If the command fails (OOM or other), break the loop.
  torchrun --nproc_per_node=$NUM_GPUS -- $(pwd)/train.py \
    --data-dir "$DATA_DIR" --preset paper --augment 3augment --batch-size $b --epochs 1 --num-workers $NUM_WORKERS $SIMPLE_FLAG $SIMPLE_ITERS_FLAG --save-dir "$outdir" \
    > "$outdir/run.log" 2>"$outdir/run.err" || {
      echo "Batch size $b failed (non-zero exit). See $outdir/run.err";
      echo "--- Last 200 lines of $outdir/run.err ---";
      tail -n 200 "$outdir/run.err" || true;
      echo "Stopping manual tests.";
      exit 0;
    }

  echo "Batch size $b succeeded. Log: $outdir/run.log"
  echo "--- Throughput summary for batch $b ---"
  grep -E "Epoch [0-9]+ finished|Throughput:" "$outdir/run.log" || true
  echo "========================================"
done

echo "Manual tests complete (no failures)."

