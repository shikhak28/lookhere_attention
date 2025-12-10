#!/usr/bin/env bash
# batch_sweep.sh - try several per-GPU batch sizes sequentially and stop at first failure
# Usage: ./batch_sweep.sh /data/imagenet
# Logs: output/debug_throughput_bs${BATCH}.log and .err

DATA_DIR=${1:-/data/imagenet}
BATCH_SIZES=(16 20 24 28 32)
NUM_GPUS=7
NUM_WORKERS=8
SAVE_DIR_BASE=output/debug_throughput_bs

set -e

for b in "${BATCH_SIZES[@]}"; do
  outdir="$SAVE_DIR_BASE$b"
  mkdir -p "$outdir"
  echo "=== Trying per-GPU batch-size $b ==="
  echo "Command: torchrun --nproc_per_node=$NUM_GPUS -- \\$(pwd)/train.py --data-dir $DATA_DIR --preset paper --augment 3augment --batch-size $b --epochs 1 --num-workers $NUM_WORKERS --save-dir $outdir"

  # Run and capture stdout+stderr to files. If the command fails (OOM or other), break the loop.
  torchrun --nproc_per_node=$NUM_GPUS -- $(pwd)/train.py \
    --data-dir "$DATA_DIR" --preset paper --augment 3augment --batch-size $b --epochs 1 --num-workers $NUM_WORKERS --save-dir "$outdir" \
    > "$outdir/run.log" 2>"$outdir/run.err" || {
      echo "Batch size $b failed (non-zero exit). See $outdir/run.err";
      # Print last 200 lines of error to stdout for quick inspection
      echo "--- Last 200 lines of $outdir/run.err ---";
      tail -n 200 "$outdir/run.err" || true;
      echo "Stopping sweep.";
      exit 0;
    }

  echo "Batch size $b succeeded. Log: $outdir/run.log"
  # Print the epoch summary lines (search in the log)
  echo "--- Throughput summary for batch $b ---"
  grep -E "Epoch [0-9]+ finished|Throughput:" "$outdir/run.log" || true
  echo "========================================"

done

echo "Sweep complete (no failures)."

