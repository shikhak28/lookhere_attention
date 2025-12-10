#!/usr/bin/env bash
# paper_run_commands.sh
# Small helper to launch paper-style screening runs tuned to the chosen per-GPU batch size.
# Edit the METHODS array to include the methods you want to screen.

DATA_DIR=${1:-/data/imagenet}
# Pass --background as second arg to run torchrun jobs under nohup so they survive logout.
# Example: nohup bash ./paper_run_commands.sh /data/imagenet --background > master.log 2>&1 &
BG_ARG=${2:-}
BG_MODE=0
if [ "$BG_ARG" = "--background" ] || [ "$BG_ARG" = "-b" ]; then
  BG_MODE=1
  echo "Background mode: torchrun jobs will be started with nohup and script will wait for each to finish."
  echo "Recommended: start this script itself under nohup so it continues launching subsequent jobs after you logout."
fi
NUM_GPUS=7
BATCH_SIZE=36
# Screening epochs for this run: 50 (per your request).
## Screening epochs for this run: set to 12 as requested.
## Using 12 epochs shortens screening to finish sooner.
EPOCHS=12
NUM_WORKERS=12
LR=3e-3
WD=0.05
AUG=3augment
# Base MASTER_PORT for torchrun rendezvous. Each method will get a unique port = BASE_PORT + idx*10
BASE_PORT=29500

# Methods to run. Change names and extra args as needed.
# Shortlist for quick screening (fits limited GPU/time budget).
## Run one baseline + three LookHere variants as requested.
## Default baseline: 2D-sincos (change if you prefer another)
METHODS=(
  "2D-sincos"
  "2D-RoPE"
  "LH-180"
  "LH-90"
)

for idx in "${!METHODS[@]}"; do
  m=${METHODS[$idx]}
  outdir=output/paper_screen_${m}_bs${BATCH_SIZE}
  mkdir -p "$outdir"
  echo "Starting screening: method=$m bs=$BATCH_SIZE -> $outdir"
  # compute unique MASTER_PORT for this method to avoid address-in-use errors
  MASTER_PORT=$((BASE_PORT + idx * 10))
  echo "Using MASTER_PORT=${MASTER_PORT} for method $m"

  echo "Running method=$m (epochs=${EPOCHS}) -> $outdir"
  if [ $BG_MODE -eq 1 ]; then
    # Start the job with nohup so it survives logout, run in background and capture PID
    nohup bash -c "MASTER_PORT=${MASTER_PORT} torchrun --nproc_per_node=${NUM_GPUS} -- /lookhere_attention/train.py \
      --data-dir '$DATA_DIR' --preset paper --augment ${AUG} \
      --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --num-workers ${NUM_WORKERS} \
      --lr ${LR} --weight-decay ${WD} --save-dir '$outdir'" \
      > "$outdir/run.log" 2>"$outdir/run.err" &

    pid=$!
    echo $pid > "$outdir/run.pid"
    echo "Launched (nohup) PID $pid for method $m; waiting for it to finish..."

    # Wait for the background process to exit and capture exit code
    wait $pid
    rc=$?
    echo $rc > "$outdir/run.exit"
    if [ $rc -ne 0 ]; then
      echo "Run for method $m exited with code $rc. Check $outdir/run.err"
      exit $rc
    fi
    echo "Completed method $m (logs: $outdir/run.log)"
  else
    # Foreground run (current behavior)
    MASTER_PORT=${MASTER_PORT} torchrun --nproc_per_node=${NUM_GPUS} -- /lookhere_attention/train.py \
      --data-dir "$DATA_DIR" --preset paper --augment ${AUG} \
      --batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --num-workers ${NUM_WORKERS} \
      --lr ${LR} --weight-decay ${WD} --save-dir "$outdir" \
      > "$outdir/run.log" 2>"$outdir/run.err"

    rc=$?
    echo $rc > "$outdir/run.exit"
    if [ $rc -ne 0 ]; then
      echo "Run for method $m exited with code $rc. Check $outdir/run.err"
      exit $rc
    fi
    echo "Completed method $m (logs: $outdir/run.log)"
  fi
  # Short pause before next run
  sleep 5
done

echo "All screening jobs launched. Monitor logs under output/paper_screen_*/run.log"

