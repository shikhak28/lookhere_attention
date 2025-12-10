#!/usr/bin/env bash
# scripts/resume_winner_150.sh
# Resume a winner to 150 epochs (preferred) or initialize from an author checkpoint.
# Usage:
#  scripts/resume_winner_150.sh <winner_outdir> [--author LH_180_weights_and_config.pth] [--gpus N] [--bg]
# Examples:
#  scripts/resume_winner_150.sh output/paper_screen_2D-sincos_bs36 --gpus 7 --bg
#  scripts/resume_winner_150.sh output/paper_screen_LH180_bs36 --author LH_180_weights_and_config.pth --gpus 7 --bg

set -euo pipefail
OUTDIR=""
AUTHOR_PTH=""
NGPUS=7
BG_MODE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --author)
      AUTHOR_PTH="$2"
      shift 2
      ;;
    --gpus)
      NGPUS="$2"
      shift 2
      ;;
    --bg)
      BG_MODE=1
      shift
      ;;
    --help|-h)
      echo "Usage: $0 <winner_outdir> [--author AUTHOR_PTH] [--gpus N] [--bg]"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1"
      exit 1
      ;;
    *)
      if [ -z "$OUTDIR" ]; then
        OUTDIR="$1"
        shift
      else
        echo "Multiple non-option args: $1"
        exit 1
      fi
      ;;
  esac
done

if [ -z "$OUTDIR" ]; then
  echo "Usage: $0 <winner_outdir> [--author AUTHOR_PTH] [--gpus N] [--bg]"
  exit 1
fi

mkdir -p "$OUTDIR"

# choose master port from timestamp to reduce collisions
BASE_PORT=29500
PORT_OFFSET=$(( $(date +%s) % 1000 ))
MASTER_PORT=$(( BASE_PORT + PORT_OFFSET ))

# If there's a resume checkpoint produced by train.py, prefer that (preserves optimizer)
RESUME_CKPT="$OUTDIR/checkpoint_last.pth.tar"
LAUNCH_ARGS=( --data-dir "/data/imagenet" --preset paper --augment 3augment --batch-size 36 --epochs 150 --num-workers 12 --lr 3e-3 --weight-decay 0.05 --save-dir "$OUTDIR" )

if [ -f "$RESUME_CKPT" ]; then
  echo "Found resume checkpoint: $RESUME_CKPT -> resuming training (keeps optimizer state)."
  LAUNCH_ARGS+=( --resume "$RESUME_CKPT" )
elif [ -n "$AUTHOR_PTH" ]; then
  if [ ! -f "$AUTHOR_PTH" ]; then
    echo "Author checkpoint not found: $AUTHOR_PTH"
    exit 1
  fi
  echo "No resume checkpoint found. Initializing from author checkpoint: $AUTHOR_PTH"
  LAUNCH_ARGS+=( --lh-weights "$AUTHOR_PTH" )
else
  echo "No resume checkpoint found and no author checkpoint specified. Either provide --author <file> or put a resume checkpoint at $RESUME_CKPT"
  exit 1
fi

# Build the torchrun command
CMD=( torchrun --nproc_per_node=${NGPUS} --master_port=${MASTER_PORT} "$(pwd)/train.py" )
CMD+=( "${LAUNCH_ARGS[@]}" )

if [ $BG_MODE -eq 1 ]; then
  echo "Launching background job in $OUTDIR with MASTER_PORT=${MASTER_PORT}"
  scripts/run_bg_torchrun.sh "$OUTDIR" ${MASTER_PORT} ${NGPUS} -- "${CMD[@]}"
  echo "Waiting for run.pid..."
  until [ -s "$OUTDIR/run.pid" ]; do sleep 1; done
  pid=$(cat "$OUTDIR/run.pid")
  echo "Launched (PID=$pid). Monitoring run.exit..."
  until [ -f "$OUTDIR/run.exit" ]; do sleep 10; done
  rc=$(cat "$OUTDIR/run.exit" || echo 1)
  if [ "$rc" -ne 0 ]; then
    echo "Run exited with code $rc. Check $OUTDIR/run.err"
    exit $rc
  fi
  echo "Run completed successfully. Logs: $OUTDIR/run.log"
else
  echo "Running foreground: MASTER_PORT=${MASTER_PORT}"
  MASTER_PORT=${MASTER_PORT} torchrun --nproc_per_node=${NGPUS} "$(pwd)/train.py" "${LAUNCH_ARGS[@]}"
  rc=$?
  echo $rc > "$OUTDIR/run.exit"
  if [ $rc -ne 0 ]; then
    echo "Run failed with code $rc"
    exit $rc
  fi
fi

echo "Finished 150-epoch job for $OUTDIR"

