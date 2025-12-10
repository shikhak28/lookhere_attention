#!/usr/bin/env bash
# scripts/run_eval_winner.sh
# Helper to run ImageNet 224x224 evaluation (Top-1/Top-5 + ECE) for a winner output dir.
# Usage:
#  scripts/run_eval_winner.sh --outdir /lookhere_attention/output/LH_180_150_author --gpus 7 --data-dir /data/imagenet
# Options:
#  --outdir    Directory containing checkpoints (default: ./output)
#  --checkpoint  Specific checkpoint path (optional). If omitted, will prefer <outdir>/checkpoint_best.pth.tar then checkpoint_last.pth.tar
#  --gpus      Number of GPUs to use (default: 7)
#  --batch     Per-process batch size for eval (default: 64)
#  --workers   DataLoader num_workers (default: 12)
#  --bg        Run under scripts/run_bg_torchrun.sh (background wrapper)

set -euo pipefail
OUTDIR="./output"
CKPT=""
NGPUS=7
BATCH=64
NUM_WORKERS=12
BG=0
DATA_DIR="/data/imagenet"
MASTER_PORT_BASE=29510

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)
      OUTDIR="$2"; shift 2;;
    --checkpoint)
      CKPT="$2"; shift 2;;
    --gpus)
      NGPUS="$2"; shift 2;;
    --batch)
      BATCH="$2"; shift 2;;
    --workers)
      NUM_WORKERS="$2"; shift 2;;
    --data-dir)
      DATA_DIR="$2"; shift 2;;
    --bg)
      BG=1; shift;;
    --help|-h)
      echo "Usage: $0 --outdir <dir> [--checkpoint <path>] [--gpus N] [--batch B] [--workers W] [--data-dir DIR] [--bg]"; exit 0;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

if [ ! -d "$OUTDIR" ]; then
  echo "Outdir not found: $OUTDIR"; exit 1
fi

# locate checkpoint if not provided
if [ -z "$CKPT" ]; then
  if [ -f "$OUTDIR/checkpoint_best.pth.tar" ]; then
    CKPT="$OUTDIR/checkpoint_best.pth.tar"
  elif [ -f "$OUTDIR/checkpoint_last.pth.tar" ]; then
    CKPT="$OUTDIR/checkpoint_last.pth.tar"
  else
    # try any checkpoint_epoch_*.pth.tar
    ck=$(ls -1 "$OUTDIR"/checkpoint_epoch_*.pth.tar 2>/dev/null | head -n 1 || true)
    if [ -n "$ck" ]; then
      CKPT="$ck"
    else
      echo "No checkpoint found in $OUTDIR. Provide --checkpoint or place a checkpoint in the outdir."; exit 1
    fi
  fi
fi

OUT_JSON="$OUTDIR/results_224.json"
OUT_CSV="$OUTDIR/results_224.csv"
LOG="$OUTDIR/run.log"
ERR="$OUTDIR/run.err"

# create a somewhat unique MASTER_PORT
PORT_OFFSET=$(( $(date +%s) % 1000 ))
MASTER_PORT=$(( MASTER_PORT_BASE + PORT_OFFSET ))

echo "Running eval for checkpoint: $CKPT"
echo "Results -> $OUT_JSON, logs -> $LOG / $ERR"

# build two variants: one that is the full torchrun invocation (for foreground)
eval_cmd=( torchrun --nproc_per_node=${NGPUS} --master_port=${MASTER_PORT} scripts/eval.py \
  --checkpoint "$CKPT" --data-dir "$DATA_DIR" --task all --img-size 224 --batch-size ${BATCH} --num-workers ${NUM_WORKERS} --out-csv "$OUT_CSV" )
# and one that is the inner script+args (for the background wrapper which will prefix torchrun)
eval_inner=( scripts/eval.py --checkpoint "$CKPT" --data-dir "$DATA_DIR" --task all --img-size 224 --batch-size ${BATCH} --num-workers ${NUM_WORKERS} --out-csv "$OUT_CSV" )

if [ $BG -eq 1 ]; then
  echo "Launching background job via scripts/run_bg_torchrun.sh"
  # pass the inner script and args (the wrapper will add torchrun)
  scripts/run_bg_torchrun.sh "$OUTDIR" ${MASTER_PORT} ${NGPUS} -- "${eval_inner[@]}"
  echo "Waiting for run.pid to appear..."
  until [ -s "$OUTDIR/run.pid" ]; do sleep 1; done
  pid=$(cat "$OUTDIR/run.pid")
  echo "Launched (PID=$pid). Monitor $OUTDIR/run.log and $OUTDIR/run.err"
else
  echo "Running foreground: MASTER_PORT=${MASTER_PORT}"
  MASTER_PORT=${MASTER_PORT} "${eval_cmd[@]}" > "$LOG" 2> "$ERR"
  rc=$?
  echo $rc > "$OUTDIR/run.exit"
  if [ $rc -ne 0 ]; then
    echo "Eval failed with code $rc. See $ERR"; exit $rc
  fi
  echo "Eval completed. Results at $OUT_JSON"
fi

# Print concise summary if results exist
if [ -f "$OUT_CSV" ]; then
  echo "Summary results (from CSV):"
  python - <<PY
import csv
path = r'''$OUT_CSV'''
res = {}
with open(path, newline='') as f:
    r=csv.reader(f)
    for row in r:
        if len(row) >= 2:
            key = row[0]
            val = row[1]
            res[key]=val
print(f"Total images: {res.get('total')}")
print(f"Top1: {res.get('top1')}")
print(f"Top5: {res.get('top5')}")
print(f"ECE: {res.get('ece')}")
PY
elif [ -f "$OUT_JSON" ]; then
  echo "Summary results (from JSON):"
  python - <<PY
import json
r=json.load(open('$OUT_JSON'))
print('Total images:', r.get('total'))
print(f"Top1: {r.get('top1'):.2f}%")
print(f"Top5: {r.get('top5'):.2f}%")
print(f"ECE: {r.get('ece'):.6f} (={r.get('ece')*100:.4f}% )")
PY
else
  echo "No result CSV/JSON found. Check $LOG and $ERR for details."
fi

