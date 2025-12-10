#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <SAVE_DIR> [MASTER_PORT] [NPROC] -- <train.py args...>"
  echo "Example: $0 output/paper_screen_2D-sincos_bs36 29500 1 -- /lookhere_attention/train.py --data-dir /data/imagenet --preset paper --batch-size 8 ..."
  exit 1
fi

SAVE_DIR=$1
shift || true

# Optional MASTER_PORT (default 29500)
MASTER_PORT=29500
if [ $# -ge 1 ] && [[ $1 =~ ^[0-9]+$ ]]; then
  MASTER_PORT=$1
  shift || true
fi

# Optional NPROC (default 1)
NPROC=1
if [ $# -ge 1 ] && [[ $1 =~ ^[0-9]+$ ]]; then
  NPROC=$1
  shift || true
fi

mkdir -p "$SAVE_DIR"
LOG="$SAVE_DIR/run.log"
ERR="$SAVE_DIR/run.err"
PIDFILE="$SAVE_DIR/run.pid"

if [ "$#" -eq 0 ]; then
  echo "No command supplied to run. Provide the train.py invocation after the args. Exiting." >&2
  exit 1
fi

# Build the torchrun command string
CMD="env MASTER_PORT=${MASTER_PORT} torchrun --nproc_per_node=${NPROC} -- $*"
echo "Starting background run in '$SAVE_DIR'" >&2
echo "Command: $CMD" >&2
echo "Logs: $LOG (stdout), $ERR (stderr)" >&2

# Run under nohup so it survives logout; redirect stdout/stderr into save dir
nohup bash -lc "$CMD" >"$LOG" 2>"$ERR" &
PID=$!
echo $PID > "$PIDFILE"
echo "Started (PID=$PID). Tail logs with: tail -F $LOG" >&2

exit 0

