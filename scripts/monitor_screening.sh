#!/usr/bin/env bash
# monitor_screening.sh
# Tail the latest logs for each screening method and show a brief summary using parse_screening.py

INTERVAL=${1:-60}
echo "Monitoring screening logs every ${INTERVAL}s. Press Ctrl-C to stop."

while true; do
  date
  echo "--- Summary (parse_screening.py) ---"
  python3 scripts/parse_screening.py
  echo "--- Tailing last 30 lines of each log ---"
  for d in output/paper_screen_*_bs*; do
    if [ -d "$d" ]; then
      echo "== $d =="
      tail -n 30 "$d/run.log" 2>/dev/null || echo "(no run.log yet)"
      echo
    fi
  done
  sleep ${INTERVAL}
done

