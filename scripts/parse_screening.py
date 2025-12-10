#!/usr/bin/env python3
"""
parse_screening.py

Scan `output/paper_screen_*_bs*/run.log` and extract the latest Minival Top-1 and Throughput values.
Usage: python scripts/parse_screening.py [--out csv]
"""
import re
import csv
import sys
from pathlib import Path

OUT_CSV = None
if len(sys.argv) > 1 and sys.argv[1] == '--out':
    OUT_CSV = sys.argv[2]

root = Path('output')
rows = []

minival_re = re.compile(r"Minival(?:.*Top-1|).*?(\d{1,3}\.\d+)")
throughput_re = re.compile(r"Throughput(?: per process)?:\s*([0-9]+\.?[0-9]*)")

for d in sorted(root.glob('paper_screen_*_bs*')):
    log = d / 'run.log'
    if not log.exists():
        continue
    text = log.read_text(errors='ignore')
    minival_matches = minival_re.findall(text)
    tp_matches = throughput_re.findall(text)
    minival = minival_matches[-1] if minival_matches else ''
    tp = tp_matches[-1] if tp_matches else ''
    rows.append((d.name, minival, tp, str(log)))

if OUT_CSV:
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method_dir', 'minival_top1', 'throughput', 'log_path'])
        for r in rows:
            w.writerow(r)
    print(f'Wrote summary to {OUT_CSV}')
else:
    print('method_dir,minival_top1,throughput,log_path')
    for r in rows:
        print(','.join(r))

