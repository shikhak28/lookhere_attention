Screening & evaluation quick guide
================================

Files added:
- `paper_run_commands.sh` - screening wrapper (already present in project). Use `--background` to start each method under `nohup` while the wrapper itself can be launched under `nohup`.
- `scripts/parse_screening.py` - extract minival Top-1 and throughput from `output/paper_screen_*/run.log`.
- `scripts/monitor_screening.sh` - simple monitor that prints a summary and tails logs every N seconds.

Quick steps to run screening (backgrounded, persistent):

1) Start the screening wrapper so it continues launching jobs after you logout:

```powershell
nohup bash ./paper_run_commands.sh /data/imagenet --background > output/screen_master.log 2>&1 &
```

2) Monitor progress from another session (or locally):

```bash
# run monitor in foreground (checks every 60s)
bash scripts/monitor_screening.sh 60

# or generate a CSV summary once jobs have produced logs
python3 scripts/parse_screening.py --out output/screening_summary.csv
```

3) When the wrapper finishes all methods, pick the winner by inspecting `screening_summary.csv` or the parse output.

Resuming the winner to 150 epochs (example for LH-180)

```bash
mkdir -p output/full_lh180_bs36
nohup bash -c "torchrun --nproc_per_node=7 /lookhere_attention/train.py \
  --data-dir /data/imagenet --preset paper --augment 3augment \
  --batch-size 36 --epochs 150 --num-workers 12 \
  --lr 3e-3 --weight-decay 0.05 \
  --save-dir output/full_lh180_bs36 \
  --resume output/paper_screen_LH-180_bs36/checkpoint_best.pth.tar" \
  > output/full_lh180_bs36/run.log 2> output/full_lh180_bs36/run.err &
```

Optional 5-epoch finetune at 384px (after 150-epoch completes):

```bash
mkdir -p output/finetune_lh180_384
nohup bash -c "torchrun --nproc_per_node=7 /lookhere_attention/train.py \
  --data-dir /data/imagenet --preset paper --augment 3augment \
  --batch-size 36 --epochs 5 --img-size 384 --num-workers 12 \
  --lr 3e-4 --weight-decay 0.05 \
  --save-dir output/finetune_lh180_384 \
  --resume output/full_lh180_bs36/checkpoint_best.pth.tar" \
  > output/finetune_lh180_384/run.log 2> output/finetune_lh180_384/run.err &
```

Notes & tips
- Make sure your `DATA_DIR` points to ImageNet training/val structure the training script expects.
- If you prefer parallelism, use GPU partitioning with `CUDA_VISIBLE_DEVICES` and unique `MASTER_PORT` per concurrent job — but that is more error-prone; sequential full-node jobs are simpler and robust.
- If a job fails, inspect `output/paper_screen_<method>_bs36/run.err` and `run.exit` for the exit code.

Want more? I can add:
- `scripts/eval.py` to run Top-1/Top-5 evaluation, ECE, and FGSM attacks automatically.
- A more robust parser that finds the best checkpoint file automatically and resumes training.

