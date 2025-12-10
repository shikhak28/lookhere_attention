#!/usr/bin/env python3
"""
scripts/eval.py

Distributed-aware evaluation utilities: classification (Top-1/Top-5), ECE (15 bins), FGSM adversarial eval, and high-res extrapolation.

This script is safe to run under `torchrun --nproc_per_node=N` or as a single-process script.

It writes per-run results to JSON/CSV and supports distributed aggregation across ranks.

Example (7-GPU eval in background):
  MASTER_PORT=29510 torchrun --nproc_per_node=7 scripts/eval.py --checkpoint output/.../checkpoint_best.pth.tar \
    --data-dir /data/imagenet --task all --img-size 224 --batch-size 64 --out-json output/results_224.json

Example HR eval:
  MASTER_PORT=29511 torchrun --nproc_per_node=7 scripts/eval.py --checkpoint output/.../checkpoint_best.pth.tar \
    --data-dir /data/imagenet --val-dir /data/imagenet_hr --task hr --img-size 1024 --batch-size 8 --out-json output/results_hr1024.json
"""

import argparse
import json
import csv
import os
import sys
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets

# Ensure repo root is on sys.path so imports like `from data_prep import ImageNetDataset`
# work when running this script from the repository (scripts/ is not a package).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_prep import ImageNetDataset
from lookhere import LookHere

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def is_dist_avail_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def world_info_from_env():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]))
    return world_size, rank, local_rank


def maybe_init_distributed():
    # Initialize process group if torchrun provided envs
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        torch.distributed.init_process_group(backend=backend)


def build_model(args, device):
    lh_config = args.lh_config if hasattr(args, 'lh_config') and args.lh_config is not None else {
        "global_slope": 1.0,
        "layer_slopes": [0.1, 0.1],
        "head_directions": "none",
    }
    model = LookHere(device=device, lh_config=lh_config, img_size=args.img_size, patch_size=args.patch_size,
                     embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, num_classes=args.num_classes)
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location='cpu')
        state = ck.get('state_dict', ck)
        try:
            model.load_state_dict(state)
            if get_rank() == 0:
                print("Loaded checkpoint:", args.checkpoint)
        except Exception as e:
            # try stripping 'module.' prefixes
            new_state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(new_state)
            if get_rank() == 0:
                print("Loaded checkpoint after key fix")
    return model


def get_rank():
    return int(os.environ.get('RANK', '0'))


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', '1'))


def make_val_loader(args, distributed=False):
    val_dir = args.val_dir if args.val_dir else os.path.join(args.data_dir, 'val')
    if not os.path.isdir(val_dir):
        raise ValueError(f"Val directory not found: {val_dir}")
    val_dataset = ImageNetDataset(datasets.ImageFolder(val_dir), do_augment=False, img_size=args.img_size)
    if distributed:
        sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        sampler = None
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True)
    return loader, sampler


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate_distributed(args, model, device, n_bins=15):
    # returns dictionary of aggregated metrics across world
    distributed = get_world_size() > 1
    loader, sampler = make_val_loader(args, distributed=distributed)
    model.to(device)
    model.eval()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # prepare tensors for reduction
    total = torch.tensor(0, dtype=torch.long, device=device)
    top1_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    top5_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
    # histograms per-bin: counts, conf_sum, correct_sum
    bin_counts = torch.zeros(n_bins, dtype=torch.long, device=device)
    bin_conf_sum = torch.zeros(n_bins, dtype=torch.float32, device=device)
    bin_correct_sum = torch.zeros(n_bins, dtype=torch.float32, device=device)

    with torch.no_grad():
        if sampler is not None:
            sampler.set_epoch(0)
        for images, targets in tqdm(loader, desc='Validate'):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)
            acc1, acc5 = accuracy(outputs, targets, topk=(1,5))
            batch = images.size(0)
            total += batch
            top1_sum += acc1.item() * batch / 100.0
            top5_sum += acc5.item() * batch / 100.0
            # bin contributions
            confs_cpu = confs.detach().cpu().numpy()
            correct_cpu = (preds == targets).detach().cpu().numpy().astype(float)
            # compute per-bin sums on CPU then accumulate to GPU tensors
            for i in range(n_bins):
                lo, hi = bins[i], bins[i+1]
                if i < n_bins - 1:
                    mask = (confs_cpu >= lo) & (confs_cpu < hi)
                else:
                    mask = (confs_cpu >= lo) & (confs_cpu <= hi)
                cnt = int(mask.sum())
                if cnt == 0:
                    continue
                conf_sum = float(confs_cpu[mask].sum())
                correct_sum = float(correct_cpu[mask].sum())
                bin_counts[i] += cnt
                bin_conf_sum[i] += conf_sum
                bin_correct_sum[i] += correct_sum

    # reduce across processes
    if distributed:
        torch.distributed.barrier()
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(top1_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(top5_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(bin_counts, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(bin_conf_sum, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(bin_correct_sum, op=torch.distributed.ReduceOp.SUM)

    # move to CPU numpy for final metrics (only meaningful after reduction)
    total_n = int(total.item())
    top1 = 100.0 * (top1_sum.item() / total_n) if total_n > 0 else 0.0
    top5 = 100.0 * (top5_sum.item() / total_n) if total_n > 0 else 0.0

    # compute ECE from aggregated bins
    ece = 0.0
    if total_n > 0:
        for i in range(n_bins):
            cnt = float(bin_counts[i].item())
            if cnt == 0:
                continue
            avg_conf = (bin_conf_sum[i].item() / cnt)
            acc = (bin_correct_sum[i].item() / cnt)
            ece += (cnt / total_n) * abs(avg_conf - acc)

    return {
        'total': total_n,
        'top1': top1,
        'top5': top5,
        'ece': ece,
        'bins': {
            'counts': [int(x) for x in bin_counts.cpu().numpy()],
            'conf_sums': [float(x) for x in bin_conf_sum.cpu().numpy()],
            'correct_sums': [float(x) for x in bin_correct_sum.cpu().numpy()]
        }
    }


def fgsm_eval_distributed(args, model, device):
    # Run FGSM attack across distributed dataset and reduce counts
    distributed = get_world_size() > 1
    loader, sampler = make_val_loader(args, distributed=distributed)
    model.to(device)
    model.eval()
    eps_list = [float(x) for x in args.eps.split(',')]
    results = {}
    for eps in eps_list:
        correct = torch.tensor(0, dtype=torch.long, device=device)
        total = torch.tensor(0, dtype=torch.long, device=device)
        for images, targets in tqdm(loader, desc=f'FGSM eps={eps}'):
            images = images.to(device)
            targets = targets.to(device)
            images.requires_grad = True
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            model.zero_grad()
            loss.backward()
            grad = images.grad.data
            # ensure eps and clamping tensors use the same dtype as the images
            eps_c = torch.tensor(eps / std, device=device, dtype=images.dtype).view(1,3,1,1)
            perturbed = images + eps_c * torch.sign(grad)
            min_norm = (0.0 - mean) / std
            max_norm = (1.0 - mean) / std
            min_t = torch.tensor(min_norm, device=device, dtype=images.dtype).view(1,3,1,1)
            max_t = torch.tensor(max_norm, device=device, dtype=images.dtype).view(1,3,1,1)
            perturbed = torch.max(torch.min(perturbed, max_t), min_t)
            with torch.no_grad():
                out2 = model(perturbed)
                _, pred = out2.max(1)
                correct += (pred == targets).sum()
                total += targets.size(0)

        if distributed:
            torch.distributed.barrier()
            torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)

        results[eps] = 100.0 * (correct.item() / total.item()) if total.item() > 0 else 0.0

    return results


def write_results(out_path, results):
    d = os.path.dirname(out_path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='/data/imagenet')
    p.add_argument('--val-dir', type=str, default=None)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--task', type=str, choices=['classify','ece','fgsm','hr','all'], default='classify')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--patch-size', type=int, default=16)
    p.add_argument('--embed-dim', type=int, default=768)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--num-heads', type=int, default=12)
    p.add_argument('--num-classes', type=int, default=1000)
    p.add_argument('--eps', type=str, default='0.00392156862745098,0.011764705882352941')
    p.add_argument('--out-csv', type=str, default=None)
    p.add_argument('--out-json', type=str, default=None)
    args = p.parse_args()

    # Setup distributed if applicable
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        maybe_init_distributed()

    world_size = get_world_size()
    rank = get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    model = build_model(args, device)

    results = {}
    if args.task in ('classify','all','ece','hr'):
        # run validate (aggregates top1/top5/ece)
        info = validate_distributed(args, model, device, n_bins=15)
        results.update({k: info[k] for k in ('total','top1','top5','ece')})

    if args.task in ('fgsm','all'):
        fgsm_res = fgsm_eval_distributed(args, model, device)
        results['fgsm'] = fgsm_res

    # write outputs (only rank 0 writes files)
    if rank == 0:
        if args.out_json:
            write_results(args.out_json, results)
            print('Wrote JSON results to', args.out_json)
        if args.out_csv:
            # flatten and write simple CSV
            with open(args.out_csv, 'w', newline='') as f:
                w = csv.writer(f)
                for k, v in results.items():
                    w.writerow([k, v])
            print('Wrote CSV results to', args.out_csv)
        # also print summary
        print('Summary results:')
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()

