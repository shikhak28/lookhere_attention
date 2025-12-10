import argparse
import json
import os
import time
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch import amp

from torchvision import datasets

from lookhere import LookHere
from data_prep import ImageNetDataset
import inspect
import data_prep as _data_prep_module

try:
    # Runtime diagnostic: print which data_prep file is being imported and the
    # ImageNetDataset.__init__ signature so we can debug argument mismatches.
    print(f"[diagnostic] data_prep module: {_data_prep_module.__file__}")
    print(f"[diagnostic] ImageNetDataset.__init__ signature: {inspect.signature(_data_prep_module.ImageNetDataset.__init__)}")
except Exception as _e:
    print("[diagnostic] Failed to inspect data_prep module:", _e)
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def parse_args():
    p = argparse.ArgumentParser(description="Train LookHere models")
    p.add_argument("--data-dir", type=str, default="/data/imagenet", help="Path to ImageNet root (contains 'train' and 'val' subfolders)")
    p.add_argument("--train-dir", type=str, default=None, help="Path to training ImageFolder (overrides --data-dir)")
    p.add_argument("--val-dir", type=str, default=None, help="Path to validation ImageFolder (overrides --data-dir/val)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--preset", type=str, default=None, help="Optional preset config: 'paper' to follow the paper's setup")
    p.add_argument("--augment", type=str, default="default", choices=["default","randaugment","3augment"], help="Augmentation choice for paper experiments")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=str, default="output")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=768)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--simple-test", action="store_true", help="Run a tiny smoke training for quick checks")
    p.add_argument("--simple-iters", type=int, default=0, help="When --simple-test, limit steps per epoch to this many iterations (0 = no limit)")
    p.add_argument("--lh-config", type=str, default=None, help="Path to LH config JSON (optional)")
    p.add_argument("--lh-weights", type=str, default=None, help="Path to authors' checkpoint to load LH config and/or weights")
    p.add_argument("--warmup-epochs", type=int, default=10, help="Linear warmup epochs before cosine schedule")
    p.add_argument("--tensorboard", action="store_true", help="Log scalars to TensorBoard if available")
    p.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for distributed training")
    return p.parse_args()


def make_dataloader(data_dir, img_size, batch_size, num_workers, do_augment=True):
    if data_dir is None:
        raise ValueError("--data-dir must point to an ImageNet root or use --train-dir/--val-dir")

    dataset = datasets.ImageFolder(root=data_dir)
    wrapped = ImageNetDataset(dataset, do_augment=do_augment, img_size=img_size)
    loader = DataLoader(wrapped, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader


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

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def save_checkpoint(state, save_dir, filename="checkpoint.pth.tar"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)


def load_lh_config_from_checkpoint(path):
    ck = torch.load(path, map_location="cpu")
    if isinstance(ck, dict) and "config" in ck:
        return ck
    return None


def train_one_epoch(train_loader, model, criterion, optimizer, device, scaler, epoch, args, log_interval=50, sampler=None, is_main_process=True):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    epoch_start = time.time()
    end = time.time()
    total_images = 0

    for i, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        batch_start = time.time()
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with amp.autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # metrics
        batch_size = images.size(0)
        total_images += batch_size
        loss_val = loss.item()
        a1, a5 = accuracy(outputs, targets, topk=(1, 5))
        a1_val = a1.item()
        a5_val = a5.item()

        loss_meter.update(loss_val, batch_size)
        acc1_meter.update(a1_val, batch_size)
        acc5_meter.update(a5_val, batch_size)

        batch_time.update(time.time() - batch_start)
        end = time.time()

        # Print per-step log. In distributed runs each process prints its own logs (one per GPU).
        if (i + 1) % log_interval == 0 or i == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})"
            )

        # Support limiting iterations when running a quick `--simple-test`.
        # If args.simple_iters > 0, stop the epoch after that many iterations.
        try:
            simple_limit = int(getattr(args, "simple_iters", 0))
        except Exception:
            simple_limit = 0
        if simple_limit > 0 and (i + 1) >= simple_limit:
            if is_main_process:
                print(f"[simple-test] Reached {i+1} iterations; stopping epoch early.")
            break

    epoch_dur = time.time() - epoch_start
    # throughput: images/sec per process
    throughput_per_proc = total_images / epoch_dur if epoch_dur > 0 else 0.0

    if is_main_process:
        print(f"Epoch {epoch} finished. Time {epoch_dur:.1f}s, Throughput per process: {throughput_per_proc:.2f} img/s")

    return {
        'epoch_time': epoch_dur,
        'throughput_per_proc': throughput_per_proc,
        'train_loss_avg': loss_meter.avg,
        'train_acc1_avg': acc1_meter.avg,
        'train_acc5_avg': acc5_meter.avg,
    }


def validate(val_loader, model, criterion, device, is_main_process=True):
    model.eval()
    total_loss = 0.0
    top1 = 0.0
    top5 = 0.0
    n = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            a1, a5 = accuracy(outputs, targets, topk=(1, 5))
            top1 += a1.item() * images.size(0) / 100.0
            top5 += a5.item() * images.size(0) / 100.0
            n += images.size(0)

    avg_loss = total_loss / n if n > 0 else 0.0
    top1 = 100.0 * top1 / n if n > 0 else 0.0
    top5 = 100.0 * top5 / n if n > 0 else 0.0
    if is_main_process:
        print(f"Validation: Loss {avg_loss:.4f} Top1 {top1:.2f}% Top5 {top5:.2f}%")
    return avg_loss, top1, top5


def main():
    args = parse_args()

    # Distributed setup
    distributed = False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        distributed = True

    local_rank = args.local_rank
    if distributed:
        # Verify CUDA is available and PyTorch has CUDA support before setting device
        if not torch.cuda.is_available():
            msg = (
                "Distributed training requested but CUDA is not available.\n"
                "This usually means you have a CPU-only PyTorch build or CUDA drivers are missing.\n"
                "Check `python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"`\n"
                "To install a CUDA-enabled PyTorch, see https://pytorch.org/get-started/locally or use one of the commands below:\n"
                "  conda (recommended): conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia\n"
                "  pip (example for CUDA 11.8): pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision --upgrade\n"
            )
            raise RuntimeError(msg)
        try:
            torch.cuda.set_device(local_rank)
        except Exception as e:
            raise RuntimeError(f"Failed to set CUDA device for local_rank={local_rank}: {e}") from e
        dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device(args.device)

    if args.simple_test:
        args.epochs = 2
        args.batch_size = min(args.batch_size, 8)

    # create model and LH config
    lh_config = {
        "global_slope": 1.0,
        "layer_slopes": [0.1, 0.1],
        "head_directions": "none",
    }

    # If user provided a JSON LH config file, load it
    if args.lh_config is not None:
        with open(args.lh_config, "r") as f:
            lh_config = json.load(f)

    # If user passed an authors' checkpoint, try to extract config and optionally weights
    preload_weights = None
    if args.lh_weights is not None:
        ck = load_lh_config_from_checkpoint(args.lh_weights)
        if ck is not None:
            if "config" in ck:
                lh_config = ck["config"]
            if "weights" in ck:
                preload_weights = ck["weights"]
            elif "state_dict" in ck:
                preload_weights = ck["state_dict"]

    model = LookHere(device=device, lh_config=lh_config, img_size=args.img_size, patch_size=args.patch_size,
                     embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, num_classes=args.num_classes)

    if preload_weights is not None:
        try:
            model.load_state_dict(preload_weights)
            print("Loaded pretrained weights from", args.lh_weights)
        except Exception as e:
            print("Warning: failed to load pretrained weights:", e)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine with linear warmup
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))

    def lr_lambda(current_epoch):
        if current_epoch < args.warmup_epochs:
            return float(current_epoch) / float(max(1, args.warmup_epochs))
        else:
            # let cosine scheduler handle remaining epochs
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # We'll step the cosine scheduler after warmup manually inside loop
    scaler = amp.GradScaler()

    writer = None
    if args.tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs"))

    # data
    # Determine training and validation folders. Prefer explicit args if provided.
    if args.train_dir:
        train_path = args.train_dir
    else:
        possible = os.path.join(args.data_dir, "train")
        train_path = possible if os.path.isdir(possible) else args.data_dir

    if args.val_dir:
        val_path = args.val_dir
    else:
        val_path = os.path.join(args.data_dir, "val")

    # Apply preset adjustments
    if args.preset == "paper":
        # Paper trains ViT-B/16 for 150 epochs on 224 px. Use best-known defaults unless overridden.
        args.epochs = getattr(args, 'epochs', 150) or 150
        args.img_size = 224
        # If user didn't override lr or wd, set paper best values
        if 'lr' in args and (args.lr is None or args.lr == 5e-4):
            args.lr = 3.0e-3
        else:
            # user supplied lr, keep it
            pass
        if 'weight_decay' in args and (args.weight_decay is None or args.weight_decay == 0.05):
            args.weight_decay = 0.05
        args.patch_size = 16

    # Build datasets and dataloaders (use DistributedSampler when distributed)
    # If the user requested the paper preset and did not provide an explicit validation folder,
    # split the training ImageFolder into 99% train / 1% minival (deterministic split using existing ordering).
    val_loader = None
    train_imagefolder = datasets.ImageFolder(train_path)
    if args.preset == "paper" and not args.val_dir:
        # compute split sizes
        total = len(train_imagefolder)
        minival_size = max(1, int(total * 0.01))
        train_size = total - minival_size
        # create indices: keep last 1% as minival to match "holding the last 1%" from paper
        indices = list(range(total))
        train_indices = indices[:train_size]
        minival_indices = indices[train_size:]
        from torch.utils.data import Subset
        train_subset = Subset(train_imagefolder, train_indices)
        minival_subset = Subset(train_imagefolder, minival_indices)
        train_dataset = ImageNetDataset(train_subset, do_augment=True, img_size=args.img_size, augment_type=args.augment)
        val_dataset = ImageNetDataset(minival_subset, do_augment=False, img_size=args.img_size)
    else:
        train_dataset = ImageNetDataset(train_imagefolder, do_augment=True, img_size=args.img_size, augment_type=args.augment)
        if os.path.isdir(val_path):
            val_dataset = ImageNetDataset(datasets.ImageFolder(val_path), do_augment=False, img_size=args.img_size)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        if os.path.isdir(val_path):
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        if os.path.isdir(val_path):
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck.get("state_dict", ck))
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"]) 
        start_epoch = ck.get("epoch", 0)

    best_top1 = 0.0

    # Wrap model with DDP after moving to device
    if distributed:
        model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        is_main_process = (dist.get_rank() == 0)
    else:
        model.to(device)
        is_main_process = True

    # training loop
    for epoch in range(start_epoch, args.epochs):
        stats = train_one_epoch(train_loader, model, criterion, optimizer, device, scaler, epoch, args, sampler=(train_sampler if distributed else None), is_main_process=is_main_process)

        # Scheduler stepping: first warmup then cosine
        if epoch < args.warmup_epochs:
            scheduler.step()
            if epoch + 1 == args.warmup_epochs:
                # replace lambda scheduler with cosine for remaining epochs
                # adjust cosine T_max to remaining epochs
                cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch - 1))
        else:
            cosine.step()

        # Validation at epoch end (only log on main process)
        if val_loader is not None:
            # if distributed, ensure evaluation uses model.module
            eval_model = model.module if distributed else model
            _, top1, top5 = validate(val_loader, eval_model, criterion, device, is_main_process=is_main_process)
            is_best = top1 > best_top1 if is_main_process else False
            if is_main_process and is_best:
                best_top1 = top1

        # Print final per-epoch summary including validation and throughput (main process)
        if is_main_process:
            tp = stats.get('throughput_per_proc', 0.0)
            # If distributed, compute global throughput
            global_tp = tp
            if distributed and dist.is_initialized():
                try:
                    world_size = dist.get_world_size()
                    global_tp = tp * world_size
                except Exception:
                    global_tp = tp

            print(f"Epoch {epoch} Summary: Train Loss {stats.get('train_loss_avg',0):.4f} "
                  f"Acc@1 {stats.get('train_acc1_avg',0):.3f} Acc@5 {stats.get('train_acc5_avg',0):.3f}")
            print(f"Throughput: {tp:.2f} img/s per process, {global_tp:.2f} img/s global")
        else:
            is_best = False

        # checkpoint (only main process writes)
        if is_main_process:
            # core state dict (may be on GPU)
            state_dict = (model.module.state_dict() if distributed else model.state_dict())

            # create a CPU-only copy of weights suitable for sharing like the author's checkpoints
            try:
                weights_cpu = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in state_dict.items()}
            except Exception:
                # fallback: save the original state_dict if CPU conversion fails
                weights_cpu = state_dict

            # include LH config and a weights field so saved checkpoints look like author checkpoints
            ck = {
                "epoch": epoch + 1,
                "state_dict": state_dict,
                "optimizer": optimizer.state_dict(),
                # author-like fields
                "config": getattr(globals().get('lh_config', None), '__dict__', lh_config) if 'lh_config' in globals() or 'lh_config' in locals() else lh_config,
                "weights": weights_cpu,
                # helpful metadata
                "img_size": getattr(args, 'img_size', None),
                "patch_size": getattr(args, 'patch_size', None),
                "grid_size": getattr(model, 'grid_size', None),
            }

            save_checkpoint(ck, args.save_dir, filename=f"checkpoint_epoch_{epoch+1}.pth.tar")
            save_checkpoint(ck, args.save_dir, filename="checkpoint_last.pth.tar")
            if is_best:
                save_checkpoint(ck, args.save_dir, filename="checkpoint_best.pth.tar")

            if writer is not None and val_loader is not None:
                writer.add_scalar("val/top1", best_top1, epoch)



if __name__ == "__main__":
    main()

