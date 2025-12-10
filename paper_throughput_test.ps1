<#
Run a single-epoch throughput test to measure epoch time and choose a safe per-GPU batch size.

Usage (PowerShell):
  .\paper_throughput_test.ps1 -DataDir 'C:\path\to\imagenet' -NumGpus 7 -PerGpuBatch 16

This runs a single-epoch training job using the `--preset paper` defaults and prints epoch throughput.
Increase `-PerGpuBatch` until you get OOM, then back off one step.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$DataDir,
    [int]$NumGpus = 7,
    [int]$PerGpuBatch = 16,
    [int]$NumWorkers = 8
)

Write-Output "Running throughput test: DataDir=$DataDir NumGpus=$NumGpus PerGpuBatch=$PerGpuBatch"

$cmd = "torchrun --nproc_per_node=$NumGpus train.py --data-dir `"$DataDir`" --preset paper --augment 3augment --lr 0.003 --weight-decay 0.05 --batch-size $PerGpuBatch --epochs 1 --num-workers $NumWorkers --save-dir output/debug_throughput"

Write-Output "Command: $cmd"
Invoke-Expression $cmd

Write-Output "Throughput test finished. Inspect the printed epoch summary for per-process and global throughput."

