<#
Run the 8 screening jobs (50 epochs) for one or more methods sequentially.

Usage (PowerShell):
  .\paper_quick_sweep.ps1 -DataDir 'C:\path\to\imagenet' -Methods @('LH-180') -NumGpus 7 -PerGpuBatch 16

Notes:
- The script runs commands sequentially and waits for each to finish.
- It uses placeholder `configs/{method}.json` as `--lh-config`. Ensure you have appropriate configs for each method or remove the flag and modify model selection in code.
- Adjust `PerGpuBatch` after running the throughput test (`paper_throughput_test.ps1`).
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$DataDir,
    [Parameter(Mandatory=$true)]
    [string[]]$Methods,
    [int]$NumGpus = 7,
    [int]$PerGpuBatch = 16,
    [int]$Epochs = 50,
    [int]$NumWorkers = 8,
    [string]$SaveRoot = "output/paper_quick_sweep"
)

$augs = @("randaugment","3augment")
$lrs = @(0.0015, 0.003)
$wds = @(0.02, 0.05)

foreach ($method in $Methods) {
    Write-Output "Starting sweeps for method: $method"
    foreach ($aug in $augs) {
        foreach ($lr in $lrs) {
            foreach ($wd in $wds) {
                $save_dir = Join-Path $SaveRoot (Join-Path $method ("{0}_aug-{1}_lr-{2}_wd-{3}" -f $method, $aug, $lr, $wd))
                # placeholder config path; adjust if your code expects other flags
                $config = "configs/$method.json"
                $cmd = "torchrun --nproc_per_node=$NumGpus train.py --data-dir `"$DataDir`" --preset paper --augment $aug --lr $lr --weight-decay $wd --batch-size $PerGpuBatch --epochs $Epochs --num-workers $NumWorkers --save-dir `"$save_dir`" --lh-config `"$config`""
                Write-Output "\n=== Running: method=$method aug=$aug lr=$lr wd=$wd ==="
                Write-Output $cmd
                try {
                    Invoke-Expression $cmd
                }
                catch {
                    Write-Output "Command failed for method=$method aug=$aug lr=$lr wd=$wd: $_"
                    # Continue to next combination
                }
            }
        }
    }
    Write-Output "Completed sweeps for method: $method"
}

Write-Output "All requested method sweeps completed. Check per-run outputs under $SaveRoot"

