# P0-5 损失矩阵三列重训（train_v1 不动数据）——顺序跑，GPU 独占。
# 用法： pwsh -File exp/07_nn_aec/run_p05_matrix.ps1 -Cols control,psycho
param([string[]]$Cols = @("control", "psycho"))

$ErrorActionPreference = "Continue"
$py = "C:\Users\14805\.conda\envs\avsuite\python.exe"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path   # repo root = avsuite/
Set-Location $root
$outroot = "exp/07_nn_aec/outputs_v4_p05"
$common = @(
  "--data", "D:/data/aec-synth/train_v1",
  "--val", "D:/data/aec-synth/eval_v3",
  "--gamma", "0.5", "--w-sisnr", "1.0", "--epochs", "240",
  "--batch-size", "16", "--lr", "2e-4", "--scenarios", "dt,nst",
  "--early-stop", "25", "--device", "cuda", "--seed", "0"
)

function Run-Col([string]$name, [string[]]$extra) {
  $log = "$outroot/train_$name.log"
  Write-Host "=== [$name] start $(Get-Date -Format o) -> $log"
  & $py exp/07_nn_aec/train.py @common "--out" "$outroot/$name" @extra *>&1 |
    Tee-Object -FilePath $log
  Write-Host "=== [$name] exit=$LASTEXITCODE $(Get-Date -Format o)"
}

foreach ($c in $Cols) {
  switch ($c) {
    "control" { Run-Col "lossctrl" @("--loss", "weighted") }
    "v4replica" {
      # 复刻校验：从 v3 的 last.ckpt 续训到 240ep（v4 的原始路径）。
      # 环境/数据逐位一致时，应复现 v4 的 best val 8.5852（否则本轮读数不可归因于损失）。
      Run-Col "lossctrl_v4replica" @("--loss", "weighted",
        "--resume", "exp/07_nn_aec/outputs_v3/nnaec_g0.5_j10/last.ckpt")
    }
    "psycho" { Run-Col "psycho" @("--loss", "psycho") }
    default { Write-Host "unknown col: $c" }
  }
}
Write-Host "=== all done $(Get-Date -Format o)"
