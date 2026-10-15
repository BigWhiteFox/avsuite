# P0-5：把损失矩阵的 AECMOS 逐文件 json 汇入 compute_ci.py 的扫描集合。
# 源：exp/07_nn_aec/outputs_v4_p05/aecmos/<set>/aecmos_<variant>.json
# 目标：exp/07_nn_aec/outputs/p05/<set>/aecmos_<variant>.json（compute_ci.py PAIRS 用相对路径寻址）
# 用复制而非移动：outputs_v4_p05/ 是训练产物落点，outputs/p05/ 是评测口径归档点。
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
$src = "exp/07_nn_aec/outputs_v4_p05/aecmos"
$dst = "exp/07_nn_aec/outputs/p05"
if (-not (Test-Path $src)) { throw "缺源目录：$src（先跑 eval_p05_matrix.py）" }
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$n = 0
Get-ChildItem -Path $src -Recurse -Filter "aecmos_*.json" | ForEach-Object {
  $rel = $_.FullName.Substring((Resolve-Path $src).Path.Length + 1)
  $target = Join-Path $dst $rel
  New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
  Copy-Item $_.FullName $target -Force
  $n++
}
Write-Host "copied $n json -> $dst"
Get-ChildItem -Path $dst -Recurse -File | Select-Object FullName | Format-Table -AutoSize
