# t13 (P1-6) evaluation-matrix parallel driver.
# Splits the <model>:<set> grid across N python worker processes (resumable: finished keys are skipped).
#
# Usage (from repo root avsuite/, bypass execution policy):
#   powershell -NoProfile -ExecutionPolicy Bypass -File exp/07_nn_aec/run_matrix_eval.ps1 -Workers 6
#   ... -OnlyTags trad,v4        # subset of tags
#   ... -OutRoot D:/data/_smoke  # alternate output root (for smoke runs)
#
# Key priority: trad (model-independent baseline, once per set) -> v4 (stage-0 anchor)
#               -> NN v5/v5x/noMUSAN -> DCCRN v5/v5x.
# Per-worker logs: outputs/eval_matrix/_logs/worker<i>.{log,err.log}
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 reads BOM-less scripts as ANSI).
param(
    [int]$Workers = 6,
    [string[]]$OnlyTags = @(),
    [int]$Limit = 0,
    [string]$OutRoot = ""
)
$ErrorActionPreference = "Stop"
$py = "C:\Users\14805\.conda\envs\avsuite\python.exe"
$R = "D:/Project_for_Python/DSH/avsuite"
$P = "D:/Project_for_Python/DSH/p1_artifacts/outputs_p1"
$out = if ($OutRoot -ne "") { $OutRoot } else { "$R/exp/07_nn_aec/outputs/eval_matrix" }
$logs = "$out/_logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$sets = @("eval_v5","eval_v8","eval_ns","eval_rir","eval_rir_sim","eval_snr_low","eval_snr_high",
          "eval_music","eval_zh","eval_zh_both","eval_far_ls","eval_long_ls")
$tagOrder = @("trad","v4","nn_v5","nn_v5x","nn_v5x_noMUSAN","dccrn_v5","dccrn_v5x","casc_v5","casc_v5x")
if ($OnlyTags.Count -gt 0) {
    $want = @()
    foreach ($x in $OnlyTags) { $want += $x.Split(",") }
    $tagOrder = @($tagOrder | Where-Object { $want -contains $_ })
}

$keys = @()
foreach ($t in $tagOrder) { foreach ($s in $sets) { $keys += ($t + ":" + $s) } }

$cascV5 = "$P/casc_v5_official/nnaec_g0.5_j10_dm_casc/best.ckpt"
$cascV5X = "$P/casc_v5x_official/nnaec_g0.5_j10_dm_casc/best.ckpt"
$ckArgs = @(
    "--ckpt", "v4=$R/exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt",
    "--ckpt", "nn_v5=$P/nn_aec_v5/nnaec_g0.5_j10_dm/best.ckpt",
    "--ckpt", "nn_v5x=$P/nn_aec_v5x/nnaec_g0.5_j10_dm/best.ckpt",
    "--ckpt", "nn_v5x_noMUSAN=$P/nn_aec_v5x_noMUSAN/nnaec_g0.5_j10_dm/best.ckpt",
    "--ckpt", "dccrn_v5=$P/dccrn_dm/dccrn_dm/best.ckpt:dccrn",
    "--ckpt", "dccrn_v5x=$P/dccrn_v5x/dccrn_dm/best.ckpt:dccrn",
    "--ckpt", "casc_v5=${cascV5}:cascade",
    "--ckpt", "casc_v5x=${cascV5X}:cascade"
)
# NOTE: 变量后紧跟 ":cascade" 时必须写 ${var}——"$var:suffix" 会被解析成"驱动器:变量"而展开为空
# （PS var:drive 语法；已踩过：路径整体变空 -> ckpt 不存在：.）
# 级联 tag 的 ckpt 不存在时跳过（新服务器回拉中）；用 .NET 判断（Test-Path 对本目录曾出现假阴性）
$ckptOf = @{ "casc_v5" = $cascV5; "casc_v5x" = $cascV5X }
$tagOrder = @($tagOrder | Where-Object {
    $c = $ckptOf[$_]
    if ($c -and -not [System.IO.File]::Exists($c)) { Write-Output ("[matrix] skip tag $_ : ckpt 未到位 $c"); $false } else { $true }
})

Write-Output ("[matrix] {0} keys across {1} workers; out={2}" -f $keys.Count, $Workers, $out)
$procs = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $slice = @()
    for ($j = $i; $j -lt $keys.Count; $j += $Workers) { $slice += $keys[$j] }
    if ($slice.Count -eq 0) { continue }
    $argList = @("$R/exp/07_nn_aec/eval_matrix.py", "--out-root", $out, "--keys", ($slice -join ","))
    $argList += $ckArgs
    if ($Limit -gt 0) { $argList += @("--limit", "$Limit") }
    $env:OMP_NUM_THREADS = "3"
    $env:PYTHONIOENCODING = "utf-8"
    $p = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru `
        -RedirectStandardOutput "$logs/worker$i.log" -RedirectStandardError "$logs/worker$i.err.log"
    $procs += $p
    Write-Output ("[matrix] worker{0} pid={1} keys={2} first={3}" -f $i, $p.Id, $slice.Count, $slice[0])
}
$procs | Wait-Process
Write-Output "[matrix] all workers finished"
foreach ($t in $tagOrder) {
    $done = 0; $rows = 0
    foreach ($s in $sets) {
        $f = "$out/$t/$s.json"
        if (Test-Path $f) {
            $txt = [System.IO.File]::ReadAllText($f, [System.Text.Encoding]::UTF8)
            $done++
            $rows += ([regex]::Matches($txt, '"name":')).Count
        }
    }
    Write-Output ("[matrix] {0,-16} {1,2}/{2} sets, {3} rows" -f $t, $done, $sets.Count, $rows)
}
