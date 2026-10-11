# eval_v5_dt300 — DT 扩样评估集（n=300，DT 154）三方重评

- 日期：2026-10-11 · 执行：WS-B · 目的：解决 DT 评估样本过少（eval_v3 n=14）问题
- 数据：`D:/data/aec-synth/eval_v5`（300 条；**DT 154 / NST 76 / FEST 70**，manifest 实测）
- 约束遵守：eval_v3 及其结果只读未动；未改 docs/NUMBERS.md 与任何既有文档。

## 1. 合成配方（与 eval_v3 差异仅 num / scenarios / seed）

| 参数 | eval_v3 | eval_v5 |
|---|---|---|
| num | 60 | **300** |
| scenarios（FEST,NST,DT） | 0.5,0.25,0.25（默认顺序） | **0.25,0.25,0.5** |
| seed | 13 | **20261011** |
| dur-sec / snr-range / soft-clip-prob / dt-leadin-sec | 8 / 20,30 / 0 / 2 | 同左 |
| ser-range / delay | 0,10（默认）/ 10–80ms（默认） | 同左 |
| RIR | pyroomacoustics 仿真（无 --rir-dir，manifest rir_src=sim） | 同左 |
| 语料目录 | voicebank train/clean（near=far）+ DNS noise | 同左 |

复现命令：

```powershell
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/scripts/aec_synth.py --num 300 --dur-sec 8 `
  --out-dir D:/data/aec-synth/eval_v5 --snr-range 20,30 --soft-clip-prob 0 --dt-leadin-sec 2 `
  --seed 20261011 --scenarios 0.25,0.25,0.5 `
  --near-dir D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --far-dir  D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --noise-dir D:/data/dns-challenge-mirror/datasets/noise
```

## 2. DT 场景三方对比（AECMOS 官方 ONNX，eval_v5 全段口径，n=154）

mean ± 95%CI（解析法 1.96·sd/√n），逐条分数来自各 aecmos_*.json 的 rows：

| 方法 | echo_mos | deg_mos |
|---|---|---|
| 传统 AEC (WP-06) | 2.525 ± 0.10 (sd 0.649) | **2.430 ± 0.08** (sd 0.531) |
| NN-AEC v4 | **3.721 ± 0.12** (sd 0.750) | 1.811 ± 0.05 (sd 0.294) |
| 级联 P1-1（MDF→NN） | **3.741 ± 0.11** (sd 0.687) | 1.851 ± 0.05 (sd 0.290) |

全场景明细（echo / deg）：

| 场景 | n | 传统 | NN v4 | 级联 |
|---|---|---|---|---|
| FEST | 70 | 4.002±0.20 / 5.00 | 3.956±0.19 / 5.00 | 4.589±0.05 / 5.00 |
| DT | 154 | 2.525±0.10 / 2.430±0.08 | 3.721±0.12 / 1.811±0.05 | 3.741±0.11 / 1.851±0.05 |
| NST | 76 | 5.00 / 2.645±0.11 | 5.00 / 2.091±0.10 | 5.00 / 2.062±0.09 |

其他指标（eval.py / eval_aec.py，eval_v5）：NN DT PESQ 1.455、DT STOI 0.751、FEST ERLE 69.84dB；传统 DT PESQ 1.717、STOI 0.805、FEST ERLE 20.18dB——与 eval_v3 同向（传统近端 PESQ 高、NN ERLE 口径性高）。

## 3. 与 eval_v3（n=14）对照

| DT | eval_v3 (n=14) | eval_v5 (n=154) | 方向 |
|---|---|---|---|
| 传统 echo / deg | 2.467 / 2.401 | 2.525±0.10 / 2.430±0.08 | 一致（略高） |
| NN v4 echo / deg | 3.754 / 1.743 | 3.721±0.12 / 1.811±0.05 | 一致 |
| 级联 echo / deg | 3.919 / 1.785 | 3.741±0.11 / 1.851±0.05 | echo 略低、deg 略高 |
| NN−传统 Δecho | +1.29 | **+1.20** | 一致 |
| NN−传统 Δdeg | −0.66 | **−0.62** | 一致 |
| 级联−NN Δecho | +0.17 | +0.02 | **级联优势缩水**（n=14 下的抽样波动） |

## 4. 结论（一句话）

扩样 11 倍（DT 14→154）后三方结论方向全部复现且 95%CI 收窄到 ±0.05–0.12：**NN/级联 echo 优势（+1.2，CI 不含 0）与 deg 结构性落后（−0.6，CI 不含 0）均为稳健结论**；唯一不复现的是 eval_v3 中级联 echo 高于纯 NN 的 +0.17——扩样后仅 +0.02，应视为小样本波动，级联的 echo 优势证据不成立。

## 5. 产物与失败/降级记录

- 数据：`D:/data/aec-synth/eval_v5`（含 manifest）、预计算 `D:/data/aec-synth/eval_v5_lin`（FLOAT，300 条，lin-check max|Δ|=0）
- NN：`exp/07_nn_aec/outputs/eval_v5_dt300/{aecmos_nn_v4.json, nn_metrics_v5.json}`
- 传统：`exp/07_nn_aec/outputs/eval_v5_dt300/{aecmos_trad_v5.json, trad_metrics_v5.json}`
- 级联：enh wav `exp/07_nn_aec/outputs/eval_v5_dt300/enh_cascade/`，打分 `exp/09_cascade_aec/results/aecmos_cascade_v5.json`
- 降级/异常：无失败。注：本机 torch 检出 CUDA，NN/级联推理实际走了 GPU（脚本自动选择）；级联未用 09 的 results/enh 默认目录，避免覆盖 eval_v3 的 enh。工具 `tools/compute_ci.py` 未调用（它会重写既有 reports/DATA_CI.md，违反禁改约束），CI 为内联同口径公式（1.96·sd/√n）直接计算。
