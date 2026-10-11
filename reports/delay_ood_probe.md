# delay_ood_probe — 基础时延整体移出训练分布（120–190ms）：NN v4 崩塌判别实验

- 日期：2026-10-11 · 执行：NN 泛化实验（子任务）· 目的：区分「NN 漂移稳健」的两种解释
- 背景：drift_sweep（σ=5ms/s、限幅 ±10ms 条内游走）中 NN v4 稳健（DT echo −0.19），仓库据此宣称
  "NN 漂移稳健是**结构性的**（逐帧复谱映射不依赖对齐）"。但 ±10ms 恰在训练 jitter 覆盖内——
  **结构性 vs 分布覆盖未被区分**。本实验把基础时延整体移出训练分布做判别。
- 约束遵守：未改任何既有文件；新产物仅 `exp/07_nn_aec/outputs/eval_ood_delay/`（本目录）、
  本报告、`D:/data/aec-synth/eval_ood_delay`（新数据目录）。未训练。

## 1. 实验设计（判别逻辑）

NN-AEC v4 训练只见过回声时延 uniform(10,80)ms（train_v1 合成）+ dataset 层 far 再加 ±10ms 抖动
（`dataset.py`，即模型见过的 far-vs-echo 失配 ≈ 0–90ms 窗）。`model.py` 的 `forward_spec` 直接吃
(mic, far) 复谱、**无任何显式对齐模块**——时延鲁棒只能来自学到的复谱映射。

判别集：与 eval_v5 **同 seed（20261011）同 rng 序列**，唯一改动 `--delay-min-ms 120 --delay-max-ms 190`
→ 每条内容（句子/RIR/SNR/SER）与 eval_v5 逐字段相同，delay_ms = delay_v5 + **110ms（逐条精确）**
（`rng.uniform` 同一 u 映射到等宽 70ms 区间，300/300 行满足线性映射，见 §2）。整体平移后 120–190ms
完全落在训练窗 0–90ms 之外（含 jitter 余量）。

- 若「结构性」成立（逐帧映射不依赖对齐）→ OOD 上 echo 应基本稳（|Δ|≤0.3）；
- 若「分布覆盖」成立 → echo 应大幅崩塌（Δ<−1.0）。

## 2. 配对核对（配对声明成立的证据）

三层核对（脚本 `exp/07_nn_aec/outputs/eval_ood_delay/check_pairing.py`，结果 `check_pairing_result.json`）：

1. **manifest 全 300 行逐字段**：name/scenario/rir_src/near_file/far_file/noise_file/ser_db/snr_db/
   soft_clip/leadin_sec **全部 0 不一致**；delay 线性映射 `(d−10)/70=(d′−120)/70` **300/300 精确成立**；
   delay 范围 10.11–79.75ms → 120.11–189.75ms；场景分布两集同為 DT 154 / NST 76 / FEST 70。
2. **波形级**：NST 无回声 → **mic 76/76 逐字节一致**；FEST nearend（全零）70/70 一致；FEST mic
   互相关峰 5/5 抽样全部精确落在 Δdelay=1760 样本（110ms@16k）处——回声路径确实只被整体平移。
3. **打分侧内置对照**：NST 行两集 NN 分数逐位相同（echo 4.999 / deg 2.091）——管线确定、
   时延操纵只作用于回声路径。

10 条抽样（均匀取 index 0/33/…/299；完整表在 `check_pairing_result.json` 的 `sample10`）：

| index | name | 场景 | near/far 句子 | noise | SER dB | SNR dB | delay_v5 | delay_OOD |
|---|---|---|---|---|---|---|---|---|
| 0 | fileid_0_nst | NST | p258_399 / p226_283 | qhSAilVtehQ | 7.717 | 24.342 | 60.45 | 170.45 |
| 33 | fileid_33_dt | DT | p228_276 / p273_042 | HsvmwXnKzB4 | 4.132 | 20.777 | 59.29 | 169.29 |
| 66 | fileid_66_dt | DT | p268_272 / p239_092 | UOD9l0nF8xQ | 8.958 | 29.471 | 44.87 | 154.87 |
| 99 | fileid_99_fest | FEST | p269_247 / p254_356 | door_…457236 | 6.135 | 29.841 | 75.93 | 185.93 |
| 132 | fileid_132_dt | DT | p273_371 / p227_166 | UmPLO_zBX6I | 1.617 | 20.386 | 65.23 | 175.23 |
| 166 | fileid_166_dt | DT | p270_217 / p279_235 | W_o_lzg5YM0 | 3.163 | 23.064 | 31.76 | 141.76 |
| 199 | fileid_199_nst | NST | p243_161 / p269_115 | door_…16941 | 9.758 | 20.961 | 25.02 | 135.02 |
| 232 | fileid_232_fest | FEST | p258_313 / p270_446 | SBr5PlzuK3I | 8.618 | 28.022 | 76.55 | 186.55 |
| 265 | fileid_265_fest | FEST | p279_379 / p231_305 | fan_…263504 | 6.752 | 20.424 | 72.08 | 182.08 |
| 299 | fileid_299_fest | FEST | p244_215 / p282_127 | UOD9l0nF8xQ | 0.758 | 22.210 | 63.40 | 173.40 |

每行：同 index 同句子（near/far/noise 文件全同）同 SER/SNR，**只 delay 不同（精确 +110.00ms）**。

**如实注记（合成协议固有，非配对破坏）**：`generate_one` 的防削波守卫在 mic 峰值>0.9 时对全部成分
同乘 k=0.9/peak；时延平移改变回声与近端/噪声的叠加对齐 → 两集 k 略不同 → 204 条含回声行的
farend/nearend 相差一个全局增益（FEST 66/70、DT 138/154 行触发）。实测该增益比 **无系统性偏置**
（FEST mean 0.992、DT mean 1.006；两集 mic 峰值均值 0.847/0.847、0.881/0.882），FEST 95% 行在
±5% 内、DT 59% 在 ±5% 内（极端 ±29%）——SER/SNR 比值不受影响，属于对称配对噪声，不构成
方向性混杂（mic 地板两集几乎相同也印证，§3）。

## 3. 主表：NN v4（best.ckpt）eval_v5 vs eval_ood_delay（AECMOS 官方 ONNX，全段口径）

ckpt 确认：eval_v5_dt300 基线用的 `exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt`
（THREE_WAY.md 同名文件命令 + aecmos_leadincrop/notes.json 探针实证；本任务用同 ckpt 复跑
eval_v5 得 **300/300 rows 与既有 `eval_v5_dt300/aecmos_nn_v4.json` 逐位一致**——同口径成立且管线确定）。

mean ± 95%CI（1.96·sd/√n）；Δ 为**逐条配对差**（OOD−v5，同 300 条）：

| 场景 | 指标 | NN @ eval_v5（10–80ms） | NN @ OOD（120–190ms） | 配对 Δ | 下降条数 | mic 地板 v5→OOD |
|---|---|---|---|---|---|---|
| DT (n=154) | echo_mos | 3.721 ± 0.118（引既有 3.721±0.12） | **1.919 ± 0.051** | **−1.802 ± 0.125** | **154/154** | 1.544±0.038 → 1.634±0.040 |
| DT (n=154) | deg_mos | 1.811 ± 0.046 | 2.294 ± 0.071 | +0.484 ± 0.080 | 23/154（回升为主） | 3.151±0.074 → 3.118±0.070 |
| FEST (n=70) | echo_mos | 3.956 ± 0.192 | **2.105 ± 0.261** | **−1.851 ± 0.301** | 60/70 | 1.352±0.039 → 1.410±0.039 |
| FEST (n=70) | deg_mos | 4.999 | 5.000 | +0.001 | —（无近端） | 5.00 → 5.00 |
| NST (n=76) | echo/deg | 4.999 / 2.091 | 4.999 / 2.091 | 0.000（逐字节一致输入） | 0/76 | 逐字节一致 |

配对解读：

1. **崩塌幅度**：DT echo −1.80（v5 的 3.721 → 1.919），FEST echo −1.85——远超 −1.0 判据；
   配对 CI ±0.125/±0.301 均不含 0，DT **154/154 无一例外下降**。
2. **地板对照排除混杂**：原始 mic 地板两集几乎不动（DT +0.089±0.024、FEST +0.059±0.033，
   后者还是略升）→ OOD 集对打分器本身不是"更难"，崩塌是 **NN 侧** 的。NN 相对 mic 地板的
   echo 余量：DT +2.18 → **+0.29**（抑制能力损失 ≈87%）、FEST +2.60 → **+0.70**——OOD 时延下
   NN 输出的回声几乎回到"没处理"水平。
3. **deg 反向回升的机制一致性**：DT deg +0.48（1.811→2.294，趋向 mic 地板 3.118）。NN 的 cRM
   门控在见过的失配窗内"压回声同时伤近端"；失配出窗后既压不住回声、也少伤近端——输出行为
   从"主动抑制"退化为近似透传，两指标同向印证崩塌源于回声定位失效。
4. NST/FEST-deg 不动为内置对照：无回声或无近端的场景对时延平移不敏感，符合"只有回声路径被平移"。

## 4. 结论（二选一 + 证据强度）

**结论：「漂移稳健=分布覆盖，结构性宣称不成立」。** DT echo 配对 Δ = −1.802（< −1.0 判据），
154/154 条全降、CI ±0.125 不含 0；FEST 同幅崩塌（−1.85）；mic 地板两集相同排除数据侧混杂。
**证据强度：强**（配对设计 + 全体一致方向 + 双场景复现 + 地板对照 + 复现校验逐位一致）。

对仓库宣称的直接修正：drift_sweep 的"NN 漂移稳健"只在**训练分布覆盖范围内**（合成 10–80ms +
jitter ±10ms ⇒ ~0–90ms 失配窗）成立——是**插值**，不是逐帧复谱映射的结构外推能力。把基础时延
整体平移 +110ms 后，模型的时间感受野内 far 参考与回声失配，cRM 无法定位回声 → 崩塌。
"逐帧复谱映射不依赖显式对齐"的说法应改为"不依赖**精确**对齐，但依赖失配落在训练覆盖的时延窗内"。

## 5. 局限

- **只测静态整体偏移**（逐条 +110ms 固定时延）；条内时变漂移（drift_sweep 已测，且其 ±10ms
  幅度恰在覆盖窗内——正因如此该实验无法区分两种解释）未在本探针范围内。
- **传统列（含 V4 修复版）另由并行任务补**——本报告只下 NN 侧结论，不做 NN vs 传统的 OOD 对比。
- 崩塌边界未扫：90–120ms 之间（训练窗边缘 + jitter 余量的过渡带）在何处失效未测。
- 单 seed 配对对（300 条同内容，内容混杂已被配对设计消除，但 seed 单一）；AECMOS 本地 ONNX
  全段送评口径（与既有全部基线同口径，绝对值有口径差、对比不受影响）。
- 防削波守卫增益抖动（§2 注记）：对称无偏的配对噪声，稀释灵敏度但不引入方向性偏差。

## 6. 重跑命令（全量）

```powershell
$py = 'C:\Users\14805\.conda\envs\avsuite\python.exe'
# ① OOD 时延集（300 条 × 8s；与 eval_v5 同 seed → 内容配对，仅 delay 10-80 → 120-190ms）
& $py scripts/aec_synth.py --num 300 --dur-sec 8 `
  --out-dir D:/data/aec-synth/eval_ood_delay --snr-range 20,30 --soft-clip-prob 0 --dt-leadin-sec 2 `
  --seed 20261011 --scenarios 0.25,0.25,0.5 --delay-min-ms 120 --delay-max-ms 190 `
  --near-dir D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --far-dir  D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --noise-dir D:/data/dns-challenge-mirror/datasets/noise
# ② NN v4 评测（ckpt 同 eval_v5_dt300：outputs_v4/nnaec_g0.5_j10/best.ckpt）
& $py exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_ood_delay `
  --nn-ckpt exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt `
  --out exp/07_nn_aec/outputs/eval_ood_delay/aecmos_nn_ood.json
# ③ 复现校验（同 ckpt 复跑 eval_v5；结果与既有 eval_v5_dt300/aecmos_nn_v4.json 逐位一致）
& $py exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v5 `
  --nn-ckpt exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt `
  --out exp/07_nn_aec/outputs/eval_ood_delay/aecmos_nn_v5_recheck.json
# ④ 原始 mic 地板（aecmos_eval 无 mic 档，复用其 load_clips/score_all 的驱动）
& $py exp/07_nn_aec/outputs/eval_ood_delay/mic_floor_driver.py --eval-dir D:/data/aec-synth/eval_ood_delay --out exp/07_nn_aec/outputs/eval_ood_delay/aecmos_mic_ood.json
& $py exp/07_nn_aec/outputs/eval_ood_delay/mic_floor_driver.py --eval-dir D:/data/aec-synth/eval_v5     --out exp/07_nn_aec/outputs/eval_ood_delay/aecmos_mic_v5_ref.json
# ⑤ 配对核对 + 汇总（主表/配对 Δ/自检）
& $py exp/07_nn_aec/outputs/eval_ood_delay/check_pairing.py
& $py exp/07_nn_aec/outputs/eval_ood_delay/aggregate.py
```

## 7. 产物清单与数值自检

- 数据：`D:/data/aec-synth/eval_ood_delay`（300 条 × 8s，manifest：DT 154 / NST 76 / FEST 70，
  delay 120.11–189.75ms）
- 打分：`exp/07_nn_aec/outputs/eval_ood_delay/{aecmos_nn_ood.json, aecmos_nn_v5_recheck.json,
  aecmos_mic_ood.json, aecmos_mic_v5_ref.json}`（tag/summary/rows 格式同 aecmos_eval）
- 校验/汇总：同目录 `{check_pairing.py, check_pairing_result.json, mic_floor_driver.py,
  aggregate.py, aggregate_result.json}`
- 自检（`aggregate_result.json`）：nn_ood 与 mic_ood 的 rows 均为 300，逐场景条数与 manifest
  一致（dt154/fest70/nst76）；summary 均值与 rows 复算一致（<5e-4）；v5 复现与既有 JSON
  300/300 rows 逐位一致；NST 内置对照逐位一致。无失败无降级；NN 推理走 CUDA（torch 自动选择）。
