# 时延漂移（drift）鲁棒性对照 — P2-1

执行时间：2026-10-12 凌晨 · 执行日志：`reports/rectification/exec/P2-1.md`
问题：回声路径时延随时间漂移（缓冲抖动/时钟偏差）时，NN-AEC（v4）相对传统线性 AEC 的
回声抑制优势是否缩小——即验「**漂移下 NN 降幅是否大于线性**」（『drift 分水岭』判断）。

## 结论（TL;DR）

**不支持「漂移下 NN 降幅大于线性」**。σ=5ms/s 慢随机游走漂移下，线性 AEC 回声分系统性
崩塌（配对：FEST −1.89、35/36 条下降；DT −0.80、17/18 条下降），NN v4 基本稳住
（FEST +0.24、DT −0.19，AECMOS 噪声级波动）；NN 相对线性的回声优势在漂移下**进一步拉大**
（DT echo：+1.54 → +2.15；FEST echo：+0.10 → +2.23）。

适用范围：σ=5ms/s、限幅 ±10ms、8s 片、同配方合成集、本地 AECMOS 口径（echo/deg）。
更大的 σ、真实设备漂移、漂移重训模型未测（见局限）。

## 数据集与口径

- **漂移集** `D:/data/aec-synth/eval_v3_drift`：eval_v3 同配方（`--num 60 --dur-sec 8
  --rir-max-ms 100 --snr-range 20,30 --soft-clip-prob 0 --dt-leadin-sec 2`，
  scenarios 0.5/0.25/0.25 与 SER 0-10 均默认）+ `--delay-drift 5.0 --seed 20261012`
  → 实现 36 FEST / 6 NST / 18 DT（seed 抽样波动；**NST 仅 6 条，NST 行统计薄**）。
- **漂移实现**（`avsuite/scripts/aec_synth.py` 新增 `--delay-drift`，默认 0 逐比特兼容）：
  delay(t) = 基础时延 + 慢随机游走——每整秒步进 N(0, 5ms)、限幅 ±10ms（±2σ）、
  节点间线性过渡；分数时延用**线性插值**实现（近似口径见 `apply_delay_drift` 注释，
  未采用 resample_poly 变比率重采样）。游走用独立 Generator（`[seed, 0x9E3779B9]`），
  **不占主 rng**——同 seed 开/关漂移的底座内容逐项一致（单测 `test_main_drift_pairs_with_base` 钉死）。
- **配对设计**（主证据）：同 seed 20261012 的 drift=0 底座与漂移集 **60/60 manifest
  底座逐字段一致**（场景/句子/基础时延/SER/SNR 全同），唯一差异是回声时延轨迹
  → 配对 delta 干净隔离漂移效应，不受样本内容混杂。
- **内置对照**：NST 样本无回声，漂移对其为空操作——mic 逐字节一致，echo/deg 与底座
  **完全相同**（线性 5.000/2.702、NN 5.000/2.259）✓，同时证明打分管线确定性、
  漂移只作用于回声路径。
- **打分**：`exp/07_nn_aec/aecmos_eval.py`（本地 AECMOS ONNX，8s 全段送评，与既有
  基线同口径）。**ovrl 口径**：本地 AECMOS 只有 echo/deg 两路输出（ONNX 输出维已核）；
  仓内惯例的 "ovrl" = **全 60 条 echo/deg 均值**（同 P1-1 日志口径），非第三路模型输出。

## 4 格表（任务规格：线性/NN × 固定/漂移，echo/deg/ovrl）

固定格 = 既有 eval_v3（seed 13，60 条 = 26/14/20）基线，直接引用
`aecmos_trad_wp06.json` / `aecmos_nn_v4.json`；漂移格 = 本次实测。
**注意**：固定与漂移两格样本内容不同（不同 seed、场景实现 26/14/20 vs 36/6/18），
此表是分布级对比；干净的逐条配对见下一节。

| 系统 | 条件 | FEST echo (n) | DT echo (n) | NST echo (n) | FEST deg | DT deg | NST deg | ovrl echo | ovrl deg |
|---|---|---|---|---|---|---|---|---|---|
| 线性 WP-06 | 固定时延 | 3.836 (26) | 2.467 (14) | 5.000 (20) | 4.999 | 2.401 | 2.665 | 3.904 | 3.615 |
| 线性 WP-06 | 漂移 σ=5 | 2.119 (36) | 1.649 (18) | 5.000 (6) | 5.000 | 3.098 | 2.702 | 2.266 | 4.200 |
| NN v4 | 固定时延 | 3.762 (26) | 3.754 (14) | 5.000 (20) | 4.998 | 1.743 | 2.073 | 4.173 | 3.264 |
| NN v4 | 漂移 σ=5 | 4.353 (36) | 3.802 (18) | 5.000 (6) | 5.000 | 1.809 | 2.259 | 4.253 | 3.768 |

来源：`exp/07_nn_aec/outputs/{aecmos_trad_wp06, aecmos_nn_v4, aecmos_trad_drift, aecmos_nn_drift}.json`

## 配对对比（主证据；同 60 条、同底座，只差漂移）

| 系统 | 场景/指标 | 底座 (drift=0) | 漂移 (σ=5) | Δ mean±std | 下降条数 |
|---|---|---|---|---|---|
| 线性 | FEST echo | 4.011 | 2.119 | **−1.892 ± 1.237** | **35/36** |
| 线性 | DT echo | 2.451 | 1.649 | **−0.802 ± 0.655** | **17/18** |
| 线性 | DT deg | 2.656 | 3.098 | +0.443 ± 0.603 | 6/18（↑为主） |
| 线性 | ovrl echo / deg | 3.642 / 4.066 | 2.266 / 4.200 | **−1.376** / +0.134 | — |
| NN v4 | FEST echo | 4.110 | 4.353 | +0.244 ± 0.599 | 9/36 |
| NN v4 | DT echo | 3.993 | 3.802 | **−0.191 ± 0.294** | 10/18 |
| NN v4 | DT deg | 1.791 | 1.809 | +0.018 ± 0.097 | 8/18 |
| NN v4 | ovrl echo / deg | 4.164 / 3.763 | 4.253 / 3.768 | **+0.089** / +0.005 | — |
| 两者 | NST echo/deg | 5.000 / 各自值 | **逐字节同文件 → 完全相同** | 0.000 ± 0.000 | 0/6 |

来源：`outputs/{aecmos_trad_driftbase, aecmos_trad_drift, aecmos_nn_driftbase, aecmos_nn_drift}.json`

## 解读（如实报数，不硬凑方向）

1. **echo**：线性对漂移高度敏感——MDF「估计一次时延→固定对齐→自适应」的结构被时变
   时延打破，FEST 35/36 条下降（均值 −1.89）；NN 未见分布上几乎不掉（FEST 反 +0.24、
   DT −0.19，幅度在 AECMOS 噪声级）。「漂移下 NN 降幅是否大于线性」→ **否，且大幅反向**。
2. **与本体 `a_drift_echo`（漂移压低回声分，decreasing）对照**：对线性成立（FEST/DT 均降）；
   对 NN 仅 DT 弱成立（−0.19），FEST 不成立（+0.24）。
3. **DT deg**：线性 +0.44（对齐失效反而少吃近端），NN +0.02（不变）——NN 的 DT 近端
   损伤（deg ~1.8，v4 既有弱项）不因漂移加重，也未好转。
4. **NN 优势**：DT echo +1.542（底座）→ +2.153（漂移）；FEST echo +0.099 → +2.234。
   训练配方只见过「每条固定随机时延 jitter」，从未见过「条内时变」——NN 仍稳，
   提示其逐帧复谱映射对时延失配的敏感性远低于传统「先对齐后滤波」结构；
   但这属于结构性猜测，因果确认需漂移重训对照（局限 1）。

## 重跑命令

```powershell
$py = 'C:\Users\14805\.conda\envs\avsuite\python.exe'
# ① 漂移集（60 条，~40s）
& $py scripts/aec_synth.py --num 60 --dur-sec 8 `
  --out-dir D:/data/aec-synth/eval_v3_drift --rir-max-ms 100 --snr-range 20,30 `
  --soft-clip-prob 0 --dt-leadin-sec 2 --seed 20261012 --delay-drift 5.0 `
  --near-dir D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --far-dir  D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --noise-dir D:/data/dns-challenge-mirror/datasets/noise
# ② 配对底座（同 seed、无 --delay-drift；临时目录，~40s；与 ① 底座逐字段一致）
& $py scripts/aec_synth.py --num 60 --dur-sec 8 `
  --out-dir $env:TEMP/p21_base_check --rir-max-ms 100 --snr-range 20,30 `
  --soft-clip-prob 0 --dt-leadin-sec 2 --seed 20261012 `
  --near-dir D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --far-dir  D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --noise-dir D:/data/dns-challenge-mirror/datasets/noise
# ③ 打分（4 条，各 ~1min）
& $py exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3_drift --trad     --out exp/07_nn_aec/outputs/aecmos_trad_drift.json
& $py exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3_drift --nn-ckpt exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/last.ckpt --out exp/07_nn_aec/outputs/aecmos_nn_drift.json
& $py exp/07_nn_aec/aecmos_eval.py --eval-dir $env:TEMP/p21_base_check --trad     --out exp/07_nn_aec/outputs/aecmos_trad_driftbase.json
& $py exp/07_nn_aec/aecmos_eval.py --eval-dir $env:TEMP/p21_base_check --nn-ckpt exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/last.ckpt --out exp/07_nn_aec/outputs/aecmos_nn_driftbase.json
# ④ 单测（7 passed）
& $py -m pytest tests/test_delay_drift.py -v
```

## 局限与后续工作

1. **未做漂移重训**——训练侧带 `--delay-drift` 的数据未生成，NN 的漂移稳健是结构性还是
   配方性未分辨；后续：train_v1 加漂移重训，对照本表再评。
2. 单一 σ=5ms/s、单 seed；更大 σ（如 20ms/s）、正弦/突发跳变等其他漂移形态未扫。
3. NST 仅 6 条（seed 20261012 抽样），NST 行统计薄；固定格 n=20 与漂移格 n=6 不等。
4. ovrl 为全 60 条均值口径（仓内惯例），AECMOS 本地模型本身无第三路输出；如需官方
   ovrl 定义须换官方长测试集口径或扩打分器。
5. 配对底座在临时目录（`$env:TEMP/p21_base_check`），命令确定可复现（~40s）；
   如需长期保留建议另落 `D:/data/aec-synth/`（超出本任务授权目录，未建）。
6. 数据集级「窗级轨迹抽验」尝试过 GCC-PHAT：真实语音+混响下语音周期结构产生显著假峰
   （固定集本身窗级跨度 0-27ms），区分度不足，未作为证据；机制验证以单测为准
   （合成信号上互相关轨迹估计误差 ≤0.6ms，见 `tests/test_delay_drift.py`）。
