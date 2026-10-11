# realrir_domain — 真实录制 RIR 池 vs 仿真 RIR 的 domain gap 评估（eval_v6）

- 日期：2026-10-11 · 执行：WS-D2 · 目的：量化"仿真 RIR（pyroomacoustics）vs 真实录制 RIR"下传统 AEC 与 NN-AEC v4 的表现差异
- 约束遵守：未改 docs/NUMBERS.md 与任何既有文档；eval_v3/v5 数据与 outputs 只读未动；新数据集写 `D:/data/aec-synth/eval_v6_realrir`，新输出写 `exp/07_nn_aec/outputs/eval_v6_realrir/`。

## 1. 真实 RIR 来源与许可清单

| 来源 | 内容 | 条数（入池） | 格式 | 许可 | URL |
|---|---|---|---|---|---|
| IR64-CAR（Zenodo 15168578） | Peugeot 208 车厢内实测 IR（Eigenmike EM64，8 个外源方位 × 64 麦克） | **32**（8 方向 × 4 个分散胶囊 c00/c08/c16/c24） | 原始 48k/24bit 64ch → 重采样 16k mono PCM16，峰值归一 | **CC BY 4.0** | https://zenodo.org/records/15168578 |
| 本地 AEC-Challenge `datasets/RIRs` | 仅有 RT60 元数据 CSV（无实际 RIR wav），不可用 | 0 | — | — | 本地 |
| 本地 `D:/data/aec-synth/rir_pool` | pyroomacoustics 仿真 RIR（60 条），非真实 | 0（不计） | 16k wav | 内部 | 本地 |

- **真实 RIR 最终条数：32（全部车厢/交通工具场景），达标（≥30）**。
- 提取工具：`avsuite/tools/make_realrir_pool.py`（新增）；池目录 `D:/data/rir_real/pool_16k_mono/`；原始包 `D:/data/rir_real/ir64car/`。

### 降级/失败记录

| 尝试 | 结果 |
|---|---|
| OpenAIR（openairlib.net） | **降级**：主站账号被托管方停用（Account Suspended）+ 证书过期，无法下载 |
| OpenSLR SLR28（RWCP/REVERB/AIR 真实房间 RIR，Apache 2.0） | **降级**：三个镜像（magicdatatech/trmal/elda）实测带宽 ~1–5 MB/min，1.3GB 需数小时；车厢 32 条已达标，放弃等待 |
| ACE Challenge | Zenodo 6257551 可直下（48k，7 个 Imperial College 房间，**无 vehicle 场景**），未采用 |
| 注册墙 | 未遇到（IR64-CAR 为 Zenodo 开放获取） |

## 2. eval_v6_realrir 合成配方

与 eval_v5 唯一差异：**RIR 来源改为真实车厢 RIR 池（--rir-dir）+ 300ms 截断 + seed**。其余完全同 eval_v5。

```powershell
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/scripts/aec_synth.py --num 100 --dur-sec 8 `
  --out-dir D:/data/aec-synth/eval_v6_realrir --snr-range 20,30 --soft-clip-prob 0 --dt-leadin-sec 2 `
  --seed 20261012 --scenarios 0.25,0.25,0.5 `
  --rir-dir D:/data/rir_real/pool_16k_mono --rir-max-ms 300 `
  --near-dir D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --far-dir  D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --noise-dir D:/data/dns-challenge-mirror/datasets/noise
```

manifest 实测：**100 条，DT 42 / FEST 34 / NST 24**（随机抽样，DT 42 而非整数配比 50；eval_v5 亦为实测口径 154/300）。16k 全程一致（IR64-CAR 已预重采样至 16k，`get_rir` 不做重采样）。注：`aec_synth.py get_rir` 对多声道 wav 取平均，故预提取为 mono。

## 3. DT 场景对照：eval_v6（真实 RIR）vs eval_v5（仿真 RIR）

AECMOS（官方 ONNX，全段口径）mean ± 95%CI（1.96·sd/√n）：

| 方法·指标 | eval_v5 仿真 RIR (n=154) | eval_v6 真实车厢 RIR (n=42) | 变化 |
|---|---|---|---|
| 传统 echo_mos | 2.525 ± 0.10 | **2.000 ± 0.15** | **−0.53（恶化）** |
| 传统 deg_mos | 2.430 ± 0.08 | 2.518 ± 0.15 | +0.09 |
| NN v4 echo_mos | 3.721 ± 0.12 | **2.565 ± 0.23** | **−1.16（恶化）** |
| NN v4 deg_mos | 1.811 ± 0.05 | 2.044 ± 0.12 | +0.23 |
| NN−传统 Δecho | +1.20 | **+0.57** | **NN 优势近乎腰斩** |

PESQ / ERLE（DT PESQ 为近端输出口径；ERLE 为 FEST）：

| 指标 | eval_v5 仿真 | eval_v6 真实 | 变化 |
|---|---|---|---|
| 传统 DT PESQ | 1.717 | 1.525 ± 0.08 | −0.19 |
| NN DT PESQ | 1.455 | 1.466 ± 0.06 | +0.01 |
| 传统 FEST ERLE | 20.18 dB | **7.95 dB** | **−12.2 dB** |
| NN FEST ERLE | 69.84 dB | 41.57 dB | −28.3 dB |

全场景 AECMOS（eval_v6）：FEST 传统 3.301±0.43 / NN 3.718±0.45；NST 传统 echo 5.00 deg 2.588±0.15 / NN echo 5.00 deg 2.051±0.13。

## 4. Domain gap 结论（一句话）

真实车厢 RIR（短混响、强早反射、48k 实测特性）相对 pyroomacoustics 仿真 RIR 造成显著 domain gap：**DT 段双方 echo AECMOS 均大幅下滑（传统 −0.53、NN −1.16，NN 对仿真的 echo 优势从 +1.20 缩水到 +0.57，CI 不含 0），传统 FEST ERLE 从 20→8 dB 崩落，而近端质量指标（deg/PESQ/STOI）基本持平——即仿真数据高估了两类 AEC 的回波抑制能力，NN-AEC 的领先幅度在真实车厢声学下明显收窄，后续应用真实 RIR 数据增强/微调。**

## 5. 产物

- RIR 池：`D:/data/rir_real/pool_16k_mono/`（32 wav）；原始 `D:/data/rir_real/ir64car/extracted/`（CC BY 4.0，引用见其 README）
- 数据集：`D:/data/aec-synth/eval_v6_realrir/`（100 条 + manifest，rir_src=dir）
- 输出：`exp/07_nn_aec/outputs/eval_v6_realrir/`
  - NN：`summary.json`、`rows.csv`、`aecmos_nn_v4.json`
  - 传统：`trad_rows_v6.csv`、`aecmos_trad_v6.json`、`compare_trad_vs_nn.md`
- 工具：`avsuite/tools/make_realrir_pool.py`（新增）
- 局限：仅单一车型（Peugeot 208、静止、门窗关闭）单一录音布置，"真实 RIR"多样性有限；SLR28 房间 RIR 未获取（带宽降级），房间场景 domain gap 未覆盖。
