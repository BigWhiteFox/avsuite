# NUMBERS.md — 全仓指标权威口径（Single Source of Truth）

> 本文件是全仓所有对外数字的**唯一权威来源**。README / ACCEPTANCE / INTERVIEW / 简历 /
> 面试口述中的任何指标，都必须与本表一致；如不一致，以本表为准并回改对应文档。
> 更新纪律：任何新训练/评估完成后，先改本表（注明 checkpoint + 数据口径 + 日期），再同步其它文档。

**统一决策（2026-10-10 定版）**：
- NN-AEC **主配置 = v4**（γ0.5 / w_sisnr 1.0 / DT-NST 专注 / 240ep），对外只报 v4；v3 等作为探索矩阵存档。
- DCCRN **主结果 = 2.1811**（devtest 150 条口径），统一弃用旧值 2.161（后者是早期 synth100 口径）。
- NN-AEC 近端保真的官方结论 = **"模型形态与 AECMOS-deg 定义之间的结构性瓶颈"**（非"欠训"），禁用"仍欠训"表述。

---

## 1. 语音增强（enhancement）

### WP-03 CRN（VoiceBank-DEMAND 测试集 824 对，16k PESQ-wb）
| 配置 | PESQ | STOI | SI-SNR | checkpoint |
|---|---|---|---|---|
| **cRM+BLSTM（主）** | **2.5288** | 0.9358 | 18.76 | `03_crn/outputs/crn_crm_blstm/best.ckpt` |
| IRM+BLSTM | 2.8428 | 0.9451 | 17.57 | `crn_irm_blstm/best.ckpt` |
| cRM+单向LSTM | 2.4999 | 0.9332 | 18.22 | `crn_crm_lstm/best.ckpt` |
| （noisy 基线） | 1.9707 | 0.921 | 8.45 | — |

### WP-04 DCCRN（DNS devtest 150 条，16k PESQ-wb，DoD≥2.6 → 走降级）
| 配置 | PESQ | STOI | SI-SNR | checkpoint |
|---|---|---|---|---|
| **DCCRN+DM（主）** | **2.1811** | 0.9384 | 15.61 | `04_dccrn/outputs/dccrn/best.ckpt` |
| --no-dm（静态混合） | 2.0151 | 0.9267 | 14.33 | `outputs_abl/dccrn_nodm` |
| --real-conv（实数卷积） | 2.1898 | 0.9372 | 15.69 | `outputs_abl/dccrn_realconv` |
| （noisy 基线） | 1.5822 | 0.9152 | 9.23 | — |

> ⚠️ real-conv ≥ 复数且参数约 2×，是本仓一个**反直觉发现**，归因分析见
> `docs/adr/0004-dccrn-real-vs-complex.md`，面试必须能讲。

### WP-05a GTCRN（DNS 自混 held-out 50 条，seed 777，SNR U(-5,15)）
| 项 | 值 | 说明 |
|---|---|---|
| PyTorch vs ONNX 对拍 | **6.32e-06** | max abs diff |
| CPU 单线程 RTF | **0.0084** | 4s 语音，算力上界 |
| CPU 单帧 T=1 | **0.673 ms** | 无 GRU 状态，算力上界非流式实测 |
| INT8 dynamic PTQ | PESQ 1.803（**-0.067**） | 可用，240KB |
| INT8 static QDQ | PESQ 1.120（崩坏） | 负面结果，教学对照 |
| fp32 基线 PESQ | 1.869 | 自混更难，勿与论文 2.x 直接比 |

## 2. 回声消除（AEC）

### WP-06 传统 AEC（AEC-Challenge 合成，eval_v3 60 条）
| 指标 | 值 |
|---|---|
| FEST ERLE | **20.55 dB** ✅（DoD≥20） |
| DT 近端 PESQ | 1.687 |
| NST 透传 PESQ | 2.446 |
| DT echo_mos | 2.467 |
| DT deg_mos | 2.401 |

### WP-07 NN-AEC（主配置 v4，eval_v3 60 条，16k；vs 传统基线）
| 指标 | 原始 mic | 传统 AEC | **NN-AEC v4** |
|---|---|---|---|
| FEST ERLE (dB) | — | 20.55 | **69.83**（掩码门控≈静音，口径见下） |
| DT 近端 PESQ | 1.414 | 1.687 | 1.403 |
| NST 透传 PESQ | 2.446 | 2.446 | 2.477 |
| DT echo_mos | 1.521 | 2.467 | **3.754**（Δ+1.29，DoD+0.3 ✅） |
| DT deg_mos | 3.18 | 2.401 | **1.743**（⚠️ 未达标，结构性瓶颈） |
| NST deg_mos | 2.665 | 2.665 | 2.073（⚠️ 轻度伪影） |

> **FEST ERLE 口径**：NN 输出经 cRM 掩码门控近似静音 → ERLE=P(mic)/P(≈0) 数值虚高，
> **不可与线性 AEC 的 ERLE 直接比**。对外表述统一为"FEST 段输出≈静音，echo_mos 4.0+
> 才是有效抑制证据"。

### WP-08 非线性 AEC（选做，DoD 较线性 +5dB → 走降级）
| 版本 | 结果 |
|---|---|
| v1 裸 GRU | ERLE 0 dB（负结果：180ms 时延+RIR 记忆超 GRU 容量） |
| v2 级联（MDF→GRU-128, 1s 段） | ERLE 5.78→6.06（**+0.28**），DT 近端无损伤 |
| v2 级联（2s 段） | ERLE 5.78→6.18（+0.4） |

## 3. 部署 / 推理
| 项 | 值 |
|---|---|
| WP-05b 流式 vs 非流式对齐 | **<1e-5**（实测 ~1e-7，单测锁死） |
| WP-05c C++ vs Python 对拍 | **3.4e-5**；RTF 0.0094（端侧**整段**推理，非逐帧流式） |
| 单元测试 | **80 passed**（`pytest tests -q`，~33s） |

---
*生成/维护：本表随每次验收更新；历史改动见 git log。*
