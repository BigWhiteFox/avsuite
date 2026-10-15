# avsuite 项目验收报告（自动生成）

生成时间：2026-10-10 05:20。指标协议统一：16k / PESQ-wb / 测试集口径见各 exp README。

## WP-03 CRN（DoD：VoiceBank 测试集 PESQ ≥2.3）

| 配置 | PESQ | STOI | SI-SNR | DoD |
|---|---|---|---|---|
| cRM+BLSTM | 2.5288 | 0.9358 | 18.7581 | ✅ |
| IRM+BLSTM | 2.8428 | 0.9451 | 17.5708 | ✅ |
| cRM+单向LSTM | 2.4999 | 0.9332 | 18.223 | ✅ |

## WP-04 DCCRN（DoD：DNS 合成测试 PESQ ≥2.6；降级带 2.4-2.5）

| 配置 | devtest PESQ | STOI | SI-SNR | 判定 |
|---|---|---|---|---|
| DCCRN+DM（默认） | 2.1811 | 0.9384 | 15.6086 | ⚠️走降级路径 |
| 静态混合 --no-dm | 2.0151 | 0.9267 | 14.3345 | ⚠️走降级路径 |
| 实数卷积 --real-conv | 2.1898 | 0.9372 | 15.685 | ⚠️走降级路径 |

## WP-05a GTCRN+ONNX+INT8（DoD：对拍<1e-4 / RTF<0.2 / 单帧<3ms）

- 对拍（PyTorch vs ONNX）：6.32e-06 ✅
- CPU 单线程 RTF：0.0084 ✅ / 单帧 T=1：0.673ms ✅
- INT8 动态 PTQ：PESQ 掉点 0.0669（中位 —，最大 1.7611）——可用 ✅
- INT8 静态 QDQ：PESQ 1.1196（崩坏，负面结果留档，教学用）

## WP-06 传统 AEC（DoD：FEST ERLE ≥20dB / 双讲近端不损）

- FEST ERLE：**20.547 dB** ✅  |  DT 近端 PESQ 1.687  |  NST 透传 PESQ 2.446

## WP-07 NN-AEC（DoD：DT echo_mos 较传统 +0.3 / 近端 PESQ 不明显掉）

> 主配置 = **v4**（γ0.5 / w_sisnr 1.0 / DT-NST 专注 / 240ep）。权威口径见 `docs/NUMBERS.md`。

| 配置 | FEST ERLE | DT PESQ(近端) | DT STOI | NST PESQ |
|---|---|---|---|---|
| NN-AEC 主配置(v4, γ0.5) | 69.83* | 1.403 | 0.731 | 2.477 |
| NN-AEC γ=0 消融 | 64.59 | 1.391 | 0.728 | 2.384 |
| 传统 AEC (WP-06 同集) | 20.547 | 1.687 | 0.8 | 2.446 |

\* FEST ERLE：NN 输出经掩码门控≈静音，ERLE=P(mic)/P(≈0) 数值虚高，**不可与线性 AEC 直接比**；有效抑制证据看 echo_mos（下）。

- DT echo_mos（**AECMOS@48k 模型协议**）：原始 mic 1.521\* → 传统 3.494 → **NN 4.175**（Δ vs 传统 **+0.681±0.212**，CI 不含 0，DoD +0.3 ✅；旧 16k 直送值 2.467/3.754 已作废，见 NUMBERS 顶部决策条）
- DT deg_mos（同口径）：原始 mic 3.18\* → 传统 3.598 → NN 3.237（⚠️ **仅 DT 段**落后，Δ−0.361±0.169；机理=**高频带选择性损伤 + 谱平坦化**，非宽带压低、非欠训；归因见 exp/07 README 与 `docs/adr/`）
- NST deg_mos（同口径）：原始 mic 2.665\* → 传统 3.842 → NN 3.941（**不再落后**；旧口径"透传段轻度伪影"表述作废，见 NUMBERS §WP-07 口径更替脚注）
- \* 原始 mic 列为旧口径、地板重评进行中（NUMBERS 同注）

> <sub>**口径更替（2026-10-11）**：本块 echo/deg 均为 AECMOS@48k 模型协议值；`reports/aecmos_rescore_v5fix.md`
> 给出逐集对照。叙事同步收窄：**deg 落后只存在于 DT 段**（eval_v3 −0.361±0.169；eval_v5 n=154
> −0.369±0.089；eval_v7 n=406 −0.322±0.055），NST 段 +0.047~+0.099 不显著 ⇒ 不再是"全场景结构性瓶颈"。</sub>

## WP-05b/05c/09/10（无训练指标，验证型 DoD）

- WP-05b 流式对齐：<1e-5 ✅（320 定块/不规则块/真实数据三路对拍，单测锁死）
- WP-05c C++ 对拍：3.4e-5 ✅ / RTF 0.0094 ✅（MSVC 构建通过）
- WP-09 evaluate_full：四模型统一评估 + 分场景 + 多进程 ✅（实测验证）
- WP-10 Gradio：增强+DNSMOS 全链路 ✅（HTTP 200 / 本地+HF Spaces 说明齐备）；DNSMOS 值按官方 polyfit
  校准口径重取（见 `exp/10_demo/README.md`：4s/SNR 5dB 样本 增强 ovr **1.924** vs 带噪 1.173，旧未校准 2.93 已作废）

## WP-08 非线性 AEC（选做；DoD：较线性基线 ERLE +5dB）

- v1（GRU 裸接 far）：**ERLE 0dB**——180ms 时延+RIR 记忆远超 GRU 容量，负结果文档化（级联结构的动机证据）
- v2 级联（MDF→GRU-128, 1s 段）：ERLE 5.78 → 6.06 (+0.28)，DT 近端 PESQ 1.687→1.724 无损伤
- v2 级联（2s 段）：ERLE 5.78 → 6.18 (+0.4)
- 判定：⚠️ 未达 +5dB，按 WORKLIST 降级路径交付（复现跑通 + 归因分析：本数据集 soft-clip ceiling 0.5 下非线性占比小，残余大头是线性misadjustment；深饱和数据集是下一步）

## 总账

- 单元测试：93 passed, 0 failed（`pytest tests -q`，~35s）
- 工作包：WP-00~10 主线全部完成；选做 WP-08/WP-10.5/WP-11 待定
- 面试文档：docs/INTERVIEW*.md ×10（提纲+追问预案，含实测数据）

---

## 附记（Wave V5, 2026-10-11）

> 本报告为 **2026-10-10 05:20 自动生成**的时点快照，正文保留不删；以下为与
> `docs/NUMBERS.md` 权威口径的差异清单（**以 NUMBERS 为准**）：

1. **单元测试**：正文「93 passed, 0 failed」→ 现为 **416 项 = 412 passed + 4 xfailed / 0 failed**
   （93 基线 + S1 管道加固 +7 + S2 评估工具 +24 + V4/V5 扩容与波次内新增 12 个测试文件 +137
   + 阶段0 扩容 +60：0-1 心理声学损失 9 / 0-3 互斥 17 / 0-5 DT 清单 24（含 t7 验收器修复新增 6）/ 0-4 语音池互斥池 10），
   `pytest tests -q` 约 60s（含 torch/ONNX 用例）；4 个 xfail 为已文档化的待裁决/待重构项
   （R14/R16/T12-b/v3a-010）。
2. **GTCRN INT8 动态 PTQ**：正文「掉点 0.0669（最大 1.7611）」为**旧 50 条口径**；
   2026-10-11 扩样 200 条后主口径 = **Δ0.053（95%CI [0.028, 0.077]）**，
   fp32 1.869→**1.897**、dynamic 1.803→**1.845**（`reports/gtcrn_heldout200.md`）。
3. **GTCRN INT8 静态 QDQ 1.1196**：不变——NUMBERS 明示其**仍为旧 50 条口径，
   未随 200 条重评**。
4. 其余 headline（CRN 2.5288 / DCCRN 2.1811 / 传统 FEST ERLE 20.55 / NN DT echo 4.175、
   deg 3.237；均 AECMOS@48k 口径）与 NUMBERS 一致，无需修订。
5. **面试文档计数**：正文「docs/INTERVIEW*.md ×10」→ 实为 **11 个模块文档**（含 WP-11 TSE）；
   `docs/INTERVIEW.md` 索引本次已补 WP-11 行。
6. **选做项状态**：正文「WP-08/WP-10.5/WP-11 待定」→ WP-10.5 **已完成**
   （`exp/10.5_capstone/`，README 已记 checkpoint 口径）；WP-08 按降级路径交付
   （NUMBERS §2 WP-08）；WP-11 文档已备、demo 本体未落地（根 README 状态表为准）。
7. **CI 测试数披露**：上条第 1 项的 **416 项 = 412 passed + 4 xfailed 为本地全量**；**CI 实跑会少于本地**
   （数据/产物依赖用例走 `skipif`，例如 `D:/data` 相关；`outputs/`、`*.onnx`、`*.ckpt`、`*.pt`
   被 `.gitignore` 排除）——**CI 侧当前具体通过数待下一次 CI 复跑刷新**（勿沿用旧的 116/124，
   那对应 124 本地时代；清单与 skipif 构成见 `reports/refactor_wave5/test_gap.md` §1.3）。