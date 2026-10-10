# CHEATSHEET.md — 项目数字速查（面试前 10 分钟扫一遍）

> 所有数字与 [`NUMBERS.md`](NUMBERS.md) 对齐。⭐=必须脱口而出；口径=被追问时的兜底。

## 一句话定位
车载语音前端全链路：AEC（回声）+ NR（降噪）+ 端侧部署，传统 + NN 双修，全程可复现。

## 必背数字（⭐）

| 项 | 值 | 口径兜底 |
|---|---|---|
| ⭐ 测试 | **80 passed** | `pytest tests -q`，~33s |
| ⭐ 降噪 PESQ | **1.97 → 2.53** | VoiceBank 824 对，wb-16k，cRM+BLSTM |
| ⭐ AEC echo_mos | 传统 2.47 → **NN 3.75** | AECMOS 官方，DT 双讲段，eval_v3 60 条 |
| ⭐ 传统 AEC ERLE | **20.55 dB** | AEC-Challenge 合成，FEST 段 |
| ⭐ 端侧 RTF | **0.0084** | GTCRN，CPU 单线程，算力上界 |
| ⭐ NN-AEC 参数 | **0.638M** | <1M 约束；GTCRN 23.67K |
| ⭐ 流式对齐 | **<1e-5** | 实测 ~1e-7，单测锁死 |

## 诚实项（主动说比被戳穿强）

| 项 | 值 | 怎么讲 |
|---|---|---|
| NN-AEC 近端 deg_mos | NN 1.74 < 传统 2.40 | 结构性瓶颈（五变体证明非欠训）→ 生产回级联 |
| FEST ERLE | NN "69.83" | 掩码门控≈静音的口径产物，不可比；看 echo_mos 4.0 |
| DCCRN | 2.18（DoD 2.6 走降级） | real-conv 2.19 反超 → 复数赢在参数效率（ADR 0004） |
| static INT8 | PESQ 1.12 崩坏 | GRU 状态误差累积（ADR 0003），dynamic -0.067 可用 |
| C++ demo | RTF 0.0094 | 端侧**整段**推理，非逐帧流式 |

## 结构速记（白板可画）
- **增强**：STFT(512/320) → CRN(3enc+BLSTM+3dec) → cRM⊙带噪谱 → ISTFT
- **NN-AEC**：mic+far 复谱(4ch) → 3×因果复数卷积(257→33) → 单向LSTM(96) → 因果反卷积 → cRM
- **传统 AEC**：GCC-PHAT → MDF → DTD 三态(自适应/软冻/硬冻) → RES
- **部署**：GTCRN(ERB 129 维) → ONNX → INT8；流式=conv cache+LSTM state

## 口径纪律（最减分点）
报任何指标前默念：**采样率 / wb-nb / 测试集构造 / 分场景 / checkpoint**。
本仓唯一权威口径 = [`NUMBERS.md`](NUMBERS.md)。
