# CHEATSHEET.md — 项目数字速查（面试前 10 分钟扫一遍）

> 所有数字与 [`NUMBERS.md`](NUMBERS.md) 对齐。⭐=必须脱口而出；口径=被追问时的兜底。

## 一句话定位
车载语音前端全链路：AEC（回声）+ NR（降噪）+ 端侧部署，传统 + NN 双修，全程可复现。

## 必背数字（⭐）

| 项 | 值 | 口径兜底 |
|---|---|---|
| ⭐ 测试 | **93 passed** | `pytest tests -q`，~35s |
| ⭐ 降噪 PESQ | **1.97 → 2.53** | VoiceBank 824 对，wb-16k，cRM+BLSTM |
| ⭐ AEC echo_mos | 传统 2.47 → **NN 3.75** | AECMOS 官方，DT 双讲段，eval_v3 60 条 |
| ⭐ 官方真实集 DT echo | 传统 2.31 → **NN 3.26**（+0.95） | ICASSP2022 真实测试集 800 段：3.256±0.124 vs 2.305±0.100（Δ+0.951±0.137） |
| ⭐ 传统 AEC ERLE | **20.55 dB** | eval_v3 FEST 段（eval_aec.py 口径；eval.py 口径 20.37，脚本间 0.18dB 口径差，均 ≥20） |
| ⭐ 端侧 RTF | **0.0084** | GTCRN，CPU 单线程，算力上界 |
| ⭐ NN-AEC 参数 | **0.638M** | <1M 约束；GTCRN 23.67K |
| ⭐ 流式对齐 | **<1e-5** | 实测 ~1e-7，单测锁死 |

> 端到端时延预算表：见 docs/LATENCY_BUDGET.md（合计 66.5 ms，实测 p99 67.8 ms,低负载复测@02:12）

## 诚实项（主动说比被戳穿强）

| 项 | 值 | 怎么讲 |
|---|---|---|
| NN-AEC 近端 deg_mos | NN 1.81 < 传统 2.43（eval_v5 DT n=154） | 结构性瓶颈（五变体+欠训+接口排除）→ 纯级联不治 deg（1.851≈1.811，ADR 0005 附记），下一步近端门控 |
| 真实车厢 RIR | NN echo 优势 +1.20 → **+0.57** | 仿真 RIR 系统性高估（IR64-CAR，eval_v6 DT n=42，CI 均不含 0）；传统 FEST ERLE 20.18→7.95dB，近端指标基本不变 |
| 漂移下线性崩塌 | 传统 FEST −1.89（35/36 条降） | 零余量一次对齐的实现缺陷（V4 整改修复中），非 MDF 本质；NN 窗内几乎不退（Δecho −0.19±0.29）但**出窗即崩**（+110ms：−1.80，154/154 降，见 delay_ood_probe——覆盖性稳健，前置粗对齐必须保留）；drift 重训反崩（3.80→2.35，负结果留档） |
| FEST ERLE | NN "69.83" | 掩码门控≈静音的口径产物，不可比；看 echo_mos 4.0 |
| DCCRN | 2.18（DoD 2.6 走降级） | real-conv 2.19 与复数相当（Δ0.009 噪声级，不称反超）→ 复数赢在参数效率（ADR 0004） |
| static INT8 | PESQ 1.12 崩坏 | GRU 状态误差累积（ADR 0003），dynamic Δ0.053 可用（200 条扩样口径，CI [0.028, 0.077]） |
| C++ demo | RTF 0.0094 | 端侧**整段**推理，非逐帧流式 |
| NN-AEC 泛化 | train/eval 同 28 说话人池（同池不同句） | 泛化结论保守解读 |

## 结构速记（白板可画）
- **增强**：STFT(512/320) → CRN(3enc+BLSTM+3dec) → cRM⊙带噪谱 → ISTFT
- **NN-AEC**：mic+far 复谱(4ch) → 3×因果复数卷积(257→33) → 单向LSTM(96) → 因果反卷积 → cRM
- **传统 AEC**：GCC-PHAT → MDF → DTD 三态(自适应/软冻/硬冻) → RES
- **部署**：GTCRN(ERB 129 维) → ONNX → INT8；流式=conv cache+LSTM state

## 口径纪律（最减分点）
报任何指标前默念：**采样率 / wb-nb / 测试集构造 / 分场景 / checkpoint**。
本仓唯一权威口径 = [`NUMBERS.md`](NUMBERS.md)。
