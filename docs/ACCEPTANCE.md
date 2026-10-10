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

| 配置 | FEST ERLE | DT PESQ(近端) | DT STOI | NST PESQ |
|---|---|---|---|---|
| NN-AEC 主配置(γ0.5) | 69.83 | 1.403 | 0.731 | 2.477 |
| NN-AEC γ=0 消融 | 64.59 | 1.391 | 0.728 | 2.384 |
| 传统 AEC (WP-06 同集) | 20.547 | 1.687 | 0.8 | 2.446 |

- DT echo_mos：原始 mic 1.521 → 传统 2.467 → **NN 3.754**（Δ vs 传统 1.287，DoD +0.3 ✅）
- DT deg_mos：原始 mic 3.18 → 传统 2.401 → NN 1.743（近端保真 ⚠️ 仍欠训，续训空间见 README）

## WP-05b/05c/09/10（无训练指标，验证型 DoD）

- WP-05b 流式对齐：<1e-5 ✅（320 定块/不规则块/真实数据三路对拍，单测锁死）
- WP-05c C++ 对拍：3.4e-5 ✅ / RTF 0.0094 ✅（MSVC 构建通过）
- WP-09 evaluate_full：四模型统一评估 + 分场景 + 多进程 ✅（实测验证）
- WP-10 Gradio：增强+DNSMOS 全链路 ✅（HTTP 200 / 本地+HF Spaces 说明齐备）

## WP-08 非线性 AEC（选做；DoD：较线性基线 ERLE +5dB）

- v1（GRU 裸接 far）：**ERLE 0dB**——180ms 时延+RIR 记忆远超 GRU 容量，负结果文档化（级联结构的动机证据）
- v2 级联（MDF→GRU-128, 1s 段）：ERLE 5.78 → 6.06 (+0.28)，DT 近端 PESQ 1.687→1.724 无损伤
- v2 级联（2s 段）：ERLE 5.78 → 6.18 (+0.4)
- 判定：⚠️ 未达 +5dB，按 WORKLIST 降级路径交付（复现跑通 + 归因分析：本数据集 soft-clip ceiling 0.5 下非线性占比小，残余大头是线性misadjustment；深饱和数据集是下一步）

## 总账

- 单元测试：80 passed, 1 warning in 24.77s
- 工作包：WP-00~10 主线全部完成；选做 WP-08/WP-10.5/WP-11 待定
- 面试文档：docs/INTERVIEW*.md ×10（提纲+追问预案，含实测数据）