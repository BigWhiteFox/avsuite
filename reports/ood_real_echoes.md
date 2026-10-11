# OOD 评测：真实录制回声上的传统 AEC vs NN-AEC v4（WS-D1）

- 日期：2026-10-11 · 执行：WS-D1 · 目的：量化仿真→真实的 domain gap
- 约束遵守：未改 docs/NUMBERS.md 与任何既有文档；产物全部在 `exp/07_nn_aec/outputs/ood_real_echoes/`（新目录），未覆盖任何既有 outputs。

## 1. 数据说明（含降级记录）

**实际使用**：AEC-Challenge 官方 **ICASSP 2022 真实录制测试集**（`D:/data/repos/AEC-Challenge/datasets/test_set_icassp2022`，本机唯一已物化的真实录制集）。真实设备（10,000+ 设备/说话人库）、真实声学环境录制，48 kHz，`lpb`=loopback 远端参考、`mic`=真实录制麦克风信号，**含真实双讲**：

| 场景子集 | 片段数 | 分层（标准 / with-movement） |
|---|---|---|
| FEST（远端单讲，含移动） | 300 | 125 / 175 |
| DT（真实双讲，含移动） | 300 | 127 / 173 |
| NST（近端单讲） | 200 | 200 / 0 |
| **合计** | **800** | |

**降级（如实记录，不顶替）**：`datasets/real`（37,588 wav）与 `datasets/real_doubled` 在本机均为 **git-lfs 指针文件**（文件头为 `version https://git-lfs.github.com/...`，无音频内容）；`blind_test_set*`、`test_set_interspeech2021`、`synthetic` 同为指针。未执行 git-lfs 拉取（任务禁止 git 操作）。本报告口径以 test_set_icassp2022 全量 800 片段为准。

**口径限制**：真实录制无干净近端参考 → **PESQ/STOI 不可算**（不硬造参考）。指标只有：
- **AECMOS**：官方 ONNX（`Run_1668423760_Stage_0.onnx`）+ 官方评级协议裁剪（FEST 送后半段 / DT 送最后 (len−15)/2 s / NST 全段），同 `official_aecmos_eval.py` 做法；
- **ERLE**：仅 FEST，官方评级段（后半段）上 `10log10(Σmic²/Σenh²)`。按 NUMBERS.md 口径注：NN 的 cRM 门控使 ERLE 虚高，**对外只能说"FEST 段输出≈静音"**，不与传统的 dB 值直接比大小。

## 2. 主表：AECMOS / ERLE（mean ± 95%CI，CI = 1.96·sd/√n）

| 场景 | n | 传统 AEC (WP-06, mdf/8/0.7) | NN-AEC v4 (best.ckpt) | NN−传统 Δ (配对, ±CI) |
|---|---|---|---|---|
| FEST echo_mos | 299/300 | 2.725 ± 0.126 | **3.143 ± 0.151** | **+0.423 ± 0.151** |
| FEST deg_mos | 299/300 | 5.000 ± 0.000 | 4.999 ± 0.000 | −0.000 ± 0.000 |
| FEST ERLE (dB) | 299/300 | 4.66 ± 1.39 | 27.09 ± 9.13* | —（口径注，不直接比） |
| DT echo_mos | 300/300 | 2.305 ± 0.100 | **3.256 ± 0.124** | **+0.951 ± 0.137** |
| DT deg_mos | 300/300 | **3.692 ± 0.088** | 3.097 ± 0.118 | −0.595 ± 0.104 |
| NST echo_mos | 200/200 | 4.999 ± 0.000 | 4.998 ± 0.000 | −0.000 ± 0.000 |
| NST deg_mos | 200/200 | **4.183 ± 0.046** | 3.759 ± 0.119 | −0.424 ± 0.106 |

\* NN FEST ERLE 受 cRM 门控口径性偏高（NUMBERS.md 注），只支持"输出≈静音"的定性说法。

**失败/降级**：仅 1 条（trad FEST `wufNfJIb3kC7OvuxutARLQ_farend-singletalk-with-movement`，传统管道输出含非有限样本/float32 溢出，打分侧发散防护拒绝送评），其余 799/800 × 2 方法全部成功。

## 3. 场景分层（movement 标签，echo_mos / deg_mos，±CI）

| 场景×分层 | n (trad/nn) | 传统 echo | NN v4 echo |
|---|---|---|---|
| FEST 标准 | 125 | 2.491 ± 0.183 / deg 5.00 | 2.821 ± 0.230 |
| FEST with-movement | 174/175 | 2.893 ± 0.168 | 3.374 ± 0.192 |
| DT 标准 | 127 | 2.318 ± 0.165 / deg 3.746 ± 0.131 | 3.298 ± 0.189 / deg 2.989 ± 0.181 |
| DT with-movement | 173 | 2.296 ± 0.123 / deg 3.653 ± 0.119 | 3.225 ± 0.164 / deg 3.176 ± 0.155 |

方向一致：真实数据上 NN 的 echo 优势在所有分层均成立；DT 的 deg 结构性落后在"with-movement"（真实可变回声路径）上略大。

## 4. Domain gap：vs 自合成 eval_v5（reports/eval_v5_dt300.md，n=300，全段口径）

| 场景·指标 | eval_v5（仿真） | 真实录制（OOD） | Δ（真实−仿真） |
|---|---|---|---|
| DT echo：传统 | 2.525 ± 0.10 | 2.305 ± 0.100 | **−0.22** |
| DT echo：NN v4 | 3.721 ± 0.12 | 3.256 ± 0.124 | **−0.47** |
| DT deg：传统 | 2.430 ± 0.08 | 3.692 ± 0.088 | **+1.26** |
| DT deg：NN v4 | 1.811 ± 0.05 | 3.097 ± 0.118 | **+1.29** |
| FEST echo：传统 | 4.002 ± 0.20 | 2.725 ± 0.126 | **−1.28** |
| FEST echo：NN v4 | 3.956 ± 0.19 | 3.143 ± 0.151 | **−0.81** |
| FEST ERLE：传统 | 20.18 dB | 4.66 dB | **−15.5 dB** |
| FEST ERLE：NN v4 | 69.84 dB | 27.09 dB | −42.8 dB（口径性指标，仅看趋势） |
| NN−传统 Δecho（DT） | **+1.20** | **+0.95 ± 0.14** | 优势缩水 −0.25，CI 仍不含 0 |
| NN−传统 Δdeg（DT） | −0.62 | −0.595 ± 0.104 | 结构性落后幅度不变 |

要点：
1. **FEST 掉得最狠**：echo_mos 传统 −1.28 / NN −0.81；传统 ERLE 从 20 dB 掉到不足 5 dB——真实环境的低 SNR、非线性扬声器、路径移动使仿真 FEST 明显高估收敛后的回声抑制（注：真实侧为官方"后半段"评级段、合成侧为 8 s 全段，口径差异加大该差距，方向不受影响）。
2. **DT echo 双方都掉**（传统 −0.22 / NN −0.47），NN 的相对优势从 +1.20 缩到 +0.95，但仍显著（CI 不含 0）。
3. **DT/NST deg_mos 在真实数据上反而大幅变好**（+1.2~+1.3）：真实录制近端信噪与音色与 AECMOS 训练分布更匹配，仿真的噪声/截幅配方对 deg 更苛刻——synthetic→real 方向上 deg 结论**不可直接外推**。
4. 与既有 `reports/official_aecmos/`（同数据、last.ckpt、官方协议）交叉验证：本结果（best.ckpt）DT echo 3.256 vs 3.239、FEST 3.143 vs 3.094，量级一致。

## 5. 结论（一句话）

在 800 段官方真实录制回声上，NN-AEC v4 的双讲 echo 优势（+0.95±0.14）从仿真的 +1.20 缩水约四分之一且 FEST 抑制量大幅缩水（传统 ERLE 20→5 dB），**仿真→真实 domain gap 实测显著（FEST echo 掉 0.8~1.3、DT echo 掉 0.2~0.5），但"NN echo 优、deg 落后"的定性结论在真实分布上仍稳健成立**。

## 6. 产物

- 脚本：`exp/07_nn_aec/ood_real_echoes_eval.py`（enumerate/infer-trad/infer-nn/score/agg，断点续跑）
- 增强 wav：`exp/07_nn_aec/outputs/ood_real_echoes/stage/enh_{trad,nn}/{fest,dt,nst}/`（各 800 条，16k FLOAT）
- 逐条打分：`exp/07_nn_aec/outputs/ood_real_echoes/scores_ood.jsonl` / `scores_ood.csv`
- 汇总（含 CI）：`exp/07_nn_aec/outputs/ood_real_echoes/summary_ood.json`
- 推理环境：NN 走 CUDA（torch 自动选择）；传统列 7 进程并行，无失败（除上表 1 条打分防护拒绝）。
