# exp/07_nn_aec — NN-AEC 神经网络回声消除（WP-07 / 模块7，简历核心）

双输入（mic + far 复数谱）压缩 CRN，**0.638M 参数（<1M 约束）**，时间维严格因果
（单测锁死），20ms 帧率。对手是 WP-06 传统 AEC——同评估集直接对比。
6 项单测全绿；全仓 80/80。

## 结构

```
mic ─STFT(512/320)─┐
                   ├─[B,4,257,T]─ CausalConv×3 [8,16,32] ─> F:257→33, T 不变
far(粗对齐)─STFT──┘        │跳连
                 LSTM(1056→96, 单向) → FC → CausalDeconv×3 → cRM
                 enh = cRM ⊙ mic谱 → ISTFT
```

**因果三件套**（流式就绪的架构保证，WP-05b cache 化即可上线）：
- conv 时间轴只左 pad 4（k=5）；deconv 输出右移 2 帧（对称转置卷积天生超前 2 帧，
  右移后依赖变为过去 4 帧）；LSTM 单向
- 单测 `test_time_causality`：扰动 2s 后的输入，2s 前输出逐位不变（±ISTFT 窗 margin）

**时延鲁棒**：数据层（WP-07D 合成已含 10-80ms 随机时延）+ 训练层（dataset 再对
far 加 ±10ms 帧级抖动）。`eval.py --jitter-sweep` 输出 ±5/10/20/40ms 的指标衰减曲线。

## 用法

```bash
# 数据（已生成）：D:/data/aec-synth/train_v1（3000 条：FEST 1205/DT 882/NST 913）
#   生成命令见 README 底部；RIR 池 60 条（pyroomacoustics，100ms 截断）

python train.py --data D:/data/aec-synth/train_v1 --val D:/data/aec-synth/eval_v3 \
    --device cuda --epochs 60                       # 主配置（γ=0.5, jitter 10ms）
python train.py --device cuda --gamma 0             # 消融：γ=0（复数谱 MSE）
python train.py --device cuda --jitter-ms 0         # 消融：无时延扰动
python train.py --smoke                             # 合成数据链路冒烟

python eval.py --ckpt outputs/nnaec_g0.5_j10/best.ckpt --compare-trad   # + WP-06 对比表
python eval.py --ckpt ... --jitter-sweep                                # 时延鲁棒性
```

损失：`weighted_distortion(enh谱, near谱, γ=0.5) + 0.1×SI-SNR`（core/losses 复用；
γ 消融开关即 WORKLIST 要求的"γ 消融"）。

## AECMOS 官方口径（已集成，`aecmos_eval.py`；模型 AECMOS_local/Run_1668423760）

```bash
python aecmos_eval.py --trad                         # 传统基线（outputs/aecmos_trad_wp06.json）
python aecmos_eval.py --nn-ckpt <ckpt>               # NN-AEC
python aecmos_eval.py --enh-dir <dir>                # 任意增强输出目录
```
口径说明：官方对 ICASSP 长测试集要求裁剪送评（FEST 后半段等）；eval_v3 是 8s 短片段
（2s DT lead-in 为滤波器收敛设计），全段送评——绝对值与官方比赛有口径差，同集对比结论不受影响。
返回值 (echo_mos, deg_mos)：echo 管回声抑制、deg 管近端退化（PESQ 抓不到的部分 AECMOS 抓得到，
实测传统 AEC DT 段 PESQ +0.27 而 deg_mos 反而 -0.78）。

## 结果（主配置 v4，2026-10-10 实测，同评估集 eval_v3 60 条，三方同口径）

**PESQ/ERLE（eval.py --compare-trad）**：

| 指标 | 传统 AEC (WP-06) | NN-AEC v4（ckpt=v4） | 备注 |
|---|---|---|---|
| FEST ERLE | 20.37 dB（同集 eval_v3） | **69.83 dB** | 掩码门控 ≈ 静音输出（口径产物，不可与线性 AEC 的 ERLE 直接比） |
| DT 近端 PESQ | **1.641** | 1.403 | 近端保真是短板（见差距分析） |
| DT STOI | 0.790 | 0.731 | |
| NST 近端 PESQ | 2.446 | **2.477** | 透传且略有增益 |

**AECMOS 官方口径（aecmos_eval.py，本地 ONNX）**：

| 场景 | 指标 | 原始 mic | 传统 AEC | NN-AEC v4（ckpt=v4） | DoD |
|---|---|---|---|---|---|
| FEST | echo_mos | 1.321 | 3.836 | **3.762** | |
| DT | echo_mos | 1.521 | 2.467 | **3.754** | ✅ +1.29（要求 +0.3） |
| DT | deg_mos | 3.180 | **2.401** | 1.743 | ⚠️ 未达标（结构性瓶颈） |
| NST | echo_mos | — | — | **5.0** | |
| NST | deg_mos | 2.665 | 2.665 | 2.073 | ⚠️ 有轻度伪影 |

**DoD 判定**：echo 维度达标（DT 3.754 vs 传统 2.467，+1.29 ≥ 要求 +0.3）；近端保真
维度未达标（deg 1.743 vs 传统 2.401）——官方结论：**模型形态与 AECMOS-deg 定义之间
的结构性瓶颈**（证据见下方探索矩阵）。

**近端保真度的完整探索矩阵（2026-10-10 汇总，每格都是真实训练+评估）**：

| 变体 | 训练配置 | DT echo | DT deg | DT PESQ | 结论 |
|---|---|---|---|---|---|
| v1 | γ0.5 w0.1, 60ep | 3.478 | 1.919 | 1.405 | 基线 |
| v2 | lr 5e-4, ~100ep | 3.277 | 2.014 | 1.451 | 欠收敛 |
| v3/night | w0.5 续训, 186ep | 3.491 | 1.931 | 1.404 | deg 不随训练量动 |
| v4 | w1.0 + DT/NST 专注, 240ep | **3.754** | 1.743 | 1.403 | echo↑但 deg 反降 |
| γ=0 消融 | γ0 w0.5, 77ep | — | — | 1.391 | γ 影响小 |

**deg 是结构性瓶颈的判断依据**：五个变体跨越了训练量(60→240ep)、损失权重
(w_sisnr 0.1→1.0)、场景配比(全场景→DT/NST 专注)、γ(0→0.5)四个维度，
deg 始终落在 1.7-2.0 窄带内——**不是欠训，是模型形态与 AECMOS-deg 定义之间的
系统性差距**。候选解释（下一步验证方向，README 纪律）：
1. NN 输出的"过干净"频谱（无自然噪声地板/房间感）被 AECMOS 判为失真；
2. DT 段起始的掩码开关瞬态（FEST 门控 → DT 打开）；
3. 官方评估协议要求 DT 只送最后 (len-15)/2 秒——我们的 8s 短片段含 2s lead-in
   过渡段，deg 可能被过渡段主导（分帧 AECMOS 可验证）。

## 训练数据复现命令

```bash
# RIR 池（60 条，pyroomacoustics 随机房间，100ms 车厢尺度截断）
python -c "import sys; sys.path.insert(0,'scripts'); import numpy as np, soundfile as sf; \
import aec_synth as S; from pathlib import Path; rng=np.random.default_rng(0); \
p=Path('D:/data/aec-synth/rir_pool'); p.mkdir(parents=True,exist_ok=True); \
[sf.write(p/f'rir_{i:03d}.wav',(lambda h:(h/abs(h).max()*0.9).astype('float32'))(S.rir_simulate_pra(rng)[:1600]),16000) for i in range(60)]"

python ../../scripts/aec_synth.py --num 3000 --dur-sec 4 \
  --out-dir D:/data/aec-synth/train_v1 --rir-dir D:/data/aec-synth/rir_pool \
  --rir-max-ms 100 --snr-range 15,30 --scenarios 0.4,0.3,0.3 \
  --soft-clip-prob 0.15 --dt-leadin-sec 1.5 --seed 21 \
  --near-dir D:/data/voicebank-demand-16k/extracted/train/clean \
  --far-dir  D:/data/voicebank-demand-16k/extracted/train/clean \
  --noise-dir D:/data/dns-challenge-mirror/datasets/noise
```

## 已知口径与局限（README 明示纪律）
- STFT 512/320：20ms 帧率 + 12ms 窗内前瞻——介于 2021 赛规格（40ms 含 20ms 前瞻）
  与 2022 规格（20ms 无前瞻）之间；严格无前瞻需 win=320 非 2 次幂 FFT，留作改进
- 流式推理 = WP-05b cache 化（conv 时间 cache + LSTM state）；架构已因果，
  等价性测试模式复用 WP-05b 的流式 vs 非流式逐帧对齐
- AECMOS 集成延后（降级路径已按 WORKLIST 授权）；非线性回声（soft-clip 15% 样本
  在训练集里）NN 天然能学一部分——WP-08 的对照组
- train/eval 同用 28 说话人池（同池不同句），未见说话人泛化此前未单独评估——
  已补全全未见说话人（DNS clean 素材）重评：exp/07_nn_aec/outputs/eval_v4_speaker_heldout/summary.md

## 流式推理（WP-05b 模式已实现，`streaming.py`）

```python
from model import NNAEC
from streaming import NNAECStreaming
st = NNAECStreaming(model, device="cpu")
out_chunk = st.process_chunk(mic_320, far_320)   # 20ms 进 20ms 出，块边界任意
```
- conv cache（每层 5 帧输入）+ LSTM state 外置逐帧推进 + 核心库新增
  `core.istft.StreamingISTFT`（滑动缓冲 WOLA，帧到即出块）
- **流式 vs 非流式逐帧对齐 <1e-5**（实测 ~1e-7，tests/test_streaming_nnaec.py 锁死；
  不定块长 800/640/960 混合推入同样一致）
- cache 内存 ~270KB 有界、与音频总长无关（`cache_size_bytes()` 记账，
  单测断言滚动不增长）
- 两个实现级细节（坑已填平进测试）：deconv 右移 2 帧的启动延迟 = 前 2 帧输出零；
  StreamingISTFT 默认窗是 torch 同款**周期** hann（np.hanning 是对称窗，对拍必翻车）
