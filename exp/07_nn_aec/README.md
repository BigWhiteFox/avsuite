# exp/07_nn_aec — NN-AEC 神经网络回声消除（WP-07 / 模块7，简历核心）

双输入（mic + far 复数谱）压缩 CRN，**0.638M 参数（<1M 约束）**，时间维严格因果
（单测锁死），20ms 帧率。对手是 WP-06 传统 AEC——同评估集直接对比。
单测全绿；全仓 261 项 = 257 passed + 4 xfailed。

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

损失：`weighted_distortion(enh谱, near谱, γ=0.5) + w_sisnr×SI-SNR`（默认 w_sisnr=0.1，
**主配置 v4 为 w_sisnr=1.0**；core/losses 复用，γ 消融开关即 WORKLIST 要求的"γ 消融"）。

## AECMOS 官方口径（已集成，`aecmos_eval.py`；模型 AECMOS_local/Run_1668423760）

```bash
python aecmos_eval.py --trad                         # 传统基线（outputs/aecmos_trad_wp06.json）
python aecmos_eval.py --nn-ckpt <ckpt>               # NN-AEC
python aecmos_eval.py --enh-dir <dir>                # 任意增强输出目录
```
口径说明：官方对 ICASSP 长测试集要求裁剪送评（FEST 后半段等）；eval_v3 是 8s 短片段
（2s DT lead-in 为滤波器收敛设计），全段送评——绝对值与官方比赛有口径差，同集对比结论不受影响。
**送评采样率（2026-10-11 口径修复，V5）**：本地模型 `Run_1668423760_Stage_0.onnx` 的实际
`sampling_rate = 48000`，官方入口用 `librosa.load(sr=48000)`；本脚本曾把 16k 波形**直送**
（绕过重采样）⇒ 自合成集分数全部失真。现已统一重采样到模型采样率（`resample_poly`，gcd 约简），
口径定名 **AECMOS@48k 模型协议**；**旧值全部作废**（见 `docs/NUMBERS.md` 顶部决策条与
`reports/aecmos_rescore_v5fix.md`）。
返回值 (echo_mos, deg_mos)：echo 管回声抑制、deg 管近端退化（PESQ 抓不到的部分 AECMOS 抓得到，
实测传统 AEC DT 段 PESQ +0.27 而 deg_mos **−0.361**（新口径；旧口径 −0.78）。

## 结果（主配置 v4，eval_v3 60 条，三方同口径；AECMOS 列为 **@48k 模型协议**）

**PESQ/ERLE（eval.py --compare-trad）**：

| 指标 | 传统 AEC (WP-06) | NN-AEC v4（ckpt=v4） | 备注 |
|---|---|---|---|
| FEST ERLE | 20.37 dB（同集 eval_v3） | **69.83 dB** | 掩码门控 ≈ 静音输出（口径产物，不可与线性 AEC 的 ERLE 直接比） |
| DT 近端 PESQ | **1.641** | 1.403 | 近端保真是短板（见差距分析）；1.641 为 `eval.py --compare-trad` 口径，`eval_aec.py` 验收口径 **1.687**（脚本间口径差，见 `docs/NUMBERS.md` WP-06 口径注） |
| DT STOI | 0.790 | 0.731 | |
| NST 近端 PESQ | 2.446 | **2.477** | 透传且略有增益 |

> <sub>**SI-SNR 口径注（V5/t24 传导）**：FEST 段的近端参考恒为零 ⇒ SI-SNR **无定义**，
> 评估链按 **NaN** 记录并跳过（`core.evaluate._si_snr_np` 对 ref/est 全零返回 NaN；
> 修正前 est 全零会得到 **0.0 dB** 的假"完美"分）。因此本表不列 FEST SI-SNR，
> `tools/evaluate_full.py` 汇总时该列全 NaN 的组也不参与平均（见 `_score_one` 注释）。</sub>

**AECMOS（aecmos_eval.py，本地 ONNX，@48k 模型协议）**：

| 场景 | 指标 | 原始 mic | 传统 AEC | NN-AEC v4（ckpt=v4） | DoD |
|---|---|---|---|---|---|
| FEST | echo_mos | 1.321\* | 3.207 | **3.194** | （新口径下 FEST echo 传统/NN 实质打平；级联 3.938 最高） |
| DT | echo_mos | 1.521\* | 3.494 | **4.175** | ✅ **+0.681±0.212**（要求 +0.3，CI 不含 0） |
| DT | deg_mos | 3.180\* | **3.598** | 3.237 | ⚠️ **仅 DT 段落后 −0.361±0.169**（机理=高频带选择性损伤+谱平坦化，非欠训） |
| NST | echo_mos | — | — | **5.0** | |
| NST | deg_mos | 2.665\* | 3.842 | **3.941** | ✅ **新口径下不再落后** |

> \* 原始 mic 列为旧口径（地板重评进行中，`docs/NUMBERS.md` 同注）。**旧口径对照（作废留档）**：DT echo 1.521/2.467/3.754、
> DT deg 3.180/2.401/1.743、NST deg 2.665/2.665/2.073、FEST echo 1.321/3.836/3.762（新口径 trad 3.207 / NN 3.194）——16k 直送错配，
> 见 `reports/aecmos_rescore_v5fix.md`；逐集新旧对照表见 `exp/07_nn_aec/outputs/aecmos48_v5fix/_t22_backfill.md`。

**DoD 判定**：echo 维度达标（DT 4.175 vs 传统 3.494，**+0.681±0.212 ≥ 要求 +0.3**）；
近端保真维度**只在 DT 段**未达标（deg 3.237 vs 传统 3.598，−0.361±0.169；NST 段 3.941 vs 3.842
**已不落后**）——结论按 V5 决策条收窄为：**DT 段存在 AECMOS-deg 落后，机理为高频带选择性损伤 +
谱平坦化**；~~"模型形态与 AECMOS-deg 定义之间的（全场景）结构性瓶颈"~~ **该表述作废**。

**近端保真度的完整探索矩阵（2026-10-10 汇总，每格都是真实训练+评估；**DT echo/deg 为旧的 16k 直送口径，已作废**，仅作变体间相对趋势留档，绝对值见上方 @48k 表）**：

| 变体 | 训练配置 | DT echo（旧口径） | DT deg（旧口径） | DT PESQ | 结论 |
|---|---|---|---|---|---|
| v1 | γ0.5 w0.1, 60ep | 3.478 | 1.919 | 1.405 | 基线 |
| v2 | lr 5e-4, ~100ep | 3.277 | 2.014 | 1.451 | 欠收敛 |
| v3/night | w0.5 续训, 186ep | 3.491 | 1.931 | 1.404 | deg 不随训练量动 |
| v4 | w1.0 + DT/NST 专注, 240ep | **4.175（新口径）** | **3.237（新口径）** | 1.403 | echo↑但 deg 仍略降 |
| γ=0 消融 | γ0 w0.5, 77ep | — | — | 1.391 | γ 影响小 |

**deg 是"DT 段 + 掩码形态"问题的判断依据**（V5 收窄）：五个变体跨越了训练量(60→240ep)、
损失权重(w_sisnr 0.1→1.0)、场景配比(全场景→DT/NST 专注)、γ(0→0.5)四个维度，
**DT 段 deg 始终低于同集传统基线**——不是欠训；而 **NST 透传段在新口径下不再落后**
（3.941 vs 3.842），故结论限定为 **DT 段**的机制性差距（高频带选择性损伤 + 谱平坦化），
不再宣称"全场景结构性瓶颈"。候选解释（下一步验证方向，README 纪律）：
1. NN 输出的"过干净"频谱（无自然噪声地板/房间感）被 AECMOS 判为失真；
2. DT 段起始的掩码开关瞬态（FEST 门控 → DT 打开）；
3. 官方评估协议要求 DT 只送最后 (len-15)/2 秒——我们的 8s 短片段含 2s lead-in
   过渡段，deg 可能被过渡段主导（分帧 AECMOS 可验证）。

## 训练数据复现命令

```bash
# 全部命令从仓库根（avsuite/）执行，脚本路径相对仓库根
# RIR 池（60 条，pyroomacoustics 随机房间，100ms 车厢尺度截断）
python -c "import sys; sys.path.insert(0,'scripts'); import numpy as np, soundfile as sf; \
import aec_synth as S; from pathlib import Path; rng=np.random.default_rng(0); \
p=Path('D:/data/aec-synth/rir_pool'); p.mkdir(parents=True,exist_ok=True); \
[sf.write(p/f'rir_{i:03d}.wav',(lambda h:(h/abs(h).max()*0.9).astype('float32'))(S.rir_simulate_pra(rng)[:1600]),16000) for i in range(60)]"

python scripts/aec_synth.py --num 3000 --dur-sec 4 \
  --out-dir D:/data/aec-synth/train_v1 --rir-dir D:/data/aec-synth/rir_pool \
  --rir-max-ms 100 --snr-range 15,30 --scenarios 0.4,0.3,0.3 \
  --soft-clip-prob 0.15 --dt-leadin-sec 1.5 --seed 21 \
  --near-dir D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean \
  --far-dir  D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean \
  --noise-dir D:/data/dns-challenge-mirror/datasets/noise
```

## 已知口径与局限（README 明示纪律）
- STFT 512/320：20ms 帧率 + 12ms 窗内前瞻——介于 2021 赛规格（40ms 含 20ms 前瞻）
  与 2022 规格（20ms 无前瞻）之间；严格无前瞻需 win=320 非 2 次幂 FFT，留作改进
- 流式推理 = WP-05b cache 化（conv 时间 cache + LSTM state）；架构已因果，
  等价性测试模式复用 WP-05b 的流式 vs 非流式逐帧对齐
- AECMOS 官方口径已集成（见上节 `aecmos_eval.py`），**送评前统一重采样到模型采样率 48k**
  （口径名 **AECMOS@48k 模型协议**；16k 直送旧值已作废）；非线性回声（soft-clip 15% 样本
  在训练集里）NN 天然能学一部分——WP-08 的对照组
- **指标 NaN 协议（V5/t24）**：静音/塌陷样本的 PESQ/STOI/SI-SNR 记 **NaN**（不写 0），
  FEST 段因近端参考为零，SI-SNR 恒为 NaN；汇总时全 NaN 的组不参与平均——读表时勿把 NaN 当 0
- 全部合成评估集（eval_v3/v5/v6）soft-clip 概率均为 0，而训练集为 0.15——非线性回声
  维度在主结果表无检验面（官方真实集除外，其回声为真实录制）
- **train/eval 数据重叠实测口径（2026-10-11；实测见 `reports/refactor_wave5/data_inventory.md` §6）**：
  同用 28 说话人池（28/28 重叠），但**语句级并非全不同**——eval_v5 行次中 near **67/300（22%）**、
  far **75/300（25%）** 与 train_v1 **是同一 wav 文件**（`fit_len` 随机裁剪使片段偏移不同，句/人/内容同源；
  DT 层最重：near 36/154、far 48/154）；**噪声池与训练 99~100% 重叠**（eval_v5 297/300，各合成集噪声
  均取自 train_v1 用过的同一 DNS 池），**两个说话人 held-out 集也共享同一 DNS 噪声池**
  （eval_v4_speaker_heldout 59/60、eval_speaker_only_holdout 60/60）——**说话人维度已由 held-out 集缓解，
  噪声维度未隔离**，泛化结论保守解读。未见说话人泛化此前未单独评估——
  已补全全未见说话人（DNS clean 素材）重评：exp/07_nn_aec/outputs/eval_v4_speaker_heldout/summary.md；
  真实分布稳健性证据：`reports/eval_v5_dt300.md`（DT 扩样 154 条）、`reports/ood_real_echoes.md`
  （ICASSP2022 真实回声）、`reports/realrir_domain.md`（IR64-CAR 真实车厢 RIR）

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
