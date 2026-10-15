# exp/04_dccrn — DCCRN 语音增强（WP-04 / 模块4）

Deep Complex CRN（Hu et al., ICASSP 2020，DNS Challenge 2020 冠军方案）复现变体。
**参数量 3.99M**（论文 3.71M 同量级，`python param_check.py` 校验）、4s 前向 ≈ 17 GMACs。
9 项单测全绿（复数代数手算对拍 / CLSTM 单步手算 / 参数量带 / 任意长度 / DM 精度 / 过拟合）。

## 阶段1：共享 DM（清单圈定，P1-5a，2026-10-13）

**为什么改**：旧数据接口 `DNSDynamicMix` 把池写死在
`D:/data/dns-challenge-mirror/datasets/{clean,noise}`，**绕过 `split_lists/` 的分割清单**
⇒ 训练会吃掉 holdout15 reader 与 noise_eval15（隔离性失效、泄漏风险，t12 发现）。
`--dynamic-mixing` 让 04 与 07 一样**清单圈定**：三池 + 三清单显式给，入口做
near/far 互斥与 noise∩speech 校验，清单版本号/池规模/配比进 `train_manifest.json`。

**语义（必须同页写）**：数据 = **AEC 合成三元组**（`exp/07_nn_aec/dm_dataset.AecDynamicMix`
经 `dccrn_dm.DccrnDynamicMix` 适配），投影成 04 的 Dataset 契约：
**`noisy = mic`**（含回声 + 噪声的麦克风信号）、**`clean = near`**（近端目标），`far` 一并
返回但 DCCRN 不消费（`data.key_map` 留痕）。⇒ 阶段1 的 DCCRN = "复数谱掩码模型在**同一条
新数据轴**上做回声+噪声抑制"，与 NN-AEC/级联**同一供给、同一 rng 流、同一 val、同一配比
口径**，三者可比（t11 的吞吐表也按同一供给折算）。

> ⚠️ 这**不是** 04 历史的 DNS 去噪任务：已有读数（devtest PESQ **2.1811**、`--no-dm`
> 2.0151、`--real-conv` 2.1898）仍属 DNS 口径，**两套口径不混比**。要回到 DNS 去噪任务
> 需要另一条数据轴（干净语音池 × 噪声池 = 03/04 的 enhancement 轴），不在本任务范围。

**口径（与 07 同构；实现在 `core/train_protocol.py` + `dm_dataset.py`，单一来源）**：

| 项 | 取值 |
|---|---|
| epoch 方案(a) | **等效更新步数对齐 04 历史**：100ep × epoch_len1500 / bs8 = **18,700 步**（`dccrn_dm.DCCRN_REF`，出处 README §结果）；`--epochs 100 --batch-size 64` ⇒ 187 步/ep、epoch_len 11,968、总 18,700（Δ=0） |
| LR / 早停 | DM 缺省 `ReduceLROnPlateau(0.5, patience=20)` + **关早停**（阶段0 §2.3b）；非 DM 保持现状 patience 3 / early-stop 10 |
| 场景配比 | `--dm-scenarios 0,0.5086,0.4914`（顺序 = **FEST,NST,DT**；= v4 实际训练配方，无 FEST）；manifest 记**具名字典** + `scenarios_explicit` |
| 从零起 | DM 拒绝从非 DM 谱系（DNS）ckpt 续训（阶段0 §2.3a）；同口径可续训，口径漂移重置 best/bad |
| val | **静态** `--val <aec_synth 目录>`（与 NN-AEC 同集，模型选择口径可比；val 不 jitter） |
| 留痕 | `out/train_manifest.json`（run/loss/data{池、清单路径+sha256+计数、互斥、dt_plan 版本/sha、key_map、配比/来源}/epoch_scheme/lr_sched/probe_mem/history_tail）+ ckpt `cfg["train_manifest"]` + stdout；`history.json` 每 epoch 加 epoch_len/steps_per_epoch/target_steps/schedule/patience/early_stop |

**阶段1 可照抄命令（bs64 口径，与 NN-AEC 命令同构；`--val` 必须与 NN-AEC 用同一集）**：

```bash
python exp/04_dccrn/train.py --device cuda --dynamic-mixing --epochs 100 \
    --batch-size 64 --num-workers 10 --probe-mem 20 \
    --dm-scenarios 0,0.5086,0.4914 \
    --near-dir <srv>/data/dns-clean/near --far-dir <srv>/data/dns-clean/far \
    --noise-dir <srv>/data/dns-noise \
    --near-list split_lists/speech_near_train70_files.txt \
    --far-list  split_lists/speech_far_train70_files.txt \
    --noise-list split_lists/noise_train70.txt \
    --rir-dir <rir-pool> --rir-max-ms 100 \
    --val D:/data/aec-synth/eval_v5 \
    --out exp/04_dccrn/outputs_dm
# 输出目录：exp/04_dccrn/outputs_dm/dccrn_dm（tag 带 _dm，不覆盖历史产物）
# 期望：187 步/epoch × 100ep = 18,700 步（方案 a，与 04 历史等效）；D1 供给按同一 DM 池折算
```

**回归护航**：不开 `--dynamic-mixing` 时**逐键不变**——ckpt `cfg` 仍是历史键集
（新增参数被过滤掉）、`early_stop` 仍 10、`epoch_len` 仍 2000、不产出 `train_manifest.json`、
tag 仍 `dccrn[_realconv][_nodm]`（单测 `tests/test_dccrn_dm.py` 锁死）。

## 结构

```
noisy [B,N] ─STFT(512/200)─> [B,2,257,T]
  ComplexConv2d ×4  ch [32,64,96,128]  k5 s(2,2)     F:257→17, T:321→21 (4s)
  flatten 2176(complex) → ComplexLSTM ×2 (hidden 64) → FC 还原
  ComplexConvTranspose2d ×4 + 跳连 → 1×1 head → cRM
  enh = cRM ⊙ noisy ─ISTFT─> [B,N]
```

复数层定义（`complex_nn.py`，Trabelsi et al. ICLR 2018）：
`(W_re+jW_im)(x_re+jx_im) = (W_re x_re − W_im x_im) + j(W_re x_im + W_im x_re)`
——两路实权重四次实矩阵乘。复数卷积能表达**相位旋转**（实数卷积做不到），
这是 cRM 系模型处理相位的核心能力来源。

## 与论文/官方 repo 的差异（复现纪律，逐条可查）

| 项 | 官方 | 本实现 | 原因 |
|---|---|---|---|
| 窗 | win 400 补零到 512 | hann(512) 全长 | core.STFT 窗长=n_fft 约定 |
| 归一化层 | 复数 LayerNorm | 实数视图 BatchNorm2d | 简化；离线训练等价可用 |
| 通道/hidden | [32,64,128,256]/LSTM 256 | [32,64,96,128]/CLSTM 64 | 压到 3.99M 且 8GB 显存 batch 8-12 |
| 损失 | SI-SNR + 多分辨率复数谱 | 复数谱 MSE + 0.1×SI-SNR | 复用 core/losses；多分辨率损失 WP-02 已有可换 |
| CLSTM 激活 | 复数 tanh（部分版本） | 分量 σ/tanh | 标准简化，实测稳定 |
| real-conv 消融 | — | 2ci→2co 实数卷积 | 论文惯例：同结构实数化（卷积参数 ×2），对照结构而非参数量 |

## 用法

```bash
python param_check.py                                  # 参数量/MACs 校验（DoD）
python train.py --device cuda --epochs 100 --epoch-len 1500   # DM 长训练
python train.py --smoke                                # 合成数据链路冒烟
python train.py --device cuda --no-dm                  # 消融：静态混合
python train.py --device cuda --real-conv              # 消融：实数卷积
python eval.py --ckpt outputs/dccrn/best.ckpt --devtest        # DNS2020 官方 devtest
python eval.py --ckpt outputs/dccrn/best.ckpt --synth 100      # DNS 素材自混 held-out
```

数据管道（`dynamic_mixing.py`）：DNS clean 10000 × noise 660 在线随机配对，
SNR U(-5,15)dB、干净语音 RMS 归一 U(-35,-20)dBFS、4s 片段、val 固定 seed 可复现。
DM 等效样本空间 ~660 万组合 >> 静态预混——DoD PESQ≥2.6 的前提。

## 硬件与时长（RTX 4060 Laptop 8GB 实测）

| 项 | 数值 |
|---|---|
| batch 8 × 4s 单步 | ~1.3 s（含 DM 数据加载） |
| epoch（epoch_len=1500，190 步） | ~37-43 s |
| 显存 | ~3.0 GB（batch 8） |
| 2 小时预算 | ~180 epochs（early-stop 12 通常更早收敛） |

## 结果（2026-10-09 实测回填，μ 默认配置，100 epochs × epoch_len 1500，4060 上 ~1h）

val loss（复数谱 MSE + 0.1×SI-SNR）：-1.269（epoch ~94 best，全程无过拟合迹象）

| 测试集 | PESQ(wb) | STOI | SI-SNR | 带噪基线 |
|---|---|---|---|---|
| **DNS2020 官方 devtest**（synthetic_no_reverb 150 条，源 `outputs/eval/dccrn/summary_devtest.json`） | **2.1811** | 0.9384 | 15.61 | 1.582 / 0.915 / 9.23 |
| DNS 素材自混 held-out（100 条，SNR U(-5,15)，seed 777） | 1.787 | 0.877 | 12.34 | 1.310 / 0.832 / 4.55 |
| （历史 run，早期口径） | 2.161 | — | — | — |

> 主结果以 `outputs/eval/dccrn/summary_devtest.json` 为准（devtest 150 条口径）；
> 2.161 为早期 run 的历史值，仅作存档，不再作为主结果引用。

**DoD 判定：未达 2.6，走 WORKLIST 降级路径**（趋势复现 + 差距分析）：
- 趋势成立：官方 devtest 上 +0.60 PESQ / +6.4dB SI-SNR，方向与论文一致
- 差距分析（按影响排序）：
  1. **训练量 ~4%**：18.7K steps（100ep×187 步）vs 官方配方 500K+ steps——
     val 曲线仍在缓降，纯粹没喂饱；续训 `--resume outputs/dccrn/last.ckpt` 即可
  2. **模型压缩**：通道 [32,64,96,128] vs 官方 [32,64,128,256]（我们 3.99M≈官方
     3.71M 是靠 CLSTM hidden 64 省的，卷积容量小于官方）
  3. **数据子集**：DNS clean 10k 条 vs 官方全量（含增广配方差异）
- 消融已回填：`--no-dm`（静态混合对照）与 `--real-conv`（复数运算价值）均已实测，
  见下表；real-conv 反直觉 ≥ 复数，归因分析见 `docs/adr/0004-dccrn-real-vs-complex.md`

| 消融 | devtest PESQ | 备注 |
|---|---|---|
| DCCRN+DM（默认，主） | **2.1811** | 已测（源 `outputs/eval/dccrn/summary_devtest.json`） |
| 静态混合 --no-dm | 2.0151 | 已测；DM 价值 ≈ +0.17 PESQ |
| 实数卷积 --real-conv | 2.1898 | 已测；反直觉：实数与复数性能相当（Δ0.009 噪声级，不称反超；参数约 2×），归因见 `docs/adr/0004-dccrn-real-vs-complex.md` |

## 降级路径（WORKLIST 预案）
PESQ 2.4-2.5 → 固定测试集趋势复现 + 完整消融 + 差距分析（DM 细节/训练时长/
通道压缩），README 明示——复现差距分析本身是面试好材料。
