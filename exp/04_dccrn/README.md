# exp/04_dccrn — DCCRN 语音增强（WP-04 / 模块4）

Deep Complex CRN（Hu et al., ICASSP 2020，DNS Challenge 2020 冠军方案）复现变体。
**参数量 3.99M**（论文 3.71M 同量级，`python param_check.py` 校验）、4s 前向 ≈ 17 GMACs。
9 项单测全绿（复数代数手算对拍 / CLSTM 单步手算 / 参数量带 / 任意长度 / DM 精度 / 过拟合）。

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
| **DNS2020 官方 devtest**（synthetic_no_reverb 150 条） | **2.161** | 0.938 | 15.50 | 1.582 / 0.915 / 9.23 |
| DNS 素材自混 held-out（100 条，SNR U(-5,15)，seed 777） | 1.787 | 0.877 | 12.34 | 1.310 / 0.832 / 4.55 |

**DoD 判定：未达 2.6，走 WORKLIST 降级路径**（趋势复现 + 差距分析）：
- 趋势成立：官方 devtest 上 +0.58 PESQ / +6.3dB SI-SNR，方向与论文一致
- 差距分析（按影响排序）：
  1. **训练量 ~4%**：18.7K steps（100ep×187 步）vs 官方配方 500K+ steps——
     val 曲线仍在缓降，纯粹没喂饱；续训 `--resume outputs/dccrn/last.ckpt` 即可
  2. **模型压缩**：通道 [32,64,96,128] vs 官方 [32,64,128,256]（我们 3.99M≈官方
     3.71M 是靠 CLSTM hidden 64 省的，卷积容量小于官方）
  3. **数据子集**：DNS clean 10k 条 vs 官方全量（含增广配方差异）
- 消融待跑：`--no-dm`（静态混合对照）与 `--real-conv`（复数运算价值）各 ~1h，
  跑完回填下表

| 消融 | devtest PESQ | 备注 |
|---|---|---|
| DCCRN+DM（默认） | 2.161 | 已测 |
| 静态混合 --no-dm | 待跑 | DM 价值量化 |
| 实数卷积 --real-conv | 待跑 | 复数运算价值 |

## 降级路径（WORKLIST 预案）
PESQ 2.4-2.5 → 固定测试集趋势复现 + 完整消融 + 差距分析（DM 细节/训练时长/
通道压缩），README 明示——复现差距分析本身是面试好材料。
