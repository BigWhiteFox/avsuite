# exp/08_nl_aec — 非线性 AEC（WP-08 选做）：Ivry 2021 级联

功放 soft-clip 失真是线性自适应滤波的物理天花板（实测纯线性 ERLE ~13dB 封顶）。
本模块用级联结构突破：**MDF 线性级消长记忆 → GRU 学记忆less 非线性残差**。

## 两个实验（v1 负结果 → v2 级联）

| | v1：GRU 裸接 far | v2：级联（MDF → GRU 残差） |
|---|---|---|
| 结构 | GRU 直接学 far→echo | GRU 学 lin→非线性残差 |
| 结果 | **ERLE 0dB**（学了空气） | 见 outputs/eval_nl_v2.json |
| 归因 | 回声 = 10-80ms 时延 + 100ms RIR ≈ **180ms 记忆**，逐样本 GRU 容量不够 | 线性级消化记忆后，GRU 只需建模 tanh 型静态非线性 |

**v1 的负结果本身就是面试素材**：它解释了 Ivry 级联结构存在的动机——
"为什么不能一个 RNN 端到端学 AEC：长记忆 + 非线性两个问题拆开各自有最优解"。

## 用法

```bash
# 数据（soft-clip 100%）：D:/data/aec-synth/{train_nl 1500条, eval_nl 60条}
# 1) 预计算线性级回声估计（CPU，~40min，幂等可断点续）
python precompute_lin.py --synth-dir D:/data/aec-synth/train_nl
python precompute_lin.py --synth-dir D:/data/aec-synth/eval_nl
# 2) 训练 GRU 残差级
python train.py --device cuda
# 3) 验收：级联 vs 纯线性 ERLE（DoD +5dB）+ DT 近端 PESQ
python eval.py --ckpt outputs/nl_aec_v2/best.ckpt
```

## 验收结果（2026-10-10 实测，eval_nl 43 条 FEST + 10 条 DT）

| 配置 | FEST ERLE | DT 近端 PESQ | DoD(+5dB) |
|---|---|---|---|
| 纯线性 (MDF 回声估计) | 5.78 dB | 1.687 | 基线 |
| 线性+GRU(128, 1s段) 级联 | 6.06 dB (**+0.28**) | 1.724 (+0.04，无损伤) | ⚠️ 未达 +5dB |
| 线性+GRU(128, 2s段) 级联 | 6.18 dB (+0.40) | 待回填 | ⚠️ 未达 +5dB |

**降级路径（WORKLIST 明示）**：开源复现跑通 ✓（级联结构端到端跑通、方向正确：
残差 MSE 减半、DT 近端无损伤）+ 对比分析文档 ✓（本文档 v1/v2 双实验）。

**为什么 +0.28dB 而不是 +5dB（诚实分析，面试可答）**：
1. 本数据集的非线性"天花板"比预期低：soft-clip ceiling 0.5 vs 回声振幅 0.1-0.3
   → 大部分样本工作在 tanh 线性区，非线性残余占比小（无 RES 时线性级 5.8dB，
   RES 可达 13dB——残余大头其实是线性 misadjustment 而非非线性）；
2. GRU 目标里混着线性级的对齐/misadjustment 伪影——不是 lin 的纯函数，
   学不动；
3. 真实验收该用"深饱和"数据集（soft-clip ceiling 压到 0.1）或真实功放录音。
   改进路径已明确，留作选做深挖。

**面试价值排序**：v1 负结果（为什么不能裸 RNN 学 AEC）> 级联结构本身 >
ERLE 数字——这是"知道什么方案为什么失败"的证据链。

## 与批内模块的关系
- 线性级 = WP-06 TradAEC 的直接复用（precompute_lin.py 一行调用）
- 数据集 = WP-07D 的 soft-clip 开关（--soft-clip-prob 1.0）
- 面试叙事：WP-06（线性）→ WP-08（非线性级联）→ 完整回答"功放失真怎么办"
