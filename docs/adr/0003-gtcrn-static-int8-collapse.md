# ADR 0003 — GTCRN 静态 INT8（QDQ）为什么崩坏：小模型量化的边界

- 状态：已确认（负面结果留档）
- 日期：2026-10-10
- 关联：`exp/05a_gtcrn_onnx/quantize.py`、`docs/NUMBERS.md` §1

## 背景

端侧部署要 INT8。两条 PTQ 路线：
- **dynamic PTQ**：只量化权重（Linear/GRU 权重→INT8），激活保持 fp32；
- **static QDQ PTQ**：权重 + 激活都量化，需校准数据统计激活范围，全 INT8 算子。

GTCRN 只有 23.67K 可训练参数，是检验"小模型量化边界"的绝佳样本。

## 实测（DNS 自混 held-out 50 条）

| 方案 | PESQ | STOI | SI-SNR | 单帧 | 体积 |
|---|---|---|---|---|---|
| fp32 | 1.869 | 0.877 | 11.54 | 0.59ms | — |
| **dynamic PTQ** | 1.803（**-0.067**） | 0.869 | 11.28 | 0.64ms | 240KB ✅ 可用 |
| **static QDQ** | 1.120（**-0.75**） | 0.197 | **-20.8** | 0.87ms | 崩坏 ❌ |

static QDQ 输出接近噪声（SI-SNR 从 +11.5 → -20.8，STOI 0.877→0.197），完全不可用。

## 崩坏机理（三要素叠加）

1. **GRU 递归状态逐帧累积量化误差。**
   static QDQ 把 GRU 的隐藏状态也量化——状态是逐帧递归的，量化误差随时间**累积放大**，
   几百帧后状态已严重偏离 fp32 轨迹。dynamic 只量化权重、状态仍 fp32，故无累积。

2. **23.67K 小模型没有冗余吸收量化噪声。**
   大模型的量化噪声可被海量参数的冗余"摊薄"，GTCRN 每个参数都"在刀刃上"，
   量化引入的相对扰动无处吸收。

3. **LayerNorm / Tanh 对 scale 敏感。**
   LN 的归一化和 Tanh 的饱和区在 INT8 的粗粒度 scale 下失真严重，进一步放大误差。

## 决策与量产路径

- **PTQ 只可用 dynamic**（精度损失 -0.067 PESQ 可接受，但提速仅 ~8%——小模型瓶颈在
  Conv 算子本身，不在被量化的 Linear/GRU）；
- 要真正 INT8 加速需 **QAT（训练时伪量化）** 或 **混合精度**（GRU/LN 留 fp16/int16、
  Conv 用 int8）——dynamic PTQ 在小模型上"提速有限"。

## 面试标准答案

> "GTCRN 静态量化崩坏是我最有价值的负面结果。崩在三点：GRU 状态量化误差逐帧累积、
> 23.67K 小模型没冗余吸收量化噪声、LayerNorm/Tanh 对 scale 敏感。dynamic 只量化权重所以
> 只掉 0.067，但提速也只有 8%——小模型瓶颈在 Conv。所以量产 INT8 不能指望 PTQ，
> 要么 QAT 让网络在训练时就适应量化噪声，要么混合精度把敏感的 GRU/LN 留在高精度。"

## 复现

```bash
python quantize.py --mode dynamic   # 可用，quant_report_dynamic.json
python quantize.py --mode static    # 复现崩坏（教学对照，gtcrn_dns3.int8_static_BAD.onnx）
```
