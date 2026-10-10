# exp/03_crn — CRN 语音增强（WP-03 / 模块3）

Tan & Wang 2018 式 CRN：卷积编码 + BLSTM + 对称解码 + cRM 复数掩码，
复用 `core/` 的 STFT/ISTFT/损失/评估。**48 项单测全绿**（含本模块 13 项）。

## 结构

```
noisy [B,N] ─STFT─> [B,257,T] ─Re/Im─> [B,2,257,T]
  Conv(2→18,k5,s(2,1)) ┐
  Conv(18→30)          ├─ 频率 257→33 降采样、时间不变（流式友好）
  Conv(30→40)          ┘ (+跳连缓存)
  flatten(40×33) → 2×BLSTM(256) → Linear → 还原
  DeConv ×3 ←跳连 concat→ 上采样回 257
  1×1 Conv → cRM Re/Im ──⊙ noisy谱──> ISTFT → enhanced [B,N]
```

## 用法

```bash
# 0) 数据（已完成）：parquet -> wav 解包
python -c "import sys; sys.path.insert(0,'exp/03_crn'); from dataset import extract_parquet; extract_parquet()"
#    -> D:/data/voicebank-demand-16k/extracted/{train,test}/{clean,noisy}（11572/824 对）

# 1) 训练（用户 GPU 侧，约 5.5h/100ep）
python exp/03_crn/train.py --device cuda            # 默认 cRM+BLSTM+hidden256
python exp/03_crn/train.py --device cuda --mask-type irm --rnn-type lstm   # 消融变体
python exp/03_crn/train.py --resume outputs/crn_crm_blstm/last.ckpt       # 断点续训
tensorboard --logdir exp/03_crn/outputs/crn_crm_blstm/tb                  # 曲线

# 2) 评估（测试集 824 条 -> PESQ/STOI/SI-SNR）
python exp/03_crn/eval.py --ckpt exp/03_crn/outputs/crn_crm_blstm/best.ckpt

# 3) 消融
python exp/03_crn/ablation.py --quick              # 冒烟（合成，秒级）
python exp/03_crn/ablation.py --real --subset 0.1  # 真实数据预览
```

## 硬件与时长实测（RTX 4060 Laptop 8GB，本机实测 2026-10-08）

| 项 | 数值 |
|---|---|
| 参数量（hidden=256 默认） | **5.60M**（hidden=64 → 1.10M） |
| 单步（batch16×3s，fp32） | 0.30 s |
| 单 epoch（11572×90%/16=650 步） | ≈ 3.3 min |
| 100 epochs（含 early stop 通常更短） | ≈ 5.5 h |
| 峰值显存 | 0.64 GB（**batch 64 可再提速**，显存充裕） |

## 结果（2026-10-09 实测回填，hidden=256 默认配置，epoch 52 best / 64 early-stop）

**VoiceBank-DEMAND 测试集全量 824 对**（协议：16k / PESQ wideband / 全长推理 batch=1 / 截断对齐不补零）：

| 配置 | PESQ(wb)↑ | STOI↑ | SI-SNR↑ | 备注 |
|---|---|---|---|---|
| 带噪基线 | 1.971 | 0.921 | 8.45 | eval.py 自动输出 |
| **cRM + BLSTM（默认）** | **2.529** | **0.936** | **18.76** | ✅ 超过 DoD 目标 ≥2.3 |

- 相对带噪：PESQ +0.56、SI-SNR +10.3dB、STOI +0.015
- 与论文参照：CRN 原文（Tan & Wang 2018，更大模型 + hall 数据协议）≈2.74；本复现用 hidden=256 小模型 + 官方 824 对测试集，2.53 符合预期量级
- val loss 曲线：0.44（ep1）→ 0.0384（best ep52）→ early stop ep64
- 明细：`outputs/eval/best/metrics.csv`（逐文件）/ `summary.json`（汇总）

| 消融变体（2026-10-09 训练回填，同协议 824 对） | PESQ | STOI | SI-SNR |
|---|---|---|---|
| IRM + BLSTM | **2.843** | **0.945** | 17.57 |
| cRM + LSTM（单向因果） | 2.500 | 0.933 | 18.22 |

消融解读：
- **IRM 反超 cRM 0.31 PESQ**：VB-DEMAND 噪声类型固定（DEMAND 10 类），幅度掩码+BCE
  的有界输出更好学；cRM 无界线性输出估计空间大、样本效率低。论文里 cRM 优势要在
  更大数据/更长训练下才显现（DCCRN 用 DNS + 动态混合）——这正是 WP-04 的看点；
- **单向 LSTM 只比 BLSTM 低 0.03 PESQ**：因果模型几乎无损 → 流式化（WP-05b/07）
  代价很小，这是车载链路最关心的结论；
- 明细：`outputs/eval/{irm_blstm,crm_lstm}/summary.json`。

开关：`--mask-type crm|irm`、`--rnn-type blstm|lstm`、`--bounded-mask`、`--no-skip`

## 已知坑（实战记录）
- **频率降采样公式**：k=5,p=2,s=2 时 F→(F−1)//2+1。曾写成 (F+4)//2 导致 LSTM
  input_size 与实际差 80——单测 `test_model_shapes_all_configs` 兜底
- **Windows DataLoader spawn**：数据集类必须定义在模块级（main() 内局部类无法
  pickle），`dataset.PairsDataset` 因此存在
- **eval 输出文件名必须与参考一致**（分目录同名配对）；曾加 enh_ 前缀导致
  evaluate_dir 交集为空
- **时间维不做降采样**（stride=(2,1)）：保留帧级对齐，为 WP-05b/07 流式铺路
- **eval 与训练解耦**：训练用 3s 随机裁剪，评估全长 batch=1
- 说话人不相交 val 切分（`speaker_split`）：p226_xxx → p226 前缀分组
