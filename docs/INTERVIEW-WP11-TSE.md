# INTERVIEW — WP-11 TSE 目标说话人提取（选做）

> 状态：演示本体未跑通（需要外源预训练模型 + 自定义接线，评估后决定）；
> 本文档先交付 WORKLIST 要求的「三流派一句话」+ 追问预案。

## 三流派一句话（必背）

| 流派 | 一句话 |
|---|---|
| **Conv-TasNet**（2019） | 时域端到端：1D 卷积做"可学习的 STFT"，TCN 分离，掩码乘编码特征——不用显式相位重建 |
| **DPRNN**（2020） | 双路径 RNN：把时频/时域特征 reshape 成块，块内(intra)与块间(inter)两个 RNN 交替建模——比 TCN 更省、更长时依赖 |
| **SepFormer**（2021） | 双路径结构不变，RNN 换自注意力（Transformer）——训练快、并行好，块级位置编码保时序 |

## TSE 与纯分离的关键差异（面试框架）

- 纯分离（blind source separation）：不知道目标是哪个说话人 → 输出通道排列歧义 →
  需要 PIT（permutation invariant training）
- **TSE = 分离 + 说话人先验**：给一段目标说话人的注册语音（enrollment），提取
  speaker embedding（ECAPA-TDNN 等），作为条件喂入分离网络——把"无序分离"变成
  "定向提取"，排列歧义消失（PIT 不再必需，可直接监督）
- VoiceFilter 思路（Wang et al. 2019）：embedding 拼到每个特征帧 + 频谱掩码
  （LSTM 主网络）——工程实现最省，所以我们按它组织

## 高频追问预案

**Q1 PIT 是什么、TSE 为什么不需要？**
分离网络输出多个通道，但"输出1 是谁"没有定义——训练时对所有排列算损失取最小
（PIT），推理时还有全局重排问题。TSE 的目标被 embedding 唯一指定，输出就是目标
说话人，损失直接对位——PIT 消失，训练目标干净得多。

**Q2 speaker embedding 怎么和分离网络结合？**
三档：①全局拼接（embedding 拼到每帧特征，VoiceFilter）；②FiLM 调制（embedding
经 MLP 生成 scale/shift 调制中间特征）；③注意力查询（embedding 作 query 引导
SepFormer 的自注意力）。复杂度递增，条件注入能力也递增。

**Q3 时域方法（Conv-TasNet 系）为什么不用显式相位？**
STFT 重建需要精确相位（相位估计是分离的老大难）；时域方法把"频谱分析"换成
可学习卷积（编码器），分离在编码域用掩码做，解码器卷积直接合成波形——相位
问题被隐式吸收进编解码器。代价：损失必须用时域（SI-SNR），谱域可解释性差。

**Q4 SI-SNRi 是什么口径？**
SI-SNR 改进 = SI-SNR(估计, 目标) − SI-SNR(混合, 目标)。报告时必须注明测试集
（WSJ0-2mix/LibriMix 的 dB 数不可比）——与我们的评估纪律一致。

**Q5 车载场景 TSE 的现实需求？**
多乘员场景："主驾指令 vs 副驾闲聊"分离、后排娱乐隔离。真实约束：enrollment
从哪来（车机声纹注册）、混响/近场多通道、时延预算。面试时把它接回车载主线。
