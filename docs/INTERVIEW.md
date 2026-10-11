# INTERVIEW 索引（批次1–7 覆盖模块 + 选做）

| 文件 | 模块 | 3 分钟主题 |
|---|---|---|
| [INTERVIEW-WP01-STFT.md](INTERVIEW-WP01-STFT.md) | 模块1 STFT/ISTFT/流式 | 从时域卷积噪声到时频掩码：为什么语音增强都活在 STFT 域 |
| [INTERVIEW-WP02-losses.md](INTERVIEW-WP02-losses.md) | 模块2 损失函数 | 五种损失怎么选：从 SI-SNR 到感知加权的设计逻辑 |
| [INTERVIEW-WP02E-eval.md](INTERVIEW-WP02E-eval.md) | 模块2.5 评估指标 | 指标体系与口径纪律：PESQ/STOI/SI-SNR/ERLE/AECMOS 各管什么 |
| [INTERVIEW-WP03-CRN.md](INTERVIEW-WP03-CRN.md) | 模块3 CRN | 编码-循环-解码 + 复数掩码：语音增强第一个完整 NN 闭环（实测 PESQ 2.53/IRM 2.84） |
| [INTERVIEW-WP07D-AEC-data.md](INTERVIEW-WP07D-AEC-data.md) | 模块7前置 数据合成 | AEC 数据从哪来：物理链路仿真与可验证的合成纪律 |
| [INTERVIEW-WP06-tradAEC.md](INTERVIEW-WP06-tradAEC.md) | 模块6 传统 AEC | 自适应滤波 + DTD 三态状态机：车载 AEC 的地基与 NN-AEC 的输入 |
| [INTERVIEW-WP04-DCCRN.md](INTERVIEW-WP04-DCCRN.md) | 模块4 DCCRN | 复数域卷积/LSTM + Dynamic Mixing：从 CRN 到 GTCRN 的技术演进主线 |
| [INTERVIEW-WP05a-GTCRN-ONNX.md](INTERVIEW-WP05a-GTCRN-ONNX.md) | 模块5a GTCRN/ONNX/INT8 | ERB 轻量化折中 + PTQ/QAT 定点化：端侧部署全链路（含静态量化崩坏负面结果） |
| [INTERVIEW-WP07-NNAEC.md](INTERVIEW-WP07-NNAEC.md) | 模块7 NN-AEC（简历核心） | 双输入因果 CRN + 时延鲁棒训练：全 NN 路线 vs 传统基线的同口径对决（V4 扩充：级联三方、官方真实集、时延 OOD 判别实验） |
| [INTERVIEW-WP10.5-capstone.md](INTERVIEW-WP10.5-capstone.md) | 模块10.5 Capstone | AEC→NR 级联：车载语音链的完整形态与三级量化叙事 |
| [INTERVIEW-WP11-TSE.md](INTERVIEW-WP11-TSE.md) | 模块11 TSE（选做） | 目标说话人提取三流派（Conv-TasNet/DPRNN/SepFormer）+ TSE vs 纯分离：demo 本体未落地，先备追问预案 |

> **V4 扩充弹药（2026-10-11 晚）**：[ADR 0005](adr/0005-cascade-verified.md)（级联三方对照+附记）、
> [ADR 0006](adr/0006-kalman-vs-mdf-boundary.md)（Kalman 结论边界）、
> `reports/{official_aecmos,realrir_domain,ood_real_echoes,delay_ood_probe,deg_mechanism}.md`
> （官方真实集 / 真实车厢 RIR / 漂移 OOD 判别 / deg 机理测量）——每个都挂着自己的实测数字。

## 复习顺序建议
1. 先过 STFT（一切的地基，追问密度最高）
2. 再过指标（口径错误在面试中最减分）
3. 车载岗主线：WP-06 传统 AEC → WP-07D 数据合成 → （批次6 后）WP-07 NN-AEC
4. 损失函数结合 WP-03/07 的模型选择再回来精读

## 通用纪律
- 任何指标必带协议：采样率 / wb-nb / 测试集 / 分场景
- 每个结论配一个自己跑出来的数字（lab 脚本的输出就是弹药）
- 手写白板自查：STFT 定义式、SI-SNR 三步、IRM 公式（批次1 的三个必默写项）
