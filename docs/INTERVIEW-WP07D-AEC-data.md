# INTERVIEW — WP-07D AEC 数据合成管道（模块7 前置）

## 3 分钟讲解提纲

**第 1 分钟：AEC 问题的数据形态**
- 输入两路：麦克风 mic = near + echo + noise；远端参考 far（回声的电信号源）
- echo = far 经过「DAC→功放→扬声器→房间(RIR)→麦克风」的畸变：线性部分是 RIR 卷积，
  非线性部分来自功放/扬声器饱和（soft-clip 近似）
- 目标：从 (mic, far) 恢复 near——监督数据因此是合成出来的（真实房间无法分离 near/echo）

**第 2 分钟：合成管道设计（scripts/aec_synth.py）**
- echo = RIR ⊛ far，可选 soft_clip(c·tanh(x/c)) 功放失真（WP-08 开关，默认 30% 样本启用）
- 时延扰动：echo 随机右移 10~80ms——模拟车载场景时延不确定（蓝牙/缓冲抖动），逼模型学时延鲁棒
- 定标协议：DT 场景 near/noise 定 SNR、echo/near 定 SER；FEST 只 echo vs noise；NST 只 near vs noise
- 三场景配比 FEST:0.5 / NST:0.25 / DT:0.25（可配）——与 AEC-Challenge 对齐；manifest.csv 全参数可追溯
- RIR 三来源：自备目录 > pyroomacoustics 随机房间仿真 > 指数衰减兜底

**第 3 分钟：验证方法论（面试最加分的部分）**
- 每个物理量有对应单测：dirac-RIR 时延互相关定位、SER/SNR 实测 vs 配置 ±0.1dB、
  soft-clip 线性区不变/饱和区压缩、同 seed 逐字节可复现
- 真实数据抽检：12 条样本三场景成分断言 + 无削波（峰值≤0.9）

## 高频追问预案

**Q1 为什么必须合成？真实 AEC 数据不行吗？**
需要 (mic, far, near-clean) 三元组做监督，但真实录音的 near 与 echo 在 mic 里已经相加、物理不可分。行业做法都是「真实 near + 真实 far + 仿真/实测 RIR」合成（AEC-Challenge 官方亦然）；微软同时用真实全链路录音做无参考测试集（我们的 test_set_icassp2022）。

**Q2 时延扰动为什么重要？不加会怎样？**
自适应滤波器（NLMS/MDF）和 NN-AEC 都隐式假设「far 的历史帧能对上 mic 里的回声」。真实链路时延会漂（蓝牙 100-300ms、缓冲抖动）。训练时见过 ±N 帧扰动，模型的等价感受野才覆盖漂移。我们造 10-80ms 随机时延 + WP-07 训练里还会加帧级扰动。

**Q3 soft-clip 为什么用 tanh？**
小信号 tanh 近似线性（功放正常区），大信号软压缩（饱和区）——是「无记忆非线性」的标准近似。真实扬声器还有记忆非线性（与历史信号相关，如悬浮非线性），Hammerstein/Wiener 模型处理；WP-08 的 GRU 功放建模级联就是往这个方向走的。

**Q4 SER 和 SNR 的定义？典型值？**
SER = 10log(P_near/P_echo)：双讲时近端被回声压多狠。AEC-Challenge 合成集 SER 分布约 -6~+6dB（回声常常比近端还响，因为扬声器增益大）。SNR 这里指近端 vs 背景噪声，取 5-20dB。两个比值独立定标、削波保护同缩不改比值——这些口径都写进了单测。

**Q5 pyroomacoustics 仿真的 RIR 可信吗？**
ISM（image source method）鞋盒模型：几何反射精确、衍射/扩散吸收近似。对 AEC 训练够用（模型学的是「卷积+非线性」映射，不挑房间真实性）；关键统计量（RT60、直接路径时延、衰减形状）可控可复现。有实测 RIR 时优先用（--rir-dir），仿真做覆盖补充。

**Q6 三场景为什么 FEST 占一半？**
FEST（远端单讲）是 ERLE 的统计主战场、也是用户最直观感受"回声消没消"的场景；DT 最难（要同时消回声+保近端）但样本可以少些；NST 保底近端保真。微软官方合成也是远端单讲占大头。

**Q7 manifest 里记什么？为什么全参数可追溯？**
每条的 scenario / delay_ms / ser_db / snr_db / near_file / far_file / noise_file / rir_src。消融和 debug 需要：比如"时延 >60ms 的子集上 ERLE 掉多少"——没有 manifest 就没法切片分析。同 seed 逐字节可复现（单测验证）保证实验可重跑。

**Q8 和 AEC-Challenge 官方合成集什么关系？**
官方集（本机已克隆，synthetic/ 10000 条）配方公开（论文 2009.04972）：真实 RIR + 真实语音/噪声 + 时延 + 非线性。我们的管道是它的"可控迷你版"：相同物理链路、参数全部暴露可调、可注入任意新条件（新时延分布、新失真强度）——官方集用于对标评测，自合成集用于消融与增广。

## 必默写公式
- 链路：`mic = near + (soft_clip(far ∗ RIR) >> delay)·g + noise`
- soft-clip：`y = c·tanh(x/c)`，线性区 y≈x − x³/(3c²)
- SER：`10·log10(P_near / P_echo)`（正数 = 近端更强）
