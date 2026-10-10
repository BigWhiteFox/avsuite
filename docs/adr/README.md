# 架构决策记录（ADR）

每篇 ADR 记录一个"为什么这么设计"的决策：背景 → 踩过的坑/实测数据 → 决策 → 面试标准答案。
这些文档同时是**面试弹药**——面试官问"你为什么这么做"时，答案就在这里。

| # | 决策 | 一句话 |
|---|---|---|
| [0001](0001-dtd-mic-vs-yecho-coherence.md) | DTD 用 mic-vs-y_echo 相干 | 时域相关被 RIR 段内泄漏打散，WebRTC 同款后置相干 |
| [0002](0002-fullnn-vs-cascade-aec.md) | NN-AEC 选全 NN 而非级联 | 为干净对照实验；实测 echo 赢 / deg 输，生产回级联 |
| [0003](0003-gtcrn-static-int8-collapse.md) | GTCRN 静态 INT8 为何崩 | GRU 状态误差累积 + 小模型无冗余 + LN/Tanh 敏感 |
| [0004](0004-dccrn-real-vs-complex.md) | DCCRN 复数 vs 实数反直觉 | 复数赢在参数效率非绝对性能，DM 才是当前瓶颈 |

## 为什么写 ADR

开源/大厂资深工程师用 ADR 记录"当时的约束与取舍"，避免后人（和未来的自己）踩同样的坑。
对求职者，ADR 证明的不只是"我做出来了"，而是"**我理解每个决策背后的 tradeoff**"——
这正是初级与资深的分界。
