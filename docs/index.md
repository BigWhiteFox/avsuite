# avsuite 文档站

> 车载语音前端全链路：**语音增强 / AEC / 端侧部署**，传统信号处理 + 神经网络 + INT8 量化，
> 全程可复现（261 项单测 = 257 passed + 4 xfailed）。

## 从这里开始

| 你是谁 | 点这里 |
|---|---|
| 🎯 面试官（30 秒） | [30 秒看懂](README_HOME.md) → [项目一页纸](STORY.md) |
| 📖 想学习这套系统 | [10 天学习路径](LEARNING.md) → 3 个 Notebook 交互教程（仓库 notebooks/ 目录） |
| 🔢 想查某个数字 | [指标权威口径 NUMBERS](NUMBERS.md) → [数字速查](CHEATSHEET.md) |
| 🧠 想知道"为什么这么设计" | [设计决策 ADR](adr/README.md) |
| ✅ 想看验收证据 | [验收报告 ACCEPTANCE](ACCEPTANCE.md) |

## 三条主线

1. **回声消除 AEC**：传统 MDF/DTD（纯 numpy）+ 全 NN 0.638M 对照实验，AECMOS 官方口径；
2. **语音增强 NR**：CRN → DCCRN（复数）→ GTCRN（23.67K 端侧）技术演进；
3. **端侧部署**：ONNX 导出对拍 6e-6、INT8 PTQ（含静态崩坏负面结果）、C++ 推理、20ms 流式。

![增强前后语谱](assets/hero_enhancement.png)

## 设计哲学

- **测出来的，不是声称的**：因果性、流式对齐、对拍精度全部由单测锁死；
- **负面结果留档**：静态 INT8 崩坏、DCCRN 实数卷积与复数性能相当（噪声级差异，不称反超），都写成 [ADR](adr/README.md)；
- **口径纪律**：所有指标带 采样率/测试集/分场景/checkpoint，唯一权威源 [NUMBERS](NUMBERS.md)。
