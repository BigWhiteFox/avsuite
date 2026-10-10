# core — 共享库（WP-01 / WP-02 / WP-02E）

被所有 `exp/*` 复用的三件套：STFT/ISTFT 工具、损失函数库、mini 评估。
单测：`pytest tests -q`（35 项，全绿，float64 对拍误差 <1e-10）。

## WP-01 STFT/ISTFT（`stft.py` / `istft.py`）

### 约定
- 谱布局 `[B, F, T]` 复数，`F = n_fft//2+1`；`spec_to_2ch` ↔ Re/Im 双通道 `[B, F, T, 2]`（网络输入布局）
- 默认 16k：`n_fft=512`（32ms）、`hop=256`（16ms，50% 重叠）
- `center=True` 两侧 pad `n_fft//2`，默认 reflect（同 torch.stft）；流式约定 zeros
- 与 `torch.stft` / `torch.istft` 逐项对拍：5 组窗长×窗型×center 配置，float64 全量误差 <1e-10

### COLA 与 WOLA —— 重构精确到底靠谁？
- **COLA**（重叠相加后窗和恒定）：`Σ_t w[n−t·hop] = 常数`。它是"**不加任何归一化**、直接重叠相加就能精确重构"的条件。hann/hamming 在 50% 重叠满足（实测偏差 ~1e-7）；blackman 在 50% **不满足**（实测偏差 0.38，要 2/3 重叠才行）。
- **WOLA**（除以窗平方和）：ISTFT 用 `y = Σ w·frame / Σ w²`。分析帧 = w·x，加窗合成 = (Σw²)·x，除回去即精确——**任意窗、任意 hop≤n_fft 都精确，不依赖 COLA**。实测 blackman@50% 往返误差同样 ~1e-15。
- 面试表述：*"COLA 是免归一化重构的充分条件；工程上更稳的是 WOLA 归一化，它把'窗是否 COLA'从重构正确性问题里解耦了。torch.istft 就是 WOLA。"*

### 相位卷绕 —— 为什么相位不能直接 MSE 回归？
1. **±π 不连续**：`angle()` 返回 (−π, π]，真值跨过 ±π 边界时（相位卷绕/wrapping），预测值 3.14 与目标 −3.14 实际几乎相等，MSE 却给出巨大误差，梯度被边界跳变主导。
2. **解不唯一**：相位差 2π 等价，回归目标本身是病态的；相位展开（unwrapping）在噪声下不可靠。
3. **感知不敏感但重构强约束**：人耳对绝对相位迟钝（所以带噪相位+幅度增强也能听），但 ISTFT 需要一致的相位才能重构——"不学相位"≠"相位不重要"。
4. **工程替代**（我们库里已实现前两种）：复数谱 Re/Im 直接 MSE（相位隐含在 Re/Im 中）；幅度掩码 + 带噪相位；更进一步有相位感知损失（PMSGE 等，后续批次再补）。

### 流式接口（`StreamingSTFT`）
- 任意 chunk 推入（含 1 样本/含跨帧大 chunk），按 hop 产出已完成帧；与离线 `pad_mode='zeros'` 逐帧一致（对拍含 chunk=7 的压测）
- **cache 有界**：缓冲 ≤ n_fft−1+chunk 个样本，与信号总长无关（`cache_size()` 实测）
- **时延账本**（n_fft=512/hop=256@16k）：帧长 32ms；超前读取 lookahead = n_fft−hop = 16ms（帧右端对应 16ms 前的音频）；左 pad 注入 n_fft//2 零。流式与离线 reflect 的差异只在最左帧边缘样本
- `flush()` 右补 n_fft//2 零结算尾帧（等价离线右 pad）

### 窗长消融（`exp/01_stft_lab/window_ablation.py` 实测）
| 窗长 | n_fft | -6dB 主瓣宽 | 频率分辨率 | 代价 |
|---|---|---|---|---|
| 20ms | 320 | 50 Hz | 分不开 100Hz 间隔的低次谐波 | 时间准 |
| 32ms | 512 | 31 Hz | 男声基频(≈100-150Hz)勉强可分 | 折中（默认） |
| 64ms | 1024 | 18 Hz | 谐波结构清晰 | 瞬态拖尾 64ms |

## WP-02 损失函数库（`losses.py`）

| 损失 | 输入 | 一句话 | 适用 |
|---|---|---|---|
| `si_snr_loss` | 时域 [B,N] | 零均值→投影消尺度→能量比 | 分离/增强端到端首选，尺度不变 |
| `complex_mse_loss` | 复谱 [B,F,T] | Re/Im 两路 MSE，相位隐含 | 复数谱模型（DCCRN 系） |
| `irm_bce_loss` | 掩码(0,1)+混/净谱 | 对 IRM 目标的 BCE | 掩码类模型（CRN 系） |
| `weighted_distortion_loss` | 复谱 + γ | 干净幅度^γ 加权谱误差，γ=0 退化为 complex MSE | AEC：响的地方错一点罚更重 |
| `multi_scale_stft_loss` | 时域 | 多窗长"谱收敛+log 幅度"求和 | 波形生成/ vocoder 式保真 |

- SI-SNR 数值细节：两侧都 clamp eps → 全零输入约定 0 dB，训练不炸
- IRM 用 sqrt(功率比)（Wiener 幅度增益）；能修正相位的是复数版 cIRM
- 加权失真的 γ 扫描消融延至 WP-07（业务向调优）

## WP-02E mini 评估（`evaluate.py`）

`evaluate_dir(est_dir, ref_dir, out_csv)` → 逐文件 PESQ(wb)/STOI/SI-SNR DataFrame。

**协议（报告任何指标都必须带口径）**：16k 单声道（内部 resample_poly 重采样）；长度截断到较短者（不补零）；PESQ wideband；按文件名配对取交集；静音文件该项 NaN 不拖垮整批。

已知教学现象（`exp/01_stft_lab/demo_zh.py` 可复现）：经典谱减在 5dB 噪声下 **SI-SNR 4.95→14.02 dB 但 STOI 0.573→0.483**——SNR 类指标大涨、可懂度反降（音乐噪声/幅度失真），这正是"为什么 2000 年代后要上 NN 掩码"的活教材。
