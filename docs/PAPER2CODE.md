# PAPER2CODE — 论文 ↔ 代码对照表 + SOTA 谱系（P3-3）

> 用途：面试「你了解哪些相关工作？你的代码对应论文哪里？」的标准答案页。
> 行号锚定：git HEAD `539cb884e04942651eaf244c0b1bdd3073cb0085`（2026-10-10 22:03 +08:00）。
> 全部行号于 2026-10-11 以 grep(-n) + read 实地核实；核实方式与交付前抽查记录见
> `reports/rectification/exec/P3-3.md`。
> 数字口径：本仓实测数字一律抄自 `docs/NUMBERS.md`（全仓唯一权威源）并随文标行号；
> 论文口径的数字单独写「论文」。本文件为 P3-3 新建文档，不改动任何既有文件。

## A. 论文 ↔ 代码对照表（13 行）

| # | 论文 / 公式·模块 | 本仓 文件:行号 | 一句话说明 |
|---|---|---|---|
| A1 | CRN（Tan & Wang, Interspeech 2018）— 卷积编码器（频率降采样、时间维不动） | `exp/03_crn/model.py:30-34`（ConvBlock，k(5,5) s(2,1) p(2,2)）、`:69`（enc 三层） | 频率 257→129→65→33、时间维不变——流式友好的关键设计（`exp/03_crn/model.py:5-6` 注释） |
| A2 | CRN — RNN 瓶颈层 | `exp/03_crn/model.py:70-73`（nn.LSTM + fc） | 频率×通道 flatten(40×33=1320) → 2 层 BLSTM(hidden=256) → Linear 还原（`exp/03_crn/model.py:7`） |
| A3 | CRN — cRM 复数比率掩码 + 增强 | `exp/03_crn/model.py:77-78`（1×1 conv 出 Re/Im 双通道）、`:84-93`（_mask_and_enhance：enh = mask⊙noisy） | cRM 默认线性无界，`--bounded` 走 1.55·tanh；IRM 消融开关在同一函数 `:86-89`（保留带噪相位） |
| A4 | DCCRN（Hu et al., Interspeech 2020）式(1) 复数卷积 | `exp/04_dccrn/complex_nn.py:31-45`（_ComplexLinear2d.forward：`yr = conv_re(xr) − conv_im(xi)`，`yi = conv_re(xi) + conv_im(xr)`）、`:48-52`（ComplexConv2d） | (W_re+j·W_im)(x_re+j·x_im) 用两路实权重、四次实乘实现（Trabelsi et al., ICLR 2018 定义，`complex_nn.py:7-11`）；torch 不吃 complex dtype，实数视图同时对 ONNX/量化友好（`complex_nn.py:3-5`） |
| A5 | DCCRN — 复数 LSTM | `exp/04_dccrn/complex_nn.py:62-115`（门控逐分量 σ/tanh 在 `:103-108`）、`exp/04_dccrn/model.py:70`（接线） | 门激活按分量作用（σ(z)=σ(re)+j·σ(im)），Trabelsi 复数 RNN 的标准简化（`complex_nn.py:65`）；逐时间步 python 循环（编码后 T≈21 步，开销可接受，`complex_nn.py:63`） |
| A6 | DCCRN — cRM 应用 enh = mask⊙noisy | `exp/04_dccrn/model.py:83-84`（STFT→[B,2,F,T] 实数视图）、`:105`（complex_mul）、`exp/04_dccrn/complex_nn.py:25-28` | 复数掩码逐点乘复数谱，显式修正相位 |
| A7 | DCCRN — 整体结构（Enc×4 + ComplexLSTM + Dec×4） | `exp/04_dccrn/model.py:41`（class DCCRN）、`:50-51`（ComplexConv2d/ComplexConvTranspose2d 选择）、`:5`（通道 1→32→64→96→128） | 编码-解码对称 + 跳连 → cRM；`--real-conv` 消融把复数卷积换实数版（见 B1 与 `docs/adr/0004-dccrn-real-vs-complex.md`） |
| A8 | GTCRN（Rong et al., ICASSP 2024）— ERB 频带压缩 | `exp/05a_gtcrn_onnx/gtcrn.py:15-53`（class ERB：hz2erb `:26-28`、erb_filter_banks `:34-53`）、`:55-59`（bm 压缩）、`:61-65`（bs 展开）、`:285`（ERB(65,64) 实例化）、`:308`（bm）、`:318`（bs） | 257 bin 压到 65 个 ERB 子带（高频段线性折叠，固定权重 requires_grad=False `:23-24`），解码后 ierb_fc 展开回去；官方 repo 移植（`gtcrn.py:2`，Xiaobin-Rong/gtcrn，MIT） |
| A9 | GTCRN — 主干（SFE + 2D UNet + DPGRNN + cRM）与 ONNX 导出 | `gtcrn.py:282-296`（GTCRN.__init__）、`:297-323`（forward）、DPGRNN `:191`、cRM Mask `:270-279`；导出 `exp/05a_gtcrn_onnx/onnx_export.py:21-28`（opset 17，动态 time 轴）、对拍 `onnx_export.py:50`（parity） | ShuffleNetV2 式分组卷积 + 2×DPGRNN；23.67K 可训练参数 / 33.0 MMACs（`gtcrn.py:6-7`；`tests/test_gtcrn_onnx.py:21`） |
| A10 | SI-SNR 损失（尺度不变信噪比） | `core/losses.py:17-29`（si_snr）、`:32-34`（si_snr_loss）；DCCRN 训练用法 `exp/04_dccrn/train.py:31,41` | 零均值 → 投影消尺度 → 投影/残差能量比；两侧 clamp eps，全零输入约定 0 dB、训练不 NaN（`core/README.md:48`） |
| A11 | γ 加权谱失真（power-law 加权） | `core/losses.py:62-75`（weighted_distortion_loss；`w = (ref_spec.abs()+eps)^γ` 在 `:72-73`） | γ=0.5：响度感知近似幅度^0.6（Stevens 幂律，`losses.py:68`）；γ=0 退化为复数谱 MSE；AEC 文献（Ivry 系列）常用（`docs/INTERVIEW-WP02-losses.md:36`） |
| A12 | Ivry et al. 2021（Interspeech，「Nonlinear Acoustic Echo Cancellation with Deep Learning」，arXiv:2106.13754）— 级联：线性级 + NN | `exp/08_nl_aec/train.py:1-7`（级联结构 ŷ_lin=MDF(far)，ŷ_res=GRU([lin,\|lin\|])，enh=mic−lin−res）、`:105`（实例化 NonlinearAEC）；`exp/08_nl_aec/model.py:1-11`（动机）、`:19-29`（GRU 残差级）、`:32-33`（enhance）；`exp/08_nl_aec/precompute_lin.py:1-6`（线性级动机）、`:32`（TradAEC(method="mdf", partitions=8, mu=0.7)）、`:40`（存 y_echo） | 线性级（MDF）先消化 10-80ms 时延+100ms RIR≈180ms 长记忆，GRU 只学记忆less 非线性残差；v1 裸 GRU ERLE 0dB 的负结果即级联存在的理由（`model.py:8-10`） |
| A13 | 流式 STFT/ISTFT：周期窗 + WOLA 窗平方归一约定 | `core/stft.py:21-33`（make_window，periodic=True 在 `:28`）、`:7`（COLA/WOLA 约定）、`:92-153`（StreamingSTFT：push `:142`、flush `:155-163`）；`core/istft.py:77-119`（StreamingISTFT：add_frame `:105`，窗平方归一 `:111-113`）、`:97`（np 周期窗 sin²(πn/N)）；对拍 `tests/test_stft.py:65`（test_streaming_matches_offline） | 周期窗（periodic=True）与 torch.stft 默认一致；ISTFT 除以窗平方和、不依赖 COLA（`core/istft.py:7-9`）；流式与离线逐帧一致（DoD <1e-5，实测 ~1e-7，`docs/NUMBERS.md:80`） |

## B. SOTA 谱系（一页标准答案）

### B1 语音增强线：CRN(2018) → DCCRN(2020) → DPCRN(2021) → GTCRN(2024)

- **CRN**（Tan & Wang, Interspeech 2018）：卷积编码 + BLSTM + 对称解码 + cRM 的端到端频域增强开山之作。
  本仓 `exp/03_crn/`（模型 `exp/03_crn/model.py:37`）——【复现过 + 做过消融】：mask_type(cRM/IRM)、rnn_type(双向/单向)消融开关在 `exp/03_crn/model.py:12-14`，消融脚本 `exp/03_crn/ablation.py`，结果 `exp/03_crn/outputs/ablation/ablation_short.md`；实测 cRM+BLSTM PESQ 2.5288 / IRM 2.8428 / cRM+单向 2.4999（VoiceBank-DEMAND 824 对，`docs/NUMBERS.md:17-22`）；论文口径 ≈2.74（`docs/INTERVIEW-WP02E-eval.md:22`），本仓小模型 2.53 量级合理（`exp/03_crn/README.md:59`）。
- **DCCRN**（Hu et al., Interspeech 2020）：复数卷积/复数 LSTM 显式建模相位，DNS Challenge 2020 冠军方案；论文 PESQ 2.68（WB，DNS 合成测试，`docs/INTERVIEW-WP04-DCCRN.md:10`）。
  本仓 `exp/04_dccrn/`（模型 `exp/04_dccrn/model.py:41`）——【复现过 + 做过消融】：`outputs_abl/dccrn_nodm`（去动态混合）与 `outputs_abl/dccrn_realconv`（实数卷积）两组消融 checkpoint 实存；实测主配置 2.1811 / nodm 2.0151 / realconv 2.1898（DNS devtest 150 条，`docs/NUMBERS.md:27-29`）；「实数卷积与复数性能相当（Δ0.009 噪声级，不称反超）且参数 2×」的反直觉发现归因见 `docs/adr/0004-dccrn-real-vs-complex.md`（`docs/NUMBERS.md:32-33`）。
  - 注（如实）：会议口径仓内不一致——`exp/04_dccrn/model.py:1` 写「ICASSP 2020」，`docs/adr/0004-dccrn-real-vs-complex.md:9` 写「Interspeech 2020」；论文实际发表于 Interspeech 2020，本表按 ADR 0004 口径（红线不改既有文件，仅在此注明）。
- **DPCRN**（Le et al., Interspeech 2021，arXiv:2107.05429）：全带 2D 卷积 + 逐子带双路径 RNN，用极小算力逼近大模型——CRN 到 GTCRN 之间「降算力」路线的中间节点。
  本仓：无模块（全仓 grep 'DPCRN' 0 命中，2026-10-11 核实）——【读过】，如实标注，不做复现/消融宣称。
- **GTCRN**（Rong et al., ICASSP 2024，「GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources」）：ERB 频带压缩 + 分组卷积 + DPGRNN 的超轻量模型（本仓权重版 23.67K 参数 / 33.0 MMACs，`exp/05a_gtcrn_onnx/gtcrn.py:6-7`）；论文 VCTK-DEMAND PESQ 2.87 / DNS3 盲测 DNSMOS OVRL 2.70（`docs/INTERVIEW-WP05a-GTCRN-ONNX.md:8`）。
  本仓 `exp/05a_gtcrn_onnx/`（官方 repo 移植 `gtcrn.py:1-4`，官方 DNS3 权重）——【复现过（ONNX/INT8 部署线）+ 对拍/量化消融】：PyTorch vs ONNX 对拍 6.32e-06（`docs/NUMBERS.md:38`）；INT8 dynamic PTQ PESQ 1.803（−0.067，240KB）/ INT8 static QDQ 1.120 崩坏（`docs/NUMBERS.md:41-42`）；static 崩因（GRU 状态误差累积 + 小模型无冗余 + LN/Tanh 敏感）见 `docs/adr/0003-gtcrn-static-int8-collapse.md`（`docs/adr/README.md:10`）；自仓 fp32 基线 1.869 为「自混 held-out，更难」口径，勿与论文直接比（`docs/NUMBERS.md:43`）。

### B2 AEC 线：WebRTC AEC → AEC-Challenge 2021/2022 冠军（级联形态）→ 深度自适应滤波方向

- **WebRTC AEC（传统 DSP 栈：自适应滤波 + DTD + 抑制器）**：NLMS/分区频域（MDF）自适应滤波消线性回声，双讲检测保近端，后置抑制器兜底。
  本仓锚点 `exp/06_trad_aec/`——【读过】：NLMS `exp/06_trad_aec/nlms.py:11`、MDF `exp/06_trad_aec/mdf.py:19`、DTD `exp/06_trad_aec/dtd.py:18`（mic-vs-y_echo 后置相干，WebRTC 同款思路 `exp/06_trad_aec/dtd.py:65`、`docs/adr/0001-dtd-mic-vs-yecho-coherence.md:30`）、单 pass 在线状态机 `exp/06_trad_aec/pipeline.py:7`（TradAEC `:53`）、残余抑制 `exp/06_trad_aec/res.py:14`；实测 FEST ERLE 20.55 dB（eval_aec.py 口径；同一 eval_v3 上 eval.py 口径 20.37，脚本间 0.18dB 口径差，见 `docs/NUMBERS.md` WP-06 口径注）、DT 近端 PESQ 1.687（eval_v3 60 条，`docs/NUMBERS.md:50-51`）。
  - 如实注：仓内引用均为「WebRTC AEC suppressor 同款思路」，全文未出现「AEC3」字样（全仓 grep 'AEC3|aec3' 0 命中，2026-10-11 核实）。
- **AEC-Challenge 2021/2022 冠军（级联形态：线性 AEC 保底 + NN 消残余/非线性）**：微软主导的赛事把「级联」定为工业主流——收敛稳、可解释、近端保真有底（`docs/adr/0002-fullnn-vs-cascade-aec.md:11`）。
  本仓锚点：【复现过(级联形态已实测:`exp/09_cascade_aec/`,ADR 0005)】全 NN 对照主线 `exp/07_nn_aec/`（0.638M 因果双输入 CRN，对标 AEC-Challenge 2022+ 规格，`exp/07_nn_aec/model.py:3`；官方 AECMOS 评估 `exp/07_nn_aec/aecmos_eval.py:28-29`）；级联实现 `exp/08_nl_aec/`（Ivry 2021 结构，v2 级联 ERLE 5.78→6.06/+0.28，`docs/NUMBERS.md:74`）与 `exp/09_cascade_aec/precompute_cascade.py:1-11`（「级联=换输入对」：y_echo 当 far 参考、e_lin 当 mic 输入，与 WP-06/WP-08 完全同款线性级）；ADR 0002 的实测对照（echo 赢 / deg 输）与 fallback 结论在 `docs/adr/0002-fullnn-vs-cascade-aec.md:26-41`（**正文数字为 16k 直送旧口径，见该 ADR 文末口径警示附记**）。全 NN v4 实测（**AECMOS@48k 模型协议**）：DT echo_mos **4.175** vs 传统 **3.494**（Δ+0.681±0.212，echo 赢），deg_mos **3.237** vs 3.598（近端输，Δ−0.361±0.169，**落后仅存在于 DT 段**；NST 段 3.941 vs 3.842 不再落后），机理=高频带选择性损伤+谱平坦化（`docs/NUMBERS.md` §WP-07 与顶部决策条）。
- **深度自适应滤波方向（NN 估自适应滤波器状态/Kalman 协方差，把收敛动态学进网络）**：把步长/协方差/正则交给 NN 估、自适应滤波器负责线性主径；Ivry 2021 的「线性级 + NN 联合」是该方向的早期形态（`docs/INTERVIEW-WP06-tradAEC.md:88`「线性滤波联合训练（Ivry 2021）」）。
  本仓：【读过/方向，未实现】——全仓 grep 'Kalman' 0 命中（2026-10-11 核实），如实不宣称；最近的本仓锚点是级联 v2 的「线性级消记忆、NN 消非线性」分工（`exp/08_nl_aec/model.py:4-6`）与 ADR 0002 的生产级联建议（`docs/adr/0002-fullnn-vs-cascade-aec.md:37-41`）。
  - 任务书偏差注：任务给的本仓锚点含「ADR 0005」；仓内 `docs/adr/` 仅有 0001–0004（`docs/adr/README.md:8-11`），ADR 0005 不存在——本表如实改用 ADR 0002（辅以 0001/0003）并在此注明，未虚构锚点。

### B3 实测数字速查（本仓实测均抄自 docs/NUMBERS.md）

| 节点 | 论文口径 | 本仓实测 | 出处 |
|---|---|---|---|
| CRN | ≈2.74（更大模型） | 2.5288（cRM+BLSTM，VoiceBank-DEMAND 824 对；IRM 2.8428） | `docs/NUMBERS.md:19-20`、`docs/INTERVIEW-WP02E-eval.md:22` |
| DCCRN | 2.68（WB，DNS 合成） | 2.1811（DNS devtest 150 条；nodm 2.0151 / realconv 2.1898） | `docs/NUMBERS.md:27-29`、`docs/INTERVIEW-WP04-DCCRN.md:10` |
| GTCRN | PESQ 2.87 / DNSMOS OVRL 2.70 | fp32 1.869（自混 held-out，更难）；INT8 dynamic 1.803（−0.067）；对拍 6.32e-06 | `docs/NUMBERS.md:38,41,43`、`docs/INTERVIEW-WP05a-GTCRN-ONNX.md:8` |
| 传统 AEC | — | FEST ERLE 20.55 dB（eval_aec 口径；eval.py 口径 20.37，同一 eval_v3）；DT 近端 PESQ 1.687 | `docs/NUMBERS.md:50-51` |
| 全 NN AEC v4 | —（对照实验） | DT echo_mos **4.175**（vs 传统 **3.494**，Δ+0.681±0.212）；deg **3.237** vs 3.598（**仅 DT 段**落后；NST 不再落后）——AECMOS@48k 口径 | `docs/NUMBERS.md` §WP-07 |
| 级联非线性 AEC v2 | — | ERLE 5.78→6.06（+0.28，1s 段）；v1 裸 GRU 0 dB | `docs/NUMBERS.md:73-74` |
| 流式部署 | — | 流式 vs 非流式 <1e-5（实测 ~1e-7）；C++ vs Python 3.4e-5 | `docs/NUMBERS.md:80-81` |

---

*口径备注：GTCRN 参数按本仓文件口径 23.67K 可训练（`gtcrn.py:7`、`tests/test_gtcrn_onnx.py:21`；ERB 滤波器组 24.6K 为 requires_grad=False）；官方 GitHub README 新版标 48.2K 为后续版本数字，不混用。*

## 时新性补注（2026-10-11）

B2 节两处陈述为**落笔时点事实**（2026-10-11 上午核实），此后仓库已新增对应锚点，原文不改以保留审计痕迹：

1. B2「深度自适应滤波方向」处 L50「全仓 grep 'Kalman' 0 命中（2026-10-11 核实）」——此后新增 `exp/06_trad_aec/kalman.py`（FDKalman 实现，P2-2 交付），该陈述现已过期。
2. B2「任务书偏差注」处「仓内 `docs/adr/` 仅有 0001–0004，ADR 0005 不存在」——此后新增 `docs/adr/0005-cascade-verified.md`（P1-1 交付），ADR 0005 现已存在。

以本补注为准；上述原文按整改决策保留原样，作为时点快照。

## 附记（Wave V5, 2026-10-11）

> 本节为文档一致性审计补充；上文正文与「时新性补注」均保留不删。

1. **`docs/NUMBERS.md` 行号锚点已失效**：本文多处按 NUMBERS **旧版**行号引用（如 `:17-22`
   CRN、`:27-29` DCCRN、`:32-33`、`:38` 对拍、`:41-42` INT8、`:43` fp32、`:50-51` WP-06、
   `:57-64` WP-07、`:62-63`、`:73-74` WP-08、`:80-81` 部署）。NUMBERS 经 V4 整改波次扩写后
   这些行号已指向无关内容。**现行行号（2026-10-11 17:30 复核）**：CRN 表 `:27-33`、DCCRN 表
   `:35-41`（real-conv 注 `:43-45`）、GTCRN 表 `:47-55`（对拍 `:50`、fp32 `:53`、
   INT8 `:54-55`）、WP-06 表 `:61-68`（口径注 `:71-78`）、WP-07 表 `:80-88`（其后新增
   **v4 checkpoint 口径** `:96-101`，钉死 best.ckpt/last.ckpt 各自适用面）、
   eval_v5 `:114-127`、WP-08 `:224-229`、部署 `:231-236`。
   ⚠ **行号随 NUMBERS 增补持续漂移**（本波次内 eval_v5 已由 `:107-120` 漂到 `:114-127`，
   WP-08/部署同步 +7）——**引用请以章节名检索为准**，本映射仅作时点快照。
2. **GTCRN 数字口径更新**：B3 表「fp32 1.869 / INT8 dynamic 1.803（−0.067）」为**旧 50 条
   口径**；NUMBERS 现行主口径 = held-out **200 条**：fp32 **1.897**、dynamic **1.845**
   （**Δ0.053**，95%CI [0.028, 0.077]），static 1.120 仍为旧 50 条口径
   （`reports/gtcrn_heldout200.md`）。
3. **仓内测试计数已从 93 → 124 → 259 → 261**（93 基线 + S1 管道加固 +7、S2 评估工具 +24、
   V5 波次扩容 +137，见 NUMBERS §3）；本文未直接引用该数字，此处仅作口径同步记录。
   另：**416 项 = 412 passed + 4 xfailed 为本地全量**（2026-10-15 阶段1 收档实测；阶段0 扩容 +60、阶段1 扩容 +95），
   **CI 实跑会少于本地（数据/产物依赖用例 skipif）**，见 `reports/refactor_wave5/test_gap.md` §1.3
   与 `docs/ACCEPTANCE.md` 附记 7。
4. **代码行号锚点（`exp/**`、`core/**`、`tests/**`）本次未复核**：V5 波次存在代码改动，
   行号可能漂移，如需引用请重新 `grep -n` 核实。
