# exp/07_nn_aec — NN-AEC 神经网络回声消除（WP-07 / 模块7，简历核心）

双输入（mic + far 复数谱）压缩 CRN，**0.638M 参数（<1M 约束）**，时间维严格因果
（单测锁死），20ms 帧率。对手是 WP-06 传统 AEC——同评估集直接对比。
单测全绿；全仓 416 项 = **412 passed + 4 xfailed**（**2026-10-15 阶段1 收档本机口径**；
其中 1 例 `--probe-mem` CUDA 用例为 GPU 门控，无 GPU 环境 skip；阶段0 收档时为 321 项、
阶段1 训练中一度为 352 项——台阶见 `docs/NUMBERS.md` §3）。

## 结构

```
mic ─STFT(512/320)─┐
                   ├─[B,4,257,T]─ CausalConv×3 [8,16,32] ─> F:257→33, T 不变
far(粗对齐)─STFT──┘        │跳连
                 LSTM(1056→96, 单向) → FC → CausalDeconv×3 → cRM
                 enh = cRM ⊙ mic谱 → ISTFT
```

**因果三件套**（流式就绪的架构保证，WP-05b cache 化即可上线）：
- conv 时间轴只左 pad 4（k=5）；deconv 输出右移 2 帧（对称转置卷积天生超前 2 帧，
  右移后依赖变为过去 4 帧）；LSTM 单向
- 单测 `test_time_causality`：扰动 2s 后的输入，2s 前输出逐位不变（±ISTFT 窗 margin）

**时延鲁棒**：数据层（WP-07D 合成已含 10-80ms 随机时延）+ 训练层（dataset 再对
far 加 ±10ms 帧级抖动）。`eval.py --jitter-sweep` 输出 ±5/10/20/40ms 的指标衰减曲线。

## 用法

```bash
# 数据（已生成）：D:/data/aec-synth/train_v1（3000 条：FEST 1205/DT 882/NST 913）
#   生成命令见 README 底部；RIR 池 60 条（pyroomacoustics，100ms 截断）

python train.py --data D:/data/aec-synth/train_v1 --val D:/data/aec-synth/eval_v3 \
    --device cuda --epochs 60                       # 主配置（γ=0.5, jitter 10ms）
python train.py --device cuda --gamma 0             # 消融：γ=0（复数谱 MSE）
python train.py --device cuda --jitter-ms 0         # 消融：无时延扰动
python train.py --smoke                             # 合成数据链路冒烟

python eval.py --ckpt outputs/nnaec_g0.5_j10/best.ckpt --compare-trad   # + WP-06 对比表
python eval.py --ckpt ... --jitter-sweep                                # 时延鲁棒性
```

损失：`weighted_distortion(enh谱, near谱, γ=0.5) + w_sisnr×SI-SNR`（默认 w_sisnr=0.1，
**主配置 v4 为 w_sisnr=1.0**；core/losses 复用，γ 消融开关即 WORKLIST 要求的"γ 消融"）。

**P0-5 损失矩阵开关（2026-10-13 新增，阶段0 0-1）**：`--loss psycho` 把上式第一项换成
心理声学损失（`core.losses.psychoacoustic_loss`：ERB 临界带 + 耳蜗压缩的可听度加权 + 4–8kHz 谱倾斜项，
无判别器、显存与基线同档），SI-SNR 项与其它超参一字不改 ⇒ 与 `--loss weighted`（v4 原列）构成单变量对照。
缩放常数 c 缺省在启动时按训练集头部 4 批实测 `mean(L_WD)/mean(L_psy)` 并写入 ckpt cfg
（也可 `--psycho-c` 显式给定）；损失量级/梯度量级实测见 `loss_scale_probe.py`。
驱逐矩阵结果、CI 与不闭环项见 `reports/p0_5_loss_matrix.md`（**train_v1 未动**）。

```bash
python train.py ... --loss psycho --out outputs_v4_p05/psycho_240ep   # 心理声学列
python exp/07_nn_aec/loss_scale_probe.py                              # c 的探针实测
python exp/07_nn_aec/eval_p05_matrix.py --ckpt ... --sets eval_v3=... eval_v5=...
python exp/07_nn_aec/bootstrap_p05.py && python exp/07_nn_aec/bootstrap_p05_pairs.py
```

## AECMOS 官方口径（已集成，`aecmos_eval.py`；模型 AECMOS_local/Run_1668423760）

```bash
python aecmos_eval.py --trad                         # 传统基线（outputs/aecmos_trad_wp06.json）
python aecmos_eval.py --nn-ckpt <ckpt>               # NN-AEC
python aecmos_eval.py --enh-dir <dir>                # 任意增强输出目录
```
口径说明：官方对 ICASSP 长测试集要求裁剪送评（FEST 后半段等）；eval_v3 是 8s 短片段
（2s DT lead-in 为滤波器收敛设计），全段送评——绝对值与官方比赛有口径差，同集对比结论不受影响。
**送评采样率（2026-10-11 口径修复，V5）**：本地模型 `Run_1668423760_Stage_0.onnx` 的实际
`sampling_rate = 48000`，官方入口用 `librosa.load(sr=48000)`；本脚本曾把 16k 波形**直送**
（绕过重采样）⇒ 自合成集分数全部失真。现已统一重采样到模型采样率（`resample_poly`，gcd 约简），
口径定名 **AECMOS@48k 模型协议**；**旧值全部作废**（见 `docs/NUMBERS.md` 顶部决策条与
`reports/aecmos_rescore_v5fix.md`）。
返回值 (echo_mos, deg_mos)：echo 管回声抑制、deg 管近端退化（PESQ 抓不到的部分 AECMOS 抓得到，
实测传统 AEC DT 段 PESQ +0.27 而 deg_mos **−0.361**（新口径；旧口径 −0.78）。

## 结果（主配置 v4，eval_v3 60 条，三方同口径；AECMOS 列为 **@48k 模型协议**）

**PESQ/ERLE（eval.py --compare-trad）**：

| 指标 | 传统 AEC (WP-06) | NN-AEC v4（ckpt=v4） | 备注 |
|---|---|---|---|
| FEST ERLE | 20.37 dB（同集 eval_v3） | **69.83 dB** | 掩码门控 ≈ 静音输出（口径产物，不可与线性 AEC 的 ERLE 直接比） |
| DT 近端 PESQ | **1.641** | 1.403 | 近端保真是短板（见差距分析）；1.641 为 `eval.py --compare-trad` 口径，`eval_aec.py` 验收口径 **1.687**（脚本间口径差，见 `docs/NUMBERS.md` WP-06 口径注） |
| DT STOI | 0.790 | 0.731 | |
| NST 近端 PESQ | 2.446 | **2.477** | 透传且略有增益 |

> <sub>**SI-SNR 口径注（V5/t24 传导）**：FEST 段的近端参考恒为零 ⇒ SI-SNR **无定义**，
> 评估链按 **NaN** 记录并跳过（`core.evaluate._si_snr_np` 对 ref/est 全零返回 NaN；
> 修正前 est 全零会得到 **0.0 dB** 的假"完美"分）。因此本表不列 FEST SI-SNR，
> `tools/evaluate_full.py` 汇总时该列全 NaN 的组也不参与平均（见 `_score_one` 注释）。</sub>

**AECMOS（aecmos_eval.py，本地 ONNX，@48k 模型协议）**：

| 场景 | 指标 | 原始 mic | 传统 AEC | NN-AEC v4（ckpt=v4） | DoD |
|---|---|---|---|---|---|
| FEST | echo_mos | 1.321\* | 3.207 | **3.194** | （新口径下 FEST echo 传统/NN 实质打平；级联 3.938 最高） |
| DT | echo_mos | 1.521\* | 3.494 | **4.175** | ✅ **+0.681±0.212**（要求 +0.3，CI 不含 0） |
| DT | deg_mos | 3.180\* | **3.598** | 3.237 | ⚠️ **仅 DT 段落后 −0.361±0.169**（机理=高频带选择性损伤+谱平坦化，非欠训） |
| NST | echo_mos | — | — | **5.0** | |
| NST | deg_mos | 2.665\* | 3.842 | **3.941** | ✅ **新口径下不再落后** |

> \* 原始 mic 列为旧口径（地板重评进行中，`docs/NUMBERS.md` 同注）。**旧口径对照（作废留档）**：DT echo 1.521/2.467/3.754、
> DT deg 3.180/2.401/1.743、NST deg 2.665/2.665/2.073、FEST echo 1.321/3.836/3.762（新口径 trad 3.207 / NN 3.194）——16k 直送错配，
> 见 `reports/aecmos_rescore_v5fix.md`；逐集新旧对照表见 `exp/07_nn_aec/outputs/aecmos48_v5fix/_t22_backfill.md`。

**DoD 判定**：echo 维度达标（DT 4.175 vs 传统 3.494，**+0.681±0.212 ≥ 要求 +0.3**）；
近端保真维度**只在 DT 段**未达标（deg 3.237 vs 传统 3.598，−0.361±0.169；NST 段 3.941 vs 3.842
**已不落后**）——结论按 V5 决策条收窄为：**DT 段存在 AECMOS-deg 落后，机理为高频带选择性损伤 +
谱平坦化**；~~"模型形态与 AECMOS-deg 定义之间的（全场景）结构性瓶颈"~~ **该表述作废**。

**近端保真度的完整探索矩阵（2026-10-10 汇总，每格都是真实训练+评估；**DT echo/deg 为旧的 16k 直送口径，已作废**，仅作变体间相对趋势留档，绝对值见上方 @48k 表）**：

| 变体 | 训练配置 | DT echo（旧口径） | DT deg（旧口径） | DT PESQ | 结论 |
|---|---|---|---|---|---|
| v1 | γ0.5 w0.1, 60ep | 3.478 | 1.919 | 1.405 | 基线 |
| v2 | lr 5e-4, ~100ep | 3.277 | 2.014 | 1.451 | 欠收敛 |
| v3/night | w0.5 续训, 186ep | 3.491 | 1.931 | 1.404 | deg 不随训练量动 |
| v4 | w1.0 + DT/NST 专注, 240ep | **4.175（新口径）** | **3.237（新口径）** | 1.403 | echo↑但 deg 仍略降 |
| γ=0 消融 | γ0 w0.5, 77ep | — | — | 1.391 | γ 影响小 |

**deg 是"DT 段 + 掩码形态"问题的判断依据**（V5 收窄）：五个变体跨越了训练量(60→240ep)、
损失权重(w_sisnr 0.1→1.0)、场景配比(全场景→DT/NST 专注)、γ(0→0.5)四个维度，
**DT 段 deg 始终低于同集传统基线**——不是欠训；而 **NST 透传段在新口径下不再落后**
（3.941 vs 3.842），故结论限定为 **DT 段**的机制性差距（高频带选择性损伤 + 谱平坦化），
不再宣称"全场景结构性瓶颈"。候选解释（下一步验证方向，README 纪律）：
1. NN 输出的"过干净"频谱（无自然噪声地板/房间感）被 AECMOS 判为失真；
2. DT 段起始的掩码开关瞬态（FEST 门控 → DT 打开）；
3. 官方评估协议要求 DT 只送最后 (len-15)/2 秒——我们的 8s 短片段含 2s lead-in
   过渡段，deg 可能被过渡段主导（分帧 AECMOS 可验证）。

## DM（动态混合）训练：口径声明（P1-3 / REMEDIATION_PLAN_V3.1 §3.2 + 附录B）

阶段1 的数据轴改动是 **DM 化**：不预生成数据，训练时由 DataLoader worker 现场跑
`scripts/aec_synth.py` 的合成基元（`exp/07_nn_aec/dm_dataset.py::AecDynamicMix`）。
**val 不变**（仍是静态 `--val` 集），所以新旧读数可比。

```bash
# 阶段1 正式训练（命令口径见 REMEDIATION_PLAN_V3.1 附录B；清单随仓 scp，服务器只按清单取子集）
python exp/07_nn_aec/train.py --device cuda --dynamic-mixing --epochs 240 \
    --batch-size 16 --num-workers 8 --probe-mem 20 \
    --near-dir <pool>/near --far-dir <pool>/far --noise-dir <pool>/noise \
    --near-list split_lists/speech_near_train70.txt \
    --far-list  split_lists/speech_far_train70.txt \
    --noise-list split_lists/noise_train70.txt \
    --rir-dir <rir-pool> --rir-max-ms 100 \
    --val D:/data/aec-synth/eval_v8 --out exp/07_nn_aec/outputs_v5gen

# DM 管线冒烟 + 吞吐预检（附录B 第 2 步；--dry-run 不落盘）
python scripts/dm_aec_synth.py --num 100 --dry-run \
    --near-dir <pool>/near --near-list split_lists/speech_near_train70.txt \
    --far-dir <pool>/far --far-list split_lists/speech_far_train70.txt \
    --noise-dir <pool>/noise --noise-list split_lists/noise_train70.txt --rir-dir <rir-pool>
```

**三条硬口径（全部写进 `train_manifest.json` + ckpt 的 `cfg["train_manifest"]` + stdout 日志）**：

1. **清单驱动（0-5）**：场景配比 / DT 的 SER / DT 的 SNR / DT 引导时长来自
   `configs/dt_plan_v1.json` 的 preset（缺省 `train_v1_repro` = 0.4/0.3/0.3、SER 0–10、
   SNR 15–30、引导 1.5s）；清单 `version` + `preset` + `sha256` 一并落盘。
   **场景配比可显式覆盖**：`--dm-scenarios FEST,NST,DT`（和 = 1），来源记作
   `scenario_ratio_source=cli_override/preset`。动因是两者的区别——
   preset 描述**数据集**配比，而 v4 的**训练**另叠了 `--scenarios dt,nst`
   （cfg 实证：0.4914/0.5086，没见过 FEST 场景）。阶段1 要严格复刻 v4 训练配比就显式
   `--dm-scenarios 0,0.4914,0.5086`；要不要借 DM 把 FEST 重新纳入训练是口径决定，
   不留在代码默认里（阶段0 §2.4：FEST echo 必须与 DT deg 并列进验收）。
2. **池隔离（0-2/0-3）**：`--near-list/--far-list/--noise-list` **必填**，实例化时做
   near/far 文件同一性 + utterance 主名互斥、noise∩speech 同一性校验；不通过直接退出
   （DM 现合成没有 manifest 兜底，泄漏只能在入口拦）。清单格式见 `split_lists/README.md`。
3. **epoch + LR/早停口径（附录B 方案 a + 阶段0 §2.3b）**：
   - `epoch_len` 按**等效更新步数**对齐 v4：目标是 v4 产物的 `step=40830`
     （`outputs_v4/nnaec_g0.5_j10/last.ckpt`；构成 186ep×187 + 54ep×112），
     `steps_per_epoch = round(40830/epochs)`、`epoch_len = steps_per_epoch × batch_size`
     ⇒ 240ep/bs16 = **170 步/ep、epoch_len 2720、总 40800 步（−30 步 = −0.073%）**；
     取整残差如实登记（`delta_steps`/`delta_frac`），不假装精确相等。换 `--epochs` 仍对齐总步数。
   - LR 调度**显式覆盖**：DM 缺省 `ReduceLROnPlateau(factor=0.5, patience=20)` +
     **关闭早停**（`--early-stop 0`）。动因是阶段0 实测：`patience=3` 会让 lr 在 20–30ep
     折半、ep55–71 就早停（`reports/phase0_acceptance_20261013.md` §2.3b），240ep 预算被截断。
     显式传 `--lr-patience/--early-stop` 永远优先，`*_explicit` 标记进 manifest。
     **实现是 `core/train_protocol.py`（单一来源）**：04（DCCRN，P1-5a）与 07 引用同一份
     `resolve_schedule` / `probe_memory`，各自只传自己的 legacy 缺省（07=8 / 04=10）；
     改口径只需改一处，避免"两处实现漂移"（阶段0 教训）。
   - `history.json` 每 epoch 追加 `epoch_len/steps_per_epoch/target_steps/schedule/patience/
     early_stop`，防 "240ep" 在新数据下被误读为等价训练量。

**训练对照必须从零起（阶段0 §2.3a）**：DM 训练拒绝从**非 DM 谱系** ckpt 续训
（`cfg` 里没有 `train_manifest` 就报错退出），因为那样既检验不了数据/损失变量，又会把
旧 LR 口径（v3 在 ep186 已降到 1.53e-8，后 54ep 近乎冻结）带进来；`from_scratch` 标志
写进 manifest。同口径 DM 续训允许（epoch/step 递增）；口径漂移（epoch_len / 清单 sha256 /
preset 变了）会重置 best/bad 并显式提示"不再等价于从零起的对照"。

**级联 DM（P1-3b）**：级联 = 同一个 NN 换一对输入 `(e_lin, y_echo)`（线性 MDF 级进循环），
由 `--cascade-linear` 打开（训练集换成 `exp/09_cascade_aec/cascade_dm.AecCascadeDynamicMix`，
线性级口径进 manifest 的 `data.linear_stage`；输出目录再带 `_casc`）；09 另有薄壳入口
`exp/09_cascade_aec/train.py`。历史静态级联集是**跨 clip 复用同一 TradAEC 实例**、DM 是
**每条 clip 新建**，两者不等价（历史级联读数不与新 DM 读数直接混比）——见
`exp/09_cascade_aec/README.md` §P1-3b。

**边界（必须同页写）**：
- **内容级改名副本不做自动判据**（同一语音换文件名拷进另一池）：判据只到路径/文件项 +
  utterance 主名，全量大池每次启动哈希代价高；内容去重按 0-2 的 sha256 清单法离线做。
- **`--dm-scenarios` 顺序 = FEST,NST,DT**（v4 训练配方 = `0,0.5086,0.4914`，即 nst 913/1795
  / dt 882/1795、无 FEST）；manifest 记**具名字典** + `scenarios_explicit` 字面键，
  收档按具名值核对（顺序写反过，见单测的非对称断言）。
- **方案 (b)（按数据量同比例放大总迭代数）未实现**：DM 池是无限组合、没有可比的"数据量"
  分母；本轮定死方案 (a) 并写入 manifest（`epoch_scheme.scheme = "a"`）。
- `--dm-epoch-len` 是冒烟/调试的显式覆盖，manifest 记 `epoch_len_source=cli_override`：
  **正式训练不要给**，否则等效步数口径不成立。
- DM 的输出目录带 `_dm` 后缀（`nnaec_g0.5_j10_dm`），不会覆盖静态数据训练的既有产物。
- **不开 `--dynamic-mixing` 时行为与现状逐字节一致**：新参数不进非 DM 的 ckpt `cfg`、
  不产出 `train_manifest.json`、`early-stop` 仍为 8、`patience` 仍为 3（单测锁死）。

### 池枚举序：平台无关（P1-5c，2026-10-13）

所有池枚举（near/far/noise/rir）统一走 `aec_synth.list_pool` / `sort_pool_paths` ——
按**池内相对路径的 UTF-8 字节序**排序，不再用 `sorted(Path)`。原因：`PurePath.__lt__`
比较的是 normcase 后的部件（Windows 折叠大小写 / POSIX 不折叠）⇒ 池内文件名大小写混合
时，两平台"下标→文件名"映射分叉，同 seed 逐行结果不同。实测（本机 Windows）：DNS 噪声池
**637/659**、eval_v8 噪声池 **82/95** 个位置与字节序不同；语音池（993/599/10000）与 RIR 池
（60）**完全不变**。UTF-8 字节序 ≡ 码点序 ⇒ **服务器（Linux）行为不变**，Windows 归一到
POSIX 口径。

> **收档标准表述（t14 可直接引用）**：
> "池枚举排序键自 2026-10-13 起统一为池内相对路径的 **UTF-8 字节序**（POSIX 语义）；
> 仅**噪声池**受影响（DNS 噪声池 637/659、eval_v8 噪声池 82/95 个位置不同；语音与 RIR 池
> 不变）。**已冻结集一律不重生成**（历史读数口径不变），但新生成集与冻结集之间
> **不可按行对齐**（集合相等，`noise_file` 映射可能不同）；服务器重放本地冻结集时的同类
> 差异（eval_v8 1234/1400 行 `noise_file` 映射不同）即该缺口的既有表现，不是数据损坏。"

## v5x 数据池扩展：DM 多源（P1X-2，2026-10-13）

**单变量纪律**：v5x 与 v5 的差异**只有数据池**（far 端多语料 + 噪声多源）；训练配方
（bs16、场景配比 `0,0.5086,0.4914`、附录B 方案(a)、patience 20/关早停、从零训练、val 同集）
**完全继承 v5**，故 v5→v5x 的增量可归因到"数据多样性/规模"。

**怎么给多源**（三侧同一套旗标；`|` 分隔——**不是** `:`，Windows 盘符含冒号）：

```bash
--far-source "DIR|LIST[|WEIGHT[|LABEL]]"      # 可重复；权重列可留空（DIR|LIST||label）
--far-source-weights 0.5,0.3,0.2              # 可选：覆盖内联权重（长度须=源数）
```
与 v5 的 `--far-dir/--far-list`（单源）**互斥**，同时给直接报错（不静默择一）。

**采样语义（必须同页写）**：多源时每样本每侧**先按归一权重抽源（1 次额外抽样）、再在源内
均匀抽文件** ⇒ `sampling_scheme=weighted_source_then_uniform_intra`；**单源时与 v5 逐位一致**
（不额外抽样，由 `tests/golden/dm_single_source_v5.json` 逐字段+逐位锁死）。因此 v5x 与 v5
**不逐样本对齐**（抽样流多了一次源选择），只保证**口径对齐**——归因看集级统计，不看逐行。

**manifest 留痕**：`data.sources`（每侧每源 `dir/list/list_sha256/n_files/weight_raw/
weight_norm/probe_sr`）始终在；单源时 `pools`/`split_lists` 与 v5 **同形**（多源时
`split_lists[side]` 给 `{n, sha256_combined, sources}`）；另有 `sampling_scheme`、
`sampling_note`、`resample`、`scene_param_policy`。

**RIR 池留痕（t26/t27，2026-10-14）**：`data.rir` = `{src, dir, list, list_sha256, n_files,
set_sha256, set_sha256_basis, enumeration, max_ms, order_matches_pool, n_in_pool,
extra_pool_files, warnings?}`。
- `src`：`simulated`（pyroomacoustics）/ `dir`（整目录枚举）/ `list`（清单驱动）；
- `set_sha256`：**实际使用**的 RIR 集合指纹，按**池内相对路径**（POSIX、排序后）算 ——
  换服务器/换挂载点不变（本轮"新服务器续跑"正踩这个场景），且当清单恰好按字节序枚举整池时
  `set_sha256 == list_sha256`（可与 teammate 已留档基线直接对账：`rir_train70.txt` 基线
  = `a9383200bf29…`，本机按新口径复算**逐位相同**）。`enumeration` 记枚举口径
  （无清单 = `list_pool` 平台无关字节序；有清单 = 清单序）——集合 + 口径唯一确定
  index→文件 映射，故不必再记有序列表；
- 新增 `--rir-list`（07/04/dm_aec_synth）：给了就**清单驱动**（训练侧只应吃
  `split_lists/rir_train70.txt`，把 eval/holdout RIR 挡在门外），清单里若有池外成员会 fail fast；
- **t27 顺序护栏**：RIR 抽样**按下标**取（`rng.integers(len(rir_files))`）⇒ 顺序也是接口。
  清单驱动时当场核对"清单序 vs 目录字节序"：**集合相同、顺序不同 ⇒ 开跑横幅 warn**
  （提示 index→文件 映射已变、同 seed 不再复现历史 RIR 序列），并把 `order_matches_pool=False`
  + `warnings` 写进 manifest；**集合不同**（池内有清单外成员）不报错——那是清单驱动的隔离
  用法——但提示"本次只使用清单内 n/m 条"，避免"切清单 = 零变化"被静默打破。
  `load_rir_pool(..., warn=False)` 可静音（脚本/一次性核对用）。
- **对已经开跑的 v5x 队列不改命令**（换 RIR 池 = 换训练数据，属第二变量）。队长 2026-10-14
  裁定：**新服务器级联续跑启用 `--rir-list split_lists/rir_train70.txt`**（清单 ≡ 原
  `rirs_train70` 目录，集合+顺序双验一致（train-eng 服务器实测）⇒ 逐样本对齐，不构成第二变量）。
- 兼容键 `rir_src`/`rir_max_ms` 保留（既有审计脚本读它们）；**不回溯改写历史 manifest**，
  新块只对之后新写的 run 生效。

**格式扩展**：清单成员可写**相对路径**（LS/AISHELL-3/MUSAN 的 ext 清单就是这种，如
`train-clean-360/1079/128631/1079-128631-0000.flac`、`music/fma-western-art/...wav`），
扩展名 `.wav` + `.flac`；**非 16k 源**（AISHELL-3 44.1k）在 worker 内用 `resample_poly`
（gcd 约简）重采样到 16k，受影响的源记入 `resample.applied_to`。

**场景参数域（P1X-2 ③）：有意不改**（`scene_param_policy=inherit_v5_dt_plan_v1`）。
中文远端/音乐噪声带来的是**数据多样性**；若同时调 SER/SNR/引导/时延/soft-clip 的参数域，
就等于给 v5→v5x 引入第二个变量，归因立刻失效。音乐噪声是否需要单独 SNR 分布属**阶段2**
议题（建议用 sweep 探，而不是改 v5x 默认）。

**v5x 命令（NN-AEC 优先；配方逐项 = v5）**：

```bash
# 0) 前置（服务器侧跑一次；池在服务器）：生成扩展语料的**训练侧文件级清单**
#    注意 --root 指向**真正含 subset 目录**的那层（LS 池多一层 LibriSpeech/，见下注）
python scripts/build_extension_file_lists.py --corpus librispeech \
    --root /root/data/librispeech/LibriSpeech \
    --readers split_lists/ext_ls_readers_train70.txt \
    --holdout-readers split_lists/ext_ls_readers_holdout15.txt \
    --out split_lists/ext_ls_far_train70_files.txt --verify
python scripts/build_extension_file_lists.py --corpus aishell3 \
    --root /root/data/aishell3 --readers split_lists/ext_aishell3_readers_train70.txt \
    --holdout-readers split_lists/ext_aishell3_readers_holdout15.txt \
    --out split_lists/ext_aishell3_far_train70_files.txt --verify

# 1) v5x（唯一变化 = far/noise 多源；DNS far 源必须与 v5 逐字相同）
python exp/07_nn_aec/train.py --device cuda --dynamic-mixing --epochs 240 \
    --batch-size 16 --num-workers 10 --probe-mem 20 \
    --dm-scenarios 0,0.5086,0.4914 \
    --near-dir /root/data/dns-clean/near \
    --near-list split_lists/speech_near_train70_files.txt \
    --far-source "/root/data/dns-clean/far|split_lists/speech_far_train70_files.txt|0.5|dns_far" \
    --far-source "/root/data/librispeech|split_lists/ext_ls_far_train70_files.txt|0.3|ls_far" \
    --far-source "/root/data/aishell3|split_lists/ext_aishell3_far_train70_files.txt|0.2|aishell3_far" \
    --noise-source "/root/data/dns-noise|split_lists/noise_train70.txt|0.7|dns_noise" \
    --noise-source "/root/data/musan|split_lists/ext_musan_train70.txt|0.3|musan" \
    --rir-dir /root/data/rir-pool --rir-max-ms 100 \
    --val <与 v5 完全相同的 val 集> --out exp/07_nn_aec/outputs_v5x
```

> **前置清单的两个实操注意（train-eng 实测，2026-10-13）**：
> ① `--root` 要给**真正含 subset 目录的那层**：LS 池是 `/root/data/librispeech/LibriSpeech/`
> （外面多一层），AISHELL-3 是 `/root/data/aishell3`；
> ② 隔离断言按**路径字段**判 reader（`LAYOUT[corpus]["reader_index"]`），**不是子串**——
> 曾用 `any(r in f for r in hold)` 被子串误伤：holdout reader `669` 命中 chapter 目录
> `121669`，导致 LS 清单生成被自己的断言拦下（生成逻辑没问题，拦下前打印的 sha256 与
> 人工产出一致）。已修 + 回归单测（`100/121669/...` 不得误报）。
> 修好后 `--verify` 可对 train-eng 已有的 LS 产物（90418 行，sha256 `7c332a209420…`）
> 做逐行确认——**断言修复不改变生成结果**（只改判据，不动 `build()`）。

> **权重（队长 2026-10-13 裁定，t24 按此执行）**：far = DNS **0.5** / LS **0.3** / AISHELL-3 **0.2**；
> noise = DNS **0.7** / MUSAN **0.3** —— DNS 保持主导，MUSAN 作为**多样性注入**而非主导
> （取 data-eng 参数文档的起点）。**明确不用**"按文件数占比"（LS tc360 有 86.5k 文件会
> 压倒性主导）；"每语料等权 1/3"作为备选**留档不启用**。权重逐源记入 manifest
> （`sources[*].weight_raw/weight_norm`），改一行 CLI 即可但需同步收档口径。
> **near 侧有意保持单源**（DNS clean train70）：目标语音分布不变，增量才能归因到
> "远端/噪声多样性"。

## 训练数据复现命令

```bash
# 全部命令从仓库根（avsuite/）执行，脚本路径相对仓库根
# RIR 池（60 条，pyroomacoustics 随机房间，100ms 车厢尺度截断）
python -c "import sys; sys.path.insert(0,'scripts'); import numpy as np, soundfile as sf; \
import aec_synth as S; from pathlib import Path; rng=np.random.default_rng(0); \
p=Path('D:/data/aec-synth/rir_pool'); p.mkdir(parents=True,exist_ok=True); \
[sf.write(p/f'rir_{i:03d}.wav',(lambda h:(h/abs(h).max()*0.9).astype('float32'))(S.rir_simulate_pra(rng)[:1600]),16000) for i in range(60)]"

python scripts/aec_synth.py --num 3000 --dur-sec 4 \
  --out-dir D:/data/aec-synth/train_v1 --rir-dir D:/data/aec-synth/rir_pool \
  --rir-max-ms 100 --snr-range 15,30 --scenarios 0.4,0.3,0.3 \
  --soft-clip-prob 0.15 --dt-leadin-sec 1.5 --seed 21 \
  --near-dir D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean \
  --far-dir  D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean \
  --noise-dir D:/data/dns-challenge-mirror/datasets/noise
```

## 已知口径与局限（README 明示纪律）
- STFT 512/320：20ms 帧率 + 12ms 窗内前瞻——介于 2021 赛规格（40ms 含 20ms 前瞻）
  与 2022 规格（20ms 无前瞻）之间；严格无前瞻需 win=320 非 2 次幂 FFT，留作改进
- 流式推理 = WP-05b cache 化（conv 时间 cache + LSTM state）；架构已因果，
  等价性测试模式复用 WP-05b 的流式 vs 非流式逐帧对齐
- AECMOS 官方口径已集成（见上节 `aecmos_eval.py`），**送评前统一重采样到模型采样率 48k**
  （口径名 **AECMOS@48k 模型协议**；16k 直送旧值已作废）；非线性回声（soft-clip 15% 样本
  在训练集里）NN 天然能学一部分——WP-08 的对照组
- **指标 NaN 协议（V5/t24）**：静音/塌陷样本的 PESQ/STOI/SI-SNR 记 **NaN**（不写 0），
  FEST 段因近端参考为零，SI-SNR 恒为 NaN；汇总时全 NaN 的组不参与平均——读表时勿把 NaN 当 0
- 全部合成评估集（eval_v3/v5/v6）soft-clip 概率均为 0，而训练集为 0.15——非线性回声
  维度在主结果表无检验面（官方真实集除外，其回声为真实录制）
- **train/eval 数据重叠实测口径（2026-10-11；实测见 `reports/refactor_wave5/data_inventory.md` §6）**：
  同用 28 说话人池（28/28 重叠），但**语句级并非全不同**——eval_v5 行次中 near **67/300（22%）**、
  far **75/300（25%）** 与 train_v1 **是同一 wav 文件**（`fit_len` 随机裁剪使片段偏移不同，句/人/内容同源；
  DT 层最重：near 36/154、far 48/154）；**噪声池与训练 99~100% 重叠**（eval_v5 297/300，各合成集噪声
  均取自 train_v1 用过的同一 DNS 池），**两个说话人 held-out 集也共享同一 DNS 噪声池**
  （eval_v4_speaker_heldout 59/60、eval_speaker_only_holdout 60/60）——**说话人维度已由 held-out 集缓解，
  噪声维度未隔离**，泛化结论保守解读。未见说话人泛化此前未单独评估——
  已补全全未见说话人（DNS clean 素材）重评：exp/07_nn_aec/outputs/eval_v4_speaker_heldout/summary.md；
  真实分布稳健性证据：`reports/eval_v5_dt300.md`（DT 扩样 154 条）、`reports/ood_real_echoes.md`
  （ICASSP2022 真实回声）、`reports/realrir_domain.md`（IR64-CAR 真实车厢 RIR）
- **隔离口径更新（2026-10-13，阶段0 工作项 0-4）**：新增隔离版 `eval_v8`（1400 条；near/far 取自 DNS clean
  `holdout15` 45 readers 的**互斥池**、噪声用 `noise_eval15`、`--nearfar-mutex utterance` + `dt_plan_v1` 留痕，
  与 train_v1/train_nl 语料 **0/1592 交集**）。存量 v4 在该集上：**DT echo 优势与 DT deg 缺口均经受隔离**
  （DiD +0.195 / +0.005），而 **FEST echo 的旧"NN 优势"被证伪为泄漏口径产物**（DiD −0.470）；
  但 `noise_eval15` 的 95 个文件历史上全被用过 ⇒ **对存量 v4 噪声轴尚未隔离**（对按 `noise_train70`
  训练的 v5 代际才是真隔离）。详见 `reports/eval_v8_isolation_20261013.md`、`docs/NUMBERS.md` §2。

## 流式推理（WP-05b 模式已实现，`streaming.py`）

```python
from model import NNAEC
from streaming import NNAECStreaming
st = NNAECStreaming(model, device="cpu")
out_chunk = st.process_chunk(mic_320, far_320)   # 20ms 进 20ms 出，块边界任意
```
- conv cache（每层 5 帧输入）+ LSTM state 外置逐帧推进 + 核心库新增
  `core.istft.StreamingISTFT`（滑动缓冲 WOLA，帧到即出块）
- **流式 vs 非流式逐帧对齐 <1e-5**（实测 ~1e-7，tests/test_streaming_nnaec.py 锁死；
  不定块长 800/640/960 混合推入同样一致）
- cache 内存 ~270KB 有界、与音频总长无关（`cache_size_bytes()` 记账，
  单测断言滚动不增长）
- 两个实现级细节（坑已填平进测试）：deconv 右移 2 帧的启动延迟 = 前 2 帧输出零；
  StreamingISTFT 默认窗是 torch 同款**周期** hann（np.hanning 是对称窗，对拍必翻车）
