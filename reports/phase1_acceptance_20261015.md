# 阶段1 验收摘要（P1-7）+ DP3 决策摘要

- 日期：2026-10-15 ｜ 执行：eval-eng（t14）｜ 依据：`REMEDIATION_PLAN_V3.1.md` §3–§6 + 队长 2026-10-15 派单
- 上游依赖：**t28**（B 档正式级联并入矩阵）、**t24**（级联正式版训练与产物）、**t13/t23**（评测矩阵与轴集）、
  **t11/t12**（DM 吞吐检查点与训练排期）、**t21**（扩量数据入库）、**t6**（阶段0 收档）
- 纪律：**先改 `docs/NUMBERS.md`（权威表）再同步文档**；历史数字只标注不删除；负结果留档。
  本报告只做汇总与决策输入，任何数字与 NUMBERS 冲突时**以 NUMBERS 为准**。

---

## 0. 结论摘要

1. **阶段1 完成且达标**：三模型两代际（v5 = DNS 单源 / v5x = 扩展池）全部训完并完成 **110 key /
   144,100 条次**的矩阵评测（12 套冻结集 × 9 个 tag：传统臂 + v4 + NN×3 + DCCRN×2 + 级联×2），
   与 t5/t16 冻结产物逐行对账 **0.000000**、QA `problems=0`、全仓 pytest **416 = 412 passed + 4 xfailed**。
2. **卡时与费用远低于预算**：实测 **≈25.4 卡时**（预算 50–90、上限 110）、**≈¥40–65**（预算 ¥100–300）
   ——含污染的 **6.42 h 返工**（RIR 池未排除评测 8 条 IR，已净池重训闭环）。
3. **三条收档级结论**（详见 NUMBERS 顶部 2026-10-15 决策条与 §2 新节）：
   ① 代际 = **用 echo 换 deg**（`v4 − nn_v5`：DT echo +0.82…+1.37、deg −0.36…−0.71），**唯一反转 `eval_rir`**；
   ② 扩数据**主要买 DT echo**（v5x − v5：12/12 集显著 −0.04…−0.31，代价 deg 略降）；
   ③ **级联是最强者**（`casc_v5x @ eval_v8` DT echo 4.324 / deg 3.998 / PESQ 2.119），
   且 **MDF-only 消融**直接测出"FEST 能力来自线性级"（仅 MDF 3.208 > trad 3.003 > v4 2.896 ≫ 纯 NN 1.625）。
4. **DM 吞吐是真瓶颈（负结果 + 缓解）**：live 供给最好 322.8 samples/s vs 硬线 552.4（1.2×GPU 460.3）
   ⇒ 必须走**预生成池 ≥36.6%**（实测 442.7–524.8 samples/s，全量 40800 步需 171.1 GB / 265.7 音频小时）。
5. **DP3 建议：续租进阶段2，但把预算压在"sim-to-real + 数据边际收益验证"上，先不做 SSM**
   （理由与假设清单见 §5；阶段2 申请 **30 卡时**，含返工余量）。

---

## 1. 阶段1 交付清单（P1-1…P1-7）

| 工作项 | 产物 | 位置 | 验收 |
|---|---|---|---|
| P1-1 数据扩量 | LibriSpeech tc100+tc360、AISHELL-3、MUSAN（ModelScope 快路，65.48 GiB） | 服务器 `/root/data/*`；留痕 `split_lists/DATA_GAP_20261013.md` §7/§8 | 字节数/`gzip -t`/sha256 抽检逐条记录；压缩包即删 |
| P1-2 扩量池切分 | `ext_*` 三分割清单 + `rir_train70`(20) 等 | `split_lists/`（服务器同版） | reader/文件级零交集；`test_split_lists_t17.py` 等 7+5 例 |
| P1-3/3b DM 化训练管线 | NN-AEC/DCCRN/级联三模型 DM 训练 + manifest | `exp/07_nn_aec`、`exp/04_dccrn`、`exp/09_cascade_aec`、`core/*` | `test_nn_aec_dm.py` 20 / `test_dccrn_dm.py` 8 / `test_cascade_dm.py` 11 / `test_dm_multisource.py` 8 |
| P1-5 DM 吞吐硬检查点 | 供给/预生成/batch 上界实测 | `p1_artifacts/outputs_d1/t11_*.json` | 检查线 1.2× 未达 ⇒ 走预生成池（§3） |
| P1-5 三模型重训（两代际） | 7 套官方 run（含净池重训） | `p1_artifacts/outputs_p1/`（0.28 GB） | `train_manifest.json` 记 `data.rir = {dir,list,list_sha256,set_sha256,n_files=20,order_matches_pool}` |
| P1-6 全套评测 | 110 key / 144,100 条次矩阵 + CI/DiD/ERLE 三分报 | `exp/07_nn_aec/outputs/eval_matrix/**` | QA `problems=0`；对账 0.000000；报告 `reports/p1_6_matrix_eval_20261014.md` |
| P1-7 收档 | 本报告 + NUMBERS 更新 + 11 份文档同步 | `docs/NUMBERS.md`、`reports/` | 见 §6 |

---

## 2. 三代际登记（分开登记；引用必须带代际名）

| 代际 | 训练数据（口径） | ckpt | 训练日期 | 备注 |
|---|---|---|---|---|
| **v4**（旧锚） | VoiceBank-DEMAND 3.3 h，非 DM | `exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt`（本机） | 2026-10-10 前 | 阶段0 基线；与 v5/v5x 的可比性由**同一冻结评测集**保证 |
| **v5**（DNS 单源） | DNS clean `train70` + `noise_train70` + `rir_train70`(20 条)；DM，`w_sisnr=1.0`，240ep/40800 步 | `nn_aec_v5/`、`dccrn_dm/`、`casc_v5_official/` | 2026-10-13→14（**净池重训**） | 污染 run 已归档 `archives/…`；截断版级联已归档 |
| **v5x**（扩展池） | v5 + LS(tc100/360, far, 权重 0.3) + AISHELL-3(far, 0.2) + MUSAN(noise, 0.3) | `nn_aec_v5x/`、`nn_aec_v5x_noMUSAN/`、`dccrn_v5x/`、`casc_v5x_official/` | 2026-10-14 | 逐池 `ext_*_train70_files.txt` + sha256 入 manifest ⇒ 评测 holdout 零泄漏 |

**eval_v8（隔离集，1400 条）关键读数**（完整表见 NUMBERS §2「阶段1 代际对照」）：

| 模型 | DT echo | DT deg | FEST echo | NST deg | PESQ_dt | SI-SNR |
|---|---|---|---|---|---|---|
| 传统 AEC | 3.105 | 3.822 | 3.003 | 4.146 | 1.563 | 9.78 |
| v4 | **4.106** | 3.458 | 2.896 | 4.186 | 1.312 | 8.05 |
| NN v5 | 3.059 | **3.824** | 1.630 | 4.366 | 1.497 | 8.69 |
| NN v5x | 3.227 | **3.874** | 1.625 | 4.371 | 1.547 | 9.09 |
| DCCRN v5 / v5x | 2.434 / 2.684 | 4.238 / 4.132 | 1.433 / 1.475 | 4.407 / 4.376 | 1.504 / 1.516 | 7.52 / 8.11 |
| **级联 v5 / v5x** | **4.280 / 4.324** | **3.915 / 3.998** | **3.532 / 3.796** | 4.365 / 4.368 | **2.066 / 2.119** | **12.99 / 13.15** |

> 读法：v4→v5 家族是 **echo ↓ / deg ↑** 的交换；级联把两者**同时**抬到全矩阵最高。
> **FEST 一列与训练样本几乎无关**（v5→v5x 仅 +0.005），与 MDF-only 消融一致 —— 见 §5 假设清单 A2。

**多轴验证矩阵（t23 七轴）**：`music`（MUSAN 噪声）/`zh`（中文远端）/`zh_both`（双端中文）/
`snr_low`(0–10 dB)/`snr_high`(30–40 dB)/`far_ls`（LS 远端）/`long_ls`(12 s)。
NN v5x 的轴效应（DT echo vs eval_v8，两样本）：snr_high **−0.438**、far_ls −0.254、ns −0.175、
music −0.140、zh −0.120、zh_both −0.119、snr_low **+0.185**；v5x 相对 v5 在各轴改善
（zh **−0.309**、long_ls −0.273、far_ls −0.254、music −0.153）。完整表见 NUMBERS §2「阶段1 多轴验证矩阵」。

---

## 3. DM 吞吐硬检查点与租卡实测（预算 vs 实测）

| 项 | 实测 | 判读 |
|---|---|---|
| GPU 纯前向（NNAEC 0.638M，bs16×4 s） | 460.3 samples/s | 天花板 |
| 硬检查线（≥1.2×） | **552.4 samples/s** | **未达**：live 最好 322.8（8 worker）= 0.70× |
| 供给+训练合体（10 worker 稳态） | 455.1 samples/s | `pass_1_2x=false` |
| 预生成池（8/10 proc） | 442.7 / 524.8 samples/s | 需覆盖 **36.6%** 样本（实测） |
| 全量 40800 步池 | 239,156 样本 / **171.1 GB** / 265.7 音频小时 / 建池 0.13 CPU·h | 磁盘纪律下可行 |

| 卡时/费用 | 预算（V3.1 §4.1/§4.3） | 实测 | 判读 |
|---|---|---|---|
| 阶段1 卡时 | 50–90（上限 110） | **≈25.4**（老机窗口 ≤17.15 含 6.42 h 返工 + 新机级联 8.23） | **约 1/3 预算** |
| 阶段1 费用（¥1.5–2.5/卡时） | ¥100–300 | **≈¥40–65** | 显著低于预算 |
| 阶段0（本机 4060） | ≤10 GPU·h | 矩阵评测 ~3.5 h + 集生成 ~1 h + 池构建 ~0.5 h | 达标 |

> **返工归因（负结果留档）**：训练 `--rir-dir /root/data/rirs` 未排除评测用 8 条 IR ⇒ 两个级联 run 与
> 五个模型的污染 run 作废（老机 07:06→13:31 共 6.42 h 窗口计入返工），已按 `rir_train70`(20 条) 净池重训；
> 训练 manifest 现记 `data.rir.*` 字段（`list_sha256 = set_sha256 = a9383200…`、`order_matches_pool=true`）防复发。
> 卡时按"单卡占用窗口"记（并发 run 不重复计）；若新机级联两跑各占 1 卡则合计 ≈33.6 卡时，仍 ≪ 预算。

---

## 4. 负结果留档清单（不得删除，引用须带口径）

1. **损失矩阵未拿到可依赖 DT deg 增益**（2026-10-13，`reports/p0_5_loss_matrix.md`）：对称对照 seed0 +0.151
   显著、seed1 −0.007 归零 ⇒ 效应不可复现；②−v4 Δdeg +0.059 CI 含 0。**继续有效。**
2. **FEST echo 的 v5「NN 优势」撤回**（2026-10-13，泄漏口径产物）；本阶段进一步给出**架构解释**：
   纯 NN 的 FEST echo 1.5–2.0 显著低于 trad/v4/级联（3.0/2.9/3.5–4.3）——是**架构缺口**，不是调参问题。
3. **MUSAN 在 0–10 dB 轴上是"用 deg 换 echo"**：去 MUSAN 后 DT echo 12/12 集更差（−0.13…−0.37），
   但 `eval_snr_low` 上 noMUSAN 的 DT deg **+0.194 [+0.164, +0.225]**。
4. **DM 供给达不到 1.2× 硬线**（live 0.70×）⇒ 预生成池为必需（§3）。
5. **RIR 轴方向分组（观察、非因果）**：真实 IR 相对仿真，DT echo 变差 = trad −0.69 / v4 −0.44 /
   级联 −0.22、−0.35；变好 = 纯 NN +0.92…+0.98 / DCCRN +1.14/+1.37。**不得**写成"抑制越强则真实 IR 越差"。
6. **污染/截断 run 作废**：`archives/casc_truncated/`（级联 `w_sisnr=0.1` + ep79/76）与污染的
   RIR 池 run；两者均只作过程证据，任何读数不得引用。
7. **AISHELL-4 渠道阻断**（parquet/无 pyarrow）与 **train-other-500 磁盘纪律跳过**：属未做项，非负结果，
   接口留痕可补（`split_lists/DATA_GAP_20261013.md` §8.2/§8.3）。

---

## 5. DP3 决策摘要（是否续租进阶段2）

### 5.1 建议

**建议续租进阶段2（申请 30 卡时 / ≈¥50–75），但主线不是换模型，而是补"sim-to-real + 数据边际收益"两块证据。**

| 阶段2 候选 | 支持证据 | 反对/风险 | 建议 |
|---|---|---|---|
| **P2-4 SSM（Mamba ~2.7M）** | 长上下文理论优势 | **本轮没有证据**：`eval_long_ls`(12 s) 与 `eval_far_ls`(8 s) 几乎无差（级联 4.30 vs 4.32 DT echo、NN v5x 3.08 vs 2.97）⇒ 长上下文不是当前瓶颈 | **暂缓**（除非先做出"更长上下文有收益"的读数） |
| **P2-7 神经 Kalman** | 流式/时延口径 | 时延预算已达标（端到端 66.5 ms、实测 p99 67.8 ms）⇒ 收益不明确 | **备选**（若阶段2 要把级联做成流式再上） |
| **RIR-as-Prompt / sim-to-real** | **唯一有实测缺口的轴**：级联/纯 NN 在 `eval_rir` 上收益消失甚至反向（−0.22…+0.11）；真实 IR 池仅 8 条、单辆车 | 需要真实多车 IR 数据（采集或公开集） | **首选**（与"多车 RIR 池"一起做） |
| 补训（tc360/other-500） | 扩数据已有正收益（v5x 改善 0.10–0.31） | 边际收益未测；other-500 会引入第二变量 | **做单变量补训**（见 5.2） |

### 5.2 LS/MUSAN/扩展池补训收益评估（基于本轮实测）

| 语料 | 轴对齐的边际收益（v5x − v5，DT echo，逐行配对） | 干净消融 | 判读 |
|---|---|---|---|
| AISHELL-3（中文） | `eval_zh` **−0.309**、`zh_both` −0.195 | — | 中文远端收益最大（轴对齐，与语料一致） |
| LibriSpeech tc100（长语音） | `eval_far_ls` −0.214、`long_ls` −0.273 | — | far 端长句收益次之 |
| MUSAN（噪声） | `eval_music` −0.153 | 去 MUSAN −0.173（12/12 集更差） | echo 上正收益；**低 SNR deg 上有负作用（+0.194）** |
| 综合（v5x vs v5） | 12/12 集 DT echo 改善 −0.04…−0.31；FEST echo 几乎不动 | — | 扩量收益集中在 **DT echo**，代价 DT deg 略降（8/12 集显著） |

**补训选项（供 DP3 定档）**：
- **A（轻量，推荐先做）**：只补 **train-clean-360**（765 spk，已入库未入池）做**单变量**补训 ——
  按本轮实测单模型成本（NN v5x 0.94 h / DCCRN 3.04 h / 级联 8.19 h）**合计 ≈12.2 卡时**；
  若只补 NN 线则 **≈1 卡时**。预期收益：far 端说话人多样性↑（可测 `eval_far_ls`/`eval_long_ls`）。
- **B（可选）**：再补 `train-other-500`（28.49 GiB，需先解决磁盘阈值）——**会引入第二变量**，
  建议在 A 验收后再决定。
- **C（不建议）**：一次把 tc360 + other-500 + AISHELL-4（若解决 parquet 依赖）全上 —— 归因纪律不允许。
- 预算口径：阶段2 申请 **30 卡时**（A 约 12 卡时 + RIR 数据/实验 ~8 卡时 + 返工余量 10 卡时）。

### 5.3 「级联 RIR 轴负方向」原因假设清单（待验证，按优先级）

> 现象（配对实测）：级联与 v4/trad 同组——真实 IR 下 DT echo 更差（级联 −0.22/−0.35、v4 −0.44、trad −0.69）；
> 纯 NN 与 DCCRN 反向（+0.92…+0.98 / +1.14…+1.37）。**纯 NN 在真实 IR 上 DT echo 甚至反超级联（+0.111）。**

1. **线性级的 sim-to-real 缺口**（最可能）：MDF 在真实车厢 IR 上残留未消回声（late-reverb 更长/更密），
   而这些残留进入 NN 的后级输入 `e_lin` ⇒ 级联整体 echo 下降；纯 NN 不含线性级，故不受此缺口影响。
   验证：对同一批真实 IR 直接对比 `linear`（MDF-only）与仿真 `linear` 的 FEST/DT echo 差（新键 `linear:eval_rir` 已有，
   只需再跑 `linear` on sim 对照集——**本轮未做**，是阶段2 的第一个实验）。
2. **真实 IR 含线性卷积模型之外的因素**（扬声器/麦克风非线性、机舱噪声非平稳）：RIR 卷积只建模线性声学路径。
   验证：真实录音的线性可预测性分析（用 MDF 残差能量谱对比仿真）。
3. **仿真池的"过易"偏置**：pyroomacoustics 8 房间 + 300 ms 上限，早期反射/EDC 形状与真车不同 ⇒
   模型在仿真上被"喂成"了对特定 EDC 敏感；真实 IR 下 AECMOS 判据更严。
   验证：用真实 IR 的 EDC 统计量（T30/C50）分桶重评。
4. **真实 IR 池过小且单一车辆**（8 条 held-out / 20 条训练，同一辆车）：对"未见的真实 IR"是小样本外推。
   验证：扩到多车 IR（阶段2 数据项）。
5. **ERLE 与 AECMOS 判据的交互**：级联 ERLE 稳态 26–34 dB、v4 65.6 dB、纯 NN 2.5 dB——
   三档与 RIR 方向分组同序（观察）。**不构成因果**，需用 `linear` 消融 + 分桶来分离。

---

## 6. 文档与留痕同步（t14 执行清单）

| 文件 | 改动 |
|---|---|
| `docs/NUMBERS.md` | 新增顶部**统一决策（2026-10-15 阶段1 收档）**（三代际登记 + 三条主结论 + 四条强制口径 + 五条负结果）；§2 新增「阶段1 代际对照」「阶段1 多轴验证矩阵」「阶段1 DM 吞吐硬检查点与租卡实测」「外部数据集留痕」；§3 单测计数 → **416 = 412 passed + 4 xfailed**（52 个测试文件） |
| `README.md`（仓根） | 测试计数 2 处 → 416 |
| `docs/ACCEPTANCE.md` | 测试计数 3 处 + 计数明细（+阶段1 扩容 +95）+ 日期口径 |
| `docs/CHEATSHEET.md` / `docs/index.md` / `docs/README_HOME.md` / `docs/STORY.md` / `docs/PAPER2CODE.md` / `docs/INTERVIEW-WP02E-eval.md` | 测试计数 → 416（口径日期 → 2026-10-15 阶段1 收档） |
| `exp/10.5_capstone/README.md` | 全仓计数 → 416 |
| `exp/07_nn_aec/README.md` | 计数 → 416（并保留 321→352→416 台阶说明） |
| `reports/README.md` | 索引登记本报告 + `p1_6_matrix_eval_20261014.md` |
| **工作区级** `D:/Project_for_Python/DSH/reports/` | `t24_cascade_official_20261014.md`（级联正式版训练与 §9 机理佐证）、`t24_v5x_cascade_bringup_20261014.md`（扩展池 bringup）—— 不在 `avsuite/` 仓内，引用时带工作区路径 |
| `reports/phase0_acceptance_20261013.md`、`reports/multi_axis_matrix_20261013.md` | **历史数字保留**，各加一行台阶注（→ 416，见 NUMBERS §3） |

> 计 11 份文档（`docs/` 8 + 仓根 README + `exp/10.5_capstone/README.md` + `exp/07_nn_aec/README.md`）+ NUMBERS 表本体。

---

## 7. 未闭环项（交阶段2 或长期）

1. `linear` on **仿真**对照集未跑 ⇒ §5.3 假设 1 尚待直接测量（成本 ≈1 卡时）。
2. 真实 IR 仅单辆车 8+20 条；**多车 IR** 是 sim-to-real 结论的前置。
3. `eval_zh_both` 的 deg 在 v4 上低至 2.96（当前最好为级联 3.87）：双端中文近端仍是弱轴。
4. MUSAN 的低 SNR deg 负作用（+0.194）未定位（是噪声类型、还是混合权重）。
5. `train-other-500`/AISHELL-4 未入库（磁盘/依赖），接口留痕可补。

---

## 8. 复现命令（逐字）

```powershell
$py='C:\Users\14805\.conda\envs\avsuite\python.exe'    # 仓库根 avsuite/ 下执行
# 评测矩阵（本机，6 进程；含 A/B 档与 MDF-only 消融）
powershell -NoProfile -ExecutionPolicy Bypass -File exp/07_nn_aec/run_matrix_eval.ps1 -Workers 6
& $py exp/07_nn_aec/eval_matrix.py --ckpt linear=<casc_v5x best.ckpt>:linear --keys linear:eval_v8
# 出表 / QA / 收档候选 / 报告
& $py exp/07_nn_aec/eval_matrix_report.py ; & $py exp/07_nn_aec/eval_matrix_qa.py
& $py exp/07_nn_aec/numbers_candidates.py
& $py exp/07_nn_aec/embed_matrix_tables.py --report reports/p1_6_matrix_eval_20261014.md `
    --tables exp/07_nn_aec/outputs/eval_matrix/_report/matrix_tables.md
# 卡时（tb events 真实 wall_time）
& $py exp/07_nn_aec/tfevents_span.py D:/Project_for_Python/DSH/p1_artifacts/outputs_p1
# 全仓单测（终值 416）
& $py -m pytest -q
```

---

## 9. 交付清单

| 类别 | 路径 |
|---|---|
| 本报告 | `reports/phase1_acceptance_20261015.md` |
| 权威表 | `docs/NUMBERS.md`（阶段1 决策条 + 4 个新节 + §3 计数） |
| 评测读数 | `exp/07_nn_aec/outputs/eval_matrix/**`（110 key + `_report/{matrix_ci.json,matrix_tables.md,qa.json,numbers_candidates.md}`） |
| 矩阵报告 | `reports/p1_6_matrix_eval_20261014.md`（3494 行）+ `reports/multi_axis_matrix_20261013.md`（轴集定义） |
| 训练产物 | `p1_artifacts/outputs_p1/`（7 套官方 run）+ `archives/casc_truncated/`（过程证据） |
| 吞吐/卡时 | `p1_artifacts/outputs_d1/t11_*.json` + `exp/07_nn_aec/tfevents_span.py` |
| 数据留痕 | `split_lists/DATA_GAP_20261013.md`、`split_lists/ext_*`、`configs/dt_plan_v{1,2}.json` |
