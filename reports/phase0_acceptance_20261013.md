# 阶段0 验收摘要（0-1 / 0-2 / 0-3 / 0-4 / 0-5）— DP1 决策输入

- 日期：2026-10-13 ｜ 执行：收档（t6，eval-eng）｜ 依据：`REMEDIATION_PLAN_V3.1.md` §2 阶段0、§6 阶段0 验收与 DP1
- 纪律：**先改表（`docs/NUMBERS.md`，注明 checkpoint + 数据口径 + 日期）再同步文档**；负结果留档、
  外部引用带获取日期、旧结论不删除只按口径标注。
- 本报告只是**汇总与索引**：任何数字与 `docs/NUMBERS.md` 冲突时以 NUMBERS 为准。
- 阶段0 全程**零租卡**（本机 RTX 4060 Laptop 8GB）；五个工作项未改动任何既有评测集/历史读数。

---

## 0. 一句话结论（DP1 用）

**阶段0 五项全部完成且可复现**；读数给出一个**方向明确的组合结论**：

> **数据侧（隔离 + 扩样）是 DT deg 缺口的唯一未证伪路线**——损失矩阵这一轴没拿到可依赖的增益（负结果留档），
> 而隔离版 eval_v8 的读数证明 **DT deg 缺口不是泄漏造成的（DiD +0.005，CI 含 0）**、
> **DT echo 优势也不是泄漏造成的（DiD +0.195，显著为正）**；同时 **FEST echo 的旧"NN 优势"被证伪为泄漏口径产物**
> （DiD −0.470），阶段1 重训必须把 FEST echo 一并纳入验收。

**建议（供用户/队长裁定）**：**按原预算进入阶段1（租 1×4090，50-90 卡时、上限 ~110 卡时 / ¥100-300）**，
并带上 §6 的 6 条前置条件；若预算紧张，优先保 **DM 化 + 数据隔离/扩样**，把损失函数的探索降到"顺带跑一列"。

---

## 1. 五工作项状态总表

| # | 工作项 | 状态 | 产出（仓内/数据） | 关键验收 | 关键数字 |
|---|---|---|---|---|---|
| 0-1 | P0-5 损失矩阵三列重训（train_v1 不动） | ✅ 完成（列③ skipped，方案留档） | ckpt `exp/07_nn_aec/outputs_v4_p05/*`；`reports/p0_5_loss_matrix.md`、`p0_5_loss_matrix_ci.md` | ckpt sha256 全表 `outputs_v4_p05/aecmos/runs_index.json`；复刻校验 v4 列与 NUMBERS 逐位一致 | **deg 增益不可复现**（seed0 +0.151 显著 / seed1 −0.007 CI 含 0）；②−v4 Δdeg +0.059 CI 含 0、Δecho −0.285 显著退化 |
| 0-2 | 噪声池三分割 | ✅ 完成 | `split_lists/noise_{train70,val15,eval15}.txt`(472/92/95)、`noise_split_manifest.csv`、`noise_split_meta.json` | `verify_split.py --reproduce` → **exit 0**（train∩eval=0、桶归属、权威表、版本哈希 7 项）；负向自检 exit 1 / 输入错 exit 2 | 池 659 全覆盖；`sha256(filename)[:8] % 1000` |
| 0-3 | near/far 分池互斥约束 | ✅ 完成 | `scripts/aec_synth.py --nearfar-mutex {off,pool,utterance}`；`tests/test_aec_synth_nearfar_mutex.py`(17 例) | 新增 17 例全过；缺省 `off` 与旧版逐比特一致（黄金基准回归）；违例在任何写盘前 fail fast | eval_v8 生成时以 `utterance` 档通过；反向对照 exit 1 且不产出 |
| 0-4 | 隔离版 eval_v8 重生成 + v4 基线读数 | ✅ 完成 | 数据 `D:/data/aec-synth/eval_v8`(1400 条/4200 wav)+`PROVENANCE.md`；`reports/eval_v8_isolation_20261013.md`；`split_lists/speech_*.txt`；脚本 `split_speech_pool.py`/`eval_v8_audit.py`/`eval_v8_isolation.py` | 准入审计 8 组 **PASS**；同命令重生成 **4200/4200 wav + manifest + profile 逐字节一致**；`verify_split --reproduce` PASS；eval_v5 重跑与仓内权威 @48k 产物 NN 600/600 值全等 | DiD：DT echo **+0.195** [+0.069,+0.318]、DT deg **+0.005** [−0.120,+0.134]、FEST echo **−0.470** [−0.769,−0.161]；CI 收窄 0.39–0.67× |
| 0-5 | DT 合成比例清单化 | ✅ 完成 | `configs/dt_plan_v1.json`(sha256 `73496c0bc877…`)+`configs/README.md`；`scripts/dt_plan.py`、`verify_dt_plan.py`；`tests/test_dt_plan.py` | `--write` 幂等、篡改一行 exit 1、坏清单 5 类 exit 2；**eval_v8 上 `--strict-preset` PASS（C 走"实跑锚定"取证）+ `--replay-sim-rir --replay-prefix 3` PASS（逐字节）**；**缺省不给 `--dt-plan` 与旧版逐字节一致（train_v1 3000 条 manifest 回归）** | preset `train_v1_repro`（0.4/0.3/0.3，1.5 s，SNR 15–30）与 `eval_v5_repro`（0.25/0.25/0.5，2.0 s，20–30）如实分列 |

**统一口径**：所有 AECMOS 数字 = **AECMOS@48k 模型协议**；配对 95%CI = bootstrap **B=2000 / seed=777**（百分位法）；
DT 场景对外引用必须带 CI（`tools/compute_ci.py` / `reports/DATA_CI.md`）。

> **0-5 验收器修复记录（t7，2026-10-13，已闭环）**：0-4 验收时发现
> `verify_dt_plan.py` 的 C/G 两项只建模 `--rir-dir` 集，对 sim-RIR 集直接报
> `LogRng.uniform() got an unexpected keyword argument 'size'`（详见
> `reports/eval_v8_isolation_20261013.md` §1.3）；该问题已由 **t7（data-eng）** 修闭，本任务复跑实测：
>
> - **C（同分布核对）PASS，exit 0**：
>   `python scripts/verify_dt_plan.py --plan configs/dt_plan_v1.json --manifest D:/data/aec-synth/eval_v8/manifest.csv --strict-preset --limit 20`
>   （需 `DT_PLAN_NEAR_DIR / DT_PLAN_FAR_DIR / DT_PLAN_NOISE_DIR` 指向 `eval_v8_pools/{near,far,noise}`）。
>   取证方式已从"离线重放随机流"换成**实跑一次 + 每样本"收尾 `random` 前恰好 6 次 `uniform`"锚定清单块 + 逐行对账**，
>   另用块首 `random`（`pick_scene` 抽值）按清单配比反推场景、与 manifest 的 `scenario` 列对账，作为**独立的切流对齐证据**，
>   强度与离线重放等价。
> - **G（产物逐字节复现）PASS，exit 0**：加 `--set-dir D:/data/aec-synth/eval_v8 --replay-prefix 3 --replay-sim-rir`，
>   manifest + wav 逐字节一致。sim 档**默认跳过 G 并打印原因**（需本机 pyroomacoustics 且版本一致）——
>   **"按设计可选"，不计失败**。
> - **门禁负向自检**：把 eval_v8 manifest 一行 DT 的 `ser_db` 改 +0.5 → **exit 1 且精确报出该行**
>   （`第 0 行（fileid_0_dt）ser_db: manifest=6.921… 实测流中该样本的采样值=6.421…`）。
> - **历史集退化路径修复**：eval_v5/v6/v7 由旧版 `aec_synth` 生成、无 `dt_plan_preset` 列 ⇒ 现在**只报一条准确原因快失败**
>   （"该集不是按 DT 比例清单生成的（历史集由旧版 aec_synth 产出），C/D/E/F 无从核对"），不再误报"区间越界/占比偏离"。
>   **收档口径 = "历史集非按清单生成，故不适用清单核对"，不是"历史集验收失败"。**
> - 适用边界与全部命令见 `configs/README.md` §3.1。

---

## 2. 0-1 损失矩阵：负结果与两条训练口径警示（必须进 DP1）

### 2.1 读数（eval_v5 DT，n=154，AECMOS@48k，配对 B=2000 seed=777）

| 列 | 谱误差项 | eval_v5 DT echo | eval_v5 DT deg |
|---|---|---|---|
| 存量锚 v4（426ep lineage） | — | 4.248 | 3.220 |
| ① V1 原列（频带均衡 + DT 加权，γ0.5，从零） | `weighted_distortion` | 4.102 | 3.129 |
| ② 心理声学列（ERB 28 带 + 4–8kHz 谱倾斜，无判别器，从零） | `psychoacoustic_loss`（c=135.17） | 3.964 | **3.279** |
| ③ flow matching 列 | — | — | **skipped**（不同构、机制先验最弱，方案留档 §7） |

### 2.2 负结果留档（不得宣称"deg 缺口已收回"）

1. **对称对照（②−①，同数据/同超参/同调度）不可复现**：seed0 Δdeg **+0.151 [+0.086, +0.217]**（显著），
   seed1 Δdeg **−0.007 [−0.069, +0.052]**（CI 含 0）；两 seed 的 echo 项甚至反号 ⇒ 本列改的是
   "echo↔deg 再平衡点"，**再平衡方向随种子漂移**，不是稳定 deg 增益。
2. **相对存量 v4 不显著**：②−v4 Δdeg **+0.059 [−0.009, +0.125]（CI 含 0）**，同时 Δecho **−0.285 显著退化**
   （以 echo 换 deg）。
3. **量级校验**：①列跨 seed 的 deg 极差 0.136、echo 极差 0.332 ⇒ **效应量 ≈ 噪声量级**。
4. **①列（原损失、从零、早停 ep60/63）本身就比 v4 低**（deg −0.092 显著、echo −0.146 显著）——
   这是**训练预算/谱系差**（v4 是 426ep lineage），不是管道漂移：复刻校验（v3→240ep 重放 V1 路径）
   与 v4 的 eval_v5 DT echo/deg 差 ≤0.001/≤0.002，best val 8.4768 vs 8.5852。

### 2.3 两条训练口径警示（阶段1 必须遵守）

- **(a) 损失/配方对照必须"从零训练"或从高 lr 检查点起**：v3 在 ep186 时 lr 已降到 **1.53e-8**
  （后 54ep 近乎冻结）⇒ "在 v3 上续训"无法检验损失函数。从零路线的诊断 run 在 lr 2e-4/1e-3 下均平台化于
  val≈36–39（`outputs_v4_p05/{lossctrl,pilot_constlr,pilot_lr1e3}` 留档），故主对照采用"从零 + 同调度两列对称"。
- **(b) epoch 口径声明须显式覆盖 LR 调度**：本轮 `ReduceLROnPlateau`（patience=3）会在 20–30ep 左右折半 lr
  并触发早停（ep55–71）⇒ 阶段1 附录B 的 `epoch_len` 换算不能只声明"240ep"，必须同时登记学习率调度与早停判据，
  否则"240ep"在新数据下不可比。

### 2.4 FEST echo 掉幅必须与 DT deg 并列进验收（DP1 只看 DT deg 会漏）

从零两列在 FEST echo 上**显著退化**：v4 **3.881** → ① **2.690**（配对 −1.19 显著）/ ② **2.348**（−1.53 显著）；
两列训练配比均为 `--scenarios dt,nst`（**没见过 FEST 场景**）且从零 ≤63ep ⇒ FEST 段抑制能力主要靠
**长谱系/训练预算**。**阶段1 重训须把 FEST echo 口径一并纳入验收**（与 eval_v8 的 FEST echo 观察一起看，见 §3）。

---

## 3. 0-4 隔离读数：三条强制口径 + 结果

**集定义**：1400 条 × 8 s@16k（FEST 310 / NST 384 / DT 706），`seed=20261014` 固死；
near/far = DNS clean `holdout15` reader **互斥池**（993 / 599，reader + 文件级零交集）；
噪声 = `noise_eval15`（95）；RIR = pyroomacoustics 仿真；`--nearfar-mutex utterance` + `dt_plan_v1/eval_v5_repro`。
ckpt = `outputs_v4/nnaec_g0.5_j10/best.ckpt`（存量 v4，未重训）。

| eval_v8（隔离） | n | 传统 AEC | NN v4 | Δ（配对 95%CI） |
|---|---|---|---|---|
| DT echo_mos | 706 | 3.105 | **4.106** | **+1.001 [+0.951, +1.049]** |
| DT deg_mos | 706 | 3.822 | 3.458 | **−0.364 [−0.404, −0.327]** |
| FEST echo_mos | 310 | 3.003 | 2.896 | −0.107 [−0.245, +0.024]（不显著） |
| NST deg_mos | 384 | 4.145 | 4.186 | **+0.041 [+0.005, +0.076]** |

| 隔离性偏差 DiD | DiD | 95%CI | 判读 |
|---|---|---|---|
| DT echo | **+0.195** | [+0.069, +0.318] | NN 优势**非泄漏**、隔离后更大 |
| DT deg | **+0.005** | [−0.120, +0.134] | 缺口**结构性**（与 deg 机理节一致） |
| FEST echo | **−0.470** | [−0.769, −0.161] | v5 的 +0.363 优势 = **泄漏口径产物** |
| NST deg | −0.011 | [−0.155, +0.135] | 持平（轻微领先保留） |

**三条强制口径（引用上述数字必须同时具备）**：

1. **FEST echo 的 v5「NN 优势」撤回**：旧结论保留、标注口径 ——
   "v5（泄漏口径）NN−传统 +0.363 显著 → v8（隔离口径）−0.107 不显著，DiD −0.470 [−0.769, −0.161]"；
   eval_v7（n=212）Δ=+0.052 同向，**两集都不支持 FEST echo 的 NN 优势**。
2. **噪声轴对存量 v4 未隔离**：`noise_eval15` 的 95 个文件全部被历史训练用过（`train_v1` 用过全池 659 中的
   652 个，含 94 个 eval15）⇒ 本 DiD 主要量化**说话人/语料**隔离；对按 `noise_train70` 训练的 v5 代际才是真噪声隔离。
3. **"绝对分下降 = 隔离偏差"不成立**：语料由 VoiceBank `train/clean` 换成 DNS clean holdout reader，
   **传统基线同期同样下降**（FEST echo 3.518→3.003，−0.515）⇒ **只有 DiD 列可作隔离偏差引用**。

**顺带的新观察（未进入任何主结论）**：隔离集上传统基线的 **NST PESQ 2.459→2.797**、NN 几乎不动（2.381→2.344）
⇒ NN 与传统在 NST PESQ 上的差由 −0.078 拉到 **−0.453**；即 AECMOS-deg 的"NST 不再落后"没有波形域 PESQ 的支持。
只作记录，留待后续机理工作（与 deg 机理同源）。

**级联 FEST echo 锚点未在隔离口径复检（必须注明）**：本阶段未对级联打分；eval_v8 上 NN 的 FEST echo 已略低于
传统（2.896 vs 3.003）。引用"级联 FEST echo 三方最高"（eval_v5 级联−NN +0.315±0.203、eval_v7 +0.562±0.128）
时须注明**该锚点未在隔离集验证**，且 `reports/eval_v7_fest_nst.md` 已限定其只在合成 RIR 域成立。

---

## 4. 0-2 / 0-3 / 0-5 清单与版本登记（`docs/NUMBERS.md` §2 已同步）

| 工作项 | 文件/接口 | 内容 | 版本哨兵 |
|---|---|---|---|
| 0-2 | `split_lists/noise_{train70,val15,eval15}.txt` | 472 / 92 / 95（池 659 全覆盖） | `noise_split_meta.json`：`sha256(filename)[:8] % 1000`，<700 / 700–849 / ≥850 |
| 0-3 | `--nearfar-mutex {off,pool,utterance}` | 缺省 `off`（默认路径逐比特不变）；eval_v8 用 `utterance` | manifest 列 `nearfar_mutex`（eval_v8 1400/1400） |
| 0-4 | `split_lists/speech_*.txt` | 305 reader → 223 / 37 / **45 holdout**；holdout 再分 near 22 人/993 wav、far 23 人/599 wav | `speech_split_meta.json`：`sha256("reader:"+id)`、`sha256("nearfar:"+id)` |
| 0-5 | `configs/dt_plan_v1.json` | `eval_v5_repro` / `train_v1_repro` 两个历史口径 preset | 清单 sha256 **73496c0bc8778432d8f18694d012dccb598d0306f0983bd08e74460440062ef4** |

**§3.1 规模下限达成情况（V3.1 新增的 DP1 检查项）**：

| 指标 | 下限 | 本机实测 | 结论 |
|---|---|---|---|
| 可用 readers 总数 | ≥300 | DNS-Challenge clean 镜像 **305**（10000 wav） | ✅ 达标（服务器侧全量池更大，下载后需按同一脚本重算并登记） |
| held-out 评测份额 | ≥40 人 | `holdout15` **45** 人 | ✅ 达标 |
| 其中 near / far 双端 | 各 ≥20 | **22 / 23** | ✅ 达标 |

---

## 5. 外部引用与留痕

- **arXiv:2608.02918**（Abreu/Miranda/Biscainho, *JAES* 74(6) 430-444, 2026）——**获取日期 2026-10-13**；
  0-1 列②仅借用"耳蜗压缩 + 临界带可听度加权"思想骨架，**未复用其代码/权重、不声称数值等价**；
  落点 `core/losses.py::psychoacoustic_loss`；登记于 `docs/NUMBERS.md` 引用节。
- 阶段0 **未新增任何数据集**；0-4 的池全部来自本机既有镜像（DNS clean / DNS noise），不涉及新下载。
- 各 run 的 ckpt sha256 与评测口径元数据：`exp/07_nn_aec/outputs_v4_p05/aecmos/runs_index.json`；
  eval_v8 的清单/清单 sha256：`D:/data/aec-synth/eval_v8/{manifest.csv,dt_plan_profile.json,PROVENANCE.md}`。

---

## 6. DP1 决策输入汇总（含阶段1 前置条件）

**支持进入阶段1 的读数**：

1. 数据隔离**已完成且可复现**（0-2/0-3/0-4 三轴清单 + 隔离集冻结 + 逐字节重生成）；
2. **DT deg 缺口不是泄漏造成的**（DiD +0.005）→ 换数据/重训是唯一未被否证的修复路线；
3. **DT echo 优势也不是泄漏造成的**（DiD +0.195）→ 阶段1 重训的 echo 结论可用同一 eval_v8 对照，
   不会因隔离而"凭空退步"；
4. 损失矩阵轴**已被否证**（负结果留档）⇒ 不必用"损失函数已解决"为由推迟数据侧投入。

**进入阶段1 的 6 条前置条件（任一缺失都会让阶段1 的对照失效）**：

1. **训练侧显式传隔离清单**：噪声用 `split_lists/noise_train70.txt`，语音 near/far 用
   `speech_readers_train70`（+ 互斥），否则 eval_v8 对 v5 的隔离性质不成立
   （见 `split_lists/README.md` §4/§6）；
2. **损失/配方对照从零或高 lr 检查点起**（v3 续训在 lr≈1.5e-8 下无效，§2.3a）；
3. **`epoch_len` 换算 + LR 调度/早停判据一并写入 manifest**（§2.3b，附录B 口径声明）；
4. **FEST echo 纳入验收表**（本轮从零两列 FEST echo 掉 1.19–1.53，§2.4）；
5. **DM 吞吐硬检查点**（供给速率 ≥ 训练消耗×1.2）达标后才排训练（V3.1 §4.2/§5）；
6. **代际对照用同一冻结集**：v4@eval_v8（本报告 §3 的数字）vs v5@eval_v8，带配对/两样本 CI 与 DiD 口径。

**预算提醒**：50-90 卡时（+20% 上浮 → 上限 ~110 卡时 / ~¥300）＝租 **1 张** 4090 跑 2-4 天；
CPU 侧首选 12-16 核机型（DM 6-8 worker + RIR 卷积 + 重采样）。

**明确不建议**：在损失函数上再投入主要精力（0-1 已给否证）；把阶段2（SSM/Kalman/RIR-as-Prompt）
提前到期内（P2 候选，不占本期资源）。

---

## 7. 收档清单（本任务改动）

| 文件 | 动作 |
|---|---|
| `docs/NUMBERS.md` | 新增「统一决策（2026-10-13 阶段0 收档：隔离版 eval_v8，0-4）」（含三条强制口径 + FEST echo 撤回 + 噪声轴未隔离 + 级联锚点未复检）；§2 新增「阶段0 隔离版 eval_v8」小节（读数表 + DiD 表 + 清单版本登记表 + 边界注）；§3 单元测试口径刷新为 **321 项 = 317 passed + 4 xfailed** |
| `reports/phase0_acceptance_20261013.md` | 本文件（阶段0 验收摘要，DP1 输入） |
| `reports/README.md` | 索引登记（本报告、0-1 报告、0-4 报告） |
| `exp/07_nn_aec/README.md` | 数据重叠小节追加 0-4 隔离结论；首部单测计数同步为 321 |
| 计数同步（NUMBERS 是 SSOT，逐处回改"当前口径"表述，**历史叙述保留**） | `README.md`（用法注释 + 计数口径注 + tests 徽章）、`docs/{ACCEPTANCE,CHEATSHEET,index,README_HOME,STORY,INTERVIEW-WP02E-eval,PAPER2CODE}.md`、`exp/10.5_capstone/README.md` |

**未改动（有意）**：根目录 `REMEDIATION_PLAN_V3.1.md`（队长的方案文件）、`docs/adr/*`（决策记录）、
各 exp 的评估报告正文与历史波次文档（`reports/refactor_wave5/*`、`exp/06_trad_aec/README.md` 的
"V5 收口时"历史表述等）——收档只改**当前口径**，历史陈述按纪律保持原样。

**并发收档注记（t7）**：本任务执行期间 data-eng 的 t7（verify_dt_plan 的 sim-RIR 适配，deps=t4）落地，
改动 `scripts/verify_dt_plan.py`（+6 测试 / `tests/test_dt_plan.py` 18→24 例）。因此：
单元测试最终计数 = **321 项 = 317 passed + 4 xfailed**（t7 落地后复跑，exit 0）（**阶段1 收档台阶：416 项 = 412 passed + 4 xfailed**，见 `docs/NUMBERS.md` §3 与 `reports/phase1_acceptance_20261015.md`）；正文 §1 的 0-5 行
与上方修复记录均已按 **t7 后**的实测（C/G PASS）书写。

**产物卫生（队长 2026-10-13 裁定，已执行）**：`reports/DATA_CI.md.bak_p05`（0-1 重生成 DATA_CI.md 前的备份）
在全仓 md/py/ps1 中无引用，且 DATA_CI.md 受 git 跟踪、历史可回溯——备份冗余，**已删除**。

> 负结果留档纪律：0-1 的"deg 增益不可复现"、列③ skipped、0-4 的"噪声轴对 v4 未隔离"、
> 0-5 的"训练/评测 DT 引导时长与 SNR 区间本就不同"全部如实入档，不做正面化改写。
> 历史数字（eval_v5 泄漏口径）**不删除**，只按数据口径标注。
