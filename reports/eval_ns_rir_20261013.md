# eval_ns / eval_rir — 两条新评测轴的落地（held-out 噪声轴 + held-out 真实 RIR 轴）

- 日期：2026-10-13 ｜ 执行：eval-eng（t16，"P1-6a 前置"）｜ 依据：`REMEDIATION_PLAN_V3.1.md` §3.2
  （评测轴互斥且冻结：eval-sp / **eval-ns** / **eval-rir**）与 §7「RIR 优先入 eval 轴」
- 上游依赖：**t9**（服务器数据五件套 + 训练侧 near/far 互斥清单）、**t5**（eval_v8 与 split_speech_pool 机具）、
  **t4**（dt_plan_v1）、**t7**（verify_dt_plan 的 sim-RIR 适配）
- 纪律：只**新增**评测集与清单；不改既有 eval 集（eval_v5/v6/v7/v8 全部冻结）、不改 NUMBERS 已收档数字
  （两集尚无模型读数，读数属 t13 预研）。所有集合定义、清单哈希与命令逐字留档在本报告与各集的
  `PROVENANCE.json`。

---

## 0. 结论摘要

1. **两套新评测集已生成并冻结（+1 个配对对照集）**：
   - `D:/data/aec-synth/eval_ns`（1400 条；FEST 352 / NST 324 / DT 724；`seed=20261015`；噪声 = `noise_eval15`）
   - `D:/data/aec-synth/eval_rir`（1400 条；FEST 323 / NST 363 / DT 714；`seed=20261016`；RIR = **held-out 真实车厢 RIR** 8 条）
   - `D:/data/aec-synth/eval_rir_sim`（**RIR 轴配对对照集**：同 seed/同清单/同池规模，RIR = 8 个仿真房间；
     与 eval_rir **1400/1400 行逐列全等**、NST 363/363 mic 逐字节相同 ⇒ 逐样本配对，队长 2026-10-13 裁定方案一）
   三集均：8 s@16k、每场景 ≥300、`--nearfar-mutex utterance`、`dt_plan_v1/eval_v5_repro`、
   manifest 留痕三列齐全（`nearfar_mutex` / `dt_plan_version` / `dt_plan_preset`）+ 侧车 `PROVENANCE.json`
   记"噪声/RIR 来源清单版本"（含名单内容 sha256）。
2. **RIR 轴第一次有了训练/评测互斥的清单**：新建 `scripts/split_rir_pool.py`（组级切分）
   → `rir_{train70,val15,holdout15}.txt`（实测 train 5 组/20 条、val 1 组/4 条、holdout 2 组/8 条）；
   eval_rir 只用 holdout 组（`IR64CAR_0_*`、`IR64CAR_315_*`），与训练份额在**位置组级与文件级零交集**。
3. **验收全绿**：三集准入审计（eval_v8 回归 + eval_ns + eval_rir + eval_rir_sim，dir 档多一组 RIR 来源闭环）全 PASS；
   `verify_dt_plan.py` 在 eval_ns（sim，C 实跑锚定 + G 逐字节）与 eval_rir / eval_rir_sim（dir，C+G 原生回放）均 **exit 0**；
   eval_ns / eval_rir / eval_v8 之间**行级内容相同 0 行**（独立样本），而 **eval_rir ↔ eval_rir_sim 为逐样本配对**。
4. **必须同页读的两条口径**：① **eval_ns 与 eval_v8 是同一配方的独立重抽样**（同扬声器池/同噪声池/同 sim RIR，
   仅 seed 不同）——它是**噪声轴读数的复现集**，不是新的隔离维度；② **eval_rir 与本池只有一辆车**，
   它衡量"未见声源位置"的泛化，**不是"未见车辆"**。

---

## 1. 三集设计对照（谁被 held out）

| 维度 | eval_v8（0-4） | **eval_ns（本文 ①）** | **eval_rir（本文 ②）** |
|---|---|---|---|
| 用途 | 隔离版基线集（sp+ns 合并） | 噪声轴（held-out 噪声）读数/复现 | RIR 轴（held-out 真实 RIR）读数 |
| 说话人 | DNS clean `holdout15` 45 reader | **同 eval_v8** | **同 eval_v8** |
| near/far 池 | `eval_v8_pools/{near,far}`（993/599，reader+文件级互斥） | 同左 | 同左 |
| 噪声 | `noise_eval15`（95） | **同左** | **同左** |
| RIR | pyroomacoustics 仿真（无 `--rir-dir`） | **同左** | **真实 IR64-CAR held-out 8 条**（`eval_rir_pool`，`--rir-max-ms 300`） |
| seed | 20261014 | **20261015** | **20261016** |
| 条数/分层 | 1400（310/384/706） | 1400（352/324/724） | 1400（323/363/714） |
| manifest sha256 | `a4f9044677b6…` | 见 `PROVENANCE.json` | 见 `PROVENANCE.json` |
| dt_plan | `dt_plan_v1/eval_v5_repro` | 同左 | 同左 |

> 三集**每场景 ≥300 条**（V3.1 §3.2 的 n 扩档口径）；AECMOS@48k 口径下建议按 t5 的协议打分
> （配对 NN−传统 B=2000 seed=777；跨集差为**两样本非配对** bootstrap——不同集是不同样本）。

### 1.1 RIR 轴的**配对对照集** `eval_rir_sim`（队长 2026-10-13 裁定方案一 → t13 配对协议用）

| 项 | eval_rir（真实） | **eval_rir_sim（对照）** |
|---|---|---|
| RIR 池 | `D:/data/aec-synth/eval_rir_pool`（8 条真实车厢 IR，held-out 位置组） | `D:/data/aec-synth/eval_sim_rir_pool`（**8 个仿真房间**，`tools/make_sim_rir_ctrl_pool.py --seed 20261017`；与训练侧仿真池 60 条**逐字节不重叠**） |
| RIR 名单 | `split_lists/rir_holdout15.txt`（sha256 `c0544ab2cdca…`） | `split_lists/rir_sim_ctrl.txt`（sha256 `b2b984145d96…`） |
| 其余全部 | seed 20261016 / num 1400 / dur 8 / 同三池 / `--rir-max-ms 300` / `dt_plan_v1/eval_v5_repro` / `--nearfar-mutex utterance` | **完全相同** |
| 条数/分层 | 1400（323/363/714） | 1400（**323/363/714**，逐场景同数） |

**配对性实测**（这是"可配对"的硬证据）：

- 两集 manifest **1400/1400 行逐列全等**，且**逐字节相同**（sha256 均为 `e3a7a3d949575c95…`；
  因为逐行 RIR 文件名不进 manifest，记录列完全一致）——是"同一样本、只差回声路径"的最强形式证据；
- **NST 363/363 条 `mic` 逐字节相同**（NST 无回声 ⇒ 内置对照，直接证明"两集只差回声路径"）；
- FEST/DT 的 `mic` 必然不同（回声路径不同）；`far`/`near` 只在抗削波统一缩放 k 触发时才不同
  （实测 far **450/1400**、near **739/1400** 逐字节相同，差异全部来自 k 随回声峰值变化——
  属回声路径的一部分，不是另一条变量）。
- 两集池规模同为 8（`pool_sizes.rir=8`）⇒ `get_rir` 的 `integers(8)` 消耗一致 ⇒ 随机流同构。

---

## 2. 生成命令（逐字留档）

```powershell
$py='C:\Users\14805\.conda\envs\avsuite\python.exe'   # 从仓库根 avsuite/ 执行
$COMMON = @('--num','1400','--dur-sec','8',
  '--near-dir','D:/data/aec-synth/eval_v8_pools/near',
  '--far-dir','D:/data/aec-synth/eval_v8_pools/far',
  '--noise-dir','D:/data/aec-synth/eval_v8_pools/noise',
  '--snr-range','20,30','--soft-clip-prob','0','--nearfar-mutex','utterance',
  '--dt-plan','configs/dt_plan_v1.json','--dt-plan-preset','eval_v5_repro')

# ① eval_ns：与 eval_v8 同配方、异 seed（仿真 RIR）
& $py scripts/aec_synth.py @COMMON --out-dir D:/data/aec-synth/eval_ns --seed 20261015
# 实测：{'nst': 324, 'fest': 352, 'dt': 724}，exit 0

# ② eval_rir：唯一差异 = RIR 换成 held-out 真实车厢 RIR 池（300ms 截断，与 eval_v6 同口径）
& $py scripts/aec_synth.py @COMMON --out-dir D:/data/aec-synth/eval_rir `
    --rir-dir D:/data/aec-synth/eval_rir_pool --rir-max-ms 300 --seed 20261016
# 实测：{'nst': 363, 'fest': 323, 'dt': 714}，exit 0

# ②b eval_rir 的**配对对照集**（队长 2026-10-13 裁定方案一）：同 seed/同清单/同池规模，只把 RIR 换成仿真池
& $py tools/make_sim_rir_ctrl_pool.py --out-dir D:/data/aec-synth/eval_sim_rir_pool --n 8 `
    --seed 20261017 --list-out split_lists/rir_sim_ctrl.txt --exclude-pool D:/data/aec-synth/rir_pool
# 实测：8 房间；与训练侧仿真池（60 条）逐字节重叠 0
& $py scripts/aec_synth.py @COMMON --out-dir D:/data/aec-synth/eval_rir_sim `
    --rir-dir D:/data/aec-synth/eval_sim_rir_pool --rir-max-ms 300 --seed 20261016
# 实测：{'nst': 363, 'fest': 323, 'dt': 714}（与 eval_rir 逐场景同数 ⇒ 配对成立），exit 0
```

**留痕**（每集目录内）：

| 文件 | 内容 |
|---|---|
| `manifest.csv` | 1400 行；列含 `nearfar_mutex=utterance`（0-3）、`dt_plan_version=dt_plan_v1` / `dt_plan_preset=eval_v5_repro`（0-5）、逐行 `near_file/far_file/noise_file`（隔离可回溯） |
| `dt_plan_profile.json` | 清单 sha256 `73496c0bc877…`、seed、num/dur、`pool_sizes`（含 `rir`）、`pool_dirs`（含 `--rir-dir` 实际路径）、`cli_args` |
| `PROVENANCE.json`（本文新增） | 集合定义 + **来源清单版本**：near/far/noise **名单内容 sha256**、RIR 名单 sha256 + 池目录 + 训练/评测组清单、三列的列名快照（由审计脚本 `--write-provenance` 生成） |

---

## 3. 验收证据（全部本任务复跑）

| # | 命令 | 结果 |
|---|---|---|
| 1 | `python scripts/split_rir_pool.py --rir-dir D:/data/rir_real/pool_16k_mono --out-dir split_lists --materialize-root D:/data/aec-synth/eval_rir_pool` | exit 0：32 条/8 组 → train 5 组·20 条 / val 1 组·4 条 / **holdout 2 组·8 条**；硬链接物化 8 条 |
| 2 | `python scripts/split_rir_pool.py … --verify --materialize-root …` | exit 0：组级互斥 / 覆盖完整 / 权威表与版本哈希 / 池目录对账 |
| 3 | `python exp/07_nn_aec/eval_v8_audit.py`（默认 = eval_v8） | **PASS exit 0**（回归：参数化未改动默认语义） |
| 4 | `python exp/07_nn_aec/eval_v8_audit.py --set-dir D:/data/aec-synth/eval_ns --expect-seed 20261015 --write-provenance D:/data/aec-synth/eval_ns/PROVENANCE.json` | **PASS exit 0**：1400 行核；每场景 ≥300；三列齐全；行内 near/far/noise ∈ 对应名单；自回声 0；历史训练语料交集 0 |
| 5 | `python exp/07_nn_aec/eval_v8_audit.py --set-dir D:/data/aec-synth/eval_rir --expect-rir-src dir --expect-seed 20261016 --write-provenance …` | **PASS exit 0**：追加 I 组 —— 评测 RIR 池 == `rir_holdout15`（8 条，sha256 `c0544ab2cdca…`）、与 `rir_train70`（20 条）**文件级与位置组级零交集**、`profile.pool_sizes.rir=8` 对账 |
| 6 | `verify_dt_plan.py --manifest …/eval_ns/manifest.csv --strict-preset --limit 20`（`DT_PLAN_{NEAR,FAR,NOISE}_DIR`） | **PASS exit 0**（C：实跑锚定 + 场景反推） |
| 7 | `verify_dt_plan.py --manifest …/eval_ns/manifest.csv --set-dir …/eval_ns --replay-prefix 3 --replay-sim-rir` | **PASS exit 0**（G：产物逐字节） |
| 8 | `verify_dt_plan.py --manifest …/eval_rir/manifest.csv --strict-preset --set-dir …/eval_rir --replay-prefix 3`（`DT_PLAN_RIR_DIR`） | **PASS exit 0**（dir 档 C+G 原生回放，`rir=dir`、`pool_sizes.rir=8`） |
| 9 | 三集行级独立性核对（键 = 场景+近/远/噪文件+ser/snr/delay，不含 RIR） | eval_ns↔eval_v8 **0 行**、eval_rir↔eval_ns **0 行**、eval_rir↔eval_v8 **0 行** |
| 10 | `pytest tests/test_split_rir_pool.py -q` | 新增 **9 例全过**（分组规则/分配保底/组级互斥/幂等/坏名单 exit 1/输入错 exit 2/物化对账） |
| 11 | 与 **t9 训练侧清单**的互斥核对（reader 级；训练侧名单存的是 reader id） | eval 池 45 reader **== `speech_readers_holdout15.txt`**；`eval ∩ speech_near_train70 = 0`、`eval ∩ speech_far_train70 = 0`、`eval ∩ train70 = 0`；对照 `train_near ∩ train_far = 0` 且并集 == train70 |
| 12 | `python tools/make_sim_rir_ctrl_pool.py --out-dir …/eval_sim_rir_pool --n 8 --seed 20261017 --list-out split_lists/rir_sim_ctrl.txt --exclude-pool …/rir_pool` | exit 0：8 房间（16k mono PCM_16、峰值 0.9、0.6 s，与真实池同格式）；**与训练侧仿真池 60 条逐字节重叠 0** |
| 13 | `python exp/07_nn_aec/eval_v8_audit.py --set-dir D:/data/aec-synth/eval_rir_sim --expect-rir-src dir --rir-list split_lists/rir_sim_ctrl.txt --expect-seed 20261016 --write-provenance …` | **PASS exit 0**（dir 档 I 组：池 == `rir_sim_ctrl`（sha256 `b2b984145d96…`）、`pool_sizes.rir=8`） |
| 14 | `verify_dt_plan.py --manifest …/eval_rir_sim/manifest.csv --strict-preset --set-dir … --replay-prefix 3`（`DT_PLAN_RIR_DIR=eval_sim_rir_pool`） | **PASS exit 0**（dir 档 C+G 原生回放） |
| 15 | **配对性核对**（eval_rir vs eval_rir_sim） | manifest **逐字节相同**（sha256 均 `e3a7a3d949575c95…`）⇒ 1400/1400 行逐列全等；**NST 363/363 条 mic 逐字节相同**（内置对照）；far 450/1400、near 739/1400 逐字节相同（差异仅来自抗削波缩放 k）⇒ **样本级配对成立** |

---

## 4. RIR 分割清单（t16 新建，与 data-eng 的训练清单对齐）

```
group  = 'ir64car_pos<angle>'（文件名命中 IR64CAR_<angle>）或上级目录名
rank   = sha256("rir_group:"+group)[:8] 升序
n_hold = max(1, ceil(0.15·G))；n_val = max(1, round(0.15·G))；其余 train（同组文件不跨 split）
```

- 池：`D:/data/rir_real/pool_16k_mono`（32 条 = 8 声源方位 × 4 胶囊，16k mono 0.6 s，Zenodo 15168578 / CC BY 4.0）
- 实测：**holdout15 = `IR64CAR_0_*` + `IR64CAR_315_*`（8 条）**；val15 = `IR64CAR_45_*`（4 条）；
  train70 = 90/135/180/225/270（20 条）
- 互斥核对：holdout ∩ train = ∅（文件级）；位置组级 `{pos0,pos315} ∩ {pos90,135,180,225,270} = ∅`；
  三份额并集 = 全池 32
- 组数少（G=8）时精确 70/15/15 不可达，规则按 §7"RIR 优先入 eval 轴"把非训练份额保底，
  实际 **62.5 / 12.5 / 25**（组级），已写进 `rir_split_meta.json` 可独立重算

**两条必须同页写的边界**：

1. **只有一个 session（一辆车）**：位置级互斥成立，**车/录音级互斥做不到**；要测"未见车辆"需第二辆车的实测 IR
   （外部数据集清单见 `realrir_domain.md`）。
2. **`eval_v6_realrir` 用了全部 32 条 RIR** ⇒ 它与本清单的训练份额**重叠 20 条**。今后用 eval_v6 与
   eval_rir/v5 代际对比时，必须注明 RIR 口径（eval_v6 = 全池、eval_rir = 8 条 held-out 位置组）。

---

## 5. 两条轴怎么读（对照协议）

| 问题 | 建议对照 | 配对性 |
|---|---|---|
| 噪声轴（held-out 噪声相对训练） | eval_ns（或 eval_v8）**vs eval_v5**（旧口径：语音与噪声都与训练同源）——用 t5 的 DiD：`[(NN−传统)@iso] − [(NN−传统)@v5]` | **两样本非配对**（不同样本；传统基线臂吸收域偏移） |
| 噪声轴的稳定性 | **eval_ns vs eval_v8**（同配方异 seed，各 1400 条）——把两集合并/分别读，看结论是否复现 | 两样本非配对；两集期望分布相同 |
| **RIR 轴（真实车厢 → 仿真）** | **eval_rir vs eval_rir_sim**（§1.1 配对对照集：同 seed/同清单/同池规模/同 300ms 截断，唯一差异 = RIR 内容） | ✅ **逐样本配对**（manifest 1400/1400 行全等、NST 363/363 mic 逐字节相同）⇒ 用**配对** bootstrap（B=2000 seed=777）报 Δ 与 CI |
| RIR 轴（与旧 sim 口径的旁证） | eval_rir vs eval_ns（`--rir-dir` vs 现场仿真，随机流结构不同） | 两样本非配对（逐行内容相同 0 行，见 §3-9）——只作旁证，主读数用上一行 |
| 说话人轴 | eval_v8/eval_ns/eval_rir **vs eval_v5**（v5 为 VoiceBank 训练同源说话人） | 两样本非配对 |

> **配对性说明**：`--rir-dir` 档每样本在 RIR 步只消耗 1 次 `integers`，现场仿真档消耗可变次 `uniform`
> ⇒ 两类档之间逐行内容不可能对齐（实测 0 行相同）。因此 RIR 轴的**主读数**用
> **eval_rir vs eval_rir_sim**（两侧都用 `--rir-dir`、池规模同为 8 ⇒ 随机流同构、逐样本可配对）。
> 非配对处必须如实报"两样本 CI"并标注配对性（队长 2026-10-13 裁定）。

---

## 6. 实施中的坑（已踩，记录下来避免复踩）

1. **`DT_PLAN_RIR_DIR` 只能对 dir-RIR 集设置**：它会被 `verify_dt_plan.py` 当作"重放时补
   `--rir-dir`"的依据；对 sim 集设置它 → 重放按 dir 模式走 → 场景标签从第 1 行起全部不符（**假失败**，
   我在本任务第一次跑 eval_ns 时踩到，去掉该环境变量后 PASS）。
2. **`--rir-max-ms` 是集合定义的一部分**：eval_rir 用 300 ms（与 eval_v6 真实 RIR 同口径），
   sim 集不设截断；profile 的 `cli_args.rir_max_ms` 已留痕，重放必须还原。
3. **RIR 池目录记在 `dt_plan_profile.json.pool_dirs.rir`**（aec_synth 的 manifest 只记 `rir_src`，
   不记逐行 RIR 文件名）⇒ 逐行 RIR 溯源靠"池目录 == 名单"这一审计项（本文 §3-5），不是靠 manifest。

---

## 7. 交付清单

| 类别 | 路径 |
|---|---|
| 评测集 | `D:/data/aec-synth/eval_ns`（1400 条 + manifest + dt_plan_profile + PROVENANCE）、`D:/data/aec-synth/eval_rir`（同）、**`D:/data/aec-synth/eval_rir_sim`（RIR 轴配对对照集，同）** |
| RIR 池 | `D:/data/aec-synth/eval_rir_pool`（8 条 held-out 真实 IR 硬链接）、**`D:/data/aec-synth/eval_sim_rir_pool`（8 个仿真房间 + `sim_rir_ctrl_meta.json`）** |
| 清单 | `split_lists/rir_{train70,val15,holdout15}.txt`、`rir_split_manifest.csv`、`rir_split_meta.json`、**`rir_sim_ctrl.txt`**（配对对照池名单，README §7） |
| 脚本 | `scripts/split_rir_pool.py`（生成/验证/物化）、**`tools/make_sim_rir_ctrl_pool.py`（配对对照池生成 + 与训练池零重叠自检）**、`tests/test_split_rir_pool.py`（9 例）、`exp/07_nn_aec/eval_v8_audit.py`（参数化为通用轴集审计，默认语义不变） |
| 读数（t13 交接） | 用 t5 的驱动即可：`aecmos_eval.py --eval-dir <集> --trad|--nn-ckpt <v4>` ×2/集 + `eval.py --compare-trad`，再用 `exp/07_nn_aec/eval_v8_isolation.py` 的口径做 CI/DiD（须把 `--dir` 指向该集的产物目录）；**RIR 轴主读数 = eval_rir vs eval_rir_sim 的逐样本配对 Δ（B=2000 seed=777）** |

## 8. 未做 / 留待 t13 决策

1. **未做模型打分**：本任务范围 = 落地评测集 + 命令留档 + verify 跑通；v4/v5 读数属 t13 预研
   （打分口径必须沿用 AECMOS@48k + B=2000/seed=777；RIR 轴用**配对** bootstrap，其余非配对处如实标注）。
2. ~~未做"配对 RIR 轴"控制集~~ → **已做**（`eval_rir_sim`，队长 2026-10-13 裁定方案一落地，见 §1.1/§3-12~15）。
3. **未做"配对噪声轴"控制集**：同思路——用 `noise_train70` 里**同长度（95 文件）**的子集替换噪声名单、
   其余参数与 eval_ns 完全相同 ⇒ 可得到逐样本配对的噪声轴读数（当前 eval_ns 只能与 eval_v5 做两样本 DiD）。
4. **未补第二辆车**（见 §4-1）。
5. **NUMBERS.md 未改动**：两集尚无读数；t13 出读数后按 t6 的口径（先改表再同步文档）登记，
   集定义与哈希可直接引用本文 §1/§2 与各集 `PROVENANCE.json`。
6. **并发漂移提示（不属于本任务，供下一轮收档）**：本任务复跑全仓 pytest 为
   **361 项 = 357 passed + 4 xfailed**（含本任务新增 9 例），而 `docs/NUMBERS.md` §3 与 11 份文档
   记的是 t6 收档时点值 **321 项**——差额来自期间落地的 DM 化测试（`tests/test_nn_aec_dm.py` 20 例、
   `tests/test_cascade_dm.py` 11 例）。**单测计数需要一次刷新收档**（本任务未改 NUMBERS：其口径属
   收档职责，且这些测试的归属任务可能仍在收敛）。
