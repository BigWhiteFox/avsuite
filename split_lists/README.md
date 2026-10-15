# split_lists — 数据集切分清单（0-2 噪声池 + 0-4 语音池 reader + t16 真实 RIR 池三分割）

> 来源：`REMEDIATION_PLAN_V3.1.md` §2 工作项 **0-2 噪声池三分割**（消除 nn_findings #13：
> 训练噪声与评测噪声 99~100% 重叠，噪声维度没有"未见噪声"泛化面；实测见
> `../reports/refactor_wave5/data_inventory.md` §6 H2/H3）。
> 目录性质：**新增**（仓内此前无 `split_lists/`，本节建立该惯例）。清单随仓 scp 到训练服务器，
> 服务器只按清单取子集，不重做切分决策。

## 1. 本目录文件

> **数据缺口留痕**：LibriSpeech/MUSAN 未入库及替代口径见 [`DATA_GAP_20261013.md`](DATA_GAP_20261013.md)
> （含链路实测数字、互斥核对、将来补入接口）。

| 文件 | 内容 |
|---|---|
| `noise_train70.txt` | 训练可用噪声文件名（472 个），每行一个文件名（非路径） |
| `noise_val15.txt` | 验证可用噪声文件名（92 个） |
| `noise_eval15.txt` | 评测专用噪声文件名（95 个）——**训练侧一律不得使用** |
| `noise_split_manifest.csv` | 单一权威表：全池 659 行（`filename,split,bucket,hash8,sha256`） |
| `noise_split_meta.json` | 算法、阈值、计数、三份名单的内容 sha256（版本哨兵） |

格式约定：UTF-8 无 BOM、LF 换行；名单按文件名升序；文件名区分大小写。
**`noise_split_meta.json` 平台无关（t17 起）**：不含任何绝对路径，只留 `pool_dir_hint`
（形如 `<noise-pool>/noise`）作文档性提示；切分只吃**文件名**，名单内容与池路径无关。
池位置由验收方显式给出：`--noise-dir <池>` 或环境变量 `NOISE_POOL_DIR`；`--reproduce`
同样需要其中之一（池不可知时给出可操作提示，而不是拿写死的本机路径去猜）。
历史 meta 里的 `noise_dir` 字段已移除；旧 meta（若手上有）仍能被读作兜底。

## 2. 切分规则（可独立重算）

```
bucket = int(sha256(filename.encode("utf-8")).hexdigest()[:8], 16) % 1000
bucket <  700            -> train70
700 <= bucket < 850      -> val15
bucket >= 850            -> eval15
```

- 只吃**文件名**（basename），不吃路径、不吃文件内容、不吃随机种子、不吃扫盘顺序。
- 一个文件名只有一个 bucket ⇒ **train70 ∩ eval15 = ∅ 是构造性成立的**，不依赖抽签后校验。
- 池规模 659（`D:\data\dns-challenge-mirror\datasets\noise` 下递归 `*.wav`；仓内 `Readme.md` 不入池）。
- 实测计数 472 / 92 / 95（0.716 / 0.140 / 0.144；哈希二项抽样对 70/15/15 的正常抖动，
  差值绝对值 ≤2 个百分点）。

## 3. 复现与验收命令

```bash
# 生成（幂等：同池同脚本 => 逐字节同名单）
python scripts/split_noise_pool.py \
    --noise-dir D:/data/dns-challenge-mirror/datasets/noise \
    --out-dir split_lists

# 只看统计不落盘
python scripts/split_noise_pool.py --noise-dir <池> --dry-run

# 验收（退出码 0 = 通过；1 = 不通过；2 = 输入/环境错误）
python scripts/verify_split.py --dir split_lists \
    --noise-dir D:/data/dns-challenge-mirror/datasets/noise --reproduce

# 池尚未下载（服务器先核清单内部一致性）
python scripts/verify_split.py --dir split_lists --skip-pool-check
```

`verify_split.py` 的判定项：① train∩eval 零重叠 ② 三组两两互斥且并集 = 全池
③ 名单成员都在池内（且池内无漏网文件）④ 逐文件名哈希桶归属正确（名单非手挑）
⑤ 权威表与名单逐行一致 ⑥ 名单/权威表内容 sha256 与 meta 一致（事后编辑哨兵）
⑦ `--reproduce`：现场重跑生成器字节一致。

## 4. 历史口径（不改写既有事实，仅作对照）

- 0-2 之前，`train_v1` 的 3000 条用了池内 **652 个唯一噪声文件**
  （`train_nl` 用 589 个；并集 = 全池 659）。
- 该历史分布在 **训练/验证/评测之间不设隔离**：`eval_v5` 300 条中 297 条（99%）的噪声文件
  与训练同池（`data_inventory.md` §6 实测）。
- 本目录的名单是**配置**（以后怎么切），不是历史统计（过去用过什么）；历史数字不回填、
  也不据此改写任何既有 eval 集。隔离效果由 0-4（隔离版 eval_v8）用读数说话。
- 因此：旧的 `train_v1` 未按本清单取样，其历史数字仍按旧口径解读；**新训练/新评测集
  必须按本目录清单取样**（`noise_eval15.txt` 对训练侧为禁区）。

## 5. 边界

- 只新增文件：不改 `scripts/aec_synth.py` 等既有代码语义，不搬/不删任何 wav。
- 本清单不能单独封锁泄漏：训练脚本是否真的按清单取子集，由 0-3（near/far 互斥）与
  0-4（eval_v8 生成）落地后一并核对；`noise_eval15.txt` 的用途纪律靠训练侧命令显式传参保证。
- 换配比/换池需重跑生成器并**登记新版本**（`noise_split_meta.json` 会记录阈值与新哈希）；
  改阈值等于换一份清单，旧读数不可直接与新清单混用。

## 6. 语音池 reader 三分割 + eval 侧 near/far 互斥池（0-4，2026-10-13 追加）

> 来源：`REMEDIATION_PLAN_V3.1.md` §3.1（near 池 70% / far 池 70% 与 near 互斥 /
> held-out 15% 评测）+ §2 0-4（隔离版 eval_v8）。生成器 `scripts/split_speech_pool.py`
> （与 0-2 同样的"哈希名单"惯例；池路径不进名单）。

| 文件 | 内容 |
|---|---|
| `speech_readers_manifest.csv` | 权威表：全池 305 reader（`reader,bucket,split,side,n_files,file_list_sha256`） |
| `speech_readers_train70.txt` | 训练份额 reader（223 个）——阶段1 near/far 训练池从中取 |
| `speech_readers_val15.txt` | 模型选择份额 reader（37 个） |
| `speech_readers_holdout15.txt` | **评测专用** reader（45 个），训练侧禁区 |
| `speech_near_eval.txt` / `speech_far_eval.txt` | 评测侧 near/far 池文件名（993 / 599）——eval_v8 的两个 `--near-dir/--far-dir` |
| `speech_split_meta.json` | 算法/阈值/计数/各名单内容 sha256（版本哨兵） |

```
reader_bucket = int(sha256("reader:"+reader_id)[:8], 16) % 1000   # <700 train70 / 700-849 val15 / >=850 holdout15
side_bucket   = int(sha256("nearfar:"+reader_id)[:8], 16) % 1000  # holdout 内：<500 near_eval / >=500 far_eval
```

实测：`D:/data/dns-challenge-mirror/datasets/clean`（10000 wav / 305 reader，文件名
`..._reader_<id>_<k>.wav`）→ 223 / 37 / 45 reader；holdout15 的 45 个 reader 再分
near 993 wav / far 599 wav。**near 与 far 在 reader 级与文件级都零交集（构造性）**，
故 `aec_synth.py --nearfar-mutex utterance` 校验通过（0-3 的运行时判据）。

```bash
# 生成名单 + 物化池目录（硬链接；near/far 用 clean 池，noise 用 0-2 的 eval15 名单）
python scripts/split_speech_pool.py --clean-dir D:/data/dns-challenge-mirror/datasets/clean \
    --out-dir split_lists --materialize-root D:/data/aec-synth/eval_v8_pools \
    --noise-src D:/data/dns-challenge-mirror/datasets/noise \
    --noise-list split_lists/noise_eval15.txt

# 验收（退出码 0/1/2）：名单可重算一致 + near/far 零交集 + 成员在池 + 桶归属 + 池目录对账
python scripts/split_speech_pool.py --clean-dir <clean> --out-dir split_lists --verify \
    --materialize-root D:/data/aec-synth/eval_v8_pools --noise-src <noise> --noise-list <eval15 名单>
```

边界与历史口径：

- 本清单只说"以后怎么切"。历史 `train_v1`/`eval_v5`/`eval_v7` 的 near/far 都取自
  VoiceBank-DEMAND `train/clean`（28 说话人，与训练同源：eval_v5 的 near/far 唯一文件里
  各有 121/295、130/295 与 train_v1 **逐文件相同**）——该历史事实不回填、既有集不动。
- eval_v8 用 holdout15 reader 的 DNS clean 素材 + `noise_eval15` 噪声生成：与 train_v1/train_nl
  的语音语料**零交集**（报告实测 0/1592）；噪声维度对**将来**按本清单训练的运行是隔离的，
  但对已存在的 v4 ckpt **不隔离**（v4 的训练用过全池 659 中的 652 个，95 个 eval15 文件里
  94 个在 train_v1 用过）——这一条是报告必须披露的边界，见
  `../reports/eval_v8_isolation_20261013.md`。
- 305 个 reader 之外 DNS clean 池若有新增文件，需重跑本脚本并登记新版本（`speech_split_meta.json`
  的计数与内容哈希是版本哨兵）。

## 7. 真实 RIR 池三分割（按录音位置组互斥）+ 训练侧 near/far 互斥（t9/t16，2026-10-13 追加）

### 7.1 真实 RIR 三分割（`scripts/split_rir_pool.py`）

> 来源：`REMEDIATION_PLAN_V3.1.md` §3.2「IR64-CAR …按录音 session/位置分割，同车 32 条高度相关，
> 随机切 = 泄漏」+ §7「RIR 优先入 eval 轴（证据价值 > 训练价值）」。t16 用它落地 **eval_rir**（held-out 真实 RIR 轴）。

| 文件 | 内容 |
|---|---|
| `rir_train70.txt` | 训练份额 RIR（**5 个位置组 / 20 条**） |
| `rir_val15.txt` | 模型选择份额（**1 组 / 4 条**） |
| `rir_holdout15.txt` | **评测专用**（**2 组 / 8 条**：`IR64CAR_0_*`、`IR64CAR_315_*`），训练侧禁区 |
| `rir_split_manifest.csv` | 权威表：全池 32 行（`filename,group,session,split,rank,sha256,group_sha8`） |
| `rir_split_meta.json` | 算法/阈值/计数/各名单内容 sha256（版本哨兵） |
| `rir_sim_ctrl.txt` | **RIR 轴配对对照池名单（t16 追加，队长 2026-10-13 裁定方案一）**：8 个**仿真**房间（`tools/make_sim_rir_ctrl_pool.py --seed 20261017` → `D:/data/aec-synth/eval_sim_rir_pool`；与训练侧仿真池 60 条逐字节不重叠）——与 `rir_holdout15.txt`（8 条真实 IR）**同规模**，使 `eval_rir` vs `eval_rir_sim` 可做**逐样本配对**的 RIR 轴读数 |

```
group = 'ir64car_pos<angle>'（文件名命中 IR64CAR_<angle>）或上级目录名
rank  = 按 sha256("rir_group:"+group)[:8] 升序排名
n_hold = max(1, ceil(0.15·G))；n_val = max(1, round(0.15·G))；其余 train（同组文件不跨 split）
```

实测池 `D:/data/rir_real/pool_16k_mono`（32 条 = 8 声源方位 × 4 胶囊；16k mono，0.6 s）→
train 5 组 / val 1 组 / holdout 2 组。**组数少（G=8）时精确 70/15/15 不可达**，规则按 §7
"RIR 优先入 eval 轴"把非训练份额保底，实际 62.5 / 12.5 / 25（组级）。

```bash
# 生成 + 物化评测 RIR 池（硬链接）
python scripts/split_rir_pool.py --rir-dir D:/data/rir_real/pool_16k_mono --out-dir split_lists \
    --materialize-root D:/data/aec-synth/eval_rir_pool
# 验收（退出码 0/1/2）：名单可重算 + 组级互斥 + 覆盖完整 + 权威表/版本哈希 + 池目录对账
python scripts/split_rir_pool.py --rir-dir <池> --out-dir split_lists --verify \
    --materialize-root D:/data/aec-synth/eval_rir_pool
```

边界：**本池只有一辆车（一个 session）**——位置级互斥已做到，**车/录音级互斥做不到**，
eval_rir 衡量的是"未见声源位置"上的泛化，不是"未见车辆"。历史事实：`train_v1`/`train_nl`
的训练 RIR 用的是**仿真池** `D:/data/aec-synth/rir_pool`（60 条 pyroomacoustics；
manifest `rir_src=dir` 但目录是仿真池），真实 IR64-CAR 此前只被 `eval_v6_realrir`（评测，
用了全部 32 条）用过 ⇒ **eval_v6 与新的训练份额重叠 20 条**，引用 eval_v6 与 v5/eval_rir
对比时须注明 RIR 口径（见 `../reports/eval_ns_rir_20261013.md`）。

### 7.2 训练侧 near/far 互斥（t9，`speech_*_train70.txt`）

| 文件 | 内容 |
|---|---|
| `speech_near_train70.txt` / `speech_far_train70.txt` | 训练份额 223 reader 再按 `sha256('nearfar_train:'+reader)[:8]` 升序取前 111 → near（3477 文件）、其余 112 → far（3675 文件） |
| `speech_nearfar_train_meta.json` | 版本/规则/计数/零重叠核对与两份名单内容 sha256 |

零重叠核对（t9 实测）：near∩far=0、两端∩holdout15=0、并集 == train70（223 reader 无碰撞）；
本机全池 10000 = train70(7152) + val15(1256) + holdout15(1592)。
**纪律**：训练侧显式传这三份名单，`--nearfar-mutex utterance` 作为二次保险保留。

### 7.3 训练侧**文件名**清单（t17 正式化，`speech_*_train70_files.txt`）

reader 级清单（§7.2）是"每行一个 reader id"，而 `aec_synth.py --near-dir/--far-dir` 要的是
**文件名清单**。t11 D1 检查点期间 train-eng 临时展开过两份文件名清单（3477/3675，逐集核对通过）
但未入库；t17 把它们正式化为仓内产物：

| 文件 | 内容 |
|---|---|
| `speech_near_train70_files.txt` | near 训练池**文件名**清单（3477 行），sha256 前缀 `1d20d5563307` |
| `speech_far_train70_files.txt` | far 训练池文件名清单（3675 行），sha256 前缀 `82a475cd7e1e` |
| `speech_nearfar_train_meta.json` → `file_lists` 段 | 生成规则、计数、两份清单完整 sha256、四条断言 |

生成/校验（幂等、可独立重算）：

```bash
python scripts/build_speech_file_lists.py \
    --clean-dir D:/data/dns-challenge-mirror/datasets/clean --out-dir split_lists
python scripts/build_speech_file_lists.py --clean-dir <池> --out-dir split_lists --verify
```

- **生成规则**：reader 清单顺序 × 该 reader 内文件名升序。跨 reader 做全局字典序会得到
  **不同顺序**（同名集），故顺序固定于此并写进 meta；语义上只关心集合。
- **四条断言**（`--verify` 与生成时都跑，任一不成立退出码 1、不写盘）：
  A `near_files ∩ far_files = ∅`；B `near ∪ far == train70 全部文件`；
  C `(near ∪ far) ∩ holdout15 文件 = ∅`（评测轴互斥）；D reader 级并集 == train70。
- 与 train-eng 临时清单的关系：**集合完全相同**（故名次无关的核对一致）。
  ⚠ **但顺序确实不同**（两份文件**字节数相同**、逐行在第 0 行即分叉），因此**文件级 sha256 不同**：
  临时版 `456e5442…/e7db5e0a…` vs 正式版 `1d20d556…/82a475cd…`。t11 在训 run 的 manifest 钉的是
  临时版哈希，复核那次 run 请以 manifest 为准；引用正式版用本目录 meta 的 `file_lists.sha256`。
  精确差异与自查命令见 [`SPEECH_FILES_ORDER_NOTE.md`](SPEECH_FILES_ORDER_NOTE.md)
  ——该文件含**收档标准表述（队长 2026-10-13 裁定，t14 可直接引用）**：在训 run 以临时版哈希为准
  （`456e5442…/e7db5e0a…`）、后续 run 一律用正式版（`1d20d556…/82a475cd…`）、
  两者集合等价证据 = `sort` 后 sha256 相等。

## 8. 扩量语料评测池（t23：MUSAN / AISHELL-3 / LibriSpeech）与本目录新增名单

> 来源：t21 的 `ext_*` 三分割清单（LS/AISHELL/MUSAN 的 readers 与文件级份额；规则与 0-2/0-4 同族）
> + t23 的多轴评测集（队长 2026-10-13 裁定方案 A：**只读**从服务器拉所需子集到本机转码建池）。

**两级名单**（先"从服务器拉什么"，后"实际用在池里的"）：

| 文件 | 内容 | 条数 | sha256 前缀 |
|---|---|---|---|
| `ext_musan_eval15_subset150.txt` | MUSAN eval15 的**等间隔确定性子集**（拉取范围） | 150 | `07fc2a6bb0d2d693` |
| `ext_pool_musan_eval15_used.txt` | 实际噪声池文件（= 子集，16k wav） | 150 | `670fde5f1d25a707` |
| `ext_aishell3_far_eval_subset800.txt` / `ext_pool_zh_far_used.txt` | AISHELL-3 远端：拉取子集 / **按说话人拼接到 ≥10 s 的池段** | 800 / **214** | `00d294aa2acca208` / `048f3a707792150b` |
| `ext_aishell3_near_eval_subset800.txt` / `ext_pool_zh_near_used.txt` | AISHELL-3 近端：同上 | 800 / **239** | `e8c9908427622e8f` / `076b67ee036e0579` |
| `ext_ls_far_eval_subset800.txt` / `ext_pool_ls_far_used.txt` | LibriSpeech 远端 | 800 / 800 | `4896289adb6c3c83` / `70f0ed499b956362` |
| `ext_ls_near_eval_subset800.txt` / `ext_pool_ls_near_used.txt` | LibriSpeech 近端 | 800 / 800 | `98512ad910036027` / `754a26830ab6f451` |

- **抽样规则**：原始名单排序后等间隔取 `target` 个（`k = len // target`）⇒ 跨说话人分散、确定性可复现；
- **为什么 AISHELL-3 要拼接**：单句 ~3.2 s，8 s 窗口会被 `fit_len` 循环补（同一句重复 = 伪周期远端）；
  拼接按**说话人**分组、贪心到 ≥10 s（不跨说话人），池段 239/214 条全部 ≥10 s；
- **互斥核对**：LS 池 191 人 ∩ `ext_ls_readers_train70` = 0；AISHELL 池 17 人 ∩ `ext_aishell3_readers_train70` = 0；
  MUSAN 池 150 文件 ∩ `ext_musan_train70` = 0（t23 实测）；
- 生成/复现：`tools/pull_ext_subsets.py`（只读 tar 流，分批 ≤150）→ `tools/build_ext_eval_pools.py`
  （16k mono PCM_16 + `_manifests/*.csv`）；池本体在 `D:/data/aec-synth/ext_pools/`，原始拉取在 `D:/data/_t23_pull/`；
- 用它们生成的评测集与 t13 协议见 [`../reports/multi_axis_matrix_20261013.md`](../reports/multi_axis_matrix_20261013.md)；
  DT 清单 v2（SNR 域外两档）见 `../configs/README.md`。
