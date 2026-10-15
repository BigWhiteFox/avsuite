# eval_v8（隔离版）重生成 + v4 基线读数（阶段0 工作项 0-4）

- 日期：2026-10-13 ｜ 执行：eval-eng（t5）｜ 依据：`REMEDIATION_PLAN_V3.1.md` §2 0-4
  （消费 t1 的 0-2 噪声清单、t2 的 0-3 互斥约束、t4 的 0-5 DT 清单）
- 纪律：**不改任何既有 eval 集 / 历史读数**（eval_v3/v5/v6/v7 全部冻结）；本报告只新增数字与
  新冻结集；收档（NUMBERS.md / reports 索引）由 t6 执行，本文 §6 给出待登记行原文。
- 数据位置：`D:/data/aec-synth/eval_v8`（1400 条 × 3 槽 = 4200 wav，2.1 GB）
  ｜ 池目录：`D:/data/aec-synth/eval_v8_pools/{near,far,noise}`（硬链接，源池不动）

---

## 0. 结论摘要

1. **隔离版 eval_v8 已冻结**：1400 条（FEST 310 / NST 384 / DT 706，**每场景 ≥300**）、
   8 s@16k、`seed=20261014` + `--nearfar-mutex utterance` + `--dt-plan dt_plan_v1/eval_v5_repro`
   三处留痕齐全；准入审计 8 组全 PASS；**同命令重生成 4200/4200 wav + manifest + profile
   逐字节一致**（可复现性硬证据，强于"前 N 条前缀回放"）。
2. **三个隔离轴的实际达成度**：语音（说话人）轴 ✅（评测 near/far 取自 DNS clean
   holdout15 reader，与 train_v1/train_nl 语料 **0/1592 交集**）；near/far 互斥轴 ✅
   （reader 级 + 文件级零交集，构造性成立）；噪声轴 ⚠️ **对 v4 ckpt 未达成**——v4 的训练
   用过全池 659 个噪声里的 652 个，`noise_eval15` 的 95 个文件里 94 个在 train_v1 里出现过
   ⇒ 本报告对 v4 的差值**主要量化"说话人/语料"隔离**，噪声隔离要等按 0-2 清单训练的
   v5 代际才能测到（这正是 0-2 的目的）。
3. **绝对分下降 = 域偏移为主，不能直接当隔离偏差**：v4 的 FEST echo_mos 从
   eval_v5 的 3.881 掉到 2.896（−0.985），但**传统基线同期也掉 0.515**（3.518→3.003，
   传统没训过任何数据）⇒ 裸差值里大半是"语料从 VoiceBank train/clean 换成 DNS clean
   holdout reader"的域偏移。**用双重差分（DiD，把传统基线也变的部分减掉）才是干净的隔离读数**。
4. **DiD 结论（v4 ckpt）**：
   - **FEST echo 的"NN 优势"是泄漏口径产物**：eval_v5 上 NN−传统 = **+0.363**
     [+0.172, +0.558]（显著），eval_v8 上 = **−0.107** [−0.245, +0.024]（不显著），
     DiD = **−0.470** [−0.769, −0.161]。与 eval_v7（n=212，Δ=+0.052）方向一致 ⇒
     该"优势"随 n 扩大/隔离而消失，**不应再对外引用**。
   - **DT echo 优势不但保留、还略微扩大**：eval_v5 +0.806 → eval_v8 **+1.001**
     [+0.951, +1.049]，DiD **+0.195** [+0.069, +0.318]（显著为正）⇒ 这条优势**不是**泄漏
     造成的，隔离后更可信。
   - **DT deg_mos 落后原样保留**：eval_v5 −0.369 → eval_v8 **−0.364** [−0.404, −0.327]，
     DiD **+0.005** [−0.120, +0.134]（含 0）⇒ 与 `reports/deg_mechanism.md` 的
     "高频带选择性损伤 + 谱平坦化"机理一致：**是结构性的，不是泄漏**，0-1 的损失矩阵方向
     仍然成立。
   - **NST deg_mos 轻微领先**：eval_v8 **+0.041** [+0.005, +0.076]（CI 收窄后转为显著）
     ⇒ "NST 透传不再落后"在隔离集上成立。
5. **CI 随 n 扩档收窄**（沿用 v7 口径，每场景 ≥300）：DT echo ±0.045→**±0.025**（0.55×）、
   DT deg ±0.088→**±0.040**（0.46×）、FEST echo ±0.183→**±0.122**（0.67×）、
   NST deg ±0.102→**±0.040**（0.39×）。
6. **对 DP1 的含义**：隔离不推翻 NN 的两条主结论（DT echo 优势 / DT deg 缺口）；
   推翻的是 **FEST echo 优势**与"评测数字乐观"的部分来源。数据扩充的紧迫性由 0-1
   损失矩阵读数决定，与本文的隔离偏差量级相互独立。

---

## 1. 数据集：配方、隔离轴与准入证据

### 1.1 配方（与 eval_v5/v7 **除隔离轴外逐字同口径**）

| 参数 | eval_v5 | eval_v7 | **eval_v8（本文）** |
|---|---|---|---|
| num / 分层 | 300（70/76/154） | 800（212/182/406） | **1400（FEST 310 / NST 384 / DT 706）** |
| seed | 20261011 | 20261013 | **20261014**（仓内未占用） |
| dur / scenarios / snr / soft-clip / leadin | 8 / 0.25,0.25,0.5 / 20–30 / 0 / 2.0 | 同左 | 同左（DT 三参数由 `dt_plan_v1` 的 `eval_v5_repro` preset 显式给出） |
| 语音语料 | VoiceBank-DEMAND `train/clean`（28 说话人，**与训练同源**） | 同左 | **DNS-Challenge clean 的 holdout15 reader（45 人，与训练零交集）** |
| near/far 池 | 同一目录（near=far 池） | 同左 | **两个互斥池**（near 993 wav / far 599 wav，reader 级 + 文件级零交集） |
| 噪声 | 训练同池（239 个唯一文件，100% 训练见过） | 同池（473 个） | **`split_lists/noise_eval15.txt`（95 个，训练侧禁区）** |
| RIR | pyroomacoustics 仿真（无 `--rir-dir`） | 同左 | **同左**（刻意不用训练侧 `rir_pool`；真实 RIR 属 eval-rir 轴，见 §4） |
| 时延/漂移 | 10–80 ms / 无 | 同左 | 同左 |

生成命令（实际执行，逐字）：

```powershell
python scripts/aec_synth.py --num 1400 --dur-sec 8 `
  --out-dir D:/data/aec-synth/eval_v8 `
  --near-dir D:/data/aec-synth/eval_v8_pools/near `
  --far-dir  D:/data/aec-synth/eval_v8_pools/far `
  --noise-dir D:/data/aec-synth/eval_v8_pools/noise `
  --snr-range 20,30 --soft-clip-prob 0 --seed 20261014 `
  --nearfar-mutex utterance `
  --dt-plan configs/dt_plan_v1.json --dt-plan-preset eval_v5_repro
```

留痕（manifest 列 + profile）：
`nearfar_mutex=utterance`（1400/1400）、`dt_plan_version=dt_plan_v1`、
`dt_plan_preset=eval_v5_repro`；`dt_plan_profile.json` 记 `seed=20261014`、`num=1400`、
`dur_sec=8.0`、`pool_sizes={near:993, far:599, noise:95, rir:0}`、
清单 sha256 = `73496c0bc8778432d8f18694d012dccb598d0306f0983bd08e74460440062ef4`
（= `configs/dt_plan_v1.json` 实际文件哈希，审计核对一致）。

池由 `scripts/split_speech_pool.py` 生成（与 0-2 同一"哈希名单"惯例）：

```
reader_bucket = int(sha256("reader:"+reader_id)[:8],16) % 1000  → <700 train70 / 700-849 val15 / >=850 holdout15
side_bucket   = int(sha256("nearfar:"+reader_id)[:8],16) % 1000 → holdout 内 <500 near_eval / >=500 far_eval
```

实测 305 reader → 223/37/45；holdout15 的 45 人再分 near 993 / far 599 个 wav。
产物：`split_lists/speech_{readers_train70,readers_val15,readers_holdout15,near_eval,far_eval}.txt`
+ `speech_readers_manifest.csv` + `speech_split_meta.json`（README §6 有格式与复现命令）。

### 1.2 准入审计（`exp/07_nn_aec/eval_v8_audit.py` → PASS，退出码 0）

| 组 | 判定 | 实测 |
|---|---|---|
| A 规模/分层 | 每场景 ≥300（§3.2）、占比不越 3σ | 310/384/706（目标 350/350/700，均在 3σ 内） |
| B 留痕/清单闭环 | 三列齐全 + profile sha256 对得上随仓清单 | ✅（见 §1.1） |
| C 隔离 | 行内 near/far/noise ∈ 对应池名单；池与 train_v1/train_nl/train_v1_drift 语料零交集 | 1400/1400 行合规；交集 **0/1592**；池重算一致 |
| D 无自回声 | 行内 near≠far、无同名 utterance | **0 行** |
| E DT 口径 | DT ser∈[0,10]、snr∈[20,30]、leadin≡2.0；FEST/NST leadin=0、snr∈[20,30] | ser [0.015, 9.973]、snr [20.016, 29.998]、leadin 706/706 | 
| F 其它参数 | delay∈[10,80]、soft_clip≡0、rir_src≡sim | delay [10.051, 79.965]、0、sim |
| G 产物完整 | 三槽各 1400、8 s@16k | 1400/1400/1400 |
| H 泄漏披露 | 必须入档的数字（不是判定项） | `noise_eval15` 95/95 个文件被历史训练用过（train_v1 94 个）；行内用到 95 个唯一噪声 |

### 1.3 可复现性与既有清单回归

| 证据 | 命令 | 结果 |
|---|---|---|
| 全量重生成逐字节一致 | 同 §1.1 命令写到 `eval_v8_regen/`，再逐文件 sha256 比对 | **manifest / profile 相同；4200/4200 wav 相同**（`outputs/eval_v8/regen_identity.json`；比对后已删临时集） |
| 0-2 噪声清单未被改动 | `python scripts/verify_split.py --dir split_lists --noise-dir <noise> --reproduce` | **PASS，退出码 0**（train∩eval=0，覆盖/桶归属/权威表/版本哈希全通过） |
| 0-4 语音池验收 | `python scripts/split_speech_pool.py --clean-dir <clean> --out-dir split_lists --verify --materialize-root <pools> --noise-src <noise> --noise-list split_lists/noise_eval15.txt` | **PASS，退出码 0**（名单可重算一致 + reader/near-far 零交集 + 成员在池 + 桶归属 + 池目录对账） |
| 0-5 DT 清单闭环 | `python scripts/verify_dt_plan.py --plan configs/dt_plan_v1.json --manifest D:/data/aec-synth/eval_v8/manifest.csv --set-dir … --strict-preset --replay-prefix 50`（带 `DT_PLAN_NEAR_DIR/FAR_DIR/NOISE_DIR`） | **A/B/D/E/F 通过**；C 706+4 项、G 1 项失败 —— 见下方说明（**验证器只建模 `--rir-dir`，对 sim-RIR 集不适用**，非数据缺陷） |
| 0-3 约束反向对照 | 故意把 `--far-dir` 指成 near 池再跑同一条命令 | **退出码 1、输出目录未创建**（fail fast，报"同一目录 + 993 个同名 utterance"）；证明确实是约束通过后才产数据 |

> **关于 `verify_dt_plan.py` 的 C/G 两项（如实登记，建议 t4 补一个说明或开关）**：
> `scripts/aec_synth.py` 在**不给 `--rir-dir`** 时走 pyroomacoustics 仿真，`rir_simulate_pra`
> 每次调用消耗**可变条数**的 `uniform(...)`（房尺寸/麦位/源位 while 循环/吸声），且会用到
> `rng.uniform(..., size=3)`；而 `verify_dt_plan.py:164` 的随机流模型只按"`--rir-dir` 时一次
> `integers(n)`"推进，`:417` 的 `LogRng.uniform` 也没有 `size=` 关键字 ⇒ C（同分布回放）与
> G（前缀回放）在 sim-RIR 集上必然失败（报错即 `TypeError: LogRng.uniform() got an unexpected
> keyword argument 'size'`）。eval_v5/v7 与本次 eval_v8 **同属 sim-RIR**，历史集同样无法用 C/G 核。
> 本报告采用的替代证据是**第 1 行那条更强的"全量重生成逐字节一致"**（覆盖 1400 条而不是
> 前缀 50 条）；`--dt-plan` 的留痕闭环（B）、区间合规（D/E/F）已通过，C/G 的缺口是**验证器
> 建模范围**问题，不是 eval_v8 的数据问题。

---

## 2. 读数：v4 ckpt 在 eval_v8 vs eval_v5（同 ckpt / 同批 / 同 AECMOS@48k 口径）

- ckpt：`exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt`（v4 主配置，**未重训**）
- 传统基线：`pipeline.TradAEC(method="mdf", partitions=8, mu=0.7)`（仓内默认）
- 统计：B=2000、seed=777、百分位法（与 `tools/compute_ci.py` 同纪律）；
  **集内 NN−传统是逐文件配对** bootstrap，**跨集差是两个不同样本集的两样本（非配对）** bootstrap
- 产物：`exp/07_nn_aec/outputs/eval_v8/{isolation_ci.json, isolation_tables.md, aecmos_*.json,
  nn_metrics_v8/, nn_metrics_v5rerun/, audit_v8.json, regen_identity.json}`
- 交叉核对：**eval_v5 的本次重跑与仓内权威 @48k 产物逐行一致** —— NN 300/300 行、600/600 个值
  `max_abs_diff = 0.0`；trad 299/300 行一致（1 行 `deg_mos` 差 0.137，集均值差 ≤0.002，
  属单行偶发，不影响结论），故本文的 v5 参照与历史 `aecmos48_v5fix/eval_v5/*` 同源可对表。

### 2.1 绝对分（NN = v4 ckpt，AECMOS@48k）

| 集 | 场景 | 指标 | n | mean | 95%CI | v5 同格 |
|---|---|---|---|---|---|---|
| eval_v8 | FEST | echo_mos | 310 | **2.896** | ±0.122 | 3.881±0.183 |
| eval_v8 | DT | echo_mos | 706 | **4.106** | ±0.025 | 4.248±0.045 |
| eval_v8 | DT | deg_mos | 706 | **3.458** | ±0.040 | 3.220±0.088 |
| eval_v8 | NST | deg_mos | 384 | **4.186** | ±0.040 | 3.827±0.102 |
| eval_v5 | FEST | echo_mos | 70 | 3.881 | ±0.183 | — |
| eval_v5 | DT | echo_mos | 154 | 4.248 | ±0.045 | — |
| eval_v5 | DT | deg_mos | 154 | 3.220 | ±0.088 | — |
| eval_v5 | NST | deg_mos | 76 | 3.827 | ±0.102 | — |

> FEST/NST 的 `deg_mos` 与 NST `echo_mos` 在 5.0 附近**饱和**（sd≈0），其"CI 不含 0"是退化情形、
> 不含信息量，判读只看 DT 两个指标与 FEST echo。上表隐含 CI 收窄：0.39–0.67×（见 §0.5）。

### 2.2 集内配对差（NN − 传统，逐文件配对）

| 集 | 场景.指标 | n | Δ | 95%CI | 判定 |
|---|---|---|---|---|---|
| eval_v8 | FEST.echo_mos | 310 | **−0.107** | [−0.245, +0.024] | 不显著 |
| eval_v8 | DT.echo_mos | 706 | **+1.001** | [+0.951, +1.049] | 显著为正 |
| eval_v8 | DT.deg_mos | 706 | **−0.364** | [−0.404, −0.327] | 显著为负 |
| eval_v8 | NST.deg_mos | 384 | **+0.041** | [+0.005, +0.076] | 显著为正 |
| eval_v5 | FEST.echo_mos | 70 | +0.363 | [+0.172, +0.558] | 显著为正 |
| eval_v5 | DT.echo_mos | 154 | +0.806 | [+0.723, +0.894] | 显著为正 |
| eval_v5 | DT.deg_mos | 154 | −0.369 | [−0.460, −0.279] | 显著为负 |
| eval_v5 | NST.deg_mos | 76 | +0.052 | [−0.035, +0.141] | 不显著 |

### 2.3 跨集差（eval_v8 − eval_v5）与双重差分

| 场景.指标 | NN Δ(v8−v5) | 传统 Δ(v8−v5) | **DiD**（=NNΔ−传统Δ） | DiD 95%CI | 判读 |
|---|---|---|---|---|---|
| FEST.echo_mos | −0.985 [−1.200, −0.757] | −0.515 [−0.713, −0.308] | **−0.470** | [−0.769, −0.161] | NN 相对传统**变差**：v5 的 FEST echo 优势不成立 |
| DT.echo_mos | −0.143 [−0.190, −0.091] | −0.338 [−0.449, −0.219] | **+0.195** | [+0.069, +0.318] | NN 优势**扩大**：不是泄漏 |
| DT.deg_mos | +0.237 [+0.145, +0.331] | +0.233 [+0.144, +0.324] | **+0.005** | [−0.120, +0.134] | 缺口**原样保留**：结构性 |
| NST.deg_mos | +0.360 [+0.255, +0.471] | +0.371 [+0.288, +0.459] | **−0.011** | [−0.155, +0.135] | 持平（轻微领先保留） |

> 这张表就是"隔离性偏差"的量化：裸跨集差（第 1 列）几乎全被域偏移占掉，**只有 DiD 列是
> 模型特有的变化**。传统基线没训练过任何数据，它在 v8 上的变化只能来自语料/噪声子集/池的
> 变化 ⇒ 用作域偏移的对照臂。

### 2.4 波形域指标（`eval.py --compare-trad`，同批同 ckpt；trad 无逐行落盘，故只有集均值）

| 指标 | trad@v8 | NN@v8 | Δ@v8 | trad@v5 | NN@v5 | Δ@v5 | DiD(Δ@v8−Δ@v5) |
|---|---|---|---|---|---|---|---|
| FEST ERLE (dB) | 19.91 | 65.56 | +45.65 | 20.18 | 69.84 | +49.66 | −4.01 dB |
| DT PESQ | 1.563 | 1.313 | −0.250 | 1.678 | 1.455 | −0.223 | −0.027 |
| DT STOI | 0.906 | 0.851 | −0.055 | 0.798 | 0.751 | −0.047 | −0.008 |
| NST PESQ | 2.797 | 2.344 | **−0.453** | 2.459 | 2.381 | −0.078 | **−0.375** |

- NN 侧跨集差（带 CI，`isolation_tables.md` §5）：DT PESQ −0.142 [−0.182, −0.105]、
  DT STOI **+0.100** [+0.084, +0.116]、NST PESQ −0.037 [−0.157, +0.090]（不显著）、
  NST STOI +0.094、FEST ERLE −4.28 dB —— **DT/NST STOI 在隔离集上反而更高**，进一步说明
  裸绝对分变化混着域难易度。
- **一个值得注意的新信息**：传统基线在 NST PESQ 上从 2.459 涨到 2.797（+0.338），NN 几乎不动
  （2.381→2.344）⇒ NN 与传统的 NST PESQ 差从 −0.078 拉到 **−0.453**。也就是说，AECMOS-deg
  上的"NST 不再落后"**没有得到波形域 PESQ 的支持**（NN 的"过干净"输出在 DNS clean 语料上
  被 PESQ 判为失真更重）。这条只作记录，判读留给后续（与 `reports/deg_mechanism.md` 同源机理）。
- FEST ERLE 的绝对值是**掩码门控≈静音**的口径产物（仓内既有纪律），不可与线性 AEC 的 ERLE 直接比。

### 2.5 跨集一致性（v5 / v7 / v8 的 NN−传统 Δ；v7 取仓内 @48k 权威产物）

| 场景.指标 | eval_v5 (n=300) | eval_v7 (n=800) | **eval_v8 (n=1400)** | 结论 |
|---|---|---|---|---|
| FEST echo | +0.363（显著） | +0.052（不显著） | **−0.107（不显著）** | 优势随 n/隔离消失 ⇒ **不成立** |
| DT echo | +0.806 | +0.799 | **+1.001** | 稳定且隔离后更大 ⇒ 成立 |
| DT deg | −0.369 | −0.321 | **−0.364** | 三集一致 ⇒ 结构性缺口 |
| NST deg | +0.052（n.s.） | +0.047 | **+0.041（显著）** | 轻微领先，隔离开不翻转 |

---

## 3. 归因与读数怎么用

1. **"预期绝对分下降"只对了一半**：echo_mos 确实下降（FEST −0.99 / DT −0.14），但 deg_mos
   在 v8 上**升高**（DT +0.24 / NST +0.36），且传统基线同向同幅 ⇒ 这是**域难易度**，不是隔离
   偏差。凡引用隔离偏差，用 §2.3 的 DiD 列，不要用裸差值。
2. **泄漏确实真实存在，但不在主结论上**：v5 的 FEST echo "+0.363 优势"在隔离集上消失
   （DiD −0.470）且与 v7（+0.052）同向 —— 这条应作为"泄漏口径产物"撤回。DT echo 优势与
   DT deg 缺口则**经受住了隔离**，说明此前把 3.3h 数据的泛化结论保守解读是对的，但两条主
   结论本身是稳的。
3. **v4 基线在 eval_v8 上的数字可以对外引用**，但必须带三件事：ckpt 路径、`eval_v8` 定义
   （1400 条 / seed 20261014 / 隔离三轴）、CI。**不要把 v8 的数字和 v5 的数字混在同一行比较
   而不写 DiD**。
4. **对阶段1 的直接用途**：v5 代际重训后，同一 eval_v8 上做"v5 vs v4"的代际对照时，
   本报告的 v4 行就是锚；同时因为 v5 将按 `noise_train70` 训练，**eval_v8 对 v5 才真正实现
   噪声轴隔离**（对 v4 不是），这一点在代际对照里要写清。

---

## 4. 边界与已知局限（如实登记）

1. **噪声轴对 v4 未隔离**（最重要的一条）：`noise_eval15` 95 个文件全部被历史训练用过
   （train_v1 94、train_nl/train_v1_drift 另 1）——因为历史 `train_v1` 用掉了全池 659 中的 652 个。
   所以 §2.3 的 DiD 量化的是**说话人/语料隔离 + 噪声子集替换**的合成效应，其中只有说话人
   部分是"泄漏"；隔离测到的那部分（DiD）已是可用上界性质的读数，但**不能声称"v4 的噪声泄漏
   已被量化"**。
2. **语料切换与说话人隔离开绑**：eval_v8 的评测语料是 DNS clean（holdout reader），eval_v5 是
   VoiceBank train/clean。两者既是"未见说话人"也是"换语料库"，本文用传统基线臂 + DiD 把
   **域偏移**减掉，但无法进一步分离"DNS clean 的域"与"说话人"两个分量（建议后续加
   一个 VoiceBank test/clean（p232/p257 未见说话人）控制集，见 §7）。
3. **跨集不是逐文件配对**：两个集是不同样本，§2.3 的 CI 是两样本 bootstrap（独立性假设），
   不能读成"同一条语音上的差值"。集内 NN−传统才是逐文件配对。
4. **RIR 轴刻意不动**：v8 与 v5/v7 一样用 pyroomacoustics 现场仿真（**不用**训练侧 `rir_pool`，
   也不用真实 RIR 池）。真实 RIR 是独立的 eval-rir 轴（`reports/realrir_domain.md`），
   不在本报告范围；若把它塞进 v8，DiD 就不再只反映说话人/噪声隔离。
5. **AECMOS 口径**：8 s 短片段全段送评（含 2 s DT 引导），AECMOS@48k 模型协议，与
   eval_v5/v7 一致；官方比赛口径（裁剪送评）绝对值不可直接对照。
6. **饱和指标**：FEST/NST 的 deg_mos 与 NST echo_mos 在 5.0 附近饱和，其显著性判定退化；
   本报告不据此下结论。
7. **`verify_dt_plan.py` 的 C/G 在 sim-RIR 集上不适用**（§1.3 已详述），替代证据为全量
   重生成逐字节一致；建议 t4 在脚本 docstring/选项里显式声明这一条。
8. **诚实性声明**：本报告的 v5 参照来自本次同批重跑（与仓内权威产物逐行核对一致），
   没有改写任何历史读数或既有 eval 集；`eval_v8` 是新冻结集，seed 固死并已留痕。

---

## 5. 复现命令（从仓库根 `avsuite/` 执行）

```powershell
$py='C:\Users\14805\.conda\envs\avsuite\python.exe'

# 1) 语音池 reader 三分割 + eval near/far 互斥池 + 物化（0-4；0-2 名单已随仓）
& $py scripts/split_speech_pool.py --clean-dir D:/data/dns-challenge-mirror/datasets/clean `
    --out-dir split_lists --materialize-root D:/data/aec-synth/eval_v8_pools `
    --noise-src D:/data/dns-challenge-mirror/datasets/noise --noise-list split_lists/noise_eval15.txt

# 2) 生成隔离版 eval_v8（命令见 §1.1）

# 3) 准入审计（8 组，退出码 0）
& $py exp/07_nn_aec/eval_v8_audit.py --json exp/07_nn_aec/outputs/eval_v8/audit_v8.json

# 4) 打分（同 ckpt 同批；每条约 6–10 min/集，AECMOS 的 trad 最慢）
$ck = "exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt"
foreach ($t in @(@('v8','D:/data/aec-synth/eval_v8'), @('v5rerun','D:/data/aec-synth/eval_v5'))) {
  & $py exp/07_nn_aec/aecmos_eval.py --eval-dir $t[1] --trad --out "exp/07_nn_aec/outputs/eval_v8/aecmos_trad_$($t[0]).json"
  & $py exp/07_nn_aec/aecmos_eval.py --eval-dir $t[1] --nn-ckpt $ck --out "exp/07_nn_aec/outputs/eval_v8/aecmos_nn_best_$($t[0]).json"
  & $py exp/07_nn_aec/eval.py --eval-dir $t[1] --ckpt $ck --compare-trad --out "exp/07_nn_aec/outputs/eval_v8/nn_metrics_$($t[0])"
}

# 5) CI / 差值 / DiD（B=2000 seed=777）
& $py exp/07_nn_aec/eval_v8_isolation.py
```

---

## 6. 给 t6 的 NUMBERS.md 待登记行（原文照抄即可，建议放在 AK 节"eval_v8 隔离版（0-4）"）

| 项 | 值 |
|---|---|
| eval_v8 定义 | 1400 条（FEST 310 / NST 384 / DT 706），8 s@16k，seed 20261014；`dt_plan_v1/eval_v5_repro`；near/far = DNS clean holdout15 reader 互斥池（993/599）；noise = `noise_eval15`（95）；RIR = pyroomacoustics 仿真；`--nearfar-mutex utterance`；命令见 `reports/eval_v8_isolation_20261013.md` §5 |
| v4 ckpt DT echo_mos @eval_v8 | **4.106**（±0.025，n=706）；传统基线 3.105；Δ **+1.001** [+0.951, +1.049] |
| v4 ckpt DT deg_mos @eval_v8 | **3.458**（±0.040）；传统基线 3.822；Δ **−0.364** [−0.404, −0.327]（**隔离后仍落后**） |
| v4 ckpt FEST echo_mos @eval_v8 | 2.896（±0.122）；传统 3.003；Δ **−0.107** [−0.245, +0.024]（不显著） |
| v4 ckpt NST deg_mos @eval_v8 | 4.186（±0.040）；传统 4.145；Δ **+0.041** [+0.005, +0.076] |
| 隔离性偏差（DiD，v8−v5） | FEST echo **−0.470** [−0.769, −0.161]（v5 的 +0.363 优势=泄漏口径产物，撤回）；DT echo **+0.195** [+0.069, +0.318]；DT deg **+0.005** [−0.120, +0.134]（缺口结构性） |
| 边界（必须同页写） | ① eval15 的 95 个噪声文件全部被历史训练用过 ⇒ **对 v4 噪声轴未隔离**，DiD 主要量化说话人/语料隔离；② 跨集差为两样本非配对 bootstrap；③ **"预期绝对分下降"的修正认知**：域偏移同时压低传统基线（FEST echo 传统 −0.515 / NN −0.985），裸绝对分差**不能**当隔离偏差引用，只有 DiD 列可作隔离性偏差；④ 8 s 全段送评 AECMOS@48k 口径 |

> **收档口径（队长 2026-10-13 裁定，三条强制）**：
> ① **FEST echo 的 v5 "NN 优势"（+0.363）确认为泄漏口径产物**：旧结论**不删除**，按数据口径标注、
> 撤回对外表述 —— 写法："v5（泄漏口径）NN−传统 +0.363 显著 → v8（隔离口径）−0.107 不显著，
> DiD −0.470 [−0.769, −0.161]"。
> ② **"噪声轴对 v4 不可隔离"**（`noise_eval15` 95/95 个文件被历史训练用过）作为 DiD 引用的
> **强制附带说明**写入。
> ③ **"预期绝对分下降"的修正认知**一并入收档：绝对分下降有一部分是域偏移（传统基线同向同幅下降），
> **只有 DiD 列可作隔离偏差引用**。

> 索引登记：`reports/README.md`「当前有效」表新增一行
> （`eval_v8_isolation_20261013.md` ｜ 隔离版 eval_v8 重生成 + v4 基线/隔离偏差 ｜ 0-4 产出）。

## 7. 下一步建议（不在本任务范围）

1. **把 FEST echo 从对外话术里撤下**（v7/v8 两集都不支持 NN 优势），改引 DT echo 与 NST deg。
2. 需要进一步分离"域 vs 说话人"时，加一个 **VoiceBank test/clean（p232/p257，未见说话人）
   控制集**（同 DT 清单、同 mutex、noise_eval15；n≥300/场景），即可把 §2.3 的 DiD 再拆一层。
3. 阶段1 v5 代际训练**必须显式传 `noise_train70` 与 near/far 训练份额**（否则 eval_v8 对 v5 的
   噪声隔离不成立）——训练命令留痕要求见 `split_lists/README.md` §4/§6。
4. t4 侧建议给 `verify_dt_plan.py` 增加"sim-RIR 集 → C/G 不适用"的显式说明或 `--skip-sim-rir` 开关，
   并把 eval_v5/v7/v8 一起补进验收矩阵。
