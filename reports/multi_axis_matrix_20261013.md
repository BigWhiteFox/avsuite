# 多轴评测集矩阵（t23）：音乐噪声 / 中文远端 / SNR 域外 / 长语音 + t13 评测协议矩阵

- 日期：2026-10-13 ｜ 执行：eval-eng（t23，P1X-3）｜ 依据：用户授权"更多情况/范围/种类" +
  `REMEDIATION_PLAN_V3.1.md` §3.2（评测轴互斥且冻结）+ 队长两次裁定（方案 A 只读拉取；AISHELL-4 阻断维持 t21 口径）
- 上游：**t21**（服务器扩量语料 + `ext_*` 三分割清单）、**t16**（eval_ns/eval_rir 与通用轴集审计）、
  **t5**（eval_v8 机具）、**t4/t7**（dt_plan 与 verify_dt_plan）、**t9**（服务器与服务纪律）
- 纪律：只新增评测集与清单；既有集（eval_v5/v6/v7/v8/ns/rir）全部冻结不动；未改 NUMBERS 已收档数字
  （新集尚无模型读数，读数属 t13；单测计数刷新按队长裁定归 t14）。

---

## 0. 结论摘要

1. **新增 7 套冻结评测集**（各 1400 条 / 每场景 ≥300）：`eval_snr_low`、`eval_snr_high`（③ SNR 域外两档）、
   `eval_music`（① MUSAN 音乐/宽带噪声）、`eval_zh` + `eval_zh_both`（② 中文远端 / 双端中文）、
   `eval_far_ls` + `eval_long_ls`（④ LS 长语音远端，8 s / 12 s）。
   全部 `seed` 固死、`--nearfar-mutex utterance`、`dt_plan` 清单留痕三列齐全、每集带 `PROVENANCE.json`。
2. **验收全绿（"audit + verify 全绿才冻结"）**：7 集准入审计 **PASS**；`verify_dt_plan` 的
   清单结构 / 留痕闭环 / 同分布核对（C）/ 区间合规（D/E/F）/ 前缀逐字节回放（G）**PASS exit 0**；
   抽一套（`eval_zh_both`）做**全量重生成逐字节核对：4200/4200 wav + manifest + profile 全同**。
3. **数据来源与互斥**：扩量语料只在服务器（musan 12G / librispeech 29G / aishell3 26G）⇒ 按队长方案 A
   **只用只读 tar 流拉回所需子集**（3350 文件 / 1.9 GB，分批 ≤150），本机转码建池；池与各语料**训练份额零交集**
   （LS 191 人 ∩ train70 = 0、AISHELL 17 人 ∩ train70 = 0、MUSAN 150 文件 ∩ train70 = 0）。
4. **AISHELL-4 阻断的替代**：维持 t21 裁定（parquet + 无 pyarrow），② 的中文远端改用 **AISHELL-3** holdout
   （22 人）；并额外做了**双端中文**集 `eval_zh_both`。
5. **一处口径修正（已解决）**：AISHELL-3 单句 ~3.2 s，直接进 8 s 窗会被 `fit_len` 循环补 ⇒ 伪周期远端。
   建池改为**按说话人拼接至 ≥10 s**（239/214 段，全部 ≥10 s）后重生成两集 ⇒ 无补环。
6. **RIR 污染发现 → 已处置（净池重训，队长 2026-10-13 裁定）**：本任务发现 P1-5 训练用
   `--rir-dir /root/data/rirs`（42 条 = 全部 32 条车厢 IR），其中含 t16 划给评测的 8 条 holdout
   ⇒ 当时代际的 `eval_rir` 轴不成立。**train-eng 已处置**：停掉污染 run → 建 `/root/data/rirs_train70`
   （`rir_train70.txt` 的 **20 条**硬链接，顺带排除 8 条不在任何名单的裸名 48k 文件）→
   NN-AEC/DCCRN/级联**全部净池重训**（自动链 ~6h）；**两个污染 run 归档 `rir_contaminated`（附 CONTAMINATION.md），
   不采信**。⇒ **v5 代际最终口径 = 训练 RIR 为 `rir_train70` 20 条（净池重训版），`eval_rir` 的 8 条 holdout
   零泄漏**；t13 等净池 v5 三模型完训后照 §4 矩阵执行。

---

## 1. 评测集矩阵总表（12 套冻结集）

| 集 | n（FEST/NST/DT） | seed | 清单 | 说话人 | 噪声 | RIR | dur | manifest sha256 |
|---|---|---|---|---|---|---|---|---|
| `eval_v5`（历史锚） | 300（70/76/154） | 20261011 | 无（早于 0-5） | VoiceBank train（与训练同源） | DNS 池（训练同源） | sim | 8 s | `8c5625cae31db621` |
| `eval_v8`（0-4 隔离） | 1400（310/384/706） | 20261014 | v1/eval_v5_repro | DNS holdout15（45 人） | `noise_eval15`(95) | sim | 8 s | `a4f9044677b67128` |
| `eval_ns`（t16 噪声轴） | 1400（352/324/724） | 20261015 | v1/eval_v5_repro | 同 v8 | `noise_eval15`(95) | sim | 8 s | `fedd3d63852cb172` |
| `eval_rir`（t16 RIR 轴） | 1400（323/363/714） | 20261016 | v1/eval_v5_repro | 同 v8 | `noise_eval15`(95) | **真实车厢 held-out 8 条** | 8 s | `e3a7a3d949575c95` |
| `eval_rir_sim`（配对对照） | 1400（323/363/714） | 20261016 | v1/eval_v5_repro | 同 v8 | `noise_eval15`(95) | 仿真 8 房间（同规模） | 8 s | `e3a7a3d949575c95` |
| **`eval_snr_low`（③）** | 1400（335/347/718） | 20261018 | **v2/eval_snr_low** | 同 v8 | `noise_eval15`(95) | sim | 8 s | `7426a3eaab3129cf` |
| **`eval_snr_high`（③）** | 1400（345/351/704） | 20261019 | **v2/eval_snr_high** | 同 v8 | `noise_eval15`(95) | sim | 8 s | `2c220ae89d3d5695` |
| **`eval_music`（①）** | 1400（359/353/688） | 20261020 | v1/eval_v5_repro | 同 v8 | **MUSAN eval15 子集 150** | sim | 8 s | `4148e11f2499e8f4` |
| **`eval_zh`（②远端中文）** | 1400（335/326/739） | 20261021 | v1/eval_v5_repro | DNS holdout15 | `noise_eval15`(95) | sim | 8 s | `a7e432c154ecf621` |
| **`eval_zh_both`（②双端中文）** | 1400（347/361/692） | 20261022 | v1/eval_v5_repro | **AISHELL-3 holdout 拼接 239 段** | `noise_eval15`(95) | sim | 8 s | `4eb0e0d26ed5d95b` |
| **`eval_far_ls`（④）** | 1400（353/361/686） | 20261023 | v1/eval_v5_repro | DNS holdout15 | `noise_eval15`(95) | sim | 8 s | `be682008f2960c15` |
| **`eval_long_ls`（④ 长段）** | 1400（345/352/703） | 20261024 | v1/eval_v5_repro | DNS holdout15 | `noise_eval15`(95) | sim | **12 s** | `f5c928b310490e5f` |

- ② 两集的**远端** = AISHELL-3 holdout（`zh_far` 214 段拼接；`eval_zh` 的 near 仍为 DNS holdout，
  `eval_zh_both` 的 near = `zh_near` 239 段）；④ 两集的**远端** = LibriSpeech holdout（`ls_far` 800 条，p50 14.1 s）。
- `eval_long_ls` 的 `dt_plan_profile.json.dur_sec` 记的是 preset 的 8（清单字段），**实际窗口 12 s**
  记在该 profile 的 `cli_args.dur_sec=12.0`（aec_synth 对"dur 与清单不一致"只打 warn，见 configs/README §4）。

---

## 2. 配方与生成命令（逐字留档）

```powershell
$py='C:\Users\14805\.conda\envs\avsuite\python.exe'      # 从仓库根 avsuite/ 执行
$V8='D:/data/aec-synth/eval_v8_pools'; $EXT='D:/data/aec-synth/ext_pools'
$C = @('--num','1400','--soft-clip-prob','0','--nearfar-mutex','utterance',
       '--dt-plan','configs/dt_plan_v1.json','--dt-plan-preset','eval_v5_repro','--snr-range','20,30')

# ③ SNR 域外两档（清单用 v2 的两档 preset；--snr-range 管 FEST/NST，preset 管 DT）
& $py scripts/aec_synth.py --num 1400 --dur-sec 8 --soft-clip-prob 0 --nearfar-mutex utterance `
    --dt-plan configs/dt_plan_v2.json --dt-plan-preset eval_snr_low  --snr-range 0,10  `
    --near-dir $V8/near --far-dir $V8/far --noise-dir $V8/noise --out-dir D:/data/aec-synth/eval_snr_low  --seed 20261018
& $py scripts/aec_synth.py ... --dt-plan-preset eval_snr_high --snr-range 30,40 ... --out-dir D:/data/aec-synth/eval_snr_high --seed 20261019

# ① 音乐/宽带噪声轴（消声=noise_eval15 → MUSAN 子集）
& $py scripts/aec_synth.py @C --dur-sec 8 --near-dir $V8/near --far-dir $V8/far `
    --noise-dir $EXT/musan_eval15 --out-dir D:/data/aec-synth/eval_music --seed 20261020
# ② 中文远端 / 双端中文
& $py scripts/aec_synth.py @C --dur-sec 8 --near-dir $V8/near   --far-dir $EXT/zh_far `
    --noise-dir $V8/noise --out-dir D:/data/aec-synth/eval_zh      --seed 20261021
& $py scripts/aec_synth.py @C --dur-sec 8 --near-dir $EXT/zh_near --far-dir $EXT/zh_far `
    --noise-dir $V8/noise --out-dir D:/data/aec-synth/eval_zh_both --seed 20261022
# ④ 长语音远端（LS far）+ 12 s 长段变体
& $py scripts/aec_synth.py @C --dur-sec 8  --near-dir $V8/near --far-dir $EXT/ls_far `
    --noise-dir $V8/noise --out-dir D:/data/aec-synth/eval_far_ls  --seed 20261023
& $py scripts/aec_synth.py @C --dur-sec 12 --near-dir $V8/near --far-dir $EXT/ls_far `
    --noise-dir $V8/noise --out-dir D:/data/aec-synth/eval_long_ls --seed 20261024
```

**DT 清单 v2（③ 用）**：`configs/dt_plan_v2.json`（sha256 `d46f8f312fd28cc6…`）由
`scripts/make_dt_plan_v2.py` 从 v1（`73496c0bc877…`）**纯派生**：原样继承两个历史 preset，
新增 `eval_snr_low`（`dt_snr_db=0..10`）/ `eval_snr_high`（`30..40`），其余字段与 `eval_v5_repro` 逐字段相同；
`--check` 可复算校验派生关系；v1 未被改动（`verify_dt_plan --plan v1` 仍 PASS）。

**扩量数据流水线（全部只读拉取 + 本机转码）**：

```powershell
# 1) 只读拉子集（tar 流，分批 ≤150；不在服务器写任何文件）
& $py tools/pull_ext_subsets.py --key D:/Project_for_Python/DSH/dsh.pem --batch 150
#    -> D:/data/_t23_pull/{musan,aishell3_near,aishell3_far,ls_near,ls_far}（3350 文件 / 1.9 GB）
# 2) 转 16k mono PCM_16 池（AISHELL 按说话人拼接到 ≥10 s，避免 8 s 窗内循环补）
& $py tools/build_ext_eval_pools.py
#    -> D:/data/aec-synth/ext_pools/{musan_eval15,zh_near,zh_far,ls_near,ls_far} + _manifests/
```

子集抽样规则（确定性、可复现）：原始名单排序后**等间隔**抽样（`k = len // target`）取 target 个；
拉取脚本写出 `split_lists/ext_*_subset<N>.txt`，池建成后写出实际使用名单 `split_lists/ext_pool_*_used.txt`：

| 名单 | 条数 | sha256 |
|---|---|---|
| `ext_musan_eval15_subset150.txt` / `ext_pool_musan_eval15_used.txt` | 150 / 150 | `07fc2a6bb0d2d693` / `670fde5f1d25a707` |
| `ext_aishell3_far_eval_subset800.txt` / `ext_pool_zh_far_used.txt` | 800 / **214**（拼接段） | `00d294aa2acca208` / `048f3a707792150b` |
| `ext_aishell3_near_eval_subset800.txt` / `ext_pool_zh_near_used.txt` | 800 / **239**（拼接段） | `e8c9908427622e8f` / `076b67ee036e0579` |
| `ext_ls_far_eval_subset800.txt` / `ext_pool_ls_far_used.txt` | 800 / 800 | `4896289adb6c3c83` / `70f0ed499b956362` |
| `ext_ls_near_eval_subset800.txt` / `ext_pool_ls_near_used.txt` | 800 / 800 | `98512ad910036027` / `754a26830ab6f451` |

> MUSAN 子集构成：music 45 / noise 72 / speech 33（如实登记；"音乐/宽带噪声轴"用的是 MUSAN eval15 份额全量，
> 其中 speech 子集是合唱/语音类噪声）。

---

## 3. 验收证据（本任务复跑）

| # | 检查 | 结果 |
|---|---|---|
| 1 | `eval_v8_audit.py` 7 集准入审计（每集带 `--expect-seed` + 池名单覆盖 + `PROVENANCE.json`） | **7/7 PASS exit 0**（规模/留痕/隔离/无自回声/DT 口径/参数域/产物） |
| 2 | `verify_dt_plan.py`（`--strict-preset --limit 20 --set-dir … --replay-prefix 3 --replay-sim-rir`，带每集 `DT_PLAN_*` 源目录） | **7/7 PASS exit 0**（C 实跑锚定 + G 前缀逐字节 + D/E/F 区间） |
| 3 | `eval_zh_both` **全量重生成**逐字节核对 | manifest + `dt_plan_profile.json` 相同；**4200/4200 wav 相同** |
| 4 | 扩量池与训练份额互斥（reader/文件级） | LS 池 191 人 ∩ `ext_ls_readers_train70` = **0**；AISHELL 池 17 人 ∩ `ext_aishell3_readers_train70` = **0**；MUSAN 池 150 ∩ `ext_musan_train70` = **0** |
| 5 | 拉取落地核对（`pull_report.json`） | 5 个语料 **3350/3350** 文件落地，逐语料记录条数/字节/子集 sha256；无服务器写入 |
| 6 | 池格式核对（`_manifests/*.csv`） | 全部 16 k mono PCM_16；`zh_*` 拼接段 239/214 条**全部 ≥10 s**（8 s 窗无补环）；`ls_*` 800 条 p50 14.1 s |
| 7 | `pytest tests/test_t23_ext_axes.py -q`（v2 派生 / 子集规则 / 池转码，5 例） | **全过**；全仓 `pytest -q` = **393 项 = 389 passed + 4 xfailed, 0 failed**（exit 0） |
| 8 | 回归：`eval_v8` / `eval_rir_sim` 用参数化后的审计脚本复跑 | **PASS**（默认语义未变） |

---

## 4. t13 评测协议矩阵（三代际 × 全部轴）

### 4.1 代际定义

| 代际 | 产物 | 训练数据口径 | 备注 |
|---|---|---|---|
| **v4** | `exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt`（本机） | VoiceBank 3.3 h | 阶段0 锚；t5 已给 eval_v5/eval_v8 读数（DiD 见 `eval_v8_isolation_20261013.md`） |
| **v5** | 服务器 `outputs_p1/{nn_aec,dccrn,cascade}_dm`（P1-5，**净池重训版**） | DNS clean `train70` + `noise_train70` + **`rir_train70`（20 条，`/root/data/rirs_train70`）** | **`eval_rir` holdout 8 条零泄漏**；污染 run 已归档 `rir_contaminated`（附 CONTAMINATION.md）**不采信** |
| **v5x** | 扩量代际（LS tc100/360 + AISHELL-3 + MUSAN） | 待 t22/DP3 定 | 语料与切分清单已在服务器（`ext_*`） |

### 4.2 对照计划（哪些**配对**、哪些**非配对**）

| 轴（集） | 要回答的问题 | 对照 | 配对性 | CI 协议 |
|---|---|---|---|---|
| 说话人 | 未见说话人上的绝对水平 | `eval_v8`（或 `eval_ns`）**vs** `eval_v5`（**旧口径集**；注意与"v5 代际"同名不同物） | **非配对**（两样本） | 两样本 bootstrap B=2000 seed=777 + 传统基线臂 DiD |
| 噪声 | 未见噪声的代价 | `eval_ns` / `eval_v8` **vs** `eval_v5` | 非配对 | 同上（DiD） |
| 噪声稳定性 | 结论是否跨抽样复现 | `eval_ns` **vs** `eval_v8`（同配方异 seed） | 非配对（同分布） | 两样本 CI；两集分别报后合并看方向 |
| **RIR（真实 vs 仿真）** | 真实车厢 RIR 的掉点 | **`eval_rir` vs `eval_rir_sim`** | ✅ **逐样本配对**（manifest 逐字节相同、NST 363/363 mic 相同） | **配对** bootstrap（逐行差）B=2000 seed=777 |
| RIR（旁证） | 与历史 sim 口径的差 | `eval_rir` vs `eval_v8`/`eval_ns` | 非配对 | 两样本 CI（仅旁证） |
| SNR 域外 | 低/高 SNR 下的退化曲线 | `eval_snr_low` / `eval_snr_high` **vs** `eval_v8` | 非配对 | 两样本 CI；建议同时报 mic 地板 |
| 音乐/宽带噪声 | 非语音噪声下的抑制与近端 | `eval_music` **vs** `eval_v8`（同说话人/同 RIR，只换噪声） | 非配对 | 两样本 CI + DiD（传统臂） |
| 中文远端 | 语言/远端换域 | `eval_zh`（远端中文）**vs** `eval_v8`；`eval_zh_both`（双端中文）**vs** `eval_zh` | 非配对 | 两样本 CI |
| 长语音远端 | 长句远端的回声抑制 | `eval_far_ls` **vs** `eval_v8`；`eval_long_ls`（12 s）**vs** `eval_far_ls`（8 s） | 非配对（且窗口不同） | 两样本 CI；**12 s 口径与 8 s 不可混读** |
| 集内三方 | NN/传统/级联谁强 | 每集内 NN−传统、级联−NN | ✅ 集内**配对**（同 (name,scenario)） | 配对 bootstrap（仓内既有 `tools/compute_ci.py` 口径） |

**统一要求（继承 V2/V3.1）**：AECMOS@48k 模型协议；所有 DT 场景引用带 95%CI；FEST/NST 的
`deg_mos` 与 NST `echo_mos` 在 5.0 附近饱和（sd≈0）⇒ 不据此下结论；**FEST echo 的"NN 优势"已撤回**
（v8 −0.107 n.s. / DiD −0.470，见 NUMBERS 顶部决策条）；噪声轴 DiD 引用必须附"对 v4/v5（训练用全噪声池）
噪声轴未隔离"的说明；**RIR 轴必须用 `eval_rir` vs `eval_rir_sim` 配对读数**。
打分流（t5 机具可直接复用）：

```powershell
& $py exp/07_nn_aec/aecmos_eval.py --eval-dir <集> --trad|--nn-ckpt <ckpt> --out <json>   # 每集 ×2
& $py exp/07_nn_aec/eval.py --eval-dir <集> --ckpt <ckpt> --compare-trad --out <dir>       # 波形域
& $py exp/07_nn_aec/eval_v8_isolation.py --dir <该集产物目录>                              # CI/DiD 表
```

### 4.3 ckpt 加载口径（t13 前置，已在本机验证并固化）

**坑（train-eng 实测，本机已复现）**：服务器（Linux）保存的 ckpt 里 `cfg` 含 `pathlib.PosixPath`，
Windows 上 `torch.load(..., weights_only=False)` 抛 `NotImplementedError: cannot instantiate
'PosixPath' on your system`（`weights_only=True` 同样不救）。**14 个 P1 ckpt 全部如此。**

**解法（已固化为模块，勿各脚本重写）**：`exp/07_nn_aec/ckpt_compat.py`

```python
import pathlib, torch
pathlib.PosixPath = pathlib.PurePosixPath          # 2 行打桩（模块内做成上下文管理器，用完恢复）
st = torch.load(p, map_location="cpu", weights_only=False)
```

- `load_ckpt(path, map_location=...)`：打桩 + 加载（**已接入** `exp/07_nn_aec/eval.py`、
  `exp/07_nn_aec/aecmos_eval.py`、`exp/09_cascade_aec/run_cascade.py` 三处 `torch.load` 调用点）；
- `python exp/07_nn_aec/ckpt_compat.py --glob <artifacts_dir>`：批量体检并落盘
  `exp/07_nn_aec/outputs/p1_ckpt_index.json`；
- 单测 `tests/test_ckpt_compat.py`（4 例）：①机制（构造引用 `pathlib.PosixPath` 的 pickle →
  Windows 上直接 loads 必崩、打桩后可 loads、**退出后 PosixPath 必须复原**）；②torch 层；
  ③真 ckpt（NN v5，`state_dict=638264`、`model.parameters()=638034`、`cfg.w_sisnr='1.0'`）。

**本机 14 个 ckpt 体检结果（`p1_artifacts/outputs_p1/`，2026-10-14 09:02 拉回的副本）**：

| run | best epoch / last epoch | w_sisnr | 能否进矩阵 |
|---|---|---|---|
| `nn_aec_v5` | 70 / 240 | 1.0 | ✅ |
| `nn_aec_v5x` | 127 / 240 | 1.0 | ✅ |
| `nn_aec_v5x_noMUSAN` | 63 / 240 | 1.0 | ✅ |
| `dccrn_dm`（v5） | 82 / 100 | （cfg 无 w_sisnr 键） | ✅ |
| `dccrn_v5x` | 91 / 100 | — | ✅ |
| `casc_dm` | 21 / 24（副本时点） | **0.1** | ⚠️ 见下 |
| `casc_v5x` | 19 / 22（副本时点） | **0.1** | ⚠️ 见下 |

> **级联口径（队长 2026-10-14 裁定）**：两个级联 ckpt 是**截断版**（服务器关机时点，续训中）且
> `w_sisnr=0.1`（与 NN/DCCRN 的 1.0 不同）⇒ t13 对级联**只跑"截断版内部对照"**（三方同集排序仍可看趋势，
> 不对外引用绝对值）；**正式读数等服务器续跑 100ep + w_sisnr=1.0 版本**。
> 其余五模型（NN v5/v5x/noMUSAN、DCCRN v5/v5x）可直接进矩阵评测。
> 注：`state_dict` 元素数 638,264 含 230 个 BN buffer；**架构参数量**（`model.parameters()`）= **638,034**，
> 引用时写清口径（0.638M 指的是后者）。
> 服务器侧另有 `*_rir_contaminated`、`*_wsisnr0.1_mismatch` 等口径不齐目录——**一律不采信**（§5-1）。

---

## 5. 边界与坑（必读）

1. **RIR 污染：已处置，v5 口径按净池重训版**。发现经过：P1-5 训练用 `--rir-dir /root/data/rirs`
   （42 条 = 全部 32 条车厢 IR + 8 条 48k 原始 + 其它），其中含 t16 划给评测的 8 条 holdout
   （`IR64CAR_0_*` / `IR64CAR_315_*`）⇒ 当时代际的 `eval_rir` 轴不成立。
   **处置（train-eng，队长 2026-10-13 裁定）**：停污染 run → 建 `/root/data/rirs_train70`
   （`rir_train70.txt` 的 **20 条**硬链接，顺带排除 8 条不在任何名单的裸名 48k 文件）→
   NN-AEC/DCCRN/级联**全部净池重训**（自动链 ~6h）；**两个污染 run 归档 `rir_contaminated`
   （附 `CONTAMINATION.md`），不采信**。
   ⇒ **t13 口径**：v5 训练 RIR = `rir_train70` **20 条**（净池重训版），`eval_rir` 的 8 条 holdout
   **零泄漏**；RIR 轴照 §4.2 用 `eval_rir` vs `eval_rir_sim` 的**逐样本配对**读数；污染 run 的任何读数不得进入报告/NUMBERS。
2. **AISHELL-4 维持阻断**（t21）：parquet + 服务器无 pyarrow ⇒ ② 用 AISHELL-3（174 人池，holdout 22 人）。
   `eval_zh_both` 的远端/近端都是中文（拼接段），`eval_zh` 只把远端换成中文。
3. **AISHELL-3 需拼接才有长段**：单句 ~3.2 s ⇒ 直接进 8 s 窗会被 `fit_len` 循环补（同一句重复 =
   伪周期远端）。本报告的两集用的是**按说话人拼接到 ≥10 s** 的池（239/214 段，全部 ≥10 s）。
   若 t13 要"原句级"中文远端（每行只含一句 + 补环），需另建池——列为选项，未做。
4. **LS 池的补环比例**：`eval_far_ls`（8 s）86.6% 的 far 文件无需补环（13.4% 短句补环）；
   `eval_long_ls`（12 s）71.5% 无需补环（28.5% 补环）——与既有集对短源文件的处理一致，如实登记。
5. **MUSAN 是 303 的确定性 150 子集**（等间隔抽样；music 45/noise 72/speech 33），不是全量；
   子集名单与池名单 sha256 已落盘（§2 表）。MUSAN 里有 16 分钟级的长音乐文件（fit_len 随机裁剪）。
6. **语料只读拉取纪律**：`tools/pull_ext_subsets.py` 只用 `tar -cf -` 把文件流回本机（服务器零写入），
   每批 ≤150 文件；本机 `D:/data/_t23_pull/` 保留原始相对目录便于按说话人核对。
7. **跨集一律非配对（除 RIR 轴与集内三方）**：不同集的样本不同；配对性已在 §4.2 逐行标注，
   非配对处不得用配对 CI 的措辞。
8. **未做模型打分**：本任务只落地评测集与协议；v4/v5/v5x 在 7 个新集上的读数属 t13，
   建议按 §4.2 逐轴执行并把结果并入 `reports/DATA_CI.md` 与 NUMBERS（收档归 t14）。
9. **单测计数刷新**按队长裁定归 t14（现 NUMBERS §3 = t6 时点值 321；本任务复跑全仓 = **393 项 =
   389 passed + 4 xfailed**，差额来自其间落地的 DM/清单类测试（t10/t15/t17 等）与本任务 +5）。
   **（阶段1 收档台阶：终值 416 项 = 412 passed + 4 xfailed，见 `docs/NUMBERS.md` §3 与
   `reports/phase1_acceptance_20261015.md`）**

---

## 6. 交付清单

| 类别 | 路径 |
|---|---|
| 评测集（新增 7） | `D:/data/aec-synth/{eval_snr_low,eval_snr_high,eval_music,eval_zh,eval_zh_both,eval_far_ls,eval_long_ls}`（各 1400 条 + manifest + `dt_plan_profile.json` + `PROVENANCE.json`） |
| 扩量池 | `D:/data/aec-synth/ext_pools/{musan_eval15,zh_near,zh_far,ls_near,ls_far}` + `_manifests/{*.csv,summary.json}` |
| 原始拉取 | `D:/data/_t23_pull/**`（3350 文件 / 1.9 GB）+ `pull_report.json` |
| 清单 | `split_lists/ext_*_subset*.txt`（子集）、`split_lists/ext_pool_*_used.txt`（实际用池）、`configs/dt_plan_v2.json`（SNR 两档） |
| 脚本 | `tools/pull_ext_subsets.py`、`tools/build_ext_eval_pools.py`、`scripts/make_dt_plan_v2.py`、`tests/test_t23_ext_axes.py`；`exp/07_nn_aec/eval_v8_audit.py`（参数化：preset/版本/区间/池名单/时长覆盖，默认语义不变） |
| 报告 | 本文件（矩阵 + 协议 + 边界）；上一阶段的轴集见 `reports/eval_ns_rir_20261013.md` |
