# eval_v7 — FEST/NST 分层补强集（n=800）三方重评

- 日期：2026-10-11 ｜ 执行：WS-V5 / data-eng（t7）｜ 依据方案：`reports/refactor_wave5/data_inventory.md` §5（队长已批准）
- 数据：`D:/data/aec-synth/eval_v7`（800 条 × 8s；**FEST 212 / NST 182 / DT 406**，manifest 实测）
- 目的：解决 `eval_v5` 的两层样本缺口（实测 FEST **70** / NST **76** < 150），并把「级联 FEST echo 优势 +0.633±0.189」这条全表相对精度最差（29.9%）的头条结论放到 FEST≥150 上复检
- 纪律：**不改任何既有报告/docs/NUMBERS.md/DATA_CI.md/tools/**；结论写在本文，不覆盖旧值（附记纪律）；`tools/compute_ci.py` 的 main 未运行（见 §7）

---

## 0. 结论摘要（一句话版）

1. **级联 FEST echo 优势保持，且从「勉强显著」变稳健**：eval_v5 `+0.633±0.189`（CI/Δ=29.9%）→ eval_v7 `+0.653±0.145`，bootstrap 95%CI **[+0.510, +0.795]**（n=212，CI 不含 0，相对精度降到 22.2%）。**同号 + 显著 + CI 收窄**，方向结论复现。
2. **两条主结论全部保持且 CI 明显收窄**：DT `NN−传统 Δecho` `+1.196±0.111 → +1.166±0.078`（−30%）；NST `NN−传统 Δdeg` `−0.554±0.077 → −0.567±0.050`（−35%）；DT `Δdeg` `−0.619±0.071 → −0.512±0.046`（−35%）。三条 CI 均不含 0。
3. **一条旧"打平"结论被大样本推翻为"极弱显著"（如实记录）**：DT `级联−NN Δecho` 由 eval_v5 的 `+0.020±0.090`（n.s.）变为 `+0.088±0.064`，bootstrap **CI [+0.024, +0.151] 刚排除 0**。但效应量仅 0.088 MOS（= NN 相对传统优势 +1.166 的 **7.5%**），且在**另外三个集上符号不定**（官方集 −0.027 / eval_v6 +0.062 / eval_v5 +0.020）——结论应写成"级联与纯 NN 在 DT echo 上**实质打平**，v7 大样本下出现 ≤0.09 MOS 的微弱、跨集不一致的正差"，**不构成"级联 echo 优于纯 NN"的恢复**。
4. FEST `NN−传统 Δecho` 仍**不显著**（`−0.130±0.189`，bootstrap [−0.319, +0.055]）——FEST 层的可判别面只有 echo，而这一层上 NN 相对传统没有优势；`NUMBERS.md:117` 把"级联 FEST echo 三方最高"作为级联保留价值的依据，**该依据在 v7 上依然成立**（级联 4.401 > 传统 3.879 > NN 3.749，级联−传统 `+0.523±0.124` 显著）。

---

## 1. 数据（合成与校验）

### 1.1 配方（与 eval_v5 逐字同配方，仅 `--num`/`--seed`/`--out-dir` 不同）

| 参数 | eval_v5 | eval_v7 |
|---|---|---|
| num | 300 | **800** |
| seed | 20261011 | **20261013**（换 seed 的理由见 1.2） |
| scenarios（FEST,NST,DT） | 0.25,0.25,0.5 | 同左（**未改配比**） |
| dur-sec / snr-range / soft-clip-prob / dt-leadin-sec | 8 / 20,30 / 0 / 2 | 同左 |
| ser-range / delay | 0,10（默认）/ 10–80ms（默认） | 同左 |
| RIR | pyroomacoustics 仿真（无 `--rir-dir`，manifest `rir_src=sim`） | 同左 |
| 语料 | VoiceBank-DEMAND 16k `train/clean`（near=far）+ DNS noise | 同左 |

```powershell
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' D:/Project_for_Python/DSH/avsuite/scripts/aec_synth.py `
  --num 800 --dur-sec 8 --out-dir D:/data/aec-synth/eval_v7 `
  --snr-range 20,30 --soft-clip-prob 0 --dt-leadin-sec 2 `
  --seed 20261013 --scenarios 0.25,0.25,0.5 `
  --near-dir D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --far-dir  D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --noise-dir D:/data/dns-challenge-mirror/datasets/noise
# 控制台实测：800 条 -> D:\data\aec-synth\eval_v7  场景分布 {'fest': 212, 'nst': 182, 'dt': 406}  耗时 23.4s
```

### 1.2 实测分层与准入校验（全部 PASS）

| 校验项 | 要求 | 实测 | 结论 |
|---|---|---|---|
| manifest 行数 | 800 | **800** | ✅ |
| 每层 ≥150 | FEST/NST/DT 各 ≥150 | **212 / 182 / 406** | ✅ |
| 分层抽中偏离设计期望 | 期望 200/200/400（sd 12.25/12.25/14.14） | FEST +1.0σ、NST −1.5σ、DT +0.4σ | ✅ 均在 ±1.5σ 内 |
| 三槽 wav 数 | 3×800 | farend/nearend/mic 各 **800** | ✅ |
| `name` 唯一 | 无重复 | 800 唯一 | ✅ |
| 参数域 | delay∈[10,80] ser∈[0,10] snr∈[20,30] | [10.078,79.861] / [0.002,9.987] / [20.007,29.970] | ✅ |
| soft_clip / rir_src | 全 0 / 全 sim | 800/800、800/800 | ✅ |
| DT lead-in / 其余 0 | DT=2.0，FEST/NST=0.0 | 406 条 2.0 / 364 条 0.0 | ✅ |
| 自回声（near==far） | 0 条 | **0** | ✅ |
| 与 eval_v5 非同源 | 前 300 条不得逐行相同 | **content 列 0/300 相同**（near/far/noise/ser/snr/delay 全 0 命中） | ✅ |

> 事前设计保证（`data_inventory.md` §5.1）：n=800 时单层 `<150` 的二项尾概率 ≈4×10⁻⁵（z=4.08），三层联合 ≈10⁻⁴。**实测抽中 NST=182 属正常抽样波动**（−1.5σ），不是设计失手。
> 非同一律：v7 与 eval_v5 的 `scenario` 序列有 113/300 巧合相同（伯努利序列的正常重叠），但**决定内容的 5 个文件/参数列 0/300 相同** ⇒ 两集样本独立。

### 1.3 磁盘（实测 vs 方案预测）

| 产物 | 方案预测（§5.5） | 实测 | 文件数 |
|---|---|---|---|
| `D:/data/aec-synth/eval_v7` | ≈586 MiB | **586.1 MiB** | 2401 |
| `D:/data/aec-synth/eval_v7_lin` | ≈977 MiB | **976.8 MiB** | 2401 |
| `D:/data/aec-synth/cascade_stage/eval_v7`（级联 enh，仓外） | ≈391 MiB | **390.7 MiB** | 800 |
| **合计** | ≈1.9 GiB | **1.91 GiB** | 5602 |

---

## 2. 评分管线与产物

### 2.1 口径（与 eval_v5 完全一致）

| 方法 | 实现 | 参数 |
|---|---|---|
| 传统 AEC | `pipeline.TradAEC`（WP-06 全链，`aecmos_eval.py --trad`） | `mdf, partitions=8, mu=0.7` |
| NN v4 | `NNAEC` + **`exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt`** | 与 eval_v5 打分同 ckpt（见 §6.1） |
| 级联 MDF→NN | `precompute_cascade.py` 同款线性级 + `exp/09_cascade_aec/outputs/nnaec_g0.5_j10/best.ckpt`（val≈23.99855） | `run_cascade.py`，lin-check mismatch=**0** |
| AECMOS | 官方 ONNX `D:/data/repos/AEC-Challenge/AECMOS/AECMOS_local/Run_1668423760_Stage_0.onnx` | 全段送评（8s 短片段口径，与 eval_v5 一致） |
| 指标 | `eval.py --compare-trad`（PESQ-wb / STOI / ERLE） | 同 eval_v5 |

### 2.2 命令与产物（新目录，未覆盖任何既有 json）

```powershell
$py='C:\Users\14805\.conda\envs\avsuite\python.exe'; $R='D:/Project_for_Python/DSH/avsuite'
& $py "$R/exp/09_cascade_aec/precompute_cascade.py" --synth-dir D:/data/aec-synth/eval_v7 --out-dir D:/data/aec-synth/eval_v7_lin
& $py "$R/exp/07_nn_aec/aecmos_eval.py" --eval-dir D:/data/aec-synth/eval_v7 --trad  --out "$R/exp/07_nn_aec/outputs/eval_v7/aecmos_trad_v7.json"
& $py "$R/exp/07_nn_aec/aecmos_eval.py" --eval-dir D:/data/aec-synth/eval_v7 --nn-ckpt "$R/exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt" --out "$R/exp/07_nn_aec/outputs/eval_v7/aecmos_nn_best_v7.json"
& $py "$R/exp/09_cascade_aec/run_cascade.py" --eval-dir D:/data/aec-synth/eval_v7 --ckpt "$R/exp/09_cascade_aec/outputs/nnaec_g0.5_j10/best.ckpt" --lin-dir D:/data/aec-synth/eval_v7_lin --enh-dir D:/data/aec-synth/cascade_stage/eval_v7/enh_cascade
& $py "$R/exp/07_nn_aec/aecmos_eval.py" --eval-dir D:/data/aec-synth/eval_v7 --enh-dir D:/data/aec-synth/cascade_stage/eval_v7/enh_cascade --out "$R/exp/07_nn_aec/outputs/eval_v7/aecmos_cascade_v7.json"
& $py "$R/exp/07_nn_aec/eval.py" --eval-dir D:/data/aec-synth/eval_v7 --ckpt "$R/exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt" --compare-trad --out "$R/exp/07_nn_aec/outputs/eval_v7/nn_metrics_v7"
& $py "$R/exp/07_nn_aec/outputs/eval_v7/ci_driver.py"          # CI：解析 + 配对 bootstrap（不跑 compute_ci.main）
& $py "$R/exp/07_nn_aec/outputs/eval_v7/metrics_compare.py"   # PESQ/STOI/ERLE 与 CI 宽度对比
```

| 产物 | 路径 | n |
|---|---|---|
| trad AECMOS | `exp/07_nn_aec/outputs/eval_v7/aecmos_trad_v7.json` | 800 |
| NN v4(best) AECMOS | `…/eval_v7/aecmos_nn_best_v7.json` | 800 |
| 级联 AECMOS | `…/eval_v7/aecmos_cascade_v7.json` | 800 |
| CI 明细（逐文件 + 配对） | `…/eval_v7/ci_eval_v7.json` | — |
| 指标对比 | `…/eval_v7/metrics_compare_v7_vs_v5.json`、`…/nn_metrics_v7/{summary.json,rows.csv,compare_trad_vs_nn.md}` | 800 |
| 驱动脚本 | `…/eval_v7/{ci_driver.py,metrics_compare.py}` | — |
| 级联 enh wav（仓外） | `D:/data/aec-synth/cascade_stage/eval_v7/enh_cascade/` | 800 |

**拒评/跳过：0 条**（三方全部 800/800 完成；无 failed、无非有限、无跳过）。参照官方集"级联拒评 2/800"的披露范式，本次**无任何拒评**，`run_cascade.py` 的 lin-check mismatch=0。

### 2.3 耗时（实测）

| 阶段 | 实测 | 时间戳（2026-10-11） |
|---|---|---|
| 合成 800 条（2400 wav） | **23.4 s** | 17:15:17 → 17:15:38 |
| lin 预计算（FLOAT） | **227 s**（0.28 s/条） | 17:15:59 → 17:19:46 |
| NN v4(best) 推理 + AECMOS | ≈90 s | → 17:16:47 |
| 传统 MDF + AECMOS | ≈3 min | → 17:19:55 |
| 级联推理（800 enh） | **195 s** | 17:20:05 → 17:23:20 |
| 级联 AECMOS | ≈50 s | → 17:24:10 |
| PESQ/STOI/ERLE（NN+trad） | ≈6 min（与前序并行） | → 17:22:36 |
| CI + 对比脚本 | <10 s | → 17:28:25 |
| **端到端（含进程启动）** | **≈13 min** | 17:15:17 → 17:28:25 |

> 方案 §5.5 预测 30–45 min，实测 **13 min**（更快；NN/last 交叉列与指标作业并行跑掉了）。磁盘与耗时两项预测均在 1% 内命中（见 §1.3）。

---

## 3. 三方 AECMOS 对照（eval_v7，附 eval_v5 旧值）

mean ± 95%CI（解析法 1.96·sd/√n），括号内为 eval_v5 旧值。**FEST/NST 的 deg 与 FEST/NST 的 echo 存在天花板（4.999/5.000），该维度无判别力**。

| 层 | n (v5) | 指标 | 传统 AEC | NN v4 | 级联 MDF→NN |
|---|---|---|---|---|---|
| **FEST** | **212** (70) | echo_mos | 3.879±0.139 (sd 1.032) ｜旧 4.002±0.199 | 3.749±0.129 (sd 0.960) ｜旧 3.956±0.192 | **4.401±0.086** (sd 0.637) ｜旧 4.589±0.052 |
| FEST | 212 (70) | deg_mos | 4.999（饱和） | 4.999（饱和） | 4.999（饱和） |
| **NST** | **182** (76) | echo_mos | 5.000（饱和） | 4.999（饱和） | 5.000（饱和） |
| **NST** | **182** (76) | deg_mos | **2.579±0.057** (sd 0.389) ｜旧 2.645±0.110 | 2.012±0.048 (sd 0.327) ｜旧 2.091±0.097 | 2.010±0.047 (sd 0.327) ｜旧 2.062±0.093 |
| **DT** | **406** (154) | echo_mos | 2.449±0.060 (sd 0.615) ｜旧 2.525±0.102 | **3.615±0.078** (sd 0.803) ｜旧 3.721±0.118 | **3.703±0.071** (sd 0.735) ｜旧 3.741±0.109 |
| DT | 406 (154) | deg_mos | **2.376±0.049** (sd 0.509) ｜旧 2.430±0.084 | 1.864±0.029 (sd 0.301) ｜旧 1.811±0.046 | 1.877±0.030 (sd 0.309) ｜旧 1.851±0.046 |

**绝对值位移**：v7 各列普遍比 v5 低 0.05–0.12（DT trad echo 2.525→2.449、NN 3.721→3.615、NST trad deg 2.645→2.579），且 FEST 的离散度显著变大（级联 sd 0.222→0.637、传统 sd 0.848→1.032）——这是**换样本**的正常波动（v5→v7 非同源），也解释了 §5 里 FEST 配对 CI 收窄幅度小于 1/√n 预测。

---

## 4. 配对差值（同条配对）与 eval_v5 对比

Δ = A − B，按 `(name, scenario)` 逐条对齐；解析 CI = 1.96·sd/√n；bootstrap 95%CI = B=2000、seed=777 百分位法（同 `tools/compute_ci.py` 约定）。

| 层 | 配对 | 指标 | n | **Δ (v7)** | 解析 CI | bootstrap 95%CI | 显著 | v5 旧 Δ | 方向 |
|---|---|---|---|---|---|---|---|---|---|
| FEST | NN − 传统 | echo | 212 | **−0.130** | ±0.189 | [−0.319, +0.055] | 否 | −0.046±0.281 | ✅ 同号 |
| FEST | **级联 − NN** | echo | 212 | **+0.653** | ±0.145 | **[+0.510, +0.795]** | **是** | +0.633±0.189 | ✅ **同号且显著** |
| FEST | 级联 − 传统 | echo | 212 | +0.523 | ±0.124 | [+0.406, +0.639] | 是 | +0.587±0.186 | ✅ 同号 |
| NST | NN − 传统 | deg | 182 | **−0.567** | ±0.050 | [−0.618, −0.519] | 是 | −0.554±0.077 | ✅ 同号 |
| NST | 级联 − NN | deg | 182 | −0.002 | ±0.035 | [−0.036, +0.031] | 否 | −0.029±0.062 | ✅ 同号(近零) |
| NST | 级联 − 传统 | deg | 182 | −0.569 | ±0.046 | [−0.617, −0.522] | 是 | −0.583±0.082 | ✅ 同号 |
| DT | NN − 传统 | echo | 406 | **+1.166** | ±0.078 | [+1.085, +1.245] | 是 | +1.196±0.111 | ✅ 同号 |
| DT | NN − 传统 | deg | 406 | **−0.512** | ±0.046 | [−0.560, −0.464] | 是 | −0.619±0.071 | ✅ 同号 |
| DT | 级联 − NN | echo | 406 | **+0.088** | ±0.064 | **[+0.024, +0.151]** | **是（边缘）** | +0.020±0.090（n.s.） | ⚠️ **由 n.s. 变边缘显著** |
| DT | 级联 − NN | deg | 406 | +0.012 | ±0.026 | [−0.014, +0.038] | 否 | +0.041±0.041 | ✅ 同号 |
| DT | 级联 − 传统 | echo | 406 | +1.254 | ±0.061 | [+1.194, +1.316] | 是 | +1.217±0.089 | ✅ 同号 |
| DT | 级联 − 传统 | deg | 406 | −0.499 | ±0.045 | [−0.545, −0.454] | 是 | −0.579±0.072 | ✅ 同号 |

> **饱和伪显著**：`NST echo`（两侧均 5.000±0.000）与 `FEST deg`（均 4.999）上的"显著"是数值噪声（sd≈0.001–0.002），不得解读为方法差异。上表已剔除。

---

## 5. 核心问题：级联 FEST echo 优势在大样本上是否保持？

**答：保持，且证据强度提升。**

| | eval_v5（n=70） | eval_v7（n=212） | 变化 |
|---|---|---|---|
| 级联 FEST echo | 4.589±0.052 | 4.401±0.086 | −0.188（换样本） |
| NN v4 FEST echo | 3.956±0.192 | 3.749±0.129 | −0.207 |
| **Δ（级联 − NN，同条配对）** | **+0.633±0.189** | **+0.653±0.145** | Δ 几乎不动（+0.020） |
| bootstrap 95%CI | — | **[+0.510, +0.795]** | 完全在 0 右侧 |
| 相对精度 CI/\|Δ\| | 29.9% | **22.2%** | 收窄 7.7pp |
| 三方排序 | 级联 > 传统 > NN | 级联 4.401 > 传统 3.879 > NN 3.749 | **一致** |

**判读**：v5 里「级联 FEST echo 最优」这条结论的主要风险是"n=70 + 相对精度 30%"，而不是"方向可疑"；v7 把它放到 212 条上后，点估计几乎不动（+0.633→+0.653），配对 CI 从 ±0.189 收到 ±0.145（因为 v7 的 FEST 样本更离散，未达到按 n 外推的 ±0.112，见下），**CI 完全不含 0**。结论从"显著但脆弱"变成"显著且稳健"。

> 与另外三个集的对照（保持诚实边界）：级联−NN FEST echo 在**官方真实集** −0.052 n.s.、**eval_v6 真实车厢 RIR** −0.313 n.s.、**eval_v5** +0.633 显著、**eval_v7** +0.653 显著。⇒ **该优势只在合成 RIR 域成立**（"级联在仿真 FEST 上更会把 echo 打到顶"），真实域上不成立；本文只声明合成域结论，真实域的负结果以既有 `reports/ood_real_echoes.md`、`reports/realrir_domain.md` 为准。这不是矛盾，是**口径分层**。

### 5.1 CI 收窄核对（对 `data_inventory.md` §5.6 的 6 个数值目标）

| 指标 | v5 CI | v7 CI | 目标 | 达标 | 1/√n 理论值 |
|---|---|---|---|---|---|
| FEST NN echo | ±0.192 | **±0.129** | ≤0.13 | ✅ | ±0.111 |
| FEST 级联−NN Δecho | ±0.189 | **±0.145** | ≤0.13 | ❌（超 12%） | ±0.112 |
| NST NN deg | ±0.097 | **±0.048** | ≤0.07 | ✅ | ±0.059 |
| NST NN−传统 Δdeg | ±0.077 | **±0.050** | ≤0.06 | ✅ | ±0.048 |
| DT NN echo | ±0.118 | **±0.078** | ≤0.09 | ✅ | ±0.073 |
| DT NN−传统 Δecho | ±0.111 | **±0.078** | ≤0.08 | ✅ | ±0.069 |

**5/6 达标，1 项未达**。原因可量化：FEST 级联−NN 的**配对 sd 由 0.807 升到 1.077（+33%）**（v5 的 FEST 级联几乎全在高位、sd 仅 0.222；v7 的 FEST 样本更分散、级联 sd 0.637），把 n 翻 3 倍的收益吃掉了一部分。整体上 6 项 CI 收窄到旧值的 **0.65–0.77×**（理论 1/√n 应为 0.57–0.62×）——**样本量不是唯一杠杆，分层内的离散度同样重要**，这条经验建议回写方案模板。

---

## 6. 方向一致性验收（对 §5.6 B1–B6）

| # | 检查 | v5 基线 | v7 实测 | 判定 |
|---|---|---|---|---|
| B1 | DT `NN−传统` Δecho > 0 且 CI 不含 0 | +1.196±0.111 | **+1.166±0.078 [+1.085,+1.245]** | ✅ 保持 |
| B2 | DT `NN−传统` Δdeg < 0 且 CI 不含 0 | −0.619±0.071 | **−0.512±0.046 [−0.560,−0.464]** | ✅ 保持 |
| B3 | NST `NN−传统` Δdeg < 0 且 CI 不含 0 | −0.554±0.077 | **−0.567±0.050 [−0.618,−0.519]** | ✅ 保持 |
| B4 | DT `级联−NN` Δecho 不显著 | +0.020±0.090 n.s. | **+0.088±0.064 [+0.024,+0.151] 显著（边缘）** | ⚠️ **未保持**（见 §6.2） |
| B5 | FEST `级联−NN` Δecho 报出符号与 CI | +0.633±0.189 显著 | **+0.653±0.145 [+0.510,+0.795] 显著** | ✅ 保持并加强（本文核心问题） |
| B6 | FEST/NST echo 饱和（三方法 ≥4.99） | 5.000/4.999/5.000 | 5.000/4.999/5.000 | ✅ 保持 |

**5/6 保持，1 项按预设规则如实记录为"未保持"（B4）。**

### 6.1 ckpt 口径（一处任务书笔误的处置）

任务书写"NN v4 = `outputs_v4/.../last.ckpt`"；实测判定 eval_v5 用的是 **`best.ckpt`**：既有产物 `exp/07_nn_aec/outputs/eval_ood_delay/aecmos_nn_v5_recheck.json`（由 `delay_ood_probe.md:121-128` 用 best.ckpt 复跑 eval_v5）与 eval_v5 基线 `aecmos_nn_v4.json` **300/300 行逐位相同**。为满足"口径与 eval_v5 完全一致"，**主列一律用 best.ckpt**（队长已确认此为任务书笔误）。副产物 `aecmos_nn_last_v7.json` 已落盘但**本文不分析**（best/last 敏感性对比归 t12）。

### 6.2 B4 的处置（不粉饰）

DT `级联−NN Δecho` 在 v7 上 **+0.088，bootstrap CI [+0.024, +0.151] 刚排除 0**。三点必须同时说：

1. **效应量极小**：+0.088 MOS ≈ NN 相对传统优势（+1.166）的 **7.5%**，也小于 AECMOS 分辨力口径下的实际意义阈值。
2. **跨集不一致**：官方集 −0.027（n.s.）、eval_v6 +0.062（n.s.）、eval_v5 +0.020（n.s.）、eval_v7 +0.088（边缘显著）——四个集里三个不显著，且符号为三正一负（官方集为负）。
3. **配对 bootstrap 的边界效应**：CI 下界 +0.024 距 0 仅 0.024（≈0.4×CI 半宽），属"统计上刚好成立、实践上仍在噪声带内"。

⇒ 结论表述维持"**级联与纯 NN 在 DT echo 上实质打平**"，仅追加一句必要条件："v7 大样本（n=406）下出现 +0.088 [0.024, 0.151] 的微弱正差，但跨 4 集符号/显著性不一致，不足以支撑'级联 echo 优于纯 NN'的宣称恢复。"`NUMBERS.md:117` 的旧表述（"级联−NN Δecho +0.02，判为小样本波动，级联 echo 优于纯 NN 证据不成立"）**不需要改写**，只需在后续统一口径时把 v7 作为补充证据附注。

---

## 7. PESQ / STOI / ERLE（`eval.py --compare-trad` 同口径）

| 指标 | v5 传统 | v7 传统 | v5 NN | v7 NN | 判读 |
|---|---|---|---|---|---|
| FEST ERLE (dB) | 20.183 | 19.530 | 69.840 | 68.490 | 同向（NN 的 68–70dB 是掩码门控口径产物，不可与线性 ERLE 直接比） |
| DT PESQ(out) | 1.717 | 1.689 | 1.455 | 1.432 | Δ 传统−NN = −0.262 → **−0.257**，一致 |
| DT PESQ(mic) | 1.410 | — | 1.410 | 1.399 | NN 处理后在 mic 附近 |
| DT STOI | 0.805 | 0.807 | 0.751 | 0.754 | 一致（传统高 ~0.053） |
| NST PESQ(out) | 2.468 | 2.581 | 2.381 | 2.441 | 传统−NN = +0.087 → **+0.140**，**同向**（NN 近端 PESQ 更低） |
| NST PESQ(mic) | 2.498 | — | 2.498 | 2.606 | 两集都是 **mic > 处理后** ⇒ NST 层上两个系统都是在"降质" |
| NST STOI | 0.895 | — | 0.869 | 0.876 | 传统高 ~0.02–0.026 |

> 注：`eval_v5_dt300.md` 只公布了 DT PESQ/STOI 与 FEST ERLE，未公布 NST PESQ；上表 v5 的 NST 值由 `trad_metrics_v5.json` / `nn_metrics_v5.json/summary.json` 逐条复算（脚本 `metrics_compare.py`）。
> **一处需要收窄的旧表述**（只记录不改）：`exp/07_nn_aec/README.md:65` 的"NST 近端 PESQ 传统 2.446 / NN 2.477（透传且略有增益）"是 eval_v3（NST n=20）口径；在 eval_v5（n=76，2.468/2.381）与 eval_v7（n=182，2.581/2.441）上 NN 的 NST PESQ **都低于传统**，且两者都低于未处理 mic。该行属既有报告，本轮不改（附记纪律），建议后续统一口径时修正。

---

## 8. 纪律与未改动声明

- **本任务未修改**：`tools/**`、`docs/**`（含 `docs/NUMBERS.md`）、`reports/DATA_CI.md` 及任何既有报告。逐项证据：
  - `reports/DATA_CI.md`：运行前后 SHA256 均为 `E36367705056D2935642674319B7D75AC62F7EAF9E9E4B4CA46462DD50A1DA68`，LastWriteTime 仍为 2026-10-11 09:34:00（**未触发 compute_ci.main 的实证**）。
  - `tools/compute_ci.py`：SHA256 `3D553351A2FCB4AB41AE40BFC68BD3325BCCB25842FF3D7D6FE814ADF4D8730D`（未改动）。
  - `docs/NUMBERS.md`：**本任务未写入**；该文件在本次运行窗口内（17:24:45）被**团队其他成员**改动，`git diff HEAD --numstat` 为 `+7/-0`，且全文不含 `eval_v7` 字样 ⇒ 与本任务产物无关（此处如实记录，避免把他人改动算作本任务、也避免声称"未变"）。
  - `reports/` 目录下本次唯一新增文件即本文（17:29:11）；`git status --porcelain` 中本任务相关新增仅 `?? reports/eval_v7_fest_nst.md`（`exp/` 路径被 .gitignore 覆盖）。
- **CI 计算未触发 `compute_ci.py` 的 main**：`exp/07_nn_aec/outputs/eval_v7/ci_driver.py` 以 `importlib` 加载 `tools/compute_ci.py`（`__name__ != "__main__"` ⇒ 不执行 main、不写任何文件），复用其 `Z/B_BOOT/SEED/METRICS/fmt_ci`，并**逐字复刻**其配对 bootstrap（`compute_ci.py:141-147`，B=2000/seed=777/百分位法），只把场景循环从 `("dt",)` 扩到 `("fest","nst","dt")`、把配对改为显式参数。
- **新增文件**：`reports/eval_v7_fest_nst.md`（本文）；`exp/07_nn_aec/outputs/eval_v7/**`（新目录）；`D:/data/aec-synth/{eval_v7,eval_v7_lin,cascade_stage/eval_v7}`（新数据）。
- **范围外只记录不修**：§7 的 07 README:65 NST PESQ 旧表述；`eval_v5` 的 FEST 分层 sd 明显偏小（0.222）导致旧 CI 偏乐观这一现象本身。

---

## 9. 附：机器可读产物索引

- 数据：`D:/data/aec-synth/eval_v7/{manifest.csv,farend,nearend,mic}`（800 条，586.1 MiB）
- 线性级：`D:/data/aec-synth/eval_v7_lin`（FLOAT，976.8 MiB）
- 级联 enh：`D:/data/aec-synth/cascade_stage/eval_v7/enh_cascade`（FLOAT，390.7 MiB）
- 打分：`exp/07_nn_aec/outputs/eval_v7/{aecmos_trad_v7.json, aecmos_nn_best_v7.json, aecmos_cascade_v7.json}`（各 800 rows）
- CI：`exp/07_nn_aec/outputs/eval_v7/ci_eval_v7.json`（逐文件 mean/sd/CI + 12 组配对 Δ 的解析与 bootstrap CI）
- 指标：`exp/07_nn_aec/outputs/eval_v7/nn_metrics_v7/{summary.json,rows.csv,compare_trad_vs_nn.md}`、`metrics_compare_v7_vs_v5.json`
- 驱动：`exp/07_nn_aec/outputs/eval_v7/{ci_driver.py, metrics_compare.py}`
- 副产物（t12 用，本文不分析）：`…/eval_v7/aecmos_nn_last_v7.json`
