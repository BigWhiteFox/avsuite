# trad_hardening_rerun — S10 加固版传统管道重跑三处受污染 headline（V4 整改第四波）

- 日期：2026-10-11 · 执行：S10（v4-rect-wave4 / s10-rerun）· 驱动：`exp/07_nn_aec/hardening_rerun.py`（新增）
- 加固配置（HANDOFF §4/§6.1）：`TradAEC(method='mdf', partitions=8, mu=0.7, align_margin_ms=25, reestimate_sec=2.0)`，其余默认（realign_min_delta_ms=5 / reestimate_win_s=2s / realign_search_pad_ms=150 / mdf_w_abs_cap=1e3）
- 数据：`D:/data/aec-synth/eval_v3_drift`（60 条：FEST 36 / DT 18 / NST 6；delay_drift σ=5ms/s、限幅 ±10ms 慢随机游走）、官方集 `farend-singletalk` 全 300 条、`D:/data/aec-synth/eval_v6_realrir`（100 条：34/42/24）
- 口径：AECMOS 沿用 `aecmos_eval.py --enh-dir`（16k 三路直送 48k 模型，仓内既有口径）；官方集按官方 proto 协议（FEST 送后半段，复用 `official_aecmos_eval.py` 的 `trim_for`/`CachedAECMOS`）；ERLE 用 `eval_aec.py` 口径（10log10(P_mic/P_out)，skip 2s）；CI = 1.96·sd/√n
- **口径校验（先于结论）**：用本驱动以旧配置（margin=0、无重估）复跑漂移集与 eval_v6，逐条配对差 **Δ=0.000（逐位复现旧 json：漂移 FEST 2.119/DT 1.649；v6 FEST 3.301/DT 2.000/ERLE 7.948=NUMBERS 7.95）**——本报告所有加固列与旧数字可直接比。

## TL;DR（三个问题三个答案）

1. **漂移集：修复后没有翻转为"可跟"**——FEST echo 2.119→**1.787**（配对 Δ−0.332±0.209，反而显著略降；判据"回升 ≥1.0"未达成），DSP 级 FEST ERLE 仅 2.0-3.1dB（静态 20.55dB）。崩塌主导机制**不是**"零余量对齐→非因果泄漏"（margin=25 后无任何回升），与 S1 单测"块内路径移动 × 块级自适应跟踪代价"归因一致。
2. **官方集 FEST：均值不动、发散条修复**——全 300 条 proto echo 2.759±0.129 vs 旧 2.725±0.126（同 299 条配对 Δ+0.028±0.067，CI 含 0 不显著）；此前唯一发散条 wufNfJIb3kC7OvuxutARLQ…movement 从"输出含 Inf、无法送评"变为**输出全有限 + echo 4.64**（发散防护 6 次复位、5 次再对齐），全列 0 条非有限输出。
3. **eval_v6 真实车厢 RIR：ERLE 回收 +2.35dB，几乎全部来自对齐余量**——FEST ERLE 7.948→**10.296±2.304**（配对 Δ+2.35±1.59，显著，19/34 条改善；margin 单独 +2.12、reestimate 单独 +0.06）。崩落 12.2dB 中对齐余量贡献 ≈2.1-2.3dB（**≈19%**），其余 ~10dB 为真实 RIR 域差本征。AECMOS（FEST 3.301→3.310、DT 2.000→1.979）与 DT PESQ（1.525→1.540）基本不变。

---

## 1. ① 漂移集（eval_v3_drift，60 条）

### 表 1-1 AECMOS（aecmos_eval 口径，mean±1.96·sd/√n）

| 列 | FEST echo（n=36） | DT echo（n=18） | NST echo（n=6） |
|---|---|---|---|
| 旧漂移（margin=0，无重估；=旧 aecmos_trad_drift.json） | 2.119±0.37 | 1.649±0.16 | 5.0 |
| **加固（margin=25+reest=2s，S10 主配置）** | **1.787±0.267** | 1.651±0.166 | 5.0 |
| 仅 margin=25 | 2.023±0.349 | 1.660±0.171 | 5.0 |
| 仅 reestimate=2s | 1.958±0.317 | 1.629±0.137 | 5.0 |
| 无漂移基线（aecmos_trad_driftbase.json） | 4.011±0.29 | 2.451±0.34 | 5.0 |

配对差（同条逐对，Δ=加固−旧）：FEST **−0.332±0.209（显著恶化）**；仅 margin −0.096±0.177（不显著）；仅 reest −0.160±0.202（不显著）；DT 全部 |Δ|≤0.02（不显著）。旧配置复跑列配对 Δ=0.000（口径校验）。

### 表 1-2 DSP 级 FEST ERLE（eval_aec 口径，skip 2s；静态无漂移参照 20.55dB）

| 列 | FEST ERLE（n=36） |
|---|---|
| 旧漂移 | 3.121±1.427 dB |
| 仅 margin | 2.574±1.224 dB |
| 仅 reestimate | 2.810±1.160 dB |
| **加固** | **2.044±1.009 dB** |

### 判据与元数据

- 任务判据"FEST echo 较旧漂移数字回升 ≥1.0（理想接近基线 4.011）"：**未达成**（实际 −0.332，方向相反）。
- 加固列 meta（60 条）：周期重估触发再对齐 **91 次**（27 条 2 次 / 16 条 1 次 / 7 条 3 次），MDF 发散防护复位 **0 次**；align_ms 均值 28.4（0-72.9）；初始时延估计均值 48.7ms（0-97.9）。
- 处理耗时：60 条 ≈4s（6 进程并行）；打分 ≈6s。

## 2. ② 官方集：发散条 + FEST 全 300

### 2.1 发散条单条断言（wufNfJIb3kC7OvuxutARLQ_farend-singletalk-with-movement，48k→resample_poly(1/3)→16k，n=382,400 样本）

| 项 | 旧行为（margin=0，无重估） | **加固行为** |
|---|---|---|
| 输出有限性 | **非有限**（divergence_scan.json：Infinity × 173,816 样本；送评报 "Audio buffer is not finite everywhere"） | **全有限 ✓**（max\|out\|=35.6） |
| AECMOS proto | failed（无法送评） | **echo 4.6388 / deg 4.9991**（同条旧 NN 列仅 1.7591） |
| n_divergence_resets | （机制不存在） | **6** |
| n_realigns / align_ms / 时延 | （机制不存在） | **5 次 / 12.56ms / 37.56ms**（23.9s 内时延 12 个估计点、路径大幅移动） |

### 2.2 FEST 全 300 条（官方 proto 协议：送后半段）

| 列 | n | FEST echo | FEST deg | 失败 |
|---|---|---|---|---|
| 旧 trad（reports/official_aecmos/summary.json） | 299 | 2.725±0.126 | 5.0 | 1（发散条） |
| **加固 trad_hard（S10）** | **300** | **2.759±0.129** | 4.999 | **0** |
| 加固（剔除发散条，同 299 条口径） | 299 | 2.753±0.129 | — | — |

- 同名 299 条配对 Δ=**+0.028±0.067**（CI 含 0，不显著）——加固对官方 FEST 均值中性；min/median/max = 1.20/2.40/4.78。
- 加固列 meta（300 条）：MDF 发散防护复位共 **10 次 / 4 条**（发散条自身 6 次 + 另 3 条共 4 次），周期重估再对齐共 516 次，align_ms 均值 25.5（0-95），**非有限输出 0 条**。旧列仅发散条 1 条即耗尽 float64——发散防护以 <1% 触发率（4/300）兜底。
- 处理耗时：300 条 ≈53s（6 进程，RTF≈0.004/条含重估）；打分 300 条 ≈22s。

## 3. ③ eval_v6 真实车厢 RIR（100 条）

### 表 3-1 AECMOS（aecmos_eval 口径）

| 列 | FEST echo（n=34） | DT echo（n=42） | NST echo（n=24） | DT deg（n=42） |
|---|---|---|---|---|
| 旧（=旧 aecmos_trad_v6.json） | 3.301±0.43 | 2.000±0.15 | 5.0 | 2.518±0.15 |
| **加固** | 3.310±0.474 | 1.979±0.131 | 5.0 | 2.503±0.143 |
| 配对 Δ（加固−旧） | +0.010±0.179 | −0.020±0.072 | 0.000 | −0.015±0.052 |

（全部不显著；旧配置复跑列配对 Δ=0.000 逐位复现。）

### 表 3-2 FEST ERLE（eval_aec 口径，skip 2s）——问题 (b) 的核心分解

| 列 | FEST ERLE（n=34） | 配对 Δ vs 旧 | DT PESQ（n=42） |
|---|---|---|---|
| 旧（margin=0，无重估） | 7.948±2.077 dB（=NUMBERS 7.95） | — | 1.525 |
| 仅 margin=25 | 10.063±2.337 dB | **+2.115±1.553（显著）** | 1.566 |
| 仅 reestimate=2s | 8.008±2.060 dB | +0.06（不显著） | 1.515 |
| **加固（margin+reest）** | **10.296±2.304 dB** | **+2.347±1.594（显著，19/34 条改善）** | 1.540 |

- 加固列 meta（100 条）：再对齐 112 次，发散防护复位 0 次，align_ms 均值 49.3（0-94.9）。
- 处理 + 打分 + ERLE/PESQ 合计每列 ≈35s（6 进程）。

## 4. 两个问题的直接回答

### (a) "线性 AEC 漂移崩塌"修复后是否翻转为"可跟"？——**否。**

FEST echo 2.119→1.787（配对 Δ−0.332±0.209，显著略降），FEST ERLE 3.12→2.04dB，距无漂移基线 4.011/静态 20.55dB 的差距一点没有收窄。margin=25（几何上保证残余路径因果）与周期重估（±10ms 游走下 1.5 次/条再对齐）都没有带来回升，两个修复各自单独还各亏 −0.10/−0.16（不显著），合起来 −0.33（显著）——再对齐复位 MDF/DTD 的暖机代价在 8s 短条上是净负项（60 条 91 次复位 ≈ 每条损失 ~1/4 收敛时间）。**漂移崩塌的主导机制不是"零余量对齐→非因果泄漏"的实现缺陷**：若它是主因，margin=25 应回收明显 ERLE（实测无）；这与 S1 噪声激励单测的预警（"margin=25 也只恢复到 5.5dB"）和 README 附记第 7 条（GCC 对真实/多径时延的固有偏差族）一致——**"块内路径移动 × 块级自适应"的跟踪代价才是主通道**（±10ms/8s ≈ 0.36 样本/块的连续移动，块级 MDF 跟不上；DTD 误冻与插值副产物为次要候选，本波未单独消融）。

### (b) 真实 RIR 下传统 ERLE 崩落有多少是零余量对齐贡献的？——**≈2.1-2.3dB，占崩落 12.2dB 的 ≈19%；其中几乎全部来自 align_margin_ms，周期重估贡献 ≈0。**

eval_v5(仿真)→eval_v6(真实) 的 FEST ERLE 崩落 20.18→7.95dB；加固后 10.30dB（配对 +2.35±1.59 显著，19/34 条改善）。消融分解：仅 margin +2.12（显著），仅 reest +0.06（≈0）——即回收量 ≈ 全部来自把残余直达径从 lag 0 挪进滤波窗内 25ms 的对齐余量（真实 RIR 下 GCC 时延偏差使零余量对齐的残余路径常落在窗沿/窗外，margin=25 修复了这一部分）；周期重估在 8s 无路径移动的静态条上无事可做（112 次再对齐多为 ≤5ms 假峰抖动边缘）。**剩余 ~10dB 是对齐修复无法触及的真实 RIR 域差本征**：300ms RIR 截断超出 2048-tap（128ms）滤波窗、车厢短混响强早反射的时变与非线性成分、20-30dB 噪声地板封顶等（README 局限节 + realrir_domain.md 结论一致）。近端指标不动（DT PESQ 1.525→1.540、DT echo 2.000→1.979）说明对齐余量只作用于回声抑制通道、不伤近端。

## 5. 解读与归因（对照 exp/06_trad_aec/README.md 附记与 S1 单测）

1. **三处 headline 的"污染"性质不同，修复效果相应不同**：官方集发散条是**实现级缺陷**（系数几何增长耗尽 float64）→ 发散防护+对齐余量+周期重估直接治愈（failed→4.64，300/300 可评）；eval_v6 的 ERLE 崩落含一个**可修复的对齐分量**（~19%）+ 域差本征（~81%）；漂移集崩塌是**算法级跟踪代价**（非本波四件套的可修范围）→ 修复后仍崩。
2. **对 HANDOFF §4 预设的修正**：S1 单测在噪声激励下发现 margin=25 只恢复部分 ERLE（5.5dB vs 静态 46dB），本波真实数据证实其外推——真实语音+RIR 下 margin 的净收益在漂移集为**零**（AECMOS −0.10 n.s. / ERLE −0.55dB），在无漂移的 v6 为 **+2.1dB**。即 margin 修复的是"静态/慢变路径下的对齐几何余量"，不是"路径移动的跟踪能力"。
3. **周期重估的双面性**：对大跳变（单测 +130ms）是救命机制（gap≈46dB）；对 8s 短条上的慢游走是净负担（每次再对齐复位 MDF/DTD，暖机损失 > 对齐收益）。生产化需要更保守的触发（如 Δ≥5ms → 提高到 ≥10ms 或加确认窗）——本报告数据支持此方向，但属后续工作。
4. **对旧结论的复核结论**：第三波/复评质疑的三个 headline 中，"官方集 1/800 发散"已修复归零（本波 300 条 FEST 0 非有限）；"传统管道漂移崩塌（FEST −1.89）"**维持成立**（修复不翻转，主因改判为跟踪代价）；"真实 RIR 传统 ERLE 崩落（−12.2dB）"**量级维持**，其中 ~2.3dB 属对齐实现缺陷、已回收，其余为域差本征。旧报告正文无需回改（只加本附记级证据），与"历史报告不改正文"策略一致。

## 6. 重跑命令（全；仓库根 D:\Project_for_Python\DSH\avsuite）

```powershell
$env:PYTHONNOUSERSITE='1'
$py = 'C:\Users\14805\.conda\envs\avsuite\python.exe'

# ① 漂移集（S10 主配置；断点续：wav/jsonl 已存在即跳过）
& $py exp\07_nn_aec\hardening_rerun.py drift --workers 6 --variant hard
# ①' 归因消融 + 口径校验（old 列应逐位复现旧 aecmos_trad_drift.json）
& $py exp\07_nn_aec\hardening_rerun.py drift --workers 4 --variant old
& $py exp\07_nn_aec\hardening_rerun.py drift --workers 4 --variant margin
& $py exp\07_nn_aec\hardening_rerun.py drift --workers 4 --variant reest
# ①" DSP 级 FEST ERLE（从已落盘 enh wav 计算）
& $py exp\07_nn_aec\hardening_rerun.py drift-erle --variant old   # margin/reest/hard 同型

# ② 发散条单条断言 → FEST 全 300 处理 → proto 打分 → 聚合
& $py exp\07_nn_aec\hardening_rerun.py official-one
& $py exp\07_nn_aec\hardening_rerun.py official-infer --workers 6
& $py exp\07_nn_aec\hardening_rerun.py official-score --workers 6
& $py exp\07_nn_aec\hardening_rerun.py official-agg

# ③ eval_v6（hard 主配置 + old 口径校验 + margin/reest 归因）
& $py exp\07_nn_aec\hardening_rerun.py v6 --workers 6 --variant hard
& $py exp\07_nn_aec\hardening_rerun.py v6 --workers 6 --variant old
& $py exp\07_nn_aec\hardening_rerun.py v6 --workers 6 --variant margin
& $py exp\07_nn_aec\hardening_rerun.py v6 --workers 6 --variant reest
```

## 7. 落盘文件清单（全部新建，未改既有文件）

- 驱动：`exp/07_nn_aec/hardening_rerun.py`
- 打分/指标 JSON：`exp/07_nn_aec/outputs/hardening_rerun/` 下
  - `divergence_clip_hard.json`（②发散条断言+delay_track 全记录）
  - `aecmos_trad_hard_drift.json`、`aecmos_trad_hard_drift_{old,margin,reest}.json`（含逐条 rows + 配对 vs 旧）
  - `erle_drift_{old,margin,reest,hard}.json`（漂移集 FEST ERLE 逐条+汇总）
  - `aecmos_trad_hard_official_fest.json`、`scores_trad_hard_official_fest.jsonl`、`meta_official_fest.jsonl`（②：300 条打分行+处理 meta）
  - `aecmos_trad_hard_v6.json`、`aecmos_trad_hard_v6_{old,margin,reest}.json`、`trad_hard_rows_v6*.csv`（列同旧 trad_rows_v6.csv）、`erle_pesq_v6*.jsonl`、`meta_v6*.jsonl`、`scores_trad_hard_v6*.jsonl`
- 增强 wav（断点续产物）：`outputs/hardening_rerun/enh_trad_hard_drift*/`（4×60 条）、`enh_trad_hard_v6*/`（4×100 条，16k FLOAT）
- 官方集增强 wav（仓外）：`D:/data/aec-synth/official_stage/enh_trad_hard/fest/*.wav`（300 条；旧 `enh_trad/` 未动）

## 8. 局限

- 漂移集仅覆盖 σ=5ms/s（限幅 ±10ms）慢游走一种幅度；>128ms 大跳变出窗场景未在本集（单测 test_reestimate_recovers_large_delay_jump 已覆盖：reestimate 在该场景恢复 ~46dB）。
- margin=25 为单一取值，未做扫描；realign_min_delta_ms=5 未消融（触发灵敏度见 §5.3）。
- NST echo AECMOS 恒 5.0（模型上限），对 NST 无区分力；本报告 NST 列仅作完整性。
- AECMOS 打分为仓内既有 16k 直送口径（与旧数字同口径可比，绝对值与官方 48k 协议有口径差）；官方列已按官方 proto 裁剪送评。
- "剩余 ~10dB 域差"的成分（滤波窗 128ms<300ms RIR、时变/非线性、噪声地板）未在本波逐项消融，属定性归因。
