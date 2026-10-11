# V4 整改 FINAL.md — 第四波（复评驱动修复）收口报告

- 写入：2026-10-11 16:25（+08:00；窗口截止 ~17:30 内完成）；写人：本轮 Lead（决策/验收/S12 亲自收口）
- 执行模式：AgentTeams 团队 `v4-rect-wave4`——3 成员并行（S2/S10/S11，任务书=HANDOFF §6.0–6.2 原文+环境块内嵌）；
  三源 API 之 ark-volcengine/GLM-5.3-Flash 一路跑通**零失败**（未触发换源备份 ark-codingplan/deepseek-official）；
  Lead 预检（上波代理零残留✅ / S1 参数与数据前置在位✅ / 基线 pytest 绿✅）→ 派发 → 亲自验收 → 亲自 S12
- 唯一权威交接入口：`reports/v4_rectification/HANDOFF.md`（本轮后 §5 状态表全 ✅）

## 1. 三任务交付与 Lead 验收

| 任务 | 成员/耗时 | 交付物 | Lead 验收方式与结论 |
|---|---|---|---|
| S2 评估工具补单测 | s2-tests / ~40min | `tests/test_official_protocol.py`（11 例）/ `test_compute_ci.py`（7 例）/ `test_precompute_semantics.py`（6 例） | 全量复跑 exit 0 + 收集计数 **124**=基线 100+新增 24；单文件 0.6–1.8s（<30s）；零 D:/data/网络依赖（grep 命中均为 docstring 对"import 无 IO"的断言）；compute_ci.py 未动；**✅ 通过** |
| S10 加固管道重跑 | s10-rerun / ~35min | `reports/trad_hardening_rerun.md` + `exp/07_nn_aec/hardening_rerun.py` + `outputs/hardening_rerun/**`（40 json/csv + 8 enh 目录）+ 仓外 `enh_trad_hard` 300 wav | 从逐条 jsonl/json **独立复算**：漂移配对 −0.3319、v6 ERLE 四列 7.948/10.296/10.063/8.008、配对 +2.347 且 19/34 改善、旧配置复现 max\|Δ\|=4.9e-05——全部与报告一致；git 零既有文件修改；**✅ 通过**（判据①"回升≥1.0"未达成=科学结论，见 §2） |
| S11 级联真实数据 | s11-cascade / ~25min | `reports/cascade_real_data.md` + `exp/09_cascade_aec/cascade_real_data.py` + `results/aecmos_cascade_{official,v6}.json` + scores jsonl + 仓外 lin/enh wav 2700 条 | 从 rows **独立重算**官方集/v6 全部均值逐位一致；paired bootstrap 块与报告表一致；拒评 2/800 与 5 条复位 13 次如实；线性级铁律核对（margin/reestimate 关、防护开、float64→float32、复用官方 trim_for/AECMOS）；**✅ 通过**（补记：v6 FEST 级联−NN −0.313 [−0.692,+0.081] n.s. 报告表未列此行，"级联≈NN"结论不变，NUMBERS 已补记） |

验收纪律（HANDOFF §8.4 执行）：三份报告的每个 headline 数字均由 Lead 从源头逐条数据复算，**不只看成员自检**；
两名成员的"诚实披露"（S10 判据未达成、S11 拒评 2/800）均被保留并升格进 NUMBERS 权威口径。

## 2. 本波关键数字（全部已入 `docs/NUMBERS.md`）

### S10 加固重跑（margin=25ms + reestimate=2s + 发散防护；partitions=8/μ=0.7 不变）
| 处 | 旧 | 加固 | 配对结论 |
|---|---|---|---|
| 漂移 FEST echo（60 条） | 2.119±0.37 | 1.787±0.267 | Δ−0.332±0.209 显著略降；判据"回升≥1.0"未达成 → **主因改判"块内路径移动×块级自适应跟踪代价"**（margin/reestimate 消融均无回升） |
| 官方 FEST echo（proto 300） | 2.725±0.126（n=299，1 条发散拒评） | 2.759±0.129（n=300，**0 非有限**） | 同 299 条配对 +0.028±0.067 n.s.（均值中性）；**发散条 failed→echo 4.64**（防护复位 6 次） |
| v6 FEST ERLE（34 条） | 7.948 dB | 10.296±2.304 dB | +2.347±1.594 显著，19/34 改善；消融 margin 单独 +2.115 / reest 单独 +0.059 → **对齐余量解释 ≈19% 崩落**，其余 ~10dB 为真实 RIR 域差本征 |
| v6 AECMOS FEST/DT echo | 3.301 / 2.000 | 3.310 / 1.979 | 均 n.s.；DT PESQ 1.525→1.540（ERLE 收益不伤近端） |

### S11 级联真实数据首验（线性级逐位复刻训练分布；NN=exp/09_cascade_aec/outputs/nnaec_g0.5_j10/best.ckpt）
| 集·指标 | 传统 | NN v4 | **级联（新）** | 级联−传统 | 级联−NN |
|---|---|---|---|---|---|
| 官方 FEST echo | 2.725±0.126 | 3.143±0.151 | **3.096±0.135** (n=299) | **+0.371 [+0.295,+0.448] 显著** | −0.052 n.s. |
| 官方 DT echo | 2.305±0.100 | 3.256±0.124 | **3.233±0.122** | **+0.932 [+0.847,+1.016] 显著** | −0.027 n.s. |
| 官方 DT deg | 3.692±0.088 | 3.097±0.118 | **2.947±0.103** | −0.751 显著（更差） | −0.151 显著（略差） |
| 官方 NST deg | 4.183±0.046 | 3.759±0.119 | **3.947±0.083** | −0.235 显著 | **+0.188 显著（优于 NN）** |
| v6 DT echo | 2.000±0.15 | 2.565±0.23 | **2.627±0.241** | **+0.628 [+0.407,+0.858] 显著** | +0.062 n.s. |
| v6 DT deg | 2.518±0.15 | 2.044±0.12 | **1.842±0.090** | −0.676 显著（更差） | −0.202 显著（略差） |
| v6 FEST echo | 3.301±0.43 | 3.718±0.45 | **3.406±0.428** (n=34) | +0.105 n.s.（CI 宽） | −0.313 n.s. |

**披露**：级联拒评 **2/800**（传统 1/800、NN 0/800）——wufNfJIb…movement 线性级靠 S1 防护保持有限（8 次复位）
但 NN 端峰值 4821.5 过 1e3 防线拒送评（with-movement 失败模式从线性级 inf **后移至 NN 级**）；
GSJyjeKff…movement（DT）新增拒评。bootstrap B=2000 seed=777。

### S2 / 全仓测试
- 全仓 pytest **124 passed / 0 failed / 0 skipped**（=93 基线 + S1 加固 7 + S2 评估工具 24；S2 junitxml 精确计数 + Lead 复跑 exit 0 + 收集 124 确认）
- 发现不修留档：`tools/compute_ci.py` `paired_bootstrap(data)` 形参未用（模块级路径耦合），测试 docstring 已注明

## 3. 四条缺陷线收口状态（对应 HANDOFF §1）

| 线 | 缺陷 | 本波结果 | 状态 |
|---|---|---|---|
| A | 漂移/Kalman 两个"证伪"headline 设计偏置 | S10：漂移崩塌维持（主因改判跟踪代价）；发散条治愈；v6 ERLE +2.35dB（≈19% 对齐贡献）；Kalman 边界=ADR 0006（上波 Lead） | ✅ 收口（旧报告正文未动，仅附记+NUMBERS 改写） |
| B | 级联（生产推荐）零真实数据验证 | S11：官方集+eval_v6 首份级联真实数据证据——echo 增益真实分布保持（≈NN、显著优于传统）；deg 交易照旧；2/800 拒评披露 | ✅ 收口 |
| C | 文档双层真相 | NUMBERS 回填 6 处（S5 机理节/S10 节/S11 节/Kalman 指针/漂移③改写/单测数 93→124）；上波 S3/S4/S9+Lead 修正 | ✅ 收口 |
| D | 治理缺口（git/测试/ADR/豁免） | S2 补测 24 例（全仓 124 passed）；本波 git 3 批本地提交；ADR 0006（上波） | ✅ 收口 |

## 4. 用户决策项（HANDOFF §9 原样呈现 + 本波事实增注）

1. （§9.1）漂移/真实 RIR 修复重跑若翻转旧结论，旧报告是否需进一步处理——**本波事实：未翻转（方向维持）**，
   维持"只加附记不改正文"，无需进一步处理。
2. （§9.2）级联真实数据无论好坏，"生产推荐级联"话术是否随数字调整——**本波事实：echo/deg 相对结论支持保留
   话术，但须随披露"级联拒评 2/800 + with-movement 失败点后移至 NN 端 + deg 交易结构"**；是否调整表述由用户定。
3. （§9.3）near-end gating（近端门控）训练仍属下一窗口（DECISIONS 决策 6 P1），本波未启动。
4. （§9.4）P2-3（多麦波束成形）豁免已补记录（PLAN 已含），是否排下窗口由用户定。
5. （§9.5）git 提交是否推送远端——本波延续策略：**只本地提交不推送**。
6. （§9.6）outputs_v4_drift*/ 两个训练目录按 .gitignore 不入库，是否另存档由用户定；本波新增 enh/中间 wav
   全部仓外（D:/data）或在 outputs/ 全局忽略规则下不入库（hardening_rerun 312MB wav），是否存档同样由用户定。
7. （本波新增）with-movement 病条根治的生产化参数（周期重估触发更保守：Δ≥10ms 或加确认窗——S10 数据支持方向）
   是否排下窗口。
8. （本波新增）`reports/DATA_CI.md` 再生成（S11 results json 已兼容 tools/compute_ci.py 口径，可自动纳入
   官方集/v6 级联列）——本波未做（超出 HANDOFF §6.3 范围），是否补做由用户定。

## 5. 遗留清单

1. `tools/compute_ci.py`：paired_bootstrap 形参未用/模块级路径耦合（S2 留档未修——任务书"发现 bug 记录不修"约束）。
2. 周期重估触发保守化（realign_min_delta_ms=5 → ≥10ms/确认窗）：S10 数据支持方向，未实现（属 06 管道行为变更，须新窗口+回归）。
3. 漂移集仅覆盖 σ=5ms/s 慢游走（±10ms 限幅）；>128ms 大跳变出窗仅 S1 单测覆盖（reestimate 恢复 ~46dB）。
4. margin=25 单值未扫描；DTD 误冻/插值副产物未单独消融（次要候选）。
5. v6 FEST n=34 CI 宽（级联−传统 +0.105 n.s.、级联−NN −0.313 n.s.）——扩样才能收紧。
6. NST AECMOS 恒 5.0（模型上限），对 NST 无区分力。
7. S10 报告称旧配置"逐位复现"，实测 max|Δ|=4.9e-05（AECMOS 浮点噪声级）——数字结论不受影响，表述从严记录。
8. INT8 static QDQ 崩坏数字仍为旧 50 条口径（未随 heldout200 重评，上波遗留）。

## 6. git 批次（本轮，全部本地未推送；hash 以 `git log` 为准）

| 批次 | 内容 |
|---|---|
| ② | [V4整改] 第四波代码与测试：hardening_rerun.py + cascade_real_data.py 驱动 + S2 评估工具 24 例单测（全仓 124 passed） |
| ③ | [V4整改] 第四波文档：NUMBERS 权威口径回填 6 处 + HANDOFF 台账收口 |
| ④ | [V4整改] 第四波报告与数据：trad_hardening_rerun / cascade_real_data / FINAL + 级联 results JSON/JSONL |

（enh/中间 wav 一律未入库：outputs/ 全局忽略 + 仓外 D:/data）
