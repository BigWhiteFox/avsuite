# reports/ 索引 — 哪份报告是"当前结论"

> **所有对外指标以 [`docs/NUMBERS.md`](../docs/NUMBERS.md) 为唯一权威口径**;
> 本目录报告是评估过程的原始记录。引用数字前先核对 NUMBERS,若冲突以 NUMBERS 为准并回改对应报告。

## 当前有效(最新口径,可对外引用)

| 报告 | 主题 | 说明 |
|---|---|---|
| [`phase0_acceptance_20261013.md`](phase0_acceptance_20261013.md) | **阶段0 验收摘要（0-1…0-5）** | DP1 决策输入（是否租卡阶段1）；汇总数字一律以 NUMBERS 为准 |
| [`phase1_acceptance_20261015.md`](phase1_acceptance_20261015.md) | **阶段1 验收摘要（P1-1…P1-7）+ DP3 决策摘要** | 三代际登记、卡时/费用实测 vs 预算、负结果清单、DP3 建议（续租但主线 = sim-to-real + 数据边际收益）、级联 RIR 轴负方向假设清单 |
| [`eval_ns_rir_20261013.md`](eval_ns_rir_20261013.md) | **新评测轴 eval_ns / eval_rir（t16）** | held-out 噪声轴 / held-out 真实 RIR 轴的集合定义 + 命令 + 验收；模型读数待 t13 |
| [`multi_axis_matrix_20261013.md`](multi_axis_matrix_20261013.md) | **多轴评测集矩阵（t23）** | 音乐噪声/中文远端/SNR 域外/长语音 7 套新集 + t13 评测协议矩阵（配对/非配对逐轴标注）+ "v5 代际 eval_rir 轴不成立"的处置口径 |
| [`p1_6_matrix_eval_20261014.md`](p1_6_matrix_eval_20261014.md) | **P1-6 全套评测（t13 + t28 B 档）** | 8 模型（v4 / NN v5·v5x·noMUSAN / DCCRN v5·v5x / 级联 v5·v5x 正式版）+ MDF-only 消融 × 12 套冻结集：AECMOS + 波形域 + 稳态/暂态 ERLE 三分报 + 配对/非配对 CI + DiD |
| [`t12_dm_retrain_20261014.md`](t12_dm_retrain_20261014.md) | **P1-5 三模型 DM 重训收档（t12）** | v5/v5x 读数总表、消融可加性分解、四条口径警示（RIR 污染/epoch-len 默认值/w_sisnr 漏传/级联供给受限）、新服务器续跑 checklist |
| [`t24_cascade_official_20261014.md`](t24_cascade_official_20261014.md) | **级联正式版收档（t24）** | w_sisnr=1.0 + 100ep 两跑读数与验收、FEST 机理（MDF-only 消融口径）、截断版隔离说明 |
| [`eval_v8_isolation_20261013.md`](eval_v8_isolation_20261013.md) | **隔离版 eval_v8（0-4）** | 新冻结隔离集 + v4 基线/隔离偏差 DiD；引用须带"噪声轴对 v4 未隔离"强制说明 |
| [`p0_5_loss_matrix.md`](p0_5_loss_matrix.md) | P0-5 损失矩阵三列（0-1） | **负结果留档**（心理声学列 deg 增益不可复现；列③ skipped）；CI 明细 [`p0_5_loss_matrix_ci.md`](p0_5_loss_matrix_ci.md) |
| [`aecmos_rescore_v5fix.md`](aecmos_rescore_v5fix.md) | AECMOS@48k 8 集全量重评分 | 2026-10-11 口径修复后的主报告;修复前所有 AECMOS 合成集分数作废 |
| [`eval_v7_fest_nst.md`](eval_v7_fest_nst.md) | FEST/NST 800 条扩容评估 | 最新扩样口径 |
| [`DATA_CI.md`](DATA_CI.md) | 全仓 95%CI 汇总 | DT 场景 AECMOS 对外引用必须带 CI(工具 `tools/compute_ci.py`) |
| [`official_aecmos.md`](official_aecmos.md) | 官方 AECMOS 链路 | 官方 48k 链,未受 16k 口径问题影响 |
| [`ood_real_echoes.md`](ood_real_echoes.md) | OOD 真实回声 | 官方链路 |
| [`trad_hardening_rerun.md`](trad_hardening_rerun.md) | 传统管道语义加固复评 | P0~P3 整改后的复跑 |
| [`gtcrn_heldout200.md`](gtcrn_heldout200.md) | GTCRN held-out 200 重评 | 量化/对拍当前口径 |
| [`realrir_domain.md`](realrir_domain.md) | RealRIR 域适应 | 现行域适应结论 |

## 历史记录(口径已过时或已被取代,勿对外引用)

| 报告 | 状态 |
|---|---|
| [`eval_v5_dt300.md`](eval_v5_dt300.md) | DT 154 条稳定口径的早期版本,数字并入 DATA_CI |
| [`cascade_real_data.md`](cascade_real_data.md) | 级联真实数据定位(S11),结论已并入 NUMBERS |
| [`deg_mechanism.md`](deg_mechanism.md) | deg 机理早期叙事,"全场景结构性碾压"表述已作废 |
| [`delay_ood_probe.md`](delay_ood_probe.md) | 延迟 OOD 探针,一次性结论 |
| [`drift_retrain.md`](drift_retrain.md) / [`drift_sweep.md`](drift_sweep.md) | drift 探索矩阵,已收敛到 v4 主配置 |
| [`v4_rectification/`](v4_rectification/README.md) | V4 整改波次的过程档案(PLAN/HANDOFF/FINAL),已闭环 |

## 纪律

- 新报告命名:`<主题>_<日期>.md`,写明 checkpoint + 数据口径 + 工具版本;
- 报告完成后,若数字需对外,先更新 `docs/NUMBERS.md`,再回来在本索引登记;
- 禁止在报告里自造"最新最优"表述,一律指向 NUMBERS。
