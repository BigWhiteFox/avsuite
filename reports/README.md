# reports/ 索引 — 哪份报告是"当前结论"

> **所有对外指标以 [`docs/NUMBERS.md`](../docs/NUMBERS.md) 为唯一权威口径**;
> 本目录报告是评估过程的原始记录。引用数字前先核对 NUMBERS,若冲突以 NUMBERS 为准并回改对应报告。

## 当前有效(最新口径,可对外引用)

| 报告 | 主题 | 说明 |
|---|---|---|
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
