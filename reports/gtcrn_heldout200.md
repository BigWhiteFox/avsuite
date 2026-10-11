# GTCRN held-out 扩样重评：50 条 → 200 条（WS-C 整改）

日期：2026-10-11 ｜ 模型：`exp/05a_gtcrn_onnx/outputs/gtcrn_dns3.sim.onnx`（fp32）与
`gtcrn_dns3.int8.onnx`（INT8 dynamic PTQ，QInt8 权重）。只评不训。

## 配方与同源性

- 同一配方：`DNSDynamicMix(seg_sec=4.0, SNR U(-5,15), level U(-35,-20) dBFS)`，seed 777。
- seed 机制为 `default_rng(seed*1000003 + idx)` 逐 idx 确定 → **扩样不换已有条目**：
  新 200 条集的前 50 条与旧 50 条集逐位一致（直接 `--num 200` 路线，无需换 seed 999）。
- 实测验证同源性：新集前 50 条 fp32 PESQ = 1.8694、INT8 = 1.8025、Δ = 0.0669，
  与旧报告 `quant_report_dynamic.json`（1.8694 / 1.8025 / 0.0669）**完全一致**。
- 合成目录：`exp/05a_gtcrn_onnx/outputs/heldout200_wav/`（前 50 条 noisy/clean wav 落盘备查）；
  旧 50 条合成是动态在线混合、无落盘目录，未被覆盖。

## PESQ(wb) 对比

| 集合 | fp32 mean±sd [95%CI] | INT8 dynamic mean±sd [95%CI] |
|---|---|---|
| 旧 50 条（新集前 50 子集，同旧集逐位一致） | 1.869 ± 0.517 [1.726, 2.013] | 1.803 ± 0.455 [1.676, 1.929] |
| **新 200 条** | **1.897 ± 0.562 [1.820, 1.975]** | **1.845 ± 0.539 [1.770, 1.920]** |

（附 STOI / SI-SNR，200 条：fp32 0.880 / 11.94 dB；INT8 0.874 / 11.83 dB。）

## 量化损耗 Δ = PESQ(fp32) − PESQ(INT8)

| 集合 | Δ mean±sd | Δ 95%CI |
|---|---|---|
| 50 条 | 0.067 ± 0.269 | [−0.008, 0.141]（含 0，不显著） |
| **200 条** | **0.053 ± 0.178** | **[0.028, 0.077]**（显著 > 0） |

## 结论

扩样后 **INT8 dynamic "可用" 结论维持**：200 条上量化损耗收紧为 Δ = 0.053
[0.028, 0.077]，显著但幅度小（<0.06 PESQ），INT8 PESQ 1.845 [1.770, 1.920]
与 fp32 的 CI 大部分重叠、绝对差约 3%；原 50 条报告的 1.869/1.803 点估计在
200 条 CI 内，未因扩样翻车。

## 产物与口径备注

- 逐条明细：`exp/05a_gtcrn_onnx/outputs/heldout200/rows.csv`
- 汇总 JSON：`exp/05a_gtcrn_onnx/outputs/heldout200/summary.json`
- 评估脚本：`exp/05a_gtcrn_onnx/eval_heldout200.py`（本次新增；未改任何既有文档/outputs）
- 已知瑕疵：旧 `quant_accuracy_rows.csv` 后来被 static 崩坏复跑覆盖（现为 static 逐条
  数据），dynamic 的旧逐条明细已不可考——故"旧 50 条"统计以新集前 50 子集为准
  （与旧集逐位一致，且其均值与旧 JSON 报告完全对上）。
