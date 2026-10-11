# DATA_CI.md — AECMOS 指标 95% 置信区间（自动生成）

- 生成日期：2026-10-11
- 生成命令：`python tools/compute_ci.py`（仓库根 avsuite/ 下）
- 方法：解析法 95%CI = mean ± 1.96·sd/√n；配对差值用 bootstrap 95%CI（B=2000，seed=777，百分位法）
- 数据源：`exp/07_nn_aec/outputs/**/aecmos_*.json`、`exp/09_cascade_aec/results/aecmos_*.json` 的逐文件 rows

## 一、逐文件解析 95%CI（1.96·sd/√n）

| 文件 | 场景 | 指标 | n | mean | sd | 95%CI |
|---|---|---|---|---|---|---|
| exp/07_nn_aec/outputs/aecmos_leadincrop/aecmos_dt_crop_t0s.json | dt | echo_mos | 14 | 3.754 | 0.572 | 3.754±0.30 |
| exp/07_nn_aec/outputs/aecmos_leadincrop/aecmos_dt_crop_t0s.json | dt | deg_mos | 14 | 1.759 | 0.292 | 1.759±0.15 |
| exp/07_nn_aec/outputs/aecmos_leadincrop/aecmos_dt_crop_t0s_run1.json | dt | echo_mos | 14 | 3.754 | 0.572 | 3.754±0.30 |
| exp/07_nn_aec/outputs/aecmos_leadincrop/aecmos_dt_crop_t0s_run1.json | dt | deg_mos | 14 | 1.759 | 0.292 | 1.759±0.15 |
| exp/07_nn_aec/outputs/aecmos_leadincrop/aecmos_dt_crop_t1s.json | dt | echo_mos | 14 | 3.113 | 0.723 | 3.113±0.38 |
| exp/07_nn_aec/outputs/aecmos_leadincrop/aecmos_dt_crop_t1s.json | dt | deg_mos | 14 | 1.984 | 0.318 | 1.984±0.17 |
| exp/07_nn_aec/outputs/aecmos_leadincrop/aecmos_dt_crop_t2s.json | dt | echo_mos | 14 | 2.711 | 0.925 | 2.711±0.48 |
| exp/07_nn_aec/outputs/aecmos_leadincrop/aecmos_dt_crop_t2s.json | dt | deg_mos | 14 | 2.077 | 0.426 | 2.077±0.22 |
| exp/07_nn_aec/outputs/aecmos_leadincrop/aecmos_dt_crop_t3s.json | dt | echo_mos | 14 | 2.776 | 0.890 | 2.776±0.47 |
| exp/07_nn_aec/outputs/aecmos_leadincrop/aecmos_dt_crop_t3s.json | dt | deg_mos | 14 | 1.942 | 0.337 | 1.942±0.18 |
| exp/07_nn_aec/outputs/aecmos_nn_drift.json | dt | echo_mos | 18 | 3.802 | 0.830 | 3.802±0.38 |
| exp/07_nn_aec/outputs/aecmos_nn_drift.json | dt | deg_mos | 18 | 1.809 | 0.253 | 1.809±0.12 |
| exp/07_nn_aec/outputs/aecmos_nn_drift.json | fest | echo_mos | 36 | 4.353 | 0.596 | 4.353±0.19 |
| exp/07_nn_aec/outputs/aecmos_nn_drift.json | fest | deg_mos | 36 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_drift.json | nst | echo_mos | 6 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_drift.json | nst | deg_mos | 6 | 2.259 | 0.476 | 2.259±0.38 |
| exp/07_nn_aec/outputs/aecmos_nn_driftbase.json | dt | echo_mos | 18 | 3.993 | 0.759 | 3.993±0.35 |
| exp/07_nn_aec/outputs/aecmos_nn_driftbase.json | dt | deg_mos | 18 | 1.791 | 0.276 | 1.791±0.13 |
| exp/07_nn_aec/outputs/aecmos_nn_driftbase.json | fest | echo_mos | 36 | 4.110 | 0.838 | 4.110±0.27 |
| exp/07_nn_aec/outputs/aecmos_nn_driftbase.json | fest | deg_mos | 36 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_driftbase.json | nst | echo_mos | 6 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_driftbase.json | nst | deg_mos | 6 | 2.259 | 0.476 | 2.259±0.38 |
| exp/07_nn_aec/outputs/aecmos_nn_night.json | dt | echo_mos | 14 | 3.562 | 0.655 | 3.562±0.34 |
| exp/07_nn_aec/outputs/aecmos_nn_night.json | dt | deg_mos | 14 | 1.910 | 0.322 | 1.910±0.17 |
| exp/07_nn_aec/outputs/aecmos_nn_night.json | fest | echo_mos | 26 | 4.728 | 0.068 | 4.728±0.03 |
| exp/07_nn_aec/outputs/aecmos_nn_night.json | fest | deg_mos | 26 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_night.json | nst | echo_mos | 20 | 5.000 | 0.001 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_night.json | nst | deg_mos | 20 | 2.163 | 0.465 | 2.163±0.20 |
| exp/07_nn_aec/outputs/aecmos_nn_v2.json | dt | echo_mos | 14 | 3.277 | 0.801 | 3.277±0.42 |
| exp/07_nn_aec/outputs/aecmos_nn_v2.json | dt | deg_mos | 14 | 2.014 | 0.399 | 2.014±0.21 |
| exp/07_nn_aec/outputs/aecmos_nn_v2.json | fest | echo_mos | 26 | 4.605 | 0.226 | 4.605±0.09 |
| exp/07_nn_aec/outputs/aecmos_nn_v2.json | fest | deg_mos | 26 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_v2.json | nst | echo_mos | 20 | 5.000 | 0.001 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_v2.json | nst | deg_mos | 20 | 2.197 | 0.472 | 2.197±0.21 |
| exp/07_nn_aec/outputs/aecmos_nn_v3.json | dt | echo_mos | 14 | 3.491 | 0.696 | 3.491±0.36 |
| exp/07_nn_aec/outputs/aecmos_nn_v3.json | dt | deg_mos | 14 | 1.931 | 0.308 | 1.931±0.16 |
| exp/07_nn_aec/outputs/aecmos_nn_v3.json | fest | echo_mos | 26 | 4.738 | 0.061 | 4.738±0.02 |
| exp/07_nn_aec/outputs/aecmos_nn_v3.json | fest | deg_mos | 26 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_v3.json | nst | echo_mos | 20 | 5.000 | 0.001 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_v3.json | nst | deg_mos | 20 | 2.200 | 0.491 | 2.200±0.22 |
| exp/07_nn_aec/outputs/aecmos_nn_v4.json | dt | echo_mos | 14 | 3.754 | 0.592 | 3.754±0.31 |
| exp/07_nn_aec/outputs/aecmos_nn_v4.json | dt | deg_mos | 14 | 1.743 | 0.254 | 1.743±0.13 |
| exp/07_nn_aec/outputs/aecmos_nn_v4.json | fest | echo_mos | 26 | 3.762 | 0.906 | 3.762±0.35 |
| exp/07_nn_aec/outputs/aecmos_nn_v4.json | fest | deg_mos | 26 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_v4.json | nst | echo_mos | 20 | 5.000 | 0.001 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_v4.json | nst | deg_mos | 20 | 2.073 | 0.377 | 2.073±0.17 |
| exp/07_nn_aec/outputs/aecmos_nn_wp07.json | dt | echo_mos | 14 | 3.478 | 0.615 | 3.478±0.32 |
| exp/07_nn_aec/outputs/aecmos_nn_wp07.json | dt | deg_mos | 14 | 1.919 | 0.342 | 1.919±0.18 |
| exp/07_nn_aec/outputs/aecmos_nn_wp07.json | fest | echo_mos | 26 | 4.727 | 0.069 | 4.727±0.03 |
| exp/07_nn_aec/outputs/aecmos_nn_wp07.json | fest | deg_mos | 26 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_wp07.json | nst | echo_mos | 20 | 5.000 | 0.001 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_nn_wp07.json | nst | deg_mos | 20 | 2.181 | 0.471 | 2.181±0.21 |
| exp/07_nn_aec/outputs/aecmos_trad_drift.json | dt | echo_mos | 18 | 1.649 | 0.340 | 1.649±0.16 |
| exp/07_nn_aec/outputs/aecmos_trad_drift.json | dt | deg_mos | 18 | 3.098 | 0.475 | 3.098±0.22 |
| exp/07_nn_aec/outputs/aecmos_trad_drift.json | fest | echo_mos | 36 | 2.119 | 1.121 | 2.119±0.37 |
| exp/07_nn_aec/outputs/aecmos_trad_drift.json | fest | deg_mos | 36 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_trad_drift.json | nst | echo_mos | 6 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_trad_drift.json | nst | deg_mos | 6 | 2.702 | 0.434 | 2.702±0.35 |
| exp/07_nn_aec/outputs/aecmos_trad_driftbase.json | dt | echo_mos | 18 | 2.451 | 0.726 | 2.451±0.34 |
| exp/07_nn_aec/outputs/aecmos_trad_driftbase.json | dt | deg_mos | 18 | 2.656 | 0.592 | 2.656±0.27 |
| exp/07_nn_aec/outputs/aecmos_trad_driftbase.json | fest | echo_mos | 36 | 4.011 | 0.878 | 4.011±0.29 |
| exp/07_nn_aec/outputs/aecmos_trad_driftbase.json | fest | deg_mos | 36 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos_trad_driftbase.json | nst | echo_mos | 6 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_trad_driftbase.json | nst | deg_mos | 6 | 2.702 | 0.434 | 2.702±0.35 |
| exp/07_nn_aec/outputs/aecmos_trad_wp06.json | dt | echo_mos | 14 | 2.467 | 0.595 | 2.467±0.31 |
| exp/07_nn_aec/outputs/aecmos_trad_wp06.json | dt | deg_mos | 14 | 2.401 | 0.338 | 2.401±0.18 |
| exp/07_nn_aec/outputs/aecmos_trad_wp06.json | fest | echo_mos | 26 | 3.836 | 0.957 | 3.836±0.37 |
| exp/07_nn_aec/outputs/aecmos_trad_wp06.json | fest | deg_mos | 26 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos_trad_wp06.json | nst | echo_mos | 20 | 5.000 | 0.001 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos_trad_wp06.json | nst | deg_mos | 20 | 2.665 | 0.485 | 2.665±0.21 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_nn_speakeronly.json | dt | echo_mos | 18 | 3.777 | 0.706 | 3.777±0.33 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_nn_speakeronly.json | dt | deg_mos | 18 | 1.727 | 0.269 | 1.727±0.12 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_nn_speakeronly.json | fest | echo_mos | 36 | 4.180 | 0.525 | 4.180±0.17 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_nn_speakeronly.json | fest | deg_mos | 36 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_nn_speakeronly.json | nst | echo_mos | 6 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_nn_speakeronly.json | nst | deg_mos | 6 | 2.067 | 0.508 | 2.067±0.41 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_trad_speakeronly.json | dt | echo_mos | 18 | 2.517 | 0.620 | 2.517±0.29 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_trad_speakeronly.json | dt | deg_mos | 18 | 2.032 | 0.459 | 2.032±0.21 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_trad_speakeronly.json | fest | echo_mos | 36 | 4.207 | 0.794 | 4.207±0.26 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_trad_speakeronly.json | fest | deg_mos | 36 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_trad_speakeronly.json | nst | echo_mos | 6 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_trad_speakeronly.json | nst | deg_mos | 6 | 2.750 | 0.653 | 2.750±0.52 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_nn_heldout.json | dt | echo_mos | 15 | 3.213 | 0.707 | 3.213±0.36 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_nn_heldout.json | dt | deg_mos | 15 | 2.007 | 0.341 | 2.007±0.17 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_nn_heldout.json | fest | echo_mos | 31 | 3.167 | 1.180 | 3.167±0.42 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_nn_heldout.json | fest | deg_mos | 31 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_nn_heldout.json | nst | echo_mos | 14 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_nn_heldout.json | nst | deg_mos | 14 | 2.333 | 0.472 | 2.333±0.25 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_trad_heldout.json | dt | echo_mos | 15 | 2.066 | 0.186 | 2.066±0.09 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_trad_heldout.json | dt | deg_mos | 15 | 2.458 | 0.597 | 2.458±0.30 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_trad_heldout.json | fest | echo_mos | 31 | 3.846 | 0.987 | 3.846±0.35 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_trad_heldout.json | fest | deg_mos | 31 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_trad_heldout.json | nst | echo_mos | 14 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_trad_heldout.json | nst | deg_mos | 14 | 2.606 | 0.546 | 2.606±0.29 |
| exp/07_nn_aec/outputs/eval_v5_dt300/aecmos_nn_v4.json | dt | echo_mos | 154 | 3.721 | 0.750 | 3.721±0.12 |
| exp/07_nn_aec/outputs/eval_v5_dt300/aecmos_nn_v4.json | dt | deg_mos | 154 | 1.811 | 0.294 | 1.811±0.05 |
| exp/07_nn_aec/outputs/eval_v5_dt300/aecmos_nn_v4.json | fest | echo_mos | 70 | 3.956 | 0.819 | 3.956±0.19 |
| exp/07_nn_aec/outputs/eval_v5_dt300/aecmos_nn_v4.json | fest | deg_mos | 70 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/eval_v5_dt300/aecmos_nn_v4.json | nst | echo_mos | 76 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/eval_v5_dt300/aecmos_nn_v4.json | nst | deg_mos | 76 | 2.091 | 0.431 | 2.091±0.10 |
| exp/09_cascade_aec/results/aecmos_cascade.json | dt | echo_mos | 14 | 3.919 | 0.397 | 3.919±0.21 |
| exp/09_cascade_aec/results/aecmos_cascade.json | dt | deg_mos | 14 | 1.785 | 0.253 | 1.785±0.13 |
| exp/09_cascade_aec/results/aecmos_cascade.json | fest | echo_mos | 26 | 4.502 | 0.312 | 4.502±0.12 |
| exp/09_cascade_aec/results/aecmos_cascade.json | fest | deg_mos | 26 | 4.999 | 0.001 | 4.999±0.00 |
| exp/09_cascade_aec/results/aecmos_cascade.json | nst | echo_mos | 20 | 5.000 | 0.001 | 5.000±0.00 |
| exp/09_cascade_aec/results/aecmos_cascade.json | nst | deg_mos | 20 | 2.081 | 0.393 | 2.081±0.17 |
| exp/09_cascade_aec/results/aecmos_cascade_long.json | dt | echo_mos | 14 | 3.898 | 0.417 | 3.898±0.22 |
| exp/09_cascade_aec/results/aecmos_cascade_long.json | dt | deg_mos | 14 | 1.846 | 0.205 | 1.846±0.11 |
| exp/09_cascade_aec/results/aecmos_cascade_long.json | fest | echo_mos | 26 | 4.340 | 0.516 | 4.340±0.20 |
| exp/09_cascade_aec/results/aecmos_cascade_long.json | fest | deg_mos | 26 | 4.999 | 0.001 | 4.999±0.00 |
| exp/09_cascade_aec/results/aecmos_cascade_long.json | nst | echo_mos | 20 | 5.000 | 0.001 | 5.000±0.00 |
| exp/09_cascade_aec/results/aecmos_cascade_long.json | nst | deg_mos | 20 | 2.190 | 0.518 | 2.190±0.23 |

## 二、NN vs 传统基线配对差值 95%CI（配对 bootstrap，B=2000，seed=777）

差值 = NN − 传统，按 (name, scenario) 对齐逐文件样本；对 DT 场景给出 echo_mos / deg_mos。

| 配对（nn vs trad） | 场景 | 指标 | n | Δmean | bootstrap 95%CI | 结论 |
|---|---|---|---|---|---|---|
| exp/07_nn_aec/outputs/aecmos_nn_v4.json vs exp/07_nn_aec/outputs/aecmos_trad_wp06.json | dt | echo_mos | 14 | +1.287 | [+0.880, +1.658] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos_nn_v4.json vs exp/07_nn_aec/outputs/aecmos_trad_wp06.json | dt | deg_mos | 14 | -0.659 | [-0.884, -0.449] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_nn_heldout.json vs exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_trad_heldout.json | dt | echo_mos | 15 | +1.147 | [+0.794, +1.492] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_nn_heldout.json vs exp/07_nn_aec/outputs/eval_v4_speaker_heldout/aecmos_trad_heldout.json | dt | deg_mos | 15 | -0.451 | [-0.776, -0.132] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_nn_speakeronly.json vs exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_trad_speakeronly.json | dt | echo_mos | 18 | +1.260 | [+0.932, +1.604] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_nn_speakeronly.json vs exp/07_nn_aec/outputs/eval_speaker_only_holdout/aecmos_trad_speakeronly.json | dt | deg_mos | 18 | -0.305 | [-0.484, -0.156] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos_nn_drift.json vs exp/07_nn_aec/outputs/aecmos_trad_drift.json | dt | echo_mos | 18 | +2.153 | [+1.745, +2.543] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos_nn_drift.json vs exp/07_nn_aec/outputs/aecmos_trad_drift.json | dt | deg_mos | 18 | -1.290 | [-1.464, -1.099] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos_nn_driftbase.json vs exp/07_nn_aec/outputs/aecmos_trad_driftbase.json | dt | echo_mos | 18 | +1.542 | [+1.149, +1.963] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos_nn_driftbase.json vs exp/07_nn_aec/outputs/aecmos_trad_driftbase.json | dt | deg_mos | 18 | -0.865 | [-1.148, -0.590] | 显著（CI 不含 0） |
| exp/09_cascade_aec/results/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos_trad_wp06.json | dt | echo_mos | 14 | +1.452 | [+1.124, +1.737] | 显著（CI 不含 0） |
| exp/09_cascade_aec/results/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos_trad_wp06.json | dt | deg_mos | 14 | -0.616 | [-0.831, -0.413] | 显著（CI 不含 0） |
| exp/09_cascade_aec/results/aecmos_cascade_long.json vs exp/07_nn_aec/outputs/aecmos_trad_wp06.json | dt | echo_mos | 14 | +1.431 | [+1.048, +1.758] | 显著（CI 不含 0） |
| exp/09_cascade_aec/results/aecmos_cascade_long.json vs exp/07_nn_aec/outputs/aecmos_trad_wp06.json | dt | deg_mos | 14 | -0.555 | [-0.774, -0.352] | 显著（CI 不含 0） |
