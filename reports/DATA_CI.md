# DATA_CI.md — AECMOS 指标 95% 置信区间（自动生成）

- 生成日期：2026-10-11
- 生成命令：`python tools/compute_ci.py`（仓库根 avsuite/ 下）
- 方法：解析法 95%CI = mean ± 1.96·sd/√n；配对差值用 bootstrap 95%CI（B=2000，seed=777，百分位法）
- 数据源：`exp/07_nn_aec/outputs/**/aecmos_*.json`、`exp/09_cascade_aec/results/aecmos_*.json` 的逐文件 rows

## 一、逐文件解析 95%CI（1.96·sd/√n）

| 文件 | 场景 | 指标 | n | mean | sd | 95%CI |
|---|---|---|---|---|---|---|
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_best.json | dt | echo_mos | 18 | 4.231 | 0.248 | 4.231±0.11 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_best.json | dt | deg_mos | 18 | 3.156 | 0.619 | 3.156±0.29 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_best.json | fest | echo_mos | 36 | 3.519 | 0.879 | 3.519±0.29 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_best.json | fest | deg_mos | 36 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_best.json | nst | echo_mos | 6 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_best.json | nst | deg_mos | 6 | 3.934 | 0.288 | 3.934±0.23 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_last.json | dt | echo_mos | 18 | 4.231 | 0.245 | 4.231±0.11 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_last.json | dt | deg_mos | 18 | 3.156 | 0.624 | 3.156±0.29 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_last.json | fest | echo_mos | 36 | 3.442 | 0.904 | 3.442±0.30 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_last.json | fest | deg_mos | 36 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_last.json | nst | echo_mos | 6 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_nn_last.json | nst | deg_mos | 6 | 3.931 | 0.290 | 3.931±0.23 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_trad.json | dt | echo_mos | 18 | 3.392 | 0.694 | 3.392±0.32 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_trad.json | dt | deg_mos | 18 | 3.614 | 0.539 | 3.614±0.25 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_trad.json | fest | echo_mos | 36 | 3.601 | 0.817 | 3.601±0.27 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_trad.json | fest | deg_mos | 36 | 4.998 | 0.001 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_trad.json | nst | echo_mos | 6 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/driftbase/aecmos_trad.json | nst | deg_mos | 6 | 3.847 | 0.417 | 3.847±0.33 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_cascade.json | dt | echo_mos | 154 | 2.853 | 0.626 | 2.853±0.10 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_cascade.json | dt | deg_mos | 154 | 3.229 | 0.488 | 3.229±0.08 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_cascade.json | fest | echo_mos | 70 | 1.760 | 0.271 | 1.760±0.06 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_cascade.json | fest | deg_mos | 70 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_cascade.json | nst | echo_mos | 76 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_cascade.json | nst | deg_mos | 76 | 3.750 | 0.465 | 3.750±0.10 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_mic.json | dt | echo_mos | 154 | 2.144 | 0.577 | 2.144±0.09 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_mic.json | dt | deg_mos | 154 | 4.165 | 0.188 | 4.165±0.03 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_mic.json | fest | echo_mos | 70 | 1.752 | 0.241 | 1.752±0.06 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_mic.json | fest | deg_mos | 70 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_mic.json | nst | echo_mos | 76 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_mic.json | nst | deg_mos | 76 | 3.795 | 0.364 | 3.795±0.08 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_nn.json | dt | echo_mos | 154 | 2.671 | 0.746 | 2.671±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_nn.json | dt | deg_mos | 154 | 3.636 | 0.430 | 3.636±0.07 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_nn.json | fest | echo_mos | 70 | 2.221 | 0.862 | 2.221±0.20 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_nn.json | fest | deg_mos | 70 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_nn.json | nst | echo_mos | 76 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_nn.json | nst | deg_mos | 76 | 3.827 | 0.455 | 3.827±0.10 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_trad.json | dt | echo_mos | 154 | 2.147 | 0.564 | 2.147±0.09 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_trad.json | dt | deg_mos | 154 | 4.146 | 0.216 | 4.146±0.03 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_trad.json | fest | echo_mos | 70 | 1.758 | 0.237 | 1.758±0.06 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_trad.json | fest | deg_mos | 70 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_trad.json | nst | echo_mos | 76 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_trad.json | nst | deg_mos | 76 | 3.776 | 0.371 | 3.776±0.08 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_cascade.json | dt | echo_mos | 18 | 4.268 | 0.287 | 4.268±0.13 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_cascade.json | dt | deg_mos | 18 | 3.242 | 0.524 | 3.242±0.24 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_cascade.json | fest | echo_mos | 36 | 4.309 | 0.352 | 4.309±0.11 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_cascade.json | fest | deg_mos | 36 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_cascade.json | nst | echo_mos | 6 | 4.996 | 0.004 | 4.996±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_cascade.json | nst | deg_mos | 6 | 3.778 | 0.794 | 3.778±0.64 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn.json | dt | echo_mos | 18 | 4.209 | 0.210 | 4.209±0.10 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn.json | dt | deg_mos | 18 | 3.178 | 0.578 | 3.178±0.27 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn.json | fest | echo_mos | 36 | 3.736 | 0.869 | 3.736±0.28 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn.json | fest | deg_mos | 36 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn.json | nst | echo_mos | 6 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn.json | nst | deg_mos | 6 | 3.972 | 0.362 | 3.972±0.29 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_trad.json | dt | echo_mos | 18 | 3.482 | 0.572 | 3.482±0.26 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_trad.json | dt | deg_mos | 18 | 3.462 | 0.446 | 3.462±0.21 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_trad.json | fest | echo_mos | 36 | 3.733 | 0.850 | 3.733±0.28 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_trad.json | fest | deg_mos | 36 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_trad.json | nst | echo_mos | 6 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_trad.json | nst | deg_mos | 6 | 3.888 | 0.542 | 3.888±0.43 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_cascade.json | dt | echo_mos | 14 | 4.261 | 0.184 | 4.261±0.10 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_cascade.json | dt | deg_mos | 14 | 3.250 | 0.481 | 3.250±0.25 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_cascade.json | fest | echo_mos | 26 | 3.938 | 0.628 | 3.938±0.24 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_cascade.json | fest | deg_mos | 26 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_cascade.json | nst | echo_mos | 20 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_cascade.json | nst | deg_mos | 20 | 3.851 | 0.515 | 3.851±0.23 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_mic.json | dt | echo_mos | 14 | 1.941 | 0.450 | 1.941±0.24 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_mic.json | dt | deg_mos | 14 | 4.186 | 0.166 | 4.186±0.09 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_mic.json | fest | echo_mos | 26 | 1.523 | 0.191 | 1.523±0.07 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_mic.json | fest | deg_mos | 26 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_mic.json | nst | echo_mos | 20 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_mic.json | nst | deg_mos | 20 | 3.842 | 0.384 | 3.842±0.17 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_nn.json | dt | echo_mos | 14 | 4.175 | 0.234 | 4.175±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_nn.json | dt | deg_mos | 14 | 3.237 | 0.392 | 3.237±0.21 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_nn.json | fest | echo_mos | 26 | 3.194 | 0.962 | 3.194±0.37 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_nn.json | fest | deg_mos | 26 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_nn.json | nst | echo_mos | 20 | 4.999 | 0.003 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_nn.json | nst | deg_mos | 20 | 3.941 | 0.430 | 3.941±0.19 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_trad.json | dt | echo_mos | 14 | 3.494 | 0.480 | 3.494±0.25 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_trad.json | dt | deg_mos | 14 | 3.598 | 0.388 | 3.598±0.20 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_trad.json | fest | echo_mos | 26 | 3.207 | 0.829 | 3.207±0.32 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_trad.json | fest | deg_mos | 26 | 4.998 | 0.001 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_trad.json | nst | echo_mos | 20 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_trad.json | nst | deg_mos | 20 | 3.842 | 0.384 | 3.842±0.17 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_cascade.json | dt | echo_mos | 18 | 3.321 | 0.693 | 3.321±0.32 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_cascade.json | dt | deg_mos | 18 | 3.210 | 0.555 | 3.210±0.26 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_cascade.json | fest | echo_mos | 36 | 2.162 | 0.490 | 2.162±0.16 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_cascade.json | fest | deg_mos | 36 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_cascade.json | nst | echo_mos | 6 | 4.998 | 0.001 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_cascade.json | nst | deg_mos | 6 | 3.982 | 0.329 | 3.982±0.26 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn.json | dt | echo_mos | 18 | 4.227 | 0.266 | 4.227±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn.json | dt | deg_mos | 18 | 3.273 | 0.444 | 3.273±0.21 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn.json | fest | echo_mos | 36 | 3.745 | 0.798 | 3.745±0.26 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn.json | fest | deg_mos | 36 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn.json | nst | echo_mos | 6 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn.json | nst | deg_mos | 6 | 3.934 | 0.288 | 3.934±0.23 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn_drift_retrain.json | dt | echo_mos | 18 | 3.542 | 0.345 | 3.542±0.16 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn_drift_retrain.json | dt | deg_mos | 18 | 3.347 | 0.430 | 3.347±0.20 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn_drift_retrain.json | fest | echo_mos | 36 | 1.965 | 0.363 | 1.965±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn_drift_retrain.json | fest | deg_mos | 36 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn_drift_retrain.json | nst | echo_mos | 6 | 4.997 | 0.001 | 4.997±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn_drift_retrain.json | nst | deg_mos | 6 | 4.157 | 0.228 | 4.157±0.18 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_trad.json | dt | echo_mos | 18 | 2.120 | 0.579 | 2.120±0.27 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_trad.json | dt | deg_mos | 18 | 4.074 | 0.421 | 4.074±0.19 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_trad.json | fest | echo_mos | 36 | 1.878 | 0.491 | 1.878±0.16 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_trad.json | fest | deg_mos | 36 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_trad.json | nst | echo_mos | 6 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_trad.json | nst | deg_mos | 6 | 3.847 | 0.417 | 3.847±0.33 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_cascade.json | dt | echo_mos | 15 | 4.222 | 0.147 | 4.222±0.07 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_cascade.json | dt | deg_mos | 15 | 3.370 | 0.514 | 3.370±0.26 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_cascade.json | fest | echo_mos | 31 | 4.047 | 0.336 | 4.047±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_cascade.json | fest | deg_mos | 31 | 4.999 | 0.000 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_cascade.json | nst | echo_mos | 14 | 4.998 | 0.001 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_cascade.json | nst | deg_mos | 14 | 4.196 | 0.226 | 4.196±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn.json | dt | echo_mos | 15 | 4.161 | 0.259 | 4.161±0.13 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn.json | dt | deg_mos | 15 | 3.299 | 0.667 | 3.299±0.34 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn.json | fest | echo_mos | 31 | 2.816 | 1.089 | 2.816±0.38 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn.json | fest | deg_mos | 31 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn.json | nst | echo_mos | 14 | 4.998 | 0.001 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn.json | nst | deg_mos | 14 | 4.325 | 0.209 | 4.325±0.11 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_trad.json | dt | echo_mos | 15 | 3.433 | 0.538 | 3.433±0.27 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_trad.json | dt | deg_mos | 15 | 3.701 | 0.415 | 3.701±0.21 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_trad.json | fest | echo_mos | 31 | 3.193 | 0.761 | 3.193±0.27 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_trad.json | fest | deg_mos | 31 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_trad.json | nst | echo_mos | 14 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_trad.json | nst | deg_mos | 14 | 4.180 | 0.195 | 4.180±0.10 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_cascade.json | dt | echo_mos | 154 | 4.317 | 0.230 | 4.317±0.04 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_cascade.json | dt | deg_mos | 154 | 3.375 | 0.567 | 3.375±0.09 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_cascade.json | fest | echo_mos | 70 | 4.196 | 0.441 | 4.196±0.10 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_cascade.json | fest | deg_mos | 70 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_cascade.json | nst | echo_mos | 76 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_cascade.json | nst | deg_mos | 76 | 3.750 | 0.465 | 3.750±0.10 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_mic.json | dt | echo_mos | 154 | 1.874 | 0.477 | 1.874±0.08 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_mic.json | dt | deg_mos | 154 | 4.252 | 0.175 | 4.252±0.03 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_mic.json | fest | echo_mos | 70 | 1.541 | 0.213 | 1.541±0.05 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_mic.json | fest | deg_mos | 70 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_mic.json | nst | echo_mos | 76 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_mic.json | nst | deg_mos | 76 | 3.795 | 0.364 | 3.795±0.08 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json | dt | echo_mos | 154 | 4.248 | 0.286 | 4.248±0.05 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json | dt | deg_mos | 154 | 3.220 | 0.559 | 3.220±0.09 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json | fest | echo_mos | 70 | 3.881 | 0.782 | 3.881±0.18 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json | fest | deg_mos | 70 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json | nst | echo_mos | 76 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json | nst | deg_mos | 76 | 3.827 | 0.455 | 3.827±0.10 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_trad.json | dt | echo_mos | 154 | 3.442 | 0.635 | 3.442±0.10 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_trad.json | dt | deg_mos | 154 | 3.590 | 0.557 | 3.590±0.09 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_trad.json | fest | echo_mos | 70 | 3.518 | 0.805 | 3.518±0.19 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_trad.json | fest | deg_mos | 70 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_trad.json | nst | echo_mos | 76 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_trad.json | nst | deg_mos | 76 | 3.776 | 0.371 | 3.776±0.08 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_cascade.json | dt | echo_mos | 42 | 3.500 | 0.772 | 3.500±0.23 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_cascade.json | dt | deg_mos | 42 | 3.453 | 0.507 | 3.453±0.15 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_cascade.json | fest | echo_mos | 34 | 3.149 | 0.981 | 3.149±0.33 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_cascade.json | fest | deg_mos | 34 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_cascade.json | nst | echo_mos | 24 | 4.998 | 0.001 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_cascade.json | nst | deg_mos | 24 | 3.770 | 0.469 | 3.770±0.19 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_nn.json | dt | echo_mos | 42 | 3.504 | 0.798 | 3.504±0.24 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_nn.json | dt | deg_mos | 42 | 3.571 | 0.406 | 3.571±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_nn.json | fest | echo_mos | 34 | 3.363 | 1.140 | 3.363±0.38 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_nn.json | fest | deg_mos | 34 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_nn.json | nst | echo_mos | 24 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_nn.json | nst | deg_mos | 24 | 3.905 | 0.373 | 3.905±0.15 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_trad.json | dt | echo_mos | 42 | 2.769 | 0.683 | 2.769±0.21 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_trad.json | dt | deg_mos | 42 | 3.951 | 0.482 | 3.951±0.15 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_trad.json | fest | echo_mos | 34 | 3.016 | 0.997 | 3.016±0.34 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_trad.json | fest | deg_mos | 34 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_trad.json | nst | echo_mos | 24 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_trad.json | nst | deg_mos | 24 | 3.792 | 0.360 | 3.792±0.14 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_cascade.json | dt | echo_mos | 406 | 4.291 | 0.257 | 4.291±0.02 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_cascade.json | dt | deg_mos | 406 | 3.451 | 0.528 | 3.451±0.05 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_cascade.json | fest | echo_mos | 212 | 4.157 | 0.545 | 4.157±0.07 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_cascade.json | fest | deg_mos | 212 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_cascade.json | nst | echo_mos | 182 | 4.998 | 0.002 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_cascade.json | nst | deg_mos | 182 | 3.846 | 0.526 | 3.846±0.08 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_nn.json | dt | echo_mos | 406 | 4.223 | 0.317 | 4.223±0.03 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_nn.json | dt | deg_mos | 406 | 3.269 | 0.539 | 3.269±0.05 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_nn.json | fest | echo_mos | 212 | 3.595 | 0.921 | 3.595±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_nn.json | fest | deg_mos | 212 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_nn.json | nst | echo_mos | 182 | 4.998 | 0.008 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_nn.json | nst | deg_mos | 182 | 3.876 | 0.514 | 3.876±0.07 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_trad.json | dt | echo_mos | 406 | 3.424 | 0.632 | 3.424±0.06 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_trad.json | dt | deg_mos | 406 | 3.590 | 0.515 | 3.590±0.05 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_trad.json | fest | echo_mos | 212 | 3.543 | 0.911 | 3.543±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_trad.json | fest | deg_mos | 212 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_trad.json | nst | echo_mos | 182 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_trad.json | nst | deg_mos | 182 | 3.829 | 0.386 | 3.829±0.06 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift.json | dt | echo_mos | 18 | 2.065 | 0.497 | 2.065±0.23 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift.json | dt | deg_mos | 18 | 4.147 | 0.253 | 4.147±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift.json | fest | echo_mos | 36 | 1.725 | 0.343 | 1.725±0.11 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift.json | fest | deg_mos | 36 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift.json | nst | echo_mos | 6 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift.json | nst | deg_mos | 6 | 3.867 | 0.379 | 3.867±0.30 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_margin.json | dt | echo_mos | 18 | 2.096 | 0.524 | 2.096±0.24 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_margin.json | dt | deg_mos | 18 | 4.156 | 0.243 | 4.156±0.11 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_margin.json | fest | echo_mos | 36 | 1.765 | 0.385 | 1.765±0.13 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_margin.json | fest | deg_mos | 36 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_margin.json | nst | echo_mos | 6 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_margin.json | nst | deg_mos | 6 | 3.867 | 0.379 | 3.867±0.30 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_old.json | dt | echo_mos | 18 | 2.120 | 0.579 | 2.120±0.27 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_old.json | dt | deg_mos | 18 | 4.074 | 0.421 | 4.074±0.19 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_old.json | fest | echo_mos | 36 | 1.878 | 0.491 | 1.878±0.16 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_old.json | fest | deg_mos | 36 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_old.json | nst | echo_mos | 6 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_old.json | nst | deg_mos | 6 | 3.847 | 0.417 | 3.847±0.33 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_reest.json | dt | echo_mos | 18 | 2.042 | 0.468 | 2.042±0.22 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_reest.json | dt | deg_mos | 18 | 4.096 | 0.363 | 4.096±0.17 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_reest.json | fest | echo_mos | 36 | 1.812 | 0.411 | 1.812±0.13 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_reest.json | fest | deg_mos | 36 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_reest.json | nst | echo_mos | 6 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_drift/aecmos_trad_hard_drift_reest.json | nst | deg_mos | 6 | 3.847 | 0.417 | 3.847±0.33 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6.json | dt | echo_mos | 42 | 2.750 | 0.664 | 2.750±0.20 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6.json | dt | deg_mos | 42 | 3.976 | 0.411 | 3.976±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6.json | fest | echo_mos | 34 | 2.946 | 0.967 | 2.946±0.33 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6.json | fest | deg_mos | 34 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6.json | nst | echo_mos | 24 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6.json | nst | deg_mos | 24 | 3.765 | 0.412 | 3.765±0.16 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_margin.json | dt | echo_mos | 42 | 2.764 | 0.676 | 2.764±0.20 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_margin.json | dt | deg_mos | 42 | 3.991 | 0.384 | 3.991±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_margin.json | fest | echo_mos | 34 | 2.963 | 0.976 | 2.963±0.33 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_margin.json | fest | deg_mos | 34 | 4.999 | 0.001 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_margin.json | nst | echo_mos | 24 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_margin.json | nst | deg_mos | 24 | 3.792 | 0.360 | 3.792±0.14 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_old.json | dt | echo_mos | 42 | 2.768 | 0.682 | 2.768±0.21 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_old.json | dt | deg_mos | 42 | 3.952 | 0.482 | 3.952±0.15 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_old.json | fest | echo_mos | 34 | 3.016 | 0.997 | 3.016±0.34 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_old.json | fest | deg_mos | 34 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_old.json | nst | echo_mos | 24 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_old.json | nst | deg_mos | 24 | 3.792 | 0.360 | 3.792±0.14 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_reest.json | dt | echo_mos | 42 | 2.733 | 0.656 | 2.733±0.20 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_reest.json | dt | deg_mos | 42 | 3.958 | 0.474 | 3.958±0.14 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_reest.json | fest | echo_mos | 34 | 3.010 | 0.986 | 3.010±0.33 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_reest.json | fest | deg_mos | 34 | 5.000 | 0.000 | 5.000±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_reest.json | nst | echo_mos | 24 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/hardening_rerun_v6/aecmos_trad_hard_v6_reest.json | nst | deg_mos | 24 | 3.777 | 0.385 | 3.777±0.15 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/leadincrop/aecmos_dt_crop_t0s.json | dt | echo_mos | 14 | 4.172 | 0.231 | 4.172±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/leadincrop/aecmos_dt_crop_t0s.json | dt | deg_mos | 14 | 3.252 | 0.366 | 3.252±0.19 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/leadincrop/aecmos_dt_crop_t1s.json | dt | echo_mos | 14 | 4.115 | 0.225 | 4.115±0.12 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/leadincrop/aecmos_dt_crop_t1s.json | dt | deg_mos | 14 | 3.250 | 0.262 | 3.250±0.14 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/leadincrop/aecmos_dt_crop_t2s.json | dt | echo_mos | 14 | 3.532 | 0.768 | 3.532±0.40 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/leadincrop/aecmos_dt_crop_t2s.json | dt | deg_mos | 14 | 3.319 | 0.533 | 3.319±0.28 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/leadincrop/aecmos_dt_crop_t3s.json | dt | echo_mos | 14 | 3.541 | 0.695 | 3.541±0.36 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/leadincrop/aecmos_dt_crop_t3s.json | dt | deg_mos | 14 | 3.341 | 0.455 | 3.341±0.24 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/zerotrain/aecmos_zerotrain.json | dt | echo_mos | 14 | 4.062 | 0.305 | 4.062±0.16 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/zerotrain/aecmos_zerotrain.json | dt | deg_mos | 14 | 3.223 | 0.617 | 3.223±0.32 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/zerotrain/aecmos_zerotrain.json | fest | echo_mos | 26 | 3.078 | 1.117 | 3.078±0.43 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/zerotrain/aecmos_zerotrain.json | fest | deg_mos | 26 | 4.999 | 0.000 | 4.999±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/zerotrain/aecmos_zerotrain.json | nst | echo_mos | 20 | 4.998 | 0.003 | 4.998±0.00 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/zerotrain/aecmos_zerotrain.json | nst | deg_mos | 20 | 3.945 | 0.463 | 3.945±0.20 |
| exp/07_nn_aec/outputs/hardening_rerun/aecmos_trad_hard_official_fest.json | ? | echo_mos | 300 | 2.759 | 1.141 | 2.759±0.13 |
| exp/07_nn_aec/outputs/hardening_rerun/aecmos_trad_hard_official_fest.json | ? | deg_mos | 300 | 4.999 | 0.001 | 4.999±0.00 |
| exp/09_cascade_aec/results/aecmos_cascade_official.json | dt | echo_mos | 299 | 3.233 | 1.076 | 3.233±0.12 |
| exp/09_cascade_aec/results/aecmos_cascade_official.json | dt | deg_mos | 299 | 2.947 | 0.912 | 2.947±0.10 |
| exp/09_cascade_aec/results/aecmos_cascade_official.json | fest | echo_mos | 299 | 3.096 | 1.189 | 3.096±0.13 |
| exp/09_cascade_aec/results/aecmos_cascade_official.json | fest | deg_mos | 299 | 4.999 | 0.001 | 4.999±0.00 |
| exp/09_cascade_aec/results/aecmos_cascade_official.json | nst | echo_mos | 200 | 4.999 | 0.002 | 4.999±0.00 |
| exp/09_cascade_aec/results/aecmos_cascade_official.json | nst | deg_mos | 200 | 3.947 | 0.595 | 3.947±0.08 |

## 二、NN vs 传统基线配对差值 95%CI（配对 bootstrap，B=2000，seed=777）

差值 = NN − 传统，按 (name, scenario) 对齐逐文件样本；对 DT 场景给出 echo_mos / deg_mos。

| 配对（nn vs trad） | 场景 | 指标 | n | Δmean | bootstrap 95%CI | 结论 |
|---|---|---|---|---|---|---|
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_trad.json | dt | echo_mos | 14 | +0.681 | [+0.494, +0.884] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_trad.json | dt | deg_mos | 14 | -0.361 | [-0.533, -0.187] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_trad.json | dt | echo_mos | 154 | +0.806 | [+0.720, +0.893] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_trad.json | dt | deg_mos | 154 | -0.369 | [-0.457, -0.281] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_trad.json | dt | echo_mos | 406 | +0.799 | [+0.743, +0.855] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_trad.json | dt | deg_mos | 406 | -0.322 | [-0.376, -0.265] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_trad.json | dt | echo_mos | 154 | +0.524 | [+0.404, +0.633] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_trad.json | dt | deg_mos | 154 | -0.510 | [-0.583, -0.441] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_trad.json | dt | echo_mos | 42 | +0.735 | [+0.500, +0.967] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_trad.json | dt | deg_mos | 42 | -0.380 | [-0.519, -0.219] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_trad.json | dt | echo_mos | 18 | +2.107 | [+1.792, +2.415] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_trad.json | dt | deg_mos | 18 | -0.800 | [-1.049, -0.541] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_trad.json | dt | echo_mos | 15 | +0.728 | [+0.485, +1.019] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_trad.json | dt | deg_mos | 15 | -0.402 | [-0.715, -0.115] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_trad.json | dt | echo_mos | 18 | +0.727 | [+0.524, +0.961] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_trad.json | dt | deg_mos | 18 | -0.284 | [-0.530, -0.037] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_nn.json | dt | echo_mos | 14 | +0.086 | [-0.016, +0.195] | CI 含 0，不显著 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3/aecmos_nn.json | dt | deg_mos | 14 | +0.013 | [-0.157, +0.175] | CI 含 0，不显著 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json | dt | echo_mos | 154 | +0.069 | [+0.027, +0.112] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json | dt | deg_mos | 154 | +0.154 | [+0.080, +0.229] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_nn.json | dt | echo_mos | 406 | +0.068 | [+0.038, +0.098] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v7/aecmos_nn.json | dt | deg_mos | 406 | +0.182 | [+0.132, +0.232] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_nn.json | dt | echo_mos | 42 | -0.004 | [-0.235, +0.224] | CI 含 0，不显著 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v6_realrir/aecmos_nn.json | dt | deg_mos | 42 | -0.118 | [-0.271, +0.030] | CI 含 0，不显著 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn.json | dt | echo_mos | 18 | -0.906 | [-1.243, -0.589] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v3_drift/aecmos_nn.json | dt | deg_mos | 18 | -0.064 | [-0.292, +0.165] | CI 含 0，不显著 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn.json | dt | echo_mos | 15 | +0.061 | [-0.079, +0.196] | CI 含 0，不显著 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn.json | dt | deg_mos | 15 | +0.072 | [-0.200, +0.397] | CI 含 0，不显著 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn.json | dt | echo_mos | 18 | +0.059 | [-0.060, +0.179] | CI 含 0，不显著 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn.json | dt | deg_mos | 18 | +0.064 | [-0.135, +0.270] | CI 含 0，不显著 |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_nn.json | dt | echo_mos | 154 | +0.182 | [+0.061, +0.301] | 显著（CI 不含 0） |
| exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_cascade.json vs exp/07_nn_aec/outputs/aecmos48_v5fix/eval_ood_delay/aecmos_nn.json | dt | deg_mos | 154 | -0.407 | [-0.493, -0.321] | 显著（CI 不含 0） |
