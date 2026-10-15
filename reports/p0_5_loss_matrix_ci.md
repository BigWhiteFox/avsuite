# P0-5 损失矩阵：逐集统计与配对 95%CI（自动生成）

- bootstrap B=2000，seed=777，百分位法（与 `tools/compute_ci.py` 同口径）
- 解析 95%CI = mean ± 1.96·sd/√n
- 数据源：`exp/07_nn_aec/outputs_v4_p05/aecmos/<set>/aecmos_<variant>.json` 的逐文件 rows

## eval_v3

### 1) 逐变体统计（解析 95%CI）

| 变体 | 场景 | 指标 | n | mean | 解析 95%CI |
|---|---|---|---|---|---|
| v4 | dt | echo_mos | 14 | 4.175 | 4.175±0.123 |
| v4 | dt | deg_mos | 14 | 3.237 | 3.237±0.205 |
| v4 | nst | echo_mos | 20 | 4.999 | 4.999±0.001 |
| v4 | nst | deg_mos | 20 | 3.941 | 3.941±0.188 |
| v4 | fest | echo_mos | 26 | 3.194 | 3.194±0.370 |
| v4 | fest | deg_mos | 26 | 4.999 | 4.999±0.000 |
| lossctrl_240ep | dt | echo_mos | 14 | 4.041 | 4.041±0.166 |
| lossctrl_240ep | dt | deg_mos | 14 | 3.089 | 3.089±0.234 |
| lossctrl_240ep | nst | echo_mos | 20 | 4.998 | 4.998±0.001 |
| lossctrl_240ep | nst | deg_mos | 20 | 3.892 | 3.892±0.202 |
| lossctrl_240ep | fest | echo_mos | 26 | 2.685 | 2.685±0.317 |
| lossctrl_240ep | fest | deg_mos | 26 | 4.999 | 4.999±0.000 |
| psycho_240ep | dt | echo_mos | 14 | 3.840 | 3.840±0.302 |
| psycho_240ep | dt | deg_mos | 14 | 3.304 | 3.304±0.205 |
| psycho_240ep | nst | echo_mos | 20 | 4.998 | 4.998±0.001 |
| psycho_240ep | nst | deg_mos | 20 | 3.925 | 3.925±0.171 |
| psycho_240ep | fest | echo_mos | 26 | 2.250 | 2.250±0.220 |
| psycho_240ep | fest | deg_mos | 26 | 5.000 | 5.000±0.000 |
| lossctrl_240ep_seed1 | dt | echo_mos | 14 | 3.724 | 3.724±0.236 |
| lossctrl_240ep_seed1 | dt | deg_mos | 14 | 3.208 | 3.208±0.185 |
| lossctrl_240ep_seed1 | nst | echo_mos | 20 | 4.998 | 4.998±0.002 |
| lossctrl_240ep_seed1 | nst | deg_mos | 20 | 3.892 | 3.892±0.185 |
| lossctrl_240ep_seed1 | fest | echo_mos | 26 | 2.248 | 2.248±0.275 |
| lossctrl_240ep_seed1 | fest | deg_mos | 26 | 5.000 | 5.000±0.000 |
| psycho_240ep_seed1 | dt | echo_mos | 14 | 3.850 | 3.850±0.196 |
| psycho_240ep_seed1 | dt | deg_mos | 14 | 3.191 | 3.191±0.186 |
| psycho_240ep_seed1 | nst | echo_mos | 20 | 4.998 | 4.998±0.001 |
| psycho_240ep_seed1 | nst | deg_mos | 20 | 3.865 | 3.865±0.203 |
| psycho_240ep_seed1 | fest | echo_mos | 26 | 2.328 | 2.328±0.222 |
| psycho_240ep_seed1 | fest | deg_mos | 26 | 5.000 | 5.000±0.000 |
| psycho_cfix_seed1 | dt | echo_mos | 14 | 3.860 | 3.860±0.316 |
| psycho_cfix_seed1 | dt | deg_mos | 14 | 3.263 | 3.263±0.215 |
| psycho_cfix_seed1 | nst | echo_mos | 20 | 4.998 | 4.998±0.001 |
| psycho_cfix_seed1 | nst | deg_mos | 20 | 3.907 | 3.907±0.175 |
| psycho_cfix_seed1 | fest | echo_mos | 26 | 2.562 | 2.562±0.311 |
| psycho_cfix_seed1 | fest | deg_mos | 26 | 5.000 | 5.000±0.000 |
| lossctrl_v4replica | dt | echo_mos | 14 | 4.173 | 4.173±0.124 |
| lossctrl_v4replica | dt | deg_mos | 14 | 3.242 | 3.242±0.202 |
| lossctrl_v4replica | nst | echo_mos | 20 | 4.999 | 4.999±0.001 |
| lossctrl_v4replica | nst | deg_mos | 20 | 3.941 | 3.941±0.188 |
| lossctrl_v4replica | fest | echo_mos | 26 | 3.219 | 3.219±0.368 |
| lossctrl_v4replica | fest | deg_mos | 26 | 4.999 | 4.999±0.000 |

### 2) 配对差值（变体 − v4，同 (name, scenario) 对齐）

| 变体 | 场景 | 指标 | n | Δmean | bootstrap 95%CI | 降/升 | 结论 |
|---|---|---|---|---|---|---|---|
| lossctrl_240ep | dt | echo_mos | 14 | -0.134 | [-0.221, -0.047] | 10/4 | 显著退化 |
| lossctrl_240ep | dt | deg_mos | 14 | -0.148 | [-0.296, -0.025] | 11/3 | 显著退化 |
| lossctrl_240ep | nst | echo_mos | 20 | -0.001 | [-0.001, -0.000] | 18/2 | 显著退化 |
| lossctrl_240ep | nst | deg_mos | 20 | -0.049 | [-0.165, +0.058] | 11/9 | CI 含 0（不显著） |
| lossctrl_240ep | fest | echo_mos | 26 | -0.509 | [-0.965, -0.057] | 18/8 | 显著退化 |
| lossctrl_240ep | fest | deg_mos | 26 | +0.000 | [-0.000, +0.000] | 8/18 | CI 含 0（不显著） |
| psycho_240ep | dt | echo_mos | 14 | -0.335 | [-0.603, -0.141] | 13/1 | 显著退化 |
| psycho_240ep | dt | deg_mos | 14 | +0.067 | [-0.113, +0.227] | 6/8 | CI 含 0（不显著） |
| psycho_240ep | nst | echo_mos | 20 | -0.001 | [-0.002, -0.000] | 17/3 | 显著退化 |
| psycho_240ep | nst | deg_mos | 20 | -0.016 | [-0.122, +0.089] | 12/8 | CI 含 0（不显著） |
| psycho_240ep | fest | echo_mos | 26 | -0.944 | [-1.329, -0.558] | 20/6 | 显著退化 |
| psycho_240ep | fest | deg_mos | 26 | +0.000 | [+0.000, +0.001] | 5/21 | 显著回升 |
| lossctrl_240ep_seed1 | dt | echo_mos | 14 | -0.451 | [-0.616, -0.299] | 13/1 | 显著退化 |
| lossctrl_240ep_seed1 | dt | deg_mos | 14 | -0.029 | [-0.128, +0.080] | 9/5 | CI 含 0（不显著） |
| lossctrl_240ep_seed1 | nst | echo_mos | 20 | -0.001 | [-0.002, -0.000] | 18/2 | 显著退化 |
| lossctrl_240ep_seed1 | nst | deg_mos | 20 | -0.049 | [-0.150, +0.056] | 12/8 | CI 含 0（不显著） |
| lossctrl_240ep_seed1 | fest | echo_mos | 26 | -0.945 | [-1.375, -0.489] | 22/4 | 显著退化 |
| lossctrl_240ep_seed1 | fest | deg_mos | 26 | +0.000 | [+0.000, +0.001] | 5/21 | 显著回升 |
| psycho_240ep_seed1 | dt | echo_mos | 14 | -0.326 | [-0.466, -0.194] | 13/1 | 显著退化 |
| psycho_240ep_seed1 | dt | deg_mos | 14 | -0.046 | [-0.174, +0.063] | 7/7 | CI 含 0（不显著） |
| psycho_240ep_seed1 | nst | echo_mos | 20 | -0.001 | [-0.002, -0.000] | 16/4 | 显著退化 |
| psycho_240ep_seed1 | nst | deg_mos | 20 | -0.076 | [-0.199, +0.047] | 12/8 | CI 含 0（不显著） |
| psycho_240ep_seed1 | fest | echo_mos | 26 | -0.866 | [-1.280, -0.419] | 20/6 | 显著退化 |
| psycho_240ep_seed1 | fest | deg_mos | 26 | +0.000 | [+0.000, +0.001] | 6/20 | 显著回升 |
| psycho_cfix_seed1 | dt | echo_mos | 14 | -0.315 | [-0.589, -0.108] | 12/2 | 显著退化 |
| psycho_cfix_seed1 | dt | deg_mos | 14 | +0.026 | [-0.090, +0.132] | 5/9 | CI 含 0（不显著） |
| psycho_cfix_seed1 | nst | echo_mos | 20 | -0.001 | [-0.002, -0.000] | 16/4 | 显著退化 |
| psycho_cfix_seed1 | nst | deg_mos | 20 | -0.034 | [-0.137, +0.073] | 12/8 | CI 含 0（不显著） |
| psycho_cfix_seed1 | fest | echo_mos | 26 | -0.631 | [-1.111, -0.158] | 18/8 | 显著退化 |
| psycho_cfix_seed1 | fest | deg_mos | 26 | +0.000 | [-0.000, +0.001] | 8/18 | CI 含 0（不显著） |
| lossctrl_v4replica | dt | echo_mos | 14 | -0.002 | [-0.005, -0.001] | 12/2 | 显著退化 |
| lossctrl_v4replica | dt | deg_mos | 14 | +0.005 | [-0.001, +0.014] | 8/6 | CI 含 0（不显著） |
| lossctrl_v4replica | nst | echo_mos | 20 | +0.000 | [-0.000, +0.000] | 3/15 | CI 含 0（不显著） |
| lossctrl_v4replica | nst | deg_mos | 20 | -0.000 | [-0.001, +0.001] | 9/11 | CI 含 0（不显著） |
| lossctrl_v4replica | fest | echo_mos | 26 | +0.025 | [+0.014, +0.036] | 4/22 | 显著回升 |
| lossctrl_v4replica | fest | deg_mos | 26 | -0.000 | [-0.000, +0.000] | 19/7 | CI 含 0（不显著） |

## eval_v5

### 1) 逐变体统计（解析 95%CI）

| 变体 | 场景 | 指标 | n | mean | 解析 95%CI |
|---|---|---|---|---|---|
| v4 | dt | echo_mos | 154 | 4.248 | 4.248±0.045 |
| v4 | dt | deg_mos | 154 | 3.220 | 3.220±0.088 |
| v4 | nst | echo_mos | 76 | 4.998 | 4.998±0.001 |
| v4 | nst | deg_mos | 76 | 3.827 | 3.827±0.102 |
| v4 | fest | echo_mos | 70 | 3.881 | 3.881±0.183 |
| v4 | fest | deg_mos | 70 | 4.999 | 4.999±0.000 |
| lossctrl_240ep | dt | echo_mos | 154 | 4.102 | 4.102±0.053 |
| lossctrl_240ep | dt | deg_mos | 154 | 3.129 | 3.129±0.083 |
| lossctrl_240ep | nst | echo_mos | 76 | 4.998 | 4.998±0.000 |
| lossctrl_240ep | nst | deg_mos | 76 | 3.741 | 3.741±0.112 |
| lossctrl_240ep | fest | echo_mos | 70 | 2.690 | 2.690±0.208 |
| lossctrl_240ep | fest | deg_mos | 70 | 5.000 | 5.000±0.000 |
| psycho_240ep | dt | echo_mos | 154 | 3.964 | 3.964±0.066 |
| psycho_240ep | dt | deg_mos | 154 | 3.279 | 3.279±0.074 |
| psycho_240ep | nst | echo_mos | 76 | 4.997 | 4.997±0.001 |
| psycho_240ep | nst | deg_mos | 76 | 3.864 | 3.864±0.092 |
| psycho_240ep | fest | echo_mos | 70 | 2.348 | 2.348±0.150 |
| psycho_240ep | fest | deg_mos | 70 | 5.000 | 5.000±0.000 |
| lossctrl_240ep_seed1 | dt | echo_mos | 154 | 3.770 | 3.770±0.074 |
| lossctrl_240ep_seed1 | dt | deg_mos | 154 | 3.265 | 3.265±0.080 |
| lossctrl_240ep_seed1 | nst | echo_mos | 76 | 4.998 | 4.998±0.000 |
| lossctrl_240ep_seed1 | nst | deg_mos | 76 | 3.812 | 3.812±0.103 |
| lossctrl_240ep_seed1 | fest | echo_mos | 70 | 2.350 | 2.350±0.163 |
| lossctrl_240ep_seed1 | fest | deg_mos | 70 | 5.000 | 5.000±0.000 |
| psycho_240ep_seed1 | dt | echo_mos | 154 | 3.925 | 3.925±0.067 |
| psycho_240ep_seed1 | dt | deg_mos | 154 | 3.258 | 3.258±0.074 |
| psycho_240ep_seed1 | nst | echo_mos | 76 | 4.997 | 4.997±0.001 |
| psycho_240ep_seed1 | nst | deg_mos | 76 | 3.835 | 3.835±0.098 |
| psycho_240ep_seed1 | fest | echo_mos | 70 | 2.373 | 2.373±0.151 |
| psycho_240ep_seed1 | fest | deg_mos | 70 | 5.000 | 5.000±0.000 |
| psycho_cfix_seed1 | dt | echo_mos | 154 | 3.936 | 3.936±0.069 |
| psycho_cfix_seed1 | dt | deg_mos | 154 | 3.241 | 3.241±0.081 |
| psycho_cfix_seed1 | nst | echo_mos | 76 | 4.997 | 4.997±0.001 |
| psycho_cfix_seed1 | nst | deg_mos | 76 | 3.837 | 3.837±0.106 |
| psycho_cfix_seed1 | fest | echo_mos | 70 | 2.540 | 2.540±0.187 |
| psycho_cfix_seed1 | fest | deg_mos | 70 | 5.000 | 5.000±0.000 |
| lossctrl_v4replica | dt | echo_mos | 154 | 4.247 | 4.247±0.045 |
| lossctrl_v4replica | dt | deg_mos | 154 | 3.222 | 3.222±0.088 |
| lossctrl_v4replica | nst | echo_mos | 76 | 4.998 | 4.998±0.001 |
| lossctrl_v4replica | nst | deg_mos | 76 | 3.826 | 3.826±0.102 |
| lossctrl_v4replica | fest | echo_mos | 70 | 3.912 | 3.912±0.180 |
| lossctrl_v4replica | fest | deg_mos | 70 | 4.999 | 4.999±0.000 |

### 2) 配对差值（变体 − v4，同 (name, scenario) 对齐）

| 变体 | 场景 | 指标 | n | Δmean | bootstrap 95%CI | 降/升 | 结论 |
|---|---|---|---|---|---|---|---|
| lossctrl_240ep | dt | echo_mos | 154 | -0.146 | [-0.187, -0.109] | 115/39 | 显著退化 |
| lossctrl_240ep | dt | deg_mos | 154 | -0.092 | [-0.152, -0.033] | 95/59 | 显著退化 |
| lossctrl_240ep | nst | echo_mos | 76 | -0.001 | [-0.001, -0.000] | 56/20 | 显著退化 |
| lossctrl_240ep | nst | deg_mos | 76 | -0.086 | [-0.160, -0.007] | 46/30 | 显著退化 |
| lossctrl_240ep | fest | echo_mos | 70 | -1.191 | [-1.446, -0.927] | 60/10 | 显著退化 |
| lossctrl_240ep | fest | deg_mos | 70 | +0.000 | [+0.000, +0.000] | 22/48 | 显著回升 |
| psycho_240ep | dt | echo_mos | 154 | -0.285 | [-0.338, -0.234] | 136/18 | 显著退化 |
| psycho_240ep | dt | deg_mos | 154 | +0.059 | [-0.006, +0.121] | 72/82 | CI 含 0（不显著） |
| psycho_240ep | nst | echo_mos | 76 | -0.001 | [-0.002, -0.001] | 58/18 | 显著退化 |
| psycho_240ep | nst | deg_mos | 76 | +0.037 | [-0.038, +0.115] | 34/42 | CI 含 0（不显著） |
| psycho_240ep | fest | echo_mos | 70 | -1.534 | [-1.748, -1.309] | 65/5 | 显著退化 |
| psycho_240ep | fest | deg_mos | 70 | +0.000 | [+0.000, +0.001] | 10/60 | 显著回升 |
| lossctrl_240ep_seed1 | dt | echo_mos | 154 | -0.478 | [-0.535, -0.425] | 147/7 | 显著退化 |
| lossctrl_240ep_seed1 | dt | deg_mos | 154 | +0.044 | [-0.021, +0.110] | 70/84 | CI 含 0（不显著） |
| lossctrl_240ep_seed1 | nst | echo_mos | 76 | -0.001 | [-0.001, -0.000] | 59/17 | 显著退化 |
| lossctrl_240ep_seed1 | nst | deg_mos | 76 | -0.015 | [-0.077, +0.044] | 36/40 | CI 含 0（不显著） |
| lossctrl_240ep_seed1 | fest | echo_mos | 70 | -1.531 | [-1.757, -1.307] | 65/5 | 显著退化 |
| lossctrl_240ep_seed1 | fest | deg_mos | 70 | +0.000 | [+0.000, +0.000] | 15/55 | 显著回升 |
| psycho_240ep_seed1 | dt | echo_mos | 154 | -0.323 | [-0.382, -0.272] | 138/16 | 显著退化 |
| psycho_240ep_seed1 | dt | deg_mos | 154 | +0.038 | [-0.032, +0.107] | 74/80 | CI 含 0（不显著） |
| psycho_240ep_seed1 | nst | echo_mos | 76 | -0.001 | [-0.002, -0.001] | 61/15 | 显著退化 |
| psycho_240ep_seed1 | nst | deg_mos | 76 | +0.008 | [-0.077, +0.090] | 37/39 | CI 含 0（不显著） |
| psycho_240ep_seed1 | fest | echo_mos | 70 | -1.508 | [-1.719, -1.292] | 65/5 | 显著退化 |
| psycho_240ep_seed1 | fest | deg_mos | 70 | +0.000 | [+0.000, +0.001] | 9/61 | 显著回升 |
| psycho_cfix_seed1 | dt | echo_mos | 154 | -0.313 | [-0.368, -0.259] | 142/12 | 显著退化 |
| psycho_cfix_seed1 | dt | deg_mos | 154 | +0.020 | [-0.057, +0.091] | 68/86 | CI 含 0（不显著） |
| psycho_cfix_seed1 | nst | echo_mos | 76 | -0.001 | [-0.002, -0.001] | 61/15 | 显著退化 |
| psycho_cfix_seed1 | nst | deg_mos | 76 | +0.010 | [-0.071, +0.097] | 35/41 | CI 含 0（不显著） |
| psycho_cfix_seed1 | fest | echo_mos | 70 | -1.341 | [-1.575, -1.080] | 59/11 | 显著退化 |
| psycho_cfix_seed1 | fest | deg_mos | 70 | +0.000 | [+0.000, +0.000] | 15/55 | 显著回升 |
| lossctrl_v4replica | dt | echo_mos | 154 | -0.001 | [-0.002, -0.000] | 85/69 | 显著退化 |
| lossctrl_v4replica | dt | deg_mos | 154 | +0.001 | [-0.000, +0.003] | 63/91 | CI 含 0（不显著） |
| lossctrl_v4replica | nst | echo_mos | 76 | +0.000 | [+0.000, +0.000] | 27/40 | 显著回升 |
| lossctrl_v4replica | nst | deg_mos | 76 | -0.000 | [-0.001, +0.000] | 40/36 | CI 含 0（不显著） |
| lossctrl_v4replica | fest | echo_mos | 70 | +0.031 | [+0.022, +0.040] | 8/62 | 显著回升 |
| lossctrl_v4replica | fest | deg_mos | 70 | +0.000 | [-0.000, +0.000] | 24/43 | CI 含 0（不显著） |

## 变体标签

- `v4` = v4（存量，对照锚）
- `lossctrl` = ①V1 原损失列（频带均衡+DT 加权）
- `lossctrl_v4replica` = ①b 复刻校验（v3 续训 240ep）
- `psycho` = ②心理声学损失列（PAQM 派生）
- `flow` = ③flow matching 列
