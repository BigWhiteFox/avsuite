# t24 · v5x 扩展代际级联正式版（新服务器执行）· 收档

- 状态：**两跑训练中**（v5 / v5x 并发从零 100ep）；本档在训练完成前写好框架，末节回填最终读数
- 服务器：`117.50.185.156:23`（与旧机同规格：Xeon Gold 6248R，**5 物理核/10 逻辑**，V100S-32GB，62G 内存，123G 盘）
- 任务口径（队长）：级联 v5 与 v5x 正式版**从零** 100ep（损失权重变过，**不可续训**）；显式 `--w-sisnr 1.0 --gamma 0.5 --loss weighted`；`--rir-list split_lists/rir_train70.txt`；净池；多源权重 far DNS 0.5/LS 0.3/AISHELL-3 0.2 + noise DNS 0.7/MUSAN 0.3

---

## 1. bring-up 全流程与证据

| 步骤 | 做法 | 结果 |
|---|---|---|
| 代码 | `avsuite_bringup.tar`（336 文件/19.37MB，含 t19–t27 全部增量）→ 解包 `/root/avsuite` | ✅ 关键文件在位断言通过 |
| tmux | `apt-get install tmux`（3.2a） | ✅（**关键**：该机 ssh 会杀掉 nohup 后台任务，一律用 tmux） |
| 环境 | venv + `torch==2.6.0 --index-url .../cu124` + requirements（torch 行注释）+ `numpy==1.26.4` + `pip cache purge` | ✅ `env_check.py --strict` = **PASS(exit 0)**；V100S 识别；日志 `t24_env.log` |
| 数据 B（ModelScope 快路） | 逐个"下载 → 校验字节数 → 解压 → 删包" | ✅ MUSAN 12G、LS 29G(132,553 flac)、AISHELL-3 26G(88,035 wav)；日志 `t24_downloads.log` |
| 数据 A（本机 scp） | 8 个 tar，**分块 300MB + 60s 退避**（该机大文件单发易断） | ✅ train70 7152 / val15 1256 / holdout15 1592 / dns-noise 659 / rirs_train70 20 / eval_v8 4200 / eval_v8_lin 4200 / eval_v8_pools 993+599+95 |
| 前置校验 | 在位 + RIR 集合/顺序/指纹 + 多源池 dry-run | ✅ `t24_preflight.log`：RIR **集合相同+顺序相同**，`set_sha256 = list_sha256 = a9383200…`（= 旧机基线） |

**ModelScope 通道（留档）**：`https://modelscope.cn/datasets/<org>/<name>/resolve/master/<path>`（HF `/tree`、`/oss/tree` 均返回空）
- `pkufool/LibriSpeech` → `train-clean-100.tar.gz`(6,387,309,499 B) / `train-clean-360.tar.gz`(23,049,477,885 B)
- `manyeyes/MUSAN` → `raw/musan.tar.gz`(11,086,114,085 B)
- `OmniData/AISHELL-3` → `raw/data_aishell3.tgz`(19,057,141,777 B)
- 实测 **~17 MB/s**（与 t21 一致），三语料合计 ~60GB ≈ 70 分钟

## 2. 两跑 manifest 关键字段（已核，`t24_preflight.log` + manifest 回读）

| 字段 | casc_v5_official | casc_v5x_official |
|---|---|---|
| loss | `{kind: weighted, gamma: 0.5, w_sisnr: 1.0}` | 同 |
| run.from_scratch | True | True |
| val | `/root/data/aec-synth/eval_v8_lin` | 同 |
| n_files | near 3477 / far 3675 / noise 472 / **rir 20** | near 3477 / **far 149,280** / **noise 1,892** / rir 20 |
| far 源 | train70 1.0 | dns_far 0.5(3675) / ls_far 0.3(90418) / aishell3_far 0.2(55187) |
| noise 源 | dns-noise 1.0 | dns_noise 0.7(472) / musan 0.3(1420) |
| **`data.rir`** | `src=list`、`list_sha256=set_sha256=a9383200bf29752a0b5b9fe6e7d858a8e06bfb39c1f451cc98380ba4dda80d5d`、`n_files 20`、`enumeration=list_order`、**`order_matches_pool=true`**、`extra_pool_files=0` | 同 |
| epoch_scheme | 方案(a) 408 步/ep × 100ep、epoch_len 6528、Δ−0.073% | 同 |
| lr_sched | plateau(0.5, patience 20) + early-stop 0 | 同 |
| linear_stage | `cascade_mdf`、`fresh_per_clip`、`equivalent_to_static_precompute=false`、`far_jitter_order=mdf_then_shift_y_echo` | 同 |

## 3. 供给/资源实测

- 两跑并发：进程 24（2 主 + 2×10 worker），load **~19.6–21**（5 物理核）；GPU 显存 ~2.6GB、利用率间歇（**CPU/MDF 受限**，与 D1 结论一致）
- 每 epoch **~293–309 s**（408 步 × 16 = 6528 样本/ep）⇒ 100ep ≈ **8.3h**（两跑同刻推进）

## 4. 最终读数（**已完训并回填**）

| run | epochs | step | **best val** | best@ep | last val | far 源 | noise 源 |
|---|---|---|---|---|---|---|---|
| **casc_v5_official** | 100 | 40800 | **7.855140** | ep96 | 7.994863 | train70 1.0 (3675) | dns-noise 1.0 (472) |
| **casc_v5x_official** | 100 | 40800 | **7.722963** | ep97 | 8.346754 | dns_far 0.5(3675)/ls_far 0.3(90418)/aishell3_far 0.2(55187) | dns_noise 0.7(472)/musan 0.3(1420) |
| **Δ（v5x − v5）** | — | — | **−0.132177** | — | — | — | — |

- 口径：`val = eval_v8_lin`（1400 clip，`fresh_per_clip` 预计算）；`loss = {weighted, gamma 0.5, w_sisnr 1.0}`；两跑均 `from_scratch`、100ep、step 40800（目标 40830，Δ−30 = −0.073%）；`lr_sched = plateau(0.5, patience 20) + early-stop 0`
- **`data.rir`（t26/t27 新块）**：`src=list`、`list_sha256 = set_sha256 = a9383200bf29752a0b5b9fe6e7d858a8e06bfb39c1f451cc98380ba4dda80d5d`（= 旧机净池基线）、`n_files=20`、**`order_matches_pool=true`**、`extra_pool_files=0`
- `best.ckpt`/`last.ckpt` 在**服务器与本机各实 load 一次**，epoch/val 与 history 逐一吻合
- 时间：08:24/08:26 UTC 起跑 → **16:43 UTC 收尾**（≈8.3h，与排期一致）；~293–309 s/epoch
- 产物：服务器 `/root/avsuite/outputs_p1/casc_{v5,v5x}_official/nnaec_g0.5_j10_dm_casc/{best,last}.ckpt,history.json,train_manifest.json,tb/`；**本机镜像 `p1_artifacts/outputs_p1/casc_{v5,v5x}_official/`**（各 ~18.6MB）
- 交付动作：① 已报 captain；② **已通知 eval-eng** 用本两行替换 t13 矩阵的截断版（截断版缺陷：`w_sisnr=0.1` + 只到 ep79/76）

## 5. 过程中的真实问题与处置（留痕）

1. **该机 ssh 会杀掉 nohup 后台任务** ⇒ 改用 **tmux**（`dl`/`env`/`v5`/`v5x` 会话）；再用 `setsid` 也无效。
2. **大文件 scp 周期断开**（`Connection closed` / `kex_exchange_identification: read: Connection reset`），根因是**本机上行被并发传输占满 + 我误留一个并行重试进程抢带宽** ⇒ 处置：串行化 + **300MB 分块 + 60s 退避**（失败只损失一块），并停掉抢带宽的旧进程。
3. **`--far-source` 的 `|` 分隔符经 ssh 嵌套引号被 shell 当管道** ⇒ v5x 首启失败（v5 正常）；改为**脚本文件 + base64 传输 + `bash -n` 语法闸**，杜绝引号风险。
4. **`rirs_train70` 只上传未解包**（前置闸报 `MISS`）⇒ 解包后 20 条到位；若没这道闸会带病开跑（DM 池校验也会拦，但会浪费一轮启动）。
5. **PowerShell 读写含中文的 .ps1/.sh 编码坑**：`Get-Content -Raw` 默认按 ANSI 读 ⇒ 莫吉乱码 + 语法错误；改用 `[IO.File]::ReadAllText(path, UTF8)` 后解决（并把服务器脚本尽量写成 ASCII）。

## 6. 与 t12 的关系（代际口径声明）

- v5x 代际的 NN/DCCRN/noMUSAN 已在**旧机**完训（t12，产物在本机 `p1_artifacts/`），**不重跑**；
- 本档补齐的是**级联的正式版**（旧机因关机只跑到 ep79/76 且 `w_sisnr=0.1`）；
- 三个口径警示在 t12 收档（`reports/t12_dm_retrain_20261014.md`）已记：RIR 池污染、04 `--epoch-len` 默认值、`--w-sisnr` 漏传；本跑全部显式带上，且 RIR 走 `--rir-list`（`order_matches_pool=true`）。

## 7. 交付后的两点澄清（t13 接口）

1. **`scenario_ratio = {fest 0.0, nst 0.5086, dt 0.4914}` 是预期口径，不是漏配**
   - 7 个 run 的 manifest 全部为该值 + `scenarios_explicit=true` + `scenario_ratio_source="cli_override"`（`nn_aec_v5`/`v5x`/`v5x_noMUSAN`/`dccrn_dm`/`dccrn_v5x`/`casc_v5_official`/`casc_v5x_official`）；
   - DT 计划 preset `train_v1_repro` 原值是 `{fest 0.4, nst 0.3, dt 0.3}`（manifest `dt_plan.scenarios` 如实记录），所有 DM run 都**显式覆盖**为 `--dm-scenarios 0,0.5086,0.4914`（复现 v4 口径的一部分）。
   - ⇒ FEST 读数应标注 **"训练侧零样本（fest 权重 0，全 v5/v5x 代际一致）"**；因各被比模型同为 FEST 零样本，**跨模型对照仍有效**。
   - 附注：`eval_v8` / `eval_v8_lin` **含 FEST 片段**（eval_v8 = dt706/nst384/**fest310**）⇒ 零样本仅指训练侧，**val/选型过程确实见过 FEST**，写报告时需加此限定。
2. **截断版已归档隔离**：`p1_artifacts/archives/casc_truncated/{casc_dm,casc_v5x}` + `README.md`（写明 `w_sisnr=0.1`、ep79/76、**不得进矩阵**）；`p1_artifacts/outputs_p1/` 现仅存 7 套正式产物（级联正式版带 `_official` 后缀）。

## 8. t13 接入口径确认（eval-eng 的 `kind=cascade`）

eval-eng 的实现与本跑 manifest 的 `data.linear_stage` **逐字一致**：`TradAEC(mdf, partitions=8, mu=0.7)` **每 clip 新实例**（=`trad_instance_policy: fresh_per_clip`）、`e_lin = mic − y_echo`（RES 前）、far jitter 在 MDF 之后作用于 `y_echo`（`far_jitter_order: mdf_then_shift_y_echo`）。其"每个测试集现算线性级、不依赖静态预计算目录"的做法是**正确**的——静态集跨 clip 复用同一实例，与本口径**不等价**（manifest `equivalent_to_static_precompute=false` 已声明）。

## 9. FEST 机理交叉引用（t13 B 档发现 + MDF-only 消融，训练侧佐证）

**结论（eval-eng t13 B 档 + t28 消融）**：正式级联的 **FEST echo 反而最好**（`casc_v5x` @ eval_v8 FEST echo **3.80**，对比纯 NN 1.63 / v4 2.90 / trad 3.00），而级联的 NN 同样**训练侧未见 FEST**。
⇒ **"FEST 弱点是纯 NN 架构的属性"，不是"训练数据没给 FEST"的后果**（否则无法解释级联）。级联 vs 同代际纯 NN 的 FEST echo 差 **−1.8…−2.4**（t13 逐行配对）。

### 9.1 MDF-only 消融（eval-eng t28，`kind=linear`：线性级输出直接当结果、不接 NN）

| eval_v8（FEST 段 310 条） | FEST echo | FEST ERLE 稳态 | DT echo | DT deg |
|---|---|---|---|---|
| **线性级（仅 MDF）** | **3.208** | 14.17 dB | 2.473 | 4.076 |
| trad（全链路 WP-06） | 3.003 | 19.91 dB | 3.105 | 3.822 |
| v4 | 2.896 | 65.56 dB | 4.106 | 3.458 |
| 纯 NN `nn_v5x` | 1.625 | 2.50 dB | 3.227 | 3.874 |
| **级联 `casc_v5x`** | **3.795** | 32.37 dB | **4.324** | 3.998 |

- **线性级单独就超过 trad / v4 / 纯 NN**（3.208 vs 3.003 / 2.896 / 1.625）⇒ "FEST 由架构里的线性级供给"由推断升为**直接测量**；级联在其上再叠 **+0.59**（FEST，n=310；`eval_rir` 同型 2.999 → 4.008，**+1.01**）。
- 逐行数据：`exp/07_nn_aec/outputs/eval_matrix/linear/{eval_v8,eval_rir}.json`；报告 `reports/p1_6_matrix_eval_20261014.md` §10-12（消融表）/ §0-5（机理句）。

### 9.2 口径收窄：**两级互补**，不是"级联只是不破坏线性级"（重要）

- 级联**不等于**"线性级 + 不破坏"：NN 有实测增量——FEST **+0.59**、**DT 2.473 → 4.324 = +1.85**。
- 正确表述：**线性级供给 FEST 抑制的底子；NN 在 DT 上贡献更大（两级互补）**。**不得**据此读成"级联里的 NN 可以省"。
- 线性级的 DT 画像是"**不伤近端但压不住回声**"（echo 低 2.473、deg 高 4.076）——正是级联 NN 要补的那一半（级联 DT 4.324/3.998）。本档原句"级联只是不破坏它"已按 eval-eng 反馈更正（原表述引自 B 档初版，t28 消融后收窄为互补）。

### 9.3 训练侧佐证（本档可查）

1. 级联的 NN 输入是 `e_lin = mic − y_echo`，即**已经过 MDF 线性消回声**的信号（`data.linear_stage.e_lin_formula`）——NN 从未直接面对回声主导的 `mic` ⇒ FEST 上"谁提供抑制底子"在数据流上就写明了是 MDF。
2. ERLE 与 FEST echo 排序**同序**：稳态 ERLE 级联 26–34 dB / trad 19.91 / 线性级 14.17 / 纯 NN 2.50 / DCCRN 0.7（v4 65.56）——但 ERLE 高 ≠ AECMOS 高（v4 ERLE 最高而 FEST echo 2.896），说明**该轴要同时看 echo 与 deg**。
3. 级联正式版两跑的训练数据仍是 `fest=0`（§7-1），却拿到 FEST 最好分 ⇒ **FEST 表现不由 FEST 训练样本决定**，而由"架构是否含线性级"决定；这与"零样本训练"并不矛盾。

**唯一反转（如实记录）**：`eval_rir`（真实车厢 IR）上纯 NN 反超（nn_v5x **+0.111**；nn_v5 +0.004 n.s.）——与该轴 ERLE 方向分组一致（t13 已按"观察、非因果"措辞写入其报告）。

