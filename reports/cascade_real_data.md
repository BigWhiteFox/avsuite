# cascade_real_data — 级联 MDF→NN 上真实数据（官方 ICASSP2022 集 800 段 + eval_v6_realrir 100 条）

- 日期：2026-10-11 · 执行：S11（V4 整改第四波）· 目的：给"生产推荐形态"（级联 MDF→NN，exp/09_cascade_aec）补第一份**真实数据**证据
- 约束遵守：未改任何既有文件（只新建驱动脚本/两份 json/jsonl/本报告）；未训练；无 git 写操作；docs/NUMBERS.md 未动（Lead 单编辑权）；enh/lin 中间 wav 全部落 `D:/data`（仓外）
- 环境：CUDA 可用（NVIDIA GeForce RTX 4060 Laptop，torch 2.6.0+cu124）；NN 前向走 GPU；MDF/打分走 CPU

## 0. 结论速览

1. **echo 优势在真实分布上保持**：官方集 FEST 级联 3.096±0.135 vs 传统 2.725±0.126（配对 Δ+0.371 [0.295,0.448]）、DT 级联 3.233±0.122 vs 传统 2.305±0.100（Δ+0.932 [0.847,1.016]）；eval_v6 DT 级联 2.627±0.241 vs 传统 2.000±0.15（Δ+0.628 [0.407,0.858]）——全部显著。级联与 NN v4 的 echo 无显著差（官方 −0.05/−0.03、v6 +0.06，CI 均含 0）：**级联把 NN 的 echo 增益完整带到了真实分布**。
2. **deg 相对档位保持**：eval_v6 DT deg 级联 1.842±0.090，与合成集同档（eval_v5 级联 1.851——真实 RIR 换掉仿真 RIR 后 deg 几乎不动）；官方集 deg 绝对量级整体上移（真实录制 vs 合成的已知口径现象，ood_real_echoes.md §4：deg +1.2~+1.3），只比同级相对差：级联 DT deg 2.947 仍坐 NN 端（比 NN −0.151 [−0.262,−0.043]、比传统 −0.751 [−0.830,−0.676]），与合成集"级联≈NN<传统"的相对次序完全一致。
3. **发散条不救**：`wufNfJIb3kC7OvuxutARLQ_farend-singletalk-with-movement` 在级联下线性级靠 S1 发散防护保持**有限**（MDF 复位 8 次，e_lin 无非有限样本），但 NN 把病态残余放大到峰值 4821.5 → 打分侧同一发散防护（非有限或 max|·|>1e3 拒送评）**拒送评**——与传统列同一条被拒、同一防线，崩溃点从"线性级 inf"后移为"NN 端爆表"。另新发现 1 条 DT movement（GSJyjeKff…，1 次复位、峰值 1240）也被拒。全 800 段共 5 条 with-movement 触发 MDF 复位（13 次），其中 3 条仍正常送评。

## 1. 管线与口径（铁律核对）

| 项 | 实现 | 训练分布一致性 |
|---|---|---|
| 线性级 | `TradAEC(method="mdf", partitions=8, mu=0.7)`（exp/06_trad_aec/pipeline.py），`res=process(mic.astype(f64), far.astype(f64))`，`y_echo=res.y_echo[:n].astype(f32)`，`e_lin=(mic−y_echo).astype(f32)` | **逐位复刻 precompute_cascade.py**（同 MDF 配置/同数据流/同 float64→float32 收口；官方侧 48k→16k 的 `resample_poly(1,3)` 实测输出 dtype=float32，减法为 float32 算术，已核对） |
| 加固参数 | `align_margin_ms=0`、`reestimate_sec=None`（**不开**）；发散防护默认开（mdf `w_abs_cap=1e3`+发散复位） | 与 HANDOFF §4.2 决策一致：防护保留、margin/reestimate 关（训练-推理一致） |
| NN | `NNAEC`（exp/07_nn_aec/model.py），`model.enhance(e_lin, y_echo)`，ckpt `exp/09_cascade_aec/outputs/nnaec_g0.5_j10/best.ckpt`（val≈24.00），加载方式同 run_cascade.py（torch.load→load_state_dict） | ✅ 同 run_cascade.py |
| 官方集打分 | librosa 48k 回采 → min_len → 发散防护（同 official_aecmos_eval.score_job）→ `trim_for("proto")`（FEST 后半段 / DT 最后 (len−15)/2 / NST 全段）→ `CachedAECMOS.run`（Run_1668423760_Stage_0.onnx）——**全部复用 official_aecmos_eval.py** | 与 ood_real_echoes.md 的 trad/NN 列同协议同口径 |
| eval_v6 打分 | sf.read float32 → min_len → `AECMOSEstimator.run`（16k 全段直送，无 trim）——aecmos_eval.py 口径逐字复刻 | 与 aecmos_trad_v6.json / aecmos_nn_v4.json 同口径 |

断点续跑：lin/enh wav 已存在(>1KB)跳过、打分行已写 jsonl 跳过（本轮一次跑通，未实际触发续跑恢复）。

## 2. 官方集 800 段（ICASSP2022 真实录制，proto 协议）

### 2.1 三列对照（echo_mos / deg_mos，mean ± 95%CI = 1.96·sd/√n）

| 场景·指标 | 传统 AEC（WP-06，旧值¹） | NN-AEC v4（旧值¹） | **级联 MDF→NN（新）** | 级联 n |
|---|---|---|---|---|
| FEST echo | 2.725 ± 0.126 (n=299) | 3.143 ± 0.151 (n=300) | **3.096 ± 0.135** | 299 |
| FEST deg | 5.000 | 4.999 | 4.999 | 299 |
| DT echo | 2.305 ± 0.100 (n=300) | 3.256 ± 0.124 (n=300) | **3.233 ± 0.122** | 299 |
| DT deg | 3.692 ± 0.088 | 3.097 ± 0.118 | **2.947 ± 0.103** | 299 |
| NST echo | 4.999 | 4.998 | 4.999 | 200 |
| NST deg | 4.183 ± 0.046 | 3.759 ± 0.119 | **3.947 ± 0.083** | 200 |

¹ 传统/NN 两列引自 `reports/ood_real_echoes.md`（同集、同 proto 协议、同发散防护），本报告未重跑这两列。

### 2.2 配对差值（级联 − 参照列，按 (subset,name) 对齐；bootstrap 95%CI，B=2000，seed=777，同 tools/compute_ci.py 约定）

| 对比 | 场景·指标 | n | Δ | bootstrap 95%CI | 显著？ |
|---|---|---|---|---|---|
| 级联−传统 | FEST echo | 299 | **+0.371** | [+0.295, +0.448] | 显著 |
| 级联−传统 | DT echo | 299 | **+0.932** | [+0.847, +1.016] | 显著 |
| 级联−传统 | DT deg | 299 | **−0.751** | [−0.830, −0.676] | 显著（更差） |
| 级联−传统 | NST deg | 200 | −0.235 | [−0.303, −0.171] | 显著（更差） |
| 级联−NN | FEST echo | 299 | −0.052 | [−0.222, +0.115] | 不显著 |
| 级联−NN | DT echo | 299 | −0.027 | [−0.166, +0.112] | 不显著 |
| 级联−NN | DT deg | 299 | −0.151 | [−0.262, −0.043] | 显著（略差于 NN） |
| 级联−NN | NST deg | 200 | +0.188 | [+0.078, +0.307] | 显著（优于 NN） |

注：配对 n=299 是交集——FEST 各自拒掉同一条 wufNfJIb（旧值列 n=299 亦因它）；DT 拒掉级联新增的 GSJyjeKff（旧值列 n=300 含它，其 trad 3.418/NN 2.268）。拒评条处理与旧口径一致（不计入均值，单独记录，见 §4）。

## 3. eval_v6_realrir 100 条（真实车厢 RIR，16k 原生，全段口径）

### 3.1 三列对照

| 场景·指标 | 传统 AEC（旧值²） | NN-AEC v4（旧值²） | **级联（新）** |
|---|---|---|---|
| FEST echo (n=34) | 3.301 ± 0.43 | 3.718 ± 0.45 | **3.406 ± 0.428** |
| FEST deg | 5.000 | 5.000 | 5.000 |
| DT echo (n=42) | 2.000 ± 0.15 | 2.565 ± 0.23 | **2.627 ± 0.241** |
| DT deg | 2.518 ± 0.15 | 2.044 ± 0.12 | **1.842 ± 0.090** |
| NST echo (n=24) | 5.000 | 5.000 | 5.000 |
| NST deg | 2.588 ± 0.15 | 2.051 ± 0.13 | **2.068 ± 0.145** |

² 引自 `reports/realrir_domain.md` §3（aecmos_trad_v6.json / aecmos_nn_v4.json，同口径全段直送）。

### 3.2 配对差值（同 §2.2 方法）

| 对比 | 场景·指标 | n | Δ | bootstrap 95%CI | 显著？ |
|---|---|---|---|---|---|
| 级联−传统 | DT echo | 42 | **+0.628** | [+0.407, +0.858] | 显著 |
| 级联−传统 | FEST echo | 34 | +0.105 | [−0.200, +0.396] | 不显著（n 小、CI 宽） |
| 级联−传统 | DT deg | 42 | −0.676 | [−0.830, −0.522] | 显著（更差） |
| 级联−NN | DT echo | 42 | +0.062 | [−0.230, +0.331] | 不显著 |
| 级联−NN | DT deg | 42 | −0.202 | [−0.296, −0.116] | 显著（略差于 NN） |
| 级联−NN | NST deg | 24 | +0.016 | [−0.073, +0.117] | 不显著 |

v6 全 100 条无发散复位（合成数据、MDF 常规收敛），级联输出全部有限。

## 4. 发散条（with-movement）专项记录

官方集 800 段中 **5 条触发 MDF 发散防护复位**（共 13 次），全部为 with-movement（真实可变回声路径）条：

| # | 条目 | subset | MDF 复位次数 | 级联 enh 峰值 | 级联处置 | 参照列处置 |
|---|---|---|---|---|---|---|
| 1 | wufNfJIb3kC7OvuxutARLQ_farest…movement | fest | **8** | **4821.5** | 拒送评（enh_diverged） | trad：非有限样本拒送评（旧）；NN：1.760/5.0 |
| 2 | GSJyjeKffECLYwJlPJQq0Q_doubletalk…movement | dt | 1 | **1239.8** | 拒送评（enh_diverged） | trad 3.418/1.939、NN 2.268/2.621（均可评） |
| 3 | LJHHM480yUqT0tBrgwTv5A_farend…movement | fest | 1 | 787.4 | 正常送评 | — |
| 4 | qM4dVLOxTUaDoIa6PnWmrw_farend…movement | fest | 2 | <1e3 | 正常送评 | — |
| 5 | d039d2dc-120f-…farend…movement | fest | 1 | 75.9 | 正常送评 | — |

（另有 SeEXVNB1… fest 峰值 118.3，无 MDF 复位、正常送评。）

要点：
- **S1 发散防护在本管线有效**：5 条病条的线性级输出（y_echo/e_lin）全部有限（0 非有限样本），wufNfJIb 不再像旧传统列那样产生 173,816 个非有限样本。
- **但级联不修复病条**：wufNfJIb 的病态残余（MDF 复位块 y=0 → e_lin=mic 段）经 NN 转写后峰值放大到 4821.5，触发打分侧同一发散防线（max|·|>1e3）拒送评——与传统列"同一条被拒"，只是崩溃点从线性级后移到 NN 级。
- **级联新增 1 条拒评**（GSJyjeKff DT，1/800）：传统/NN 在该条均可评；其线性级本就有 1 次复位，NN 输出峰值 1239.8 过线。合计级联拒评 2/800（传统 1/800、NN 0/800）。
- 含义：对 with-movement 病条，根治手段是线性级的 margin/reestimate（S10 验证方向），级联结构本身不提供额外防护。

## 5. 三个问题的直接回答

**（a）级联的 FEST/echo 优势在真实分布上是否保持？——保持。**
官方集 FEST +0.371 [+0.295,+0.448]、DT +0.932 [+0.847,+1.016]（vs 传统，配对显著），与 NN v4 的 +0.423/+0.951 同量级；v6 DT +0.628 [+0.407,+0.858]（真实车厢 RIR 上显著）。级联与 NN 的 echo 差在两套真实数据上均不显著（官方 −0.052/−0.027、v6 +0.062，CI 含 0）——级联在真实分布上完整保住了 NN 的 echo 增益，没有因换输入对（e_lin/y_echo）而损失。唯一不显著的是 v6 FEST（+0.105 [−0.200,+0.396]），n=34、CI 宽，方向仍为正。

**（b）deg 是否仍 ~1.8-1.9 档？——合成同配方集上精确保持；真实录制上按已知量级上移、相对次序不变。**
eval_v6（与 eval_v5 唯一差异=真实 RIR）DT deg 1.842±0.090，与 eval_v5 级联 1.851 几乎逐位同档；官方集真实录制的 deg 绝对量级整体上移（对齐 ood_real_echoes.md §4 记录的 +1.2~+1.3 口径现象，传统 2.43→3.69、NN 1.81→3.10），只比同级相对差：级联 2.947 比 NN −0.151 [−0.262,−0.043]、比传统 −0.751 [−0.830,−0.676]，而合成集（v5）为比 NN −0.04、比传统 −0.58——"级联≈NN<传统"的相对次序与量级在真实分布上复现；v6 上级联对传统的 deg 恶化幅度（−0.676）也与官方集（−0.751）一致。级联仍以 deg 换 echo，交易结构与合成集一致。

**（c）发散条在级联下的表现？——防护保住线性级、救不回送评（§4）。**
wufNfJIb…movement：级联线性级 8 次复位、输出有限（S1 防护生效），但 NN 端峰值 4821.5 被同一发散防线拒送评；GSJyjeKff…movement（DT）新增拒评。级联把该族条的失败模式从"线性级 inf"改为"NN 端爆表"，5/800 条受影响（2 拒评 + 3 正常），与"级联不解决 with-movement 病条，根治靠线性级 margin/reestimate"的判断一致。

## 6. 与合成集级联数字的对照（同管线同 ckpt）

| 集 | FEST echo | DT echo | DT deg | 说明 |
|---|---|---|---|---|
| eval_v3（合成,8s 全段） | 4.502 | 3.919 | 1.785 | results/aecmos_cascade.json |
| eval_v5（合成 RIR） | 4.589 | 3.741 | 1.851 | results/aecmos_cascade_v5.json |
| eval_v6（真实车厢 RIR） | 3.406 | 2.627 | 1.842 | 本报告 |
| 官方集（真实录制,proto） | 3.096 | 3.233 | 2.947 | 本报告（deg 量级口径不同,见 §5b） |

真实 RIR（v5→v6）主要打掉 echo（DT −1.11）而 deg 几乎不动（1.851→1.842）；真实录制（官方集）再叠加 deg 口径量级上移。

## 7. 重跑命令（全部，workdir=仓库根 D:\Project_for_Python\DSH\avsuite）

```powershell
$env:PYTHONNOUSERSITE='1'
$py = 'C:\Users\14805\.conda\envs\avsuite\python.exe'
# 官方集 800 段（>60s 的子命令后台跑）
& $py exp/09_cascade_aec/cascade_real_data.py lin-official --workers 7    # MDF 线性级 → D:/data/aec-synth/official_stage/cascade_lin/
& $py exp/09_cascade_aec/cascade_real_data.py enh-official                # 级联 NN(GPU) → D:/data/aec-synth/official_stage/enh_cascade/
& $py exp/09_cascade_aec/cascade_real_data.py score-official --workers 6  # proto 协议打分 → results/scores_cascade_official.jsonl
& $py exp/09_cascade_aec/cascade_real_data.py agg-official                # 汇总+CI+配对 → results/aecmos_cascade_official.json
# eval_v6_realrir 100 条
& $py exp/09_cascade_aec/cascade_real_data.py lin-v6 --workers 7
& $py exp/09_cascade_aec/cascade_real_data.py enh-v6
& $py exp/09_cascade_aec/cascade_real_data.py score-v6 --workers 6
& $py exp/09_cascade_aec/cascade_real_data.py agg-v6                      # → results/aecmos_cascade_v6.json
# 冒烟：任一子命令加 --limit N（每子集/总数前 N 条）；--subsets fest dt nst 可选子集
```

## 8. 产物清单与资源消耗

| 产物 | 路径 |
|---|---|
| 驱动脚本（新建） | `exp/09_cascade_aec/cascade_real_data.py` |
| 官方集汇总（新建） | `exp/09_cascade_aec/results/aecmos_cascade_official.json`（含逐条 rows + n_div） |
| v6 汇总（新建） | `exp/09_cascade_aec/results/aecmos_cascade_v6.json` |
| 官方集逐条打分 | `exp/09_cascade_aec/results/scores_cascade_official.jsonl` |
| v6 逐条打分 | `exp/09_cascade_aec/results/scores_cascade_v6.jsonl` |
| 线性级中间 wav（仓外） | `D:/data/aec-synth/official_stage/cascade_lin/{fest,dt,nst}/{mic,farend}/*.wav`（e_lin/y_echo，16k FLOAT）+ `manifest_lin.jsonl` |
| 级联增强 wav（仓外） | `D:/data/aec-synth/official_stage/enh_cascade/{fest,dt,nst}/*.wav`（800 条，16k FLOAT）+ `D:/data/aec-synth/official_stage/manifest_cascade_enh.jsonl` |
| v6 线性级/增强 wav（仓外） | `D:/data/aec-synth/cascade_stage/eval_v6/{lin_cascade,enh_cascade}/` + `manifest_{lin,enh}.jsonl` |

资源：官方集线性级 172s（794 条并行 ×7 进程，~1.5s/条 CPU float64 MDF）；NN 前向 15s（GPU，~0.02s/条）；打分 30s（×6 进程）；v6 三阶段合计 ~30s。总实验耗时 ~25 min（含冒烟/修 bug/汇总），GPU 仅用于 NN 前向（峰值显存低），MDF 与打分为 CPU。断点续跑已内置（wav 存在性 + 打分 jsonl 去重）。
