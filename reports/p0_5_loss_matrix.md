# P0-5 损失矩阵三列重训（阶段0 · 0-1）

> 生成：2026-10-13（train-eng）｜数据：**train_v1 不动**（`D:/data/aec-synth/train_v1`，1795 条 dt/nst）
> 评测口径：AECMOS@48k 模型协议（复用 `exp/07_nn_aec/aecmos_eval.py` 打分会话）；配对 95%CI = bootstrap
> **B=2000 / seed=777**（与 `tools/compute_ci.py` 同口径）｜硬件：本机 RTX 4060 Laptop 8GB
> 单变量纪律：只改**谱误差项的加权方式**；数据/模型架构/超参/SI-SNR 项不动。

## 1. 结论先读（TL;DR）

**总判：本轮不支持"心理声学损失列可收回 DT deg 缺口"的假设——效应不稳定、且不在与 v4 的显著区间内。**

| 列 | 谱误差项 | ckpt（`exp/07_nn_aec/`） | eval_v5 DT echo（n=154） | eval_v5 DT deg |
|---|---|---|---|---|
| 对照锚 | —（存量 v4，426ep lineage） | `outputs_v4/nnaec_g0.5_j10/best.ckpt` | **4.248** | 3.220 |
| ① V1 原列（频带均衡 + DT 加权） | `weighted_distortion(γ0.5)` | `outputs_v4_p05/lossctrl_240ep/nnaec_g0.5_j10/best.ckpt` | 4.102 | 3.129 |
| ② 心理声学列（PAQM 派生） | `psychoacoustic_loss`（c=135.17） | `outputs_v4_p05/psycho_240ep/nnaec_g0.5_j10_psycho/best.ckpt` | 3.964 | **3.279** |
| ③ flow matching 类备选 | — | — | — | **skipped**（方案与理由见 §7） |

**四条读数（按证据强度排序）**：

1. **②相对存量 v4 的 deg 差仍不显著**：②−v4 DT deg = **+0.059 [−0.009, +0.125]（CI 含 0）**，
   同时 echo −0.285 [−0.336, −0.236] 显著退化。⇒ 按仓内"CI 不含 0 才算数"的纪律，
   **不得宣称"deg 缺口已收回"**；相对传统基线的缺口从 v4 的 −0.378（3.220 vs 3.598）缩到约 −0.32，仍在同量级。
2. **对称对照（②−①）只在 seed 0 显著，seed 1 归零——效应不可复现**：
   seed0 Δdeg **+0.151 [+0.086, +0.217]**（103/154 改善）；seed1 Δdeg **−0.007 [−0.069, +0.052]（CI 含 0）**。
   两 seed 的 echo 项甚至反号（seed0 −0.139 显著、seed1 +0.155 显著）⇒ 本列改的是
   "echo↔deg 的再平衡点"，而**再平衡的方向/幅度随种子漂移**，不是稳定的 deg 增益。
3. **量级校验**：①列跨 seed 的 deg 极差 0.136、echo 极差 0.332；而 ②−① 的效应量在 seed0 为 +0.151
   —— **效应量≈噪声量级**。这解释了为什么两 seed 不一致，也是"不予确认"的直接依据。
4. **①（原损失、从零训练）比 v4 低**：DT deg 3.129 vs 3.220（−0.092 [−0.152, −0.031] 显著）、
   echo 4.102 vs 4.248（−0.146 显著）。这是**训练预算/谱系差**（v4 是 426ep lineage，本轮两列从零、早停在 ep60/63），
   不是管道漂移——§4 的复刻校验（v3→240ep 重放）把环境钉死在同一读数上。

> **给 DP1 的直接含义**：损失矩阵这一轴**没有拿到可依赖的 deg 增益**（最优情形也只是"与 v4 等同/略高但 CI 含 0"，
> 且伴随 echo 显著下降）。数据侧（隔离 + 扩样）的优先级不应因此下调。

## 2. 损失函数自检（先证"损失指向机理"，再看数字）

V4-S5 实测的 DT 段 deg 短板机理 = **高频带选择性损伤（4–8kHz） + 谱平坦化**（`reports/deg_mechanism.md`）。
基线 `weighted_distortion_loss` 的权重 `w∝|S|^γ`（γ=0.5）在参考能量已衰减的高频带权重自然趋零
⇒ 高频带误差在目标里几乎不可见。心理声学列把"耳蜗压缩 + 临界带可听度加权"显式写进目标
（`core/losses.py::psychoacoustic_loss`）：

```
L_psy = mean_{f,t}[ a[f]·| |S_ref|^α − |S_est|^α | ] / mean[|S_ref|^α]        (α=0.3)
      + λ·E_t[ (ΔE_lo − ΔE_hi)_ref − (ΔE_lo − ΔE_hi)_est ]²                   (λ=0.1, 4–8kHz)
a[f]：ERB(28 带) 三角组 → 逐文件 softmax(L_ref/τ)，τ=1.5（随 s 缩放）
```

单测（`tests/test_losses_psycho.py`，9 例）锁死四条性质：
1. ERB 带中心单调、ERB 域等距（gap 离散度 < 1e-3）；三角组逐 bin 归一（无带宽偏置）、支撑 ≤3 带；
2. 全零/退化形状(1×1)/常量输入恒有限；几何尺度不变（同乘常数损失不变）；
3. 对 est 可微、梯度有限非零（est==ref 时次梯度为 0，属数学事实，单测用受损 est 验证）；
4. **行为一致性**：人为把 est 的 4–8kHz 压低 12dB（=V4-S5 机理的人工复现），损失必须上升——
   这是"损失定义指向该机理"的可证伪断言（若该断言不成立，就没理由期待它拉回 deg）。

> **诚实边界**：①机理自检通过 ≠ 训练后指标回升。本轮把两者的关系切开看：§1 的第 2 条就是"机理成立但效应不稳"。

## 3. 缩放常数 c 的标定（两列谱项量级可比，且如实披露非唯一性）

心理声学损失按参考响度归一（无量纲、幅度不变），与基线 `weighted_distortion`（能量幂律域）
天然不同量级。c 由固定探针实测确定（`exp/07_nn_aec/loss_scale_probe.py` →
`outputs_v4_p05/scale_probe.json`；32 条固定训练样本，seed 1234，jitter 10ms）：

| 状态 | WD 损失 \| 梯度范数 | PSY 损失 \| 梯度范数 | SI-SNR 损失 \| 梯度范数 |
|---|---|---|---|
| 随机初始化 | 298.32 \| 2339.59 | 0.6224 \| 0.4264 | 27.75 \| 320.87 |
| 冻结 v4 | 0.6089 \| 10.43 | 0.3905 \| 6.90 | −16.25 \| 17.26 |

- **c 定义（写入 ckpt cfg）**：c = mean(L_WD)/mean(L_psy)，在**同批训练数据、同一模型状态（初始）**上实测。
  训练启动时按训练集头部 4 批重算 ⇒ seed0 实测 **c = 135.168856**（WD=86.1751 / PSY=0.6375）。
- 同一探针定义下：随机 32 条为 **479.28**、冻结 v4 为 **1.56** ⇒ 三值跨两个数量级。
  根因：**WD 随训练降 ~490×，而 PSY 只降 ~1.6×**（绝对测度 vs 相对测度的固有差异）。
- **自标定在 seed 间也漂移 2.5×**：seed1 的头部 4 批给出 c = **53.71**（WD=35.47/PSY=0.6603）。
  根因：初始模型的 WD 本身方差大、且头部子集与 seed 无关（漂移来自初始化）。
  ⇒ 为切出"纯方差"，另跑了 `psycho_cfix_seed1`（`--psycho-c 135.17 --seed 1`）。
- **留给阶段1的动作**：c 只做了单点（+ 一处固定 c 复测），未做 c∈{1.56, 135, 479} 网格；
  自标定应改为"多 probe 批 + 固定种子"，或直接把 c 当超参网格化。

## 4. 训练（240ep 上限，train_v1 不动）

| run | 输出目录（tag） | 设置 | early-stop | best val | ckpt sha256(前 16) |
|---|---|---|---|---|---|
| ① 原损失 seed0 | `outputs_v4_p05/lossctrl_240ep/nnaec_g0.5_j10` | lr1e-3, bs16, γ0.5, w_sisnr1.0, dt+nst | ep60（best@35） | 31.9724 | `33ff095e0fdeb013` |
| ② 心理声学 seed0 | `outputs_v4_p05/psycho_240ep/nnaec_g0.5_j10_psycho` | 同上 + `--loss psycho`（c 自标定 135.17） | ep63（best@34） | 165.7133¹ | `6284698f1c46c31c` |
| ① 原损失 seed1 | `outputs_v4_p05/lossctrl_240ep_seed1/nnaec_g0.5_j10` | seed1 | ep71（best@45） | 32.7633 | `4e8cde1ee798c5d7` |
| ② 心理声学 seed1 | `outputs_v4_p05/psycho_240ep_seed1/nnaec_g0.5_j10_psycho` | seed1（c 自标定 53.71） | ep55（best@29） | 84.8017¹ | `6581b9cc3d475362` |
| ② 心理声学 seed1（c 固定） | `outputs_v4_p05/psycho_cfix_seed1/nnaec_g0.5_j10_psycho` | seed1 + `--psycho-c 135.168856` | ep63（best@20） | 156.9994¹ | `790b4e1c91e46c92` |
| 复刻校验（V1 路径重放） | `outputs_v4_p05/lossctrl_v4replica/nnaec_g0.5_j10` | v3 last.ckpt(ep186)→240ep, lr2e-4 | 跑到 240 | 8.4768（v4：8.5852） | `babc37b6de5ae129` |
| （诊断用）从零 240ep lr1e-3 中断版 | `outputs_v4_p05/lossctrl/nnaec_g0.5_j10` | 早期诊断跑（val≈36 平台） | 手动中断 ep48 | 35.8446 | `2d3ac9fa32bac93e` |

> 全量 ckpt 元数据（路径 + 完整 sha256 + epoch/val/损失口径/seed）：`outputs_v4_p05/aecmos/runs_index.json`
> （由 `exp/07_nn_aec/p05_index.py` 生成；`eval_p05_matrix.py` 的单次 index.json 会被后续调用覆盖，
> 故留痕以该表为准）。

> ¹ **psycho 列的 val 绝对值与①不可比**：总损失 = c·L_psy + SI-SNR，c 把谱项抬到另一个量级。
> 跨列比较一律用 AECMOS（§5），不用 val。

**为什么两列都在 ep60–71 早停**：ReduceLROnPlateau(patience=3, factor=0.5) 每次减半重置 patience，
平台期每 4–5 epoch 提升一点 ⇒ LR 在 20–30 个 epoch 内折半到 ~1e-5，随后 bad 累积到 25 触发早停。
①②各 seed 的早停点一致（ep60/63/71/55/63），说明这是**调度器的共同行为**，非某一列异常。

**复刻校验（钉死环境）**：按 v4 原始路径重放（v3 `last.ckpt`（ep186）→ 跑到 240ep，lr 2e-4，`--early-stop 240`，
与原命令一致，含"损失签名变化→best/bad 重置"这一步），best val **8.4768**（v4：**8.5852**），
AECMOS 与 v4 差 ≤0.04（eval_v5 DT echo 4.247 vs 4.248、deg 3.222 vs 3.220；FEST echo 3.912 vs 3.881）
⇒ **环境、数据、评测链路无漂移**，①/②与 v4 的差值可归因于模型/配方本身。

> **必须披露的负结果（决定了主对照的选择）**：v3 的 scheduler 在 ep186 已把 lr 压到 **1.53e-8**，
> v4 的"187–240ep"其实是在近乎冻结的 lr 下微调（复刻证明权重与 v3 差异仅噪声级）
> ⇒ **"在 v3 上续训"无法检验损失函数**（没有可学空间）。故损失对照采用**从零 240ep、两列对称**的协议，
> `lossctrl_v4replica` 只作环境复刻证据，不作损失对照。

## 5. AECMOS 对照（冻结集 eval_v3 60 条 / eval_v5 300 条）

数据源：`outputs_v4_p05/aecmos/<set>/aecmos_<variant>.json`（逐文件 rows）；
完整表：`reports/p0_5_loss_matrix_ci.md`、`outputs_v4_p05/aecmos/bootstrap_p05{,_pairs}.json`、
回归后的 `reports/DATA_CI.md`（新增 P0-5 配对节）。

### 5.1 点估计（eval_v5 DT，n=154）

| run | DT echo | DT deg |
|---|---|---|
| v4（存量锚） | **4.248** | 3.220 |
| ① 原损失 seed0 | 4.102 | 3.129 |
| ② 心理声学 seed0 | 3.964 | **3.279** |
| ① 原损失 seed1 | 3.770 | 3.265 |
| ② 心理声学 seed1（c 自标定 53.71） | 3.925 | 3.258 |
| ② 心理声学 seed1（c 固定 135.17） | 3.936 | 3.241 |
| 复刻校验 | 4.247 | 3.222 |
| （参考）传统 AEC @48k | 3.442 | **3.590** |

### 5.2 配对差值（bootstrap B=2000 / seed=777，同 (name, scenario) 对齐）

| 配对 | 场景 | 指标 | n | Δmean | 95%CI | 降/升 | 判定 |
|---|---|---|---|---|---|---|---|
| **②−① seed0** | DT | deg | 154 | **+0.151** | **[+0.086, +0.217]** | 51/103 | **显著回升** |
| ②−① seed0 | DT | echo | 154 | −0.139 | [−0.180, −0.094] | 107/47 | 显著退化 |
| **②−① seed1** | DT | deg | 154 | **−0.007** | **[−0.069, +0.052]** | 77/77 | **CI 含 0（不显著）** |
| ②−① seed1 | DT | echo | 154 | +0.155 | [+0.095, +0.210] | 44/110 | 显著**回升**（反号） |
| ②−v4 seed0 | DT | deg | 154 | +0.059 | [−0.009, +0.125] | 72/82 | **CI 含 0：不得宣称缺口收回** |
| ②−v4 seed0 | DT | echo | 154 | −0.285 | [−0.336, −0.236] | 136/18 | 显著退化 |
| ①−v4 seed0 | DT | deg | 154 | −0.092 | [−0.152, −0.031] | 95/59 | 显著退化（谱系/预算差） |
| ①−v4 seed0 | DT | echo | 154 | −0.146 | [−0.183, −0.108] | 115/39 | 显著退化 |
| 复刻−v4 | DT | deg | 154 | +0.001 | [−0.000, +0.003] | 63/91 | 环境复刻成立 |
| 复刻−v4 | DT | echo | 154 | −0.001 | [−0.002, −0.000] | 85/69 | 数值级（AECMOS 浮点） |

### 5.3 eval_v3（DT n=14，与 NUMBERS 主表同集）

配对（同口径）：②−① seed0 deg **+0.215 [+0.026, +0.383]**（显著）、echo −0.201 [−0.400, −0.029]；
②−v4 seed0 deg +0.067（CI 含 0）、echo −0.335（显著）。方向与 eval_v5 的 seed0 一致，
但小集（n=14）不足以单独支撑结论。

### 5.4 同列跨 seed 方差（决定"效应是否可复现"的标尺）

| 列 | DT deg（seed0 / seed1） | 极差 | DT echo（seed0 / seed1） | 极差 |
|---|---|---|---|---|
| ① 原损失 | 3.129 / 3.265 | **0.136** | 4.102 / 3.770 | **0.332** |
| ② 心理声学（c 自标定） | 3.279 / 3.258 | 0.021 | 3.964 / 3.925 | 0.038 |
| ② 心理声学（c 固定） | — / 3.241 | — | — / 3.936 | — |

> **读法**：②列的 seed 间极差看起来更小（0.021/0.038），但**这是"用自标定 c 补偿了 seed 差异"的假象**
> ——c 从 135.17 掉到 53.71（2.5×），把谱项权重整体改小了；固定 c 的 seed1 复测得 3.241（与 seed0 3.279 差 0.038）。
> 而判定"损失项是否真有 deg 增益"的标尺是**①列的跨 seed 波动**（deg 0.136），不是②列自己的波动——
> 因为"再平衡方向"本身随 seed 变。seed0 的 +0.151 与 ① 的 seed 波动同量级 ⇒ 不构成可靠增益。

### 5.5 自校验

v4 列由本轮评测脚本重算，与 `docs/NUMBERS.md` 现行权威值**逐位一致**
（eval_v3 4.175/3.237；eval_v5 4.248/3.220）⇒ 评测链路（@48k 重采样 + 打分会话 + 汇总）自校验通过。

## 6. 引用与留痕（V3.1 新增纪律）

| 项 | 值 |
|---|---|
| 文献 | Abreu, W.; Miranda, B. V.; Biscainho, L. W. P. "Efficient Audio Enhancement with a Differentiable Psychoacoustic Loss", *J. Audio Eng. Soc.* **74**(6), 430–444, 2026（JAES 2026-06 接收）；arXiv:2608.02918（eess.AS，v1 提交 2026-08-03） |
| **获取日期** | **2026-10-13**（`https://arxiv.org/abs/2608.02918`；摘要页 + `arxiv.org/pdf/2608.02918`） |
| 借用范围 | 仅"耳蜗压缩 + 临界带可听度加权"思想骨架（该文 PAQM 派生损失、无判别器、显存友好）；**未复用其代码/权重**；不做 PAQM 全套（无时域掩蔽模板、无绝对听阈），**不声称数值等价** |
| 实现落点 | `core/losses.py::psychoacoustic_loss` + 单测 `tests/test_losses_psycho.py`（9 例） |
| 数据集 | 本轮**未新增数据集**：沿用既有 `D:/data/aec-synth/{train_v1,eval_v3,eval_v5}`；逐 run ckpt sha256 记于 `outputs_v4_p05/aecmos/index.json` |

## 7. ③ flow matching 列：skipped（方案留档，理由）

**状态：skipped（未训练）**，依据计划 §2 的"若时间不够可标注 skipped"豁免条款。

- **不同构**：现模型是 cRM 掩码、谱进谱出的确定性回归器；flow matching 学的是"噪声→数据"的
  随机插值速度场。不换架构（计划明令不换模型）时只能把 mask 头当速度场：
  `m_t = (1−t)·ε + t·m*`，`L = E_t‖v_θ(·,t,x_t) − (m*−ε)‖²`；采样需 ODE 多步（推理 ×N），
  且要新增时间条件输入并复核流式/因果单测。
- **先验期望最低、代价最高**：它改的是生成式重建，不是感知加权；对"DT deg 缺口"没有直接机制
  （②列至少还有 §2 的机理路径——而本轮结果表明那条路径也不稳）。
- **留作阶段1候选**：最小可行实现与验收口径（与①/②同超参、同冻结集、同 B=2000/seed=777 CI）已在上文钉死。

## 8. 未闭环清单（下一棒）

1. **c 的网格化**（本列最大不确定性）：只做了单点 135.17 + 一处固定 c 复测；需补 c∈{1.56, 135, 479}
   三点（每点约 7 分钟训练 + 6 分钟评测）。自标定建议改"多 probe 批 + 固定种子"。
2. **多 seed 扩样**：当前①/②各 2 seed（+ ②固定 c 1 点）。要判定"再平衡方向是否可复现"，
   至少需要每列 ≥3 seed；本轮 2 seed 已足以**否证**"稳定 deg 增益"的强表述。
3. **与 v4 的差距不是损失能单独补上的**：②−v4 的 deg CI 含 0；且②的 echo 掉 0.285。
   若目标是"既保 echo 又收 deg"，需在损失外加 echo 约束项，或在数据侧（阶段1 隔离+扩样）解决。
4. **FEST echo 掉幅显著**（v4 3.881 → ① 2.690 / ② 2.348，配对 −1.19/−1.53 显著）：
   两列都是从零 ≤63ep 的产物，说明 FEST 段抑制能力主要靠长谱系/训练预算 ⇒ 阶段1 重训须把
   FEST 口径一并纳入验收（DP1 只看 DT deg 会漏掉这个退步）。
5. **训练预算不对称（口径声明）**：对照锚 v4 是 426ep lineage 的产物，①②两列从零训练且早停；
   ①②之间的比较对称，①②与 v4 的比较含"谱系/预算"差，已在 §1/§4 注明。

## 9. 目录与产物清单（`exp/07_nn_aec/outputs_v4_p05/`）

| 子目录 | 内容 |
|---|---|
| `lossctrl_240ep/`、`lossctrl_240ep_seed1/` | ①列（原损失）主 run × 2 seed：`best.ckpt`/`last.ckpt`/`history.json`/`tb/` + `../train_*.log` |
| `psycho_240ep/`、`psycho_240ep_seed1/`、`psycho_cfix_seed1/` | ②列（心理声学）主 run + seed1 复测 + 固定 c 复测 |
| `lossctrl_v4replica/` | 复刻校验（v3→240ep 重放 V1 路径） |
| `lossctrl/`、`pilot_constlr/`、`pilot_lr1e3/` | **诊断 run**（fresh-从零路线的平台化证据：lr2e-4/1e-3 均停在 val≈36–39；正因如此主对照改用"同调度两列对称"，并把复刻校验当作可比性证据） |
| `aecmos/<set>/aecmos_<variant>.json` | 评测逐文件 rows（14 个变体×2 集），`runs_index.json` 为全量 ckpt 元数据表 |
| `scale_probe.json` | 损失量级/梯度量级实测（c 的来源） |

## 10. 复现命令（完整）

```powershell
$PY='C:\Users\14805\.conda\envs\avsuite\python.exe'   # torch 2.6.0+cu124 / 4060 Laptop 8GB
# 0) 损失量级标定（结果固化进 ckpt cfg）
& $PY exp/07_nn_aec/loss_scale_probe.py
# 1) 两列训练（从零，240ep 上限；train_v1 不动）。--seed 换 1 得到第二组
$BASE = @("--data","D:/data/aec-synth/train_v1","--val","D:/data/aec-synth/eval_v3","--gamma","0.5",
          "--w-sisnr","1.0","--epochs","240","--batch-size","16","--lr","1e-3","--scenarios","dt,nst",
          "--early-stop","25","--device","cuda","--seed","0")
& $PY exp/07_nn_aec/train.py @BASE --loss weighted --out exp/07_nn_aec/outputs_v4_p05/lossctrl_240ep
& $PY exp/07_nn_aec/train.py @BASE --loss psycho   --out exp/07_nn_aec/outputs_v4_p05/psycho_240ep
& $PY exp/07_nn_aec/train.py @BASE --loss psycho --psycho-c 135.168856 --seed 1 --out exp/07_nn_aec/outputs_v4_p05/psycho_cfix_seed1
# 2) 复刻校验（V1 路径重放，证明环境未漂移）
& $PY exp/07_nn_aec/train.py @BASE --lr 2e-4 --early-stop 240 `
     --resume exp/07_nn_aec/outputs_v3/nnaec_g0.5_j10/last.ckpt `
     --out exp/07_nn_aec/outputs_v4_p05/lossctrl_v4replica
# 3) 评测（两套冻结集）→ 4) CI（对 v4 与 ②−① 两个方向）→ 5) 汇入全仓 DATA_CI
& $PY exp/07_nn_aec/eval_p05_matrix.py --ckpt v4=... --ckpt lossctrl_240ep=... --ckpt psycho_240ep=... `
     --sets eval_v3=D:/data/aec-synth/eval_v3 eval_v5=D:/data/aec-synth/eval_v5
& $PY exp/07_nn_aec/bootstrap_p05.py --variants lossctrl_240ep psycho_240ep lossctrl_240ep_seed1 psycho_240ep_seed1 psycho_cfix_seed1 lossctrl_v4replica
& $PY exp/07_nn_aec/bootstrap_p05_pairs.py
& $PY exp/07_nn_aec/p05_seed_variance.py           # 逐 seed 对称效应 + 跨 seed 方差
# 6) 复制 json 到 exp/07_nn_aec/outputs/p05/（tools/compute_ci.py 的寻址约定），再回归全仓 CI
& $PY tools/compute_ci.py
```
