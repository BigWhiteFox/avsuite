# deg_mechanism — DT 段近端衰减/回声抑制/分频带/伪影代理三方物理测量（eval_v5, n=300）

- 日期：2026-10-11 · 执行：语音信号分析（接手 exp/07_nn_aec/deg_mechanism.py 未竟工作）
- 目的：用合成集 ground truth（干净 nearend 已知）直接测量三方（传统/NN v4/级联）在 DT 段的近端衰减与回声抑制，替代此前仅靠 AECMOS deg_mos 的间接推理。
- 约束遵守：未做任何 git 操作；未训练模型；未改任何既有文件（仅改前任新写的 `exp/07_nn_aec/deg_mechanism.py`，见 §6）；未触碰 `D:/data/aec-synth/eval_ood_delay`。
- 数据：`D:/data/aec-synth/eval_v5`（300 条；DT 154 / NST 76 / FEST 70）。FEST 无近端激活，不测量。

## 1. 口径与 enh 来源（实地核查）

- 帧长 20ms/320 样本不重叠；近端激活 VAD：对**干净 nearend** 求帧能量，阈值=帧能量峰值 −40dB。
- `nla_db`（DT）= 10·log10(Σ|enh|²/Σ|nearend|²)，仅近端激活帧且 t≥lead-in（负=近端被压）；分频带 NLA 同帧集、逐帧 Hann 加窗 rFFT(320) 后 <500 / 500-2k / 2k-4k / 4k-8k Hz 四带能量比。
- `leadin_er_db` = 10·log10(Σ|mic|²/Σ|enh|²)（t<leadin_sec=2s 纯远端段，正=压得多）。
- NST 片段：近端激活帧 NLA + 谱平坦度比 10·log10(SFM_enh/SFM_near)（SFM=功率谱几何/算术平均，逐帧求、激活帧均值；0=谱形不变）。
- 三方 enh 位置核查结果：
  - **级联**：`exp/07_nn_aec/outputs/eval_v5_dt300/enh_cascade/`（v5 评估原盘，300 条，dt+nst 230 条全覆盖）。
  - **传统 / NN**：v5 评估时 `aecmos_eval.py` 内存推理、enh wav 未落盘（已核 `outputs/eval_v5_dt300/` 仅有 aecmos/metrics 存档，与 reports/eval_v5_dt300.md §5 产物清单一）→ 按仓库自身确定性管线**重建**：trad = `TradAEC(method="mdf", partitions=8, mu=0.7)`（与 aecmos_eval.py `--trad`、eval_aec.py 默认完全同参，use_res=True）；nn = `NNAEC` + `outputs_v4/nnaec_g0.5_j10/best.ckpt`（v5 打分同款 v4 权重，自动 CUDA——与 v5 实际执行同设备）。
  - **重建保真自检**：3 固定样本 × 2 方法，PESQ(nearend, enh_regen) 对 v5 存档逐条核对——**6/6 全部 Δ=0.0000（trad 1.4422/2.0257/1.7897，nn 1.3862/1.4773/1.2794），PASS**。重建与 v5 打分信号逐位一致，非降级替代。

## 2. 主表（DT n=154，median [p25, p75]）

| 指标 | 传统 AEC (WP-06) | NN-AEC v4 | 级联 P1-1 (MDF→NN) |
|---|---|---|---|
| lead-in 回声抑制 dB | 10.43 [7.34, 13.27] | **51.85** [33.03, 67.40] | 29.30 [25.06, 32.89] |
| DT 近端衰减 NLA dB | **+0.07** [+0.03, +0.17] | −0.22 [−0.42, −0.04] | −0.05 [−0.13, +0.00] |
| NLA <500 Hz dB | +0.07 [+0.02, +0.21] | −0.21 [−0.46, −0.02] | −0.07 [−0.21, +0.02] |
| NLA 500-2k Hz dB | +0.05 [+0.01, +0.13] | −0.17 [−0.43, −0.02] | −0.01 [−0.08, +0.03] |
| NLA 2k-4k Hz dB | +0.05 [+0.01, +0.23] | −0.64 [−0.84, −0.36] | −0.10 [−0.23, −0.03] |
| NLA 4k-8k Hz dB | +0.13 [+0.03, +0.50] | **−2.41** [−3.50, −1.91] | −0.15 [−0.34, −0.05] |

NST（n=76，纯透传段伪影代理）：

| 指标 | 传统 | NN v4 | 级联 |
|---|---|---|---|
| NST NLA dB | +0.02 [+0.01, +0.03] | −0.11 [−0.17, −0.06] | −0.03 [−0.04, −0.01] |
| NST SFM 比 dB | +0.81 [+0.24, +2.25] | **−1.46** [−2.14, −0.58] | −0.50 [−0.97, −0.12] |

分布尾（DT，逐条）：NLA≤−1dB 占比 trad 2.0% / nn 9.7% / cascade 0%；NLA≤−3dB 仅 nn 出现（2.0%，最深 −6.79dB）。

逐条相关（rows.csv × aecmos rows，analysis.json）：DT 池化(n=462) corr(deg_mos, 宽带NLA)=0.245 < corr(deg_mos, 4k-8k NLA)=**0.366**；NST(n=228) corr(deg_mos, SFM 比)=**0.375**；NN 方法内 corr(deg, NLA)=0.274（trad/cascade 内 ≈0，因二者逐条几乎无变异）。

## 3. 问题 a)：NN deg_mos 落后（1.81 vs 传统 2.43）主要对应"近端能量被过度压低"还是"谱结构伪影"？

**测量级答案：主要是谱结构损伤（高频带选择性过压 + 谱平坦化），不是宽带能量被过度压低。**

- 宽带能量面：NN DT NLA 中位仅 −0.22 dB、p75 −0.04 dB，90.3% 条目 NLA > −1dB——近端总能量几乎未少，"整体压低"量级（≤0.3 dB）不足以解释 0.62 MOS 差。
- 谱形面（三条独立证据同向）：
  1. 分频带：NN 衰减随频率单调加深（<500：−0.21 → 4k-8k：**−2.41 dB** 中位、p25 −3.50）——擦音/清音能量被系统性削掉，属谱结构损伤而非整体衰减；
  2. NST 纯透传段（无远端、不涉及双讲抑制）NN 仍把谱压平（SFM 比 −1.46 dB vs 传统 +0.81 dB），即伪影在无回声语境下也持续存在；
  3. 三方 deg 排序 2.43 > 1.85 ≈ 1.81 与谱形损伤轴严格单调（SFM 比 +0.81 > −0.50 > −1.46；4k-8k +0.13 > −0.15 > −2.41），而宽带 NLA 轴上 cascade(−0.05) 与 nn(−0.22) 几乎并列、deg 却差在别处——级联近端能量几乎零损伤仍 deg 1.85，说明能量面不是主因。
- 逐条相关同样指向谱轴：deg~4k-8k NLA r=0.366、deg~SFM 比 r=0.375，均高于 deg~宽带 NLA r=0.245。

## 4. 问题 b)：传统的 deg 优势是否来自冻结透传（NLA≈0dB）？

**测量级答案：是，与冻结透传一致。** 传统在 DT 近端激活帧 NLA 中位 **+0.066 dB**（p25 +0.025 / p75 +0.165），四频带中位 +0.05~+0.13 dB，NST 段 +0.017 dB——近端激活时输出与干净参考能量差 ≤0.2 dB 量级，即 DTD 冻结把近端近似原样透传（NLA≈0dB 成立，且全频带成立、无高频暗伤）。其 deg 优势（2.43）与该零损伤透传同现；代价是 lead-in 回声抑制中位仅 10.4 dB（NN 51.9 dB）。附注：传统 NST SFM 比 +0.81 dB（MDF 失配噪声使谱略平）不构成 deg 惩罚（NST deg 2.645 仍最高），进一步说明 AECMOS deg 惩罚的是"压平/高频损失"方向的形变而非任何微小偏离。

## 5. 随机抽查手工复算（seed=20261011，5 条 DT × 3 方法）

独立实现（逐帧显式循环重算 VAD 与 NLA，不复用 deg_mechanism.py 函数），对照 rows.csv：

| 条目 | 方法 | 激活帧 | NLA 复算 dB | rows.csv dB | \|Δ\| |
|---|---|---|---|---|---|
| fileid_132_dt | trad / nn / cascade | 300 | −0.037 / +0.350 / −0.045 | −0.037 / +0.350 / −0.045 | 0.0000×3 |
| fileid_163_dt | trad / nn / cascade | 299 | +0.023 / −0.128 / −0.099 | 同左 | 0.0000×3 |
| fileid_166_dt | trad / nn / cascade | 300 | −0.039 / **−1.541** / −0.143 | 同左 | 0.0000×3 |
| fileid_242_dt | trad / nn / cascade | 295 | +0.029 / −0.032 / −0.053 | 同左 | 0.0000×3 |
| fileid_243_dt | trad / nn / cascade | 252 | +0.030 / +0.010 / −0.039 | 同左 | 0.0000×3 |

15/15 全对，max|ΔNLA|=0.0000 → PASS（明细 `outputs/deg_mechanism/recheck_nla.json`）。抽查亦直观示出典型形态：NN 多数条目近端能量接近 0dB（fileid_163/242/243），但个别条（fileid_166）达 −1.54 dB；trad/cascade 全部 |NLA|<0.15 dB。

## 6. 脚本相对前任版本的改动（exp/07_nn_aec/deg_mechanism.py）

前任脚本主体（指标定义、regen 配置、测量流程）审查为**正确**，与任务规格和 v5 推理配置（trad: mdf/8/0.7 含 use_res=True、nn: v4 best.ckpt）一致，未改算法。仅做 4 处健棒性修正（防"长命令中途死亡"场景）：

1. regen 续跑跳过条件由"文件存在且 >1024B"改为"存在且 `sf.info` 帧数==mic 帧数"（`_wav_ok`）——防上次中断留下的半截 wav 被误当作完成（本次 116 条存量 trad wav 全部通过帧数校验，未重写）；
2. trad regen 输出长度防御：`out[:len(mic)]` 改为截断/补零到恰好 len(mic)；
3. `validate()` 对缺失 enh wav 由崩溃改为 SKIP 记录（`--validate-only` 现可在 nn 未重建时单测 trad）；
4. `measure()` 逐行 try/except（错误记入 summary `measure_errors` 而非整跑报废）；末尾 `validate()` 调用改为 try/except 并去掉原来的magic 阈值条件。

新增（均为新文件，未动既有文件）：`outputs/deg_mechanism/recheck_nla.py|json`（§5 复算）、`analysis.py|json`（§2 相关性）、`regen.log|measure.log`。

## 7. 复现命令（workdir=仓库根 avsuite/）

```powershell
$env:PYTHONNOUSERSITE='1'
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' exp/07_nn_aec/deg_mechanism.py --stage regen     # 续跑式：trad 剩余114条21s + nn 230条2s(cuda) + 6样本PESQ自检
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' exp/07_nn_aec/deg_mechanism.py --stage measure   # 690行 → rows.csv + summary.json（内嵌自检）
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' exp/07_nn_aec/outputs/deg_mechanism/recheck_nla.py  # 随机5条手工复算
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' exp/07_nn_aec/outputs/deg_mechanism/analysis.py    # deg_mos 相关性
```

## 8. 产物清单与偏离记录

- 逐条测量：`exp/07_nn_aec/outputs/deg_mechanism/rows.csv`（690 行=3 方法×(DT154+NST76)）
- 分位汇总：`exp/07_nn_aec/outputs/deg_mechanism/summary.json`（含重建自检 PASS、aecmos 引用、missing=[]、measure_errors=[]）
- 重建 enh wav：`exp/07_nn_aec/outputs/deg_mechanism/enh_wavs/{trad,nn}/`（各 230 条，FLOAT）
- 复算/分析：同目录 `recheck_nla.{py,json}`、`analysis.{py,json}`、`regen.log`、`measure.log`
- 偏离/如实记录：① trad/nn enh v5 未落盘 → 确定性重建 + PESQ 逐位自检 Δ=0.0000（三方齐全，未降级为两方）；② NN 重建沿用 v5 实际设备 CUDA（本机有卡；`--cpu` 可切，预期千分位差）；③ FEST 未测（任务口径外）；④ 本报告为新增文件 reports/deg_mechanism.md。
