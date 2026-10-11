# drift（时延漂移）重训闭环 — P2-1 遗留项（WS-E）

执行时间：2026-10-11 · 上游：`reports/drift_sweep.md`（其中"train_v1 加漂移重训，对照本表再评"即本任务）
结论先读：**漂移重训不支持"重训收复"假设**——同配方（v4 超参）+ σ=5ms/s 漂移训练集
重训后，NN 在漂移集上的 echo 反而**大幅低于**固定时延训练的 v4（DT echo 3.802 → 2.352，
FEST echo 4.353 → 1.880），且 240ep 训练 val 在 ~ep50 后彻底平台化。漂移下 NN 的稳健
更可能是**结构性**（逐帧复谱映射），而非可通过同分布重训复现的**配方性**收益；带漂移
数据训练反而损伤回声抑制能力。

## σ 取值

- 漂移集 `D:/data/aec-synth/eval_v3_drift` manifest `delay_drift_ms` 列唯一值 **5.0**（ms/s
  随机游走、限幅 ±10ms = ±2σ，节点线性过渡），与 `drift_sweep.md` 记载一致。
- 训练集重训使用**同一 σ=5.0**。

## 数据：train_v1_drift（新建，未覆盖任何既有数据集）

- 命令 = README `train_v1` 复现命令逐参数同配方（3000 条 / dur 4s / rir-dir rir_pool /
  rir-max-ms 100 / snr 15,30 / scenarios 0.4,0.3,0.3 / soft-clip 0.15 / dt-leadin 1.5s /
  seed 21 / 同 near-far-noise 目录）+ `--delay-drift 5.0`。
- 场景分布 FEST 1205 / DT 882 / NST 913 = train_v1 逐项一致；**manifest 底座逐字段
  0/3000 失配**（漂移游走用独立 Generator，不占主 rng，同 seed 开/关漂移底座可对照）。

## 训练（outputs_v4_drift*，新目录，未覆盖 outputs_v4）

- v4 同超参：γ0.5 / w_sisnr 1.0 / batch 16 / lr 2e-4 / jitter 10ms / scenarios dt,nst /
  240ep（读 v4 ckpt cfg 确认，含其 lr 2e-4 与 dt,nst 过滤）。val = eval_v3_drift。
- **两条 run**（因第一条提前早停，为公平补跑满 240ep）：
  1. `outputs_v4_drift/nnaec_g0.5_j10`：early-stop 25（同 v4），best val **58.364 @ep46**，
     ep71 早停，耗时 ~9 min（RTX 4060 Laptop，~6-7s/ep）。
  2. `outputs_v4_drift_full/nnaec_g0.5_j10`：禁早停跑满 240ep，best val **59.436 @ep34**，
     耗时 ~26 min；**ep~50 起 val 死平台（59.7-59.9 抖动，200+ epoch 无改善）**，
     ReduceLROnPlateau 已把 lr 压到下限仍不动——按止损条款如实记录：这是一次真收敛
     平台，不是"训练量不够"。
- 参考：v4（固定时延训练，~426ep 累计 lineage）在 eval_v3 上 val 8.585。val 绝对值
  不可跨集比较（漂移集损失尺度不同），仅列出备查。
- 两条 run 的 AECMOS 结论一致（full 略更差），取 **full 240ep** 为主结果（v4 同规格），
  early-stop run 附注。

## 三列 drift 对照（eval_v3_drift，DT 场景为主；±CI = 1.96·sd/√n）

| 系统 | DT echo (n=18) | DT deg (n=18) | FEST echo (n=36) | NST echo (n=6) | ovrl echo / deg (60) |
|---|---|---|---|---|---|
| 传统 AEC（WP-06 线性） | **1.649 ± 0.157** | **3.098 ± 0.219** | 2.119 ± 0.366 | 5.000 | 2.266 / 4.200 |
| NN v4（固定时延训练） | **3.802 ± 0.384** | 1.809 ± 0.117 | **4.353 ± 0.195** | 5.000 | 4.253 / 3.768 |
| NN drift 重训（240ep） | 2.352 ± 0.204 | 1.769 ± 0.080 | 1.880 ± 0.189 | 5.000 | 2.334 / 3.764 |
| NN drift 重训（ep71 早停 run，附注） | 2.571 ± 0.279 | 1.807 ± 0.105 | 2.114 ± 0.245 | 5.000 | 2.540 / 3.780 |

（传统 / v4 两列自既有 `outputs/aecmos_trad_drift.json`、`aecmos_nn_drift.json` 汇入，
只读未重跑；重训两格为本 batch 新测：`outputs/eval_drift_retrain/aecmos_nn_drift_retrain_full.json`
与 `..._retrain.json`。FEST deg 三方均 5.000（近端=干净远端），NST 仅 6 条统计薄。）

补充（eval.py，eval_v3_drift，重训 240ep ckpt）：FEST ERLE 仅 **3.92 dB**（ep71 run
4.64 dB；对比 v4 在固定集的 69.83 dB 口径产物）、DT PESQ 1.440（mic 1.406）、DT STOI
0.728——重训模型对远端参考的利用明显弱化。

## 关键问题回答

1. **固定时延训练的 v4 在 drift 下退化多少（vs 其 driftbase）？**
   几乎不退化（引 `drift_sweep.md` 配对表）：DT echo 3.993 → 3.802（**−0.191 ± 0.294**，
   AECMOS 噪声级），FEST echo 反 +0.244，DT deg +0.018。v4 的 drift 优势保持。
2. **drift 重训收复多少？**
   **不收复，反向**：重训（240ep）DT echo 2.352 vs v4 3.802（**−1.45**，远超各自 CI），
   FEST echo 1.880 vs 4.353（**−2.47**）；相对传统线性 AEC 的 DT echo 优势从 +2.153 缩到
   +0.703。带时变时延的 far 参考使"对齐-滤波"映射的学习目标内在歧义，模型学到的回声
   抑制反而更弱（ERLE 3.9 dB、val 平台佐证）。早停 run 同方向（DT 2.571），结论稳健。
   与"漂移重训能进一步收复"的预设相反——`drift_sweep.md` 局限 1 的"结构性 vs 配方性"
   之问就此有了答案：**NN 的漂移稳健是结构性的，且不可用同分布漂移数据再训放大**。
3. **线性 AEC 是否天然稳态？**
   否——线性 AEC 在 drift 下系统性崩塌（FEST echo 4.011 → 2.119，−1.89，35/36 条下降；
   DT 2.451 → 1.649，−0.80，17/18 条下降；本表 DT 1.649 为三列最低）。其"估计一次时延 →
   固定对齐 → 自适应"结构被条内时变时延打破。三方排序在 drift-DT echo 上变为
   **v4 (3.80) > 重训 (2.35) > 线性 (1.65)**。

## 训练曲线要点与复现

- 曲线：前 ~30ep 快速下降（67.4 → 59.4），ep34 best 后单调平台化；早停 25 于 ep71 触发；
  禁早停的 240ep run 证实 ep50 后无任何趋势性改善 → **early-stop 判定为真平台**。
- 配方/复现（数据 ~4 min，训练 ~26 min，评测 ~2 min）：

```powershell
$py = 'C:\Users\14805\.conda\envs\avsuite\python.exe'
# ① 训练集（train_v1 同配方同 seed + drift）
& $py scripts/aec_synth.py --num 3000 --dur-sec 4 --out-dir D:/data/aec-synth/train_v1_drift `
  --rir-dir D:/data/aec-synth/rir_pool --rir-max-ms 100 --snr-range 15,30 `
  --scenarios 0.4,0.3,0.3 --soft-clip-prob 0.15 --dt-leadin-sec 1.5 --seed 21 `
  --delay-drift 5.0 `
  --near-dir D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --far-dir  D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean `
  --noise-dir D:/data/dns-challenge-mirror/datasets/noise
# ② 重训（v4 同超参，val=eval_v3_drift）
& $py exp/07_nn_aec/train.py --data D:/data/aec-synth/train_v1_drift `
  --val D:/data/aec-synth/eval_v3_drift --out exp/07_nn_aec/outputs_v4_drift_full `
  --gamma 0.5 --w-sisnr 1.0 --epochs 240 --batch-size 16 --lr 2e-4 `
  --scenarios dt,nst --early-stop 240 --device cuda
# ③ 评测
& $py exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3_drift `
  --nn-ckpt exp/07_nn_aec/outputs_v4_drift_full/nnaec_g0.5_j10/best.ckpt `
  --out exp/07_nn_aec/outputs/eval_drift_retrain/aecmos_nn_drift_retrain_full.json
```

## 局限

- 未改训练侧对 far 的时变对齐机制（如逐帧时延估计前端）；当前仅验证"数据层加漂移"，
  结论限于该配方。v4 有 ~426ep resume lineage 而重训为 fresh 240ep，但 full-run 200+ ep
  平台表明追加 epoch 不改变结论。
- 单一 σ=5、单 seed（与上游一致）；NST 仅 6 条。

## 一句话结论

**σ=5ms/s 漂移重训不仅没有收复、反而显著削弱 NN 的回声抑制（DT echo 3.80→2.35），
证明 v4 的漂移稳健来自逐帧复谱映射的结构而非训练分布，"漂移重训"路线就此证伪关账。**

## 附记（V4 · 时延 OOD 探针对本文机制解释的修正，2026-10-11 晚）

本文"结构性（逐帧复谱映射）"的机制宣称已被 `reports/delay_ood_probe.md` 修正：
+110ms 出窗时延下 NN DT echo 3.721→1.919（配对 Δ−1.802±0.125，154/154 全降）——
v4 的漂移稳健实为**分布覆盖性**（漂移集 ±10ms 游走落在训练失配窗 0–90ms 内），
不是"不依赖对齐的结构性质"。本文**读数仍全部成立**（重训不收复、反而倒退；固定时延
训练的 v4 在漂移下几乎不退），但结论句应改为："同分布漂移重训不收复，且出窗即崩——
覆盖性稳健的天花板就是训练失配窗，生产必须保留前置粗对齐"。本文对重训倒退的机制
解释（"对齐-滤波目标内在歧义"）未排除替代解释（帧级标签噪声/分数时延插值副产物），
以覆盖性视角重审。详见 `docs/NUMBERS.md` 时延 OOD 探针小节。
