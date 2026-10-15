# 09 · 级联 AEC(线性 MDF → NN 残差)实验区

一句话定位:级联 AEC(线性 MDF → NN 残差)实验区——传统线性级擦主回声,同一个 NN 换一对输入吃非线性残余;
对应整改任务 P1-1(仓根 `EXTENSION_PLAN.md`)与决策记录 `avsuite/docs/adr/0005-cascade-verified.md`。
完整三方对照表、DoD 判定与逐列重跑命令以 `results/THREE_WAY.md` 为准,本页只做导览。

## 原理(三行)

- `far → MDF(DTD 门控, partitions=8, mu=0.7)→ (y_echo, e_lin = mic − y_echo)`:线性级输出已 GCC-PHAT 对齐的回声估计 y_echo 与其线性残差 e_lin(RES 前的 MDF 误差输出);
- `(e_lin, y_echo) → NNAEC 0.638M 双输入 CRN`:把 (e_lin, y_echo) 当作该模型的 (mic, far) 输入对送进去;
- 「级联 = 同一个模型换一对输入」:架构/损失/训练代码一字不改(`exp/07_nn_aec` 的 train.py/dataset.py/model.py 只读复用),换数据不换模型。

## 目录导览

| 路径 | 内容 |
|---|---|
| `precompute_cascade.py` | 线性级预计算:每条 clip 跑 MDF,产三槽目录(mic/=e_lin、farend/=y_echo、nearend/=原目标),断点续跑,`--force` 原地覆盖;P1-3b 追加 `--fresh-instance-per-clip`(DM 同约定)与 `--measure N`(只测 ms/clip) |
| `cascade_dm.py` | P1-3b 级联 DM 数据集:`AecCascadeDynamicMix(AecDynamicMix)` = 合成 → MDF → (e_lin, y_echo, near) |
| `train.py` | P1-3b 训练入口(薄壳):补 `--cascade-linear` 后转交 `exp/07_nn_aec/train.py`(训练循环不复制) |
| `run_cascade.py` | 级联推理:eval_v3 逐条 MDF→(e_lin,y_echo)→NN(默认 best.ckpt),前 3 条 lin-check 与预计算逐位核对 |
| `zerotrain_baseline.py` | 零训练基线:v4 权重原封不动直接吃 (e_lin, y_echo),结果见 `results/ZEROTRAIN.md` |
| `probe_env.py` / `probe_clip.py` / `probe_compare.py` | 探针:环境与 wav 子类型、削波影响面、逐条 DT deg 对比 |
| `outputs/` | 40ep 主训:`train.log` + `nnaec_g0.5_j10/{best,last}.ckpt`(best@epoch 39,val 23.99855) |
| `outputs_long/` | 250ep 收敛口径:epoch 64 早停,best@39 val 23.57189(DT deg 仅 +0.06 → 「没训够」已排除) |
| `results/` | `THREE_WAY.md`(三方对照)、`ZEROTRAIN.md`(零训练基线)、`aecmos_*.json` 与 `zerotrain_aecmos.json`(逐条打分)、`enh/`(级联增强 wav 60 条 FLOAT;另有 `enh_long/`、`zerotrain_enh/` 对照) |

## 数据

- `D:/data/aec-synth/train_cascade_lin/` — 训练 3000 条全量(源 train_v1 经 MDF 预计算;首版随源 PCM_16 落盘有削波,已改 FLOAT 重写);
- `D:/data/aec-synth/eval_v3_lin/` — 验证 60 条(源 eval_v3 同法预计算,FLOAT)。

## 结果速览(AECMOS 官方 ONNX 同集同口径,**AECMOS@48k 模型协议**,eval_v3 60 条全段,DT 口径)

| DT | 传统 WP-06 | 全 NN v4 | 零训练级联 | 级联 P1-1(40ep) |
|---|---|---|---|---|
| echo_mos | 3.494 | 4.175 | 4.062±0.160 | **4.261** |
| deg_mos | **3.598** | 3.237 | 3.223 | 3.250 |

- DoD(echo≥3.5 且 deg≥2.2,DT 口径):echo **4.261 ✅**(三方最高);deg **3.250 ✅** —— 按原门槛字面判定达成,
  ⚠️ 但须注意 **deg 绝对尺度在新口径下整体上移 ≈+1.3**(旧口径 1.785→新 3.250),门槛 2.2 是旧口径标定的,
  **可靠的是相对结论**:级联 deg ≈ 纯 NN(3.250 vs 3.237)且均低于传统 3.598(差距 −0.35±0.16);
  是否重标定 DoD 门槛请以 `docs/NUMBERS.md` 为准(本 README 只报读数)。
- 结论:deg 病理在掩码不在接口——全 NN 3.237 / 零训练 3.223 / 训练级联 3.250 三条 NN 路同档(传统的 3.598 靠 DTD 冻结保近端),250ep 收敛口径同结论;下一步近端门控。

> <sub>**口径更替(2026-10-11)**:上表为 @48k 模型协议重评值;旧值(echo 2.467/3.754/3.174/3.919、
> deg 2.401/1.743/1.887/1.785)为 **16k 直送错配口径,已作废**,逐集对照见
> `reports/aecmos_rescore_v5fix.md` 与 `exp/07_nn_aec/outputs/aecmos48_v5fix/_t22_backfill.md`。</sub>

## 结果速览·稳健口径(eval_v5,DT n=154,95%CI,**AECMOS@48k 模型协议**)——V4 整改补

| DT | 传统 WP-06 | 全 NN v4 | 级联 P1-1 |
|---|---|---|---|
| echo_mos | 3.442±0.10 | **4.248±0.05** | 4.317±0.04 |
| deg_mos | **3.590±0.09** | 3.220±0.09 | 3.375±0.09 |

- **修正(ADR 0005 附记)**:eval_v3 n=14 中"级联 echo 三方最高(+0.17)"系小样本波动——扩样后
  级联−NN Δecho **+0.069±0.043**(效应量仅 NN 优势的 6%,v7 复验 +0.068±0.031);FEST echo 级联
  **4.196±0.103** 仍三方最高(v7:4.157 > NN 3.595 ≈ 传统 3.543,级联−NN +0.562±0.128);deg 三条 NN 路
  同档在 11 倍样本上成立(**3.220/3.375 vs 传统 3.590**,级联−NN +0.154±0.079 仅略优,不推广为"级联治 deg")。
  上表与本节旧值(2.525/3.721/3.741 与 2.430/1.811/1.851)按新口径替换,旧值为 16k 直送口径已作废。
- **deg 机理测量(V4,reports/deg_mechanism.md)**:NN 近端短板=高频带选择性损伤(4k-8k 中位
  −2.41dB)+谱平坦化(SFM −1.46dB),非宽带能量压低(NLA 中位仅 −0.22dB)——近端门控靶点=高频保持。

## P1-3b：级联 DM（训练时现合成 + 线性级入循环，2026-10-13）

**修掉的缺口**：本目录此前**没有训练入口**（三方对照的级联训练是把 `train_v1` 经
`precompute_cascade.py` 物化成静态目录 `train_cascade_lin/`，再用 `exp/07_nn_aec/train.py`
去吃），因此附录B 的 `python exp/09_cascade_aec/train.py --dynamic-mixing ...` 会失败；
DM 化后训练数据不再落盘，级联必须有"合成后接线性级"的能力。

**两个入口（薄壳，训练循环只有一份）**：

```bash
# ① 09 的薄壳入口（推荐；自动补 --cascade-linear 并转交 07 的 main）
python exp/09_cascade_aec/train.py --device cuda --dynamic-mixing --epochs 240 \
    --batch-size 16 --num-workers 8 --probe-mem 20 \
    --dm-scenarios 0,0.5086,0.4914 \
    --near-dir <pool>/near --far-dir <pool>/far --noise-dir <pool>/noise \
    --near-list split_lists/speech_near_train70.txt \
    --far-list  split_lists/speech_far_train70.txt \
    --noise-list split_lists/noise_train70.txt \
    --rir-dir <rir-pool> \
    --val <eval集>_lin --out exp/09_cascade_aec/outputs_dm
# ② 等价写法（直接走 07 + 开关）
python exp/07_nn_aec/train.py --cascade-linear --dynamic-mixing ...（其余同上）

# val 侧：必须与训练**同 per-clip 约定**预计算（否则 MDF 约定不一致）
python exp/09_cascade_aec/precompute_cascade.py --synth-dir <eval集> \
    --out-dir <eval集>_lin --fresh-instance-per-clip      # partitions=8 / mu=0.7 为缺省
```

**实现**：`cascade_dm.AecCascadeDynamicMix(AecDynamicMix)` —— `__getitem__` 里 DM 合成
(mic, far, near) 后跑 `TradAEC(method="mdf", partitions=8, mu=0.7)`，返回
`(mic=e_lin, far=y_echo, near)`；far 的相对平移（jitter 10ms）**施加在 y_echo 上**
（与静态集一致：静态是先跑 MDF、再由 `AecDataset` 对 `farend/` 做 jitter）。
模型/损失/训练循环一字不改（"级联 = 同一个 NN 换一对输入"）。

**口径与边界（必须同页写）**：

- 全部线性级参数 + 实例策略写进训练 manifest 的 `data.linear_stage`（并进 ckpt cfg）：
  `method/partitions/mu/block/use_dtd/use_res/align_margin_ms/reestimate_sec/... +
  trad_instance_policy="fresh_per_clip" + equivalent_to_static_precompute=False`。
- **每条 clip 新建 TradAEC 实例**（队长 2026-10-13 裁定）：DM 可复现、worker 可并行；
  历史静态集是**跨 clip 复用同一实例**（MDF 谱队列/对齐锁存/DTD EMA 跨 clip 延续）
  ⇒ **历史级联读数不与新 DM 读数直接混比**，收档须注明。
- val 侧 `precompute_cascade.py` 缺省仍是历史口径（跨 clip 复用，逐位兼容既有产物）；
  带 `--fresh-instance-per-clip` 才与 DM 训练同约定，输出目录落
  `cascade_lin_profile.json` 记录策略与参数。
- 训练配比沿用 v4 实际配方：`--dm-scenarios 0,0.5086,0.4914`（顺序 = **FEST,NST,DT**；
  即 nst 913/1795 = 0.5086、dt 882/1795 = 0.4914，无 FEST）；manifest 记具名字典 +
  `scenarios_explicit=true`。
- 输出目录带 `_dm_casc` 后缀（`nnaec_g0.5_j10_dm_casc`），不与纯 DM / 静态级联的 run 混合。

**单条线性级耗时实测（供 D1 吞吐按"合成+MDF"折算；t11/train-eng 用）**

| 项 | 口径 | 实测（本机 CPU） |
|---|---|---|
| 线性级 MDF | `TradAEC(mdf, partitions=8, mu=0.7)`，**每 clip 新建实例**，8s clip | mean **165.9 ms/clip**（p50 165.9 / p95 174.6）→ 6.0 clip/s/worker |
| 线性级 MDF | 同上，4s clip（训练片段长度） | mean **83.2 ms/clip** → 12.0 clip/s/worker |
| DM 合成 | 真实池 993/599/95 + `rir_pool`，4s 片段，dir-RIR | ≈ **5.4 ms/clip**（184 样本/s/worker） |

→ 4s 片段下单条总成本 ≈ **88.6 ms**（MDF 占 ~94%）：**线性级是级联 DM 的 CPU 瓶颈，
不是合成**。复跑命令：

```bash
python exp/09_cascade_aec/precompute_cascade.py --synth-dir <集> --measure 20   # 只测耗时、不落盘
python scripts/dm_aec_synth.py --num 30 --dry-run --seg-sec 4 ...               # 合成侧速率
```

## 重跑命令

命令抄自 `results/THREE_WAY.md` 与 `reports/rectification/exec/P1-1.md`,参数未动
(预计算条 P1-1.md 原文以 `python` 简写,此处补全为 avsuite 环境解释器全路径):

```powershell
# 预计算(P1-1.md §2;eval_v3 同法:--synth-dir D:/data/aec-synth/eval_v3 --out-dir D:/data/aec-synth/eval_v3_lin)
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite\exp\09_cascade_aec\precompute_cascade.py --synth-dir D:/data/aec-synth/train_v1 --out-dir D:/data/aec-synth/train_cascade_lin --force

# 训练 40ep(P1-1.md §3;250ep 收敛口径:--epochs 250 --early-stop 25 --out avsuite/exp/09_cascade_aec/outputs_long)
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite\exp\07_nn_aec\train.py `
  --data D:/data/aec-synth/train_cascade_lin --val D:/data/aec-synth/eval_v3_lin `
  --out avsuite/exp/09_cascade_aec/outputs `
  --gamma 0.5 --w-sisnr 1.0 --scenarios dt,nst --jitter-ms 10.0 --val-jitter-ms 0.0 `
  --batch-size 16 --lr 1e-3 --epochs 40 --early-stop 8 --num-workers 4 --device cuda --seed 0

# 级联推理 + AECMOS(THREE_WAY.md「级联(P1-1,两步)」)
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/exp/09_cascade_aec/run_cascade.py    # eval_v3 逐条 MDF→(e_lin,y_echo)→NN(best.ckpt)
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3 --enh-dir avsuite/exp/09_cascade_aec/results/enh --out avsuite/exp/09_cascade_aec/results/aecmos_cascade.json

# 零训练基线(ZEROTRAIN.md)
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/exp/09_cascade_aec/zerotrain_baseline.py --eval-dir D:/data/aec-synth/eval_v3
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3 --enh-dir avsuite/exp/09_cascade_aec/results/zerotrain_enh --out avsuite/exp/09_cascade_aec/results/zerotrain_aecmos.json
```
