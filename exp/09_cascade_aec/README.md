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
| `precompute_cascade.py` | 线性级预计算:每条 clip 跑 MDF,产三槽目录(mic/=e_lin、farend/=y_echo、nearend/=原目标),断点续跑,`--force` 原地覆盖 |
| `run_cascade.py` | 级联推理:eval_v3 逐条 MDF→(e_lin,y_echo)→NN(默认 best.ckpt),前 3 条 lin-check 与预计算逐位核对 |
| `zerotrain_baseline.py` | 零训练基线:v4 权重原封不动直接吃 (e_lin, y_echo),结果见 `results/ZEROTRAIN.md` |
| `probe_env.py` / `probe_clip.py` / `probe_compare.py` | 探针:环境与 wav 子类型、削波影响面、逐条 DT deg 对比 |
| `outputs/` | 40ep 主训:`train.log` + `nnaec_g0.5_j10/{best,last}.ckpt`(best@epoch 39,val 23.99855) |
| `outputs_long/` | 250ep 收敛口径:epoch 64 早停,best@39 val 23.57189(DT deg 仅 +0.06 → 「没训够」已排除) |
| `results/` | `THREE_WAY.md`(三方对照)、`ZEROTRAIN.md`(零训练基线)、`aecmos_*.json` 与 `zerotrain_aecmos.json`(逐条打分)、`enh/`(级联增强 wav 60 条 FLOAT;另有 `enh_long/`、`zerotrain_enh/` 对照) |

## 数据

- `D:/data/aec-synth/train_cascade_lin/` — 训练 3000 条全量(源 train_v1 经 MDF 预计算;首版随源 PCM_16 落盘有削波,已改 FLOAT 重写);
- `D:/data/aec-synth/eval_v3_lin/` — 验证 60 条(源 eval_v3 同法预计算,FLOAT)。

## 结果速览(AECMOS 官方 ONNX 同集同口径,eval_v3 60 条全段,DT 口径)

| DT | 传统 WP-06 | 全 NN v4 | 零训练级联 | 级联 P1-1(40ep) |
|---|---|---|---|---|
| echo_mos | 2.467 | 3.754 | 3.174 | **3.919** |
| deg_mos | **2.401** | 1.743 | 1.887 | 1.785 |

- DoD(echo≥3.5 且 deg≥2.2,DT 口径):echo **3.919 ✅**(三方最高);deg **1.785 ❌** → DoD 未达成(echo 达标 / deg 未达标)。
- 结论:deg 病理在掩码不在接口——全 NN 1.743 / 零训练 1.887 / 训练级联 1.785 三条 NN 路同档(传统的 2.401 靠 DTD 冻结保近端),250ep 收敛口径同结论;下一步近端门控。

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
