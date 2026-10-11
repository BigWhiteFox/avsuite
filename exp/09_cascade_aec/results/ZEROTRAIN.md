# P1-1 止损基线:零训练级联(WP-06 线性输出直送现成 v4 NN 推断)

- 时间:2026-10-11 01:06–01:25 · 设备:CPU(避让并行训练)· Python:`C:\Users\14805\.conda\envs\avsuite\python.exe`
- 数据:D:/data/aec-synth/eval_v3,60 条(fest 26 / dt 14 / nst 20),与 trad / v4 对照同集同口径
- 执行件:`avsuite/exp/09_cascade_aec/zerotrain_baseline.py` · 执行日志:`reports/rectification/exec/P1-1-baseline.md`

## 方法(零训练,不碰训练代码/权重)

```
(mic, far) ─GCC-PHAT 对齐 + MDF(DTD 门控, partitions=8, mu=0.7, 与 aecmos --trad 同参)─> y_echo
e_lin = mic − y_echo                       # 线性级残差(= MDF 误差输出,不含 RES)
(e_lin, y_echo) ──当作 (mic, far)──> NNAEC v4(exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/last.ckpt)─> enh
```

口径与 07_nn_aec 一致:float32 波形原样进出(dataset.py/train.py 无波形归一化),全段批处理,不加抖动、不加缩放。

## AECMOS 结果(aecmos_eval.py --enh-dir,同集同口径)

| 场景 | n | 零训练级联 echo | 零训练级联 deg | 传统 WP-06 echo | 传统 WP-06 deg | 全 NN v4 echo | 全 NN v4 deg |
|---|---|---|---|---|---|---|---|
| FEST | 26 | 3.003 | 5.000 | 3.836 | 4.999 | 3.762 | 4.998 |
| DT | 14 | **3.174** | **1.887** | 2.467 | 2.401 | 3.754 | 1.743 |
| NST | 20 | 5.000 | 2.013 | 5.000 | 2.665 | 5.000 | 2.073 |
| **OVRL(60 条逐条均值)** | 60 | **3.708** | **3.278** | 3.904 | 3.615 | 4.173 | 3.264 |

对照来源:exp/07_nn_aec/outputs/aecmos_trad_wp06.json(任务书写作 "acedmos_trad_wp06.json",仓内实为此文件)、exp/07_nn_aec/outputs/aecmos_nn_v4.json;级联本体:results/zerotrain_aecmos.json(tag=enhdir,60 rows)。

## 解读

零训练级联的 deg 由线性级保底:DT 场景 deg 1.887 较全 NN v4 的 1.743 回升 +0.144(线性残差作主输入,近端损伤被兜住),同时 DT echo 3.174 较传统线性级 2.467 提升 +0.707(NN 级在线性残差上继续压回声)——双讲上同时优于两级的各自弱项;代价是 FEST echo 3.003 低于两个单级(单讲下 (e_lin, y_echo) 的分布偏移让 NN 掩蔽过头)。OVRL:echo 3.708(低于传统 −0.196)、deg 3.278(与 v4 持平 +0.014)。

线性级保底检查:fest 段 e_lin ERLE(skip 2s)均值 14.2 dB(逐条 5.8–20.9 dB)。

## 重跑命令

```powershell
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/exp/09_cascade_aec/zerotrain_baseline.py --eval-dir D:/data/aec-synth/eval_v3
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3 --enh-dir avsuite/exp/09_cascade_aec/results/zerotrain_enh --out avsuite/exp/09_cascade_aec/results/zerotrain_aecmos.json
```

冒烟:`zerotrain_baseline.py --limit 2`。

## 红线核查

仅新建:exp/09_cascade_aec/zerotrain_baseline.py、results/{zerotrain_enh/(60 wav), ZEROTRAIN.md, zerotrain_aecmos.json}、reports/rectification/exec/P1-1-baseline.md;未修改、未删除任何既有文件;git 未写操作。
