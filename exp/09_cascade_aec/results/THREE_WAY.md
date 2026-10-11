# P1-1 三方对照表:传统 WP-06 / 全 NN v4 / 级联(线性 MDF→NN 残差)

- 日期:2026-10-11 · 评估集:D:/data/aec-synth/eval_v3,60 条(fest 26 / dt 14 / nst 20)
- 口径:AECMOS 官方 ONNX(`aecmos_eval.py`),三方同集同口径;eval_v3 为 8s 短片段全段送评
  (与官方绝对值有口径差,三方对比不受影响,见 exp/07_nn_aec/aecmos_eval.py 头注)
- 级联列 = P1-1 训练级联:**40 epoch 全程跑完(未触发 04:00 硬止损),best@epoch 39(val 23.99855)**,
  超参照 v4 配方(gamma 0.5 / w_sisnr 1.0 / scenarios dt,nst / jitter 10ms / batch 16 / seed 0,lr 1e-3 全新起步)

## 主表

| 场景 | 指标 | 传统 WP-06 | 全 NN v4 | 级联(P1-1) |
|---|---|---|---|---|
| **DT(双讲)** | echo_mos | 2.467 | 3.754 | **3.919** |
| **DT(双讲)** | deg_mos | **2.401** | 1.743 | 1.785 |
| FEST(远端单讲) | echo_mos | 3.836 | 3.762 | **4.502** |
| FEST(远端单讲) | deg_mos | 4.999 | 4.998 | 4.999 |
| NST(近端单讲) | echo_mos | 5.000 | 5.000 | 5.000 |
| NST(近端单讲) | deg_mos | **2.665** | 2.073 | 2.081 |
| OVRL(60 条逐条均值) | echo_mos | 3.904 | 4.173 | **4.532** |
| OVRL(60 条逐条均值) | deg_mos | **3.615** | 3.264 | 3.276 |

(传统/全NN 数字分别抄自 `avsuite/exp/07_nn_aec/outputs/aecmos_trad_wp06.json`、
`avsuite/exp/07_nn_aec/outputs/aecmos_nn_v4.json`;级联数字来自
`avsuite/exp/09_cascade_aec/results/aecmos_cascade.json`;OVRL 为 60 条逐条均值,
与 `results/ZEROTRAIN.md` 同口径。)

口径注记(补):全NN v4 列基线 json 的 DT deg 1.743 为 `best.ckpt` 所产,last.ckpt 全段重跑 1.759
(见 `avsuite/exp/07_nn_aec/outputs/aecmos_leadincrop/RESULT.md`);级联列与 v4-last 同 ckpt 口径
(enh 落盘 → 全段送评,各列 enh 与打分同源同一 ckpt)。

## DoD(echo≥3.5 且 deg≥2.2,DT 口径)

- DT echo **3.919 ≥ 3.5 ✅** —— 三方最高(较传统 +1.452,较全 NN v4 +0.165)
- DT deg **1.785 < 2.2 ❌** —— 与全 NN(1.743)同档,距传统(2.401)差 0.616
- **DoD 未达成(echo 达标 / deg 未达标)**
- 止损注记:**非止损截断**——40 epoch 全程训完,early-stop 未触发,best@epoch 39(任务硬止损 04:00 未启用)

## 与零训练级联的差分(回声收益来自训练;deg 病理与训练无关)

零训练级联 = v4 权重原封不动直接吃 (e_lin, y_echo)(并行任务,数字与解读见 `results/ZEROTRAIN.md`):

| DT | 零训练级联 | P1-1 训练级联 | Δ(训练带来) |
|---|---|---|---|
| echo_mos | 3.174 | 3.919 | **+0.745** |
| deg_mos | 1.887 | 1.785 | −0.102 |

FEST echo:零训练 3.003 → 训练 4.502(+1.499,训练修复了 (e_lin,y_echo) 分布偏移下 FEST 掩蔽过头)。
即:「换输入对」让现成 v4 权重零训练就拿到 DT echo 3.174,级联训练再 +0.745 至三方最高;
但 DT deg 在 零训练 1.887 / 训练 1.785 / 全 NN 1.743 之间不动——**近端保真病理是掩码+损失的结构性问题,
级联接口本身不治**(逐条看 DT deg 普遍 1.5~1.8、无单条崩塌,传统依靠 DTD 冻结保 deg,NN 无此机制)。

## 每列重跑命令

传统 WP-06:
```powershell
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3 --trad --out avsuite/exp/07_nn_aec/outputs/aecmos_trad_wp06.json
```

全 NN v4:
```powershell
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3 --nn-ckpt avsuite/exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt --out avsuite/exp/07_nn_aec/outputs/aecmos_nn_v4.json
```

级联(P1-1,两步:级联推理 → AECMOS):
```powershell
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/exp/09_cascade_aec/run_cascade.py    # eval_v3 逐条 MDF→(e_lin,y_echo)→NN(best.ckpt)
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' avsuite/exp/07_nn_aec/aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3 --enh-dir avsuite/exp/09_cascade_aec/results/enh --out avsuite/exp/09_cascade_aec/results/aecmos_cascade.json
```

## 级联列产物与数据指针

- 级联训练数据:`D:/data/aec-synth/train_cascade_lin/`(3000 条全量,`precompute_cascade.py` 实测 283s;三槽
  `mic/`=e_lin、`farend/`=y_echo、`nearend/`=原目标;FLOAT 子类型) + 验证 `D:/data/aec-synth/eval_v3_lin/`(60 条)
- ckpt:`avsuite/exp/09_cascade_aec/outputs/nnaec_g0.5_j10/best.ckpt`(best@39,val 23.99855)· 训练日志:`outputs/train.log`
- 增强 wav:`avsuite/exp/09_cascade_aec/results/enh/`(60 条,FLOAT);推理含 lin-check:前 3 条与 eval_v3_lin 逐位一致(max|Δ|=0)
