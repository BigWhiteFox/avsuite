# P1-2 · 官方 AEC-Challenge ICASSP2022 测试集 + 官方 AECMOS 协议评估报告

- 日期:2026-10-11(01:36–02:10 +08:00)· 执行:整改执行者(P1-2)
- 脚本:`avsuite/exp/07_nn_aec/official_aecmos_eval.py`(新建,未改任何既有文件)
- 打分模型:官方 `Run_1668423760_Stage_0.onnx`(README 标注 48kHz 版本 1;
  `AECMOS_local/aecmos.py::AECMOSEstimator` 官方推理逻辑,仅缓存 session)
- 数据:官方 `test_set_icassp2022` 三子集原生 48k wav,800 clips
  (DT 300 / FEST 300 / NST 200;600/600/400 wav 全覆盖)
- 两列:传统 = WP-06 管道(`TradAEC mdf/8/0.7`,与 aecmos_eval.py --trad 同配置);
  NN = v4(`outputs_v4/nnaec_g0.5_j10/last.ckpt`,加载同 capstone/cascade.py)

## 1. 结论(官方协议口径,主表——可与赛题榜单直接比)

**proto = 官方协议裁剪**(FEST 送后半段 / DT 送最后 (len−15)/2 s / NST 全段),
三路同裁,官方 ONNX 本地推理:

| 方法 | FEST echo | FEST deg | DT echo | DT deg | NST echo | NST deg |
|---|---|---|---|---|---|---|
| 传统 WP-06 | 2.725 (n=299) | 5.000 | 2.305 (n=300) | 3.692 | 4.999 (n=200) | 4.183 |
| NN v4 | 3.094 (n=300) | 4.999 | 3.239 (n=300) | 3.114 | 4.998 (n=200) | 3.755 |
| **Δ (NN−trad)** | **+0.369** | −0.001 | **+0.934** | **−0.578** | −0.001 | **−0.428** |

- **DT echo:NN 优势 +0.93**(官方口径下仍保持;eval_v3 全段口径为 +1.29);
- **DT deg / NST deg:NN 落后**(−0.58 / −0.43,方向与 eval_v3 一致);
- FEST deg 与 NST echo 双方都在 5.0 上限(FEST 无近端语音,deg 无从损伤;
  NST 无回声);FEST echo NN 由 eval_v3 的持平转为 +0.37。
- 传统列 FEST n=299:1 条 clip 推理发散被排除(见 §5),其余全量覆盖。

## 2. 官方协议说明(注记,来源已核实)

来源:`AEC-Challenge/AECMOS/README.md` 与 `AECMOS_local/README.md`(两处一致),
实现样例 `AECMOS/aecmos.py::process_interspeech2021`;打分入口
`AECMOS_local/aecmos.py::AECMOSEstimator`。原话:测试集被加长以让模型收敛,
只送真正要评级的段。

1. **FEST**:只送**后半段**(最后 len/2 秒);本集 21.8–24.4s → 送 10.9–12.2s。
2. **DT**:只送**最后 (len_s−15)/2 秒**(按 48k 样本数 int;rating≤0 才不裁);
   本集 37.4–39.2s → 送 11.2–12.1s。
3. **NST**:**全段**送评;本集 14.8–16.9s(<20s,不触 estimator 截断)。
4. 三路(lpb/mic/enh)同一长度同样裁剪;**ECHOMOS/DEGMOS 是同一次模型前向的
   两个输出——官方脚本没有对 DEGMOS 单独再切「近端激活后的后半段」**。
5. 官方 estimator 对 >20s 输入截**前** 20s;对三路一律 `librosa.load(sr=48000)`
   后取 min_len。子目录含 `-with-movement` 变体,官方按同场景同 talk_type
   (fest→st / dt→dt / nst→nst),本执行按子目录归并,与官方一致。
6. 仓内 P1-2 条目的后备口径「DT 只送后半段」**未被使用**——官方协议在仓内
   README 明确可查,按官方公式执行;任务书中的「t=2s 裁剪」是 P1-3 对自合成
   eval_v3(2s DT lead-in)的口径,官方集无此对应,已作 `t2s` 对照版保留。

## 3. 送评版本对照(proto / full / t2s)

| 版本 | 方法 | FEST echo/deg | DT echo/deg | NST echo/deg |
|---|---|---|---|---|
| **proto(官方协议,主表)** | trad | 2.725 / 5.000 | 2.305 / 3.692 | 4.999 / 4.183 |
| | nn | 3.094 / 4.999 | 3.239 / 3.114 | 4.998 / 3.755 |
| full(t=0 全段,>20s 截前 20s) | trad | 2.798 / 5.000 | 2.707 / 3.310 | 4.999 / 4.183 |
| | nn | 3.313 / 4.999 | 3.969 / 2.727 | 4.998 / 3.755 |
| t2s(去前 2s,eval_v3 口径对照) | trad | 2.801 / 4.999 | 2.654 / 3.490 | 4.998 / 4.146 |
| | nn | 3.212 / 4.999 | 3.920 / 2.933 | 4.997 / 3.709 |

观察:DT 上 full/t2s 的数字普遍好于 proto(NN echo 3.97/3.92 vs 3.24)——
前 20s 含收敛期(远端独播、回声易压),官方协议只送更难的双讲评级段。
**与榜单比只用 proto;full/t2s 仅用于与本仓其他实验对读。**

## 4. 与自合成 eval_v3 的口径差

eval_v3(8s 短片段,DT 含 2s lead-in,**全段送评**,同一官方 ONNX):

| 方法 | 集合 | FEST echo/deg (n) | DT echo/deg (n) | NST echo/deg (n) |
|---|---|---|---|---|
| trad | eval_v3 全段 | 3.836 / 4.999 (26) | 2.467 / 2.401 (14) | 5.000 / 2.665 (20) |
| | **官方 proto** | 2.725 / 5.000 (299) | 2.305 / 3.692 (300) | 4.999 / 4.183 (200) |
| NN v4 | eval_v3 全段 | 3.762 / 4.998 (26) | 3.754 / 1.743 (14) | 5.000 / 2.073 (20) |
| | **官方 proto** | 3.094 / 4.999 (300) | 3.239 / 3.114 (300) | 4.998 / 3.755 (200) |

**哪些数字从此可与赛题榜单直接比**:本报告 §1 proto 主表——官方测试集 +
官方协议裁剪 + 官方 ONNX 模型。前提:榜单由官方 web API 打分(版本化
scoring_url),本执行用官方本地 48kHz v1 模型离线推理,模型版本可能与榜单
所用版本有差;其余口径(测试集、裁剪协议、talk_type 映射)与官方一致。

绝对值不可跨集直比的点:
- **评测段不同**:eval_v3 是 8s 全段(短、易);官方是长片段按协议裁剪
  (DT 只送双讲段,更难)→ 同方法官方集 echo/deg 普遍更严;
- **采样率路径**:官方原生 48k → 处理侧 `resample_poly(1/3)` 降 16k 送处理器 →
  送评 `librosa.load(sr=48000)`(librosa 0.11,soxr_hq)重采样回 48k;
  官方 README 钉的 librosa 0.9.1 默认 kaiser_best,重采样滤波器略有差别;
- **数据分布**:官方集含 `-with-movement` 变体、真实设备/内容分布,规模
  300/300/200 vs eval_v3 的 26/14/20——官方数字方差小得多,置信度高;
- **方向性结论两口径一致**:DT echo NN 大幅领先(+0.93 vs +1.29)、
  DT/NST deg NN 落后(−0.58/−0.43 vs −0.66/−0.59)。唯一方向变化:FEST echo
  由持平(eval_v3 −0.07)转为 NN +0.37。

## 5. 数据量 / 耗时 / 失败清单

| 阶段 | 规模 | 耗时 | 备注 |
|---|---|---|---|
| 冒烟(每子集 2 条) | 6 clips | ~2 min | 两列推理 + proto 打分通过 |
| 全量推理 trad | 800 clips,6 workers | 111 s | 0.83 s/clip(含 48k→16k 重采样) |
| 全量推理 nn | 800 clips,CUDA 单进程 | 50 s | 0.06 s/clip,全程 GPU,未触发退避 |
| 打分 proto | 1600 行 | ~40 s | 6 workers,官方逻辑 session 缓存 |
| 打分 full+t2s | 3200 行 | 85 s | 同上 |
| 发散扫描 | 1600 wav 全检 | ~40 s | 只 trad 1 条异常 |

**推理 1600 条全覆盖**(800 clips × 2 方法全部落盘,manifest_trad/manifest_nn
各 800 行)。**送评缺口 1 条**:trad FEST `wufNfJIb3kC7OvuxutARLQ_farend-singletalk-with-movement`
MDF 自适应滤波发散(输出 173,816 个非有限样本,float64→float32 cast overflow),
librosa 载入即拒(`Audio buffer is not finite everywhere`),三个 variant 均记
failed 排除,不污染均值 → trad FEST n=299。NN 列 800/800 全部有限。扫描底稿:
`D:/data/aec-synth/official_stage/divergence_scan.json`。

打分行:4800 = 3 variant × 1600;ok 4797 / failed 3(同一条发散 clip)。

## 6. 重跑命令

```powershell
$py = 'C:\Users\14805\.conda\envs\avsuite\python.exe'
cd D:\Project_for_Python\DSH
& $py avsuite\exp\07_nn_aec\official_aecmos_eval.py --stage infer-trad --workers 6
& $py avsuite\exp\07_nn_aec\official_aecmos_eval.py --stage infer-nn
& $py avsuite\exp\07_nn_aec\official_aecmos_eval.py --stage score --variants proto
& $py avsuite\exp\07_nn_aec\official_aecmos_eval.py --stage score --variants full t2s
& $py avsuite\exp\07_nn_aec\official_aecmos_eval.py --stage agg --variants proto full t2s
# 可选:--limit N 每子集抽样;--subsets dt fest nst;--methods trad nn
# 断点续跑:enh wav 已存在(>1KB)跳过;打分行已在 jsonl 跳过
```

## 7. 产物清单

- `D:/data/aec-synth/official_stage/enh_trad|enh_nn/{dt,fest,nst}/*.wav` — 1600 条增强 wav(16k FLOAT)
- `D:/data/aec-synth/official_stage/manifest_trad.csv` / `manifest_nn.csv` — 逐条状态(1 条发散已标注)
- `D:/data/aec-synth/official_stage/divergence_scan.json` — 全量非有限样本扫描底稿
- `avsuite/reports/official_aecmos/scores_official.jsonl` — 逐条送评分(4800 行,增量写)
- `avsuite/reports/official_aecmos/scores_official.csv` / `summary.json` — 汇总
- `avsuite/reports/official_aecmos.md` — 本报告
- `reports/rectification/exec/P1-2.md` — 执行日志
