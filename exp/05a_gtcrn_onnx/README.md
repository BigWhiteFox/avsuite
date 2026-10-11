# exp/05a_gtcrn_onnx — GTCRN 移植 + ONNX + INT8（WP-05a / 模块5a）

GTCRN（Rong et al., ICASSP 2024）：23.67K 可训练参数、33 MMACs 的超轻量增强模型，
车载/嵌入式端侧的现役标杆。官方 repo 移植 + DNS3 官方权重 + 导出/对拍/基准/量化全链路。
**4 项单测全绿**（参数量带 / 权重加载推理 / ONNX 对拍 / 时间因果性）。

## DoD 验收（全部本机实测，2026-10-09）

| 项 | DoD | 实测 | 结果 |
|---|---|---|---|
| PyTorch vs ONNX 对拍 | <1e-4 | **6.32e-06** | ✅ |
| CPU RTF（4s 语音，单线程） | <0.2 | **0.0084** | ✅（24× 余量） |
| CPU 单帧（T=1，16ms 帧移） | <3ms | **0.673ms** | ✅（帧预算余量 23.8×） |
| INT8 PTQ 精度损失 | 报告 | 见下表 | ✅ |

INT8 量化对比（DNS 自混 held-out，seed 777，SNR U(-5,15)）：

**主口径 = 200 条扩样（2026-10-11 重评，`reports/gtcrn_heldout200.md`；前 50 条与旧集逐位一致）**：

| 方案 | PESQ（95%CI） | Δ PESQ（95%CI） | 结论 |
|---|---|---|---|
| fp32（sim 版） | **1.897**（[1.820, 1.975]） | — | 基线；旧 50 条口径 1.869 落在新 CI 内 |
| **dynamic PTQ (QInt8)** | **1.845**（[1.770, 1.920]） | **0.053**（[0.028, 0.077]） | **可用**；240KB（旧 50 条 Δ 0.067，CI 含 0） |
| static QDQ PTQ | 1.120 | — | **崩坏，弃用**；**旧 50 条口径，未随 200 条重评** |

旧 50 条口径明细（历史对照，`quant_report_{dynamic,static}.json`）：

| 方案 | PESQ | STOI | SI-SNR | 单帧 | 结论 |
|---|---|---|---|---|---|
| fp32（sim 版） | 1.869 | 0.877 | 11.54 | 0.59ms | 基线 |
| **dynamic PTQ (QInt8)** | 1.803 (-0.067) | 0.869 | 11.28 | 0.64ms | **可用**；240KB |
| static QDQ PTQ | 1.120 (-0.750) | 0.197 | -20.8 | 0.87ms | **崩坏，弃用** |

**静态 QDQ 崩坏是本 WP 最有价值的负面结果**：GRU 递归状态逐帧累积量化误差 +
23.67K 小模型无冗余吸收量化噪声 + LayerNorm/Tanh 对 scale 敏感 → 输出变噪声。
量产 INT8 需要 QAT（训练时伪量化）或混合精度（GRU 留 fp16/int16、Conv int8）——
详见 INTERVIEW-WP05a 定点化概念卡。dynamic 量化只动 Linear/GRU 权重（激活仍
fp32），故精度损失小，但提速也小（8%）——小模型瓶颈在 Conv 算子本身。

## 文件与用法

```bash
python onnx_export.py --check        # 导出 fp32 ONNX（动态 T 轴）+ onnxsim 简化 + 对拍
python bench.py                      # CPU RTF / 单帧时延 / RSS
python quantize.py --mode dynamic    # INT8 动态 PTQ（旧 50 条精度报告）
python quantize.py --mode static     # 静态 QDQ（复现崩坏结果用，教学对照）
python eval_heldout200.py            # 200 条主口径重评（fp32 vs INT8 dynamic，含 95%CI）
python infer_demo.py                 # 官方 mix.wav + DNS 自混 -> 增强 wav + 指标
```

产物按量化模式分文件（追溯性：dynamic / static 互不覆盖）：
`outputs/gtcrn_dns3.onnx`（0.40MB）/ `gtcrn_dns3.sim.onnx`（0.29MB，1484→455 节点）/
`gtcrn_dns3.int8.onnx`（dynamic PTQ 主产物，240KB）/
`gtcrn_dns3.int8_static_BAD.onnx`（static QDQ 崩坏对照，留档）/
`quant_report_{dynamic,static}.json` / `quant_accuracy_rows_{dynamic,static}.csv`（逐条明细）。
历史遗留 `outputs/quant_accuracy_rows.csv`（旧 static 跑逐条明细，曾把 dynamic 明细覆盖掉）
原样保留为存档；`quant_accuracy_rows_static.csv` 是该存档的逐位副本（SHA256 一致，使 static
侧也有模式专属文件名），`quant_accuracy_rows_dynamic.csv` 为 2026-10-11 本机重跑
（summary 与旧 `quant_report_dynamic.json` 逐位一致：PESQ 1.8694→1.8025 / SI-SNR 11.541→11.283）。
新跑只写模式专属文件名，不再出现"dynamic 明细被 static 覆盖"的追溯性缺口。

## 结构与移植说明

- `gtcrn.py`：官方原样移植（署名+MIT 见文件头），ShuffleNetV2 式 GTConvBlock ×8 +
  ERB 频带压缩（257→65+64=129 维）+ SFE + TRA 时序注意力 + 2×DPGRNN（分组双路 GRU）
- **参数量口径**（单测锁死）：可训练 23669（官方标称 23.67K）+ ERB 固定滤波器组
  24576（requires_grad=False 的查表 Parameter）= 总 48245
- 权重：`D:/data/repos/gtcrn/checkpoints/model_trained_on_dns3.tar`（官方 DNS3 训练版，
  另有 VCTK 版）；官方 stream/ 目录含流式 ONNX（WP-05b 参考实现）
- STFT 约定：**sqrt-hann 双端**（stft 与 istft 各乘 hann^0.5，等效整窗 WOLA），
  n_fft=512/hop=256；torch≥2.x 的 istft 只吃 complex，common.py 做视图转换
- 时间因果性已单测验证（GTConv 左侧 padding + inter-GRU 单向）：扰动未来帧，
  过去帧输出逐位不变——流式部署的合法性前提

## 与批次内其他模块的连接
- 对拍/基准方法学复用 WP-01（对拍纪律）与 core/evaluate（指标口径）
- 本模型是 WP-10 Gradio demo 与 WP-10.5 级联 capstone 的默认端侧引擎
- C++ 单帧推理（WP-05c 选做）直接用 `gtcrn_dns3.sim.onnx` + onnxruntime C++ API
