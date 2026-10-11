# exp/10.5_capstone — 全链路 AEC→NR 级联 Capstone（WP-10.5，强烈建议项）

车载语音链的完整形态演示：**NN-AEC（20ms 流式）→ GTCRN 降噪**，三级效果可听可比可量化。

```
mic + far(lpb) ──[NN-AEC 流式, 20ms 块进块出]──> aec_out ──[GTCRN ONNX 整段批]──> nr_out
        └─────────────── ① 原始 mic      └── ② AEC 后        └── ③ AEC+NR 后
```

组件复用（零新训练）：WP-07 NN-AEC 权重 + WP-05b 流式引擎 + WP-05a GTCRN ONNX +
AECMOS 官方口径。**2 项单测全绿**（93/93 全仓）。

## 运行

```bash
python demo_app.py            # http://127.0.0.1:7861（上传 mic+far 两路音频）
python cascade.py 内嵌 CLI    # 或用 Python API：
```
```python
from cascade import Cascade
c = Cascade()
sigs = c.process(mic_wav, far_wav)                      # {mic, aec, nr} 三路等长
scores = c.aecmos_scores(far_wav, sigs, talk_type="dt") # 三级 echo/deg 分数
```

## 测试数据（eval_v3 的 DT 片段，fileid_0_dt）

| 级 | echo_mos↑ | 说明 |
|---|---|---|
| ① 原始 mic | ~1.5 | 回声清晰可闻 |
| ② NN-AEC 后 | ~3.5 | 回声基本消除（DoD 级性能） |
| ③ AEC+NR 后 | ~3.5 | 残余噪声再降，echo 不回退（单测断言） |

## 设计说明与已知边界（诚实口径）

- **AEC 级是真流式**（20ms 块进块出，NNAECStreaming 复用 WP-05b 的 cache 引擎，
  <1e-5 离线等价已由单测锁死）；**NR 级是整段批处理**（GTCRN 官方 stream/ 版
  可替换成全流式——替换点在 `cascade.py:run_nr`，接口已隔离）
- 级联器**每次调用自动重置流状态**（Gradio 连续请求安全，单测覆盖）
- 内存账：AEC 流式 cache ~270KB + GTCRN 权重 290KB + ONNX runtime——总 <10MB，
  嵌入式叙事成立
- AECMOS 分数是整段口径（同 WP-07 评估协议）；DEMO 场景用 dt 分
