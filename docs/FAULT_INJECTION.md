# 故障注入练习集(P3-1):把踩过的坑做成可复现 Bug 实验

六个实战坑 → 六个 pytest(`tests/test_bug_01…06`),每个都**真的复现**该坑的
特征失败:断言坏路径表现出文档记载的失败模式,同时护栏断言正确路径不受影响。
被测源码零改动——注入一律走 monkeypatch / 参数替换 / 包装函数。
全部数字为本机实测(avsuite 环境,Python 3.11.17;种子固定,可复现),
实测时间 2026-10-11 01:15–02:00。

全量复跑(在 `avsuite/` 下;PowerShell 不展开通配符,用显式路径):

```powershell
& 'C:\Users\14805\.conda\envs\avsuite\python.exe' -m pytest `
  tests/test_bug_01_hanning_window.py tests/test_bug_02_dtd_no_latch.py `
  tests/test_bug_03_dtd_mic_vs_far.py tests/test_bug_04_no_silence_freeze.py `
  tests/test_bug_05_gtcrn_static_qdq.py tests/test_bug_06_stft_reflect_stream.py -v
```

2026-10-11 实测:**6 passed in 3.46s**(唯一 warning 来自既有 `common.py`
的 `return_complex=False` 弃用提示,与本练习无关)。

## 总表

| 编号 | 坑 | 注入方式 | 特征失败(实测) | 结果 |
|---|---|---|---|---|
| BUG-01 | StreamingISTFT 误用 np.hanning 对称窗 | 参数替换(`window=`) | 流式对拍 f64 7.6e-11→**1.18e-2**;f32 2.2e-7→1.18e-2 | ✅ 忠实复现 |
| BUG-02 | DTD 去掉收敛闩锁 | monkeypatch 子类(converged 恒 True) | 冷启动 ERLE 锁 **1.03/1.32dB**(正确 10.50/17.77),dt_frac 0.632 vs 0.060 | ✅ 忠实复现 |
| BUG-03 | DTD 相干参考改回 mic-vs-far(v2 形态) | monkeypatch 子类(update 单行差 + 无闩锁) | dt_frac **0.989**(历史档 0.93–1.00),ERLE 卡 6.35dB vs 15.68 | ✅ 忠实复现(见 §3 变体说明) |
| BUG-04 | 去掉远端静默硬冻结(IDLE 0→1) | monkeypatch 常量 | far 恢复段 ERLE 30.70→**-16.22dB**(塌 47dB);静默段 y/mic 漂移 3.4× | ✅ 复现(PESQ→ERLE 替代,见 §4) |
| BUG-05 | GTCRN 静态 QDQ 量化崩坏 | 零注入(留档产物同音频对比) | SI-SNR 4.61/10.04/10.37 → **-12.52/-5.07/-6.67**(掉 15–17dB) | ✅ 忠实复现 |
| BUG-06 | STFT center=True+reflect 搬进流式循环 | 包装函数(测试内错误封装) | 对拍误差 0(逐位一致)→**1.688**;互相关峰滞后 145 样本 | ✅ 忠实复现 |

---

## BUG-01 StreamingISTFT 换 np.hanning(对称窗)

- **注入方式**:参数替换,不改源码。`core.istft.StreamingISTFT(n_fft=512, hop=320,
  window=np.hanning(512))`(对称窗)对照默认 `window=None`(周期 hann,
  `sin²(πk/N)`,与 `torch.hann_window(periodic=True)` 同款)。同一份离线
  STFT(周期窗分析)的 irfft 帧喂进两种窗的流式 WOLA,与离线 ISTFT 输出对拍。
- **观察到的失败模式**(seed=0,3s 信号):对称窗与周期窗系数差
  max|Δw|=4.7e-3(集中在窗尾),WOLA 合成窗 ≠ 分析窗 → 归一化比率失配:
  - f64 链:正确对拍 **7.6e-11**,对称窗 **1.18e-2**(爆 8 个量级);
  - f32 链:正确 **2.2e-7**(~1e-6 级),对称窗仍 **1.18e-2**
    (误差由窗失配主导,与浮点精度无关)。
- **对应实战坑**:`exp/07_nn_aec/README.md` L143「StreamingISTFT 默认窗是
  torch 同款**周期** hann(np.hanning 是对称窗,对拍必翻车)」;
  `exp/07_nn_aec/streaming.py` L102-103 是正确写法(从 `stft.window` 取窗)。
- **复跑命令**:`& …python.exe -m pytest tests/test_bug_01_hanning_window.py -v`

## BUG-02 DTD 去掉收敛闩锁

- **注入方式**:monkeypatch `pipeline.DTDStreaming` → 子类 `_NoLatchDTD`,
  `__init__` 末尾 `self.converged = True`(闩锁永闭)。状态机里 converged
  分支位于 warmup 之前,故 warmup 分支同时被短路(前 8 块改由 DT 判决接管,
  结局同为冻结,等价于无闩锁形态)。
- **观察到的失败模式**(seed=11,4s 纯 FEST,直达主导 RIR,ERL≈3dB,SNR≈20dB):
  冷启动 y≈0 → γ²≈0 < coh_thr,凡 Geigel 触发的块(实测 geigel 率 0.892)
  全部误判双讲 → 步长 ×0.02 + hangover 保持 → 整个收敛期锁死:
  - 正确路径:dt_frac **0.060**,ERLE[0.5–1s]=10.50、[1–2s]=**17.77dB**;
  - 无闩锁:dt_frac **0.632**,ERLE[0.5–1s]=**1.03**、[1–2s]=**1.32dB**
    (锁在 ~1–2dB 量级,2–4s 也只爬到 5.35dB)。
- **对应实战坑**:`exp/06_trad_aec/dtd.py` DTDStreaming docstring:「收敛闩锁:
  块级 ERLE 的 EMA 连续 ≥ latch_blocks 块超过 erle_thr 才允许判 DT。冷启动
  y≈0、γ²≈0,不闩锁会把整个收敛期误判成双讲而永久冻结(鸡生蛋问题)」。
- **复跑命令**:`& …python.exe -m pytest tests/test_bug_02_dtd_no_latch.py -v`

## BUG-03 DTD 相干参考改回 mic-vs-far(时域判据家族)

- **注入方式**:monkeypatch `pipeline.DTDStreaming` → `_MicVsFarV2DTD`:
  `update()` 是 `dtd.py` L119-196 的逐行副本,唯一差异是把相干第二通道从
  y_echo 换回 far(`Y = np.fft.rfft(far_b * self.win)`);并令
  `converged = True`——即 **v2 历史形态**(收敛闩锁是 v3 才引入的,v2 判据无闩锁)。
- **观察到的失败模式**(seed=14,6s 纯 FEST,扩散型 RIR:直达 480 + 混响尾主导,
  帧内能量占比 0.169,ERL≈-2dB,SNR≈16.5dB):帧长 256 << RIR 1600 taps,
  段内泄漏把 mic-vs-far 分段相干压到 ~0.17 < coh_thr=0.6 → 纯 FEST 全程误报:
  - 正确路径:dt_frac **0.040**,ERLE[1–3s]=**15.68dB**;
  - v2 注入:dt_frac **0.989**(落在历史实测档 0.93–1.00),ERLE[1–3s]=**6.35dB**,
    滤波器被误报永久压住、永不收敛。
  - **变体说明(最接近的忠实变体之对照)**:只改相干、保留 v3 闩锁的注入,
    实测 dt_frac **0.909**、ERLE 卡在半收敛 **12.01dB**——闩锁让滤波器先收敛、
    之后才被相干误报冻住。这正是 v3 补闩锁的原因;dt_frac 0.93–1.00 的原始
    出处(带 ERLE 冻在 2dB)是 v2 无闩锁形态,故本练习以 v2 形态为主断言。
- **对应实战坑**:`exp/06_trad_aec/README.md`「DTD 判据演进」:v1 mic-vs-far
  时域相关(随机 RIR 打散,FEST 误报 95%)→ v2 mic-vs-far 分段相干(段内
  泄漏,纯回声相干 ~0.16–0.2,实测 dt_frac 0.93–1.00、ERLE 冻结 2dB)→
  v3 mic-vs-y_echo 后置相干(WebRTC suppressor 思路)+ 收敛闩锁。
- **复跑命令**:`& …python.exe -m pytest tests/test_bug_03_dtd_mic_vs_far.py -v`

## BUG-04 去掉远端静默硬冻结

- **注入方式**:monkeypatch `pipeline.IDLE` 0.0 → 1.0(远端静默门失效,静默块
  按活跃远端**全速自适应**)。走 `use_dtd=False` 基线分支做单点注入,不与
  DTD 自身的静默早退(`return 2`)纠缠——全链中两层静默冻结需同时拆除,
  文档在此说明。源码零改动。
- **观察到的失败模式**(seed=21,2s FEST → 6s 远端静默(数字底噪 -70dBFS)
  + 近端噪声 → 2s FEST):静默期谱队列 X[p] 还留着活跃 far 数据,滤波器把
  近端学成虚假相关(随机游走);毒化藏在系数里——静默段 y/mic 仅 **0.0092**
  (正确 0.0027,3.4×;X 是底噪所以读数小),far 恢复时爆发:
  - 正确路径:恢复段前 0.25s ERLE **30.70dB**;
  - 去硬冻结:**-16.22dB**(塌 47dB;0.25–1s -8.42dB,1–2s 才爬回 8.60dB)。
  - **PESQ 替代说明**:按任务许可,用 ERLE 塌陷 + 静默段 y/mic 漂移替代 PESQ
    断言(PESQ 在 10s×2 路径上过慢,不适合单测;两指标捕获同一损伤:
    近端被虚假系数吃掉/输出漂移)。
- **对应实战坑**:`pipeline.py` IDLE 注释「远端静默块硬冻结:谱队列 X[p] 还留着
  旧 far 数据,慢速自适应会把近端学成虚假相关(实测 NST y/mic=0.17、
  PESQ 2.30→1.73 的元凶)」;`dtd.py` docstring「软冻结的随机游走 8s 也能
  积累出 40% 量级的虚假系数,必须硬冻结」;README 设计决策 5(绝对下限)。
- **复跑命令**:`& …python.exe -m pytest tests/test_bug_04_no_silence_freeze.py -v`

## BUG-05 GTCRN 静态 QDQ 量化崩坏

- **注入方式**:零注入——崩坏产物已留档:
  `exp/05a_gtcrn_onnx/outputs/gtcrn_dns3.int8_static_BAD.onnx`(静态 QDQ)
  对照 `gtcrn_dns3.sim.onnx`(fp32),同段音频 onnxruntime 推理
  (`common.stft_convention` / sqrt-hann istft,官方约定)。
- **观察到的失败模式**(DNSDynamicMix seed=777 与 quantize.py 同源,3 条 4s):
  | clip | fp32 sim | 静态 QDQ BAD | 掉点 |
  |---|---|---|---|
  | 0 | 4.61 | **-12.52** | 17.1dB |
  | 1 | 10.04 | **-5.07** | 15.1dB |
  | 2 | 10.37 | **-6.67** | 17.0dB |
  逐条与留档 `outputs/quant_accuracy_rows.csv` 前三行一致;整档 50 条均值
  SI-SNR **11.541 → -20.764**(PESQ 1.869→1.120,STOI 0.877→0.197,
  见 `outputs/quant_report_static.json`)。方向断言:sim>0、BAD<0、掉点>10dB。
- **对应实战坑**:`exp/05a_gtcrn_onnx/README.md`「静态 QDQ 崩坏是本 WP 最有
  价值的负面结果」:GRU 递归状态逐帧累积量化误差 + 23.67K 小模型无冗余吸收
  量化噪声 + LayerNorm/Tanh 对 scale 敏感 → 输出变噪声;量产 INT8 需 QAT 或
  混合精度,dynamic PTQ(只动 Linear/GRU 权重)仅掉 0.26dB。
- **复跑命令**:`& …python.exe -m pytest tests/test_bug_05_gtcrn_static_qdq.py -v`
  (缺 onnx 产物或本机 DNS 数据时按仓内惯例 skip)

## BUG-06 STFT center=True 配 reflect 做流式

- **注入方式**:包装函数(测试内实现错误流式封装,不改被测源码):逐 chunk
  `torch.stft(chunk, 512, 256, …, center=True, pad_mode="reflect")` 拼谱后过
  统一 ISTFT——把离线整段约定直接搬进流式循环的实战写法。对照正确路径:
  全局帧网格 + zeros 约定(`core.stft.StreamingSTFT`)。
- **观察到的失败模式**(torch.manual_seed(6),1.6s 白噪声):每个 chunk 被当成
  独立信号——镜像 pad 复制 chunk 自身边缘(首帧错位),hop 网格按 chunk 局部
  重启(k·320 而非 t·256)→ 帧落点系统性错位,传播到整条链:
  - 正确路径:流式帧 vs 离线帧 max|Δ|=**0.0**(逐位一致),往返重构 5.6e-16;
  - 注入路径:max|y_bad − y_ref| = **1.688**(信号峰值幅度的 5 倍+,差 9 个
    量级以上);互相关峰滞后 **145 样本**(全链偏移签名,非逐点噪声)。
- **对应实战坑**:`core/stft.py` 模块注释「center=True 两侧 pad n_fft//2,
  pad_mode 默认 'reflect'(同 torch.stft);**流式接口无法预知未来样本,
  约定 pad_mode='zeros',对拍时离线用同模式**」;`tests/test_streaming_nnaec.py`
  的对拍约定即按此建立。
- **复跑命令**:`& …python.exe -m pytest tests/test_bug_06_stft_reflect_stream.py -v`

---

## 边界与诚实说明

1. BUG-02/03/04 的场景为受控合成(白噪声 far + 合成 RIR + 固定种子)。Geigel
   触发依赖 ERL 踩破 alpha=0.5(6dB)假设或噪声底抬高 mic 峰——这正是该判据
   在真实数据上的失效条件;历史实测数字(dtd.py docstring、exp/06 README)来自
   真实 eval_v3 集,本练习用合成场景复现同一机制,量级吻合(见各节)。
2. BUG-03 的"忠实变体"问题:现版代码含 v3 闩锁,只改相干参考时 dt_frac 0.909
   (闩锁先把滤波器救到半收敛);v2 历史形态(相干改 far + 无闩锁)复现出
   历史档位的 dt_frac 0.93–1.00,故以此为断言主体,两套实测数字均在上文。
3. BUG-04 的 PESQ→ERLE 替代与 `use_dtd=False` 单点注入理由见 §4;毒化读数
   在静默段很小是机制本身(毒化在系数里、X 是底噪),断言落在恢复段爆发。
4. 全程未修改任何既有文件:`git status` 中本次新增仅
   `tests/test_bug_01…06` 六个文件、本文档与执行日志
   `reports/rectification/exec/P3-1.md`(其余 M/?? 条目为本次任务之前已存在的
   其他改动,与本练习无关)。
