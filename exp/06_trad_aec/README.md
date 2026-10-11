# exp/06_trad_aec — 传统 AEC（WP-06 / 模块6，纯 numpy，零 GPU）

GCC-PHAT 时延估计 → MDF/NLMS 自适应滤波（在线 DTD 三态冻结）→ RES 残余抑制。
**DoD 验收通过**：eval_v3（60 条受控合成集）FEST ERLE 20.55dB（≥20 目标）、双讲近端 PESQ +0.27、
近端单讲完全透传。7 项单测全绿（FIR 辨识/时延/三场景/端到端）。

## 结构与用法

```
far ─GCC-PHAT(时延+回声证据)─> 对齐 ─┐
                                     ├─MDF(DTDStreaming 逐块钩子)─> e ─RES(DT门控)─> out
mic ────────────────────────────────┘
```

```bash
python eval_aec.py --synth-dir D:/data/aec-synth/eval_v3        # 受控集验收
python eval_aec.py --official --num 40                          # AEC-Challenge 官方 synthetic
python ablation.py                                              # 收敛曲线 + 步长扫描
python eval_aec.py --synth-dir ... --method nlms                # NLMS 基线
python eval_aec.py --synth-dir ... --no-dtd / --no-res          # 组件消融
```

评估集生成（WP-07D 管道，车厢尺度）：
```bash
python ../../scripts/aec_synth.py --num 60 --dur-sec 8 \
  --out-dir D:/data/aec-synth/eval_v3 --rir-max-ms 100 --snr-range 20,30 \
  --soft-clip-prob 0 --dt-leadin-sec 2 --seed 13 \
  --near-dir .../train/clean --far-dir .../train/clean --noise-dir .../noise
```

## 验收结果（eval_v3：60 条 = 26 FEST / 14 DT / 20 NST，8s，100ms RIR，SNR 20-30dB）

| 配置 | FEST ERLE↑ | DT PESQ(近端)↑ | NST PESQ(透传)↑ | 结论 |
|---|---|---|---|---|
| **完整链路 MDF+DTD+RES (μ0.7,P8)** | **20.55** ✅ | **1.687**（基线1.414）| **2.446**（=基线） | DoD 通过 |
| 无 DTD | 26.64 | 1.784 | 1.702 ⚠️ | FEST"更好看"但近端单讲被吃 0.74 |
| 无 RES | 14.49 | 1.709 | 2.446 | RES 贡献 +6dB FEST |
| NLMS（无在线 DTD） | 23.95 | 1.180 ⚠️ | 1.103 ⚠️⚠️ | 无 DTD 的自适应=近端灾难 |

**读表要点（面试常考）**：无 DTD 时 FEST ERLE 反而更高（滤波器把一切都当回声学），
代价是 NST/DT 近端被当成回声消掉——ERLE 单独看会骗人，必须三场景成组评。

## 消融：NLMS vs MDF（`ablation.py`，2048 taps，真实语音+100ms 房间 RIR）

| | 稳态 ERLE | t(15dB) | RTF | 备注 |
|---|---|---|---|---|
| NLMS μ=0.5 | 18.8dB | 0.5s | 0.10 | 逐样本标量归一，语音非平稳下稳 |
| MDF μ=0.7 | 14.0dB | 3.0s | **0.01** | 逐 bin 归一，计算快 10 倍 |

**诚实发现**：教科书说 MDF 收敛快于 NLMS——那是对**平稳/有色平稳**信号
（单测 `test_mdf_beats_nlms_on_colored_input` 里 AR(1) 噪声 MDF 赢 3dB+ ✓）。
真实**语音**（强非平稳）下逐 bin 白化让每个音素都把 W 往自己方向拉，
||W|| 长到真值的 15 倍（过拟合），稳态反输 NLMS 4-5dB。生产级 MDF
（speexdsp）用 prop 逐 bin 梯度加权 + 更细的步长控制补这个差距——已知改进方向。
管道仍默认 MDF：块结构是 DSP/流式的正确形态，线性部分 14dB + RES 6dB 达标。

步长扫描（MDF）：μ∈[0.3,1.0] 稳态 13-14dB 平坦、μ=1.3 发散——信赖域+对角加载
把稳定边界推到 ~1.2（朴素 FDAF 语音上 μ>0.8 就震荡）。

## 关键设计决策（每一条都是实测教训）

1. **DTD 判据演进**（三版）：
   - v1 mic-vs-far 时域相关：随机 RIR 把相关打散 → FEST 误报 95% ❌
   - v2 mic-vs-far 分段相干：帧长 256 << RIR 1600 taps，段内泄漏使纯回声相干
     也只有 ~0.16 ❌；累积相关被混响稀释（corr≈直达/总能量比 0.1-0.3），与
     NST 噪声带重叠 ❌
   - v3（现版）**三层证据**：整段 GCC 峰旁瓣比（FEST 19-84 / NST 4.6-13.4，
     阈值 14，混响不稀释峰）做回声存在性先验 + mic-vs-**y_echo** 后置相干
     （时间对齐无泄漏，WebRTC suppressor 思路）+ ERLE 收敛闩锁（带回声证据
     门槛，防 NST 假收敛）✅
2. **三态冻结**：0=自适应 / 1=DT 软冻结(×0.02，跟踪路径漂移) / 2=近端主导
   硬冻结(=0)。软冻结不够：0.02 步长的随机游走 8s 积累出 40% 虚假系数
   （NST PESQ -0.4）；hangover 必须保持原状态语义（硬不被软化）。
3. **MDF 归一化**：跨度总功率 Σ_p|X_p|² 的慢 EMA(0.97) + 0.5×瞬时下限。
   当前块 EMA 在语音停顿时过期偏低 → 爆发点步长×10 → 发散（实测 -87dB）；
   纯瞬时 span 同样爆（-39dB）。
4. **对角加载用 max 不用 +**：加性 loading 把语音高频段（天然低于均值 20-30dB）
   步长压小几十倍 → 收敛慢 6 倍；max(span, 5%·mean) 只托底深谷 bin。
5. **远端活跃门要绝对下限**（-60dBFS）：安静 far 文件的"停顿"相对门分不清，
   静音块 corr 分母塌缩产生假证据 → 14 个块误开全速自适应 → 垃圾系数毒化
   整段近端（冻结只能保留垃圾不能清除，PESQ -1.8）。
6. **ERLE 统计跳过收敛瞬态**（2s），且 FEST 评估集 SNR 设 20-30dB——
   噪声地板会封顶 ERLE 读数（SNR 15dB 时 ERLE 永远 ≤15dB）。
7. **GCC-PHAT 对无直达径 RIR 会偏 20+ 样本** → 残余路径非因果 → ERLE 物理
   封顶 ~11dB。真实/仿真房间 RIR 直达径主导时锁定精确（±1ms）。

## 局限（README 明示，同样值钱）
- DT-from-start（通话一开始就双讲）无 lead-in：经典 AEC 结构性弱点，
  滤波器没机会收敛——NN-AEC（WP-07）的核心卖点之一
- MDF 语音稳态 misadjustment（见消融节，prop 加权是改进方向）
- 非线性回声（功放失真）线性滤波封顶 ~10-15dB —— WP-08 的入口

## 附记（V4 整改：对齐余量 / 周期重估 / DTD 复位 / 发散防护，2026-10-11 晚）

代码审查定位三个实现级缺陷（详见 `tests/test_trad_aec_hardening.py` 与
`reports/v4_rectification/`），本次修复，**默认参数逐位兼容**（原 93 项测试不改一字全绿）：

1. **对齐余量** `align_margin_ms`（默认 0）：对齐右移量 = max(估计时延 − margin, 0)，
   使残余直达径落在 MDF 窗口内部而非 lag 0——修复"负向时延漂移/路径移动 → 残余路径
   非因果 → FIR 不可表示"的实现缺陷（漂移集崩塌与官方集 with-movement 发散的共享根因，
   与本 README 上文第 7 条自记失效同族）。
2. **周期重估再对齐** `reestimate_sec`（默认关）+ `realign_min_delta_ms` /
   `reestimate_win_s` / `realign_search_pad_ms`：每 N 秒用最近窗口原始 mic/far 做窗ed
   GCC，超阈值即再对齐并复位 MDF+DTD。单测实测：+130ms 跳变（残差 155ms 越出 128ms 窗）
   下固定对齐 ERLE 崩至 ~3dB，重估列恢复 ~49dB（gap ≈46dB）。
3. **DTD 可复位** `DTDStreaming.reset()`：清闩锁/EMA/暖机计数，重对齐后暖机期重新冻结
   ——修复"闩锁只置位永不复位 + 路径突变后误冻正反馈"。
4. **MDF 发散防护** `w_abs_cap`（默认 1e3，谱域绝对界）+ 有限性检查：触发即复位并计数
   （`n_divergence_resets`），输出恒有限——修复官方集 with-movement 1/800 条系数几何
   增长耗尽 float64 的问题（相对信赖域随 ||w|| 水涨船高、leak 只界随机游走）。

**单元测试运动的如实发现（影响漂移归因的强度）**：噪声激励下，慢速时延游走
（±10ms/8s ≈ 0.36 样本/块）使 MDF 的 ERLE 从静态 ~46dB 跌到 4-5.5dB——**margin=25
也只恢复到 5.5dB**（块内路径移动 × 块级自适应的跟踪代价，大于纯几何的"非因果泄漏"
解释）；而周期激励（6 稳定正弦）下滤波器可"周期性预测"补偿超窗回声（残差 155ms 仍
19-27dB）。两个效应都说明：**漂移崩溃的定量归因（非因果泄漏 vs 跟踪代价 vs DTD 误冻
vs 插值副产物）必须靠真实语音+RIR 的重跑定位**（V4 任务 S10：
`reports/trad_hardening_rerun.md`），单元测试只锁几何语义与不倒退。修复配置的漂移/
官方集/真实 RIR 三列重跑数字以 S10 报告为准。

回归：全仓 pytest **100 passed / 0 failed**（93 + 7 新增）。

> 后续增补（Wave V5, 2026-10-11）：S2 评估工具 +24 单测后，全仓一度 = **124 passed / 0 failed**；
> V5 收口时全仓 = **261 项 = 257 passed + 4 xfailed / 0 failed**
> （`docs/NUMBERS.md` §3）；上述「93 / 100 passed」为本附记时点快照。

## 附记（V5 语义加固：非有限输入 / DTD 闩锁 / 跨文件状态 / GCC 证据口径，2026-10-11 深夜）

来源：`reports/refactor_wave5/deep_audit.md`（T4 深度审计，G1/G22/G23/G24/G7/G14）。
**默认有限输入路径逐位不变**（对拍指纹见 `reports/refactor_wave5/_t10_before.json` 与
`_t10_probe.py` 的 before/after）；本次只改"先前静默劣化"的输入与文档口径。

1. **G1 非有限输入显式降级**（`pipeline.process` / `gcc.gcc_phat_stats`）：
   far（参考端）含 NaN/Inf 时，旧行为是 FFT 把非有限值扩散到整条频带 →
   `peak_ratio=nan` → `nan >= 14` 为 False → `echo_prior=False` → DTD **整段硬冻结**
   （输出退化为直通、ERLE≈0dB，且全程无告警）；若 `delay_ms` 变 NaN，`align_far` 还会
   抛 `ValueError: cannot convert float NaN to integer` 直接崩。
   现在：`gcc_phat_stats` 返回 `(delay_ms, peak_ratio, evidence_valid)`，`evidence_valid=False`
   表示参考端被污染、证据不可用；`pipeline` 显式降级为「一次性 RuntimeWarning +
   `AECResult.n_gcc_fallback` 计数 + 按'可能有回声'放行自适应」，并跳过非有限对齐量。
   mic 侧含 NaN 仍照常给证据（丢失几个 mic 样本不影响参考端），保证真实回声链路的
   ERLE 不受影响（硬化用例实测 32.5dB 不变）。
2. **G22 DTD 闩锁双向化**（`dtd.py`）：`converged` 闩锁现在可**失锁**——判据刻意保守到只
   覆盖"回声估计本身塌掉"这一种病态：① 曾经真收敛过（`run` 累计 ≥ `reunlatch_blocks`，
   默认 30 块 ≈0.5s，挡住冷启动）；② `rms(y) < y_collapse_ratio·rms(mic)`（默认 0.05，
   即滤波器输出已塌缩——"系数被复位/估计彻底错误"的特征）；③ ERLE EMA 掉到
   `erle_thr − unlatch_hyst_db`（默认 3dB）以下；④ mic 有能量。失锁即重新暖机（`nb=0`）
   并把 `n_unlatched` 加一。
   修复前实测：收敛后把 `y` 置 0（回声估计彻底错误）连喂 10+ 块仍全部返回 0（全速自适应）；
   修复后同样输入在约 16 块内失锁并回到硬冻结。
   为什么不用"Geigel 未触发 + 相干低"当失锁条件：实测真实语音**单讲**块也会触发 Geigel
   （回声衰减 6dB 时 |mic| 峰值≈0.5·|far| 峰值，正好压在阈值上），用它会漏掉该修的病态；
   而"真双讲"时 `y`（回声估计）仍然强、`rms(y)` 与 `rms(mic)` 同量级，`y_collapsed` 不成立，
   故该判据天然不会在双讲/路径漂移时误解锁——**实测三条默认指标路径（有限输入）的输出
   指纹逐位不变**（对拍脚本与证据：`reports/refactor_wave5/_t10_probe.py`、`_t10_before.json`
   与 `_t10_after_final.json`）。
3. **G22 暖机计数的取舍（如实记录）**：`nb`（块序号，含远端静默块）仍是暖机判据用的计数器，
   **保持与 HEAD 逐位一致**；另外单独维护 `n_warm`（仅远端活跃块）作为可观测项。审计建议的
   "改用 `n_warm` 做暖机判据"实测会改变默认路径（`finite_static` 的 `dt_frac` 0.0400→0.0453、
   输出指纹全变），与本任务"默认有限输入逐位不变"的纪律冲突，故**未采用**；需要该语义的
   调用方可读 `n_warm` 自行门控，或在会话边界显式 `reset()`。
   代价（已知）：前置静默块会占用 `warm_blocks` 窗口（`nb` 也被静默块推进），极端情况下
   冷启动暖机缩短；由于该分支只在 `echo_prior` 为 False 时才影响冻结时长，
   而当前所有调用点都传 bool，实际暴露面很小。
4. **G23 跨文件状态继承陷阱**（`dtd.py` docstring + `kalman_eval.run_kalman`）：
   `far_rms_max`（每块只衰减 0.9995，≈35s 才降一半）、`far_peaks`、`converged` 都是
   跨调用持久状态。**一个 `DTDStreaming` 实例只服务一个文件**；逐文件循环的调用方
   （评估脚本）必须在换文件前显式 `reset()`，否则会继承上一文件的历史峰值 →
   `far_active` 长期 False → 整文件硬冻结。`kalman_eval.run_kalman` 已在返回前 reset。
5. **G24 `echo_prior=None` 自动判据删除**（`dtd.py`）：该回退靠"mic·far 块内积累积相关"
   判断有没有回声，实测**两个方向都不成立**——含 40ms 时延回声时 `corr_cum=0.0023`
   （< 0.1 门槛，真实回声被误判为无回声）；无回声且近端主导时 `corr_cum=0.80`
   （> 0.1 门槛，会全速自适应去学近端，正是它声称要防的场景）。现 `echo_prior`
   默认 `True`、传 `None` 直接 `ValueError`；`corr_cum` 仅留作诊断属性。
6. **G7 GCC 证据的尺度依赖（阈值 14 的适用范围）**：`peak_ratio = 相关峰/|旁瓣|中位数`
   **随分析窗长单调增大**（本机合成：0.1s→404、1s→1328、2s→2141、4s→2262、10s→2423、
   20s→2490；无回声型 0.1s→5.6~7.5、≥1s→4.2~6.2）。阈值 14 只在原标定窗长下有效；
   换窗长（统一 10s/整段/极小片段）须重新标定，短段（≤0.1s）无回声侧已实测到 7.5。
   本次**不改阈值**（会动 headline 口径），只在 `gcc.gcc_phat_stats` docstring 与
   `pipeline.EVIDENCE_THR` 注释披露。
7. **G14 RES 最小长度**：`residual_suppress` 要求 `len > n_fft`（默认 512）——短输入时
   scipy 会先把 `nperseg` 降到输入长度，使 `noverlap = nperseg - hop` 非法，抛
   `ValueError: noverlap must be less than nperseg`（看不出根因）。现在 `res.py` 与
   `pipeline.process` 都在入口给出含最小长度与原因的明确错误。
8. **API 变更：`gcc_phat_stats` 返回值由 2 元组变 3 元组**
   （`(delay_ms, peak_ratio, evidence_valid)`）。本目录内调用点已全部同步：
   `pipeline.py`（首段判决 + 周期重估两处）、`kalman_eval.run_kalman`、`ablation.py`；
   `gcc_phat_delay` 取 `[0]` 不受影响。**范围外仍待同步**（本任务 inScope 不含它们，
   仅记录）：`notebooks/06_aec_lab/06_aec_lab_run.py:181`、`notebooks/06_aec_lab/probe_samples.py:30`
   仍是 2 值解包 ⇒ 独立运行会 `ValueError`；`06_aec_lab.ipynb` 内嵌同款代码亦同。
   测试套件不覆盖这三个入口，故 CI 不会暴露。


