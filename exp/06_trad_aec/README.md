# exp/06_trad_aec — 传统 AEC（WP-06 / 模块6，纯 numpy，零 GPU）

GCC-PHAT 时延估计 → MDF/NLMS 自适应滤波（在线 DTD 三态冻结）→ RES 残余抑制。
**DoD 验收通过**：受控合成集 FEST ERLE 20.55dB（≥20 目标）、双讲近端 PESQ +0.27、
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
