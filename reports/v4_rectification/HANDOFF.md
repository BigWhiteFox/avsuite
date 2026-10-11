# V4 整改交接书（HANDOFF）— 供新会话继承

- 写入时间：2026-10-11 ~14:50（+08:00）；窗口截止约 17:30，剩余约 2.5h
- 写人：本轮 Lead（决策/验收）；本文件是新会话的**唯一权威交接入口**
- 用户指示：本轮因 API 速率限制未能用满并行预算；**新会话已配三源 API（失败可切换）**，
  并行子代理失败率应显著下降

---

## 0. 新会话行动顺序（10 分钟读完本节即可开工）

1. 读本文件 §1–§5（状态与验收要点）
2. 若 S1（管道加固）已交付且验收通过 → 立即并行发 §6.1（S10）与 §6.2（S11）
   （任务书原文可复制；**注意把相对路径按 §7 环境块澄清**）
3. S10/S11 回报后 → Lead 亲自执行 S12（§6.3：NUMBERS 整合 + git 分批 + FINAL.md）
4. 若某子代理失败：按 §8 运维教训重发（先查残留文件，再原样重发；三源 API 下失败应少见）

---

## 1. 本波目标（复评驱动修复，第三波之后的第四波）

第三方复评定四大缺陷，本波对应四条修复线：

| 线 | 缺陷 | 修复 | 状态 |
|---|---|---|---|
| A | 漂移/Kalman 两个"证伪"headline 实验设计偏置 | S1 管道四件套 + S10 重跑 + S6 判别探针 + ADR 0006 | S6✅ ADR0006✅；S1 进行中；S10 待发 |
| B | 级联（生产推荐）零真实数据验证 | S11 级联上官方集+eval_v6 | 待发（依赖 S1 的发散防护） |
| C | 文档双层真相（80/93、AECMOS 三分量、ADR 0005 未更新…） | S3/S4/S9 + Lead 修正 | S3✅ S4✅；S9 进行中；Lead 已修 NUMBERS/CHEATSHEET/PAPER2CODE/README |
| D | 治理缺口（git 未提交、脚本零测试、ADR 0006 缺、P2-3 无豁免） | S2 + S12 + ADR 0006 | ADR 0006✅；S2 进行中；git 待 S12 |

## 2. 已完成并验收（新会话不必重做）

### 2.1 S6 时延 OOD 判别探针（★本波最重要成果，改写第三波结论）
- **结论**：NN v4 在 +110ms 出窗时延下 DT echo 3.721→1.919（配对 Δ−1.802±0.125，
  154/154 全降，对 mic 地板余量 +2.18→+0.29）——**"漂移稳健是结构性的"被证伪，
  实为分布覆盖性稳健**（训练失配窗 0–90ms，漂移集 ±10ms 游走恰在窗内无判别力）。
  生产含义：NN 前置 GCC 粗对齐必须保留。
- 配对三层验证（manifest 全 300 行仅 delay +110.00ms；NST 76/76 逐字节一致；v5 复跑
  300/300 逐位一致）。
- 产物：`reports/delay_ood_probe.md`、`D:/data/aec-synth/eval_ood_delay`、
  `exp/07_nn_aec/outputs/eval_ood_delay/`（含 check_pairing 等自检脚本）
- **已回写**：NUMBERS drift 节重写（修正①的机制宣称 + 新增"时延 OOD 判别探针"小节）、
  CHEATSHEET 漂移行、（S9 正在给 drift_sweep/drift_retrain 文末加附记 + INTERVIEW-WP07
  新增 Q10）

### 2.2 S4 权威层修复（Lead 已二次修正一处口径）
- NUMBERS.md：删"级联价值在 deg 保底叙事"矛盾表述→改为准确版；WP-06 表 20.55 口径注
- ADR 0005 新增附记：**收回**"DT echo 三方最高"（+0.17→+0.02 噪声级），FEST 4.589 维持，
  新版 60-90 秒面试稿
- PAPER2CODE 两处 20.55 口径同步
- ⚠️ Lead 复核修正：S4 初版脚注按"集属标错"口径写，实际是**同一 eval_v3 上
  eval_aec.py(20.55) 与 eval.py(20.37) 的 0.18dB 脚本口径差**（06 README L16/L31 证明
  其验收对象就是 eval_v3）——已在 NUMBERS 口径注/CHEATSHEET/PAPER2CODE 三处更正。

### 2.3 S3 面试层文档同步
- CHEATSHEET（93、官方集行、deg 话术、真实 RIR/漂移诚实行、INT8 Δ0.053）、STORY
  （93、真实集数字、"级联+近端门控"话术）、根 README（徽章/能力表/三行新模块）、
  07 README（93、w_sisnr 口径、soft-clip 披露）——与 NUMBERS 逐位对齐自检通过
- Lead 修补：README "WP-09 级联"改名"级联 AEC（P1-1，exp/09）"避免与 WP-09 评估工具撞号

### 2.4 S8 级联时延补账
- MDF 单块 mean 0.274 / p99 0.613 ms（逐块 vs 整段逐位等价已证）；级联合计（流式口径）
  **50.9 / 52.4 ms**，G.114 余量 ≈2.9×；块重缓冲未入账上界 62.9/64.4 亦披露
- 产物：`docs/LATENCY_BUDGET.md` 文末附记、`tools/latency_bench.py` 扩展、
  `exp/06_trad_aec/outputs/mdf_block_latency.json`

### 2.5 Lead 亲自完成（S7 三次启动失败后接管）
- **ADR 0006**（docs/adr/0006-kalman-vs-mdf-boundary.md）：Kalman 结论边界限定版——
  固定 q/r 外壳输 MDF 成立、不外推完整 FKF（三件本质件未装）、L3"不确定"裁决如实披露、
  复测前置条件、面试稿
- adr/README.md 索引补 0005+0006 两行（0005 原本就漏索引）
- `reports/v4_rectification/{README.md,PLAN.md}` 台账

## 3. 进行中（本轮已发、回报未到或刚到——新会话先核对实际状态）

| # | 内容 | 写域 | 验收要点 |
|---|---|---|---|
| S1 | 传统管道四件套：align_margin_ms / reestimate_sec+realign / DTD.reset() / MDF w_abs_cap+发散复位 | exp/06_trad_aec/*.py、tests/test_trad_aec_hardening.py、06 README 附记 | ①默认参数逐位兼容（原 93 测试不变绿则拒收）②新测试 b：margin=25 比 margin=0 ERLE 高 ≥6dB ③发散防护输出恒有限 ④README 附记在位。**先 `git status` + 读 06 目录确认是否已改** |
| S2 | 评估工具补测：trim_for/load16 往返/compute_ci（允许抽纯函数重构）/precompute 三槽 | tests/test_official_protocol.py、test_compute_ci.py、test_precompute_semantics.py、tools/compute_ci.py | 新测试全绿且单文件 <30s；compute_ci main 行为不变 |
| S5' | deg 机理量化（DT 近端衰减 NLA/lead-in 回声抑制/分频带，三方） | exp/07_nn_aec/deg_mechanism.py（前任半成品，重发接管）、outputs/deg_mechanism/、reports/deg_mechanism.md | 主表+5 条手工抽查；回答"过度抑制 vs 伪影"；前任半成品在 exp/07_nn_aec/deg_mechanism.py（已存在） |
| S9 | 残留清理+面试刷新：6 处 80-passed、3 处 AECMOS 三分量、INTERVIEW-WP07（Q1/Q6/Q7/Q8+新 Q10）、09 README eval_v5 表、THREE_WAY/ZEROTRAIN 横幅、drift_sweep/drift_retrain 附记、INTERVIEW.md 索引 | 见其任务书 | grep 职责文件内"80 passed/三分量"清零；数字与 NUMBERS 逐位一致 |

**若回报已到**：按验收要点抽查后把结论填进 §5 状态表；**若未到且文件已出现**：视为半成品，
按任务书重发接管（S5 的先例）；**若失败且无文件**：原样重发（§8 教训）。

## 4. 关键设计决策（已定，新会话遵照执行）

1. **S10 修复配置**：margin=25ms + reestimate_sec=2.0（窗口 2s，±40ms 搜索，Δ≥5ms 才
   重对齐并复位 MDF/DTD）+ 发散防护默认开；partitions=8 保持与旧基线可比。
2. **S11 级联真实数据**：线性级必须**逐位复刻** precompute_cascade.py 的路径
   （TradAEC(mdf,8,0.7) 全链、y_echo/e_lin= mic−y_echo、FLOAT）——级联 NN 训练分布如此；
   发散防护（S1 默认开）允许保留，margin/reestimate **不开**（保持训练-推理一致）。
3. **NUMBERS.md 单一编辑权归 Lead**（S12），子代理只写各自 reports。
4. **历史报告不改正文，只加附记/横幅**（ADR 0005/THREE_WAY/drift_* 的处理先例）。
5. 用户决策项挂起清单见 §9。

## 5. 状态总表（新会话先更新此表再动手）

| 任务 | 状态 | 交付物位置 |
|---|---|---|
| S3 文档同步 A | ✅ 验收通过 | docs/CHEATSHEET.md 等 4 文件 |
| S4 权威层修复 | ✅ 验收通过（含 Lead 口径二次修正） | NUMBERS/ADR0005/PAPER2CODE |
| S6 OOD 探针 | ✅ 验收通过（重大发现） | reports/delay_ood_probe.md 等 |
| S8 时延补账 | ✅ 验收通过 | LATENCY_BUDGET 附记等 |
| ADR 0006 + 台账 | ✅ Lead 完成 | docs/adr/0006*、reports/v4_rectification/ |
| S5 deg 机理 | ✅ 验收通过（Lead 收尾验收） | reports/deg_mechanism.md（高频带选择性损伤 −2.41dB@4-8k + 谱平坦化，非宽带能量压低；重建自检 6/6 逐位 PASS、复算 15/15 PASS） |
| S9 残留清理 | ✅ Lead 补完验收（S9 完成任务 1/2/部分3 后夭折，Lead 完成 Q6/Q7/Q8/Q10、09 README、THREE_WAY/ZEROTRAIN 横幅、drift 双附记、INTERVIEW 索引） | 见 git 批次③ |
| S1 管道加固 | ✅ **Lead 亲自实现并验收**（两次代理夭折零残留后） | pipeline.py（margin/reestimate/新字段）、mdf.py（w_abs_cap+发散防护）、dtd.py（reset）、tests/test_trad_aec_hardening.py（7 项全绿）、06 README 附记；**全仓 100 passed/0 failed** |
| S2 评估工具测试 | ❌ 代理夭折零残留 → 任务书移交新会话（§6.0） | （未开始） |
| S10 修复版重跑 | ❌ 待新会话发（S1 已就绪，任务书 §6.1） | 含 S1 单测发现的归因注意事项（§6.1 末尾） |
| S11 级联真实数据 | ❌ 待新会话发（任务书 §6.2） | |
| S12 整合+git+终报 | 🔶 git 已完成（5 批提交 205cfeb→05d0c8e，工作树干净）；NUMBERS 已更新 OOD/漂移修正（deg 机理/加固重跑数字待入）；FINAL.md 待新会话在 S10/S11 后写 | 提交清单见 §10 |

## 6. 未启动任务的完整任务书（可直接复制给子代理；发前按 §7 澄清路径）

### 6.0 S2：评估工具补单测（代理两次夭折零残留，任务书原样移交）
```
你是 avsuite 仓的测试工程师。仓库 D:\Project_for_Python\DSH\avsuite（git 仓，禁止 git 操作）。
Python：& 'C:\Users\14805\.conda\envs\avsuite\python.exe'（先 $env:PYTHONNOUSERSITE='1'），workdir 仓库根。
只允许：新建 tests/test_official_protocol.py、tests/test_compute_ci.py、
tests/test_precompute_semantics.py；允许小幅重构 tools/compute_ci.py（仅限抽纯函数，main 行为不变）。
1. test_official_protocol.py（被测 exp/07_nn_aec/official_aecmos_eval.py，先通读）：
   trim_for（约 L253-273）三分支（FEST 后半段 / DT 最后(len−15)/2 / rating≤0 不裁）+ 边界；
   load16 往返（48k→16k：长度≈n48/3、7.5kHz 以上能量≈0、与参考重采样逐位一致）。
   注意 import 副作用（路径常量/sys.path 注入）——确保 import 安全。
2. test_compute_ci.py（tools/compute_ci.py）：解析 CI 手工算例；n=1/全同值/空边界；
   配对 bootstrap 用固定 seed 小样本断言 lo ≤ Δmean ≤ hi 合理近似。
3. test_precompute_semantics.py（exp/09_cascade_aec/precompute_cascade.py）：三槽语义——
   合成 1s 案例（far 多正弦，echo=far 右移 20ms×0.5，mic=echo+near），断言 e_lin 对 far
   的相干/能量比 mic 显著下降（ERLE>10dB）、y_echo 与 echo 相关>0.9；若整脚本 CLI 耦合，
   写等价语义测试（直接驱动 exp/06_trad_aec 的 MDF）并注明。
4. 全量 pytest 回归（当前基线 100 passed），汇报最终数字。
约束：测试不依赖 D:/data 与网络；单文件 <30s；不改被测脚本本体（发现 bug 记录不修）。
背景：这些脚本产出仓库全部 headline 数字，当前零测试；注意 S1 已加固 pipeline
（ TradAEC 新参数 align_margin_ms/reestimate_sec 等，默认逐位兼容）。
```

### 6.1 S10：修复版传统管道重跑三个受污染 headline
```
你是 avsuite 仓的实验工程师。仓库 D:\Project_for_Python\DSH\avsuite（git 仓，禁止 git 操作）。
Python：& 'C:\Users\14805\.conda\envs\avsuite\python.exe'（先 $env:PYTHONNOUSERSITE='1'），workdir 仓库根。
>60s 的命令一律后台跑（run_in_background + job_output 收），分小步落盘，可断点续。

前置：S1 已给 TradAEC 加 align_margin_ms / reestimate_sec / 发散防护（先读 exp/06_trad_aec/
pipeline.py 的实际 API 与 tests/test_trad_aec_hardening.py 确认参数名）。

任务：用加固配置 TradAEC(method='mdf', partitions=8, mu=0.7, align_margin_ms=25,
reestimate_sec=2.0) 重跑三处，对比旧数字，回答"修复后还崩吗"：
① 漂移集：D:/data/aec-synth/eval_v3_drift（60 条）→ aecmos_eval.py --enh-dir 口径打分
   （先读 exp/07_nn_aec/aecmos_eval.py 用法；把增强 wav 落 exp/07_nn_aec/outputs/hardening_rerun/
   下新目录，勿覆盖既有 json）→ 与 aecmos_trad_drift.json（旧：FEST 2.119/DT 1.649）和
   aecmos_trad_driftbase.json（无漂移基线：FEST 4.011/DT 2.451）三方对比。成功判据：
   FEST echo 较旧漂移数字回升 ≥1.0（理想：接近基线）。
② 官方集发散条：先只跑 D:/data/aec-synth/official_stage 下那条
   wufNfJIb3kC7OvuxutARLQ_farend-singletalk-with-movement（lpb/mic 在
   D:/data/repos/AEC-Challenge/datasets/test_set_icassp2022/farend-singletalk/ 下同名前缀，
   48k→16k 用 resample_poly(1/3)，参考 exp/07_nn_aec/official_aecmos_eval.py 的 load16）→
   断言输出全有限、n_divergence_resets 有记录；再对官方 FEST 全 300 条跑加固列并按官方
   proto 协议打分（复用 official_aecmos_eval.py 的 trim_for/打分函数，新写小驱动），对比旧
   FEST echo 2.725±0.126。
③ eval_v6 真实车厢 RIR：D:/data/aec-synth/eval_v6_realrir（100 条）→ aecmos_eval 打分，
   对比旧 aecmos_trad_v6.json（DT echo 2.000±0.15）与旧 FEST ERLE 7.95dB（eval_aec 口径）。
报告：avsuite/reports/trad_hardening_rerun.md——三表（旧/加固/基线）+ 两个问题的直接回答：
(a) "线性 AEC 漂移崩塌"修复后是否翻转为"可跟"；(b) 真实 RIR 下传统 ERLE 崩落有多少是
零余量对齐贡献的。所有对比带 CI（1.96·sd/√n），重跑命令写全。
只新建 hardening_rerun 相关文件与报告；不改既有文件；不训练。
⚠️ S1 单测运动的归因注意（必读，写进报告的解读节）：噪声激励单测发现慢游走
（±10ms/8s）使 MDF ERLE 从静态 ~46dB 跌到 4-5.5dB 且 margin=25 仅恢复 5.5dB——
"块内路径移动 × 块级自适应"的跟踪代价可能与"非因果泄漏"叠加；而周期激励下滤波器
可周期性预测补偿超窗回声。真实语音+RIR 的崩溃归因（非因果泄漏 vs 跟踪代价 vs DTD
误冻 vs 插值副产物）请以重跑数字为准，解读时对照 exp/06_trad_aec/README.md 附记。
```

### 6.2 S11：级联上真实数据（官方集 + eval_v6）
```
你是 avsuite 仓的 NN 实验工程师。仓库 D:\Project_for_Python\DSH\avsuite（git 仓，禁止 git 操作）。
Python 同上（GPU 可用，torch 自动选）；>60s 命令后台跑。
目标：给"生产推荐形态"（级联 MDF→NN，exp/09_cascade_aec）补第一份真实数据证据。

铁律：线性级逐位复刻训练分布——读 exp/09_cascade_aec/precompute_cascade.py：TradAEC(mdf,8,0.7)
全链 → y_echo=res.y_echo、e_lin=mic−y_echo（float64→float32）；发散防护默认开可保留，
margin/reestimate 一律不开。NN 用 exp/09_cascade_aec/outputs/nnaec_g0.5_j10/best.ckpt。
推理方式参考 exp/09_cascade_aec/run_cascade.py（含 NN 前向的模型加载方式）。

任务：
① 官方集 800 段（D:/data/repos/AEC-Challenge/datasets/test_set_icassp2022，48k→16k load16
   同 official_aecmos_eval.py）：逐条 MDF→(e_lin,y_echo)→NN→enh（16k FLOAT 落
   D:/data/aec-synth/official_stage/enh_cascade/{fest,dt,nst}/，避免写仓内大 wav）；
   按官方 proto 协议（FEST 后半段/DT 最后(len−15)/2/NST 全段；复用 official_aecmos_eval.py
   的 trim_for 与 AECMOS 推理，模型 D:/data/repos/AEC-Challenge/AECMOS/AECMOS_local/
   Run_1668423760_Stage_0.onnx）打分 → exp/09_cascade_aec/results/aecmos_cascade_official.json。
② eval_v6_realrir 100 条：同管线（16k 原生）→ aecmos_eval.py 口径打分 →
   aecmos_cascade_v6.json。
③ 报告 avsuite/reports/cascade_real_data.md：官方集 DT/FEST 的三列对照（传统 2.305/2.725、
   NN 3.256/3.143 vs 新级联，带 CI，引 ood_real_echoes.md 旧值）；eval_v6 同（传统 2.000、
   NN 2.565）。回答：级联的 FEST/echo 优势在真实分布上是否保持？deg 是否仍 ~1.8-1.9 档
   （对照官方 NN deg 3.097、传统 3.692——注意真实集 deg 量级与合成不同，只比同级相对差）？
   发散条（wufNfJIb…movement）在级联下的表现单独记录。重跑命令写全。
只新建上述 json/驱动脚本/报告；不改既有文件；不训练。
```

### 6.3 S12：Lead 亲自执行的收口（不派代理）
1. **NUMBERS.md 整合**（单编辑权）：补 S5 deg 机理小节、S10 加固重跑小节（漂移节③改写
   "实现缺陷已修复，修复后数字见…"）、S11 级联真实数据小节、Kalman 一行指针（→ADR 0006）。
2. **pytest 全量回归**：预期 93+S1 新增+S2 新增全绿，数字写进 FINAL.md。
3. **.gitignore 追加**：`exp/07_nn_aec/outputs_v4_drift/`、`exp/07_nn_aec/outputs_v4_drift_full/`
   （含 ckpt/事件流，规则 *.ckpt/*.tfevents.* 已在但目录内有非忽略文件）；确认
   official_stage 的 enh wav 均在 D:/data（仓外）。
4. **git 分 4 批提交**（message 前缀 "[V4整改]"）：①3.5 波存量（文档 M+报告 ??——先于本波
   的全部未提交物）②本波代码+测试（06_trad、tests、tools、latency）③本波文档（NUMBERS/
   ADR/INTERVIEW 等）④本波报告与 JSON。提交前 `git status` 逐项核对无大文件/临时物。
5. **reports/v4_rectification/FINAL.md**：结果汇总 + §9 用户决策项 + 遗留清单。

## 7. 环境块（所有子代理任务书都要内嵌）

- 仓库：D:\Project_for_Python\DSH\avsuite（git 仓，禁止 git 操作）；整改台账在仓外
  D:\Project_for_Python\DSH\reports\rectification\（**给子代理的路径一律写绝对路径**）
- Python：`& 'C:\Users\14805\.conda\envs\avsuite\python.exe'`，先 `$env:PYTHONNOUSERSITE='1'`
- GPU：RTX 4060 Laptop，torch 自动选 CUDA
- 数据根：D:/data/aec-synth/{eval_v3,eval_v5,eval_v3_drift,eval_v6_realrir,eval_ood_delay,
  train_v1,…}；官方集 D:/data/repos/AEC-Challenge/datasets/test_set_icassp2022；AECMOS ONNX
  D:/data/repos/AEC-Challenge/AECMOS/AECMOS_local/Run_1668423760_Stage_0.onnx
- VB 语料 D:/data/voicebank-demand-16k/voicebank-demand-16k/extracted/train/clean；
  噪声 D:/data/dns-challenge-mirror/datasets/noise；真实 RIR 池 D:/data/rir_real/pool_16k_mono
- Windows/pwsh；>60s 命令必须后台跑（run_in_background=true + job_output 收）

## 8. 运维教训（本轮实证，新会话必读）

1. **子代理失败模式 A（路径）**：把仓外路径写成相对 avsuite 的路径 → 代理找不到文件即死
   （S7 两次因此挂）。对策：任务书内一律绝对路径。
2. **失败模式 B（无消息夭折）**：8 路并行时 3 个代理无 closing message 直接死（S1/S7/S8
   首发）。对策：失败后先 `git status`+`Test-Path` 查残留（本轮三次失败均零残留，原样重发
   成功）；新会话三源 API 下应更稳，建议单波并行 ≤6。
3. **失败模式 C（长命令中途死）**：S5 首发死于前台长命令。对策：任务书强制">60s 后台跑+
   断点续"；S5 二次重发时明确"接管半成品"。
4. **验收纪律**：子代理的口径表述必须抽查源头（S4 按我给的错误前提写了错误脚注，靠我复核
   06 README 才纠正——**Lead 的验收不能只看自检 grep**）。

## 9. 留给用户的决策项（FINAL.md 需原样呈现）

1. 漂移/真实 RIR 修复重跑（S10）若翻转旧结论：旧报告（drift_sweep/drift_retrain/
   realrir_domain 正文）是否需要除附记外的进一步处理——当前策略=只加附记不改正文。
2. 级联真实数据（S11）结果无论好坏，"生产推荐级联"话术是否随数字调整。
3. near-end gating（近端门控）训练仍属下一窗口（DECISIONS 决策 6 P1），本波不启动。
4. P2-3（多麦波束成形）在本波台账中补豁免记录（PLAN 已含），是否排下窗口由用户定。
5. git 提交是否推送远端：本轮策略=只本地提交不推送。
6. outputs_v4_drift*/ 两个训练目录：按 .gitignore 不入库，是否需要另存档由用户定。

## 10. git 提交清单（本轮，全部本地未推送）

| commit | 内容 |
|---|---|
| 205cfeb | [3.5波补录] 数据充分性整改存量（eval_v5/DATA_CI/GTCRN200/真实集OOD/真实RIR/drift重训负结果/工具） |
| 62f2e9f | [V4整改] 传统管道四件套加固 + 7 项新测试 + latency_bench MDF 基准（全仓 100 passed） |
| e1b43e7 | [V4整改] 文档双层真相止血（权威层+ADR0005附记+ADR0006+面试层全套） |
| 68060c3 | [V4整改] 报告与台账（HANDOFF/PLAN、deg 机理、时延 OOD 探针、drift_sweep 附记） |
| 05d0c8e 附近 | [V4整改] 补录 deg_mechanism.py 脚本（最新一笔，`git log` 为准） |

## 11. 新会话开工清单（压缩版）

1. 读 HANDOFF §1-§5 → 更新 §5 状态表
2. 发 S10（§6.1，含归因注意事项）+ S11（§6.2）——S1 已就绪，可立即并行；S2（§6.0）可同时发
3. 数字回填 NUMBERS（S12a）→ 全量 pytest → FINAL.md（含 §9 用户决策项）
4. 三源 API 下并行建议 ≤6；所有任务书路径已是绝对路径
