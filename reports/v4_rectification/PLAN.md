# V4 整改任务分解（PLAN）

> 写域互斥表；验收全部由 Lead 承担（子代理只交付，不定论）。

| # | 任务 | 写域 | 依赖 | 验收标准 |
|---|---|---|---|---|
| S1 | 传统管道加固（对齐余量/重估/闩锁复位/发散防护） | exp/06_trad_aec/*.py、tests/test_trad_aec_hardening.py、06 README 附记 | 无 | 默认参数逐位兼容；新增测试全绿；漂移测试中 margin 修复增益 ≥6dB；全量 pytest 通过 |
| S2 | 评估工具补单测 | tests/test_official_protocol.py、test_compute_ci.py、test_precompute_semantics.py（tools/compute_ci.py 允许抽纯函数重构） | 无 | 新测试全绿、<30s/文件；compute_ci 重构行为等价 |
| S3 | 面试层文档同步 A | docs/CHEATSHEET.md、docs/STORY.md、README.md、exp/07_nn_aec/README.md | 无 | 与 NUMBERS 逐位一致；无 80 passed/0.1×SI-SNR/AECMOS 三分量/生产回级联残留 |
| S4 | 权威层修复 B | docs/NUMBERS.md（两处）、docs/adr/0005（附记）、docs/PAPER2CODE.md | 无 | "deg 保底"矛盾消除；ADR 0005 附记收回 DT echo 结论；grep 自检通过 |
| S5 | deg 机理直接量化 | exp/07_nn_aec/deg_mechanism.py、outputs/deg_mechanism/、reports/deg_mechanism.md | 无 | 三方 NLA/回声抑制主表 + 5 条手工抽查；结论只报测量事实 |
| S6 | NN 时延 OOD 探针（120–190ms） | D:/data/aec-synth/eval_ood_delay、exp/07_nn_aec/outputs/eval_ood_delay/、reports/delay_ood_probe.md | 无 | 配对声明核实（同 seed 同内容）；NN echo Δ 结论二选一有数 |
| S7 | ADR 0006 + V4 台账 | docs/adr/0006*、docs/adr/README.md、reports/v4_rectification/ | 无 | ADR 体例一致、边界表述准确、L3"不确定"裁决如实披露 |
| S8 | 时延账本补级联行 | tools/latency_bench.py、docs/LATENCY_BUDGET.md（附记） | 无 | MDF 单块实测 mean/p99；级联合计与 G.114 对标 |
| S10 | 修复版重跑受污染 headline | 新驱动脚本 + 新输出目录 + reports/trad_hardening_rerun.md | S1 | 漂移配对/官方集发散条/eval_v6 传统列三个数字；发散条消除 |
| S11 | 级联上真实数据 | exp/09_cascade_aec/official_cascade_eval.py 等 + reports/cascade_real_data.md | S1（仅发散防护） | 官方集或 eval_v6 至少一列级联数字（AECMOS 同口径） |
| S12 | NUMBERS 整合 + git 分批提交 + 终报 | docs/NUMBERS.md、git、reports/v4_rectification/FINAL.md | S1/S2/S3/S4/S5/S6/S8/S10/S11 | 新数字入 NUMBERS；工作树提交分批可回滚；终报含遗留决策项 |

## 时间线

- T+0：S1/S2/S3/S4/S5/S6/S7/S8 并行铺开（S1 三次启动失败后 Lead 亲自接管 S7）
- T+40min 起：S10/S11 依赖 S1 落地后启动
- 收口：S12 由 Lead 亲自执行（NUMBERS 单一编辑权、git、终报）
