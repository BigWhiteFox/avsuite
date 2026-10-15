# configs — 合成口径清单

> 来源：`REMEDIATION_PLAN_V3.1.md` §2 工作项 **0-5 DT 合成比例清单化**
> （动因见附录C-1：DM 化后"默认比例不可追溯"）。

## 1. 本目录文件

| 文件 | 内容 |
|---|---|
| `dt_plan_v1.json` | DT（双讲）合成比例清单 v1：场景配比、DT 引导时长（= 重叠段补）、DT 的 SER/SNR 分布、**只读历史审计区** |
| `dt_plan_v2.json` | **v1 的派生版（t23 ③ 新增）**：原样继承 v1 的两个历史 preset，新增 `eval_snr_low` / `eval_snr_high` 两档（SNR 域外评测轴）；含 `derived_from`（v1 路径 + sha256 + 派生规则）。生成器 `scripts/make_dt_plan_v2.py`（幂等，`--check` 校验派生关系） |

清单 v1 由 `scripts/dt_plan.py --write configs/dt_plan_v1.json [--history-from-dir <合成集根>]` 生成
（幂等；历史审计区从既有 `manifest.csv` 实测）。校验/复现入口：`scripts/verify_dt_plan.py`。
v2 由 `scripts/make_dt_plan_v2.py` 从 v1 **派生**（不改 v1；`--check` 可复算校验）。

## 2. 清单里的 preset

| 清单 | preset | 场景配比 FEST/NST/DT | DT 引导 | DT SER | DT SNR | 用于 |
|---|---|---|---|---|---|---|
| v1 | `train_v1_repro` | 0.4 / 0.3 / 0.3 | 1.5 s | 0–10 dB | 15–30 dB | `train_v1`、`train_v1_drift`（seed 21） |
| v1 | `eval_v5_repro` | 0.25 / 0.25 / 0.5 | 2.0 s | 0–10 dB | 20–30 dB | `eval_v5`、`eval_ood_delay`、`eval_v6_realrir`、`eval_v7`、`eval_v8`/`eval_ns`/`eval_rir`(±sim) |
| v2 | `eval_snr_low` | 0.25 / 0.25 / 0.5 | 2.0 s | 0–10 dB | **0–10 dB** | `eval_snr_low`（t23 ③ 低 SNR 域外；配 `--snr-range 0,10`） |
| v2 | `eval_snr_high` | 0.25 / 0.25 / 0.5 | 2.0 s | 0–10 dB | **30–40 dB** | `eval_snr_high`（t23 ③ 高 SNR 域外；配 `--snr-range 30,40`） |

> v2 与 v1 的 `eval_v5_repro` **逐字段相同**（因此按 v2 的 `eval_v5_repro` 生成的集合与按 v1 生成
> 的清单值一致）；v2 只多两档 SNR 域外 preset。`--snr-range` 管 FEST/NST 行的 SNR，DT 行的
> SNR 由 preset 的 `dt_snr_db` 决定 —— 两者**必须同时设置**才能得到"全场景域外"的集合。

- 出入：两个 preset 的**引导时长与 SNR 区间不同**（训练 1.5s/15–30，评测 2.0s/20–30）——这是历史
  事实，清单如实分别登记；**不要**把两者混成一个口径。
- 原始命令出处：`train_v1_repro` 见 `exp/07_nn_aec/README.md` §训练数据复现命令；
  `eval_v5_repro` 见 `reports/eval_v5_dt300.md` 与 `reports/eval_v7_fest_nst.md`（同配方）。
- 语义：`dt_ser_db` = DT 的 SER（P_near/P_echo，越大回声越弱）；`dt_snr_db` = DT 近端相对噪声的
  SNR；`dt_leadin_sec` = 近端引导时长，**双讲重叠段 = dur_sec − dt_leadin_sec**。
- 分布种类只有 `uniform`（单区间均匀；点分布 = lo == hi 的退化情形）；未知 kind 直接报错。

## 3. 用法

```bash
# 合成时显式按清单采样（manifest 追加 dt_plan_version / dt_plan_preset 两列，
# 并写出同目录 dt_plan_profile.json：清单 sha256 + preset + 随机流消耗序 + 池规模 + CLI 留痕）
python scripts/aec_synth.py --num 300 --dur-sec 8 \
    --near-dir <near-pool> --far-dir <far-pool> --noise-dir <dns-noise> \
    --rir-dir <rir-pool> --rir-max-ms 100 --snr-range 20,30 --soft-clip-prob 0 \
    --seed <未占用 seed> --dt-plan configs/dt_plan_v1.json --dt-plan-preset eval_v5_repro \
    --nearfar-mutex utterance

# 看清单里有哪些 preset
python scripts/aec_synth.py --dt-plan configs/dt_plan_v1.json --dt-plan-list

# 验收（结构 / 留痕闭环 / 同分布核对 / 区间合规；退出码 0=通过 1=不通过 2=输入错误）
# sim-RIR 集（eval_v5/v6/v7/v8）的 C 项需要实跑，故要源目录；大集配 --limit 取前缀
DT_PLAN_NEAR_DIR=<near> DT_PLAN_FAR_DIR=<far> DT_PLAN_NOISE_DIR=<noise> \
python scripts/verify_dt_plan.py --plan configs/dt_plan_v1.json \
    --manifest <合成集>/manifest.csv --strict-preset --limit 20

# 追加"重生成前 N 条逐字节一致"（G 项；sim-RIR 集要显式加 --replay-sim-rir）
DT_PLAN_RIR_DIR=... python scripts/verify_dt_plan.py --plan configs/dt_plan_v1.json \
    --manifest <合成集>/manifest.csv --set-dir <合成集> --replay-prefix 3
```

### 3.1 验收项的适用边界（sim-RIR vs dir-RIR）

| 项 | 含义 | `--rir-dir`（dir） | pyroomacoustics 仿真（sim，**eval_v5/v6/v7/v8 都属这一档**） |
|---|---|---|---|
| A/B/D/E/F | 清单结构 / 留痕闭环 / 区间 / 占比 / preset 专项 | 适用 | 适用 |
| C | 同分布核对（DT 三列逐行等于随机流中的真实抽样值 + 场景反推） | 适用 | **适用**（同一实现：实跑 + 收尾结构锚定 + 逐行对账） |
| G | 重生成前 N 条与既有产物逐字节一致 | 适用 | 需显式 `--replay-sim-rir`（要本机 pyroomacoustics 且版本一致） |

要点：**C 不是"跳过"，而是换了取证方式**——RIR 段在流里的长度取决于来源（dir=固定一次
抽取；sim=pyroomacoustics 自取若干个），故不去离线复刻内部细节，而是直接观察真实现场流；
证据强度与离线重放等价（同样逐行、同样要求精确相等）。G 在 sim 档的跳过属**环境限制**
（缺 pyroomacoustics），不要当成数据失败。脚本会打印 `rir=dir|sim` 与跳过说明。

**缺省不给 `--dt-plan` 时行为与改动前逐比特一致**（不读清单、不追加列、不额外抽随机数）——
这是 0-5 的硬约束，由单测与"重跑 `train_v1` 逐字节一致"的回归实测守住。

## 4. 采样口径（复现与验收都依赖这份描述）

`--dt-plan` 路径下，每条样本的随机流结构（**逐句对照 `aec_synth.main` 实测**，不是推测；
`verify_dt_plan.py` 的 C 项按"收尾 random 前恰好 6 次 uniform"锚定块，就是照这份结构）：

```
1 random            pick_scene（按清单 scenarios 抽场景）
3 integers          近端/远端/噪声池下标
n integers(条件)    fit_len 裁剪偏移：**每个**源长于 dur 才抽一次（短的走循环补，不抽）
                    → dir-RIR 集实测 3 次；sim-RIR 集实测 2 次（噪声源通常已足长）
RIR 段              dir: 1 integers（池规模）｜sim: pyroomacoustics 自取若干个
                    （实测 4 次 uniform，含 uniform(..., size=3) 数组抽样）
2 uniform           ser/snr 各抽一次（值被清单采样覆盖，但随机流照旧前进）
3 uniform           清单采样：ser -> snr -> leadin（顺序固定）
1 uniform           delay_ms
1 random            soft_clip
```

三个坑（都踩过并已固化）：

1. **调用的消耗量依赖取值范围**：numpy `integers(n)` 在 n 为 2 的幂时只取一次 32 位，
   否则取两次；`integers(0, 32001)` 同理。所以 `dt_plan_profile.json` 记**真实池规模**
   （`pool_sizes`），供归档与排查。
2. **别漏参数**：复现时漏一个 `--dur-sec` 就会换掉 `fit_len` 的分支与抽样次数，
   manifest 大面积对不上（验收脚本初始版本就漏了这个）。
3. **RIR 段长度随来源变**（见 §3.1）：所以 C 项不去离线复刻 RIR 的内部细节，而是实跑一次
   直接观察真实流；`LogRng` 也必须完整转发 `uniform(..., size=n)` 这类数组抽样调用
   （t7 修复：早期版本一遇 `size=` 就抛 TypeError、整条回放崩掉）。

## 5. 边界

- 只新增：`dt_plan_v1` 不改任何既有 eval 集，也不改历史读数；0-4 的 eval_v8 按本清单生成后，
  其"隔离性偏差"读数才与旧集可比。
- `history.measured` 是**只读审计区**（各历史集合的实测占比/区间），**不参与采样**；
  它与 preset 声明值的差异属正常抽样抖动（例：`eval_v5` 实测 DT 占比 0.513 vs 声明 0.5，
  n=300 下 3σ 界约 ±0.087）。
- 换配比/换区间 = 新版本（`dt_plan_v2.json` + `version` 字段），旧集读数不可与新清单混用。
