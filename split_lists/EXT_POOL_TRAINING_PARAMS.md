# 扩展池 → 训练命令参数组合（t21 交付，t22/t24 直接消费）

> 2026-10-13 ｜ 依据：t21 扩量入库（LibriSpeech tc100+tc360、AISHELL-3、MUSAN）与 `ext_*`
> 扩展池清单（与全部冻结评测轴零交集，见同目录 `ext_pool_meta.json`）。
> 本文件只给"怎么用这批池子跑训练"的参数组合，**不含实验结论**。

## 0. 两条裁定（避免误用未入库语料，队长 2026-10-13）

| 语料 | 裁定 | 理由 |
|---|---|---|
| **AISHELL-4** | **不推**（留痕"渠道形态阻断，阶段2 若有官方 tar 源再议"） | ModelScope 该仓库**只有 test 划分的 11 个 parquet 分片（~5GB）**，无 train/无 wav；既不满足训练量级（评测下限都要 ≥300 条/场景），又需额外解析链路。**AISHELL-3（174 spk / 88k 文件 / 25GB 中文普通话）已覆盖"中文远端"轴多样性** |
| **train-other-500** | **不补下**（留痕 DP3 选项） | 磁盘 35G 余 < 峰值需求（28.5 包 + ~28 解压），会击穿 30GB 阈值；tc360(765 spk)+tc100+AISHELL-3 已把语音池从 305 readers 扩到 **~1500+ speakers**，"更大数量级"实质达成；其边际价值（带噪/口音变体）与磁盘风险不成比例 ⇒ 阶段2 换大盘实例可再下 |

## 1. 服务器源池路径（/root/data）

| 轴 | 训练源 | 评测源（冻结，训练侧禁区） |
|---|---|---|
| near/far 语音（英文） | `librispeech/LibriSpeech/{train-clean-100,train-clean-360}/` | `eval_v8_pools/near/`(993)、`eval_v8_pools/far/`(599) |
| near/far 语音（中文） | `aishell3/{train,dev}/`（官方 test 已排除） | AISHELL 侧暂无冻结评测池（如需须新建并过互斥核对） |
| 噪声 | `dns-noise/`（训用 564 = `noise_train70`+`noise_val15`）、`musan/musan/{music,noise,speech}/` | `eval_v8_pools/noise/`(95 = `noise_eval15`)；MUSAN 评测 = `ext_musan_eval15.txt` |
| RIR | `rirs/`（本机 rir_real 40 wav）+ pyroomacoustics 仿真 | `split_lists/rir_holdout15.txt`(8 条，禁区) |

## 2. 清单文件名（全部在 `split_lists/`，仓内与服务器同一套）

| 用途 | 清单文件 | 行数 |
|---|---|---|
| LS speaker 三分 | `ext_ls_readers_{train70,val15,holdout15}.txt` | 798 / 176 / 198 |
| **LS near/far 训练池（已生成，直接用）** | **`ext_ls_near_train70.txt` / `ext_ls_far_train70.txt`** | **45,474 / 44,944**（reader 各 399） |
| LS 评测文件 | `ext_ls_{near,far}_eval.txt` | 11,786 / 10,606 |
| AISHELL-3 speaker 三分 | `ext_aishell3_readers_{train70,val15,holdout15}.txt` | 130 / 22 / 22 |
| **AISHELL-3 near/far 训练池（已生成，直接用）** | **`ext_aishell3_near_train70.txt` / `ext_aishell3_far_train70.txt`** | **23,835 / 22,988**（reader 各 65） |
| AISHELL-3 评测文件 | `ext_aishell3_{near,far}_eval.txt` | 4,612 / 3,229 |
| MUSAN 文件三分 | `ext_musan_{train70,val15,eval15}.txt` | 1,420 / 293 / 303 |
| 元数据（计数 / sha256 / 被排除项 / 近远切分记录） | `ext_pool_meta.json` | — |
| 既有 DNS 噪声三分 | `noise_{train70,val15,eval15}.txt` | 472 / 92 / 95 |
| 既有语音池（DNS clean 305 readers） | `speech_*`（near/far 训练侧 = `speech_near_train70.txt` / `speech_far_train70.txt`） | 111 / 112 readers |

**near/far 训练池的 reader 级互斥是已做的，不是待办**：`ext_*_near_train70.txt` 与
`ext_*_far_train70.txt` 由 `sha256('<corpus>:nearfar_train:'+speaker)[:8]` 升序对半切产出，
两文件 reader 集合零交集（LS 399/399、AISHELL-3 65/65），且与各自 holdout15 零交集。

## 3. 训练命令参数组合（模板）

```bash
# 英文为主（LibriSpeech 扩展池）+ 中文补充（AISHELL-3），噪声 DNS+MUSAN
python exp/07_nn_aec/train.py \
  --data <train 输出目录> --val <val 输出目录> \
  --near-list split_lists/ext_ls_near_train70.txt \
  --far-list  split_lists/ext_ls_far_train70.txt \
  --near-dir  /root/data/librispeech/LibriSpeech \
  --far-dir   /root/data/librispeech/LibriSpeech \
  --noise-dir /root/data/dns-noise \
  --noise-list split_lists/noise_train70.txt \
  --rir-dir   /root/data/rirs \
  --nearfar-mutex utterance \
  --dt-plan configs/dt_plan_v1.json --dt-plan-preset train_v1_repro \
  --scenarios 0.4,0.3,0.3 --dt-leadin-sec 1.5 \
  --snr-range 15,30 --soft-clip-prob 0.15 --seed <未占用>
```

- 中文列（AISHELL-3）：把 `--near-list/--far-list/--near-dir/--far-dir` 换成
  `ext_aishell3_near_train70.txt` / `ext_aishell3_far_train70.txt` + `/root/data/aishell3`。
- 若 runner 只接受**目录**而非清单：把清单先物化成交集池目录
  （`scripts/split_speech_pool.py --materialize-root`，或本次 `split_extension_pools.py` 类似做法），
  再传池目录。

## 4. 与 dt_plan / mutex 的搭配（口径纪律）

- **DT 清单**：`--dt-plan configs/dt_plan_v1.json`，preset **必须显式二选一**：
  `train_v1_repro`（0.4/0.3/0.3、引导 1.5s、SER 0–10、SNR 15–30，与 train_v1 同分布）或
  `eval_v5_repro`（0.25/0.25/0.5、引导 2.0s、SNR 20–30，与 eval_v5/v7/v8 同分布）。
  给 `--dt-plan` 时**不要**再显式传冲突的 `--scenarios`（脚本会报错；缺省值被清单覆盖）。
- **mutex**：`--nearfar-mutex utterance`（池已 reader 级互斥，此项为二次保险）。
- **噪声**：训练侧只用 `noise_train70`+`noise_val15` 圈定的 564；**`noise_eval15` 与
  `eval_v8_pools/noise/` 是禁区**。MUSAN 训练侧用 `ext_musan_train70`+`ext_musan_val15`。
- **RIR**：训练用 `rirs/` 或仿真；**`rir_holdout15.txt` 8 条为评测禁区**。

## 5. 建议源权重起点（起手参考，非结论）

| 源 | 起点 | 说明 |
|---|---|---|
| LibriSpeech 扩展池（1172 spk） | 0.5 | 体量最大，覆盖 far 端长语音特性 |
| AISHELL-3（174 spk） | 0.3 | 补中文/口音多样性；量级小，权重不宜过高 |
| 既有 DNS clean（305 readers） | 0.2 | 与 eval_v8 同源英文池，保与历史代际可比 |
| 噪声：DNS 噪声 | 0.7 | 与评测同域，主噪声源 |
| 噪声：MUSAN（music/noise/speech） | 0.3 | 补音乐/宽带噪声多样性（历史缺口） |

每次只动一个变量（归因纪律）；评测集**不参与**权重——`ext_*_eval` / `eval_v8_pools/*` /
`noise_eval15` / `rir_holdout15` 全部禁区。

## 6. 复现与验收

```bash
# 重算扩展池清单（幂等；空份额拒绝落盘）
python scripts/split_extension_pools.py --split-lists split_lists --out-dir split_lists \
    --ls-root /root/data/librispeech --aishell3-root /root/data/aishell3 \
    --musan-root /root/data/musan/musan
# 验收：三分割自洽 + 与全部冻结评测轴零交集 + meta 一致（exit 0/1/2）
python scripts/verify_extension_pools.py --split-lists split_lists --derive \
    --ls-root /root/data/librispeech --musan-root /root/data/musan/musan
```
