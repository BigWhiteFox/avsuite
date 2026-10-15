"""训练口径协议（模型无关，供各 exp 训练入口共用）：LR/早停解析 + 显存探针。

**为什么放在 `core/`**：`exp/0X_*` 各目录之间存在同名模块（model / train / dataset）的
历史作用域冲突（见 `tests/conftest.py` §3.5），跨实验复用只能走仓库级共享位置。
把这两段口径代码放在这里，`exp/07_nn_aec/train.py` 与 `exp/04_dccrn/train.py`
**引用同一份实现**——阶段0 的教训是"两处实现早晚漂移"，而口径漂移会直接污染对照。

口径来源
--------
`reports/phase0_acceptance_20261013.md` §2.3b：`ReduceLROnPlateau(patience=3)` 会让 lr
在 20–30ep 折半、ep55–71 就早停 ⇒ 240ep 预算被截断（"240ep" 在新数据下不可比）。
因此 **DM 长训缺省 = patience 20 + 关闭早停**；非 DM（历史路径）缺省保持各入口现状
（NN-AEC：patience 3 / early-stop 8；DCCRN：patience 3 / early-stop 10）。
"""
import torch

#: DM 长训缺省（阶段0 §2.3b）
DM_LR_PATIENCE = 20
DM_EARLY_STOP = 0                     # 0 = 关闭早停（跑满 epochs）
DM_LR_KIND = "plateau"
DM_LR_FACTOR = 0.5


def resolve_schedule(dynamic_mixing, lr_patience=None, early_stop=None, *,
                     legacy_early_stop=8, legacy_patience=3):
    """解析 LR/早停口径（None = 未显式指定 → 按 DM / 非 DM 缺省）。

    显式给值永远优先，`*_explicit` 如实登记（口径可回读、不靠"默认值"叙事）。
    `legacy_*` 由各入口传入自己的历史缺省（NN-AEC 8 / DCCRN 10），DM 缺省统一为
    patience 20 + 关早停。
    """
    return {
        "kind": DM_LR_KIND, "factor": DM_LR_FACTOR,
        "patience": int(lr_patience if lr_patience is not None
                        else (DM_LR_PATIENCE if dynamic_mixing else legacy_patience)),
        "early_stop": int(early_stop if early_stop is not None
                          else (DM_EARLY_STOP if dynamic_mixing else legacy_early_stop)),
        "patience_explicit": lr_patience is not None,
        "early_stop_explicit": early_stop is not None,
        "basis": ("DM 长训缺省：patience 20 + 关早停（阶段0 §2.3b）" if dynamic_mixing
                  else f"现状缺省：patience {legacy_patience} + early-stop "
                       f"{legacy_early_stop}"),
    }


def probe_memory(batch_size, device, step, tb=None):
    """显存探针（`--probe-mem N`，阶段1 定 batch 用）：打印并返回峰值显存记录。

    GPU：`torch.cuda.max_memory_allocated()` / `max_memory_reserved()` 峰值 + 总显存，
    并按"峰值 ÷ batch × 0.85 安全系数"给一个**粗**线性外推的 batch 上界（激活随 batch
    近似线性，但固定开销不随 batch 变，故只作数量级参考，别当保证）。CPU：显存探针无意义。

    注意（实测踩过）：本函数只写**本机控制台可编码**的字符——Windows GBK 控制台下
    一个非 GBK 字符（如 "≲"）就会 UnicodeEncodeError 把训练打断。
    """
    rec = {"step": int(step), "batch_size": int(batch_size), "device": str(device)}
    if device.type != "cuda":
        rec.update(available=False,
                   note="非 CUDA 设备：显存探针无意义（max_memory_allocated 恒 0）")
        print(f"[probe-mem] step {step}：{rec['note']}", flush=True)
        return rec
    alloc = int(torch.cuda.max_memory_allocated())
    res = int(torch.cuda.max_memory_reserved())
    total = int(torch.cuda.get_device_properties(device).total_memory)
    rec.update(available=True, max_memory_allocated=alloc, max_memory_reserved=res,
               device_total=total,
               suggest_batch=int(0.85 * total / max(alloc / max(batch_size, 1), 1)))
    print(f"[probe-mem] step {step}  batch {batch_size}: "
          f"max_memory_allocated {alloc / 2**20:.1f} MiB（{alloc / 2**30:.2f} GiB）| "
          f"reserved {res / 2**20:.1f} MiB | total {total / 2**30:.1f} GiB | "
          f"0.85 安全系数粗估 batch <= {rec['suggest_batch']}（线性外推，仅数量级参考）",
          flush=True)
    if tb is not None:
        tb.add_scalar("probe/max_memory_allocated_mib", alloc / 2**20, step)
        tb.add_scalar("probe/max_memory_reserved_mib", res / 2**20, step)
    return rec
