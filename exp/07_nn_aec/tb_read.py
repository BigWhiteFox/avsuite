"""读取 tensorboard events 里的 val/loss 序列（含 epoch 与 LR 还原），用于对照两条训练轨迹。

v4 的 tb events 是唯一存下来的训练侧时间线（train.log 只截了尾部）；
本脚本按 (val/loss, train/loss) 摘要输出，并反推 ReduceLROnPlateau 的 LR 阶梯：
patience=3 / factor=0.5 ⇒ 连续 3 个 epoch 未刷新 best 就减半。

    python exp/07_nn_aec/tb_read.py exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/tb \
        exp/07_nn_aec/outputs_v4_p05/lossctrl/nnaec_g0.5_j10/tb
"""
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def read(path: Path):
    ea = event_accumulator.EventAccumulator(str(path), size_guidance={0: 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for t in tags:
        ev = ea.Scalars(t)
        out[t] = [(e.step, e.value, e.wall_time) for e in ev]
    return out


def lr_steps(vals, patience=3, factor=0.5, lr0=1e-3):
    """按 ReduceLROnPlateau(mode=min, patience, factor) 反推每个 epoch 后的 LR。"""
    lr, best, bad, hist = lr0, float("inf"), 0, []
    for v in vals:
        if v < best:
            best, bad = v, 0
        else:
            bad += 1
        hist.append(lr)
        if bad >= patience:
            bad = 0
            lr *= factor
    return hist


def main():
    for p in sys.argv[1:]:
        p = Path(p)
        ev = read(p)
        val = [v for _s, v, _w in ev.get("val/loss", [])]
        tr = [v for _s, v, _w in ev.get("train/loss", [])]
        print(f"== {p}  val epochs={len(val)} train steps={len(tr)}")
        if val:
            print(f"   val: e1={val[0]:.4f} e20={val[min(19,len(val)-1)]:.4f} "
                  f"emin={min(val):.4f}@{val.index(min(val))+1} elast={val[-1]:.4f}")
        if tr:
            print(f"   train/loss: step1={tr[0]:.4f} last={tr[-1]:.4f} "
                  f"n_steps/e≈{len(tr)/max(len(val),1):.0f}")


if __name__ == "__main__":
    main()
