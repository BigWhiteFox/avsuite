#!/usr/bin/env python
"""级联 AEC 的 DM 训练入口（P1-3b，薄壳）：等价于

    python exp/07_nn_aec/train.py --cascade-linear <其余参数原样透传>

为什么是薄壳而不是第二份训练循环
--------------------------------
级联 = 「同一个 NN 换一对输入」（`(e_lin, y_echo)` 取代 `(mic, far)`），模型/损失/
optimizer/LR 调度/断点续训三件套一字不改。把训练循环复制到 09 会立刻产生两套口径
（阶段0 刚吃过"两处实现漂移"的亏），所以这里只做一件事：**补上 `--cascade-linear`
并转交 07 的 `main`**。级联的数据侧（合成 + MDF 线性级）在
`exp/09_cascade_aec/cascade_dm.py`。

用法（口径继承 v4 + 队长 2026-10-13 裁定；注意 `--dm-scenarios` 顺序 = FEST,NST,DT）：
    python exp/09_cascade_aec/train.py --device cuda --dynamic-mixing --epochs 240 \
        --batch-size 16 --num-workers 8 \
        --dm-scenarios 0,0.5086,0.4914 \
        --near-dir <pool>/near --far-dir <pool>/far --noise-dir <pool>/noise \
        --near-list split_lists/speech_near_train70.txt \
        --far-list  split_lists/speech_far_train70.txt \
        --noise-list split_lists/noise_train70.txt \
        --rir-dir <rir-pool> \
        --val <eval集>_lin --out exp/09_cascade_aec/outputs_dm

    # val 侧（必须与训练同 per-clip 约定；见 precompute_cascade.py --fresh-instance-per-clip）
    python exp/09_cascade_aec/precompute_cascade.py --synth-dir <eval集> \
        --out-dir <eval集>_lin --fresh-instance-per-clip

`--cascade-linear` 缺省由本入口补上；显式再给一次也不会重复。想跑**纯 DM**（非级联）
请直接用 07 的 train.py。
"""
import runpy
import sys
from pathlib import Path

TRAIN_07 = Path(__file__).resolve().parents[1] / "07_nn_aec" / "train.py"


def main():
    argv = list(sys.argv[1:])
    if "--cascade-linear" not in argv:
        argv.append("--cascade-linear")
    print(f"[cascade] 转交 {TRAIN_07.name}（已补 --cascade-linear；线性级口径见 manifest "
          f"的 data.linear_stage）", flush=True)
    sys.argv = [str(TRAIN_07), *argv]
    runpy.run_path(str(TRAIN_07), run_name="__main__")


if __name__ == "__main__":
    main()
