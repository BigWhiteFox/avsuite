"""预计算级联 AEC 训练/验证数据(P1-1 / 级联=换输入对):

对 aec_synth 目录每条 clip 跑 TradAEC(MDF, partitions=8, mu=0.7——与 WP-06 评估、
WP-08 precompute_lin 完全同款线性级),取
    y_echo = 线性回声估计(已经 GCC-PHAT 对齐的滤波器输出)
    e_lin  = mic − y_echo(MDF 误差信号,RES 之前)

产出与 aec_synth 完全同构的三槽目录,AecDataset(dataset.py,只读复用)直接可读:
    mic/     <- e_lin    (NN 的 mic 输入:线性级擦完主回声后的残余)
    farend/  <- y_echo   (NN 的 far 参考:已对齐回声估计,NN 免学时延/回声路径)
    nearend/ <- 原目标(逐字节复制,目标不动)
    manifest.csv 原样复制(scenario 标签不变,dt,nst 过滤照常可用)。

    wav 子类型 = FLOAT:实测 40 条随机 clip 中 4 条 y_echo 峰值 >1.0(max 1.78)、
    3 条 e_lin >1.0(max 1.48),PCM_16 会削波(≈0.03% 样本)且与推理期内存
    float32 不一致——FLOAT 免量化免削波,dataset.py 按 dtype='float32' 读取,
    与子类型无关。--force 覆盖已有输出(断点续跑时不传)。

用法:
    python precompute_cascade.py --synth-dir D:/data/aec-synth/train_v1 \
        --out-dir D:/data/aec-synth/train_cascade_lin
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_trad_aec"))

from pipeline import TradAEC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--partitions", type=int, default=8)
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--force", action="store_true", help="覆盖已有输出(默认跳过三槽齐的 clip)")
    args = ap.parse_args()

    root, out = args.synth_dir, args.out_dir
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8")))
    for sub in ("mic", "farend", "nearend"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    aec = TradAEC(method="mdf", partitions=args.partitions, mu=args.mu)
    t0 = time.perf_counter()
    n_done = n_skip = 0
    for i, r in enumerate(rows):
        name = r["name"]
        p_mic = out / "mic" / f"{name}.wav"
        p_far = out / "farend" / f"{name}.wav"
        p_near = out / "nearend" / f"{name}.wav"
        if not args.force and p_mic.exists() and p_far.exists() and p_near.exists():
            n_skip += 1
            continue
        mic = sf.read(root / "mic" / f"{name}.wav", dtype="float32")[0]
        far = sf.read(root / "farend" / f"{name}.wav", dtype="float32")[0]
        n = min(len(mic), len(far))
        mic, far = mic[:n], far[:n]
        res = aec.process(mic.astype(np.float64), far.astype(np.float64))
        y_echo = res.y_echo[:n].astype(np.float32)
        e_lin = (mic - y_echo).astype(np.float32)   # = res.error(MDF 误差,RES 前)
        sf.write(p_mic, e_lin, 16000, subtype="FLOAT")
        sf.write(p_far, y_echo, 16000, subtype="FLOAT")
        shutil.copyfile(root / "nearend" / f"{name}.wav", p_near)   # 目标逐字节复制
        n_done += 1
        if (i + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(rows)}  {el:.0f}s  "
                  f"({el / max(n_done, 1) * 1000:.0f}ms/clip)", flush=True)
    shutil.copyfile(root / "manifest.csv", out / "manifest.csv")
    el = time.perf_counter() - t0
    print(f"[cascade_lin] {root.name}: {n_done} done + {n_skip} skipped / {len(rows)} "
          f"-> {out}  subtype=FLOAT  {el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
