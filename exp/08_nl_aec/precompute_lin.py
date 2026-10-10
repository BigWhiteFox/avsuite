"""预计算线性回声估计（WP-08 级联结构的前置）：每条 clip 跑 TradAEC(MDF)，
把 y_echo（线性滤波器回声估计）存成 wav。

动机（负实验结果文档化）：GRU 直接吃原始 far 学 echo 学到 ERLE 0dB——
回声含 10-80ms 时延 + 100ms RIR ≈ 180ms 记忆，远超 GRU 容量。Ivry 2021 的
级联设计即为此：线性级消化长记忆，NN 只学记忆less 的非线性残差。
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_trad_aec"))

from pipeline import TradAEC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=Path, required=True)
    ap.add_argument("--out-sub", default="lin_est")
    args = ap.parse_args()

    root = Path(args.synth_dir)
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8")))
    out = root / args.out_sub
    out.mkdir(parents=True, exist_ok=True)
    aec = TradAEC(method="mdf", partitions=8, mu=0.7)
    done = {p.stem for p in out.glob("*.wav")}
    for i, r in enumerate(rows):
        if r["name"] in done:
            continue
        mic = sf.read(root / "mic" / f"{r['name']}.wav", dtype="float32")[0]
        far = sf.read(root / "farend" / f"{r['name']}.wav", dtype="float32")[0]
        res = aec.process(mic.astype(np.float64), far.astype(np.float64))
        sf.write(out / f"{r['name']}.wav", res.y_echo.astype(np.float32), 16000)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    print(f"[lin_est] {len(rows)} 条 -> {out}")


if __name__ == "__main__":
    main()
