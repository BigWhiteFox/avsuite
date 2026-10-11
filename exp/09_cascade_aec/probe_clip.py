"""削波影响面探针:40 条随机 clip 的 y_echo/e_lin 峰值分布(决定 wav 子类型)。"""
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_trad_aec"))
from pipeline import TradAEC  # noqa: E402

with open(r"D:/data/aec-synth/train_v1/manifest.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
rng = np.random.default_rng(7)
pick = [rows[i] for i in rng.choice(len(rows), 40, replace=False)]
aec = TradAEC(method="mdf", partitions=8, mu=0.7)
ny = ne = 0.0
mx_y = mx_e = 0.0
k_y = k_e = 0
nc_y = nc_e = 0
for r in pick:
    n = r["name"]
    mic = sf.read(rf"D:/data/aec-synth/train_v1/mic/{n}.wav", dtype="float32")[0]
    far = sf.read(rf"D:/data/aec-synth/train_v1/farend/{n}.wav", dtype="float32")[0]
    res = aec.process(mic.astype(np.float64), far.astype(np.float64))
    y = res.y_echo
    e = mic.astype(np.float64) - y
    ny += np.abs(y).max()
    ne += np.abs(e).max()
    mx_y = max(mx_y, np.abs(y).max())
    mx_e = max(mx_e, np.abs(e).max())
    if np.abs(y).max() > 1:
        k_y += 1
        nc_y += int((np.abs(y) > 1).sum())
    if np.abs(e).max() > 1:
        k_e += 1
        nc_e += int((np.abs(e) > 1).sum())
print(f"40 clips: y_echo>1 in {k_y} clips (mean max|y|={ny / 40:.2f}, max={mx_y:.2f}, "
      f"{nc_y} samples clipped); e_lin>1 in {k_e} clips (max={mx_e:.2f}, {nc_e} samples)")
