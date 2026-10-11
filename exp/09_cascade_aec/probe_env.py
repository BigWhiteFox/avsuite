"""环境探针(P1-1):wav subtype/片长、单条 MDF 耗时 -> 决定预计算是否并行。"""
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_trad_aec"))
from pipeline import TradAEC  # noqa: E402

i1 = sf.info(r"D:/data/aec-synth/train_v1/mic/fileid_0_dt.wav")
print("train_v1 subtype:", i1.subtype, i1.samplerate, f"{i1.duration:.2f}s")
d3 = r"D:/data/aec-synth/eval_v3/mic"
i2 = sf.info(os.path.join(d3, os.listdir(d3)[0]))
print("eval_v3 subtype:", i2.subtype, i2.samplerate, f"{i2.duration:.2f}s")
print("D: free GB:", round(shutil.disk_usage("D:/").free / 2**30, 1))

aec = TradAEC(method="mdf", partitions=8, mu=0.7)
rows = list(csv.DictReader(open(r"D:/data/aec-synth/eval_v3/manifest.csv",
                                encoding="utf-8")))[:5]
t0 = time.perf_counter()
for r in rows:
    name = r["name"]
    mic = sf.read(rf"D:/data/aec-synth/eval_v3/mic/{name}.wav", dtype="float32")[0]
    far = sf.read(rf"D:/data/aec-synth/eval_v3/farend/{name}.wav", dtype="float32")[0]
    res = aec.process(mic.astype(np.float64), far.astype(np.float64))
t = time.perf_counter() - t0
print(f"MDF per clip: {t / len(rows) * 1000:.0f} ms "
      f"-> 3000 clips ~ {t / len(rows) * 3000 / 60:.1f} min (single proc)")
