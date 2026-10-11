"""级联推理(P1-1):eval_v3 每条先跑 MDF 得 (e_lin, y_echo),再送级联 NN。

    mic ──MDF──> y_echo, e_lin = mic − y_echo ──NN(e_lin, y_echo)──> enh

与训练数据完全同构(precompute_cascade.py 同款 MDF 配置:TradAEC(mdf, 8, 0.7)),
enh wav 写 results/enh/{name}.wav(16kHz, FLOAT 子类型,免量化)供
aecmos_eval.py --enh-dir 用。前 3 条另与预计算目录 eval_v3_lin 逐位核对
(PCM_16 量化容差内应一致,MDF 为确定性过程)。

用法:
    python run_cascade.py --ckpt outputs/nnaec_g0.5_j10/best.ckpt
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "07_nn_aec"))      # avsuite/exp/07_nn_aec
sys.path.insert(0, str(HERE.parent / "06_trad_aec"))    # avsuite/exp/06_trad_aec
sys.path.insert(0, str(HERE.parents[1]))                # avsuite(core 包)

from pipeline import TradAEC  # noqa: E402
from model import NNAEC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("D:/data/aec-synth/eval_v3"))
    ap.add_argument("--ckpt", type=Path,
                    default=HERE / "outputs" / "nnaec_g0.5_j10" / "best.ckpt")
    ap.add_argument("--enh-dir", type=Path, default=HERE / "results" / "enh")
    ap.add_argument("--lin-dir", type=Path, default=Path("D:/data/aec-synth/eval_v3_lin"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.eval_dir / "manifest.csv", encoding="utf-8")))
    args.enh_dir.mkdir(parents=True, exist_ok=True)

    aec = TradAEC(method="mdf", partitions=8, mu=0.7)   # 线性级:与 WP-06/08 同配置
    st = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = NNAEC().to(args.device).eval()
    model.load_state_dict(st["model"])
    print(f"[cascade] ckpt {args.ckpt} val={st.get('val')} -> {args.device}", flush=True)

    n_bad = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, r in enumerate(rows):
            name = r["name"]
            mic = sf.read(args.eval_dir / "mic" / f"{name}.wav", dtype="float32")[0]
            far = sf.read(args.eval_dir / "farend" / f"{name}.wav", dtype="float32")[0]
            n = min(len(mic), len(far))
            mic, far = mic[:n], far[:n]
            res = aec.process(mic.astype(np.float64), far.astype(np.float64))
            y_echo = res.y_echo[:n].astype(np.float32)
            e_lin = (mic - y_echo).astype(np.float32)
            if i < 3:                       # 与预计算目录逐位核对(确定性 MDF)
                y2 = sf.read(args.lin_dir / "farend" / f"{name}.wav", dtype="float32")[0]
                e2 = sf.read(args.lin_dir / "mic" / f"{name}.wav", dtype="float32")[0]
                if len(y2) == len(y_echo):
                    d = max(np.abs(y2 - y_echo).max(), np.abs(e2 - e_lin).max())
                    ok = d <= 2.0 / 32768   # PCM_16 往返量化容差
                    n_bad += (not ok)
                    print(f"  [lin-check] {name} max|d|={d:.2e} {'OK' if ok else 'MISMATCH'}",
                          flush=True)
            enh = model.enhance(torch.from_numpy(e_lin).unsqueeze(0).to(args.device),
                                torch.from_numpy(y_echo).unsqueeze(0).to(args.device))
            enh = enh[0].cpu().numpy().astype(np.float32)
            sf.write(args.enh_dir / f"{name}.wav", enh, 16000, subtype="FLOAT")
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(rows)}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[cascade] {len(rows)} enh -> {args.enh_dir} "
          f"({time.perf_counter() - t0:.0f}s, lin-check mismatch={n_bad})", flush=True)


if __name__ == "__main__":
    main()
