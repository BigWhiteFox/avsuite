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

from ckpt_compat import load_ckpt  # noqa: E402 - 服务器 ckpt 含 PosixPath，须打桩加载
from model import NNAEC  # noqa: E402
from pipeline import TradAEC  # noqa: E402

# lin-check 容差：预计算目录现为 FLOAT 子类型（无 PCM_16 量化），同一确定性 MDF 下
# 实测 max|d|≈0。保留 2/32768 是**兼容旧 PCM_16 时代预计算目录**的宽松上界
# （V5 v3b-049：原注释把它当成"必经的 PCM_16 往返误差"，与当前写盘口径不符）。
LIN_CHECK_TOL = 2.0 / 32768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("D:/data/aec-synth/eval_v3"))
    ap.add_argument("--ckpt", type=Path,
                    default=HERE / "outputs" / "nnaec_g0.5_j10" / "best.ckpt")
    ap.add_argument("--enh-dir", type=Path, default=HERE / "results" / "enh")
    ap.add_argument("--lin-dir", type=Path, default=None,
                    help="预计算线性级目录（默认按 --eval-dir 派生：<eval-dir>_lin）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.lin_dir is None:
        # V5 v3b-045：默认从 --eval-dir 派生（旧实现写死 D:/data/aec-synth/eval_v3_lin，
        # 与 --eval-dir 完全脱钩 → 换数据集时静默核对错目录）
        args.lin_dir = args.eval_dir.parent / (args.eval_dir.name + "_lin")

    with open(args.eval_dir / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    args.enh_dir.mkdir(parents=True, exist_ok=True)

    aec = TradAEC(method="mdf", partitions=8, mu=0.7)   # 线性级:与 WP-06/08 同配置
    st = load_ckpt(args.ckpt, map_location=args.device)  # PosixPath 打桩在 ckpt_compat 内
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
                if len(y2) == len(y_echo) and len(e2) == len(e_lin):
                    d = max(np.abs(y2 - y_echo).max(), np.abs(e2 - e_lin).max())
                    # V5 v3b-049：预计算目录现由 precompute_cascade 以 FLOAT 子类型写盘
                    # （无 PCM_16 量化），同一确定性 MDF 实测 max|d|≈0；此容差是兼容
                    # 旧 PCM_16 时代预计算目录的宽松上界，不是"必经量化误差"。
                    ok = d <= LIN_CHECK_TOL
                    n_bad += (not ok)
                    print(f"  [lin-check] {name} max|d|={d:.2e} {'OK' if ok else 'MISMATCH'}",
                          flush=True)
                else:
                    # V5 v3b-043：长度不等时旧实现**静默跳过**核对却仍打印 mismatch=0
                    # （看起来全对）。现在显式计入 mismatch 并打印长度，不掩盖。
                    n_bad += 1
                    print(f"  [lin-check] {name} SKIPPED(长度不等: y {len(y2)}/{len(y_echo)}, "
                          f"e {len(e2)}/{len(e_lin)}) -> 计入 mismatch", flush=True)
            enh = model.enhance(torch.from_numpy(e_lin).unsqueeze(0).to(args.device),
                                torch.from_numpy(y_echo).unsqueeze(0).to(args.device))
            enh = enh[0].cpu().numpy().astype(np.float32)
            # V5 v3b-048：写盘前长度/有限性断言（与 zerotrain_baseline 同款守卫，
            # 避免把截断/NaN 的 enh 静默写进下游 aecmos_eval 的输入目录）
            assert len(enh) == n, f"{name}: enh len {len(enh)} != {n}"
            assert np.isfinite(enh).all(), f"{name}: enh 含 NaN/Inf"
            sf.write(args.enh_dir / f"{name}.wav", enh, 16000, subtype="FLOAT")
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(rows)}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[cascade] {len(rows)} enh -> {args.enh_dir} "
          f"({time.perf_counter() - t0:.0f}s, lin-check mismatch={n_bad})", flush=True)


if __name__ == "__main__":
    main()
