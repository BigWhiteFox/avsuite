"""非线性 AEC 评估 v2（WP-08 验收）：线性级 vs 线性+GRU 残差级 的 ERLE 对比。

    python eval.py --ckpt outputs/nl_aec_v2/best.ckpt --eval-dir D:/data/aec-synth/eval_nl
验收（WORKLIST）：合成非线性数据上较线性 NLMS/MDF +5dB。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import NonlinearAEC  # noqa: E402

SR = 16000
SKIP_S = 0.5


def erle_db(mic, out, skip_s=SKIP_S):
    k = int(skip_s * SR)
    return 10 * np.log10((np.sum(mic[k:] ** 2) + 1e-12) / (np.sum(out[k:] ** 2) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, default=Path("D:/data/aec-synth/eval_nl"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = NonlinearAEC().eval()
    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model"])

    rows = [r for r in csv.DictReader(
        open(args.eval_dir / "manifest.csv", encoding="utf-8")) if r["scenario"] == "fest"]
    results = []
    with torch.no_grad():
        for r in rows:
            name = r["name"]
            mic = sf.read(args.eval_dir / "mic" / f"{name}.wav", dtype="float32")[0]
            lin = sf.read(args.eval_dir / "lin_est" / f"{name}.wav", dtype="float32")[0]
            n = min(len(mic), len(lin))
            mic, lin = mic[:n], lin[:n]
            res = model(torch.from_numpy(lin).unsqueeze(0)).squeeze(0).numpy()
            lin_out = (mic - lin).astype(np.float32)
            casc_out = (mic - lin - res).astype(np.float32)
            results.append({"name": name, "erle_linear": erle_db(mic, lin_out),
                            "erle_cascade": erle_db(mic, casc_out)})
    mean_lin = float(np.mean([r["erle_linear"] for r in results]))
    mean_cas = float(np.mean([r["erle_cascade"] for r in results]))
    summary = {"n": len(results), "erle_linear_mean": round(mean_lin, 2),
               "erle_cascade_mean": round(mean_cas, 2),
               "delta": round(mean_cas - mean_lin, 2),
               "doD_plus5db": bool(mean_cas - mean_lin >= 5.0)}
    out = args.out or Path(__file__).parent / "outputs" / "eval_nl_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"DoD（级联较纯线性 +5dB）：{'✅ PASS' if summary['doD_plus5db'] else '⚠️ 未达标'}")

    # DT 近端保真参考（级联对近端的损伤应≤线性级）
    from core.evaluate import _safe_pesq
    dt = [r for r in csv.DictReader(
        open(args.eval_dir / "manifest.csv", encoding="utf-8")) if r["scenario"] == "dt"][:5]
    pesq_lin, pesq_cas = [], []
    with torch.no_grad():
        for r in dt:
            name = r["name"]
            mic = sf.read(args.eval_dir / "mic" / f"{name}.wav", dtype="float32")[0]
            lin = sf.read(args.eval_dir / "lin_est" / f"{name}.wav", dtype="float32")[0]
            near = sf.read(args.eval_dir / "nearend" / f"{name}.wav", dtype="float32")[0]
            n = min(len(mic), len(lin), len(near))
            mic, lin, near = mic[:n], lin[:n], near[:n]
            res = model(torch.from_numpy(lin).unsqueeze(0)).squeeze(0).numpy()
            pesq_lin.append(_safe_pesq(near, (mic - lin).astype(np.float32)))
            pesq_cas.append(_safe_pesq(near, (mic - lin - res).astype(np.float32)))
    print(f"DT 近端 PESQ（前5条）：线性 {np.mean(pesq_lin):.3f} → 级联 {np.mean(pesq_cas):.3f}")


if __name__ == "__main__":
    main()
