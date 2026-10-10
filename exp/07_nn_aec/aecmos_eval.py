"""AECMOS 官方口径评估（WP-07 DoD：双讲 echo MOS 较传统基线 +0.3）。

用 AEC-Challenge 仓库 AECMOS_local 的 ONNX 模型（本地推理，无 Web API）。
场景映射：fest->st / dt->dt / nst->nst（官方 talk_type 命名）。

    python aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3 --nn-ckpt outputs/nnaec_g0.5_j10/best.ckpt
    python aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3 --trad      # WP-06 基线
    python aecmos_eval.py --eval-dir ... --enh-dir <已增强 wav 目录>

口径说明（README 纪律）：官方对 ICASSP 长测试集要求裁剪送评（FEST 后半段 /
DT 最后 (len-15)/2 秒）；我们的 eval_v3 是 8s 短片段（2s DT lead-in 为滤波器
收敛设计），全段送评——与官方绝对值有口径差，**对比结论（NN vs 传统同集同口径）
不受影响**。返回 (echo_mos, deg_mos)。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))                       # repo root
sys.path.insert(0, str(HERE.parents[0] / "06_trad_aec"))       # WP-06
AECMOS_DIR = Path("D:/data/repos/AEC-Challenge/AECMOS/AECMOS_local")
AECMOS_MODEL = AECMOS_DIR / "Run_1668423760_Stage_0.onnx"      # 最新版（v2 有单讲 bug）
TALK = {"fest": "st", "dt": "dt", "nst": "nst"}


def score_all(pairs, est):
    """pairs: [(name, scenario, lpb, mic, enh)] -> per-scenario 汇总。"""
    rows = []
    for name, sc, lpb, mic, enh in pairs:
        echo_mos, deg_mos = est.run(TALK[sc], lpb, mic, enh)
        rows.append(dict(name=name, scenario=sc, echo_mos=echo_mos, deg_mos=deg_mos))
    summary = {}
    for sc in ("fest", "dt", "nst"):
        sub = [r for r in rows if r["scenario"] == sc]
        if sub:
            summary[sc] = {
                "n": len(sub),
                "echo_mos": round(float(np.mean([r["echo_mos"] for r in sub])), 3),
                "deg_mos": round(float(np.mean([r["deg_mos"] for r in sub])), 3),
            }
    return rows, summary


def load_clips(eval_dir):
    eval_dir = Path(eval_dir)
    rows = list(csv.DictReader(open(eval_dir / "manifest.csv", encoding="utf-8")))
    clips = []
    for r in rows:
        lpb, _ = sf.read(eval_dir / "farend" / f"{r['name']}.wav", dtype="float32")
        mic, _ = sf.read(eval_dir / "mic" / f"{r['name']}.wav", dtype="float32")
        clips.append((r["name"], r["scenario"], lpb, mic))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", type=Path, default=Path("D:/data/aec-synth/eval_v3"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--nn-ckpt", type=Path, default=None)
    g.add_argument("--trad", action="store_true")
    g.add_argument("--enh-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(AECMOS_DIR))
    from aecmos import AECMOSEstimator
    est = AECMOSEstimator(str(AECMOS_MODEL))
    clips = load_clips(args.eval_dir)

    if args.enh_dir:
        pairs = []
        for name, sc, lpb, mic in clips:
            enh, _ = sf.read(args.enh_dir / f"{name}.wav", dtype="float32")
            n = min(len(lpb), len(mic), len(enh))
            pairs.append((name, sc, lpb[:n], mic[:n], enh[:n]))
        tag = "enhdir"
    elif args.trad:
        from pipeline import TradAEC
        aec = TradAEC(method="mdf", partitions=8, mu=0.7)
        pairs = []
        for name, sc, lpb, mic in clips:
            out = aec.process(mic.astype(np.float64), lpb.astype(np.float64)).out
            pairs.append((name, sc, lpb, mic, out.astype(np.float32)))
        tag = "trad_wp06"
    else:
        import torch
        from model import NNAEC
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st = torch.load(args.nn_ckpt, map_location=device, weights_only=False)
        model = NNAEC().to(device).eval()
        model.load_state_dict(st["model"])
        pairs = []
        with torch.no_grad():
            for name, sc, lpb, mic in clips:
                enh = model.enhance(torch.from_numpy(mic).unsqueeze(0).to(device),
                                    torch.from_numpy(lpb).unsqueeze(0).to(device))
                pairs.append((name, sc, lpb, mic, enh[0].cpu().numpy()))
        tag = "nn_wp07"

    rows, summary = score_all(pairs, est)
    out = args.out or HERE / "outputs" / f"aecmos_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": tag, "summary": summary, "rows": rows}, indent=2))
    print(f"[aecmos:{tag}]")
    for sc, s in summary.items():
        print(f"  {sc.upper():5s} n={s['n']:3d}  echo_mos {s['echo_mos']:.3f}  "
              f"deg_mos {s['deg_mos']:.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
