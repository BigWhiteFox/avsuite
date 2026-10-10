"""传统 AEC 验收评估（WP-06）：分场景 ERLE / 近端 PESQ / STOI。

DoD 对应（WORKLIST WP-06）：
  - FEST（远端单讲）稳态 ERLE ≥ 20 dB —— 只统计 far-end 活跃段、跳过收敛瞬态
  - DT（双讲）近端无明显损伤 —— PESQ(out vs nearend) 与 mic-vs-nearend 基线对比
  - NST（近端单讲）透传 —— PESQ/STOI 不劣化

两种数据模式：
  --synth-dir   自合成受控集（aec_synth.py 产物，manifest 带场景标签）
  --official    AEC-Challenge 官方 synthetic 集（四目录 + meta.csv，
                场景按 nearend 能量占比自动判定；默认取 test split 前 N 条）

用法：
  python eval_aec.py --synth-dir D:/data/aec-synth/eval_v1 --method mdf
  python eval_aec.py --official --num 40 --method mdf
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))               # 本目录模块
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # repo root (core)

from pipeline import TradAEC  # noqa: E402

SR = 16000
SKIP_S = 2.0          # ERLE 统计跳过收敛瞬态（4096-tap 级滤波器在语音上需 ~2s）


def erle_db(mic, out, skip_s=SKIP_S, sr=SR):
    """稳态 ERLE：10log10(P_mic/P_out)，跳过前 skip_s 秒。"""
    k = int(skip_s * sr)
    return 10 * np.log10((np.sum(mic[k:] ** 2) + 1e-12) / (np.sum(out[k:] ** 2) + 1e-12))


def pesq_safe(ref, deg):
    from core.evaluate import _safe_pesq, _safe_stoi
    n = min(len(ref), len(deg))
    return _safe_pesq(ref[:n].astype(np.float32), deg[:n].astype(np.float32)), \
        _safe_stoi(ref[:n].astype(np.float32), deg[:n].astype(np.float32))


def run_one(aec, mic, far):
    r = aec.process(mic, far)
    return r


def eval_synth_dir(aec, synth_dir, out_csv):
    """自合成受控集：manifest 场景标签直接分组。"""
    synth_dir = Path(synth_dir)
    rows = list(csv.DictReader(open(synth_dir / "manifest.csv", encoding="utf-8")))
    results = []
    for r in rows:
        name, sc = r["name"], r["scenario"]
        mic, _ = sf.read(synth_dir / "mic" / f"{name}.wav", dtype="float32")
        far, _ = sf.read(synth_dir / "farend" / f"{name}.wav", dtype="float32")
        near, _ = sf.read(synth_dir / "nearend" / f"{name}.wav", dtype="float32")
        res = run_one(aec, mic.astype(np.float64), far.astype(np.float64))
        row = {"name": name, "scenario": sc}
        if sc == "fest":
            row["erle_db"] = erle_db(mic, res.out)
        else:  # dt / nst：近端保真
            p_out, s_out = pesq_safe(near, res.out)
            p_mic, s_mic = pesq_safe(near, mic)     # 基线：不做 AEC 的近端可懂度
            row.update(pesq_out=p_out, stoi_out=s_out, pesq_mic=p_mic, stoi_mic=s_mic)
            row["erle_db"] = erle_db(mic, res.out)  # DT 段 ERLE 参考（含近端，仅趋势）
        results.append(row)
    return _summarize(results, out_csv)


def eval_official(aec, num, out_csv):
    """AEC-Challenge synthetic：fileid 关联四信号，nearend 能量占比判场景。"""
    root = Path("D:/data/repos/AEC-Challenge/datasets/synthetic")
    meta = list(csv.DictReader(open(root / "meta.csv", encoding="utf-8")))
    meta = [m for m in meta if m["split"] == "test"][:num]
    results = []
    for m in meta:
        fid = m["fileid"]
        mic, _ = sf.read(root / "nearend_mic_signal" / f"nearend_mic_fileid_{fid}.wav",
                         dtype="float32")
        far, _ = sf.read(root / "farend_speech" / f"farend_speech_fileid_{fid}.wav",
                         dtype="float32")
        near, _ = sf.read(root / "nearend_speech" / f"nearend_speech_fileid_{fid}.wav",
                          dtype="float32")
        n = min(len(mic), len(far), len(near))
        mic, far, near = mic[:n], far[:n], near[:n] * float(m["nearend_scale"])
        p_near = np.mean(near ** 2) + 1e-12
        p_mic = np.mean(mic ** 2) + 1e-12
        scenario = "fest" if 10 * np.log10(p_near / p_mic) < -25 else "dt"
        res = run_one(aec, mic.astype(np.float64), far.astype(np.float64))
        row = {"name": f"fileid_{fid}", "scenario": scenario, "ser_db": m["ser"],
               "nonlinear": m["is_farend_nonlinear"]}
        if scenario == "fest":
            row["erle_db"] = erle_db(mic, res.out)
        else:
            p_out, s_out = pesq_safe(near, res.out)
            p_mic_, s_mic = pesq_safe(near, mic)
            row.update(pesq_out=p_out, stoi_out=s_out, pesq_mic=p_mic_, stoi_mic=s_mic)
        results.append(row)
    return _summarize(results, out_csv)


def _summarize(results, out_csv):
    import pandas as pd
    df = pd.DataFrame(results)
    if out_csv:
        df.to_csv(out_csv, index=False)
    summary = {}
    fest = df[df.scenario == "fest"]
    if len(fest):
        summary["fest"] = {"n": len(fest), "erle_db_mean": round(fest.erle_db.mean(), 2),
                           "erle_db_p10": round(fest.erle_db.quantile(0.1), 2)}
        if "soft_clip" in df.columns:      # 线性/非线性切片（非线性是线性 AEC 的物理上限）
            lin = fest[fest.soft_clip.astype(int) == 0]
            nl = fest[fest.soft_clip.astype(int) == 1]
            if len(lin):
                summary["fest_linear_only"] = {"n": len(lin),
                                               "erle_db_mean": round(lin.erle_db.mean(), 2)}
            if len(nl):
                summary["fest_nonlinear"] = {"n": len(nl),
                                             "erle_db_mean": round(nl.erle_db.mean(), 2)}
    for sc in ("dt", "nst"):
        sub = df[df.scenario == sc]
        if len(sub) and "pesq_out" in sub:
            summary[sc] = {"n": len(sub),
                           "pesq_out": round(sub.pesq_out.mean(), 3),
                           "pesq_mic_baseline": round(sub.pesq_mic.mean(), 3),
                           "stoi_out": round(sub.stoi_out.mean(), 3)}
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=Path, default=None)
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--num", type=int, default=40, help="official 模式条数")
    ap.add_argument("--method", default="mdf", choices=["mdf", "nlms"])
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--partitions", type=int, default=8,
                    help="MDF 分块数（滤波器长 = partitions×256 taps；≥ RIR 截断长度，车厢 100ms→8）")
    ap.add_argument("--no-dtd", action="store_true")
    ap.add_argument("--no-res", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    aec = TradAEC(method=args.method, mu=args.mu, partitions=args.partitions,
                  use_dtd=not args.no_dtd, use_res=not args.no_res)
    tag = f"{args.method}_mu{args.mu}_p{args.partitions}" \
        + ("_nodtd" if args.no_dtd else "") + ("_nores" if args.no_res else "")
    out_csv = args.out or Path(__file__).parent / "outputs" / f"eval_{tag}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.official:
        df, summary = eval_official(aec, args.num, out_csv)
    elif args.synth_dir:
        df, summary = eval_synth_dir(aec, args.synth_dir, out_csv)
    else:
        ap.error("需要 --synth-dir 或 --official")

    print(f"[eval_aec:{tag}] {len(df)} 条 -> {out_csv}")
    for sc, s in summary.items():
        print(f"  {sc.upper():5s} {s}")
    if "fest" in summary:
        ok = summary["fest"]["erle_db_mean"] >= 20
        print(f"  DoD FEST ERLE>=20dB: {'PASS' if ok else 'FAIL'} "
              f"({summary['fest']['erle_db_mean']}dB)")
    if "dt" in summary:
        d = summary["dt"]
        print(f"  DT 近端: PESQ out {d['pesq_out']} vs mic基线 {d['pesq_mic_baseline']}")


if __name__ == "__main__":
    main()
