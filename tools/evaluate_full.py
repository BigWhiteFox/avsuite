"""完整评估工具（WP-09 / 模块9）：checkpoint + 测试集 -> 分场景 CSV + 柱状图。

    python tools/evaluate_full.py --model crn --ckpt exp/03_crn/outputs/crn_crm_blstm/best.ckpt \
        --noisy-dir D:/data/voicebank-demand-16k/extracted/test/noisy \
        --clean-dir D:/data/voicebank-demand-16k/extracted/test/clean
    python tools/evaluate_full.py --model dccrn --ckpt exp/04_dccrn/outputs/dccrn/best.ckpt --synth 50
    python tools/evaluate_full.py --model nnaec --ckpt ... --eval-dir D:/data/aec-synth/eval_v3
    python tools/evaluate_full.py --model gtcrn --onnx exp/05a_gtcrn_onnx/outputs/gtcrn_dns3.sim.onnx --synth 50

指标：PESQ(wb)/STOI/SI-SNR（有参考）+ DNSMOS（无参考，--dnsmos 开启，需下载模型）。
分组：--group-by speaker（p226 前缀）/ scenario（AEC 场景 manifest）。
并行：指标计算多进程（--workers，Windows 需 __main__ 保护，测试里用 0）。
输出：outputs/eval_full/<模型名>/ 下 rows.csv + summary.json + charts.png。
"""
import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from core.evaluate import _safe_pesq, _safe_stoi, _si_snr_np  # noqa: E402

SR = 16000


# ---------------------------------------------------------------- 模型加载
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_model(args):
    """返回 (infer_fn, meta)，infer_fn(batch dict) -> enhanced wav [N] float32。

    统一成 dict 输入以兼容单路（noisy）与双路（mic+far）模型。
    """
    if args.model == "gtcrn":
        sys.path.insert(0, str(ROOT / "exp" / "05a_gtcrn_onnx"))
        common = _load_module("evfull_common", ROOT / "exp/05a_gtcrn_onnx/common.py")
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        import torch

        def infer(b):
            spec = common.stft_convention(torch.from_numpy(b["noisy"])).numpy()
            enh = sess.run(None, {"spec": spec})[0]
            e = torch.complex(torch.from_numpy(enh[0, ..., 0]),
                              torch.from_numpy(enh[0, ..., 1]))
            w = torch.hann_window(512).pow(0.5)
            return torch.istft(e, 512, 256, 512, w,
                               length=len(b["noisy"])).numpy().astype(np.float32)
        return infer, "gtcrn-onnx"
    # torch 系：crn / dccrn / nnaec
    import torch
    device = "cpu"
    if args.model == "crn":
        mod = _load_module("evfull_crn", ROOT / "exp/03_crn/eval.py")
        m = mod.load_model(Path(args.ckpt), device)
        return (lambda b: m.enhance_wav(torch.from_numpy(b["noisy"]).unsqueeze(0))[0]
                .numpy().astype(np.float32)), "crn"
    if args.model == "dccrn":
        sys.path.insert(0, str(ROOT / "exp/04_dccrn"))
        mod = _load_module("evfull_dccrn", ROOT / "exp/04_dccrn/eval.py")
        m = mod.load_model(Path(args.ckpt), device)
        return (lambda b: m.enhance_wav(torch.from_numpy(b["noisy"]).unsqueeze(0))[0]
                .numpy().astype(np.float32)), "dccrn"
    if args.model == "nnaec":
        sys.path.insert(0, str(ROOT / "exp/07_nn_aec"))
        mod = _load_module("evfull_nnaec", ROOT / "exp/07_nn_aec/eval.py")
        m = mod.load_model(Path(args.ckpt), device)
        return (lambda b: m.enhance(torch.from_numpy(b["mic"]).unsqueeze(0),
                                    torch.from_numpy(b["far"]).unsqueeze(0))[0]
                .numpy().astype(np.float32)), "nnaec"
    raise ValueError(f"unknown model {args.model}")


# ---------------------------------------------------------------- 数据装配
def collect_cases(args):
    """返回 [(name, group, noisy/mic, clean/near, far|None)]。"""
    if args.model == "nnaec":
        eval_dir = Path(args.eval_dir)
        rows = list(csv.DictReader(open(eval_dir / "manifest.csv", encoding="utf-8")))
        cases = []
        for r in rows:
            mic = sf.read(eval_dir / "mic" / f"{r['name']}.wav", dtype="float32")[0]
            far = sf.read(eval_dir / "farend" / f"{r['name']}.wav", dtype="float32")[0]
            near = sf.read(eval_dir / "nearend" / f"{r['name']}.wav", dtype="float32")[0]
            cases.append((r["name"], r["scenario"], mic, near, far))
        return cases
    if args.synth:
        sys.path.insert(0, str(ROOT / "exp" / "04_dccrn"))
        dm = _load_module("evfull_dm", ROOT / "exp/04_dccrn/dynamic_mixing.py")
        ds = dm.DNSDynamicMix(epoch_len=args.synth, seg_sec=6.0, seed=777)
        return [(f"synth_{i:04d}", "synth", item["noisy"].numpy(), item["clean"].numpy(),
                 None) for i, item in enumerate([ds[i] for i in range(args.synth)])]
    noisy_dir, clean_dir = Path(args.noisy_dir), Path(args.clean_dir)
    names = sorted({p.name for p in noisy_dir.glob("*.wav")} &
                   {p.name for p in clean_dir.glob("*.wav")})
    if args.max_files:
        names = names[: args.max_files]
    cases = []
    for n in names:
        noisy = sf.read(noisy_dir / n, dtype="float32")[0]
        clean = sf.read(clean_dir / n, dtype="float32")[0]
        group = n.split("_")[0] if args.group_by == "speaker" else "all"
        cases.append((n, group, noisy, clean, None))
    return cases


# ---------------------------------------------------------------- 指标计算
def _score_one(job):
    """worker 函数（多进程）：(name, group, enh, ref, dnsmos) -> 指标 dict。"""
    name, group, enh, ref = job
    n = min(len(enh), len(ref))
    enh, ref = enh[:n].astype(np.float32), ref[:n].astype(np.float32)
    row = {"name": name, "group": group,
           "pesq_wb": _safe_pesq(ref, enh), "stoi": _safe_stoi(ref, enh),
           "si_snr": _si_snr_np(ref, enh)}
    return row


def dnsmos_score_all(cases_enh, dnsmos):
    out = {}
    for name, _group, enh, _ in cases_enh:
        s = dnsmos.score(enh)
        if s:
            out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["crn", "dccrn", "nnaec", "gtcrn"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--noisy-dir", type=Path, default=None)
    ap.add_argument("--clean-dir", type=Path, default=None)
    ap.add_argument("--eval-dir", type=Path, default=Path("D:/data/aec-synth/eval_v3"))
    ap.add_argument("--synth", type=int, default=0, help="DNS 自混 N 条（seed777）")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--group-by", default="speaker", choices=["speaker", "all", "scenario"],
                    help="speaker=p226 前缀；scenario=AEC 场景（nnaec 默认）")
    ap.add_argument("--dnsmos", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    infer, tag = load_model(args)
    cases = collect_cases(args)
    print(f"[evaluate_full] model={tag}  cases={len(cases)}  workers={args.workers}",
          flush=True)

    # 1) 推理（单进程，模型加载一次）
    enhanced = []
    for i, (name, group, noisy, ref, far) in enumerate(cases):
        b = {"noisy": noisy} if far is None else {"mic": noisy, "far": far, "near": ref}
        enh = infer(b)
        enhanced.append((name, group, enh, ref))
        if (i + 1) % 50 == 0:
            print(f"  enhance {i+1}/{len(cases)}", flush=True)

    # 2) 指标（多进程；Windows spawn 需可 pickle 的模块级 worker）
    jobs = [(name, group, enh, ref) for name, group, enh, ref in enhanced]
    if args.workers > 1 and len(jobs) > 2:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            rows = pool.map(_score_one, jobs, chunksize=max(1, len(jobs) // args.workers // 4))
    else:
        rows = [_score_one(j) for j in jobs]

    # 3) DNSMOS（可选，无参考）
    dnsmos_res = {}
    if args.dnsmos:
        from dnsmos import DNSMOS
        dm = DNSMOS()
        if dm.available:
            dnsmos_res = dnsmos_score_all(enhanced, dm)
            for r in rows:
                if r["name"] in dnsmos_res:
                    r.update(dnsmos_res[r["name"]])
            print("[dnsmos] ok")
        else:
            print("[dnsmos] 模型下载失败（代理/网络），跳过该指标")

    # 4) 汇总 + 图表
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    import pandas as pd
    df = pd.DataFrame(rows)
    out = args.out or ROOT / "tools" / "outputs" / "eval_full" / f"{tag}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    summary = {}
    for g, sub in df.groupby("group"):
        summary[g] = {k: round(float(sub[k].mean()), 4)
                      for k in ("pesq_wb", "stoi", "si_snr") if not sub[k].isna().all()}
        if "ovr" in df:
            summary[g]["ovr"] = round(float(sub["ovr"].dropna().mean()), 4)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(max(4, 1.8 * len(summary)), 4.5))
    groups = list(summary)
    for i, k in enumerate(("pesq_wb", "stoi", "si_snr")):
        vals = [summary[g].get(k, 0) for g in groups]
        ax.bar(np.arange(len(groups)) + i * 0.25, vals, width=0.25,
               label=("PESQ" if k == "pesq_wb" else "STOI" if k == "stoi" else "SI-SNR"))
    ax.set_xticks(np.arange(len(groups)) + 0.25)
    ax.set_xticklabels(groups, rotation=15)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"evaluate_full: {tag}（{len(df)} 条）")
    fig.tight_layout()
    fig.savefig(out / "charts.png", dpi=150)
    print(json.dumps(summary, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
