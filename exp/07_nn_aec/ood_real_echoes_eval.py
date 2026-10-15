"""WS-D1: 真实录制回声 OOD 评测（传统 AEC vs NN-AEC v4），量化仿真→真实 domain gap。

数据：AEC-Challenge 官方 ICASSP2022 真实测试集（本机已物化的唯一真实录制集；
datasets/real 与 real_doubled 在本机均为 git-lfs 指针文件，无音频内容——如实降级，
不顶替）。片段为真实设备/真实环境录制，48kHz，lpb=loopback 远端参考，mic=真实
录制麦克风；含真实双讲。**无干净近端参考 → 只算 AECMOS（官方口径）与 FEST 段
ERLE，不算 PESQ/STOI**。

口径：
- AECMOS：官方协议裁剪（FEST 送后半段 / DT 送最后 (len-15)/2 s / NST 全段），
  官方 Run_1668423760_Stage_0.onnx，librosa 48k 加载，同 official_aecmos_eval.py；
- ERLE：仅 FEST，官方评级段（后半段）上 10log10(Σmic²/Σenh²)，16k；
  注意 NUMBERS.md 口径注：NN 的 cRM 门控可使该 ERLE 虚高，只说明"FEST 段输出≈静音"。
- CI：95%CI = 1.96·sd/√n。

用法（avsuite 环境）：
  python ood_real_echoes_eval.py --stage infer-trad --workers 7
  python ood_real_echoes_eval.py --stage infer-nn
  python ood_real_echoes_eval.py --stage score --methods trad nn
  python ood_real_echoes_eval.py --stage agg
断点续跑：enh wav 已存在(>1KB)即跳过；打分行已在 jsonl 即跳过。
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "06_trad_aec"))

from official_aecmos_eval import (  # noqa: E402
    AECMOS_DIR,
    AECMOS_MODEL,
    SR_PROC,
    TALK,
    CachedAECMOS,
    enumerate_clips,
    load16,
    trim_for,
)

OUT = HERE / "outputs" / "ood_real_echoes"          # 新目录，不覆盖既有产物
STAGE = OUT / "stage"
JSONL = OUT / "scores_ood.jsonl"
CKPT = HERE / "outputs_v4" / "nnaec_g0.5_j10" / "best.ckpt"   # 任务指定 best.ckpt
SUBSETS = {"fest": "farend-singletalk", "dt": "doubletalk", "nst": "nearend-singletalk"}


# ---------------------------------------------------------------- 传统列推理
def trad_job(clip):
    t0 = time.time()
    dst = STAGE / "enh_trad" / clip["subset"] / f"{clip['name']}.wav"
    try:
        if dst.exists() and dst.stat().st_size > 1024:
            return dict(subset=clip["subset"], name=clip["name"], status="ok",
                        sec=0.0, error="cached")
        from pipeline import TradAEC
        l16, m16 = load16(clip)
        aec = TradAEC(method="mdf", partitions=8, mu=0.7)
        out = aec.process(m16.astype(np.float64), l16.astype(np.float64)).out
        dst.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf
        sf.write(dst, out.astype(np.float32), SR_PROC, subtype="FLOAT")
        return dict(subset=clip["subset"], name=clip["name"], status="ok",
                    sec=round(time.time() - t0, 2), error="")
    except Exception as e:
        return dict(subset=clip["subset"], name=clip["name"], status="failed",
                    sec=round(time.time() - t0, 2), error=f"{type(e).__name__}: {e}")


def stage_infer_trad(clips, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    man = STAGE / "manifest_trad.csv"
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(trad_job, c): c for c in clips}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            if i % 50 == 0 or i == len(clips):
                ok = sum(x["status"] == "ok" for x in rows)
                print(f"[trad {i}/{len(clips)}] ok={ok} fail={i - ok}", flush=True)
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "name", "status", "sec", "error"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["subset"], r["name"])))
    print(f"[trad done] {time.time() - t0:.0f}s -> {man}", flush=True)


# ---------------------------------------------------------------- NN 列推理
def stage_infer_nn(clips, ckpt):
    import soundfile as sf
    import torch
    torch.set_num_threads(min(16, os.cpu_count() or 16))
    from model import NNAEC
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(ckpt, map_location=device, weights_only=False)
    model = NNAEC().to(device).eval()
    model.load_state_dict(st["model"])
    print(f"[nn] device={device} ckpt={Path(ckpt).name}", flush=True)
    man = STAGE / "manifest_nn.csv"
    rows, t0 = [], time.time()
    with torch.no_grad():
        for i, clip in enumerate(clips, 1):
            tc = time.time()
            dst = STAGE / "enh_nn" / clip["subset"] / f"{clip['name']}.wav"
            try:
                if dst.exists() and dst.stat().st_size > 1024:
                    rows.append(dict(subset=clip["subset"], name=clip["name"],
                                     status="ok", sec=0.0, error="cached"))
                    continue
                l16, m16 = load16(clip)
                enh = model.enhance(torch.from_numpy(m16).unsqueeze(0).to(device),
                                    torch.from_numpy(l16).unsqueeze(0).to(device))
                enh = enh[0].cpu().numpy()
                assert len(enh) == len(m16)
                assert np.all(np.isfinite(enh)), "non-finite output"
                dst.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dst, enh.astype(np.float32), SR_PROC, subtype="FLOAT")
                rows.append(dict(subset=clip["subset"], name=clip["name"], status="ok",
                                 sec=round(time.time() - tc, 2), error=""))
            except Exception as e:
                rows.append(dict(subset=clip["subset"], name=clip["name"], status="failed",
                                 sec=round(time.time() - tc, 2),
                                 error=f"{type(e).__name__}: {e}"))
            if i % 50 == 0 or i == len(clips):
                ok = sum(x["status"] == "ok" for x in rows)
                print(f"[nn {i}/{len(clips)}] ok={ok} fail={i - ok}", flush=True)
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "name", "status", "sec", "error"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["subset"], r["name"])))
    print(f"[nn done] {time.time() - t0:.0f}s -> {man}", flush=True)


# ---------------------------------------------------------------- 打分
def erle_db(sig, ref):
    return 10 * float(np.log10((np.sum(ref ** 2) + 1e-12) / (np.sum(sig ** 2) + 1e-12)))


def score_job(clip, methods):
    """AECMOS（官方 proto 裁剪，48k librosa）+ FEST ERLE（官方评级段，16k）。"""
    import librosa
    import soundfile as sf
    rows = []
    try:
        lpb, _ = librosa.load(clip["lpb"], sr=48000)
        mic, _ = librosa.load(clip["mic"], sr=48000)
    except Exception as e:
        for m in methods:
            rows.append(dict(method=m, subset=clip["subset"], name=clip["name"],
                             status="failed", echo_mos=None, deg_mos=None, erle=None,
                             error=f"load lpb/mic: {type(e).__name__}: {e}"))
        return rows
    for m in methods:
        enhp = STAGE / ("enh_trad" if m == "trad" else "enh_nn") \
            / clip["subset"] / f"{clip['name']}.wav"
        try:
            enh, _ = librosa.load(enhp, sr=48000)
        except Exception as e:
            rows.append(dict(method=m, subset=clip["subset"], name=clip["name"],
                             status="failed", echo_mos=None, deg_mos=None, erle=None,
                             error=f"load enh: {type(e).__name__}: {e}"))
            continue
        n = min(len(lpb), len(mic), len(enh))
        if not np.all(np.isfinite(enh)) or float(np.max(np.abs(enh))) > 1e3:
            rows.append(dict(method=m, subset=clip["subset"], name=clip["name"],
                             status="failed", echo_mos=None, deg_mos=None, erle=None,
                             error="enh_diverged"))
            continue
        base = [lpb[:n], mic[:n], enh[:n]]
        try:
            trimmed, k, _note = trim_for(clip["subset"], "proto", base)
            a, b, c = trimmed
            echo, deg = _SCORER.run(TALK[clip["subset"]], a, b, c)
        except Exception as e:
            rows.append(dict(method=m, subset=clip["subset"], name=clip["name"],
                             status="failed", echo_mos=None, deg_mos=None, erle=None,
                             error=f"score: {type(e).__name__}: {e}"))
            continue
        erle = None
        if clip["subset"] == "fest":
            try:  # ERLE：官方评级段（后半段），16k
                l16, m16 = load16(clip)
                e16, _ = sf.read(enhp, dtype="float32")
                nn = min(len(l16), len(m16), len(e16))
                tri, _, _ = trim_for("fest", "proto", [m16[:nn], e16[:nn]])
                erle = round(erle_db(tri[1], tri[0]), 2)
            except Exception as e:
                erle = None
                rows[-1:] = [dict(method=m, subset=clip["subset"], name=clip["name"],
                                  status="ok", echo_mos=round(echo, 4),
                                  deg_mos=round(deg, 4), erle=None,
                                  error=f"erle: {type(e).__name__}: {e}")]
                continue
        rows.append(dict(method=m, subset=clip["subset"], name=clip["name"],
                         status="ok", echo_mos=round(echo, 4), deg_mos=round(deg, 4),
                         erle=erle, error=""))
    return rows


_SCORER = None


def _score_init():
    global _SCORER
    sys.path.insert(0, str(AECMOS_DIR))
    _SCORER = CachedAECMOS(AECMOS_MODEL, intra_threads=2)


def stage_score(clips, methods, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if JSONL.exists():
        with open(JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        done.add((r["method"], r["subset"], r["name"]))
                except json.JSONDecodeError:
                    pass
    todo = [c for c in clips
            if any((m, c["subset"], c["name"]) not in done for m in methods)]
    print(f"[score] methods={methods} todo={len(todo)}/{len(clips)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_score_init) as ex:
        futs = {ex.submit(score_job, c, methods): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            with open(JSONL, "a", encoding="utf-8") as f:
                for r in fut.result():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(futs):
                print(f"[score {i}/{len(futs)}] {time.time() - t0:.0f}s", flush=True)
    print(f"[score done] {time.time() - t0:.0f}s -> {JSONL}", flush=True)


# ---------------------------------------------------------------- 聚合
def stage_agg(methods):
    rows = []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    summary = {}
    for m in methods:
        for sc in SUBSETS:
            sub = [r for r in rows if r["method"] == m and r["subset"] == sc
                   and r["status"] == "ok"]
            if not sub:
                continue
            entry = dict(n=len(sub))
            for key in ("echo_mos", "deg_mos"):
                vals = [r[key] for r in sub if r.get(key) is not None]
                if vals:
                    mean = float(np.mean(vals))
                    ci = 1.96 * float(np.std(vals, ddof=0)) / np.sqrt(len(vals))
                    entry[key] = round(mean, 3)
                    entry[key + "_ci95"] = round(ci, 3)
                    entry[key + "_sd"] = round(float(np.std(vals, ddof=0)), 3)
            ev = [r["erle"] for r in sub if r.get("erle") is not None]
            if ev:
                entry["erle_db"] = round(float(np.mean(ev)), 2)
                entry["erle_ci95"] = round(
                    1.96 * float(np.std(ev, ddof=0)) / np.sqrt(len(ev)), 2)
                entry["erle_med_db"] = round(float(np.median(ev)), 2)
            fails = sum(1 for r in rows if r["method"] == m and r["subset"] == sc
                        and r["status"] != "ok")
            if fails:
                entry["failed"] = fails
            summary[f"{m}:{sc}"] = entry
    (OUT / "summary_ood.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT / "scores_ood.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "subset", "name", "status",
                                          "echo_mos", "deg_mos", "erle", "error"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["method"], r["subset"], r["name"])))
    for m in methods:
        print(f"[agg {m}]")
        for sc in SUBSETS:
            s = summary.get(f"{m}:{sc}")
            if s:
                print(f"  {sc.upper():4s} n={s['n']} echo={s.get('echo_mos')}±{s.get('echo_mos_ci95')}"
                      f" deg={s.get('deg_mos')}±{s.get('deg_mos_ci95')}"
                      f" erle={s.get('erle_db')}" + (f" fail={s['failed']}" if s.get("failed") else ""))
    print(f"-> {OUT / 'summary_ood.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["enumerate", "infer-trad", "infer-nn", "score", "agg"])
    ap.add_argument("--methods", nargs="+", default=["trad", "nn"], choices=["trad", "nn"])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    clips = enumerate_clips()
    if args.limit:
        by = {}
        for c in clips:
            by.setdefault(c["subset"], []).append(c)
        clips = sum([v[: args.limit] for v in by.values()], [])

    if args.stage == "enumerate":
        from collections import Counter
        print("clips:", dict(Counter(c["subset"] for c in clips)), "total:", len(clips))
    elif args.stage == "infer-trad":
        stage_infer_trad(clips, args.workers)
    elif args.stage == "infer-nn":
        stage_infer_nn(clips, args.ckpt)
    elif args.stage == "score":
        stage_score(clips, args.methods, min(args.workers, 6))
    elif args.stage == "agg":
        stage_agg(args.methods)


if __name__ == "__main__":
    main()
