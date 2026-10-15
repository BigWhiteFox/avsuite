"""P1-2(整改执行):官方 AEC-Challenge ICASSP2022 测试集 + 官方 AECMOS 协议评估。

官方协议注记(来源 D:/data/repos/AEC-Challenge/AECMOS/README.md 与
AECMOS_local/README.md,两处一致;打分实现样例见上层 aecmos.py 的
process_interspeech2021;官方原话:测试集被加长以让模型收敛,只送真正要评级
的段):
  1. farend-singletalk(FEST):只送后半段(最后 len/2 秒);
  2. doubletalk(DT):只送最后 (len_s-15)/2 秒(int 样本数,rating_dt_length<=0
     才不裁;本集 37-39s → 约 11-12s,含滤波收敛后的双讲评级段);
  3. nearend-singletalk(NST):全段送评;
  4. 三路(lpb/mic/enh)按同一长度同样裁剪;ECHOMOS/DEGMOS 是**同一次模型前向**
     的两个输出——官方脚本没有对 DEGMOS 单独再切「近端激活后的后半段」。
  5. 官方模型 Run_1668423760_Stage_0.onnx 按 README 是 48kHz 版本 1;官方脚本对
     lpb/mic/enh 一律 librosa.load(sr=48000) 后取 min_len,再送
     AECMOSEstimator.run(talk_type, ...) → (echo_mos, deg_mos);>20s 的输入官方
     estimator 内部截**前** 20s。本测试集原生 48k,与该模型同源配对。

送评版本(variants):
  - proto: 上面的官方协议裁剪(主表口径,可与赛题榜单直接比);
  - full : t=0 全段(>20s 时官方 estimator 截前 20s——官方行为,如实保留对照);
  - t2s  : 去前 2s(本仓 P1-3 对自合成 eval_v3 的 lead-in 口径;官方集无此要求,
           仅作对照列)。

处理侧口径(如实记录,不补造):官方集 48k → scipy.resample_poly(1/3) 降到 16k
送两列处理器(传统 WP-06 管道 = aecmos_eval.py --trad 同配置 mdf/8/0.7;NN v4 =
outputs_v4/nnaec_g0.5_j10/last.ckpt,加载方式同 capstone/cascade.py:load_nnaec),
增强 wav 以 16k FLOAT 落盘;送评时按官方脚本 librosa.load(sr=48000) 重采样回 48k。

用法(avsuite 环境):
  python official_aecmos_eval.py --stage infer-trad --workers 7
  python official_aecmos_eval.py --stage infer-nn
  python official_aecmos_eval.py --stage score --variants proto
  python official_aecmos_eval.py --stage score --variants full t2s
  python official_aecmos_eval.py --stage agg
  可加 --limit N 冒烟 / --subsets dt fest nst / --methods trad nn
断点续跑:增强 wav 已存在(>1KB)即跳过;打分行已写入 jsonl 即跳过。
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
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
AVROOT = HERE.parents[1]                                   # avsuite/
sys.path.insert(0, str(HERE))                              # model.py
sys.path.insert(0, str(HERE.parents[0] / "06_trad_aec"))   # WP-06 pipeline

TEST_ROOT = Path("D:/data/repos/AEC-Challenge/datasets/test_set_icassp2022")
AECMOS_DIR = Path("D:/data/repos/AEC-Challenge/AECMOS/AECMOS_local")
AECMOS_MODEL = AECMOS_DIR / "Run_1668423760_Stage_0.onnx"  # 官方 48kHz 版本 1
CKPT = HERE / "outputs_v4" / "nnaec_g0.5_j10" / "last.ckpt"  # NN v4

STAGE = Path("D:/data/aec-synth/official_stage")           # 中间产物目录
REPORTS = AVROOT / "reports" / "official_aecmos"           # 打分产物目录
JSONL = REPORTS / "scores_official.jsonl"

SUBSETS = {"fest": "farend-singletalk", "dt": "doubletalk", "nst": "nearend-singletalk"}
TALK = {"fest": "st", "dt": "dt", "nst": "nst"}            # 官方 talk_type 命名
SR_PROC = 16000                                            # 处理侧采样率
SR_AECMOS = 48000                                          # 官方模型采样率


# ---------------------------------------------------------------- 数据组织
def enumerate_clips(subsets=None):
    """官方三子集 → (subset, name=官方 hash 前 22 位, lpb, mic)。"""
    clips = []
    for sc, dirname in SUBSETS.items():
        if subsets and sc not in subsets:
            continue
        wavs = sorted((TEST_ROOT / dirname).glob("*.wav"))
        lpbs = {p.name[:-8]: p for p in wavs if p.name.endswith("_lpb.wav")}
        mics = {p.name[:-8]: p for p in wavs if p.name.endswith("_mic.wav")}
        assert set(lpbs) == set(mics), f"{dirname}: lpb/mic 不成对"
        for h in sorted(lpbs):
            clips.append(dict(subset=sc, name=h, lpb=str(lpbs[h]), mic=str(mics[h])))
    return clips


def load16(clip):
    """官方 48k → resample_poly(1/3) → 16k,两路取 min_len。"""
    lpb48, _ = sf.read(clip["lpb"], dtype="float32")
    mic48, _ = sf.read(clip["mic"], dtype="float32")
    l16 = resample_poly(lpb48, 1, 3)
    m16 = resample_poly(mic48, 1, 3)
    n = min(len(l16), len(m16))
    return l16[:n], m16[:n]


# ---------------------------------------------------------------- 传统列推理
def trad_job(clip):
    """WP-06 传统 AEC(与 aecmos_eval.py --trad 分支同配置)。"""
    t0 = time.time()
    dst = STAGE / "enh_trad" / clip["subset"] / f"{clip['name']}.wav"
    try:
        if dst.exists() and dst.stat().st_size > 1024:
            return dict(subset=clip["subset"], name=clip["name"], status="ok",
                        sec=0.0, out=str(dst), error="cached")
        from pipeline import TradAEC
        l16, m16 = load16(clip)
        aec = TradAEC(method="mdf", partitions=8, mu=0.7)
        out = aec.process(m16.astype(np.float64), l16.astype(np.float64)).out
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, out.astype(np.float32), SR_PROC, subtype="FLOAT")
        return dict(subset=clip["subset"], name=clip["name"], status="ok",
                    sec=round(time.time() - t0, 2), out=str(dst), error="")
    except Exception as e:  # 单条失败记 manifest 继续
        return dict(subset=clip["subset"], name=clip["name"], status="failed",
                    sec=round(time.time() - t0, 2), out="", error=f"{type(e).__name__}: {e}")


def _write_manifest(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["subset", "name", "status", "sec", "out", "error"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["subset"], r["name"])))


def stage_infer_trad(clips, workers):
    man = STAGE / "manifest_trad.csv"
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(trad_job, c): c for c in clips}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            if i % 25 == 0 or i == len(clips):
                ok = sum(x["status"] == "ok" for x in rows)
                print(f"[trad {i}/{len(clips)}] ok={ok} fail={i - ok} "
                      f"avg={np.mean([x['sec'] for x in rows if x['status'] == 'ok']):.2f}s/clip",
                      flush=True)
    _write_manifest(rows, man)
    fails = [r for r in rows if r["status"] == "failed"]
    print(f"[trad done] {time.time() - t0:.0f}s ok={len(rows) - len(fails)} "
          f"fail={len(fails)} -> {man}", flush=True)
    return rows


# ---------------------------------------------------------------- NN 列推理
def stage_infer_nn(clips, ckpt, vram_floor_gib=1.5):
    import torch
    torch.set_num_threads(min(16, os.cpu_count() or 16))
    sys.path.insert(0, str(HERE))
    from model import NNAEC
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(ckpt, map_location=device, weights_only=False)
    model = NNAEC().to(device).eval()
    model.load_state_dict(st["model"])
    print(f"[nn] device={device} torch={torch.__version__} ckpt={Path(ckpt).name}", flush=True)

    man = STAGE / "manifest_nn.csv"
    rows, note = [], []
    t0 = time.time()
    # V5 v3b-019（high，读码实测）：当前推理设备必须**跨 clip 持久化**。旧实现每轮
    # `dev = device` 重置回 cuda，而 VRAM 守卫/OOM 已把 model 永久搬到 cpu →
    # 下一轮把输入送 cuda、权重在 cpu，整列 nn 打分全失败。降级后不再回 cuda。
    dev = device
    with torch.no_grad():
        for i, clip in enumerate(clips, 1):
            tc = time.time()
            dst = STAGE / "enh_nn" / clip["subset"] / f"{clip['name']}.wav"
            try:
                if dst.exists() and dst.stat().st_size > 1024:
                    rows.append(dict(subset=clip["subset"], name=clip["name"], status="ok",
                                     sec=0.0, out=str(dst), error="cached"))
                    continue
                l16, m16 = load16(clip)
                if dev == "cuda":
                    free, _ = torch.cuda.mem_get_info(0)
                    if free < vram_floor_gib * 2**30:   # 显存紧张:整列退 CPU
                        model = model.to("cpu")
                        dev = "cpu"
                        torch.cuda.empty_cache()
                        note.append(f"clip#{i} VRAM<{vram_floor_gib}GiB -> CPU")
                mic_t = torch.from_numpy(m16).unsqueeze(0).to(dev)
                far_t = torch.from_numpy(l16).unsqueeze(0).to(dev)
                try:
                    enh = model.enhance(mic_t, far_t)[0].cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    model = model.to("cpu")
                    dev = "cpu"
                    torch.cuda.empty_cache()
                    note.append(f"clip#{i} OOM -> CPU")
                    enh = model.enhance(mic_t.cpu(), far_t.cpu())[0].cpu().numpy()
                assert len(enh) == len(m16), f"{clip['name']}: enh {len(enh)} != mic {len(m16)}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dst, enh.astype(np.float32), SR_PROC, subtype="FLOAT")
                rows.append(dict(subset=clip["subset"], name=clip["name"], status="ok",
                                 sec=round(time.time() - tc, 2), out=str(dst), error=""))
            except Exception as e:
                rows.append(dict(subset=clip["subset"], name=clip["name"], status="failed",
                                 sec=round(time.time() - tc, 2), out="",
                                 error=f"{type(e).__name__}: {e}"))
            if i % 25 == 0 or i == len(clips):
                ok = sum(x["status"] == "ok" for x in rows)
                done = [x["sec"] for x in rows if x["status"] == "ok" and x["sec"] > 0]
                print(f"[nn {i}/{len(clips)}] ok={ok} fail={i - ok} "
                      f"avg={np.mean(done) if done else 0:.2f}s/clip dev={dev}", flush=True)
    _write_manifest(rows, man)
    fails = [r for r in rows if r["status"] == "failed"]
    print(f"[nn done] {time.time() - t0:.0f}s ok={len(rows) - len(fails)} fail={len(fails)} "
          f"final_dev={dev} -> {man}", flush=True)
    if note:
        print("[nn notes] " + "; ".join(note[:5]) + (f" (+{len(note) - 5} more)" if len(note) > 5 else ""),
              flush=True)
    return rows


# ---------------------------------------------------------------- 送评打分
class CachedAECMOS:
    """官方 AECMOSEstimator(AECMOS_local/aecmos.py),仅把 per-call
    InferenceSession 换成缓存 session;run() 逻辑逐行同官方(含 20s 截断、
    mel 变换、scenario marker),h0 用 np.zeros(与官方 torch.zeros 同值)。"""

    def __init__(self, model_path, intra_threads=2):
        import onnxruntime as ort
        from aecmos import AECMOSEstimator
        self.est = AECMOSEstimator(str(model_path))
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), so,
                                         providers=["CPUExecutionProvider"])

    def run(self, talk_type, lpb_sig, mic_sig, enh_sig):
        est = self.est
        assert len(lpb_sig) == len(mic_sig) == len(enh_sig)
        seg = est.max_len * est.sampling_rate
        if len(lpb_sig) >= seg:                        # 官方: 截前 20s
            lpb_sig, mic_sig, enh_sig = lpb_sig[:seg], mic_sig[:seg], enh_sig[:seg]
        lpb_sig = est.transform(lpb_sig, est.sampling_rate)
        mic_sig = est.transform(mic_sig, est.sampling_rate)
        enh_sig = est.transform(enh_sig, est.sampling_rate)
        if est.need_scenario_marker:                   # 官方 scenario marker
            assert talk_type in ("nst", "st", "dt")
            ne_st = 1 if talk_type == "nst" else 0
            fe_st = 1 if talk_type == "st" else 0
            mic_sig = np.concatenate((mic_sig, np.ones((20, mic_sig.shape[1])) * (1 - fe_st),
                                      np.zeros((20, mic_sig.shape[1]))), axis=0)
            lpb_sig = np.concatenate((lpb_sig, np.ones((20, lpb_sig.shape[1])) * (1 - ne_st),
                                      np.zeros((20, lpb_sig.shape[1]))), axis=0)
            enh_sig = np.concatenate((enh_sig, np.ones((20, enh_sig.shape[1])),
                                      np.zeros((20, enh_sig.shape[1]))), axis=0)
        feats = np.stack((lpb_sig, mic_sig, enh_sig)).astype(np.float32)
        feats = np.expand_dims(feats, axis=0)
        h0 = np.zeros(est.hidden_size, dtype=np.float32)
        result = self.sess.run([], {self.sess.get_inputs()[0].name: feats, "h0": h0})[0]
        return float(result[0]), float(result[1])


def trim_for(subset, variant, sigs):
    """送评裁剪。输入三路等长(48k 样本);返回 (裁后三路, 裁掉样本数, 口径说明)。"""
    n = len(sigs[0])
    if variant == "full":
        return sigs, 0, "t=0 全段"
    if variant == "t2s":
        k = min(2 * SR_AECMOS, n // 2)
        return [a[k:] for a in sigs], k, "去前 2s(仓内 P1-3 lead-in 口径,官方集无此要求)"
    if variant == "proto":
        if subset == "nst":
            return sigs, 0, "官方: NST 全段"
        if subset == "fest":
            k = n // 2                                 # 官方: last half
            note = "官方: FEST 送后半段"
        else:                                          # dt
            k = int((n - 15 * SR_AECMOS) / 2)          # 官方: (len_s-15)/2
            note = "官方: DT 送最后 (len_s-15)/2 s"
            if k <= 0:
                return sigs, 0, note + "(rating<=0 不裁)"
        return [a[-k:] for a in sigs], n - k, note
    raise ValueError(variant)


_SCORER = None


def _score_init():
    global _SCORER
    sys.path.insert(0, str(AECMOS_DIR))
    _SCORER = CachedAECMOS(AECMOS_MODEL, intra_threads=2)


def score_job(clip, variants, methods):
    """一条 clip × 两列方法 × 各 variant;官方 read_and_process 流程
    (三路 librosa.load sr=48000 → min_len → 裁剪 → estimator.run)。"""
    import librosa
    rows = []
    try:
        lpb, _ = librosa.load(clip["lpb"], sr=SR_AECMOS)
        mic, _ = librosa.load(clip["mic"], sr=SR_AECMOS)
    except Exception as e:
        for m in methods:
            for v in variants:
                rows.append(dict(variant=v, method=m, subset=clip["subset"], name=clip["name"],
                                 status="failed", echo_mos=None, deg_mos=None, n48=0,
                                 trim=0, error=f"load lpb/mic: {type(e).__name__}: {e}"))
        return rows
    for m in methods:
        enhp = STAGE / ("enh_trad" if m == "trad" else "enh_nn") / clip["subset"] / f"{clip['name']}.wav"
        try:
            enh, _ = librosa.load(enhp, sr=SR_AECMOS)
        except Exception as e:
            for v in variants:
                rows.append(dict(variant=v, method=m, subset=clip["subset"], name=clip["name"],
                                 status="failed", echo_mos=None, deg_mos=None, n48=0,
                                 trim=0, error=f"load enh: {type(e).__name__}: {e}"))
            continue
        n = min(len(lpb), len(mic), len(enh))          # 官方 min_len
        if not np.all(np.isfinite(enh)) or float(np.max(np.abs(enh))) > 1e3:
            for v in variants:                         # 发散防护:脏 enh 不送评
                rows.append(dict(variant=v, method=m, subset=clip["subset"], name=clip["name"],
                                 status="failed", echo_mos=None, deg_mos=None, n48=n,
                                 trim=0, error="enh_diverged: 推理列输出含非有限/爆表样本"))
            continue
        base = [lpb[:n], mic[:n], enh[:n]]
        for v in variants:
            try:
                trimmed, k, _note = trim_for(clip["subset"], v, base)
                a, b, c = trimmed
            except Exception as e:
                rows.append(dict(variant=v, method=m, subset=clip["subset"], name=clip["name"],
                                 status="failed", echo_mos=None, deg_mos=None, n48=n,
                                 trim=0, error=f"trim: {type(e).__name__}: {e}"))
                continue
            try:
                echo, deg = _SCORER.run(TALK[clip["subset"]], a, b, c)
                rows.append(dict(variant=v, method=m, subset=clip["subset"], name=clip["name"],
                                 status="ok", echo_mos=round(echo, 4), deg_mos=round(deg, 4),
                                 n48=int(len(a)), trim=int(k), error=""))
            except Exception as e:
                rows.append(dict(variant=v, method=m, subset=clip["subset"], name=clip["name"],
                                 status="failed", echo_mos=None, deg_mos=None, n48=len(a),
                                 trim=int(k), error=f"score: {type(e).__name__}: {e}"))
    return rows


def stage_score(clips, variants, methods, workers):
    REPORTS.mkdir(parents=True, exist_ok=True)
    done = set()
    if JSONL.exists():
        with open(JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        done.add((r["variant"], r["method"], r["subset"], r["name"]))
                except json.JSONDecodeError:
                    pass
    todo = [c for c in clips
            if any((v, m, c["subset"], c["name"]) not in done for v in variants for m in methods)]
    print(f"[score] variants={variants} methods={methods} todo={len(todo)}/{len(clips)} "
          f"(已有 {len(done)} 行)", flush=True)
    t0 = time.time()
    nrow = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_score_init) as ex:
        futs = {ex.submit(score_job, c, variants, methods): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rows = fut.result()
            with open(JSONL, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            nrow += len(rows)
            if i % 25 == 0 or i == len(futs):
                print(f"[score {i}/{len(futs)}] rows={nrow} {time.time() - t0:.0f}s", flush=True)
    print(f"[score done] {time.time() - t0:.0f}s rows={nrow} -> {JSONL}", flush=True)


# ---------------------------------------------------------------- 聚合
def stage_agg(variants, methods):
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    summary = {}
    for v in variants:
        for m in methods:
            for sc in SUBSETS:
                sub = [r for r in rows if r["variant"] == v and r["method"] == m
                       and r["subset"] == sc and r["status"] == "ok"]
                if not sub:
                    continue
                summary[f"{v}:{m}:{sc}"] = dict(
                    n=len(sub),
                    echo_mos=round(float(np.mean([r["echo_mos"] for r in sub])), 3),
                    deg_mos=round(float(np.mean([r["deg_mos"] for r in sub])), 3))
                fails = sum(1 for r in rows if r["variant"] == v and r["method"] == m
                            and r["subset"] == sc and r["status"] != "ok")
                if fails:
                    summary[f"{v}:{m}:{sc}"]["failed"] = fails
    out = REPORTS / "summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 逐行 csv(全 variants)
    with open(REPORTS / "scores_official.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "method", "subset", "name",
                                          "status", "echo_mos", "deg_mos", "n48", "trim", "error"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["variant"], r["method"], r["subset"], r["name"])))
    for v in variants:
        print(f"[agg {v}]")
        for m in methods:
            for sc in SUBSETS:
                s = summary.get(f"{v}:{m}:{sc}")
                if s:
                    print(f"  {m:4s} {sc.upper():4s} n={s['n']:3d} echo={s['echo_mos']:.3f} "
                          f"deg={s['deg_mos']:.3f}" + (f" fail={s['failed']}" if s.get("failed") else ""))
    print(f"-> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["enumerate", "infer-trad", "infer-nn", "score", "agg"])
    ap.add_argument("--methods", nargs="+", default=["trad", "nn"], choices=["trad", "nn"])
    ap.add_argument("--variants", nargs="+", default=["proto"], choices=["proto", "full", "t2s"])
    ap.add_argument("--subsets", nargs="+", default=None, choices=list(SUBSETS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    args = ap.parse_args()

    clips = enumerate_clips(args.subsets)
    if args.limit:
        # 每子集取前 limit 条,保持三子集都有覆盖
        by = {}
        for c in clips:
            by.setdefault(c["subset"], []).append(c)
        clips = sum([v[: args.limit] for v in by.values()], [])

    if args.stage == "enumerate":
        from collections import Counter
        cnt = Counter(c["subset"] for c in clips)
        print("clips:", dict(cnt), "total:", len(clips))
        for c in clips[:3]:
            print(" ", c)
    elif args.stage == "infer-trad":
        STAGE.mkdir(parents=True, exist_ok=True)
        stage_infer_trad(clips, args.workers)
    elif args.stage == "infer-nn":
        STAGE.mkdir(parents=True, exist_ok=True)
        stage_infer_nn(clips, args.ckpt)
    elif args.stage == "score":
        stage_score(clips, args.variants, args.methods, min(args.workers, 6))
    elif args.stage == "agg":
        stage_agg(args.variants, args.methods)


if __name__ == "__main__":
    main()
