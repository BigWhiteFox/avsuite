#!/usr/bin/env python
"""S11（V4 整改第四波）: 级联 MDF→NN 上真实数据 —— 官方 ICASSP2022 集 800 段 + eval_v6_realrir 100 条。

线性级逐位复刻训练分布（= exp/09_cascade_aec/precompute_cascade.py 同款路径）:
    TradAEC(method="mdf", partitions=8, mu=0.7)   # margin/reestimate 不开（默认关，训练-推理一致）
    res = aec.process(mic.astype(np.float64), far.astype(np.float64))
    y_echo = res.y_echo[:n].astype(np.float32)    # NN 的 far 输入（对齐回声估计）
    e_lin  = (mic - y_echo).astype(np.float32)    # NN 的 mic 输入（MDF 误差,RES 之前）
发散防护（S1 默认开: MDF w_abs_cap=1e3 + 发散复位）保留；n_divergence_resets 逐条记录。
NN: outputs/nnaec_g0.5_j10/best.ckpt（加载方式同 run_cascade.py: torch.load → NNAEC.load_state_dict）。

子命令（全部支持断点续跑：wav 已存在(>1KB) 跳过；打分行已写入 jsonl 跳过）:
  lin-official   官方 800 段 48k→16k（load16 = resample_poly(1/3)，同 official_aecmos_eval.py）
                 → MDF → e_lin/y_echo 16k FLOAT → D:/data/aec-synth/official_stage/cascade_lin/{sub}/{mic,farend}/
  lin-v6         eval_v6 100 条 16k 原生 → MDF → D:/data/aec-synth/cascade_stage/eval_v6/lin_cascade/
  enh-official   级联 NN 前向 → D:/data/aec-synth/official_stage/enh_cascade/{sub}/   (16k FLOAT)
  enh-v6         级联 NN 前向 → D:/data/aec-synth/cascade_stage/eval_v6/enh_cascade/
  score-official 官方 proto 协议打分（复用 official_aecmos_eval.py 的 trim_for/CachedAECMOS）→
                 results/scores_cascade_official.jsonl（逐条,断点续）
  score-v6       aecmos_eval.py 口径（16k 全段 AECMOSEstimator.run,无 trim）→
                 results/scores_cascade_v6.jsonl
  agg-official / agg-v6   汇总+95%CI+配对 bootstrap（可单独重跑,不重打分）→
                 results/aecmos_cascade_official.json / results/aecmos_cascade_v6.json

用法（avsuite 环境,workdir=仓库根;>60s 的子命令后台跑）:
  python exp/09_cascade_aec/cascade_real_data.py lin-official --workers 7
  python exp/09_cascade_aec/cascade_real_data.py enh-official
  python exp/09_cascade_aec/cascade_real_data.py score-official --workers 6
  python exp/09_cascade_aec/cascade_real_data.py agg-official
  （v6 同理:*-v6;冒烟加 --limit N）
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent                  # exp/09_cascade_aec
AVROOT = HERE.parents[1]                                # avsuite/
sys.path.insert(0, str(HERE.parent / "06_trad_aec"))    # WP-06 pipeline
sys.path.insert(0, str(HERE.parent / "07_nn_aec"))      # model.py + official_aecmos_eval.py
sys.path.insert(0, str(AVROOT))                         # core 包

import official_aecmos_eval as OAE  # noqa: E402  (常量/load16/enumerate/trim_for/CachedAECMOS)
from pipeline import TradAEC  # noqa: E402

CKPT = HERE / "outputs" / "nnaec_g0.5_j10" / "best.ckpt"
STAGE = Path("D:/data/aec-synth/official_stage")
LIN_OFF = STAGE / "cascade_lin"                          # {sub}/{mic,farend}/{name}.wav
ENH_OFF = STAGE / "enh_cascade"                          # {sub}/{name}.wav
V6_ROOT = Path("D:/data/aec-synth/eval_v6_realrir")
V6_STAGE = Path("D:/data/aec-synth/cascade_stage/eval_v6")
LIN6 = V6_STAGE / "lin_cascade"                          # {mic,farend}/{name}.wav
ENH6 = V6_STAGE / "enh_cascade"                          # {name}.wav
RESULTS = HERE / "results"
SCORE_JSONL_OFF = RESULTS / "scores_cascade_official.jsonl"
SCORE_JSONL_V6 = RESULTS / "scores_cascade_v6.jsonl"
LIN_JSONL_OFF = LIN_OFF / "manifest_lin.jsonl"
LIN_JSONL_V6 = LIN6 / "manifest_lin.jsonl"
ENH_JSONL_OFF = STAGE / "manifest_cascade_enh.jsonl"
ENH_JSONL_V6 = V6_STAGE / "manifest_enh.jsonl"
OOD_JSONL = AVROOT / "exp" / "07_nn_aec" / "outputs" / "ood_real_echoes" / "scores_ood.jsonl"
TRAD_V6 = AVROOT / "exp" / "07_nn_aec" / "outputs" / "eval_v6_realrir" / "aecmos_trad_v6.json"
NN_V6 = AVROOT / "exp" / "07_nn_aec" / "outputs" / "eval_v6_realrir" / "aecmos_nn_v4.json"
SUBSETS = ("fest", "dt", "nst")
DIV_CLIP = "wufNfJIb3kC7OvuxutARLQ_farend-singletalk-with-movement"   # 官方集唯一发散条（trad 列）
Z95 = 1.96


# ---------------------------------------------------------------- 通用
def _load_jsonl(p: Path) -> list[dict]:
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, list):        # 兼容早期误写的批量行
                rows.extend(o for o in obj if isinstance(o, dict))
            elif isinstance(obj, dict):
                rows.append(obj)
    return rows


def mean_sd_ci(vals) -> tuple[int, float, float, float]:
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    mean = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    ci = Z95 * sd / math.sqrt(n) if n else 0.0
    return n, mean, sd, ci


def boot_ci(diffs, b: int = 2000, seed: int = 777) -> tuple[float, float, float]:
    """配对 bootstrap 95%CI（百分位法）——同 tools/compute_ci.py 约定（B=2000, seed=777）。"""
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    rng = random.Random(seed)
    boots = np.empty(b)
    for i in range(b):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        boots[i] = s / n
    boots.sort()
    lo, hi = float(boots[int(0.025 * b)]), float(boots[int(0.975 * b) - 1])
    return float(d.mean()), lo, hi


def enum_official(subsets, limit) -> list[dict]:
    clips = OAE.enumerate_clips(list(subsets) if subsets else None)
    if limit:
        by: dict[str, list] = {}
        for c in clips:
            by.setdefault(c["subset"], []).append(c)
        clips = sum([v[:limit] for v in by.values()], [])
    return clips


def enum_v6(limit, subsets) -> list[dict]:
    with open(V6_ROOT / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if subsets:
        rows = [r for r in rows if r["scenario"] in subsets]
    return rows[:limit] if limit else rows


def _run_pool(jobs, job_fn, init_fn, jsonl_path: Path, label: str, workers: int) -> None:
    """断点续跑驱动：jsonl 已有 ok 行跳过；新完成行追加 jsonl。"""
    done = set()
    for r in _load_jsonl(jsonl_path):
        if r.get("status") == "ok":
            done.add((r.get("subset") or r.get("scenario"), r["name"]))
    todo = [j for j in jobs if ((j["subset"] if "subset" in j else j["scenario"]),
                                j["name"]) not in done]
    print(f"[{label}] todo={len(todo)}/{len(jobs)} (jsonl 已有 {len(done)} 行) workers={workers}",
          flush=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init_fn) as ex:
        futs = {ex.submit(job_fn, j): j for j in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_ok += r.get("status") == "ok"
            if i % 25 == 0 or i == len(futs):
                print(f"[{label} {i}/{len(futs)}] ok={n_ok} {time.time() - t0:.0f}s", flush=True)
    print(f"[{label} done] {time.time() - t0:.0f}s ok={n_ok} -> {jsonl_path}", flush=True)


# ---------------------------------------------------------------- 线性级（MDF）
_LIN_AEC: TradAEC | None = None


def _lin_init():
    global _LIN_AEC
    # 训练同款配置；align_margin_ms/reestimate_sec 保持默认（不开），发散防护默认开
    _LIN_AEC = TradAEC(method="mdf", partitions=8, mu=0.7)


def _lin_run(mic: np.ndarray, far: np.ndarray, p_e: Path, p_y: Path) -> tuple[int, object]:
    """precompute_cascade.py 逐位同款: float64 全链 → y_echo/e_lin(float32) → FLOAT wav。"""
    n = min(len(mic), len(far))
    mic, far = mic[:n], far[:n]
    res = _LIN_AEC.process(mic.astype(np.float64), far.astype(np.float64))
    y_echo = res.y_echo[:n].astype(np.float32)
    e_lin = (mic - y_echo).astype(np.float32)
    p_e.parent.mkdir(parents=True, exist_ok=True)
    p_y.parent.mkdir(parents=True, exist_ok=True)
    sf.write(p_e, e_lin, 16000, subtype="FLOAT")
    sf.write(p_y, y_echo, 16000, subtype="FLOAT")
    return n, res


def lin_job_official(clip: dict) -> dict:
    sub, name = clip["subset"], clip["name"]
    p_e = LIN_OFF / sub / "mic" / f"{name}.wav"
    p_y = LIN_OFF / sub / "farend" / f"{name}.wav"
    t0 = time.time()
    try:
        if p_e.exists() and p_y.exists() and p_e.stat().st_size > 1024 and p_y.stat().st_size > 1024:
            return dict(subset=sub, name=name, status="cached", sec=0.0, error="")
        l16, m16 = OAE.load16(clip)          # 官方 48k→16k，同 official_aecmos_eval.py
        n, res = _lin_run(m16, l16, p_e, p_y)
        return dict(subset=sub, name=name, status="ok", sec=round(time.time() - t0, 2), n=int(n),
                    n_div=int(res.n_divergence_resets), n_realigns=int(res.n_realigns),
                    delay_ms=round(float(res.delay_ms), 2), align_ms=round(float(res.align_ms), 2),
                    evidence=round(float(res.evidence), 2), dt_frac=round(float(res.dt_frac), 4),
                    error="")
    except Exception as e:  # 单条失败记 manifest 继续
        return dict(subset=sub, name=name, status="failed", sec=round(time.time() - t0, 2),
                    error=f"{type(e).__name__}: {e}")


def lin_job_v6(r: dict) -> dict:
    name, sc = r["name"], r["scenario"]
    p_e = LIN6 / "mic" / f"{name}.wav"
    p_y = LIN6 / "farend" / f"{name}.wav"
    t0 = time.time()
    try:
        if p_e.exists() and p_y.exists() and p_e.stat().st_size > 1024 and p_y.stat().st_size > 1024:
            return dict(scenario=sc, name=name, status="cached", sec=0.0, error="")
        mic = sf.read(V6_ROOT / "mic" / f"{name}.wav", dtype="float32")[0]
        far = sf.read(V6_ROOT / "farend" / f"{name}.wav", dtype="float32")[0]
        n, res = _lin_run(mic, far, p_e, p_y)
        return dict(scenario=sc, name=name, status="ok", sec=round(time.time() - t0, 2), n=int(n),
                    n_div=int(res.n_divergence_resets), n_realigns=int(res.n_realigns),
                    delay_ms=round(float(res.delay_ms), 2), align_ms=round(float(res.align_ms), 2),
                    evidence=round(float(res.evidence), 2), dt_frac=round(float(res.dt_frac), 4),
                    error="")
    except Exception as e:
        return dict(scenario=sc, name=name, status="failed", sec=round(time.time() - t0, 2),
                    error=f"{type(e).__name__}: {e}")


def stage_lin_official(args) -> None:
    clips = enum_official(args.subsets, args.limit)
    _run_pool(clips, lin_job_official, _lin_init, LIN_JSONL_OFF, "lin-official", args.workers)


def stage_lin_v6(args) -> None:
    rows = enum_v6(args.limit, args.subsets)
    _run_pool(rows, lin_job_v6, _lin_init, LIN_JSONL_V6, "lin-v6", args.workers)


# ---------------------------------------------------------------- 级联 NN 前向
def _enh_run(items, lin_root_of, enh_root_of, jsonl_path: Path, key_field: str) -> None:
    import torch
    from model import NNAEC
    if not torch.cuda.is_available():
        raise SystemExit("[enh] CUDA 不可用 —— 按任务书立即停止报告（不静默退 CPU）")
    device = "cuda"
    torch.set_num_threads(min(16, os.cpu_count() or 16))
    st = torch.load(CKPT, map_location=device, weights_only=False)
    model = NNAEC().to(device).eval()
    model.load_state_dict(st["model"])
    print(f"[enh] device={device} ({torch.cuda.get_device_name(0)}) "
          f"ckpt={CKPT} val={st.get('val')}", flush=True)
    # V5 v3b-052：cached 也算"已完成"。旧实现只把 status=="ok" 计入 done →
    # 上一轮补记的 cached 行本轮不被承认 → 每轮都再 append 一条 cached 行，manifest 无限增长。
    done = {(r[key_field], r["name"]) for r in _load_jsonl(jsonl_path)
            if r.get("status") in ("ok", "cached") and r.get(key_field)}
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_new = n_skip = 0
    with torch.no_grad():
        for i, it in enumerate(items, 1):
            key, name = it[key_field], it["name"]
            dst = enh_root_of(key) / f"{name}.wav"
            if (key, name) in done or (dst.exists() and dst.stat().st_size > 1024):
                n_skip += 1
                if dst.exists() and (key, name) not in done:
                    # 补记 cached 行（断点续跑后 manifest 仍完整；finite/max_abs 从存量 wav 读回）
                    w = sf.read(dst, dtype="float32")[0]
                    row = {key_field: key, "name": name, "status": "cached", "sec": 0.0,
                           "n": int(len(w)), "finite": bool(np.isfinite(w).all()),
                           "max_abs": round(float(np.abs(w).max()), 4), "error": ""}
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            tc = time.time()
            p_e, p_y = lin_root_of(key) / "mic" / f"{name}.wav", lin_root_of(key) / "farend" / f"{name}.wav"
            e_lin = sf.read(p_e, dtype="float32")[0]
            y_echo = sf.read(p_y, dtype="float32")[0]
            n = min(len(e_lin), len(y_echo))
            enh = model.enhance(torch.from_numpy(e_lin[:n]).unsqueeze(0).to(device),
                                torch.from_numpy(y_echo[:n]).unsqueeze(0).to(device))
            enh = enh[0].cpu().numpy().astype(np.float32)
            assert len(enh) == n, f"{name}: enh {len(enh)} != e_lin {n}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(dst, enh, 16000, subtype="FLOAT")
            row = {key_field: key, "name": name, "status": "ok", "sec": round(time.time() - tc, 2),
                   "n": int(n), "finite": bool(np.isfinite(enh).all()),
                   "max_abs": round(float(np.abs(enh).max()), 4),
                   "error": ""}
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_new += 1
            if i % 25 == 0 or i == len(items):
                secs = [r["sec"] for r in _load_jsonl(jsonl_path)
                        if isinstance(r.get("sec"), (int, float)) and r["sec"] > 0]
                print(f"[enh {i}/{len(items)}] new={n_new} skip={n_skip} "
                      f"avg={np.mean(secs) if secs else 0:.2f}s/clip", flush=True)
    print(f"[enh done] new={n_new} skip={n_skip} {time.time() - t0:.0f}s -> {jsonl_path}", flush=True)


def stage_enh_official(args) -> None:
    clips = enum_official(args.subsets, args.limit)
    _enh_run(clips, lambda sub: LIN_OFF / sub, lambda sub: ENH_OFF / sub,
             ENH_JSONL_OFF, "subset")


def stage_enh_v6(args) -> None:
    rows = enum_v6(args.limit, args.subsets)
    _enh_run(rows, lambda sc: LIN6, lambda sc: ENH6, ENH_JSONL_V6, "scenario")


# ---------------------------------------------------------------- 打分
_SC = None
_EST = None


def _score_init_official():
    global _SC
    sys.path.insert(0, str(OAE.AECMOS_DIR))
    _SC = OAE.CachedAECMOS(str(OAE.AECMOS_MODEL), intra_threads=2)


def score_job_official(clip: dict) -> dict:
    """官方 proto 协议（official_aecmos_eval.score_job 同流程: librosa 48k → min_len →
    发散防护 → trim_for('proto') → CachedAECMOS.run）。"""
    import librosa
    sub, name = clip["subset"], clip["name"]

    def fail(err, **kw):
        return dict(variant="proto", method="cascade", subset=sub, name=name, status="failed",
                    echo_mos=None, deg_mos=None, n48=kw.get("n48", 0), trim=kw.get("trim", 0),
                    error=err)

    try:
        lpb, _ = librosa.load(clip["lpb"], sr=OAE.SR_AECMOS)
        mic, _ = librosa.load(clip["mic"], sr=OAE.SR_AECMOS)
        enh, _ = librosa.load(ENH_OFF / sub / f"{name}.wav", sr=OAE.SR_AECMOS)
    except Exception as e:
        return fail(f"load: {type(e).__name__}: {e}")
    n = min(len(lpb), len(mic), len(enh))
    if not np.all(np.isfinite(enh)) or float(np.max(np.abs(enh))) > 1e3:
        return fail("enh_diverged: 推理列输出含非有限/爆表样本", n48=n)
    base = [lpb[:n], mic[:n], enh[:n]]
    try:
        trimmed, k, _note = OAE.trim_for(sub, "proto", base)
        echo, deg = _SC.run(OAE.TALK[sub], trimmed[0], trimmed[1], trimmed[2])
        return dict(variant="proto", method="cascade", subset=sub, name=name, status="ok",
                    echo_mos=round(echo, 4), deg_mos=round(deg, 4), n48=int(len(trimmed[0])),
                    trim=int(k), error="")
    except Exception as e:
        return fail(f"score: {type(e).__name__}: {e}", n48=n)


def stage_score_official(args) -> None:
    clips = enum_official(args.subsets, args.limit)
    _run_pool(clips, score_job_official, _score_init_official, SCORE_JSONL_OFF,
              "score-official", min(args.workers, 6))


def _score_init_v6():
    global _EST
    sys.path.insert(0, str(OAE.AECMOS_DIR))
    from aecmos import AECMOSEstimator
    _EST = AECMOSEstimator(str(OAE.AECMOS_MODEL))


def score_job_v6(r: dict) -> dict:
    """aecmos_eval.py 口径逐字复刻：sf.read float32 → min_len → est.run(talk, lpb, mic, enh)
    （16k 全段直送、无 trim；AECMOSEstimator 内部 >20s 截前 20s）。"""
    name, sc = r["name"], r["scenario"]
    try:
        lpb = sf.read(V6_ROOT / "farend" / f"{name}.wav", dtype="float32")[0]
        mic = sf.read(V6_ROOT / "mic" / f"{name}.wav", dtype="float32")[0]
        enh = sf.read(ENH6 / f"{name}.wav", dtype="float32")[0]
        n = min(len(lpb), len(mic), len(enh))
        # V5 v3b-050：与 official 路径同款发散/爆表防护（见 score_job_official）。
        # 旧实现 v6 路径缺这道门 → 爆表的 enh 会被直接送评，得到无意义的 AECMOS 数字
        # 而不是显式的 failed 行。
        if not np.all(np.isfinite(enh)) or float(np.max(np.abs(enh))) > 1e3:
            return dict(variant="full", method="cascade", scenario=sc, name=name,
                        status="failed", echo_mos=None, deg_mos=None, n=int(n),
                        finite=False, error="enh_diverged: 推理列输出含非有限/爆表样本")
        echo, deg = _EST.run(OAE.TALK[sc], lpb[:n], mic[:n], enh[:n])
        return dict(variant="full", method="cascade", scenario=sc, name=name, status="ok",
                    echo_mos=float(echo), deg_mos=float(deg), n=int(n),
                    finite=bool(np.isfinite(enh).all()), error="")
    except Exception as e:
        return dict(variant="full", method="cascade", scenario=sc, name=name, status="failed",
                    echo_mos=None, deg_mos=None, n=0,
                    error=f"{type(e).__name__}: {e}")


def stage_score_v6(args) -> None:
    rows = enum_v6(args.limit, args.subsets)
    _run_pool(rows, score_job_v6, _score_init_v6, SCORE_JSONL_V6, "score-v6", min(args.workers, 6))


# ---------------------------------------------------------------- 汇总
def _dedupe(rows: list[dict], key2: str) -> dict:
    out = {}
    for r in rows:
        if r.get("status", "ok") == "ok" and r.get("echo_mos") is not None:
            out[(r[key2], r["name"])] = r
    return out


def _paired(cas: dict, other: dict, scs) -> dict:
    """配对差值 = 级联 − other，按 (subset, name) 对齐；bootstrap 95%CI。"""
    out = {}
    for tag, rows in other.items():
        per = {}
        for sc in scs:
            for metric in ("echo_mos", "deg_mos"):
                diffs = [cas[k][metric] - rows[k][metric]
                         for k in cas if k in rows and k[0] == sc
                         and cas[k].get(metric) is not None and rows[k].get(metric) is not None]
                if diffs:
                    d, lo, hi = boot_ci(diffs)
                    per.setdefault(sc, {})[metric] = dict(
                        n=len(diffs), delta=round(d, 4), lo=round(lo, 4), hi=round(hi, 4))
        out[tag] = per
    return out


def _lin_div_stats(lin_rows: list[dict], key_field: str) -> dict:
    divs = [r for r in lin_rows if r.get("status") == "ok" and r.get("n_div", 0) > 0]
    return dict(n_clips_with_resets=len(divs), total_resets=sum(r["n_div"] for r in divs),
                clips=[{key_field: r[key_field], "name": r["name"], "n_div": r["n_div"],
                        "n_realigns": r.get("n_realigns", 0)} for r in divs])


def _dedupe_lin(rows: list[dict], key2: str) -> dict:
    """lin manifest 去重（无 echo_mos 字段；status 默认 ok）。"""
    out = {}
    for r in rows:
        if r.get("status", "ok") == "ok" and "n_div" in r:
            out[(r[key2], r["name"])] = r
    return out


def agg_official() -> None:
    score_rows_raw = _load_jsonl(SCORE_JSONL_OFF)
    rows = _dedupe(score_rows_raw, "subset")
    lin = _dedupe_lin(_load_jsonl(LIN_JSONL_OFF), "subset")
    ood = _load_jsonl(OOD_JSONL)
    other = {"trad": _dedupe([r for r in ood if r.get("method") == "trad"], "subset"),
             "nn": _dedupe([r for r in ood if r.get("method") == "nn"], "subset")}
    summary, paired = {}, {}
    for sc in SUBSETS:
        keys = [k for k in rows if k[0] == sc]
        if not keys:
            continue
        n, em, esd, eci = mean_sd_ci([rows[k]["echo_mos"] for k in keys])
        _, dm, dsd, dci = mean_sd_ci([rows[k]["deg_mos"] for k in keys])
        # V5 v3b-051：失败统计按**最新行**口径（旧实现从原始行 score_rows_raw 统计 →
        # 上一轮失败、本轮已成功的 clip 仍永久计为 failed，failed 数只增不减）。
        latest = {}
        for r in score_rows_raw:
            if r.get("subset") == sc and r.get("name"):
                latest[r["name"]] = r
        failed = [r for r in latest.values() if r.get("status") != "ok"]
        summary[sc] = dict(n=n, echo_mos=round(em, 3), echo_sd=round(esd, 3),
                           echo_ci95=round(eci, 3), deg_mos=round(dm, 3), deg_sd=round(dsd, 3),
                           deg_ci95=round(dci, 3),
                           failed=len(failed),
                           failed_clips=[dict(name=r["name"], error=r.get("error", ""))
                                         for r in failed])
        lin_ok = [k for k in lin if k[0] == sc]
        summary[sc]["n_lin"] = len(lin_ok)
        summary[sc]["n_div_clips"] = sum(1 for k in lin_ok if lin[k].get("n_div", 0) > 0)
    paired = _paired(rows, other, SUBSETS)
    div = _lin_div_stats(list(lin.values()), "subset")
    # 发散条：级联打分行可能是 failed（enh 爆表被拒），单独取原始行
    wuf_score = next((r for r in score_rows_raw
                      if r.get("subset") == "fest" and r.get("name") == DIV_CLIP), None)
    wuf_lin = lin.get(("fest", DIV_CLIP), {})
    wuf_nn = other["nn"].get(("fest", DIV_CLIP), {})
    wuf = dict(name=DIV_CLIP, subset="fest",
               cascade_status=(wuf_score or {}).get("status"),
               cascade_error=(wuf_score or {}).get("error"),
               cascade_n48=(wuf_score or {}).get("n48"),
               cascade_echo_mos=(wuf_score or {}).get("echo_mos"),
               cascade_deg_mos=(wuf_score or {}).get("deg_mos"),
               lin_n_div=wuf_lin.get("n_div"), lin_delay_ms=wuf_lin.get("delay_ms"),
               lin_evidence=wuf_lin.get("evidence"),
               trad="failed（ood_real_echoes: enh_diverged 拒送评,173,816 非有限样本）",
               nn=dict(echo_mos=wuf_nn.get("echo_mos"), deg_mos=wuf_nn.get("deg_mos")))
    out = dict(tag="cascade_official_proto",
               pipeline="TradAEC(mdf,8,0.7)→(e_lin,y_echo)→NNAEC g0.5_j10 best.ckpt（发散防护开,margin/reestimate 关）",
               ckpt=str(CKPT), scoring="官方 proto 协议（trim_for + CachedAECMOS,48k）",
               summary=summary, paired=paired, divergence=div, divergence_clip=wuf,
               rows=[dict(name=r["name"], scenario=k[0], echo_mos=r["echo_mos"], deg_mos=r["deg_mos"],
                          n48=r.get("n48"), trim=r.get("trim"),
                          n_div=lin.get(k, {}).get("n_div")) for k, r in sorted(rows.items())])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "aecmos_cascade_official.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "paired": paired, "divergence": div,
                      "divergence_clip": wuf}, ensure_ascii=False, indent=1))
    print(f"-> {RESULTS / 'aecmos_cascade_official.json'}", flush=True)


def agg_v6() -> None:
    rows = _dedupe(_load_jsonl(SCORE_JSONL_V6), "scenario")
    lin = _dedupe_lin(_load_jsonl(LIN_JSONL_V6), "scenario")

    def load_ref(p: Path) -> dict:
        if not p.exists():
            return {}
        return _dedupe(json.loads(p.read_text(encoding="utf-8")).get("rows", []), "scenario")

    other = {"trad": load_ref(TRAD_V6), "nn": load_ref(NN_V6)}
    scs = ("fest", "dt", "nst")
    summary, paired = {}, {}
    for sc in scs:
        keys = [k for k in rows if k[0] == sc]
        if not keys:
            continue
        n, em, esd, eci = mean_sd_ci([rows[k]["echo_mos"] for k in keys])
        _, dm, dsd, dci = mean_sd_ci([rows[k]["deg_mos"] for k in keys])
        summary[sc] = dict(n=n, echo_mos=round(em, 3), echo_sd=round(esd, 3), echo_ci95=round(eci, 3),
                           deg_mos=round(dm, 3), deg_sd=round(dsd, 3), deg_ci95=round(dci, 3))
    paired = _paired(rows, other, scs)
    div = _lin_div_stats(list(lin.values()), "scenario")
    out = dict(tag="cascade_v6_full",
               pipeline="TradAEC(mdf,8,0.7)→(e_lin,y_echo)→NNAEC g0.5_j10 best.ckpt（发散防护开,margin/reestimate 关）",
               ckpt=str(CKPT), scoring="aecmos_eval.py 口径（16k 全段 AECMOSEstimator,无 trim）",
               summary=summary, paired=paired, divergence=div,
               rows=[dict(name=r["name"], scenario=k[0], echo_mos=r["echo_mos"], deg_mos=r["deg_mos"],
                          n_div=lin.get(k, {}).get("n_div")) for k, r in sorted(rows.items())])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "aecmos_cascade_v6.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "paired": paired, "divergence": div},
                     ensure_ascii=False, indent=1))
    print(f"-> {RESULTS / 'aecmos_cascade_v6.json'}", flush=True)


# ---------------------------------------------------------------- 入口
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=[
        "lin-official", "lin-v6", "enh-official", "enh-v6",
        "score-official", "score-v6", "agg-official", "agg-v6", "env"])
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None, help="每子集/总数取前 N 条（冒烟）")
    ap.add_argument("--subsets", nargs="+", default=None, choices=list(SUBSETS))
    args = ap.parse_args()

    if args.stage == "env":
        import shutil

        import torch
        print("cuda:", torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
        print("disk D free GB:", round(shutil.disk_usage("D:/").free / 2**30, 1))
        print("ckpt exists:", CKPT.exists())
    elif args.stage == "lin-official":
        stage_lin_official(args)
    elif args.stage == "lin-v6":
        stage_lin_v6(args)
    elif args.stage == "enh-official":
        stage_enh_official(args)
    elif args.stage == "enh-v6":
        stage_enh_v6(args)
    elif args.stage == "score-official":
        stage_score_official(args)
    elif args.stage == "score-v6":
        stage_score_v6(args)
    elif args.stage == "agg-official":
        agg_official()
    elif args.stage == "agg-v6":
        agg_v6()


if __name__ == "__main__":
    main()
