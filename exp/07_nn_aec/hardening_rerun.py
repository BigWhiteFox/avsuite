"""S10 加固版传统管道重跑驱动（V4 整改第四波，2026-10-11）。

用 S1 加固配置 TradAEC(method='mdf', partitions=8, mu=0.7, align_margin_ms=25,
reestimate_sec=2.0)（其余参数取默认：realign_min_delta_ms=5 / reestimate_win_s=2 /
realign_search_pad_ms=150 / mdf_w_abs_cap=1e3）重跑三处受污染 headline：

  drift          ① 漂移集 D:/data/aec-synth/eval_v3_drift（60 条）：处理→增强 wav 落
                 outputs/hardening_rerun/enh_trad_hard_drift/ → AECMOS 打分
                 （aecmos_eval.py --enh-dir 同口径：16k 信号直送 48k 模型）→
                 outputs/hardening_rerun/aecmos_trad_hard_drift.json
  official-one   ② 官方集发散条 wufNfJIb3kC7OvuxutARLQ_farend-singletalk-with-movement：
                 单条处理 + 全有限断言 + n_divergence_resets 记录 →
                 outputs/hardening_rerun/divergence_clip_hard.json
  official-infer ②' 官方 FEST 全 300 条：增强 wav 落 D:/data/aec-synth/official_stage/
                 enh_trad_hard/fest/（新目录，不动旧 enh_trad）+ meta jsonl
  official-score ②'' 官方 proto 协议打分（复用 official_aecmos_eval.trim_for +
                 CachedAECMOS；FEST 送后半段）→ scores_trad_hard_official_fest.jsonl
  official-agg   ②''' 聚合（总 300 / 同名 299 配对 vs 旧列）→
                 aecmos_trad_hard_official_fest.json
  v6             ③ D:/data/aec-synth/eval_v6_realrir（100 条）：AECMOS +
                 eval_aec 口径 ERLE（skip 2s）/ PESQ → aecmos_trad_hard_v6.json +
                 trad_hard_rows_v6.csv（列同旧 trad_rows_v6.csv）

断点续：处理侧 dst wav 已存在(>1KB)即跳过；meta / 打分行 jsonl 追加写，行级去重。
打分统一用 CachedAECMOS（official_aecmos_eval.py 同款：est.run 逻辑逐行同官方，
仅缓存 InferenceSession；16k 输入直送 48k 模型与 aecmos_eval.py 旧口径一致）。
写域：仅本文件与 outputs/hardening_rerun/**、D:/data/aec-synth/official_stage/enh_trad_hard/**。
"""
import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
AVROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))                                # aecmos_eval / official_aecmos_eval
sys.path.insert(0, str(HERE.parents[0] / "06_trad_aec"))     # WP-06 pipeline

from official_aecmos_eval import (  # noqa: E402
    AECMOS_DIR, AECMOS_MODEL, CachedAECMOS, enumerate_clips, load16, trim_for)
from aecmos_eval import TALK  # noqa: E402
from pipeline import TradAEC  # noqa: E402
from eval_aec import erle_db, pesq_safe  # noqa: E402

SR = 16000
OUT = HERE / "outputs" / "hardening_rerun"
STAGE_HARD = Path("D:/data/aec-synth/official_stage/enh_trad_hard")
DRIFT_DIR = Path("D:/data/aec-synth/eval_v3_drift")
V6_DIR = Path("D:/data/aec-synth/eval_v6_realrir")
DIV_NAME = "wufNfJIb3kC7OvuxutARLQ_farend-singletalk-with-movement"
Z = 1.96
# S1 加固配置（HANDOFF §4 / §6.1；其余参数默认）
HARD_CFG = dict(method="mdf", partitions=8, mu=0.7,
                align_margin_ms=25.0, reestimate_sec=2.0)
# 归因消融变体（S10 报告解读节用）："old" 兼作驱动口径回归校验（应复现旧 json）
VARIANTS = {
    "hard":   dict(margin=25.0, reest=2.0),   # S10 主配置
    "margin": dict(margin=25.0, reest=None),  # 仅对齐余量
    "reest":  dict(margin=0.0, reest=2.0),    # 仅周期重估
    "old":    dict(margin=0.0, reest=None),   # 旧配置（WP-06 原样，口径校验）
}


def _variant_cfg(variant):
    v = VARIANTS[variant]
    return dict(method="mdf", partitions=8, mu=0.7,
                align_margin_ms=v["margin"], reestimate_sec=v["reest"])


def _venh_dir(variant, set_name):
    """变体增强 wav 目录；hard 保持 S10 主产出目录名。"""
    return OUT / (f"enh_{set_name}" if variant == "hard"
                  else f"enh_{set_name}_{variant}")


def _ci_half(vals):
    n = len(vals)
    if n < 2:
        return 0.0, 0.0
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    return mean, Z * sd / math.sqrt(n)


# ---------------------------------------------------------------- 处理侧 job
def _synth_job(args):
    """合成集一条：读 wav → 指定配置 TradAEC → enh wav + 元数据。"""
    name, sc, lpb_path, mic_path, dst, cfg = args
    t0 = time.time()
    try:
        if Path(dst).exists() and Path(dst).stat().st_size > 1024:
            return dict(name=name, scenario=sc, status="cached", sec=0.0)
        lpb, _ = sf.read(lpb_path, dtype="float32")
        mic, _ = sf.read(mic_path, dtype="float32")
        n = min(len(lpb), len(mic))
        aec = TradAEC(**cfg)
        r = aec.process(mic[:n].astype(np.float64), lpb[:n].astype(np.float64))
        out = r.out[:n]
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, out.astype(np.float32), SR, subtype="FLOAT")
        return dict(name=name, scenario=sc, status="ok", sec=round(time.time() - t0, 2),
                    delay_ms=round(r.delay_ms, 2), align_ms=round(r.align_ms, 2),
                    n_realigns=int(r.n_realigns), n_divergence_resets=int(r.n_divergence_resets),
                    dt_frac=round(r.dt_frac, 3), evidence=round(r.evidence, 1),
                    finite=bool(np.isfinite(out).all()))
    except Exception as e:  # 单条失败记行继续
        return dict(name=name, scenario=sc, status="failed", sec=round(time.time() - t0, 2),
                    error=f"{type(e).__name__}: {e}")


def _official_job(clip):
    """官方 FEST 一条：load16（48k→resample_poly 1/3→16k）→ 加固 TradAEC → enh wav。"""
    subset, name = clip["subset"], clip["name"]
    dst = STAGE_HARD / subset / f"{name}.wav"
    t0 = time.time()
    try:
        if dst.exists() and dst.stat().st_size > 1024:
            return dict(subset=subset, name=name, status="cached", sec=0.0)
        l16, m16 = load16(clip)
        aec = TradAEC(**HARD_CFG)
        r = aec.process(m16.astype(np.float64), l16.astype(np.float64))
        out = r.out
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, out.astype(np.float32), SR, subtype="FLOAT")
        realigns = [(t, round(est, 1)) for t, est, ra in r.delay_track if ra]
        return dict(subset=subset, name=name, status="ok", sec=round(time.time() - t0, 2),
                    delay_ms=round(r.delay_ms, 2), align_ms=round(r.align_ms, 2),
                    n_realigns=int(r.n_realigns), n_divergence_resets=int(r.n_divergence_resets),
                    dt_frac=round(r.dt_frac, 3), evidence=round(r.evidence, 1),
                    len=int(len(out)), finite=bool(np.isfinite(out).all()),
                    delay_first_last=[round(r.delay_track[0][1], 1),
                                      round(r.delay_track[-1][1], 1)],
                    realign_events=realigns)
    except Exception as e:
        return dict(subset=subset, name=name, status="failed", sec=round(time.time() - t0, 2),
                    error=f"{type(e).__name__}: {e}")


def _run_infer(jobs, meta_path, workers, log_every=25):
    """并行处理 + meta jsonl 追加（断点续）。返回 rows。"""
    done = set()
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") in ("ok", "cached"):
                        done.add(r["name"])
                except json.JSONDecodeError:
                    pass
    todo = [j for j in jobs if j[0] not in done]
    print(f"[infer] todo={len(todo)}/{len(jobs)} (已有 {len(done)} 行)", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_synth_job, j): j for j in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % log_every == 0 or i == len(futs):
                ok = sum(x["status"] in ("ok", "cached") for x in rows)
                print(f"[infer {i}/{len(futs)}] ok={ok} fail={i - ok} "
                      f"{time.time() - t0:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------- 打分侧 job
_SCORER = None


def _score_init():
    global _SCORER
    if str(AECMOS_DIR) not in sys.path:
        sys.path.insert(0, str(AECMOS_DIR))   # aecmos 模块（CachedAECMOS.__init__ 内 import）
    _SCORER = CachedAECMOS(str(AECMOS_MODEL), intra_threads=2)


def _synth_score_job(item):
    """合成集一条打分：16k 三路直送（aecmos_eval --enh-dir 同口径）。"""
    name, sc, lpb_path, mic_path, enh_path = item
    try:
        lpb, _ = sf.read(lpb_path, dtype="float32")
        mic, _ = sf.read(mic_path, dtype="float32")
        enh, _ = sf.read(enh_path, dtype="float32")
        n = min(len(lpb), len(mic), len(enh))
        echo, deg = _SCORER.run(TALK[sc], lpb[:n], mic[:n], enh[:n])
        return dict(name=name, scenario=sc, status="ok", echo_mos=round(echo, 4),
                    deg_mos=round(deg, 4), error="")
    except Exception as e:
        return dict(name=name, scenario=sc, status="failed", echo_mos=None,
                    deg_mos=None, error=f"{type(e).__name__}: {e}")


def _official_score_job(clip):
    """官方 FEST 一条：官方 read_and_process 流程（librosa 48k → min_len →
    trim_for(proto) → scorer.run）。"""
    import librosa
    subset, name = clip["subset"], clip["name"]
    enhp = STAGE_HARD / subset / f"{name}.wav"
    try:
        lpb, _ = librosa.load(clip["lpb"], sr=48000)
        mic, _ = librosa.load(clip["mic"], sr=48000)
    except Exception as e:
        return dict(variant="proto", method="trad_hard", subset=subset, name=name,
                    status="failed", echo_mos=None, deg_mos=None,
                    error=f"load lpb/mic: {type(e).__name__}: {e}")
    try:
        enh, _ = librosa.load(enhp, sr=48000)
    except Exception as e:
        return dict(variant="proto", method="trad_hard", subset=subset, name=name,
                    status="failed", echo_mos=None, deg_mos=None,
                    error=f"load enh: {type(e).__name__}: {e}")
    n = min(len(lpb), len(mic), len(enh))
    if not np.all(np.isfinite(enh)) or float(np.max(np.abs(enh))) > 1e3:
        return dict(variant="proto", method="trad_hard", subset=subset, name=name,
                    status="failed", echo_mos=None, deg_mos=None,
                    error="enh_diverged: 推理列输出含非有限/爆表样本")
    base = [lpb[:n], mic[:n], enh[:n]]
    try:
        trimmed, k, _note = trim_for(subset, "proto", base)
        a, b, c = trimmed
    except Exception as e:
        return dict(variant="proto", method="trad_hard", subset=subset, name=name,
                    status="failed", echo_mos=None, deg_mos=None,
                    error=f"trim: {type(e).__name__}: {e}")
    try:
        echo, deg = _SCORER.run(TALK[subset], a, b, c)
        return dict(variant="proto", method="trad_hard", subset=subset, name=name,
                    status="ok", echo_mos=round(echo, 4), deg_mos=round(deg, 4),
                    n48=int(len(a)), trim=int(k), error="")
    except Exception as e:
        return dict(variant="proto", method="trad_hard", subset=subset, name=name,
                    status="failed", echo_mos=None, deg_mos=None,
                    error=f"score: {type(e).__name__}: {e}")


def _run_score(items, rows_path, job_fn, workers, log_every=25):
    """并行打分 + jsonl 追加（行级去重）。返回 rows。"""
    done = set()
    if rows_path.exists():
        with open(rows_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        done.add(r["name"])
                except json.JSONDecodeError:
                    pass
    todo = [it for it in items if it[0] not in done]
    print(f"[score] todo={len(todo)}/{len(items)} (已有 {len(done)} 行)", flush=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_score_init) as ex:
        futs = {ex.submit(job_fn, it): it for it in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % log_every == 0 or i == len(futs):
                ok = sum(x["status"] == "ok" for x in rows)
                print(f"[score {i}/{len(futs)}] ok={ok} fail={i - ok} "
                      f"{time.time() - t0:.0f}s", flush=True)
    return rows


# ---------------------------------------------------------------- 聚合
def _agg_synth(rows_path, out_json, tag, old_rel, extra_rows_path=None):
    """合成集聚合：per-scenario mean±CI + 配对差 vs 旧 json（按 (name, scenario)）。"""
    rows = []
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    summary = {}
    for sc in ("fest", "dt", "nst"):
        sub = [r for r in rows if r["scenario"] == sc and r["status"] == "ok"]
        if not sub:
            continue
        em = [r["echo_mos"] for r in sub]
        dm = [r["deg_mos"] for r in sub]
        me, ce = _ci_half(em)
        md, cd = _ci_half(dm)
        summary[sc] = dict(n=len(sub), echo_mos=round(me, 3), echo_ci95=round(ce, 3),
                           deg_mos=round(md, 3), deg_ci95=round(cd, 3),
                           echo_sd=round((ce * math.sqrt(len(em)) / Z), 3) if len(em) > 1 else 0.0)
    out = dict(tag=tag, config=HARD_CFG, summary=summary, rows=rows)
    if old_rel:
        old_p = HERE / "outputs" / old_rel
        if old_p.exists():
            old_rows = json.loads(old_p.read_text(encoding="utf-8"))["rows"]
            old_map = {(r["name"], r["scenario"]): r for r in old_rows}
            paired = {}
            for sc in ("fest", "dt", "nst"):
                keys = [k for k in old_map if k[1] == sc
                        and any(r["name"] == k[0] and r["scenario"] == sc
                                and r["status"] == "ok" for r in rows)]
                de = [next(r["echo_mos"] for r in rows
                           if r["name"] == k[0] and r["scenario"] == sc and r["status"] == "ok")
                      - float(old_map[k]["echo_mos"]) for k in keys]
                dd = [next(r["deg_mos"] for r in rows
                           if r["name"] == k[0] and r["scenario"] == sc and r["status"] == "ok")
                      - float(old_map[k]["deg_mos"]) for k in keys]
                if de:
                    mde, cde = _ci_half(de)
                    mdd, cdd = _ci_half(dd)
                    paired[sc] = dict(n=len(de), d_echo=round(mde, 3), d_echo_ci95=round(cde, 3),
                                      d_deg=round(mdd, 3), d_deg_ci95=round(cdd, 3))
            out["paired_vs_old"] = dict(old_file=str(old_p), diff=paired)
    if extra_rows_path:  # v6：erle/pesq 行
        erows = []
        with open(extra_rows_path, encoding="utf-8") as f:
            for line in f:
                try:
                    erows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        out["erle_pesq_rows"] = erows
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"[agg:{tag}] {json.dumps(summary, ensure_ascii=False)}")
    if "paired_vs_old" in out:
        print(f"[agg:{tag}] paired={json.dumps(out['paired_vs_old']['diff'], ensure_ascii=False)}")
    print(f"-> {out_json}", flush=True)


def _agg_official(rows_path, out_json):
    """官方 FEST 聚合：总 300 / 同名 299 配对 vs 旧 proto:trad 列。"""
    rows = []
    with open(rows_path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    ok = [r for r in rows if r["status"] == "ok" and r["subset"] == "fest"]
    em = [r["echo_mos"] for r in ok]
    dm = [r["deg_mos"] for r in ok]
    me, ce = _ci_half(em)
    md, cd = _ci_half(dm)
    summary = dict(n=len(ok), echo_mos=round(me, 3), echo_ci95=round(ce, 3),
                   deg_mos=round(md, 3), deg_ci95=round(cd, 3),
                   failed=len(rows) - len(ok))
    excl = [r for r in ok if r["name"] != DIV_NAME]
    em2 = [r["echo_mos"] for r in excl]
    me2, ce2 = _ci_half(em2)
    summary["n_excl_divergence"] = len(excl)
    summary["echo_mos_excl_divergence"] = round(me2, 3)
    summary["echo_ci95_excl"] = round(ce2, 3)
    # 配对 vs 旧 proto:trad:fest（reports/official_aecmos/scores_official.jsonl）
    old_jsonl = AVROOT / "reports" / "official_aecmos" / "scores_official.jsonl"
    paired = None
    if old_jsonl.exists():
        old_map = {}
        with open(old_jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (r.get("variant") == "proto" and r.get("method") == "trad"
                        and r.get("subset") == "fest" and r.get("status") == "ok"):
                    old_map[r["name"]] = r["echo_mos"]
        new_map = {r["name"]: r["echo_mos"] for r in ok}
        common = sorted(set(old_map) & set(new_map))
        de = [new_map[k] - float(old_map[k]) for k in common]
        mde, cde = _ci_half(de)
        m_old = sum(float(old_map[k]) for k in common) / len(common)
        paired = dict(n=len(de), old_mean_same_n=round(m_old, 3),
                      d_echo=round(mde, 3), d_echo_ci95=round(cde, 3))
    out = dict(tag="trad_hard_official_fest_proto", config=HARD_CFG,
               summary=summary, paired_vs_old=paired, rows=rows)
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"[agg:official] {json.dumps(summary, ensure_ascii=False)}")
    if paired:
        print(f"[agg:official] paired={json.dumps(paired, ensure_ascii=False)}")
    print(f"-> {out_json}", flush=True)


# ---------------------------------------------------------------- 各 stage
def _manifest_rows(eval_dir):
    return list(csv.DictReader(open(Path(eval_dir) / "manifest.csv", encoding="utf-8")))


def _vp(variant, base, ext, suffix=""):
    """hard 保持 S10 主产出名；其余带变体后缀。suffix 形如 '_margin'。"""
    if variant == "hard":
        return OUT / f"{base}{suffix}.{ext}"
    return OUT / f"{base}_{variant}.{ext}"


def stage_drift(workers, variant="hard"):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _variant_cfg(variant)
    enh_dir = _venh_dir(variant, "trad_hard_drift")
    rows_m = _manifest_rows(DRIFT_DIR)
    jobs = [(r["name"], r["scenario"],
             str(DRIFT_DIR / "farend" / f"{r['name']}.wav"),
             str(DRIFT_DIR / "mic" / f"{r['name']}.wav"),
             str(enh_dir / f"{r['name']}.wav"), cfg) for r in rows_m]
    _run_infer(jobs, _vp(variant, "meta_drift", "jsonl"), workers)
    items = [(r["name"], r["scenario"],
              str(DRIFT_DIR / "farend" / f"{r['name']}.wav"),
              str(DRIFT_DIR / "mic" / f"{r['name']}.wav"),
              str(enh_dir / f"{r['name']}.wav")) for r in rows_m]
    _run_score(items, _vp(variant, "scores_trad_hard_drift", "jsonl"),
               _synth_score_job, workers)
    _agg_synth(_vp(variant, "scores_trad_hard_drift", "jsonl"),
               _vp(variant, "aecmos_trad_hard_drift", "json"),
               f"trad_hard_drift_{variant}", "aecmos_trad_drift.json")


def stage_v6(workers, variant="hard"):
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _variant_cfg(variant)
    enh_dir = _venh_dir(variant, "trad_hard_v6")
    rows_m = _manifest_rows(V6_DIR)
    jobs = [(r["name"], r["scenario"],
             str(V6_DIR / "farend" / f"{r['name']}.wav"),
             str(V6_DIR / "mic" / f"{r['name']}.wav"),
             str(enh_dir / f"{r['name']}.wav"), cfg) for r in rows_m]
    _run_infer(jobs, _vp(variant, "meta_v6", "jsonl"), workers)
    items = [(r["name"], r["scenario"],
              str(V6_DIR / "farend" / f"{r['name']}.wav"),
              str(V6_DIR / "mic" / f"{r['name']}.wav"),
              str(enh_dir / f"{r['name']}.wav")) for r in rows_m]
    _run_score(items, _vp(variant, "scores_trad_hard_v6", "jsonl"),
               _synth_score_job, workers)
    _v6_erle_pesq(workers, variant, enh_dir)
    _agg_synth(_vp(variant, "scores_trad_hard_v6", "jsonl"),
               _vp(variant, "aecmos_trad_hard_v6", "json"),
               f"trad_hard_v6_{variant}", "eval_v6_realrir/aecmos_trad_v6.json",
               extra_rows_path=_vp(variant, "erle_pesq_v6", "jsonl"))


def _v6_erle_pesq(workers, variant="hard", enh_dir=None):
    """eval_aec 口径 ERLE（skip 2s，FEST）+ DT/NST 近端 PESQ（列同旧 trad_rows_v6.csv）。"""
    outp = _vp(variant, "erle_pesq_v6", "jsonl")
    csvp = _vp(variant, "trad_hard_rows_v6", "csv")
    if enh_dir is None:
        enh_dir = _venh_dir(variant, "trad_hard_v6")
    done = set()
    if outp.exists():
        with open(outp, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(r["name"])
                except json.JSONDecodeError:
                    pass
    rows_m = [r for r in _manifest_rows(V6_DIR) if r["name"] not in done]
    print(f"[erle_pesq] todo={len(rows_m)} (已有 {len(done)} 行)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_v6_metric_job, (r, str(enh_dir))): r for r in rows_m}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with open(outp, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(futs):
                print(f"[erle_pesq {i}/{len(futs)}] {time.time() - t0:.0f}s", flush=True)
    # CSV（列同旧 trad_rows_v6.csv：name,scenario,pesq,pesq_mic,erle_db）
    erows = []
    with open(outp, encoding="utf-8") as f:
        for line in f:
            try:
                erows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "scenario", "pesq", "pesq_mic", "erle_db"])
        w.writeheader()
        for r in sorted(erows, key=lambda x: x["name"]):
            w.writerow(dict(name=r["name"], scenario=r["scenario"],
                            pesq=r.get("pesq", ""), pesq_mic=r.get("pesq_mic", ""),
                            erle_db=r.get("erle_db", "")))
    print(f"-> {csvp}", flush=True)


def _v6_metric_job(item):
    m, enh_dir = item
    name, sc = m["name"], m["scenario"]
    try:
        mic, _ = sf.read(V6_DIR / "mic" / f"{name}.wav", dtype="float32")
        enh, _ = sf.read(Path(enh_dir) / f"{name}.wav", dtype="float32")
        n = min(len(mic), len(enh))
        mic, enh = mic[:n].astype(np.float64), enh[:n].astype(np.float64)
        row = dict(name=name, scenario=sc)
        if sc == "fest":
            row["erle_db"] = round(float(erle_db(mic, enh)), 4)
        else:
            near, _ = sf.read(V6_DIR / "nearend" / f"{name}.wav", dtype="float32")
            k = min(len(near), n)
            p_out, _s_out = pesq_safe(near[:k], enh[:k].astype(np.float32))
            p_mic, _s_mic = pesq_safe(near[:k], mic[:k].astype(np.float32))
            row["pesq"] = round(float(p_out), 4)
            row["pesq_mic"] = round(float(p_mic), 4)
        return row
    except Exception as e:
        return dict(name=name, scenario=sc, error=f"{type(e).__name__}: {e}")


def stage_official_one():
    """发散条单条：处理 + 断言全有限 + n_divergence_resets 记录。"""
    OUT.mkdir(parents=True, exist_ok=True)
    clip = dict(subset="fest", name=DIV_NAME,
                lpb=str(Path("D:/data/repos/AEC-Challenge/datasets/test_set_icassp2022")
                        / "farend-singletalk" / f"{DIV_NAME}_lpb.wav"),
                mic=str(Path("D:/data/repos/AEC-Challenge/datasets/test_set_icassp2022")
                        / "farend-singletalk" / f"{DIV_NAME}_mic.wav"))
    t0 = time.time()
    l16, m16 = load16(clip)
    aec = TradAEC(**HARD_CFG)
    r = aec.process(m16.astype(np.float64), l16.astype(np.float64))
    out = r.out
    dst = STAGE_HARD / "fest" / f"{DIV_NAME}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, out.astype(np.float32), SR, subtype="FLOAT")
    finite = bool(np.isfinite(out).all())
    max_abs = float(np.max(np.abs(out))) if finite else float("nan")
    info = dict(name=DIV_NAME, subset="fest", config=HARD_CFG,
                n=len(out), finite=finite, max_abs_out=round(max_abs, 4),
                n_divergence_resets=int(r.n_divergence_resets),
                n_realigns=int(r.n_realigns), align_ms=round(r.align_ms, 2),
                delay_ms=round(r.delay_ms, 2),
                delay_track=[(t, round(est, 1), bool(ra)) for t, est, ra in r.delay_track],
                old_behavior="旧列（margin=0，无重估）输出含非有限样本（divergence_scan.json: Infinity, 173816 样本）",
                sec=round(time.time() - t0, 2), out=str(dst))
    assert finite, "加固列输出含非有限样本！"
    (OUT / "divergence_clip_hard.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "delay_track"},
                     ensure_ascii=False, indent=2))
    print(f"delay_track: {len(r.delay_track)} 项, realign {r.n_realigns} 次")
    print(f"-> {OUT / 'divergence_clip_hard.json'}", flush=True)


def stage_official_infer(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    clips = [c for c in enumerate_clips(["fest"])]
    done = set()
    meta_path = OUT / "meta_official_fest.jsonl"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") in ("ok", "cached"):
                        done.add(r["name"])
                except json.JSONDecodeError:
                    pass
    todo = [c for c in clips if c["name"] not in done]
    print(f"[official-infer] todo={len(todo)}/{len(clips)}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_official_job, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(futs):
                print(f"[official-infer {i}/{len(futs)}] {time.time() - t0:.0f}s", flush=True)
    fails = 0
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                try:
                    if json.loads(line).get("status") == "failed":
                        fails += 1
                except json.JSONDecodeError:
                    pass
    print(f"[official-infer done] fails={fails}", flush=True)


def _run_score_official(items, rows_path, workers):
    done = set()
    if rows_path.exists():
        with open(rows_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        done.add(r["name"])
                except json.JSONDecodeError:
                    pass
    todo = [it for it in items if it[0] not in done]
    print(f"[official-score] todo={len(todo)}/{len(items)} (已有 {len(done)} 行)", flush=True)
    if not todo:
        return
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_score_init) as ex:
        futs = {ex.submit(_official_score_job, it[1]): it for it in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            with open(rows_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 25 == 0 or i == len(futs):
                ok = sum(x["status"] == "ok" for x in rows)
                print(f"[official-score {i}/{len(futs)}] ok={ok} fail={i - ok} "
                      f"{time.time() - t0:.0f}s", flush=True)


def stage_drift_erle(variant):
    """漂移集 FEST 条 DSP 级 ERLE（eval_aec 口径：10log10(P_mic/P_out)，skip 2s）。
    从已落盘 enh wav 计算，不重跑处理。"""
    enh_dir = _venh_dir(variant, "trad_hard_drift")
    rows_m = [r for r in _manifest_rows(DRIFT_DIR) if r["scenario"] == "fest"]
    erles = {}
    for r in rows_m:
        mic, _ = sf.read(DRIFT_DIR / "mic" / f"{r['name']}.wav", dtype="float32")
        enh, _ = sf.read(enh_dir / f"{r['name']}.wav", dtype="float32")
        n = min(len(mic), len(enh))
        erles[r["name"]] = round(float(erle_db(mic[:n].astype(np.float64),
                                               enh[:n].astype(np.float64))), 4)
    vals = list(erles.values())
    m, c = _ci_half(vals)
    out = dict(variant=variant, n=len(vals), erle_mean=round(m, 3),
               erle_ci95=round(c, 3), per_clip=erles)
    (OUT / f"erle_drift_{variant}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[drift-erle:{variant}] n={len(vals)} mean={m:.3f}±{c:.3f} dB")
    print(f"-> {OUT / f'erle_drift_{variant}.json'}", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["drift", "official-one", "official-infer",
                                      "official-score", "official-agg", "v6",
                                      "drift-erle"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--variant", choices=list(VARIANTS), default="hard",
                    help="drift/v6 消融变体（hard=S10 主配置；old 兼作口径回归校验）")
    args = ap.parse_args()
    if args.stage == "drift":
        stage_drift(args.workers, args.variant)
    elif args.stage == "v6":
        stage_v6(args.workers, args.variant)
    elif args.stage == "official-one":
        stage_official_one()
    elif args.stage == "official-infer":
        stage_official_infer(args.workers)
    elif args.stage == "official-score":
        clips = enumerate_clips(["fest"])
        items = [(c["name"], c) for c in clips]
        _run_score_official(items, OUT / "scores_trad_hard_official_fest.jsonl", args.workers)
    elif args.stage == "official-agg":
        _agg_official(OUT / "scores_trad_hard_official_fest.jsonl",
                      OUT / "aecmos_trad_hard_official_fest.json")
    elif args.stage == "drift-erle":
        stage_drift_erle(args.variant)


if __name__ == "__main__":
    main()
