#!/usr/bin/env python
"""P1-6（t13）评测矩阵驱动：一次遍历同时出 AECMOS@48k + 波形域指标 + 稳态/暂态 ERLE。

为什么合并成一支
----------------
仓内既有 `aecmos_eval.py` 与 `eval.py` 各跑一遍 NN 推理、各跑一遍 TradAEC；矩阵是
**6 模型 × 12 套集**，分开跑等于把最贵的两段（AECMOS ONNX ~233 ms/条、TradAEC ~174 ms/条）
重复 2 遍。本驱动：
- 每个 `<模型>:<集>` 只读一次 wav、只前向一次；
- 传统基线做成**独立 key**（`trad:<集>`，与模型无关）⇒ 全矩阵只跑 12 次而非 72 次；
- AECMOS 送评口径与 `aecmos_eval.py` 完全一致（@48k 模型协议、lpb=far、mic、enh 三路）；
- ERLE 按 V3.1 要求**分报**：`erle_full`（整段）/ `erle_transient`（0–2 s 收敛段）/
  `erle_steady`（2 s 之后，= 既有 `eval.py` 的 SKIP_S=2.0 口径，历史值即此列）；
- 逐行 JSON 落盘 + 断点续跑（`.part.jsonl` 记录已完成行）。

用法
----
    # 级联（B 档）：--ckpt casc=<best.ckpt>:cascade
    python exp/07_nn_aec/eval_matrix.py --ckpt casc=<ckpt>:cascade --keys casc:eval_v8
    # 单个 key（冒烟）
    python exp/07_nn_aec/eval_matrix.py --ckpt v4=<best.ckpt> --keys v4:eval_v8 --limit 12
    # 传统基线（每集一次）
    python exp/07_nn_aec/eval_matrix.py --keys trad:eval_v8,trad:eval_ns
    # 一条命令跑一批（供多进程并行分片；key 之间顺序执行，已完成的 key 自动跳过）
    python exp/07_nn_aec/eval_matrix.py --ckpt v5=<...> --keys v5:eval_v8,v5:eval_ns --out-root <dir>

输出：`<out-root>/<key 的模型部分>/<集>.json`（如 `v5/eval_v8.json`），含 rows 与 by-scenario summary。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / "exp" / "06_trad_aec"))

from ckpt_compat import load_ckpt  # noqa: E402

from core.evaluate import _safe_pesq, _safe_stoi, _si_snr_np  # noqa: E402

SR = 16000
SKIP_S = 2.0          # 收敛瞬态边界（与 WP-06 eval_aec / eval.py 同口径）
SETS_ROOT = Path("D:/data/aec-synth")
AECMOS_DIR = Path("D:/data/repos/AEC-Challenge/AECMOS/AECMOS_local")
TALK = {"fest": "st", "dt": "dt", "nst": "nst"}


# ---------------------------------------------------------------- 指标
def erle_split(est: np.ndarray, ref: np.ndarray, skip_s: float = SKIP_S) -> dict:
    """回声抑制比（FEST 段：ref=mic、est=增强输出）。分报整段/瞬态/稳态三个口径。"""
    k = int(skip_s * SR)
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    kk = min(k, n)

    def _db(a, b):
        num = float(np.sum(b ** 2)) + 1e-12
        den = float(np.sum(a ** 2)) + 1e-12
        return 10.0 * np.log10(num / den)

    return {
        "erle_full": round(_db(est, ref), 4),
        "erle_transient": round(_db(est[:kk], ref[:kk]), 4) if kk else None,
        "erle_steady": round(_db(est[kk:], ref[kk:]), 4) if n > kk else None,
    }


# ---------------------------------------------------------------- 模型
_MOD_CACHE: dict = {}


def _load_module(path: Path, alias: str, extra_dir: Path | None = None):
    """按文件路径加载模块（唯一别名，避免 exp/03、04、07 各有一个 `model.py` 互相顶替）。"""
    if alias in _MOD_CACHE:
        return _MOD_CACHE[alias]
    if extra_dir is not None:
        sys.path.insert(0, str(extra_dir))
    try:
        spec = importlib.util.spec_from_file_location(alias, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    finally:
        if extra_dir is not None:
            sys.path.remove(str(extra_dir))
    _MOD_CACHE[alias] = mod
    return mod


def load_nn(ckpt: Path, device: str):
    cls = _load_module(HERE / "model.py", "nnaec07_model").NNAEC
    st = load_ckpt(ckpt, map_location=device)
    m = cls().to(device).eval()
    m.load_state_dict(st["model"])
    return m


def load_cascade(ckpt: Path, device: str):
    """级联（B 档）：NN 架构与 NN-AEC 相同（同一个 NNAEC），差别只在输入换成 (e_lin, y_echo)。

    线性级由 `process_key` 按 key 建一个 `TradAEC(mdf/8/0.7)` 实例（训练侧
    `linear_stage.trad_instance_policy = fresh_per_clip`）。
    """
    return load_nn(ckpt, device)


def load_dccrn(ckpt: Path, device: str):
    """DCCRN（exp/04_dccrn 的 model.py；ckpt cfg 决定 clstm_hidden/real_conv）。"""
    d04 = HERE.parents[1] / "exp" / "04_dccrn"
    cls = _load_module(d04 / "model.py", "dccrn04_model", extra_dir=d04).DCCRN
    st = load_ckpt(ckpt, map_location=device)
    cfg = st.get("cfg", {}) or {}
    m = cls(clstm_hidden=int(cfg.get("clstm_hidden", 64)),
            real_conv=bool(cfg.get("real_conv", False))).to(device).eval()
    m.load_state_dict(st["model"])
    return m


def enhance(kind: str, model, mic: np.ndarray, far: np.ndarray, device: str,
            aec=None) -> np.ndarray:
    """按 kind 出增强输出。

    - `nn`：`NNAEC.enhance(mic, far)`（直接 AEC）；
    - `dccrn`：`DCCRN.enhance_wav(mic)`（NR 模型，单输入）；
    - `cascade`：先线性级 `TradAEC(mdf/8/0.7)`（**每 clip 独立实例** = fresh_per_clip），
      NN 输入换成 `(e_lin, y_echo)`——口径依据 `exp/09_cascade_aec/run_cascade.py`
      与 `train.py` 头注「级联 = 同一个 NN 换一对输入」。
    """
    t_mic = torch.from_numpy(mic).unsqueeze(0).to(device)
    if kind == "linear":
        # MDF-only 消融（t28 附加）：线性级输出直接当结果，不接 NN 后级
        res = aec.process(mic.astype(np.float64), far.astype(np.float64))
        y_echo = res.y_echo[:len(mic)].astype(np.float32)
        return (mic - y_echo).astype(np.float32)
    if kind == "dccrn":
        with torch.no_grad():
            return model.enhance_wav(t_mic)[0].cpu().numpy()
    if kind == "cascade":
        res = aec.process(mic.astype(np.float64), far.astype(np.float64))
        y_echo = res.y_echo[:len(mic)].astype(np.float32)
        e_lin = (mic - y_echo).astype(np.float32)
        with torch.no_grad():
            return model.enhance(torch.from_numpy(e_lin).unsqueeze(0).to(device),
                                 torch.from_numpy(y_echo).unsqueeze(0).to(device))[0].cpu().numpy()
    with torch.no_grad():
        t_far = torch.from_numpy(far).unsqueeze(0).to(device)
        return model.enhance(t_mic, t_far)[0].cpu().numpy()


def trad_enhance(mic: np.ndarray, far: np.ndarray, aec) -> np.ndarray:
    return aec.process(mic.astype(np.float64), far.astype(np.float64)).out.astype(np.float32)


# ---------------------------------------------------------------- AECMOS
def make_aecmos():
    sys.path.insert(0, str(AECMOS_DIR))
    from aecmos import AECMOSEstimator
    return AECMOSEstimator(str(AECMOS_DIR / "Run_1668423760_Stage_0.onnx"))


def aecmos_score(est, name, scenario, far, mic, enh):
    from math import gcd

    from scipy.signal import resample_poly
    sr_model = int(getattr(est, "sampling_rate", SR))

    def rs(x):
        if sr_model == SR:
            return x
        g = gcd(SR, sr_model)
        return resample_poly(x, sr_model // g, SR // g).astype(np.float32)

    n = min(len(far), len(mic), len(enh))
    echo, deg = est.run(TALK[scenario], rs(far[:n]), rs(mic[:n]), rs(enh[:n]))
    return float(echo), float(deg)


# ---------------------------------------------------------------- 单 key
def process_key(key: str, ckpts: dict, out_root: Path, device: str, limit: int | None,
                est, force: bool) -> dict:
    tag, set_name = key.split(":", 1)
    set_dir = SETS_ROOT / set_name
    man = set_dir / "manifest.csv"
    if not man.is_file():
        raise FileNotFoundError(f"集不存在：{set_dir}")
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{set_name}.json"
    part = out_dir / f"{set_name}.part.jsonl"
    if out_path.is_file() and not force:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"[skip] {key} 已完成（{len(prev['rows'])} 行）")
        return prev

    import csv
    with open(man, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("scenario")]
    if limit:
        rows = rows[:limit]

    done = {}
    if part.is_file() and not force:
        for ln in part.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                done[r["name"]] = r

    model = None
    aec = None
    if tag == "trad":
        from pipeline import TradAEC
        aec = TradAEC(method="mdf", partitions=8, mu=0.7)
        kind = "trad"
    elif tag in ckpts:
        kind = ckpts[tag]["kind"]
        if kind == "nn":
            model = load_nn(ckpts[tag]["path"], device)
        elif kind == "dccrn":
            model = load_dccrn(ckpts[tag]["path"], device)
        elif kind == "cascade":
            from pipeline import TradAEC
            aec = TradAEC(method="mdf", partitions=8, mu=0.7)   # fresh_per_clip：本 key 一个实例
            model = load_cascade(ckpts[tag]["path"], device)
        elif kind == "linear":
            from pipeline import TradAEC
            aec = TradAEC(method="mdf", partitions=8, mu=0.7)   # MDF-only 消融（无 NN）
        else:
            raise KeyError(f"未知 kind {kind!r}（可选 nn/dccrn/cascade）")
    else:
        raise KeyError(f"未知 key 前缀 {tag!r}（既不是 trad，也未在 --ckpt 里声明）")

    print(f"[{key}] {len(rows)} 行（已完成 {len(done)}）kind={kind} device={device}", flush=True)
    t0 = time.time()
    with open(part, "a", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            name, sc = r["name"], r["scenario"]
            if name in done:
                continue
            mic, _ = sf.read(set_dir / "mic" / f"{name}.wav", dtype="float32")
            far, _ = sf.read(set_dir / "farend" / f"{name}.wav", dtype="float32")
            near, _ = sf.read(set_dir / "nearend" / f"{name}.wav", dtype="float32")
            enh = (trad_enhance(mic, far, aec) if kind == "trad"
                   else enhance(kind, model, mic, far, device, aec=aec))
            out = {"name": name, "scenario": sc}
            if sc == "fest":
                out.update(erle_split(np.asarray(enh, dtype=np.float64),
                                      np.asarray(mic, dtype=np.float64)))
            else:
                k = min(len(near), len(enh))
                out["pesq"] = _safe_pesq(near[:k], enh[:k])
                out["stoi"] = _safe_stoi(near[:k], enh[:k])
                out["sisnr"] = float(_si_snr_np(near[:k], enh[:k]))
                out["pesq_mic"] = _safe_pesq(near[:k], mic[:k])
            echo, deg = aecmos_score(est, name, sc, far, mic, enh)
            out["echo_mos"], out["deg_mos"] = echo, deg
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            done[name] = out
            if i % 50 == 0 or i == len(rows):
                el = time.time() - t0
                rate = el / max(1, i - 0) if i else 0
                eta = rate * (len(rows) - len(done))
                print(f"  {i}/{len(rows)}  {el:.0f}s  ETA {eta / 60:.1f} min", flush=True)

    all_rows = [done[r["name"]] for r in rows if r["name"] in done]
    summary = {}
    for sc in ("fest", "dt", "nst"):
        sub = [r for r in all_rows if r["scenario"] == sc]
        if not sub:
            continue
        s = {"n": len(sub),
             "echo_mos": round(float(np.mean([r["echo_mos"] for r in sub])), 4),
             "deg_mos": round(float(np.mean([r["deg_mos"] for r in sub])), 4)}
        for k in ("erle_full", "erle_transient", "erle_steady", "pesq", "stoi", "sisnr",
                  "pesq_mic"):
            vals = [r[k] for r in sub if r.get(k) is not None]
            if vals:
                s[k] = round(float(np.mean(vals)), 4)
        summary[sc] = s
    payload = {"key": key, "tag": tag, "set": set_name, "rows": all_rows, "summary": summary,
               "protocol": {"aecmos": "AECMOS@48k 模型协议（lpb=far, mic, enh）",
                            "erle": "ref=mic；full=整段 / transient=0-2s / steady=2s 之后",
                            "pesq": "wb-16k（dt/nst 段 near vs enh）",
                            "skip_s": SKIP_S}}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"[{key}] 完成 {len(all_rows)} 行 -> {out_path}  用时 {(time.time() - t0) / 60:.1f} min",
          flush=True)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", default=[],
                    help="模型声明 name=path[:kind]（kind 缺省 nn，可选 dccrn）")
    ap.add_argument("--keys", required=True, help="逗号分隔：<name>:<set>（trad:<set> 为传统基线）")
    ap.add_argument("--out-root", type=Path, default=HERE / "outputs" / "eval_matrix")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ckpts = {}
    for spec in args.ckpt:
        parts = spec.split("=", 1)
        if len(parts) != 2:
            print(f"[eval_matrix] --ckpt 需形如 name=path[:kind]：{spec}", file=sys.stderr)
            return 2
        name, rest = parts
        kind = "nn"
        for suffix, k in ((":dccrn", "dccrn"), (":cascade", "cascade"), (":linear", "linear")):
            if rest.endswith(suffix):
                rest, kind = rest[:-len(suffix)], k
        ckpts[name] = {"path": Path(rest), "kind": kind}

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    need_aecmos = any(k.split(":", 1)[0] != "skip" for k in keys)
    est = make_aecmos() if need_aecmos else None
    print(f"[eval_matrix] {len(keys)} 个 key；device={args.device}；out={args.out_root}", flush=True)
    for key in keys:
        try:
            process_key(key, ckpts, args.out_root, args.device, args.limit, est, args.force)
        except (FileNotFoundError, KeyError) as e:
            print(f"[eval_matrix] {key} 失败：{e}", file=sys.stderr)
            return 2
    print("[eval_matrix] 全部 key 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
