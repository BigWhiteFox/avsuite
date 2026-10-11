"""deg_mos 机制物理测量：DT 段近端衰减(NLA) / lead-in 回声抑制 / 分频带 NLA / NST 伪影代理。

背景：eval_v5_dt300 结论"NN deg 结构性落后(1.81 vs 传统 2.43)"此前全靠 AECMOS 间接
推理。本脚本用合成集 ground truth（干净 nearend 参考已知）直接测量增强信号在近端
激活段的能量/谱变化，三方(trad / nn / cascade)对比。

 enh 来源（重要，如实记录）：
  - cascade：exp/07_nn_aec/outputs/eval_v5_dt300/enh_cascade/（v5 评估时落盘，直接读）。
  - trad / nn：v5 评估时 aecmos_eval.py 在内存推理、enh wav 未落盘，盘上不存在（已核查
    仓库与 D:/data 全域）。本脚本 --stage regen 用**仓库自身确定性推理管线**按 v5 同款
    配置重建——trad = TradAEC(mdf, 8, 0.7)（与 aecmos_eval.py --trad / eval.py --compare-trad
    完全一致）；nn = NNAEC + outputs_v4/nnaec_g0.5_j10/best.ckpt（v5 打分所用 v4 权重，
    见 exp/09_cascade_aec/results/THREE_WAY.md 复现命令）。重建 wav 写入
    outputs/deg_mechanism/enh_wavs/{trad,nn}/（仅 dt+nst，FEST 无需测量），并自动与
    v5 存档逐条指标(nn_metrics_v5.json/rows.csv 的 pesq、trad_metrics_v5.json 的
    pesq_out)核对——确定性 MDF 应逐位一致，NN 应在 PESQ 千分位一致(CPU/GPU 舍入差)。

 指标定义（帧=20ms/320 样本，不重叠；VAD 阈值=帧能量峰值 −40dB，对干净 nearend 求）：
  - leadin_er_db   : DT，t<leadin_sec 纯远端段 10·log10(Σ|mic|²/Σ|enh|²)（回声抑制量，正=压得多）；
  - nla_db         : DT 段近端激活帧 10·log10(Σ|enh|²/Σ|nearend|²)（负=近端被压）；
  - nla_band_db    : 同上按 4 频带（<500 / 500-2k / 2k-4k / 4k-8k Hz；逐帧 Hann 加窗
                     rFFT(320)，bin 宽 50Hz，激活帧谱能量带内求和后取比值）；
  - nst_nla_db     : NST 段（无远端，纯透传）近端激活帧 NLA——透传伪影的能量面；
  - nst_sfm_ratio_db: NST 近端激活帧谱平坦度 SFM=geometric/arithmetic mean(功率谱)，
                     比值 10·log10(SFM_enh/SFM_near)（0=谱形不变，负=谱被压平/变形）。

 用法（avsuite 环境，workdir=仓库根 avsuite/，PYTHONNOUSERSITE=1）：
  python exp/07_nn_aec/deg_mechanism.py --stage regen     # 重建 trad/nn enh + 自检（一次）
  python exp/07_nn_aec/deg_mechanism.py --stage measure   # 测量 → outputs/deg_mechanism/{rows.csv,summary.json}
  python exp/07_nn_aec/deg_mechanism.py --stage regen --validate-only   # 只跑自检
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent          # exp/07_nn_aec/
AVROOT = HERE.parents[1]                        # avsuite/
sys.path.insert(0, str(AVROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "06_trad_aec"))

EVAL_DIR = Path("D:/data/aec-synth/eval_v5")
CASCADE_ENH = HERE / "outputs" / "eval_v5_dt300" / "enh_cascade"
OUT = HERE / "outputs" / "deg_mechanism"
NN_CKPT = HERE / "outputs_v4" / "nnaec_g0.5_j10" / "best.ckpt"   # v5 打分同款 v4 权重

SR = 16000
FRAME = 320                                     # 20 ms
VAD_THRESH_DB = 40.0                            # 峰值帧能量 −40dB
BANDS = [("lt500", 0.0, 500.0), ("500_2k", 500.0, 2000.0),
         ("2k_4k", 2000.0, 4000.0), ("4k_8k", 4000.0, 8000.0)]
NEED_SC = ("dt", "nst")                         # 测量所需场景（FEST 不需要）
VAL_NAMES = ["fileid_1_dt", "fileid_2_dt", "fileid_0_nst"]       # 自检样本（固定，可复现）

METHODS = {"trad": OUT / "enh_wavs" / "trad", "nn": OUT / "enh_wavs" / "nn",
           "cascade": CASCADE_ENH}

WIN = np.hanning(FRAME)
FREQS = np.arange(FRAME // 2 + 1) * (SR / FRAME)  # rfft bin 频率，50 Hz/bin


# ------------------------------------------------------------------ 基元
def _frames(x):
    n = len(x) // FRAME
    return x[: n * FRAME].reshape(n, FRAME)


def frame_energy(x):
    return (_frames(x) ** 2).mean(axis=1)


def _wav_ok(path, n_frames):
    """续跑跳过前校验：wav 可读且帧数与期望一致（防上次中途写坏的半截文件被误跳过）。"""
    try:
        return path.exists() and path.stat().st_size > 1024 \
            and sf.info(str(path)).frames == n_frames
    except Exception:                                            # noqa: BLE001
        return False


def active_mask(near):
    """近端激活帧：帧能量 ≥ 峰值帧能量 −40dB（对干净 nearend）。"""
    e = frame_energy(near)
    return e >= e.max() * 10 ** (-VAD_THRESH_DB / 10.0)


def nla_db(enh, near, mask):
    """激活帧能量比 10·log10(Σ|enh|²/Σ|near|²)。"""
    fe, fn = _frames(enh), _frames(near)
    m = mask[: len(fe)]
    if m.sum() == 0 or len(fe) != len(fn):
        return float("nan")
    return 10.0 * np.log10((fe[m] ** 2).sum() / max((fn[m] ** 2).sum(), 1e-20))


def band_powers(x, mask):
    """逐帧 Hann 加窗 rFFT 功率谱，激活帧带内求和 → dict band→Σ|X|²。"""
    fr = _frames(x) * WIN
    spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    m = mask[: len(fr)]
    out = {}
    for bname, lo, hi in BANDS:
        sel = (FREQS >= lo) & (FREQS < hi)
        out[bname] = float(spec[np.ix_(m, sel)].sum())
    return out


def sfm_mean(x, mask):
    """激活帧谱平坦度(geometric/arithmetic mean of power spectrum)线性均值。"""
    fr = _frames(x) * WIN
    spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-12
    m = mask[: len(fr)]
    if m.sum() == 0:
        return float("nan")
    s = spec[m]
    gm = np.exp(np.log(s).mean(axis=1))
    am = s.mean(axis=1)
    return float((gm / am).mean())


# ------------------------------------------------------------------ regen
def regen(args):
    import csv as _csv
    with open(EVAL_DIR / "manifest.csv", encoding="utf-8") as f:
        rows = [r for r in _csv.DictReader(f) if r["scenario"] in NEED_SC]
    for key in ("trad", "nn"):
        METHODS[key].mkdir(parents=True, exist_ok=True)

    # ---- trad：MDF 确定性推理（与 v5 打分同配置 aecmos_eval.py --trad）
    from pipeline import TradAEC
    aec = TradAEC(method="mdf", partitions=8, mu=0.7)
    t0 = time.perf_counter()
    n_done = 0
    for r in rows:
        wav = METHODS["trad"] / f"{r['name']}.wav"
        mic = sf.read(EVAL_DIR / "mic" / f"{r['name']}.wav", dtype="float32")[0]
        if _wav_ok(wav, len(mic)):
            continue
        far = sf.read(EVAL_DIR / "farend" / f"{r['name']}.wav", dtype="float32")[0]
        out = aec.process(mic.astype(np.float64), far.astype(np.float64)).out.astype(np.float32)
        out = out[: len(mic)] if len(out) >= len(mic) else np.pad(out, (0, len(mic) - len(out)))
        sf.write(wav, out, SR, subtype="FLOAT")
        n_done += 1
        if n_done % 40 == 0:
            print(f"  [trad] {n_done} regen, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[trad] done ({n_done} regen, {time.perf_counter() - t0:.0f}s total)", flush=True)

    # ---- nn：NNAEC v4 best.ckpt（v5 打分同款）
    import torch
    from model import NNAEC
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(NN_CKPT, map_location=device, weights_only=False)
    model = NNAEC().to(device).eval()
    model.load_state_dict(st["model"])
    print(f"[nn] ckpt {NN_CKPT.name} val={st.get('val')} -> {device}", flush=True)
    t0, n_done = time.perf_counter(), 0
    with torch.no_grad():
        for r in rows:
            wav = METHODS["nn"] / f"{r['name']}.wav"
            mic = sf.read(EVAL_DIR / "mic" / f"{r['name']}.wav", dtype="float32")[0]
            far = sf.read(EVAL_DIR / "farend" / f"{r['name']}.wav", dtype="float32")[0]
            n = min(len(mic), len(far))
            if _wav_ok(wav, n):
                continue
            enh = model.enhance(torch.from_numpy(mic[:n]).unsqueeze(0).to(device),
                                torch.from_numpy(far[:n]).unsqueeze(0).to(device))
            enh = enh[0].cpu().numpy().astype(np.float32)
            sf.write(wav, enh, SR, subtype="FLOAT")
            n_done += 1
            if n_done % 20 == 0:
                print(f"  [nn] {n_done} regen, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[nn] done ({n_done} regen, {time.perf_counter() - t0:.0f}s total)", flush=True)
    validate()


def validate():
    """重建 enh 与 v5 存档逐条指标核对：PESQ(nearend, enh) vs 存档 pesq/pesq_out。"""
    from core.evaluate import _safe_pesq
    with open(HERE / "outputs/eval_v5_dt300/nn_metrics_v5.json/rows.csv", encoding="utf-8") as f:
        stored_nn = {r["name"]: r for r in csv.DictReader(f)}
    with open(HERE / "outputs/eval_v5_dt300/trad_metrics_v5.json", encoding="utf-8") as f:
        stored_tr = {r["name"]: r for r in csv.DictReader(f)}
    res = {"sample": VAL_NAMES, "trad": [], "nn": []}
    for nm in VAL_NAMES:
        near = sf.read(EVAL_DIR / "nearend" / f"{nm}.wav", dtype="float32")[0]
        for meth, store, key in (("trad", stored_tr, "pesq_out"), ("nn", stored_nn, "pesq")):
            enh_path = METHODS[meth] / f"{nm}.wav"
            if not enh_path.exists():
                res[meth].append({"name": nm, "error": "enh wav 缺（regen 未跑/未完）"})
                print(f"  [validate/{meth}] {nm}: SKIP（enh wav 缺）", flush=True)
                continue
            enh = sf.read(enh_path, dtype="float32")[0]
            n = min(len(near), len(enh))
            p = float(_safe_pesq(near[:n], enh[:n]))
            ref = float(store[nm][key])
            res[meth].append({"name": nm, "pesq_regen": round(p, 4),
                              "pesq_stored_v5": ref, "abs_diff": round(abs(p - ref), 4)})
            print(f"  [validate/{meth}] {nm}: regen {p:.4f} vs stored {ref:.4f} "
                  f"(Δ{abs(p - ref):.4f})", flush=True)
    mx = {}
    for meth in ("trad", "nn"):
        d = [r["abs_diff"] for r in res[meth] if "abs_diff" in r]
        mx[meth] = max(d) if d else None
    res["max_abs_diff"] = mx
    res["pass"] = (mx["trad"] is not None and mx["trad"] <= 0.01
                   and mx["nn"] is not None and mx["nn"] <= 0.05)
    print(f"[validate] max|Δpesq| trad={res['max_abs_diff']['trad']} "
          f"nn={res['max_abs_diff']['nn']} -> {'PASS' if res['pass'] else 'FAIL'}", flush=True)
    return res


# ------------------------------------------------------------------ measure
def measure_row(name, scenario, leadin_sec, method):
    near = sf.read(EVAL_DIR / "nearend" / f"{name}.wav", dtype="float32")[0]
    enh = sf.read(METHODS[method] / f"{name}.wav", dtype="float32")[0]
    n = min(len(near), len(enh))
    near, enh = near[:n], enh[:n]
    mask = active_mask(near)
    row = {"name": name, "scenario": scenario, "method": method,
           "leadin_sec": leadin_sec, "n_active_frames": int(mask.sum())}
    if scenario == "dt":
        mic = sf.read(EVAL_DIR / "mic" / f"{name}.wav", dtype="float32")[0][:n]
        k = int(leadin_sec * SR)
        if k > 0:
            row["leadin_er_db"] = 10.0 * np.log10(
                (mic[:k] ** 2).sum() / max((enh[:k] ** 2).sum(), 1e-20))
        dt_mask = mask.copy()
        dt_mask[: k // FRAME] = False           # 只保留 t≥leadin 的激活帧
        row["nla_db"] = nla_db(enh, near, dt_mask)
        pb, pn = band_powers(enh, dt_mask), band_powers(near, dt_mask)
        for bname, _, _ in BANDS:
            row[f"nla_{bname}_db"] = 10.0 * np.log10(pb[bname] / max(pn[bname], 1e-20))
    else:                                       # nst：纯透传伪影代理
        row["nla_db"] = nla_db(enh, near, mask)
        se, sn = sfm_mean(enh, mask), sfm_mean(near, mask)
        row["sfm_ratio_db"] = 10.0 * np.log10(se / sn) if (se and sn) else float("nan")
    return row


def pct(vals):
    v = [x for x in vals if x == x]             # 去 NaN
    if not v:
        return None
    p25, med, p75 = np.percentile(v, [25, 50, 75])
    return {"median": round(float(med), 3), "p25": round(float(p25), 3),
            "p75": round(float(p75), 3), "n": len(v)}


def measure():
    with open(EVAL_DIR / "manifest.csv", encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f))
    out_rows, missing, errors = [], [], []
    for r in rows_in:
        if r["scenario"] not in NEED_SC:
            continue
        for method in METHODS:
            wav = METHODS[method] / f"{r['name']}.wav"
            if wav.exists():
                try:
                    out_rows.append(measure_row(r["name"], r["scenario"],
                                                float(r["leadin_sec"]), method))
                except Exception as ex:                          # noqa: BLE001
                    errors.append({"method": method, "name": r["name"], "error": str(ex)})
                    print(f"[measure] ERROR {method}/{r['name']}: {ex}", flush=True)
            else:
                missing.append(f"{method}/{r['name']}")

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["name", "scenario", "method", "leadin_sec", "n_active_frames",
            "leadin_er_db", "nla_db"] + [f"nla_{b}_db" for b, _, _ in BANDS] + \
           ["sfm_ratio_db"]
    with open(OUT / "rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in row.items()})

    summ = {}
    for method in METHODS:
        ms = [r for r in out_rows if r["method"] == method]
        entry = {}
        for sc in NEED_SC:
            sub = [r for r in ms if r["scenario"] == sc]
            e = {}
            for k in (["leadin_er_db", "nla_db"] if sc == "dt" else ["nla_db"]):
                e[k] = pct([r.get(k, float("nan")) for r in sub])
            if sc == "dt":
                e["nla_band_db"] = {b: pct([r.get(f"nla_{b}_db", float("nan"))
                                            for r in sub]) for b, _, _ in BANDS}
            else:
                e["sfm_ratio_db"] = pct([r.get("sfm_ratio_db", float("nan")) for r in sub])
            entry[sc] = e
        summ[method] = entry

    aecmos_ref = {}
    for tag, p in (("trad", HERE / "outputs/eval_v5_dt300/aecmos_trad_v5.json"),
                   ("nn", HERE / "outputs/eval_v5_dt300/aecmos_nn_v4.json"),
                   ("cascade", AVROOT / "exp/09_cascade_aec/results/aecmos_cascade_v5.json")):
        try:
            with open(p, encoding="utf-8") as f:
                j = json.load(f)
            aecmos_ref[tag] = j.get("summary", {}).get("dt", {})
        except Exception as ex:                                  # noqa: BLE001
            aecmos_ref[tag] = {"error": str(ex)}

    try:
        regen_val = validate()
    except Exception as ex:                                      # noqa: BLE001
        regen_val = {"error": str(ex)}
    summary = {
        "config": {"sr": SR, "frame_ms": FRAME * 1000 // SR, "frame_hop_ms": FRAME * 1000 // SR,
                   "vad_thresh_below_peak_db": VAD_THRESH_DB,
                   "bands_hz": [[b, lo, hi] for b, lo, hi in BANDS],
                   "eval_dir": str(EVAL_DIR)},
        "enh_sources": {
            "cascade": str(CASCADE_ENH) + " (v5 评估原盘)",
            "trad": str(METHODS["trad"]) + " (regen: TradAEC mdf/8/0.7, 确定性)",
            "nn": str(METHODS["nn"]) + f" (regen: NNAEC {NN_CKPT}, v5 同款权重)",
            "missing": missing},
        "regen_validation": regen_val,
        "measure_errors": errors,
        "aecmos_ref_dt": aecmos_ref,
        "summary": summ,
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[measure] {len(out_rows)} rows -> {OUT/'rows.csv'} ; summary -> {OUT/'summary.json'}")
    if missing:
        print(f"[measure] WARNING missing enh: {missing[:10]}{' ...' if len(missing) > 10 else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["regen", "measure"], default="measure")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="nn regen 强制 CPU")
    args = ap.parse_args()
    if args.validate_only:
        validate()
        return
    if args.stage == "regen":
        regen(args)
    else:
        measure()


if __name__ == "__main__":
    main()
