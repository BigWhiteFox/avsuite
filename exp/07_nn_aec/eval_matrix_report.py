#!/usr/bin/env python
"""P1-6（t13）评测矩阵汇总：CI / 配对与非配对差 / DiD / 稳态-暂态 ERLE。

读 `exp/07_nn_aec/eval_matrix.py` 的逐 key 产物（`<tag>/<set>.json`，rows 含逐行
AECMOS/波形指标/ERLE 三分报），按 `reports/multi_axis_matrix_20261013.md` §4.2 的协议出表：

1. **绝对分**（每 tag × 集 × 场景 × 指标）：n / mean / 解析 95%CI（1.96·sd/√n）；
2. **配对差（集内）**：tag − trad，按 (name, scenario) 逐行配对 + bootstrap 95%CI（B=2000 seed=777）；
3. **跨集差**：轴集 − 参照集；`eval_rir` vs `eval_rir_sim` 是**逐样本配对**（manifest 逐字节相同），
   其余为**两样本非配对**（协议要求如实标注）；
4. **DiD**：[(tag−trad)@轴集] − [(tag−trad)@参照集]，把域偏移扣掉（继承 t5 的口径）；
5. **ERLE 三分报**（FEST）：full / transient(0-2s) / steady(>2s)，含 transient−steady 差。

用法：
    python exp/07_nn_aec/eval_matrix_report.py \
        --out-root exp/07_nn_aec/outputs/eval_matrix \
        --json exp/07_nn_aec/outputs/eval_matrix/_report/matrix_ci.json \
        --md exp/07_nn_aec/outputs/eval_matrix/_report/matrix_tables.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

TAGS = ("trad", "v4", "nn_v5", "nn_v5x", "nn_v5x_noMUSAN", "dccrn_v5", "dccrn_v5x",
        "casc_v5", "casc_v5x", "linear")
MODEL_TAGS = TAGS[1:]
SETS = ("eval_v5", "eval_v8", "eval_ns", "eval_rir", "eval_rir_sim", "eval_snr_low",
        "eval_snr_high", "eval_music", "eval_zh", "eval_zh_both", "eval_far_ls", "eval_long_ls")
SCEN = ("fest", "dt", "nst")
AEC = ("echo_mos", "deg_mos")
WAVE_DT = ("pesq", "stoi", "sisnr")
ERLE = ("erle_full", "erle_transient", "erle_steady")
#: 逐样本配对（两个集的 manifest 逐字节相同：同 seed/同池/只差 RIR 内容）
PAIRED = {("eval_rir", "eval_rir_sim")}
#: 轴集 -> 参照集（协议 §4.2；第一条是 RIR 轴的**逐样本配对**对照）
AXIS_REF = (("eval_rir", "eval_rir_sim"), ("eval_v8", "eval_v5"), ("eval_ns", "eval_v8"),
            ("eval_rir", "eval_v8"), ("eval_rir_sim", "eval_v8"),
            ("eval_snr_low", "eval_v8"), ("eval_snr_high", "eval_v8"),
            ("eval_music", "eval_v8"), ("eval_zh", "eval_v8"), ("eval_zh_both", "eval_v8"),
            ("eval_far_ls", "eval_v8"), ("eval_long_ls", "eval_far_ls"))
#: 模型间配对对照（同一集逐行配对；回答"扩数据/去 MUSAN 是否有效"）
MODEL_PAIRS = (("nn_v5", "nn_v5x"), ("nn_v5x_noMUSAN", "nn_v5x"), ("v4", "nn_v5"),
               # MDF-only 消融（t28 附加）：线性级 vs 传统全链路 / vs 级联
               ("trad", "linear"), ("linear", "casc_v5x"), ("linear", "casc_v5"),
               ("v4", "nn_v5x"), ("dccrn_v5", "dccrn_v5x"),
               # B 档（级联正式版）：级联 vs 纯 NN 同代际、级联 v5x vs v5
               ("nn_v5", "casc_v5"), ("nn_v5x", "casc_v5x"), ("casc_v5", "casc_v5x"),
               ("v4", "casc_v5x"))
B_BOOT = 2000
SEED = 777
Z = 1.96


def _sig(lo: float, hi: float, eps: float = 1e-9) -> bool:
    """CI 是否排除 0（带 epsilon：常数差经浮点运算会得到 ~1e-16 的"假显著"）。"""
    return bool(lo > eps or hi < -eps)


def _f(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def load(out_root: Path) -> dict:
    data = {}
    for tag in TAGS:
        for s in SETS:
            p = out_root / tag / f"{s}.json"
            if p.is_file():
                obj = json.loads(p.read_text(encoding="utf-8"))
                data[(tag, s)] = {(r["name"], r["scenario"]): r for r in obj["rows"]}
    return data


def boot_paired(diffs: list[float], b: int = B_BOOT, seed: int = SEED):
    """逐行差的重采样均值区间（numpy 向量化：B×n 索引矩阵一次算完）。"""
    a = np.asarray(diffs, dtype=np.float64)
    n = a.size
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, n, size=(b, n))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def boot_two(a: list[float], c: list[float], b: int = B_BOOT, seed: int = SEED):
    """两样本均值差的重采样区间（独立重采样两臂）。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(c, dtype=np.float64)
    rng = np.random.default_rng(seed)
    mx = x[rng.integers(0, x.size, size=(b, x.size))].mean(axis=1)
    my = y[rng.integers(0, y.size, size=(b, y.size))].mean(axis=1)
    d = mx - my
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def boot_did(aa, bb, at, bt, b: int = B_BOOT, seed: int = SEED):
    """DiD = [(tag−trad)@轴] − [(tag−trad)@参照]，四臂独立重采样。"""
    rng = np.random.default_rng(seed)

    def m(v):
        arr = np.asarray(v, dtype=np.float64)
        return arr[rng.integers(0, arr.size, size=(b, arr.size))].mean(axis=1)

    d = (m(aa) - m(bb)) - (m(at) - m(bt))
    return (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)),
            float(d.mean()))


def _vals(rows: dict, metric: str, scenario: str | None = None,
          names: set | None = None) -> list[float]:
    out = []
    for (nm, sc), r in sorted(rows.items()):
        if scenario and sc != scenario:
            continue
        if names is not None and (nm, sc) not in names:
            continue
        v = _f(r.get(metric))
        if v is not None:
            out.append(v)
    return out


def _cell(vals: list[float]) -> dict:
    n = len(vals)
    if not n:
        return {"n": 0}
    m = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    out = {"n": n, "mean": round(m, 4), "sd": round(sd, 4),
           "ci_half": round(Z * sd / n ** 0.5, 4)}
    # 饱和守卫（继承 NUMBERS 决策条：FEST deg / NST echo 在 5.0 附近 sd≈0 时不可据此下结论）
    if 4.9 <= m <= 5.0 and sd < 0.02:
        out["saturated"] = True
    return out


def build(data: dict) -> dict:
    res: dict = {"protocol": {"bootstrap": {"B": B_BOOT, "seed": SEED, "method": "percentile"},
                              "paired": "集内 tag−trad 与 (eval_rir, eval_rir_sim) 为逐行配对；"
                                        "其余跨集为两样本非配对",
                              "erle": "ref=mic；full=整段 / transient=0-2s / steady=>2s"},
                 "absolute": {}, "paired_vs_trad": {}, "cross_set": {}, "did": {},
                 "model_pairs": {}, "erle": {}}

    for tag in TAGS:
        for s in SETS:
            if (tag, s) not in data:
                continue
            for sc in SCEN:
                for m in AEC:
                    c = _cell(_vals(data[(tag, s)], m, sc))
                    if c["n"]:
                        res["absolute"].setdefault(f"{tag}", {}).setdefault(s, {})[f"{sc}.{m}"] = c
                if sc == "fest":
                    for m in ERLE:
                        c = _cell(_vals(data[(tag, s)], m, sc))
                        if c["n"]:
                            res["erle"].setdefault(tag, {}).setdefault(s, {})[m] = c
                else:
                    for m in WAVE_DT:
                        c = _cell(_vals(data[(tag, s)], m, sc))
                        if c["n"]:
                            res["absolute"].setdefault(tag, {}).setdefault(s, {})[f"{sc}.{m}"] = c

    # 2) 集内配对差 tag − trad
    for tag in MODEL_TAGS:
        for s in SETS:
            if (tag, s) not in data or ("trad", s) not in data:
                continue
            common = sorted(set(data[(tag, s)]) & set(data[("trad", s)]))
            for sc in SCEN:
                for m in AEC + (() if sc == "fest" else WAVE_DT) + (ERLE if sc == "fest" else ()):
                    diffs = []
                    for k in common:
                        if k[1] != sc:
                            continue
                        a, b = _f(data[(tag, s)][k].get(m)), _f(data[("trad", s)][k].get(m))
                        if a is not None and b is not None:
                            diffs.append(a - b)
                    if len(diffs) < 5:
                        continue
                    lo, hi = boot_paired(diffs)
                    res["paired_vs_trad"].setdefault(tag, {}).setdefault(s, {})[f"{sc}.{m}"] = {
                        "n": len(diffs), "delta": round(statistics.fmean(diffs), 4),
                        "ci": [round(lo, 4), round(hi, 4)],
                        "sig": _sig(lo, hi)}

    # 3) 跨集差（配对 / 非配对）+ 4) DiD
    for tag in MODEL_TAGS:
        for axis, ref in AXIS_REF:
            if (tag, axis) not in data or (tag, ref) not in data:
                continue
            is_paired = (axis, ref) in PAIRED or (ref, axis) in PAIRED
            for sc in SCEN:
                for m in AEC:
                    if is_paired:
                        common = sorted(set(data[(tag, axis)]) & set(data[(tag, ref)]))
                        diffs = []
                        for k in common:
                            if k[1] != sc:
                                continue
                            a, b = _f(data[(tag, axis)][k].get(m)), _f(data[(tag, ref)][k].get(m))
                            if a is not None and b is not None:
                                diffs.append(a - b)
                        if len(diffs) < 5:
                            continue
                        lo, hi = boot_paired(diffs)
                        d = statistics.fmean(diffs)
                        n_txt = [len(diffs), len(diffs)]
                    else:
                        a = _vals(data[(tag, axis)], m, sc)
                        b = _vals(data[(tag, ref)], m, sc)
                        if len(a) < 5 or len(b) < 5:
                            continue
                        lo, hi = boot_two(a, b)
                        d = statistics.fmean(a) - statistics.fmean(b)
                        n_txt = [len(a), len(b)]
                    res["cross_set"].setdefault(tag, {})[f"{axis}-{ref}.{sc}.{m}"] = {
                        "paired": is_paired, "n": n_txt, "delta": round(d, 4),
                        "ci": [round(lo, 4), round(hi, 4)], "sig": _sig(lo, hi)}

                    # DiD：需要两个集上都有 trad
                    if (tag, axis) in data and (tag, ref) in data \
                            and ("trad", axis) in data and ("trad", ref) in data:
                        aa = _vals(data[(tag, axis)], m, sc)
                        bb = _vals(data[(tag, ref)], m, sc)
                        at = _vals(data[("trad", axis)], m, sc)
                        bt = _vals(data[("trad", ref)], m, sc)
                        if min(len(aa), len(bb), len(at), len(bt)) < 5:
                            continue
                        lo_b, hi_b, did = boot_did(aa, bb, at, bt)
                        res["did"][f"{tag}|{axis}-{ref}.{sc}.{m}"] = {
                            "did": round(did, 4),
                            "ci": [round(lo_b, 4), round(hi_b, 4)],
                            "sig": _sig(lo_b, hi_b)}
    # 5) 模型间配对差（a − b，同一集逐行配对）
    for a_tag, b_tag in MODEL_PAIRS:
        for s in SETS:
            if (a_tag, s) not in data or (b_tag, s) not in data:
                continue
            common = sorted(set(data[(a_tag, s)]) & set(data[(b_tag, s)]))
            for sc in SCEN:
                metrics = AEC + (ERLE if sc == "fest" else WAVE_DT)
                for m in metrics:
                    diffs = []
                    for k in common:
                        if k[1] != sc:
                            continue
                        x, y = _f(data[(a_tag, s)][k].get(m)), _f(data[(b_tag, s)][k].get(m))
                        if x is not None and y is not None:
                            diffs.append(x - y)
                    if len(diffs) < 5:
                        continue
                    lo, hi = boot_paired(diffs)
                    res["model_pairs"].setdefault(f"{a_tag}-{b_tag}", {}).setdefault(s, {})[
                        f"{sc}.{m}"] = {"n": len(diffs), "delta": round(statistics.fmean(diffs), 4),
                                        "ci": [round(lo, 4), round(hi, 4)], "sig": _sig(lo, hi)}

    return res


def _fmt_ci(cell: dict | None) -> str:
    return f"{cell['mean']:.3f}±{cell['ci_half']:.3f}" if cell else "—"


def _fmt_delta(entry: dict | None) -> str:
    return (f"{entry['delta']:+.3f} [{entry['ci'][0]:+.3f},{entry['ci'][1]:+.3f}]"
            if entry else "—")


def to_md(res: dict) -> str:
    L = ["# 评测矩阵读数（自动生成；协议见 reports/multi_axis_matrix_20261013.md §4）", "",
         f"- bootstrap B={B_BOOT} seed={SEED}（百分位法）；集内 tag−trad 与 eval_rir/eval_rir_sim 为**逐行配对**，"
         "其余跨集为**两样本非配对**", "- ERLE：ref=mic；full=整段 / transient=0-2s / steady=>2s（既有口径）", "",
         "## 3. 绝对分（AECMOS，均值 ± 解析 95%CI）", "",
         "| tag | 集 | DT echo | DT deg | FEST echo | NST deg |", "|---|---|---|---|---|---|"]
    for tag in TAGS:
        for s in SETS:
            a = res["absolute"].get(tag, {}).get(s)
            if not a:
                continue
            L.append(f"| {tag} | {s} | {_fmt_ci(a.get('dt.echo_mos'))} | {_fmt_ci(a.get('dt.deg_mos'))} "
                     f"| {_fmt_ci(a.get('fest.echo_mos'))} | {_fmt_ci(a.get('nst.deg_mos'))} |")
    L += ["", "## 4. 集内配对差（tag − trad，DT/FEST/NST）", "",
          "| tag | 集 | DT echo Δ | DT deg Δ | FEST echo Δ | NST deg Δ |", "|---|---|---|---|---|---|"]
    for tag in MODEL_TAGS:
        for s in SETS:
            p = res["paired_vs_trad"].get(tag, {}).get(s)
            if not p:
                continue
            L.append(f"| {tag} | {s} | {_fmt_delta(p.get('dt.echo_mos'))} | "
                     f"{_fmt_delta(p.get('dt.deg_mos'))} | {_fmt_delta(p.get('fest.echo_mos'))} "
                     f"| {_fmt_delta(p.get('nst.deg_mos'))} |")
    L += ["", "## 5. 跨集差（轴集 − 参照集；逐行标注配对性）", "",
          "| tag | 对照 | 场景.指标 | Δ | 95%CI | 配对性 |", "|---|---|---|---|---|---|"]
    for tag in MODEL_TAGS:
        for key, v in sorted(res["cross_set"].get(tag, {}).items()):
            pair, rest = key.split(".", 1)
            L.append(f"| {tag} | {pair} | {rest} | {v['delta']:+.3f} | "
                     f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | "
                     f"{'✅ 逐样本配对' if v['paired'] else '两样本非配对'} |")
    L += ["", "## 6. DiD（[(tag−trad)@轴] − [(tag−trad)@参照]）", "",
          "| 键 | DiD | 95%CI | 显著 |", "|---|---|---|---|"]
    for k, v in sorted(res["did"].items()):
        L.append(f"| {k} | {v['did']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | "
                 f"{'是' if v['sig'] else '否'} |")
    L += ["", "## 7. 模型间配对差（DT echo/deg 等，逐行配对；回答扩数据/去 MUSAN 是否有效）", "",
          "| 对照 | 集 | 场景.指标 | Δ | 95%CI | 显著 |", "|---|---|---|---|---|---|"]
    for pair, sets_d in sorted(res.get("model_pairs", {}).items()):
        for s, cells in sorted(sets_d.items()):
            for k, v in sorted(cells.items()):
                L.append(f"| {pair} | {s} | {k} | {v['delta']:+.3f} | "
                         f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {'是' if v['sig'] else '否'} |")
    L += ["", "## 8. ERLE 三分报（FEST，均值 ± 解析 95%CI）", "",
          "| tag | 集 | full | transient(0-2s) | steady(>2s) | transient−steady |", "|---|---|---|---|---|---|"]
    for tag in TAGS:
        for s in SETS:
            e = res["erle"].get(tag, {}).get(s)
            if not e:
                continue
            f = e.get("erle_full", {}).get("mean")
            tr = e.get("erle_transient", {}).get("mean")
            st = e.get("erle_steady", {}).get("mean")
            ci = e.get("erle_steady", {}).get("ci_half", 0.0)
            gap = f"{tr - st:+.2f}" if (tr is not None and st is not None) else "—"
            L.append(f"| {tag} | {s} | {f:.2f} | {tr:.2f} | {st:.2f}±{ci:.2f} | {gap} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=Path("exp/07_nn_aec/outputs/eval_matrix"))
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()
    data = load(args.out_root)
    have = sorted({f"{t}:{s}" for (t, s) in data})
    print(f"[report] 读到 {len(have)} 个 (tag,set) 产物")
    res = build(data)
    js = args.json or (args.out_root / "_report" / "matrix_ci.json")
    md = args.md or (args.out_root / "_report" / "matrix_tables.md")
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                  newline="\n")
    text = to_md(res)
    md.write_text(text, encoding="utf-8", newline="\n")
    print(f"  -> {js}\n  -> {md}（{len(text.splitlines())} 行）")
    if len(have) < 12:
        print(f"[report] 注意：只读到 {len(have)} 个 key，可能是部分完成（矩阵共 84 个 key）",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
