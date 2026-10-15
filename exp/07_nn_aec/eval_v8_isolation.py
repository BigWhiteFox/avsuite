#!/usr/bin/env python
"""0-4 读数：v4 ckpt 在隔离版 eval_v8 与旧口径 eval_v5 上的配对对照 + 95%CI（差值=隔离性偏差）。

读两套**同一批命令、同一 ckpt、同一 AECMOS@48k 口径**的逐文件产物：
    outputs/eval_v8/aecmos_nn_best_{v8,v5rerun}.json     NN（v4 ckpt）AECMOS
    outputs/eval_v8/aecmos_trad_{v8,v5rerun}.json        传统 AEC（WP-06）
    outputs/eval_v8/nn_metrics_{v8,v5rerun}/rows.csv     eval.py：PESQ/STOI/ERLE/SI-SNR

四类统计（全部 B=2000、seed=777、百分位法，与 `tools/compute_ci.py` 同纪律）：

1. **集内绝对分**：每 (集, 场景, 指标) 的 n/mean/sd/解析 95%CI（1.96·sd/√n）。
2. **集内配对差**：NN − 传统，按 (name, scenario) 逐文件配对（同一条样本同场景），bootstrap CI。
3. **跨集差**：mean@eval_v8 − mean@eval_v5 —— 两个集是**不同样本**，只能做两样本 bootstrap
   （各自独立重采样），这一点在报告里必须写明（不是逐文件配对）。
4. **双重差分（DiD）**：[(NN−传统)@v8] − [(NN−传统)@v5] —— 把"域偏移"（传统基线也会变的
   那部分）从"模型靠泄漏吃到的优势"里减掉，是隔离性偏差最干净的一个刻画。

另有交叉核对：本次 eval_v5 重跑的 NN AECMOS 逐文件行与仓内既有权威产物
``outputs/aecmos48_v5fix/eval_v5/aecmos_nn.json`` 必须逐行一致（同 ckpt/同口径复核）。

用法：
    python exp/07_nn_aec/eval_v8_isolation.py            # 打印表格 + 写 json/md
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "outputs" / "eval_v8"
AECMOS_METRICS = ("echo_mos", "deg_mos")
SCENARIOS = ("fest", "dt", "nst")
#: eval.py 侧指标（rows.csv 列 -> 场景）：fest 只看 ERLE；dt/nst 看近端 PESQ/STOI
WAVE_METRICS = {"fest": ("erle_db",), "dt": ("pesq", "stoi"), "nst": ("pesq", "stoi")}
REF_V5_NN = (HERE / "outputs" / "aecmos48_v5fix" / "eval_v5" / "aecmos_nn.json",
             HERE / "outputs" / "aecmos48_v5fix" / "eval_v5" / "aecmos_trad.json")
B_BOOT = 2000
SEED = 777
Z = 1.96


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None          # 丢 NaN/Inf


def load_aecmos(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in obj.get("rows", []):
        out[(r["name"], r["scenario"])] = {m: _f(r.get(m)) for m in AECMOS_METRICS}
    return out


def load_rows_csv(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        out[(r["name"], r["scenario"])] = {k: _f(v) for k, v in r.items()
                                           if k not in ("name", "scenario")}
    return out


def boot_ci(diffs: list[float], b: int = B_BOOT, seed: int = SEED) -> tuple[float, float]:
    """单样本（配对差）bootstrap 百分位 CI。"""
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return boots[int(0.025 * b)], boots[int(0.975 * b) - 1]


def boot_ci_two(a: list[float], b_: list[float], n_boot: int = B_BOOT, seed: int = SEED):
    """两样本（独立）bootstrap 百分位 CI：mean(a) − mean(b)。"""
    rng = random.Random(seed)
    na, nb = len(a), len(b_)
    boots = sorted((sum(a[rng.randrange(na)] for _ in range(na)) / na
                    - sum(b_[rng.randrange(nb)] for _ in range(nb)) / nb) for _ in range(n_boot))
    return boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def _verdict(lo: float, hi: float) -> str:
    return "显著（CI 不含 0）" if lo > 0 or hi < 0 else "CI 含 0，不显著"


def _cell(vals: list[float]) -> dict:
    n = len(vals)
    mean = statistics.fmean(vals)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    return {"n": n, "mean": round(mean, 4), "sd": round(sd, 4),
            "ci_half": round(Z * sd / n ** 0.5, 4) if n else 0.0}


def _series(rows: dict, scenario: str, metric: str, names=None) -> tuple[list[str], list[float]]:
    keys = sorted(k for k in rows if k[1] == scenario and rows[k].get(metric) is not None)
    if names is not None:
        keys = [k for k in keys if k in names]
    return keys, [rows[k][metric] for k in keys]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()
    d = args.dir
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    needed = [d / "aecmos_nn_best_v8.json", d / "aecmos_trad_v8.json",
              d / "aecmos_nn_best_v5rerun.json", d / "aecmos_trad_v5rerun.json",
              d / "nn_metrics_v8" / "rows.csv", d / "nn_metrics_v5rerun" / "rows.csv"]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        print("[isolation] 缺输入：\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2

    nn = {"eval_v8": load_aecmos(d / "aecmos_nn_best_v8.json"),
          "eval_v5": load_aecmos(d / "aecmos_nn_best_v5rerun.json")}
    td = {"eval_v8": load_aecmos(d / "aecmos_trad_v8.json"),
          "eval_v5": load_aecmos(d / "aecmos_trad_v5rerun.json")}
    wave = {"eval_v8": load_rows_csv(d / "nn_metrics_v8" / "rows.csv"),
            "eval_v5": load_rows_csv(d / "nn_metrics_v5rerun" / "rows.csv")}

    result: dict = {"bootstrap": {"B": B_BOOT, "seed": SEED, "method": "percentile"},
                    "note": "跨集差为两样本（非配对）bootstrap；集内 NN−传统为逐文件配对 bootstrap",
                    "absolute": {}, "paired_within_set": {}, "cross_set": {}, "did": {},
                    "wave": {}, "crosscheck": {}}

    # ---- 0) 交叉核对：eval_v5 重跑 vs 仓内既有权威产物（同 ckpt 同口径应逐行一致）
    for tag, src, ref in (("nn", nn["eval_v5"], REF_V5_NN[0]), ("trad", td["eval_v5"], REF_V5_NN[1])):
        if not ref.is_file():
            result["crosscheck"][tag] = "缺参照文件，跳过"
            continue
        ref_rows = load_aecmos(ref)
        common = sorted(set(src) & set(ref_rows))
        diffs = [abs(src[k][m] - ref_rows[k][m]) for k in common for m in AECMOS_METRICS
                 if src[k][m] is not None and ref_rows[k][m] is not None]
        result["crosscheck"][tag] = {"ref": str(ref.relative_to(HERE.parents[1])),
                                     "common_rows": len(common), "n_values": len(diffs),
                                     "max_abs_diff": round(max(diffs), 9) if diffs else None}

    # ---- 1) 集内绝对分
    for s in ("eval_v8", "eval_v5"):
        result["absolute"][s] = {}
        for sc in SCENARIOS:
            _, v = _series(nn[s], sc, "echo_mos")
            if not v:
                continue
            result["absolute"][s][sc] = {m: _cell(_series(nn[s], sc, m)[1])
                                         for m in AECMOS_METRICS}

    # ---- 2) 集内配对差 NN − 传统
    result["paired_within_set"] = {}
    for s in ("eval_v8", "eval_v5"):
        result["paired_within_set"][s] = {}
        for sc in SCENARIOS:
            for m in AECMOS_METRICS:
                keys = sorted(set(nn[s]) & set(td[s]))
                keys = [k for k in keys if k[1] == sc and nn[s][k][m] is not None
                        and td[s][k][m] is not None]
                if not keys:
                    continue
                diffs = [nn[s][k][m] - td[s][k][m] for k in keys]
                lo, hi = boot_ci(diffs)
                result["paired_within_set"][s][f"{sc}.{m}"] = {
                    "n": len(diffs), "delta": round(statistics.fmean(diffs), 4),
                    "ci": [round(lo, 4), round(hi, 4)], "verdict": _verdict(lo, hi)}

    # ---- 3/4) 跨集差 + DiD（AECMOS）
    for sc in SCENARIOS:
        for m in AECMOS_METRICS:
            _, a = _series(nn["eval_v8"], sc, m)
            _, b = _series(nn["eval_v5"], sc, m)
            if not a or not b:
                continue
            lo, hi = boot_ci_two(a, b)
            result["cross_set"][f"nn.{sc}.{m}"] = {
                "n": [len(a), len(b)], "delta_v8_minus_v5": round(statistics.fmean(a)
                                                                - statistics.fmean(b), 4),
                "ci": [round(lo, 4), round(hi, 4)], "verdict": _verdict(lo, hi)}
            _, at = _series(td["eval_v8"], sc, m)
            _, bt = _series(td["eval_v5"], sc, m)
            if at and bt:
                lo2, hi2 = boot_ci_two(at, bt)
                result["cross_set"][f"trad.{sc}.{m}"] = {
                    "n": [len(at), len(bt)], "delta_v8_minus_v5": round(
                        statistics.fmean(at) - statistics.fmean(bt), 4),
                    "ci": [round(lo2, 4), round(hi2, 4)], "verdict": _verdict(lo2, hi2)}
                # DiD：把传统基线也变的那部分减掉
                rng = random.Random(SEED)
                na, nb = len(a), len(b)
                nat, nbt = len(at), len(bt)
                boots = []
                for _ in range(B_BOOT):
                    dn = (sum(a[rng.randrange(na)] for _ in range(na)) / na
                          - sum(b[rng.randrange(nb)] for _ in range(nb)) / nb)
                    dt = (sum(at[rng.randrange(nat)] for _ in range(nat)) / nat
                          - sum(bt[rng.randrange(nbt)] for _ in range(nbt)) / nbt)
                    boots.append(dn - dt)
                boots.sort()
                lo3, hi3 = boots[int(0.025 * B_BOOT)], boots[int(0.975 * B_BOOT) - 1]
                result["did"][f"{sc}.{m}"] = {
                    "did": round((statistics.fmean(a) - statistics.fmean(b))
                                 - (statistics.fmean(at) - statistics.fmean(bt)), 4),
                    "ci": [round(lo3, 4), round(hi3, 4)], "verdict": _verdict(lo3, hi3),
                    "n": [len(a), len(b), len(at), len(bt)]}

    # ---- 5) eval.py 侧（NN，跨集）
    for sc, metrics in WAVE_METRICS.items():
        for m in metrics:
            _, a = _series(wave["eval_v8"], sc, m)
            _, b = _series(wave["eval_v5"], sc, m)
            if not a or not b:
                continue
            lo, hi = boot_ci_two(a, b)
            result["wave"][f"{sc}.{m}"] = {
                "n": [len(a), len(b)],
                "mean_v8": round(statistics.fmean(a), 4), "mean_v5": round(statistics.fmean(b), 4),
                "delta_v8_minus_v5": round(statistics.fmean(a) - statistics.fmean(b), 4),
                "ci": [round(lo, 4), round(hi, 4)], "verdict": _verdict(lo, hi)}

    # ---- 输出
    js = args.json or (d / "isolation_ci.json")
    js.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8", newline="\n")
    lines = ["# eval_v8 隔离读数（v4 ckpt，eval_v8 vs eval_v5；B=2000 seed=777 百分位 bootstrap）",
             "",
             "## 0. 交叉核对（eval_v5 重跑 vs 仓内权威 @48k 产物）", ""]
    for tag, cc in result["crosscheck"].items():
        lines.append(f"- {tag}: {cc if isinstance(cc, str) else cc}")
    lines += ["", "## 1. 集内绝对分（NN=v4 ckpt，AECMOS@48k）", "",
              "| 集 | 场景 | 指标 | n | mean | 95%CI |", "|---|---|---|---|---|---|"]
    for s in ("eval_v8", "eval_v5"):
        for sc, cells in result["absolute"][s].items():
            for m, c in cells.items():
                lines.append(f"| {s} | {sc} | {m} | {c['n']} | {c['mean']:.3f} | "
                             f"±{c['ci_half']:.3f} |")
    lines += ["", "## 2. 集内配对差（NN − 传统，逐文件配对）", "",
              "| 集 | 场景.指标 | n | Δ | 95%CI | 判定 |", "|---|---|---|---|---|---|"]
    for s in ("eval_v8", "eval_v5"):
        for key, c in result["paired_within_set"][s].items():
            lines.append(f"| {s} | {key} | {c['n']} | {c['delta']:+.3f} | "
                         f"[{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | {c['verdict']} |")
    lines += ["", "## 3. 跨集差（eval_v8 − eval_v5；两样本非配对 bootstrap）", "",
              "| 对象 | 场景.指标 | n(v8/v5) | Δ | 95%CI | 判定 |", "|---|---|---|---|---|---|"]
    for key, c in result["cross_set"].items():
        obj, rest = key.split(".", 1)
        lines.append(f"| {obj} | {rest} | {c['n'][0]}/{c['n'][1]} | "
                     f"{c['delta_v8_minus_v5']:+.3f} | [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | "
                     f"{c['verdict']} |")
    lines += ["", "## 4. 双重差分（DiD = [(NN−传统)@v8] − [(NN−传统)@v5]）", "",
              "| 场景.指标 | DiD | 95%CI | 判定 | n(NN v8/v5, trad v8/v5) |", "|---|---|---|---|---|"]
    for key, c in result["did"].items():
        lines.append(f"| {key} | {c['did']:+.3f} | [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | "
                     f"{c['verdict']} | {c['n'][0]}/{c['n'][1]}/{c['n'][2]}/{c['n'][3]} |")
    lines += ["", "## 5. eval.py 波形域指标（NN，跨集差）", "",
              "| 场景.指标 | mean@v8 | mean@v5 | Δ | 95%CI | 判定 |", "|---|---|---|---|---|---|"]
    for key, c in result["wave"].items():
        lines.append(f"| {key} | {c['mean_v8']:.3f} | {c['mean_v5']:.3f} | "
                     f"{c['delta_v8_minus_v5']:+.3f} | [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] | "
                     f"{c['verdict']} |")
    text = "\n".join(lines) + "\n"
    md = args.md or (d / "isolation_tables.md")
    md.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    print(f"-> {js}\n-> {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
