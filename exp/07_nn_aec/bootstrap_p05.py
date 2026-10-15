"""P0-5 置信区间与配对差值（仓内统一口径：bootstrap B=2000, seed=777）。

输入：`eval_p05_matrix.py` 产出的 `aecmos_<variant>.json`（含逐文件 rows）。
输出：markdown 表（写 `reports/p0_5_loss_matrix.md` 引用之 CSV/markdown + stdout）。

两类数字：
  1. 逐集逐场景 mean ± 1.96·sd/√n（解析 95%CI）——与 tools/compute_ci.py 同式；
  2. **配对差值**（variant − v4，同 (name, scenario) 对齐）：bootstrap 百分位 95%CI，
     B=2000 / seed=777——CI 不含 0 才叫"回升/退化"。
     DT 段另给"每场景配对 + 逐场景 n 降/升计数"。

用法：
    python exp/07_nn_aec/bootstrap_p05.py \
        --root exp/07_nn_aec/outputs_v4_p05/aecmos \
        --baseline v4 --variants lossctrl psycho [flow ...]
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

B_BOOT, SEED, Z = 2000, 777, 1.96
METRICS = ("echo_mos", "deg_mos")
SCENARIOS = ("dt", "nst", "fest")
VARIANT_LABEL = {
    "v4": "v4（存量，对照锚）",
    "lossctrl": "①V1 原损失列（频带均衡+DT 加权）",
    "lossctrl_v4replica": "①b 复刻校验（v3 续训 240ep）",
    "psycho": "②心理声学损失列（PAQM 派生）",
    "flow": "③flow matching 列",
}


def load(root: Path, eval_set: str, variant: str):
    p = root / eval_set / f"aecmos_{variant}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def per_set_stats(rows):
    """{scenario: {metric: (n, mean, sd)}}"""
    out = {}
    for sc in SCENARIOS:
        sub = [r for r in rows if r["scenario"] == sc]
        if not sub:
            continue
        for m in METRICS:
            vals = [float(r[m]) for r in sub if r.get(m) is not None]
            n = len(vals)
            mean = sum(vals) / n
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
            out.setdefault(sc, {})[m] = (n, mean, sd)
    return out


def paired_bootstrap(rows_a, rows_b, scenario, metric, rng):
    """variant(a) − baseline(b) 的配对 bootstrap 95%CI（同 (name, scenario) 对齐）。"""
    a = {(r["name"], r["scenario"]): r for r in rows_a}
    b = {(r["name"], r["scenario"]): r for r in rows_b}
    keys = sorted(k for k in set(a) & set(b) if k[1] == scenario)
    diffs = [float(a[k][metric]) - float(b[k][metric]) for k in keys
             if a[k].get(metric) is not None and b[k].get(metric) is not None]
    if not diffs:
        return None
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B_BOOT))
    lo, hi = boots[int(0.025 * B_BOOT)], boots[int(0.975 * B_BOOT) - 1]
    return {"n": n, "delta": sum(diffs) / n, "lo": lo, "hi": hi,
            "sig": (lo > 0 or hi < 0), "n_up": sum(1 for d in diffs if d > 0),
            "n_down": sum(1 for d in diffs if d < 0),
            "sd": math.sqrt(sum((d - sum(diffs) / n) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("exp/07_nn_aec/outputs_v4_p05/aecmos"))
    ap.add_argument("--sets", nargs="+", default=["eval_v3", "eval_v5"])
    ap.add_argument("--baseline", default="v4")
    ap.add_argument("--variants", nargs="+", default=["lossctrl", "psycho"])
    ap.add_argument("--out", type=Path, default=Path("reports/p0_5_loss_matrix_ci.md"))
    ap.add_argument("--json-out", type=Path,
                    default=Path("exp/07_nn_aec/outputs_v4_p05/aecmos/bootstrap_p05.json"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rng = random.Random(SEED)
    result = {"B": B_BOOT, "seed": SEED, "baseline": args.baseline, "sets": {}}
    lines = [
        "# P0-5 损失矩阵：逐集统计与配对 95%CI（自动生成）",
        "",
        f"- bootstrap B={B_BOOT}，seed={SEED}，百分位法（与 `tools/compute_ci.py` 同口径）",
        f"- 解析 95%CI = mean ± {Z}·sd/√n",
        "- 数据源：`exp/07_nn_aec/outputs_v4_p05/aecmos/<set>/aecmos_<variant>.json` 的逐文件 rows",
        "",
    ]
    for s in args.sets:
        base = load(args.root, s, args.baseline)
        if base is None:
            print(f"[skip] {s}: 缺 {args.baseline}", file=sys.stderr)
            continue
        result["sets"][s] = {"per_variant": {}, "paired": {}}
        lines += [f"## {s}", "", "### 1) 逐变体统计（解析 95%CI）", "",
                  "| 变体 | 场景 | 指标 | n | mean | 解析 95%CI |", "|---|---|---|---|---|---|"]
        variants = [args.baseline] + [v for v in args.variants if load(args.root, s, v)]
        for v in variants:
            obj = load(args.root, s, v)
            stats = per_set_stats(obj["rows"])
            result["sets"][s]["per_variant"][v] = {
                sc: {m: {"n": st[0], "mean": st[1], "sd": st[2]} for m, st in mm.items()}
                for sc, mm in stats.items()}
            for sc in SCENARIOS:
                for m in METRICS:
                    if sc in stats and m in stats[sc]:
                        n, mean, sd = stats[sc][m]
                        half = Z * sd / math.sqrt(n)
                        lines.append(f"| {v} | {sc} | {m} | {n} | {mean:.3f} | {mean:.3f}±{half:.3f} |")
        lines += ["", f"### 2) 配对差值（变体 − {args.baseline}，同 (name, scenario) 对齐）", "",
                  "| 变体 | 场景 | 指标 | n | Δmean | bootstrap 95%CI | 降/升 | 结论 |",
                  "|---|---|---|---|---|---|---|---|"]
        for v in variants:
            if v == args.baseline:
                continue
            obj = load(args.root, s, v)
            result["sets"][s]["paired"][v] = {}
            for sc in SCENARIOS:
                for m in METRICS:
                    r = paired_bootstrap(obj["rows"], base["rows"], sc, m, rng)
                    if r is None:
                        continue
                    result["sets"][s]["paired"][v][f"{sc}.{m}"] = r
                    verdict = "显著回升" if (r["sig"] and r["delta"] > 0) else \
                              "显著退化" if (r["sig"] and r["delta"] < 0) else "CI 含 0（不显著）"
                    lines.append(
                        f"| {v} | {sc} | {m} | {r['n']} | {r['delta']:+.3f} | "
                        f"[{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['n_down']}/{r['n_up']} | {verdict} |")
        lines.append("")
    lines += ["## 变体标签", ""]
    for k, v in VARIANT_LABEL.items():
        lines.append(f"- `{k}` = {v}")
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(text)
    print(f"[written] {args.out}\n[written] {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
