"""P0-5 补充统计：**同源两列之间的配对差值**（psycho − lossctrl）与分场景一致性。

为什么需要这一步：①/②两列的训练条件完全对称（同数据/同超参/同种子/同调度），
所以"损失项的作用"的干净对照是 **② − ①**；而与存量 v4 的差值是"代际+配方"混合差。
置信区间口径沿用 `bootstrap_p05.py`（bootstrap B=2000, seed=777，百分位法）。

    python exp/07_nn_aec/bootstrap_p05_pairs.py
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

B_BOOT, SEED = 2000, 777
ROOT = Path("exp/07_nn_aec/outputs_v4_p05/aecmos")


def rows(eval_set, variant):
    p = ROOT / eval_set / f"aecmos_{variant}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["rows"]


def paired(a, b, sc, m, rng):
    A = {(r["name"], r["scenario"]): r for r in a}
    B = {(r["name"], r["scenario"]): r for r in b}
    keys = sorted(k for k in set(A) & set(B) if k[1] == sc)
    d = [float(A[k][m]) - float(B[k][m]) for k in keys
         if A[k].get(m) is not None and B[k].get(m) is not None]
    if not d:
        return None
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B_BOOT))
    lo, hi = boots[int(0.025 * B_BOOT)], boots[int(0.975 * B_BOOT) - 1]
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "delta": mean, "lo": lo, "hi": hi, "sd": sd,
            "sig": lo > 0 or hi < 0, "n_up": sum(1 for x in d if x > 0),
            "n_down": sum(1 for x in d if x < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="psycho_240ep")
    ap.add_argument("--right", default="lossctrl_240ep")
    ap.add_argument("--also-v4", action="store_true", default=True)
    ap.add_argument("--out", type=Path,
                    default=Path("exp/07_nn_aec/outputs_v4_p05/aecmos/bootstrap_p05_pairs.json"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(SEED)
    res = {"B": B_BOOT, "seed": SEED, "left": args.left, "right": args.right, "results": {}}
    print(f"# 配对差值（{args.left} − {args.right}），B={B_BOOT} seed={SEED}\n")
    print("| eval 集 | 场景 | 指标 | n | Δmean | 配对 sd | bootstrap 95%CI | 降/升 | 结论 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in ("eval_v3", "eval_v5"):
        a = rows(s, args.left)
        b = rows(s, args.right)
        if a is None or b is None:
            continue
        for sc in ("dt", "nst", "fest"):
            for m in ("echo_mos", "deg_mos"):
                r = paired(a, b, sc, m, rng)
                if r is None:
                    continue
                res["results"][f"{s}.{sc}.{m}"] = r
                v = "显著回升" if (r["sig"] and r["delta"] > 0) else \
                    "显著退化" if (r["sig"] and r["delta"] < 0) else "CI 含 0（不显著）"
                print(f"| {s} | {sc} | {m} | {r['n']} | {r['delta']:+.3f} | {r['sd']:.3f} | "
                      f"[{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['n_down']}/{r['n_up']} | {v} |")
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
