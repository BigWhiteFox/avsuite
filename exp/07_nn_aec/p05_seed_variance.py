"""P0-5：逐 seed 的对称效应量与全部 run 点估计（表格化，写入报告）。"""
import json
import random
import sys
from pathlib import Path

R = Path("exp/07_nn_aec/outputs_v4_p05/aecmos")
B = 2000
SEED = 777


def rows(eval_set, variant):
    p = R / eval_set / f"aecmos_{variant}.json"
    return json.loads(p.read_text(encoding="utf-8"))["rows"] if p.exists() else None


def paired(a, b, sc, m, rng, B=B):
    A = {(r["name"], r["scenario"]): r for r in a}
    Bk = {(r["name"], r["scenario"]): r for r in b}
    ks = sorted(k for k in set(A) & set(Bk) if k[1] == sc)
    d = [float(A[k][m]) - float(Bk[k][m]) for k in ks]
    n = len(d)
    mean = sum(d) / n
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return {"n": n, "delta": mean, "lo": boots[int(0.025 * B)], "hi": boots[int(0.975 * B) - 1],
            "up": sum(1 for x in d if x > 0)}


def point(v, ev):
    r = rows(ev, v)
    if r is None:
        return None
    dt = [x for x in r if x["scenario"] == "dt"]
    return {"n": len(dt), "echo": sum(x["echo_mos"] for x in dt) / len(dt),
            "deg": sum(x["deg_mos"] for x in dt) / len(dt)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(SEED)
    print("=== eval_v5 DT：②−① 逐 seed（对称对照，B=2000 seed=777）===")
    for tag, a, b in (("seed0", "psycho_240ep", "lossctrl_240ep"),
                      ("seed1", "psycho_240ep_seed1", "lossctrl_240ep_seed1")):
        d = paired(rows("eval_v5", a), rows("eval_v5", b), "dt", "deg_mos", rng)
        e = paired(rows("eval_v5", a), rows("eval_v5", b), "dt", "echo_mos", rng)
        print(f"{tag}: Δdeg={d['delta']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}] n={d['n']} "
              f"改善 {d['up']}/{d['n']} | Δecho={e['delta']:+.3f} [{e['lo']:+.3f},{e['hi']:+.3f}]")
    print()
    print("=== 各 run 的 DT 点估计（eval_v5 n=154 / eval_v3 n=14）===")
    print("| run | v5 echo | v5 deg | v3 echo | v3 deg |")
    print("|---|---|---|---|---|")
    for v in ("v4", "lossctrl_240ep", "psycho_240ep", "lossctrl_240ep_seed1",
              "psycho_240ep_seed1", "psycho_cfix_seed1", "lossctrl_v4replica", "freshctrl_60ep"):
        p5, p3 = point(v, "eval_v5"), point(v, "eval_v3")
        if p5 is None:
            continue
        print(f"| {v} | {p5['echo']:.3f} | {p5['deg']:.3f} | {p3['echo']:.3f} | {p3['deg']:.3f} |")
    print()
    print("=== 同列跨 seed 方差（eval_v5 DT）===")
    for col, vs in (("① 原损失", ("lossctrl_240ep", "lossctrl_240ep_seed1")),
                    ("② 心理声学（自标定 c）", ("psycho_240ep", "psycho_240ep_seed1"))):
        ps = [point(v, "eval_v5") for v in vs]
        print(f"{col}: deg {ps[0]['deg']:.3f} / {ps[1]['deg']:.3f}（极差 "
              f"{abs(ps[0]['deg']-ps[1]['deg']):.3f}）| echo {ps[0]['echo']:.3f} / {ps[1]['echo']:.3f}（极差 "
              f"{abs(ps[0]['echo']-ps[1]['echo']):.3f}）")


if __name__ == "__main__":
    main()
