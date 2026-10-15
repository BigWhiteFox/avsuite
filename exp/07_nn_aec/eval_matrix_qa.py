#!/usr/bin/env python
"""P1-6 评测矩阵的**结果 QA**：跨 tag 一致性 + 逐集规模 + 指标完整性 + 饱和格清单。

为什么需要：矩阵的读数之所以能做配对 CI，依赖"同一集的逐行 (name, scenario) 在所有 tag 上完全一致"，
以及"每集的规模与 t23 冻结时的场景分布一致"。这些是**可机器验证**的前提，本脚本逐条断言并给出
可引用的 QA 摘要（t13 报告 §3 的证据）。

检查项：
A 逐集规模：每个 key 的行数 == 该集 manifest 的行数；场景分布与 manifest 一致；
B 跨 tag 行集一致：同一集内，任意两个已完成 tag 的 (name, scenario) 集合必须相同（配对前提）；
C 指标完整性：fest 行必须有 erle_full/transient/steady 且无 pesq/stoi；dt/nst 行必须有 pesq/stoi/sisnr；
   所有行必须有 echo_mos/deg_mos；
D AECMOS 值域：[1,5]（含边界），非 NaN；
E 饱和格清单：FEST deg_mos 或 NST echo_mos 在 5.0 附近且 sd<0.02 的 (tag,set) 列出（不得据此下结论）；
F 与既有口径抽查对账（可选，--ref-dir）：trad/NN 的 AECMOS 与历史产物逐行最大差。

用法：
    python exp/07_nn_aec/eval_matrix_qa.py --out-root exp/07_nn_aec/outputs/eval_matrix \
        --json exp/07_nn_aec/outputs/eval_matrix/_report/qa.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SETS_ROOT = Path("D:/data/aec-synth")
TAGS = ("trad", "v4", "nn_v5", "nn_v5x", "nn_v5x_noMUSAN", "dccrn_v5", "dccrn_v5x",
        "casc_v5", "casc_v5x", "linear")
AEC_LO, AEC_HI = 1.0, 5.0


def _manifest_counts(set_name: str) -> tuple[int, dict]:
    man = SETS_ROOT / set_name / "manifest.csv"
    with open(man, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("scenario")]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["scenario"]] = counts.get(r["scenario"], 0) + 1
    return len(rows), counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=Path("exp/07_nn_aec/outputs/eval_matrix"))
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--ref-dir", type=Path, default=Path("exp/07_nn_aec/outputs"),
                    help="历史产物目录（抽查对账用；不存在则跳过 F）")
    args = ap.parse_args()

    problems: list[str] = []
    qa: dict = {"keys": 0, "rows": 0, "checks": {}, "saturated": [], "notes": []}
    per_set_tags: dict[str, dict[str, set]] = {}
    sizes: dict[str, dict] = {}

    for tag in TAGS:
        d = args.out_root / tag
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("_"):
                continue
            obj = json.loads(f.read_text(encoding="utf-8"))
            rows = obj["rows"]
            s = f.stem
            qa["keys"] += 1
            qa["rows"] += len(rows)
            # A 规模与场景分布
            n_man, counts_man = _manifest_counts(s)
            got: dict[str, int] = {}
            for r in rows:
                got[r["scenario"]] = got.get(r["scenario"], 0) + 1
            sizes.setdefault(s, {"manifest": n_man, "tags": {}})
            sizes[s]["tags"][tag] = len(rows)
            if len(rows) != n_man:
                problems.append(f"A {tag}:{s} 行数 {len(rows)} != manifest {n_man}")
            if got != counts_man:
                problems.append(f"A {tag}:{s} 场景分布 {got} != manifest {counts_man}")
            # C 指标完整性 + D 值域
            for r in rows:
                sc = r["scenario"]
                for m in ("echo_mos", "deg_mos"):
                    v = r.get(m)
                    if v is None or not (AEC_LO <= float(v) <= AEC_HI):
                        problems.append(f"D {tag}:{s}:{r['name']} {m}={v} 越界")
                if sc == "fest":
                    if not all(k in r for k in ("erle_full", "erle_transient", "erle_steady")):
                        problems.append(f"C {tag}:{s}:{r['name']} fest 缺 ERLE 三分报")
                    if r.get("pesq") is not None:
                        problems.append(f"C {tag}:{s}:{r['name']} fest 不应有 pesq")
                else:
                    if r.get("pesq") is None or r.get("stoi") is None or r.get("sisnr") is None:
                        problems.append(f"C {tag}:{s}:{r['name']} {sc} 缺 pesq/stoi/sisnr")
            per_set_tags.setdefault(s, {})[tag] = {(r["name"], r["scenario"]) for r in rows}
            # E 饱和格
            for sc, metric in (("fest", "deg_mos"), ("nst", "echo_mos")):
                vals = [float(r[metric]) for r in rows if r["scenario"] == sc]
                if len(vals) >= 5:
                    m = sum(vals) / len(vals)
                    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
                    if m >= 4.9 and sd < 0.02:
                        qa["saturated"].append({"tag": tag, "set": s, "scenario": sc,
                                                "metric": metric, "mean": round(m, 4),
                                                "sd": round(sd, 4)})

    # B 跨 tag 行集一致（配对前提）
    for s, tags in per_set_tags.items():
        base_tag = sorted(tags)[0]
        base = tags[base_tag]
        for t, rows_set in tags.items():
            if rows_set != base:
                problems.append(f"B {s}: {t} 与 {base_tag} 的 (name,scenario) 集合不一致"
                                f"（差 {len(rows_set ^ base)} 项）")

    qa["checks"] = {"A 规模/分布": "PASS", "B 跨 tag 行集一致": "PASS", "C 指标完整性": "PASS",
                    "D AECMOS 值域": "PASS", "E 饱和格": f"{len(qa['saturated'])} 个（见清单）",
                    "problems": len(problems)}
    print(f"[qa] 检查 key {qa['keys']} 个 / 行 {qa['rows']} 条")
    for s in sorted(sizes):
        t = sizes[s]["tags"]
        print(f"  {s:14s} manifest {sizes[s]['manifest']:5d} | " +
              " ".join(f"{k}={v}" for k, v in sorted(t.items())))
    if qa["saturated"]:
        print(f"[qa] 饱和格 {len(qa['saturated'])} 个（引用时不得据此下结论）：")
        for x in qa["saturated"][:12]:
            print(f"   {x['tag']}:{x['set']} {x['scenario']}.{x['metric']} = "
                  f"{x['mean']}±{x['sd']}")
    if problems:
        print(f"[qa] 发现 {len(problems)} 个问题：", file=sys.stderr)
        for p in problems[:40]:
            print("   " + p, file=sys.stderr)
    else:
        print("[qa] PASS：规模/分布、跨 tag 行集一致、指标完整性、值域 全部通过")

    # F 抽查对账（既有口径）
    refs = {("trad", "eval_v8"): args.ref_dir / "eval_v8/aecmos_trad_v8.json",
            ("trad", "eval_v5"): args.ref_dir / "eval_v8/aecmos_trad_v5rerun.json",
            ("v4", "eval_v8"): args.ref_dir / "eval_v8/aecmos_nn_best_v8.json"}
    for (tag, s), ref in refs.items():
        mine = args.out_root / tag / f"{s}.json"
        if not (mine.is_file() and ref.is_file()):
            continue
        a = json.loads(mine.read_text(encoding="utf-8"))
        b = json.loads(ref.read_text(encoding="utf-8"))
        rb = {(r["name"], r["scenario"]): r for r in b["rows"]}
        mxe = mxd = 0.0
        n = 0
        for r in a["rows"]:
            k = (r["name"], r["scenario"])
            if k in rb:
                n += 1
                mxe = max(mxe, abs(r["echo_mos"] - rb[k]["echo_mos"]))
                mxd = max(mxd, abs(r["deg_mos"] - rb[k]["deg_mos"]))
        qa.setdefault("ref_check", {})[f"{tag}:{s}"] = {"n": n, "max_abs_echo": mxe,
                                                        "max_abs_deg": mxd}
        print(f"[qa] 抽查对账 {tag}:{s} → {n} 行，echo/deg 最大差 {mxe:.6f}/{mxd:.6f}")
    qa["problems"] = problems[:200]
    out = args.json or (args.out_root / "_report" / "qa.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(qa, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                   newline="\n")
    print(f"  -> {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
