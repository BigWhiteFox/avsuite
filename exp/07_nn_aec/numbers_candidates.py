#!/usr/bin/env python
"""P1-6（t13）收档候选行：把矩阵里**要进 NUMBERS 的头条数字**按可引用格式抖出来（t14 用）。

只做"取值 + 排版"，不做判断；所有值取自 `eval_matrix_report.py` 的 `matrix_ci.json`
（绝对分 / 集内配对差 / 集间配对差 / 模型间配对差 / ERLE），并附 95%CI 与口径注释。

用法：
    python exp/07_nn_aec/numbers_candidates.py \
        --json exp/07_nn_aec/outputs/eval_matrix/_report/matrix_ci.json \
        --md exp/07_nn_aec/outputs/eval_matrix/_report/numbers_candidates.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWSPEC = [
    # (小节, 键路径, 说明)
    ("绝对分（DT 场景，AECMOS@48k）", ("absolute", "v4", "eval_v8", "dt.echo_mos"),
     "v4 @ eval_v8 DT echo"),
    ("绝对分（DT 场景，AECMOS@48k）", ("absolute", "v4", "eval_v8", "dt.deg_mos"),
     "v4 @ eval_v8 DT deg"),
    ("绝对分（DT 场景，AECMOS@48k）", ("absolute", "nn_v5", "eval_v8", "dt.echo_mos"),
     "nn_v5 @ eval_v8 DT echo"),
    ("绝对分（DT 场景，AECMOS@48k）", ("absolute", "nn_v5", "eval_v8", "dt.deg_mos"),
     "nn_v5 @ eval_v8 DT deg"),
    ("绝对分（DT 场景，AECMOS@48k）", ("absolute", "nn_v5x", "eval_v8", "dt.echo_mos"),
     "nn_v5x @ eval_v8 DT echo"),
    ("绝对分（DT 场景，AECMOS@48k）", ("absolute", "nn_v5x", "eval_v8", "dt.deg_mos"),
     "nn_v5x @ eval_v8 DT deg"),
    ("绝对分（DT 场景，AECMOS@48k）", ("absolute", "nn_v5x_noMUSAN", "eval_v8", "dt.echo_mos"),
     "noMUSAN @ eval_v8 DT echo"),
    ("绝对分（DT 场景，AECMOS@48k）", ("absolute", "trad", "eval_v8", "dt.echo_mos"),
     "trad @ eval_v8 DT echo"),
    ("绝对分（B 档级联，eval_v8）", ("absolute", "casc_v5", "eval_v8", "dt.echo_mos"),
     "casc_v5 @ eval_v8 DT echo"),
    ("绝对分（B 档级联，eval_v8）", ("absolute", "casc_v5", "eval_v8", "dt.deg_mos"),
     "casc_v5 @ eval_v8 DT deg"),
    ("绝对分（B 档级联，eval_v8）", ("absolute", "casc_v5x", "eval_v8", "dt.echo_mos"),
     "casc_v5x @ eval_v8 DT echo"),
    ("绝对分（B 档级联，eval_v8）", ("absolute", "casc_v5x", "eval_v8", "dt.deg_mos"),
     "casc_v5x @ eval_v8 DT deg"),
    ("绝对分（B 档级联，eval_v8）", ("absolute", "casc_v5x", "eval_v8", "dt.pesq"),
     "casc_v5x @ eval_v8 DT PESQ"),
    ("MDF-only 消融（t28 附加）", ("absolute", "linear", "eval_v8", "fest.echo_mos"),
     "线性级(仅 MDF) @ eval_v8 FEST echo"),
    ("MDF-only 消融（t28 附加）", ("absolute", "linear", "eval_v8", "dt.echo_mos"),
     "线性级(仅 MDF) @ eval_v8 DT echo"),
    ("MDF-only 消融（t28 附加）", ("absolute", "linear", "eval_rir", "fest.echo_mos"),
     "线性级(仅 MDF) @ eval_rir FEST echo"),
]
PAIRSPEC = [
    ("MDF-only 消融（t28 附加）", ("model_pairs", "linear-casc_v5x", "eval_v8", "fest.echo_mos"),
     "linear − casc_v5x @ eval_v8 FEST echo（负号 = NN 在其上再加分）"),
    ("MDF-only 消融（t28 附加）", ("model_pairs", "trad-linear", "eval_v8", "fest.echo_mos"),
     "trad − linear @ eval_v8 FEST echo（正号 = 线性级更好）"),
    ("MDF-only 消融（t28 附加）", ("model_pairs", "linear-casc_v5x", "eval_v8", "dt.echo_mos"),
     "linear − casc_v5x @ eval_v8 DT echo（负号 = NN 在其上再加分）"),
    ("逐行配对：代际（本集同批样本）", ("model_pairs", "v4-nn_v5", "eval_v8", "dt.echo_mos"),
     "v4 − nn_v5 @ eval_v8 DT echo"),
    ("逐行配对：代际（本集同批样本）", ("model_pairs", "v4-nn_v5", "eval_v8", "dt.deg_mos"),
     "v4 − nn_v5 @ eval_v8 DT deg"),
    ("逐行配对：扩数据", ("model_pairs", "nn_v5-nn_v5x", "eval_v8", "dt.echo_mos"),
     "nn_v5 − nn_v5x @ eval_v8 DT echo"),
    ("逐行配对：扩数据", ("model_pairs", "nn_v5-nn_v5x", "eval_zh", "dt.echo_mos"),
     "nn_v5 − nn_v5x @ eval_zh（中文远端）DT echo"),
    ("逐行配对：扩数据", ("model_pairs", "nn_v5-nn_v5x", "eval_music", "dt.echo_mos"),
     "nn_v5 − nn_v5x @ eval_music（MUSAN 噪声）DT echo"),
    ("逐行配对：MUSAN 消融", ("model_pairs", "nn_v5x_noMUSAN-nn_v5x", "eval_music", "dt.echo_mos"),
     "noMUSAN − nn_v5x @ eval_music DT echo"),
    ("逐行配对：MUSAN 消融", ("model_pairs", "nn_v5x_noMUSAN-nn_v5x", "eval_snr_low", "dt.deg_mos"),
     "noMUSAN − nn_v5x @ eval_snr_low（0–10 dB）DT deg"),
    ("逐行配对：RIR 轴（真实 vs 仿真）", ("cross_set", "nn_v5", "eval_rir-eval_rir_sim.dt.echo_mos"),
     "nn_v5 @ eval_rir − eval_rir_sim DT echo"),
    ("逐行配对：RIR 轴（真实 vs 仿真）", ("cross_set", "nn_v5", "eval_rir-eval_rir_sim.dt.deg_mos"),
     "nn_v5 @ eval_rir − eval_rir_sim DT deg"),
    ("逐行配对：RIR 轴（真实 vs 仿真）", ("cross_set", "v4", "eval_rir-eval_rir_sim.dt.echo_mos"),
     "v4 @ eval_rir − eval_rir_sim DT echo（方向反转）"),
    ("逐行配对：B 档级联 vs 纯 NN", ("model_pairs", "nn_v5x-casc_v5x", "eval_v8", "dt.echo_mos"),
     "nn_v5x − casc_v5x @ eval_v8 DT echo（负=级联更好）"),
    ("逐行配对：B 档级联 vs 纯 NN", ("model_pairs", "nn_v5x-casc_v5x", "eval_v8", "fest.echo_mos"),
     "nn_v5x − casc_v5x @ eval_v8 FEST echo"),
    ("逐行配对：B 档级联 vs 纯 NN", ("model_pairs", "nn_v5x-casc_v5x", "eval_rir", "dt.echo_mos"),
     "nn_v5x − casc_v5x @ eval_rir DT echo（真实 IR 上反转）"),
    ("逐行配对：B 档级联迭代", ("model_pairs", "casc_v5-casc_v5x", "eval_v8", "dt.echo_mos"),
     "casc_v5 − casc_v5x @ eval_v8 DT echo"),
    ("逐行配对：B 档级联迭代", ("model_pairs", "casc_v5-casc_v5x", "eval_v8", "dt.deg_mos"),
     "casc_v5 − casc_v5x @ eval_v8 DT deg"),
    ("逐行配对：B 档级联迭代", ("model_pairs", "casc_v5-casc_v5x", "eval_v8", "fest.echo_mos"),
     "casc_v5 − casc_v5x @ eval_v8 FEST echo（训练侧零样本场景）"),
    ("逐行配对：B 档级联 vs v4", ("model_pairs", "v4-casc_v5x", "eval_v8", "dt.deg_mos"),
     "v4 − casc_v5x @ eval_v8 DT deg"),
    ("逐行配对：B 档级联的 RIR 轴", ("cross_set", "casc_v5x", "eval_rir-eval_rir_sim.dt.echo_mos"),
     "casc_v5x @ eval_rir − eval_rir_sim DT echo（与纯 NN 反向）"),
]
ERLE_SPEC = [
    ("ERLE 三分报（FEST，eval_v8，dB）", ("erle", "v4", "eval_v8"), "v4"),
    ("ERLE 三分报（FEST，eval_v8，dB）", ("erle", "trad", "eval_v8"), "trad"),
    ("ERLE 三分报（FEST，eval_v8，dB）", ("erle", "nn_v5", "eval_v8"), "nn_v5"),
    ("ERLE 三分报（FEST，eval_v8，dB）", ("erle", "nn_v5x", "eval_v8"), "nn_v5x"),
    ("ERLE 三分报（FEST，eval_v8，dB）", ("erle", "casc_v5x", "eval_v8"), "casc_v5x（B 档）"),
    ("ERLE 三分报（FEST，eval_v8，dB）", ("erle", "casc_v5", "eval_v8"), "casc_v5（B 档）"),
]


def _get(d: dict, path: tuple):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path,
                    default=Path("exp/07_nn_aec/outputs/eval_matrix/_report/matrix_ci.json"))
    ap.add_argument("--md", type=Path,
                    default=Path("exp/07_nn_aec/outputs/eval_matrix/_report/numbers_candidates.md"))
    ap.add_argument("--qa", type=Path,
                    default=Path("exp/07_nn_aec/outputs/eval_matrix/_report/qa.json"))
    args = ap.parse_args()
    d = json.loads(args.json.read_text(encoding="utf-8"))
    L = ["| 组 | 条目 | 值 / 差（95%CI） | 口径 |", "|---|---|---|---|"]
    for group, path, note in ROWSPEC:
        c = _get(d, path)
        if not c:
            continue
        L.append(f"| {group} | {note} | **{c['mean']:.3f}** ± {c['ci_half']:.3f}（n={c['n']}） | "
                 f"均值±解析 95%CI |")
    for group, path, note in PAIRSPEC:
        c = _get(d, path)
        if not c:
            continue
        star = "✅显著" if c.get("sig") else "n.s."
        L.append(f"| {group} | {note} | **{c['delta']:+.3f}** [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}]"
                 f"（{star}） | 逐行配对 + bootstrap B=2000 seed=777 |")
    for group, path, note in ERLE_SPEC:
        c = _get(d, path)
        if not c:
            continue
        L.append(f"| {group} | {note} | full {c['erle_full']['mean']:.2f} / transient "
                 f"{c['erle_transient']['mean']:.2f} / steady {c['erle_steady']['mean']:.2f}"
                 f"±{c['erle_steady']['ci_half']:.2f} | ref=mic，三分报 |")
    if args.qa.is_file():
        qa = json.loads(args.qa.read_text(encoding="utf-8"))
        L.append(f"| 数据质量 | 饱和格（FEST deg / NST echo 在 5.0 附近且 sd<0.02） | "
                 f"{len(qa.get('saturated', []))} 格 | 不得据此下结论 |")
        parts = []
        for k, v in qa.get("ref_check", {}).items():
            mx = max(v["max_abs_echo"], v["max_abs_deg"])
            parts.append(f"{k}: {v['n']} 行 max|Δ|={mx:.6f}")
        L.append(f"| 数据质量 | 与既有口径抽查对账 | {'; '.join(parts)} | 逐行 |")
    out = "\n".join(L) + "\n"
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(out, encoding="utf-8", newline="\n")
    print(f"[numbers] {len(L) - 1} 行候选 -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
