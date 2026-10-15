#!/usr/bin/env python
"""DT 合成比例清单（阶段0 工作项 0-5）：双讲参数显式化、可读取、可复现同分布。

为什么需要它
------------
`scripts/aec_synth.py` 的 DT（双讲）合成口径此前是**命令行默认值 + 分散在报告里的历史命令**：
FEST/NST/DT 场景配比、DT 近端引导（引导期内近端为 0，故 `dur - leadin` 即"双讲重叠段"）、
DT 的 SER（near/echo 比）与 SNR（near/noise 比）区间。DM 化（REMEDIATION_PLAN_V3.1 §3.2）
改为训练时现合成后，这些默认值不再有 manifest 兜底，于是存在计划点名的风险：
"默认比例不可追溯"。本模块把上述参数抽成一个带**版本号**的 JSON 清单：

    configs/dt_plan_v1.json     （清单本体；随仓 scp，训练/评测同源引用）

并由 `aec_synth.py --dt-plan <json>` 显式采样；生成物 manifest 追加
`dt_plan_version` / `dt_plan_preset` 两列留痕（**缺省不加列、行为逐比特不变**）。

清单内容（每条 DT 相关参数都有来源，不写"默认值"这种不可追溯的表述）
--------------------------------------------------------------------
- ``scenarios``：FEST/NST/DT 场景配比 = **每场景 DT 占比**；
- ``dt_leadin_sec``：DT 近端引导时长分布（引导段纯远端，用于自适应滤波收敛；
  重叠段 = ``dur_sec - leadin``）。v1 记为**点分布**（退化）——它精确刻画历史行为，
  又为将来改成区间/三角分布留了同一 schema；
- ``dt_ser_db``：DT 回声相对近端的 SER 分布（near 相对 noise 定 SNR 后，
  echo 相对 near 定 -SER；SER 越大回声越弱）；
- ``dt_snr_db``：DT 近端相对噪声的 SNR 分布；
- ``history``：**只读审计区**——各历史合成集的实测占比与分布统计（含出处的历史命令），
  供"清单 vs 实际"核对，**不参与采样**。

口径纪律（可与 0-4 对齐的可复现性）
-----------------------------------
1. **默认路径逐比特不变**：不给 ``--dt-plan`` 时完全不读清单、不追加列、不额外抽随机数。
2. **采样只换范围、不换顺序**：``--dt-plan`` 下 SER/SNR 仍各消耗一次 ``rng.uniform``
   （单区间均匀分布，与旧口径同型），因此 DT 波形的随机流与老配方**同构**；
   同 seed + 同参数 ⇒ 同 manifest、同波形（由 ``scripts/verify_dt_plan.py`` 回放核对）。
3. 分布种类目前只支持 ``uniform``（点分布 = lo == hi 的退化情形）；未知 kind 直接报错，
   宁可停也不静默用别的分布。

用法
----
    # 生成/覆盖清单（把当前 CLI 默认值写死成 v1；含实测历史审计区）
    python scripts/dt_plan.py --write configs/dt_plan_v1.json --history-from-dir D:/data/aec-synth

    # 查看清单摘要（供下游选 preset）
    python scripts/dt_plan.py --show configs/dt_plan_v1.json

    # 合成时显式按清单采样
    python scripts/aec_synth.py --num 300 ... --dt-plan configs/dt_plan_v1.json \\
        --dt-plan-preset eval_v5_repro

    # 校验（清单结构 + 与 manifest 的分布一致性 + 可复现回放）
    python scripts/verify_dt_plan.py --plan configs/dt_plan_v1.json --manifest <set>/manifest.csv \\
        --set-dir <set> --replay
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

SCENARIOS = ("fest", "nst", "dt")
SCHEMA = "dt_plan/1"
VERSION_RE = r"^dt_plan_v\d+$"
DIST_KINDS = ("uniform",)

#: v1 默认清单里 preset 的键序（也决定 --write 的采样来源）
DEFAULT_PRESET = "train_v1_repro"


class DtPlanError(ValueError):
    """清单缺失/结构非法/取值不可用（调用方应 fail fast，不静默兜底）。"""


# ---------------------------------------------------------------- 取值原语
def _dist(d, field):
    """取一个分布/标量并归一成 {kind, lo, hi}；标量 x -> {uniform, x, x}（点分布）。"""
    if isinstance(d, dict):
        kind = d.get("kind", "uniform")
        if kind not in DIST_KINDS:
            raise DtPlanError(f"{field}: 不支持的分布 kind={kind!r}"
                              f"（v1 只支持 {DIST_KINDS}）")
        if "lo" not in d or "hi" not in d:
            raise DtPlanError(f"{field}: 分布缺 lo/hi")
        lo, hi = float(d["lo"]), float(d["hi"])
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        kind, lo, hi = "uniform", float(d), float(d)
    else:
        raise DtPlanError(f"{field}: 需为标量或 {{kind,lo,hi}} 字典，实为 {type(d).__name__}")
    if not (lo <= hi):
        raise DtPlanError(f"{field}: lo({lo}) > hi({hi})")
    return {"kind": kind, "lo": lo, "hi": hi}


def dist_repr(d):
    """分布的人读字符串：`0..10`（区间）或 `2.0`（点）。"""
    return f"{d['lo']}..{d['hi']}" if d["lo"] != d["hi"] else f"{d['lo']}"


# ---------------------------------------------------------------- 载入与校验
def load_plan(path):
    """读清单 JSON 并校验；返回 (plan, sha256)。任何结构问题抛 DtPlanError。"""
    p = Path(path)
    if not p.is_file():
        raise DtPlanError(f"清单不存在：{p}")
    raw = p.read_bytes()
    try:
        plan = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise DtPlanError(f"清单不是合法 UTF-8 JSON：{p}（{e}）") from None
    validate_plan(plan)
    return plan, hashlib.sha256(raw).hexdigest()


def validate_plan(plan):
    """结构校验（就地抛 DtPlanError；返回值 = 归一化后的 plan 便于推演）。"""
    if plan.get("schema") != SCHEMA:
        raise DtPlanError(f"schema 应为 {SCHEMA!r}，实为 {plan.get('schema')!r}")
    ver = plan.get("version")
    if not isinstance(ver, str) or not re.match(VERSION_RE, ver):
        raise DtPlanError(f"version 需形如 dt_plan_v1（清单版本号），实为 {ver!r}")
    presets = plan.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise DtPlanError("presets 必须是非空对象")
    for name, ps in presets.items():
        if not isinstance(ps, dict):
            raise DtPlanError(f"preset {name}: 必须是对象")
        sc = ps.get("scenarios")
        if not isinstance(sc, dict) or set(sc) != set(SCENARIOS):
            raise DtPlanError(f"preset {name}.scenarios 必须恰好含 {SCENARIOS}")
        vals = {k: float(v) for k, v in sc.items()}
        if any(v < 0 for v in vals.values()) or abs(sum(vals.values()) - 1.0) > 1e-9:
            raise DtPlanError(f"preset {name}.scenarios 需非负且和为 1，实为 {vals}")
        for field in ("dt_leadin_sec", "dt_ser_db", "dt_snr_db"):
            if field not in ps:
                raise DtPlanError(f"preset {name} 缺字段 {field}")
            _dist(ps[field], f"preset {name}.{field}")
        if "dur_sec" in ps and float(ps["dur_sec"]) <= 0:
            raise DtPlanError(f"preset {name}.dur_sec 需 > 0")
    return plan


def resolve_preset(plan, name=None, sha=None):
    """挑一个 preset 并解析成纯数值字典（采样只需这个）。

    返回 dict：preset/version/plan_sha256/scenarios/dt_leadin_sec/dt_ser_db/dt_snr_db
    （分布字段为 {kind,lo,hi}；scenarios 为三键 float）。
    """
    presets = plan["presets"]
    key = name or DEFAULT_PRESET
    if key not in presets:
        raise DtPlanError(f"preset {key!r} 不在清单中（可选：{sorted(presets)}）")
    ps = presets[key]
    return {
        "version": plan["version"],
        "preset": key,
        "plan_sha256": sha or "",
        "scenarios": {k: float(ps["scenarios"][k]) for k in SCENARIOS},
        "dt_leadin_sec": _dist(ps["dt_leadin_sec"], f"{key}.dt_leadin_sec"),
        "dt_ser_db": _dist(ps["dt_ser_db"], f"{key}.dt_ser_db"),
        "dt_snr_db": _dist(ps["dt_snr_db"], f"{key}.dt_snr_db"),
        "dur_sec": float(ps.get("dur_sec", 0.0)) or None,
    }


# ---------------------------------------------------------------- 采样
def sample_dist(rng, dist):
    """从分布抽一个值（每样本一次 rng 调用，与旧 `rng.uniform(lo, hi)` 同型）。"""
    if dist["kind"] == "uniform":
        return float(rng.uniform(dist["lo"], dist["hi"]))
    raise DtPlanError(f"不支持的分布 kind={dist['kind']!r}")      # pragma: no cover


def sample_dt(rng, rp):
    """按 preset 抽一条 DT 样本的全套参数（顺序固定：ser -> snr -> leadin）。

    顺序是清单的一部分：**换顺序 = 换整条随机流**，故显式写死在此处，
    并由 verify_dt_plan.py 的 --replay 回放核对（同 seed 必须逐行重现）。
    """
    return {
        "ser_db": sample_dist(rng, rp["dt_ser_db"]),
        "snr_db": sample_dist(rng, rp["dt_snr_db"]),
        "leadin_sec": sample_dist(rng, rp["dt_leadin_sec"]),
    }


def sample_ratio(rng, rp):
    """按清单的 scenarios 抽场景（与 `aec_synth.pick_scene` 同型：一次 uniform + 累积比较）。

    本函数供**其它**合成驱动（DM 化、独立集生成）复用；`aec_synth.py --dt-plan` 走的是
    现有 main 循环里的 `pick_scene(rng, ratio)`——两者同型，故随机流一致。
    """
    r = float(rng.random())
    acc = 0.0
    for s in SCENARIOS:
        acc += rp["scenarios"][s]
        if r < acc:
            return s
    return SCENARIOS[-1]


# ---------------------------------------------------------------- 历史审计（只读）
def _quantiles(xs, qs=(0.05, 0.5, 0.95)):
    if not xs:
        return {}
    xs = sorted(xs)
    out = {}
    for q in qs:
        pos = q * (len(xs) - 1)
        lo = int(pos)
        frac = pos - lo
        hi = min(lo + 1, len(xs) - 1)
        out[f"p{int(q * 100)}"] = round(xs[lo] * (1 - frac) + xs[hi] * frac, 4)
    return out


def measure_manifest(path):
    """实测一份合成 manifest 的场景占比与 DT 参数分布（只读审计用）。"""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    if not rows:
        raise DtPlanError(f"manifest 为空：{path}")
    n = len(rows)
    by = {s: [r for r in rows if r["scenario"] == s] for s in SCENARIOS}
    dt = by["dt"]

    def nums(rs, k):
        return [float(r[k]) for r in rs if r.get(k) not in (None, "", "None")]

    lead = nums(dt, "leadin_sec")
    out = {
        "manifest": str(path),
        "rows": n,
        "scene_counts": {s: len(by[s]) for s in SCENARIOS},
        "scene_frac": {s: round(len(by[s]) / n, 4) for s in SCENARIOS},
        "dt_ser_db": {"min": round(min(nums(dt, "ser_db")), 4),
                      "max": round(max(nums(dt, "ser_db")), 4),
                      "mean": round(statistics.mean(nums(dt, "ser_db")), 4)},
        "dt_snr_db": {"min": round(min(nums(dt, "snr_db")), 4),
                      "max": round(max(nums(dt, "snr_db")), 4),
                      "mean": round(statistics.mean(nums(dt, "snr_db")), 4)},
        "leadin_sec_unique": sorted({round(v, 6) for v in lead}),
        "delay_ms": {"min": round(min(nums(rows, "delay_ms")), 3),
                     "max": round(max(nums(rows, "delay_ms")), 3)},
        "soft_clip_frac": round(statistics.mean(nums(rows, "soft_clip")), 4),
        "dur_sec": None,
    }
    return out


# ---------------------------------------------------------------- 写清单
def build_plan(history_dir=None, sets=None):
    """构造 v1 清单（两个 preset：训练配方 / 实测评测配方；可选附实测审计区）。

    preset 的数值来源（**不是** CLI 默认值，是各自留档过的历史命令）：
      - ``train_v1_repro``：`exp/07_nn_aec/README.md` 的 train_v1 复现命令
        （`--snr-range 15,30 --scenarios 0.4,0.3,0.3 --dt-leadin-sec 1.5`，SER 用默认 0,10）；
      - ``eval_v5_repro``：`reports/eval_v5_dt300.md` / `reports/eval_v7_fest_nst.md`
        （`--snr-range 20,30 --dt-leadin-sec 2 --scenarios 0.25,0.25,0.5`，SER 用默认 0,10）。
    """
    plan = {
        "schema": SCHEMA,
        "version": "dt_plan_v1",
        "created": "2026-10-13",
        "task": "0-5 DT 合成比例清单化（REMEDIATION_PLAN_V3.1.md §2/§3.2）",
        "notes": [
            "值语义：dt_ser_db = DT 的 SER（P_near/P_echo，越大回声越弱）；"
            "dt_snr_db = DT 近端相对噪声的 SNR；dt_leadin_sec = DT 近端引导时长，"
            "双讲重叠段 = dur_sec - dt_leadin_sec。",
            "采样口径：单区间均匀，每样本每参数消耗一次 rng（与旧 rng.uniform 同型）；"
            "采样顺序固定 ser -> snr -> leadin，换顺序等于换整条随机流。",
            "缺省（不给 --dt-plan）时 aec_synth.py 不读本清单、不追加列、行为逐比特不变。",
            "media/dataset 无关：本清单只描述 DT 合成分布，不含噪声/语音池切分（见 split_lists/）。",
        ],
        "presets": {
            "train_v1_repro": {
                "scenarios": {"fest": 0.4, "nst": 0.3, "dt": 0.3},
                "dt_leadin_sec": {"kind": "uniform", "lo": 1.5, "hi": 1.5},
                "dt_ser_db": {"kind": "uniform", "lo": 0.0, "hi": 10.0},
                "dt_snr_db": {"kind": "uniform", "lo": 15.0, "hi": 30.0},
                "dur_sec": 4.0,
                "used_by": "train_v1 / train_v1_drift（seed 21；train_v1_drift 逐行同底座）",
                "source": "exp/07_nn_aec/README.md §训练数据复现命令",
            },
            "eval_v5_repro": {
                "scenarios": {"fest": 0.25, "nst": 0.25, "dt": 0.5},
                "dt_leadin_sec": {"kind": "uniform", "lo": 2.0, "hi": 2.0},
                "dt_ser_db": {"kind": "uniform", "lo": 0.0, "hi": 10.0},
                "dt_snr_db": {"kind": "uniform", "lo": 20.0, "hi": 30.0},
                "dur_sec": 8.0,
                "used_by": "eval_v5 / eval_ood_delay / eval_v6_realrir / eval_v7",
                "source": "reports/eval_v5_dt300.md、reports/eval_v7_fest_nst.md（同配方）",
            },
        },
        "history": {
            "role": "只读审计区：实测统计，不参与采样；用于核对清单与历史实际是否一致",
            "measured": {},
        },
    }
    if history_dir:
        sets = sets or ("train_v1", "train_v1_drift", "eval_v5", "eval_v7")
        measured = {}
        root = Path(history_dir)
        for name in sets:
            mp = root / name / "manifest.csv"
            if mp.is_file():
                measured[name] = measure_manifest(mp)
        plan["history"]["history_from_dir"] = str(root)
        plan["history"]["measured"] = measured
    return plan


# ---------------------------------------------------------------- CLI
def cmd_write(args):
    plan = build_plan(args.history_from_dir)
    out = Path(args.write)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"[dt_plan] 写出 {out}")
    print(f"  version={plan['version']} schema={plan['schema']} "
          f"presets={sorted(plan['presets'])}")
    print(f"  仅文件名哈希无关；清单 sha256={hashlib.sha256(text.encode('utf-8')).hexdigest()}")
    if plan["history"]["measured"]:
        for name, m in plan["history"]["measured"].items():
            print(f"  实测 {name}: n={m['rows']} frac={m['scene_frac']} "
                  f"leadin={m['leadin_sec_unique']} snr={m['dt_snr_db']['min']}-{m['dt_snr_db']['max']}")
    return 0


def cmd_show(args):
    plan, sha = load_plan(args.show)
    print(f"[dt_plan] {args.show}")
    print(f"  schema={plan['schema']} version={plan['version']} sha256={sha}")
    for name, ps in plan["presets"].items():
        rp = resolve_preset(plan, name, sha)
        print(f"  preset {name}:")
        print(f"    scenarios={rp['scenarios']} (DT 占比 {rp['scenarios']['dt']})")
        print(f"    leadin={dist_repr(rp['dt_leadin_sec'])}s "
              f"ser={dist_repr(rp['dt_ser_db'])}dB snr={dist_repr(rp['dt_snr_db'])}dB"
              + (f" dur={rp['dur_sec']}s" if rp["dur_sec"] else ""))
        print(f"    used_by={ps.get('used_by', '')}")
    hist = (plan.get("history") or {}).get("measured") or {}
    if hist:
        print(f"  history（只读审计，{len(hist)} 集）：")
        for name, m in hist.items():
            print(f"    {name}: n={m['rows']} frac={m['scene_frac']} "
                  f"leadin={m['leadin_sec_unique']}")
    return 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 默认 GBK 控制台
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="JSON", help="写出/覆盖清单（含实测历史审计区）")
    g.add_argument("--show", metavar="JSON", help="打印清单摘要（presets + 历史实测）")
    ap.add_argument("--history-from-dir", type=Path, default=None,
                    help="--write 时从该目录下的 <集>/manifest.csv 采集实测审计区")
    args = ap.parse_args()
    try:
        return cmd_write(args) if args.write else cmd_show(args)
    except DtPlanError as e:
        print(f"[dt_plan] ERROR {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
