#!/usr/bin/env python
"""compute_ci.py — 为 AECMOS 评估结果补算 95% 置信区间。

读 exp/07_nn_aec/outputs/（含 eval_v4_speaker_heldout、eval_speaker_only_holdout、
aecmos_leadincrop 子目录）与 exp/09_cascade_aec/results/ 下所有 aecmos_*.json 的
逐文件 rows，按 (文件, 场景, 指标) 输出 n / mean / sd / 解析 95%CI（1.96*sd/sqrt(n)）；
并对 NN vs 传统基线的同名配对做配对 bootstrap 95%CI（B=2000, seed=777）。

用法：
    python tools/compute_ci.py            # 打印 markdown 表并写 reports/DATA_CI.md
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIRS = [
    REPO / "exp" / "07_nn_aec" / "outputs",
    REPO / "exp" / "09_cascade_aec" / "results",
]
REPORT = REPO / "reports" / "DATA_CI.md"
GENERATE_DATE = "2026-10-11"
B_BOOT = 2000
SEED = 777
Z = 1.96
METRICS = ["echo_mos", "deg_mos"]

# NN vs 传统基线配对（(nn_json 相对 outputs 的 key, trad_json key)）。
# V5 口径更替（2026-10-11）：全部换为 AECMOS@48k 新口径（aecmos48_v5fix/<set>/aecmos_*.json）；
# 旧口径键（aecmos_nn_v4/aecmos_trad_wp06 等）已随 44 份旧 json 一并归档 _legacy_aecmos16k/。
PAIRS = [
    ("aecmos48_v5fix/eval_v3/aecmos_nn", "aecmos48_v5fix/eval_v3/aecmos_trad"),
    ("aecmos48_v5fix/eval_v5/aecmos_nn", "aecmos48_v5fix/eval_v5/aecmos_trad"),
    ("aecmos48_v5fix/eval_v7/aecmos_nn", "aecmos48_v5fix/eval_v7/aecmos_trad"),
    ("aecmos48_v5fix/eval_ood_delay/aecmos_nn", "aecmos48_v5fix/eval_ood_delay/aecmos_trad"),
    ("aecmos48_v5fix/eval_v6_realrir/aecmos_nn", "aecmos48_v5fix/eval_v6_realrir/aecmos_trad"),
    ("aecmos48_v5fix/eval_v3_drift/aecmos_nn", "aecmos48_v5fix/eval_v3_drift/aecmos_trad"),
    ("aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn", "aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_trad"),
    ("aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn", "aecmos48_v5fix/eval_speaker_only_holdout/aecmos_trad"),
    ("aecmos48_v5fix/eval_v3/aecmos_cascade", "aecmos48_v5fix/eval_v3/aecmos_nn"),
    ("aecmos48_v5fix/eval_v5/aecmos_cascade", "aecmos48_v5fix/eval_v5/aecmos_nn"),
    ("aecmos48_v5fix/eval_v7/aecmos_cascade", "aecmos48_v5fix/eval_v7/aecmos_nn"),
    ("aecmos48_v5fix/eval_v6_realrir/aecmos_cascade", "aecmos48_v5fix/eval_v6_realrir/aecmos_nn"),
    ("aecmos48_v5fix/eval_v3_drift/aecmos_cascade", "aecmos48_v5fix/eval_v3_drift/aecmos_nn"),
    ("aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_cascade", "aecmos48_v5fix/eval_v4_speaker_heldout/aecmos_nn"),
    ("aecmos48_v5fix/eval_speaker_only_holdout/aecmos_cascade", "aecmos48_v5fix/eval_speaker_only_holdout/aecmos_nn"),
    ("aecmos48_v5fix/eval_ood_delay/aecmos_cascade", "aecmos48_v5fix/eval_ood_delay/aecmos_nn"),
    # ---- P0-5 损失矩阵（0-1，2026-10-13）：各列 vs v4 主配置，同冻结集同 @48k 口径 ----
    # 生成：exp/07_nn_aec/eval_p05_matrix.py；判读见 reports/p0_5_loss_matrix.md
    ("p05/eval_v3/aecmos_lossctrl_240ep", "p05/eval_v3/aecmos_v4"),
    ("p05/eval_v5/aecmos_lossctrl_240ep", "p05/eval_v5/aecmos_v4"),
    ("p05/eval_v3/aecmos_psycho_240ep", "p05/eval_v3/aecmos_v4"),
    ("p05/eval_v5/aecmos_psycho_240ep", "p05/eval_v5/aecmos_v4"),
    ("p05/eval_v3/aecmos_lossctrl_v4replica", "p05/eval_v3/aecmos_v4"),
    ("p05/eval_v5/aecmos_lossctrl_v4replica", "p05/eval_v5/aecmos_v4"),
    # 官方 48k 原生链不入 PAIRS：cascade_official 仅 summary（rows 空）、
    # hard_official_fest 行结构为 subset/variant（无 scenario 键），与 compute_ci 行协议不兼容；
    # 官方数字以 docs/NUMBERS.md 真实数据节 + reports/{official_aecmos,ood_real_echoes}.md 为权威。
]


def find_json(key: str) -> Path | None:
    """key 支持两种形式：仓库根相对路径，或 outputs 目录下的相对路径（如 eval_v4_speaker_heldout/...）。"""
    for base in (REPO, OUT_DIRS[0]):
        p = (base / (key + ".json")).resolve()
        if p.exists():
            return p
    return None


# t22：口径隔离（默认排除 legacy 归档；这些目录不存在时，文件集合与过滤前**逐字节相同**）。
#   背景：AECMOS 16k 直送错配口径的旧产物已移入 exp/07_nn_aec/outputs/_legacy_aecmos16k/
#   （清单见该目录 MANIFEST.csv），新口径为 exp/07_nn_aec/outputs/aecmos48_v5fix/（@48k 模型
#   协议）。若不排除，DATA_CI 会把两套口径混排。
#   过滤规则 1：路径中任一目录名以这些前缀开头即跳过（通用；不依赖具体目录是否存在于本机）。
EXCLUDE_DIR_PREFIXES = ("_legacy", "_archive", "_stale")
#   过滤规则 2：**过渡期**显式跳过清单（仓内相对路径，不含 .json）。这些是 exp/09_cascade_aec/
#   results/ 下的旧口径级联产物——该目录不在本次任务的改动范围（归 09 模块 owner 迁移），
#   故先用显式清单挡住，避免新旧混排；待其迁移后本集合应清空。
LEGACY_FILES = {
    "exp/09_cascade_aec/results/aecmos_cascade",           # eval_v3 60 条旧口径
    "exp/09_cascade_aec/results/aecmos_cascade_long",      # 旧口径（长段）
    "exp/09_cascade_aec/results/aecmos_cascade_v5",        # eval_v5 旧口径
    "exp/09_cascade_aec/results/aecmos_cascade_v6",        # eval_v6 旧口径
    # 注：aecmos_cascade_official（官方 48k 原生链）**不在**此清单，属正确口径，保留。
}


def _is_excluded(rel: str) -> bool:
    """t22 过滤：legacy 归档目录 / 过渡期显式清单 → True（不进入 DATA_CI）。"""
    if any(part.startswith(EXCLUDE_DIR_PREFIXES) for part in rel.split("/")[:-1]):
        return True
    return rel[: -len(".json")] in LEGACY_FILES if rel.endswith(".json") else False


def load_all() -> dict[str, dict]:
    """返回 {label: {scenario: {metric: [values]}}}，label 为相对仓库根的显示路径。

    t22：按 `_is_excluded` 跳过 legacy/归档产物（无此类文件时行为与过滤前完全一致）。
    """
    data = {}
    files = []
    for d in OUT_DIRS:
        files.extend(sorted(d.rglob("aecmos_*.json")))
    for f in files:
        rel = f.relative_to(REPO).as_posix()
        if _is_excluded(rel):
            print(f"[skip] legacy/归档口径: {rel}", file=sys.stderr)
            continue
        obj = json.loads(f.read_text(encoding="utf-8"))
        rows = obj.get("rows")
        if not rows:
            print(f"[skip] 无逐文件 rows: {rel}", file=sys.stderr)
            continue
        agg: dict[str, dict[str, list[float]]] = {}
        for r in rows:
            sc = r.get("scenario", "?")
            for m in METRICS:
                if m in r and r[m] is not None:
                    agg.setdefault(sc, {}).setdefault(m, []).append(float(r[m]))
        data[rel] = agg
    return data


def fmt_ci(mean: float, sd: float, n: int) -> str:
    half = Z * sd / math.sqrt(n)
    return f"{mean:.3f}±{half:.2f}"


def stats_table(data: dict[str, dict]) -> list[str]:
    lines = [
        "## 一、逐文件解析 95%CI（1.96·sd/√n）",
        "",
        "| 文件 | 场景 | 指标 | n | mean | sd | 95%CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for rel in sorted(data):
        for sc in sorted(data[rel]):
            for m in METRICS:
                vals = data[rel].get(sc, {}).get(m)
                if not vals:
                    continue
                n = len(vals)
                mean = sum(vals) / n
                sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
                lines.append(
                    f"| {rel} | {sc} | {m} | {n} | {mean:.3f} | {sd:.3f} | {fmt_ci(mean, sd, n)} |"
                )
    return lines


def paired_bootstrap(data: dict[str, dict]) -> list[str]:
    rng = random.Random(SEED)
    lines = [
        "## 二、NN vs 传统基线配对差值 95%CI（配对 bootstrap，B=2000，seed=777）",
        "",
        "差值 = NN − 传统，按 (name, scenario) 对齐逐文件样本；对 DT 场景给出 echo_mos / deg_mos。",
        "",
        "| 配对（nn vs trad） | 场景 | 指标 | n | Δmean | bootstrap 95%CI | 结论 |",
        "|---|---|---|---|---|---|---|",
    ]
    for nn_key, trad_key in PAIRS:
        nn_p, trad_p = find_json(nn_key), find_json(trad_key)
        if nn_p is None or trad_p is None:
            lines.append(f"| {nn_key} vs {trad_key} | — | — | — | — | — | ⚠️ 文件缺失 |")
            continue
        nn_rel = nn_p.relative_to(REPO).as_posix()
        trad_rel = trad_p.relative_to(REPO).as_posix()
        nn_rows = {  # noqa: E731
            (r["name"], r["scenario"]): r for r in json.loads(nn_p.read_text(encoding="utf-8"))["rows"]
        }
        trad_rows = {
            (r["name"], r["scenario"]): r for r in json.loads(trad_p.read_text(encoding="utf-8"))["rows"]
        }
        common = sorted(set(nn_rows) & set(trad_rows))
        for sc in ("dt",):
            keys = [k for k in common if k[1] == sc]
            for m in METRICS:
                diffs = [
                    float(nn_rows[k][m]) - float(trad_rows[k][m])
                    for k in keys
                    if nn_rows[k].get(m) is not None and trad_rows[k].get(m) is not None
                ]
                if not diffs:
                    continue
                n = len(diffs)
                dmean = sum(diffs) / n
                boots = []
                for _ in range(B_BOOT):
                    s = [diffs[rng.randrange(n)] for _ in range(n)]
                    boots.append(sum(s) / n)
                boots.sort()
                lo, hi = boots[int(0.025 * B_BOOT)], boots[int(0.975 * B_BOOT) - 1]
                sig = "显著（CI 不含 0）" if lo > 0 or hi < 0 else "CI 含 0，不显著"
                pair_name = f"{nn_rel} vs {trad_rel}"
                lines.append(
                    f"| {pair_name} | {sc} | {m} | {n} | {dmean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sig} |"
                )
    return lines


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 默认 GBK 控制台
    data = load_all()
    lines = [
        "# DATA_CI.md — AECMOS 指标 95% 置信区间（自动生成）",
        "",
        f"- 生成日期：{GENERATE_DATE}",
        "- 生成命令：`python tools/compute_ci.py`（仓库根 avsuite/ 下）",
        f"- 方法：解析法 95%CI = mean ± 1.96·sd/√n；配对差值用 bootstrap 95%CI（B={B_BOOT}，seed={SEED}，百分位法）",
        "- 数据源：`exp/07_nn_aec/outputs/**/aecmos_*.json`、`exp/09_cascade_aec/results/aecmos_*.json` 的逐文件 rows",
        "",
    ]
    lines += stats_table(data)
    lines.append("")
    lines += paired_bootstrap(data)
    lines.append("")
    text = "\n".join(lines)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[written] {REPORT}", file=sys.stderr)


if __name__ == "__main__":
    main()
