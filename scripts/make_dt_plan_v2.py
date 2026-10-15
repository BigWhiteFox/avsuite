#!/usr/bin/env python
"""派生 DT 比例清单 v2：在 v1 的两个历史 preset 之上，加"SNR 域外"两档 preset（t23 ③）。

为什么要新版本而不是改 v1
--------------------------
`configs/README.md` §5 的版本纪律：**换配比/换区间 = 新版本**（`dt_plan_v2.json` + `version`
字段），旧集读数不可与新清单混用。本脚本因此不改 `dt_plan_v1.json`，而是从它派生 v2：
- `presets` 原样保留 v1 的两个历史 preset（`train_v1_repro` / `eval_v5_repro`，
  `verify_dt_plan.py` 的清单结构检查也要求它们存在）；
- 新增两档（其余字段与 `eval_v5_repro` **逐字段相同**，只改 `dt_snr_db` 区间）：
  - `eval_snr_low`  ：`dt_snr_db = 0..10`（配合 `--snr-range 0,10` ⇒ 全场景低 SNR 域外）
  - `eval_snr_high` ：`dt_snr_db = 30..40`（配合 `--snr-range 30,40` ⇒ 全场景高 SNR 域外）
- 写入 `derived_from`（v1 路径 + sha256 + 派生规则）与 `history` 继承说明，供收档核对。

用法
----
    python scripts/make_dt_plan_v2.py                       # -> configs/dt_plan_v2.json
    python scripts/make_dt_plan_v2.py --out <path> --check   # 只校验既有 v2 与 v1 的派生关系

边界：只新增清单文件；不动 v1、不动任何既有 eval 集；采样口径（单区间均匀 + 固定顺序
ser->snr->leadin）与 v1 完全一致，故随机流结构与旧配方同构。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import dt_plan as DP  # noqa: E402

V1 = REPO / "configs" / "dt_plan_v1.json"
V2 = REPO / "configs" / "dt_plan_v2.json"
SNR_PRESETS = {
    "eval_snr_low": (0.0, 10.0, "eval_snr_low（t23 ③ SNR 域外·低档；配 --snr-range 0,10）"),
    "eval_snr_high": (30.0, 40.0, "eval_snr_high（t23 ③ SNR 域外·高档；配 --snr-range 30,40）"),
}


def build(v1: dict, v1_sha: str) -> dict:
    plan = json.loads(json.dumps(v1))                    # 深拷贝
    plan["version"] = "dt_plan_v2"
    plan["created"] = "2026-10-13"
    plan["task"] = (v1["task"] + "；t23 追加 SNR 域外两档 preset"
                    "（REMEDIATION_PLAN_V3.1.md §3.2 多轴评测）")
    base = plan["presets"]["eval_v5_repro"]
    for name, (lo, hi, used_by) in SNR_PRESETS.items():
        ps = json.loads(json.dumps(base))                # 以 eval_v5_repro 为底
        ps["dt_snr_db"] = {"kind": "uniform", "lo": lo, "hi": hi}
        ps["used_by"] = used_by
        ps["source"] = ("t23 任务书（队长 2026-10-13 指派）：低档 0–10 / 高档 30–40，"
                        "其余字段同 eval_v5_repro")
        plan["presets"][name] = ps
    plan["derived_from"] = {
        "path": "configs/dt_plan_v1.json",
        "sha256": v1_sha,
        "rule": ("presets 原样继承 v1；新增 eval_snr_low/high = eval_v5_repro 逐字段复制后"
                 "仅改 dt_snr_db 区间；schema/采样顺序不变"),
        "history": "继承 v1 的只读审计区（不参与采样）",
    }
    return plan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=V2)
    ap.add_argument("--check", action="store_true",
                    help="只校验 --out 确实等于由 V1 派生的结果（不写盘）")
    args = ap.parse_args()

    v1, v1_sha = DP.load_plan(V1)
    plan = build(v1, v1_sha)
    DP.validate_plan(plan)                                # schema/version/preset 结构
    text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        got = args.out.read_text(encoding="utf-8")
        if got != text:
            print(f"[make_dt_plan_v2] {args.out} 与派生结果不一致（v1 sha256={v1_sha[:12]}…）",
                  file=sys.stderr)
            return 1
        print(f"[make_dt_plan_v2] OK：{args.out} == 由 {V1.name}（sha256 {v1_sha[:12]}…）派生的结果")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    out_sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    loaded, sha2 = DP.load_plan(args.out)                 # 回读校验
    print(f"[make_dt_plan_v2] 写出 {args.out}（version={loaded['version']}，sha256 {out_sha[:16]}…）")
    print(f"  derived_from v1 sha256 {v1_sha[:16]}…；presets={sorted(loaded['presets'])}")
    for name in SNR_PRESETS:
        rp = DP.resolve_preset(loaded, name, sha2)
        print(f"  {name}: DT 占比 {rp['scenarios']['dt']} | 引导 "
              f"{DP.dist_repr(rp['dt_leadin_sec'])}s | SER {DP.dist_repr(rp['dt_ser_db'])}dB"
              f" | SNR {DP.dist_repr(rp['dt_snr_db'])}dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
