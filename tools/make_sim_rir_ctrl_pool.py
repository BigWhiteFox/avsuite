#!/usr/bin/env python
"""为 RIR 轴**配对对照**造固定 n 房间的仿真 RIR 池（与真实池同规模、同格式、同截断口径）。

用途（t16 追加，队长 2026-10-13 裁定）
------------------------------------
`eval_rir`（真实车厢 RIR 8 条）与它的对照集必须**逐样本配对**（同 seed、同近/远/噪清单、
同 `--rir-max-ms`、**同池规模**），否则 `--rir-dir` 与仿真档的随机流结构不同、行内容无法对齐。
本脚本按 `scripts/aec_synth.py::rir_simulate_pra`（同一函数、同一房间参数域）现场造 n 个房间，
写成与真实池 **同格式**（16k mono PCM_16、峰值归一到 0.9、保留 0.6 s）的 wav，
供 `aec_synth.py --rir-dir` 使用 —— 于是"真实 vs 仿真"成为**唯一**变量。

用法
----
    python tools/make_sim_rir_ctrl_pool.py --out-dir D:/data/aec-synth/eval_sim_rir_pool \
        --n 8 --seed 20261017 --list-out split_lists/rir_sim_ctrl.txt \
        --exclude-pool D:/data/aec-synth/rir_pool

留痕与自检
----------
- 池内文件与 ``--exclude-pool``（默认 = 训练侧仿真池）**逐字节不重叠**（sha256 比对）；
- 写出名单文件（UTF-8/LF，每行一个文件名）与 ``sim_rir_ctrl_meta.json``
  （seed/n/文件 sha256/房间参数域），供配对实验引用版本。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import aec_synth as S  # noqa: E402

SR = 16000
PEAK = 0.9
TRIM_S = 0.6          # 与真实池一致（真实池 0.6 s，合成时再由 --rir-max-ms 截断）


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("D:/data/aec-synth/eval_sim_rir_pool"))
    ap.add_argument("--n", type=int, default=8, help="房间数（必须与配对真实池的条数相同）")
    ap.add_argument("--seed", type=int, default=20261017, help="池生成种子（独立于集合 seed）")
    ap.add_argument("--list-out", type=Path, default=None, help="名单落盘（UTF-8/LF）")
    ap.add_argument("--meta-out", type=Path, default=None, help="版本哨兵 JSON")
    ap.add_argument("--exclude-pool", type=Path, default=Path("D:/data/aec-synth/rir_pool"),
                    help="必须与它逐字节不重叠（默认训练侧仿真池）")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names, shas = [], {}
    for i in range(args.n):
        h = S.rir_simulate_pra(rng)                      # 与 aec_synth 仿真档同一生成器
        h = np.asarray(h, dtype=np.float64)[: int(SR * TRIM_S)]
        m = float(np.max(np.abs(h)))
        if m > 0:
            h = h * (PEAK / m)
        name = f"sim_room_{i:02d}.wav"
        sf.write(str(args.out_dir / name), h.astype(np.float32), SR, subtype="PCM_16")
        names.append(name)
        shas[name] = sha256(args.out_dir / name)

    excl = {sha256(p) for p in args.exclude_pool.glob("**/*.wav")} if args.exclude_pool.is_dir() \
        else set()
    clash = sorted(n for n, s in shas.items() if s in excl)
    if clash:
        print(f"[make_sim_rir_ctrl_pool] 与 {args.exclude_pool} 有 {len(clash)} 条重复：{clash}",
              file=sys.stderr)
        return 1

    if args.list_out:
        args.list_out.parent.mkdir(parents=True, exist_ok=True)
        args.list_out.write_text("".join(f"{n}\n" for n in sorted(names)), encoding="utf-8",
                                 newline="\n")
    meta = {
        "version": "sim_rir_ctrl_v1",
        "purpose": "eval_rir 的配对对照池（同规模/同格式/同截断口径；唯一变量 = 真实 vs 仿真 RIR）",
        "generator": "scripts/aec_synth.py::rir_simulate_pra（同一函数、同一参数域）",
        "seed": args.seed, "n": args.n, "sr": SR, "subtype": "PCM_16",
        "trim_sec": TRIM_S, "peak": PEAK,
        "files_sha256": shas,
        "excluded_pool": str(args.exclude_pool),
        "excluded_pool_n": len(excl),
        "overlap_with_excluded": len(clash),
    }
    meta_p = args.meta_out or (args.out_dir / "sim_rir_ctrl_meta.json")
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                      newline="\n")
    print(f"[make_sim_rir_ctrl_pool] {args.n} 个房间 -> {args.out_dir}（seed={args.seed}）")
    print(f"  与 {args.exclude_pool} 逐字节重叠：{len(clash)}（训练侧仿真池 {len(excl)} 条）")
    print(f"  名单 -> {args.list_out}；meta -> {meta_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
