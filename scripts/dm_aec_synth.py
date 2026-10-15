#!/usr/bin/env python
"""DM 管线冒烟 / 吞吐预检 / 小集预生成（REMEDIATION_PLAN_V3.1 附录B 第 2 步）。

用途（P1-3 配套）：
1. **端到端冒烟**：`--num 100 --dry-run` —— 用训练同一套 DM 数据集（`exp/07_nn_aec/
   dm_dataset.py`）现场合成 100 条，打印吞吐（样本/秒）、场景配比与 DT 分布对账，
   **不落盘**。附录B 用它验收"合成基元 + worker 吞吐"；
2. **吞吐硬检查点**（§4.1/附录B）：`--num N --dry-run` 的样本/秒 × worker 数 vs 训练
   消耗（样本/秒）≥1.2 倍——数字由本脚本给，判定在阶段1 执行；
3. **小集预生成**：给 `--out-dir` 则落盘 (farend/nearend/mic) + `manifest.csv`
   （aec_synth 同列 + `dt_plan_version/dt_plan_preset/nearfar_mutex`）+ `dm_profile.json`，
   用于抽样试听/回归对账。**正式训练不用它**：训练走 `train.py --dynamic-mixing` 现合成。

纪律：
- 池/清单与训练完全同口径（互斥校验复用 `aec_synth` 的 0-3 判据，清单互斥不通过直接退出）；
- 清单版本号 + sha256 + 池规模 + 清单内容哈希全部写进 `dm_profile.json`（可追溯）；
- `--seed` 固定时逐 (seed, idx) 确定（可复现对账）；缺省吃熵（和训练一致）。
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "exp" / "07_nn_aec"))

import aec_synth as S  # noqa: E402
import dm_dataset as DM  # noqa: E402
import dt_plan as DP  # noqa: E402


def build_dataset(args):
    preset, plan_info = DM.load_plan(args.plan, args.preset)
    near, far, noise, pool_info = DM.build_pools_from_args(args)
    seg = float(args.seg_sec or preset["dur_sec"] or 4.0)
    rir_files = DM.load_rir_pool(args.rir_dir, args.rir_list)   # 有清单走清单驱动（t26）
    if args.rir_dir and not rir_files:
        raise SystemExit(f"[dm_aec_synth] --rir-dir 内无 wav：{args.rir_dir}")
    ds = DM.AecDynamicMix(
        near, far, noise, preset, plan_info=plan_info, pool_info=pool_info,
        epoch_len=args.num, seg_sec=seg, rir_files=rir_files,
        rir_max_ms=args.rir_max_ms, rir_dir=args.rir_dir, rir_list=args.rir_list,
        delay_ms_range=(args.delay_min_ms, args.delay_max_ms),
        soft_clip_prob=args.soft_clip_prob, jitter_ms=args.jitter_ms, seed=args.seed)
    return ds, preset, plan_info, pool_info


def print_stats(ds, items, seconds):
    scenes = Counter(it["scenario"] for it in items)
    n = max(len(items), 1)
    print(f"[dm_aec_synth] {len(items)} 条现合成完成：{seconds:.2f}s "
          f"→ {len(items) / max(seconds, 1e-9):.1f} 样本/秒"
          f"（{len(items) * ds.seg_sec / max(seconds, 1e-9):.1f} 语音秒/秒）")
    print(f"  场景分布 {dict(scenes)}（清单目标 {ds.preset['scenarios']}）")
    dt = [it for it in items if it["scenario"] == "dt"]
    if dt:
        ser = [it["ser_db"] for it in dt]
        snr = [it["snr_db"] for it in dt]
        print(f"  DT n={len(dt)}：SER {min(ser):.2f}–{max(ser):.2f}（均值 "
              f"{sum(ser) / len(ser):.2f}，清单 {DP.dist_repr(ds.preset['dt_ser_db'])}）｜"
              f"SNR {min(snr):.2f}–{max(snr):.2f}（均值 {sum(snr) / len(snr):.2f}，清单 "
              f"{DP.dist_repr(ds.preset['dt_snr_db'])}）")
        print(f"  引导 {sorted({it['leadin_sec'] for it in dt})}s"
              f"（清单 {DP.dist_repr(ds.preset['dt_leadin_sec'])}）")
    print(f"  片段 {ds.seg_sec}s，soft-clip 命中 {sum(it['soft_clip'] for it in items)}/"
          f"{n}，delay {min(it['delay_ms'] for it in items):.1f}–"
          f"{max(it['delay_ms'] for it in items):.1f}ms")
    # 吞吐硬检查点（§4.1）：单进程供给速率 × worker 数 才是训练可用供给
    for w in (1, 4, 8):
        print(f"  worker={w} 供给 {w * len(items) / max(seconds, 1e-9):.0f} 样本/秒"
              f"（训练消耗需 ≤ 供给/1.2）")


def write_set(ds, items, out, preset, plan_info, pool_info, args):
    for sub in ("farend", "nearend", "mic"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for i, it in enumerate(items):
        name = f"dm_{i:05d}_{it['scenario']}"
        sf.write(out / "farend" / f"{name}.wav", it["far"].numpy(), S.SR)
        sf.write(out / "nearend" / f"{name}.wav", it["near"].numpy(), S.SR)
        sf.write(out / "mic" / f"{name}.wav", it["mic"].numpy(), S.SR)
        rows.append({"name": name, "scenario": it["scenario"],
                     "rir_src": ds.provenance()["rir_src"],
                     "near_file": "dm", "far_file": "dm", "noise_file": "dm",
                     "delay_ms": it["delay_ms"], "ser_db": it["ser_db"],
                     "snr_db": it["snr_db"], "soft_clip": it["soft_clip"],
                     "leadin_sec": it["leadin_sec"],
                     "nearfar_mutex": "utterance",
                     "dt_plan_version": plan_info["version"],
                     "dt_plan_preset": plan_info["preset"]})
    fields = ["name", "scenario", "rir_src", "near_file", "far_file", "noise_file",
              "delay_ms", "ser_db", "snr_db", "soft_clip", "leadin_sec",
              "nearfar_mutex", "dt_plan_version", "dt_plan_preset"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    profile = {
        "producer": "scripts/dm_aec_synth.py",
        "note": ("DM 预生成/冒烟产物：口径与 train.py --dynamic-mixing 同源。"
                 "verify_dt_plan.py 的 C/G 项针对 aec_synth 的静态随机流，不适用于本产物"),
        "dt_plan": plan_info, "pools": pool_info["pools"],
        "split_lists": pool_info["split_lists"], "mutex": pool_info["mutex"],
        "sampling": {"mode": ds.provenance()["seed_mode"], "seed": args.seed,
                     "num": args.num, "seg_sec": ds.seg_sec,
                     "soft_clip_prob": ds.soft_clip_prob,
                     "delay_ms_range": list(ds.delay_ms_range),
                     "jitter_ms": ds.jitter_ms, "rir_max_ms": ds.rir_max_ms,
                     "rir_src": ds.provenance()["rir_src"]},
        "cli_args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
    }
    with open(out / "dm_profile.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"  产物 -> {out}（manifest.csv {len(rows)} 行 + dm_profile.json）")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # Windows 默认 GBK 控制台
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=100, help="现合成条数（冒烟/预生成）")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="落盘目录；缺省 + --dry-run = 只测吞吐不落盘")
    ap.add_argument("--dry-run", action="store_true", help="不落盘（吞吐/分布预检）")
    ap.add_argument("--near-dir", type=Path, default=None,
                    help="单源 near 池目录（多源用 --near-source 重复给；二者互斥）")
    ap.add_argument("--far-dir", type=Path, default=None, help="单源 far 池目录")
    ap.add_argument("--noise-dir", type=Path, default=None, help="单源 noise 池目录")
    ap.add_argument("--near-list", type=Path, default=None, help="near 池分割清单")
    ap.add_argument("--far-list", type=Path, default=None)
    ap.add_argument("--noise-list", type=Path, default=None,
                    help="训练侧只能用 noise_train70.txt（0-2 纪律）")
    DM.add_dm_source_args(ap)      # P1X-2：多源池（--{near,far,noise}-source 可重复）
    ap.add_argument("--rir-dir", type=Path, default=None)
    ap.add_argument("--rir-list", type=Path, default=None,
                    help="RIR 池清单（训练侧只应吃 split_lists/rir_train70.txt）；"
                         "给了 => 清单驱动并把清单名+sha256 记入 profile（t26）")
    ap.add_argument("--rir-max-ms", type=float, default=100.0)
    ap.add_argument("--plan", type=Path, default=None,
                    help="DT 比例清单（缺省 configs/dt_plan_v1.json）")
    ap.add_argument("--preset", default=None, help="清单 preset（缺省 train_v1_repro）")
    ap.add_argument("--seg-sec", type=float, default=None, help="片段时长（缺省清单 dur_sec）")
    ap.add_argument("--soft-clip-prob", type=float, default=0.15)
    ap.add_argument("--delay-min-ms", type=float, default=10.0)
    ap.add_argument("--delay-max-ms", type=float, default=80.0)
    ap.add_argument("--jitter-ms", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=None,
                    help="固定 = 逐 (seed, idx) 可复现；缺省吃熵（与训练一致）")
    args = ap.parse_args()
    if args.out_dir is None and not args.dry_run:
        raise SystemExit("[dm_aec_synth] 既没给 --out-dir 也没给 --dry-run："
                         "不知道要不要落盘（拒绝静默什么都不做）")

    ds, preset, plan_info, pool_info = build_dataset(args)
    print(f"[dm_aec_synth] 清单 {plan_info['version']}/{plan_info['preset']} "
          f"sha256={plan_info['sha256'][:12]}…｜清单文件 near/far/noise="
          f"{pool_info['split_lists']['near']['n']}/"
          f"{pool_info['split_lists']['far']['n']}/"
          f"{pool_info['split_lists']['noise']['n']}｜互斥校验通过"
          f"（{pool_info['mutex']['level']}）")
    t0 = time.perf_counter()
    items = [ds[i] for i in range(args.num)]
    seconds = time.perf_counter() - t0
    print_stats(ds, items, seconds)
    if args.out_dir is not None and not args.dry_run:
        write_set(ds, items, args.out_dir, preset, plan_info, pool_info, args)
    else:
        print("  --dry-run：不落盘")
    return 0


if __name__ == "__main__":
    sys.exit(main())
