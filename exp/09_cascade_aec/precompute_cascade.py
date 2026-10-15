"""预计算级联 AEC 训练/验证数据(P1-1 / 级联=换输入对):

对 aec_synth 目录每条 clip 跑 TradAEC(MDF, partitions=8, mu=0.7——与 WP-06 评估、
WP-08 precompute_lin 完全同款线性级),取
    y_echo = 线性回声估计(已经 GCC-PHAT 对齐的滤波器输出)
    e_lin  = mic − y_echo(MDF 误差信号,RES 之前)

产出与 aec_synth 完全同构的三槽目录,AecDataset(dataset.py,只读复用)直接可读:
    mic/     <- e_lin    (NN 的 mic 输入:线性级擦完主回声后的残余)
    farend/  <- y_echo   (NN 的 far 参考:已对齐回声估计,NN 免学时延/回声路径)
    nearend/ <- 原目标(逐字节复制,目标不动)
    manifest.csv 原样复制(scenario 标签不变,dt,nst 过滤照常可用)。

    wav 子类型 = FLOAT:实测 40 条随机 clip 中 4 条 y_echo 峰值 >1.0(max 1.78)、
    3 条 e_lin >1.0(max 1.48),PCM_16 会削波(≈0.03% 样本)且与推理期内存
    float32 不一致——FLOAT 免量化免削波,dataset.py 按 dtype='float32' 读取,
    与子类型无关。--force 覆盖已有输出(断点续跑时不传)。

用法:
    python precompute_cascade.py --synth-dir D:/data/aec-synth/train_v1 \
        --out-dir D:/data/aec-synth/train_cascade_lin

P1-3b 追加（DM 化级联，2026-10-13）
----------------------------------
- ``--fresh-instance-per-clip``：**每条 clip 新建 TradAEC 实例**（DM 训练同约定，见
  `cascade_dm.AecCascadeDynamicMix`）。缺省仍是历史口径（跨 clip 复用同一实例，逐位
  兼容既有产物）；**val 侧若用于 DM 训练的 `--val`，必须带本开关**，否则训练/MDF
  约定不一致。策略与实际参数写进输出目录的 `cascade_lin_profile.json`。
- ``--measure N``：只测前 N 条 clip 的**单条线性级耗时**（含实例创建，单位为 ms/clip），
  打印 mean/p50/p95 与折算供给速率，**不写任何文件**——供 D1 吞吐预检把"合成+MDF"
  口径折算进去（t11/train-eng 用）。
"""
import argparse
import csv
import json
import shutil
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_trad_aec"))

from pipeline import TradAEC  # noqa: E402


def _read_clip(root, name, sr=16000):
    """读一条 clip 的 (mic, far) float64（含采样率校验，与写盘口径同源）。"""
    mic, sr_m = sf.read(root / "mic" / f"{name}.wav", dtype="float32")
    far, sr_f = sf.read(root / "farend" / f"{name}.wav", dtype="float32")
    assert sr_m == sr_f == sr, \
        f"{name}: 采样率不一致 mic={sr_m} far={sr_f}（期望 {sr}）"
    n = min(len(mic), len(far))
    return mic[:n].astype(np.float64), far[:n].astype(np.float64)


def measure(synth_dir, n_measure, partitions, mu):
    """单条线性级耗时实测（含 TradAEC 实例创建；不写盘）。返回统计 dict。"""
    with open(Path(synth_dir) / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = [r["name"] for r in rows][:n_measure]
    times, durs = [], []
    for name in names:
        mic, far = _read_clip(Path(synth_dir), name)
        t0 = time.perf_counter()
        aec = TradAEC(method="mdf", partitions=partitions, mu=mu)   # 每条 clip 新建（DM 约定）
        aec.process(mic, far)
        times.append((time.perf_counter() - t0) * 1000.0)           # ms/clip
        durs.append(len(mic) / 16000.0)
    stats = {
        "clips": len(times),
        "seg_sec_mean": round(float(np.mean(durs)), 3),
        "ms_per_clip_mean": round(float(np.mean(times)), 2),
        "ms_per_clip_p50": round(float(statistics.median(times)), 2),
        "ms_per_clip_p95": round(float(np.percentile(times, 95)), 2),
        "clips_per_sec_1worker": round(1000.0 / float(np.mean(times)), 2),
        "trad_params": {**asdict(TradAEC()), "partitions": partitions, "mu": mu},
        "trad_instance_policy": "fresh_per_clip",
        "note": "含实例创建；不写盘。合成侧速率另由 scripts/dm_aec_synth.py --dry-run 给",
    }
    print(f"[measure] {stats['clips']} 条（seg {stats['seg_sec_mean']}s）：线性级 "
          f"mean {stats['ms_per_clip_mean']} ms/clip（p50 {stats['ms_per_clip_p50']} / "
          f"p95 {stats['ms_per_clip_p95']}）→ 单 worker {stats['clips_per_sec_1worker']} clip/s")
    print(f"[measure] 口径：TradAEC(mdf, partitions={partitions}, mu={mu})，"
          f"trad_instance_policy=fresh_per_clip（每 clip 新建实例）；"
          f"供 D1 吞吐按'合成+MDF'折算")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="三槽输出目录；--measure 模式下不需要")
    ap.add_argument("--partitions", type=int, default=8)
    ap.add_argument("--mu", type=float, default=0.7)
    ap.add_argument("--force", action="store_true", help="覆盖已有输出(默认跳过三槽齐的 clip)")
    ap.add_argument("--fresh-instance-per-clip", action="store_true",
                    help="每条 clip 新建 TradAEC 实例（P1-3b：DM 训练同约定；缺省=历史口径"
                         "跨 clip 复用同一实例，逐位兼容既有产物）")
    ap.add_argument("--measure", type=int, default=0, metavar="N",
                    help="只测前 N 条 clip 的线性级 ms/clip（含实例创建，不写盘）后退出")
    args = ap.parse_args()

    root, out = args.synth_dir, args.out_dir
    SR = 16000                       # V5 v3b-046：读入校验与写盘统一用同一常量
    if args.measure:
        measure(root, args.measure, args.partitions, args.mu)
        return 0
    if out is None:
        raise SystemExit("precompute_cascade: 需要一个 --out-dir（或改用 --measure N 只测耗时）")
    with open(root / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for sub in ("mic", "farend", "nearend"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # 线性级实例策略（P1-3b）：缺省 = 历史口径（跨 clip 复用）；--fresh-instance-per-clip = DM 约定
    aec = None if args.fresh_instance_per_clip else TradAEC(
        method="mdf", partitions=args.partitions, mu=args.mu)
    t0 = time.perf_counter()
    n_done = n_skip = 0
    for i, r in enumerate(rows):
        name = r["name"]
        p_mic = out / "mic" / f"{name}.wav"
        p_far = out / "farend" / f"{name}.wav"
        p_near = out / "nearend" / f"{name}.wav"
        if not args.force and p_mic.exists() and p_far.exists() and p_near.exists():
            n_skip += 1
            continue
        mic, sr_m = sf.read(root / "mic" / f"{name}.wav", dtype="float32")
        far, sr_f = sf.read(root / "farend" / f"{name}.wav", dtype="float32")
        # V5 v3b-046：读入采样率必须显式校验。旧实现写盘硬编码 16000 而读入完全不校验 →
        # 44.1k/48k 数据集会被静默按 16k 处理（时延/ERLE 全错且无任何告警）。
        assert sr_m == sr_f == SR, \
            f"{name}: 采样率不一致 mic={sr_m} far={sr_f}（期望 {SR}）"
        n = min(len(mic), len(far))
        mic, far = mic[:n], far[:n]
        if args.fresh_instance_per_clip:
            aec = TradAEC(method="mdf", partitions=args.partitions, mu=args.mu)
        res = aec.process(mic.astype(np.float64), far.astype(np.float64))
        y_echo = res.y_echo[:n].astype(np.float32)
        e_lin = (mic - y_echo).astype(np.float32)   # = res.error(MDF 误差,RES 前)
        sf.write(p_mic, e_lin, SR, subtype="FLOAT")
        sf.write(p_far, y_echo, SR, subtype="FLOAT")
        shutil.copyfile(root / "nearend" / f"{name}.wav", p_near)   # 目标逐字节复制
        n_done += 1
        if (i + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  {i + 1}/{len(rows)}  {el:.0f}s  "
                  f"({el / max(n_done, 1) * 1000:.0f}ms/clip)", flush=True)
    shutil.copyfile(root / "manifest.csv", out / "manifest.csv")
    el = time.perf_counter() - t0
    policy = "fresh_per_clip" if args.fresh_instance_per_clip else "shared_across_clips"
    profile = {
        "producer": "exp/09_cascade_aec/precompute_cascade.py",
        "source_synth_dir": str(root), "out_dir": str(out),
        "clips": {"done": n_done, "skipped": n_skip, "total": len(rows)},
        "elapsed_sec": round(el, 1),
        "trad_params": {**asdict(TradAEC()), "method": "mdf",
                        "partitions": args.partitions, "mu": args.mu},
        "trad_instance_policy": policy,
        "equals_dm_training_policy": bool(args.fresh_instance_per_clip),
        "wav_subtype": "FLOAT",
        "e_lin_formula": "e_lin = mic − y_echo（TradAEC.process 的 y_echo，RES 前）",
        "note": ("P1-3b：DM 训练（cascade_dm.AecCascadeDynamicMix）逐 clip 新建实例；"
                 "val 侧 --val 若服务 DM 训练，必须带 --fresh-instance-per-clip 以对齐约定。"
                 "shared_across_clips 与 fresh_per_clip 不等价（前者 MDF 谱队列/对齐锁存/"
                 "DTD EMA 跨 clip 延续），两套读数不直接混比"),
    }
    with open(out / "cascade_lin_profile.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[cascade_lin] {root.name}: {n_done} done + {n_skip} skipped / {len(rows)} "
          f"-> {out}  subtype=FLOAT  policy={policy}  {el:.0f}s", flush=True)
    print(f"[cascade_lin] profile -> {out / 'cascade_lin_profile.json'}"
          f"（含 TradAEC 生效参数与实例策略；P1-3b）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
