#!/usr/bin/env python
"""把 t23 从服务器只读拉回的扩量语料子集，转成 aec_synth 能直接吃的 16k mono PCM_16 池（t23）。

输入（`tools/pull_ext_subsets.py` 的产物，保留原始相对目录）
------------------------------------------------------------
    D:/data/_t23_pull/musan/...            (music|noise|speech/*.wav，多为 16k/44.1k)
    D:/data/_t23_pull/aishell3_{near,far}/train/wav/SSB*/*.wav   (44.1k mono，单句 ~3.2s)
    D:/data/_t23_pull/ls_{near,far}/train-clean-100|360/*/*/*.flac  (16k mono flac，p50 ~14s)

输出（每池扁平化 + 重采样到 16k + PCM_16；池目录只放 `.wav`，清单放 `_manifests/`）
------------------------------------------------------------------------------
    D:/data/aec-synth/ext_pools/{musan_eval15,zh_near,zh_far,ls_near,ls_far}/
    D:/data/aec-synth/ext_pools/_manifests/<pool>.csv      (rel_path,out_name,sr_in,ch,frames,bytes,sha256)
    D:/data/aec-synth/ext_pools/_manifests/summary.json    (每池计数/字节/时长分布/去重统计)

**短句语料的拼接（`POOLS[...]['concat_target_sec']`）**：AISHELL-3 单句 ~3.2 s，若直接进 8 s
窗口会被 `fit_len` 循环补（同一句重复 ⇒ 伪周期远端）。故按**说话人分组、贪心拼到 ≥10 s**
（尾料并入上一段；不跨说话人），拼接段命名 `<说话人>_catNNNN.wav`，清单 `rel_path` 记
"a+b+c" 原始句列表 ⇒ 8 s 窗口不再补环。

用法
----
    python tools/build_ext_eval_pools.py [--pull-dir D:/data/_t23_pull] \
        [--out-root D:/data/aec-synth/ext_pools] [--pools musan_eval15 zh_far ...]

边界：只做格式/采样率/声道规范化（**不做**峰值归一，电平交给合成时的 db_scale）；
同名冲突自动加顶层目录前缀并在清单里留痕；不删任何源文件。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
DEFAULT_PULL = Path("D:/data/_t23_pull")
DEFAULT_OUT = Path("D:/data/aec-synth/ext_pools")
#: 池名 -> dict(拉回目录 / 扩展名 / 可选拼接目标秒数；仅对短文件语料用拼接代替"窗口内循环补")
POOLS = {
    "musan_eval15": {"src": "musan", "exts": (".wav",)},
    "zh_near": {"src": "aishell3_near", "exts": (".wav",), "concat_target_sec": 10.0},
    "zh_far": {"src": "aishell3_far", "exts": (".wav",), "concat_target_sec": 10.0},
    "ls_near": {"src": "ls_near", "exts": (".flac",)},
    "ls_far": {"src": "ls_far", "exts": (".flac",)},
}


def resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR:
        return x
    g = math.gcd(int(sr_in), SR)
    return resample_poly(x, SR // g, int(sr_in) // g)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _chunks_by_speaker(files: list[Path], target_sec: float) -> list[list[Path]]:
    """按上级目录（= 说话人）分组，组内**贪心拼到 ≥ target_sec** 一段（尾料并入上一段）。

    目的：AISHELL-3 单句 ~3.2 s，8 s 窗口会被 fit_len 循环补（重复同一句）——对"AEC 远端"是
    伪周期信号。先按说话人拼接成 ≥10 s 的自然长段，合成侧就不再补环（且不跨说话人）。
    """
    groups: dict[str, list[Path]] = {}
    for p in files:
        groups.setdefault(p.parent.name, []).append(p)
    chunks: list[list[Path]] = []
    for g in sorted(groups):
        cur: list[Path] = []
        acc = 0.0
        for p in sorted(groups[g]):
            cur.append(p)
            acc += sf.info(str(p)).frames / sf.info(str(p)).samplerate
            if acc >= target_sec:
                chunks.append(cur)
                cur, acc = [], 0.0
        if cur:
            if chunks:
                chunks[-1].extend(cur)        # 尾料并入上一段，避免出现短段
            else:
                chunks.append(cur)
    return chunks


def build_pool(pull_dir: Path, out_root: Path, pool: str, src_sub: str,
               exts: tuple[str, ...], concat_target_sec: float | None = None) -> dict:
    src = pull_dir / src_sub
    if not src.is_dir():
        raise FileNotFoundError(f"缺拉回目录：{src}")
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"{src} 下没有 {exts} 文件")
    names = [p.name for p in files]
    dup = {n for n, c in Counter(names).items() if c > 1}
    dest = out_root / pool
    dest.mkdir(parents=True, exist_ok=True)
    rows, durs, srs = [], [], Counter()
    units = (_chunks_by_speaker(files, concat_target_sec) if concat_target_sec
             else [[p] for p in files])
    for ui, unit in enumerate(units):
        waves, srs_in = [], []
        for p in unit:
            data, sr_in = sf.read(str(p), dtype="float32", always_2d=True)
            mono = data.mean(1) if data.shape[1] > 1 else data[:, 0]
            waves.append(resample_to_16k(mono, sr_in))
            srs_in.append(int(sr_in))
        y = np.concatenate(waves) if len(waves) > 1 else waves[0]
        if concat_target_sec:                              # 拼接段用 说话人_catNNN 命名
            out_name = f"{unit[0].parent.name}_cat{ui:04d}.wav"
        else:
            out_name = Path(unit[0].name).with_suffix(".wav").name   # 统一 .wav（池只放 wav）
            if unit[0].name in dup:                        # 同名冲突 -> 加顶层目录前缀
                out_name = f"{unit[0].relative_to(src).parts[0]}_{out_name}"
        sf.write(str(dest / out_name), y.astype(np.float32), SR, subtype="PCM_16")
        durs.append(len(y) / SR)
        srs[srs_in[0]] += 1
        rows.append(dict(rel_path="+".join(p.relative_to(src).as_posix() for p in unit),
                         out_name=out_name, sr_in=srs_in[0], channels=1,
                         frames=int(len(y)), bytes=(dest / out_name).stat().st_size,
                         sha256=_sha256(dest / out_name)))
    man_dir = out_root / "_manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    with open(man_dir / f"{pool}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rel_path", "out_name", "sr_in", "channels",
                                          "frames", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    return {"pool": pool, "src": str(src), "n": len(rows), "out_dir": str(dest),
            "bytes": sum(r["bytes"] for r in rows),
            "sr_in": dict(srs), "dup_names": sorted(dup),
            "dur_sec": {"min": round(min(durs), 3), "max": round(max(durs), 3),
                        "mean": round(sum(durs) / len(durs), 3)},
            "names_sha256": hashlib.sha256(
                ("\n".join(sorted(r["out_name"] for r in rows)) + "\n").encode("utf-8")
            ).hexdigest(),
            "manifest": str(man_dir / f"{pool}.csv")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pull-dir", type=Path, default=DEFAULT_PULL)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pools", nargs="*", default=list(POOLS))
    args = ap.parse_args()
    summary = {}
    try:
        for pool in args.pools:
            spec = POOLS[pool]
            info = build_pool(args.pull_dir, args.out_root, pool, spec["src"], spec["exts"],
                              spec.get("concat_target_sec"))
            summary[pool] = info
            print(f"[{pool}] {info['n']} 文件 / {info['bytes'] / 1e6:.1f} MB；"
                  f"输入采样率 {info['sr_in']}；时长 {info['dur_sec']}s；"
                  f"同名冲突 {len(info['dup_names'])}；"
                  f"拼接目标 {spec.get('concat_target_sec')}")
            print(f"   -> {info['out_dir']}（清单 {info['manifest']}）")
    except (FileNotFoundError, ValueError) as e:
        print(f"[build_ext_eval_pools] 输入错误：{e}", file=sys.stderr)
        return 2
    out = args.out_root / "_manifests" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pools": summary, "sr": SR, "subtype": "PCM_16"},
                              ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"[build_ext_eval_pools] summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
