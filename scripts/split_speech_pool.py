#!/usr/bin/env python
"""0-4 语音池 reader 级三分割 + eval 侧 near/far 互斥池（可复现、可落盘、可验证）。

为什么
------
REMEDIATION_PLAN_V3.1.md §3.1：语音池按 **reader（说话人）** 三分——near 池 70% /
far 池 70%（与 near 互斥）/ held-out 15% 评测。0-3 的 ``--nearfar-mutex`` 只做**运行时
校验**（喂错池直接退出），本脚本提供被校验的那份池本身；0-4 的 eval_v8 用 held-out
reader 生成**评测侧 near/far 互斥池**，与训练侧（历史 train_v1 的 VoiceBank 28 说话人、
将来的 DNS reader 训练份额）在 reader 维度上零交集。

切分规则（可独立重算，不依赖扫盘顺序/随机种子）
------------------------------------------------
    reader_bucket  = int(sha256("reader:" + reader_id)[:8], 16) % 1000
    < 700 -> train70（训练份额，阶段1 用）  |  700..849 -> val15（模型选择）
    >= 850 -> holdout15（**评测专用**，训练侧禁区）

holdout reader 再按第二个命名空间分 near/far（本脚本只对 holdout 分 side）：

    side_bucket = int(sha256("nearfar:" + reader_id)[:8], 16) % 1000
    < 500 -> near_eval   |   >= 500 -> far_eval

两个命名空间 + 两个阈值写进 ``speech_split_meta.json``；同一条 reader 只落一桶，
故 ``train70 ∩ holdout15 = ∅``、``near_eval ∩ far_eval = ∅`` 均为**构造性**成立。

产出（默认 ``avsuite/split_lists/``）
------------------------------------
- ``speech_readers_manifest.csv``：全池 reader 权威表
  （``reader,bucket,split,side,n_files,file_list_sha256``）。
- ``speech_readers_{train70,val15,holdout15}.txt``：reader id 名单。
- ``speech_near_eval.txt`` / ``speech_far_eval.txt``：评测侧 near/far 池的**文件名**
  （非路径；每行一个 basename，UTF-8/LF），直接喂 ``aec_synth.py --near-dir/--far-dir``
  的池目录清单。
- ``speech_split_meta.json``：算法/阈值/计数/各名单内容 sha256（版本哨兵）。

用法
----
    # 1) 生成名单
    python scripts/split_speech_pool.py --clean-dir D:/data/dns-challenge-mirror/datasets/clean \
        --out-dir split_lists

    # 2) 物化池目录（默认硬链接；跨卷/无权限时自动退化为复制）
    python scripts/split_speech_pool.py --clean-dir <clean> --out-dir split_lists \
        --materialize-root D:/data/aec-synth/eval_v8_pools \
        --noise-src D:/data/dns-challenge-mirror/datasets/noise \
        --noise-list split_lists/noise_eval15.txt

    # 3) 验收（退出码 0=通过 1=不通过 2=输入错误；含 --reproduce 重算对账）
    python scripts/split_speech_pool.py --clean-dir <clean> --out-dir split_lists --verify

    # 4) 纯预览
    python scripts/split_speech_pool.py --clean-dir <clean> --out-dir split_lists --dry-run

边界（不做的事）
----------------
- 不搬/不删源 wav：``--materialize-root`` 只建池目录（硬链接/复制），源池原样不动。
- 不改既有 eval 集与历史读数：本清单是"以后怎么切"的配置；eval_v8 是唯一消费方。
- 内容级去重不做（与 0-3 的 ``--nearfar-mutex`` 同一边界）：本脚本只按文件名/reader
  切分；改名副本需按 0-2 的 sha256 清单法离线核（评测池实测 0 条，见报告）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

#: 桶分辨率与阈值（与 0-2 噪声切分同分辨率；命名空间前缀不同，互不影响）
BUCKETS = 1000
TRAIN_HI = 700          # bucket < 700        -> train70
VAL_HI = 850            # 700 <= bucket < 850 -> val15；>= 850 -> holdout15
READER_SEED = "reader:"
SIDE_SEED = "nearfar:"
SIDE_HI = 500           # holdout 内部：< 500 -> near_eval，>= 500 -> far_eval
HASH8_LEN = 8

TRAIN_NAME, VAL_NAME, HOLDOUT_NAME = "train70", "val15", "holdout15"
SPLIT_NAMES = (TRAIN_NAME, VAL_NAME, HOLDOUT_NAME)
SIDE_NEAR, SIDE_FAR = "near_eval", "far_eval"
READER_RE = re.compile(r"reader_(\d+)")

READER_MANIFEST = "speech_readers_manifest.csv"
META_NAME = "speech_split_meta.json"
READER_LISTS = {
    TRAIN_NAME: "speech_readers_train70.txt",
    VAL_NAME: "speech_readers_val15.txt",
    HOLDOUT_NAME: "speech_readers_holdout15.txt",
}
SIDE_LISTS = {SIDE_NEAR: "speech_near_eval.txt", SIDE_FAR: "speech_far_eval.txt"}
SCHEME = ("sha256('reader:'+reader_id)[:8] % 1000 -> <700 train70 / 700-849 val15 / "
          ">=850 holdout15；holdout 内 sha256('nearfar:'+reader_id)[:8] % 1000 "
          "-> <500 near_eval / >=500 far_eval")


# ---------------------------------------------------------------- 哈希与解析
def hash_bucket(key: str, prefix: str, mod: int = BUCKETS) -> int:
    """``sha256(prefix + key)[:8] % mod``。只吃字符串，不吃路径/内容/扫盘顺序。"""
    return int(hashlib.sha256((prefix + key).encode("utf-8")).hexdigest()[:HASH8_LEN], 16) % mod


def reader_split(reader: str) -> str:
    b = hash_bucket(reader, READER_SEED)
    if b < TRAIN_HI:
        return TRAIN_NAME
    return VAL_NAME if b < VAL_HI else HOLDOUT_NAME


def reader_side(reader: str) -> str:
    return SIDE_NEAR if hash_bucket(reader, SIDE_SEED) < SIDE_HI else SIDE_FAR


def reader_of(name: str) -> str | None:
    """从 DNS clean 文件名解析 reader id（``..._reader_06709_0.wav``）。"""
    m = READER_RE.search(Path(name).name)
    return m.group(1) if m else None


def content_sha256(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 构建
def scan_pool(clean_dir: Path) -> dict[str, list[str]]:
    """扫池 -> {reader: [文件名...]}（按文件名排序）。无 reader 命名的文件直接报错。"""
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"--clean-dir 不存在或不是目录：{clean_dir}")
    per: dict[str, list[str]] = {}
    orphans = []
    for p in sorted(clean_dir.glob("**/*.wav")):
        r = reader_of(p.name)
        if r is None:
            orphans.append(p.name)
            continue
        per.setdefault(r, []).append(p.name)
    if orphans:
        raise ValueError(f"{len(orphans)} 个 wav 文件名里没有 reader_<id>（例：{orphans[0]}）")
    if not per:
        raise FileNotFoundError(f"--clean-dir 下没有 wav：{clean_dir}")
    return per


def build_split(per_reader: dict[str, list[str]]) -> dict:
    """按哈希把 reader 分桶、holdout 再分 near/far；返回名单集合（纯函数）。"""
    readers = sorted(per_reader)
    by_split = {s: [] for s in SPLIT_NAMES}
    near_files: list[str] = []
    far_files: list[str] = []
    rows = []
    for r in readers:
        sp = reader_split(r)
        side = reader_side(r) if sp == HOLDOUT_NAME else ""
        by_split[sp].append(r)
        files = sorted(per_reader[r])
        if sp == HOLDOUT_NAME:
            (near_files if side == SIDE_NEAR else far_files).extend(files)
        rows.append(dict(reader=r, bucket=hash_bucket(r, READER_SEED), split=sp, side=side,
                         n_files=len(files), file_list_sha256=content_sha256(files)))
    return dict(readers=readers, by_split=by_split, near_files=sorted(near_files),
                far_files=sorted(far_files), rows=rows)


# ---------------------------------------------------------------- 落盘
def _write_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{n}\n" for n in names), encoding="utf-8", newline="\n")


def write_outputs(out_dir: Path, split: dict, clean_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in ((READER_MANIFEST, split["rows"]),):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["reader", "bucket", "split", "side",
                                              "n_files", "file_list_sha256"])
            w.writeheader()
            w.writerows(rows)
    lists = {}
    for sp, fname in READER_LISTS.items():
        _write_list(out_dir / fname, split["by_split"][sp])
        lists[fname] = split["by_split"][sp]
    lists[SIDE_LISTS[SIDE_NEAR]] = split["near_files"]
    _write_list(out_dir / SIDE_LISTS[SIDE_NEAR], split["near_files"])
    lists[SIDE_LISTS[SIDE_FAR]] = split["far_files"]
    _write_list(out_dir / SIDE_LISTS[SIDE_FAR], split["far_files"])
    meta = dict(
        version="speech_split_v1", scheme=SCHEME,
        source={"clean_dir": str(clean_dir)},
        thresholds={"buckets": BUCKETS, "train_hi": TRAIN_HI, "val_hi": VAL_HI,
                    "side_hi": SIDE_HI, "reader_seed": READER_SEED, "side_seed": SIDE_SEED},
        counts={"readers": len(split["readers"]),
                "readers_" + TRAIN_NAME: len(split["by_split"][TRAIN_NAME]),
                "readers_" + VAL_NAME: len(split["by_split"][VAL_NAME]),
                "readers_" + HOLDOUT_NAME: len(split["by_split"][HOLDOUT_NAME]),
                "files": sum(r["n_files"] for r in split["rows"]),
                "files_" + SIDE_NEAR: len(split["near_files"]),
                "files_" + SIDE_FAR: len(split["far_files"])},
        lists_sha256={k: content_sha256(v) for k, v in sorted(lists.items())},
    )
    (out_dir / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta


# ---------------------------------------------------------------- 验收
def _read_list(path: Path) -> list[str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path.name}: 含 BOM（要求 UTF-8 无 BOM）")
    if b"\r" in raw:
        raise ValueError(f"{path.name}: 含 CR（要求 LF）")
    return [ln for ln in raw.decode("utf-8").split("\n") if ln]


def verify(out_dir: Path, clean_dir: Path, noise_src: Path | None,
           noise_list: Path | None, pool_root: Path | None) -> list[str]:
    """返回问题列表（空 = 通过）。输入缺失/不可读由调用方转成退出码 2。"""
    problems: list[str] = []
    for fname in list(READER_LISTS.values()) + list(SIDE_LISTS.values()) + [READER_MANIFEST,
                                                                           META_NAME]:
        if not (out_dir / fname).exists():
            raise FileNotFoundError(f"缺清单文件：{out_dir / fname}")

    per = scan_pool(clean_dir)
    built = build_split(per)
    meta = json.loads((out_dir / META_NAME).read_text(encoding="utf-8"))

    # 1) 名单与"独立重算"逐字节一致（--reproduce 语义内置）
    for sp, fname in READER_LISTS.items():
        if _read_list(out_dir / fname) != built["by_split"][sp]:
            problems.append(f"{fname}: 与重算名单不一致（手改或算法变更）")
    for side, fname in SIDE_LISTS.items():
        want = built["near_files"] if side == SIDE_NEAR else built["far_files"]
        if _read_list(out_dir / fname) != want:
            problems.append(f"{fname}: 与重算名单不一致（手改或算法变更）")

    # 2) 池级互斥（对**盘上的名单**判定：near ∩ far = ∅，reader 级与文件级）
    near_names = _read_list(out_dir / SIDE_LISTS[SIDE_NEAR])
    far_names = _read_list(out_dir / SIDE_LISTS[SIDE_FAR])
    near_r = {reader_of(f) for f in near_names}
    far_r = {reader_of(f) for f in far_names}
    if near_r & far_r:
        problems.append(f"near/far reader 交集 {len(near_r & far_r)} 个"
                        f"（例：{sorted(near_r & far_r)[0]}）")
    if set(near_names) & set(far_names):
        problems.append(f"near/far 文件名交集 {len(set(near_names) & set(far_names))} 个"
                        f"（例：{sorted(set(near_names) & set(far_names))[0]}）")
    hold = set(built["by_split"][HOLDOUT_NAME])
    if (near_r | far_r) != hold:
        problems.append("holdout reader 未被 near∪far 恰好覆盖"
                        f"（漏 {len(hold - (near_r | far_r))} / 多 {len((near_r | far_r) - hold)}）")

    # 3) 成员在源池 + reader 桶归属正确（名单非手挑）
    pool_names = {n for files in per.values() for n in files}
    for side, fname in ((SIDE_NEAR, SIDE_LISTS[SIDE_NEAR]),
                        (SIDE_FAR, SIDE_LISTS[SIDE_FAR])):
        names = _read_list(out_dir / fname)
        if len(names) != len(set(names)):
            problems.append(f"{fname}: 有重复行")
        bad = [n for n in names if "/" in n or "\\" in n]
        if bad:
            problems.append(f"{fname}: {bad[0]} 不是 basename")
        miss = [n for n in names if n not in pool_names]
        if miss:
            problems.append(f"{fname}: {len(miss)} 个文件不在源池（例：{miss[0]}）")
        wrong = [n for n in names if n in pool_names and (
            reader_split(reader_of(n)) != HOLDOUT_NAME or reader_side(reader_of(n)) != side)]
        if wrong:
            problems.append(f"{fname}: {len(wrong)} 个文件的 reader 桶归属不是 {side}"
                            f"（例：{wrong[0]}）")

    # 4) 权威表与重算一致
    got_rows = list(csv.DictReader(open(out_dir / READER_MANIFEST, encoding="utf-8")))
    want_rows = [{k: str(v) for k, v in r.items()} for r in built["rows"]]
    if got_rows != want_rows:
        problems.append(f"{READER_MANIFEST}: 与重算的权威表不一致")

    # 5) meta 自洽（计数 + 名单内容哈希）
    for fname, sha in meta.get("lists_sha256", {}).items():
        if content_sha256(_read_list(out_dir / fname)) != sha:
            problems.append(f"{META_NAME}: {fname} 的内容 sha256 与 meta 不符")
    if meta.get("scheme") != SCHEME:
        problems.append(f"{META_NAME}: scheme 字段与脚本常量不一致")

    # 6) 物化目录（可选）
    if pool_root is not None:
        for side, sub in ((SIDE_NEAR, "near"), (SIDE_FAR, "far")):
            d = pool_root / sub
            if not d.is_dir():
                problems.append(f"池目录缺失：{d}")
                continue
            want = sorted(p.name for p in d.glob("*.wav"))
            have = sorted(_read_list(out_dir / SIDE_LISTS[side]))
            if want != have:
                problems.append(f"{d}: 文件与 {SIDE_LISTS[side]} 不一致"
                                f"（{len(want)} vs {len(have)}）")
        nd = pool_root / "noise"
        if nd.is_dir() and noise_list is not None:
            want = sorted(p.name for p in nd.glob("*.wav"))
            have = sorted(_read_list(noise_list))
            if want != have:
                problems.append(f"{nd}: 文件与 {noise_list.name} 不一致"
                                f"（{len(want)} vs {len(have)}）")

    # 7) 噪声名单可选核对（成员在池）
    if noise_src is not None or noise_list is not None:
        if noise_src is None or noise_list is None:
            problems.append("--noise-src 与 --noise-list 必须成对给")
        else:
            names = _read_list(noise_list)
            have = {p.name for p in noise_src.glob("**/*.wav")}
            miss = [n for n in names if n not in have]
            if miss:
                problems.append(f"{noise_list.name}: {len(miss)} 个噪声不在 {noise_src}"
                                f"（例：{miss[0]}）")
    return problems


# ---------------------------------------------------------------- 物化
def materialize(pool_root: Path, out_dir: Path, clean_dir: Path, noise_src: Path | None,
                noise_list: Path | None) -> dict:
    """按名单建池目录（硬链接优先，失败退化为复制）。返回 {子目录: 文件数}。"""
    jobs = [("near", clean_dir, out_dir / SIDE_LISTS[SIDE_NEAR]),
            ("far", clean_dir, out_dir / SIDE_LISTS[SIDE_FAR])]
    if noise_src is not None and noise_list is not None:
        jobs.append(("noise", noise_src, noise_list))
    counts = {}
    for sub, src_dir, list_path in jobs:
        names = _read_list(list_path)
        d = pool_root / sub
        d.mkdir(parents=True, exist_ok=True)
        src_index = {p.name: p for p in src_dir.glob("**/*.wav")}
        linked = copied = 0
        for n in names:
            src = src_index.get(n)
            if src is None:
                raise FileNotFoundError(f"{sub}: 源池缺 {n}")
            dst = d / n
            if dst.exists():
                continue
            try:
                os.link(src, dst)
                linked += 1
            except OSError:
                shutil.copy2(src, dst)
                copied += 1
        counts[sub] = dict(linked=linked, copied=copied, total=len(names))
    return counts


# ---------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", type=Path, required=True, help="DNS clean 语音池（含 reader_<id>）")
    ap.add_argument("--out-dir", type=Path, default=Path("split_lists"))
    ap.add_argument("--write", action="store_true", help="生成名单（默认动作）")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写盘")
    ap.add_argument("--verify", action="store_true", help="验收既有名单（退出码 0/1/2）")
    ap.add_argument("--materialize-root", type=Path, default=None,
                    help="物化池目录根（在其下建 near/far[/noise]）")
    ap.add_argument("--noise-src", type=Path, default=None, help="0-2 噪声池源目录")
    ap.add_argument("--noise-list", type=Path, default=None,
                    help="0-2 评测噪声名单（默认 split_lists/noise_eval15.txt）")
    args = ap.parse_args()

    try:
        if args.verify:
            problems = verify(args.out_dir, args.clean_dir, args.noise_src,
                              args.noise_list, args.materialize_root)
        else:
            per = scan_pool(args.clean_dir)
            split = build_split(per)
            print(f"[split_speech_pool] 池 {args.clean_dir}：{len(split['readers'])} readers / "
                  f"{sum(len(v) for v in per.values())} wav")
            for sp in SPLIT_NAMES:
                print(f"  {sp}: {len(split['by_split'][sp])} readers")
            print(f"  eval near/far：{len(split['near_files'])} / {len(split['far_files'])} wav"
                  f"（holdout reader 专供评测）")
            if args.dry_run:
                return 0
            meta = write_outputs(args.out_dir, split, args.clean_dir)
            print(f"  名单 -> {args.out_dir}（meta counts={meta['counts']}）")
            problems = []
            if args.materialize_root is not None:
                counts = materialize(args.materialize_root, args.out_dir, args.clean_dir,
                                     args.noise_src, args.noise_list)
                print(f"  池目录 -> {args.materialize_root}：{counts}")
            if args.materialize_root is not None or args.noise_list is not None:
                problems = verify(args.out_dir, args.clean_dir, args.noise_src,
                                  args.noise_list, args.materialize_root)
    except (FileNotFoundError, ValueError) as e:
        print(f"[split_speech_pool] 输入错误：{e}", file=sys.stderr)
        return 2

    if problems:
        print(f"[split_speech_pool] 不通过（{len(problems)} 项）：", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if args.verify or args.materialize_root is not None or args.noise_list is not None:
        print("[split_speech_pool] PASS：reader 桶归属=名单、near/far 零交集、名单可重算一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
