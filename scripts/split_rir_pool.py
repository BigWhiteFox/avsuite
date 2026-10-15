#!/usr/bin/env python
"""真实 RIR 三分割：按**录音 session / 声源位置**分组做 train70/val15/holdout15 互斥切分。

为什么必须按组切（而不是按文件随机切）
----------------------------------------
`REMEDIATION_PLAN_V3.1.md` §3.2 明写：IR64-CAR「同车 32 条高度相关，随机切 = 泄漏」——
32 条 IR 来自**同一辆车、同一次录音**，只有 8 个声源方位 × 4 个麦克胶囊的差异；按文件随机切
会把同一方位/同一录音的近邻 IR 同时放进训练与评测。故本脚本以
**位置组（`IR64CAR_<angle>`）为最小切分单位**，同组文件永不跨 split。

切分规则（可独立重算，吃文件名/相对路径，不吃扫盘顺序）
--------------------------------------------------------
    group_key = 'ir64car_pos<angle>'（文件名命中 IR64CAR_<angle>） 或 上级目录名
    rank      = sha256("rir_group:" + group_key)[:8] 的十六进制值（升序排名；同值按组名）
    G = 组数：n_hold = max(1, ceil(0.15·G))；n_val = max(1, round(0.15·G))；其余 train
    rank 最大的 n_hold 组 -> holdout15（**评测专用**；训练侧禁区）
    其次 n_val 组          -> val15
    其余                   -> train70

> 组数很少时（本池 G=8）精确 70/15/15 不可达；规则按 §7「RIR 优先入 eval 轴（证据价值 > 训练价值）」
> 把非训练份额保底（实测 train 5 组=20 条 / val 1 组=4 条 / holdout 2 组=8 条，即 62.5/12.5/25）。
> 阈值与分配公式写进 ``rir_split_meta.json``，可被独立重算核对。

产出（默认 ``avsuite/split_lists/``）
------------------------------------
- ``rir_train70.txt`` / ``rir_val15.txt`` / ``rir_holdout15.txt``：每行一个**文件名**（UTF-8/LF），
  下发给 aec_synth 的池目录用（``--rir-dir``）。
- ``rir_split_manifest.csv``：全池权威表
  （``filename,group,session,split,rank,sha256,group_sha8``）。
- ``rir_split_meta.json``：算法/阈值/计数/各名单内容 sha256（版本哨兵）。

用法
----
    # 1) 生成名单（幂等）
    python scripts/split_rir_pool.py --rir-dir D:/data/rir_real/pool_16k_mono --out-dir split_lists

    # 2) 物化评测 RIR 池（硬链接；建议 D:/data/aec-synth/eval_rir_pool）
    python scripts/split_rir_pool.py --rir-dir <池> --out-dir split_lists \
        --materialize-root D:/data/aec-synth/eval_rir_pool

    # 3) 验收（退出码 0=通过 1=不通过 2=输入错误）
    python scripts/split_rir_pool.py --rir-dir <池> --out-dir split_lists --verify \
        --materialize-root D:/data/aec-synth/eval_rir_pool

    # 4) 纯预览
    python scripts/split_rir_pool.py --rir-dir <池> --out-dir split_lists --dry-run

边界（如实声明）
----------------
- **本池只有一个车（一个 session）**：位置级互斥已做到，但**车/录音级互斥做不到**——
  eval_rir 衡量的是"未见声源位置"上的泛化，不是"未见车辆"。要更严的轴需第二辆车的实测 IR
  （见 `reports/realrir_domain.md` 的外部数据集清单）。
- 不改既有 eval 集/训练集：本清单是"以后怎么切"的配置。历史事实：`train_v1`/`train_nl` 的
  训练 RIR 用的是 **仿真池** `D:/data/aec-synth/rir_pool`（60 条 pyroomacoustics，manifest
  ``rir_src=dir`` 但目录是仿真池），真实 IR64-CAR 此前只被 `eval_v6_realrir`（评测）用过。
- 与 0-4 的语音池/0-2 的噪声池是三个独立命名空间，互不影响。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

HASH8_LEN = 8
GROUP_SEED = "rir_group:"
HOLD_FRAC = 0.15
VAL_FRAC = 0.15
TRAIN_NAME, VAL_NAME, HOLDOUT_NAME = "train70", "val15", "holdout15"
SPLIT_NAMES = (TRAIN_NAME, VAL_NAME, HOLDOUT_NAME)
ANGLE_RE = re.compile(r"IR64CAR[_-](\d+)", re.IGNORECASE)
CAR_RE = re.compile(r"IR64CAR", re.IGNORECASE)

LIST_FILES = {
    TRAIN_NAME: "rir_train70.txt",
    VAL_NAME: "rir_val15.txt",
    HOLDOUT_NAME: "rir_holdout15.txt",
}
MANIFEST_NAME = "rir_split_manifest.csv"
META_NAME = "rir_split_meta.json"
SCHEME = ("group=IR64CAR_<angle> 或上级目录名；按 sha256('rir_group:'+group)[:8] 升序排名；"
          "n_hold=max(1,ceil(0.15G)) 取最高名次组、n_val=max(1,round(0.15G)) 取次高，其余 train；"
          "同组文件不跨 split")


# ---------------------------------------------------------------- 分组与命名
def group_of(path: Path, root: Path) -> str:
    """文件 -> 切分最小单位（位置组）。IR64-CAR 命中声源方位，其余退化为上级目录名。"""
    m = ANGLE_RE.search(path.name)
    if m:
        return f"ir64car_pos{m.group(1)}"
    try:
        rel_parent = path.parent.relative_to(root)
    except ValueError:                                       # pragma: no cover - 池外文件
        rel_parent = Path(".")
    return f"dir:{rel_parent.as_posix() or '.'}"


def session_of(path: Path) -> str:
    """录音 session（车辆）标识；本池只有一辆车，如实记为 ir64car。"""
    return "ir64car" if CAR_RE.search(path.name) else "unknown"


def group_hash8(group: str) -> str:
    return hashlib.sha256((GROUP_SEED + group).encode("utf-8")).hexdigest()[:HASH8_LEN]


def content_sha256(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def scan_pool(rir_dir: Path) -> list[Path]:
    if not rir_dir.is_dir():
        raise FileNotFoundError(f"--rir-dir 不存在或不是目录：{rir_dir}")
    files = sorted(p for p in rir_dir.glob("**/*.wav"))
    if not files:
        raise FileNotFoundError(f"--rir-dir 下没有 wav：{rir_dir}")
    return files


def allocate(n_groups: int) -> dict[str, int]:
    """组数 -> 三个份额的组数（保底：有 ≥3 组时 holdout/val 各至少 1 组，train 至少 1 组）。

    组数极少（≤2）时不强留评测份额（否则训练份额为空）；该情形会打印告警，且 meta 记下实际分配。
    """
    if n_groups <= 2:
        return {HOLDOUT_NAME: max(0, n_groups - 1), VAL_NAME: 0,
                TRAIN_NAME: min(1, n_groups)}
    n_hold = max(1, min(math.ceil(HOLD_FRAC * n_groups), n_groups - 2))
    n_val = max(1, min(round(VAL_FRAC * n_groups), n_groups - n_hold - 1))
    return {HOLDOUT_NAME: n_hold, VAL_NAME: n_val,
            TRAIN_NAME: n_groups - n_hold - n_val}


def build_split(files: list[Path], root: Path) -> dict:
    """纯函数：文件列表 -> 名单集合 + 权威表行（不落盘）。"""
    groups: dict[str, list[Path]] = {}
    for f in files:
        groups.setdefault(group_of(f, root), []).append(f)
    ranked = sorted(groups, key=lambda g: (group_hash8(g), g))     # 升序：名次越低越靠前
    n = allocate(len(ranked))
    hold = set(ranked[len(ranked) - n[HOLDOUT_NAME]:])
    rest = [g for g in ranked if g not in hold]
    val = set(rest[len(rest) - n[VAL_NAME]:]) if n[VAL_NAME] else set()
    names = {s: [] for s in SPLIT_NAMES}
    rows = []
    for rank, g in enumerate(ranked):
        split = (HOLDOUT_NAME if g in hold else VAL_NAME if g in val else TRAIN_NAME)
        for f in sorted(groups[g]):
            names[split].append(f.name)
            rows.append(dict(filename=f.name, group=g, session=session_of(f), split=split,
                             rank=rank, sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                             group_sha8=group_hash8(g)))
    for s in names:
        names[s].sort()
    return dict(names=names, rows=sorted(rows, key=lambda r: r["filename"]),
                groups={g: sorted(p.name for p in v) for g, v in groups.items()},
                counts=n, n_groups=len(ranked))


# ---------------------------------------------------------------- 落盘
def _write_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{n}\n" for n in names), encoding="utf-8", newline="\n")


def write_outputs(out_dir: Path, split: dict, rir_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / MANIFEST_NAME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "group", "session", "split", "rank",
                                          "sha256", "group_sha8"])
        w.writeheader()
        w.writerows(split["rows"])
    for s, fname in LIST_FILES.items():
        _write_list(out_dir / fname, split["names"][s])
    meta = dict(
        version="rir_split_v1", scheme=SCHEME,
        source={"rir_dir": str(rir_dir)},
        thresholds={"hold_frac": HOLD_FRAC, "val_frac": VAL_FRAC, "hash8_len": HASH8_LEN,
                    "group_seed": GROUP_SEED, "min_hold_groups": 1, "min_val_groups": 1},
        counts={"files": sum(len(v) for v in split["names"].values()),
                "groups": split["n_groups"],
                "groups_" + TRAIN_NAME: split["counts"][TRAIN_NAME],
                "groups_" + VAL_NAME: split["counts"][VAL_NAME],
                "groups_" + HOLDOUT_NAME: split["counts"][HOLDOUT_NAME],
                "files_" + TRAIN_NAME: len(split["names"][TRAIN_NAME]),
                "files_" + VAL_NAME: len(split["names"][VAL_NAME]),
                "files_" + HOLDOUT_NAME: len(split["names"][HOLDOUT_NAME])},
        lists_sha256={LIST_FILES[s]: content_sha256(split["names"][s]) for s in SPLIT_NAMES},
    )
    (out_dir / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta


def _read_list(path: Path) -> list[str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path.name}: 含 BOM（要求 UTF-8 无 BOM）")
    if b"\r" in raw:
        raise ValueError(f"{path.name}: 含 CR（要求 LF）")
    return [ln for ln in raw.decode("utf-8").split("\n") if ln]


# ---------------------------------------------------------------- 验收
def verify(out_dir: Path, rir_dir: Path, pool_root: Path | None) -> list[str]:
    problems: list[str] = []
    for fname in list(LIST_FILES.values()) + [MANIFEST_NAME, META_NAME]:
        if not (out_dir / fname).exists():
            raise FileNotFoundError(f"缺清单文件：{out_dir / fname}")
    files = scan_pool(rir_dir)
    built = build_split(files, rir_dir)
    meta = json.loads((out_dir / META_NAME).read_text(encoding="utf-8"))

    # 1) 名单 == 独立重算（--reproduce 语义）
    for s, fname in LIST_FILES.items():
        if _read_list(out_dir / fname) != built["names"][s]:
            problems.append(f"{fname}: 与重算名单不一致（手改或池变更）")
    train = set(built["names"][TRAIN_NAME])
    val = set(built["names"][VAL_NAME])
    hold = set(built["names"][HOLDOUT_NAME])

    # 2) 覆盖与两两互斥
    pool = {f.name for f in files}
    if train | val | hold != pool:
        problems.append(f"三份额并集 != 全池（差 {len(pool - (train | val | hold))} 个）")
    for a, b, la, lb in ((train, hold, "train", "holdout"), (train, val, "train", "val"),
                         (val, hold, "val", "holdout")):
        if a & b:
            problems.append(f"{la} ∩ {lb} = {len(a & b)} 个（例：{sorted(a & b)[0]}）")

    # 3) 同组不跨 split（泄漏红线：位置/录音级互斥）
    g_split = {}
    for row in built["rows"]:
        g_split.setdefault(row["group"], set()).add(row["split"])
    mixed = sorted(g for g, s in g_split.items() if len(s) > 1)
    if mixed:
        problems.append(f"{len(mixed)} 个位置组跨 split（同组文件必须在同一份额）：{mixed[0]}")

    # 4) 权威表一致 + meta 自洽
    got = list(csv.DictReader(open(out_dir / MANIFEST_NAME, encoding="utf-8")))
    want = [{k: str(v) for k, v in r.items()} for r in built["rows"]]
    if got != want:
        problems.append(f"{MANIFEST_NAME}: 与重算的权威表不一致")
    for fname, sha in meta.get("lists_sha256", {}).items():
        if content_sha256(_read_list(out_dir / fname)) != sha:
            problems.append(f"{META_NAME}: {fname} 的内容 sha256 与 meta 不符")
    if meta.get("scheme") != SCHEME:
        problems.append(f"{META_NAME}: scheme 字段与脚本常量不一致")

    # 5) 物化目录（可选）
    if pool_root is not None:
        d = pool_root
        if not d.is_dir():
            problems.append(f"评测 RIR 池目录缺失：{d}")
        else:
            got_names = sorted(p.name for p in d.glob("**/*.wav"))
            if got_names != built["names"][HOLDOUT_NAME]:
                problems.append(f"{d}: 文件与 {LIST_FILES[HOLDOUT_NAME]} 不一致"
                                f"（{len(got_names)} vs {len(built['names'][HOLDOUT_NAME])}）")
    return problems


# ---------------------------------------------------------------- 物化
def materialize(pool_root: Path, out_dir: Path, rir_dir: Path) -> dict:
    names = _read_list(out_dir / LIST_FILES[HOLDOUT_NAME])
    src_index = {p.name: p for p in rir_dir.glob("**/*.wav")}
    pool_root.mkdir(parents=True, exist_ok=True)
    linked = copied = 0
    for n in names:
        src = src_index.get(n)
        if src is None:
            raise FileNotFoundError(f"源 RIR 池缺 {n}")
        dst = pool_root / n
        if dst.exists():
            continue
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            shutil.copy2(src, dst)
            copied += 1
    return {"linked": linked, "copied": copied, "total": len(names)}


# ---------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rir-dir", type=Path, required=True, help="真实 RIR 池（递归 *.wav）")
    ap.add_argument("--out-dir", type=Path, default=Path("split_lists"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--materialize-root", type=Path, default=None,
                    help="评测 RIR 池目录（放 holdout15 的硬链接）")
    args = ap.parse_args()
    try:
        if args.verify:
            problems = verify(args.out_dir, args.rir_dir, args.materialize_root)
        else:
            files = scan_pool(args.rir_dir)
            split = build_split(files, args.rir_dir)
            print(f"[split_rir_pool] 池 {args.rir_dir}：{len(files)} 条 / "
                  f"{split['n_groups']} 个位置组")
            for s in SPLIT_NAMES:
                print(f"  {s}: {split['counts'][s]} 组 / {len(split['names'][s])} 条")
            if args.dry_run:
                return 0
            meta = write_outputs(args.out_dir, split, args.rir_dir)
            print(f"  名单 -> {args.out_dir}（meta counts={meta['counts']}）")
            problems = []
            if args.materialize_root is not None:
                print(f"  评测 RIR 池 -> {args.materialize_root}："
                      f"{materialize(args.materialize_root, args.out_dir, args.rir_dir)}")
            if args.materialize_root is not None:
                problems = verify(args.out_dir, args.rir_dir, args.materialize_root)
    except (FileNotFoundError, ValueError) as e:
        print(f"[split_rir_pool] 输入错误：{e}", file=sys.stderr)
        return 2

    if problems:
        print(f"[split_rir_pool] 不通过（{len(problems)} 项）：", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if args.verify or args.materialize_root is not None:
        print("[split_rir_pool] PASS：组级互斥 / 覆盖完整 / 桶归属=名单 / 权威表与版本哈希一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
