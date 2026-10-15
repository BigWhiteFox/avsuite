#!/usr/bin/env python
"""0-2 噪声池三分割：DNS 噪声按文件名哈希切 train70 / val15 / eval15（可复现、可落盘）。

为什么是哈希切分
----------------
REMEDIATION_PLAN_V3.1.md §2 0-2：训练用噪声与评测用噪声此前 99~100% 重叠
（实测见 ``reports/refactor_wave5/data_inventory.md`` §6 H2/H3）——噪声维度没有
"未见噪声"泛化面。本脚本把同一个 DNS 噪声池按**文件名的哈希**切成三份，切分
只依赖文件名本身（不依赖盘上路径、不依赖扫盘顺序、不依赖随机种子）：

    bucket = int(sha256(filename)[:8], 16) % 1000
    bucket < 700            -> train70
    700 <= bucket < 850     -> val15
    bucket >= 850           -> eval15

因此 **train70 ∩ eval15 = ∅ 是构造性成立的**（一个文件名只有一个 bucket），
也不需要"先分组再抽签"的中间状态——服务器上重跑得到同一份名单。

产出（默认 ``avsuite/split_lists/``）
------------------------------------
- ``noise_train70.txt`` / ``noise_val15.txt`` / ``noise_eval15.txt``：UTF-8、
  每行一个**文件名**（非路径；池路径在服务器上不同）、按文件名排序、LF 换行。
- ``noise_split_manifest.csv``：全池 659 行，逐文件给
  ``filename,split,bucket,hash8,sha256``——单一权威表，供抽检与并集核对。
- ``noise_split_meta.json``：算法 + 桶阈值 + 计数 + 三份名单的内容哈希，
  使"桶分配规则"本身可被独立重算与核对。
- ``README.md``：复现命令、验证命令与边界说明。

用法
----
    python scripts/split_noise_pool.py \\
        --noise-dir D:/data/dns-challenge-mirror/datasets/noise \\
        --out-dir split_lists

    python scripts/split_noise_pool.py --noise-dir <池> --out-dir <目录> --dry-run
    python scripts/verify_split.py --dir split_lists --noise-dir <池>

边界（不做的事）
----------------
- 不改仓内既有代码语义：本文件与 ``verify_split.py`` 均为新增文件。
- 不搬/不删任何 wav：切分是**命名清单**，训练侧按清单取子集即可。
- 不覆盖"训练实际用过哪些噪声"的历史事实：该事实由 train 侧 manifest 给出，
  见 README"历史口径"一节；本脚本产出的名单是**配置**，不是历史统计。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: 桶分辨率与分段阈值（改这三处即改配比；改后必须重跑并登记新版本）
BUCKETS = 1000
TRAIN_HI = 700          # bucket < 700        -> train70
VAL_HI = 850            # 700 <= bucket < 850 -> val15；>= 850 -> eval15
TRAIN_NAME = "train70"
VAL_NAME = "val15"
EVAL_NAME = "eval15"
SPLIT_NAMES = (TRAIN_NAME, VAL_NAME, EVAL_NAME)
HASH8_LEN = 8

LIST_FILES = {
    TRAIN_NAME: "noise_train70.txt",
    VAL_NAME: "noise_val15.txt",
    EVAL_NAME: "noise_eval15.txt",
}
MANIFEST_NAME = "noise_split_manifest.csv"
META_NAME = "noise_split_meta.json"
SCHEME = "sha256(filename)[:8] % 1000"


def filename_hash8(filename: str, seed: str = "") -> str:
    """文件名 -> sha256 hexdigest 前 8 位（小写）。

    只吃 basename 字符串，不吃路径、不吃文件内容；``seed`` 默认空串，
    非空时前缀拼进哈希输入（用于换配比版本，默认路径不使用）。
    """
    payload = (seed + filename).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:HASH8_LEN]


def bucket_of(filename: str, seed: str = "") -> int:
    """文件名 -> [0, BUCKETS) 的桶号。"""
    return int(filename_hash8(filename, seed), 16) % BUCKETS


def split_of(filename: str, seed: str = "") -> str:
    """文件名 -> train70 / val15 / eval15（哈希唯一决定，故三分必然互斥）。"""
    b = bucket_of(filename, seed)
    if b < TRAIN_HI:
        return TRAIN_NAME
    if b < VAL_HI:
        return VAL_NAME
    return EVAL_NAME


def pool_filenames(noise_dir: Path) -> list[str]:
    """池目录下的 wav 文件名（**按文件名排序**，与扫盘顺序无关）。

    递归 ``**/*.wav`` 以兼容将来按类目建子目录的噪声池；非 wav 文件（例如仓库
    自带的 ``Readme.md``）不入池。
    """
    noise_dir = Path(noise_dir)
    if not noise_dir.is_dir():
        raise FileNotFoundError(f"噪声池目录不存在: {noise_dir}")
    names = [p.name for p in noise_dir.glob("**/*.wav") if p.is_file()]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"池内存在重名 wav（递归目录下必须唯一）: {dupes[:5]}")
    return sorted(names)


def build_split(names: list[str], seed: str = "") -> dict[str, list[str]]:
    """文件名列表 -> {split: [filename, ...]}（每组按文件名排序）。"""
    groups: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for n in names:
        groups[split_of(n, seed)].append(n)
    return {k: sorted(v) for k, v in groups.items()}


def verify_groups(groups: dict[str, list[str]], pool: list[str]) -> list[str]:
    """本地自检：覆盖完整 + 两两互斥。返回问题列表（空 = 通过）。"""
    problems: list[str] = []
    sets = {g: set(groups.get(g, [])) for g in SPLIT_NAMES}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                problems.append(f"{a} ∩ {b} = {len(inter)} 个（应为 0）: {sorted(inter)[:3]}")
    union = set().union(*sets.values())
    if union != set(pool):
        missing = sorted(set(pool) - union)[:5]
        extra = sorted(union - set(pool))[:5]
        problems.append(f"并集 != 全池（缺 {missing} / 多 {extra}）")
    return problems


def list_text(names: list[str]) -> str:
    """名单文件的字节内容（UTF-8 无 BOM、LF 换行；空名单也是合法文本）。"""
    return "".join(f"{n}\n" for n in names)


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_text(names: list[str], seed: str = "") -> str:
    """权威表 CSV：filename,split,bucket,hash8,sha256（每行一个池内文件）。"""
    lines = ["filename,split,bucket,hash8,sha256"]
    for n in names:
        h8 = filename_hash8(n, seed)
        lines.append(f"{n},{split_of(n, seed)},{int(h8, 16) % BUCKETS},{h8},"
                     f"{hashlib.sha256(n.encode('utf-8')).hexdigest()}")
    return "".join(f"{ln}\n" for ln in lines)


def build_artifacts(noise_dir: Path, seed: str = "") -> dict:
    """算出全部产出（不落盘）。返回 {"groups", "manifest", "meta", ...}。"""
    pool = pool_filenames(noise_dir)
    groups = build_split(pool, seed)
    lists = {g: list_text(groups[g]) for g in SPLIT_NAMES}
    manifest = manifest_text(pool, seed)
    meta = {
        "task": "0-2 噪声池三分割（REMEDIATION_PLAN_V3.1 §2）",
        "scheme": SCHEME,
        "hash": "sha256",
        "hash_input": "filename(utf-8, basename, no path)" + (" + seed prefix" if seed else ""),
        "seed": seed,
        "buckets": BUCKETS,
        "thresholds": {"train70": f"bucket < {TRAIN_HI}",
                       "val15": f"{TRAIN_HI} <= bucket < {VAL_HI}",
                       "eval15": f"bucket >= {VAL_HI}"},
        # 平台无关：切分只吃**文件名**，池路径与名单内容无关。这里只留一个不含绝对路径的
        # 文档性提示（末级目录名），避免名单被绑死在某台机器的 D:\... 上。
        # 复现/校验请显式 --noise-dir（或由调用方自行记录路径）。
        "pool_dir_hint": f"<noise-pool>/{noise_dir.name}",
        "pool_files": len(pool),
        "counts": {g: len(groups[g]) for g in SPLIT_NAMES},
        "ratios": {g: round(len(groups[g]) / len(pool), 6) if pool else 0.0
                   for g in SPLIT_NAMES},
        "list_files": {g: LIST_FILES[g] for g in SPLIT_NAMES},
        "list_sha256": {g: content_sha256(lists[g]) for g in SPLIT_NAMES},
        "manifest_file": MANIFEST_NAME,
        "manifest_sha256": content_sha256(manifest),
        "mutual_exclusion": "by construction: one filename -> one bucket -> one group",
    }
    return {"pool": pool, "groups": groups, "lists": lists,
            "manifest": manifest, "meta": meta}


def write_artifacts(art: dict, out_dir: Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for g in SPLIT_NAMES:
        p = out / LIST_FILES[g]
        p.write_text(art["lists"][g], encoding="utf-8", newline="\n")
        written.append(p)
    p = out / MANIFEST_NAME
    p.write_text(art["manifest"], encoding="utf-8", newline="\n")
    written.append(p)
    p = out / META_NAME
    p.write_text(json.dumps(art["meta"], ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8", newline="\n")
    written.append(p)
    return written


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 默认 GBK 控制台
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise-dir", type=Path, required=True,
                    help="DNS 噪声池目录（递归取 *.wav；文件名即切分依据）")
    ap.add_argument("--out-dir", type=Path, default=Path("split_lists"),
                    help="清单输出目录（默认 avsuite/split_lists）")
    ap.add_argument("--seed", default="", help="哈希前缀种子（换版本用；默认空 = 纯文件名哈希）")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计与问题，不落盘")
    ap.add_argument("--quiet", action="store_true", help="只打印一行汇总")
    args = ap.parse_args()

    art = build_artifacts(args.noise_dir, args.seed)
    problems = verify_groups(art["groups"], art["pool"])
    if problems:
        print("[split_noise_pool] FAIL 自检未通过：")
        for p in problems:
            print(f"  - {p}")
        return 2

    counts = art["meta"]["counts"]
    ratios = art["meta"]["ratios"]
    line = (f"[split_noise_pool] pool={len(art['pool'])} -> "
            f"train70={counts[TRAIN_NAME]}({ratios[TRAIN_NAME]:.3f}) "
            f"val15={counts[VAL_NAME]}({ratios[VAL_NAME]:.3f}) "
            f"eval15={counts[EVAL_NAME]}({ratios[EVAL_NAME]:.3f})")
    if args.dry_run:
        print(line + "  [dry-run，未落盘]")
        return 0

    written = write_artifacts(art, args.out_dir)
    if args.quiet:
        print(line + f"  -> {Path(args.out_dir)}")
    else:
        print(line)
        for p in written:
            print(f"  写出 {p}")
        print(f"  train∩eval = 0（构造性）；清单即配置，训练侧按 {LIST_FILES[TRAIN_NAME]} 取子集")
    return 0


if __name__ == "__main__":
    sys.exit(main())
