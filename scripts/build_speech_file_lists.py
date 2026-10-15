#!/usr/bin/env python
"""0-3/t10 接口收尾：把 reader 级 near/far 训练清单展开成**文件名清单**并落盘校验。

背景
----
训练侧（t10/t12）需要的是"喂给 `aec_synth.py --near-dir/--far-dir` 的文件清单"，而
`split_lists/speech_{near,far}_train70.txt` 是 **reader 级**清单（每行一个 reader id）。
t11 D1 检查点期间 train-eng 临时展开过两份文件名清单（3477/3675，逐集核对通过）但未入库；
本脚本把它**正式化**：同一规则、可重算、带 meta 与断言。

生成规则（可独立重算，幂等）
--------------------------
1. 读 `split_lists/speech_{near,far}_train70.txt`（reader 级，由 0-3 裁决产出）；
2. 在 `<clean-dir>` 下按 `reader_<id>` 解析每个 wav 归到 reader（与 `split_speech_pool.py`
   的 `reader_of` 同口径）；
3. 展开顺序 = **reader 清单顺序 × 该 reader 内文件名升序**（与训练侧临时清单一致：
   集合相同、顺序相同；仅"跨 reader 全局排序"会得到不同顺序，故顺序写死在此并记入 meta）；
4. 写 `speech_{near,far}_train70_files.txt`（UTF-8/LF/无 BOM），并更新
   `speech_nearfar_train_meta.json` 的 `file_lists` 段。

三条硬断言（不通过即退出码 1，不写盘）
--------------------------------------
A. `near_files ∩ far_files = ∅`；
B. `near_files ∪ far_files == train70 readers 的全部文件`（即 reader 级并集 == train70）；
C. `(near_files ∪ far_files) ∩ holdout15 readers 的文件 = ∅`（评测轴互斥）。

用法
----
    python scripts/build_speech_file_lists.py \\
        --clean-dir D:/data/dns-challenge-mirror/datasets/clean --out-dir split_lists
    python scripts/build_speech_file_lists.py --clean-dir <池> --out-dir split_lists --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

READER_RE = re.compile(r"reader_(\d+)")
NEAR_READERS = "speech_near_train70.txt"
FAR_READERS = "speech_far_train70.txt"
TRAIN_READERS = "speech_readers_train70.txt"
HOLDOUT_READERS = "speech_readers_holdout15.txt"
NEAR_FILES = "speech_near_train70_files.txt"
FAR_FILES = "speech_far_train70_files.txt"
META_NAME = "speech_nearfar_train_meta.json"


def reader_of(name: str) -> str | None:
    """与 split_speech_pool.reader_of 同口径：从文件名取 `reader_<id>`。"""
    m = READER_RE.search(Path(name).name)
    return m.group(1) if m else None


def read_list(path: Path) -> list[str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{path.name}: 含 BOM（要求 UTF-8 无 BOM）")
    return [ln.strip() for ln in raw.decode("utf-8").splitlines() if ln.strip()]


def content_sha256(names: list[str]) -> str:
    return hashlib.sha256("".join(f"{n}\n" for n in names).encode("utf-8")).hexdigest()


def scan_pool(clean_dir: Path) -> dict[str, list[str]]:
    """扫池 -> {reader: [文件名...]}（每个 reader 内按文件名升序）。"""
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"--clean-dir 不存在或不是目录：{clean_dir}")
    per: dict[str, list[str]] = {}
    for p in sorted(clean_dir.glob("**/*.wav")):
        r = reader_of(p.name)
        if r is None:
            raise ValueError(f"文件名里没有 reader_<id>：{p.name}")
        per.setdefault(r, []).append(p.name)
    if not per:
        raise FileNotFoundError(f"--clean-dir 下没有 wav：{clean_dir}")
    return {r: sorted(v) for r, v in per.items()}


def expand(reader_list: list[str], per_reader: dict[str, list[str]]) -> list[str]:
    """reader 清单顺序 × reader 内文件名升序 -> 文件名清单。"""
    out: list[str] = []
    for r in reader_list:
        out.extend(per_reader.get(r, []))
    return out


def build(out_dir: Path, per_reader: dict[str, list[str]]) -> dict:
    near_r = read_list(out_dir / NEAR_READERS)
    far_r = read_list(out_dir / FAR_READERS)
    train_r = read_list(out_dir / TRAIN_READERS)
    hold_r = read_list(out_dir / HOLDOUT_READERS)
    near_f = expand(near_r, per_reader)
    far_f = expand(far_r, per_reader)
    train_f = expand(train_r, per_reader)
    hold_f = expand(hold_r, per_reader)

    problems: list[str] = []
    if set(near_r) & set(far_r):
        problems.append("A. reader 级 near ∩ far 非空")
    if set(near_f) & set(far_f):
        problems.append(f"A. 文件级 near ∩ far = {len(set(near_f) & set(far_f))} 个（应为 0）")
    if set(near_f) | set(far_f) != set(train_f):
        d1 = sorted(set(train_f) - (set(near_f) | set(far_f)))[:3]
        d2 = sorted((set(near_f) | set(far_f)) - set(train_f))[:3]
        problems.append(f"B. near ∪ far != train70 文件集（缺 {d1} / 多 {d2}）")
    inter_h = (set(near_f) | set(far_f)) & set(hold_f)
    if inter_h:
        problems.append(f"C. 训练文件 ∩ holdout15 文件 = {len(inter_h)} 个（应为 0）")
    if sorted(set(near_r) | set(far_r)) != sorted(set(train_r)):
        problems.append("D. reader 级 near ∪ far != train70 readers")

    return dict(problems=problems, near_files=near_f, far_files=far_f,
                near_readers=near_r, far_readers=far_r, holdout_files=hold_f)


def read_json(path: Path) -> dict:
    """读 JSON：容忍 BOM（`speech_nearfar_train_meta.json` 早期由 PowerShell 写出，带 BOM）。"""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_outputs(out_dir: Path, built: dict, clean_dir: Path) -> dict:
    near_path = out_dir / NEAR_FILES
    far_path = out_dir / FAR_FILES
    near_path.write_text("".join(f"{n}\n" for n in built["near_files"]),
                         encoding="utf-8", newline="\n")
    far_path.write_text("".join(f"{n}\n" for n in built["far_files"]),
                        encoding="utf-8", newline="\n")
    meta_path = out_dir / META_NAME
    meta = read_json(meta_path)
    meta["file_lists"] = {
        "rule": ("reader 清单顺序 × reader 内文件名升序（与训练侧 t11 临时清单逐行一致）；"
                 "跨 reader 全局字典序会得到不同顺序，故顺序固定于此"),
        "near_file_list": NEAR_FILES,
        "far_file_list": FAR_FILES,
        "counts": {"near_files": len(built["near_files"]), "far_files": len(built["far_files"])},
        "sha256": {NEAR_FILES: content_sha256(built["near_files"]),
                   FAR_FILES: content_sha256(built["far_files"])},
        "assertions": {
            "A_near_inter_far_files": 0,
            "B_union_equals_train70_files": True,
            "C_no_overlap_with_holdout15_files": True,
            "D_readers_union_equals_train70": True,
        },
        "source_pool_basename": clean_dir.name,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8", newline="\n")
    return meta


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 默认 GBK 控制台
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", type=Path, required=True, help="DNS clean 池（含 reader_<id>）")
    ap.add_argument("--out-dir", type=Path, default=Path("split_lists"))
    ap.add_argument("--verify", action="store_true",
                    help="只核对既有 _files 清单与源池（不写盘）")
    args = ap.parse_args()

    try:
        per = scan_pool(args.clean_dir)
        built = build(args.out_dir, per)
    except (FileNotFoundError, ValueError) as e:
        print(f"[build_speech_file_lists] 输入错误：{e}", file=sys.stderr)
        return 2

    print(f"[build_speech_file_lists] 池 {args.clean_dir}：{len(per)} readers")
    print(f"  near={len(built['near_files'])} far={len(built['far_files'])} "
          f"(readers {len(built['near_readers'])}/{len(built['far_readers'])})")
    if built["problems"]:
        print(f"  不通过（{len(built['problems'])} 项）：")
        for p in built["problems"]:
            print(f"   - {p}")
        return 1

    if args.verify:
        for fname, names in ((NEAR_FILES, built["near_files"]), (FAR_FILES, built["far_files"])):
            p = args.out_dir / fname
            if not p.is_file():
                print(f"  不通过：缺 {fname}")
                return 1
            on_disk = read_list(p)
            if on_disk != names:
                if set(on_disk) == set(names):
                    print(f"  不通过：{fname} 与重算结果集合相同但顺序不同（应逐行一致）")
                else:
                    print(f"  不通过：{fname} 与重算结果内容不一致")
                return 1
        print("  PASS：A/B/C/D 四条断言成立，且与既有 _files 清单逐行一致")
        return 0

    meta = write_outputs(args.out_dir, built, args.clean_dir)
    shas = {k: v[:12] for k, v in meta["file_lists"]["sha256"].items()}
    print(f"  PASS：A/B/C/D 断言成立；写入 {NEAR_FILES} / {FAR_FILES} 与 meta.file_lists")
    print(f"  sha256 = {shas}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
