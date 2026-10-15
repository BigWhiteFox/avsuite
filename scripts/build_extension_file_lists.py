#!/usr/bin/env python
"""扩展语料的**训练侧文件名清单**生成/校验（P1X-2 配套，2026-10-13）。

为什么需要它
------------
t21 的 `split_extension_pools.py` 产出了扩展池的 **reader 级**训练清单
（`ext_ls_readers_train70.txt` / `ext_aishell3_readers_train70.txt`）与**评测侧**文件清单，
但 DM 训练要的是**训练侧文件级清单**（`dm_dataset.load_pool` 按清单取子集，隔离靠清单）。
本脚本按与 `build_speech_file_lists.py`（t17）同族的规则补齐：

    行序 = reader 清单顺序 × 该 reader 内文件名字节序（t19 的平台无关键）
    行内容 = 相对池根的 POSIX 路径（LS: `<subset>/<reader>/<chapter>/<file>.flac`；
             AISHELL-3: `<train|dev>/wav/<spk>/<file>.wav`）

隔离性由构造保证：只输出**清单里那些 reader** 的文件 ⇒ holdout/val reader 天然不在其中；
`--verify` 再显式断言"与 holdout reader 零交集"（双保险）。

用法（**在数据所在机器上跑**，池在服务器 ⇒ 服务器侧生成后随仓 scp）
    python scripts/build_extension_file_lists.py --corpus librispeech \
        --root /root/data/librispeech \
        --readers split_lists/ext_ls_readers_train70.txt \
        --holdout-readers split_lists/ext_ls_readers_holdout15.txt \
        --out split_lists/ext_ls_far_train70_files.txt --verify
    python scripts/build_extension_file_lists.py --corpus aishell3 \
        --root /root/data/aishell3 \
        --readers split_lists/ext_aishell3_readers_train70.txt \
        --holdout-readers split_lists/ext_aishell3_readers_holdout15.txt \
        --out split_lists/ext_aishell3_far_train70_files.txt --verify
"""
import argparse
import hashlib
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import aec_synth as S  # noqa: E402   （池枚举的平台无关排序键：pool_sort_key）

AUDIO_EXTS = (".wav", ".flac")
#: 各语料的"reader 在相对路径里的位置"（前缀部分的下标）与音频扩展名
LAYOUT = {
    "librispeech": {"reader_index": 1, "exts": (".flac",), "audio_dir": None,
                    "note": "<subset>/<reader>/<chapter>/<file>.flac"},
    "aishell3": {"reader_index": 2, "exts": (".wav",), "audio_dir": "wav",
                 "note": "<train|dev>/wav/<spk>/<file>.wav"},
}


def read_list(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def scan(root, layout):
    """一次遍历：reader → [相对路径（POSIX）]，扩展名按语料限定。"""
    idx, exts = layout["reader_index"], layout["exts"]
    per: dict = {}
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"[ext-lists] 池根不存在：{root}")
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        parts = p.relative_to(root).parts
        if len(parts) <= idx + 1:
            continue                                   # 目录结构不符（如根下的散文件）
        per.setdefault(parts[idx], []).append("/".join(parts))
    return per


def reader_of_rel(rel, layout):
    """相对路径 → reader 字段（**唯一**判据；不要用子串匹配，见下）。

    踩过的坑（train-eng 实锤，2026-10-13）：隔离断言曾写成 `any(r in f for r in hold)`——
    那是**子串**判断，而 LS 的 chapter 目录名会撞车：holdout reader `669` 是 chapter 目录
    `121669` 的子串 ⇒ 路径 `train-clean-360/100/121669/100-121669-0000.flac`（reader=`100`，
    正常训练文件）被误报成"含 holdout reader"，整个 LS 清单生成被自己的断言拦下（生成逻辑
    本身没问题：被拦前打印的 sha256 与人工复现的产出一致）。⇒ 一律按**路径字段**判。
    """
    parts = str(rel).split("/")
    idx = layout["reader_index"]
    return parts[idx] if len(parts) > idx else ""


def leaked_files(files, hold, layout):
    """生成清单里**真正**属于 holdout reader 的行（按 reader 字段相等判，**绝不能用子串**）。

    抽成命名函数是为了可直接单测（t25）：`reader_of_rel(f) in hold` 这条判据既不能**误报**
    （chapter `121669` 撞 holdout `669`），也不能**漏报**（真 holdout 行必须抓出来）。
    """
    return [f for f in files if reader_of_rel(f, layout) in hold]


def suggest_root(root, layout, missing):
    """`--root` 给错层时的候选：在 root 的**下一层**找能覆盖缺失 reader 的目录。

    LibriSpeech 池常见形态是多一层（`.../librispeech/LibriSpeech/train-clean-*/...`）；
    给错层时扫不到音频/reader，错误信息里直接把候选目录打出来（t25 ④）。
    """
    root = Path(root)
    cands = []
    try:
        subs = [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return cands
    for sub in subs:
        per = scan(sub, layout)
        if not per:
            continue
        hit = sum(1 for r in missing if r in per)
        if hit:
            cands.append((hit, str(sub), sum(len(v) for v in per.values())))
    cands.sort(key=lambda c: (-c[0], c[1]))
    return cands[:3]


def build(root, readers, layout):
    """读者清单顺序 × 读者内字节序 → 文件级清单（不含任何非清单 reader 的文件）。"""
    per = scan(root, layout)
    missing = [r for r in readers if r not in per]
    if missing:
        hint = ""
        cands = suggest_root(root, layout, missing)
        if cands:
            hint = ("\n  提示：--root 可能多/少了一层——下层这些目录能覆盖缺失的 reader："
                    + "".join(f"\n    {c[1]}（命中 {c[0]}/{len(missing)} 个缺失 reader，"
                              f"共 {c[2]} 条音频）" for c in cands))
        elif not per:
            hint = ("\n  提示：该目录下没扫到任何音频（布局应为 "
                    f"{layout['note']}）。LibriSpeech 池常多一层目录"
                    "（…/librispeech/LibriSpeech），--root 要给**含 subset 那层**")
        raise SystemExit(f"[ext-lists] 有 {len(missing)} 个 reader 在池内没有音频"
                         f"（例：{missing[:3]}）——reader 清单与池不匹配{hint}")
    out = []
    for r in readers:
        # 平台无关序（t19 的同一实现/同一键）：池内相对路径的 UTF-8 字节序
        out.extend(sorted(per[r], key=lambda rel: S.pool_sort_key(root, root / rel)))
    return out, per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=sorted(LAYOUT), required=True)
    ap.add_argument("--root", type=Path, required=True, help="语料池根目录")
    ap.add_argument("--readers", type=Path, required=True, help="训练份额 reader 清单")
    ap.add_argument("--holdout-readers", type=Path, default=None,
                    help="评测份额 reader 清单（--verify 时断言零交集）")
    ap.add_argument("--out", type=Path, default=None, help="输出清单路径（--verify 时可省）")
    ap.add_argument("--verify", action="store_true",
                    help="只校验：与已存在清单（--out）比对 + 打印统计，不写盘")
    args = ap.parse_args()
    if args.out is None and not args.verify:
        raise SystemExit("[ext-lists] 需要 --out（或 --verify 只校验已存在的清单）")

    layout = LAYOUT[args.corpus]
    readers = read_list(args.readers)
    files, per = build(args.root, readers, layout)
    if len(files) != len(set(files)):
        raise SystemExit("[ext-lists] 生成结果内有重复行（池内路径不唯一？）")
    stats = {
        "corpus": args.corpus, "layout": layout["note"], "root": str(args.root),
        "readers": len(readers), "files": len(files),
        "sha256": hashlib.sha256(("\n".join(files) + "\n").encode("utf-8")).hexdigest(),
    }
    print(f"[ext-lists] {args.corpus}: {len(readers)} readers → {len(files)} files "
          f"（sha256 {stats['sha256'][:12]}…）")
    if args.holdout_readers:
        hold = set(read_list(args.holdout_readers))
        leak = sorted(set(readers) & hold)
        if leak:
            raise SystemExit(f"[ext-lists] 训练 reader 与 holdout reader 有交集：{leak[:5]}")
        # 文件级双保险：按**路径字段**判 reader（不是子串！见 reader_of_rel 的坑）。
        # 走命名函数 = 单测锁定的那条实现，避免"测试过的判据"与"线上判据"分叉。
        bad = leaked_files(files, hold, layout)
        if bad:
            raise SystemExit(f"[ext-lists] 生成清单含 holdout reader 的文件：{bad[:3]}")
        print(f"[ext-lists] 隔离断言通过：train∩holdout readers = 0（holdout {len(hold)}），"
              f"生成清单 {len(files)} 行中 0 行属于 holdout reader（按路径字段判 reader）")
    if args.verify:
        if args.out is None:
            return 0
        if not args.out.is_file():
            raise SystemExit(f"[ext-lists] --verify 需要已存在的清单：{args.out}")
        old = read_list(args.out)
        if old != files:
            diff = next((i for i, (a, b) in enumerate(zip(old, files)) if a != b),
                        min(len(old), len(files)))
            raise SystemExit(f"[ext-lists] 与既有清单不一致（首个差异在第 {diff + 1} 行；"
                             f"既有 {len(old)} 行 vs 现场 {len(files)} 行）")
        print(f"[ext-lists] verify PASS：{args.out} 与现场重建逐行一致（{len(files)} 行）")
        return 0
    text = "".join(f"{n}\n" for n in files)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"[ext-lists] 写出 {args.out}（{len(files)} 行，UTF-8 无 BOM/LF）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
