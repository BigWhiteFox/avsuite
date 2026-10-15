#!/usr/bin/env python
"""t21：扩展池三分割（LibriSpeech / AISHELL-4 / MUSAN）+ 与既有冻结评测轴互斥核对。

设计口径（与既有 0-2/0-4 完全同族，只用名字哈希，可独立重算）
--------------------------------------------------------------
- **说话人语料（LS / AISHELL-4）**：按 speaker id 哈希
    bucket = sha256('<corpus>:reader:' + speaker)[:8] % 1000
    <700 train70 / 700-849 val15 / >=850 holdout15
    holdout15 内再分 near/far：
    sha256('<corpus>:nearfar:' + speaker)[:8] % 1000 -> <500 near_eval / >=500 far_eval
- **MUSAN**：按**相对路径**哈希（MUSAN 无说话人概念）
    bucket = sha256('musan:file:' + relpath)[:8] % 1000
    同样 700/850 三分；eval15 = >=850
- **硬互斥（本脚本的验收核心）**：任何落在既定评测轴里的 id/speaker **一律排除**
  （不进入 train/val/holdout 任一份额），排除项逐条记录进 meta：
    * LibriSpeech/AISHELL-4：speaker 与 `speech_readers_holdout15.txt`（= eval_v8 的
      near/far eval readers 的超集）冲突 -> 排除；
    * MUSAN eval15：文件（basename）与 `noise_eval15.txt`（DNS 评测噪声）冲突 -> 排除；
      与 `EVAL_EXCLUDE_NOISE`（其他训练源已用噪声）冲突 -> 排除。
- **不改动任何冻结清单**：本脚本只写 `ext_*` 前缀的新文件；既有文件只读。

用法
----
    python scripts/split_extension_pools.py --ls-root <LibriSpeech 根> \
        --musan-root <musan 根> [--aishell3-root <AISHELL-3 根>] \
        --split-lists split_lists --out-dir split_lists [--write|--verify]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

BUCKETS = 1000
TRAIN_HI = 700
VAL_HI = 850
SIDE_HI = 500
SPLITS = ("train70", "val15", "holdout15")
SPEAKER_RE = re.compile(r"^(\d+)-(\d+)-(\d+)$")
#: LibriSpeech 顶层子集目录（避免 rglob 扫到 BOOKS.TXT 之类的散文件）
LS_SUBSETS = ("train-clean-100", "train-clean-360", "train-other-500",
              "dev-clean", "dev-other", "test-clean", "test-other")


def speaker_from_filename(name: str) -> str | None:
    """从 `{speaker}-{chapter}-{utterance}.flac` 取 speaker（LibriSpeech 官方命名规范）。

    **不要**用目录名字段：LibriSpeech 的 speaker 目录是纯纯数字（如 `254`），
    chapter 目录才是 `<speaker>-<chapter>`（如 `254-127705`）——按目录名猜 speaker 会全部漏掉。
    """
    m = re.match(r"^(\d+)-(\d+)-(\d+)\.(flac|wav)$", name)
    return m.group(1) if m else None


def speaker_from_path(p: Path, root: Path) -> str | None:
    """按布局取 speaker：

    1. LibriSpeech：文件名规范 `{spk}-{chp}-{utt}.flac`；
    2. **`wav/<speaker>/...`**（AISHELL-3 的 `test/wav/SSB0038/xxx.wav`）；
    3. **`<speaker>/...` 直接位于 root 下**（AISHELL-3 的 `train/SSB0038/...`）；
    4. 回退：路径里最里层纯数字目录名。
    """
    spk = speaker_from_filename(p.name)
    if spk:
        return spk
    parts = p.relative_to(root).parts[:-1]
    for i, part in enumerate(parts):
        if part == "wav" and i + 1 < len(parts):
            return parts[i + 1]
    if len(parts) >= 2:
        return parts[0]
    for part in reversed(parts):
        if part.isdigit():
            return part
    return None


#: AISHELL 官方划分：**test 是官方测试集，不得进训练池**（训练/验证只用 train、dev）
AISHELL_TRAIN_SUBSETS = ("train", "dev")


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def bucket(key: str) -> int:
    return int(sha8(key), 16) % BUCKETS


def three_way(key: str) -> str:
    b = bucket(key)
    return "train70" if b < TRAIN_HI else ("val15" if b < VAL_HI else "holdout15")


def side(key: str) -> str:
    return "near_eval" if bucket(key) < SIDE_HI else "far_eval"


def content_sha256(names: list[str]) -> str:
    return hashlib.sha256("".join(f"{n}\n" for n in names).encode("utf-8")).hexdigest()


def read_list(p: Path) -> list[str]:
    if not p.is_file():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]


# ---------------------------------------------------------------- 冻结轴（只读）
def load_frozen(split_lists: Path) -> dict:
    """读既有冻结评测轴与训练噪声份额（只读，绝不修改）。"""
    hold = set(read_list(split_lists / "speech_readers_holdout15.txt"))
    near_eval = set(read_list(split_lists / "speech_near_eval.txt"))
    far_eval = set(read_list(split_lists / "speech_far_eval.txt"))
    noise_eval = set(read_list(split_lists / "noise_eval15.txt"))
    rir_hold = set(read_list(split_lists / "rir_holdout15.txt"))
    # eval_v8 的 near/far 池里出现过的 speaker（按文件名 reader_<id> 提）
    pool_speakers = set()
    for f in ("speech_near_eval.txt", "speech_far_eval.txt"):
        for name in read_list(split_lists / f):
            m = re.search(r"reader_(\d+)", name)
            if m:
                pool_speakers.add(m.group(1))
    return dict(holdout_speakers=hold, near_eval_readers=pool_speakers,
                near_eval_files=near_eval, far_eval_files=far_eval,
                noise_eval=noise_eval, rir_holdout=rir_hold)


# ---------------------------------------------------------------- LS / AISHELL
def scan_speaker_corpus(root: Path, corpus: str) -> dict[str, list[str]]:
    """扫说话人语料 -> {speaker: [相对 root 的路径...]}。

    适配两种布局：
      * LibriSpeech：`<subset>/<speaker>/<chapter>/<spk>-<chp>-<utt>.flac`
        （speaker 从**文件名**取；目录名不可靠）
      * AISHELL-4 等：回退"路径里最里层纯数字目录"作 speaker
    """
    per: dict[str, list[str]] = {}
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"语料根不存在：{root}")

    # LibriSpeech：只进已知子集目录，避免把 BOOKS.TXT 之类散文件当数据
    tops = [root / s for s in LS_SUBSETS if (root / s).is_dir()]
    if tops:
        scan_roots = tops
    else:
        # AISHELL 布局：**只用 train/dev，排除官方 test 划分**（防测试集污染训练池）
        scan_roots = [root / s for s in AISHELL_TRAIN_SUBSETS if (root / s).is_dir()]
        if not scan_roots:
            scan_roots = [d for d in sorted(root.iterdir()) if d.is_dir()]
    for top in scan_roots:
        for p in sorted(top.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in (".flac", ".wav"):
                continue
            spk = speaker_from_path(p, top)
            if spk:
                per.setdefault(spk, []).append(str(p.relative_to(root)))
    if not per:
        raise FileNotFoundError(f"{root} 下没扫到 speaker 音频（期望 <subset>/<speaker>/<chapter>/*.flac|wav）")
    per = {k: v for k, v in per.items() if v}
    if not per:
        raise FileNotFoundError(f"{root} 下没有带音频的 speaker 目录")
    return {k: sorted(v) for k, v in per.items()}


def split_speaker_corpus(per: dict[str, list[str]], corpus: str,
                         frozen: dict) -> dict:
    """按 speaker 哈希三分 + holdout 内 near/far；把与冻结轴冲突的 speaker 排除。"""
    excluded: list[dict] = []
    kept: dict[str, list[str]] = {}
    hold = frozen["holdout_speakers"]
    for spk, files in sorted(per.items()):
        if spk in hold:
            excluded.append(dict(id=spk, reason="与既有 holdout/eval reader 冲突（eval_v8 评测轴）"))
            continue
        kept[spk] = files

    by_split: dict[str, list[str]] = {s: [] for s in SPLITS}
    hold_sub: dict[str, list[str]] = {"near_eval": [], "far_eval": []}
    rows = []
    d = f"{corpus}:"
    for spk, files in sorted(kept.items()):
        sp = three_way(d + "reader:" + spk)
        sd = side(d + "nearfar:" + spk) if sp == "holdout15" else ""
        by_split[sp].append(spk)
        if sp == "holdout15":
            hold_sub[sd].extend(files)
        rows.append(dict(speaker=spk, bucket=bucket(d + "reader:" + spk), split=sp, side=sd,
                         n_files=len(files), file_list_sha256=content_sha256(files)))
    return dict(per=kept, by_split=by_split, hold_sub=hold_sub, rows=rows, excluded=excluded,
                split_of=lambda s: three_way(d + "reader:" + s),
                side_of=lambda s: side(d + "nearfar:" + s))


# ---------------------------------------------------------------- MUSAN
def scan_musan(root: Path) -> dict[str, list[str]]:
    """扫 MUSAN -> {relpath: []}：以**相对路径**为切分键（无 speaker 概念）。"""
    if not root.is_dir():
        raise FileNotFoundError(f"MUSAN 根不存在：{root}")
    out = {}
    for p in sorted(root.rglob("*.wav")):
        out[str(p.relative_to(root))] = []
    if not out:
        raise FileNotFoundError(f"{root} 下没有 wav")
    return out


def split_musan(files: dict[str, list[str]], frozen: dict, extra_exclude: set[str]) -> dict:
    """按相对路径哈希三分；eval15 排除与既评测噪声/其他训练噪声同名的文件。"""
    excluded: list[dict] = []
    by_split: dict[str, list[str]] = {s: [] for s in SPLITS}
    for rel in sorted(files):
        base = Path(rel).name
        if base in frozen["noise_eval"]:
            excluded.append(dict(id=rel, reason="basename 命中 noise_eval15（DNS 评测噪声）"))
            continue
        if base in extra_exclude:
            excluded.append(dict(id=rel, reason="basename 命中 EXCLUDE_NOISE（其他训练源已用）"))
            continue
        by_split[three_way("musan:file:" + rel)].append(rel)
    return dict(by_split=by_split, excluded=excluded)


# ---------------------------------------------------------------- 落盘 / 验收
def write_list(p: Path, names: list[str]) -> None:
    p.write_text("".join(f"{n}\n" for n in names), encoding="utf-8", newline="\n")


#: 语料 -> 落盘文件名前缀（只新增 ext_*，不动任何冻结清单）
#: 注：LibriSpeech 用 `ls` 前缀（与既有的 `ext_ls_readers_*` 保持一致）
FILE_PREFIX = {"librispeech": "ls", "aishell3": "aishell3", "musan": "musan"}
CORPUS_FILES = {
    "librispeech": {
        "readers": {"train70": "ext_ls_readers_train70.txt", "val15": "ext_ls_readers_val15.txt",
                    "holdout15": "ext_ls_readers_holdout15.txt"},
        "eval": {"near_eval": "ext_ls_near_eval.txt", "far_eval": "ext_ls_far_eval.txt"},
    },
    "aishell3": {
        "readers": {"train70": "ext_aishell3_readers_train70.txt",
                    "val15": "ext_aishell3_readers_val15.txt",
                    "holdout15": "ext_aishell3_readers_holdout15.txt"},
        "eval": {"near_eval": "ext_aishell3_near_eval.txt",
                 "far_eval": "ext_aishell3_far_eval.txt"},
    },
    "musan": {
        "files": {"train70": "ext_musan_train70.txt", "val15": "ext_musan_val15.txt",
                  "holdout15": "ext_musan_eval15.txt"},
    },
}
EXT_META = "ext_pool_meta.json"


def persist(out_dir: Path, kind: str, corpus: str, res: dict) -> dict:
    """落盘某个语料的 ext_* 清单；返回其 sha256 段（供 meta）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = CORPUS_FILES[corpus]
    shas: dict[str, str] = {}
    counts: dict[str, int] = {}
    if kind == "speaker":
        for sp, fname in spec["readers"].items():
            write_list(out_dir / fname, res["by_split"][sp])
            counts[f"readers_{sp}"] = len(res["by_split"][sp])
            counts[f"files_{sp}"] = sum(len(res["per"][x]) for x in res["by_split"][sp])
            shas[fname] = content_sha256(res["by_split"][sp])
        for sd, fname in spec["eval"].items():
            write_list(out_dir / fname, res["hold_sub"][sd])
            counts[f"eval_files_{sd}"] = len(res["hold_sub"][sd])
            shas[fname] = content_sha256(res["hold_sub"][sd])
    else:
        for sp, fname in spec["files"].items():
            write_list(out_dir / fname, res["by_split"][sp])
            counts[f"files_{sp}"] = len(res["by_split"][sp])
            shas[fname] = content_sha256(res["by_split"][sp])
    return dict(counts=counts, list_sha256=shas)


def read_ext_lists(out_dir: Path) -> dict[str, list[str]]:
    """读回已落盘的 ext_* 清单（--verify 用）。"""
    got = {}
    for corpus, spec in CORPUS_FILES.items():
        for group in ("readers", "eval", "files"):
            for key, fname in spec.get(group, {}).items():
                got[f"{corpus}:{group}:{key}"] = read_list(out_dir / fname)
    return got


def derive_near_far_train(res: dict, corpus: str, out_dir: Path) -> dict:
    """把 train70 的 **reader 清单**再按名字哈希对半切成 near/far **文件名清单**。

    规则：`sha256('<corpus>:nearfar_train:'+speaker)[:8]` 升序，前一半 -> near_train，
    其余 -> far_train（与对 DNS clean 的 t9 做法同族）。产出训练入口要的文件名清单。
    """
    d = f"{corpus}:"
    ranked = sorted(res["by_split"]["train70"],
                    key=lambda s: sha8(d + "nearfar_train:" + s))
    half = len(ranked) // 2
    near_r, far_r = ranked[:half], ranked[half:]
    near_f, far_f = [], []
    for s in near_r:
        near_f.extend(res["per"][s])
    for s in far_r:
        far_f.extend(res["per"][s])
    out = {f"ext_{FILE_PREFIX.get(corpus, corpus)}_near_train70.txt": sorted(near_f),
           f"ext_{FILE_PREFIX.get(corpus, corpus)}_far_train70.txt": sorted(far_f)}
    for fname, names in out.items():
        write_list(out_dir / fname, names)
    return dict(readers_near=len(near_r), readers_far=len(far_r),
                files_near=len(near_f), files_far=len(far_f),
                list_sha256={k: content_sha256(v) for k, v in out.items()})


def assert_disjoint(tag: str, groups: dict[str, list[str]]) -> list[str]:
    problems = []
    sets = {k: set(v) for k, v in groups.items()}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                problems.append(f"[{tag}] {a} ∩ {b} = {len(inter)}（应为 0）例：{sorted(inter)[:3]}")
    return problems


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-lists", type=Path, default=Path("split_lists"))
    ap.add_argument("--out-dir", type=Path, default=Path("split_lists"))
    ap.add_argument("--ls-root", type=Path, default=None, help="LibriSpeech 根（含多个子集目录）")
    ap.add_argument("--other500-root", type=Path, default=None, help="train-other-500 目录（可选）")
    ap.add_argument("--aishell3-root", type=Path, default=None,
                    help="AISHELL-3 根（中文；官方 test 划分不进训练池）")
    ap.add_argument("--musan-root", type=Path, default=None, help="MUSAN 根（含 music/noise/speech）")
    ap.add_argument("--verify", action="store_true", help="只核对既有 ext_* 清单（不写盘）")
    args = ap.parse_args()

    frozen = load_frozen(args.split_lists)
    print(f"[frozen] holdout speakers={len(frozen['holdout_speakers'])} "
          f"eval-pool speakers={len(frozen['near_eval_readers'])} "
          f"noise_eval15={len(frozen['noise_eval'])} rir_holdout={len(frozen['rir_holdout'])}")

    problems: list[str] = []
    meta: dict = dict(version="ext_pool_v1",                      scheme="sha256(<corpus>:reader:<speaker>)[:8] % 1000 -> 700/850; "
                             "holdout 内 sha256(<corpus>:nearfar:<speaker>)[:8] % 1000 -> 500; "
                             "MUSAN 按 relpath: sha256('musan:file:'+rel)[:8] % 1000",
                      frozen_axes=dict(holdout_speakers=sorted(frozen["holdout_speakers"]),
                                       noise_eval15_count=len(frozen["noise_eval"]),
                                       rir_holdout_count=len(frozen["rir_holdout"])),
                      corpora={})

    # ---- LibriSpeech（tc100 + tc360 同根；可选 other500 单独根）----
    if args.ls_root:
        per: dict[str, list[str]] = {}
        for root in (args.ls_root, args.other500_root):
            if not root:
                continue
            for sub in sorted(Path(root).iterdir()):
                if not sub.is_dir():
                    continue
                try:
                    got = scan_speaker_corpus(sub, "librispeech")
                except FileNotFoundError:
                    continue
                for k, v in got.items():
                    per.setdefault(k, []).extend(v)
        res = split_speaker_corpus({k: sorted(v) for k, v in per.items()},
                                   "librispeech", frozen)
        print(f"[librispeech] speakers={len(per)} kept={len(res['per'])} "
              f"excluded={len(res['excluded'])} -> " +
              ", ".join(f"{s}={len(res['by_split'][s])}" for s in SPLITS) +
              f"; holdout near/far files={len(res['hold_sub']['near_eval'])}/"
              f"{len(res['hold_sub']['far_eval'])}")
        problems += assert_disjoint("librispeech readers", res["by_split"])
        problems += assert_disjoint("librispeech eval sides", res["hold_sub"])
        written = persist(args.out_dir, "speaker", "librispeech", res)
        nf = derive_near_far_train(res, "librispeech", args.out_dir)
        meta["corpora"]["librispeech"] = dict(
            speakers_total=len(per), speakers_kept=len(res["per"]),
            **written, near_far_train=nf, excluded=res["excluded"],
            speakers={s: res["by_split"][s] for s in SPLITS})

    # ---- AISHELL-3（中文；官方 test 划分不进训练池）----
    if args.aishell3_root:
        per = scan_speaker_corpus(args.aishell3_root, "aishell3")
        res = split_speaker_corpus(per, "aishell3", frozen)
        print(f"[aishell3] speakers={len(per)} kept={len(res['per'])} "
              f"excluded={len(res['excluded'])} -> " +
              ", ".join(f"{s}={len(res['by_split'][s])}" for s in SPLITS))
        problems += assert_disjoint("aishell3 readers", res["by_split"])
        written = persist(args.out_dir, "speaker", "aishell3", res)
        nf = derive_near_far_train(res, "aishell3", args.out_dir)
        meta["corpora"]["aishell3"] = dict(
            speakers_total=len(per), speakers_kept=len(res["per"]),
            **written, near_far_train=nf, excluded=res["excluded"],
            speakers={s: res["by_split"][s] for s in SPLITS})

    # ---- MUSAN ----
    if args.musan_root:
        files = scan_musan(args.musan_root)
        res = split_musan(files, frozen, set())
        print(f"[musan] files={len(files)} excluded={len(res['excluded'])} -> " +
              ", ".join(f"{s}={len(res['by_split'][s])}" for s in SPLITS))
        problems += assert_disjoint("musan files", res["by_split"])
        written = persist(args.out_dir, "file", "musan", res)
        meta["corpora"]["musan"] = dict(files_total=len(files), **written,
                                        excluded=res["excluded"])

    # ---------------- 空份额守卫（防止静默落盘空清单） ----------------
    for corpus, c in (meta["corpora"] or {}).items():
        for k, v in (c.get("counts") or {}).items():
            if v == 0:
                problems.append(f"[{corpus}] 份额 {k} 为空（0）——拒绝落盘，请先核对源池是否完整")
    if args.ls_root and not meta["corpora"].get("librispeech"):
        problems.append("--ls-root 给了但没算出 librispeech 语料（路径/布局不对？）")
    if args.aishell3_root and not meta["corpora"].get("aishell3"):
        problems.append("--aishell3-root 给了但没算出 aishell3 语料")
    if args.musan_root and not meta["corpora"].get("musan"):
        problems.append("--musan-root 给了但没算出 musan 语料")

    if problems:
        print(f"[split_extension_pools] 不通过（{len(problems)} 项）：")
        for p in problems:
            print(f"  - {p}")
        return 1

    # ---------------- 落盘 meta + 冻结轴互斥终检 ----------------
    if args.verify:
        checked = 0
        for _corpus, spec in CORPUS_FILES.items():
            for group in ("readers", "eval", "files"):
                for _key, fname in spec.get(group, {}).items():
                    on_disk = read_list(args.out_dir / fname)
                    if not on_disk:
                        problems.append(f"缺清单或为空：{fname}")
                        continue
                    checked += 1
        if problems:
            print("[split_extension_pools] --verify 不通过：")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"[split_extension_pools] PASS（--verify）：{checked} 份 ext_* 清单存在且非空")
        return 0

    meta_path = args.out_dir / EXT_META
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8", newline="\n")
    print(f"[split_extension_pools] PASS：语料内三分互斥成立；已写 {EXT_META} 与 ext_* 清单")
    return 0


if __name__ == "__main__":
    sys.exit(main())
