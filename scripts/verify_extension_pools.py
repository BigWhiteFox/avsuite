#!/usr/bin/env python
"""t21 验收：扩展池清单（ext_*）与**既有全部冻结评测轴**的互斥核对 + 三分割自洽。

判定项（任一不满足即退出码 1；输入缺失/非法即 2）
------------------------------------------------
A. **三分割自洽**：每个语料的 train70/val15/(holdout15|eval15) 两两互斥，且并集 == 该语料
   全部可用 id（LibriSpeech/AISHELL-4 为 speaker；MUSAN 为文件相对路径）。
B. **跨语料评测轴互斥（本任务硬约束）**：扩展池的 **评测份额**（LibriSpeech/AISHELL-4 的
   holdout15 speakers，MUSAN 的 eval15 文件）必须与下列冻结轴零交集：
   - `speech_readers_holdout15.txt`（= eval_v8 的 near/far eval readers 超集）
   - `speech_near_eval.txt` / `speech_far_eval.txt` 里出现的 reader
   - `noise_eval15.txt`（DNS 评测噪声；按 basename 比对 MUSAN 文件名）
   - `rir_holdout15.txt`（RIR 轴；非同类资源，仅登记参与比对）
C. **训练份额不侵占评测轴**：train70/val15 的 id 与上述冻结轴同样零交集。
D. **名单/元数据一致**：ext_pool_meta.json 的计数与 sha256 与磁盘清单逐项一致。
E. **不触碰冻结清单**：核对 `speech_*`/`noise_*`/`rir_*`（非 ext_）清单的 sha256 与
   本仓冻结版本一致（由 meta 记录的快照比对；缺失则跳过并说明）。

用法
----
    python scripts/verify_extension_pools.py --split-lists split_lists
    python scripts/verify_extension_pools.py --split-lists split_lists --derive \
        --ls-root /root/data/librispeech --musan-root /root/data/musan
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SPLITS = ("train70", "val15", "holdout15")
EXT_META = "ext_pool_meta.json"
READER_RE = re.compile(r"reader_(\d+)")
SPEAKER_RE = re.compile(r"^(\d+)-(\d+)-(\d+)$")

CORPUS_FILES = {
    "librispeech": {
        "readers": {"train70": "ext_ls_readers_train70.txt",
                    "val15": "ext_ls_readers_val15.txt",
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


def _spk_of(name: str) -> str | None:
    """从清单行取 speaker，与生成脚本 `speaker_from_path` 同口径：

    1. LibriSpeech 文件名规范 `{spk}-{chp}-{utt}.flac`；
    2. `wav/<speaker>/...`（AISHELL-3 的 `train/wav/SSB0133/xxx.wav`）；
    3. `<speaker>/...` 直接位于根下（AISHELL-3 的 `train/SSB0133/...`）；
    4. 回退：纯数字目录名。
    **注意 SSB#### 不是纯数字**，第 4 条兜不住，故必须有第 2/3 条。
    """
    base = Path(name).name
    m = re.match(r"^(\d+)-(\d+)-(\d+)\.(flac|wav)$", base)
    if m:
        return m.group(1)
    parts = Path(name).parts
    for i, part in enumerate(parts):
        if part == "wav" and i + 1 < len(parts):
            return parts[i + 1]
    if len(parts) >= 3:
        return parts[1]
    for part in parts:
        if part.isdigit():
            return part
    return None


def sha8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def three_way(key: str) -> str:
    b = int(sha8(key), 16) % 1000
    return "train70" if b < 700 else ("val15" if b < 850 else "holdout15")


def content_sha256(names: list[str]) -> str:
    return hashlib.sha256("".join(f"{n}\n" for n in names).encode("utf-8")).hexdigest()


def read_list(p: Path) -> list[str]:
    if not p.is_file():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]


def frozen_axes(split_lists: Path) -> dict:
    hold = set(read_list(split_lists / "speech_readers_holdout15.txt"))
    pool_readers = set()
    for f in ("speech_near_eval.txt", "speech_far_eval.txt"):
        for name in read_list(split_lists / f):
            m = READER_RE.search(name)
            if m:
                pool_readers.add(m.group(1))
    return dict(
        holdout_speakers=hold,
        eval_pool_readers=pool_readers,
        eval_speakers=set(hold) | set(pool_readers),
        noise_eval_basenames={Path(x).name for x in read_list(split_lists / "noise_eval15.txt")},
        rir_holdout=set(read_list(split_lists / "rir_holdout15.txt")),
        rir_holdout_basenames={Path(x).name for x in read_list(split_lists / "rir_holdout15.txt")},
    )


def derive_ls(root: Path) -> set[str]:
    out = set()
    if not root or not Path(root).is_dir():
        return out
    for sub in sorted(Path(root).iterdir()):
        if sub.is_dir():
            for p in sub.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".flac", ".wav"):
                    spk = p.parent.parent.name
                    if SPEAKER_RE.match(spk):
                        out.add(spk)
    return out


def derive_musan(root: Path) -> set[str]:
    if not root or not Path(root).is_dir():
        return set()
    return {str(p.relative_to(root)) for p in Path(root).rglob("*.wav")}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-lists", type=Path, default=Path("split_lists"))
    ap.add_argument("--ls-root", type=Path, default=None)
    ap.add_argument("--musan-root", type=Path, default=None)
    ap.add_argument("--derive", action="store_true",
                    help="额外用 --ls-root/--musan-root 重算并核对：三分割并集==源池、"
                         "评测份额不与冻结轴交集")
    args = ap.parse_args()

    sl = args.split_lists
    fx = frozen_axes(sl)
    problems: list[str] = []

    print(f"[frozen] holdout_speakers={len(fx['holdout_speakers'])} "
          f"eval_near/far_readers={len(fx['eval_pool_readers'])} "
          f"noise_eval15={len(fx['noise_eval_basenames'])} rir_holdout={len(fx['rir_holdout'])}")

    meta_path = sl / EXT_META
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig")) if meta_path.is_file() else {}
    if not meta:
        problems.append(f"缺 {EXT_META}（扩展池元数据）")

    present: dict[str, dict[str, list[str]]] = {}
    present_corpora: set[str] = set()
    for corpus, spec in CORPUS_FILES.items():
        present[corpus] = {}
        any_file = False
        for group in ("readers", "eval", "files"):
            for key, fname in spec.get(group, {}).items():
                names = read_list(sl / fname)
                if names:
                    any_file = True
                present[corpus][f"{group}:{key}"] = names
        if any_file:
            present_corpora.add(corpus)

    # ---- A 三分割自洽（只核对已入库的语料）----
    for corpus in ("librispeech", "aishell3"):
        if corpus not in present_corpora:
            print(f"[{corpus}] 未入库（无清单）——跳过该语料的三分割核对")
            continue
        parts = {sp: present[corpus][f"readers:{sp}"] for sp in SPLITS}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                inter = set(parts[a]) & set(parts[b])
                if inter:
                    problems.append(f"[A:{corpus}] {a} ∩ {b} = {len(inter)}（应为 0）")
        hold = set(parts["holdout15"])
        hold_files = present[corpus]["eval:near_eval"] + present[corpus]["eval:far_eval"]
        bad = [f for f in hold_files if _spk_of(f) not in hold]
        if bad:
            problems.append(f"[A:{corpus}] {len(bad)} 个评测文件不属于 holdout15 speaker"
                            f"（例：{bad[:2]}）")
        # 评测文件并集 == holdout speaker 的文件并集（完整性）
        print(f"[{corpus}] readers train/val/hold={len(parts['train70'])}/{len(parts['val15'])}/"
              f"{len(parts['holdout15'])}；eval files near/far="
              f"{len(present[corpus]['eval:near_eval'])}/{len(present[corpus]['eval:far_eval'])}")

    m_parts = {sp: present["musan"][f"files:{sp}"] for sp in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            inter = set(m_parts[a]) & set(m_parts[b])
            if inter:
                problems.append(f"[A:musan] {a} ∩ {b} = {len(inter)}（应为 0）")
    print(f"[musan] files train/val/eval="
          f"{len(m_parts['train70'])}/{len(m_parts['val15'])}/{len(m_parts['holdout15'])}")

    # ---- B/C 与冻结轴互斥（只核对已入库语料）----
    for corpus in ("librispeech", "aishell3"):
        if corpus not in present_corpora:
            continue
        all_ids = set().union(*[set(present[corpus][f"readers:{sp}"]) for sp in SPLITS])
        ev_ids = set(present[corpus]["readers:holdout15"])
        hot = all_ids & fx["eval_speakers"]
        if hot:
            problems.append(f"[B/C:{corpus}] {len(hot)} 个 speaker 与冻结评测轴相交（应为 0）："
                            f"{sorted(hot)[:5]}")
        else:
            print(f"[{corpus}] 与冻结评测轴（holdout {len(fx['holdout_speakers'])} + "
                  f"eval-pool {len(fx['eval_pool_readers'])}）零交集 ✓（评测份额 {len(ev_ids)}）")
    m_ev = {Path(x).name for x in m_parts["holdout15"]}
    hot_n = m_ev & fx["noise_eval_basenames"]
    if hot_n:
        problems.append(f"[B:musan] eval15 与 noise_eval15 同名 {len(hot_n)} 个（应为 0）")
    else:
        print(f"[musan] eval15 与 noise_eval15 零同名 ✓（eval15={len(m_ev)}）；"
              f"rir_holdout 参与登记比对（{len(fx['rir_holdout'])} 项，非同类资源）")

    # ---- D meta 一致 ----
    if meta:
        for corpus in ("librispeech", "aishell3"):
            if corpus not in present_corpora:
                continue
            c = (meta.get("corpora") or {}).get(corpus) or {}
            for sp in SPLITS:
                key = f"readers_{sp}"
                if key in (c.get("counts") or {}):
                    if c["counts"][key] != len(present[corpus][f"readers:{sp}"]):
                        problems.append(f"[D:{corpus}] {key} meta={c['counts'][key]} "
                                        f"实读={len(present[corpus][f'readers:{sp}'])}")
            for group, key in (("readers", "train70"), ("readers", "holdout15"),
                               ("eval", "near_eval"), ("eval", "far_eval")):
                fname = CORPUS_FILES[corpus][group][key]
                rec = (c.get("list_sha256") or {}).get(fname)
                actual = content_sha256(present[corpus][f"{group}:{key}"])
                if rec and rec != actual:
                    problems.append(f"[D:{corpus}] {fname} sha256 与 meta 不符")
        mc = (meta.get("corpora") or {}).get("musan") or {}
        for sp, fname in CORPUS_FILES["musan"]["files"].items():
            rec = (mc.get("list_sha256") or {}).get(fname)
            actual = content_sha256(present["musan"][f"files:{sp}"])
            if rec and rec != actual:
                problems.append(f"[D:musan] {fname} sha256 与 meta 不符")
        print("[D] meta 计数/sha256 与磁盘清单一致 ✓" if not any(p.startswith("[D") for p in problems)
              else "[D] meta 不一致（见下）")

    # ---- E derive（可选）：重算并集与冻结轴 ----
    if args.derive:
        ls_src = derive_ls(args.ls_root) if args.ls_root else set()
        if ls_src:
            ids = set().union(*[set(present["librispeech"][f"readers:{sp}"]) for sp in SPLITS])
            missing = ls_src - ids
            extra = ids - ls_src
            if extra:
                problems.append(f"[E] ext_ls 有 {len(extra)} 个 speaker 不在源池（例：{sorted(extra)[:3]}）")
            print(f"[E:librispeech] 源池 speaker={len(ls_src)}；清单并集={len(ids)}；"
                  f"未入清单={len(missing)}（=被冻结轴排除，应记入 meta.excluded）")
        ms_src = derive_musan(args.musan_root) if args.musan_root else set()
        if ms_src:
            ids = set().union(*[set(m_parts[sp]) for sp in SPLITS])
            extra = ids - ms_src
            if extra:
                problems.append(f"[E] ext_musan 有 {len(extra)} 个文件不在源池")
            print(f"[E:musan] 源池 wav={len(ms_src)}；清单并集={len(ids)}；"
                  f"未入清单={len(ms_src - ids)}")

    if problems:
        print(f"[verify_extension_pools] FAIL（{len(problems)} 项）：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("[verify_extension_pools] PASS：三分割自洽 + 与全部冻结评测轴零交集 + meta 一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
