#!/usr/bin/env python
"""0-2 验收脚本：核对噪声池三分割清单，训练用噪声与 eval 噪声必须零重叠。

判定项（任一不满足即 **退出码非零**）
-------------------------------------
1. **零重叠（本任务验收核心）**：``train70`` 与 ``eval15`` 的文件名集合交集必须为 0
   （并顺带核 ``val15`` 与两端的交集，槽位名任意）。
2. **覆盖完整且两两互斥**：三份名单并集 == 池内 wav 全量，且无重复行、无同名跨组。
3. **成员确实存在于池**：每行列出的文件名都能在 ``--noise-dir`` 下找到（只读 stat）。
4. **切分可复现（防手工改名单）**：逐文件名重算 ``sha256(filename)[:8] % 1000``，
   必须落回它所属清单声明的分组桶；即名单不是"手挑"的。
5. **权威表一致**：``noise_split_manifest.csv`` 逐行与三份名单一致（split/bucket/hash）。
6. **版本哈希一致**：名单/权威表的内容 sha256 与 ``noise_split_meta.json`` 记录相同
   （任何事后编辑都会被抓住）。meta **平台无关**：不写死本机绝对路径，只留
   ``pool_dir_hint`` 文档性提示；池位置由 ``--noise-dir`` / ``NOISE_POOL_DIR`` 决定。
7. **可重现（可选，``--reproduce``）**：现场用 ``scripts/split_noise_pool.py`` 生成到临时
   目录，与仓库内名单逐字节比对——证明"同一池 + 同一脚本 = 同一名单"。

退出码：0 = 全部通过；1 = 有不通过项；2 = 输入/环境错误（清单缺失、池目录不存在等）。

用法
----
    python scripts/verify_split.py --dir split_lists \\
        --noise-dir D:/data/dns-challenge-mirror/datasets/noise

    # 只核清单内部一致性（服务器上池尚未下载时可先跑这档）
    python scripts/verify_split.py --dir split_lists --skip-pool-check

    # 连"重跑生成器字节一致"一起核
    python scripts/verify_split.py --dir split_lists --noise-dir <池> --reproduce

    # 手工制造重叠，确认本脚本真的会失败（负向自检）
    python scripts/verify_split.py --dir <被污染副本> --noise-dir <池>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRODUCER = REPO / "scripts" / "split_noise_pool.py"
DEFAULT_DIR = REPO / "split_lists"


def load_producer(path: Path = PRODUCER):
    """按文件路径加载生成器模块——复用其哈希与阈值，避免两处实现漂移。"""
    if not path.exists():
        raise FileNotFoundError(f"生成器不存在: {path}")
    spec = importlib.util.spec_from_file_location("split_noise_pool_producer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_list(path: Path) -> list[str]:
    """读名单：UTF-8（容忍 BOM），忽略空行与行尾空白，保留原顺序以查重。"""
    if not path.exists():
        raise FileNotFoundError(f"名单缺失: {path}")
    text = path.read_text(encoding="utf-8-sig")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_lists(lists: dict[str, list[str]], P) -> tuple[list[str], dict[str, set[str]]]:
    """判定项 1+2：互斥与覆盖（不依赖池）。返回 (问题, 集合表)。"""
    problems: list[str] = []
    sets: dict[str, set[str]] = {}
    for g in P.SPLIT_NAMES:
        names = lists[g]
        if len(names) != len(set(names)):
            dupes = sorted({n for n in names if names.count(n) > 1})
            problems.append(f"{g} 名单内有重复行 {len(dupes)} 个: {dupes[:3]}")
        sets[g] = set(names)
        if not names:
            problems.append(f"{g} 名单为空（三分割不应有空组）")
    for i, a in enumerate(P.SPLIT_NAMES):
        for b in P.SPLIT_NAMES[i + 1:]:
            inter = sets[a] & sets[b]
            if inter:
                problems.append(f"{a} ∩ {b} = {len(inter)} 个（应为 0）: {sorted(inter)[:3]}")
    # 验收核心单列一条，便于报告直接引用
    tr_ev = sets[P.TRAIN_NAME] & sets[P.EVAL_NAME]
    if tr_ev:
        problems.append(f"[验收项] train 与 eval 噪声重叠 {len(tr_ev)} 个: {sorted(tr_ev)[:3]}")
    return problems, sets


def pool_dir_from_meta(dir_: Path, P) -> Path | None:
    """从 meta 推断池目录（**平台无关**）。

    优先顺序：``NOISE_POOL_DIR`` 环境变量 -> ``pool_dir_hint`` 相对当前工作目录解析
    -> 旧字段 ``noise_dir``（历史 meta 才有，绝对值、换机器即失效）。都不成立返回 None，
    由调用方给出"请显式 --noise-dir"的可操作提示，而不是拿写死的本机路径去猜。
    """
    mpath = dir_ / P.META_NAME
    if not mpath.is_file():
        return None
    meta = json.loads(mpath.read_text(encoding="utf-8-sig"))
    env = os.environ.get("NOISE_POOL_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    hint = (meta.get("pool_dir_hint") or "").strip()
    if hint:
        rel = hint.split("/", 1)[1] if hint.startswith("<") and "/" in hint else hint
        for base in (Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent):
            cand = (base / rel).resolve()
            if cand.is_dir():
                return cand
    legacy = (meta.get("noise_dir") or "").strip()
    if legacy and Path(legacy).is_dir():
        return Path(legacy)
    return None


def check_pool(sets: dict[str, set[str]], noise_dir: Path, P) -> list[str]:
    """判定项 3+4：成员存在 + 哈希桶归属正确。"""
    problems: list[str] = []
    pool = set(P.pool_filenames(noise_dir))
    listed = set().union(*sets.values())
    missing_on_disk = sorted(listed - pool)
    extra_on_disk = sorted(pool - listed)
    if missing_on_disk:
        problems.append(f"名单里有 {len(missing_on_disk)} 个文件不在池内: {missing_on_disk[:3]}")
    if extra_on_disk:
        problems.append(f"池内 {len(extra_on_disk)} 个 wav 未进任何名单: {extra_on_disk[:3]}")
    bad_bucket: list[str] = []
    for g in P.SPLIT_NAMES:
        for n in sets[g]:
            if P.split_of(n) != g:
                bad_bucket.append(f"{n}(声明 {g} -> 实算 {P.split_of(n)})")
    if bad_bucket:
        problems.append(f"{len(bad_bucket)} 个文件不满足哈希桶归属（名单被手工改过？）: "
                        f"{bad_bucket[:3]}")
    return problems


def check_manifest(dir_: Path, lists: dict[str, list[str]], P) -> list[str]:
    """判定项 5：权威表与名单一致。"""
    problems: list[str] = []
    mpath = dir_ / P.MANIFEST_NAME
    if not mpath.exists():
        return [f"权威表缺失: {mpath}"]
    rows: dict[str, dict] = {}
    with open(mpath, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = (row.get("filename") or "").strip()
            if not name:
                problems.append("权威表存在空 filename 行")
                continue
            if name in rows:
                problems.append(f"权威表重复行: {name}")
            rows[name] = row
    listed = set().union(*[set(v) for v in lists.values()])
    if set(rows) != listed:
        miss = sorted(listed - set(rows))[:3]
        extra = sorted(set(rows) - listed)[:3]
        problems.append(f"权威表与名单集合不一致（表缺 {miss} / 表多 {extra}）")
    for g in P.SPLIT_NAMES:
        for n in lists[g]:
            row = rows.get(n)
            if row is None:
                continue
            if (row.get("split") or "").strip() != g:
                problems.append(f"权威表 split 不符: {n} 表={row.get('split')} 名单={g}")
            h8 = P.filename_hash8(n)
            if (row.get("hash8") or "").strip().lower() != h8:
                problems.append(f"权威表 hash8 不符: {n} 表={row.get('hash8')} 实算={h8}")
            if str(row.get("bucket") or "").strip() != str(P.bucket_of(n)):
                problems.append(f"权威表 bucket 不符: {n}")
            if (row.get("sha256") or "").strip().lower() != hashlib.sha256(
                    n.encode("utf-8")).hexdigest():
                problems.append(f"权威表 sha256 不符: {n}")
    return problems


def check_meta(dir_: Path, lists: dict[str, list[str]], P) -> list[str]:
    """判定项 6：内容哈希与 meta 记录一致（事后编辑哨兵）。"""
    problems: list[str] = []
    mpath = dir_ / P.META_NAME
    if not mpath.exists():
        return [f"元数据缺失: {mpath}"]
    try:
        meta = json.loads(mpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"元数据不是合法 JSON: {e}"]
    rec = meta.get("list_sha256") or {}
    for g in P.SPLIT_NAMES:
        actual = sha256_text("".join(f"{n}\n" for n in lists[g]))
        if rec.get(g) and rec[g] != actual:
            problems.append(f"{g} 内容哈希与 meta 不符（名单被改过）: meta={rec[g][:12]} 实算={actual[:12]}")
    mpath2 = dir_ / P.MANIFEST_NAME
    if mpath2.exists() and meta.get("manifest_sha256"):
        actual = sha256_text(mpath2.read_text(encoding="utf-8-sig").replace("\r\n", "\n"))
        if meta["manifest_sha256"] != actual:
            problems.append("权威表内容哈希与 meta 不符（表被改过）")
    counts = meta.get("counts") or {}
    for g in P.SPLIT_NAMES:
        if counts.get(g) is not None and counts[g] != len(lists[g]):
            problems.append(f"{g} 计数与 meta 不符: meta={counts[g]} 实读={len(lists[g])}")
    return problems


def check_reproduce(dir_: Path, P, noise_dir: Path | None = None) -> list[str]:
    """判定项 7：现场重跑生成器，名单必须逐字节一致。

    池目录：优先用调用方解析出的 ``noise_dir``（--noise-dir / 环境变量 / meta hint），
    不再依赖 meta 里的绝对路径——meta 已平台无关化（只留 ``pool_dir_hint``）。
    """
    problems: list[str] = []
    meta = json.loads((dir_ / P.META_NAME).read_text(encoding="utf-8-sig"))
    noise_dir = noise_dir or pool_dir_from_meta(dir_, P)
    if noise_dir is None or not Path(noise_dir).is_dir():
        return ["--reproduce 需要可用的噪声池目录：请显式 --noise-dir <池>"
                "（或设 NOISE_POOL_DIR 环境变量）；meta 只留文档性 hint，不再写死绝对路径"]
    # 生成器只算不落盘（写盘是 main 的事），故这里无需临时目录
    art = P.build_artifacts(Path(noise_dir), meta.get("seed", ""))
    for g in P.SPLIT_NAMES:
        new = art["lists"][g]
        old = "".join(f"{n}\n" for n in read_list(dir_ / P.LIST_FILES[g]))
        if new != old:
            problems.append(f"重跑生成器后 {P.LIST_FILES[g]} 不一致（不可复现）")
    if art["manifest"] != (dir_ / P.MANIFEST_NAME).read_text(
            encoding="utf-8-sig").replace("\r\n", "\n"):
        problems.append("重跑生成器后权威表不一致（不可复现）")
    return problems


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 默认 GBK 控制台
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="清单目录（默认 avsuite/split_lists）")
    ap.add_argument("--noise-dir", type=Path, default=None,
                    help="噪声池目录（默认取 meta 记录值）")
    ap.add_argument("--producer", type=Path, default=PRODUCER, help="生成器脚本路径")
    ap.add_argument("--skip-pool-check", action="store_true", help="池不在本机时跳过 3/4 两项")
    ap.add_argument("--reproduce", action="store_true",
                    help="追加「重跑生成器字节一致」检查")
    args = ap.parse_args()

    try:
        P = load_producer(args.producer)
    except Exception as e:                                   # noqa: BLE001 - CLI 边界
        print(f"[verify_split] ERROR 无法加载生成器: {e}")
        return 2

    dir_ = Path(args.dir)
    problems: list[str] = []
    try:
        lists = {g: read_list(dir_ / P.LIST_FILES[g]) for g in P.SPLIT_NAMES}
    except FileNotFoundError as e:
        print(f"[verify_split] ERROR 输入不完整: {e}")
        return 2

    p, sets = check_lists(lists, P)
    problems += p
    noise_dir = args.noise_dir if args.noise_dir is not None else pool_dir_from_meta(dir_, P)
    if noise_dir is not None:
        try:
            problems += check_pool(sets, Path(noise_dir), P)
        except FileNotFoundError as e:
            print(f"[verify_split] ERROR 池目录不可用: {e}")
            return 2
    elif not args.skip_pool_check:
        problems.append("未能确定噪声池目录（--noise-dir 未给、NOISE_POOL_DIR 未设、"
                        "meta 只有文档性 hint）：池不在本机时请显式 --skip-pool-check，"
                        "否则请用 --noise-dir <池> 指定")
    problems += check_manifest(dir_, lists, P)
    problems += check_meta(dir_, lists, P)
    if args.reproduce:
        problems += check_reproduce(dir_, P, noise_dir)

    counts = {g: len(lists[g]) for g in P.SPLIT_NAMES}
    overlap = len(sets[P.TRAIN_NAME] & sets[P.EVAL_NAME])
    print(f"[verify_split] dir={dir_}")
    print(f"  train70={counts[P.TRAIN_NAME]} val15={counts[P.VAL_NAME]} "
          f"eval15={counts[P.EVAL_NAME]} 合计={sum(counts.values())}")
    print(f"  train∩eval = {overlap}（验收线 0）")
    if problems:
        print(f"  FAIL 共 {len(problems)} 项：")
        for x in problems:
            print(f"   - {x}")
        return 1
    print("  PASS 全部检查通过（零重叠 / 覆盖完整 / 哈希桶归属 / 权威表一致 / 版本哈希一致）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
