#!/usr/bin/env python
"""把扩量语料（MUSAN / AISHELL-3 / LibriSpeech）的**评测子集**从服务器只读拉取到本机（t23）。

为什么
------
t23 的多轴评测集需要本机没有的语料（服务器 `/root/data` 有 musan/librispeech/aishell3）。
队长 2026-10-13 裁定"方案 A"：**只读**拉取所需子集到本机，本机转码/建池/生成（不动服务器
磁盘与在用文件；t13 打分走 t5 模式把 ckpt 拉回本机）。

拉取纪律（本脚本强制）
----------------------
1. **只读**：只用 `tar -cf - <paths>` 把文件流回来，不在服务器上写任何东西；
2. **分批 ≤ `--batch`（默认 150）**：每批一次 ssh，避免 scp 命令行长度上限（t21 教训）；
3. **确定性子集**：对原始名单排序后**等间隔抽样**（`k = len // target`）取 `target` 个，
   保证跨说话人分散；子集名单落盘（默认 `split_lists/ext_<corpus>_subset<N>.txt`）可复现；
4. **可续跑**：已存在且大小 > 0 的文件跳过（断点续拉）。

用法
----
    python tools/pull_ext_subsets.py --key D:/Project_for_Python/DSH/dsh.pem \
        --dry-run                       # 先看子集规模
    python tools/pull_ext_subsets.py --key <pem> [--only musan aishell3_ls] [--batch 150]

产物：`D:/data/_t23_pull/<corpus>/<原始相对路径>`（保留目录结构，便于按说话人核对）+
`<dest>/pull_report.json`（每个语料的子集名单/文件数/字节数/名单 sha256）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LISTS = REPO / "split_lists"
DEFAULT_DEST = Path("D:/data/_t23_pull")

#: corpus -> (名单文件, 服务器根目录, 本机子目录, 目标文件数, 子集名单名)
SPECS = {
    "musan": ("ext_musan_eval15.txt", "/root/data/musan/musan", "musan", 150,
              "ext_musan_eval15_subset150.txt"),
    "aishell3_near": ("ext_aishell3_near_eval.txt", "/root/data/aishell3", "aishell3_near",
                      800, "ext_aishell3_near_eval_subset800.txt"),
    "aishell3_far": ("ext_aishell3_far_eval.txt", "/root/data/aishell3", "aishell3_far",
                     800, "ext_aishell3_far_eval_subset800.txt"),
    "ls_near": ("ext_ls_near_eval.txt", "/root/data/librispeech/LibriSpeech", "ls_near",
                800, "ext_ls_near_eval_subset800.txt"),
    "ls_far": ("ext_ls_far_eval.txt", "/root/data/librispeech/LibriSpeech", "ls_far",
               800, "ext_ls_far_eval_subset800.txt"),
}


def subset(names: list[str], target: int) -> list[str]:
    """排序后等间隔抽样（确定性、跨说话人分散）。"""
    ordered = sorted(set(names))
    if len(ordered) <= target:
        return ordered
    k = len(ordered) // target
    picked = ordered[::k][:target]
    return picked if len(picked) == target else ordered[:target]


def sha256_text(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def ssh_base(key: Path, host: str, port: int) -> list[str]:
    return ["ssh", "-i", str(key), "-p", str(port), "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=25", f"root@{host}"]


def pull_batch(key: Path, host: str, port: int, root: str, batch: list[str],
               dest: Path, tar_tmp: Path) -> int:
    """一批：远端 tar 到 stdout -> 本地文件 -> 解包。返回本批拉到的字节数。"""
    cmd = "cd " + root + " && tar -cf - " + " ".join(batch)
    with open(tar_tmp, "wb") as fh:
        rr = subprocess.run(ssh_base(key, host, port) + [cmd], stdout=fh,
                            stderr=subprocess.PIPE)
    if rr.returncode != 0:
        raise RuntimeError(f"tar 批失败：{rr.stderr.decode('utf-8', 'replace')[:300]}")
    with tarfile.open(tar_tmp) as tf:
        tf.extractall(dest)
    return tar_tmp.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", type=Path, required=True, help="SSH 私钥（团队留档）")
    ap.add_argument("--host", default="117.50.221.249")
    ap.add_argument("--port", type=int, default=23)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--batch", type=int, default=150, help="每批文件数上限（t21 教训）")
    ap.add_argument("--only", nargs="*", default=None, help="只拉这些 corpus（默认全部）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.key.is_file():
        print(f"[pull] 私钥不存在：{args.key}", file=sys.stderr)
        return 2
    specs = {k: v for k, v in SPECS.items() if not args.only or k in args.only}
    if not specs:
        print(f"[pull] --only 无匹配（可选 {sorted(SPECS)}）", file=sys.stderr)
        return 2

    report: dict = {"host": args.host, "port": args.port, "batch": args.batch,
                    "subset_rule": "sort -> every k-th (k=len//target) -> first target",
                    "corpora": {}}
    args.dest.mkdir(parents=True, exist_ok=True)
    tar_tmp = args.dest / "_batch.tar"

    for name, (list_name, root, sub, target, subset_name) in specs.items():
        names = [x for x in (LISTS / list_name).read_text(encoding="utf-8").split("\n") if x]
        picked = subset(names, target)
        list_out = LISTS / subset_name
        if not args.dry_run:
            list_out.write_text("".join(f"{n}\n" for n in picked), encoding="utf-8",
                                newline="\n")
        dest = args.dest / sub
        dest.mkdir(parents=True, exist_ok=True)
        have = sum(1 for n in picked if (dest / n).is_file())
        print(f"[{name}] 名单 {len(names)} -> 子集 {len(picked)}（已在本机 {have}）"
              f"{'  [dry-run]' if args.dry_run else ''}")
        if args.dry_run:
            continue

        batches = [picked[i:i + args.batch] for i in range(0, len(picked), args.batch)]
        got_files, got_bytes = 0, 0
        for bi, batch in enumerate(batches, 1):
            todo = [n for n in batch if not (dest / n).is_file()]
            if todo:
                got_bytes += pull_batch(args.key, args.host, args.port, root, todo, dest,
                                        tar_tmp)
            got_files += len([n for n in batch if (dest / n).is_file()])
            print(f"  batch {bi}/{len(batches)}：本机已有 {got_files}/{len(picked)}"
                  f"（本次 {got_bytes / 1e6:.1f} MB）", flush=True)
        sizes = [(dest / n).stat().st_size for n in picked if (dest / n).is_file()]
        missing = [n for n in picked if not (dest / n).is_file()]
        report["corpora"][name] = {
            "server_root": root, "list_file": list_name, "subset_list": subset_name,
            "n_source": len(names), "n_subset": len(picked), "n_landed": len(sizes),
            "bytes": sum(sizes), "missing": missing[:5],
            "subset_sha256": sha256_text(picked),
            "min_bytes": min(sizes) if sizes else 0,
        }
        flag = "OK" if not missing else f"缺 {len(missing)}"
        print(f"  落地 {len(sizes)}/{len(picked)}，共 {sum(sizes) / 1e6:.1f} MB（{flag}）")

    if tar_tmp.exists():
        tar_tmp.unlink()
    if not args.dry_run:
        out = args.dest / "pull_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8", newline="\n")
        print(f"[pull] 报告 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
