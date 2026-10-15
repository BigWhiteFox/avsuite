#!/usr/bin/env python
"""把 `eval_matrix_report.py` 生成的读数表**嵌入**报告模板的占位符（t13 收尾用）。

报告 `reports/p1_6_matrix_eval_20261014.md` 里以 `<<TAB_ABS>>`… 六个占位符标记要嵌入的章节；
本脚本按表格文件的 `## N. 标题` 切段，取**段体**（去掉标题行）替换对应占位符。

用法：
    python exp/07_nn_aec/embed_matrix_tables.py \
        --report reports/p1_6_matrix_eval_20261014.md \
        --tables exp/07_nn_aec/outputs/eval_matrix/_report/matrix_tables.md \
        [--check]        # 只检查占位符与章节是否齐备，不写盘
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: 占位符 -> 表格文件的章节号
MAP = {"<<TAB_ABS>>": "3", "<<TAB_PAIRED>>": "4", "<<TAB_CROSS>>": "5",
       "<<TAB_DID>>": "6", "<<TAB_PAIRS>>": "7", "<<TAB_ERLE>>": "8"}


def split_sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(\d+)\.", line)
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur is not None:
            out[cur].append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--strip", action="store_true",
                    help="反向：把 §3-§8 的表格行换回占位符（报告本身当模板源用）")
    args = ap.parse_args()
    rep = args.report.read_text(encoding="utf-8")

    if args.strip:
        # 反向操作：把 §3-§8 的表格行抽掉、原位换回占位符（保留各节标题与其间的说明文字）。
        # 用途：报告本体即"模板源"——改完正文后先 strip 回带占位符的版本，再由脚本重新嵌入。
        num_to_ph = {num: ph for ph, num in MAP.items()}
        out, cur = [], None
        for line in rep.splitlines():
            m = re.match(r"^##\s+(\d+)\.", line)
            if m:
                if cur and num_to_ph.get(cur) not in out:
                    out.append("")
                    out.append(num_to_ph[cur])
                cur = m.group(1) if m.group(1) in num_to_ph else None
                out.append(line)
                continue
            if cur and line.startswith("|"):
                continue                      # 丢掉表格行（表头/分隔/数据）
            out.append(line)
        if cur and num_to_ph.get(cur) not in out:
            out.append("")
            out.append(num_to_ph[cur])
        stripped = "\n".join(out).rstrip("\n") + "\n"
        left = re.findall(r"<<TAB_\w+>>", stripped)
        print(f"[embed --strip] 抽出表格后 {len(stripped.splitlines())} 行，占位符 {len(left)}/6：{left}")
        if not args.check:
            args.report.write_text(stripped, encoding="utf-8", newline="\n")
            print(f"   -> {args.report}")
        return 0 if len(left) == len(MAP) else 2

    secs = split_sections(args.tables.read_text(encoding="utf-8"))
    missing = [n for n in MAP.values() if n not in secs]
    if missing:
        print(f"[embed] 表格缺章节：{missing}", file=sys.stderr)
        return 2
    filled = rep
    for ph, num in MAP.items():
        body = "\n".join(secs[num]).strip("\n")
        if ph not in filled:
            print(f"[embed] 报告缺占位符 {ph}", file=sys.stderr)
            return 2
        filled = filled.replace(ph, body)
    left = re.findall(r"<<TAB_\w+>>", filled)
    if left:
        print(f"[embed] 仍有未替换占位符：{left}", file=sys.stderr)
        return 2
    n_rows = sum(1 for ln in filled.splitlines() if ln.startswith("| "))
    print(f"[embed] 6 个章节已嵌入；报告 {len(filled.splitlines())} 行，其中表格行 {n_rows}")
    for ph, num in MAP.items():
        print(f"   {ph} <- §{num}（{len(secs[num])} 行）")
    if not args.check:
        args.report.write_text(filled, encoding="utf-8", newline="\n")
        print(f"   -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
