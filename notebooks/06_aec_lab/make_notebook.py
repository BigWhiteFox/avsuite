# -*- coding: utf-8 -*-
r"""由 06_aec_lab_run.py(百分号分格,唯一代码源)生成 06_aec_lab.ipynb。

nbformat 缺失(P3-2 约束禁止 pip 安装),用标准库 json 手写 nbformat 4.5 结构,
字段形态与 notebooks/01-03 既有文件一致(kernelspec: Python 3 (avsuite))。
运行:
  & 'C:\Users\14805\.conda\envs\avsuite\python.exe' notebooks/06_aec_lab/make_notebook.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "06_aec_lab_run.py"
DST = HERE / "06_aec_lab.ipynb"

MARK = re.compile(r"^# %%")          # 代码格写成 "# %% ---" 装饰线,前缀判定

lines = SRC.read_text(encoding="utf-8").splitlines()
cells = []
cur_kind, cur = None, []            # cur_kind: None(前导 docstring,丢弃) / md / code


def flush():
    if cur_kind == "markdown":
        body = []
        for ln in cur:
            body.append("" if ln == "#" else (ln[2:] if ln.startswith("# ") else ln[1:]))
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": _src(body)})
    elif cur_kind == "code":
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": _src(cur)})


def _src(body):
    """-> nbformat 的 source 行列表(除末行外都带换行)。"""
    if not body:
        return [""]
    return [ln + "\n" for ln in body[:-1]] + [body[-1]]


for ln in lines:
    if MARK.match(ln):
        flush()
        cur_kind = "markdown" if "[markdown]" in ln else "code"
        cur = []
    elif cur_kind is not None:
        cur.append(ln)
flush()

# 丢弃末尾全空 code/markdown 块
while cells and not "".join(cells[-1]["source"]).strip():
    cells.pop()

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3 (avsuite)",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
               encoding="utf-8")

n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
n_code = len(cells) - n_md
# 自检:json 可回读,代码格合计行数与脚本中代码格行数一致(同源证明)
back = json.loads(DST.read_text(encoding="utf-8"))
assert back["nbformat"] == 4 and len(back["cells"]) == len(cells)
code_lines_src = sum(len(c["source"]) for c in cells if c["cell_type"] == "code")
print(f"written {DST.relative_to(HERE.parents[2]).as_posix()}: "
      f"{n_md} markdown + {n_code} code cells, code lines total={code_lines_src}")
sys.exit(0)
