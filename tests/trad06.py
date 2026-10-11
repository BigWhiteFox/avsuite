"""把 exp/06_trad_aec（数字开头目录）以包别名 `trad06` 暴露给 pytest。

sys.path 只在加载期临时插入（v3a-026）：`res`/`pipeline`/`gcc` 等裸名很通用，
永久留在 sys.path 上会遮蔽同名模块。加载完成后移除；已加载模块留在
sys.modules，后续 `from pipeline import ...`（含 exp/07 等处的惰性导入）走缓存。
"""
import importlib
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parents[1] / "exp" / "06_trad_aec"

sys.path.insert(0, str(_DIR))
try:
    gcc = importlib.import_module("gcc")
    nlms = importlib.import_module("nlms")
    mdf = importlib.import_module("mdf")
    dtd = importlib.import_module("dtd")
    res = importlib.import_module("res")
    pipeline = importlib.import_module("pipeline")
finally:
    if str(_DIR) in sys.path:
        sys.path.remove(str(_DIR))
