"""把 exp/06_trad_aec（数字开头目录）以包别名 `trad06` 暴露给 pytest。"""
import importlib
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parents[1] / "exp" / "06_trad_aec"
sys.path.insert(0, str(_DIR))

gcc = importlib.import_module("gcc")
nlms = importlib.import_module("nlms")
mdf = importlib.import_module("mdf")
dtd = importlib.import_module("dtd")
res = importlib.import_module("res")
pipeline = importlib.import_module("pipeline")
