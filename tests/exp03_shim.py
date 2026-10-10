# -*- coding: utf-8 -*-
"""pytest 收集辅助：把 exp/03_crn 以包别名 `exp03` 暴露（目录名以数字开头不能直接 import）。"""
import importlib
import sys
from pathlib import Path

_CRN_DIR = Path(__file__).resolve().parents[1] / "exp" / "03_crn"
sys.path.insert(0, str(_CRN_DIR))

STFT = importlib.import_module("core.stft").STFT

dataset = importlib.import_module("dataset")
model = importlib.import_module("model")
train = importlib.import_module("train")
CRN = model.CRN
