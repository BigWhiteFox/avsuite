"""pytest 收集辅助：exp/07_nn_aec 作用域加载（防与 03/04 的 model/train 冲突）。"""
import importlib
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parents[1] / "exp" / "07_nn_aec"
_COLLIDING = ("model", "train", "dataset")     # 'eval' 不入列：与内建名冲突且测试不依赖

_saved = {}
for name in _COLLIDING:
    if name in sys.modules:
        _saved[name] = sys.modules.pop(name)

sys.path.insert(0, str(_DIR))
try:
    model = importlib.import_module("model")
    dataset = importlib.import_module("dataset")
    train = importlib.import_module("train")
    NNAEC = model.NNAEC
finally:
    for alias in _COLLIDING:
        mod = sys.modules.pop(alias, None)
        if mod is not None:
            sys.modules[f"nnaec_{alias}"] = mod
    sys.modules.update(_saved)
