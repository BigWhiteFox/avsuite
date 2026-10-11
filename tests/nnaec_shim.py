"""pytest 收集辅助：exp/07_nn_aec 作用域加载（防与 03/04 的 model/train 冲突）。

sys.path 只在加载期临时插入（v3a-026），加载完成后移除，避免 exp/07_nn_aec
永久留在 sys.path 上遮蔽同名模块。
"""
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
    if str(_DIR) in sys.path:
        sys.path.remove(str(_DIR))
    for alias in _COLLIDING:
        mod = sys.modules.pop(alias, None)
        if mod is not None:
            sys.modules[f"nnaec_{alias}"] = mod
    sys.modules.update(_saved)
