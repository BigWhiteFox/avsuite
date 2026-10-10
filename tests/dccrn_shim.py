"""pytest 收集辅助：exp/04_dccrn 模块加载（作用域隔离版）。

exp/03_crn 与 exp/04_dccrn 都有 model.py/train.py——sys.modules 先到先得。
本 shim 在加载 DCCRN 模块时临时弹出已注册的同名模块（CRN 版），加载完成后
恢复；已绑定的对象引用（X.DCCRN 等）不受 sys.modules 恢复影响。
"""
import importlib
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parents[1] / "exp" / "04_dccrn"
_COLLIDING = ("model", "train", "complex_nn", "dynamic_mixing")

_saved = {}
for name in _COLLIDING:
    if name in sys.modules:
        _saved[name] = sys.modules.pop(name)

sys.path.insert(0, str(_DIR))
try:
    complex_nn = importlib.import_module("complex_nn")
    model = importlib.import_module("model")
    dynamic_mixing = importlib.import_module("dynamic_mixing")
    train = importlib.import_module("train")
    DCCRN = model.DCCRN
finally:
    # 恢复 CRN 版同名模块（exp03_shim / test_crn 可能依赖），DCCRN 版以别名保留
    for alias in _COLLIDING:
        mod = sys.modules.pop(alias, None)
        if mod is not None:
            sys.modules[f"dccrn_{alias}"] = mod
    sys.modules.update(_saved)
