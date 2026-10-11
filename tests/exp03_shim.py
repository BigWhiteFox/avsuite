"""pytest 收集辅助：把 exp/03_crn 以包别名 `exp03` 暴露（目录名以数字开头不能直接 import）。

sys.path 只在**加载期**临时插入（v3a-026）：`dataset`/`model`/`train` 都是极通用的
裸名，把 exp/03_crn 永久留在 sys.path 上会遮蔽后续任何同名模块（作用域泄漏）。
加载完成后立即移除该条目；已加载模块留在 sys.modules，后续 `from model import ...`
走 sys.modules 缓存，不需要 sys.path。

同时按 §3.5 做 **sys.modules 防污染**（与 dccrn_shim/nnaec_shim 同款）：加载前把已注册的
同名裸模块（可能是 exp/07 的 model）临时弹出，加载完成后把 CRN 版以 `crn_*` 别名保留、
并把原对象恢复回去。这样收集顺序（例如先收集到 exp/09 探针把 `model` 注册成 exp/07 版）
不再影响本 shim —— 这是全量 pytest 下 test_crn 收集失败的根因所在。
"""
import importlib
import sys
from pathlib import Path

_CRN_DIR = Path(__file__).resolve().parents[1] / "exp" / "03_crn"
_COLLIDING = ("model", "train", "dataset")

_saved = {}
for _name in _COLLIDING:
    if _name in sys.modules:
        _saved[_name] = sys.modules.pop(_name)

sys.path.insert(0, str(_CRN_DIR))
try:
    STFT = importlib.import_module("core.stft").STFT

    dataset = importlib.import_module("dataset")
    model = importlib.import_module("model")
    train = importlib.import_module("train")
    CRN = model.CRN
finally:
    if str(_CRN_DIR) in sys.path:
        sys.path.remove(str(_CRN_DIR))
    for _alias in _COLLIDING:
        _mod = sys.modules.pop(_alias, None)
        if _mod is not None:
            sys.modules[f"crn_{_alias}"] = _mod
    sys.modules.update(_saved)
