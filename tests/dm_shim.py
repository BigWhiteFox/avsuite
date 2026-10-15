"""pytest 收集辅助：共享 DM 模块的**单实例**加载。

为什么需要（实测踩过）
----------------------
`importlib` 按文件路径加载同名模块会生成**两个不同的类对象**，而 `sys.modules[name]`
只保留后一个。DM 数据集要在单测里做 pickle 往返（DataLoader worker 可 pickle 的前提），
而 pickle 要求 `sys.modules["dm_dataset"].AecDynamicMix is <被 pickle 的类>`——
多文件各载一次时直接 `PicklingError`（现象：单文件跑全绿、多文件连着跑红，
且是否触发取决于 pytest 收集顺序）。

因此三个 DM 测试文件（test_nn_aec_dm / test_cascade_dm / test_dccrn_dm）统一走本 shim：
同名模块已在 `sys.modules` 中就直接复用，不做第二次加载。

注意：**只对"共享同名模块"用本 shim**；各 exp 目录自己的裸名 `model`/`dynamic_mixing`
等仍由各测试文件按需加载（`tests/conftest.py` 的 per-test 隔离负责清理）。
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP04 = ROOT / "exp" / "04_dccrn"
EXP07 = ROOT / "exp" / "07_nn_aec"
EXP09 = ROOT / "exp" / "09_cascade_aec"


def load(name: str, path: Path):
    """加载仓库内模块；`name` 已在 sys.modules 中则**复用**（不重复加载）。"""
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        if str(path.parent) in sys.path:
            sys.path.remove(str(path.parent))
    return mod


def dm_module():
    """07 的 `dm_dataset`（三个 DM 测试文件共用的单实例）。"""
    return load("dm_dataset", EXP07 / "dm_dataset.py")


def cascade_module():
    """09 的 `cascade_dm`（依赖 dm_module() 已加载）。"""
    return load("cascade_dm", EXP09 / "cascade_dm.py")


def dccrn_dm_module():
    """04 的 `dccrn_dm`（依赖 dm_module() 已加载）。"""
    return load("dccrn_dm", EXP04 / "dccrn_dm.py")
