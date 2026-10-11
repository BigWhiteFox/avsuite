"""pytest 全局配置：sys.path 注入 + 用例间仓库模块隔离 + 环境自检。

三项职责（Wave-B / t14 卫生前置，依据 reports/refactor_wave5/test_gap.md §3.5/§3.6）：

1) **sys.path**：把仓库根加进来，使 `import core.*` 在 pytest 下可用。
2) **仓库模块隔离**（§3.5）：tests/*_shim.py 与部分测试用 importlib 动态装载
   exp/0X 下的同名模块（model / train / dataset / pipeline / …），靠 sys.modules
   "先到先得"。若某测试在用例内 import 了裸名 model，后续 test_crn / test_dccrn /
   test_nn_aec 可能取到别的模块——隐性依赖 pytest 收集顺序。下面的 autouse
   fixture 在每个用例前后快照/恢复"仓库内模块"（__file__ 位于仓库根之下）与
   sys.path，使用例之间不再互相污染；对 torch / numpy / pytest 等第三方模块
   不做任何干预（避免破坏 C 扩展状态、避免重复 import 开销）。
   注意：shim 在 **import 期** 的 save/restore 仍是第一道防线，本 fixture 补的是
   **用例运行期** 的泄漏（例如新增测试文件在函数内 import 裸名 model）。
3) **环境自检**（§3.6）：未设 PYTHONNOUSERSITE 时用户站点包会静默顶替 conda 包
   （实证：scipy 1.17.1 → Roaming 1.10.1、pandas 3.0.6 → 1.5.3）。session 级
   fixture 发 EnvSanityWarning 显式告警（不 fail；CI 无用户站点故静默）。
"""
import importlib.util
import site
import sys
import warnings
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# 自检覆盖的关键包：本地实测会被 Roaming 用户站点顶替的那几个
_ENV_PROBE_PKGS = ("numpy", "scipy", "pandas", "torch")

# 路径 -> 是否仓库内模块 的缓存（避免每个用例对上千个模块反复 stat）
_REPO_MODULE_CACHE: dict = {}


class EnvSanityWarning(UserWarning):
    """环境自检告警。

    单独立类（而非裸 UserWarning）是为了让 wave-C 的 `-W error` 白名单能**定点**
    针对本类，而不误伤 torch/pesq 等第三方 UserWarning。
    """


def _is_repo_module(mod: object) -> bool:
    """模块是否来自本仓库（__file__ 位于仓库根之下）。"""
    f = getattr(mod, "__file__", None)
    if not f:
        return False
    cached = _REPO_MODULE_CACHE.get(f)
    if cached is None:
        try:
            cached = Path(f).resolve().is_relative_to(_REPO_ROOT)
        except (OSError, ValueError):
            cached = False
        _REPO_MODULE_CACHE[f] = cached
    return cached


def env_warnings() -> list:
    """返回环境异常描述清单（空列表 = 正常）。纯函数，可被单测直接调用。

    判据（用 find_spec 只解析路径、不导入包，避免给每个会话增加 import 成本）：
    - `site.ENABLE_USER_SITE` 为真 ⇒ PYTHONNOUSERSITE 未生效；
    - 关键包的解析路径落在用户站点目录内 ⇒ 该包已被用户站点顶替。
    """
    out = []
    if site.ENABLE_USER_SITE:
        out.append(
            "PYTHONNOUSERSITE 未生效（site.ENABLE_USER_SITE=True）：用户站点包会参与解析并"
            "可能静默顶替 conda 包（实证 scipy 1.17.1→1.10.1、pandas 3.0.6→1.5.3）。"
            "请在启动 Python 前设置 PYTHONNOUSERSITE=1。")
    try:
        usersite = Path(site.getusersitepackages()).resolve()
    except (OSError, ValueError):
        usersite = None
    if usersite is not None:
        for name in _ENV_PROBE_PKGS:
            origin = None
            if name in sys.modules:
                origin = getattr(sys.modules[name], "__file__", None)
            else:
                try:
                    spec = importlib.util.find_spec(name)
                except (ImportError, ValueError, AttributeError):
                    spec = None
                origin = getattr(spec, "origin", None) if spec is not None else None
            if not origin:
                continue
            try:
                if usersite in Path(origin).resolve().parents:
                    out.append(f"{name} 来自用户站点（{origin}）→ 已顶替 conda 版本。")
            except (OSError, ValueError):
                continue
    return out


@pytest.fixture(autouse=True)
def _isolate_repo_modules():
    """每个用例前后快照/恢复"仓库内模块"与 sys.path（见模块 docstring 第 2 条）。"""
    saved_modules = {k: v for k, v in sys.modules.items() if _is_repo_module(v)}
    saved_path = list(sys.path)
    try:
        yield
    finally:
        # 用例期间新装入的仓库内模块 → 移除；被替换的 → 还原为快照对象
        for key in [k for k, v in sys.modules.items()
                    if k not in saved_modules and _is_repo_module(v)]:
            del sys.modules[key]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


@pytest.fixture(scope="session", autouse=True)
def _env_sanity_check():
    """会话开始时做一次环境自检；异常以 EnvSanityWarning 显式告警（不 fail）。"""
    for msg in env_warnings():
        warnings.warn(f"环境自检：{msg}", EnvSanityWarning, stacklevel=1)
    yield
