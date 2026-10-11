"""t14 卫生前置的自证用例：conftest 的"仓库模块隔离"与"环境自检"确实生效。

背景见 tests/conftest.py 模块 docstring 与 reports/refactor_wave5/test_gap.md
§3.5（shim 的 sys.modules 顺序依赖）/§3.6（用户站点包静默顶替 conda 包）。
本文件不依赖 D:/data、不联网，全部为进程内断言。
"""
import site
import sys
import types
from pathlib import Path

import conftest  # tests/ 已在 sys.path（pytest rootdir 导入模式）
import pytest

_REPO = Path(__file__).resolve().parents[1]
_FAKE_MOD_NAME = "t14_hygiene_probe"
_FAKE_PATH = "/nonexistent/t14_hygiene_probe_path"

# 记录"污染用例是否已跑过"，让第二个用例在收集顺序被改动时可跳过而非误报
_STATE: dict = {}


# ---------------------------------------------------------------- 仓库模块隔离
def test_a_pollutes_repo_modules_and_sys_path():
    """在用例内装入一个"仓库内模块"裸名并污染 sys.path：本用例内应当可见。"""
    mod = types.ModuleType(_FAKE_MOD_NAME)
    mod.__file__ = str(_REPO / "tests" / "_t14_hygiene_probe.py")   # 伪造为仓库内路径
    sys.modules[_FAKE_MOD_NAME] = mod
    sys.path.insert(0, _FAKE_PATH)
    _STATE["polluted"] = True
    assert _FAKE_MOD_NAME in sys.modules
    assert sys.path[0] == _FAKE_PATH


def test_b_pollution_is_reverted_between_tests():
    """上一个用例的污染必须被 conftest 的 autouse fixture 清理干净。"""
    if not _STATE.get("polluted"):
        pytest.skip("污染用例未先运行（收集顺序被改动）——本用例跳过而非误报")
    assert _FAKE_MOD_NAME not in sys.modules, \
        "conftest 未清理用例间 sys.modules 污染（test_gap §3.5 的隐性顺序依赖）"
    assert _FAKE_PATH not in sys.path, "conftest 未还原 sys.path"


def test_repo_module_classification():
    """_is_repo_module：仓库内模块判 True，第三方判 False。"""
    import numpy  # 第三方
    assert conftest._is_repo_module(numpy) is False
    assert conftest._is_repo_module(object()) is False          # 无 __file__
    fake = types.ModuleType("t14_probe_classify")
    fake.__file__ = str(_REPO / "core" / "stft.py")
    assert conftest._is_repo_module(fake) is True
    fake.__file__ = str(_REPO.parent / "outside.py")
    assert conftest._is_repo_module(fake) is False


# ---------------------------------------------------------------- 环境自检
def test_env_warnings_silent_when_user_site_disabled(monkeypatch):
    """用户站点关闭（PYTHONNOUSERSITE=1 的正常态）时不产生任何告警。"""
    monkeypatch.setattr(site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(site, "getusersitepackages",
                        lambda: "/nonexistent-t14/user/site-packages")
    assert conftest.env_warnings() == []


def test_env_warnings_flags_enabled_user_site(monkeypatch):
    """PYTHONNOUSERSITE 未生效时必须显式告警（t14 实证：scipy/pandas 会被顶替）。"""
    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(site, "getusersitepackages",
                        lambda: "/nonexistent-t14/user/site-packages")
    msgs = conftest.env_warnings()
    assert any("PYTHONNOUSERSITE" in m for m in msgs), msgs


def test_env_warnings_flags_shadowed_package(monkeypatch, tmp_path):
    """模拟 scipy 被用户站点顶替（t14 实证场景：conda 1.17.1 → Roaming 1.10.1）。"""
    usersite = tmp_path / "Roaming" / "Python311" / "site-packages"
    fake_init = usersite / "scipy" / "__init__.py"
    fake_init.parent.mkdir(parents=True)
    fake_init.write_text("", encoding="utf-8")
    monkeypatch.setattr(site, "ENABLE_USER_SITE", False)
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(usersite))
    shadowed = types.ModuleType("scipy")
    shadowed.__file__ = str(fake_init)
    monkeypatch.setitem(sys.modules, "scipy", shadowed)
    msgs = conftest.env_warnings()
    assert any("scipy" in m and "用户站点" in m for m in msgs), msgs
    # 未被顶替的包不应误报
    assert not any("torch" in m for m in msgs), msgs
