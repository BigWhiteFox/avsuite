"""t13 前置单测：跨平台 ckpt 加载兼容层（`exp/07_nn_aec/ckpt_compat.py`）。

覆盖三件事：
(a) **机制**：手工构造一个"引用 `pathlib.PosixPath`"的 pickle（protocol 0，把
    `pathlib\\nPurePosixPath\\n` 逐字节改成 `pathlib\\nPosixPath\\n`）——在 Windows 上直接
    `pickle.loads` 抛 `NotImplementedError: cannot instantiate 'PosixPath'`，
    而 `posixpath_shim()` 内可以正常 loads；**退出后 `pathlib.PosixPath` 必须被恢复**
    （否则会污染同进程其它代码）；
(b) **torch 层**：`load_ckpt` 能加载一个含 pathlib 对象的 torch.save 文件，且 `ckpt_summary`
    能抽出 epoch/best/参数量/口径字段；
(c) **真 ckpt 体检（缺文件则 skip）**：服务器拉回的 NN v5 best.ckpt 必须能加载，
    参数量 638,264（0.638M 架构不变），且 `cfg.w_sisnr == '1.0'`（净池重训口径）。

用法：`pytest tests/test_ckpt_compat.py -q`
"""
import pathlib
import pickle
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp" / "07_nn_aec"))
sys.path.insert(0, str(ROOT))

import ckpt_compat as CC  # noqa: E402

P1_NN_V5 = (Path("D:/Project_for_Python/DSH/p1_artifacts/outputs_p1/nn_aec_v5/"
                 "nnaec_g0.5_j10_dm/best.ckpt"))


# ---------------------------------------------------------------- (a) 机制
def _posixpath_pickle() -> bytes:
    """构造引用 `pathlib.PosixPath` 的 pickle 字节（跨平台可构造：改写 protocol 0 的 GLOBAL 行）。"""
    raw = pickle.dumps(pathlib.PurePosixPath("/root/data/x"), protocol=0)
    assert b"pathlib\nPurePosixPath\n" in raw
    return raw.replace(b"pathlib\nPurePosixPath\n", b"pathlib\nPosixPath\n")


def test_posixpath_pickle_is_broken_without_shim():
    data = _posixpath_pickle()
    if sys.platform != "win32":
        pytest.skip("仅在 Windows 上 PosixPath 不可实例化")
    with pytest.raises(NotImplementedError):
        pickle.loads(data)


def test_shim_makes_pickle_loadable_and_is_restored():
    data = _posixpath_pickle()
    original = pathlib.PosixPath
    with CC.posixpath_shim():
        obj = pickle.loads(data)
        assert str(obj) == "/root/data/x"
        assert pathlib.PosixPath is pathlib.PurePosixPath
    assert pathlib.PosixPath is original          # 必须恢复


# ---------------------------------------------------------------- (b) torch 层
def test_load_ckpt_and_summary(tmp_path):
    p = tmp_path / "fake.ckpt"
    torch.save({"model": {"w": torch.zeros(3)}, "epoch": 7, "best": 1.25, "step": 42,
                "cfg": {"loss": "psycho", "w_sisnr": "1.0", "val": pathlib.PurePosixPath("/root/v")}}, p)
    st = CC.load_ckpt(p)
    s = CC.ckpt_summary(st, p)
    assert s["epoch"] == 7 and s["best"] == 1.25 and s["step"] == 42
    assert s["numel_state_dict"] == 3
    assert s["cfg"]["loss"] == "psycho" and s["cfg"]["w_sisnr"] == "1.0"
    assert "cfg" in s["top_keys"]


def test_load_ckpt_missing_file():
    with pytest.raises(FileNotFoundError):
        CC.load_ckpt("Z:/nope/none.ckpt")


# ---------------------------------------------------------------- (c) 真 ckpt 体检
@pytest.mark.skipif(not P1_NN_V5.is_file(), reason="本机无服务器拉回的 P1 ckpt")
def test_real_server_ckpt_loads_and_matches_arch():
    st = CC.load_ckpt(P1_NN_V5)
    s = CC.ckpt_summary(st, P1_NN_V5)
    assert s["numel_state_dict"] == 638264             # 状态字典元素数（含 230 个 BN buffer）
    assert s["cfg"].get("w_sisnr") == "1.0"           # 净池重训版口径
    assert {k for k in ("model", "cfg", "epoch")} <= set(s["top_keys"])
    # 架构参数量（model.parameters()）比状态字典少 230（BN running stats）
    sys.path.insert(0, str(ROOT / "exp" / "07_nn_aec"))
    from model import NNAEC
    n_param = sum(p.numel() for p in NNAEC().parameters())
    assert n_param == 638034 and s["numel_state_dict"] - n_param == 230
