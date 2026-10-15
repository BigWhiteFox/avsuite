"""T12：`exp/09_cascade_aec/` 探针脚本覆盖（test_gap.md §2 T12）。

分两层：
- **T12-a（直接测）**：`zerotrain_baseline.erle_db` 的口径（跳过收敛瞬态 + 零信号保护
  + float32/list 兼容）。
- **T12-b（结构契约）**：`probe_clip.py` / `probe_env.py` / `probe_compare.py` 是
  **模块级脚本（import 即执行、直接读 D:/data）**，无法在"不改本体"的前提下单测
  → 用严格标注的 xfail 固化这一重构需求（t14 inScope 不含 exp/09，故不重构本体）。
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SR = 16000

# exp/09 的脚本按"裸名"导入同目录/邻目录模块（pipeline/model/...），会在 sys.modules
# 留下会与 exp/03、exp/04、exp/07 冲突的条目；本文件在模块级加载后立即清理这些**新增**裸名
# （§3.5 卫生），避免污染后续收集的 test_crn/test_dccrn。
_BARE = {"model", "train", "dataset", "pipeline", "gcc", "mdf", "dtd", "nlms", "res",
         "eval", "kalman", "complex_nn", "dynamic_mixing"}


def _load(name: str, rel: str):
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        for key in [k for k in sys.modules if k not in before and k in _BARE]:
            del sys.modules[key]
    return mod


ZB = _load("t12_zerotrain", "exp/09_cascade_aec/zerotrain_baseline.py")
erle_db = ZB.erle_db


# ------------------------------------------------------------------ T12-a
def test_erle_db_hand_computed():
    ref = np.ones(4 * SR)
    sig = ref * 0.1
    assert abs(erle_db(sig, ref) - 20.0) < 1e-6
    assert abs(erle_db(sig, ref, skip_s=0.0) - 20.0) < 1e-6


def test_erle_db_skip_semantics():
    """前 2s 无消回声、后 2s 消 20dB：skip_s=2 得 20dB，skip_s=0 明显更低。"""
    ref = np.ones(4 * SR)
    sig = ref.copy()
    sig[2 * SR:] = ref[2 * SR:] * 0.1
    assert abs(erle_db(sig, ref, skip_s=2.0) - 20.0) < 1e-6
    assert erle_db(sig, ref, skip_s=0.0) < erle_db(sig, ref, skip_s=2.0) - 3.0


def test_erle_db_handles_zero_signal():
    v = erle_db(np.zeros(3 * SR), np.ones(3 * SR))
    assert np.isfinite(v) and v > 100.0


def test_erle_db_accepts_float32():
    ref = np.ones(4 * SR)
    sig = ref * 0.1
    v64 = erle_db(sig, ref)
    v32 = erle_db(sig.astype(np.float32), ref.astype(np.float32))
    assert abs(v32 - v64) < 1e-3


def test_erle_db_rejects_python_list():
    """现状固化：`erle_db` 只接受 ndarray（list 会 `**` TypeError）。

    test_gap §2 T12-a #4 期望 list 也可用；实测不成立。管线一律传 ndarray，影响低，
    若要支持 list 应在入口 `np.asarray`（属待裁决的小健壮性改进）。
    """
    ref = np.ones(4 * SR)
    with pytest.raises(TypeError):
        erle_db(list(ref * 0.1), list(ref))


# ------------------------------------------------------------------ T12-b
_HAS_D_DATA = Path("D:/data/aec-synth/eval_v3").is_dir()   # Linux/CI 无数据 ⇒ 跳过

@pytest.mark.parametrize("script", [
    pytest.param("probe_clip.py", marks=pytest.mark.skipif(
        not _HAS_D_DATA, reason="探针模块级直读 D:/data（Windows 素材）；无数据环境跳过")),
    pytest.param("probe_env.py", marks=pytest.mark.skipif(
        not _HAS_D_DATA, reason="探针模块级直读 D:/data（Windows 素材）；无数据环境跳过")),
    pytest.param("probe_compare.py", marks=pytest.mark.xfail(
        strict=False, reason="probe_compare.py **模块级**读取 "
                             "exp/07_nn_aec/outputs/aecmos_trad_wp06.json（import 即执行、"
                             "产物缺失即 FileNotFoundError）→ 需最小可测化重构"
                             "（模块级代码 → main()）；t14 inScope 不含 exp/09")),
])
def test_probe_scripts_import_without_side_effects(script):
    """import 成功、不读仓库外产物、不打印、不建 results/（子进程隔离，避免污染本进程）。

    probe_clip / probe_env 当前已满足该契约（作为硬断言锁死）；probe_compare 尚不满足
    （见上方 xfail 原因）——它是 T12-b 要固化的重构需求。
    """
    code = (
        "import importlib.util,sys\n"
        f"spec=importlib.util.spec_from_file_location('probe', r'{ROOT / 'exp/09_cascade_aec' / script}')\n"
        "m=importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "print('IMPORT_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(ROOT), timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.strip().endswith("IMPORT_OK")
    assert "results" not in r.stdout
