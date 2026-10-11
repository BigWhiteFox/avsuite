"""T3：`exp/06_trad_aec/{gcc,nlms,res}.py` 边界/异常语义补测（test_gap.md §2 T3）。

这三者有直接单测（test_trad_aec.py），但只覆盖"正常路径"；V5 正在改这些边界
（t10 已落地 gcc 非有限防护 + 3 元组返回、res 最小长度校验），改动需要兜底。

**与 test_gap §2 T3 的偏差（以 t10 落定 + t29/F4 更正后的状态为准）**：
- `gcc_phat_stats` 现在返回 **3 元组** `(delay_ms, peak_ratio, evidence_valid)`；
- `residual_suppress` 的**真实**长度下限是 `n_fft - hop`（默认 256）而非 `n_fft`（512）：
  scipy 的约束是 `noverlap < 有效 nperseg`，len(x) < nperseg 时 scipy 会下调 nperseg。
  V5 终审 F4 实测（HEAD 版）：n≤256 → scipy `noverlap must be less than nperseg`；
  257≤n≤511 → istft 形状 broadcast 报错（另一处缺陷，已随 F4 零补修法一并修好）；
  n≥512 → 成功（旧守卫 `len <= n_fft` 会误拒 n=512）。故边界用例现在钉 n=257 可用；
- NaN 注入的断言口径：mic 脏 → 证据仍有效（`evidence_valid=True`），far 脏 → 失效。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def _load(name: str, rel: str):
    sys.path.insert(0, str(ROOT / "exp" / "06_trad_aec"))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GCC = _load("t3_gcc", "exp/06_trad_aec/gcc.py")
NL = _load("t3_nlms", "exp/06_trad_aec/nlms.py")
RES = _load("t3_res", "exp/06_trad_aec/res.py")


# ---------------------------------------------------------------- gcc
def test_gcc_phat_stats_peak_ratio_separates_echo_from_independent():
    """peak_ratio 是 echo_prior（EVIDENCE_THR=14）的唯一判据，此前无直接断言。"""
    rng = np.random.default_rng(7)
    far = rng.standard_normal(SR) * 0.3
    mic_echo = 0.3 * far
    mic_ind = np.random.default_rng(8).standard_normal(SR) * 0.3

    d_echo, r_echo, ok_echo = GCC.gcc_phat_stats(mic_echo, far, SR)
    assert ok_echo is True
    assert r_echo > 14.0
    assert abs(d_echo) < 1.0

    d_ind, r_ind, ok_ind = GCC.gcc_phat_stats(mic_ind, far, SR)
    assert ok_ind is True
    assert r_ind < 14.0


def test_gcc_phat_delay_clamps_negative_to_zero():
    rng = np.random.default_rng(0)
    far = rng.standard_normal(SR) * 0.3
    mic = np.concatenate([far[300:], np.zeros(300)])       # far 滞后 mic（非物理）
    assert GCC.gcc_phat_delay(mic, far, SR) == 0.0


def test_gcc_phat_delay_respects_max_delay_ms():
    rng = np.random.default_rng(0)
    far = rng.standard_normal(SR) * 0.3
    mic = np.concatenate([np.zeros(int(0.100 * SR)), far])[:SR]     # 真值 100ms
    d = GCC.gcc_phat_delay(mic, far, SR, max_delay_ms=50.0)
    assert 0.0 <= d <= 50.0


def test_gcc_phat_stats_nonfinite_input_degrades_explicitly():
    """V5/t10 修复固化：mic 含 NaN 时仍给有限证据（参考端有效）；
    far 含 NaN 时 `evidence_valid=False`（调用方据此降级，而非把 nan 当"无回声"）。"""
    rng = np.random.default_rng(5)
    far = rng.standard_normal(SR) * 0.3
    mic = 0.3 * far + 0.01 * rng.standard_normal(SR)
    mic_bad = mic.copy()
    mic_bad[1000:1004] = np.nan
    d, r, ok = GCC.gcc_phat_stats(mic_bad, far, SR)
    assert np.isfinite(d) and np.isfinite(r) and ok is True

    far_bad = far.copy()
    far_bad[1000:1004] = np.nan
    d2, r2, ok2 = GCC.gcc_phat_stats(mic, far_bad, SR)
    assert ok2 is False and np.isfinite(d2)


# ---------------------------------------------------------------- nlms
def test_nlms_mu_validation():
    with pytest.raises(ValueError):
        NL.NLMS(length=16, mu=0.0)
    with pytest.raises(ValueError):
        NL.NLMS(length=16, mu=2.5)
    NL.NLMS(length=16, mu=2.0)          # 闭区间边界合法


def test_nlms_reset_clears_state():
    rng = np.random.default_rng(0)
    a = NL.NLMS(length=16, mu=0.5)
    a.process(rng.standard_normal(500), rng.standard_normal(500))
    assert not np.allclose(a.w, 0.0)
    a.reset()
    assert np.allclose(a.w, 0.0)
    assert np.allclose(a.xbuf, 0.0)
    assert a.x2_ema is None


def test_nlms_silence_normalization_floor():
    """远端静默（NST 式）：归一化下限保证不往近端注垃圾——输出≈mic。"""
    rng = np.random.default_rng(1)
    far = np.zeros(2000)
    mic = rng.standard_normal(2000) * 0.01
    e = NL.NLMS(length=64, mu=0.5).process(far, mic)
    assert np.isfinite(e).all()
    assert np.abs(e - mic).max() < 1e-6


# ---------------------------------------------------------------- res
def test_residual_suppress_min_length_contract():
    """F4 更正后边界：真实下限 n > n_fft - hop = 256；n=257 起可用，n=512 也恢复可用。

    （旧断言钉的是 t10 的 `len <= n_fft` 过严口径：会误拒 HEAD 本可成功的 n=512。）
    """
    for n in (100, 256):
        with pytest.raises(ValueError, match="需要输入长度"):
            RES.residual_suppress(np.zeros(n), np.zeros(n), gate=np.ones(n))
    for n in (257, 512, 513):                      # 新可用面（含恢复的 512）
        out = RES.residual_suppress(np.zeros(n), np.zeros(n), gate=np.ones(n))
        assert len(out) == n, f"n={n}: 输出长度应恒等于输入（不静默截短）"


def test_residual_suppress_gmin_floor():
    rng = np.random.default_rng(2)
    n = 4 * 512
    echo = rng.standard_normal(n) * 0.1
    out = RES.residual_suppress(echo, echo, gate=np.ones(n), alpha=1.5, g_min=0.1)
    erle = 10 * np.log10(np.sum(echo[512:-512] ** 2) / (np.sum(out[512:-512] ** 2) + 1e-12))
    assert 17.0 < erle < 23.0          # g_min=0.1 -> ≈20dB 地板


def test_residual_suppress_gate_is_frame_majority():
    """gate 按 hop(=256) 帧内均值 ≥0.5 判决：第 1 帧抑制、第 2 帧直通。"""
    rng = np.random.default_rng(3)
    n = 768
    echo = rng.standard_normal(n) * 0.1
    gate = np.concatenate([np.ones(256), np.zeros(256), np.zeros(256)])
    out = RES.residual_suppress(echo, echo, gate=gate)
    assert not np.allclose(out[0:256], echo[0:256])          # 第 1 帧被抑制
    assert np.allclose(out[256:768], echo[256:768])          # 其余帧 G=1 直通


def test_residual_suppress_gate_tail_passthrough():
    """gate 只覆盖前 1 帧时，尾帧走"保守直通"分支（G=1）。"""
    rng = np.random.default_rng(4)
    n = 1024
    echo = rng.standard_normal(n) * 0.1
    gate = np.ones(n)
    gate[256:] = 0.0
    out = RES.residual_suppress(echo, echo, gate=gate)
    assert np.allclose(out[512:], echo[512:])
