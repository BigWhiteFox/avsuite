"""T1：`exp/06_trad_aec/kalman.py::FDKalman` 直接覆盖（test_gap.md §2 T1）。

`FDKalman` 是全仓唯一既零直接覆盖、又不在任何间接链上的自适应滤波器
（`MDF`/`NLMS` 都有直接单测；`pipeline.py` 只支持 mdf/nlms，不会顺带覆盖它）。
本文件按 test_gap §2 T1 的 7 条用例固化：FIR 辨识 ERLE、taps 契约、硬冻结语义、
adapt_scale=0 直通、等长校验、reset 契约、逐块 vs 整段逐位等价。

约束：纯 numpy + 合成信号，无 D:/data、无网络，实测 < 3s。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SR = 16000

def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


KF = _load("t1_kalman", "exp/06_trad_aec/kalman.py")
FDKalman = KF.FDKalman


def _erle(mic, err, skip_frac=0.25):
    """与 tests/test_trad_aec.py::_erle 同口径（跳过收敛瞬态）。"""
    k = int(len(mic) * skip_frac)
    return 10 * np.log10(np.sum(mic[k:] ** 2) / (np.sum(err[k:] ** 2) + 1e-12) + 1e-12)


def _scene(seed=1, n=2 * SR):
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(2048) * np.exp(-np.arange(2048) / 300)
    far = rng.standard_normal(n) * 0.3
    mic = np.convolve(far, h)[:n]
    return far, mic


def _erle_curve(mic, e, block=256, win_blocks=40):
    """按窗口 ERLE 曲线（沿用 t23 诊断脚本口径，窗口默认 40 块=0.64s）。"""
    xs, ys = [], []
    for i in range(0, len(mic) - win_blocks * block + 1, win_blocks * block):
        m, ee = mic[i:i + win_blocks * block], e[i:i + win_blocks * block]
        xs.append((i + len(m)) / 2 / SR)
        ys.append(10 * np.log10(np.sum(m ** 2) / (np.sum(ee ** 2) + 1e-12) + 1e-12))
    return np.array(xs), np.array(ys)


def _first_cross(xs, ys, thr):
    """首次 ≥thr 的时刻（s）；未达标返回 None。"""
    for x, y in zip(xs, ys):
        if y >= thr:
            return float(x)
    return None


def test_fd_kalman_identifies_fir():
    """默认超参（q=1e-4, r=1e-2）下能抑制回声；短时程读数**不可用作收敛判据**。

    t23 诊断结论（报告 reports/refactor_wave5/kalman_convergence.md）：
    实测 2s 窗口 ERLE 仅 3.3–4.5dB，但那是**里卡蒂稳态尚未建立**的早段读数
    （P∞≈sqrt(q·r)≈1e-3 ⇒ 速率慢，与 ADR 0006「上限不缺、缺速率」一致），
    20s 窗口即 >15dB、60s 达机器精度。故本用例只 pin「有限 + 确有抑制」下限。
    """
    far, mic = _scene()
    kf = FDKalman(block=256, partitions=8, q=1e-4, r=1e-2)
    e = kf.process(far, mic)
    assert np.isfinite(e).all()
    assert _erle(mic, e) > 2.0


def test_fd_kalman_improves_with_larger_q():
    """q=1e-2（仍在 kalman_eval 的网格内）应显著优于默认 q=1e-4。"""
    far, mic = _scene()
    e_slow = FDKalman(block=256, partitions=8, q=1e-4, r=1e-2).process(far, mic)
    e_fast = FDKalman(block=256, partitions=8, q=1e-2, r=1e-2).process(far, mic)
    assert _erle(mic, e_fast) > 10.0
    assert _erle(mic, e_fast) > _erle(mic, e_slow) + 5.0


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_fd_kalman_converges_within_documented_horizon(seed):
    """t23 诊断后的正确期望：**上限不缺、缺的是速率**（与 ADR 0006 一致）。

    t14 曾以"2s 内 >15dB"为期望并 xfail——该窗口对 q=1e-4 的里卡蒂稳态
    （P∞≈sqrt(q·r)≈1e-3）本就过短。本用例按 ADR 口径断言"给足时长能收敛"：
    20s 场景 75% 之后窗口 ERLE ≥ 15dB，且 8s 窗口显著高于 1s 窗口（仍在爬升）。
    """
    far, mic = _scene(n=20 * SR)
    kf = FDKalman(block=256, partitions=8, q=1e-4, r=1e-2)
    e = kf.process(far, mic)
    xs, ys = _erle_curve(mic, e, win_blocks=5)
    tail = _erle(mic, e, skip_frac=0.75)
    assert tail > 15.0, f"20s 未收敛到 15dB（尾段 {tail:.2f}dB）——与 ADR 0006「上限不缺」矛盾"
    assert ys[np.argmin(np.abs(xs - 8.0))] > ys[np.argmin(np.abs(xs - 1.0))] + 5.0


def test_fd_kalman_rate_vs_mdf_is_documented_boundary():
    """ADR 0006 的边界：同场景 MDF **更快**（t≈1s 领先 >10dB），但 Kalman 终会收敛。

    把"fixed-q Kalman 输在速率"这一已文档化结论钉进测试：若将来有人调大默认 q 或修
    更新公式，本用例会红 → 提醒同步更新 ADR 0006 附记与结论行。
    """
    far, mic = _scene(n=20 * SR)
    e_kal = FDKalman(block=256, partitions=8, q=1e-4, r=1e-2).process(far, mic)
    e_mdf = _load("t1_mdf", "exp/06_trad_aec/mdf.py").MDF(
        block=256, partitions=8, mu=0.7).process(far, mic)
    xs_k, ys_k = _erle_curve(mic, e_kal, win_blocks=5)
    xs_m, ys_m = _erle_curve(mic, e_mdf, win_blocks=5)
    t15_k, t15_m = _first_cross(xs_k, ys_k, 15.0), _first_cross(xs_m, ys_m, 15.0)
    assert t15_m is not None and t15_k is not None, (t15_m, t15_k)
    assert t15_m < t15_k, f"MDF t15={t15_m}s 应快于 Kalman t15={t15_k}s（ADR 0006 边界）"
    at1_k = ys_k[np.argmin(np.abs(xs_k - 1.0))]
    at1_m = ys_m[np.argmin(np.abs(xs_m - 1.0))]
    assert at1_m - at1_k > 10.0, f"t≈1s 时 MDF 只领先 {at1_m - at1_k:.1f}dB（预期 >10dB）"


def test_fd_kalman_q_scaling_is_sqrt_law():
    """q 标度律（t23 定量结论）：增益 ∝ sqrt(q) ⇒ 收敛更快、P 稳态更高。

    实测（8s 场景）：q=1e-2 相对 1e-4（100×）t15 由 2.76s→0.92s、ERLE(t≈1s)
    由 2.8dB→15.9dB、Pc 均值由 1.44e-3→2.72e-2（≈19×，与 sqrt 律同数量级）。
    """
    far, mic = _scene(n=8 * SR)
    kf_slow = FDKalman(block=256, partitions=8, q=1e-4, r=1e-2)
    kf_fast = FDKalman(block=256, partitions=8, q=1e-2, r=1e-2)
    e_slow = kf_slow.process(far, mic)
    e_fast = kf_fast.process(far, mic)
    xs_s, ys_s = _erle_curve(mic, e_slow, win_blocks=5)
    xs_f, ys_f = _erle_curve(mic, e_fast, win_blocks=5)
    assert _first_cross(xs_f, ys_f, 15.0) < _first_cross(xs_s, ys_s, 15.0)
    assert ys_f[np.argmin(np.abs(xs_f - 1.0))] > ys_s[np.argmin(np.abs(xs_s - 1.0))] + 5.0
    ratio = float(np.abs(kf_fast.Pc).mean() / np.abs(kf_slow.Pc).mean())
    assert ratio > 3.0, f"q 100× 后 P 稳态比仅 {ratio:.1f}×（预期 ≈10×，取同数量级下限 3×）"


def test_taps_property_and_length():
    kf = FDKalman(block=256, partitions=8)
    assert kf.taps.shape == (2048,)
    assert np.allclose(kf.taps, 0.0)
    far, mic = _scene()
    kf.process(far[: 8 * 256], mic[: 8 * 256])
    assert not np.allclose(kf.taps, 0.0)


def test_block_hook_zero_hard_freezes():
    """block_hook 返回 0（= pipeline 的 IDLE 硬冻结）时 H/P 必须逐位不动。"""
    far, mic = _scene()
    kf = FDKalman(block=256, partitions=8, q=1e-4, r=1e-2)
    kf.process(far[: 16000], mic[: 16000])
    H0, P0 = kf.H.copy(), kf.Pc.copy()
    kf.process(far[16000: 20000], mic[16000: 20000], block_hook=lambda *a: 0.0)
    assert np.array_equal(kf.H, H0)
    assert np.array_equal(kf.Pc, P0)


def test_adapt_scale_zero_passes_through():
    """H 恒为 0 时回声估计为 0 -> 输出逐点等于 mic。"""
    far, mic = _scene()
    e_frozen = FDKalman(block=256, partitions=8).process(far[: 16000], mic[: 16000],
                                                        adapt_scale=0.0)
    e_active = FDKalman(block=256, partitions=8).process(far[: 16000], mic[: 16000],
                                                        adapt_scale=1.0)
    assert np.array_equal(e_frozen, mic[: 16000])
    assert not np.allclose(e_active, mic[: 16000])


def test_equal_length_required():
    kf = FDKalman(block=256, partitions=8)
    with pytest.raises(ValueError, match="equal length"):
        kf.process(np.zeros(100), np.zeros(101))


def test_reset_restores_initial_state():
    far, mic = _scene()
    kf = FDKalman(block=256, partitions=8)
    kf.process(far[: 16000], mic[: 16000])
    kf.reset()
    assert np.allclose(kf.H, 0.0)
    assert kf.span_ph is None
    # nb = N//2+1，N = 2*block = 512 -> 257 个 bin（注意：不是 129，129 属于 N=256 的误解）
    assert np.allclose(kf.Pc, np.repeat(np.eye(8)[None], 257, axis=0))


def test_blockwise_vs_segment_bitwise():
    """逐块 process(256) 拼接 == 整段 process（与 latency_bench 的 MDF 同款口径）。"""
    rng = np.random.default_rng(11)
    n = 40 * 256
    far = rng.standard_normal(n) * 0.05
    mic = rng.standard_normal(n) * 0.05
    kf_blk = FDKalman(block=256, partitions=8, mu=0.7) if False else FDKalman(
        block=256, partitions=8, q=1e-4, r=1e-2)
    e_blk = np.concatenate([
        kf_blk.process(far[i * 256:(i + 1) * 256], mic[i * 256:(i + 1) * 256])
        for i in range(40)])
    kf_seg = FDKalman(block=256, partitions=8, q=1e-4, r=1e-2)
    e_seg = kf_seg.process(far, mic)
    assert np.array_equal(e_blk, e_seg)
