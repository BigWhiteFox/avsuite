"""V4 整改：TradAEC 加固的单测（对齐余量 / 周期重估 / DTD 复位 / 发散防护）。

全部测试内合成信号，不依赖 D:/data 外部数据。
背景（reports/v4_rectification/ 与代码审查）：原实现三个缺陷——
① align_far 把直达径放在滤波器 lag 0（零余量），负向时延漂移/路径移动时残余路径
   非因果、任何 FIR 不可表示 → 回声泄漏崩塌（漂移集 FEST −1.89 的主通道）；
② DTD 收敛闩锁只置位永不复位，路径突变后误冻正反馈；
③ MDF 无系数绝对界/发散复位（官方集 1/800 条耗尽 float64）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp" / "06_trad_aec"))

from dtd import DTDStreaming  # noqa: E402
from mdf import MDF  # noqa: E402
from pipeline import AECResult, TradAEC  # noqa: E402

SR = 16000


def _voice_like(rng: np.random.Generator, n: int) -> np.ndarray:
    """宽带类语音信号（多正弦谐波 + 音节率包络 + 低噪），供 GCC/MDF 收敛。"""
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f0 in (140.0, 210.0, 330.0, 510.0, 830.0, 1200.0):
        x += (0.3 + 0.2 * np.sin(2 * np.pi * 0.7 * t + f0)) * np.sin(2 * np.pi * f0 * t)
    env = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 4 * t)) ** 0.3
    return (x * env + 0.01 * rng.standard_normal(n)).astype(np.float64)


def _shift_frac(x: np.ndarray, tau_ms: np.ndarray) -> np.ndarray:
    """逐样本分数时延读取（测试内版，仿 aec_synth.apply_delay_drift 的线性插值）。"""
    n = len(x)
    t = np.arange(n)
    pos = t - tau_ms * SR / 1000.0
    i0 = np.clip(np.floor(pos).astype(int), 0, n - 2)
    frac = np.clip(pos - i0, 0.0, 1.0)
    i1 = np.clip(i0 + 1, 0, n - 1)
    out = (1 - frac) * x[i0] + frac * x[i1]
    out[pos < 0] = 0.0
    return out


def _erle_db(mic: np.ndarray, out: np.ndarray, seg: slice) -> float:
    return 10 * np.log10(np.mean(mic[seg] ** 2) / (np.mean(out[seg] ** 2) + 1e-20))


def test_defaults_backward_compatible_static_path():
    """默认参数（margin=0、无重估）：静态回声路径行为不变——输出有限、无复位/重对齐、
    收敛后 ERLE 达标（对齐既有验收口径 >15dB）。"""
    rng = np.random.default_rng(3)
    n = 6 * SR
    far = _voice_like(rng, n)
    echo = _shift_frac(far, np.full(n, 40.0)) * 0.4
    mic = echo + 0.01 * rng.standard_normal(n)
    r = TradAEC(method="mdf", partitions=8, mu=0.7).process(mic, far)
    assert isinstance(r, AECResult)
    assert np.isfinite(r.out).all() and np.isfinite(r.error).all()
    assert r.n_realigns == 0 and r.n_divergence_resets == 0
    assert r.delay_track == [(0.0, pytest.approx(40.0, abs=2.0), False)]
    assert _erle_db(mic, r.out, slice(2 * SR, None)) > 15.0


def test_align_margin_semantics_and_no_regression():
    """margin 语义：align = max(est − margin, 0)；慢速游走下 margin=25 不劣于 margin=0
    （行为收益的量化交给 S10 真实数据重跑——单元测试发现：噪声激励下慢游走的 MDF
    跟踪代价(块内路径移动)与周期激励的周期性预测补偿相互纠缠，纯几何断言会误导，
    见 exp/06_trad_aec/README.md 附记）。"""
    rng = np.random.default_rng(7)
    n = 8 * SR
    far = _voice_like(rng, n)
    t = np.arange(n) / SR
    tau = 40.0 - 10.0 * np.abs(np.sin(2 * np.pi * t / 4.0))   # 0↔−10ms 确定性游走
    echo = _shift_frac(far, tau) * 0.35
    near = 0.25 * _voice_like(rng, n)                          # 双讲近端
    mic = echo + near + 0.005 * rng.standard_normal(n)
    r0 = TradAEC(method="mdf", partitions=8, mu=0.7).process(mic, far)
    r25 = TradAEC(method="mdf", partitions=8, mu=0.7,
                  align_margin_ms=25.0).process(mic, far)
    # 几何语义：margin 把对齐点从估计值内收 25ms（负向游走时残余路径保持因果）
    assert r0.align_ms == pytest.approx(r0.delay_ms, abs=0.01)
    assert r25.align_ms == pytest.approx(max(r25.delay_ms - 25.0, 0.0), abs=0.01)
    assert r0.n_realigns == 0 and r25.n_realigns == 0
    assert r0.n_divergence_resets == 0 and r25.n_divergence_resets == 0
    assert np.isfinite(r0.out).all() and np.isfinite(r25.out).all()
    # 行为不倒退（噪声式紧断言留待 S10 真实漂移集重跑）
    assert _erle_db(mic, r25.out, slice(3 * SR, None)) >= \
        _erle_db(mic, r0.out, slice(3 * SR, None)) - 1.0


def test_align_margin_clamps_at_zero():
    """估计时延小于 margin 时对齐量钳到 0（不产生负对齐）。"""
    rng = np.random.default_rng(9)
    n = 3 * SR
    far = _voice_like(rng, n)
    echo = _shift_frac(far, np.full(n, 8.0)) * 0.4
    mic = echo + 0.01 * rng.standard_normal(n)
    r = TradAEC(method="mdf", partitions=8, mu=0.7,
                align_margin_ms=25.0).process(mic, far)
    assert r.delay_ms <= 10.0
    assert r.align_ms == 0.0


def test_mdf_divergence_guard_keeps_output_finite():
    """发散防护：病态输入（巨幅 far + NaN 突发）下输出恒有限、复位计数 ≥1。"""
    rng = np.random.default_rng(5)
    n = 3 * SR
    far = _voice_like(rng, n) * 1e6
    mic = far.copy()
    for k in range(7, 10):
        mic[k * 256:k * 256 + 8] = np.nan
    aec = TradAEC(method="mdf", partitions=8, mu=0.7)
    r = aec.process(mic, far)
    assert np.isfinite(r.out).all() and np.isfinite(r.error).all()
    assert r.n_divergence_resets >= 1
    # MDF 级直接验证
    m = MDF(block=256, partitions=8, mu=0.7)
    far_b = rng.standard_normal(256) * 1e6
    mic_b = far_b * 0.4
    mic_b[100:104] = np.nan
    for _ in range(30):
        e = m.process(far_b, mic_b)
    assert np.isfinite(e).all() and m.n_divergence_resets >= 1


def test_reestimate_recovers_large_delay_jump():
    """周期重估再对齐：+130ms 跳变（残差 155ms 越出 128ms 滤波器窗，固定对齐
    不可救——非周期宽带激励下 ERLE 崩到 ~3dB），reestimate_sec=2 在下个窗口捕获
    新时延并再对齐——末端 ERLE 差 ≥6dB（实测 ~46dB）。"""
    rng = np.random.default_rng(11)
    n = 12 * SR
    t = np.arange(n) / SR
    far = (rng.standard_normal(n) * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 1.0 * t))))
    d_ms = np.full(n, 40.0)
    d_ms[4 * SR:] = 170.0
    echo = _shift_frac(far.astype(np.float64), d_ms) * 0.4
    mic = echo + 0.01 * rng.standard_normal(n)
    r_fix = TradAEC(method="mdf", partitions=8, mu=0.7,
                    align_margin_ms=25.0).process(mic, far)
    r_dyn = TradAEC(method="mdf", partitions=8, mu=0.7, align_margin_ms=25.0,
                    reestimate_sec=2.0).process(mic, far)
    assert r_fix.n_realigns == 0 and r_dyn.n_realigns >= 1
    assert any(tr[2] for tr in r_dyn.delay_track)
    assert np.isfinite(r_dyn.out).all()
    gap = _erle_db(mic, r_dyn.out, slice(9 * SR, None)) - _erle_db(mic, r_fix.out, slice(9 * SR, None))
    assert gap >= 6.0, f"重估修复增益不足: {gap:.2f} dB"


def test_dtd_reset_clears_latch_and_rewarmups():
    """DTDStreaming.reset()：清闩锁/EMA/暖机计数——重对齐后暖机期重新冻结，
    修复"闩锁只置位永不复位 + 误冻正反馈"。"""
    dtds = DTDStreaming(frame=256, sr=SR, echo_prior=True)
    rng = np.random.default_rng(0)
    far_b = rng.standard_normal(256) * 0.1
    mic_b = far_b * 0.4                      # 纯回声（健康收敛态）
    y_b = far_b * 0.4
    for _ in range(40):
        dtds.update(mic_b, far_b, y_b)
    assert dtds.converged
    dtds.reset()
    assert not dtds.converged and dtds.nb == 0 and dtds.erle_ema is None
    assert dtds.update(mic_b, far_b, y_b) == 2   # 暖机期（前 warm_blocks 块）重新冻结


def test_aecresult_new_fields_default():
    """AECResult 新字段带默认值——旧构造（位置/关键字）不受影响。"""
    r = AECResult(error=np.zeros(4), out=np.zeros(4), y_echo=np.zeros(4),
                  delay_ms=1.0, dt_frac=0.0, method="mdf")
    assert r.n_realigns == 0 and r.n_divergence_resets == 0
    assert r.delay_track == [] and r.align_ms == 0.0
