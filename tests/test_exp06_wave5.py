"""Wave V5 · exp/06 语义加固回归（T10）。

钉住 deep_audit.md 的 G1（非有限输入显式降级）与 G22（DTD 闩锁双向化 + 暖机计数拆分），
以及 G14（RES 最小长度）与 G24（echo_prior=None 删除）的边界行为。

全部离线合成信号，<30s；不依赖 D:/data 外部数据。
对应实现在 pipeline.py / dtd.py / gcc.py / res.py，证据见
reports/refactor_wave5/deep_audit.md 与 _t10_probe.py（before/after 指纹对拍）。
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp" / "06_trad_aec"))
sys.path.insert(0, str(ROOT))

from dtd import DTDStreaming  # noqa: E402
from gcc import align_far, gcc_phat_stats  # noqa: E402
from pipeline import TradAEC  # noqa: E402
from res import residual_suppress  # noqa: E402

SR = 16000
B = 256


def _voice_like(rng, n):
    """宽带类语音信号（多正弦谐波 + 音节率包络 + 低噪），与 hardening 用例同族。"""
    t = np.arange(n) / SR
    x = np.zeros(n)
    for f0 in (140.0, 210.0, 330.0, 510.0, 830.0, 1200.0):
        x += (0.3 + 0.2 * np.sin(2 * np.pi * 0.7 * t + f0)) * np.sin(2 * np.pi * f0 * t)
    env = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 4 * t)) ** 0.3
    return (x * env + 0.01 * rng.standard_normal(n)).astype(np.float64)


# --------------------------------------------------------------------- G1：非有限输入
def test_g1_nonfinite_far_degrades_explicitly_not_silently():
    """far（参考端）含 Inf：显式降级（告警 + n_gcc_fallback + 不为 nan 的有限时延），
    不得把不可用证据当"无回声"而不告警。"""
    rng = np.random.default_rng(7)
    n = 3 * SR
    far = _voice_like(rng, n)
    far[1000:1004] = np.inf
    mic = 0.4 * far

    delay_ms, ratio, valid = gcc_phat_stats(mic, far, SR)
    assert valid is False, "far 含 Inf 时必须报告证据不可用"
    assert np.isfinite(delay_ms)

    aec = TradAEC(method="mdf", partitions=8, mu=0.7)
    with pytest.warns(RuntimeWarning, match="GCC-PHAT 证据失效"):
        r = aec.process(mic, far)
    assert r.n_gcc_fallback == 1
    assert np.isfinite(r.out).all() and np.isfinite(r.error).all()
    assert np.isfinite(r.align_ms), "对齐量不得被 NaN 污染（否则 align_far 会崩）"
    assert all(np.isfinite(float(t)) for t, _, _ in r.delay_track)


def test_g1_finite_far_with_nan_mic_keeps_evidence():
    """mic 侧 NaN 不影响参考端：证据照常给出（不下调级、不告警），
    且输出逐位有限——区分"mic 脏"与"far 脏"是 G1 的关键设计。"""
    rng = np.random.default_rng(5)
    n = 3 * SR
    far = _voice_like(rng, n)
    mic = far.copy()
    mic[7 * B:7 * B + 8] = np.nan

    _, _, valid = gcc_phat_stats(mic, far, SR)
    assert valid is True, "mic 含 NaN 时 far 参考端仍有效，不应判为证据不可用"

    aec = TradAEC(method="mdf", partitions=8, mu=0.7)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        r = aec.process(mic, far)          # 不应产生本模块的降级告警
    assert r.n_gcc_fallback == 0
    assert np.isfinite(r.out).all()


def test_g1_nan_delay_never_reaches_align_far():
    """align_far 对非有限 delay 显式报错（而不是 int(round(nan)) 的隐晦崩溃）。"""
    with pytest.raises(ValueError, match="必须有限"):
        align_far(np.zeros(1024), float("nan"), SR, 1024)
    # 负对齐量按 0 处理（far 早于 mic 无物理意义）
    assert align_far(np.arange(10.0), -5.0, SR, 10)[0] == 0.0


def test_g1_empty_and_tiny_input_do_not_crash_gcc():
    """G3：空/单样本输入不再抛 OverflowError / IndexError，而是返回不可用证据。"""
    for n in (0, 1):
        d, ratio, valid = gcc_phat_stats(np.zeros(n), np.zeros(n), SR)
        assert valid is False and d == 0.0 and np.isnan(ratio)


def test_g4_huge_max_delay_ms_does_not_index_error():
    """G4：max_delay_ms 超过可搜索范围时钳制，不再 IndexError。"""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(SR) * 0.3
    b = rng.standard_normal(SR) * 0.3
    d, ratio, valid = gcc_phat_stats(a, b, SR, max_delay_ms=4000.0)
    assert valid is True and np.isfinite(ratio)
    assert 0.0 <= d <= SR / 2 / SR * 1000.0        # 钳到 N//2 以内


# --------------------------------------------------------------------- G22：闩锁与暖机
def _healthy_blocks(n_blocks, echo_gain=0.25, noise_db=40.0):
    """健康单讲回声：mic = echo_gain·far + 噪声（约 17dB ERLE，Geigel 不误报单讲为双讲）。

    echo_gain 取 0.25 是为了让 |mic| 峰值明显低于 alpha·far 峰值（0.5×），
    即"近端不存在"与 Geigel 判据自洽——这正是 G22 解锁条件的单讲前提。
    """
    rng = np.random.default_rng(0)
    far = rng.standard_normal(n_blocks * B) * 0.3
    echo = echo_gain * far
    mic = echo + 10 ** (-noise_db / 20) * rng.standard_normal(n_blocks * B)
    return mic, far, echo


def test_g22_latch_unlocks_when_echo_estimate_breaks():
    """G22 核心：收敛后回声估计变 0（块级 ERLE 变负）不得长期继续全速自适应。

    修复前实测：60 块健康后连喂 10 块 y=0，10 块全部返回 0（全速自适应、闩锁不复位）；
    修复后：单讲块 + ERLE EMA 掉出滞后带 -> 失锁（n_unlatched+1）-> 重新暖机冻结。
    ERLE EMA 以 0.1/块下降，故失锁不是瞬时（阈 6dB、滞后 3dB -> 约十余块）。
    """
    ds = DTDStreaming(frame=B, sr=SR, echo_prior=True)
    mic, far, y = _healthy_blocks(100)
    for k in range(70):
        sl = slice(k * B, (k + 1) * B)
        ds.update(mic[sl], far[sl], y[sl])
    assert ds.converged, "健康回声下应先收敛（闩锁生效）"

    states = []
    for k in range(70, 100):
        sl = slice(k * B, (k + 1) * B)
        states.append(ds.update(mic[sl], far[sl], np.zeros(B)))
    assert states[0] == 0, "失锁前（ERLE 仍在滞后带内）应保持自适应"
    assert ds.n_unlatched >= 1, f"回声估计失效后必须解锁，实测 states={states}"
    assert states[-1] == 2, f"失锁后应回到硬冻结（重新暖机），实测 states={states}"
    assert ds.converged is False


def test_g22_latch_kept_during_double_talk():
    """G22 反面：真双讲（Geigel 触发）**不得**解锁——DT 段 ERLE 被近端拉低是正常现象，
    据此解锁会让闩锁保护失效。"""
    ds = DTDStreaming(frame=B, sr=SR, echo_prior=True)
    mic, far, y = _healthy_blocks(80)
    for k in range(70):
        sl = slice(k * B, (k + 1) * B)
        ds.update(mic[sl], far[sl], y[sl])
    assert ds.converged

    rng = np.random.default_rng(1)
    for k in range(70, 80):
        sl = slice(k * B, (k + 1) * B)
        near = rng.standard_normal(B) * 0.6          # 强近端 -> Geigel 触发
        st = ds.update(mic[sl] + near, far[sl], y[sl])
        assert st != 2, "双讲块应走软冻结（state=1）或自适应，而不是解锁后硬冻结"
    assert ds.converged, "双讲期间闩锁必须保留"
    assert ds.n_unlatched == 0


def test_g22_warmup_counter_observable_and_reset():
    """G22：静默块会推进块序号 nb（与 HEAD 行为一致，默认路径逐位兼容），
    单独的 n_warm 只统计远端活跃块供诊断；reset() 必须把两者都归零。"""
    ds = DTDStreaming(frame=B, sr=SR, echo_prior=True)
    silent = np.zeros(B)
    for _ in range(20):
        st = ds.update(silent, silent, silent)
        assert st == 2                       # 静默 -> 硬冻结
    assert ds.nb == 20                       # 块序号含静默块（暖机判据使用）
    assert ds.n_warm == 0, "n_warm 只统计远端活跃块（诊断用途）"
    ds.reset()
    assert ds.nb == 0 and ds.n_warm == 0


def test_g22_reset_clears_both_counters_and_latch():
    """reset() 仍是唯一的"重对齐/换文件"清理入口，两个计数器都要归零。"""
    ds = DTDStreaming(frame=B, sr=SR, echo_prior=True)
    mic, far, y = _healthy_blocks(40)
    for k in range(40):
        sl = slice(k * B, (k + 1) * B)
        ds.update(mic[sl], far[sl], y[sl])
    assert ds.converged
    ds.reset()
    assert not ds.converged and ds.nb == 0 and ds.n_warm == 0 and ds.erle_ema is None
    assert ds.update(mic[:B], far[:B], y[:B]) == 2    # 暖机期重新冻结
    assert ds.nb == 1


def test_g22_state_semantics_unchanged_on_healthy_echo():
    """行为改变只限"先前静默劣化"的输入：健康单讲回声下仍走收敛->放行自适应。"""
    ds = DTDStreaming(frame=B, sr=SR, echo_prior=True)
    mic, far, y = _healthy_blocks(80)
    states = [ds.update(mic[k * B:(k + 1) * B], far[k * B:(k + 1) * B], y[k * B:(k + 1) * B])
              for k in range(80)]
    assert states[:8] == [2] * 8            # 暖机期硬冻结
    assert states[-1] == 0                  # 暖机后放行自适应
    assert ds.converged


# --------------------------------------------------------------------- G24 / G14
def test_g24_echo_prior_none_is_rejected():
    """G24：数值上不成立的"自动回声存在性"回退判据已删除，None 直接报错。"""
    with pytest.raises(ValueError, match="echo_prior=None"):
        DTDStreaming(frame=B, sr=SR, echo_prior=None)
    ds = DTDStreaming(frame=B, sr=SR)          # 默认 True
    assert ds.echo_prior is True
    ds_false = DTDStreaming(frame=B, sr=SR, echo_prior=False)
    assert ds_false.echo_prior is False


def test_g14_res_and_pipeline_reject_too_short_input():
    """G14 + V5 终审 F4：真实下限是 n_fft - hop = 256（非 n_fft=512），错误文本含原因。

    F4 更正：旧断言把"刚好过线"钉在 513，且隐含"512 不可用"——实测 HEAD 版 n=512 可成功；
    257..511 在 HEAD 下因 istft 形状不匹配崩（另一处缺陷），本波随 F4 零补修法一并修好。
    """
    short = np.zeros(256)
    with pytest.raises(ValueError, match="需要输入长度 >"):
        residual_suppress(short, short, gate=np.ones(256))
    with pytest.raises(ValueError, match="需要长度 >"):
        TradAEC(method="mdf").process(np.zeros(256), np.zeros(256))
    # 新可用面：257（真实下限+1）、512（HEAD 可用、t10 曾误拒）、513 都必须可用且长度不变
    for n in (257, 512, 513):
        ok = residual_suppress(np.zeros(n), np.zeros(n), gate=np.ones(n))
        assert len(ok) == n, f"n={n}: 输出长度应等于输入"
    r = TradAEC(method="mdf", partitions=4).process(np.zeros(512), np.zeros(512))
    assert r.out.shape == (512,) and np.isfinite(r.out).all()


# ------------------------------------------------- F5a：gcc ratio 的 lag 集口径（终审 F5a）
def test_f5a_gcc_ratio_uses_searched_peak_lag_set():
    """F5a 固化（V5 终审）：时延超出搜索窗时 ratio 必须反映**搜索到的峰**。

    构造 far 滞后 mic 100ms（非物理、搜索峰落负 lag 环回区）：旧实现分子写死
    `cc[:maxlag].max()`（只含正 lag 区）→ ratio ≈ 4.8 < 阈值 14 ⇒ 把"有回声"误判成
    "无回声"（DTD 持续硬冻结、ERLE≈0）；修复后分子与延迟搜索同 lag 集 ⇒ ratio 反映真实峰。
    **变异验证**：把 gcc.py 的分子改回 `cc[:maxlag].max()`，本用例必须变红。
    """
    rng = np.random.default_rng(0)
    n = 32000
    mic = (rng.standard_normal(n) * 0.3).astype(np.float64)
    far = np.concatenate([np.zeros(1600), mic])[:n]          # far 滞后 mic 100ms（超窗）
    _d_est, ratio, valid = gcc_phat_stats(mic, far, SR)
    assert valid
    # 旧实现此处实测 ~4.8（<14，会误判"无回声"）；新实现反映环回峰（实测 ~990）
    assert ratio >= 14.0, \
        f"ratio={ratio:.3f} 应反映搜索峰（旧实现被系统性低估到 ~4.8 < 阈值 14）"


# ------------------------------------- F5b：cascade failed 统计 latest 行口径（终审 F5b）
def test_f5b_cascade_failed_stats_use_latest_row(tmp_path, monkeypatch):
    """F5b 固化（V5 终审）：failed 统计按**最新行**口径。

    造"同一 clip 上一轮 failed、本轮 ok"的 jsonl：新口径 failed=0；
    旧实现（从原始行 `score_rows_raw` 统计）会得 1 ⇒ 已修复的 clip 永久留在失败清单。
    **变异验证**：把 cascade_real_data.agg_official 的 failed 统计改回原始行口径，本用例必须变红。
    """
    import importlib.util
    import json as _json

    here = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "crd_f5b", here / "exp" / "09_cascade_aec" / "cascade_real_data.py")
    crd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(crd)

    score = tmp_path / "scores.jsonl"
    rows = [
        dict(variant="proto", method="cascade", subset="dt", name="clipA", status="failed",
             echo_mos=None, deg_mos=None, n48=100, trim=0, error="boom(上一轮)"),
        dict(variant="proto", method="cascade", subset="dt", name="clipA", status="ok",
             echo_mos=3.0, deg_mos=1.5, n48=100, trim=0, error=""),
    ]
    score.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(crd, "SCORE_JSONL_OFF", score)
    monkeypatch.setattr(crd, "LIN_JSONL_OFF", tmp_path / "none_lin.jsonl")
    monkeypatch.setattr(crd, "OOD_JSONL", tmp_path / "none_ood.jsonl")
    monkeypatch.setattr(crd, "RESULTS", tmp_path / "results")

    crd.agg_official()

    payload = _json.loads((tmp_path / "results" / "aecmos_cascade_official.json")
                          .read_text(encoding="utf-8"))
    dt = payload["summary"]["dt"]
    assert dt["n"] == 1, "本轮 ok 行应被计入"
    assert dt["failed"] == 0, f"最新行是 ok，failed 应为 0（旧口径会得 1）：{dt}"
    assert dt["failed_clips"] == []
