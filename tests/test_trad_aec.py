# -*- coding: utf-8 -*-
"""WP-06 单测：GCC 精度 / NLMS 与 MDF 已知 FIR 辨识 / DTD 判决 / pipeline 端到端 ERLE / RES 门控。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trad06 import gcc, dtd as dtd_mod, mdf as mdf_mod, nlms as nlms_mod, pipeline as pipe_mod  # noqa: E402

TradAEC = pipe_mod.TradAEC

SR = 16000


def _erle(mic, err, skip_frac=0.25):
    k = int(len(mic) * skip_frac)          # 跳过收敛瞬态，只看稳态
    return 10 * np.log10(np.sum(mic[k:] ** 2) / (np.sum(err[k:] ** 2) + 1e-12) + 1e-12)


def test_gcc_exact_delay_dirac():
    rng = np.random.default_rng(0)
    far = rng.standard_normal(SR) * 0.3
    d = 600                                   # 37.5ms
    mic = np.concatenate([np.zeros(d), far])[:SR]
    est = gcc.gcc_phat_delay(mic, far, SR)
    assert abs(est - d / SR * 1000) < 0.2     # 亚样本地rounding误差


def test_nlms_identifies_fir():
    rng = np.random.default_rng(1)
    h = np.array([0.0, 0.0, 0.5, 0.3, -0.2, 0.1, 0.05, 0.02])
    far = rng.standard_normal(SR) * 0.3
    mic = np.convolve(far, h)[:SR]
    af = nlms_mod.NLMS(length=16, mu=0.5)
    e = af.process(far, mic)
    assert _erle(mic, e) > 20
    np.testing.assert_allclose(af.w[:len(h)], h, atol=5e-2)


def test_mdf_identifies_fir():
    """已知 512-tap FIR：MDF(B=256,P=4) 稳态 ERLE>25dB 且等效 taps 收敛到真值。"""
    rng = np.random.default_rng(2)
    h = rng.standard_normal(512) * np.exp(-np.arange(512) / 200)
    far = rng.standard_normal(2 * SR) * 0.3
    mic = np.convolve(far, h)[:2 * SR]
    af = mdf_mod.MDF(block=256, partitions=4, mu=0.5)
    e = af.process(far, mic)
    assert _erle(mic, e) > 25
    taps = af.taps[:512]
    np.testing.assert_allclose(taps / (np.linalg.norm(taps) + 1e-12),
                               h / (np.linalg.norm(h) + 1e-12), atol=2e-2)


def test_mdf_beats_nlms_on_colored_input():
    """有色输入（AR(1) 语音谱近似）下：MDF 逐频点归一化不受病态相关矩阵拖累，
    稳态 ERLE 应显著优于同时长预算的 NLMS（教科书结论的实测版）。"""
    from scipy.signal import lfilter
    rng = np.random.default_rng(3)
    h = rng.standard_normal(1024) * np.exp(-np.arange(1024) / 300)
    far = lfilter([1.0], [1.0, -0.9], rng.standard_normal(2 * SR) * 0.3)  # 有色
    mic = np.convolve(far, h)[:2 * SR]
    e_mdf = mdf_mod.MDF(block=256, partitions=4, mu=0.5).process(far, mic)
    e_nlms = nlms_mod.NLMS(length=1024, mu=0.5).process(far, mic)
    assert _erle(mic, e_mdf) > _erle(mic, e_nlms) + 3


def test_dtd_decisions():
    """FEST 不误报（<20%）；真双讲高检出（>80%）；NST 判为双讲（保护近端）。"""
    rng = np.random.default_rng(4)
    fr = 256
    d = dtd_mod.DTD(frame=fr)
    far = rng.standard_normal(SR) * 0.3
    # FEST：回声=0.3×far（路径衰减>6dB，相关高）
    fest_mic = 0.3 * far
    flags = d.decide_frames(fest_mic, far)
    assert flags.mean() < 0.2
    # DT：近端(独立信号)叠加在衰减回声上，相关骤降 + 能量抬高
    near = rng.standard_normal(SR) * 0.5
    dt_mic = 0.3 * far + near
    flags = d.decide_frames(dt_mic, far)
    assert flags.mean() > 0.8
    # NST：far 静默、只有近端 -> 判双讲（冻结自适应+RES 直通，保护近端）
    nst_mic = near.copy()
    flags = d.decide_frames(nst_mic, far * 1e-6)
    assert flags.mean() > 0.8


def test_pipeline_fest_erle_and_dt_protection():
    """端到端：FEST（30ms时延+直达径主导RIR，ERL≈6dB）稳态 ERLE≥20dB；
    FEST收敛后近端加入 -> DTD检出+冻结 -> 近端不被消掉（SI-SNR>15dB）。

    RIR 用直达径主导（真实房间/pyroomacoustics 皆如此）：随机 iid 长 RIR 无直达峰，
    GCC-PHAT 会偏 20+ 样本 -> 残余路径非因果 -> ERLE 被封顶（实测教训 11.5dB）。
    """
    rng = np.random.default_rng(5)
    n = SR * 4
    far = rng.standard_normal(n) * 0.3
    h = np.zeros(1600)
    h[0] = 1.0                                            # 直达径主导
    h[8:] = 0.03 * rng.standard_normal(1592) * np.exp(-np.arange(1592) / 300)
    echo = 0.5 * np.convolve(far, h / np.linalg.norm(h))[:n]   # ERL≈6dB
    d = 480                                               # 30ms 时延
    echo_d = np.concatenate([np.zeros(d), echo])[:n]

    aec = TradAEC(method="mdf", partitions=8, mu=0.5, use_res=False)
    res = aec.process(echo_d, far)
    assert abs(res.delay_ms - 30.0) <= 1.5                # 直达径 RIR 下 GCC 应锁准
    assert _erle(echo_d, res.error, skip_frac=0.4) > 20   # DoD 线性 AEC 目标
    assert res.dt_frac < 0.2                              # FEST 不应误报双讲

    # FEST(2s) -> DT(2s)：滤波器先收敛，近端再加入
    near = np.zeros(n)
    near[2 * SR:] = rng.standard_normal(2 * SR) * 0.3     # SER≈+6dB（近端占优）
    mic_dt = near + echo_d
    res_dt = aec.process(mic_dt, far)
    assert res_dt.dt_frac > 0.25                          # DT 段被检出

    def si_snr(x, s):
        x, s = x - x.mean(), s - s.mean()
        p = (x @ s) / (s @ s) * s
        return 10 * np.log10((p @ p) / ((x - p) @ (x - p) + 1e-12) + 1e-12)

    k = int(2.5 * SR)                                     # DT 稳态段
    assert si_snr(res_dt.error[k:], near[k:]) > 15        # 近端保留（冻结生效）


def test_res_gated_protects_nearend_and_boosts_fest():
    """RES：FEST 段 ERLE 提升；门控下 DT 段近端误差信号几乎不变。"""
    from trad06 import res as res_mod
    residual_suppress = res_mod.residual_suppress
    rng = np.random.default_rng(6)
    n = SR
    far = rng.standard_normal(n) * 0.3
    echo = 0.3 * far + 0.02 * rng.standard_normal(n)     # 残余 5% 的回声
    # 全程允许抑制（FEST 场景）：g_min=0.1 地板 -> ERLE≈20dB
    out = residual_suppress(echo, 0.3 * far, gate=np.ones(n))
    assert 10 * np.log10(np.sum(echo ** 2) / (np.sum(out ** 2) + 1e-12)) > 3
    # 门控关闭（DT 场景）：直通（比较内部区，STFT 边界样本除外）
    out2 = residual_suppress(echo, 0.3 * far, gate=np.zeros(n))
    assert np.allclose(out2[512:-512], echo[512:-512], atol=1e-3)
