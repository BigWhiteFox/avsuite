"""P3-1 故障注入练习 BUG-03:DTD 相干参考改回 mic-vs-far -> 双讲段误报 dt_frac 0.93+。

对应实战坑(exp/06_trad_aec/README.md「DTD 判据演进」、dtd.py DTDStreaming docstring):
DTD 判据三代:v1 mic-vs-far 时域相关(随机 RIR 打散,FEST 误报 95%)→
v2 mic-vs-far **分段相干**(帧长 256 << RIR 1600+ taps,段内泄漏使纯回声相干
只有 ~0.16-0.2 → 全被误判双讲,实测 dt_frac 0.93-1.00、ERLE 被冻结)→
v3(现版)mic-vs-**y_echo** 后置相干(逐块时间对齐、无段内泄漏)+ ERLE 收敛闩锁。
本练习把相干参考改回 far,复现 v2 形态的失败模式。

注入方式:monkeypatch pipeline.DTDStreaming -> 子类。update() 为 dtd.py
L119-196 的**逐行副本**,唯一差异是把相干第二通道从 y_echo 换回 far_b
(dtd.py L169 `Y = np.fft.rfft(y_b * self.win)` -> `far_b * self.win`);
并令 converged 恒 True(v2 历史形态——收敛闩锁是 v3 才引入的,故 v2 判据
无闩锁)。
观察到的失败模式(实测,seed=14,6s 纯 FEST,扩散型 RIR(直达 480+混响尾主导,
帧内能量占比 0.169),ERL≈-2dB,SNR≈16.5dB):
    正确:      dt_frac=0.040,ERLE[1-3s]=15.68dB
    v2 注入:   dt_frac=0.989(历史档 0.93-1.00),ERLE[1-3s]= 6.35dB(永不收敛)
    附:只改相干、保留闩锁的变体 dt_frac=0.909,ERLE 卡在半收敛 ~12dB
    (闩锁让滤波器先收敛、之后才被 coh 误报冻住——这正是 v3 补闩锁的原因)。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trad06 import dtd as dtd_mod  # noqa: E402
from trad06 import pipeline as pipe_mod  # noqa: E402

SR = 16000


class _MicVsFarDTD(dtd_mod.DTDStreaming):
    """dtd.py::DTDStreaming.update 逐行副本,唯一差异:相干第二通道 y_b -> far_b。"""

    def update(self, mic_b: np.ndarray, far_b: np.ndarray, y_b: np.ndarray) -> int:
        eps = 1e-12
        # 1) 远端活跃度(同原版)
        frms = float(np.sqrt(np.mean(far_b ** 2)))
        self.far_rms_max = max(self.far_rms_max * 0.9995, frms, 1e-10)
        self.far_active = (frms > self.far_rel_thr * self.far_rms_max
                           and frms > self.far_abs_floor)
        self.far_peaks = np.roll(self.far_peaks, 1)
        self.far_peaks[0] = np.abs(far_b).max()
        # 2) mic-vs-far 累积相关(同原版,闩锁证据门槛用)
        if self.far_active:
            cw = 0.995
            self.cmxy = cw * self.cmxy + float(mic_b @ far_b)
            self.cmxx = cw * self.cmxx + float(mic_b @ mic_b)
            self.cmyy = cw * self.cmyy + float(far_b @ far_b)
            self.n_active += 1
        corr_cum = abs(self.cmxy) / (np.sqrt(self.cmxx * self.cmyy) + eps)
        corr_reliable = self.n_active >= self.corr_min_blocks
        # 3) 收敛闩锁(同原版,吃真实 y_b 的 ERLE)
        e_b = mic_b - y_b
        erle_inst = 10 * np.log10((np.mean(mic_b ** 2) + eps) / (np.mean(e_b ** 2) + eps))
        if self.erle_ema is None:
            self.erle_ema = erle_inst
        else:
            self.erle_ema = 0.9 * self.erle_ema + 0.1 * erle_inst
        if self.erle_ema > self.erle_thr:
            self.run += 1
        else:
            self.run = max(self.run - 1, 0)
        if self.run >= self.latch_blocks and (self.echo_prior or
                                              corr_cum >= self.warmup_corr_thr):
            self.converged = True
        # 4) 远端静默硬冻结(同原版)
        if not self.far_active:
            self.nb += 1
            return 2
        # 5) mic-vs-far 相干(BUG-03 注入点:原版此处第二通道为 y_b)
        X = np.fft.rfft(mic_b * self.win)
        Y = np.fft.rfft(far_b * self.win)
        b = 1.0 - self.ema
        self.sxx = self.ema * self.sxx + b * (X * X.conj()).real
        self.syy = self.ema * self.syy + b * (Y * Y.conj()).real
        self.sxy = self.ema * self.sxy + b * (X * Y.conj())
        coh = float(np.mean((np.abs(self.sxy) ** 2 / (self.sxx * self.syy + eps))[self.bins]))
        # 6) Geigel(同原版)
        geigel = np.abs(mic_b).max() > self.alpha * self.far_peaks.max() + 1e-10
        # 7) 状态机(同原版)
        if self.converged:
            state = 1 if (self.far_active and geigel and coh < self.coh_thr) else 0
        elif self.nb < self.warm_blocks:
            state = 2
        elif self.echo_prior is True:
            state = 0
        elif self.echo_prior is False:
            state = 2
        else:
            state = 2 if (not corr_reliable or
                          (self.far_active and corr_cum < self.warmup_corr_thr)) else 0
        self.nb += 1
        if state:
            self.hang = self.hangover
            self.last_state = state
        else:
            self.hang = max(self.hang - 1, 0)
        return state if not self.hang or state else self.last_state


class _MicVsFarV2DTD(_MicVsFarDTD):
    """v2 历史形态:mic-vs-far 相干 + 无收敛闩锁(闩锁 v3 才加)。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.converged = True


def _scenario():
    """6s 纯 FEST:扩散型 RIR(帧内能量占比 0.169 -> mic-vs-far 分段相干被段内泄漏压低)。"""
    r3 = np.random.default_rng(13)
    h = np.zeros(1600)
    h[480] = 1.0
    h[481:] = 0.15 * r3.standard_normal(1119) * np.exp(-np.arange(1119) / 400)
    rng = np.random.default_rng(14)
    n = SR * 6
    far = rng.standard_normal(n) * 0.3
    echo = 0.8 * np.convolve(far, h / np.linalg.norm(h))[:n]
    mic = echo + 0.15 * echo.std() * rng.standard_normal(n)
    return mic, far


def _erle(mic, err, a, b) -> float:
    return 10 * np.log10(np.sum(mic[a:b] ** 2) / (np.sum(err[a:b] ** 2) + 1e-12))


def test_bug03_mic_vs_far_coherence_false_dt(monkeypatch):
    """v2 形态(相干回 far + 无闩锁):纯 FEST 被误判双讲 dt_frac 0.93+,ERLE 永不收敛。"""
    mic, far = _scenario()

    res_ok = pipe_mod.TradAEC(method="mdf", partitions=8, mu=0.5,
                              use_res=False).process(mic, far)
    monkeypatch.setattr(pipe_mod, "DTDStreaming", _MicVsFarV2DTD)
    res_bug = pipe_mod.TradAEC(method="mdf", partitions=8, mu=0.5,
                               use_res=False).process(mic, far)

    # 纯 FEST 场景:双讲占比即误报率
    assert res_bug.dt_frac > 0.9, \
        f"mic-vs-far 相干 dt_frac {res_bug.dt_frac:.3f} 应 0.93+(实测 0.989)"
    assert res_ok.dt_frac < 0.2, \
        f"正确路径 dt_frac {res_ok.dt_frac:.3f} 应 <0.2(实测 0.040)"

    # 误报连锁后果:步长被 FREEZE 压住,ERLE 永不收敛(实测 6.35 vs 15.68)
    erle_ok = _erle(mic, res_ok.error, SR, 3 * SR)
    erle_bug = _erle(mic, res_bug.error, SR, 3 * SR)
    assert erle_ok > 12.0, f"正确路径 1-3s ERLE {erle_ok:.2f}dB 应 >12(实测 15.68)"
    assert erle_bug < 10.0, f"v2 注入 1-3s ERLE {erle_bug:.2f}dB 应被误报压住(实测 6.35)"
