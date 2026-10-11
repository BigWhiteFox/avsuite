"""P3-1 故障注入练习 BUG-02:DTD 去掉收敛闩锁 -> 冷启动 ERLE 锁在低位。

对应实战坑(exp/06_trad_aec/dtd.py DTDStreaming docstring「收敛闩锁」段):
收敛闩锁要求块级 ERLE 的 EMA 连续 >= latch_blocks 块超过 erle_thr 才允许判 DT;
冷启动 y≈0、γ²≈0,不闩锁会把整个收敛期误判成双讲而永久冻结(鸡生蛋问题)。

注入方式:monkeypatch pipeline.DTDStreaming -> 子类,converged 恒 True
(= 闩锁永闭;状态机里 converged 分支位于 warmup 之前,故 warmup 分支被短路,
前 8 块由 DT 判决接管——结局同为冻结,等价于无闩锁形态)。
观察到的失败模式(实测,seed=11,4s 纯 FEST,直达主导 RIR,ERL≈3dB,SNR≈20dB):
    正确:   dt_frac=0.060,ERLE[.5-1s]=10.50dB、[1-2s]=17.77dB
    无闩锁: dt_frac=0.632,ERLE[.5-1s]= 1.03dB、[1-2s]= 1.32dB(冷启动锁死)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trad06 import dtd as dtd_mod  # noqa: E402
from trad06 import pipeline as pipe_mod  # noqa: E402

SR = 16000


class _NoLatchDTD(dtd_mod.DTDStreaming):
    """BUG-02 注入:收敛闩锁失效(converged 恒 True == 闩锁永闭)。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.converged = True


def _scenario():
    """4s 纯 FEST:直达主导 RIR + ERL≈3dB(踩 Geigel alpha=0.5 假设边界)+ 20dB 噪声底。"""
    rng = np.random.default_rng(11)
    n = SR * 4
    far = rng.standard_normal(n) * 0.3
    h = np.zeros(1600)
    h[0] = 1.0
    h[8:] = 0.03 * np.random.default_rng(12).standard_normal(1592) * \
        np.exp(-np.arange(1592) / 300)
    echo = 0.7 * np.convolve(far, h / np.linalg.norm(h))[:n]
    mic = echo + 0.1 * echo.std() * rng.standard_normal(n)
    return mic, far


def _erle(mic, err, a, b) -> float:
    return 10 * np.log10(np.sum(mic[a:b] ** 2) / (np.sum(err[a:b] ** 2) + 1e-12))


def test_bug02_removing_convergence_latch_locks_coldstart_erle(monkeypatch):
    """无闩锁:冷启动段(y≈0,γ²≈0)被误判双讲 -> 步长×0.02+hangover -> ERLE 锁死低位。"""
    mic, far = _scenario()

    res_ok = pipe_mod.TradAEC(method="mdf", partitions=8, mu=0.5,
                              use_res=False).process(mic, far)
    monkeypatch.setattr(pipe_mod, "DTDStreaming", _NoLatchDTD)
    res_bug = pipe_mod.TradAEC(method="mdf", partitions=8, mu=0.5,
                               use_res=False).process(mic, far)

    # 冷启动之后、正确路径已收敛的窗:1-2s
    erle_ok = _erle(mic, res_ok.error, SR, 2 * SR)
    erle_bug = _erle(mic, res_bug.error, SR, 2 * SR)
    assert erle_ok > 12.0, f"正确路径 1-2s ERLE {erle_ok:.2f}dB 应 >12(实测 17.77)"
    assert erle_bug < 5.0, f"无闩锁 1-2s ERLE {erle_bug:.2f}dB 应锁在低位(实测 1.32)"
    assert erle_ok - erle_bug > 8.0, "闩锁应有 >8dB 的冷启动 ERLE 差"

    # 冷启动窗本身(0.5-1s):正确路径已爬到 10dB,无闩锁仍在 1dB 级
    assert _erle(mic, res_bug.error, SR // 2, SR) < 4.0, "无闩锁 0.5-1s 应仍锁死(实测 1.03)"

    # dt_frac:无闩锁大量误报(实测 0.632),正确路径仅 warmup 计入(实测 0.060)
    assert res_bug.dt_frac > 0.4, f"无闩锁 dt_frac {res_bug.dt_frac:.3f} 应大量误报"
    assert res_ok.dt_frac < 0.2, f"正确路径 dt_frac {res_ok.dt_frac:.3f} 应 <0.2"
