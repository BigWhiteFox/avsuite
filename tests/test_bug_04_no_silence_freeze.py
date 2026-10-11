"""P3-1 故障注入练习 BUG-04:去掉远端静默硬冻结 -> 滤波器被随机游走毒化。

对应实战坑(exp/06_trad_aec/pipeline.py IDLE 注释、README「关键设计决策 5」):
远端静默块硬冻结(IDLE=0.0):谱队列 X[p] 还留着旧 far 数据,静默期继续自适应
会把近端学成虚假相关(实测 NST y/mic=0.17、PESQ 2.30->1.73 的元凶);
dtd.py DTDStreaming 同款注释:「软冻结的随机游走 8s 也能积累出 40% 量级的
虚假系数,必须硬冻结」。

注入方式:monkeypatch pipeline.IDLE 0.0 -> 1.0(远端静默门失效,静默块按活跃
远端全速自适应;use_dtd=False 基线分支,单点注入、不与 DTD 自身的静默早退
纠缠)。被测源码零改动。
观察到的失败模式(实测,seed=21,2s FEST -> 6s 远端静默+近端 -> 2s FEST):
    毒化藏在系数里:静默段 y/mic 仅 0.0092 vs 正确 0.0027(X 是底噪,读数小);
    far 恢复段爆发:前 0.25s ERLE -16.22dB vs 正确 30.70dB(塌 47dB)。
    按任务许可,用 ERLE 塌陷替代 PESQ 断言(PESQ 在 10s 信号上过慢)。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trad06 import pipeline as pipe_mod  # noqa: E402

SR = 16000
SEG_A, SEG_B, SEG_C = SR * 2, SR * 6, SR * 2


def _scenario():
    """FEST 收敛 -> 远端静默(数字底噪 -70dBFS)+ 近端噪声 -> FEST 恢复。"""
    rng = np.random.default_rng(21)
    n = SEG_A + SEG_B + SEG_C
    far = rng.standard_normal(n) * 0.3
    far[SEG_A:SEG_A + SEG_B] = 3e-4 * rng.standard_normal(SEG_B)   # 远端静默(底噪)
    h = np.zeros(1600)
    h[0] = 1.0
    h[8:] = 0.03 * np.random.default_rng(22).standard_normal(1592) * \
        np.exp(-np.arange(1592) / 300)
    echo = 0.5 * np.convolve(far, h / np.linalg.norm(h))[:n]
    near = 0.05 * rng.standard_normal(n)                            # 近端噪声底
    mic = echo + 0.03 * echo.std() * rng.standard_normal(n)
    mic[SEG_A:SEG_A + SEG_B] = near[SEG_A:SEG_A + SEG_B] + 0.02 * rng.standard_normal(SEG_B)
    return mic, far


def _erle_seg(mic, err, a, b) -> float:
    return 10 * np.log10(np.mean(mic[a:b] ** 2) / (np.mean(err[a:b] ** 2) + 1e-12))


def test_bug04_no_silence_freeze_poisons_filter(monkeypatch):
    """静默期全速自适应:陈旧谱队列把近端学成虚假相关;far 恢复时 ERLE 塌陷。"""
    mic, far = _scenario()

    res_ok = pipe_mod.TradAEC(method="mdf", partitions=8, mu=0.5,
                              use_dtd=False, use_res=False).process(mic, far)
    monkeypatch.setattr(pipe_mod, "IDLE", 1.0)                      # BUG-04 注入
    res_bug = pipe_mod.TradAEC(method="mdf", partitions=8, mu=0.5,
                               use_dtd=False, use_res=False).process(mic, far)

    # 静默段:毒化累积进系数(近端被学成虚假相关),y 漂移大于正确路径
    b2a, b2b = SEG_A + SEG_B // 2, SEG_A + SEG_B
    y_ratio_ok = float(np.sqrt(np.mean(res_ok.y_echo[b2a:b2b] ** 2))
                       / np.sqrt(np.mean(mic[b2a:b2b] ** 2)))
    y_ratio_bug = float(np.sqrt(np.mean(res_bug.y_echo[b2a:b2b] ** 2))
                        / np.sqrt(np.mean(mic[b2a:b2b] ** 2)))
    assert y_ratio_bug > 2.0 * y_ratio_ok, \
        f"静默段 y/mic 漂移 {y_ratio_bug:.4f} 应 > 2× 正确 {y_ratio_ok:.4f}(实测 0.0092/0.0027)"

    # far 恢复段前 0.25s:被毒化的系数输出垃圾回声估计 -> ERLE 塌陷
    c0 = SEG_A + SEG_B
    erle_ok = _erle_seg(mic, res_ok.error, c0, c0 + SR // 4)
    erle_bug = _erle_seg(mic, res_bug.error, c0, c0 + SR // 4)
    assert erle_ok > 25.0, f"正确路径恢复段 ERLE {erle_ok:.2f}dB 应 >25(实测 30.70)"
    assert erle_bug < 10.0, \
        f"毒化路径恢复段 ERLE {erle_bug:.2f}dB 应塌陷(实测 -16.22)"
    assert erle_ok - erle_bug > 15.0, "硬冻结应保住恢复段 >15dB 的 ERLE 差"
