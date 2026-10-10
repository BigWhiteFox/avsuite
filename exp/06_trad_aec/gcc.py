"""GCC-PHAT 时延估计（WP-06）。

互相关 + PHAT 白化（每频点单位模长）——语音宽带相关函数会被谐波结构主导，
白化后峰更尖锐、对混响更鲁棒。
"""
import numpy as np


def gcc_phat_stats(mic: np.ndarray, far: np.ndarray, sr: int,
                   max_delay_ms: float = 120.0):
    """GCC-PHAT 时延 + 峰旁瓣比（回声存在性证据）。

    peak_ratio = 相关峰 / 旁瓣中位数：有回声（线性卷积关系）时白化相关峰尖锐
    （实测 FEST 19-84 / DT 15-66）；mic 与 far 独立时相关是平坦噪声
    （实测 NST 4.6-13.4）。阈值 14 可分——比累积相关系数 robust 得多：
    后者被混响稀释（corr ≈ 直达径/总能量比，仅 0.1-0.3），与 NST 噪声带重叠。
    返回 (delay_ms, peak_ratio)。
    """
    n = len(mic)
    N = 1 << int(np.ceil(np.log2(2 * n)))
    M, F = np.fft.rfft(mic, N), np.fft.rfft(far, N)
    R = M * np.conj(F)
    R /= np.abs(R) + 1e-12                       # PHAT 白化
    cc = np.fft.irfft(R, N)
    maxlag = int(max_delay_ms * sr / 1000)
    lags = np.concatenate([np.arange(0, maxlag + 1), np.arange(N - maxlag, N)])
    k = lags[np.argmax(cc[lags])]
    d = k if k <= maxlag else k - N              # 负 lag = far 滞后 mic（非物理，截 0）
    seg = np.abs(cc[:maxlag])
    ratio = float(cc[:maxlag].max() / (np.median(seg) + 1e-12))
    return max(d, 0) / sr * 1000.0, ratio


def gcc_phat_delay(mic: np.ndarray, far: np.ndarray, sr: int,
                   max_delay_ms: float = 120.0) -> float:
    """估计 mic 中回声相对 far 参考的时延（ms，正数=mic 滞后）。"""
    return gcc_phat_stats(mic, far, sr, max_delay_ms)[0]


def align_far(far: np.ndarray, delay_ms: float, sr: int, n_out: int) -> np.ndarray:
    """把 far 右移 delay_ms（对齐到 mic 坐标系），定长 n_out（不足补零）。"""
    d = int(round(delay_ms * sr / 1000))
    shifted = np.concatenate([np.zeros(d), far])[:n_out]
    if len(shifted) < n_out:
        shifted = np.concatenate([shifted, np.zeros(n_out - len(shifted))])
    return shifted
