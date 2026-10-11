"""P3-1 故障注入练习 BUG-01:StreamingISTFT 误用 np.hanning(对称窗)。

对应实战坑(exp/07_nn_aec/README.md L143、core/istft.py 注释):
StreamingISTFT 默认窗必须是与 torch.stft 同款的**周期** hann
(np.sin(pi*k/N)**2);np.hanning 是对称窗、系数不同(实测 max|Δw|=4.7e-3),
流式 vs 离线对拍必翻车。

注入方式:参数替换(不改被测源码)——StreamingISTFT(window=np.hanning(n_fft))。
观察到的失败模式(实测,torch.manual_seed(0),3s 信号):
    f64:正确周期窗对拍 max|Δ|=7.6e-11;换对称窗 1.18e-2(爆 8 个量级)
    f32:正确 2.2e-7(~1e-6 级);换对称窗 1.18e-2
"""
import numpy as np
import torch

from core.istft import ISTFT, StreamingISTFT
from core.stft import STFT

N_FFT, HOP, SR = 512, 320, 16000
N = SR * 3


def _signal() -> torch.Tensor:
    torch.manual_seed(0)
    t = torch.arange(N, dtype=torch.float64) / SR
    return 0.8 * torch.sin(2 * torch.pi * 440 * t) + 0.2 * torch.randn(N).double()


def _stream_out(frames_td: torch.Tensor, window) -> np.ndarray:
    s = StreamingISTFT(N_FFT, HOP, window=window)
    return np.concatenate([s.add_frame(frames_td[0, i].numpy())
                           for i in range(frames_td.shape[1])])


def _aligned(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # 与 tests/test_streaming_nnaec.py::_align 同约定:流式 out[j] = 离线 y_full[j],
    # 离线 center=True 裁剪自 n_fft//2 起
    return out[N_FFT // 2: N_FFT // 2 + len(ref)]


def test_bug01_symmetric_window_breaks_streaming_parity():
    """对称窗合成 vs 周期窗分析:对拍误差爆到 1e-2 量级(正确路径 ~1e-10/f64、~1e-7/f32)。"""
    x = _signal()
    spec = STFT(N_FFT, HOP, "hann", True, pad_mode="zeros").transform(x)
    ref = ISTFT(N_FFT, HOP, "hann", True).inverse(spec)[0].numpy()
    frames_td = torch.fft.irfft(spec.transpose(1, 2), n=N_FFT)      # [T, n_fft]

    # 正确路径:window=None -> 默认周期 hann(sin^2)
    err_ok = float(np.abs(_aligned(_stream_out(frames_td, None), ref) - ref).max())
    # 注入路径:np.hanning 对称窗(实战坑原样)
    err_bad = float(np.abs(_aligned(_stream_out(frames_td, np.hanning(N_FFT)),
                                    ref) - ref).max())
    assert err_ok < 1e-8, f"正确周期窗对拍 {err_ok:.2e} 应 <1e-8(实测基线 7.6e-11)"
    assert err_bad > 5e-3, f"对称窗对拍 {err_bad:.2e} 应爆到 1e-2 量级(实测 1.18e-2)"
    assert err_bad > 1e4 * err_ok, "坏路径应比正确路径差 4 个量级以上"

    # f32 分析链:正确路径 ~1e-6 级,坏路径仍 1e-2 量级(误差由窗失配主导,非精度)
    spec32 = STFT(N_FFT, HOP, "hann", True, pad_mode="zeros").transform(x.float())
    ref32 = ISTFT(N_FFT, HOP, "hann", True).inverse(spec32)[0].numpy()
    fr32 = torch.fft.irfft(spec32.transpose(1, 2), n=N_FFT)
    err32_ok = float(np.abs(_aligned(_stream_out(fr32, None), ref32) - ref32).max())
    err32_bad = float(np.abs(_aligned(_stream_out(fr32, np.hanning(N_FFT)),
                                      ref32) - ref32).max())
    assert err32_ok < 1e-6, f"f32 正确对拍 {err32_ok:.2e} 应 ~1e-6 级(实测 2.2e-7)"
    assert err32_bad > 5e-3, f"f32 对称窗对拍 {err32_bad:.2e} 应仍爆(实测 1.18e-2)"
