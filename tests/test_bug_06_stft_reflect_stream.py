"""P3-1 故障注入练习 BUG-06:STFT center=True 配 reflect 搬进流式循环 -> 全链偏移。

对应实战坑(core/stft.py 模块注释「流式接口无法预知未来样本,约定
pad_mode='zeros'」、tests/test_streaming_nnaec.py 对拍约定):离线 STFT 默认
center=True + reflect 是**整段信号**约定;把 torch.stft(center=True,
pad_mode='reflect') 直接搬进逐块循环,每个 chunk 被当成独立信号:
镜像 pad 复制的是 chunk 自身边缘(首帧错位),且 hop 网格按 chunk 局部重启
(k*320 而非 t*256)——帧网格错位传播到整条链,重构输出整体偏移。

注入方式:包一层函数(测试内实现错误流式封装,不改被测源码)——
逐 chunk torch.stft(center=True, pad_mode='reflect') 拼谱后过统一 ISTFT。
观察到的失败模式(实测,torch.manual_seed(6),1.6s 白噪声):
    正确路径:流式帧 vs 离线帧逐元素一致(<1e-9,重构误差 ~1e-16);
    注入路径:max|y_bad - y_ref| = 1.688(信号幅度的 5 倍+),互相关峰
    滞后 145 样本(全链偏移签名)。
"""
import numpy as np
import torch

from core.istft import ISTFT
from core.stft import STFT, StreamingSTFT

N_FFT, HOP, CHUNK = 512, 256, 320
N = 25600                                   # 1.6s,CHUNK 整除


def test_bug06_center_reflect_per_chunk_shifts_whole_chain():
    torch.manual_seed(6)
    x = (0.3 * torch.randn(1, N)).double()

    # ---- 正确路径:全局帧网格 + zeros 约定(流式无法预知未来样本)----
    stft_off = STFT(N_FFT, HOP, "hann", True, pad_mode="zeros")
    spec_ref = stft_off.transform(x)
    sstream = StreamingSTFT(stft_off)
    frames_ok = []
    for i in range(0, N, CHUNK):
        frames_ok.extend(sstream.push(x[0, i:i + CHUNK]))
    frames_ok.extend(sstream.flush())
    spec_ok = torch.stack(frames_ok, dim=-1)                 # [1, F, T]
    frame_err = float((spec_ok - spec_ref).abs().max())
    assert frame_err < 1e-9, \
        f"正确约定下流式帧 vs 离线帧 {frame_err:.2e} 应逐元素一致(<1e-9)"

    y_ref = ISTFT(N_FFT, HOP, "hann", True).inverse(spec_ref)[0].numpy()
    rt_err = float(np.abs(y_ref - x[0, :len(y_ref)].numpy()).max())
    assert rt_err < 1e-9, f"参考路径往返重构 {rt_err:.2e} 应精确(~1e-16)"

    # ---- 注入路径:逐 chunk torch.stft(center=True, reflect)(实战坑原样)----
    w = torch.hann_window(N_FFT, periodic=True, dtype=torch.float64)
    frames_bad = []
    for i in range(0, N, CHUNK):
        ch = x[0, i:i + CHUNK]
        frames_bad.append(torch.stft(ch, N_FFT, HOP, N_FFT, w, center=True,
                                     pad_mode="reflect", return_complex=True))  # [F,2]
    spec_bad = torch.cat(frames_bad, dim=1).unsqueeze(0)     # [1, F, T_bad]
    y_bad = ISTFT(N_FFT, HOP, "hann", True).inverse(spec_bad)[0].numpy()

    L = min(len(y_bad), len(y_ref))
    err_bad = float(np.abs(y_bad[:L] - y_ref[:L]).max())
    assert err_bad > 0.5, \
        f"reflect 逐块流式对拍误差 {err_bad:.2e} 应远超正确路径(实测 1.688)"
    assert err_bad > 1e6 * max(frame_err, 1e-15), "坏路径应比正确路径差 6 个量级以上"

    # 全链偏移签名:互相关峰不在 0 滞后(实测 145 样本)
    seg = 4000
    a = y_ref[2000:2000 + seg] - y_ref[2000:2000 + seg].mean()
    c = np.correlate(y_bad[2000:2000 + seg + 600], a, mode="valid")
    lag = int(np.argmax(np.abs(c)))
    assert lag != 0, "互相关峰应偏离 0 滞后(实测 145 样本),否则不是偏移型失败"
