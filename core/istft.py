# -*- coding: utf-8 -*-
"""ISTFT：overlap-add 重构（WP-01 / 模块1）。

WOLA（加权重叠相加）公式：
    y[n] = Σ_t w[n - t·hop] · frame_t[n - t·hop] / Σ_t w²[n - t·hop]

要点：
- 分母是"窗平方"的 overlap-add。分析帧 = w·x，加窗合成 Σ w·frame = (Σw²)·x，
  除以窗平方和即精确还原——因此不依赖 COLA 条件，任意窗 + hop<=n_fft 均可无损
  （COLA 是"合成时不加归一化"才需要的条件，见 core/README.md 的辨析）；
- 分母趋零只发生在信号两端（窗端点为 0），用 eps 下限 + where 置零处理；
- center=True 时与 torch.istft 一致地裁掉两侧各 n_fft//2 的 pad 区；
- 对拍 torch.istft：float64 全量 <1e-10，float32 <1e-5（tests/test_istft.py）。
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .stft import make_window

__all__ = ["ISTFT", "StreamingISTFT"]


class ISTFT(nn.Module):
    """逆 STFT（nn.Module，window 随模型 .to(device)）。

    Args:
        n_fft / hop / window: 必须与分析端 STFT 严格一致。
        center: 对齐 torch.istft 默认（裁掉两侧 pad）。
        eps:    窗平方和下限，低于它的时间点输出 0（仅出现在信号两端）。
    """

    def __init__(self, n_fft: int = 512, hop: int = 256, window: str = "hann",
                 center: bool = True, eps: float = 1e-11):
        super().__init__()
        if not 0 < hop <= n_fft:
            raise ValueError(f"hop must be in (0, {n_fft}], got {hop}")
        self.n_fft = n_fft
        self.hop = hop
        self.center = center
        self.eps = eps
        self.register_buffer("window", make_window(n_fft, window), persistent=False)

    def inverse(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """复数谱 [B, F, T] -> 时域 [B, L]。

        length: 目标点数。None 时与 torch.istft 一致（center=True 输出 (T-1)*hop 点）；
        给定时截断 / 右侧零补到该长度。
        """
        if spec.dim() != 3:
            raise ValueError(f"expected [B, F, T], got {tuple(spec.shape)}")
        bsz, _, n_frames = spec.shape
        n_fft, hop = self.n_fft, self.hop
        w = self.window.to(spec.real.dtype)

        frames = torch.fft.irfft(spec.transpose(1, 2), n=n_fft, dim=-1)   # [B, T, n_fft]
        frames = frames * w                                               # 加窗合成
        out_len = (n_frames - 1) * hop + n_fft
        y = F.fold(frames.transpose(1, 2), output_size=(1, out_len),
                   kernel_size=(1, n_fft), stride=(1, hop)).reshape(bsz, out_len)
        w2 = w.square().view(1, n_fft, 1).expand(bsz, n_fft, n_frames).contiguous()
        wsum = F.fold(w2, output_size=(1, out_len),
                      kernel_size=(1, n_fft), stride=(1, hop)).reshape(bsz, out_len)
        y = torch.where(wsum > self.eps, y / wsum.clamp_min(self.eps),
                        torch.zeros_like(y))

        if self.center:
            y = y[:, n_fft // 2: y.shape[-1] - n_fft // 2]
        if length is not None:
            if length <= y.shape[-1]:
                y = y[..., :length]
            else:
                y = F.pad(y, (0, length - y.shape[-1]))
        return y


class StreamingISTFT:
    """流式 ISTFT（WOLA overlap-add，滑动缓冲）。

    网格约定：帧 t 的窗落在 [t·hop, t·hop + n_fft)——与离线 ISTFT(center=True)
    的帧网格一致（离线 center 的 256 零区加在输入信号上、被窗乘掉，帧网格本身
    从 0 开始；离线只是最后裁剪 [256, 256+(T-1)hop)）。因此流式输出 out[j] 即
    离线 y_full[j]，对齐时取 out[256 : 256+(T-1)·hop]（pad 区 [0,256) 两侧都≈0）。

    每 add_frame 一帧立即发射 hop 个样本：其最后覆盖窗就是当前帧
    （下一窗起点 (t+1)·hop 已越过该段），WOLA 分母此刻收齐，与离线逐位一致
    （parity 单测：core 级 + exp/07_nn_aec 模型级）。
    """

    def __init__(self, n_fft: int = 512, hop: int = 320, window: "np.ndarray|None" = None,
                 eps: float = 1e-11):
        import numpy as np
        if window is None:
            # 与 torch.hann_window(periodic=True) 一致（周期窗；np.hanning 是对称窗，
            # 系数不同——对拍必须同窗）
            window = np.sin(np.pi * np.arange(n_fft) / n_fft) ** 2
        self.n_fft, self.hop = n_fft, hop
        self.w = np.asarray(window, dtype=np.float64)
        self.eps = eps
        self.buf = np.zeros(n_fft + hop)
        self.wn = np.zeros(n_fft + hop)

    def add_frame(self, frame: "np.ndarray") -> "np.ndarray":
        """入一帧（n_fft 实数样本），出 hop 个样本。"""
        import numpy as np
        n_fft, hop = self.n_fft, self.hop
        f = np.asarray(frame, dtype=np.float64)
        assert f.shape == (n_fft,), f"frame 长度须为 {n_fft}"
        self.buf[:n_fft] += self.w * f
        self.wn[:n_fft] += self.w ** 2
        out = self.buf[:hop] / (self.wn[:hop] + self.eps)
        # 窗左移 hop：位置 j 的新内容 = 旧位置 j+hop
        self.buf[:n_fft] = self.buf[hop:hop + n_fft]
        self.buf[n_fft:] = 0.0
        self.wn[:n_fft] = self.wn[hop:hop + n_fft]
        self.wn[n_fft:] = 0.0
        return out
