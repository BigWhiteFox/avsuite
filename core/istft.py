"""ISTFT：overlap-add 重构（WP-01 / 模块1）。

WOLA（加权重叠相加）公式：
    y[n] = Σ_t w[n - t·hop] · frame_t[n - t·hop] / Σ_t w²[n - t·hop]

要点：
- 分母是"窗平方"的 overlap-add。分析帧 = w·x，加窗合成 Σ w·frame = (Σw²)·x，
  除以窗平方和即精确还原——因此不依赖 COLA 条件（COLA 是"合成时不加归一化"
  才需要的条件，见 core/README.md 的辨析）；
- **可重构的精确条件**（deep_audit G6 实测更正：原文档"任意 hop<=n_fft 均可无损"
  不成立）：`hop < n_fft`，或窗端点在被覆盖区内非零（即每个样本的 Σw² > 0）。
  `hop == n_fft` 时帧间无重叠，每个样本只被**一个**窗覆盖，端点 w 直接决定可逆性：
  hann（w[0]=0）与 blackman（periodic 的 w[0] 数学上为 0；float32 下为 -2.98e-8）
  在端点处 Σw²=0 → 分母置零、样本丢失，实测 512/512 往返 maxerr = **1.176**
  （deep_audit G6 值；maxerr 恰等于被置零的端点样本的幅度、随信号变化——本仓另用
  440Hz 音+噪声复现得 1.22，机制与量级不变，恒为 ~1）；而 hamming（端点 0.08
  非零）同条件 maxerr = 5.7e-15，可精确重构。故 hop == n_fft 且窗为
  hann/blackman 时，ISTFT 构造即发一次性 warning
  （不 raise，见 ISTFT.__init__ 与 _warn_hop_eq_nfft_once）；
- 分母趋零除上述结构性不可逆区外只发生在信号两端，用 eps 下限 + where 置零处理；
- center=True 时与 torch.istft 一致地裁掉两侧各 n_fft//2 的 pad 区；
- 对拍 torch.istft：float64 全量 <1e-10，float32 <1e-5（tests/test_istft.py）。
"""

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from .stft import make_window

__all__ = ["ISTFT", "StreamingISTFT"]

# hop == n_fft 且窗端点为 0 时的"一次性"告警账本（key = (window, n_fft)）：
# 同一进程内同配置只告警一次，避免批量构造刷屏。
_WARNED_HOP_EQ_NFFT: set = set()

# 端点值为 0（数学意义上）的窗：hop == n_fft 时端点样本不可逆
_ZERO_ENDPOINT_WINDOWS = ("hann", "blackman")


def _warn_hop_eq_nfft_once(window: str, n_fft: int) -> None:
    """hop == n_fft + hann/blackman 的一次性告警（不 raise）。

    触发条件（deep_audit G6 实测）：无重叠帧时端点样本的 Σw² = 0，会被静默置零，
    实测往返 maxerr = 1.176（信号量级 ~1）。只告警不抛错——既有调用方仍可用，
    是否改用 hop < n_fft 或 hamming 由调用方决定。
    """
    key = (window, n_fft)
    if key in _WARNED_HOP_EQ_NFFT:
        return
    _WARNED_HOP_EQ_NFFT.add(key)
    warnings.warn(
        f"ISTFT(hop={n_fft}) 即 hop == n_fft，配合 window={window!r} 无法精确重构："
        f"无重叠帧时窗端点样本的 Σw² = 0（{window} 端点为 0），这些样本会被静默置零"
        f"（实测往返 maxerr ≈ 1.18，信号量级 ~1）。请改用 hop < n_fft"
        f"（推荐 50% 重叠 hop = n_fft//2），或换端点非零的窗（如 'hamming'）。",
        UserWarning,
        stacklevel=3,
    )


class ISTFT(nn.Module):
    """逆 STFT（nn.Module，window 随模型 .to(device)）。

    Args:
        n_fft / hop / window: 必须与分析端 STFT 严格一致。
        center: 对齐 torch.istft 默认（裁掉两侧 pad）。
        eps:    窗平方和下限，低于它的时间点输出 0（仅出现在信号两端，
                以及 hop==n_fft + hann/blackman 的结构性不可逆端点）。

    可重构性：`hop < n_fft` 时任意窗都精确重构（WOLA 窗平方归一）；
    `hop == n_fft` 时仅端点非零的窗（如 hamming）可重构——hann/blackman
    端点为 0，构造时会发一次性 UserWarning（不 raise，见 G6 说明）。
    """

    def __init__(self, n_fft: int = 512, hop: int = 256, window: str = "hann",
                 center: bool = True, eps: float = 1e-11):
        super().__init__()
        if not 0 < hop <= n_fft:
            raise ValueError(f"hop must be in (0, {n_fft}], got {hop}")
        if hop == n_fft and window in _ZERO_ENDPOINT_WINDOWS:
            _warn_hop_eq_nfft_once(window, n_fft)
        self.n_fft = n_fft
        self.hop = hop
        self.center = center
        self.eps = eps
        self.register_buffer("window", make_window(n_fft, window), persistent=False)

    def inverse(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """复数谱 [B, F, T] -> 时域 [B, L]。

        length: 目标点数，**必须 >= 0 或 None**。
            None   —— 与 torch.istft 一致（center=True 输出 (T-1)*hop 点）；
            >= 0   —— 给定值时截断 / 右侧零补到该长度（length=0 返回空张量）；
            < 0    —— 无意义（负长度不是"从尾部截取"），直接 raise ValueError。
                      修正前 -1 会被 `y[..., :length]` 当成负索引**静默截掉尾部
                      1 个样本**（实测 1536→1535），调用方把长度算错成负值时
                      会静默丢样点而非报错（test_gap §3.7 #1b）。
        """
        if spec.dim() != 3:
            raise ValueError(f"expected [B, F, T], got {tuple(spec.shape)}")
        if length is not None and length < 0:
            raise ValueError(f"length must be >= 0 or None, got {length}")
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
    （下一窗起点 (t+1)·hop 已越过该段），WOLA 分母此刻收齐。与离线是**相对
    eps/Σw² 量级的一致**、并非逐位相同：add_frame 用 `buf/(Σw²+eps)`，离线用
    `where(Σw²>eps)` + `clamp_min`。hop=320 时主体区 Σw² 最低约 0.19，相对偏差
    ≈ 1e-11/0.19 ≈ 5.2e-11（|x|≈3.2 处绝对 ≈1.7e-10，实测）。parity 单测
    （core 级 + exp/07_nn_aec 模型级）即按该容差断言。

    尾部结算（deep_audit G5）：T 帧后流式共出 T·hop 个样本，而离线全网格长度是
    (T-1)·hop + n_fft ⇒ 尾部 n_fft - hop 个样本仍留在缓冲里，须调用 flush() 取回
    （不调用 flush 的既有路径行为逐位不变）。
    """

    # np 只出现在字符串注解里作类型提示（numpy 在函数内延迟导入、模块顶层不引入），运行时不会求值
    def __init__(self, n_fft: int = 512, hop: int = 320, window: "np.ndarray|None" = None,  # noqa: F821
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
        self._n_frames = 0        # 已入帧数（flush 的尾部语义需要）
        self._closed = False      # flush() 之后流关闭

    # 同上：字符串注解不做运行时求值，np 未在模块顶层导入属有意的延迟导入设计
    def add_frame(self, frame: "np.ndarray") -> "np.ndarray":  # noqa: F821
        """入一帧（n_fft 实数样本），出 hop 个样本。

        flush() 之后再调用会抛 RuntimeError（流已关闭）。
        """
        import numpy as np
        if self._closed:
            raise RuntimeError("flush() 之后流已关闭")
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
        self._n_frames += 1
        return out

    # 同上：字符串注解不做运行时求值
    def flush(self) -> "np.ndarray":  # noqa: F821
        """结算尾帧：返回全网格尾部残留的 n_fft - hop 个样本，之后流关闭。

        触发条件（deep_audit G5，实测）：add_frame 每帧只发射 hop 个样本，而离线
        全网格长度是 (T-1)·hop + n_fft ⇒ 尾部 n_fft - hop 个样本在 T 帧后仍留在
        buf 里，不调用本方法就永远拿不到。实测 n_fft=512/hop=320/T=20：离线
        6592 样本，流式只出 6400 → 尾部 192 样本（12ms）被静默丢弃。

        归一化用**离线 ISTFT.inverse 的口径**（`where(Σw² > eps)` 才相除，否则置 0），
        而不是 add_frame 的 `buf/(Σw²+eps)`：尾部落在"窗覆盖不全"区、Σw² 最小可到
        ~1.4e-9（窗端点 w[511]=3.765e-5 的平方），eps=1e-11 进分母会带来最高
        ~0.7% 的相对偏差（实测 eps/(min Σw²+eps) = 0.7005%）。用离线口径后
        `concat(add_frame…) + flush()` 的长度恰为 (T-1)·hop + n_fft；**尾段**
        （本方法返回的 n_fft-hop 个样本）与离线全网格实测 max|diff| ≈ 2.3e-24，
        而**已发射前缀**仍带 add_frame 分母口径造成的 ~eps/Σw² 相对差（实测 T=20：
        全网格 max|diff| = 1.677e-10，落在 Σw²≈0.19、|x|≈3.2 处，恰等于
        |x|·eps/Σw² = 1.677e-10）。故整体是"长度精确 + 尾段精确 + 前缀既有
        5e-11 级相对口径差"，**并非逐位相同**。

        返回: float64 数组，长度 n_fft - hop
              （hop >= n_fft 时为 0；未入任何帧时为空数组）。
        重复调用抛 RuntimeError；不调用本方法的既有路径行为逐位不变。
        """
        import numpy as np
        if self._closed:
            raise RuntimeError("flush() 只能调用一次")
        self._closed = True
        n_tail = self.n_fft - self.hop
        if self._n_frames == 0 or n_tail <= 0:
            return np.zeros(0, dtype=np.float64)
        den = self.wn[:n_tail]
        return np.where(den > self.eps, self.buf[:n_tail] / np.maximum(den, self.eps), 0.0)
