"""STFT 工具类（WP-01 / 模块1）。

与 torch.stft 的约定逐项对齐（tests/test_stft.py 对拍 <1e-5）：
- 输出单边复数谱 [B, F, T]，F = n_fft//2 + 1；另提供 Re/Im 双通道视图 [B, F, T, 2]；
- center=True 时两侧各 pad n_fft//2，pad_mode 默认 'reflect'（同 torch.stft）；
  流式接口无法预知未来样本，约定 pad_mode='zeros'，对拍时离线用同模式；
- 周期窗（periodic=True），hann 50% 重叠满足 COLA；ISTFT 侧用窗平方归一（WOLA），
  任意满足 hop<=n_fft 的窗均可精确重构（见 core/istft.py 说明）。

流式接口 StreamingSTFT：任意大小 chunk 推入、按 hop 产出已完成帧；
cache 内存固定 O(B·n_fft) 与信号长度无关；flush() 以零右填充结算尾帧。
"""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["make_window", "STFT", "StreamingSTFT", "spec_to_2ch", "spec_from_2ch"]


def make_window(n_fft: int, window: str = "hann", dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """周期窗，periodic=True 与 torch.stft 默认一致（首尾不平滑会导致帧间跳变）。

    默认以 float64 计算窗系数（消除 float32 舍入带来的对拍偏差），
    使用端再 cast 到输入精度（见 STFT.transform）。
    """
    if window == "hann":
        return torch.hann_window(n_fft, periodic=True, dtype=dtype)
    if window == "hamming":
        return torch.hamming_window(n_fft, periodic=True, dtype=dtype)
    if window == "blackman":
        return torch.blackman_window(n_fft, periodic=True, dtype=dtype)
    raise ValueError(f"unsupported window: {window!r}")


def spec_to_2ch(spec: torch.Tensor) -> torch.Tensor:
    """复数谱 [B, F, T] -> Re/Im 双通道 [B, F, T, 2]（网络输入常用布局）。"""
    return torch.stack([spec.real, spec.imag], dim=-1)


def spec_from_2ch(spec2: torch.Tensor) -> torch.Tensor:
    """Re/Im 双通道 [B, F, T, 2] -> 复数谱 [B, F, T]。"""
    return torch.complex(spec2[..., 0], spec2[..., 1])


class STFT(nn.Module):
    """短时傅里叶变换（nn.Module，window 随模型 .to(device)）。

    等价于：
        torch.stft(wav, n_fft, hop_length=hop, window=w, center=center,
                   pad_mode='reflect', onesided=True, return_complex=True)

    Args:
        n_fft:    FFT 点数，16k 语音默认 512（32ms）。
        hop:      帧移，默认 256（16ms，50% 重叠）。
        window:   'hann' | 'hamming' | 'blackman'。
        center:   两侧 pad n_fft//2（对齐 torch.stft 默认）。
        pad_mode: 'reflect'（离线默认）| 'zeros'（流式对拍约定）。
    """

    def __init__(self, n_fft: int = 512, hop: int = 256, window: str = "hann",
                 center: bool = True, pad_mode: str = "reflect"):
        super().__init__()
        if not 0 < hop <= n_fft:
            raise ValueError(f"hop must be in (0, {n_fft}], got {hop}")
        self.n_fft = n_fft
        self.hop = hop
        self.center = center
        self.pad_mode = pad_mode
        self.register_buffer("window", make_window(n_fft, window), persistent=False)

    def transform(self, wav: torch.Tensor) -> torch.Tensor:
        """[B, N]（或 [N]）-> 复数谱 [B, F, T]。"""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2:
            raise ValueError(f"expected [B, N] or [N], got {tuple(wav.shape)}")
        if self.center:
            pad = (self.n_fft // 2,) * 2
            if self.pad_mode == "reflect":
                if wav.shape[-1] <= self.n_fft // 2:
                    raise ValueError("reflect pad 要求 N > n_fft//2；短信号请用 pad_mode='zeros'")
                wav = F.pad(wav.unsqueeze(1), pad, mode="reflect").squeeze(1)
            else:  # 'zeros'（流式对拍约定）——F.pad 的零填充名为 constant
                wav = F.pad(wav.unsqueeze(1), pad).squeeze(1)
        w = self.window.to(wav.dtype)
        frames = wav.unfold(-1, self.n_fft, self.hop)        # [B, T, n_fft]
        spec = torch.fft.rfft(frames * w, dim=-1)            # [B, T, F]
        return spec.transpose(1, 2)                          # [B, F, T]


class StreamingSTFT:
    """流式 STFT：逐 chunk 推入样本、按 hop 产出帧，与离线逐帧一致。

    对拍约定：离线 STFT(center=..., pad_mode='zeros') 的第 t 帧 与 流式第 t 帧
    逐元素一致（tests/test_stft.py::test_streaming_matches_offline）。

    时序约定（center=True，16k 下 n_fft=512/hop=256）：
    - 内部先注入 n_fft//2 个零样本（等价离线左 pad；离线 reflect 模式仅最左帧
      边缘样本不同，这是流式的固有差异，故对拍统一用 zeros 模式）；
    - 第 t 帧在缓冲长度达到 t*hop + n_fft 时产出（帧右端样本已到齐）；
    - flush() 右填充 n_fft//2 个零（等价离线右 pad），结算全部尾帧后流关闭。

    内存：cache 为缓冲样本 [B, <= n_fft-1+chunk]，与信号总长无关（cache_size()）。
    算法时延：帧长 n_fft（32ms）；其中"超前读取"lookahead = n_fft - hop（16ms），
    即帧右端对应的是 16ms 之前的音频——20ms 帧移的车载链路按此核算端到端时延。
    """

    def __init__(self, stft: STFT):
        self.stft = stft
        self._buf: list[torch.Tensor] = []
        self._len = 0          # 缓冲样本数（含注入的左 pad 零）
        self._next_start = 0   # 下一帧在缓冲坐标系中的起点
        self._primed = False
        self._closed = False

    @property
    def _pad(self) -> int:
        return self.stft.n_fft // 2 if self.stft.center else 0

    def _append(self, x: torch.Tensor) -> None:
        self._buf.append(x)
        self._len += x.shape[-1]

    def _cat(self) -> torch.Tensor:
        return self._buf[0] if len(self._buf) == 1 else torch.cat(self._buf, dim=-1)

    def _drain(self) -> list[torch.Tensor]:
        out = []
        n_fft = self.stft.n_fft
        buf = self._cat()
        w = self.stft.window.to(buf.dtype)
        while self._len - self._next_start >= n_fft:
            frame = buf[..., self._next_start:self._next_start + n_fft]
            out.append(torch.fft.rfft(frame * w, dim=-1))    # [B, F]
            self._next_start += self.stft.hop
        if self._next_start > 0:                             # 丢弃历史，cache 有界
            buf = buf[..., self._next_start:]
            self._buf, self._len, self._next_start = [buf], buf.shape[-1], 0
        return out

    def push(self, chunk: torch.Tensor) -> list[torch.Tensor]:
        """推入一段样本 [B, C]（或 [C]），返回本次新产出的复数帧列表（每个 [B, F]）。"""
        if self._closed:
            raise RuntimeError("flush() 之后流已关闭")
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        if not self._primed:
            if self._pad:
                self._append(chunk.new_zeros(chunk.shape[0], self._pad))
            self._primed = True
        self._append(chunk)
        return self._drain()

    def flush(self) -> list[torch.Tensor]:
        """右填充 n_fft//2 个零结算尾帧（对应离线右 pad），之后流关闭。"""
        if self._closed:
            raise RuntimeError("flush() 只能调用一次")
        self._closed = True
        if self._pad:
            buf = self._cat()
            self._append(buf.new_zeros(buf.shape[0], self._pad))
        return self._drain()

    def cache_size(self) -> int:
        """当前缓冲样本数（流式 cache 占用，含注入的左 pad 零）。"""
        return self._len
