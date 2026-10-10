"""NN-AEC 流式推理（WP-05b 模式）：20ms 块进块出，conv cache + LSTM state。

架构前提（WP-07 设计时埋好）：时间维严格因果——conv 左 pad 4、deconv 右移 2、
LSTM 单向。流式只需：
- conv 层：cache 最近 5 帧输入，每新帧算 1 帧输出（感受野 12 帧 ≈ 240ms）；
- LSTM：状态外置逐帧推进；
- ISTFT：StreamingISTFT 窗口滑过 hop=320，帧到即出块。

与离线逐帧对拍 <1e-5 由 tests/test_streaming_nnaec.py 锁死（帧来源同为
StreamingSTFT 的 zero-pad 约定；离线 reflect 左边缘差异见 core 文档）。

内存账（cache 总量，float32）：
  5 帧 × (4·257 + 8·129 + 16·65 + 64·33 + 48·65 + 24·129 + 8·257) ≈ 67K floats
  ≈ 270 KB + LSTM state(2×96) ≈ 1 KB——与音频总长无关（有界 cache 纪律）。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.istft import StreamingISTFT  # noqa: E402
from core.stft import STFT, StreamingSTFT  # noqa: E402

# model 模块名与 exp/03、exp/04 冲突（sys.modules 先到先得，pytest 全量收集时
# 可能是 CRN/DCCRN 的 model）——按文件路径直载本目录的 model，规避冲突。
_SPEC = __import__("importlib.util", fromlist=["spec_from_file_location"]).spec_from_file_location(
    "nnaec_stream_model", Path(__file__).resolve().parent / "model.py")
_MODEL = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODEL)
NNAEC = _MODEL.NNAEC

__all__ = ["NNAECStreaming"]


class _ConvStream:
    """因果卷积逐帧：cache 最近 5 帧输入。

    注意绕过 CausalConvBlock.forward（它内部会再左 pad 一次——离线路径的职责），
    流式直接调 conv/bn/act 三件套，padding 由本类完成。
    """

    def __init__(self, block: torch.nn.Module, cin: int, fin: int):
        self.block = block
        self.cache = torch.zeros(1, cin, fin, 5)

    def step(self, x: torch.Tensor):        # x: [1, C, F]
        self.cache = torch.cat([self.cache[..., 1:], x[..., None]], dim=-1)
        w = torch.nn.functional.pad(self.cache, (4, 0, 0, 0))   # 左 pad 4（同离线）
        y = self.block.act(self.block.bn(self.block.conv(w)))
        return y[..., -1]                                       # [1, C_out, F_out]

    def n_cached(self) -> int:
        return self.cache.numel()


class _DeconvStream:
    """因果转置卷积逐帧：cache 最近 5 帧输入，输出 6 帧取第 2 帧（右移 2 等价）。

    离线语义：out_t = convT_full[t-2]（t<2 时为零——右移的启动延迟）。流式需
    显式复刻：前 2 帧直接输出零（cache 里是零+首帧，直接取第 2 帧会算出非零）。
    """

    def __init__(self, block: torch.nn.Module, cin: int, fin: int):
        self.block = block
        self.cache = torch.zeros(1, cin, fin, 5)
        self.t = 0

    def step(self, x: torch.Tensor):
        self.cache = torch.cat([self.cache[..., 1:], x[..., None]], dim=-1)
        if self.t < 2:                       # 离线右移启动延迟：前 2 帧输出零
            self.t += 1
            fin = (self.cache.shape[2] - 1) * 2 + 1
            cout = self.block.conv.weight.shape[1]   # ConvTranspose2d 权重 (cin, cout, k, k)
            return self.cache.new_zeros(1, cout, fin)
        self.t += 1
        y = self.block.act(self.block.bn(self.block.conv(self.cache)))
        return y[..., 2]

    def n_cached(self) -> int:
        return self.cache.numel()


class NNAECStreaming:
    """20ms 块流式 NN-AEC。

    process_chunk(mic_chunk, far_chunk) -> enhanced 同长块（int16 可写）。
    块长约定 320 样本（模型 hop）；任意长度推入时内部按帧结算（首尾对齐由
    StreamingSTFT 保证，与离线一致）。
    """

    def __init__(self, model: NNAEC, device="cpu"):
        model = model.to(device).eval()
        self.model = model
        self.device = device
        stft = STFT(model.n_fft, model.hop, pad_mode="zeros")
        self.s_mic = StreamingSTFT(stft)
        self.s_far = StreamingSTFT(stft)
        w = np.asarray(stft.window.to(torch.float64).numpy())
        self.istft = StreamingISTFT(model.n_fft, model.hop, window=w)

        # 逐层流缓存（通道/频点数与离线各层输入一致）
        self.enc = [
            _ConvStream(model.enc[0], 4, 257),
            _ConvStream(model.enc[1], 8, 129),
            _ConvStream(model.enc[2], 16, 65),
        ]
        self.dec = [
            _DeconvStream(model.dec[0], 64, 33),
            _DeconvStream(model.dec[1], 48, 65),
            _DeconvStream(model.dec[2], 24, 129),
        ]
        h = model.lstm.hidden_size
        self.h = torch.zeros(1, 1, h, device=device)
        self.c = torch.zeros(1, 1, h, device=device)

    def cache_size_bytes(self) -> int:
        return 4 * (sum(e.n_cached() for e in self.enc) +
                    sum(d.n_cached() for d in self.dec) +
                    2 * self.h.numel())

    def _frame_step(self, mf: torch.Tensor, ff: torch.Tensor) -> torch.Tensor:
        """一帧：复数谱 -> 增强复谱。mf/ff: [F] complex，在 device 上。"""
        with torch.no_grad():
            x = torch.cat([torch.stack([mf.real, mf.imag]),
                           torch.stack([ff.real, ff.imag])], dim=0).unsqueeze(0)  # [1,4,F]
            e1 = self.enc[0].step(x)
            e2 = self.enc[1].step(e1)
            e3 = self.enc[2].step(e2)
            z = e3.reshape(1, 1, -1)
            z, (self.h, self.c) = self.model.lstm(z, (self.h, self.c))
            x = self.model.fc(z).reshape(1, e3.shape[1], e3.shape[2])
            d0 = self.dec[0].step(torch.cat([x, e3], dim=1))
            d1 = self.dec[1].step(torch.cat([d0, e2], dim=1))
            d2 = self.dec[2].step(torch.cat([d1, e1], dim=1))
            mask = self.model.head(d2.unsqueeze(-1))[..., 0]     # [1,2,F]
            m = torch.complex(mask[0, 0], mask[0, 1])
            return m * mf

    def process_chunk(self, mic_chunk: np.ndarray, far_chunk: np.ndarray) -> np.ndarray:
        assert len(mic_chunk) == len(far_chunk), "mic/far 块须等长"
        mic_frames = self.s_mic.push(torch.from_numpy(mic_chunk.astype(np.float32)))
        far_frames = self.s_far.push(torch.from_numpy(far_chunk.astype(np.float32)))
        outs = []
        for mf, ff in zip(mic_frames, far_frames):
            mf = mf.to(self.device)
            ff = ff.to(self.device)
            enh = self._frame_step(mf[0], ff[0])                # [F] complex
            frame_td = torch.fft.irfft(enh, n=self.model.n_fft)
            outs.append(self.istft.add_frame(frame_td.cpu().numpy()))
        if not outs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outs)

    def flush(self) -> np.ndarray:
        """结算尾帧（右零 pad，等价离线 center 右边缘），返回剩余增强样本。"""
        mf = self.s_mic.flush()
        ff = self.s_far.flush()
        outs = []
        for m, f in zip(mf, ff):
            enh = self._frame_step(m[0].to(self.device), f[0].to(self.device))
            frame_td = torch.fft.irfft(enh, n=self.model.n_fft)
            outs.append(self.istft.add_frame(frame_td.cpu().numpy()))
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    def reset(self):
        for e in self.enc:
            e.cache.zero_()
        for d in self.dec:
            d.cache.zero_()
            d.t = 0
        self.h.zero_()
        self.c.zero_()
        self.s_mic = StreamingSTFT(STFT(self.model.n_fft, self.model.hop, pad_mode="zeros"))
        self.s_far = StreamingSTFT(STFT(self.model.n_fft, self.model.hop, pad_mode="zeros"))
        w = np.asarray(self.model.stft.window.to(torch.float64).numpy())
        self.istft = StreamingISTFT(self.model.n_fft, self.model.hop, window=w)
