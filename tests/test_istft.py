# -*- coding: utf-8 -*-
"""WP-01 单测：ISTFT 往返重构 + vs torch.istft 对拍。"""
import pytest
import torch

from core.istft import ISTFT
from core.stft import STFT

CONFIGS = [
    dict(n_fft=512, hop=256, window="hann", center=True, N=16000),
    dict(n_fft=512, hop=128, window="hamming", center=True, N=16000),
    dict(n_fft=1024, hop=256, window="blackman", center=True, N=16000),
    # center=False + 端点非零窗：hann 端点为 0，torch.istft 会直接 raise（重叠归一分母为 0），
    # 我们的处理是置零；hamming 端点非零，两边都能算，才可对拍
    dict(n_fft=256, hop=64, window="hamming", center=False, N=16000),
]


def _roundtrip(wav, cfg):
    stft = STFT(cfg["n_fft"], cfg["hop"], cfg["window"], cfg["center"])
    istft = ISTFT(cfg["n_fft"], cfg["hop"], cfg["window"], cfg["center"])
    return istft.inverse(stft.transform(wav))


def test_roundtrip_exact_f64():
    """WOLA 归一化下往返精确重构（不依赖 COLA，blackman 非严格 COLA 一样过）。"""
    torch.manual_seed(0)
    for cfg in CONFIGS:
        wav = torch.randn(2, cfg["N"], dtype=torch.float64)
        rec = _roundtrip(wav, cfg)
        # center=True: 输出 (T-1)*hop 点（裁掉两侧 pad），与 wav 前 L 点逐元素一致
        # center=False: 首尾窗端点处窗平方和趋零，属结构性不可重构区，比较内部
        if cfg["center"]:
            assert rec.shape[-1] == (wav.shape[-1] // cfg["hop"]) * cfg["hop"]
            torch.testing.assert_close(rec, wav[..., :rec.shape[-1]], rtol=1e-7, atol=1e-10)
        else:
            inner = rec[..., cfg["n_fft"]:-cfg["n_fft"]]
            torch.testing.assert_close(inner, wav[..., cfg["n_fft"]:wav.shape[-1] - cfg["n_fft"]],
                                       rtol=1e-7, atol=1e-10)


def test_roundtrip_f32():
    torch.manual_seed(1)
    wav = torch.randn(2, 16000, dtype=torch.float32)
    rec = _roundtrip(wav, CONFIGS[0])
    torch.testing.assert_close(rec, wav[..., :rec.shape[-1]], rtol=1e-5, atol=1e-6)


def _torch_istft(spec, cfg):
    w = torch.hann_window if cfg["window"] == "hann" else (
        torch.hamming_window if cfg["window"] == "hamming" else torch.blackman_window)
    return torch.istft(spec, cfg["n_fft"], hop_length=cfg["hop"],
                       win_length=cfg["n_fft"], center=cfg["center"],
                       window=w(cfg["n_fft"], periodic=True, dtype=spec.real.dtype))


@pytest.mark.parametrize("cfg", CONFIGS)
def test_matches_torch_istft(cfg):
    """与 torch.istft 全量对拍（hamming 端点非零，center=False 也无边界退化）。"""
    torch.manual_seed(3)
    wav = torch.randn(2, cfg["N"], dtype=torch.float64)
    spec = STFT(cfg["n_fft"], cfg["hop"], cfg["window"], cfg["center"]).transform(wav)
    ours = ISTFT(cfg["n_fft"], cfg["hop"], cfg["window"], cfg["center"]).inverse(spec)
    ref = _torch_istft(spec, cfg)
    torch.testing.assert_close(ours, ref, rtol=1e-7, atol=1e-10)


def test_length_argument():
    """length=8000 截断一致；length 超长右侧零补。"""
    torch.manual_seed(4)
    wav = torch.randn(2, 16000, dtype=torch.float64)
    spec = STFT().transform(wav)
    ours = ISTFT().inverse(spec, length=8000)
    ref = torch.istft(spec, 512, hop_length=256, window=torch.hann_window(512, periodic=True,
                                                                          dtype=torch.float64))
    torch.testing.assert_close(ours, ref[..., :8000], rtol=1e-7, atol=1e-10)
    long_ = ISTFT().inverse(spec, length=20000)
    assert long_.shape[-1] == 20000
    torch.testing.assert_close(long_[..., 15872:], torch.zeros(2, 4128, dtype=torch.float64))


def test_gradient_through_roundtrip():
    """spec requires_grad -> ISTFT 可反传（irfft/fold 均可微，为谱域损失铺路）。"""
    torch.manual_seed(5)
    spec = STFT().transform(torch.randn(1, 4000)).requires_grad_(True)
    loss = ISTFT().inverse(spec).pow(2).mean()
    loss.backward()
    assert spec.grad is not None
    assert torch.isfinite(spec.grad).all()
