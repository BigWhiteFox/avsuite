"""WP-01 单测：自实现 STFT vs torch.stft 对拍 + 流式接口逐帧对齐。"""
import pytest
import torch

from core.stft import STFT, StreamingSTFT, spec_from_2ch, spec_to_2ch

CONFIGS = [
    dict(n_fft=512, hop=256, window="hann", center=True, N=16000),
    dict(n_fft=512, hop=128, window="hann", center=True, N=16001),
    dict(n_fft=1024, hop=256, window="hamming", center=True, N=8000),
    dict(n_fft=512, hop=256, window="blackman", center=False, N=16000),
    dict(n_fft=256, hop=64, window="hann", center=True, N=4096),
]


def _torch_stft(wav, cfg):
    w = torch.hann_window if cfg["window"] == "hann" else (
        torch.hamming_window if cfg["window"] == "hamming" else torch.blackman_window)
    kw = dict(hop_length=cfg["hop"], win_length=cfg["n_fft"], center=cfg["center"],
              onesided=True, return_complex=True)
    if cfg["center"]:
        kw["pad_mode"] = "reflect"
    return torch.stft(wav, cfg["n_fft"], window=w(cfg["n_fft"], periodic=True,
                                                 dtype=wav.dtype), **kw)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_matches_torch_stft_f64(cfg):
    """float64 对拍：同一路 FFT/加窗，误差应 <1e-10（满足 WORKLIST 的 <1e-5 要求）。"""
    torch.manual_seed(0)
    wav = torch.randn(2, cfg["N"], dtype=torch.float64)
    ours = STFT(cfg["n_fft"], cfg["hop"], cfg["window"], cfg["center"]).transform(wav)
    ref = _torch_stft(wav, cfg)
    assert ours.shape == ref.shape
    torch.testing.assert_close(ours, ref, rtol=1e-7, atol=1e-10)


def test_matches_torch_stft_f32():
    """float32 对拍：有效精度 1e-6 量级，放 rtol=1e-5/atol=1e-4（相对谱峰 1e-6）。"""
    torch.manual_seed(0)
    wav = torch.randn(2, 16000, dtype=torch.float32)
    ours = STFT().transform(wav)
    ref = _torch_stft(wav, CONFIGS[0])
    torch.testing.assert_close(ours, ref, rtol=1e-5, atol=1e-4)


def test_shape_and_layout():
    """[B,F,T] 布局 + center=True 时 T = N//hop + 1 + F = n_fft//2+1；一维输入自动升维。"""
    torch.manual_seed(0)
    wav = torch.randn(16000)
    spec = STFT().transform(wav)
    assert spec.shape == (1, 257, 63)          # F=257, T=16000//256+1
    assert spec.is_complex()
    assert spec.dtype == torch.complex64


def test_2ch_view_roundtrip():
    """Re/Im 双通道视图 <-> 复数谱 往返无损（网络输入布局约定）。"""
    torch.manual_seed(0)
    spec = STFT().transform(torch.randn(2, 8000))
    assert spec_to_2ch(spec).shape == (2, 257, 32, 2)
    torch.testing.assert_close(spec_from_2ch(spec_to_2ch(spec)), spec)


def test_streaming_matches_offline():
    """流式（任意 chunk 大小）与离线 zeros-pad 模式逐帧一致，flush 结算尾帧。"""
    torch.manual_seed(1)
    wav = torch.randn(2, 16000, dtype=torch.float64)
    stft = STFT(n_fft=512, hop=256, center=True, pad_mode="zeros")
    offline = stft.transform(wav)

    for chunk in (320, 333, 7, 1000):          # 320=20ms@16k；333 制造不整除；7 压测缓冲
        s = StreamingSTFT(stft)
        frames = []
        for i in range(0, wav.shape[-1], chunk):
            frames += s.push(wav[:, i:i + chunk])
            assert s.cache_size() <= stft.n_fft + chunk   # cache 与信号总长无关
        frames += s.flush()
        ours = torch.stack(frames, dim=-1)
        assert ours.shape == offline.shape
        torch.testing.assert_close(ours, offline, rtol=1e-7, atol=1e-10)


def test_streaming_center_false_and_f32():
    """center=False 流式（无注入 pad，flush 不补帧）+ float32 路径。"""
    torch.manual_seed(2)
    wav = torch.randn(1, 8000, dtype=torch.float32)
    stft = STFT(n_fft=512, hop=256, center=False, pad_mode="zeros")
    offline = stft.transform(wav)
    s = StreamingSTFT(stft)
    frames = []
    for i in range(0, 8000, 320):
        frames += s.push(wav[:, i:i + 320])
    frames += s.flush()
    ours = torch.stack(frames, dim=-1)
    assert ours.shape == offline.shape
    torch.testing.assert_close(ours, offline, rtol=1e-5, atol=1e-4)


def test_streaming_flush_once_and_push_after_close():
    stft = STFT()
    s = StreamingSTFT(stft)
    s.push(torch.zeros(320))
    s.flush()
    with pytest.raises(RuntimeError):
        s.push(torch.zeros(320))
    with pytest.raises(RuntimeError):
        s.flush()
