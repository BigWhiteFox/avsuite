# -*- coding: utf-8 -*-
"""WP-05b 单测：流式 vs 非流式逐帧对齐（<1e-5）+ cache 有界性。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nnaec_shim as X  # noqa: E402

NNAEC = X.NNAEC


def _load_model():
    m = NNAEC().eval()
    return m


def _offline_ref(model, mic, far):
    """离线参考：zeros-pad STFT（与 StreamingSTFT 同约定）-> forward_spec -> ISTFT。"""
    from core.istft import ISTFT
    from core.stft import STFT
    stft = STFT(512, 320, pad_mode="zeros")
    ms = stft.transform(torch.from_numpy(mic.astype(np.float32)).unsqueeze(0))
    fs = stft.transform(torch.from_numpy(far.astype(np.float32)).unsqueeze(0))
    with torch.no_grad():
        mask, enh = model.forward_spec(ms, fs)
    return ISTFT(512, 320).inverse(enh, length=None)[0].numpy()


def _run_streaming(model, mic, far, chunk=320):
    from streaming import NNAECStreaming
    st = NNAECStreaming(model, device="cpu")
    outs = []
    for i in range(0, len(mic), chunk):
        outs.append(st.process_chunk(mic[i:i + chunk], far[i:i + chunk]))
    outs.append(st.flush())
    return np.concatenate(outs)


def _align(stream_out, ref):
    """流式输出 out[j] = 离线 y_full[j]（帧网格从 0 起，pad 区被窗自然乘零）；
    离线 center 裁剪 = y_full[256 : 256+(T-1)·hop]——对齐取同区间。"""
    assert len(stream_out) >= 256 + len(ref), (len(stream_out), len(ref))
    return stream_out[256: 256 + len(ref)]


def test_streaming_matches_offline_320chunks():
    """320 定块：流式输出 [256:256+(T-1)hop] 与离线一致（<1e-5，实测应 ~1e-10）。"""
    rng = np.random.default_rng(0)
    n = 16000 * 3                                   # 3s -> T=151 帧
    mic = (0.3 * rng.standard_normal(n)).astype(np.float32)
    far = (0.3 * rng.standard_normal(n)).astype(np.float32)
    model = _load_model()
    ref = _offline_ref(model, mic, far)            # 长度 (T-1)*hop = 48000
    out = _run_streaming(model, mic, far, 320)
    got = _align(out, ref)
    assert got.shape == ref.shape
    err = np.abs(got - ref).max()
    assert err < 1e-5, f"逐帧对齐误差 {err:.2e} >= 1e-5"


def test_streaming_matches_offline_irregular_chunks():
    """不规则块（800/640/320 混合）：块边界不影响逐帧等价性。"""
    rng = np.random.default_rng(1)
    n = 16000 * 3
    mic = (0.3 * rng.standard_normal(n)).astype(np.float32)
    far = (0.3 * rng.standard_normal(n)).astype(np.float32)
    model = _load_model()
    ref = _offline_ref(model, mic, far)
    from streaming import NNAECStreaming
    st = NNAECStreaming(model, device="cpu")
    outs = []
    i = 0
    pattern = [800, 640, 320, 960, 320]
    while i < n:
        c = pattern[len(outs) % len(pattern)]
        outs.append(st.process_chunk(mic[i:i + c], far[i:i + c]))
        i += c
    outs.append(st.flush())
    out = np.concatenate(outs)
    got = _align(out, ref)
    err = np.abs(got - ref).max()
    assert err < 1e-5, f"不规则块对齐误差 {err:.2e} >= 1e-5"


def test_cache_bounded_and_accounted():
    """cache 内存与信号时长无关；账本 ≈ 270KB + LSTM state。"""
    from streaming import NNAECStreaming
    model = _load_model()
    st = NNAECStreaming(model, device="cpu")
    rng = np.random.default_rng(2)
    s1 = st.cache_size_bytes()
    for _ in range(50):                             # 1s
        st.process_chunk(rng.standard_normal(320).astype(np.float32),
                         rng.standard_normal(320).astype(np.float32))
    s2 = st.cache_size_bytes()
    assert s1 == s2, "cache 应随帧数滚动而非增长"
    assert 200_000 < s1 < 400_000, f"cache 账本 {s1} B 超出预期带（~270KB）"
    # 层内 5 帧 × 通道 × 频点 手算对照（保守带内即可）
    assert st.enc[0].cache.shape == (1, 4, 257, 5)
    assert st.dec[0].cache.shape == (1, 64, 33, 5)


def test_real_data_end_to_end():
    """真实数据（eval_v3 2s 段）：流式输出与离线一致 + 全零静音段输出非 NaN。"""
    import soundfile as sf
    mic, _ = sf.read("D:/data/aec-synth/eval_v3/mic/fileid_0_dt.wav", dtype="float32")
    far, _ = sf.read("D:/data/aec-synth/eval_v3/farend/fileid_0_dt.wav", dtype="float32")
    n = 32000
    mic, far = mic[:n], far[:n]
    model = _load_model()
    ref = _offline_ref(model, mic, far)
    out = _run_streaming(model, mic, far, 320)
    got = _align(out, ref)
    err = np.abs(got - ref).max()
    assert err < 1e-5, f"真实数据对齐误差 {err:.2e}"
    assert np.isfinite(got).all()
