"""WP-05a 单测：GTCRN 参数量 / 权重加载推理 / ONNX 对拍(<1e-4 DoD) / 时间因果性。"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / "exp" / "05a_gtcrn_onnx"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CKPT_DNS3, istft_convention, load_model, stft_convention  # noqa: E402
from gtcrn import GTCRN  # noqa: E402

ONNX_SIM = HERE / "outputs" / "gtcrn_dns3.sim.onnx"
TEST_WAV = Path("D:/data/repos/gtcrn-repo/test_wavs/mix.wav")


def test_param_count_23k():
    """官方标称 23.67K = 可训练参数（ERB 滤波器组 24.6K 是 requires_grad=False
    的固定查表 Parameter，不计入）。总量 48.2K。轻量化卖点，两个数都锁死。"""
    m = GTCRN()
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    assert 23000 <= trainable <= 24500, f"可训练 {trainable} 超出 23.67K 带"
    assert 47000 <= total <= 49500, f"总量 {total} 超出 48.2K 带"


def test_ckpt_loads_and_enhances_official_wav():
    """DNS3 官方权重 + 官方测试 wav：链路通、输出有限非零、噪声能量下降。"""
    if not (CKPT_DNS3.exists() and TEST_WAV.exists()):
        pytest.skip("权重或官方测试 wav 缺失")
    import soundfile as sf
    model = load_model(CKPT_DNS3)
    mix, sr = sf.read(str(TEST_WAV), dtype="float32")
    assert sr == 16000
    wav = torch.from_numpy(mix[: 16000 * 3])
    with torch.no_grad():
        spec = stft_convention(wav)
        enh_spec = model(spec)
        enh = istft_convention(enh_spec, length=len(wav))[0].numpy()
    assert np.isfinite(enh).all() and np.abs(enh).max() > 1e-3
    assert np.mean(enh ** 2) < np.mean(mix[:len(enh)] ** 2)      # 含噪能量应下降


def test_onnx_parity_dod():
    """PyTorch vs ONNXRuntime max|diff| < 1e-4（WORKLIST DoD）。"""
    if not ONNX_SIM.exists():
        pytest.skip("先跑 onnx_export.py")
    import onnxruntime as ort
    model = load_model(CKPT_DNS3)
    torch.manual_seed(0)
    x = torch.randn(1, 257, 101, 2) * 0.1
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(ONNX_SIM), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"spec": x.numpy()})[0]
    err = float(np.abs(out - ref).max())
    assert err < 1e-4, f"parity {err:.2e} >= 1e-4"


@pytest.mark.skipif(not CKPT_DNS3.exists(), reason="需要本机 DNS3 官方权重(CI 跳过)")
def test_time_causality():
    """时间因果性：改动未来帧输入，过去帧输出必须逐位不变（流式部署的前提，
    官方 repo 同款检查）。GTConv 时间维只做左侧 padding、inter-GRU 单向。"""
    model = load_model(CKPT_DNS3)
    torch.manual_seed(1)
    x = torch.randn(1, 257, 12, 2) * 0.1
    x2 = x.clone()
    x2[:, :, 8:] = torch.randn_like(x2[:, :, 8:]) * 0.1     # 扰动第 8 帧起
    with torch.no_grad():
        y1, y2 = model(x), model(x2)
    torch.testing.assert_close(y1[:, :, :8], y2[:, :, :8], rtol=0, atol=1e-6)
    assert not torch.allclose(y1[:, :, 8:], y2[:, :, 8:])    # 未来帧确实变了
