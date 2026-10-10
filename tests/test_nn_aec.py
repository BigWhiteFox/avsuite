# -*- coding: utf-8 -*-
"""WP-07 单测：参数量 <1M / 时间因果性（核心约束）/ 形状 / 梯度 / 过拟合 / jitter。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nnaec_shim as X  # noqa: E402

NNAEC = X.NNAEC
dataset = X.dataset
train_mod = X.train


def test_param_count_under_1m():
    n = sum(p.numel() for p in NNAEC().parameters())
    assert n < 1_000_000, f"{n/1e6:.3f}M 超出 <1M 约束"
    assert n > 200_000, f"{n} 过小，结构可疑"


def test_shapes():
    torch.manual_seed(0)
    m = NNAEC()
    mic = torch.randn(2, 32000)
    far = torch.randn(2, 32000)
    enh, mask, mic_spec, enh_spec = m(mic, far)
    assert enh.shape == mic.shape
    assert mask.shape == (2, 2, 257, 101)      # T = 32000/320 + 1
    assert mic_spec.shape == (2, 257, 101)


def test_time_causality():
    """核心约束：扰动未来（t>=2s）的 mic/far，t<2s-窗长 的输出必须逐位不变。

    时间因果 = conv 左 pad + 单向 LSTM + deconv 右移 2 帧的组合性质；
    ISTFT 重叠会向后泄漏一窗（512 样本），比较区留 margin。
    """
    torch.manual_seed(1)
    m = NNAEC().eval()
    mic = torch.randn(1, 64000)
    far = torch.randn(1, 64000)
    mic2, far2 = mic.clone(), far.clone()
    cut = 32000                                   # 2s
    g = torch.Generator().manual_seed(9)
    mic2[:, cut:] = torch.randn(mic2[:, cut:].shape, generator=g)
    far2[:, cut:] = torch.randn(far2[:, cut:].shape, generator=g)
    with torch.no_grad():
        y1 = m.enhance(mic, far)
        y2 = m.enhance(mic2, far2)
    margin = 512 + 320                            # ISTFT 窗 + 帧移
    torch.testing.assert_close(y1[:, :cut - margin], y2[:, :cut - margin],
                               rtol=0, atol=1e-6)
    assert not torch.allclose(y1[:, cut + margin:], y2[:, cut + margin:])


def test_gradient_flow():
    torch.manual_seed(2)
    m = NNAEC()
    far = 0.2 * torch.randn(1, 16000)
    near = 0.1 * torch.randn(1, 16000)
    mic = far * 0.5 + near
    loss = train_mod.make_loss_fn(0.5, 0.1, "cpu")(m, {"mic": mic, "far": far,
                                                       "near": near})
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 10 and all(torch.isfinite(g).all() for g in grads)


def test_overfit_single_batch():
    """合成回声对（far 衰减+近端）单 batch 过拟合：loss 应显著下降。"""
    torch.manual_seed(3)
    m = NNAEC()
    g = torch.Generator().manual_seed(0)
    far = 0.2 * torch.randn(2, 16000, generator=g)
    near = 0.1 * torch.randn(2, 16000, generator=torch.Generator().manual_seed(5))
    mic = 0.5 * far + near                        # 理想线性回声
    batch = {"mic": mic, "far": far, "near": near}
    loss_fn = train_mod.make_loss_fn(0.5, 0.1, "cpu")
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(80):
        loss = loss_fn(m, batch)
        if first is None:
            first = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.7 * first, f"{first:.4f} -> {float(loss):.4f}"


def test_dataset_jitter_and_shift():
    """_shift 语义 + AecDataset jitter 只动 far。"""
    x = np.arange(10, dtype=np.float32)
    np.testing.assert_allclose(dataset._shift(x, 3), [0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(dataset._shift(x, -2), [2, 3, 4, 5, 6, 7, 8, 9, 0, 0])
    ds = dataset.AecDataset("D:/data/aec-synth/eval_v3", jitter_ms=10, seed=1)
    a = ds[0]
    assert a["mic"].shape == a["far"].shape == a["near"].shape
    ds0 = dataset.AecDataset("D:/data/aec-synth/eval_v3", jitter_ms=0)
    b = ds0[0]
    assert not torch.allclose(a["far"], b["far"])   # jitter 生效
    assert torch.allclose(a["mic"], b["mic"])       # mic 不受影响
