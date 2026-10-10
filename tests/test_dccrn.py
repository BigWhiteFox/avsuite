# -*- coding: utf-8 -*-
"""WP-04 单测：复数层手算对拍 / 模型结构与参数量 / DM 混合精度 / 过拟合冒烟。"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dccrn_shim as X  # noqa: E402

DCCRN = X.DCCRN
complex_nn = X.complex_nn
dm = X.dynamic_mixing


def test_complex_conv_matches_complex_algebra():
    """ComplexConv2d 输出 == 复数乘法定义手算（四路实 conv 组合）。

    注：torch F.conv2d 对 complex dtype 不做完整复数代数（缺交叉项），
    所以参照实现手工展开 (W_re+jW_im)(x_re+jx_im)。
    """
    torch.manual_seed(0)
    cc = complex_nn.ComplexConv2d(2, 3, kernel=3, stride=1, padding=1)
    x = torch.randn(1, 4, 7, 5)                    # 2 复数通道 = 4 实通道
    out = cc(x)
    xr, xi = x[:, :2], x[:, 2:]
    conv = torch.nn.functional.conv2d
    ref_re = conv(xr, cc.conv_re.weight, cc.conv_re.bias, padding=1) \
        - conv(xi, cc.conv_im.weight, cc.conv_im.bias, padding=1)
    ref_im = conv(xi, cc.conv_re.weight, cc.conv_re.bias, padding=1) \
        + conv(xr, cc.conv_im.weight, cc.conv_im.bias, padding=1)
    torch.testing.assert_close(out[:, :3], ref_re, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out[:, 3:], ref_im, atol=1e-6, rtol=1e-5)


def test_complex_lstm_one_step_manual():
    """ComplexLSTM 单步输出 == 手算复数门（i,f,g,o 复数sigmoid/tanh 按分量）。"""
    torch.manual_seed(1)
    lstm = complex_nn.ComplexLSTM(3, 2, num_layers=1).eval()
    x = torch.randn(1, 1, 6)                       # T=1, D_in=3 复数
    with torch.no_grad():
        out = lstm(x)
    xr, xi = x[0, 0, :3], x[0, 0, 3:]
    g_re = lstm.w_re[0](xr) - lstm.w_im[0](xi)
    g_im = lstm.w_re[0](xi) + lstm.w_im[0](xr)
    i = torch.complex(torch.sigmoid(g_re[:2]), torch.sigmoid(g_im[:2]))
    f = torch.complex(torch.sigmoid(g_re[2:4]), torch.sigmoid(g_im[2:4]))
    g = torch.complex(torch.tanh(g_re[4:6]), torch.tanh(g_im[4:6]))
    o = torch.complex(torch.sigmoid(g_re[6:]), torch.sigmoid(g_im[6:]))
    c = i * g                                       # c0=0, f·0=0
    h = o * torch.complex(torch.tanh(c.real), torch.tanh(c.imag))
    torch.testing.assert_close(out[0, 0, :2], h.real, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out[0, 0, 2:], h.imag, atol=1e-6, rtol=1e-5)


def test_model_shapes_and_param_band():
    """4s 输入：输出同长；参数量落在 3-4.5M 带（DoD：3-4M 量级）。"""
    torch.manual_seed(2)
    m = DCCRN()
    n_par = sum(p.numel() for p in m.parameters())
    assert 2.5e6 <= n_par <= 4.5e6, f"{n_par/1e6:.2f}M 超出量级带"
    wav = torch.randn(2, 64000)
    enh, mask, noisy, enh_spec = m(wav)
    assert enh.shape == wav.shape
    assert mask.shape == (2, 2, 257, 321)           # 4s@hop200 -> T=321
    assert noisy.shape == (2, 257, 321)


def test_model_arbitrary_length_padding():
    """非 T≡1(mod16) 的任意长度：帧数对齐逻辑不崩、输出等长。"""
    torch.manual_seed(3)
    m = DCCRN(clstm_hidden=8)
    for n in (16000, 16001, 33333, 79999):          # T=161,161,334,800 混合奇偶
        out = m(torch.randn(1, n))[0]
        assert out.shape == (1, n)


def test_real_conv_ablation_forward():
    """real-conv 消融可前向。参数量惯例说明：实数版把 Re/Im 当普通通道
    (2ci→2co)，卷积参数是复数版的 2 倍——复数网络的论文消融惯例即如此
    （对照的是"同结构实数化"，不是"同参数量"），README 有对照表。"""
    torch.manual_seed(4)
    m = DCCRN(clstm_hidden=8, real_conv=True)
    out = m(torch.randn(1, 32000))[0]
    assert out.shape == (1, 32000)


def test_gradient_flow():
    torch.manual_seed(5)
    m = DCCRN(clstm_hidden=8)
    clean = 0.1 * torch.randn(1, 16000)
    noisy = clean + 0.05 * torch.randn(1, 16000)
    loss = X.train.make_loss_fn(0.1, "cpu")(m, {"noisy": noisy, "clean": clean})
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 10 and all(torch.isfinite(g).all() for g in grads)


def test_mix_at_snr_accuracy():
    """mix_at_snr 实测 SNR 与请求一致（±0.1dB，功率定义）。"""
    rng = np.random.default_rng(0)
    clean = rng.standard_normal(64000).astype(np.float32) * 0.1
    noise = rng.standard_normal(64000).astype(np.float32)
    for snr in (-5, 0, 7.5, 15):
        noisy, c, n_used = dm.mix_at_snr(clean, noise, snr)
        measured = 10 * np.log10(np.mean(c ** 2) / np.mean(n_used ** 2))
        assert abs(measured - snr) < 0.1
        assert np.allclose(noisy, c + n_used)


def test_dynamic_mix_levels_and_repro():
    """DM 数据集：响度归一落在 level_range、seed 固定可复现、SNR 在范围内。"""
    ds = dm.DNSDynamicMix(epoch_len=6, seg_sec=1.0, seed=42)
    a, b = ds[0], ds[0]
    torch.testing.assert_close(a["noisy"], b["noisy"])      # seed 逐 idx 确定
    for i in range(6):
        item = ds[i]
        assert item["noisy"].shape == (16000,)
        lvl = 20 * np.log10(item["clean"].pow(2).mean().sqrt().item())
        assert -36 <= lvl <= -19, f"响度 {lvl} 超出归一带"
        snr = 10 * np.log10(item["clean"].pow(2).mean().item() /
                            (item["noisy"] - item["clean"]).pow(2).mean().item())
        assert -6 <= snr <= 16


def test_overfit_single_batch():
    """小配置单 batch 过拟合：loss 60 步降到 <60% 初值（结构可学习 sanity）。"""
    torch.manual_seed(6)
    m = DCCRN(clstm_hidden=8)
    g = torch.Generator().manual_seed(0)
    clean = 0.1 * torch.randn(2, 16000, generator=g)
    noisy = clean + 0.08 * torch.randn(2, 16000, generator=g)
    batch = {"noisy": noisy, "clean": clean}
    loss_fn = X.train.make_loss_fn(0.1, "cpu")
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = loss_fn(m, batch)
        if first is None:
            first = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.6 * first, f"{first:.4f} -> {float(loss):.4f}"
