"""WP-02 单测：手算对拍 / 尺度不变性 / 梯度有限 / 数值稳定 / 退化关系。"""
import math

import pytest
import torch

from core.losses import (
    complex_mse_loss,
    irm_bce_loss,
    irm_target,
    multi_scale_stft_loss,
    si_snr,
    si_snr_loss,
    weighted_distortion_loss,
)


def test_si_snr_hand_computed():
    """手算对拍。注意 SI-SNR 的投影取"最优尺度"而非生成尺度，
    所以构造时让噪声与 ref' 正交（此时最优尺度 = 生成尺度 2）：
    est' = 2·ref' + [1,-2,1]，[1,-2,1] 零均值且 ⊥ ref'=[-1,0,1]。
    SI-SNR = 10·log10( ||2·ref'||² / ||[1,-2,1]||² ) = 10·log10(8/6)。"""
    ref = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    est = 2 * (ref - ref.mean()) + torch.tensor([[1.0, -2.0, 1.0]], dtype=torch.float64)
    assert abs(si_snr(est, ref).item() - 10 * math.log10(4 / 3)) < 1e-9


def test_si_snr_scale_invariant_and_dc_removal():
    """尺度不变：est×100 同分；DC 偏置被零均值化消掉。"""
    torch.manual_seed(0)
    ref = torch.randn(4, 1600, dtype=torch.float64)
    est = ref + 0.01 * torch.randn_like(ref)
    s0 = si_snr(est, ref)
    assert torch.allclose(si_snr(est * 100, ref), s0, atol=1e-9)
    assert torch.allclose(si_snr(est + 5.0, ref), s0, atol=1e-9)   # 加常数 DC


def test_si_snr_zero_input_finite():
    """全零输入约定 0 dB，不产生 NaN/inf。"""
    z = torch.zeros(2, 100)
    v = si_snr(z, z)
    assert torch.isfinite(v).all() and torch.all(v == 0)


def test_complex_mse_value():
    est = torch.tensor([[1 + 2j]], dtype=torch.complex128)
    ref = torch.tensor([[0 - 1j]], dtype=torch.complex128)
    assert complex_mse_loss(est, ref).item() == pytest.approx(1.0 + 9.0)  # |Δ|² = 1+9


def test_irm_target_hand_computed():
    """单 bin 手算：|clean|²=4, |noise|²=12 -> IRM = sqrt(4/16)=0.5。"""
    clean = torch.tensor([[2.0 + 0j]], dtype=torch.complex128)
    noise = torch.tensor([[0.0 + (math.sqrt(12)) * 1j]], dtype=torch.complex128)
    irm = irm_target(clean + noise, clean)
    assert irm.item() == pytest.approx(0.5, abs=1e-12)
    assert irm.item() <= 1.0 and irm.item() >= 0.0


def test_irm_bce_perfect_mask_near_zero():
    """极端用例：前半 bin 纯语音（IRM=1）、后半 bin 纯噪声（IRM≈0）。
    est_mask == IRM -> BCE ≈ 0；常数 0.5 掩码 -> 明显大于 0。"""
    torch.manual_seed(0)
    clean = torch.zeros(1, 32, dtype=torch.complex128)
    noise = torch.zeros(1, 32, dtype=torch.complex128)
    clean[0, :16] = 2.0        # 纯语音 bin -> IRM = 1
    noise[0, 16:] = 1.0        # 纯噪声 bin -> IRM = 0
    mix = clean + noise
    irm = irm_target(mix, clean)
    assert torch.all(irm[0, :16] > 0.9999) and torch.all(irm[0, 16:] < 1e-4)
    assert irm_bce_loss(irm, mix, clean).item() < 1e-6
    assert irm_bce_loss(torch.full_like(irm, 0.5), mix, clean).item() > 0.1


def test_weighted_distortion_gamma0_equals_complex_mse():
    """γ=0 严格退化为复数谱 MSE（加权设计的关键校验）。"""
    torch.manual_seed(1)
    ref = torch.randn(2, 33, 20, dtype=torch.complex128)
    est = ref + 0.1 * torch.randn(2, 33, 20, dtype=torch.complex128)
    assert weighted_distortion_loss(est, ref, gamma=0.0).item() == pytest.approx(
        complex_mse_loss(est, ref).item(), rel=1e-12)


def test_weighted_distortion_emphasizes_loud_bins():
    """同样的谱误差，落在高能量 bin 上应罚得更重（γ>0 的设计意图）。
    两 bin 幅度 10 / 0.1，误差 0.1 只放在其中一个 bin：
    γ=0.5 时 loud 误差的损失 ≈ (w_loud/w_quiet)² ≈ 100 倍于 quiet 误差。"""
    torch.manual_seed(2)
    ref = torch.zeros(1, 1, 2, dtype=torch.complex128)
    ref[..., 0] = 10.0 + 0j      # 高能量 bin
    ref[..., 1] = 0.1 + 0j       # 低能量 bin
    est_loud = ref.clone()
    est_loud[..., 0] += 0.1
    est_quiet = ref.clone()
    est_quiet[..., 1] += 0.1
    loss_loud = weighted_distortion_loss(est_loud, ref, gamma=0.5)
    loss_quiet = weighted_distortion_loss(est_quiet, ref, gamma=0.5)
    assert loss_loud.item() > loss_quiet.item()
    # 权重比 = sqrt(10.1/0.1) ≈ 10（power-law γ=0.5 把能量比开方）
    assert loss_loud.item() / loss_quiet.item() > 5


@pytest.mark.parametrize("loss_fn", [
    lambda e, r: si_snr_loss(e, r),
])
def test_gradients_finite(loss_fn):
    """时域损失：随机张量上可反传、梯度有限。谱域损失的反传见 test_gradient_spectral_losses。"""
    torch.manual_seed(3)
    ref = torch.randn(2, 8000, dtype=torch.float64)
    est = (ref + 0.1 * torch.randn_like(ref)).requires_grad_(True)
    loss = loss_fn(est, ref)
    assert torch.isfinite(loss)
    loss.backward()
    assert est.grad is not None and torch.isfinite(est.grad).all()


def test_gradient_spectral_losses():
    torch.manual_seed(4)
    ref = torch.randn(2, 33, 50, dtype=torch.complex128)
    est = (ref + 0.05 * torch.randn_like(ref)).requires_grad_(True)
    for loss in (complex_mse_loss(est, ref),
                 weighted_distortion_loss(est, ref, gamma=0.5),
                 irm_bce_loss(torch.sigmoid(est.real), ref + est, ref)):
        loss.backward(retain_graph=True)
    assert torch.isfinite(est.grad).all()


def test_multi_scale_stft_loss():
    """相同信号 -> 0；不同信号 -> >0；可反传。"""
    torch.manual_seed(5)
    x = torch.randn(2, 8000, dtype=torch.float64)
    assert multi_scale_stft_loss(x, x).item() == pytest.approx(0.0, abs=1e-12)
    y = torch.randn(2, 8000, dtype=torch.float64).requires_grad_(True)
    loss = multi_scale_stft_loss(y, x)
    assert loss.item() > 0
    loss.backward()
    assert torch.isfinite(y.grad).all()


def test_numerical_stability_degenerate_inputs():
    """全零 / 极短信号输入不炸（训练早期常见）。"""
    z = torch.zeros(1, 1000)
    assert torch.isfinite(si_snr_loss(z, z))
    assert torch.isfinite(complex_mse_loss(torch.zeros(1, 33, 10, dtype=torch.complex128),
                                           torch.zeros(1, 33, 10, dtype=torch.complex128)))
    assert torch.isfinite(multi_scale_stft_loss(z, z))          # 1000 点 < 最大窗 2048
    assert torch.isfinite(multi_scale_stft_loss(torch.zeros(1, 8000),
                                                torch.zeros(1, 8000)))
