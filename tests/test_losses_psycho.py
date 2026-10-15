"""P0-5 训练工程单测：心理声学损失（`core.losses.psychoacoustic_loss`）。

覆盖四件事（对应产物 `reports/p0_5_loss_matrix.md` 的“损失函数自检”节）：
1. ERB 滤波器组的结构性质（权重归一 / 非负 / 单调带中心 / ERB 域等距）；
2. 数值稳定（全零、退化形状恒有限）；
3. 可微性（backward 得到有限非零梯度）；
4. **行为一致性**：人为压低 4–8kHz（V4-S5 实测的 deg 短板机理）必须抬高损失——
   这是“损失定义指向该机理”的可证伪断言，而不是“训练完数字变好”的事后解释。
"""
import torch

from core.losses import (
    _erb_band_centers,
    _erb_projection,
    perceptual_band_weight,
    psychoacoustic_loss,
)

SR, N_FFT, HOP = 16000, 512, 320
F_BINS = N_FFT // 2 + 1


def _ploss(est, ref, **kw):
    """psychoacoustic_loss 返回 (loss, 尺度项)；测试只关心 loss 分量。"""
    return psychoacoustic_loss(est, ref, **kw)[0]


def _spec(n_batch=2, n_frames=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    wav = torch.randn(n_batch, N_FFT + HOP * (n_frames - 1), generator=g)
    frames = wav.unfold(-1, N_FFT, HOP) * torch.hann_window(N_FFT, periodic=True)
    return torch.fft.rfft(frames, dim=-1).transpose(1, 2)


def test_erb_band_centers_monotone_and_equispaced():
    """带中心单调递增、落在 [0, 8k]，且 ERB 域近似等距（gap 离散度 < 1e-3）。"""
    c = _erb_band_centers(F_BINS, 28, SR, N_FFT)
    assert c.shape == (28,)
    assert torch.all(c[1:] > c[:-1])
    assert float(c.min()) > 0.0 and float(c.max()) <= SR / 2 + 1e-6
    erb = 21.4 * torch.log10(4.37 * c / 1000 + 1)
    gaps = erb[1:] - erb[:-1]
    assert float(gaps.max() - gaps.min()) < 1e-3


def test_projection_weights_normalized_and_local():
    """滤波器组逐 bin 归一（和=1）且非负；每个 bin 只被相邻带覆盖（局部性）。"""
    w = _erb_projection(F_BINS, 28, SR, N_FFT, torch.device("cpu"), torch.float32)
    assert w.shape == (F_BINS, 28)
    assert torch.all(w >= 0)
    s = w.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    assert int((w > 0).sum(dim=1).max()) <= 3


def test_perceptual_band_weight_normalized_nonneg():
    """感知权重在频轴上归一（和为 1）、非负；纯低频信号的低频带权重更高。"""
    ref = _spec(seed=1)
    a = perceptual_band_weight(ref)
    assert torch.isfinite(a).all() and torch.all(a >= 0)
    assert torch.allclose(a.sum(dim=-1), torch.ones(a.shape[0], dtype=a.dtype), atol=1e-4)
    g = torch.Generator().manual_seed(2)
    low = torch.zeros(1, F_BINS, 20, dtype=torch.complex64)
    low[:, :8, :] = torch.randn(1, 8, 20, generator=g)
    a_low = perceptual_band_weight(low)
    freqs = torch.linspace(0, SR / 2, F_BINS)
    assert float(a_low[0, freqs >= 4000].mean()) < float(a_low[0, freqs < 1000].mean())


def test_psychoacoustic_zero_and_identical_finite():
    """全零输入有限；est==ref 时损失为 0（可达到的下界）。"""
    z = torch.zeros(2, F_BINS, 30, dtype=torch.complex64)
    v = _ploss(z, z)
    assert torch.isfinite(v) and float(v) < 1e-6
    ref = _spec(seed=3)
    v_same = _ploss(ref, ref)
    assert torch.isfinite(v_same) and float(v_same) < 1e-6


def test_psychoacoustic_scale_invariance():
    """几何尺度不变：est/ref 同乘常数，带加权相对误差基本不变。"""
    ref = _spec(seed=7)
    est = ref * torch.exp(torch.linspace(0, -1.0, ref.shape[1]))[None, :, None]
    v1 = float(_ploss(est, ref))
    v2 = float(_ploss(est * 7.5, ref * 7.5))
    assert abs(v1 - v2) < 1e-3 * max(abs(v1), 1e-6)


def test_psychoacoustic_hf_suppression_raises_loss():
    """（机理断言）把 est 的 4-8kHz 段压低 12dB，损失必须上升。

    V4-S5 的 deg 亏损机理正是“高频带选择性损伤”，本损失若对该扰动不敏感，
    就没有理由期待它把 DT deg 拉回来。
    """
    ref = _spec(seed=4)
    est = ref.clone()
    freqs = torch.linspace(0, SR / 2, F_BINS)
    hf = (freqs >= 4000) & (freqs <= 8000)
    est[:, hf, :] = est[:, hf, :] * 10 ** (-12 / 20)
    base = float(_ploss(ref, ref))
    damaged = float(_ploss(est, ref))
    level_only = float(_ploss(est, ref, w_tilt=0.0))
    assert damaged > base and level_only > 0
    assert damaged != level_only                      # 倾斜项确实参与


def test_psychoacoustic_differentiable():
    """对 est 可微：backward 后梯度存在、有限、非全零（est≠ref）。

    注意 est==ref 时 |Δ| 的次梯度取 0，梯度恰为 0——所以此处用受损 est 测梯度。
    """
    ref = _spec(seed=6)
    est = ref * torch.exp(torch.linspace(0, -1.0, ref.shape[1]))[None, :, None]
    est = est.clone().requires_grad_(True)
    loss = _ploss(est, ref)
    loss.backward()
    g = est.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0


def test_psychoacoustic_degenerate_shapes():
    """单帧 / 单样本 / 全常量输入不 NaN（训练尾声与静音段的健壮性）。"""
    for b, t in ((1, 1), (1, 5), (3, 1)):
        s = torch.full((b, F_BINS, t), 1e-7, dtype=torch.complex64)
        assert torch.isfinite(_ploss(s, s))


def test_psychoacoustic_returns_scale_term():
    """第二返回值 = mean|S_ref|^alpha，供训练侧缩放对齐；有限正值、已 detach。"""
    ref = _spec(seed=8)
    loss, scale = psychoacoustic_loss(ref, ref)
    assert torch.isfinite(scale) and float(scale) > 0
    assert not scale.requires_grad
    assert torch.isfinite(loss)
