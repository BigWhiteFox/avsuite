"""损失函数库（WP-02 / 模块2）。

五个损失：SI-SNR / 复数谱 MSE / IRM 掩码 BCE / 加权语音失真（power-law）/ 多尺度 STFT。
适用场景速查表见 core/README.md。数值约定：
- 时域输入 [B, N]，谱输入 [B, F, T] 复数，掩码 (0,1) 实数；
- 所有损失对全零输入有限（eps 下限），梯度有限；tests/test_losses.py 逐项覆盖。
"""
import math

import torch
from torch import nn

from .stft import STFT

__all__ = ["si_snr", "si_snr_loss", "complex_mse_loss", "irm_target", "irm_bce_loss",
           "weighted_distortion_loss", "multi_scale_stft_loss",
           "perceptual_band_weight", "psychoacoustic_loss"]

# ERB 滤波器组缓存（纯几何量，按 (F, 带数, sr, n_fft, device, dtype) 复用）
_ERB_CACHE: dict = {}


def si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """尺度不变信噪比（dB）。[..., N] -> [...]。

    三步：零均值化（DC 不是干扰）-> 向参考方向投影消尺度 -> 投影/残差 能量比。
    全零输入约定返回 0 dB（两侧都 clamp 到 eps，训练中不会 NaN/inf）。
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    ref_e = torch.sum(ref ** 2, dim=-1, keepdim=True).clamp_min(eps)
    proj = torch.sum(est * ref, dim=-1, keepdim=True) / ref_e * ref
    noise = est - proj
    return 10 * torch.log10(torch.sum(proj ** 2, dim=-1).clamp_min(eps)
                            / torch.sum(noise ** 2, dim=-1).clamp_min(eps))


def si_snr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SNR 取负取平均（越小越好）。"""
    return -si_snr(est, ref, eps).mean()


def complex_mse_loss(est_spec: torch.Tensor, ref_spec: torch.Tensor) -> torch.Tensor:
    """复数谱 MSE：|Ŝ - S|² 的均值（Re/Im 两路等权，相位信息隐含在内）。"""
    return ((est_spec.real - ref_spec.real) ** 2 +
            (est_spec.imag - ref_spec.imag) ** 2).mean()


def irm_target(mix_spec: torch.Tensor, clean_spec: torch.Tensor,
               eps: float = 1e-8) -> torch.Tensor:
    """理想比率掩码 IRM：sqrt(P_speech / (P_speech + P_noise))，取值 (0,1)。

    P_noise = |mix|² - |clean|²（下限 eps）。开方使掩码作用在幅度域
    （Wiener 增益），复数版 cIRM 才能同时修正相位——WP-03 消融会用到。
    """
    p_speech = clean_spec.abs() ** 2
    p_noise = (mix_spec.abs() ** 2 - p_speech).clamp_min(eps)
    return torch.sqrt(p_speech / (p_speech + p_noise)).clamp(0.0, 1.0)


def irm_bce_loss(est_mask: torch.Tensor, mix_spec: torch.Tensor,
                 clean_spec: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """预测掩码（sigmoid 输出）对 IRM 目标的 BCE。目标天然在 (0,1)，BCE 梯度陡。"""
    target = irm_target(mix_spec, clean_spec)
    return nn.functional.binary_cross_entropy(est_mask.clamp(eps, 1.0 - eps), target)


def weighted_distortion_loss(est_spec: torch.Tensor, ref_spec: torch.Tensor,
                             gamma: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    """加权语音失真（AEC 式）：干净幅度 power-law 压缩后加权谱误差。

    L = mean( w · |S - Ŝ|² )，w = (|S|+eps)^γ / mean((|S|+eps)^γ)
    - γ=0 退化为普通复数谱 MSE（单测校验）；
    - γ 越大越强调高能量区——响度感知近似 幅度^0.6（Stevens 幂律），
      等价于人耳对"响的地方错一点"更敏感；
    - γ 的业务向调优（扫描/响度域加权）延至 WP-07 的消融。
    """
    w = (ref_spec.abs() + eps).pow(gamma)
    w = w / w.mean().clamp_min(eps)
    err2 = (est_spec.real - ref_spec.real) ** 2 + (est_spec.imag - ref_spec.imag) ** 2
    return (w * err2).mean()


def _erb_band_centers(f_bins: int, n_bands: int, sr: int, n_fft: int,
                      f_max: float | None = None) -> torch.Tensor:
    """ERB 尺度（Glasberg & Moore）上等间隔的带中心（Hz），f_max 默认奈奎斯特。

    ERB(f) = 24.7·(4.37·f/1000 + 1)；可听带按 ERB 等间隔划分 → 低频带窄、
    高频带宽，与人耳临界带分辨率一致（PAQM/PEAQ 类感知测度共用的第一步）。
    不依赖输入，纯函数（单测可直接查手算值）。
    """
    f_max = float(f_max) if f_max is not None else sr / 2
    e_max = 21.4 * math.log10(4.37 * f_max / 1000 + 1)
    e = torch.linspace(0.0, e_max, n_bands + 1, dtype=torch.float64)
    centers = (10 ** (e / 21.4) - 1) * 1000 / 4.37
    # 带中心取相邻边界的算术中点（ERB 带在 f 上近似等宽 ⇒ 中点≈真中心；
    # 用几何中点会把第 0 带推到 ~5mHz，破坏"ERB 域近似等距"的结构性质）
    c = 0.5 * (centers[:-1] + centers[1:])
    return c.clamp(0.0, f_max)


def _erb_projection(f_bins: int, n_bands: int, sr: int, n_fft: int,
                    device, dtype) -> torch.Tensor:
    """[F, n_bands] 三角 ERB 滤波器组。频率轴按 n_fft/sr 的 FFT 栅格铺开。

    纯几何量、与输入无关 ⇒ 按 (f_bins, n_bands, sr, dtype) 缓存（同一次训练里
    每个 batch 复用同一份矩阵，避免 257×28 的重复构造进入热点路径）。
    """
    key = (f_bins, n_bands, sr, n_fft, str(device), str(dtype))
    cached = _ERB_CACHE.get(key)
    if cached is not None:
        return cached
    centers = _erb_band_centers(f_bins, n_bands, sr, n_fft)
    freqs = torch.linspace(0.0, sr / 2.0, f_bins, dtype=torch.float64)
    w = torch.zeros(f_bins, n_bands, dtype=torch.float64)
    for b in range(n_bands):
        lo = centers[b - 1] if b > 0 else torch.zeros((), dtype=torch.float64)
        hi = centers[b + 1] if b < n_bands - 1 else centers.new_tensor(sr / 2.0)
        c = centers[b]
        left = ((freqs - lo) / (c - lo).clamp_min(1e-9)).clamp(0.0, 1.0) if b > 0 \
            else torch.ones_like(freqs)
        right = ((hi - freqs) / (hi - c).clamp_min(1e-9)).clamp(0.0, 1.0) \
            if b < n_bands - 1 else torch.ones_like(freqs)
        w[:, b] = torch.minimum(left, right)
    # 逐 bin 归一（每个 FFT bin 的总权重为 1）：带内取加权平均而非求和 ⇒ 不带宽偏置
    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    out = w.to(device=device, dtype=dtype)
    _ERB_CACHE[key] = out
    return out


def perceptual_band_weight(ref_spec: torch.Tensor, *,
                           n_bands: int = 28, sr: int = 16000, n_fft: int = 512,
                           alpha: float = 0.3, tau: float = 1.5,
                           eps: float = 1e-8) -> torch.Tensor:
    """参考谱的感知带权重 a[f]（PAQM 式"可听度加权"的可微简化）：[B, F]。

    三步（全部可微、无判别器、无额外 STFT）：
      1. ERB 滤波器组把 |S| 压成 ~n_bands 个临界带电平（带内加权平均 ⇒ 无带宽偏置）；
      2. 幂律压缩 L = mean(P)^alpha（alpha≈0.3 近似耳蜗压缩 + Stevens 响度幂律），
         这是 PAQM 里"耳蜗压缩后比较"的最小实现；
      3. 逐文件 softmax(l_ref/tau) 作权重——参考越响的带越"听得见"，误差越该被罚，
         同时逐文件归一（权重和为 1）⇒ 不奖励单纯放大输出。
    权重逐文件、逐时帧重算（[B,F]，T 上共享），显存开销 O(B·F·n_bands)。
    """
    ref_pow = ref_spec.abs().pow(2).mean(dim=-1)                # [B, F]
    return _band_weight_from_power(ref_pow, ref_pow.shape[-1], n_bands, sr, n_fft,
                                   ref_pow.device, ref_pow.dtype,
                                   alpha=alpha, tau=tau, eps=eps)


def _band_weight_from_power(ref_pow: torch.Tensor, n_freq: int, n_bands: int, sr: int,
                            n_fft: int, device, dtype, *, alpha: float, tau: float,
                            eps: float) -> torch.Tensor:
    """由参考带电平算 [B, F] 感知权重（供 loss 与单测共用）。"""
    proj = _erb_projection(n_freq, n_bands, sr, n_fft, device, dtype)
    band_pow = ref_pow @ proj                                   # [B, n_bands]
    level = band_pow.clamp_min(eps).pow(alpha).log()            # 压缩后取对数域
    a = torch.softmax(level / max(tau, 1e-6), dim=-1)           # [B, n_bands]
    w = a @ proj.transpose(0, 1)                                # [B, F]
    return w / w.sum(dim=-1, keepdim=True).clamp_min(eps)


def psychoacoustic_loss(est_spec: torch.Tensor, ref_spec: torch.Tensor, *,
                        n_bands: int = 28, sr: int = 16000, n_fft: int = 512,
                        alpha: float = 0.3, tau: float = 1.5,
                        w_level: float = 1.0, w_tilt: float = 0.1,
                        tilt_lo_hz: float = 4000.0, tilt_hi_hz: float = 8000.0,
                        eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """心理声学失真损失（PAQM 派生的可微感知测度，无判别器、8GB 友好）。

    动机：V4-S5 实测 NN-AEC 的 DT 段 deg 短板 = **4–8kHz 高频带选择性损伤 + 谱平坦化**，
    而基线 `weighted_distortion_loss`（w∝|S|^γ）在 |S| 衰减快的频带权重自然趋零 ⇒
    高频带的误差在目标里本就近乎不可见。本损失把"耳蜗压缩 + 可听度加权"显式写进目标：

        L_perc = L_level + λ·L_tilt
        L_level = mean_{f,t}[ a[f] · |S_ref|^α − |S_est|^α | ] / mean_{f,t}[ |S_ref|^α ]
        L_tilt  = E_t[( (E_lo − E_hi)/E_tot )_ref − (…)_est]²，4–8kHz 能量占比回归

    返回值 **（L_perc, 尺度项）**：L_perc 已按参考响度归一（无量纲、幅度不变），
    第二个值是 mean_ref_abs = mean|S_ref^α|，供训练侧把本损失整体缩放到与原损失同量级
    （见 `exp/07_nn_aec/train.py --loss psycho`，缩放常数在启动时按同批数据实测并写入日志/ckpt）。

    与基线损失的区别只在**谱误差项的加权方式**（响度域可听度加权 vs 能量幂律加权），
    共同项 SI-SNR 不变，故与 v4 列构成单变量对照。

    参考：Abreu, Miranda & Biscainho, "Efficient Audio Enhancement with a Differentiable
    Psychoacoustic Loss"（JAES v74 n6 p430-444, 2026；arXiv:2608.02918，获取日期 2026-10-13）
    ——PAQM 派生可微感知损失、训练显存比基线低 2-4×。本实现只借用"耳蜗压缩 + 临界带
    可听度加权"的思想骨架，不做 PAQM 全套（无时域掩蔽模板/无绝对听阈），**不声称数值等价**；
    代码在本仓独立实现，未复用该文代码或权重。
    """
    est_mag = est_spec.abs()
    ref_mag = ref_spec.abs()
    est_c = est_mag.clamp_min(eps).pow(alpha)
    ref_c = ref_mag.clamp_min(eps).pow(alpha)
    ref_pow = ref_mag.pow(2).mean(dim=-1)                       # [B, F]
    a = _band_weight_from_power(ref_pow, est_mag.shape[1], n_bands, sr, n_fft,
                                est_mag.device, est_mag.dtype,
                                alpha=alpha, tau=tau, eps=eps)  # [B, F]
    num = ((ref_c - est_c).abs() * a.unsqueeze(-1)).sum(dim=1).mean()
    den = ref_c.mean().clamp_min(eps)
    lvl = num / den

    if w_tilt > 0:
        freqs = torch.linspace(0.0, sr / 2.0, est_mag.shape[1],
                               device=est_mag.device, dtype=est_mag.dtype)
        m_hf = ((freqs >= tilt_lo_hz) & (freqs < tilt_hi_hz))[None, :, None]
        m_lo = (freqs < tilt_lo_hz)[None, :, None]
        r_tot = ref_mag.mean(dim=1, keepdim=True).clamp_min(eps)   # [B, 1, T]
        e_tot = est_mag.mean(dim=1, keepdim=True).clamp_min(eps)
        r_hf = (ref_mag * m_hf).sum(dim=1) / (m_hf.sum() * r_tot.squeeze(1))
        e_hf = (est_mag * m_hf).sum(dim=1) / (m_hf.sum() * e_tot.squeeze(1))
        r_lo = (ref_mag * m_lo).sum(dim=1) / (m_lo.sum() * r_tot.squeeze(1))
        e_lo = (est_mag * m_lo).sum(dim=1) / (m_lo.sum() * e_tot.squeeze(1))
        tilt = ((r_lo - r_hf) - (e_lo - e_hf)).pow(2).mean()
    else:
        tilt = est_mag.new_zeros(())
    return w_level * lvl + w_tilt * tilt, den.detach()


def multi_scale_stft_loss(est: torch.Tensor, ref: torch.Tensor,
                          scales=((512, 128), (1024, 256), (2048, 512)),
                          window: str = "hann") -> torch.Tensor:
    """多尺度 STFT loss（HiFi-GAN 式）：逐尺度 谱收敛项 + log 幅度项 后求和。

    est/ref: [B, N] 时域。单窗受时频不确定性约束——短窗保瞬态、长窗保谐波包络，
    多尺度同时兼顾。谱收敛 L_sc = |||E|-|R|||_F / (|||R|||_F + |||E|||_F)，
    幅度项用 log 域（对低能量区公平），log 下限 1e-5 数值保底。
    内部 STFT 用 zeros pad（而非 reflect）：损失对边缘不敏感，但短信号更稳。
    """
    loss = est.new_zeros(())
    for n_fft, hop in scales:
        stft = STFT(n_fft=n_fft, hop=hop, window=window, pad_mode="zeros").to(est.device)
        me, mr = stft.transform(est).abs(), stft.transform(ref).abs()
        sc = torch.norm(me - mr, p="fro") / torch.norm(mr, p="fro").add(torch.norm(me, p="fro")).clamp_min(1e-8)
        mag = (me.clamp_min(1e-5).log() - mr.clamp_min(1e-5).log()).abs().mean()
        loss = loss + sc + mag
    return loss
