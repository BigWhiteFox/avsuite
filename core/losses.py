"""损失函数库（WP-02 / 模块2）。

五个损失：SI-SNR / 复数谱 MSE / IRM 掩码 BCE / 加权语音失真（power-law）/ 多尺度 STFT。
适用场景速查表见 core/README.md。数值约定：
- 时域输入 [B, N]，谱输入 [B, F, T] 复数，掩码 (0,1) 实数；
- 所有损失对全零输入有限（eps 下限），梯度有限；tests/test_losses.py 逐项覆盖。
"""
import torch
from torch import nn

from .stft import STFT

__all__ = ["si_snr", "si_snr_loss", "complex_mse_loss", "irm_target", "irm_bce_loss",
           "weighted_distortion_loss", "multi_scale_stft_loss"]


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
