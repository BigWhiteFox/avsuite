# -*- coding: utf-8 -*-
"""中文语音 demo（WP-01）：共振峰合成"啊-咿-乌" -> 加噪 -> 谱减增强 -> 指标。

全链路只用 core 自己的组件：STFT -> 幅度谱处理（谱减）-> ISTFT -> evaluate。
零数据依赖（无音频文件也能跑）；有真实中文语音时：python demo_zh.py --wav xxx.wav

输出：outputs/demo_{clean,noisy,enhanced}.wav + demo_spectrograms.png，打印 PESQ 对比。
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
from core.evaluate import _safe_pesq, _safe_stoi, _si_snr_np, load16k  # noqa: E402
from core.istft import ISTFT  # noqa: E402
from core.stft import STFT, spec_from_2ch, spec_to_2ch  # noqa: E402

try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

SR = 16000
# Peterson & Barney 经典共振峰 (F1, F2, F3) + 带宽
VOWELS = {"啊": [(730, 90), (1090, 110), (2440, 170)],
          "咿": [(270, 60), (2290, 120), (3010, 200)],
          "乌": [(300, 80), (870, 110), (2240, 180)]}


def synth_vowels(sr=SR, dur_each=0.4, f0_start=150.0, f0_end=110.0):
    """声门脉冲串(F0 滑落+微颤) -> 双二阶共振器级联 -> 音节级 AM（源-滤波模型）。"""
    from scipy.signal import lfilter
    pieces, n_total = [], 0
    for v in VOWELS.values():
        n = int(sr * dur_each)
        n_total += n
        t_rel = np.arange(n_total - n, n_total) / sr                 # 全局时间轴
        f0_t = np.linspace(f0_start, f0_end, n) * (1 + 0.03 * np.sin(2 * np.pi * 5 * t_rel))
        phase = 2 * np.pi * np.cumsum(f0_t) / sr
        src = sum(np.sin(h * phase) / h ** 1.3 for h in range(1, 13))   # 低通状声门谱
        y = src.astype(np.float64)
        for f, bw in v:                                             # 共振器级联塑形
            r = np.exp(-np.pi * bw / sr)
            theta = 2 * np.pi * f / sr
            y = lfilter([1 - r], [1, -2 * r * np.cos(theta), r ** 2], y)
        t_loc = np.arange(n) / sr
        syll = 0.55 + 0.45 * np.sin(2 * np.pi * 2 * t_loc) ** 2     # 2Hz 音节包络
        y *= syll
        fade = int(0.02 * sr)
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
        pieces.append(y)
    return np.concatenate(pieces).astype(np.float32) * 0.2


def spectral_subtraction(spec: torch.Tensor, noise_mag: torch.Tensor,
                         alpha=2.0, beta=0.05) -> torch.Tensor:
    """幅度谱减：|Ŝ| = max(|X| - α·|N|, β·|X|)，相位保持带噪相位（经典 NR 基线）。"""
    mag = spec.abs()
    enh_mag = (mag - alpha * noise_mag.unsqueeze(-1)).clamp_min(beta * mag)
    return spec_from_2ch(torch.stack([enh_mag * spec.real / (mag + 1e-8),
                                      enh_mag * spec.imag / (mag + 1e-8)], dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None, help="真实中文语音路径（默认共振峰合成）")
    ap.add_argument("--snr-db", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    import soundfile as sf

    if args.wav:
        clean = load16k(args.wav)
    else:
        clean = synth_vowels()
        sf.write(out_dir / "demo_clean.wav", clean, SR)
    clean_t = torch.from_numpy(clean).unsqueeze(0)

    # 加噪（白噪声，前 0.2s 留作噪声估计窗）
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= np.linalg.norm(clean) / (10 ** (args.snr_db / 20) * np.linalg.norm(noise) + 1e-12)
    noisy = clean + noise
    sf.write(out_dir / "demo_noisy.wav", noisy, SR)
    noisy_t = torch.from_numpy(noisy).unsqueeze(0)

    stft, istft = STFT(), ISTFT()
    spec = stft.transform(noisy_t)
    noise_mag = spec[..., : int(0.2 * SR / stft.hop)].abs().mean(-1)   # 噪声谱轮廓
    enh = istft.inverse(spectral_subtraction(spec, noise_mag), length=len(clean))
    sf.write(out_dir / "demo_enhanced.wav", enh[0].numpy(), SR)

    # 三联谱图
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (name, sig) in zip(axes, [("干净", clean), ("加噪 5dB", noisy), ("谱减增强", enh[0].numpy())]):
        s = stft.transform(torch.from_numpy(sig).unsqueeze(0).float()).abs()[0].numpy()
        ax.imshow(20 * np.log10(s + 1e-8), origin="lower", aspect="auto", cmap="magma",
                  extent=[0, len(sig) / SR, 0, SR / 2])
        ax.set_title(f"{name}   PESQ(wb) vs 干净 = {_safe_pesq(clean, sig.astype(np.float32)):.3f}"
                     if name != "干净" else f"{name}（参考）")
        ax.set_ylabel("Hz")
    axes[-1].set_xlabel("s")
    fig.suptitle("共振峰合成中文元音 /a-i-u/：加噪 -> 谱减（幅度谱减、相位保留）")
    fig.tight_layout()
    fig.savefig(out_dir / "demo_spectrograms.png", dpi=150)
    print(f"outputs -> {out_dir}")
    for name, sig in (("noisy", noisy), ("enh", enh[0].numpy())):
        sig32 = sig.astype(np.float32)
        print(f"{name:5s}  PESQ={_safe_pesq(clean, sig32):.3f}  "
              f"STOI={_safe_stoi(clean, sig32):.3f}  SI-SNR={_si_snr_np(clean, sig32):.2f}dB")


if __name__ == "__main__":
    main()
