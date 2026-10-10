# -*- coding: utf-8 -*-
"""窗长(20/32/64ms) × 窗型(hann/hamming/blackman) 消融（WP-01）。

每个配置报告三件事：
  1. 频率分辨率：主瓣宽度（null-to-null，Hz）——决定能否分开相邻谐波；
  2. COLA 偏差：Σ_t w[n-t·hop] 的归一后波动——"不加归一化直接 overlap-add"的误差上界；
  3. 往返重构误差：STFT->ISTFT（WOLA 归一）后的最大绝对误差——应始终 ~1e-7 量级。
并输出"短时傅里叶的时间模糊"对比图：1kHz 纯音突发的起振沿在不同窗长下的拖尾。

用法：python window_ablation.py
输出：outputs/window_ablation.png + stdout 的 markdown 表
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root
from core.istft import ISTFT  # noqa: E402
from core.stft import STFT, make_window  # noqa: E402

try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

SR = 16000
WIN_MS = (20, 32, 64)
WINDOWS = ("hann", "hamming", "blackman")


def mainlobe_width_hz(window: torch.Tensor, sr: int) -> float:
    """-6dB 主瓣宽度（Hz）。比"找第一零点"稳健（零点附近数值抖动会让游走提前停）。"""
    spec = torch.fft.fft(window, n=1 << 16).abs()
    peak = int(spec.argmax())
    half = spec[peak] * 0.5
    left = peak
    while left > 0 and spec[left] > half:
        left -= 1
    right = peak
    while right < len(spec) - 1 and spec[right] > half:
        right += 1
    return sr * (right - left) / len(spec)


def cola_deviation(window: torch.Tensor, hop: int) -> float:
    """Σ_t w[n-t·hop] 在内部区域的归一波动（0 = 完美 COLA）。"""
    n_fft = window.numel()
    n = 64 * hop + n_fft
    s = torch.zeros(n)
    for t in range(0, n - n_fft + 1, hop):
        s[t:t + n_fft] += window
    inner = s[n_fft:-n_fft]
    return float((inner.max() - inner.min()) / inner.mean())


def roundtrip_err(n_fft: int, hop: int, window: str) -> float:
    torch.manual_seed(0)
    wav = torch.randn(1, 16000, dtype=torch.float64)
    rec = ISTFT(n_fft, hop, window, center=True).inverse(
        STFT(n_fft, hop, window, center=True).transform(wav))
    return float((rec - wav[..., :rec.shape[-1]]).abs().max())


def main():
    torch.manual_seed(0)
    rows = []
    print("| 窗长 | 窗型 | n_fft/hop | -6dB主瓣宽(Hz) | COLA偏差 | 往返误差(max, f64) |")
    print("|---|---|---|---|---|---|")
    for ms in WIN_MS:
        n_fft = ms * SR // 1000          # 320 / 512 / 1024（20ms 非二次幂，rfft 一样可用）
        hop = n_fft // 2
        for wname in WINDOWS:
            w = make_window(n_fft, wname)
            ml, cola, rt = mainlobe_width_hz(w, SR), cola_deviation(w, hop), roundtrip_err(n_fft, hop, wname)
            rows.append((ms, wname, ml, cola, rt))
            print(f"| {ms}ms | {wname} | {n_fft}/{hop} | {ml:.1f} | {cola:.2e} | {rt:.2e} |")

    # 时间模糊对比：1kHz 纯音在 t=0.2s 突发，看起振沿被窗拖多宽
    t = torch.arange(16000) / SR
    burst = 0.5 * torch.sin(2 * np.pi * 1000 * t) * (t >= 0.2)
    fig, axes = plt.subplots(len(WIN_MS), 1, figsize=(10, 2.2 * len(WIN_MS)), sharex=True)
    for ax, ms in zip(axes, WIN_MS):
        n_fft = ms * SR // 1000
        spec = STFT(n_fft, n_fft // 2).transform(burst.unsqueeze(0)).abs()[0]
        bin_1k = int(round(1000 / SR * n_fft))
        env = spec[bin_1k - 1: bin_1k + 2].max(0).values.numpy()
        t_frames = (np.arange(len(env)) * n_fft // 2) / SR
        ax.plot(t_frames, env / env.max() + 0.0, marker="o", ms=3)
        ax.axvline(0.2, color="r", ls="--", lw=1, label="真实起振 t=0.2s")
        ax.set_ylabel(f"{ms}ms窗\n1kHz幅度")
        ax.set_ylim(0, 1.3)
        ax.legend(loc="right", fontsize=8)
    axes[-1].set_xlabel("时间 (s)")
    fig.suptitle("窗长 vs 时间分辨率：起振沿拖尾 ≈ 窗长（频率分辨率与时间分辨率的交换）")
    fig.tight_layout()
    out = Path(__file__).parent / "outputs" / "window_ablation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
