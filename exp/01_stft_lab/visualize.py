"""STFT 四联图：幅度(dB) / 相位 / 实部 / 虚部（WP-01）。

用法：
    python visualize.py                       # 无数据依赖：合成调频谐波演示信号
    python visualize.py --wav your.wav        # 用自己的语音
输出：exp/01_stft_lab/outputs/stft_viz.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core.stft import STFT  # noqa: E402

try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def synth_demo(sr=16000, dur=1.0):
    """调频谐波+幅度调制，覆盖频率扫掠与瞬态包络，便于读图。"""
    t = torch.arange(int(sr * dur)) / sr
    f0 = 150 + 100 * torch.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * torch.cumsum(f0, 0) / sr
    wav = 0.3 * torch.sin(phase) + 0.15 * torch.sin(2 * phase) + 0.08 * torch.sin(3 * phase)
    env = 0.55 + 0.45 * torch.sin(2 * np.pi * 2.0 * t)
    return (wav * env).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None, help="16k(或任意采样率) wav/flac 路径")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs" / "stft_viz.png"))
    args = ap.parse_args()

    if args.wav:
        import soundfile as sf
        y, sr = sf.read(args.wav, dtype="float32", always_2d=True)
        wav = torch.from_numpy(y.mean(1)).unsqueeze(0)
    else:
        wav, sr = synth_demo(), 16000

    stft = STFT(n_fft=512, hop=256)
    spec = stft.transform(wav)[0].numpy()          # [F, T] complex
    mag_db = 20 * np.log10(np.abs(spec) + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    panels = [
        (axes[0][0], mag_db, "幅度 (dB)", "viridis"),
        (axes[0][1], np.angle(spec), "相位 (rad, [-π, π] 卷绕)", "twilight"),
        (axes[1][0], spec.real, "实部 Re", "RdBu_r"),
        (axes[1][1], spec.imag, "虚部 Im", "RdBu_r"),
    ]
    hop_s = stft.hop / sr
    for ax, data, title, cmap in panels:
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap,
                       extent=[0, spec.shape[1] * hop_s, 0, sr / 2])
        ax.set_title(title)
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("频率 (Hz)")
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(f"STFT  n_fft=512(32ms) hop=256(16ms)  sr={sr}", fontsize=13)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
