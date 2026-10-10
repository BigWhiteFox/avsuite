# -*- coding: utf-8 -*-
"""GTCRN 推理 demo（WP-05a）：官方测试 wav + DNS 自混样本，fp32 ONNX 增强 + 指标。

    python infer_demo.py                 # 输出 outputs/demo/*.wav + 指标表
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import onnxruntime as ort  # noqa: E402

from common import SR, istft_convention, stft_convention, synth_noisy  # noqa: E402
from core.evaluate import _safe_pesq, _safe_stoi, _si_snr_np  # noqa: E402

ONNX = HERE / "outputs" / "gtcrn_dns3.sim.onnx"
OUT = HERE / "outputs" / "demo"


def enhance_onnx(sess, noisy: np.ndarray) -> np.ndarray:
    spec = stft_convention(torch.from_numpy(noisy)).numpy()
    enh = sess.run(None, {"spec": spec})[0]
    e = torch.complex(torch.from_numpy(enh[0, ..., 0]), torch.from_numpy(enh[0, ..., 1]))
    w = torch.hann_window(512).pow(0.5)
    return torch.istft(e, 512, 256, 512, w, length=len(noisy)).numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])

    cases = []
    official = Path("D:/data/repos/gtcrn-repo/test_wavs/mix.wav")
    if official.exists():
        mix, _ = sf.read(str(official), dtype="float32")
        cases.append(("official_mix", mix, None))
    noisy, clean = synth_noisy(n=SR * 4, snr_db=5.0, seed=777)
    cases.append(("dns_synth_snr5", noisy, clean))

    for name, noisy_i, clean_i in cases:
        enh = enhance_onnx(sess, noisy_i)
        sf.write(OUT / f"{name}_noisy.wav", noisy_i, SR)
        sf.write(OUT / f"{name}_enh.wav", enh, SR)
        if clean_i is not None:
            n = min(len(clean_i), len(enh))
            row = dict(
                pesq_noisy=_safe_pesq(clean_i[:n], noisy_i[:n]),
                pesq_enh=_safe_pesq(clean_i[:n], enh[:n].astype(np.float32)),
                stoi_noisy=_safe_stoi(clean_i[:n], noisy_i[:n]),
                stoi_enh=_safe_stoi(clean_i[:n], enh[:n].astype(np.float32)),
                sisnr_noisy=_si_snr_np(clean_i[:n], noisy_i[:n]),
                sisnr_enh=_si_snr_np(clean_i[:n], enh[:n]))
            print(f"{name}:")
            print(f"  PESQ  {row['pesq_noisy']:.3f} -> {row['pesq_enh']:.3f}")
            print(f"  STOI  {row['stoi_noisy']:.3f} -> {row['stoi_enh']:.3f}")
            print(f"  SISNR {row['sisnr_noisy']:.2f} -> {row['sisnr_enh']:.2f} dB")
        else:
            print(f"{name}: 无参考（试听对比 {OUT}/{name}_*.wav）")
    print(f"wavs -> {OUT}")


if __name__ == "__main__":
    main()
