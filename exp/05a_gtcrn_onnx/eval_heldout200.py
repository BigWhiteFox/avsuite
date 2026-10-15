"""WP-05a held-out 扩样重评（WS-C 整改）：50 -> 200 条 DNS 自混 held-out。

同一配方（DNSDynamicMix seg_sec=4.0, SNR U(-5,15)）、同 seed 777；
seed 机制为 seed*1000003+idx 逐 idx 确定 -> 前 50 条与旧集逐位一致，
扩样不换已有条目。只评已有 fp32(sim)/INT8 dynamic ONNX，不重训不覆盖旧产物。

    python eval_heldout200.py

输出：outputs/heldout200_wav/（200 条合成 wav）+ outputs/heldout200/{rows.csv,summary.json}
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from common import stft_convention  # noqa: E402

from core.evaluate import _safe_pesq, _safe_stoi, _si_snr_np  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "exp" / "04_dccrn"))
from dynamic_mixing import DNSDynamicMix  # noqa: E402

OUT = HERE / "outputs"
WAV_DIR = OUT / "heldout200_wav"
RES_DIR = OUT / "heldout200"
FP32_ONNX = OUT / "gtcrn_dns3.sim.onnx"
INT8_ONNX = OUT / "gtcrn_dns3.int8.onnx"
N = 200
SUBSET_OLD = 50
SEED = 777


def enhance(sess, noisy):
    spec = stft_convention(torch.from_numpy(noisy)).numpy()
    enh = sess.run(None, {"spec": spec})[0]
    e = torch.complex(torch.from_numpy(enh[0, ..., 0]), torch.from_numpy(enh[0, ..., 1]))
    w = torch.hann_window(512).pow(0.5)
    return torch.istft(e, 512, 256, 512, w, length=len(noisy)).numpy()


def mean_sd_ci(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    mean = x.mean()
    sd = x.std(ddof=1)
    ci = 1.96 * sd / np.sqrt(len(x))       # 大样本正态近似 95% CI
    return dict(n=int(len(x)), mean=round(mean, 4), sd=round(sd, 4),
                ci95=[round(mean - ci, 4), round(mean + ci, 4)])


def main():
    import onnxruntime as ort

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    ds = DNSDynamicMix(epoch_len=N, seg_sec=4.0, seed=SEED)
    sess32 = ort.InferenceSession(str(FP32_ONNX), providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(str(INT8_ONNX), providers=["CPUExecutionProvider"])

    rows = []
    for i in range(N):
        item = ds[i]
        noisy, clean = item["noisy"].numpy(), item["clean"].numpy()
        if i < SUBSET_OLD:
            # 抽查与旧集同源：落盘前 50 条供复核
            sf.write(str(WAV_DIR / f"noisy_{i:03d}.wav"), noisy, 16000)
            sf.write(str(WAV_DIR / f"clean_{i:03d}.wav"), clean, 16000)
        e32 = enhance(sess32, noisy).astype(np.float32)
        e8 = enhance(sess8, noisy).astype(np.float32)
        rows.append(dict(
            idx=i,
            pesq32=_safe_pesq(clean, e32), pesq8=_safe_pesq(clean, e8),
            stoi32=_safe_stoi(clean, e32), stoi8=_safe_stoi(clean, e8),
            sisnr32=_si_snr_np(clean, e32), sisnr8=_si_snr_np(clean, e8)))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{N}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "rows.csv", index=False)

    old = df[df.idx < SUBSET_OLD]
    d200 = df.pesq32 - df.pesq8
    d50 = old.pesq32 - old.pesq8
    summary = {
        "config": {"n": N, "seed": SEED, "seg_sec": 4.0, "snr_range": [-5, 15],
                   "note": "seed 逐 idx 确定(seed*1000003+idx)，前 50 条与旧 50 条集逐位一致"},
        "fp32": {"200": mean_sd_ci(df.pesq32), "first50": mean_sd_ci(old.pesq32)},
        "int8_dynamic": {"200": mean_sd_ci(df.pesq8), "first50": mean_sd_ci(old.pesq8)},
        "quant_delta": {"200": mean_sd_ci(d200), "first50": mean_sd_ci(d50)},
        "stoi": {"fp32_200": mean_sd_ci(df.stoi32), "int8_200": mean_sd_ci(df.stoi8)},
        "sisnr": {"fp32_200": mean_sd_ci(df.sisnr32), "int8_200": mean_sd_ci(df.sisnr8)},
        "old_report_50_ref": {"pesq_fp32": 1.8694, "pesq_int8": 1.8025, "pesq_drop": 0.0669,
                              "src": "outputs/quant_report_dynamic.json (run 2026-10-09)"},
    }
    out_json = RES_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {out_json}")


if __name__ == "__main__":
    main()
