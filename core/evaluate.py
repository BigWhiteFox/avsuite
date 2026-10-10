# -*- coding: utf-8 -*-
"""mini 评估（WP-02E / 模块2.5）：增强目录 vs 参考目录 -> PESQ(wb)/STOI/SI-SNR 的 CSV。

协议约定（后续所有模块统一沿用，报告指标时必须带口径）：
- 一律 16k 单声道：读入后立体声转单声道、重采样到 16k（scipy resample_poly）；
- 长度对齐：截断到两者较短者（不补零——PESQ/STOI 对 padding 段敏感）；
- 按文件名配对（.wav/.flac，取交集），PESQ 用 wideband（16k）协议；
- 静音/无语音活动的文件 PESQ 会抛异常 -> 该项记 NaN，不拖垮整批评估。
"""
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

__all__ = ["evaluate_dir", "load16k"]

SR = 16000
AUDIO_EXTS = (".wav", ".flac")


def load16k(path) -> np.ndarray:
    """读音频 -> float32 单声道 16k。"""
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        y = resample_poly(y, SR // g, sr // g).astype("float32")
    return y


def _si_snr_np(ref, est, eps=1e-8):
    ref = ref - ref.mean()
    est = est - est.mean()
    ref_e = max(np.sum(ref ** 2), eps)
    proj = np.sum(est * ref) / ref_e * ref
    noise = est - proj
    return float(10 * np.log10(max(np.sum(proj ** 2), eps) / max(np.sum(noise ** 2), eps)))


def _safe_pesq(ref, est):
    try:
        from pesq import pesq as pesq_fn
        return float(pesq_fn(SR, ref, est, "wb"))
    except Exception:
        return float("nan")  # NoUtterancesError / 采样率不匹配等


def _safe_stoi(ref, est):
    try:
        from pystoi import stoi as stoi_fn
        return float(stoi_fn(ref, est, SR, extended=False))
    except Exception:
        return float("nan")


def evaluate_dir(est_dir, ref_dir, out_csv=None) -> pd.DataFrame:
    """评估目录对，返回逐文件指标 DataFrame（列：file / pesq_wb / stoi / si_snr / samples）。"""
    est_dir, ref_dir = Path(est_dir), Path(ref_dir)
    est_files = {p.name: p for p in est_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS}
    ref_files = {p.name: p for p in ref_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS}
    rows = []
    for name in sorted(set(est_files) & set(ref_files)):
        est, ref = load16k(est_files[name]), load16k(ref_files[name])
        n = min(len(est), len(ref))
        est, ref = est[:n], ref[:n]
        rows.append({"file": name, "pesq_wb": _safe_pesq(ref, est),
                     "stoi": _safe_stoi(ref, est), "si_snr": _si_snr_np(ref, est),
                     "samples": n})
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df
