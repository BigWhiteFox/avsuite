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
    """SI-SNR（dB，numpy 版）。

    口径（V5 wave-C 加固）：
    - **入口长度对齐**：按较短者截断（此前不做对齐，长度不等时 numpy 广播会直接
      抛 ValueError；evaluate_dir 自己先截断过，但直接调用方没有这层保护）；
    - **静音/塌陷 -> NaN**：ref 或 est 全零（静音参考 / 增强塌陷）时返回 NaN，
      与 PESQ 的 NaN 协议一致。修正前 est 全零会得到 **0.0dB** 的假"完美"分
      （proj=0、noise=0，分子分母同时被 eps 兜底 ⇒ 比值 1）。
    """
    ref = np.asarray(ref, dtype=np.float64).ravel()
    est = np.asarray(est, dtype=np.float64).ravel()
    n = min(len(ref), len(est))
    if n == 0:
        return float("nan")
    ref, est = ref[:n], est[:n]
    if not np.any(ref) or not np.any(est):
        return float("nan")
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
