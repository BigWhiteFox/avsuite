"""WP-02E 单测：多档 SNR 合成退化工况下指标单调合理 + 协议行为（重采样/对齐/CSV）。"""
import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from core.evaluate import evaluate_dir

SNRS = [20, 10, 0, -10]     # 10dB 步进，单调性判定才干净


def _pseudo_speech(n=16000, sr=16000, f0=170.0):
    """类语音信号：声门脉冲串 + 共振峰滤波 + 音节级 AM（无静音段，PESQ 需语音活动）。

    纯谐波音对 PESQ 几乎无响应（~1.0 平坦），共振峰结构才能驱动它。
    """
    t = np.arange(n) / sr
    # sin²(2π·4t) 的周期是 1/(2·4)=1/8 s ⇒ 包络调制率 8Hz（不是 4Hz）
    env = 0.4 + 0.6 * np.sin(2 * np.pi * 4.0 * t) ** 2          # 8Hz 音节包络
    phase = 2 * np.pi * f0 * t
    src = sum(np.sin(h * phase) / h ** 1.2 for h in (1, 2, 3, 4, 5, 6)).astype(np.float64)
    y = src
    for f, bw in ((500, 100), (1500, 150), (2500, 200)):        # 粗共振峰级联
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        y = lfilter([1 - r], [1, -2 * r * np.cos(theta), r ** 2], y)
    return (0.2 * env * y).astype(np.float32)


def _make_dirs(tmp_path):
    ref = tmp_path / "ref"
    ref.mkdir()
    clean = _pseudo_speech()
    sf.write(ref / "a.wav", clean, 16000)
    sf.write(ref / "b.wav", (clean * 0.8).astype(np.float32), 16000)
    per_snr = {}
    for snr in SNRS:
        est = tmp_path / f"est_{snr}db"
        est.mkdir()
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(len(clean)).astype(np.float32)
        noise *= np.linalg.norm(clean) / (10 ** (snr / 20) * np.linalg.norm(noise) + 1e-12)
        sf.write(est / "a.wav", (clean + noise).astype(np.float32), 16000)
        sf.write(est / "b.wav", (clean * 0.8 + noise).astype(np.float32), 16000)
        per_snr[snr] = evaluate_dir(est, ref)
    return per_snr


def test_metrics_monotonic_with_snr(tmp_path):
    """SNR 从 20 -> -10（SNRS 列表四档）：SI-SNR 严格单调降；PESQ/STOI 容差内单调降。"""
    per_snr = _make_dirs(tmp_path)
    si = [per_snr[s]["si_snr"].mean() for s in SNRS]
    assert all(si[i] > si[i + 1] + 8 for i in range(len(SNRS) - 1))   # 每 -10dB 至少掉 8dB
    pe = [per_snr[s]["pesq_wb"].mean() for s in SNRS]
    assert all(pe[i] >= pe[i + 1] - 0.3 for i in range(len(SNRS) - 1))
    st = [per_snr[s]["stoi"].mean() for s in SNRS]
    assert all(st[i] >= st[i + 1] - 0.05 for i in range(len(SNRS) - 1))
    assert all(0.9 <= p < 4.7 for p in pe)          # PESQ(wb) 量程 sanity
    assert all(0.0 <= s <= 1.0 for s in st)         # STOI 量程 sanity


def test_no_nan_on_active_signals(tmp_path):
    per_snr = _make_dirs(tmp_path)
    for df in per_snr.values():
        assert not df[["pesq_wb", "stoi", "si_snr"]].isna().any().any()
        assert (df["samples"] == 16000).all()


def test_resample_and_length_alignment(tmp_path):
    """8k 参考自动重采样（12000 采样点 @8k -> 24000 @16k）；est 更短 -> 截断到较短者。"""
    ref, est = tmp_path / "ref", tmp_path / "est"
    ref.mkdir(), est.mkdir()
    clean = _pseudo_speech()
    sf.write(ref / "x.wav", (clean[:12000] * 0.5).astype(np.float32), 8000)   # 8k + 更短
    sf.write(est / "x.wav", clean, 16000)                                     # 16k 全长
    df = evaluate_dir(est, ref)
    assert len(df) == 1
    assert df.iloc[0]["samples"] == 16000          # min(24000, 16000) 对齐
    assert np.isfinite(df.iloc[0][["pesq_wb", "stoi", "si_snr"]].astype(float)).all()


def test_csv_written_and_unpaired_files_skipped(tmp_path):
    ref, est = tmp_path / "ref", tmp_path / "est"
    ref.mkdir(), est.mkdir()
    clean = _pseudo_speech()
    sf.write(ref / "a.wav", clean, 16000)
    sf.write(est / "a.wav", (clean * 0.9).astype(np.float32), 16000)
    sf.write(est / "only_est.wav", clean, 16000)   # 无配对 -> 应被忽略
    csv = tmp_path / "metrics.csv"
    df = evaluate_dir(est, ref, out_csv=csv)
    assert len(df) == 1 and csv.exists()
    assert list(df.columns) == ["file", "pesq_wb", "stoi", "si_snr", "samples"]
