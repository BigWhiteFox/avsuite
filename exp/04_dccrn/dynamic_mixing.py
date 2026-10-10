"""DNS 子集数据管道 + 动态混合 Dynamic Mixing（WP-04 / 模块4）。

Dynamic Mixing（Koizumi et al. 2019，DNS 冠军方案标配）：
每个 batch 在线随机配对 干净语音×噪声、随机 SNR、随机裁剪——等效把 1 万条
干净语音 × 660 条噪声组合成 ~660 万种混合，远大于静态预混数据集，
是 DCCRN 达标 PESQ≥2.6 的前提（WORKLIST 明示）。

响度/重采样约定（管道级纪律）：
- 一律 16k 单声道 float32（DNS 素材原生 16k，加载时断言）；
- 干净语音 RMS 归一到 U(-35,-20) dBFS（模拟录音电平多样性）；
- 噪声按目标 SNR=U(-5,15)dB 相对归一后的语音缩放；
- 语音短于片段则拼接自身，噪声短则 tile；片段默认 4s；
- 混合峰值 >0.99 时整体缩放（防 int16 写出削波；训练内为 float 无此虑，
  但保持与 wav 落盘评估一致的习惯）。

模块级类定义（Windows DataLoader spawn 需要可 pickle）。
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000
DNS_ROOT = Path("D:/data/dns-challenge-mirror/datasets")


def load_16k(path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, f"{path}: 期望 16k，实际 {sr}（管道纪律：入口断言）"
    return y.mean(1)


def fit_length(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """裁到 n：长则随机裁剪，短则自拼接。"""
    if len(x) >= n:
        off = rng.integers(0, len(x) - n + 1)
        return x[off:off + n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float):
    """按 SNR 混合（噪声缩放到目标功率差）。返回 (noisy, clean, noise_used)。"""
    pc = np.mean(clean ** 2) + 1e-12
    pn = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(pc / pn / (10 ** (snr_db / 10)))
    noise = noise * scale
    return clean + noise, clean, noise


class DNSDynamicMix(Dataset):
    """虚拟数据集：__len__ = epoch_len，每次取样现场随机配对混合（DM 精髓）。

    seed=None：每 worker 独立随机（训练）；seed 固定：可复现（val）。
    """

    def __init__(self, clean_dir=DNS_ROOT / "clean", noise_dir=DNS_ROOT / "noise",
                 epoch_len: int = 2000, seg_sec: float = 4.0,
                 snr_range=(-5.0, 15.0), level_range=(-35.0, -20.0), seed=None):
        self.clean_files = sorted(Path(clean_dir).glob("*.wav"))
        self.noise_files = sorted(Path(noise_dir).glob("*.wav"))
        if not self.clean_files or not self.noise_files:
            raise FileNotFoundError(f"DNS 素材缺失: {clean_dir} / {noise_dir}")
        self.epoch_len = epoch_len
        self.n = int(seg_sec * SR)
        self.snr_range, self.level_range = snr_range, level_range
        self.seed = seed

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        if self.seed is not None:
            rng = np.random.default_rng(self.seed * 1000003 + idx)   # 逐 idx 确定
        else:
            rng = np.random.default_rng()
        clean = fit_length(load_16k(rng.choice(self.clean_files)), self.n, rng)
        noise = fit_length(load_16k(rng.choice(self.noise_files)), self.n, rng)
        # 响度归一：clean RMS -> U(level_range) dBFS
        target = rng.uniform(*self.level_range)
        clean = clean * (10 ** ((target - rms_db(clean)) / 20))
        snr = rng.uniform(*self.snr_range)
        noisy, clean, _ = mix_at_snr(clean, noise, snr)
        peak = np.abs(noisy).max()
        if peak > 0.99:
            noisy, clean = noisy * (0.99 / peak), clean * (0.99 / peak)
        return {"noisy": torch.from_numpy(noisy.copy()),
                "clean": torch.from_numpy(clean.copy()),
                "name": f"dm_{idx}"}


class StaticPairs(Dataset):
    """--no-dm 消融用：预生成固定混合对（seed 定死，epoch 间不变）。"""

    def __init__(self, clean_dir=DNS_ROOT / "clean", noise_dir=DNS_ROOT / "noise",
                 n_pairs: int = 2000, seg_sec: float = 4.0, seed=0):
        rng = np.random.default_rng(seed)
        clean_files = sorted(Path(clean_dir).glob("*.wav"))
        noise_files = sorted(Path(noise_dir).glob("*.wav"))
        n = int(seg_sec * SR)
        self.items = []
        for _i in range(n_pairs):
            clean = fit_length(load_16k(rng.choice(clean_files)), n, rng)
            noise = fit_length(load_16k(rng.choice(noise_files)), n, rng)
            clean = clean * (10 ** ((rng.uniform(-35, -20) - rms_db(clean)) / 20))
            noisy, clean, _ = mix_at_snr(clean, noise, rng.uniform(-5, 15))
            self.items.append((torch.from_numpy(noisy.copy()),
                               torch.from_numpy(clean.copy())))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        noisy, clean = self.items[i]
        return {"noisy": noisy, "clean": clean, "name": f"static_{i}"}
