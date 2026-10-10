# -*- coding: utf-8 -*-
"""VoiceBank-DEMAND 数据管道（WP-03 / 模块3）。

两层结构：
1. extract_parquet()：把 HF 格式的 6 个 parquet（D:/data/voicebank-demand-16k）解包成
   wav 对目录（train/clean, train/noisy, test/clean, test/noisy）——一次性预处理，
   之后训练直接读 wav（快，且可被 AEC 合成等复用）。
2. VoiceBankDataset / speaker split：定长训练片段、按说话人前缀（p226_xxx -> p226）
   切 train/val，防止说话人泄漏。

约定：16k 单声道 float32；训练片段 SEG_SEC=3.0s，不足补零、过 3s 随机裁剪。
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000
SEG_SEC = 3.0

VB_ROOT = Path("D:/data/voicebank-demand-16k")


def extract_parquet(parquet_dir=VB_ROOT, out_dir=None, overwrite=False):
    """解包 parquet -> wav 对目录。返回 {split: {clean: dir, noisy: dir}}。

    out_dir 默认 parquet_dir/extracted。文件名用 id（如 p226_001.wav）。
    幂等：已存在且非 overwrite 则跳过。
    """
    import io

    import pyarrow.parquet as pq

    parquet_dir = Path(parquet_dir)
    out_dir = Path(out_dir or parquet_dir / "extracted")
    plan = {"train": sorted(parquet_dir.glob("train-*.parquet")),
            "test": sorted(parquet_dir.glob("test-*.parquet"))}
    result = {}
    for split, files in plan.items():
        out_clean, out_noisy = out_dir / split / "clean", out_dir / split / "noisy"
        result[split] = {"clean": out_clean, "noisy": out_noisy}
        n_done = len(list(out_clean.glob("*.wav"))) if out_clean.exists() else 0
        if n_done and not overwrite:
            continue
        for d in (out_clean, out_noisy):
            d.mkdir(parents=True, exist_ok=True)
        n = 0
        for pf in files:
            table = pq.read_table(pf, columns=["id", "clean", "noisy"])
            for i in range(table.num_rows):
                uid = table["id"][i].as_py()
                for col, out in (("clean", out_clean), ("noisy", out_noisy)):
                    rec = table[col][i].as_py()
                    (out / f"{uid}.wav").write_bytes(rec["bytes"])
                n += 1
        print(f"[extract] {split}: {n} 对 -> {out_dir/split}")
    return result


def list_pairs(clean_dir, noisy_dir):
    """配对文件列表（交集，按名字排序）。"""
    clean_dir, noisy_dir = Path(clean_dir), Path(noisy_dir)
    names = sorted({p.name for p in clean_dir.glob("*.wav")} &
                   {p.name for p in noisy_dir.glob("*.wav")})
    return [(noisy_dir / n, clean_dir / n) for n in names]


def speaker_of(filename: str) -> str:
    """p226_001.wav -> p226（测试集为 [E123, p257...] 前缀同样适用）。"""
    return Path(filename).stem.split("_")[0]


def speaker_split(pairs, val_speakers=0.1, seed=0):
    """按说话人切 train/val（说话人不相交，防泄漏）。

    val_speakers: float<1 按比例取说话人数，int 取个数。
    返回 (train_pairs, val_pairs)。
    """
    spk_to_pairs = {}
    for noisy, clean in pairs:
        spk_to_pairs.setdefault(speaker_of(noisy.name), []).append((noisy, clean))
    spks = sorted(spk_to_pairs)
    rng = np.random.default_rng(seed)
    n_val = int(round(val_speakers * len(spks))) if isinstance(val_speakers, float) else val_speakers
    n_val = max(1, min(n_val, len(spks) - 1))
    val_spks = set(rng.choice(spks, size=n_val, replace=False).tolist())
    train = [p for s in spks if s not in val_spks for p in spk_to_pairs[s]]
    val = [p for s in sorted(val_spks) for p in spk_to_pairs[s]]
    return train, val


class VoiceBankDataset(Dataset):
    """(noisy, clean) 波形对。训练时 3s 定长（随机裁剪/补零），评估取全长。

    __getitem__ 返回 dict(noisy [N], clean [N], name str)。
    """

    def __init__(self, clean_dir, noisy_dir, segment_sec=None, length=None):
        self.pairs = list_pairs(clean_dir, noisy_dir)
        if not self.pairs:
            raise FileNotFoundError(f"no paired wavs under {clean_dir} / {noisy_dir}")
        self.seg = int((segment_sec or SEG_SEC) * SR) if segment_sec != "full" else None
        self.length = length or self.seg
        if self.seg is None and length is None:
            self.length = None   # 全长模式，collate 前需逐条处理

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        y, _ = sf.read(str(path), dtype="float32", always_2d=True)
        return y.mean(1)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy, clean = self._load(noisy_path), self._load(clean_path)
        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]
        if self.length is not None:
            if len(noisy) >= self.length:
                off = np.random.randint(0, len(noisy) - self.length + 1)
                noisy, clean = noisy[off:off + self.length], clean[off:off + self.length]
            else:
                pad = self.length - len(noisy)
                noisy, clean = np.pad(noisy, (0, pad)), np.pad(clean, (0, pad))
        return {"noisy": torch.from_numpy(np.ascontiguousarray(noisy)),
                "clean": torch.from_numpy(np.ascontiguousarray(clean)),
                "name": Path(noisy_path).stem}


class PairsDataset(VoiceBankDataset):
    """(noisy, clean) 路径对 -> 定长训练片段。

    模块级定义（Windows DataLoader spawn 要求可 pickle；勿移入函数内）。
    """

    def __init__(self, pairs, segment_sec=SEG_SEC):
        self.pairs = list(pairs)
        self.seg = int(segment_sec * SR)
        self.length = self.seg

