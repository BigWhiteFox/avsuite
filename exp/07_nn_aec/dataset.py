"""NN-AEC 数据管道（WP-07）：aec_synth 产物目录 -> (mic, far, near) 训练三元组。

训练期时延扰动（WORKLIST：复用 WP-07D 思想）：far 参考相对文件内对齐再随机
平移 ±jitter_ms——逼模型学"残余时延鲁棒"，对标车载链路时延漂移（蓝牙/缓冲）。
"""
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000


class AecDataset(Dataset):
    """aec_synth 输出目录（farend/nearend/mic + manifest.csv）。

    jitter_ms: far 随机平移范围（±），0 关闭。
    scenarios: 可选过滤（如只训 DT）。
    """

    def __init__(self, synth_dir, jitter_ms: float = 10.0, scenarios=None, seed=None):
        synth_dir = Path(synth_dir)
        import csv
        rows = list(csv.DictReader(open(synth_dir / "manifest.csv", encoding="utf-8")))
        if scenarios:
            rows = [r for r in rows if r["scenario"] in scenarios]
        self.dir = synth_dir
        self.rows = rows
        self.jitter = int(jitter_ms * SR / 1000)
        self.seed = seed
        if not rows:
            raise FileNotFoundError(f"{synth_dir}: manifest 空或缺失")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        name = r["name"]
        mic = sf.read(self.dir / "mic" / f"{name}.wav", dtype="float32")[0]
        far = sf.read(self.dir / "farend" / f"{name}.wav", dtype="float32")[0]
        near = sf.read(self.dir / "nearend" / f"{name}.wav", dtype="float32")[0]
        n = min(len(mic), len(far), len(near))
        mic, far, near = mic[:n], far[:n], near[:n]
        if self.jitter and n > 2 * self.jitter:
            rng = (np.random.default_rng(self.seed * 1000003 + idx)
                   if self.seed is not None else np.random.default_rng())
            k = int(rng.integers(-self.jitter, self.jitter + 1))
            far = _shift(far, k)
        return {"mic": torch.from_numpy(mic.copy()), "far": torch.from_numpy(far.copy()),
                "near": torch.from_numpy(near.copy()), "name": name,
                "scenario": r["scenario"]}


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """正 k 右移（延迟增大），负 k 左移；越界补零。"""
    out = np.zeros_like(x)
    if k >= 0:
        out[k:] = x[:len(x) - k]
    else:
        out[:len(x) + k] = x[-k:]
    return out
