"""DCCRN 的 DM 适配（P1-5a）：吃**与 NN-AEC / 级联同一条** DM 供给，输出 04 的契约。

为什么要它（t12 发现）
----------------------
`exp/04_dccrn/train.py` 原来是旧数据接口：`DNSDynamicMix` 把池写死在
`D:/data/dns-challenge-mirror/datasets/{clean,noise}`，**绕过 `split_lists/` 的分割清单**
⇒ 训练会吃掉 holdout15 reader 与 noise_eval15（隔离性失效、泄漏风险）。
本模块让 04 能像 07 一样**清单圈定**：`--near-list/--far-list/--noise-list` 显式给池内
成员，入口做 near/far 互斥与 noise∩speech 校验，清单版本号入 manifest。

语义（口径，需与收档同页写）
--------------------------
- **数据 = AEC 合成三元组**（`dm_dataset.AecDynamicMix`，与 07/09 同一实现、同一 rng 流），
  按 04 现有 Dataset 契约投影：**`noisy = mic`**（含回声 + 噪声的麦克风信号）、
  **`clean = near`**（近端目标）；**`far`（远端参考）一并返回**，DCCRN 不消费它（保留给
  审计/多输入变体）。
  ⇒ 阶段1 的 DCCRN 是"复数谱掩码模型在**同一条新数据轴**上做回声+噪声抑制"，与
  NN-AEC/级联同供给、同 val、同配比口径，三者可比（t11 的吞吐表也是按同一供给折算的）。
  这不是 04 历史的 DNS 去噪任务（历史读数 2.1811 PESQ@devtest 仍属 DNS 口径，两套口径
  不混比）；要回到 DNS 去噪任务需要另一条数据轴（干净语音 × 噪声池），不在本任务范围。
- **不动的部分**：模型（`model.DCCRN`）、损失（`core.losses`）、优化器/调度骨架、
  断点续训三件套。本模块只改"数据从哪来、以什么键交给模型"。

key 映射（写进 manifest 的 `key_map`，可回读）：
    noisy <- mic    clean <- near    far <- far（不消费，仅留痕）
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_EXP = Path(__file__).resolve().parent.parent
_EXP07 = str(_EXP / "07_nn_aec")
_TEMP_INSERT = _EXP07 not in sys.path
if _TEMP_INSERT:
    sys.path.insert(0, _EXP07)
try:
    from dm_dataset import AecDynamicMix  # noqa: E402
finally:
    # 只在导入期需要 07 的路径：导入完就摘掉，避免把 exp/07 永久挂在 sys.path 上
    # 遮蔽 03/04 的裸名 model/train/dataset（conftest §3.5 的同类问题）
    if _TEMP_INSERT and _EXP07 in sys.path:
        sys.path.remove(_EXP07)

#: DCCRN 的「等效更新步数」参照（方案 a 的目标）：04 README §结果 的主结果口径
#: = 100 epochs × epoch_len 1500 / bs 8 → floor(1500/8) = 187 步/ep → **18700 步**
#: （`outputs/dccrn` 是更长的续训谱系：203ep × 187 = 37961 步，不作对照锚）。
DCCRN_REF = {
    "name": "dccrn_hist",
    "steps": 18700,
    "epochs": 100,
    "batch_size": 8,
    "steps_per_epoch": 187,
    "source": "exp/04_dccrn/README.md §结果（100 epochs × epoch_len 1500，bs8；主结果 "
              "PESQ 2.1811 口径）",
    "note": "outputs/dccrn 含更长谱系（203ep/37961 步），不用于对照锚",
}

#: 04 的 Dataset 契约 → DM 三元组的键映射
KEY_MAP = {"noisy": "mic", "clean": "near", "far": "far"}


class DccrnDynamicMix(AecDynamicMix):
    """AEC DM 三元组 → 04 契约（`noisy/clean`，外加 `far` 留痕）。

    除键名映射外，采样/清单/互斥/留痕全部继承 `AecDynamicMix`（=07/09 同一实现）；
    因此同 seed 下 DCCRN 与 NN-AEC 看到**逐位相同**的合成内容（可配对对照）。
    """

    def __getitem__(self, idx):
        it = super().__getitem__(idx)
        it["noisy"] = it["mic"]
        it["clean"] = it["near"]
        return it

    def provenance(self) -> dict:
        p = super().provenance()
        p["mode"] = "dccrn_aec_dynamic_mixing"
        p["key_map"] = dict(KEY_MAP)
        p["note"] = ("DCCRN 与 NN-AEC/级联同一条 DM 供给（AEC 三元组投影：noisy=mic、"
                     "clean=near；far 仅留痕不消费）。历史 DNS 去噪读数（devtest PESQ "
                     "2.1811）属另一口径，不与本数据轴混比。")
        return p


class AecValAsDccrn(Dataset):
    """静态验证集适配：aec_synth 目录（farend/nearend/mic + manifest）→ 04 契约。

    与 07 的 `dataset.AecDataset` 读同一份目录/同一列（不 jitter：val 口径恒定），
    只把 mic/near 投影成 noisy/clean。DM 训练的 val 保持**静态**（与 07 一致），
    这样三个模型的"模型选择口径"可比。
    """

    def __init__(self, synth_dir, scenarios=None):
        import csv
        synth_dir = Path(synth_dir)
        with open(synth_dir / "manifest.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if scenarios:
            rows = [r for r in rows if r["scenario"] in scenarios]
        if not rows:
            raise FileNotFoundError(f"{synth_dir}: manifest 空或缺失")
        self.dir, self.rows = synth_dir, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import soundfile as sf
        name = self.rows[idx]["name"]
        mic = sf.read(self.dir / "mic" / f"{name}.wav", dtype="float32")[0]
        near = sf.read(self.dir / "nearend" / f"{name}.wav", dtype="float32")[0]
        far = sf.read(self.dir / "farend" / f"{name}.wav", dtype="float32")[0]
        n = min(len(mic), len(near), len(far))
        return {"noisy": torch.from_numpy(np.ascontiguousarray(mic[:n])),
                "clean": torch.from_numpy(np.ascontiguousarray(near[:n])),
                "far": torch.from_numpy(np.ascontiguousarray(far[:n])),
                "name": name, "scenario": self.rows[idx]["scenario"]}
