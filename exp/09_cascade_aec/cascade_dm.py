"""级联 AEC 的 DM 数据集（P1-3b）：合成 → 线性级(MDF) → (e_lin, y_echo, near)。

复用关系（零复制）
------------------
- 合成基元：`exp/07_nn_aec/dm_dataset.AecDynamicMix`（t10）——池/清单互斥、清单采样、
  DT 引导段、防腐削波等口径全继承；
- 线性级：`exp/06_trad_aec/pipeline.TradAEC`——与 WP-06 评估、P1-1 三方对照、以及
  `precompute_cascade.py` 完全同一实现；
- 思路沿用 `exp/09_cascade_aec/README.md` 的"级联 = 同一个 NN 换一对输入"：
  NN 的 (mic, far) 输入对换成 (e_lin, y_echo)，模型/损失/训练循环一字不改。

口径（队长 2026-10-13 裁定，全部进 manifest 的 `linear_stage`）
--------------------------------------------------------------
1. **每条 clip 新建 TradAEC 实例**（``trad_instance_policy="fresh_per_clip"``）：
   DM 可复现、worker 可并行（num_workers>0 下同一 idx 的产物与 worker 数无关）。
   历史静态集 `train_cascade_lin/` 是**跨 clip 复用同一实例**（`precompute_cascade.py`
   在循环外建一次；内部 MDF 谱队列 / 对齐锁存 / DTD EMA 跨 clip 延续）⇒ 两者**不等价**，
   **历史级联读数不与新 DM 读数直接混比**（manifest 记
   ``equivalent_to_static_precompute=False`` 与说明文字）。
2. **far 平移在线性级之后**施加于 ``y_echo``（与静态级联训练一致：jitter 作用于
   `farend/`），见 `AecDynamicMix._post_synth`。
3. 线性级参数缺省与 P1-1 三方对照同款：``method="mdf", partitions=8, mu=0.7``
   （其余 TradAEC 参数走 dataclass 缺省，生效值全量记入 manifest）。
"""
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
for _d in (str(_EXP / "07_nn_aec"), str(_EXP / "06_trad_aec")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from dm_dataset import AecDynamicMix  # noqa: E402  （07：t10 的 DM 合成器）
from pipeline import TradAEC  # noqa: E402  （06；其 import 会自挂同目录 sys.path）

#: 级联线性级缺省（与 P1-1 三方对照 / precompute_cascade 同款）
LINEAR_STAGE_DEFAULTS = {"method": "mdf", "partitions": 8, "mu": 0.7}
_TRAD_FIELDS = tuple(f.name for f in fields(TradAEC))


class AecCascadeDynamicMix(AecDynamicMix):
    """DM 合成 → TradAEC(MDF) → 返回 (mic=e_lin, far=y_echo, near)。

    ``trad_kwargs`` 只允许 TradAEC 的 dataclass 字段（未知键直接报错，不静默丢弃——
    线性级超参进错值等于整批数据换口径）。
    """

    def __init__(self, *args, trad_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trad_kwargs = dict(LINEAR_STAGE_DEFAULTS if trad_kwargs is None
                                else trad_kwargs)
        unknown = sorted(set(self.trad_kwargs) - set(_TRAD_FIELDS))
        if unknown:
            raise ValueError(f"trad_kwargs 含 TradAEC 不认识的字段：{unknown}"
                             f"（可用：{list(_TRAD_FIELDS)}）")
        # 静态级联口径：线性级吃**未平移**的 far，jitter 施加在 y_echo 上（_post_synth 覆盖）
        self.linear_calls = 0

    # ---------------------------------------------------------- 线性级
    def run_linear(self, mic_r, far_r):
        """跑一次线性级（**每条 clip 新建实例**）→ (y_echo, e_lin)。

        与 `precompute_cascade.py` 逐式一致：``y_echo = res.y_echo[:n]``、
        ``e_lin = mic − y_echo``（float32 落盘/进网络）；MDF 本身确定性无随机性。
        """
        aec = TradAEC(**self.trad_kwargs)
        res = aec.process(np.asarray(mic_r, dtype=np.float64),
                          np.asarray(far_r, dtype=np.float64))
        n = min(len(mic_r), len(res.y_echo))
        y_echo = res.y_echo[:n].astype(np.float32)
        e_lin = (np.asarray(mic_r, dtype=np.float32)[:n] - y_echo).astype(np.float32)
        self.linear_calls += 1
        return y_echo, e_lin

    def _post_synth(self, mic_r, far_r, near_r, rng):
        """插入线性级：合成 → MDF → far 平移作用于 y_echo（顺序同静态级联集）。"""
        y_echo, e_lin = self.run_linear(mic_r, far_r)
        return e_lin, self._augment_far(y_echo, rng), near_r

    def linear_stage_info(self) -> dict:
        """线性级口径留痕（进 `train_manifest.json` 的 ``data.linear_stage``）。"""
        effective = asdict(TradAEC())            # dataclass 缺省全景
        effective.update(self.trad_kwargs)       # 显式覆盖后的生效值
        return {
            "kind": "cascade_mdf",
            "trad_params": effective,
            "trad_instance_policy": "fresh_per_clip",
            "equivalent_to_static_precompute": False,
            "e_lin_formula": "e_lin = mic − y_echo（TradAEC.process 的 y_echo，RES 前）",
            "far_jitter_order": "mdf_then_shift_y_echo",
            "note": ("每条 clip 新建 TradAEC 实例：与历史静态集 train_cascade_lin/ 的"
                     "跨 clip 复用同一实例**不等价**（后者 MDF 谱队列/对齐锁存/DTD EMA 跨"
                     "clip 延续）⇒ 历史级联读数不与新 DM 读数直接混比；"
                     "见 REMEDIATION_PLAN_V3.1 P1-3b 与 exp/09_cascade_aec/README.md"),
        }

    def provenance(self) -> dict:
        base = super().provenance()
        base["mode"] = "cascade_dynamic_mixing"
        base["linear_stage"] = self.linear_stage_info()
        return base
