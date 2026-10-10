# -*- coding: utf-8 -*-
"""全链路 Capstone 级联器（WP-10.5）：NN-AEC(20ms 流式) → GTCRN NR。

    mic + far(lpb) ──► [NN-AEC 流式，WP-07+WP-05b] ──► [GTCRN ONNX，WP-05a] ──► 最终输出
    20ms 块进块出的 AEC 级（真流式）+ 整段批处理的 NR 级（离线；官方 stream/ 版
    可替换为全流式，README 注明）。

输出三路信号（原始 / AEC 后 / AEC+NR 后）+ AECMOS 三分量（echo/deg 逐级变化）
——"三级处理效果"的量化演示，对车载团队的说服力超过任何单个模块。
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp" / "07_nn_aec"))
sys.path.insert(0, str(ROOT / "exp" / "05a_gtcrn_onnx"))

from streaming import NNAECStreaming  # noqa: E402  (按路径直载 model，无名字冲突)

NNAEC_CKPT = ROOT / "exp/07_nn_aec/outputs_v3/nnaec_g0.5_j10/last.ckpt"
GTCRN_ONNX = ROOT / "exp/05a_gtcrn_onnx/outputs/gtcrn_dns3.sim.onnx"
SR = 16000
CHUNK = 320                                    # 20ms


def load_nnaec(ckpt_path=None):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cap_model", ROOT / "exp/07_nn_aec/model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.NNAEC().eval()
    st = torch.load(ckpt_path or NNAEC_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(st["model"])
    return m


class Cascade:
    """NN-AEC（流式）→ GTCRN（批）级联。模型各加载一次。"""

    def __init__(self, nnaec_ckpt=None, gtcrn_onnx=None, device="cpu"):
        self.device = device
        self.aec = NNAECStreaming(load_nnaec(nnaec_ckpt), device=device)
        import onnxruntime as ort
        self.nr = ort.InferenceSession(str(gtcrn_onnx or GTCRN_ONNX),
                                       providers=["CPUExecutionProvider"])

    def run_aec(self, mic: np.ndarray, far: np.ndarray) -> np.ndarray:
        """流式 NN-AEC：20ms 块进块出。返回 aec 输出（与 mic 等长）。

        每次调用前重置流状态（Gradio 连续请求 / 测试复用同一级联器都需要）。
        """
        self.aec.reset()
        outs = []
        for i in range(0, len(mic), CHUNK):
            outs.append(self.aec.process_chunk(mic[i:i + CHUNK], far[i:i + CHUNK]))
        tail = self.aec.flush()
        if len(tail):
            outs.append(tail)
        out = np.concatenate(outs)
        # 与离线对齐：流输出前 n_fft//2 是 pad 区，中心裁剪取 mic 长度
        start = 512 // 2
        if len(out) >= start + len(mic):
            return out[start: start + len(mic)]
        return np.pad(out[start:], (0, len(mic) - (len(out) - start)))

    def run_nr(self, aec_out: np.ndarray) -> np.ndarray:
        """GTCRN ONNX 整段批处理（sqrt-hann 约定，见 WP-05a common）。"""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cap_common", ROOT / "exp/05a_gtcrn_onnx/common.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        spec_in = mod.stft_convention(torch.from_numpy(aec_out.astype(np.float32))).numpy()
        enh = self.nr.run(None, {"spec": spec_in})[0]
        e = torch.complex(torch.from_numpy(enh[0, ..., 0]),
                          torch.from_numpy(enh[0, ..., 1]))
        w = torch.hann_window(512).pow(0.5)
        return torch.istft(e, 512, 256, 512, w, length=len(aec_out)).numpy().astype(np.float32)

    def process(self, mic: np.ndarray, far: np.ndarray):
        """完整级联：返回 dict(mic, aec, nr)。"""
        aec_out = self.run_aec(mic.astype(np.float32), far.astype(np.float32))
        nr_out = self.run_nr(aec_out)
        return {"mic": mic, "aec": aec_out, "nr": nr_out}

    def aecmos_scores(self, lpb: np.ndarray, signals: dict, talk_type="dt"):
        """三路信号的 AECMOS (echo_mos, deg_mos)——三级效果的量化演示。"""
        sys.path.insert(0, "D:/data/repos/AEC-Challenge/AECMOS/AECMOS_local")
        from aecmos import AECMOSEstimator
        est = AECMOSEstimator("D:/data/repos/AEC-Challenge/AECMOS/AECMOS_local/"
                              "Run_1668423760_Stage_0.onnx")
        out = {}
        for name, sig in signals.items():
            n = min(len(lpb), len(sig))
            out[name] = est.run(talk_type, lpb[:n], sig[:n], sig[:n])
        return out
