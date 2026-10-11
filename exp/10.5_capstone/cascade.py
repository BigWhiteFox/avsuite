"""全链路 Capstone 级联器（WP-10.5）：NN-AEC(20ms 流式) → GTCRN NR。

    mic + far(lpb) ──► [NN-AEC 流式，WP-07+WP-05b] ──► [GTCRN ONNX，WP-05a] ──► 最终输出
    20ms 块进块出的 AEC 级（真流式）+ 整段批处理的 NR 级（离线；官方 stream/ 版
    可替换为全流式，README 注明）。

输出三路信号（原始 / AEC 后 / AEC+NR 后）+ AECMOS echo/deg 两路分数（逐级变化）
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

# NN-AEC 权重口径：与 docs/NUMBERS.md「统一决策（2026-10-10 定版）」对齐——
# NN-AEC 主配置 = v4（γ0.5 / w_sisnr 1.0 / DT-NST 专注 / 240ep），对外只报 v4，
# v3 及更早（outputs_v3/...）仅作探索矩阵存档，capstone 不加载。
# 绝对路径：D:\Project_for_Python\DSH\avsuite\exp\07_nn_aec\outputs_v4\nnaec_g0.5_j10\last.ckpt
# 来源：exp/07_nn_aec/train.py 主配置 v4 训练产物（2026-10-11 V4 整改波次；同目录
# best.ckpt 被 deg_mechanism.py / ood_real_echoes_eval.py 引用，last.ckpt 被
# official_aecmos_eval.py / leadin_crop_eval.py 引用，加载口径一致）。
NNAEC_CKPT = ROOT / "exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/last.ckpt"
GTCRN_ONNX = ROOT / "exp/05a_gtcrn_onnx/outputs/gtcrn_dns3.sim.onnx"
SR = 16000
CHUNK = 320                                    # 20ms
AECMOS_ONNX = "D:/data/repos/AEC-Challenge/AECMOS/AECMOS_local/Run_1668423760_Stage_0.onnx"
_AECMOS_SINGLETON = {"est": None}


def _get_aecmos():
    """AECMOS 估计器进程内单例（t20：不做每请求重建）。"""
    if _AECMOS_SINGLETON["est"] is None:
        sys.path.insert(0, "D:/data/repos/AEC-Challenge/AECMOS/AECMOS_local")
        from aecmos import AECMOSEstimator
        _AECMOS_SINGLETON["est"] = AECMOSEstimator(AECMOS_ONNX)
    return _AECMOS_SINGLETON["est"]


def _resample_to_model_sr(y, sr_in, sr_out):
    """重采样到 AECMOS 模型采样率（t20 R1 口径，与 aecmos_eval.py 同法）。

    官方 AECMOS_local/aecmos.py 对 Run_1668423760 设 sampling_rate=48000，其官方入口
    用 librosa.load(sr=sampling_rate) 统一重采样；本仓 16k 管线需显式对齐。
    """
    if int(sr_in) == int(sr_out):
        return y
    from math import gcd

    from scipy.signal import resample_poly
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(y, int(sr_out) // g, int(sr_in) // g).astype(np.float32)


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

    def aecmos_scores(self, lpb: np.ndarray, signals: dict, talk_type="dt", sr=SR):
        """三路信号的 AECMOS (echo_mos, deg_mos)——三级效果的量化演示。

        t20（R1）修复三点：
        1. **mic/enh 传参**：官方签名是 `run(talk_type, lpb, mic, enh)`——`mic` 是**未处理的
           原始 mic**（退化参照），`enh` 才是本轮输出。旧实现把同一路 sig 同时当 mic 与 enh
           传入（`est.run(talk_type, lpb[:n], sig[:n], sig[:n])`），等于拿本级输出与自身比，
           分阶段 deg 分数失去含义（恒等于"零处理"参照）。现显式取 `signals["mic"]` 作 mic。
        2. **采样率对齐**：三路在送评前统一重采样到模型采样率（Run_1668423760 = 48000），
           口径与 `exp/07_nn_aec/aecmos_eval.py::resample_to_model_sr` 一致（gcd 约简 +
           scipy.signal.resample_poly，16k→48k 即 (3,1)）；此前 16k 直送 48k 模型会系统偏移。
        3. **估计器单例**：AECMOSEstimator 只在进程内构造一次（不做每请求重建）。
        """
        est = _get_aecmos()
        sr_model = int(getattr(est, "sampling_rate", sr))
        mic_raw = np.asarray(signals["mic"], dtype=np.float32)   # 退化参照（未处理 mic）
        out = {}
        for name, sig in signals.items():
            n = min(len(lpb), len(mic_raw), len(sig))            # 三路截齐
            a = _resample_to_model_sr(np.asarray(lpb[:n], dtype=np.float32), sr, sr_model)
            m = _resample_to_model_sr(mic_raw[:n], sr, sr_model)
            e = _resample_to_model_sr(np.asarray(sig[:n], dtype=np.float32), sr, sr_model)
            out[name] = est.run(talk_type, a, m, e)
        return out
