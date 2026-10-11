"""DNSMOS 无参考 MOS 预测（WP-09/10 共用）。

微软 DNS-Challenge 官方 ONNX 模型（sig_bak_ovr = P.835 式 SIG/BAK/OVR 三分量）。
下载：GitHub raw（走代理，3 次重试）缓存到 D:/data/models/dnsmos/。
无参考场景（真实录音没有 clean 参考）用 OVR；Gradio demo 里即此用途。

协议（V5 wave-C：对齐官方 dnsmos_local.py 口径）：
- 16k 单声道 float32；固定输入 9.01s（144160 样本）；
- **短于 9.01s**：按官方做法"**重复自身**"补齐（`while len < len_samples: append(self)`），
  不是零补——零补会让尾部静音段把 OVR 拉低/偏移（实测同一 8s clip 差 0.12 OVR）；
- **长音频**：官方 hop = fs（1s）滑窗，段数 `int(floor(len/fs) − 9.01) + 1`，逐段打分后取均值；
- **校准**：模型 raw 输出经官方三条 polyfit 映射为 SIG/BAK/OVR（raw 值同时以
  `*_raw` 保留，便于对照未校准口径）。
"""
import time
from pathlib import Path

import numpy as np

SR = 16000
INPUT_LENGTH = 9.01         # 官方 dnsmos_local.py 的固定输入时长（秒）
FIXED_LEN = 144160          # sig_bak_ovr.onnx 固定输入长 144160 样本（9.01s）
URL = ("https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/"
       "DNSMOS/DNSMOS/sig_bak_ovr.onnx")
CACHE = Path("D:/data/models/dnsmos/sig_bak_ovr.onnx")
PROXY = "http://127.0.0.1:7890"

# 官方三条 polyfit 校准系数（DNS-Challenge DNSMOS/dnsmos_local.py::ComputeScore）
# 修正前本仓直接返回 raw 三段（无校准），实测同一 8s clip ovr 3.166 vs 官方 2.814。
_POLY_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
_POLY_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_POLY_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])


def _repeat_pad(audio: np.ndarray, n: int) -> np.ndarray:
    """官方填充口径：不足 n 时**重复自身**直到够长（非零补）。

    注意空数组会死循环——调用方（`DNSMOS.score`）已保证长度 >= SR//2，
    这里再兜一次底。
    """
    if len(audio) == 0:
        return np.zeros(n, dtype=audio.dtype)
    while len(audio) < n:
        audio = np.append(audio, audio)
    return audio


def _download():
    import requests
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(3):
        try:
            r = requests.get(URL, proxies={"http": PROXY, "https": PROXY}, timeout=60)
            if r.status_code == 200 and len(r.content) > 100_000:
                CACHE.write_bytes(r.content)
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def _ensure_model():
    if not CACHE.exists() or CACHE.stat().st_size < 100_000:
        if not _download():
            return None
    return str(CACHE)


class DNSMOS:
    """DNSMOS 封装：score(wav) -> dict(ovr, sig, bak)。模型不可用时 score 返回 None。"""

    def __init__(self):
        self.sess = None
        self.input_name = None
        self.model_path = _ensure_model()
        if self.model_path:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(self.model_path,
                                             providers=["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name

    @property
    def available(self) -> bool:
        return self.sess is not None

    def score(self, wav: np.ndarray) -> dict:
        """wav -> {ovr, sig, bak}（已校准）+ {ovr_raw, sig_raw, bak_raw}。

        官方口径（dnsmos_local.py）：不足 9.01s 先"重复自身"补足，再按 hop=1s
        取 `int(floor(len/fs) − 9.01) + 1` 段，逐段打分后取均值。
        """
        if not self.available or len(wav) < SR // 2:
            return None
        audio = np.asarray(wav, dtype=np.float64)
        audio = _repeat_pad(audio, FIXED_LEN)                # 官方：重复自身补齐
        num_hops = int(np.floor(len(audio) / SR) - INPUT_LENGTH) + 1
        outs = []
        for idx in range(max(num_hops, 1)):
            seg = audio[int(idx * SR):int((idx + INPUT_LENGTH) * SR)]
            if len(seg) < FIXED_LEN:
                continue
            outs.append(self._run(seg))
        if not outs:                                         # 兜底：至少评一段
            outs.append(self._run(audio[:FIXED_LEN]))
        return {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}

    def _run(self, seg: np.ndarray) -> dict:
        """单段打分：模型 raw 三段 + 官方 polyfit 校准三段（raw 以 *_raw 保留）。"""
        seg = np.asarray(seg, dtype=np.float32)
        if len(seg) < FIXED_LEN:                             # 官方：重复自身（非零补）
            seg = _repeat_pad(seg, FIXED_LEN)
        x = seg[None, :FIXED_LEN]                            # [B,N]
        sig, bak, ovr = self.sess.run(None, {self.input_name: x})[0][0]
        return {"sig": float(_POLY_SIG(sig)), "bak": float(_POLY_BAK(bak)),
                "ovr": float(_POLY_OVR(ovr)),
                "sig_raw": float(sig), "bak_raw": float(bak), "ovr_raw": float(ovr)}
