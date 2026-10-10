# -*- coding: utf-8 -*-
"""DNSMOS 无参考 MOS 预测（WP-09/10 共用）。

微软 DNS-Challenge 官方 ONNX 模型（sig_bak_ovr = P.835 式 SIG/BAK/OVR 三分量）。
下载：GitHub raw（走代理，3 次重试）缓存到 D:/data/models/dnsmos/。
无参考场景（真实录音没有 clean 参考）用 OVR；Gradio demo 里即此用途。

协议：16k 单声道 float32；长音频按 10s 滑动窗分块评分后取均值
（模型按固定时长设计，整段喂入会 OOM/失真）。
"""
import time
from pathlib import Path

import numpy as np

SR = 16000
FIXED_LEN = 144160          # sig_bak_ovr.onnx 固定输入长 144160 样本（9.01s）
URL = ("https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/"
       "DNSMOS/DNSMOS/sig_bak_ovr.onnx")
CACHE = Path("D:/data/models/dnsmos/sig_bak_ovr.onnx")
PROXY = "http://127.0.0.1:7890"


def _download():
    import requests
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
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
        if not self.available or len(wav) < SR // 2:
            return None
        wav = np.asarray(wav, dtype=np.float32)
        win, hop = FIXED_LEN, FIXED_LEN // 2
        outs = []
        if len(wav) <= win:
            outs.append(self._run(wav))
        else:
            for i in range(0, len(wav) - win + 1, hop):
                outs.append(self._run(wav[i:i + win]))
        m = {k: float(np.mean([o[k] for o in outs])) for k in outs[0]}
        return m

    def _run(self, seg: np.ndarray) -> dict:
        if len(seg) < FIXED_LEN:                     # 模型固定输入长：右侧零补
            seg = np.pad(seg, (0, FIXED_LEN - len(seg)))
        x = seg[None, :FIXED_LEN]                    # [B,N]
        sig, bak, ovr = self.sess.run(None, {self.input_name: x})[0][0]
        return {"sig": sig, "bak": bak, "ovr": ovr}
