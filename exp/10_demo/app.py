# -*- coding: utf-8 -*-
"""Gradio 在线 Demo（WP-10 / 模块10）：上传带噪音频 -> GTCRN 增强 -> A/B 对比 + DNSMOS。

设计要点：
- 模型加载一次复用（模块级单例）；GTCRN ONNX 默认（WP-05a 产物），可切 CRN/DCCRN 权重
- 无参考场景只给 DNSMOS（有参考才 PESQ——WORKLIST 明示）
- 任意采样率输入自动重采样 16k；手机浏览器友好（Gradio 默认响应式）

本地运行：python app.py（http://127.0.0.1:7860）
HF Spaces 部署：README（gradio sdk，requirements 复用仓库根 requirements）
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SR = 16000
DEFAULT_ONNX = ROOT / "exp/05a_gtcrn_onnx/outputs/gtcrn_dns3.sim.onnx"

_session = {"sess": None, "dnsmos": None}


def get_sessions():
    import onnxruntime as ort
    if _session["sess"] is None:
        _session["sess"] = ort.InferenceSession(str(DEFAULT_ONNX),
                                                providers=["CPUExecutionProvider"])
    return _session["sess"]


def get_dnsmos():
    if _session["dnsmos"] is None:
        from dnsmos import DNSMOS
        _session["dnsmos"] = DNSMOS()
    return _session["dnsmos"]


def load16k(path_or_tuple):
    """Gradio Audio 返回 (sr, ndarray)；文件路径输入则直接读。"""
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    if isinstance(path_or_tuple, (tuple, list)):
        sr_in, y = path_or_tuple
    else:
        y, sr_in = sf.read(str(path_or_tuple), dtype="float32", always_2d=True)
        y = y.mean(1)
    if sr_in != SR:
        g = gcd(SR, sr_in)
        y = resample_poly(y, SR // g, sr_in // g).astype("float32")
    return y.astype("float32")


def enhance(noisy: np.ndarray) -> np.ndarray:
    import torch
    if "exp05a_common" not in sys.modules:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "exp05a_common", ROOT / "exp/05a_gtcrn_onnx/common.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["exp05a_common"] = mod
        spec.loader.exec_module(mod)
    from exp05a_common import stft_convention
    spec = stft_convention(torch.from_numpy(noisy)).numpy()
    enh = get_sessions().run(None, {"spec": spec})[0]
    e = torch.complex(torch.from_numpy(enh[0, ..., 0]), torch.from_numpy(enh[0, ..., 1]))
    w = torch.hann_window(512).pow(0.5)
    return torch.istft(e, 512, 256, 512, w, length=len(noisy)).numpy().astype("float32")


def process(audio):
    if audio is None:
        return None, "请先上传音频"
    try:
        y = load16k(audio)
        enh = enhance(y)
    except Exception as e:  # 网络/模型缺失等：友好报错，不崩页面
        return None, f"增强失败：{e}"
    d = get_dnsmos().score(enh)
    dnsmos_txt = (f"DNSMOS  ovr {d['ovr']:.3f} | sig {d['sig']:.3f} | bak {d['bak']:.3f}"
                  if d else "DNSMOS 模型未下载（需网络）")
    return (SR, enh), dnsmos_txt


def build_ui():
    import gradio as gr
    with gr.Blocks(title="语音增强 GTCRN Demo (avsuite)") as demo:
        gr.Markdown("""
# 语音增强 Demo — GTCRN (23.67K 参数，DNS3 官方权重)

上传 16k（或任意采样率，自动重采样）单声道音频 -> ONNX 推理增强。
A/B 对比试听 + 无参考 DNSMOS 分（OVR=整体质量）。
""")
        with gr.Row():
            audio_in = gr.Audio(label="带噪音频", sources=["upload", "microphone"],
                                type="filepath")
            audio_out = gr.Audio(label="增强结果")
        btn = gr.Button("增强", variant="primary")
        score = gr.Textbox(label="DNSMOS（无参考）", interactive=False)
        btn.click(process, inputs=[audio_in], outputs=[audio_out, score])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, show_error=True)
