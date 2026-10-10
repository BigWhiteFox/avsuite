# -*- coding: utf-8 -*-
"""全链路 Capstone 演示（WP-10.5）：三级处理效果 Gradio 页。

上传 mic（带回声的麦克风录音）+ far（远端参考）→
    [原始 mic] → [NN-AEC 20ms 流式] → [GTCRN 降噪] 三路 A/B 对比 + AECMOS 逐级分数。
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cascade import Cascade, SR  # noqa: E402

_session = {"cascade": None, "last": None}


def get_cascade():
    if _session["cascade"] is None:
        _session["cascade"] = Cascade()
    return _session["cascade"]


def load16k(x):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    if isinstance(x, (tuple, list)):
        sr_in, y = x
    else:
        y, sr_in = sf.read(str(x), dtype="float32", always_2d=True)
        y = y.mean(1)
    if sr_in != SR:
        g = gcd(SR, sr_in)
        y = resample_poly(y, SR // g, sr_in // g).astype("float32")
    return y.astype("float32")


def process(mic_audio, far_audio):
    if mic_audio is None or far_audio is None:
        return None, None, None, "请上传 mic 与 far（远端参考）两个音频"
    try:
        mic = load16k(mic_audio)
        far = load16k(far_audio)
        n = min(len(mic), len(far))
        sigs = get_cascade().process(mic[:n], far[:n])
        scores = get_cascade().aecmos_scores(far[:n], sigs, talk_type="dt")
        txt = "\n".join(
            f"{k}: echo {v[0]:.3f} / deg {v[1]:.3f}"
            for k, v in scores.items())
    except Exception as e:
        return None, None, None, f"处理失败：{e}"
    return ((SR, sigs["mic"]), (SR, sigs["aec"]), (SR, sigs["nr"]), txt)


def build_ui():
    import gradio as gr
    with gr.Blocks(title="全链路 AEC→NR Capstone (avsuite)") as demo:
        gr.Markdown("""
# 全链路演示：回声消除 → 噪声抑制（车载语音链）

上传**麦克风录音（含回声）**与**远端参考（扬声器信号）**：
NN-AEC（23.67K 级 20ms 流式）先消回声，GTCRN 再降残余噪声。
AECMOS 分数逐级变化 = 每一级贡献的量化证据。
""")
        with gr.Row():
            mic_in = gr.Audio(label="麦克风 mic", sources=["upload", "microphone"],
                              type="filepath")
            far_in = gr.Audio(label="远端参考 far(lpb)", type="filepath")
        with gr.Row():
            raw_out = gr.Audio(label="① 原始 mic")
            aec_out = gr.Audio(label="② NN-AEC 后")
            nr_out = gr.Audio(label="③ AEC+NR 后")
        btn = gr.Button("运行级联", variant="primary")
        score = gr.Textbox(label="AECMOS（echo 越高越好 / deg 越高越好）", interactive=False)
        btn.click(process, inputs=[mic_in, far_in],
                  outputs=[raw_out, aec_out, nr_out, score])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7861, show_error=True)
