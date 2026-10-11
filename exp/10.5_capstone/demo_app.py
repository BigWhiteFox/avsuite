"""全链路 Capstone 演示（WP-10.5）：三级处理效果 Gradio 页。

上传 mic（带回声的麦克风录音）+ far（远端参考）→
    [原始 mic] → [NN-AEC 20ms 流式] → [GTCRN 降噪] 三路 A/B 对比 + AECMOS 逐级分数。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cascade import SR, Cascade  # noqa: E402

_session = {"cascade": None, "last": None}


def get_cascade():
    if _session["cascade"] is None:
        _session["cascade"] = Cascade()
    return _session["cascade"]


def load16k(x):
    from math import gcd

    import soundfile as sf
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


def _fmt_score(v):
    """AECMOS 分数格式化（t20 加固）。

    正常返回 (echo_mos, deg_mos) 二元组；但若上游返回标量/长度不足，
    旧写法 `v[0]` 会抛 "float is not subscriptable" 并被外层 except 吞掉，
    连三级音频一起丢掉（v3b-054 描述的失败模式）。这里显式容错：
    只有拿得到二元组才格式化成 echo/deg，否则原样显示。
    """
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        return f"echo {float(v[0]):.3f} / deg {float(v[1]):.3f}"
    return f"{v}"


def process(mic_audio, far_audio):
    if mic_audio is None or far_audio is None:
        return None, None, None, "请上传 mic 与 far（远端参考）两个音频"
    # 阶段 1：三级处理——失败则确实没有音频可返回
    try:
        mic = load16k(mic_audio)
        far = load16k(far_audio)
        n = min(len(mic), len(far))
        sigs = get_cascade().process(mic[:n], far[:n])
    except Exception as e:
        return None, None, None, f"处理失败：{e}"
    # 阶段 2：AECMOS 评分——失败只降级文本，三级音频照常返回（t20 加固）
    try:
        scores = get_cascade().aecmos_scores(far[:n], sigs, talk_type="dt")
        txt = "\n".join(f"{k}: {_fmt_score(v)}" for k, v in scores.items())
    except Exception as e:
        txt = f"音频已处理，但 AECMOS 评分失败：{type(e).__name__}: {e}"
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
