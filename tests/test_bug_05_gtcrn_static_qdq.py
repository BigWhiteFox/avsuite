"""P3-1 故障注入练习 BUG-05:GTCRN 静态 QDQ 量化崩坏(SI-SNR 11.5 -> 负值)。

对应实战坑(exp/05a_gtcrn_onnx/README.md「静态 QDQ 崩坏是本 WP 最有价值的
负面结果」+ outputs/quant_report_static.json):静态 QDQ PTQ 在 GTCRN 上崩坏——
GRU 递归状态逐帧累积量化误差 + 23.67K 小模型无冗余吸收量化噪声 +
LayerNorm/Tanh 对 scale 敏感 → 输出变噪声。50 条 DNS 自混实测:
PESQ 1.869→1.120、STOI 0.877→0.197、SI-SNR 11.54→-20.8(dynamic PTQ 只掉 0.26)。

注入方式:零注入——崩坏产物已留档(outputs/gtcrn_dns3.int8_static_BAD.onnx,
「崩坏对照,留档」),与 fp32 sim 版同段音频 onnxruntime 推理对比。
观察到的失败模式(实测,DNSDynamicMix seed=777 与 quantize.py 同源,3 条 4s):
    clip0: sim 4.610  -> BAD -12.517(掉 17.1dB)
    clip1: sim 10.041 -> BAD  -5.071(掉 15.1dB)
    clip2: sim 10.369 -> BAD  -6.674(掉 17.0dB)
    (与 outputs/quant_accuracy_rows.csv 前三行逐条一致)
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "exp" / "05a_gtcrn_onnx"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import sqrt_hann  # noqa: E402

SIM_ONNX = HERE / "outputs" / "gtcrn_dns3.sim.onnx"
BAD_ONNX = HERE / "outputs" / "gtcrn_dns3.int8_static_BAD.onnx"


def _enhance(sess, noisy, w):
    import torch
    # 官方 STFT 约定 (B,F,T,2)，与 exp/05a common.stft_convention 数值逐位一致；
    # 但走 torch>=2.x 推荐写法：return_complex=True 再 view_as_real 取实虚视图。
    # 原因：return_complex=False 已弃用（"stft with return_complex=False is
    # deprecated ... will raise an error"，实测 torch 2.6.0 每次调用都告警），
    # 未来版本会直接报错。等价性已实测：同一输入下两种写法 bitwise equal。
    x = torch.from_numpy(noisy)
    spec = torch.view_as_real(
        torch.stft(x, 512, 256, 512, w, return_complex=True)).unsqueeze(0).numpy()
    enh = sess.run(None, {"spec": spec})[0]                    # (1,F,T,2)
    e = torch.complex(torch.from_numpy(enh[0, ..., 0]), torch.from_numpy(enh[0, ..., 1]))
    return torch.istft(e, 512, 256, 512, w, length=len(noisy)).numpy()


def test_bug05_static_qdq_si_snr_collapse():
    """静态 QDQ 产物 vs fp32 sim:同段音频 SI-SNR 从正值塌到负值,差值方向断言。"""
    if not (SIM_ONNX.exists() and BAD_ONNX.exists()):
        pytest.skip("缺少 exp/05a 产物(gtcrn_dns3.sim.onnx / int8_static_BAD.onnx)")
    import numpy as np
    import onnxruntime as ort

    from core.evaluate import _si_snr_np

    sys.path.insert(0, str(HERE.parents[1] / "exp" / "04_dccrn"))
    try:
        from dynamic_mixing import DNSDynamicMix
        ds = DNSDynamicMix(epoch_len=3, seg_sec=4.0, seed=777)   # quantize.py 同源
        _ = ds[0]
    except FileNotFoundError as e:
        # 只吞"本机 DNS 素材缺失"这一类（DNSDynamicMix.__init__ 对空 glob 显式抛
        # FileNotFoundError）。历史写法 `except Exception` 会把真实回归——例如加载
        # 逻辑被改坏后抛 KeyError/IndexError、或 wav 采样率断言失败——一并静默 skip，
        # 使本用例在坏机器上永远"绿"。其余异常必须外抛。
        pytest.skip(f"DNS 本机数据缺失:{e}")

    sess32 = ort.InferenceSession(str(SIM_ONNX), providers=["CPUExecutionProvider"])
    sessBAD = ort.InferenceSession(str(BAD_ONNX), providers=["CPUExecutionProvider"])
    w = sqrt_hann()

    pairs = []
    for i in range(3):
        item = ds[i]
        noisy, clean = item["noisy"].numpy(), item["clean"].numpy()
        s32 = float(_si_snr_np(clean, _enhance(sess32, noisy, w)))
        sBAD = float(_si_snr_np(clean, _enhance(sessBAD, noisy, w).astype(np.float32)))
        pairs.append((s32, sBAD))

    for i, (s32, sBAD) in enumerate(pairs):
        assert s32 > 3.0, f"clip{i}: fp32 sim SI-SNR {s32:.2f} 应为正基线(实测 4.6-10.4)"
        assert sBAD < 0.0, f"clip{i}: 静态 QDQ SI-SNR {sBAD:.2f} 应塌到负值(实测 -5~-12.5)"
        assert s32 - sBAD > 10.0, \
            f"clip{i}: SI-SNR 掉点 {s32 - sBAD:.2f}dB 应 >10dB(实测 15-17dB)"
