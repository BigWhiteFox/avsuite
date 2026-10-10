"""WP-05a 公共工具：模型加载 / STFT 约定 / 测试信号。

官方推理约定（infer.py 同款，必须严格一致否则权重对不上）：
    spec = torch.stft(wav, 512, 256, 512, hann(512)**0.5, return_complex=False)
    enh  = model(spec[None])           # (B,F,T,2) -> (B,F,T,2) 复数谱实虚视图
    wav  = torch.istft(enh, 512, 256, 512, hann(512)**0.5)
sqrt-hann 双端各乘一次 = 整窗 hann 的 WOLA 重构（与 core/istft 的窗平方归一等价路径）。
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))                      # repo root

from gtcrn import GTCRN  # noqa: E402

CKPT_DNS3 = Path("D:/data/repos/gtcrn/checkpoints/model_trained_on_dns3.tar")
CKPT_VCTK = Path("D:/data/repos/gtcrn/checkpoints/model_trained_on_vctk.tar")
OFFICIAL_STREAM_ONNX = Path("D:/data/repos/gtcrn-repo/stream/onnx_models/gtcrn.onnx")
SR = 16000


def sqrt_hann(n=512):
    return torch.hann_window(n).pow(0.5)


def load_model(ckpt=CKPT_DNS3, device="cpu") -> GTCRN:
    model = GTCRN().to(device).eval()
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return model


def stft_convention(wav: torch.Tensor) -> torch.Tensor:
    """[N] 或 [B,N] -> (B,F,T,2) 实虚视图（官方约定，逐条 stft 后堆叠）。"""
    w = sqrt_hann().to(wav.device)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return torch.stack([torch.stft(x, 512, 256, 512, w, return_complex=False)
                        for x in wav])


def istft_convention(spec: torch.Tensor, length=None) -> torch.Tensor:
    """(B,F,T,2) -> [B,N]（官方约定；torch>=2.x 的 istft 只吃 complex，
    这里把实虚视图转回 complex——旧版 torch 的 return_complex=False 等价路径）。"""
    w = sqrt_hann().to(spec.device)
    outs = []
    for s in spec:
        sc = torch.complex(s[..., 0], s[..., 1])      # (F,T) complex
        outs.append(torch.istft(sc, 512, 256, 512, w, length=length))
    return torch.stack(outs)


def synth_noisy(n=SR * 4, snr_db=5.0, seed=0):
    """合成带噪测试信号（谐波语音 + DNS 风格噪声），返回 (noisy, clean) float32。"""
    sys.path.insert(0, str(HERE.parents[1] / "exp" / "04_dccrn"))
    from dynamic_mixing import DNSDynamicMix
    ds = DNSDynamicMix(epoch_len=1, seg_sec=n / SR, seed=seed)
    item = ds[0]
    return item["noisy"].numpy(), item["clean"].numpy()
