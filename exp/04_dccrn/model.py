"""DCCRN 模型（WP-04 / 模块4）：Deep Complex CRN（Hu et al., ICASSP 2020 复现变体）。

结构（@16k, n_fft=512, hop=200 → F=257, T=帧数）：
    noisy 复谱 [B,2,257,T]（实数视图）
    Enc: ComplexConv2d ×4，通道 [1→32→64→96→128]，k5 s(2,2) p2
         F: 257→129→65→33→17，T: 321→161→81→41→21（4s 输入）
    CLSTM: flatten 128×17=2176 复数维 → 2 层复数 LSTM(hidden 64) → FC 还原
    Dec: ComplexConvTranspose2d ×4 + 跳连 concat → cRM [B,2,257,T]
    enh = cRM ⊙ noisy谱 → ISTFT

与论文/官方 repo 的差异（README 有完整对照表）：
    - 窗：本实现 hann(512) 全长窗（core.STFT 约束），官方 win 400 补零到 512
    - BN：实数视图 BatchNorm2d，官方用复数 LayerNorm
    - 通道 [32,64,96,128] + CLSTM hidden 64：压缩到 ~3.8M（官方 3.71M 同量级），
      8GB 显存可 batch 8-12
    - CLSTM 门激活按分量（σ/tanh 作用在 Re/Im 两路）

消融开关：real_conv=True 把复数卷积换成实数卷积（同通道，验证复数运算的价值）。
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from complex_nn import ComplexConv2d, ComplexConvTranspose2d, ComplexLSTM, complex_mul, complex_view  # noqa: E402

from core.istft import ISTFT  # noqa: E402
from core.stft import STFT  # noqa: E402

__all__ = ["DCCRN"]

ENC_CH = (32, 64, 96, 128)
DEC_CH = (64, 32, 16, 16)
CLSTM_H = 64


class DCCRN(nn.Module):
    def __init__(self, n_fft: int = 512, hop: int = 200, clstm_hidden: int = CLSTM_H,
                 clstm_layers: int = 2, real_conv: bool = False):
        super().__init__()
        self.n_fft, self.hop, self.real_conv = n_fft, hop, real_conv
        stft = STFT(n_fft, hop)
        self.register_buffer("window", stft.window, persistent=False)
        self.stft, self.istft = stft, ISTFT(n_fft, hop)

        conv_cls = ComplexConv2d
        deconv_cls = ComplexConvTranspose2d
        if real_conv:      # 消融：实数卷积直接吃 2ch 视图（无复数代数）
            # 消融分支的卷积工厂保留 lambda 闭包写法，改写成 def 属纯风格调整，豁免 E731
            conv_cls = lambda ci, co, **kw: nn.Conv2d(2 * ci, 2 * co, 5,  # noqa: E731
                                                      stride=kw.get("stride", (2, 2)),
                                                      padding=kw.get("padding", 2))
            # 同上：反卷积工厂 lambda，豁免 E731
            deconv_cls = lambda ci, co, **kw: nn.ConvTranspose2d(2 * ci, 2 * co, 5,  # noqa: E731
                                                                 stride=kw.get("stride", (2, 2)),
                                                                 padding=kw.get("padding", 2))

        chs = (1,) + ENC_CH
        self.enc = nn.ModuleList([conv_cls(chs[i], chs[i + 1], stride=(2, 2), padding=2)
                                  for i in range(4)])
        self.enc_bn = nn.ModuleList([nn.BatchNorm2d(2 * c) for c in ENC_CH])
        self.act = nn.LeakyReLU(0.1, inplace=True)

        self.f_bot = n_fft // 2 // 16 + 1          # 257→…→17
        flat = ENC_CH[-1] * self.f_bot             # 2176 复数维
        self.clstm = ComplexLSTM(flat, clstm_hidden, clstm_layers)
        self.fc = nn.Linear(2 * clstm_hidden, 2 * flat)

        dchs = ENC_CH[-1]
        dec_in = [dchs * 2, DEC_CH[0] + ENC_CH[2], DEC_CH[1] + ENC_CH[1],
                  DEC_CH[2] + ENC_CH[0]]
        self.dec = nn.ModuleList([deconv_cls(dec_in[i], DEC_CH[i], stride=(2, 2), padding=2)
                                  for i in range(4)])
        self.dec_bn = nn.ModuleList([nn.BatchNorm2d(2 * c) for c in DEC_CH])
        self.head = nn.Conv2d(2 * DEC_CH[-1], 2, 1)     # cRM Re/Im

    def forward(self, wav: torch.Tensor):
        """[B,N] -> (enh_wav [B,N], mask [B,2,F,T], noisy_spec, enh_spec)。"""
        spec = self.stft.transform(wav)                 # [B,F,T] complex
        x = complex_view(spec)                          # [B,2,F,T]
        # 帧数对齐：4 次 stride-2 下采样要求 T ≡ 1 (mod 16) 才能精确对称还原
        # （训练定长 4s=321 帧天然满足；评估任意长文件在此右侧零填充）
        T = x.shape[-1]
        T_pad = ((T - 1 + 15) // 16) * 16 + 1
        if T_pad > T:
            x = nn.functional.pad(x, (0, T_pad - T))
        skips = []
        for conv, bn in zip(self.enc, self.enc_bn):
            x = self.act(bn(conv(x)))
            skips.append(x)
        B, _, F4, T4 = x.shape
        z = x.permute(0, 3, 1, 2).reshape(B, T4, -1)    # [B,T,2·128·17]
        z = self.clstm(z)
        z = self.fc(z)                                  # [B,T,2·2176]
        x = z.reshape(B, T4, 2, ENC_CH[-1], F4).permute(0, 2, 3, 4, 1)
        x = x.reshape(B, 2 * ENC_CH[-1], F4, T4)
        for i, (dec, bn) in enumerate(zip(self.dec, self.dec_bn)):
            x = torch.cat([x, skips[3 - i]], dim=1)
            x = self.act(bn(dec(x)))
        mask = self.head(x)[..., :T]                    # [B,2,257,T]（裁回原帧数）
        enh = complex_mul(mask, spec)
        enh_wav = self.istft.inverse(enh, length=wav.shape[-1])
        return enh_wav, mask, spec, enh

    @torch.no_grad()
    def enhance_wav(self, wav: torch.Tensor) -> torch.Tensor:
        return self.forward(wav)[0]
