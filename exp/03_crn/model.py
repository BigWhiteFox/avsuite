# -*- coding: utf-8 -*-
"""CRN 模型（WP-03 / 模块3）：卷积编码 + 循环层 + 对称解码 + 复数比率掩码。

结构（Tan & Wang 2018 CRN 变体，@16k n_fft=512）：
    输入 noisy 复谱 Re/Im [B, 2, F=257, T]
    Enc: Conv(2→18) → Conv(18→30) → Conv(30→40)，kernel(5,5)、stride(2,1)、pad(2,2)
         —— 频率 257→129→65→33 降采样，时间维度不变（流式友好的关键设计）
    RNN: 频率×通道 flatten(40×33=1320) → 2 层 BLSTM(hidden=256) → Linear → 还原 [B,40,33,T]
    Dec: Deconv ×3 上采样回 257，跳连 concat encoder 特征
    Mask: 1×1 conv 输出 cRM Re/Im（线性无界；--bounded 可选 tanh 界定）
    增强: enh = mask ⊙ noisy（复数乘）

消融开关（ablation.py / train.py 暴露）：
    mask_type: 'crm'（复数掩码，2 通道，默认）| 'irm'（幅度掩码，1 通道 sigmoid，保留带噪相位）
    rnn_type:  'blstm'（默认，双向，非因果）| 'lstm'（单向，因果/流式友好）
"""
import torch
from torch import nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.istft import ISTFT          # noqa: E402
from core.stft import STFT, spec_from_2ch, spec_to_2ch  # noqa: E402

__all__ = ["CRN"]


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, deconv=False):
        cls = nn.ConvTranspose2d if deconv else nn.Conv2d
        super().__init__(cls(cin, cout, (5, 5), stride=(2, 1), padding=(2, 2)),
                         nn.ReLU(inplace=True))


class CRN(nn.Module):
    """CRN 增强模型。

    Args:
        n_fft/hop:    STFT 参数（16k 默认 512/256）。
        hidden:       每层 RNN 隐单元数（默认 256，~5M 参数；512 ≈ 论文规模 ~17M）。
        rnn_layers:   循环层数（默认 2）。
        rnn_type:     'blstm' | 'lstm'。
        mask_type:    'crm' | 'irm'。
        bounded_mask: True 时 cRM 输出过 1.55·tanh（降低估计空间；默认线性无界）。
        skip:         decoder 是否 concat encoder 跳连（默认 True）。
    """

    ENC_CH = (18, 30, 40)

    def __init__(self, n_fft: int = 512, hop: int = 256, hidden: int = 256,
                 rnn_layers: int = 2, rnn_type: str = "blstm",
                 mask_type: str = "crm", bounded_mask: bool = False, skip: bool = True):
        super().__init__()
        if rnn_type not in ("blstm", "lstm"):
            raise ValueError(f"rnn_type: {rnn_type}")
        if mask_type not in ("crm", "irm"):
            raise ValueError(f"mask_type: {mask_type}")
        self.n_fft, self.hop = n_fft, hop
        self.mask_type, self.skip = mask_type, skip
        f_rnn = n_fft // 2 + 1                       # 257
        for _ in self.ENC_CH:                        # 三次频率降采样后的 bin 数
            f_rnn = (f_rnn - 1) // 2 + 1             # k=5 p=2 s=2 -> (F-1)//2+1
        self.f_rnn = f_rnn                           # 33
        ch_last = self.ENC_CH[-1]                    # 40

        e1, e2, e3 = self.ENC_CH
        self.enc = nn.ModuleList([ConvBlock(2, e1), ConvBlock(e1, e2), ConvBlock(e2, e3)])
        self.rnn = nn.LSTM(ch_last * f_rnn, hidden, rnn_layers, batch_first=True,
                           bidirectional=(rnn_type == "blstm"))
        rnn_out = hidden * (2 if rnn_type == "blstm" else 1)
        self.fc = nn.Linear(rnn_out, ch_last * f_rnn)
        self.dec = nn.ModuleList([ConvBlock(ch_last, e2, deconv=True),
                                  ConvBlock(e2 * (2 if skip else 1), e1, deconv=True),
                                  ConvBlock(e1 * (2 if skip else 1), e1, deconv=True)])
        mask_ch = 2 if mask_type == "crm" else 1
        self.head = nn.Conv2d(e1, mask_ch, 1)
        self.bounded = bounded_mask

        self.stft = STFT(n_fft, hop)
        self.istft = ISTFT(n_fft, hop)

    def _mask_and_enhance(self, mask: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        """mask [B,ch,F,T] ⊙ noisy 复谱 [B,F,T] -> 增强复谱。"""
        if self.mask_type == "irm":
            m = torch.sigmoid(mask).squeeze(1)            # [B,F,T] 幅度增益 (0,1)
            enh = torch.complex(m * noisy.real, m * noisy.imag)   # 保留带噪相位
            return enh, m
        if self.bounded:
            mask = 1.55 * torch.tanh(mask)
        m = spec_from_2ch(mask.permute(0, 2, 3, 1).contiguous())   # [B,F,T] 复数
        return m * noisy, m

    def forward(self, wav: torch.Tensor):
        """输入波形 [B, N] -> (enh_wav [B, N_in], mask, noisy_spec, enh_spec)。"""
        noisy = self.stft.transform(wav)                  # [B,F,T]
        x = spec_to_2ch(noisy).permute(0, 3, 1, 2)        # [B,2,F,T]
        skips = []
        for conv in self.enc:
            x = conv(x)
            skips.append(x)
        b, c, f, t = x.shape
        z, _ = self.rnn(x.reshape(b, c * f, t).transpose(1, 2))     # [B,T,rnn_out]
        x = self.fc(z).transpose(1, 2).reshape(b, c, f, t)
        x = self.dec[0](x)                                          # [B,30,65,T]
        x = torch.cat([x, skips[1]], dim=1) if self.skip else x
        x = self.dec[1](x)                                          # [B,18,129,T]
        x = torch.cat([x, skips[0]], dim=1) if self.skip else x
        x = self.dec[2](x)                                          # [B,18,257,T]
        mask = self.head(x)                                         # [B,ch,257,T]
        enh_spec, m = self._mask_and_enhance(mask, noisy)
        enh_wav = self.istft.inverse(enh_spec, length=wav.shape[-1])
        return enh_wav, m, noisy, enh_spec

    @torch.no_grad()
    def enhance_wav(self, wav: torch.Tensor) -> torch.Tensor:
        """推理入口：波形进波形出（单条/批均可）。"""
        return self.forward(wav)[0]
