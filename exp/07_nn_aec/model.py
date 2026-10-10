"""NN-AEC 模型（WP-07 / 模块7，简历核心）：远端参考 + 麦克风 双输入复数谱 CRN 压缩版。

设计约束（对标 AEC-Challenge 2022+ 规格与车载部署）：
- 参数 <1M（WP-06 传统基线的对手要能在车机跑）
- **时间维严格因果**：conv 时间轴只左 padding、LSTM 单向、STFT hop=320(20ms)——
  架构级流式就绪（cache 化的流式推理走 WP-05b 模式，本 WP 交付离线等价实现，
  因果性由单测锁死：扰动未来帧过去输出逐位不变）
- 输入 4 通道：mic 复谱 Re/Im + far(对齐后)复谱 Re/Im——far 参考让网络自己学
  回声路径（替代线性滤波），时延鲁棒性靠训练期 ±帧扰动（WP-07D 管道同款思想）

结构：Enc 3×CausalComplexConv [4→8→16→32] k5 s(2,1)（F:257→33，T 不变）
      → LSTM(32·33=1056, hidden 96, 1 层单向) → FC 还原
      → Dec 3×CausalDeconv + 跳连 → head cRM(2ch) → enh = M ⊙ mic谱
"""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.istft import ISTFT  # noqa: E402
from core.stft import STFT, spec_from_2ch  # noqa: E402

__all__ = ["NNAEC"]

N_FFT, HOP = 512, 320          # 32ms 窗 / 20ms 帧率（12ms 窗内前瞻，README 记口径）
ENC_CH = (8, 16, 32)
LSTM_H = 96


class CausalConvBlock(nn.Module):
    """频率轴 stride-2 下采样、时间轴严格因果。

    conv 路径：时间轴左 pad 4（k=5）→ 输出帧只依赖当前+过去 4 帧。
    deconv 路径：对称 ConvTranspose 的 out[t] 依赖 in[t-2..t+2]（超前 2 帧）——
    右移 2 帧（前补零+裁尾）把依赖变成 in[t-4..t]，保持因果；流式启动期
    前 2 帧输出为零（warmup 瞬态，与 WP-05b cache 方案语义一致）。
    """

    def __init__(self, cin, cout, deconv=False):
        super().__init__()
        cls = nn.ConvTranspose2d if deconv else nn.Conv2d
        # conv 路径：时间轴 padding=0（手动左 pad 4 实现因果，conv 再 pad 会双重
        # 填充每层漂 +4 帧）；deconv 路径：对称 pad 2，输出右移 2 帧因果化
        pad = (2, 2) if deconv else (2, 0)
        self.conv = cls(cin, cout, (5, 5), stride=(2, 1), padding=pad)
        self.deconv = deconv
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout)

    def forward(self, x):
        if not self.deconv:
            x = nn.functional.pad(x, (4, 0, 0, 0))       # 时间轴左 pad 4（因果）
            x = self.conv(x)
        else:
            T_in = x.shape[-1]
            x = self.conv(x)                              # T_out = T_in + 1
            x = nn.functional.pad(x, (2, 0, 0, 0))[..., :T_in]   # 右移 2 = 因果化
        return self.act(self.bn(x))


class NNAEC(nn.Module):
    def __init__(self, n_fft=N_FFT, hop=HOP, lstm_hidden=LSTM_H):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        f = n_fft // 2 + 1
        for _ in ENC_CH:
            f = (f - 1) // 2 + 1
        self.f_bot = f                                    # 33

        chs = (4,) + ENC_CH
        self.enc = nn.ModuleList([CausalConvBlock(chs[i], chs[i + 1]) for i in range(3)])
        self.lstm = nn.LSTM(ENC_CH[-1] * f, lstm_hidden, 1, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, ENC_CH[-1] * f)
        dec_in = [ENC_CH[-1] * 2, ENC_CH[2] + ENC_CH[1], ENC_CH[1] + ENC_CH[0]]
        self.dec = nn.ModuleList([CausalConvBlock(dec_in[i], ENC_CH[2 - i], deconv=True)
                                  for i in range(3)])
        self.head = nn.Conv2d(ENC_CH[0], 2, 1)
        self.stft = STFT(n_fft, hop)
        self.istft = ISTFT(n_fft, hop)

    def forward(self, mic: torch.Tensor, far: torch.Tensor):
        """mic/far: [B,N] 波形（far 已粗对齐，训练期加帧级抖动）。

        返回 (enh_wav, mask, mic_spec, enh_spec)。
        """
        mic_spec = self.stft.transform(mic)
        far_spec = self.stft.transform(far)
        mask, enh = self.forward_spec(mic_spec, far_spec)
        enh_wav = self.istft.inverse(enh, length=mic.shape[-1])
        return enh_wav, mask, mic_spec, enh

    def forward_spec(self, mic_spec: torch.Tensor, far_spec: torch.Tensor):
        """网络主体（谱进谱出）：训练主路径与流式逐帧对拍的公共基座。"""
        # 通道优先布局 [B,4,F,T]：mic(Re,Im) + far(Re,Im)
        # （注意 core.spec_to_2ch 是 [B,F,T,2] 末维布局，conv 输入需 stack 到 dim1）
        x = torch.cat([torch.stack([mic_spec.real, mic_spec.imag], dim=1),
                       torch.stack([far_spec.real, far_spec.imag], dim=1)], dim=1)
        skips = []
        for conv in self.enc:
            x = conv(x)
            skips.append(x)
        B, C, Fb, T = x.shape
        z, _ = self.lstm(x.reshape(B, C * Fb, T).transpose(1, 2))
        x = self.fc(z).transpose(1, 2).reshape(B, C, Fb, T)
        for i, dec in enumerate(self.dec):
            x = torch.cat([x, skips[2 - i]], dim=1)
            x = dec(x)
        mask = self.head(x)                                   # cRM [B,2,F,T]
        m = spec_from_2ch(mask.permute(0, 2, 3, 1).contiguous())
        enh = m * mic_spec
        return mask, enh

    @torch.no_grad()
    def enhance(self, mic: torch.Tensor, far: torch.Tensor) -> torch.Tensor:
        return self.forward(mic, far)[0]
