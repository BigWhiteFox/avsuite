"""复数神经网络层（WP-04 / 模块4）：复数卷积 / 复数 LSTM。

复数张量的存储约定：全部用**实数张量双通道拼接** (B, 2C, ...) / (B, T, 2D)，
前 2C 维的前半是 Re、后半是 Im——torch 的 nn.Conv2d/nn.Linear 不吃 complex dtype，
且实数视图对 ONNX/量化友好（WP-05a 伏笔）。

复数线性算子的定义（Trabelsi et al., ICLR 2018）：
    (W_re + j·W_im)(x_re + j·x_im)
      = (W_re·x_re − W_im·x_im) + j·(W_re·x_im + W_im·x_re)
即两路实权重、四次实矩阵乘的线性组合——参数量是实数版的 2 倍，
但建模能力覆盖复平面旋转（实数卷积做不到相位旋转）。
"""
import torch
from torch import nn

__all__ = ["ComplexConv2d", "ComplexConvTranspose2d", "ComplexLSTM", "complex_view",
           "complex_mul"]


def complex_view(x: torch.Tensor) -> torch.Tensor:
    """torch.complex 张量 [B,F,T] -> 实数视图 [B,2,F,T]（[Re;Im] 通道拼接）。"""
    return torch.cat([x.real.unsqueeze(1), x.imag.unsqueeze(1)], dim=1)


def complex_mul(mask: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
    """复数掩码（实数视图 [B,2,F,T]）⊙ 复数谱 [B,F,T] -> 复数谱。"""
    m = torch.complex(mask[:, 0], mask[:, 1])
    return m * spec


class _ComplexLinear2d(nn.Module):
    """复数 Conv2d / ConvTranspose2d 共用骨架：两路实 conv 的线性组合。"""

    def __init__(self, cls, cin, cout, kernel, stride, padding, bias=True):
        super().__init__()
        self.conv_re = cls(cin, cout, kernel, stride=stride, padding=padding, bias=bias)
        self.conv_im = cls(cin, cout, kernel, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        # x: [B, 2C, ...] -> re/im 各 [B, C, ...]
        c = x.shape[1] // 2
        xr, xi = x[:, :c], x[:, c:]
        yr = self.conv_re(xr) - self.conv_im(xi)
        yi = self.conv_re(xi) + self.conv_im(xr)
        return torch.cat([yr, yi], dim=1)


class ComplexConv2d(_ComplexLinear2d):
    """复数 2D 卷积。输入/输出均为 (B, 2C, F, T) 实数视图。"""

    def __init__(self, cin, cout, kernel=5, stride=(2, 2), padding=2, bias=True):
        super().__init__(nn.Conv2d, cin, cout, kernel, stride, padding, bias)


class ComplexConvTranspose2d(_ComplexLinear2d):
    """复数 2D 转置卷积（解码器上采样）。"""

    def __init__(self, cin, cout, kernel=5, stride=(2, 2), padding=2, bias=True):
        super().__init__(nn.ConvTranspose2d, cin, cout, kernel, stride, padding, bias)


class ComplexLSTM(nn.Module):
    """复数 LSTM（逐时间步 python 循环——DCCRN 编码后 T 仅 ~21 步，开销可接受）。

    门激活按分量作用（σ(z)=σ(re)+jσ(im)，tanh 同），Trabelsi 复数 RNN 的标准简化；
    全复数 tanh（有界全纯版）是更理论化的选择，实测差异小、代价高，未采用。
    输入 (B, T, 2D_in)，输出 (B, T, 2D_hidden)（取每层输出，非仅末态）。
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden, self.layers = hidden_size, num_layers
        d_in = input_size
        self.w_re = nn.ModuleList()
        self.w_im = nn.ModuleList()
        self.u_re = nn.ModuleList()
        self.u_im = nn.ModuleList()
        for _l in range(num_layers):
            self.w_re.append(nn.Linear(d_in, 4 * hidden_size))
            self.w_im.append(nn.Linear(d_in, 4 * hidden_size))
            self.u_re.append(nn.Linear(hidden_size, 4 * hidden_size, bias=False))
            self.u_im.append(nn.Linear(hidden_size, 4 * hidden_size, bias=False))
            d_in = hidden_size

    def forward(self, x):
        B, T, _ = x.shape
        H = self.hidden
        for layer in range(self.layers):
            h_re = x.new_zeros(B, H)
            h_im = x.new_zeros(B, H)
            c_re = x.new_zeros(B, H)
            c_im = x.new_zeros(B, H)
            outs_re, outs_im = [], []
            for t in range(T):
                xt_re, xt_im = x[:, t, : x.shape[2] // 2], x[:, t, x.shape[2] // 2:]
                # 复数线性：输入权重 + 递归权重
                g_re = (self.w_re[layer](xt_re) - self.w_im[layer](xt_im)
                        + self.u_re[layer](h_re) - self.u_im[layer](h_im))
                g_im = (self.w_re[layer](xt_im) + self.w_im[layer](xt_re)
                        + self.u_re[layer](h_im) + self.u_im[layer](h_re))
                i_re, f_re, g_re2, o_re = g_re.chunk(4, dim=-1)
                i_im, f_im, g_im2, o_im = g_im.chunk(4, dim=-1)
                i = torch.complex(torch.sigmoid(i_re), torch.sigmoid(i_im))
                f = torch.complex(torch.sigmoid(f_re), torch.sigmoid(f_im))
                o = torch.complex(torch.sigmoid(o_re), torch.sigmoid(o_im))
                gz = torch.complex(torch.tanh(g_re2), torch.tanh(g_im2))
                c = f * torch.complex(c_re, c_im) + i * gz
                h = o * torch.complex(torch.tanh(c.real), torch.tanh(c.imag))
                c_re, c_im = c.real, c.imag
                h_re, h_im = h.real, h.imag
                outs_re.append(h_re)
                outs_im.append(h_im)
            x = torch.cat([torch.stack(outs_re, dim=1), torch.stack(outs_im, dim=1)],
                          dim=-1)                      # (B, T, 2H)
        return x
