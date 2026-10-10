# -*- coding: utf-8 -*-
"""非线性 AEC 模型（WP-08 / 模块8，选做）：GRU 非线性残差建模（Ivry 2021 级联）。

结构（级联第二级）：
    lin  = MDF(far)            # 线性级消化时延+RIR 长记忆（precompute_lin.py 预计算）
    res  = GRU([lin, |lin|])   # 记忆less 非线性残差（tanh 型功放失真），2×GRU-128
    enh  = mic - lin - res

负实验（v1，已文档化于 README）：GRU 直接吃原始 far → ERLE 0dB。回声含
10-80ms 时延 + 100ms RIR ≈ 180ms 记忆，远超逐样本 GRU 容量——这就是
Ivry 级联结构存在的理由：非线性建模前必须先线性对齐消记忆。
验收：合成非线性数据上较纯线性级 ERLE +5dB（WORKLIST）。
"""
import torch
from torch import nn

__all__ = ["NonlinearAEC"]


class NonlinearAEC(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(2, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, lin: torch.Tensor):
        """lin: [B,N] 线性回声估计 -> 非线性残差估计 [B,N]（同长逐样本）。"""
        x = torch.stack([lin, lin.abs()], dim=-1)          # [B,N,2]
        z, _ = self.gru(x)
        return self.fc(z).squeeze(-1)                      # [B,N]

    @torch.no_grad()
    def enhance(self, mic: torch.Tensor, lin: torch.Tensor) -> torch.Tensor:
        return mic - lin - self.forward(lin)
