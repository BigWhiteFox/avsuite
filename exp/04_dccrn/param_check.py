# -*- coding: utf-8 -*-
"""DCCRN 参数量校验（WP-04 DoD：3-4M 量级）+ 逐模块参数表 + MACs 估算。

    python param_check.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model import DCCRN  # noqa: E402


def main():
    m = DCCRN()
    groups = {}
    for name, p in m.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + p.numel()
    total = sum(groups.values())
    print("| 模块 | 参数量 | 占比 |")
    print("|---|---|---|")
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"| {k} | {v/1e6:.3f}M | {v/total*100:.1f}% |")
    print(f"| **总计** | **{total/1e6:.3f}M** | 100% |")
    assert 2.5e6 <= total <= 4.5e6, f"参数量 {total/1e6:.2f}M 超出 3-4M 量级带"
    print("\nPASS: 参数量在 3-4M 量级（论文 DCCRN 3.71M）")

    # 实数卷积消融版对照
    mr = DCCRN(real_conv=True)
    tr = sum(p.numel() for p in mr.parameters())
    print(f"real-conv 消融版: {tr/1e6:.3f}M（复数版 {total/1e6:.3f}M 的 {tr/total:.0%}）")

    # MACs 估算（4s 输入单条）：conv MACs = k²·Cin·Cout·F_out·T_out（复数×4 实乘）
    macs = 0
    with torch.no_grad():
        x = torch.zeros(1, 2, 257, 321)
        skips = []
        for conv in m.enc:
            x = conv(x)
            skips.append(x)
            w = conv.conv_re.weight if hasattr(conv, "conv_re") else conv.weight
            cin = w.shape[1]
            cout = w.shape[0]
            n_complex = 4 if hasattr(conv, "conv_re") else 1     # 复数=4 实乘
            macs += 5 * 5 * cin * cout * x.shape[2] * x.shape[3] * n_complex
        flat_dim = x.shape[1] * x.shape[2]
        T4 = x.shape[3]
        # CLSTM: 每步每层 2 个复数矩阵乘(4H·D + 4H·H)×4 实乘
        H = m.clstm.hidden
        for li in range(m.clstm.layers):
            D = flat_dim // 2 if li == 0 else H
            macs += T4 * 4 * (4 * H * D + 4 * H * H) * 2
        macs += T4 * (2 * H) * (2 * flat_dim)                    # FC
        for dec in m.dec:
            x = torch.cat([x, skips.pop()], dim=1)
            x = dec(x)
            w = dec.conv_re.weight if hasattr(dec, "conv_re") else dec.weight
            n_complex = 4 if hasattr(dec, "conv_re") else 1
            macs += 5 * 5 * w.shape[1] * w.shape[0] * x.shape[2] * x.shape[3] * n_complex
    print(f"\n4s 单条前向 ≈ {macs/1e9:.2f} GMACs（含复数 4 实乘展开）")


if __name__ == "__main__":
    main()
