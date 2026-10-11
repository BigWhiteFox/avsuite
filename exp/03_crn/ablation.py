"""CRN 消融驱动（WP-03）：IRM↔cRM、BLSTM↔LSTM、跳连 on/off、bounded mask。

两种模式：
    --quick（默认，冒烟）：合成数据短训，验证开关都工作 + 趋势表模板
    --real：真实 VoiceBank 子集短训（--subset 控制规模），最终消融表由
            完整训练回填（README 表格）。

    python ablation.py --quick
    python ablation.py --real --subset 0.2
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import CRN  # noqa: E402

RUNS = [  # (mask_type, rnn_type, bounded, skip)
    ("crm", "blstm", False, True),      # 默认配置
    ("irm", "blstm", False, True),      # 掩码类型
    ("crm", "lstm", False, True),       # 循环结构（因果）
    ("crm", "blstm", True, False),      # bounded mask + 无跳连
]


def _train_inline(mask_type, rnn_type, bounded, skip, a):
    """单组配置短训练，返回 (val_loss, wall_sec, params)。"""
    from core.losses import complex_mse_loss, irm_bce_loss
    from core.stft import STFT
    if a.real:
        from dataset import list_pairs, speaker_split
        tr_pairs, _ = speaker_split(list_pairs(a.data_root / "train" / "clean",
                                               a.data_root / "train" / "noisy"))
        n = max(int(len(tr_pairs) * a.subset), 2)
        pairs = tr_pairs[:n]

        class _Pairs(torch.utils.data.Dataset):
            def __len__(self):
                return len(pairs)

            def __getitem__(self, i):
                import soundfile as sf
                noisy, clean = pairs[i]
                ny, _ = sf.read(str(noisy), dtype="float32", always_2d=True)
                cy, _ = sf.read(str(clean), dtype="float32", always_2d=True)
                n_ = min(len(ny), len(cy), 48000)
                return {"noisy": torch.from_numpy(ny[:n_, 0]),
                        "clean": torch.from_numpy(cy[:n_, 0]), "name": noisy.stem}
        ds = _Pairs()
    else:
        class _Syn(torch.utils.data.Dataset):
            def __len__(self):
                return 16

            def __getitem__(self, i):
                torch.manual_seed(i)
                clean = 0.3 * torch.randn(48000)
                noisy = clean + 0.05 * torch.randn(48000)
                return {"noisy": noisy, "clean": clean, "name": f"s{i}"}
        ds = _Syn()

    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True)
    device = torch.device(a.device)
    model = CRN(hidden=a.hidden, rnn_type=rnn_type, mask_type=mask_type,
                bounded_mask=bounded, skip=skip).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    stft = STFT().to(device)

    def loss_fn(batch):
        # V5 v3b-023：输入张量必须搬到 device。旧实现模型/STFT 在 device 而 batch 仍在
        # cpu → `--device cuda` 直接 device mismatch 崩（cpu 下 .to 是恒等，行为不变）。
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        _, mask, noisy_spec, enh_spec = model(noisy)
        clean_spec = stft.transform(clean)
        if mask_type == "irm":
            return irm_bce_loss(mask, noisy_spec, clean_spec)
        return complex_mse_loss(enh_spec, clean_spec)

    t0 = time.perf_counter()
    for _ in range(a.epochs):
        model.train()
        for i, batch in enumerate(dl):
            if a.batches and i >= a.batches:
                break
            loss = loss_fn(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for batch in dl:
            tot += float(loss_fn(batch)) * len(batch["noisy"])
    return tot / max(len(ds), 1), time.perf_counter() - t0, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="合成数据 2 epoch × 3 batch")
    ap.add_argument("--real", action="store_true", help="真实数据（需已解包）")
    ap.add_argument("--data-root", type=Path,
                    default=Path("D:/data/voicebank-demand-16k/extracted"))
    ap.add_argument("--subset", type=float, default=0.1, help="真实模式取的训练对比例")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(Path(__file__).parent / "outputs" / "ablation"))
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.batches = 2, 3

    rows = []
    for mask_type, rnn_type, bounded, skip in RUNS:
        val, sec, par = _train_inline(mask_type, rnn_type, bounded, skip, args)
        name = f"{mask_type}+{rnn_type}" + ("+bound" if bounded else "") + ("+skip" if skip else "")
        rows.append((name, val, sec, par))
        print(f"{name:22s} val {val:.5f}  {sec:6.1f}s  {par/1e6:.2f}M")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["| 配置 | val loss | wall(s) | params |", "|---|---|---|---|"]
    for name, val, sec, par in rows:
        lines.append(f"| {name} | {val:.5f} | {sec:.1f} | {par/1e6:.2f}M |")
    (out / "ablation_short.md").write_text(
        "\n".join(lines) + "\n\n注：短训冒烟表（验证开关工作 + 趋势参考）；"
        "正式消融表见 README，由完整训练回填。\n")
    print(f"saved -> {out/'ablation_short.md'}")


if __name__ == "__main__":
    main()
