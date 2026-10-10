# -*- coding: utf-8 -*-
"""NN-AEC 训练（WP-07 / 模块7，简历核心）。

损失（复用 core/losses，WORKLIST：加权失真 γ 消融）：
    L = weighted_distortion(enh谱, near谱, γ) + w_sisnr × SI-SNR(enh波形, near波形)
    γ 默认 0.5（AEC 式响度加权）；γ=0 退化为复数谱 MSE（消融开关 --gamma 0）

用法：
    python train.py --data D:/data/aec-synth/train_v1 --val D:/data/aec-synth/eval_v3 \
        --device cuda --epochs 60
    python train.py --smoke            # 合成张量链路验证
    python train.py ... --gamma 0.3    # γ 消融
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.losses import si_snr_loss, weighted_distortion_loss  # noqa: E402
from core.stft import STFT  # noqa: E402

from dataset import AecDataset  # noqa: E402
from model import HOP, NNAEC, N_FFT  # noqa: E402


def collate(batch):
    """变长保护：批内截到最短（aec_synth 定长生成时恒等长）。"""
    n = min(b["mic"].shape[-1] for b in batch)
    out = {k: torch.stack([b[k][..., :n] for b in batch]) for k in ("mic", "far", "near")}
    out["name"] = [b["name"] for b in batch]
    out["scenario"] = [b["scenario"] for b in batch]
    return out


def make_loss_fn(gamma, w_sisnr, device):
    stft = STFT(N_FFT, HOP).to(device)

    def loss_fn(model, batch):
        enh_wav, mask, mic_spec, enh_spec = model(batch["mic"], batch["far"])
        near_spec = stft.transform(batch["near"])
        loss = weighted_distortion_loss(enh_spec, near_spec, gamma=gamma)
        if w_sisnr > 0:
            loss = loss + w_sisnr * si_snr_loss(enh_wav, batch["near"])
        return loss

    return loss_fn


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        total += float(loss_fn(model, batch)) * len(batch["mic"])
        n += len(batch["mic"])
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("D:/data/aec-synth/train_v1"))
    ap.add_argument("--val", type=Path, default=Path("D:/data/aec-synth/eval_v3"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--w-sisnr", type=float, default=0.1)
    ap.add_argument("--jitter-ms", type=float, default=10.0)
    ap.add_argument("--val-jitter-ms", type=float, default=0.0)
    ap.add_argument("--scenarios", default=None,
                    help="训练场景过滤（逗号分隔，如 dt,nst；默认全场景）")
    ap.add_argument("--early-stop", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.num_workers, args.synthetic = 1, 0, True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    tag = f"nnaec_g{args.gamma}_j{int(args.jitter_ms)}"
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        class _Syn(torch.utils.data.Dataset):
            def __init__(self, n, seed):
                self.n, self.seed = n, seed

            def __len__(self):
                return self.n

            def __getitem__(self, i):
                g = torch.Generator().manual_seed(self.seed * 100003 + i)
                far = 0.2 * torch.randn(32000, generator=g)
                near = 0.1 * torch.randn(32000, generator=torch.Generator().manual_seed(i + 7))
                mic = far * 0.5 + near
                return {"mic": mic, "far": far, "near": near, "name": f"s{i}",
                        "scenario": "dt"}
        tr, va = _Syn(16, 1), _Syn(8, 999)
        print("[data] SYNTHETIC")
    else:
        sc_filter = args.scenarios.split(",") if args.scenarios else None
        tr = AecDataset(args.data, jitter_ms=args.jitter_ms, scenarios=sc_filter)
        va = AecDataset(args.val, jitter_ms=args.val_jitter_ms, seed=123)
        from collections import Counter
        print(f"[data] train {len(tr)} {dict(Counter(r['scenario'] for r in tr.rows))} "
              f"/ val {len(va)} (jitter {args.jitter_ms}/{args.val_jitter_ms}ms)")
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate, drop_last=True,
                    persistent_workers=args.num_workers > 0)
    vl = DataLoader(va, batch_size=args.batch_size, num_workers=0, collate_fn=collate)

    model = NNAEC().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] NNAEC {n_par/1e6:.3f}M params -> {device}", flush=True)
    assert n_par < 1e6, f"参数量 {n_par/1e6:.2f}M 超出 <1M 约束"
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=3)
    loss_fn = make_loss_fn(args.gamma, args.w_sisnr, device)
    tb = SummaryWriter(out / "tb")

    start_ep, best, bad, step = 1, float("inf"), 0, 0
    ckpt_path, best_path = out / "last.ckpt", out / "best.ckpt"
    loss_sig = (args.gamma, args.w_sisnr, args.scenarios)
    if args.resume and Path(args.resume).exists():
        st = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_ep, best, bad, step = st["epoch"] + 1, st["best"], st["bad"], st.get("step", 0)
        if st.get("cfg", {}).get("loss_sig") != loss_sig:
            # 损失函数配置变了（如 w-sisnr 0.1→0.5）：旧 best 口径失效，重置
            # （否则 val 尺度不同，best.ckpt 永远不会被新损失下的更优模型覆盖）
            best, bad = float("inf"), 0
            print("[resume] 检测到损失配置变化，best/bad 已重置", flush=True)
        print(f"[resume] epoch {start_ep} best {best:.5f}", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        for batch in tl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            loss = loss_fn(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tb.add_scalar("train/loss", float(loss), step)
            step += 1
        val_loss = validate(model, vl, loss_fn, device)
        sched.step(val_loss)
        tb.add_scalar("val/loss", val_loss, ep)
        mark = ""
        if val_loss < best:
            best, bad, mark = val_loss, 0, " *best*"
            torch.save({"model": model.state_dict(), "cfg": dict(vars(args), loss_sig=loss_sig),
                        "val": val_loss}, best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best, "bad": bad,
                    "step": step, "cfg": dict(vars(args), loss_sig=loss_sig)}, ckpt_path)
        (out / "history.json").write_text(json.dumps(
            {"epoch": ep, "val": val_loss, "best": best, "bad": bad, "step": step}))
        print(f"epoch {ep:3d}  val {val_loss:.5f}  ({time.perf_counter()-t0:.0f}s){mark}",
              flush=True)
        if bad >= args.early_stop:
            print(f"[early-stop] best {best:.5f}")
            break
    print(f"[done] best {best:.5f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
