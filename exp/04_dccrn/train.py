"""DCCRN 训练脚本（WP-04 / 模块4）。

与 CRN train.py 同骨架（TensorBoard / Plateau / early stop / resume），差异：
- 数据：DNSDynamicMix 在线混合（epoch_len 虚拟长度），val 用固定 seed 的可复现子集
- 损失：复数谱 MSE + si_snr_weight × SI-SNR（官方是 SI-SNR + 多分辨率复数谱损，
  我们简化并文档化）
- 默认 4s 片段 / hop 200 / batch 8（8GB 显存）

用法：
    python train.py --device cuda --epochs 100 --batch-size 8        # 长训练
    python train.py --smoke                                          # 合成数据 1ep×2batch
    python train.py --device cuda --no-dm                            # 静态混合消融
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

from dynamic_mixing import DNSDynamicMix, StaticPairs  # noqa: E402
from model import DCCRN  # noqa: E402

from core.losses import complex_mse_loss, si_snr_loss  # noqa: E402
from core.stft import STFT  # noqa: E402


def make_loss_fn(si_snr_weight, device):
    stft = STFT(n_fft=512, hop=200).to(device)

    def loss_fn(model, batch):
        enh_wav, mask, noisy_spec, enh_spec = model(batch["noisy"])
        clean_spec = stft.transform(batch["clean"])
        return complex_mse_loss(enh_spec, clean_spec) \
            + si_snr_weight * si_snr_loss(enh_wav, batch["clean"])

    return loss_fn


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        total += float(loss_fn(model, batch)) * len(batch["noisy"])
        n += len(batch["noisy"])
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--epoch-len", type=int, default=2000, help="DM 虚拟 epoch 样本数")
    ap.add_argument("--val-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seg-sec", type=float, default=4.0)
    ap.add_argument("--si-snr-weight", type=float, default=0.1)
    ap.add_argument("--clstm-hidden", type=int, default=64)
    ap.add_argument("--real-conv", action="store_true", help="消融：实数卷积")
    ap.add_argument("--no-dm", action="store_true", help="消融：静态混合对")
    ap.add_argument("--early-stop", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true", help="随机张量（无 DNS 冒烟）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.epoch_len, args.val_len, args.num_workers = 1, 8, 4, 0
        args.synthetic = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    tag = "dccrn" + ("_realconv" if args.real_conv else "") + ("_nodm" if args.no_dm else "")
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
                clean = 0.1 * torch.randn(64000, generator=g)
                noisy = clean + 0.05 * torch.randn(64000, generator=g)
                return {"noisy": noisy, "clean": clean, "name": f"s{i}"}
        tr, va = _Syn(args.epoch_len, 1), _Syn(args.val_len, 999)
        print("[data] SYNTHETIC")
    elif args.no_dm:
        tr = StaticPairs(n_pairs=args.epoch_len, seg_sec=args.seg_sec, seed=args.seed)
        va = StaticPairs(n_pairs=args.val_len, seg_sec=args.seg_sec, seed=12345)
        print(f"[data] STATIC pairs train {len(tr)} / val {len(va)}")
    else:
        tr = DNSDynamicMix(epoch_len=args.epoch_len, seg_sec=args.seg_sec)
        va = DNSDynamicMix(epoch_len=args.val_len, seg_sec=args.seg_sec, seed=123)
        print(f"[data] DYNAMIC MIXING train {len(tr)}/ep / val {len(va)} (seed 固定)")
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    vl = DataLoader(va, batch_size=args.batch_size, num_workers=0)

    model = DCCRN(clstm_hidden=args.clstm_hidden, real_conv=args.real_conv).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] DCCRN{'(real-conv)' if args.real_conv else ''} {n_par/1e6:.2f}M -> {device}",
          flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=3)
    loss_fn = make_loss_fn(args.si_snr_weight, device)
    tb = SummaryWriter(out / "tb")

    start_ep, best, bad, step = 1, float("inf"), 0, 0
    ckpt_path, best_path = out / "last.ckpt", out / "best.ckpt"
    if args.resume and Path(args.resume).exists():
        st = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_ep, best, bad, step = st["epoch"] + 1, st["best"], st["bad"], st.get("step", 0)
        print(f"[resume] epoch {start_ep} best {best:.5f} step {step}", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        for batch in tl:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
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
        tb.add_scalar("lr", opt.param_groups[0]["lr"], ep)
        mark = ""
        if val_loss < best:
            best, bad, mark = val_loss, 0, " *best*"
            torch.save({"model": model.state_dict(), "cfg": vars(args), "val": val_loss},
                       best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best, "bad": bad,
                    "step": step, "cfg": vars(args)}, ckpt_path)
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
