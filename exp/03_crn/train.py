"""CRN 训练脚本（WP-03 / 模块3）。

特性：TensorBoard、ReduceLROnPlateau、early stop、断点续训（--resume）、
说话人不相交 val 切分、best-by-val-loss checkpoint。

用法：
    python train.py --data-root D:/data/voicebank-demand-16k/extracted \
        --epochs 100 --batch-size 16 --device cuda
    python train.py --smoke        # CPU 合成数据 1 epoch × 2 batch，验证链路

训练时长预估（RTX 4060 8GB，默认 hidden=256，batch 16，3s 片段）：
    约 3-4 s/epoch-step 百量级 -> 全量 ~2-4 h/100 epochs（README 有详细表）。
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import PairsDataset, extract_parquet, list_pairs, speaker_split  # noqa: E402
from model import CRN  # noqa: E402

from core.losses import complex_mse_loss, irm_bce_loss, si_snr_loss  # noqa: E402
from core.stft import STFT  # noqa: E402

DEFAULT_DATA = Path("D:/data/voicebank-demand-16k/extracted")


def make_loss_fn(mask_type, si_snr_weight=0.0, device=None):
    stft = STFT()
    if device is not None:
        stft = stft.to(device)

    def loss_fn(model, batch):
        enh_wav, mask, noisy_spec, enh_spec = model(batch["noisy"])
        clean_spec = stft.transform(batch["clean"])
        if mask_type == "irm":
            loss = irm_bce_loss(mask, noisy_spec, clean_spec)
        else:
            loss = complex_mse_loss(enh_spec, clean_spec)
        if si_snr_weight > 0:
            loss = loss + si_snr_weight * si_snr_loss(enh_wav, batch["clean"])
        return loss

    return loss_fn


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        total += float(loss_fn(model, batch)) * len(batch["noisy"])
        n += len(batch["noisy"])
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--rnn-type", default="blstm", choices=["blstm", "lstm"])
    ap.add_argument("--mask-type", default="crm", choices=["crm", "irm"])
    ap.add_argument("--bounded-mask", action="store_true")
    ap.add_argument("--no-skip", action="store_true", help="关掉 encoder->decoder 跳连")
    ap.add_argument("--si-snr-weight", type=float, default=0.0)
    ap.add_argument("--early-stop", type=int, default=12, help="val loss 连续 N epoch 不降则停")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=Path, default=None, help="从 checkpoint 续训")
    ap.add_argument("--max-train-batches", type=int, default=0, help=">0 时截断训练批数（冒烟用）")
    ap.add_argument("--synthetic", action="store_true", help="随机张量合成数据（无真实数据冒烟）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 epoch × 2 batch 的链路验证")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.max_train_batches, args.num_workers = 1, 2, 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    tag = (f"crn_{args.mask_type}_{args.rnn_type}"
           f"{'_bound' if args.bounded_mask else ''}{'_noskip' if args.no_skip else ''}")
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)

    # 数据
    if args.synthetic or args.smoke:
        class _Syn(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, i):
                torch.manual_seed(i)
                n = 48000
                clean = 0.3 * torch.randn(n)
                noisy = clean + 0.05 * torch.randn(n)
                return {"noisy": noisy, "clean": clean, "name": f"syn{i}"}
        tr, va = _Syn(), _Syn()
        print("[data] SYNTHETIC random-pairs (smoke)")
    else:
        if not (args.data_root / "train").exists():
            print(f"[data] {args.data_root} 无解包目录，先解包 parquet ...")
            extract_parquet()
        tr_pairs, va_pairs = speaker_split(list_pairs(args.data_root / "train" / "clean",
                                                      args.data_root / "train" / "noisy"))
        print(f"[data] train {len(tr_pairs)} / val {len(va_pairs)} (speaker-disjoint)")
        tr, va = PairsDataset(tr_pairs), PairsDataset(va_pairs)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CRN(hidden=args.hidden, rnn_type=args.rnn_type, mask_type=args.mask_type,
                bounded_mask=args.bounded_mask, skip=not args.no_skip).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] CRN {args.rnn_type}/{args.mask_type}  {n_par/1e6:.2f}M params -> {device}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # torch>=2.9 移除了 ReduceLROnPlateau 的 verbose 参数（原 verbose=True 仅打印降 lr 日志）
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=4)
    loss_fn = make_loss_fn(args.mask_type, args.si_snr_weight, device)
    tb = SummaryWriter(out / "tb")

    start_ep, best, bad = 1, float("inf"), 0
    ckpt_path, best_path = out / "last.ckpt", out / "best.ckpt"
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_ep, best, bad = state["epoch"] + 1, state["best"], state["bad"]
        print(f"[resume] epoch {start_ep}, best={best:.5f}")

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        for i, batch in enumerate(tl):
            if args.max_train_batches and i >= args.max_train_batches:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss = loss_fn(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tb.add_scalar("train/loss", float(loss), (ep - 1) * len(tl) + i)
        val_loss = validate(model, vl, loss_fn, device)
        sched.step(val_loss)
        tb.add_scalar("val/loss", val_loss, ep)
        tb.add_scalar("lr", opt.param_groups[0]["lr"], ep)
        mark = ""
        if val_loss < best:
            best, bad, mark = val_loss, 0, " *best*"
            torch.save({"model": model.state_dict(), "cfg": vars(args)}, best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best, "bad": bad,
                    "cfg": vars(args)}, ckpt_path)
        print(f"epoch {ep:3d}  val_loss {val_loss:.5f}{mark}")
        (out / "history.json").write_text(json.dumps({"best": best, "bad": bad,
                                                      "epoch": ep, "val": val_loss}))
        if bad >= args.early_stop:
            print(f"[early-stop] {args.early_stop} epochs no improve (best {best:.5f})")
            break
    print(f"[done] best val {best:.5f}  ckpt -> {out}")


if __name__ == "__main__":
    main()
