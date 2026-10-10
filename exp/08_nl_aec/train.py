"""非线性 AEC 训练 v2（WP-08）：Ivry 2021 级联结构。

    级联：ŷ_lin = MDF(far)（预计算，消 180ms 长记忆）
          ŷ_res = GRU([lin, |lin|])（记忆less 非线性残差，2 层 GRU-128）
          enh   = mic - lin - res
负实验（v1，已文档化）：GRU 直接吃原始 far → ERLE 0dB——回声时延+RIR 的
180ms 记忆远超 GRU 容量；线性级消化长记忆后，GRU 只需建模 tanh 型静态非线性。

用法同 v1：python train.py --device cuda（--data 需含 precompute_lin.py 的 lin_est）
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import NonlinearAEC  # noqa: E402

SR = 16000


class NLCascadeSet(Dataset):
    """mic / far / near / lin_est 四目录 + manifest。"""

    def __init__(self, synth_dir, scenarios=None, seed=None):
        self.dir = Path(synth_dir)
        rows = list(csv.DictReader(open(self.dir / "manifest.csv", encoding="utf-8")))
        if scenarios:
            rows = [r for r in rows if r["scenario"] in scenarios]
        self.rows = rows
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        name = r["name"]
        mic = sf.read(self.dir / "mic" / f"{name}.wav", dtype="float32")[0]
        far = sf.read(self.dir / "farend" / f"{name}.wav", dtype="float32")[0]
        near = sf.read(self.dir / "nearend" / f"{name}.wav", dtype="float32")[0]
        lin = sf.read(self.dir / "lin_est" / f"{name}.wav", dtype="float32")[0]
        n = min(len(mic), len(far), len(near), len(lin))
        return {"mic": torch.from_numpy(mic[:n].copy()),
                "lin": torch.from_numpy(lin[:n].copy()),
                "near": torch.from_numpy(near[:n].copy())}


def collate(batch):
    n = min(b["mic"].shape[-1] for b in batch)
    return {k: torch.stack([b[k][..., :n] for b in batch]) for k in ("mic", "lin", "near")}


@torch.no_grad()
def validate(model, loader, device, seg_n=None):
    model.eval()
    total, n = 0.0, 0
    for b in loader:
        lin = b["lin"].to(device)
        mic = b["mic"].to(device)
        near = b["near"].to(device)
        if seg_n and lin.shape[-1] > seg_n:
            lin, mic, near = lin[:, :seg_n], mic[:, :seg_n], near[:, :seg_n]
        res = model(lin)
        total += float(torch.nn.functional.mse_loss(res, (mic - near) - lin)) * len(lin)
        n += len(lin)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("D:/data/aec-synth/train_nl"))
    ap.add_argument("--val", type=Path, default=Path("D:/data/aec-synth/eval_nl"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seg", type=float, default=1.0)
    ap.add_argument("--early-stop", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    out = args.out / "nl_aec_v2"
    out.mkdir(parents=True, exist_ok=True)
    seg_n = int(args.seg * SR)

    tr = NLCascadeSet(args.data, scenarios=["fest", "dt"])
    va = NLCascadeSet(args.val)
    print(f"[data] train {len(tr)} / val {len(va)}", flush=True)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate, drop_last=True, persistent_workers=args.num_workers > 0)
    vl = DataLoader(va, batch_size=args.batch_size, num_workers=0, collate_fn=collate)

    model = NonlinearAEC().to(device)
    print(f"[model] NonlinearAEC(级联残差) {sum(p.numel() for p in model.parameters())/1e6:.3f}M",
          flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    start_ep, best, bad = 1, float("inf"), 0
    ckpt_path, best_path = out / "last.ckpt", out / "best.ckpt"
    if args.resume and Path(args.resume).exists():
        st = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_ep, best, bad = st["epoch"] + 1, st["best"], st["bad"]
        print(f"[resume] epoch {start_ep}", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        for b in tl:
            lin = b["lin"].to(device)
            mic = b["mic"].to(device)
            near = b["near"].to(device)
            n = lin.shape[-1]
            if n > seg_n:
                off = torch.randint(0, n - seg_n + 1, (1,)).item()
                lin, mic, near = lin[:, off:off + seg_n], mic[:, off:off + seg_n], \
                    near[:, off:off + seg_n]
            res = model(lin)
            loss = torch.nn.functional.mse_loss(res, (mic - near) - lin)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val = validate(model, vl, device, seg_n=seg_n)
        sched.step(val)
        mark = ""
        if val < best:
            best, bad, mark = val, 0, " *best*"
            torch.save({"model": model.state_dict(), "cfg": vars(args)}, best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best": best, "bad": bad}, ckpt_path)
        (out / "history.json").write_text(json.dumps(
            {"epoch": ep, "val": val, "best": best, "bad": bad}))
        print(f"epoch {ep:3d}  val {val:.6f}  ({time.perf_counter()-t0:.0f}s){mark}",
              flush=True)
        if bad >= args.early_stop:
            break
    print(f"[done] best {best:.6f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
