"""P0-5 训练工程探针①：损失项量级/梯度量级实测（决定心理声学列的缩放常数 c）。

在**冻结的 v4 ckpt** 与随机初始化两处状态上，对固定的 32 条训练样本探针批，
分别实测 WD / PSY / SISNR 三项的损失值与梯度范数——这样"两列的谱误差项量级可比"
不是拍的，而是有实测锚点的（数字原样写入 reports/p0_5_loss_matrix.md 与 ckpt 元数据）。

    python exp/07_nn_aec/loss_scale_probe.py --out exp/07_nn_aec/outputs_v4_p05/scale_probe.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

from dataset import AecDataset  # noqa: E402
from model import HOP, N_FFT, NNAEC  # noqa: E402

from core.losses import (  # noqa: E402
    psychoacoustic_loss,
    si_snr_loss,
    weighted_distortion_loss,
)
from core.stft import STFT  # noqa: E402


def grad_norm(model):
    return float(sum(float(p.grad.pow(2).sum()) for p in model.parameters()
                     if p.grad is not None) ** 0.5)


def probe(tag, ckpt, mic, far, near, near_spec, device):
    model = NNAEC().to(device)
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    model.train()
    out = {}
    for name, fn in (
        ("wd", lambda enh, enh_spec: weighted_distortion_loss(enh_spec, near_spec, gamma=0.5)),
        ("psy", lambda enh, enh_spec: psychoacoustic_loss(enh_spec, near_spec)[0]),
        ("sisnr", lambda enh, enh_spec: si_snr_loss(enh, near)),
    ):
        model.zero_grad(set_to_none=True)
        enh, _mask, _mic_spec, enh_spec = model(mic, far)
        loss = fn(enh, enh_spec)
        loss.backward()
        out[name] = {"loss": float(loss), "grad_norm": grad_norm(model)}
    out["psy_scale_term"] = float(psychoacoustic_loss(model(mic, far)[3], near_spec)[1])
    out["tag"] = tag
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("D:/data/aec-synth/train_v1"))
    ap.add_argument("--ckpt", type=Path,
                    default=HERE / "outputs_v4" / "nnaec_g0.5_j10" / "best.ckpt",
                    help="对标锚点 ckpt（默认 v4，用于把 PSY 的缩放常数钉在 v4 的谱项量级上）")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--target-ratio", type=float, default=1.0,
                    help="c = target_ratio * WD / PSY（1.0 = 谱项总量级对齐）")
    ap.add_argument("--out", type=Path, default=HERE / "outputs_v4_p05" / "scale_probe.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    tr = AecDataset(args.data, jitter_ms=10.0, scenarios=["dt", "nst"], seed=0)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(tr), generator=g)[:args.batch].tolist()
    mic = torch.from_numpy(np.stack([tr[i]["mic"].numpy() for i in idx])).to(device)
    far = torch.from_numpy(np.stack([tr[i]["far"].numpy() for i in idx])).to(device)
    near = torch.from_numpy(np.stack([tr[i]["near"].numpy() for i in idx])).to(device)
    near_spec = STFT(N_FFT, HOP).to(device).transform(near)

    res = {"data": str(args.data), "probe": {"n": len(idx), "seed": args.seed,
                                             "indices": idx[:16], "jitter_ms": 10.0}}
    init = probe("init", None, mic, far, near, near_spec, device)
    ref = probe("v4", args.ckpt, mic, far, near, near_spec, device)
    res["init"], res["v4"] = init, ref
    c = args.target_ratio * ref["wd"]["loss"] / ref["psy"]["loss"]
    res["c"] = float(c)
    res["c_from_init"] = float(args.target_ratio * init["wd"]["loss"] / init["psy"]["loss"])
    res["c_definition"] = ("c = target_ratio * WD(v4) / PSY(v4)，在 32 条固定训练探针批"
                           "（seed 1234）上实测；psycho 列总损失 = c*L_psy + SI-SNR")
    res["target_ratio"] = args.target_ratio
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for tag, r in (("init", init), ("v4", ref)):
        print(f"[{tag:5s}] WD L={r['wd']['loss']:9.4f} |g|={r['wd']['grad_norm']:9.3f}  "
              f"PSY L={r['psy']['loss']:.5f} |g|={r['psy']['grad_norm']:.4f}  "
              f"SISNR L={r['sisnr']['loss']:8.4f} |g|={r['sisnr']['grad_norm']:8.3f}")
        print(f"        |g| 比  PSY/SISNR={r['psy']['grad_norm']/r['sisnr']['grad_norm']:.4f}  "
              f"WD/SISNR={r['wd']['grad_norm']/r['sisnr']['grad_norm']:.4f}")
    print(f"[scale] c = {c:.6f}  (PSY 列谱项 L 将从 {ref['psy']['loss']:.5f} 变为 "
          f"{c*ref['psy']['loss']:.5f}，v4 的 WD 为 {ref['wd']['loss']:.5f})")
    print(f"[scale] 若改用 init 状态标定 c'={res['c_from_init']:.4f}（量级差 "
          f"{res['c_from_init']/c:.1f}×，记录在案）")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
