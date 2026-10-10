"""CRN 评估脚本（WP-03）：checkpoint + noisy/clean 目录 -> 增强 wav + 指标 CSV。

    python eval.py --ckpt outputs/crm_blstm/best.ckpt \
        --noisy-dir D:/data/voicebank-demand-16k/extracted/test/noisy \
        --clean-dir D:/data/voicebank-demand-16k/extracted/test/clean

输出 outputs/eval/<ckpt名>/enh_*.wav + metrics.csv + 均值表（PESQ/STOI/SI-SNR，
噪声前后对比）。全长推理（batch=1），与训练 3s 裁剪解耦。
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import list_pairs  # noqa: E402
from model import CRN  # noqa: E402

from core.evaluate import evaluate_dir  # noqa: E402


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("cfg", {})
    model = CRN(hidden=cfg.get("hidden", 256), rnn_type=cfg.get("rnn_type", "blstm"),
                mask_type=cfg.get("mask_type", "crm"),
                bounded_mask=cfg.get("bounded_mask", False),
                skip=not cfg.get("no_skip", False)).to(device).eval()
    model.load_state_dict(state["model"])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--noisy-dir", type=Path, required=True)
    ap.add_argument("--clean-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-files", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or Path(__file__).parent / "outputs" / "eval" / args.ckpt.stem
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, device)
    pairs = list_pairs(args.clean_dir, args.noisy_dir)
    if args.max_files:
        pairs = pairs[:args.max_files]
    import soundfile as sf
    with torch.no_grad():
        for i, (noisy_p, _clean_p) in enumerate(pairs):
            noisy, _ = sf.read(str(noisy_p), dtype="float32", always_2d=True)
            noisy = torch.from_numpy(noisy.mean(1)).unsqueeze(0).to(device)
            enh = model.enhance_wav(noisy)[0].cpu().numpy()
            sf.write(out / noisy_p.name, enh, 16000)   # 与参考同名（分目录），配对靠文件名
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(pairs)}")

    # 指标：增强 vs 干净（主指标）+ 带噪 vs 干净（基线参考）
    df_enh = evaluate_dir(out, args.clean_dir, out_csv=out / "metrics.csv")
    noisy_tmp = out / "_noisy_link"
    import shutil
    if noisy_tmp.exists():
        shutil.rmtree(noisy_tmp)
    noisy_tmp.mkdir(parents=True)
    for noisy_p, _ in pairs:          # list_pairs 返回 (noisy, clean)
        shutil.copy(noisy_p, noisy_tmp / noisy_p.name)
    df_noisy = evaluate_dir(noisy_tmp, args.clean_dir, out_csv=out / "metrics_noisy_baseline.csv")
    shutil.rmtree(noisy_tmp)

    summary = {
        "ckpt": str(args.ckpt),
        "n_files": len(df_enh),
        "enh": {k: round(float(df_enh[k].mean()), 4) for k in ("pesq_wb", "stoi", "si_snr")},
        "noisy_baseline": {k: round(float(df_noisy[k].mean()), 4)
                           for k in ("pesq_wb", "stoi", "si_snr")},
    }
    (out / "summary.json").write_text(str(summary).replace("'", '"'))
    print(f"[eval] {len(df_enh)} files -> {out}")
    print(f"  enh:    PESQ {summary['enh']['pesq_wb']:.3f}  "
          f"STOI {summary['enh']['stoi']:.3f}  SI-SNR {summary['enh']['si_snr']:.2f}")
    print(f"  noisy:  PESQ {summary['noisy_baseline']['pesq_wb']:.3f}  "
          f"STOI {summary['noisy_baseline']['stoi']:.3f}  SI-SNR {summary['noisy_baseline']['si_snr']:.2f}")


if __name__ == "__main__":
    main()
