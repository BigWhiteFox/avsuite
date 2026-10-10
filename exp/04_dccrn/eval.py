"""DCCRN 评估（WP-04）：DNS2020 devtest（官方 parquet）+ 自混 held-out，PESQ/STOI/SI-SNR。

    python eval.py --ckpt outputs/dccrn/best.ckpt --devtest --num 824
    python eval.py --ckpt outputs/dccrn/best.ckpt --synth 100      # DNS 素材自混（seed 777）

协议：16k / PESQ-wb / 全长推理 batch=1 / core.evaluate 统一口径。
"""
import argparse
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynamic_mixing import DNSDynamicMix, load_16k  # noqa: E402
from model import DCCRN  # noqa: E402

from core.evaluate import evaluate_dir  # noqa: E402

DEVTEST = Path("D:/data/dns2020-devtest/data/synthetic_no_reverb-00000-of-00001.parquet")


def load_model(ckpt_path, device):
    st = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = st.get("cfg", {})
    m = DCCRN(clstm_hidden=cfg.get("clstm_hidden", 64),
              real_conv=cfg.get("real_conv", False)).to(device).eval()
    m.load_state_dict(st["model"])
    return m


def dump_devtest(tmpdir: Path, num: int):
    """官方 DNS2020 devtest parquet -> (noisy, clean) wav 目录对。"""
    import pyarrow.parquet as pq
    table = pq.read_table(DEVTEST)
    cols = table.column_names
    print(f"[devtest] columns: {cols}, rows: {table.num_rows}")
    noisy_dir, clean_dir = tmpdir / "noisy", tmpdir / "clean"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    n = min(num, table.num_rows)
    for i in range(n):
        row = {c: table[c][i].as_py() for c in cols}
        name = f"devtest_{i:04d}.wav"
        nb = row.get("noisy") or row.get("deg")
        cb = row.get("clean") or row.get("ref")
        (noisy_dir / name).write_bytes(nb["bytes"] if isinstance(nb, dict) else nb)
        (clean_dir / name).write_bytes(cb["bytes"] if isinstance(cb, dict) else cb)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n}")
    return noisy_dir, clean_dir


def dump_synth(tmpdir: Path, num: int):
    """DNS 素材自混 held-out（seed 777，SNR -5~15，与训练同分布不同样本）。"""
    ds = DNSDynamicMix(epoch_len=num, seg_sec=6.0, seed=777)
    noisy_dir, clean_dir = tmpdir / "noisy", tmpdir / "clean"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num):
        item = ds[i]
        name = f"synth_{i:04d}.wav"
        sf.write(noisy_dir / name, item["noisy"].numpy(), 16000)
        sf.write(clean_dir / name, item["clean"].numpy(), 16000)
    return noisy_dir, clean_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--devtest", action="store_true")
    ap.add_argument("--synth", type=int, default=0)
    ap.add_argument("--num", type=int, default=824)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or Path(__file__).parent / "outputs" / "eval" / args.ckpt.parent.name
    out.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="dccrn_eval_"))
    if args.devtest:
        noisy_dir, clean_dir = dump_devtest(tmp, args.num)
    elif args.synth:
        noisy_dir, clean_dir = dump_synth(tmp, args.synth)
    else:
        ap.error("需要 --devtest 或 --synth N")

    model = load_model(args.ckpt, device)
    for i, np_ in enumerate(sorted(noisy_dir.glob("*.wav"))):
        y = load_16k(np_)
        with torch.no_grad():
            enh = model.enhance_wav(torch.from_numpy(y).unsqueeze(0).to(device))[0]
        sf.write(out / np_.name, enh.cpu().numpy(), 16000)
        if (i + 1) % 100 == 0:
            print(f"  enhanced {i+1}", flush=True)

    df_enh = evaluate_dir(out, clean_dir, out_csv=out / "metrics.csv")
    df_noisy = evaluate_dir(noisy_dir, clean_dir, out_csv=out / "metrics_noisy.csv")
    import json
    summary = {
        "ckpt": str(args.ckpt), "n": len(df_enh),
        "mode": "devtest" if args.devtest else f"synth{args.synth}",
        "enh": {k: round(float(df_enh[k].mean()), 4) for k in ("pesq_wb", "stoi", "si_snr")},
        "noisy": {k: round(float(df_noisy[k].mean()), 4) for k in ("pesq_wb", "stoi", "si_snr")},
    }
    mode = summary["mode"]
    (out / f"summary_{mode}.json").write_text(json.dumps(summary, indent=2))  # 分模式存，
    (out / "summary.json").write_text(json.dumps(summary, indent=2))          # 兼容旧引用
    print(json.dumps(summary, indent=2))
    print("DoD 参考：DNS 合成测试 PESQ ≥2.6（DM+充分训练）；2.4-2.5 走降级路径")


if __name__ == "__main__":
    main()
