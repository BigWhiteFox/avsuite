"""P1-1 止损基线:零训练级联 —— WP-06 线性输出直送现成 v4 NN 推断(整改执行件)。

级联(零训练,不碰任何训练代码/checkpoint):
    (mic, far) ─GCC-PHAT 对齐 + MDF(DTD 门控, WP-06 现成管道)─> y_echo
    e_lin = mic − y_echo                        # 线性级残差(= MDF 误差输出,不含 RES)
    (e_lin, y_echo) ──当作 (mic, far)──> NNAEC v4(last.ckpt 现成推理)─> enh

口径(与 07_nn_aec/eval.py、aecmos_eval.py 保持一致):
- 线性级 = pipeline.TradAEC(method="mdf", partitions=8, mu=0.7)——与
  aecmos_eval.py --trad / eval.py 对比口径同参;只取 y_echo,RES 不参与
  (送 NN 的是线性级产物)。
- NN 侧无任何归一化:dataset.py/train.py 均为 float32 波形原样进出
  (训练期仅 far 抖动增强;零训练基线不加抖动、不加缩放),全段批处理。
- 输出 {name}.wav 与 eval_v3 manifest 的 mic 一一对应,16k float32。

用法(重跑):
    & 'C:\\Users\\14805\\.conda\\envs\\avsuite\\python.exe' zerotrain_baseline.py `
        --eval-dir D:/data/aec-synth/eval_v3
之后送评(aecmos_eval.py --enh-dir 模式):
    & 'C:\\Users\\14805\\.conda\\envs\\avsuite\\python.exe' ../07_nn_aec/aecmos_eval.py `
        --eval-dir D:/data/aec-synth/eval_v3 `
        --enh-dir ../09_cascade_aec/results/zerotrain_enh `
        --out ../09_cascade_aec/results/zerotrain_aecmos.json
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent                  # avsuite/exp/09_cascade_aec
ROOT = HERE.parents[1]                                  # avsuite 仓根
sys.path.insert(0, str(ROOT))                           # model.py 的 core.* 导入需要
sys.path.insert(0, str(ROOT / "exp" / "06_trad_aec"))   # pipeline/mdf
sys.path.insert(0, str(ROOT / "exp" / "07_nn_aec"))     # model(NNAEC)

from model import NNAEC  # noqa: E402
from pipeline import TradAEC  # noqa: E402

SR = 16000
SKIP_S = 2.0                                            # ERLE 跳收敛瞬态(与 eval.py 同口径)


def erle_db(sig, ref, skip_s=SKIP_S):
    k = int(skip_s * SR)
    return 10 * np.log10((np.sum(ref[k:] ** 2) + 1e-12) / (np.sum(sig[k:] ** 2) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, default=Path("D:/data/aec-synth/eval_v3"))
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/last.ckpt")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results" / "zerotrain_enh")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                    help="默认 CPU:避免与并行训练抢 GPU;60 条 CPU 也很快")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条(冒烟用,0=全部)")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")

    # ---- 线性级(WP-06 现成管道,与 aecmos --trad 基线同参)----
    aec = TradAEC(method="mdf", partitions=8, mu=0.7)

    # ---- NN 级(现成 v4 checkpoint,零训练)----
    st = torch.load(args.ckpt, map_location=device, weights_only=False)
    nn = NNAEC().to(device).eval()
    nn.load_state_dict(st["model"])
    print(f"[zerotrain] ckpt={args.ckpt}")
    print(f"[zerotrain] device={device}  clips_dir={args.eval_dir}  out={args.out_dir}")

    with open(args.eval_dir / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    erle_fest = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            name, sc = r["name"], r["scenario"]
            mic, sr = sf.read(args.eval_dir / "mic" / f"{name}.wav", dtype="float32")
            far, sr2 = sf.read(args.eval_dir / "farend" / f"{name}.wav", dtype="float32")
            assert sr == sr2 == SR, f"{name}: sr={sr}/{sr2}"
            n = min(len(mic), len(far))
            mic, far = mic[:n], far[:n]

            # 线性级:MDF(mic, far) -> y_echo;e_lin = mic - y_echo
            res = aec.process(mic.astype(np.float64), far.astype(np.float64))
            y_echo = res.y_echo
            e_lin = mic.astype(np.float64) - y_echo      # 按 P1-1 定义;数值上 == res.error

            # NN 级:(e_lin, y_echo) 当作 (mic, far) 原样送 v4(换输入对,零训练)
            enh = nn.enhance(torch.from_numpy(e_lin.astype(np.float32)).unsqueeze(0).to(device),
                             torch.from_numpy(y_echo.astype(np.float32)).unsqueeze(0).to(device))
            enh = enh[0].cpu().numpy().astype(np.float32)
            assert len(enh) == n, f"{name}: enh len {len(enh)} != {n}"
            assert np.isfinite(enh).all(), f"{name}: enh 有 NaN/Inf"
            # V5 v3b-044：写盘子类型显式 FLOAT（与 run_cascade/precompute_cascade 一致）。
            # 旧实现未指定 subtype → soundfile 默认 PCM_16 → 增强信号被量化，
            # 下游 AECMOS/PESQ 吃的是量化后的 wav（实测该目录 60/60 为 PCM_16）。
            sf.write(args.out_dir / f"{name}.wav", enh, SR, subtype="FLOAT")

            line = f"[{i + 1}/{len(rows)}] {name:18s} {sc:4s}"
            if sc == "fest":
                e_db = erle_db(e_lin, mic)
                erle_fest.append(e_db)
                line += f"  e_lin_erle {e_db:6.1f} dB"
            print(line)

    print(f"[zerotrain] done: {len(rows)} clips -> {args.out_dir}")
    if erle_fest:
        print(f"[zerotrain] fest e_lin ERLE(skip2s) mean = {np.mean(erle_fest):.1f} dB"
              f"(线性级保底检查,越正越好)")


if __name__ == "__main__":
    main()
