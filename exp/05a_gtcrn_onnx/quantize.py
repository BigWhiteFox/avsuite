"""GTCRN INT8 量化（WP-05a）：PTQ 后训练量化 + 精度损失报告。

    python quantize.py                # dynamic PTQ（GRU/Linear 权重量化）
    python quantize.py --mode static  # 静态 QDQ（含 Conv，需校准集，慢）

报告内容（DoD：PESQ/STOI 掉点）：
  - 50 条 DNS 自混 held-out（seed 777，与 eval.py --synth 同源）
  - fp32 vs int8：PESQ(wb)/STOI/SI-SNR 均值 + 逐条最大掉点
  - 模型体积 + 单帧时延对比

产物按模式分文件（追溯性：两种模式互不覆盖）：
  - 模型：dynamic -> outputs/gtcrn_dns3.int8.onnx（主产物）
          static  -> outputs/gtcrn_dns3.int8_static_BAD.onnx（崩坏对照，ADR-0003）
  - 逐条：outputs/quant_accuracy_rows_{dynamic,static}.csv
  - 汇总：outputs/quant_report_{dynamic,static}.json
历史遗留 outputs/quant_accuracy_rows.csv（旧 static 跑逐条明细）原样保留为存档，
新跑不再写该文件名（旧脚本曾用它，dynamic 明细被 static 覆盖即此坑）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import stft_convention  # noqa: E402

OUT = Path(__file__).parent / "outputs"
FP32_ONNX = OUT / "gtcrn_dns3.sim.onnx"

# 产物按量化模式分文件——杜绝 static 重跑覆盖 dynamic 的模型/逐条追溯行
# （历史坑：dynamic 的 quant_accuracy_rows.csv 曾被后续 static 跑覆盖）。
# 命名兼容既有引用：dynamic 主产物沿用 gtcrn_dns3.int8.onnx（eval_heldout200.py /
# bench.py / reports 引用此名），static 崩坏对照沿用 ADR-0003 约定的
# gtcrn_dns3.int8_static_BAD.onnx（tests/test_bug_05 引用此名）。
INT8_ONNX = {
    "dynamic": OUT / "gtcrn_dns3.int8.onnx",
    "static": OUT / "gtcrn_dns3.int8_static_BAD.onnx",
}
CSV_ROWS = {
    "dynamic": OUT / "quant_accuracy_rows_dynamic.csv",
    "static": OUT / "quant_accuracy_rows_static.csv",
}


def quantize_dynamic():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    int8_onnx = INT8_ONNX["dynamic"]
    quantize_dynamic(str(FP32_ONNX), str(int8_onnx), weight_type=QuantType.QInt8)
    print(f"[quantize] dynamic -> {int8_onnx.name} "
          f"({int8_onnx.stat().st_size/1024:.0f} KB vs "
          f"{FP32_ONNX.stat().st_size/1024:.0f} KB)")


def quantize_static(n_calib=50):
    """静态 QDQ：Conv 也量化，需要校准数据读入器。"""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class Reader(CalibrationDataReader):
        def __init__(self, n):
            torch.manual_seed(0)
            self.data = iter([{"spec": (torch.randn(1, 257, 201, 2) * 0.1).numpy()}
                              for _ in range(n)])

        def get_next(self):
            return next(self.data, None)

    int8_onnx = INT8_ONNX["static"]
    quantize_static(str(FP32_ONNX), str(int8_onnx), Reader(n_calib),
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    print(f"[quantize] static QDQ -> {int8_onnx.name} "
          f"({int8_onnx.stat().st_size/1024:.0f} KB)")


def accuracy_report(mode, n_clips=50):
    """fp32 vs int8：DNS 自混集上 PESQ/STOI/SI-SNR（逐条明细按模式分文件落盘）。"""
    import onnxruntime as ort

    from core.evaluate import _safe_pesq, _safe_stoi, _si_snr_np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_dccrn"))
    from dynamic_mixing import DNSDynamicMix

    sess32 = ort.InferenceSession(str(FP32_ONNX), providers=["CPUExecutionProvider"])
    sess8 = ort.InferenceSession(str(INT8_ONNX[mode]), providers=["CPUExecutionProvider"])
    ds = DNSDynamicMix(epoch_len=n_clips, seg_sec=4.0, seed=777)

    def enhance(sess, noisy):
        spec = stft_convention(torch.from_numpy(noisy)).numpy()
        enh = sess.run(None, {"spec": spec})[0]                 # (1,F,T,2)
        e = torch.complex(torch.from_numpy(enh[0, ..., 0]), torch.from_numpy(enh[0, ..., 1]))
        w = torch.hann_window(512).pow(0.5)
        return torch.istft(e, 512, 256, 512, w, length=len(noisy)).numpy()

    rows = []
    for i in range(n_clips):
        item = ds[i]
        noisy, clean = item["noisy"].numpy(), item["clean"].numpy()
        e32, e8 = enhance(sess32, noisy), enhance(sess8, noisy)
        rows.append(dict(
            pesq32=_safe_pesq(clean, e32.astype(np.float32)),
            pesq8=_safe_pesq(clean, e8.astype(np.float32)),
            stoi32=_safe_stoi(clean, e32.astype(np.float32)),
            stoi8=_safe_stoi(clean, e8.astype(np.float32)),
            sisnr32=_si_snr_np(clean, e32), sisnr8=_si_snr_np(clean, e8)))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_clips}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows).dropna()
    csv_path = CSV_ROWS[mode]
    df.to_csv(csv_path, index=False)
    rep = {
        "n": len(df),
        "pesq_fp32": round(df.pesq32.mean(), 4), "pesq_int8": round(df.pesq8.mean(), 4),
        "pesq_drop": round((df.pesq32 - df.pesq8).mean(), 4),
        "pesq_drop_median": round((df.pesq32 - df.pesq8).median(), 4),
        "pesq_drop_max": round((df.pesq32 - df.pesq8).max(), 4),
        "stoi_fp32": round(df.stoi32.mean(), 4), "stoi_int8": round(df.stoi8.mean(), 4),
        "stoi_drop": round((df.stoi32 - df.stoi8).mean(), 4),
        "sisnr_fp32": round(df.sisnr32.mean(), 3), "sisnr_int8": round(df.sisnr8.mean(), 3),
    }
    return rep


def latency_compare(mode="dynamic"):
    import onnxruntime as ort
    x1 = np.random.randn(1, 257, 1, 2).astype(np.float32) * 0.1
    times = {}
    for tag, path in (("fp32", FP32_ONNX), ("int8", INT8_ONNX[mode])):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        for _ in range(10):
            sess.run(None, {"spec": x1})
        t0 = time.perf_counter()
        for _ in range(200):
            sess.run(None, {"spec": x1})
        times[tag] = (time.perf_counter() - t0) / 200 * 1000
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dynamic", choices=["dynamic", "static"])
    ap.add_argument("--clips", type=int, default=50)
    args = ap.parse_args()

    if args.mode == "dynamic":
        quantize_dynamic()
    else:
        quantize_static()

    rep = accuracy_report(args.mode, args.clips)
    times = latency_compare(args.mode)
    rep["frame_ms_fp32"] = round(times["fp32"], 3)
    rep["frame_ms_int8"] = round(times["int8"], 3)
    out_json = OUT / f"quant_report_{args.mode}.json"
    out_json.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"-> {out_json}")
    print(f"-> {CSV_ROWS[args.mode]}")


if __name__ == "__main__":
    main()
