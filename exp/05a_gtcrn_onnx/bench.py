"""GTCRN CPU 基准（WP-05a DoD：RTF < 0.2，单帧 < 3ms）。

测三件事：
  1. 整段 RTF：4s 语音端到端（stft+model+istft，ONNXRuntime CPU 单线程）
  2. 单帧时延：T=1 逐帧调用（GTCRN 时间维因果，帧级调用合法；真流式带
     GRU 状态缓存的版本在官方 stream/ 目录，WP-05b 范畴）
  3. 峰值内存：推理期间进程 RSS 增量

    python bench.py                       # fp32 sim 模型
    python bench.py --model outputs/gtcrn_dns3.int8.onnx
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 16000


def bench(onnx_path: Path, threads: int = 1, dur: float = 4.0):
    import onnxruntime as ort
    import psutil
    proc = psutil.Process()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    n = int(dur * SR)
    T = n // 256 + 1
    x_full = np.random.randn(1, 257, T, 2).astype(np.float32) * 0.1
    x_frame = np.random.randn(1, 257, 1, 2).astype(np.float32) * 0.1

    # 预热
    for _ in range(3):
        sess.run(None, {"spec": x_full})
    rss0 = proc.memory_info().rss

    # 1) 整段 RTF（模型部分；stft/istft 另计）
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, {"spec": x_full})
    model_t = (time.perf_counter() - t0) / reps
    rtf = model_t / dur

    # torch stft/istft 开销（端到端账本要算上）
    import torch
    wav = torch.from_numpy(np.random.randn(n).astype(np.float32))
    w = torch.hann_window(512).pow(0.5)
    t0 = time.perf_counter()
    for _ in range(reps):
        spec = torch.stft(wav, 512, 256, 512, w, return_complex=True)
        torch.istft(spec, 512, 256, 512, w, length=n)
    stft_t = (time.perf_counter() - t0) / reps

    # 2) 单帧时延（T=1，含会话开销）
    for _ in range(10):
        sess.run(None, {"spec": x_frame})
    reps_f = 200
    t0 = time.perf_counter()
    for _ in range(reps_f):
        sess.run(None, {"spec": x_frame})
    frame_ms = (time.perf_counter() - t0) / reps_f * 1000

    rss1 = proc.memory_info().rss
    print(f"[bench] {onnx_path.name}  (threads={threads})")
    print(f"  模型整段:  {model_t*1000:7.1f} ms / {dur}s  -> RTF {rtf:.4f}  "
          f"DoD(<0.2): {'PASS' if rtf < 0.2 else 'FAIL'}")
    print(f"  stft+istft: {stft_t*1000:6.1f} ms  (端到端 RTF {(model_t+stft_t)/dur:.4f})")
    print(f"  单帧(T=1): {frame_ms:7.3f} ms  DoD(<3ms): {'PASS' if frame_ms < 3 else 'FAIL'}"
          f"   帧移 16ms -> 帧预算余量 {16/frame_ms:.1f}x")
    print(f"  RSS 增量:  {(rss1-rss0)/1024**2:.1f} MB")
    return rtf, frame_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path,
                    default=Path(__file__).parent / "outputs" / "gtcrn_dns3.sim.onnx")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    bench(args.model, args.threads)


if __name__ == "__main__":
    main()
