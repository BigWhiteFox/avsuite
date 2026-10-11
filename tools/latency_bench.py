"""P2-7 端到端时延账本实测：NN-AEC v4 流式单块 + GTCRN ONNX 单帧/整段基准。

整改项 P2-7（时延叙事 AEC 32ms + NR 32ms = 64ms 的实测底账）：
  1. NN-AEC v4 流式（20ms 块进块出）：加载 exp/07_nn_aec/outputs_v4/
     nnaec_g0.5_j10/last.ckpt，预热 50 块后连续 1000 次 process_chunk(320)
     逐次 perf_counter 计时 -> mean/p50/p95/p99（ms）；默认线程与单线程各测
     一轮（单线程与仓内 RTF 口径一致，作预算主数字）。
  2. GTCRN ONNX（fp32 sim，同 capstone）：单帧等效耗时（T=1，16ms 帧移，
     intra_op=1，同 exp/05a bench.py 口径）+ 整段批处理 RTF（4s，stft+模型
     +istft 端到端——capstone run_nr 的实际口径是整段批，非流式，如实分列）。
  3. 记录 CPU 型号 / 线程配置 / 测量时背景负载（psutil cpu_percent 采样）。

输出：控制台表 + JSON（默认打印到 stdout；--json PATH 可落盘）。

用法：
    & 'C:\\Users\\14805\\.conda\\envs\\avsuite\\python.exe' tools/latency_bench.py
    python tools/latency_bench.py --json outputs.json
"""
import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent        # avsuite/tools
ROOT = HERE.parent                            # avsuite/ 仓库根
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp" / "07_nn_aec"))

SR = 16000
CHUNK = 320                      # 20ms @16k（模型 hop）
N_WARM = 50                      # 预热块数（=50 帧）
N_ITERS = 1000                   # 计时块数
NNAEC_CKPT = ROOT / "exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/last.ckpt"
GTCRN_ONNX = ROOT / "exp/05a_gtcrn_onnx/outputs/gtcrn_dns3.sim.onnx"


def cpu_name() -> str:
    """CPU 型号：Windows 注册表最可靠，失败退 platform.processor()。"""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        return platform.processor() or "unknown"


def stats_ms(samples) -> dict:
    a = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": round(float(a.mean()), 4),
        "p50_ms": round(float(np.percentile(a, 50)), 4),
        "p95_ms": round(float(np.percentile(a, 95)), 4),
        "p99_ms": round(float(np.percentile(a, 99)), 4),
        "min_ms": round(float(a.min()), 4),
        "max_ms": round(float(a.max()), 4),
    }


def load_aec():
    """同 exp/10.5_capstone/cascade.py::load_nnaec + NNAECStreaming 用法。"""
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location(
        "p27_nnaec_model", ROOT / "exp/07_nn_aec/model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.NNAEC().eval()
    st = torch.load(NNAEC_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(st["model"])
    from streaming import NNAECStreaming
    return NNAECStreaming(m, device="cpu")


def bench_aec(aec, label: str) -> dict:
    """预热 50 块后连续 1000 次 process_chunk(320) 逐块计时（同一流实例，
    状态跨块延续——即部署形态）。"""
    mic = (np.random.default_rng(7).standard_normal((N_WARM + N_ITERS + 1) * CHUNK)
           * 0.05).astype(np.float32)
    far = (np.random.default_rng(8).standard_normal((N_WARM + N_ITERS + 1) * CHUNK)
           * 0.05).astype(np.float32)
    for i in range(N_WARM):
        aec.process_chunk(mic[i * CHUNK:(i + 1) * CHUNK],
                          far[i * CHUNK:(i + 1) * CHUNK])
    samples, out_lens = [], set()
    gc.disable()
    try:
        for i in range(N_ITERS):
            off = (N_WARM + i) * CHUNK
            t0 = time.perf_counter()
            out = aec.process_chunk(mic[off:off + CHUNK], far[off:off + CHUNK])
            samples.append((time.perf_counter() - t0) * 1000.0)
            out_lens.add(int(len(out)))
    finally:
        gc.enable()
    r = {
        "config": label,
        "chunk_samples": CHUNK,
        "chunk_ms": 20.0,
        "warmup_chunks": N_WARM,
        "timed_chunks": N_ITERS,
        "audio_processed_s": round(N_ITERS * CHUNK / SR, 1),
        "out_len_per_chunk": sorted(out_lens),
        "cache_bytes_after": int(aec.cache_size_bytes()),
    }
    r.update(stats_ms(samples))
    return r


def bench_gtcrn_frame(iters: int = N_ITERS) -> dict:
    """GTCRN ONNX 单帧等效（T=1，16ms 帧移；intra_op=1 同 exp/05a bench.py）。"""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(GTCRN_ONNX), opts,
                                providers=["CPUExecutionProvider"])
    x = (np.random.default_rng(9).standard_normal((1, 257, 1, 2)) * 0.1).astype(np.float32)
    for _ in range(10):
        sess.run(None, {"spec": x})
    samples = []
    gc.disable()
    try:
        for _ in range(iters):
            t0 = time.perf_counter()
            sess.run(None, {"spec": x})
            samples.append((time.perf_counter() - t0) * 1000.0)
    finally:
        gc.enable()
    r = {"config": "onnxruntime CPU intra_op=1, T=1 单帧（16ms 帧移）",
         "warmup": 10, "timed": iters, "frame_budget_ms": 16.0}
    r.update(stats_ms(samples))
    return r


def bench_gtcrn_batch(dur: float = 4.0, reps: int = 5) -> dict:
    """GTCRN 整段批处理（capstone run_nr 的实际口径）：4s 音频 stft+模型+istft。"""
    import torch
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(GTCRN_ONNX), opts,
                                providers=["CPUExecutionProvider"])
    n = int(dur * SR)
    T = n // 256 + 1
    x = (np.random.default_rng(10).standard_normal((1, 257, T, 2)) * 0.1).astype(np.float32)
    for _ in range(3):
        sess.run(None, {"spec": x})
    t0 = time.perf_counter()
    for _ in range(reps):
        sess.run(None, {"spec": x})
    model_ms = (time.perf_counter() - t0) / reps * 1000.0

    wav = torch.from_numpy(
        (np.random.default_rng(11).standard_normal(n) * 0.1).astype(np.float32))
    w = torch.hann_window(512).pow(0.5)
    for _ in range(2):
        torch.stft(wav, 512, 256, 512, w, return_complex=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        spec = torch.stft(wav, 512, 256, 512, w, return_complex=True)
        torch.istft(spec, 512, 256, 512, w, length=n)
    stft_ms = (time.perf_counter() - t0) / reps * 1000.0
    return {
        "config": f"整段批处理 {dur}s（intra_op=1）",
        "dur_s": dur, "reps": reps, "frames_T": T,
        "model_ms_per_call": round(model_ms, 2),
        "stft_istft_ms_per_call": round(stft_ms, 2),
        "rtf_model": round(model_ms / (dur * 1000.0), 5),
        "rtf_e2e": round((model_ms + stft_ms) / (dur * 1000.0), 5),
        "note": "capstone run_nr 为整段批处理（非流式）；无 lookahead 概念，"
                "整段时延 = RTF × 段长；流式版需官方 stream/ 的 GRU 状态缓存实现",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None, help="JSON 落盘路径（默认仅 stdout）")
    args = ap.parse_args()

    import torch
    import psutil
    import onnxruntime

    meta = {
        "task": "P2-7 端到端时延账本实测",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cpu": cpu_name(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "torch_num_threads_default": torch.get_num_threads(),
        "cpu_percent_sample_start": psutil.cpu_percent(interval=0.3),
        "machine_note": "测量时机器有并行训练负载（任务背景说明）；"
                        "cpu_percent_* 为测量前后瞬时采样，p99 异常时按此解读",
    }

    result = {"meta": meta}
    print(f"[P2-7] CPU: {meta['cpu']}  ({meta['cores_physical']}C/{meta['cores_logical']}T)  "
          f"torch {meta['torch']} / ort {meta['onnxruntime']}  "
          f"torch_threads={meta['torch_num_threads_default']}  cpu%={meta['cpu_percent_sample_start']}")

    aec = load_aec()
    result["aec_stream_default_threads"] = bench_aec(
        aec, f"torch 默认线程({torch.get_num_threads()})")
    torch.set_num_threads(1)
    result["aec_stream_1thread"] = bench_aec(aec, "torch 单线程（与 RTF 0.0084 口径一致）")
    t0 = time.perf_counter()
    tail = aec.flush()
    result["aec_flush_ms_single_call"] = round((time.perf_counter() - t0) * 1000.0, 4)
    result["aec_flush_out_samples"] = int(len(tail))

    result["gtcrn_frame_1thread"] = bench_gtcrn_frame()
    result["gtcrn_batch_1thread"] = bench_gtcrn_batch()
    meta["cpu_percent_sample_end"] = psutil.cpu_percent(interval=0.3)

    print()
    print("=" * 76)
    print(f"{'项':<38}{'mean':>8}{'p50':>8}{'p95':>8}{'p99':>8}  (ms)")
    print("-" * 76)
    for key, tag in [("aec_stream_default_threads", "AEC 流式 20ms 块 (默认线程)"),
                     ("aec_stream_1thread", "AEC 流式 20ms 块 (单线程)")]:
        r = result[key]
        print(f"{tag:<38}{r['mean_ms']:>8.3f}{r['p50_ms']:>8.3f}"
              f"{r['p95_ms']:>8.3f}{r['p99_ms']:>8.3f}")
    g = result["gtcrn_frame_1thread"]
    print(f"{'GTCRN 单帧 T=1 (16ms 帧移)':<38}{g['mean_ms']:>8.3f}{g['p50_ms']:>8.3f}"
          f"{g['p95_ms']:>8.3f}{g['p99_ms']:>8.3f}")
    b = result["gtcrn_batch_1thread"]
    print(f"GTCRN 整段批 {b['dur_s']}s: model {b['model_ms_per_call']}ms + "
          f"stft/istft {b['stft_istft_ms_per_call']}ms -> RTF(e2e) {b['rtf_e2e']}")
    print(f"AEC flush 单次: {result['aec_flush_ms_single_call']}ms "
          f"(出 {result['aec_flush_out_samples']} 样本；仅流结束调用，不计稳态预算)")
    print(f"CPU% 采样: start {meta['cpu_percent_sample_start']} / end {meta['cpu_percent_sample_end']}")
    print("=" * 76)

    js = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(js, encoding="utf-8")
        print(f"[P2-7] JSON -> {args.json}")
    print(js)


if __name__ == "__main__":
    main()
