# -*- coding: utf-8 -*-
"""GTCRN ONNX 导出（WP-05a）：PyTorch 官方权重 -> fp32 ONNX（动态 T 轴）-> 简化。

    python onnx_export.py                     # 导出 outputs/gtcrn_dns3.onnx
    python onnx_export.py --ckpt vctk         # VCTK 权重版
    python onnx_export.py --check             # 导出后立即对拍（<1e-4 DoD）
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CKPT_DNS3, CKPT_VCTK, load_model  # noqa: E402

OUT = Path(__file__).parent / "outputs"


def export(ckpt: Path, onnx_path: Path, opset=17):
    model = load_model(ckpt)
    dummy = torch.randn(1, 257, 201, 2)               # (B,F,T,2)，T≈4s@hop256
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(onnx_path), opset_version=opset,
        input_names=["spec"], output_names=["enh"],
        dynamic_axes={"spec": {0: "batch", 2: "time"}, "enh": {0: "batch", 2: "time"}},
        do_constant_folding=True)
    size_mb = onnx_path.stat().st_size / 1024 ** 2
    print(f"[export] {onnx_path} ({size_mb:.2f} MB, opset {opset})")

    # onnx 简化（onnxsim）：常量折叠/冗余节点消除
    try:
        import onnx
        from onnxsim import simplify
        m = onnx.load(str(onnx_path))
        m_sim, ok = simplify(m)
        if ok:
            simp_path = onnx_path.with_suffix(".sim.onnx")
            onnx.save(m_sim, str(simp_path))
            n0, n1 = len(m.graph.node), len(m_sim.graph.node)
            print(f"[simplify] 节点 {n0} -> {n1}  {simp_path.name} "
                  f"({simp_path.stat().st_size/1024**2:.2f} MB)")
    except ImportError:
        print("[simplify] onnxsim 不可用，跳过（不影响对拍/基准）")
    return onnx_path


def parity(onnx_path: Path, ckpt: Path):
    """PyTorch vs ONNXRuntime 对拍（DoD：<1e-4）。"""
    import onnxruntime as ort
    model = load_model(ckpt)
    x = torch.randn(1, 257, 101, 2)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"spec": x.numpy()})[0]
    err = np.abs(out - ref).max()
    print(f"[parity] max|onnx - torch| = {err:.2e}  DoD(<1e-4): "
          f"{'PASS' if err < 1e-4 else 'FAIL'}")
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="dns3", choices=["dns3", "vctk"])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ckpt = CKPT_DNS3 if args.ckpt == "dns3" else CKPT_VCTK
    onnx_path = OUT / f"gtcrn_{args.ckpt}.onnx"
    t0 = time.perf_counter()
    export(ckpt, onnx_path)
    print(f"[export] {time.perf_counter()-t0:.1f}s")
    if args.check:
        parity(onnx_path, ckpt)


if __name__ == "__main__":
    main()
