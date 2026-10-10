# -*- coding: utf-8 -*-
"""环境自检（WP-00）：Python / GPU / 关键库 / 数据目录 报告。

用法：
    python env_check.py            # 汇总报告（退出码恒 0）
    python env_check.py --strict   # 任一关键库缺失 -> 退出码 1（CI/脚本用）
"""
import argparse
import importlib
import platform
import sys
from pathlib import Path

CRITICAL = ("torch", "numpy", "scipy", "soundfile", "pesq", "pystoi",
            "matplotlib", "pandas", "pytest")
OPTIONAL = ("librosa", "onnxruntime", "tensorboard", "gradio")

DATA_ROOT = Path("D:/data")
DATA_DIRS = {
    "voicebank-demand-16k (WP-03)": "voicebank-demand-16k",
    "dns-challenge-mirror (WP-04)": "dns-challenge-mirror",
    "aec-challenge (WP-06/07)": "aec-challenge",
    "repos (RIR/官方repo)": "repos",
}


def probe(mod):
    try:
        m = importlib.import_module(mod)
        return True, str(getattr(m, "__version__", "?"))
    except Exception as e:
        return False, f"MISSING({type(e).__name__})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    print("=" * 64)
    print(f"avsuite 环境自检   python {platform.python_version()}  [{platform.platform()}]")
    print("=" * 64)

    ok = True
    print("\n[1] 关键库（批次1-3 必需）")
    for mod in CRITICAL:
        hit, ver = probe(mod)
        ok &= hit
        print(f"  {'OK ' if hit else 'MISS':4s} {mod:12s} {ver}")

    print("\n[2] 可选库（后续批次用）")
    for mod in OPTIONAL:
        hit, ver = probe(mod)
        print(f"  {'OK ' if hit else 'MISS':4s} {mod:12s} {ver}")

    print("\n[3] GPU / torch")
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"  torch {torch.__version__}  cuda_available={cuda}")
        if cuda:
            p = torch.cuda.get_device_properties(0)
            print(f"  GPU: {p.name}  {p.total_memory / 1024**3:.1f} GB  cc{p.major}.{p.minor}")
        else:
            print("  !! GPU 不可用：批次2/4 训练类 WP 需要 CUDA；批次1 代码与单测不受影响")
    except ImportError:
        ok = False
        print("  MISS torch")

    print("\n[4] 数据目录（D:/data，顶层条目数）")
    for label, sub in DATA_DIRS.items():
        p = DATA_ROOT / sub
        if p.exists():
            try:
                n = len(list(p.iterdir()))
                print(f"  OK  {label:30s} {p}  ({n} 项)")
            except OSError as e:
                print(f"  WARN {label} {e}")
        else:
            print(f"  MISS {label:30s} {p} 未创建")

    print("\n结论:", "PASS" if ok else "FAIL（缺关键库，见 [1]/[3]）")
    return (0 if ok else 1) if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
