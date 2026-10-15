"""P0-5：汇总全部 run 的 ckpt 元数据（路径 / sha256 / epoch / val / loss 口径）到 index 表。

`eval_p05_matrix.py` 每次调用只写当次 runs 的 index.json（会被后续调用覆盖），
本脚本按已知 run 清单重建**全量**元数据表，作为产物留痕（V3.1 引用/清单纪律）。

    python exp/07_nn_aec/p05_index.py
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_v4_p05"
RUNS = [
    ("v4", HERE / "outputs_v4" / "nnaec_g0.5_j10" / "best.ckpt"),
    ("lossctrl_240ep", OUT / "lossctrl_240ep" / "nnaec_g0.5_j10" / "best.ckpt"),
    ("psycho_240ep", OUT / "psycho_240ep" / "nnaec_g0.5_j10_psycho" / "best.ckpt"),
    ("lossctrl_240ep_seed1", OUT / "lossctrl_240ep_seed1" / "nnaec_g0.5_j10" / "best.ckpt"),
    ("psycho_240ep_seed1", OUT / "psycho_240ep_seed1" / "nnaec_g0.5_j10_psycho" / "best.ckpt"),
    ("psycho_cfix_seed1", OUT / "psycho_cfix_seed1" / "nnaec_g0.5_j10_psycho" / "best.ckpt"),
    ("lossctrl_v4replica", OUT / "lossctrl_v4replica" / "nnaec_g0.5_j10" / "best.ckpt"),
    ("freshctrl_60ep_aborted", OUT / "lossctrl" / "nnaec_g0.5_j10" / "best.ckpt"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for name, p in RUNS:
        if not p.exists():
            print(f"[skip] {name}: {p} 不存在")
            continue
        st = torch.load(p, map_location="cpu", weights_only=False)
        cfg = st.get("cfg") or {}
        rows.append({
            "variant": name, "ckpt": p.relative_to(HERE.parents[1]).as_posix(),
            "ckpt_sha256": sha256(p), "epoch": st.get("epoch"), "val": st.get("val"),
            "loss_kind": cfg.get("loss_kind", "weighted"),
            "psycho_scale": cfg.get("psycho_scale"),
            "psycho_probe": cfg.get("psycho_probe"),
            "lr": cfg.get("lr"), "seed": cfg.get("seed"), "epochs": cfg.get("epochs"),
            "early_stop": cfg.get("early_stop"), "scenarios": cfg.get("scenarios"),
        })
    out = OUT / "aecmos" / "runs_index.json"
    out.write_text(json.dumps({"runs": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("| run | ckpt | sha256 | ep | best val | loss | c |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['variant']} | `{r['ckpt']}` | `{r['ckpt_sha256'][:16]}…` | {r['epoch']} | "
              f"{r['val']:.4f} | {r['loss_kind']}{'' if r['psycho_scale'] is None else ' c='+format(r['psycho_scale'],'.3f')} | "
              f"{r['seed']} |")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
