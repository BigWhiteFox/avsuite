#!/usr/bin/env python
"""跨平台 ckpt 加载兼容层（t13 打分前置；服务器 ckpt 在 Windows 上的 PosixPath 坑）。

问题（train-eng 实测复现）
--------------------------
服务器（Linux）保存的 ckpt 里 `cfg` 含 `pathlib.PosixPath` 对象；在 Windows 上
`torch.load(..., weights_only=False)` 反序列化时抛
`NotImplementedError: cannot instantiate 'PosixPath' on your system`
（`weights_only=True` 也救不了——它并不改变 pickle 对 pathlib 类的构造）。

解法（2 行打桩，加载前把 PosixPath 指成 PurePosixPath）
--------------------------------------------------------
    import pathlib
    pathlib.PosixPath = pathlib.PurePosixPath
    st = torch.load(p, map_location="cpu", weights_only=False)

本模块把这件事固化成可复用入口，避免每个打分脚本各写一遍、也避免忘记：
- `posixpath_shim()`：上下文管理器，**用完恢复**原 `PosixPath`（避免污染同进程其它逻辑）；
- `load_ckpt(path, ...)`：打桩 + `torch.load`；失败时给出带路径的可读报错；
- `ckpt_summary(st)`：抽取 epoch/best/参数量/loss 口径（w_sisnr、loss_kind、rir 相关）供矩阵登记；
- `audit_ckpts(paths)`：批量加载 + 汇总（**不**跑推理，只做元数据体检）。

CLI
---
    python exp/07_nn_aec/ckpt_compat.py --ckpt <best.ckpt> [--ckpt ...] [--json out.json]
    python exp/07_nn_aec/ckpt_compat.py --glob D:/Project_for_Python/DSH/p1_artifacts/outputs_p1

边界：只解决"能否在本机 load"；ckpt 的**训练口径**（epoch 数、w_sisnr、RIR 池是否净池）
需要按 t13 协议另行核对（见 `reports/multi_axis_matrix_20261013.md` §4.3）。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys
from pathlib import Path

import torch

#: 需要打桩的类：Linux pickle 里出现的是 PosixPath；Windows 上对应的是 WindowsPath
_SHIM = {"PosixPath": pathlib.PurePosixPath}


@contextlib.contextmanager
def posixpath_shim():
    """临时把 `pathlib.PosixPath` 指成 `PurePosixPath`，退出时恢复。"""
    saved = {name: getattr(pathlib, name) for name in _SHIM}
    for name, repl in _SHIM.items():
        setattr(pathlib, name, repl)
    try:
        yield
    finally:
        for name, orig in saved.items():
            setattr(pathlib, name, orig)


def load_ckpt(path, map_location="cpu", weights_only=False, patch=True):
    """加载 ckpt；`patch=True` 时自动打 PosixPath 桩。返回 torch.load 的结果。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"ckpt 不存在：{p}")
    ctx = posixpath_shim() if patch else contextlib.nullcontext()
    try:
        with ctx:
            return torch.load(p, map_location=map_location, weights_only=weights_only)
    except NotImplementedError as e:                       # pragma: no cover - 极端环境
        raise RuntimeError(
            f"ckpt 反序列化失败（{p}）：{e}；若报 'PosixPath'，说明打桩未生效") from None


def ckpt_summary(st, path=None) -> dict:
    """从 state dict 抽元数据（供矩阵登记；不做推理）。

    ``numel_state_dict`` = 保存的 `model` 状态字典元素总数（**含 BN buffer**）；
    架构参数量（`model.parameters()`）比它少（NN-AEC 实测 638,264 vs **638,034**，
    差 230 = 各 BN 的 running_mean/var/num_batches_tracked）——引用时写清是哪一个。
    """
    cfg = st.get("cfg") or {}
    cfg = dict(cfg) if isinstance(cfg, dict) else {"raw": str(cfg)}
    model = st.get("model") or {}
    numel_sd = int(sum(v.numel() for v in model.values())) if hasattr(model, "values") else 0
    keys_of_interest = ("loss", "loss_kind", "loss_sig", "w_sisnr", "gamma", "epochs",
                        "epoch_len", "lr", "batch_size", "seed", "rir_dir", "rir_max_ms",
                        "near_dir", "far_dir", "noise_dir", "dm_scenarios", "out", "val")
    return {
        "path": str(path) if path else None,
        "epoch": st.get("epoch"),
        "best": st.get("best"),
        "step": st.get("step"),
        "numel_state_dict": numel_sd,
        "top_keys": sorted(st.keys()),
        "cfg": {k: str(cfg[k]) for k in keys_of_interest if k in cfg},
        "cfg_keys": sorted(cfg),
    }


def audit_ckpts(paths) -> list[dict]:
    out = []
    for p in paths:
        st = load_ckpt(p)
        s = ckpt_summary(st, p)
        out.append(s)
        print(f"[ckpt] {Path(p).parent.parent.name}/{Path(p).parent.name}/{Path(p).name}: "
              f"state_dict={s['numel_state_dict']} epoch={s['epoch']} best={s['best']} "
              f"loss={s['cfg'].get('loss')} w_sisnr={s['cfg'].get('w_sisnr')}")
    return out


def _expand(a) -> list[Path]:
    paths: list[Path] = list(a.ckpt or [])
    for g in a.glob or []:
        base = Path(g)
        paths.extend(sorted(base.rglob("*.ckpt")) if base.is_dir() else [base])
    seen, uniq = set(), []
    for p in paths:
        rp = Path(p).resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(Path(p))
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, action="append", default=None, help="单个 ckpt（可重复）")
    ap.add_argument("--glob", action="append", default=None, help="目录或 glob（可重复，递归找 *.ckpt）")
    ap.add_argument("--json", type=Path, default=None, help="汇总落盘")
    args = ap.parse_args()
    paths = _expand(args)
    if not paths:
        print("[ckpt_compat] 未给 --ckpt/--glob", file=sys.stderr)
        return 2
    try:
        rows = audit_ckpts(paths)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ckpt_compat] 输入错误：{e}", file=sys.stderr)
        return 2
    print(f"[ckpt_compat] 共 {len(rows)} 个 ckpt 全部可加载（PosixPath 打桩生效）")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
        print(f"  -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
