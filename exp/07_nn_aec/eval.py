"""NN-AEC 评估（WP-07）：分场景 ERLE / 近端 PESQ / STOI + 与 WP-06 传统基线对比表。

    python eval.py --ckpt outputs/nnaec_g0.5_j10/best.ckpt \
        --eval-dir D:/data/aec-synth/eval_v3

对比表（WORKLIST DoD：双讲较传统基线 +0.3 / 近端 PESQ 不明显掉）：
    python eval.py --ckpt ... --compare-trad     # 同时跑 TradAEC 生成对比 markdown

时延鲁棒性（WORKLIST 降级路径也要求的实验）：
    python eval.py --ckpt ... --jitter-sweep     # far 平移 ±5/10/20/40ms 的 PESQ 衰减曲线
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "06_trad_aec"))

from ckpt_compat import load_ckpt  # noqa: E402 - 服务器 ckpt 含 PosixPath，须打桩加载
from dataset import _shift  # noqa: E402
from model import NNAEC  # noqa: E402

from core.evaluate import _safe_pesq, _safe_stoi, _si_snr_np  # noqa: E402

SR = 16000
SKIP_S = 2.0        # ERLE 跳收敛瞬态——与 WP-06 eval_aec 同口径（对比公平性）


def erle_db(sig, ref, skip_s=SKIP_S):
    k = int(skip_s * SR)
    return 10 * np.log10((np.sum(ref[k:] ** 2) + 1e-12) / (np.sum(sig[k:] ** 2) + 1e-12))


def load_model(ckpt, device):
    st = load_ckpt(ckpt, map_location=device)      # PosixPath 打桩在 ckpt_compat 内
    m = NNAEC().to(device).eval()
    m.load_state_dict(st["model"])
    return m


@torch.no_grad()
def enhance(model, mic, far, device):
    enh = model.enhance(torch.from_numpy(mic).unsqueeze(0).to(device),
                        torch.from_numpy(far).unsqueeze(0).to(device))
    return enh[0].cpu().numpy()


def eval_dir(model, eval_dir, device, jitter_ms=0.0, seed=0):
    eval_dir = Path(eval_dir)
    with open(eval_dir / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    # V5 v3b-012（high）：jitter RNG 必须在循环**外**创建。旧实现每轮都以同一 seed
    # 重建 RNG → 全评估集共享同一个平移量，"±5/10/20/40ms 衰减曲线"实为单点重复
    # （伪曲线，结论不可用）。移出后每个样本拿到独立抽样；首个样本与旧实现同值
    # （首个抽样未变），便于与既有数字对照。
    rng = np.random.default_rng(seed)
    for r in rows:
        mic = sf.read(eval_dir / "mic" / f"{r['name']}.wav", dtype="float32")[0]
        far = sf.read(eval_dir / "farend" / f"{r['name']}.wav", dtype="float32")[0]
        near = sf.read(eval_dir / "nearend" / f"{r['name']}.wav", dtype="float32")[0]
        if jitter_ms:
            far = _shift(far, int(rng.integers(-jitter_ms, jitter_ms + 1) * SR / 1000))
        enh = enhance(model, mic.astype(np.float32), far.astype(np.float32), device) \
            .astype(np.float32)
        row = {"name": r["name"], "scenario": r["scenario"]}
        if r["scenario"] == "fest":
            row["erle_db"] = erle_db(enh, mic)
        else:
            n = min(len(near), len(enh))
            row["pesq"] = _safe_pesq(near[:n], enh[:n])
            row["stoi"] = _safe_stoi(near[:n], enh[:n])
            row["sisnr"] = _si_snr_np(near[:n], enh[:n])
            row["pesq_mic"] = _safe_pesq(near[:n], mic[:n])
        out.append(row)
    return out


def summarize(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    s = {}
    fest = df[df.scenario == "fest"]
    if len(fest):
        s["fest_erle_db"] = round(float(fest.erle_db.mean()), 2)
    for sc in ("dt", "nst"):
        sub = df[df.scenario == sc].dropna(subset=["pesq"]) if "pesq" in df else df.iloc[:0]
        if len(sub):
            s[f"{sc}_pesq"] = round(float(sub.pesq.mean()), 3)
            s[f"{sc}_pesq_mic"] = round(float(sub.pesq_mic.mean()), 3)
            s[f"{sc}_stoi"] = round(float(sub.stoi.mean()), 3)
    return s


def compare_trad(eval_dir, device):
    """跑 WP-06 TradAEC 同评估集，返回 (trad_rows_summary)。"""
    from pipeline import TradAEC
    aec = TradAEC(method="mdf", partitions=8, mu=0.7)
    eval_dir = Path(eval_dir)
    rows = []
    with open(eval_dir / "manifest.csv", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))     # 先物化：语义与原逐行流式一致（顺序不变）
    for r in manifest_rows:
        mic = sf.read(eval_dir / "mic" / f"{r['name']}.wav", dtype="float32")[0]
        far = sf.read(eval_dir / "farend" / f"{r['name']}.wav", dtype="float32")[0]
        near = sf.read(eval_dir / "nearend" / f"{r['name']}.wav", dtype="float32")[0]
        res = aec.process(mic.astype(np.float64), far.astype(np.float64))
        out = res.out.astype(np.float32)
        row = {"name": r["name"], "scenario": r["scenario"]}
        if r["scenario"] == "fest":
            row["erle_db"] = erle_db(out, mic)
        else:
            n = min(len(near), len(out))
            row["pesq"] = _safe_pesq(near[:n], out[:n])
            row["stoi"] = _safe_stoi(near[:n], out[:n])
            row["pesq_mic"] = _safe_pesq(near[:n], mic[:n])
        rows.append(row)
    return summarize(rows)


def _json_safe(obj):
    """递归把非有限浮点（NaN/±Inf）换成 None。

    V5 终审 F11：SI-SNR 口径改为对静音/塌陷条目返回 NaN 后，`json.dumps` 默认会把
    它们写成非标准 JSON 的 `NaN` token（严格解析器/其它语言会拒绝）。落盘前先 sanitize，
    并配 `allow_nan=False` 让"漏网的 NaN"直接报错而不是静默写出非标文件。
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--eval-dir", type=Path, default=Path("D:/data/aec-synth/eval_v3"))
    ap.add_argument("--compare-trad", action="store_true")
    ap.add_argument("--jitter-sweep", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = args.out or Path(__file__).parent / "outputs" / "eval" / args.ckpt.parent.name
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, device)
    rows = eval_dir(model, args.eval_dir, device)
    s = summarize(rows)
    print("[nn-aec]", json.dumps(s, indent=2))
    (out / "summary.json").write_text(
        json.dumps(_json_safe(s), indent=2, allow_nan=False))
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)

    if args.compare_trad:
        t = compare_trad(args.eval_dir, device)
        lines = ["| 指标 | 传统 AEC (WP-06) | NN-AEC (WP-07) | Δ |", "|---|---|---|---|"]
        for k in ("fest_erle_db", "dt_pesq", "dt_stoi", "nst_pesq"):
            tv, nv = t.get(k), s.get(k)
            if tv is not None and nv is not None:
                lines.append(f"| {k} | {tv} | {nv} | {nv - tv:+.3f} |")
        (out / "compare_trad_vs_nn.md").write_text(
            "\n".join(lines) +
            "\n\nDoD：双讲指标较传统基线显著提升且近端 PESQ 不明显掉。\n")
        print("\n".join(lines))

    if args.jitter_sweep:
        sweep = {}
        for j in (5, 10, 20, 40):
            rj = eval_dir(model, args.eval_dir, device, jitter_ms=j, seed=7)
            sweep[f"+-{j}ms"] = summarize(rj)
            print(f"jitter ±{j}ms:", sweep[f"+-{j}ms"])
        (out / "jitter_sweep.json").write_text(
            json.dumps(_json_safe(sweep), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
