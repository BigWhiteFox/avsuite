"""P0-5 评测驱动：对若干 ckpt × 若干冻结评测集跑 AECMOS@48k 模型协议。

口径与仓内一致（复用 `aecmos_eval.py` 的会话与打分函数，不另写一套打分逻辑）：
    - 送评前统一重采样到模型采样率 48k（resample_poly，gcd 约减）；
    - 场景映射 fest->st / dt->dt / nst->nst；
    - 逐文件 rows 落盘，供 bootstrap（B=2000, seed=777）复用。

用法：
    python exp/07_nn_aec/eval_p05_matrix.py \
        --ckpt v4=exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/best.ckpt \
        --ckpt lossctrl=exp/07_nn_aec/outputs_v4_p05/lossctrl/nnaec_g0.5_j10/best.ckpt \
        --ckpt psycho=exp/07_nn_aec/outputs_v4_p05/psycho/nnaec_g0.5_j10_psycho/best.ckpt \
        --sets eval_v3=D:/data/aec-synth/eval_v3 eval_v5=D:/data/aec-synth/eval_v5 \
        --out-dir exp/07_nn_aec/outputs_v4_p05/aecmos
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import aecmos_eval as AE  # noqa: E402  （复用口径与打分会话）
from model import NNAEC  # noqa: E402


def sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="name=path（可重复）")
    ap.add_argument("--sets", nargs="+", required=True, help="name=dir（可重复）")
    ap.add_argument("--out-dir", type=Path,
                    default=HERE / "outputs_v4_p05" / "aecmos")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpts = []
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"[ckpt] 不存在：{p}")
        ckpts.append((name, p))
    sets = []
    for spec in args.sets:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not (p / "manifest.csv").exists():
            raise SystemExit(f"[set] 缺 manifest：{p}")
        sets.append((name, p))
    device = torch.device(args.device)

    sys.path.insert(0, str(AE.AECMOS_DIR))
    from aecmos import AECMOSEstimator
    est = AECMOSEstimator(str(AE.AECMOS_MODEL))
    sr_model = int(getattr(est, "sampling_rate", 16000))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index = {"aecmos_model": str(AE.AECMOS_MODEL), "aecmos_sr_model": sr_model,
             "resample": "scipy.signal.resample_poly(gcd-reduced) -> model sr",
             "runs": []}
    for cname, cpath in ckpts:
        st = torch.load(cpath, map_location=device, weights_only=False)
        model = NNAEC().to(device).eval()
        model.load_state_dict(st["model"])
        cfg = st.get("cfg", {})
        rec = {"variant": cname, "ckpt": str(cpath), "ckpt_sha256": sha256(cpath),
               "epoch": st.get("epoch"), "val": st.get("val"),
               "loss_kind": cfg.get("loss_kind", "weighted"),
               "psycho_scale": cfg.get("psycho_scale"),
               "psycho_probe": cfg.get("psycho_probe"), "sets": {}}
        for sname, sdir in sets:
            clips = AE.load_clips(sdir)
            sr_in = int(sf.info(sdir / "farend" / f"{clips[0][0]}.wav").samplerate)
            pairs = []
            with torch.no_grad():
                for name, sc, lpb, mic in clips:
                    enh = model.enhance(torch.from_numpy(mic).unsqueeze(0).to(device),
                                        torch.from_numpy(lpb).unsqueeze(0).to(device))
                    pairs.append((name, sc, lpb, mic, enh[0].cpu().numpy()))
            rows, summary = AE.score_all(pairs, est, sr_in=sr_in)
            out = args.out_dir / sname / f"aecmos_{cname}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "tag": f"p05_{cname}", "variant": cname, "eval_set": sname,
                "aecmos_model": str(AE.AECMOS_MODEL), "aecmos_sr_model": sr_model,
                "input_sr": sr_in, "ckpt": str(cpath),
                "ckpt_sha256": sha256(cpath), "epoch": st.get("epoch"),
                "val": st.get("val"), "loss_kind": cfg.get("loss_kind", "weighted"),
                "psycho_scale": cfg.get("psycho_scale"),
                "summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                encoding="utf-8")
            rec["sets"][sname] = summary
            msg = "  ".join(f"{sc}:n={v['n']} echo={v['echo_mos']:.3f} deg={v['deg_mos']:.3f}"
                            for sc, v in summary.items())
            print(f"[{cname}@{sname}] {msg}", flush=True)
            print(f"    -> {out}", flush=True)
        index["runs"].append(rec)
    (args.out_dir / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[index] -> {args.out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
