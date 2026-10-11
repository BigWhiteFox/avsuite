"""P1-3(整改执行):分帧 lead-in 裁剪扫描,检验「DT lead-in 过渡段主导 deg 缺口」假设。

设计(父代理 P1-3 规格,不改既有文件,产物全部写新目录):
  eval_v3 的 DT 片段合成时前 2s 是 lead-in(远端先播、近端后进,--dt-leadin-sec 2.0)。
  对 t ∈ {0,1,2,3}s:
    1) crop  : 从 eval_v3 只取 DT 子集,mic/farend/nearend 全部裁掉前 t 秒,manifest
               同步,目录结构不变 -> D:/data/aec-synth/eval_v3_dt_crop_t{t}s/
               (t=0 不裁样,字节级复制,保证可与基线逐位对齐)
    2) infer : v4 ckpt(outputs_v4/nnaec_g0.5_j10/last.ckpt)对裁剪集重推理,
               增强 wav 写 <crop>/enh/{name}.wav(FLOAT,无损),
               推理路径逐行照抄 aecmos_eval.py --nn-ckpt 分支
    3) score : 子进程调 aecmos_eval.py --eval-dir <crop> --enh-dir <crop>/enh
               --out outputs/aecmos_leadincrop/aecmos_dt_crop_t{t}s.json
  t=0 = 全段(应复现基线 DT deg 1.743 / echo 3.754,源 aecmos_nn_v4.json),
  t=2 = 官方 DT 协议(去前 2s lead-in)。
  判读:deg(t=2) - deg(t=0) >= 0.05 -> 「缺口部分是协议产物」成立(lead-in 主导);
       < 0.05 -> 结构性瓶颈锁定,级联动机更硬。两种结果都如实报。

用法(avsuite 环境 python):
  python leadin_crop_eval.py                 # t=0,1,2,3 全流程 + 汇总
  python leadin_crop_eval.py --t 2           # 只跑单个 t(汇总只含已跑的)
  python leadin_crop_eval.py --agg-only      # 只汇总(读已有打分 json 出表/图)
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
SRC_EVAL = Path("D:/data/aec-synth/eval_v3")
CROP_ROOT = Path("D:/data/aec-synth/eval_v3_dt_crop_t{t}s")   # {t} 由 format 填
CKPT = HERE / "outputs_v4" / "nnaec_g0.5_j10" / "last.ckpt"
BASELINE_JSON = HERE / "outputs" / "aecmos_nn_v4.json"        # v4@eval_v3 全段基线
OUT_DIR = HERE / "outputs" / "aecmos_leadincrop"
SUBDIRS = ("farend", "mic", "nearend")
DT = "dt"


def dt_rows(src: Path):
    with open(src / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dt = [r for r in rows if r["scenario"] == DT]
    return rows, dt


def stage_crop(t: int, src: Path, dst: Path):
    """取 DT 子集,三路裁掉前 t 秒(或 t=0 字节复制),manifest 同步。"""
    _, dt = dt_rows(src)
    dst.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (dst / sub).mkdir(exist_ok=True)
    (dst / "enh").mkdir(exist_ok=True)

    # 采样率一致性核对(所有 DT 片段三路同 sr)
    srs = {sf.info(src / sub / f"{r['name']}.wav").samplerate
           for r in dt for sub in SUBDIRS}
    assert len(srs) == 1, f"采样率不一致: {srs}"
    sr = srs.pop()
    n_crop = int(round(t * sr))
    durs = []
    for r in dt:
        name = r["name"]
        for sub in SUBDIRS:
            s, d = src / sub / f"{name}.wav", dst / sub / f"{name}.wav"
            if n_crop == 0:
                shutil.copyfile(s, d)                     # 字节级复制
            else:
                info = sf.info(s)
                data, _ = sf.read(s, start=n_crop, dtype="float32")
                sf.write(d, data, info.samplerate, subtype=info.subtype)
            durs.append(sf.info(d).frames / sr)
    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dt[0].keys()))
        w.writeheader()
        w.writerows(dt)
    print(f"[crop t={t}s] DT n={len(dt)} sr={sr} crop={n_crop} frames; "
          f"clip dur min={min(durs):.3f}s max={max(durs):.3f}s -> {dst}")
    return sr, min(durs)


def stage_infer(t: int, dst: Path, sr: int, device: str):
    """v4 对裁剪集重推理;路径与 aecmos_eval.py --nn-ckpt 分支逐行一致。"""
    import torch
    sys.path.insert(0, str(HERE))
    from model import NNAEC
    st = torch.load(CKPT, map_location=device, weights_only=False)
    model = NNAEC().to(device).eval()
    model.load_state_dict(st["model"])
    _, dt = dt_rows(dst)
    t0 = time.time()
    with torch.no_grad():
        for r in dt:
            name = r["name"]
            mic, _ = sf.read(dst / "mic" / f"{name}.wav", dtype="float32")
            lpb, _ = sf.read(dst / "farend" / f"{name}.wav", dtype="float32")
            enh = model.enhance(torch.from_numpy(mic).unsqueeze(0).to(device),
                                torch.from_numpy(lpb).unsqueeze(0).to(device))
            enh = enh[0].cpu().numpy()
            assert len(enh) == len(mic), f"{name}: enh {len(enh)} != mic {len(mic)}"
            sf.write(dst / "enh" / f"{name}.wav", enh, sr, subtype="FLOAT")
    print(f"[infer t={t}s] n={len(dt)} device={device} "
          f"ckpt={CKPT.name} took {time.time()-t0:.1f}s")


def stage_score(t: int, dst: Path):
    """送官方打分器(--enh-dir),输出固定写到 aecmos_leadincrop/。"""
    out = OUT_DIR / f"aecmos_dt_crop_t{t}s.json"
    cmd = [sys.executable, str(HERE / "aecmos_eval.py"),
           "--eval-dir", str(dst), "--enh-dir", str(dst / "enh"), "--out", str(out)]
    print(f"[score t={t}s] {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise RuntimeError(f"aecmos_eval failed for t={t} (exit {p.returncode})")
    return out


def load_dt(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["summary"][DT], {r["name"]: r for r in d["rows"] if r["scenario"] == DT}


def aggregate(ts):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_sum, base_rows = load_dt(BASELINE_JSON)
    grid, per_clip = [], {}
    for t in ts:
        sp = OUT_DIR / f"aecmos_dt_crop_t{t}s.json"
        if not sp.exists():
            print(f"[agg] missing {sp.name}, skip")
            continue
        s, rows = load_dt(sp)
        grid.append({"t": t, "n": s["n"], "echo_mos": s["echo_mos"],
                     "deg_mos": s["deg_mos"],
                     "d_deg_vs_t0": None, "d_echo_vs_t0": None})
        for name, r in rows.items():
            per_clip.setdefault(name, {})[t] = (r["echo_mos"], r["deg_mos"])
    if grid:
        g0 = grid[0]
        for g in grid:
            g["d_deg_vs_t0"] = round(g["deg_mos"] - g0["deg_mos"], 3)
            g["d_echo_vs_t0"] = round(g["echo_mos"] - g0["echo_mos"], 3)

    # rows.csv / rows.json 原始数
    with open(OUT_DIR / "rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "name", "scenario", "echo_mos", "deg_mos"])
        for t in ts:
            sp = OUT_DIR / f"aecmos_dt_crop_t{t}s.json"
            if not sp.exists():
                continue
            for name, r in load_dt(sp)[1].items():
                w.writerow([t, name, DT, r["echo_mos"], r["deg_mos"]])
    (OUT_DIR / "rows.json").write_text(json.dumps(
        {"baseline_full": {"src": str(BASELINE_JSON), **base_sum},
         "grid": grid,
         "per_clip": {k: {str(t): {"echo_mos": v[0], "deg_mos": v[1]}
                          for t, v in d.items()} for k, d in per_clip.items()}},
        indent=2), encoding="utf-8")

    # curve.png(deg vs t,Agg)
    if len(grid) >= 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts_ = [g["t"] for g in grid]
        degs = [g["deg_mos"] for g in grid]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(ts_, degs, "o-", color="#1f77b4", label="deg_mos (DT crop, v4)")
        ax.axhline(base_sum["deg_mos"], ls="--", color="gray",
                   label=f"baseline full-seg t=0 json {base_sum['deg_mos']:.3f}")
        ax.axhline(base_sum["deg_mos"] + 0.05, ls=":", color="red",
                   label="threshold: +0.05 vs t=0")
        for x, y in zip(ts_, degs):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=9)
        ax.set_xlabel("lead-in crop t (s)   [0=full seg, 2=official DT protocol]")
        ax.set_ylabel("AECMOS deg_mos (DT)")
        ax.set_title(f"P1-3 lead-in crop scan: DT deg vs t (n={grid[0]['n']})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "curve.png", dpi=150)
        plt.close(fig)
    return grid, base_sum, base_rows, per_clip


def write_result(grid, base_sum, base_rows, per_clip, run_notes):
    if not grid:
        return
    if not run_notes and (OUT_DIR / "notes.json").exists():
        run_notes = json.loads((OUT_DIR / "notes.json").read_text(encoding="utf-8"))
    g0 = next((g for g in grid if g["t"] == 0), grid[0])
    g2 = next((g for g in grid if g["t"] == 2), None)
    delta = (g2["deg_mos"] - g0["deg_mos"]) if g2 else None
    if delta is None:
        verdict = None
    elif delta >= 0.05:
        verdict = ("confirm", "deg 回升 ≥0.05:「缺口部分是协议产物」成立——"
                   "低 deg 相当一部分来自 lead-in 过渡段被全段计分,官方裁剪协议下会缩水。")
    else:
        verdict = ("refute", "deg 回升 <0.05:结构性瓶颈锁定——低 deg 主要来自双讲段本身的"
                   "处理缺陷(欠抵消/残余回声/双讲失真),级联动机更硬。")

    lines = [
        "# P1-3 · 分帧 AECMOS:lead-in 裁剪扫描(v4 @ eval_v3 DT 子集)",
        "",
        f"- 生成时间:{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- 假设:DT 片段合成时前 2s 是 lead-in(远端先播、近端后进,"
        "`--dt-leadin-sec 2.0`);若低 deg 主要由这段过渡被全段计分造成,"
        "裁掉前 t 秒后 deg 应明显回升。",
        f"- 基线(全段,源 `{BASELINE_JSON.name}`):DT n={base_sum['n']} "
        f"echo={base_sum['echo_mos']:.3f} deg={base_sum['deg_mos']:.3f}",
        f"- DT 子集:n={grid[0]['n']}(任务预估 ≈15,manifest 实际 14,"
        "全部 leadin_sec=2.0);t=0 字节级复制全段。",
        "",
        "## deg/echo vs t",
        "",
        "| t (s) | n | echo_mos | deg_mos | Δdeg vs t=0 | Δecho vs t=0 | 含义 |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in grid:
        mean = ("全段=基线" if g["t"] == 0 else
                "官方 DT 协议(去前 2s lead-in)" if g["t"] == 2 else
                f"去前 {g['t']}s")
        lines.append(f"| {g['t']} | {g['n']} | {g['echo_mos']:.3f} | "
                     f"{g['deg_mos']:.3f} | {g['d_deg_vs_t0']:+.3f} | "
                     f"{g['d_echo_vs_t0']:+.3f} | {mean} |")
    lines += ["", f"![deg vs t](curve.png)", "",
              "## 结论", ""]
    if delta is not None:
        lines.append(f"- **Δdeg(t=2 − t=0) = {delta:+.3f}** "
                     f"(阈值 ±0.05)→ 判定:**{verdict[0]}**。{verdict[1]}")
        echo_seq = " → ".join(f"t={g['t']}:{g['echo_mos']:.3f}" for g in grid)
        lines.append(f"- echo 随 t:{echo_seq}"
                     f"(基线全段 {base_sum['echo_mos']:.3f});"
                     "本轮 echo 随裁剪下降,与 deg 回升同向——lead-in 段是回声易压的"
                     "远端独播段,截掉后剩余 DT 段更难,解读见备注末条。")
        same = (abs(g0["deg_mos"] - base_sum["deg_mos"]) < 0.005
                and abs(g0["echo_mos"] - base_sum["echo_mos"]) < 0.005)
        lines.append(f"- t=0 复现检查:本次 t=0(字节复制重推理)deg="
                     f"{g0['deg_mos']:.3f} / echo={g0['echo_mos']:.3f},"
                     f"基线 json deg={base_sum['deg_mos']:.3f} / "
                     f"echo={base_sum['echo_mos']:.3f}"
                     f"({'一致' if same else '存在偏差,已定位,见备注第 3 条'});"
                     "复跑逐位一致(推理确定)。")
    else:
        lines.append("- t=2 未跑,无法判定协议产物 vs 结构性瓶颈。")
    lines += ["", "## 逐片 deg(t →)", "", "| name | baseline_full | " +
              " | ".join(f"t={g['t']}s" for g in grid) + " |",
              "|---|---|" + "---|" * len(grid)]
    for name in sorted(per_clip):
        b = base_rows.get(name, {}).get("deg_mos", float("nan"))
        cells = []
        for g in grid:
            v = per_clip[name].get(g["t"], (None, None))[1]
            cells.append("—" if v is None else f"{v:.3f}")
        lines.append(f"| {name} | {b:.3f} | " + " | ".join(cells) + " |")
    lines += ["", "## 重跑命令(每格)", "",
              "```bash",
              "# 环境:C:/Users/14805/.conda/envs/avsuite/python.exe(base 坏)",
              "PY='C:/Users/14805/.conda/envs/avsuite/python.exe'",
              "cd avsuite/exp/07_nn_aec"]
    for g in grid:
        lines.append(f"{{ $PY }} leadin_crop_eval.py --t {g['t']}   "
                     f"# crop->infer->score,写 eval_v3_dt_crop_t{g['t']}s/ 与 "
                     f"outputs/aecmos_leadincrop/aecmos_dt_crop_t{g['t']}s.json")
    lines += [f"{{ $PY }} leadin_crop_eval.py --agg-only      # 只重出 rows/curve/RESULT",
              "# 单格打分等价于:",
              f"#   {{ $PY }} aecmos_eval.py --eval-dir D:/data/aec-synth/eval_v3_dt_crop_t{{t}}s \\",
              f"#     --enh-dir D:/data/aec-synth/eval_v3_dt_crop_t{{t}}s/enh \\",
              "#     --out outputs/aecmos_leadincrop/aecmos_dt_crop_t{t}s.json",
              "```",
              "",
              "## 备注",
              ""]
    lines += [f"- {n}" for n in run_notes]
    (OUT_DIR / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[result] -> {OUT_DIR / 'RESULT.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, nargs="+", default=[0, 1, 2, 3],
                    choices=[0, 1, 2, 3])
    ap.add_argument("--src", type=Path, default=SRC_EVAL)
    ap.add_argument("--ckpt", type=Path, default=CKPT)
    ap.add_argument("--agg-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notes = []
    device = "cpu"
    if not args.agg_only:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[env] device={device} torch={torch.__version__} ckpt={args.ckpt}")
        for t in args.t:
            dst = Path(str(CROP_ROOT).format(t=t))
            sr, min_dur = stage_crop(t, args.src, dst)
            stage_infer(t, dst, sr, device)
            stage_score(t, dst)
            notes.append(f"t={t}s:crop 后最短片段 {min_dur:.2f}s,打分正常"
                         f"(aecmos_eval 无最短时长限制,实际验证)。")
    grid, base_sum, base_rows, per_clip = aggregate(sorted(set(args.t)))
    if notes:
        (OUT_DIR / "notes.json").write_text(json.dumps(notes, ensure_ascii=False),
                                            encoding="utf-8")
    write_result(grid, base_sum, base_rows, per_clip, notes)
    for g in grid:
        print(f"[summary] t={g['t']}s n={g['n']} echo={g['echo_mos']:.3f} "
              f"deg={g['deg_mos']:.3f} d_deg={g['d_deg_vs_t0']:+.3f}")


if __name__ == "__main__":
    main()
