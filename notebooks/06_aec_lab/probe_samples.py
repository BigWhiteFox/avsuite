# -*- coding: utf-8 -*-
r"""P3-2 选样探针:在 eval_v3 全集上快跑 MDF+DTD 链路,输出选样依据。

只读数据 + 只读仓内代码;产物仅 stdout 表格。运行:
  & 'C:\Users\14805\.conda\envs\avsuite\python.exe' notebooks/06_aec_lab/probe_samples.py
"""
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # .../avsuite
sys.path.insert(0, str(REPO / "exp" / "06_trad_aec"))
sys.path.insert(0, str(REPO))

import numpy as np                                  # noqa: E402
import soundfile as sf                              # noqa: E402

from gcc import gcc_phat_stats, align_far           # noqa: E402
from mdf import MDF                                 # noqa: E402
from dtd import DTDStreaming                        # noqa: E402
from pipeline import FREEZE, EVIDENCE_THR, TradAEC  # noqa: E402

SR, BLOCK, PARTS, MU = 16000, 256, 12, 0.7
DATA = Path("D:/data/aec-synth/eval_v3")


def run_probe(mic, far):
    """镜像 TradAEC.process 的 MDF 分支,记录三态/γ²/ERLE EMA(无系数快照)。"""
    n = len(mic)
    delay_ms, evidence = gcc_phat_stats(mic, far, SR)
    echo_prior = evidence >= EVIDENCE_THR
    far_al = align_far(far, delay_ms, SR, n)
    n_blocks = int(np.ceil(n / BLOCK))
    aec = TradAEC(method="mdf", mu=MU, block=BLOCK, partitions=PARTS,
                  use_dtd=True, use_res=False)
    base_scale = aec._far_gate(far_al, n_blocks)
    af = MDF(block=BLOCK, partitions=PARTS, mu=MU)
    dtds = DTDStreaming(frame=BLOCK, sr=SR, echo_prior=echo_prior)
    state = np.zeros(n_blocks, dtype=int)
    coh = np.full(n_blocks, np.nan)
    erle = np.full(n_blocks, np.nan)
    far_act = np.zeros(n_blocks, dtype=bool)

    def hook(bi, fb, db, y):
        st = dtds.update(db, fb, y)
        state[bi] = st
        far_act[bi] = dtds.far_active
        if dtds.far_active:
            coh[bi] = float(np.mean((np.abs(dtds.sxy) ** 2 /
                                     (dtds.sxx * dtds.syy + 1e-12))[dtds.bins]))
        erle[bi] = dtds.erle_ema
        return base_scale[bi] if st == 0 else (FREEZE if st == 1 else 0.0)

    e, y = af.process(far_al, mic, block_hook=hook, return_y=True)
    res = aec.process(mic, far)                      # 官方链路对照(奇偶校验)
    par = float(np.max(np.abs(e - res.error)))
    return dict(delay=delay_ms, ev=evidence, state=state, coh=coh, erle=erle,
                far_act=far_act, e=e, y=y, parity=par)


def main():
    rows = list(csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")))
    print(f"{'name':16s} {'sc':4s} {'ser':>5s} {'dly':>5s} {'ev':>6s} "
          f"{'st0':>5s} {'st1':>5s} {'st2':>5s} {'cohPre':>6s} {'cohPost':>7s} "
          f"{'erle@8s':>7s} {'nearAct':>7s} {'parity':>9s}")
    for r in rows:
        name, sc = r["name"], r["scenario"]
        if sc == "nst":
            continue                                  # 本任务只看 dt / fest
        mic, _ = sf.read(DATA / "mic" / f"{name}.wav", dtype="float32")
        far, _ = sf.read(DATA / "farend" / f"{name}.wav", dtype="float32")
        near, _ = sf.read(DATA / "nearend" / f"{name}.wav", dtype="float32")
        mic, far, near = (x.astype(np.float64) for x in (mic, far, near))
        out = run_probe(mic, far)
        st = out["state"]
        f0, f1, f2 = (float((st == k).mean()) for k in (0, 1, 2))
        nb = len(st)
        brms = np.array([np.sqrt(np.mean(near[i * BLOCK:(i + 1) * BLOCK] ** 2))
                         for i in range(nb)])
        near_act = brms > max(0.05 * brms.max(), 1e-4)
        lead = float(r["leadin_sec"])
        pre = out["coh"][:int(lead * SR / BLOCK)]
        post = out["coh"][int(lead * SR / BLOCK):]
        erle_end = float(np.nanmean(out["erle"][-40:]))
        if sc == "dt":
            # 细分:近端活跃块(真 DT)上的检测率与 γ²,近端停顿块上的 γ²
            act = near_act & out["far_act"]
            pau = (~near_act) & out["far_act"] & (np.arange(nb) >= lead * SR // BLOCK)
            det = float(np.mean(out["state"][act] == 1)) if act.any() else float("nan")
            g_na = float(np.nanmean(out["coh"][act])) if act.any() else float("nan")
            g_pa = float(np.nanmean(out["coh"][pau])) if pau.any() else float("nan")
            extra = f"{det:5.2f} {g_na:7.2f} {g_pa:7.2f}"
        else:
            extra = " " * 21
        print(f"{name:16s} {sc:4s} {float(r['ser_db']):5.1f} "
              f"{float(r['delay_ms']):5.1f} {out['ev']:6.1f} "
              f"{f0:5.2f} {f1:5.2f} {f2:5.2f} "
              f"{np.nanmean(pre) if len(pre) else float('nan'):6.2f} "
              f"{np.nanmean(post) if len(post) else float('nan'):7.2f} "
              f"{erle_end:7.1f} {near_act.mean():7.2f} {out['parity']:9.2e} {extra}")


if __name__ == "__main__":
    main()
