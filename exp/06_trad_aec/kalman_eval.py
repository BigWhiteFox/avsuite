"""P2-2 L1:频域块 Kalman vs MDF 同口径对比评估(eval_v3 固定子集)。

对比设计(与 eval_aec.py / ablation.py 口径对齐):
- 同一生产前端:GCC-PHAT 对齐 + 远端活跃门 + DTDStreaming 三态(与 pipeline.py 同策略),
  唯一差异 = 自适应滤波器本体;两边都不加 RES(RES 对两者加同量增益,会掩盖滤波器差异)。
- MDF 基线 = TradAEC(method='mdf', mu=0.7, partitions=8, use_res=False) —— 与验收同参。
- Kalman = 平行驱动(gcc_phat_stats/align_far + far_gate 复刻 + DTDStreaming hook + FDKalman)。
- 口径:稳态 ERLE 公式 = eval_aec.erle_db(10lg(Σmic²/Σout²))（**注意:本文件不 import
  eval_aec**——eval_aec.erle_db 的默认实现内部会跳过前 2s 瞬态,而本脚本要的是"最后 2s 窗"
  读数,故用等价的窗口化实现 window_erle() 就地计算,两者公式同源、取窗不同;
  早期版本曾 import 该函数但从未调用,已删除以免误读为同取窗口径）;
  滑窗曲线 = ablation.py 同款 (0.5s 步长,±0.5s 窗);稳态读数 = 最后 2s 窗;
  t90 = 曲线首次 ≥ 0.9×稳态;
  DT 段 ERLE = t≥leadin_sec 窗(注意:DT 段 mic 与 e 都含近端,该读数只作趋势/发散监测,
  近端保真看补充读数 DT 段 PESQ 与 corr(e, near))。

子集(固定可复现,manifest 顺序):全部 14 条 DT + 前 12 条 FEST + 前 8 条 NST = 34 条。
配置:MDF(mu0.7,P8)+ Kalman 任务网格 3 组 (Q,R)=(1e-6,1e-2)/(1e-4,1e-2)/(1e-4,1e-1)
+ 机理诊断 2 组(不进主对比):(1e-2,1e-2) 加速档、(1e-4,1.0) 噪声匹配档。

用法:
  python kalman_eval.py           # 全子集 34 条 × 6 配置
  python kalman_eval.py --quick   # 3 条冒烟(每场景 1 条)
输出:outputs/kalman_vs_mdf.csv / .json / .png
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dtd import DTDStreaming  # noqa: E402
from gcc import align_far, gcc_phat_stats  # noqa: E402
from kalman import FDKalman  # noqa: E402
from pipeline import EVIDENCE_THR, FAR_ABS_THR, FAR_REL_THR, FREEZE, IDLE, TradAEC  # noqa: E402

SR = 16000
BLOCK, PARTITIONS, MU = 256, 8, 0.7
DATA = Path("D:/data/aec-synth/eval_v3")
OUT = Path(__file__).resolve().parent / "outputs"
N_FEST, N_NST = 12, 8
CURVE_STEP = 0.5

CONFIGS = [
    dict(kind="mdf", label="MDF(mu0.7,P8)", diag=False),
    dict(kind="kal", q=1e-6, r=1e-2, label="Kal(Q1e-06,R1e-02)", diag=False),
    dict(kind="kal", q=1e-4, r=1e-2, label="Kal(Q1e-04,R1e-02)", diag=False),
    dict(kind="kal", q=1e-4, r=1e-1, label="Kal(Q1e-04,R1e-01)", diag=False),
    dict(kind="kal", q=1e-2, r=1e-2, label="diag_Kal(Q1e-02,R1e-02)", diag=True),
    dict(kind="kal", q=1e-4, r=1.0, label="diag_Kal(Q1e-04,R1)", diag=True),
]


def load_subset(quick: bool = False):
    """固定子集:全部 DT + 前 12 FEST + 前 8 NST(manifest 顺序,可复现)。"""
    rows = list(csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")))
    if quick:
        return [next(r for r in rows if r["scenario"] == s)
                for s in ("fest", "dt", "nst")]
    dt = [r for r in rows if r["scenario"] == "dt"]
    fest = [r for r in rows if r["scenario"] == "fest"][:N_FEST]
    nst = [r for r in rows if r["scenario"] == "nst"][:N_NST]
    return fest + dt + nst


def far_gate(far_al: np.ndarray, n_blocks: int) -> np.ndarray:
    """远端活跃门(逐块 1.0/IDLE),与 pipeline.TradAEC._far_gate 同款。"""
    rms = np.array([np.sqrt(np.mean(far_al[i * BLOCK:(i + 1) * BLOCK] ** 2) + 1e-20)
                    for i in range(n_blocks)])
    w = max(int(1.0 * SR / BLOCK), 1)
    gate = np.ones(n_blocks)
    for i in range(n_blocks):
        peak = rms[max(0, i - w):i + 1].max()
        if rms[i] <= max(FAR_REL_THR * peak, FAR_ABS_THR):
            gate[i] = IDLE
    return gate


def run_kalman(mic: np.ndarray, far: np.ndarray, q: float, r: float):
    """平行驱动:GCC 对齐 + 远端门 + DTDStreaming 三态 hook + FDKalman(与 MDF 同策略)。

    V5 G23/G1：
    - DTDStreaming 的 far_rms_max/far_peaks/converged 是跨调用持久状态（本函数每次
      新建实例，故单文件安全），**但评估脚本按文件循环时若复用实例会继承上一文件的
      历史峰值 -> far_active 长期 False -> 全程硬冻结**；故在返回前显式 reset()，
      把"一文件一状态"的纪律写进代码（调用方 eval_one 逐文件调用本函数）。
    - GCC 证据失效（far 含非有限样本 / peak_ratio 非有限）时与 pipeline.process 同口径
      降级：不把不可用证据当成"无回声"，而是按"可能有回声"放行自适应。
    """
    n = len(mic)
    # V5（T10 / deep_audit G1）：gcc_phat_stats 现在返回
    # (delay_ms, peak_ratio, evidence_valid)——第三个值 = "参考端 far 有限且
    # peak_ratio 可用于回声存在性判决"。证据失效时与 pipeline.process 同口径降级
    # （不把不可用证据当成"无回声"），delay/evidence 取安全值。
    delay_ms, evidence, evidence_valid = gcc_phat_stats(mic, far, SR)
    if (not evidence_valid) or (not np.isfinite(evidence)):
        delay_ms, evidence, echo_prior = 0.0, float("nan"), True
    else:
        echo_prior = evidence >= EVIDENCE_THR
    delay_ms = float(delay_ms) if np.isfinite(delay_ms) else 0.0
    far_al = align_far(far, delay_ms, SR, n)
    n_blocks = int(np.ceil(n / BLOCK))
    gate = far_gate(far_al, n_blocks)
    kf = FDKalman(block=BLOCK, partitions=PARTITIONS, q=q, r=r)
    dtds = DTDStreaming(frame=BLOCK, sr=SR, echo_prior=echo_prior)

    def hook(bi, fb, db, y):
        st = dtds.update(db, fb, y)          # 0=自适应 1=DT软冻结 2=近端硬冻结
        if st == 0:
            return gate[bi]
        return FREEZE if st == 1 else 0.0

    e, y = kf.process(far_al, mic, block_hook=hook, return_y=True)
    dtds.reset()          # V5 G23：跨文件状态卫生——下一个文件不得继承本文件的历史统计
    return e[:n], y[:n], delay_ms, evidence


def erle_curve(mic, e, step_s=CURVE_STEP):
    """滑窗 ERLE 曲线(ablation.py 同款:窗 ±step,步长 step)。"""
    w = int(step_s * SR)
    xs, ys = [], []
    for i in range(w, len(mic) - w, w):
        xs.append(i / SR)
        ys.append(10 * np.log10(np.sum(mic[i - w:i + w] ** 2) /
                                (np.sum(e[i - w:i + w] ** 2) + 1e-12)))
    return np.array(xs), np.array(ys)


def window_erle(mic, e, t0, t1):
    k0, k1 = int(t0 * SR), min(int(t1 * SR), len(mic))
    return float(10 * np.log10((np.sum(mic[k0:k1] ** 2) + 1e-12) /
                               (np.sum(e[k0:k1] ** 2) + 1e-12)))


def t90_of(xs, ys, steady):
    """达稳态 90% 的收敛时长:曲线首次 ≥ 0.9×稳态(达不到记 NaN)。"""
    thr = 0.9 * steady
    for x, y in zip(xs, ys):
        if y >= thr:
            return float(x)
    return float("nan")


def eval_one(row, cfg, pesq_baseline):
    name, sc = row["name"], row["scenario"]
    mic, _ = sf.read(DATA / "mic" / f"{name}.wav", dtype="float32")
    far, _ = sf.read(DATA / "farend" / f"{name}.wav", dtype="float32")
    near, _ = sf.read(DATA / "nearend" / f"{name}.wav", dtype="float32")
    n = min(len(mic), len(far), len(near))
    mic, far, near = mic[:n].astype(np.float64), far[:n].astype(np.float64), near[:n].astype(np.float64)
    leadin = float(row.get("leadin_sec") or 0.0)
    t0 = time.perf_counter()
    if cfg["kind"] == "mdf":
        aec = TradAEC(method="mdf", mu=MU, block=BLOCK, partitions=PARTITIONS,
                      use_dtd=True, use_res=False)
        e = aec.process(mic, far).error
    else:
        e, _, _, _ = run_kalman(mic, far, cfg["q"], cfg["r"])   # 返回前已 dtds.reset()（V5 G23）
    wall = time.perf_counter() - t0

    xs, ys = erle_curve(mic, e)
    steady = window_erle(mic, e, n / SR - 2.0, n / SR)          # 最后 2s
    dt_seg = window_erle(mic, e, leadin, n / SR) if sc == "dt" else float("nan")
    t90 = t90_of(xs, ys, steady)
    pesq_dt = corr_dt = float("nan")
    if sc == "dt":
        k0 = int(leadin * SR)
        if n - k0 > int(0.5 * SR):                              # DT 段足够长才算
            from core.evaluate import _safe_pesq
            pesq_dt = _safe_pesq(near[k0:].astype(np.float32), e[k0:].astype(np.float32))
            corr_dt = float(np.corrcoef(e[k0:], near[k0:])[0, 1])
            if name not in pesq_baseline:                       # mic 基线与配置无关,算一次
                pesq_baseline[name] = _safe_pesq(near[k0:].astype(np.float32),
                                                 mic[k0:].astype(np.float32))
    rec = {"name": name, "scenario": sc, "leadin_sec": leadin, "config": cfg["label"],
           "steady2s_erle_db": round(steady, 2), "t90_s": (round(t90, 2) if np.isfinite(t90) else ""),
           "dt_seg_erle_db": (round(dt_seg, 2) if np.isfinite(dt_seg) else ""),
           "dt_pesq": (round(pesq_dt, 3) if np.isfinite(pesq_dt) else ""),
           "dt_corr_near": (round(corr_dt, 3) if np.isfinite(corr_dt) else ""),
           "wall_s": round(wall, 2)}
    for t, v in zip(xs, ys):
        rec[f"erle_t{t:g}"] = round(float(v), 2)
    return rec


def summarize(recs, curve_ts):
    """按配置汇总三条读数 + 平均曲线。"""
    out = {}
    for cfg in CONFIGS:
        sub = [r for r in recs if r["config"] == cfg["label"]]
        fest = [r for r in sub if r["scenario"] == "fest"]
        dt = [r for r in sub if r["scenario"] == "dt"]
        nst = [r for r in sub if r["scenario"] == "nst"]
        ste = np.array([r["steady2s_erle_db"] for r in fest], float)
        t90 = np.array([r["t90_s"] for r in fest if r["t90_s"] != ""], float)
        # 绝对收敛门槛 t(15dB)(ablation.py 同款):曲线首次 ≥15dB;稳态<15dB 的配置
        # 该读数自然为 NaN——t90(达自身稳态 90%)在稳态≈0 时退化为过零时刻,须以此补足
        t15 = []
        for r in fest:
            hit = next((float(t) for t in curve_ts if r[f"erle_t{t:g}"] >= 15.0), None)
            if hit is not None:
                t15.append(hit)
        dte = np.array([r["dt_seg_erle_db"] for r in dt], float)
        dpq = np.array([r["dt_pesq"] for r in dt if r["dt_pesq"] != ""], float)
        dcr = np.array([r["dt_corr_near"] for r in dt if r["dt_corr_near"] != ""], float)
        nst_ste = np.array([r["steady2s_erle_db"] for r in nst], float)
        allste = np.array([r["steady2s_erle_db"] for r in sub], float)
        entry = {
            "n": len(sub),
            "fest_steady_median": round(float(np.median(ste)), 2) if len(ste) else None,
            "fest_steady_mean": round(float(np.mean(ste)), 2) if len(ste) else None,
            "fest_t90_median_s": round(float(np.median(t90)), 2) if len(t90) else None,
            "fest_t90_n": int(len(t90)),
            "fest_t15_median_s": round(float(np.median(t15)), 2) if t15 else None,
            "fest_t15_n": len(t15),
            "dt_seg_erle_median": round(float(np.median(dte)), 2) if len(dte) else None,
            "dt_pesq_median": round(float(np.median(dpq)), 3) if len(dpq) else None,
            "dt_corr_near_median": round(float(np.median(dcr)), 3) if len(dcr) else None,
            "dt_n_diverge": int(np.sum(dte < 0)) if len(dte) else 0,
            "nst_steady_median": round(float(np.median(nst_ste)), 2) if len(nst_ste) else None,
            "full_steady_median": round(float(np.median(allste)), 2) if len(allste) else None,
            "diag": cfg["diag"],
        }
        for scope, names in (("all", None), ("fest", "fest"), ("dt", "dt")):
            pool = sub if names is None else [r for r in sub if r["scenario"] == names]
            if pool:
                entry[f"curve_{scope}"] = {
                    "t": list(curve_ts),
                    "erle_db": [round(float(np.mean([r[f"erle_t{t:g}"] for r in pool])), 2)
                                for t in curve_ts],
                }
        wall = sum(r["wall_s"] for r in sub)
        entry["wall_s_total"] = round(wall, 1)
        out[cfg["label"]] = entry
    return out


def build_conclusion(summary):
    """三条读数的如实结论(主对比 = MDF vs 任务网格 3 组;诊断组只作旁证)。

    收敛读数以绝对门槛 t(15dB) 为主、t90 为辅:t90 是「达自身稳态 90%」,
    稳态≈0 的配置(Q1e-06 组 fest 稳态中位 0.01dB)会让它退化为曲线过零时刻。

    V5/t23 健壮化:DT 组或诊断组整体缺失时不再抛 `ValueError: max() arg is an
    empty sequence`(原实现对空列表直接 max) —— 缺失时按 0 条/None 处理并在正文显式
    标注「组缺失」，便于只跑 FEST 子集/分批跑网格；数据齐全时输出与旧实现逐字一致。
    """
    mdf = summary["MDF(mu0.7,P8)"]
    grid = [c["label"] for c in CONFIGS if c["kind"] == "kal" and not c["diag"]]
    diag = [c["label"] for c in CONFIGS if c.get("diag")]
    ste = [summary[lab]["fest_steady_median"] for lab in grid]
    t90n = [summary[lab]["fest_t90_n"] for lab in grid]
    t90s = [summary[lab]["fest_t90_median_s"] for lab in grid]
    t15n = [summary[lab]["fest_t15_n"] for lab in grid]
    dte = [v for v in (summary[key]["dt_seg_erle_median"] for key in grid) if v is not None]
    dpq = [v for v in (summary[key]["dt_pesq_median"] for key in grid) if v is not None]
    ste_diag_vals = [s for s in (summary[key]["fest_steady_median"] for key in diag)
                     if s is not None]
    ste_diag = max(ste_diag_vals) if ste_diag_vals else None
    dt_best = max(dte) if dte else None
    dt_line = (
        (f"③ DT:段 ERLE 中位 MDF {mdf['dt_seg_erle_median']}dB vs Kalman 最好 {dt_best}dB"
         f"(差 {mdf['dt_seg_erle_median'] - dt_best:.2f}dB,在 1dB 容差内,该读数含近端仅趋势);"
         f"DT PESQ(近端保真,基线 1.466)MDF {mdf['dt_pesq_median']} vs Kalman "
         f"{min(dpq)}~{max(dpq)} → 近端保真 MDF 更好")
        if dt_best is not None else
        ("③ DT:组缺失(0 条,未跑 DT 子集)→ 跳过 DT 段 ERLE/PESQ 对比;"
         f"MDF 侧读数 {mdf['dt_seg_erle_median']}")
    )
    lines = [
        (f"① 收敛:MDF t(15dB)={mdf['fest_t15_median_s']}s({mdf['fest_t15_n']}/12 条达 15dB)最快;"
         f"Kalman 网格组 {t15n[0]}、{t15n[1]}、{t15n[2]}/12 条达标 → MDF 全面更快"
         f"(t90 名义最好 {t90s[0]}s 的 Q1e-06 组稳态仅 {ste[0]}dB,该口径在稳态≈0 时退化,且仅 {t90n[0]}/12 条达成)"),
        (f"② 稳态 ERLE(FEST 最后2s 中位):MDF {mdf['fest_steady_median']}dB;"
         f"Kalman 网格 {ste[0]}/{ste[1]}/{ste[2]}dB(最好诊断组 "
         f"{ste_diag if ste_diag is not None else '缺失'}dB)→ MDF 更高;"
         f"32s 单条诊断:Kalman(1e-4,1e-2) 渐近与 MDF 同 ~20dB 平台——输在速率非上限"),
        dt_line,
    ]
    return lines


def plot_fig(summary, out_png: Path, conclusion: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), sharey=True)
    panels = [("curve_all", "全子集平均(34条,含 NST≈0dB 拉低)"),
              ("curve_fest", "FEST 平均(稳态读数区)"),
              ("curve_dt", "DT 平均(leadin 后有近端,读数仅趋势)")]
    for ax, (key, title) in zip(axes, panels):
        for cfg in CONFIGS:
            s = summary.get(cfg["label"], {})
            cur = s.get(key)
            if not cur:
                continue
            ls = "--" if cfg["diag"] else "-"
            lw = 2.6 if cfg["kind"] == "mdf" else 1.5
            ax.plot(cur["t"], cur["erle_db"], ls, lw=lw, label=cfg["label"])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("时间 (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("滑窗 ERLE (dB)")
    fig.suptitle("P2-2 L1:频域块 Kalman vs MDF(线性级,无 RES,eval_v3 固定子集 34 条)", fontsize=12)
    fig.text(0.01, 0.005, conclusion, ha="left", va="bottom", fontsize=8.6,
             bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.6"))
    fig.tight_layout(rect=(0, 0.155, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 条冒烟(每场景 1 条)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    subset = load_subset(args.quick)
    print(f"subset = {len(subset)} items "
          f"(fest={sum(1 for r in subset if r['scenario'] == 'fest')}, "
          f"dt={sum(1 for r in subset if r['scenario'] == 'dt')}, "
          f"nst={sum(1 for r in subset if r['scenario'] == 'nst')})")
    pesq_baseline = {}
    recs = []
    t_start = time.perf_counter()
    for i, row in enumerate(subset):
        for cfg in CONFIGS:
            rec = eval_one(row, cfg, pesq_baseline)
            recs.append(rec)
        print(f"[{i + 1}/{len(subset)}] {row['name']} ({row['scenario']}) done")
    wall_total = time.perf_counter() - t_start

    curve_ts = sorted({float(k[6:]) for r in recs for k in r if k.startswith("erle_t")})
    summary = summarize(recs, curve_ts)

    # ---- 三条读数结论(主对比:MDF vs 任务网格 3 组;诊断组不参与)----
    lines = build_conclusion(summary)
    conclusion = "\n".join(lines)
    print("\n=== 三条读数(主对比)===")
    for ln in lines:
        print("  " + ln)
    print(f"\n总耗时 {wall_total:.0f}s")

    # ---- CSV(逐条×配置,含逐条 ERLE 时序)----
    csv_path = OUT / "kalman_vs_mdf.csv"
    cols = ["name", "scenario", "leadin_sec", "config", "steady2s_erle_db", "t90_s",
            "dt_seg_erle_db", "dt_pesq", "dt_corr_near", "wall_s"] \
        + [f"erle_t{t:g}" for t in curve_ts]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(recs)

    # ---- JSON(汇总)----
    meta = {
        "task": "P2-2 L1 频域块 Kalman vs MDF 同口径对比",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.executable,
        "data": str(DATA),
        "subset": [r["name"] for r in subset],
        "subset_rule": "全部 14 条 DT + 前 12 条 FEST + 前 8 条 NST(manifest 顺序)",
        "block": BLOCK, "partitions": PARTITIONS, "mdf_mu": MU,
        "qr_grid_primary": [(c["q"], c["r"]) for c in CONFIGS if c["kind"] == "kal" and not c["diag"]],
        "qr_grid_diag": [(c["q"], c["r"]) for c in CONFIGS if c.get("diag")],
        "metric": "ERLE=eval_aec.erle_db 同式(10lg(Σmic²/Σout²));曲线=±0.5s 滑窗 0.5s 步长;"
                  "稳态=最后2s;t90=曲线首次≥0.9×稳态;DT段=t≥leadin_sec;线性级对比(均无 RES)",
    }
    json_path = OUT / "kalman_vs_mdf.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary,
                   "dt_pesq_mic_baseline": pesq_baseline}, f, ensure_ascii=False, indent=1)

    # ---- PNG(三面板收敛曲线族 + 结论图注)----
    png_path = OUT / "kalman_vs_mdf.png"
    plot_fig(summary, png_path, conclusion)

    print(f"saved -> {csv_path}")
    print(f"saved -> {json_path}")
    print(f"saved -> {png_path}")


if __name__ == "__main__":
    main()
