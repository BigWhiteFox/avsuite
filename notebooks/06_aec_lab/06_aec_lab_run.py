# -*- coding: utf-8 -*-
r"""P3-2 整改交付:DTD / MDF 可视化实验室(06_aec_lab)。

本文件是唯一代码源(jupytext 风格百分号分格):
  - `06_aec_lab.ipynb` 由 `make_notebook.py` 从本文件逐格生成(代码同源);
  - 因 avsuite 环境缺 nbformat/nbconvert/ipykernel 且禁止 pip 安装(P3-2 约束),
    本脚本承担实际执行:图件落 `outputs/`,notebook 保持未执行并在首格注明。
运行:
  & 'C:\Users\14805\.conda\envs\avsuite\python.exe' notebooks/06_aec_lab/06_aec_lab_run.py
"""
import csv
import sys
from pathlib import Path

# %% [markdown]
# # 06 路 DTD / MDF 可视化实验室(P3-2)
#
# 在受控合成集 `D:/data/aec-synth/eval_v3`(60 条,dt/fest/nst ≈ 0.5/0.25/0.25)上,
# 对 `exp/06_trad_aec` 的 **MDF(分块频域自适应滤波)** + **DTD(三态状态机)** 链路做四部分可视化:
#
# | 部分 | 内容 | 样本 |
# |---|---|---|
# | ① | mic / far / y_echo / e_lin 波形叠图 | DT:`fileid_49_dt` |
# | ② | 相干函数 γ² + ERLE EMA 时序,标注双讲来临段 | 同上 |
# | ③ | DTD 三态状态机着色时间轴(NE/DT/FES 三色横带)+ 决策量曲线 | 同上 |
# | ④ | MDF 逐 bin 收敛曲线(5 个 bin 的 \|H\|² vs 时间) | ST:`fileid_7_fest` |
#
# **选样依据**(全集探针 `probe_samples.py`,40 条 dt/fest 全跑):
# - `fileid_49_dt`:三态齐全(块占比 0.23/0.23/0.54),FEST 段 γ²≈0.70、近端活跃段 γ²≈0.42
#   (跌落清晰),DTD 对近端活跃块检出率 0.94,SER 3.4dB / 时延 18.1ms 居中;
# - `fileid_7_fest`:末段 ERLE EMA ≈ 22dB(过 DoD ≥20dB 线),γ²≈0.89,收敛过程最干净。
#
# > ⚠ **执行方式(P3-2 首格注记)**:本环境缺 nbformat/nbconvert/ipykernel 且禁止 pip 安装,
# > 故本 notebook **未执行**;同目录 `06_aec_lab_run.py`(与本 notebook 逐格同码)已用
# > avsuite python 跑通,图件落 `outputs/`——详见下格「执行方式注记」。

# %% [markdown]
# ## ⚠ 执行方式注记(P3-2 降级方案)
#
# avsuite 环境(Python 3.11.17)经核查 **缺 nbformat / nbconvert / ipykernel**,
# 且整改约束禁止 pip 安装,故:
#
# - 本 notebook 保持**未执行**状态(与 `notebooks/01–03` 既有风格一致);
# - 同目录 `06_aec_lab_run.py` 为与本 notebook **逐格一致**的执行脚本
#   (`make_notebook.py` 由脚本反向生成 ipynb,保证代码同源),已用 avsuite python 跑通;
# - 图件落 `notebooks/06_aec_lab/outputs/`(每格代码末尾统一 `save_fig(...)` + `plt.show()`);
# - 环境装上 nbconvert 后可原地执行:
#   `python -m nbconvert --to notebook --execute --inplace notebooks/06_aec_lab/06_aec_lab.ipynb`。

# %% ------------------------------------------------------------------------
import sys
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent          # 以脚本运行
except NameError:                                   # notebook:cwd 通常即本目录
    HERE = Path.cwd().resolve()
    if HERE.name != "06_aec_lab":
        HERE = HERE / "06_aec_lab"
REPO = HERE.parents[1]                              # .../avsuite

sys.path.insert(0, str(REPO / "exp" / "06_trad_aec"))
sys.path.insert(0, str(REPO))

import importlib                                    # noqa: E402

gcc = importlib.import_module("gcc")                # 同 tests/trad06.py 的 shim 手法:
mdf = importlib.import_module("mdf")                # 数字开头目录以模块名直接 import
dtd = importlib.import_module("dtd")
pipeline = importlib.import_module("pipeline")

import matplotlib                                   # noqa: E402
import numpy as np                                  # noqa: E402
import soundfile as sf                              # noqa: E402

try:                                                # notebook 有 get_ipython
    get_ipython                                     # noqa: B018
    IN_NB = True
except NameError:                                   # 脚本:无显示环境
    IN_NB = False
    matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.patches import Patch                # noqa: E402
from matplotlib import font_manager                 # noqa: E402

_cjk = [f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
        if any(f.lower() == x.name.lower()
               for x in font_manager.fontManager.ttflist)]
if _cjk:                                            # 图内中文可读(Windows 有雅黑)
    matplotlib.rcParams["font.sans-serif"] = _cjk + matplotlib.rcParams["font.sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False

if not IN_NB:                                       # Agg 下 show 是告警 no-op,图已落盘
    plt.show = lambda *a, **k: None

MDF, DTDStreaming = mdf.MDF, dtd.DTDStreaming
gcc_phat_stats, align_far = gcc.gcc_phat_stats, gcc.align_far
TradAEC, FREEZE, IDLE, EVIDENCE_THR = (pipeline.TradAEC, pipeline.FREEZE,
                                       pipeline.IDLE, pipeline.EVIDENCE_THR)

SR, BLOCK, PARTS, MU = 16000, 256, 12, 0.7          # TradAEC 默认口径(mdf: 12*256 taps)
DATA = Path("D:/data/aec-synth/eval_v3")
OUTDIR = HERE / "outputs"


def save_fig(fig, name):
    """图件统一落 outputs/(notebook 与脚本行为一致:落盘 + 内联/无操作显示)。"""
    OUTDIR.mkdir(exist_ok=True)
    p = OUTDIR / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print("saved ->", p.relative_to(REPO).as_posix())


print(f"repo={REPO}  IN_NB={IN_NB}  matplotlib={matplotlib.__version__}")

# %% ------------------------------------------------------------------------
from csv import DictReader                          # noqa: E402

DT_ID, ST_ID = "fileid_49_dt", "fileid_7_fest"      # 选样依据见首格 + probe_samples.py

rows = {r["name"]: r for r in
        DictReader(open(DATA / "manifest.csv", encoding="utf-8"))}
print(f"manifest: {len(rows)} 条 "
      f"(dt={sum(1 for r in rows.values() if r['scenario'] == 'dt')}, "
      f"fest={sum(1 for r in rows.values() if r['scenario'] == 'fest')}, "
      f"nst={sum(1 for r in rows.values() if r['scenario'] == 'nst')})")
print(f"DT 样本 {DT_ID}: {dict(rows[DT_ID])}")
print(f"ST 样本 {ST_ID}: {dict(rows[ST_ID])}")


def load_sample(sid):
    mic, _ = sf.read(DATA / "mic" / f"{sid}.wav", dtype="float32")
    far, _ = sf.read(DATA / "farend" / f"{sid}.wav", dtype="float32")
    near, _ = sf.read(DATA / "nearend" / f"{sid}.wav", dtype="float32")
    return tuple(x.astype(np.float64) for x in (mic, far, near))


def near_active(near, block=BLOCK):
    """近端活跃块掩码(真双讲段):块 rms > 5% 峰值(近端文件无混响/噪声,阈值干净)。"""
    nb = int(np.ceil(len(near) / block))
    brms = np.array([np.sqrt(np.mean(near[i * block:(i + 1) * block] ** 2) + 1e-20)
                     for i in range(nb)])
    return brms > max(0.05 * brms.max(), 1e-4)


def mask_spans(mask, block=BLOCK):
    """块掩码 -> [(t0, t1), ...] 秒区间,画 axvspan 用。"""
    spans, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            spans.append((start * block / SR, i * block / SR))
            start = None
    if start is not None:
        spans.append((start * block / SR, len(mask) * block / SR))
    return spans


mic_dt, far_dt, near_dt = load_sample(DT_ID)
mic_st, far_st, near_st = load_sample(ST_ID)
T_DT, T_ST = len(mic_dt) / SR, len(mic_st) / SR
near_dt_act, near_st_act = near_active(near_dt), near_active(near_st)
lead_dt = float(rows[DT_ID]["leadin_sec"])          # 合成时近端引导前静音秒数
print(f"{DT_ID}: {T_DT:.1f}s, 近端活跃块占比 {near_dt_act.mean():.2f}, leadin={lead_dt}s")
print(f"{ST_ID}: {T_ST:.1f}s(fest 全程无近端)")

# %% ------------------------------------------------------------------------
def run_instrumented(mic, far, snap_w=False):
    """镜像 `TradAEC.process`(mdf 分支)的在线链路,只**加记录、不改行为**:

    - 逐块 DTD 三态 state:0=NE 正常自适应,1=DT 软冻结(步长×FREEZE=0.02),
      2=硬冻结(远端静默 FES / 暖机 / 无回声)——步长规则与 pipeline.py 逐字一致;
    - DTD 决策量(直接取自 DTDStreaming 自身状态,与判决零拷贝):
      * γ²:mic-vs-y_echo EMA 谱相干(语音频段均值;仅远端活跃块更新,静默块为 NaN);
      * Geigel 边距:|mic|峰 / (α·远端滚动峰),>1 即 Geigel 判真;
      * ERLE EMA(0.9/0.1)与收敛闩锁 converged;
    - snap_w=True 时逐块快照 MDF 系数(本块更新前),供 ④ 逐 bin 收敛曲线。
    """
    n = len(mic)
    # V5 R8（blocker）：gcc_phat_stats 现返回 3 值 (delay_ms, peak_ratio, evidence_valid)
    # （t10 为 G1 非有限降级所加）——旧的 2 值解包会直接 ValueError。
    # 语义与 pipeline.process 对齐：证据失效/非有限时按"回声可能存在"降级（放行自适应，
    # 而不是把不可用证据当成"无回声"而永久硬冻结），对齐量取 0 避免 int(round(nan)) 崩溃。
    delay_ms, evidence, ev_valid = gcc_phat_stats(mic, far, SR)
    if (not ev_valid) or (not np.isfinite(evidence)):
        delay_ms, evidence, echo_prior = 0.0, float("nan"), True
    elif not np.isfinite(delay_ms):
        delay_ms, echo_prior = 0.0, True
    else:
        echo_prior = evidence >= EVIDENCE_THR        # GCC 峰旁瓣比整段回声存在性
    far_al = align_far(far, delay_ms, SR, n)
    n_blocks = int(np.ceil(n / BLOCK))
    aec = TradAEC(method="mdf", mu=MU, block=BLOCK, partitions=PARTS,
                  use_dtd=True, use_res=False)
    base_scale = aec._far_gate(far_al, n_blocks)     # 远端活跃门(1.0 / IDLE=0.0)
    af = MDF(block=BLOCK, partitions=PARTS, mu=MU)
    dtds = DTDStreaming(frame=BLOCK, sr=SR, echo_prior=echo_prior)

    state = np.zeros(n_blocks, dtype=int)
    coh = np.full(n_blocks, np.nan)                  # γ²(远端静默块不更新 → NaN 断线)
    geigel_m = np.full(n_blocks, np.nan)             # Geigel 边距
    erle = np.full(n_blocks, np.nan)                 # ERLE EMA
    far_act = np.zeros(n_blocks, dtype=bool)
    conv = np.zeros(n_blocks, dtype=bool)            # 收敛闩锁
    W_hist = [] if snap_w else None

    def hook(bi, fb, db, y):
        st = dtds.update(db, fb, y)
        state[bi] = st
        far_act[bi] = dtds.far_active
        conv[bi] = dtds.converged
        if snap_w:
            W_hist.append(af.W.copy())               # 本块系数更新前快照
        if dtds.far_active:
            coh[bi] = float(np.mean((np.abs(dtds.sxy) ** 2 /
                                     (dtds.sxx * dtds.syy + 1e-12))[dtds.bins]))
            geigel_m[bi] = float(np.abs(db).max() /
                                 (dtds.alpha * dtds.far_peaks.max() + 1e-10))
        erle[bi] = dtds.erle_ema
        # 与 pipeline.py 的 hook 相同的步长规则:
        return base_scale[bi] if st == 0 else (FREEZE if st == 1 else 0.0)

    e, y = af.process(far_al, mic, block_hook=hook, return_y=True)
    return dict(e=e, y=y, far_al=far_al, state=state, coh=coh, geigel=geigel_m,
                erle=erle, far_act=far_act, conv=conv, W_hist=W_hist,
                delay_ms=delay_ms, evidence=evidence, echo_prior=echo_prior,
                base_scale=base_scale)


# %% ------------------------------------------------------------------------
run_dt = run_instrumented(mic_dt, far_dt)
run_st = run_instrumented(mic_st, far_st, snap_w=True)

# 奇偶校验:镜像链路 vs 官方 TradAEC.process,应完全一致(证明"只加记录、不改行为")
for tag, m, f, r in (("DT", mic_dt, far_dt, run_dt), ("ST", mic_st, far_st, run_st)):
    res = TradAEC(method="mdf", mu=MU, block=BLOCK, partitions=PARTS,
                  use_dtd=True, use_res=False).process(m, f)
    de = float(np.max(np.abs(r["e"] - res.error)))
    dy = float(np.max(np.abs(r["y"] - res.y_echo)))
    assert de == 0.0 and dy == 0.0, f"{tag} 镜像链路与官方 pipeline 不一致: {de}, {dy}"
    print(f"[parity {tag}] max|de|={de:.1e}  max|dy|={dy:.1e}  [OK]")

st = run_dt["state"]
print(f"{DT_ID}: 时延估计 {run_dt['delay_ms']:.1f}ms, GCC 证据 {run_dt['evidence']:.1f}"
      f"(阈值 {EVIDENCE_THR:.0f}) -> echo_prior={run_dt['echo_prior']}")
print(f"三态块占比: NE={np.mean(st == 0):.2f}  DT={np.mean(st == 1):.2f}  "
      f"硬冻结={np.mean(st == 2):.2f};  dt_frac={np.mean(st > 0):.2f}")

# %% [markdown]
# ## ① 波形叠图:mic / far / y_echo / e_lin(DT 样本)
#
# **观察点**:leadin 段(0–2s,纯远端)mic≈回声,y_echo 收敛后逐块贴住 mic,e_lin 掉到
# 底噪量级(ERLE 上升);2s 双讲来临后 mic=回声+近端,y_echo 只解释回声部分,
# e_lin≈近端——**冻结保住了近端**。红影 = 近端活跃段(真双讲),虚线 = leadin 边界。

# %% ------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
t_dt = np.arange(len(mic_dt)) / SR
for ax in axes:
    for t0, t1 in mask_spans(near_dt_act):
        ax.axvspan(t0, t1, color="crimson", alpha=0.12, lw=0)
    ax.axvline(lead_dt, color="crimson", ls="--", lw=1, alpha=0.8)
    ax.grid(alpha=0.3)

axes[0].plot(t_dt, far_dt, c="tab:blue", lw=0.6)
axes[0].set_ylabel("far")
axes[1].plot(t_dt, mic_dt, c="tab:orange", lw=0.6)
axes[1].set_ylabel("mic")
axes[2].plot(t_dt, run_dt["y"], c="tab:green", lw=0.6)
axes[2].set_ylabel("y_echo")
axes[3].plot(t_dt, run_dt["e"], c="tab:red", lw=0.6)
axes[3].set_ylabel("e_lin")
axes[3].set_xlabel("time [s]")
axes[0].set_title(f"{DT_ID}(dt)  delay≈{run_dt['delay_ms']:.0f}ms  "
                  f"GCC evidence={run_dt['evidence']:.0f}  红影=近端活跃(双讲)")
fig.align_ylabels(axes)
fig.tight_layout()
save_fig(fig, f"p1_waveforms_{DT_ID}.png")
plt.show()

# %% [markdown]
# ## ② 相干函数 γ² 与 ERLE EMA 时序(同一样本,标注双讲来临段)
#
# γ² 是 **mic-vs-y_echo** 的 EMA 谱相干(DTDStreaming 的第二判据;远端静默块不更新→断线):
# leadin 段收敛后 γ²→0.9+(纯线性回声相干高);双讲来临后 y_echo 解释不了近端,γ² 跌破
# coh_thr=0.6 → 判 DT。ERLE EMA 是收敛闩锁的依据(连续 ≥6dB 才允许判 DT,防冷启动误判);
# 注意 ERLE 在双讲段被近端能量主导(虚高或拉低),只在远端单讲段可信——验收口径也只在
# FEST 段统计 ERLE(eval_aec.py 跳过收敛瞬态)。

# %% ------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
tt = np.arange(len(st)) * BLOCK / SR
for ax in (ax1, ax2):
    for t0, t1 in mask_spans(near_dt_act):
        ax.axvspan(t0, t1, color="crimson", alpha=0.12, lw=0)
    ax.axvline(lead_dt, color="crimson", ls="--", lw=1, alpha=0.8)
    ax.grid(alpha=0.3)

ax1.plot(tt, run_dt["coh"], c="tab:purple", lw=1.2, label="γ²  mic vs y_echo")
ax1.axhline(DTDStreaming().coh_thr, c="k", ls=":", lw=1, label="coh_thr = 0.6")
ax1.set_ylabel("coherence γ²")
ax1.set_ylim(0, 1.05)
ax1.annotate("双讲来临", xy=(lead_dt, 1.02), xytext=(lead_dt + 0.3, 0.9),
             arrowprops=dict(arrowstyle="->", color="crimson"), color="crimson")
ax1.legend(loc="lower left", fontsize=9)

ax2.plot(tt, run_dt["erle"], c="tab:brown", lw=1.2, label="ERLE EMA (0.9/0.1)")
ax2.axhline(DTDStreaming().erle_thr, c="k", ls=":", lw=1, label="erle_thr = 6 dB")
ax2.set_ylabel("ERLE [dB]")
ax2.set_xlabel("time [s]")
ax2.legend(loc="lower right", fontsize=9)
ax1.set_title(f"{DT_ID}(dt):γ² 在双讲来临段跌破 0.6;ERLE EMA 仅远端单讲段可信"
              f"(双讲段被近端能量扰动)")
fig.tight_layout()
save_fig(fig, f"p2_coh_erle_{DT_ID}.png")
plt.show()

# %% [markdown]
# ## ③ DTD 三态状态机着色时间轴(NE / DT / FES)+ 决策量曲线
#
# `DTDStreaming.update` 的三态:0=NE 正常自适应(绿),1=DT 软冻结、步长×0.02(红),
# 2=硬冻结、步长=0(灰蓝;覆盖远端静默 FES / 暖机 / hangover)。横带下方是两个决策量:
# γ²(双讲判据,`<0.6` 才判 DT)+ Geigel 边距(能量判据,`>1` 判真);灰底 = 远端静默块。
#
# **观察点**:本样本 SER 3.4dB(回声能量高),Geigel 边距全程 >1——能量判据单独会全程
# 误报,这正是要配第二判据的原因(dtd.py 头注:时域相关会被混响打散而相干性不会);
# leadin 段 γ²>0.6 → NE 绿带(收敛早期 γ² 未稳,偶发细红条误报,hangover/闩锁就是压它的);
# 双讲段 γ²<0.6 且 Geigel>1 → DT 红带;近端停顿后 γ² 回 0.9+ 自恢复 NE;
# 远端停顿 → FES 灰带,两个判定量同时断线(不更新)。

# %% ------------------------------------------------------------------------
NE, DT_, FES = 0, 1, 2
state_rgb = np.array([(0.16, 0.63, 0.32), (0.85, 0.26, 0.21), (0.45, 0.55, 0.68)])
fig, (ax0, ax1, ax2) = plt.subplots(
    3, 1, figsize=(11, 6.5), sharex=True,
    gridspec_kw=dict(height_ratios=[0.7, 1.6, 1.6]))

ax0.imshow(state_rgb[st][None, :, :], aspect="auto",
           extent=[0, tt[-1], 0, 1], interpolation="none")
ax0.set_yticks([])
ax0.legend(handles=[Patch(color=state_rgb[NE], label="0 NE 自适应"),
                    Patch(color=state_rgb[DT_], label="1 DT 软冻结(×0.02)"),
                    Patch(color=state_rgb[FES], label="2 硬冻结(FES/暖机)")],
           loc="upper right", ncols=3, fontsize=8, framealpha=0.9)
ax0.set_title(f"{DT_ID}(dt):DTD 三态时间轴 + 决策量")

for t0, t1 in mask_spans(~run_dt["far_act"]):
    for ax in (ax1, ax2):
        ax.axvspan(t0, t1, color="0.55", alpha=0.20, lw=0)
for t0, t1 in mask_spans(near_dt_act):
    for ax in (ax1, ax2):
        ax.axvspan(t0, t1, color="crimson", alpha=0.10, lw=0)

ax1.plot(tt, run_dt["coh"], c="tab:purple", lw=1.2)
ax1.axhline(DTDStreaming().coh_thr, c="k", ls=":", lw=1)
ax1.set_ylabel("γ²")
ax1.set_ylim(0, 1.05)
ax2.plot(tt, run_dt["geigel"], c="tab:cyan", lw=1.0, label="Geigel 边距")
ax2.axhline(1.0, c="k", ls=":", lw=1, label="判真线 =1")
ax2.set_yscale("log")
ax2.set_ylabel("Geigel |mic|峰/α·远端峰")
ax2.set_xlabel("time [s]")
ax2.legend(loc="upper left", fontsize=8)
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
fig.align_ylabels([ax0, ax1, ax2])
fig.tight_layout()
save_fig(fig, f"p3_dtd_states_{DT_ID}.png")
plt.show()

# %% [markdown]
# ## ④ MDF 逐 bin 收敛曲线(|H|² vs 时间,fest 样本)
#
# 每块快照一次系数(本块更新前),H(f)=Σ_p W_p(f) 为等效回声路径估计;取
# 250/500/1k/2k/3k Hz 共 5 个 bin 画 20log10|H|²。**观察点**:远端活跃块步长放开,
# 各 bin |H|² 快速爬升;远端停顿(灰底)时远端门=IDLE,曲线持平(冻结不学);
# 收敛后曲线进入平台,ERLE EMA 稳在 20dB 量级(末段 ≈22dB,过 DoD 线)。

# %% ------------------------------------------------------------------------
W_hist = np.asarray(run_st["W_hist"])                # (n_blocks, P, N/2+1) 复数
Hf = W_hist.sum(axis=1)                              # 等效频响 H(f)=Σ_p W_p(f)
H_db = 20 * np.log10(np.abs(Hf) + 1e-6)              # = 10log10|H|², dB
freqs = np.fft.rfftfreq(2 * BLOCK, 1 / SR)
pick = [250, 500, 1000, 2000, 3000]                  # 语音段等比取 5 个 bin
tt_st = np.arange(H_db.shape[0]) * BLOCK / SR        # 快照=块更新前状态

fig, ax = plt.subplots(figsize=(11, 4.5))
for t0, t1 in mask_spans(~run_st["far_act"]):
    ax.axvspan(t0, t1, color="0.55", alpha=0.18, lw=0)
for f0 in pick:
    k = int(np.argmin(np.abs(freqs - f0)))
    ax.plot(tt_st, H_db[:, k], lw=1.2, label=f"{freqs[k]:.0f} Hz")
ax.set_ylim(-80, 40)
ax.set_xlabel("time [s]")
ax.set_ylabel(r"$20\log_{10}|H(f)|^2$  [dB]")
ax.set_title(f"{ST_ID}(fest):MDF 逐 bin 收敛(灰底=远端静默块,不自适应)")
ax.legend(loc="lower right", fontsize=9, ncols=2)
ax.grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, f"p4_mdf_bins_{ST_ID}.png")
plt.show()

print(f"{ST_ID}: 末 40 块 ERLE EMA 均值 "
      f"{np.nanmean(run_st['erle'][-40:]):.1f} dB(DoD FEST ≥20dB)")

# %% [markdown]
# ## 读图小结
#
# - **②③ 是一件事的两面**:γ² 在双讲来临段跌破 0.6 + Geigel 边距 >1 + 收敛闩锁已闭合
#   → 状态机切 DT 软冻结;近端停顿后 γ² 回到 0.9+,状态机自动回到 NE(自恢复)。
# - **FES(灰带)不是浪费**:远端静默期无可学习,硬冻结防谱队列旧 far 与近端误差
#   形成虚假梯度(NST/静默期随机游走是实测过的坑,见 dtd.py 头注)。
# - **④ 的平台高度 = 收敛质量**:fest 样本末段 ERLE EMA ≈22dB,过 DoD;
#   远端停顿段曲线严格持平,验证"冻结只冻学习、不冻滤波"的口径。
#
# ### 复现
#
# ```powershell
# & 'C:\Users\14805\.conda\envs\avsuite\python.exe' notebooks/06_aec_lab/06_aec_lab_run.py
# # 环境具备 nbconvert 后:
# & 'C:\Users\14805\.conda\envs\avsuite\python.exe' -m nbconvert --to notebook --execute --inplace notebooks/06_aec_lab/06_aec_lab.ipynb
# ```
