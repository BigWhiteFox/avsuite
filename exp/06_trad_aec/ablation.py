"""WP-06 消融实验：NLMS vs MDF 收敛曲线 + MDF 步长扫描。

受控信号（确定性种子）：真实语音 far + pyroomacoustics 房间 RIR（100ms 截断）
+ 30ms 时延 + FEST（无近端无噪声，纯回声路径辨识问题）。

    python ablation.py            # 全部（~1-2 分钟 CPU）
输出：outputs/ablation_convergence.png + ablation_stepsize.png + stdout markdown 表
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import soundfile as sf  # noqa: E402
from gcc import align_far, gcc_phat_stats  # noqa: E402
from mdf import MDF  # noqa: E402
from nlms import NLMS  # noqa: E402
from scipy.signal import fftconvolve  # noqa: E402

try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

SR = 16000
OUT = Path(__file__).parent / "outputs"
TAPS = 2048


def make_signal(dur=8.0, seed=0):
    """FEST 受控信号：语音 far -> pra RIR(100ms) -> 30ms 时延 -> mic。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import aec_synth as S
    rng = np.random.default_rng(seed)
    far, _ = sf.read("D:/data/voicebank-demand-16k/extracted/train/clean/p226_001.wav",
                     dtype="float32")
    far = far.astype(np.float64)
    while len(far) < dur * SR:
        far = np.concatenate([far, far])
    far = far[: int(dur * SR)] * 0.3
    rir = S.rir_simulate_pra(rng)[: int(0.1 * SR)]
    echo = fftconvolve(far, rir)[: len(far)]
    d = int(0.03 * SR)
    mic = np.concatenate([np.zeros(d), echo])[: len(far)]
    mic = mic / np.std(mic) * 0.1
    return far, mic


def erle_curve(mic, e, step_s=0.5):
    """滑窗 ERLE 曲线（dB vs 秒）。"""
    w = int(step_s * SR)
    xs, ys = [], []
    for i in range(w, len(mic) - w, w):
        xs.append(i / SR)
        ys.append(10 * np.log10(np.sum(mic[i - w:i + w] ** 2) /
                                (np.sum(e[i - w:i + w] ** 2) + 1e-12)))
    return np.array(xs), np.array(ys)


def run_filter(kind, far_al, mic, mu):
    if kind == "mdf":
        return MDF(block=256, partitions=TAPS // 256, mu=mu).process(far_al, mic)
    return NLMS(length=TAPS, mu=mu).process(far_al, mic)


def main():
    OUT.mkdir(exist_ok=True)
    far, mic = make_signal()
    d_est, ratio = gcc_phat_stats(mic, far, SR)
    far_al = align_far(far, d_est, SR, len(mic))
    print(f"GCC: delay {d_est:.1f}ms (真值 30ms+直达径), 峰旁瓣比 {ratio:.1f}\n")

    # ---- 1) NLMS vs MDF 收敛曲线 ----
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for kind, mu in (("nlms", 0.5), ("mdf", 0.7)):
        t0 = time.perf_counter()
        af = (MDF(block=256, partitions=TAPS // 256, mu=mu) if kind == "mdf"
              else NLMS(length=TAPS, mu=mu))
        e = af.process(far_al, mic)
        wall = time.perf_counter() - t0
        xs, ys = erle_curve(mic, e)
        ax.plot(xs, ys, label=f"{kind.upper()} μ={mu}", lw=2)
        steady = ys[len(ys) // 2:].mean()
        t15 = next((x for x, y in zip(xs, ys) if y >= 15), float("nan"))
        rtf = wall / (len(mic) / SR)
        w_norm = np.linalg.norm(af.taps if kind == "mdf" else af.w)
        rows.append((kind.upper(), mu, steady, t15, wall, rtf, w_norm))
        print(f"{kind.upper():5s} μ={mu}: steady {steady:5.1f}dB  t(15dB)={t15:4.1f}s  "
              f"wall {wall:5.1f}s  RTF {rtf:.2f}  ||W||={w_norm:.2f}")
    # 真实路径范数（过拟合诊断：MDF 逐 bin 白化在非平稳语音上 ||W|| 会远超真值）
    print("（参考：真实回声路径 ||W_true|| ≈ 0.34）")
    ax.axhline(20, color="gray", ls="--", lw=1, label="DoD 20dB")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("滑窗 ERLE (dB)")
    ax.set_title("NLMS vs MDF 收敛曲线（2048 taps，真实语音 + 100ms 房间 RIR）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_convergence.png", dpi=150)
    print(f"saved -> {OUT/'ablation_convergence.png'}")

    # ---- 2) MDF 步长扫描 ----
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    sweep = []
    for mu in (0.3, 0.5, 0.7, 1.0, 1.3):
        e = run_filter("mdf", far_al, mic, mu)
        xs, ys = erle_curve(mic, e)
        ax2.plot(xs, ys, label=f"μ={mu}", lw=1.5)
        steady = ys[len(ys) // 2:].mean()
        t15 = next((x for x, y in zip(xs, ys) if y >= 15), float("nan"))
        sweep.append((mu, steady, t15))
    ax2.axhline(20, color="gray", ls="--", lw=1)
    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("滑窗 ERLE (dB)")
    ax2.set_title("MDF 步长扫描：收敛速度 vs 稳态误调（信赖域+对角加载保驾，μ=1.3 也不发散）")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / "ablation_stepsize.png", dpi=150)
    print(f"saved -> {OUT/'ablation_stepsize.png'}")

    print("\n| μ | 稳态 ERLE(dB) | t(15dB) |")
    print("|---|---|---|")
    for mu, s, t in sweep:
        print(f"| {mu} | {s:.1f} | {t:.1f}s |")


if __name__ == "__main__":
    main()
