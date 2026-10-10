# -*- coding: utf-8 -*-
"""AEC 数据合成管道（WP-07D / 模块7 前置）：造 (farend, nearend, mic) 三元组 + manifest。

物理链路（每条样本）：
    far  = 远端语音（电信号参考，AEC 的输入之一）
    echo = soft_clip( conv(far, RIR) )[右移 delay 样本] × echo_gain   # 扬声器->麦克风
    mic  = near + echo + noise                                       # 麦克风信号

三场景（配比 --scenarios，默认 FEST:0.5 / NST:0.25 / DT:0.25）：
    FEST 远端单讲: near=0,        mic = echo + noise（ERLE 统计段）
    NST 近端单讲: echo=0,        mic = near + noise（近端保真段）
    DT  双讲:     都有,          mic = near + echo + noise（SER 可配）

RIR 来源（三选一，自动降级）：
    1. --rir-dir  用户自备 RIR wav 目录（官方 AEC-Challenge RIR / 自采）
    2. pyroomacoustics 随机 shoe-box 房间仿真（默认；尺寸/吸声随机）
    3. 指数衰减噪声近似（无依赖兜底，仅冒烟用）

时延扰动：echo 相对 far 参考随机右移 delay∈[--delay-min-ms,--delay-max-ms]，
模拟 loopback 时延不确定性，训练 AEC 的时延鲁棒性（WP-07 用）。

用法：
    python aec_synth.py --num 100 --out-dir D:/data/aec-synth/v1 \
        --near-dir D:/data/voicebank-demand-16k/extracted/train/clean \
        --far-dir  D:/data/voicebank-demand-16k/extracted/train/clean \
        --noise-dir D:/data/dns-challenge-mirror/datasets/noise
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

SR = 16000
SCENARIOS = ("fest", "nst", "dt")


# ---------------------------------------------------------------- RIR 来源
def rir_simulate_pra(rng, sr=SR):
    """pyroomacoustics 随机房间：尺寸 3-8m、吸声 0.10-0.45、源/麦距>=0.5m。"""
    import pyroomacoustics as pra
    dim = rng.uniform(3, 8, size=3)
    mic = dim / 2 + rng.uniform(-0.5, 0.5, size=3)
    while True:
        src = rng.uniform(0.2, 1.0, size=3) * dim
        if np.linalg.norm(src - mic) >= 0.5:
            break
    room = pra.ShoeBox(dim, fs=sr, materials=pra.Material(rng.uniform(0.10, 0.45)),
                       max_order=8)
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    return room.rir[0][0].astype(np.float64)


def rir_exponential(rng, sr=SR):
    """兜底：直接路径脉冲 + 指数衰减尾（RT60 0.2-0.6s），仅冒烟/单测用。"""
    n = int(sr * rng.uniform(0.2, 0.6))
    t = np.arange(n) / sr
    tau = n / sr / 3
    h = rng.standard_normal(n) * np.exp(-t / tau)
    h[0] = 0.0
    d0 = rng.integers(0, int(0.03 * sr))       # 直接路径时延 0-30ms
    h = np.concatenate([np.zeros(d0), h])[: n + int(0.03 * sr)]
    return h / (np.linalg.norm(h) + 1e-12) * 10


def load_rir_dir(rir_dir):
    files = sorted(Path(rir_dir).glob("**/*.wav"))
    if not files:
        raise FileNotFoundError(f"--rir-dir 无 wav: {rir_dir}")
    return files


def get_rir(rir_files, rng, max_ms=None):
    """按可用来源取一条 RIR；max_ms 截断（车厢 RIR 短，RT60>300ms 的部分截掉）。"""
    if rir_files is not None:
        h, _ = sf.read(str(rir_files[rng.integers(len(rir_files))]), dtype="float64",
                       always_2d=True)
        h = h[:, 0] if h.shape[1] == 1 else h.mean(1)
    else:
        try:
            h = rir_simulate_pra(rng)
        except ImportError:
            h = rir_exponential(rng)
    if max_ms:
        h = h[: int(SR * max_ms / 1000)]
    return h


# ---------------------------------------------------------------- 合成基元
def soft_clip(x, ceiling=0.5):
    """功放饱和近似（WP-08 非线性 AEC 的失真源开关）：y = c·tanh(x/c)。

    ceiling 以下信号近似线性，接近/超过 ceiling 处压缩——模拟小扬声器满幅失真。
    """
    return ceiling * np.tanh(x / ceiling)


def fit_len(x, n, rng):
    """裁到 n：长了随机裁剪，短了循环补。"""
    if len(x) >= n:
        off = rng.integers(0, len(x) - n + 1)
        return x[off:off + n].copy()
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def db_scale(x, ref_power, target_db, eps=1e-12):
    """把 x 缩放到 相对 ref_power 的目标 dB（SER/SNR 通用）。

    ref_power≈0（无参考，如 FEST 中噪声为零）时跳过定标、保持原电平。
    """
    if ref_power <= eps:
        return x
    p = np.mean(x ** 2) + eps
    g = 10 ** (target_db / 10) * ref_power / p
    return x * np.sqrt(g)


def generate_one(near, far, noise, rir, scenario, ser_db, snr_db,
                 delay_ms, use_soft_clip):
    """单样本合成 -> (far_ref, near_clean, mic, noise, aux)。

    前四条等长 float32 波形（noise 一并返回便于测试对账/后续 ERLE 抽检）；
    aux 为参数 dict（manifest 用）。

    定标协议（全在函数内完成，输入为原始电平）：
      FEST: echo 相对 noise 定 snr_db；NST: near 相对 noise 定 snr_db；
      DT:   near 相对 noise 定 snr_db，echo 相对 near 定 -ser_db（SER=P_near/P_echo）。
    防削波：mic 峰值超 0.9 时全部成分同缩 k（SER/SNR 口径不变）。
    """
    n = len(near)
    echo = fftconvolve(far, rir)[: n + int(SR * 0.1)]      # RIR 尾部截断缓冲
    if use_soft_clip:
        echo = soft_clip(echo)
    d = int(round(delay_ms * SR / 1000))
    echo = np.concatenate([np.zeros(d), echo])[:n]          # 时延扰动（far 参考不移）
    noise_p = np.mean(noise ** 2)
    if scenario == "fest":
        near = np.zeros(n)
        echo = db_scale(echo, noise_p, snr_db)              # 回声 vs 噪声
    elif scenario == "nst":
        echo = np.zeros(n)
        near = db_scale(near, noise_p, snr_db)              # 近端 vs 噪声
    else:  # dt
        near = db_scale(near, noise_p, snr_db)
        echo = db_scale(echo, np.mean(near ** 2), -ser_db)  # SER=10log(P_near/P_echo)
    mic = near + echo + noise
    peak = np.abs(mic).max()
    if peak > 0.9:                                          # 防削波统一缩放（SER/SNR 不变）
        k = 0.9 / peak
        mic, near, echo, noise, far = mic * k, near * k, echo * k, noise * k, far * k
    return (far.astype(np.float32), near.astype(np.float32), mic.astype(np.float32),
            noise.astype(np.float32),
            dict(delay_ms=float(delay_ms), ser_db=float(ser_db), snr_db=float(snr_db),
                 soft_clip=int(bool(use_soft_clip))))


# ---------------------------------------------------------------- 批量驱动
def pick_scene(rng, scenario_ratio):
    r = rng.random()
    acc = 0.0
    for s, w in zip(SCENARIOS, scenario_ratio):
        acc += w
        if r < acc:
            return s
    return SCENARIOS[-1]


def load_wav_16k(path):
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, f"{path} 采样率 {sr} != 16000（管道统一 16k）"
    return y.mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--near-dir", type=Path, required=True)
    ap.add_argument("--far-dir", type=Path, required=True)
    ap.add_argument("--noise-dir", type=Path, required=True)
    ap.add_argument("--rir-dir", type=Path, default=None,
                    help="自备 RIR wav 目录；缺省用 pyroomacoustics 仿真")
    ap.add_argument("--rir-max-ms", type=float, default=None,
                    help="RIR 截断长度（车厢场景 ~300ms；None 不截断）")
    ap.add_argument("--scenarios", default="0.5,0.25,0.25",
                    help="FEST,NST,DT 配比（逗号分隔）")
    ap.add_argument("--dur-sec", type=float, default=4.0)
    ap.add_argument("--ser-range", default="0,10", help="DT 场景 SER dB 范围")
    ap.add_argument("--snr-range", default="5,20", help="噪声 SNR dB 范围")
    ap.add_argument("--delay-min-ms", type=float, default=10.0)
    ap.add_argument("--delay-max-ms", type=float, default=80.0)
    ap.add_argument("--soft-clip-prob", type=float, default=0.3,
                    help="启用功放软剪切的样本比例（WP-08 开关）")
    ap.add_argument("--dt-leadin-sec", type=float, default=2.0,
                    help="DT 样本近端加入前的纯远端引导秒数（真实通话时序：先 FEST 收敛再双讲）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ratio = [float(x) for x in args.scenarios.split(",")]
    assert len(ratio) == 3 and abs(sum(ratio) - 1) < 1e-6, "配比需 3 个数且和为 1"
    ser_lo, ser_hi = map(float, args.ser_range.split(","))
    snr_lo, snr_hi = map(float, args.snr_range.split(","))
    rng = np.random.default_rng(args.seed)

    near_files = sorted(Path(args.near_dir).glob("**/*.wav"))
    far_files = sorted(Path(args.far_dir).glob("**/*.wav"))
    noise_files = sorted(Path(args.noise_dir).glob("**/*.wav"))
    assert near_files and far_files and noise_files, "输入目录缺 wav"
    rir_files = load_rir_dir(args.rir_dir) if args.rir_dir else None
    if rir_files:
        rir_src = "dir"
    else:
        try:
            import pyroomacoustics  # noqa: F401
            rir_src = "sim"
        except ImportError:
            rir_src = "exp"      # 指数衰减近似（仅冒烟）

    out = args.out_dir
    for sub in ("farend", "nearend", "mic"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n = int(args.dur_sec * SR)
    rows = []
    for i in range(args.num):
        scene = pick_scene(rng, ratio)
        near_f = near_files[rng.integers(len(near_files))]
        far_f = far_files[rng.integers(len(far_files))]
        noise_f = noise_files[rng.integers(len(noise_files))]
        near = fit_len(load_wav_16k(near_f), n, rng)
        far = fit_len(load_wav_16k(far_f), n, rng)
        noise = fit_len(load_wav_16k(noise_f), n, rng)
        rir = get_rir(rir_files, rng, max_ms=args.rir_max_ms)
        if scene == "dt" and args.dt_leadin_sec > 0:
            lead = int(args.dt_leadin_sec * SR)
            near[:lead] = 0.0            # 引导段纯远端：给自适应滤波收敛窗口
        far_r, near_r, mic_r, _, par = generate_one(
            near, far, noise, rir, scene,
            ser_db=rng.uniform(ser_lo, ser_hi),
            snr_db=rng.uniform(snr_lo, snr_hi),
            delay_ms=rng.uniform(args.delay_min_ms, args.delay_max_ms),
            use_soft_clip=bool(rng.random() < args.soft_clip_prob))
        uid = f"fileid_{i}_{scene}"
        sf.write(out / "farend" / f"{uid}.wav", far_r, SR)
        sf.write(out / "nearend" / f"{uid}.wav", near_r, SR)
        sf.write(out / "mic" / f"{uid}.wav", mic_r, SR)
        rows.append(dict(name=uid, scenario=scene, rir_src=rir_src,
                         near_file=near_f.name, far_file=far_f.name,
                         noise_file=noise_f.name,
                         leadin_sec=args.dt_leadin_sec if scene == "dt" else 0.0,
                         **par))
    fields = ["name", "scenario", "rir_src", "near_file", "far_file", "noise_file",
              "delay_ms", "ser_db", "snr_db", "soft_clip", "leadin_sec"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"[aec_synth] {args.num} 条 -> {out}")
    print(f"  场景分布 {dict(Counter(r['scenario'] for r in rows))}（目标 {ratio}）")
    print(f"  manifest: {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
