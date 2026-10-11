"""P2-1 单测：--delay-drift 时延漂移（回归 / 特性 / 开关语义 / CLI 接线）。

口径说明：
(a) 回归——drift=0（默认）时同 seed 输出与「旧版」逐字节一致。旧版 =
    `git show HEAD:scripts/aec_synth.py`（P2-1 编辑前已核对工作树与 HEAD 无差异），
    取到临时目录、同参数各跑一次小规模合成（num=2、1.0s、同 fixture），对
    farend/nearend/mic 全部 wav 与 manifest.csv **逐字节比较**——比"关键统计量"
    更强的口径。
(b) 特性——drift=5ms、基础时延 40ms、8s 单条 FEST（DIRAC RIR + 零噪声，
    mic = 纯时变时延回声）：0.25s 窗 / 0.5s 步互相关估计时延轨迹，极差 > 0
    （本机实测 ~10-14ms）、落在 [40-2σ-2, 40+2σ+2] ms 限幅带内、与声明游走
    轨迹偏差 ≤2ms；对照 drift=0 的同信号轨迹极差恒 0。
(c) 开关语义——drift=0 不消耗 rng 抽样（默认路径逐比特兼容的前提）；
    drift>0 输出与文档口径（限幅随机游走 + 线性插值分数时延）逐位一致。
(d) 配对保证——游走走独立 Generator：同 seed 开/关漂移的 manifest 底座
    （场景/文件/时延/SER/SNR 序列）逐字段一致，漂移对照可做同内容配对。
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aec_synth as S  # noqa: E402

SR = 16000
DIRAC = np.array([1.0])            # 纯延迟"房间"：echo = far 平移
REPO = Path(__file__).resolve().parents[1]


def _sig(n=16000, seed=0, f0=200.0):
    t = np.arange(n) / SR
    return 0.3 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 3 * f0 * t)


def _make_fixture(root):
    """小语音/噪声 fixture：near/far/noise 各 2 条 1.5s 噪声 wav。"""
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for sub in ("near", "far", "noise"):
        d = root / sub
        d.mkdir()
        for i in range(2):
            y = (0.2 * rng.standard_normal(24000)).astype(np.float32)
            sf.write(d / f"{sub}{i}.wav", y, SR)
    return root


def _traj(mic, far, win_s=0.25, hop_s=0.5, lag_lo=400, lag_hi=880):
    """滑窗互相关时延估计（样本）。mic[m] ≈ far[m-d]：参考窗向左对齐，
    far 前补 lag_hi 个零处理段首；慢漂移下窗内时延近似恒定。
    估计范围 [lag_lo, lag_hi] = [25, 55] ms（基础 40ms ± 2σ=10ms + 裕量）。"""
    win, hop = int(win_s * SR), int(hop_s * SR)
    pad = np.concatenate([np.zeros(lag_hi), far])
    out = []
    for start in range(0, len(mic) - win + 1, hop):
        ref = pad[start: start + win + (lag_hi - lag_lo)]
        seg = mic[start: start + win].astype(np.float64)
        c = correlate(ref, seg, mode="valid", method="fft")
        out.append(lag_hi - int(np.argmax(np.abs(c))))
    return np.array(out, dtype=np.float64)


def test_drift0_matches_head_bitwise(tmp_path):
    """(a) 回归：--delay-drift 缺省(0) 时同 seed 输出与旧版逐字节一致。"""
    r = subprocess.run(["git", "show", "HEAD:scripts/aec_synth.py"],
                       capture_output=True, cwd=str(REPO), timeout=60)
    if r.returncode != 0:
        pytest.skip(f"git show 不可用，无法取旧版基准: {r.stderr[-200:]!r}")
    old = tmp_path / "aec_synth_head.py"
    old.write_bytes(r.stdout)
    fx = _make_fixture(tmp_path / "fx")
    common = ["--num", "2", "--dur-sec", "1.0",
              "--near-dir", str(fx / "near"), "--far-dir", str(fx / "far"),
              "--noise-dir", str(fx / "noise"), "--seed", "7"]
    outs = {}
    for tag, script in (("old", old), ("new", Path(S.__file__))):
        o = tmp_path / f"out_{tag}"
        rr = subprocess.run([sys.executable, "-s", str(script), *common,
                             "--out-dir", str(o)],
                            capture_output=True, text=True, timeout=600)
        assert rr.returncode == 0, f"[{tag}] {rr.stderr[-2000:]}"
        outs[tag] = o
    assert (outs["old"] / "manifest.csv").read_bytes() == \
           (outs["new"] / "manifest.csv").read_bytes(), "manifest 与旧版不一致"
    for sub in ("farend", "nearend", "mic"):
        wavs = sorted(p.name for p in (outs["old"] / sub).glob("*.wav"))
        assert len(wavs) == 2
        for name in wavs:
            assert (outs["old"] / sub / name).read_bytes() == \
                   (outs["new"] / sub / name).read_bytes(), \
                   f"drift=0 输出与旧版不一致: {sub}/{name}"


def test_drift0_leaves_rng_untouched():
    """(c) drift=0 时 generate_one 不消耗 drift_rng 抽样，且与默认路径逐位一致。"""
    far = _sig(16000)
    r1 = np.random.default_rng(7)
    a = S.generate_one(np.zeros(16000), far, np.zeros(16000), DIRAC, "fest",
                       0, 20, 10.0, False, drift_ms=0.0, drift_rng=r1)[2]
    assert r1.bit_generator.state == np.random.default_rng(7).bit_generator.state
    b = S.generate_one(np.zeros(16000), far, np.zeros(16000), DIRAC, "fest",
                       0, 20, 10.0, False)[2]
    assert np.array_equal(a, b)


def test_drift_walk_bounded_two_sided():
    """(c) 游走单元：w0=0、全程 |w|<=2σ 限幅、固定 seed 下非单向（方向随 seed 变）。"""
    walks = {}
    for seed in (123, 124):
        nt, nw = S.delay_drift_walk(8 * SR, 40.0, 5.0, np.random.default_rng(seed))
        walks[seed] = nw
        assert nt[0] == 0.0 and nw[0] == 0.0          # 从基础时延出发
        assert nt[-1] == 8.0                          # 节点覆盖整段
        assert np.all(np.abs(nw) <= 2 * 5.0 + 1e-9), f"seed {seed} 游走越出 ±2σ 限幅"
        d = np.diff(nw)
        assert (d > 0).any() and (d < 0).any(), f"seed {seed} 游走单向（方向不随机?）"
    assert not np.allclose(walks[123], walks[124])    # 轨迹随 seed 变化


def test_drift_matches_documented_formula():
    """(c) drift>0 输出与文档口径（限幅随机游走+线性插值分数时延）逐位一致
    （0.05 电平小信号避免防削波缩放，isolates 漂移公式本身）。"""
    n = 2 * SR
    far = 0.05 * np.random.default_rng(11).standard_normal(n)
    nt, nw = S.delay_drift_walk(n, 40.0, 5.0, np.random.default_rng(123))
    mic = S.generate_one(np.zeros(n), far, np.zeros(n), DIRAC, "fest", 0, 20,
                         40.0, False, drift_ms=5.0,
                         drift_rng=np.random.default_rng(123))[2]
    tau = np.interp(np.arange(n) / SR, nt, nw) + 40.0  # 文档口径复现
    pos = np.arange(n) - tau * SR / 1000.0
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    i0c = np.clip(i0, 0, len(far) - 2)
    y = (1.0 - frac) * far[i0c] + frac * far[i0c + 1]
    y[pos < 0] = 0.0
    np.testing.assert_allclose(mic, y.astype(np.float32), atol=1e-7)


def test_drift_trajectory_varies_bounded_tracks():
    """(b) 特性：drift>0 时窗互相关时延轨迹随时间变化（极差>0）、限幅带内、
    跟踪声明游走 <=2ms；drift=0 对照轨迹极差恒 0。"""
    n = 8 * SR
    far = 0.2 * np.random.default_rng(11).standard_normal(n)
    # 对照：drift=0（固定 40ms）轨迹应恒定
    mic0 = S.generate_one(np.zeros(n), far, np.zeros(n), DIRAC, "fest", 0, 20,
                          40.0, False)[2]
    est0 = _traj(mic0, far) / SR * 1000
    assert np.ptp(est0) == 0.0 and abs(est0[0] - 40.0) <= 2.0
    for seed in (123, 124):
        nt, nw = S.delay_drift_walk(n, 40.0, 5.0, np.random.default_rng(seed))
        out = S.generate_one(np.zeros(n), far, np.zeros(n), DIRAC, "fest", 0, 20,
                             40.0, False, drift_ms=5.0,
                             drift_rng=np.random.default_rng(seed))
        mic, aux = out[2], out[4]
        assert aux["delay_drift_ms"] == 5.0
        assert not np.array_equal(mic, mic0)          # 漂移确实施加
        est_ms = _traj(mic, far) / SR * 1000
        assert np.ptp(est_ms) > 2.0, \
            f"seed {seed} 轨迹极差 {np.ptp(est_ms):.2f}ms，时延未随时间变化?"
        assert est_ms.min() >= 40.0 - 10.0 - 2.0 and est_ms.max() <= 40.0 + 10.0 + 2.0
        centers = np.arange(len(est_ms)) * 0.5 + 0.125
        true_ms = np.interp(centers, nt, nw) + 40.0
        assert np.abs(est_ms - true_ms).max() <= 2.0, \
            f"seed {seed} 轨迹偏离声明游走 {np.abs(est_ms - true_ms).max():.2f}ms"


def test_main_drift_pairs_with_base(tmp_path):
    """配对保证：同 seed 下 drift=5.0 与 drift=0 的 manifest 底座逐字段一致
    （场景/文件/时延/SER/SNR 序列不动，仅多 delay_drift_ms 列）——
    漂移对照实验可做同内容配对的前提。"""
    fx = _make_fixture(tmp_path / "fx")
    common = ["--num", "3", "--dur-sec", "1.0", "--seed", "5",
              "--near-dir", str(fx / "near"), "--far-dir", str(fx / "far"),
              "--noise-dir", str(fx / "noise")]
    mans = {}
    for tag, extra in (("base", []), ("drift", ["--delay-drift", "5.0"])):
        o = tmp_path / f"out_{tag}"
        rr = subprocess.run([sys.executable, "-s", str(Path(S.__file__)), *common,
                             *extra, "--out-dir", str(o)],
                            capture_output=True, text=True, timeout=600)
        assert rr.returncode == 0, f"[{tag}] {rr.stderr[-2000:]}"
        with open(o / "manifest.csv", encoding="utf-8") as f:
            mans[tag] = list(csv.DictReader(f))
    cols = ["name", "scenario", "near_file", "far_file", "noise_file",
            "delay_ms", "ser_db", "snr_db", "soft_clip", "leadin_sec"]
    for rb, rd in zip(mans["base"], mans["drift"]):
        for c in cols:
            assert rb[c] == rd[c], f"配对破裂: {c}: {rb[c]} != {rd[c]}"
    assert all(r["delay_drift_ms"] == "5.0" for r in mans["drift"])
    assert "delay_drift_ms" not in mans["base"][0]


def test_main_drift_cli_and_manifest(tmp_path):
    """CLI 接线：--delay-drift 5.0 全链路可跑，manifest 增加 delay_drift_ms 列。"""
    fx = _make_fixture(tmp_path / "fx")
    out = tmp_path / "out_drift"
    rr = subprocess.run(
        [sys.executable, "-s", str(Path(S.__file__)),
         "--num", "2", "--dur-sec", "1.0", "--delay-drift", "5.0", "--seed", "3",
         "--near-dir", str(fx / "near"), "--far-dir", str(fx / "far"),
         "--noise-dir", str(fx / "noise"), "--out-dir", str(out)],
        capture_output=True, text=True, timeout=600)
    assert rr.returncode == 0, rr.stderr[-2000:]
    with open(out / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert all(r["delay_drift_ms"] == "5.0" for r in rows)
    for sub in ("farend", "nearend", "mic"):
        assert len(list((out / sub).glob("*.wav"))) == 2
