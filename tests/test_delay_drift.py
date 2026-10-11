"""P2-1 单测：--delay-drift 时延漂移（回归 / 特性 / 开关语义 / CLI 接线）。

口径说明：
(a) 回归——drift=0（默认）时同 seed 输出与 **tests/golden/ 固化黄金基准** 逐字节
    一致（t14 改造：原口径是 `git show HEAD:scripts/aec_synth.py` 现场取旧版对比）。
    改造理由：那个口径只在**未提交**改动时有效——一旦把 aec_synth.py 的改动提交，
    HEAD 与工作树相同，比较退化为"自己 vs 自己"，保护归零；且它把 HEAD 当正确
    基准、没有正规的"更新基准"入口，还依赖 git 可执行文件。固化基准与 HEAD 解耦，
    详见 test_drift0_matches_golden_baseline 的 docstring。
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
import os
import shutil
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


# ---------------------------------------------------------------- 黄金基准（t14）
_GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "aec_synth_drift0"
_GOLDEN_RIR_DIR = _GOLDEN_DIR / "rir"
_GOLDEN_EXPECT = _GOLDEN_DIR / "expected"
_GOLDEN_SUBS = ("farend", "nearend", "mic")
_GOLDEN_NUM, _GOLDEN_DUR, _GOLDEN_SEED = 2, 0.5, 7


def _golden_inputs(root):
    """确定性输入 fixture：near/far/noise 各 2 条 0.75s 噪声 wav（seed 0）。

    长度 12000 > dur_sec*SR=8000，使 fit_len 走"随机裁剪"分支（覆盖 rng 消耗）。
    """
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for sub in ("near", "far", "noise"):
        d = root / sub
        d.mkdir()
        for i in range(2):
            y = (0.2 * rng.standard_normal(12000)).astype(np.float32)
            sf.write(d / f"{sub}{i}.wav", y, SR)
    return root


def _make_golden_rir(path):
    """确定性合成 RIR：直达径 + 指数衰减尾（固定 seed）。

    用自备 RIR 而非默认的 pyroomacoustics 仿真，是为了让基准**与环境无关**：
    默认路径在 pyroomacoustics 缺失时退化为 rir_exponential，装了它就换成房间
    仿真——那样基准会随"是否装了 pra / pra 版本"漂移。
    """
    rng = np.random.default_rng(12345)
    n = 512
    t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-t / 0.02)
    h[0] = 1.0                                   # 直达径
    h = (h / np.linalg.norm(h) * 0.7).astype(np.float32)
    sf.write(path, h, SR)


def _run_golden_synth(out_dir, fx, rir_dir, extra=()):
    """按固定参数跑一次 aec_synth.py（子进程，与基准生成时完全同参）。"""
    argv = [sys.executable, "-s", str(Path(S.__file__)),
            "--num", str(_GOLDEN_NUM), "--dur-sec", str(_GOLDEN_DUR),
            "--seed", str(_GOLDEN_SEED),
            "--near-dir", str(fx / "near"), "--far-dir", str(fx / "far"),
            "--noise-dir", str(fx / "noise"),
            "--rir-dir", str(rir_dir), "--out-dir", str(out_dir), *extra]
    rr = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    assert rr.returncode == 0, rr.stderr[-2000:]
    return out_dir


def _regen_golden(fx):
    """重写 tests/golden/aec_synth_drift0/（RIR + expected/）。仅 REGEN_GOLDEN 路径调用。"""
    if _GOLDEN_DIR.exists():
        shutil.rmtree(_GOLDEN_DIR)
    _GOLDEN_RIR_DIR.mkdir(parents=True)
    _make_golden_rir(_GOLDEN_RIR_DIR / "rir_seed12345.wav")
    _run_golden_synth(_GOLDEN_EXPECT, fx, _GOLDEN_RIR_DIR)


def test_drift0_matches_golden_baseline(tmp_path):
    """(a) 回归：--delay-drift 缺省(0) 输出与 tests/golden/ 固化基准逐字节一致。

    与旧口径（`git show HEAD:scripts/aec_synth.py` 现场对比）的差别：
    1) 旧口径只在**未提交**改动时有效——改动一旦提交，HEAD == 工作树，比较退化成
       "自己 vs 自己"，回归保护归零；固化基准入库后与 HEAD 无关，跨提交持续有效；
    2) 旧口径把"HEAD 版"当正确基准，**有意的** drift=0 输出变化会误报，且没有正规
       的更新入口；固化基准的更新是一次显式、可 review 的提交（见下方命令）；
    3) 旧口径依赖 git 可执行文件与仓库历史（浅克隆/无 git 时只能 skip）；本用例
       不再需要 git。
    因此改 aec_synth.py 的**非默认路径**（例如 drift>0 的游走分支）不会误报；
    只有 drift=0 的实际输出字节变了才红。

    生成/更新基准（需要时手动跑一次，review 后提交 tests/golden/）：
        $env:PYTHONNOUSERSITE='1'
        $env:REGEN_GOLDEN='1'; & 'C:\\Users\\14805\\.conda\\envs\\avsuite\\python.exe' `
            -m pytest tests/test_delay_drift.py -q -k golden
    基准构成：自备合成 RIR（固定 seed，不依赖 pyroomacoustics）+ num=2 / 0.5s /
    seed=7 的 farend/nearend/mic 全部 wav 与 manifest.csv。
    """
    fx = _golden_inputs(tmp_path / "fx")
    if os.environ.get("REGEN_GOLDEN"):
        _regen_golden(fx)
        pytest.skip("REGEN_GOLDEN 已置位：黄金基准已重写，请 review 后提交 tests/golden/")

    assert _GOLDEN_RIR_DIR.is_dir() and _GOLDEN_EXPECT.is_dir(), (
        f"缺少黄金基准目录 {_GOLDEN_DIR}（生成命令见本用例 docstring）")
    out = _run_golden_synth(tmp_path / "out", fx, _GOLDEN_RIR_DIR)

    assert (out / "manifest.csv").read_bytes() == \
        (_GOLDEN_EXPECT / "manifest.csv").read_bytes(), "manifest.csv 与黄金基准不一致"
    names = sorted(p.name for p in (_GOLDEN_EXPECT / "mic").glob("*.wav"))
    assert len(names) == _GOLDEN_NUM, f"基准内 mic wav 数 {len(names)} != {_GOLDEN_NUM}"
    for sub in _GOLDEN_SUBS:
        for name in names:
            assert (out / sub / name).read_bytes() == \
                (_GOLDEN_EXPECT / sub / name).read_bytes(), \
                f"drift=0 输出与黄金基准不一致: {sub}/{name}"


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
