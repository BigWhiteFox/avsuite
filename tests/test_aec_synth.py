"""WP-07D 单测：时延正确性 / 三场景成分 / SER-SNR 定标 / soft-clip / 配比 / manifest 可复现。"""
import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aec_synth as S  # noqa: E402

SR = 16000
DIRAC = np.array([1.0])   # 纯延迟"房间"：echo = far 平移


def _sig(n=16000, seed=0, f0=200.0):
    t = np.arange(n) / SR
    return (0.3 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 3 * f0 * t))


def test_delay_correctness():
    """dirac RIR + 已知 delay：echo 严格等于 far 右移 d 样本（far 参考不移）。"""
    far = _sig(16000)
    delay_ms = 37.5                                   # -> d = 600 样本
    _, _, mic, _, _ = S.generate_one(np.zeros(16000), far, np.zeros(16000), DIRAC,
                                     "fest", ser_db=0, snr_db=20, delay_ms=delay_ms,
                                     use_soft_clip=False)
    d = int(round(delay_ms * SR / 1000))
    assert d == 600
    np.testing.assert_allclose(mic[d:], far[:len(far) - d], atol=1e-6)
    assert np.abs(mic[:d]).max() == 0


def test_scenario_composition():
    """零回声/零近端的成分守恒：FEST 近端全零；NST mic = near+noise；DT 两者都有。"""
    far = _sig(16000, f0=180)
    near = _sig(16000, seed=1, f0=320)
    noise = _sig(16000, seed=3, f0=97)
    noise = (noise / np.sqrt(np.mean(noise ** 2)) * 0.01).astype(np.float64)  # 低电平静音无关
    _, n_f, mic_f, _, _ = S.generate_one(near.copy(), far, noise, DIRAC, "fest",
                                         0, 20, 10, False)
    assert np.abs(n_f).max() == 0 and np.abs(mic_f).max() > 0        # 纯回声+噪声
    _, n_n, mic_n, noise_n, _ = S.generate_one(near.copy(), far, noise, DIRAC, "nst",
                                               0, 20, 10, False)
    np.testing.assert_allclose(mic_n, n_n + noise_n, atol=1e-6)      # 无回声成分
    assert np.abs(n_n).max() > 0
    _, n_d, mic_d, _, _ = S.generate_one(near.copy(), far, noise, DIRAC, "dt",
                                         6, 20, 10, False)
    echo_d = mic_d - n_d
    assert np.abs(n_d).max() > 0 and np.abs(echo_d).max() > 0        # 双讲


def test_ser_and_snr_calibration():
    """SER/SNR 定标。注意 mic 里无法分离 echo 与 noise（测试不引入测量误差的方法）：
    SER 用零噪声样本测（mic-near=echo 精确）；SNR 用成分对成分测。"""
    far = _sig(16000, f0=170)
    # SER：零噪声 -> echo = mic - near 精确可测
    _, near_d, mic_d, _, _ = S.generate_one(_sig(16000, seed=1, f0=310), far,
                                            np.zeros(16000), DIRAC, "dt", ser_db=6.0,
                                            snr_db=12, delay_ms=10, use_soft_clip=False)
    echo_d = mic_d - near_d
    ser = 10 * math.log10(np.mean(near_d ** 2) / np.mean(echo_d ** 2))
    assert abs(ser - 6.0) < 0.1
    # SNR：单位功率噪声 -> near/noise 与配置一致
    noise = _sig(16000, seed=3, f0=97)
    noise = noise / np.sqrt(np.mean(noise ** 2))
    _, near_s, _, noise_s, _ = S.generate_one(_sig(16000, seed=1, f0=310), far, noise,
                                              DIRAC, "dt", ser_db=6.0, snr_db=12,
                                              delay_ms=10, use_soft_clip=False)
    snr = 10 * math.log10(np.mean(near_s ** 2) / np.mean(noise_s ** 2))
    assert abs(snr - 12.0) < 0.5
    # FEST：回声/噪声比 = snr_db
    _, _, mic_f, noise_f, _ = S.generate_one(np.zeros(16000), far, noise, DIRAC,
                                             "fest", 0, 15, 10, False)
    fest_snr = 10 * math.log10(np.mean(mic_f ** 2) / np.mean(noise_f ** 2))
    assert abs(fest_snr - 15.0) < 0.5


def test_soft_clip_switch():
    """开关生效：大信号时剪切改变波形且不超 ceiling；小信号在线性区不变。"""
    x_big = np.linspace(-3, 3, 16000)
    assert not np.allclose(S.soft_clip(x_big), x_big, atol=1e-4)
    assert np.abs(S.soft_clip(x_big)).max() <= 0.5 + 1e-6
    x_small = x_big * 0.001      # 线性区：|c·tanh(x/c)-x| ≈ x³/(3c²)，x=0.003 时 ~2e-8
    assert np.allclose(S.soft_clip(x_small), x_small, atol=1e-6)
    far = _sig(16000, f0=150)
    _, _, m1, _, _ = S.generate_one(np.zeros(16000), far * 4, np.zeros(16000), DIRAC,
                                    "fest", 0, 10, 10, True)
    _, _, m2, _, _ = S.generate_one(np.zeros(16000), far * 4, np.zeros(16000), DIRAC,
                                    "fest", 0, 10, 10, False)
    assert not np.allclose(m1, m2)


def test_scenario_ratio():
    """pick_scene 统计：0.5/0.25/0.25 下 3000 次抽样各场景 ±4%。"""
    rng = np.random.default_rng(7)
    from collections import Counter
    c = Counter(S.pick_scene(rng, (0.5, 0.25, 0.25)) for _ in range(3000))
    for s, target in zip(S.SCENARIOS, (0.5, 0.25, 0.25)):
        assert abs(c[s] / 3000 - target) < 0.04


def _make_fixture(tmp_path):
    rng = np.random.default_rng(0)
    for sub, n_files in (("near", 3), ("far", 3), ("noise", 2)):
        d = tmp_path / sub
        d.mkdir(parents=True)
        for i in range(n_files):
            y = (0.2 * rng.standard_normal(24000)).astype(np.float32)
            sf.write(d / f"{sub}{i}.wav", y, SR)


def test_main_manifest_and_repro(tmp_path):
    """main() 全链路：目录结构 + manifest 字段完整 + 同 seed 输出逐字节可复现。

    --dur-sec 必须 > --dt-leadin-sec（默认 2.0s）：脚本用 `near[:lead] = 0.0` 实现
    引导段，lead 超过片段长度时 numpy 切片会把**整段** near 清零，DT 样本静默退化
    为 FEST（manifest 仍写 ser_db/snr_db，但近端恒 0、SER 口径失去意义）。
    这里用 3.0s 保证引导段之外仍有近端，并显式断言 DT 近端非零作为回归守卫。
    """
    fx = tmp_path / "fx"
    _make_fixture(fx)
    base = [sys.executable, "-s", str(Path(S.__file__)),
            "--num", "6", "--dur-sec", "3.0",
            "--near-dir", str(fx / "near"), "--far-dir", str(fx / "far"),
            "--noise-dir", str(fx / "noise"), "--seed", "0"]
    r = subprocess.run(base + ["--out-dir", str(tmp_path / "out1")],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    out1 = tmp_path / "out1"
    for sub in ("farend", "nearend", "mic"):
        assert len(list((out1 / sub).glob("*.wav"))) == 6
    with open(out1 / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    for row in rows:
        assert row["scenario"] in S.SCENARIOS and row["near_file"] and row["far_file"]
        assert float(row["delay_ms"]) >= 0 and row["rir_src"] in ("dir", "sim", "exp")
    # DT 近端不得被引导段清零：引导段内恒 0、引导段外必须非零
    dt_rows = [r_ for r_ in rows if r_["scenario"] == "dt"]
    assert dt_rows, "seed=0 的 6 个样本应至少含 1 条 dt，否则本守卫形同虚设"
    for row in dt_rows:
        near = sf.read(out1 / "nearend" / f"{row['name']}.wav", dtype="float32")[0]
        lead = int(float(row["leadin_sec"]) * SR)
        assert lead == 32000 and len(near) == 48000, (lead, len(near))
        assert np.abs(near[:lead]).max() == 0, f"{row['name']}: 引导段应纯远端"
        assert np.abs(near[lead:]).max() > 1e-4, \
            f"{row['name']}: dt 近端被整体清零（--dur-sec 小于 dt-leadin 时会退化 FEST）"
    # 可复现：同 seed 重跑，所有 wav 一致
    subprocess.run(base + ["--out-dir", str(tmp_path / "out2")],
                   capture_output=True, text=True, timeout=300)
    for row in rows:
        a = (out1 / "mic" / f"{row['name']}.wav").read_bytes()
        b = (tmp_path / "out2" / "mic" / f"{row['name']}.wav").read_bytes()
        assert a == b


@pytest.mark.xfail(strict=True, reason=(
    "aec_synth.main 未对引导段做 lead=min(lead,n) 夹取：--dur-sec 小于 "
    "--dt-leadin-sec(2.0) 时 near[:lead]=0 会清零整段近端，dt 静默退化为 fest。"
    "脚本本体属 R16 批次；修好后本用例转 xpass，届时删除本标记。"))
def test_dt_leadin_longer_than_clip_keeps_nearend(tmp_path):
    """--dur-sec 1.0 < dt-leadin 2.0：DT 近端仍应保留非零尾部（当前失败，见 xfail）。"""
    fx = tmp_path / "fx"
    _make_fixture(fx)
    out = tmp_path / "out_short"
    r = subprocess.run([sys.executable, "-s", str(Path(S.__file__)),
                        "--num", "6", "--dur-sec", "1.0",
                        "--near-dir", str(fx / "near"), "--far-dir", str(fx / "far"),
                        "--noise-dir", str(fx / "noise"), "--seed", "0",
                        "--out-dir", str(out)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    with open(out / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dt_rows = [r_ for r_ in rows if r_["scenario"] == "dt"]
    assert dt_rows, "应含 dt 样本"
    for row in dt_rows:
        near = sf.read(out / "nearend" / f"{row['name']}.wav", dtype="float32")[0]
        assert np.abs(near).max() > 1e-4, f"{row['name']}: dt 近端被整体清零（退化 FEST）"
