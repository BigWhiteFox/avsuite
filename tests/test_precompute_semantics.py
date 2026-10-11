"""S2 单测：exp/09_cascade_aec/precompute_cascade.py 的三槽语义（合成 1s 案例）。

合成案例（无 D:/data / 网络依赖）：1s@16k——far=多正弦（300/700/1300/1900Hz），
echo = far 右移 20ms × 0.5，near = 500/900Hz 小幅正弦，mic = echo + near。
脚本本体按 CLI 驱动：保存/还原 sys.argv 调 main()，在临时 synth/out 目录上跑，
不修改被测脚本。断言：
- 槽位语义：out mic = e_lin = mic − y_echo（float32）、out farend = y_echo——均与测试内
  同配置 MDF 参考（TradAEC mdf/8/0.7，与脚本完全同款）逐位一致；out nearend 与
  manifest.csv 为源文件逐字节复制；
- 线性级效果（稳态段=跳过前 25% 收敛瞬态，仓内 test_trad_aec._erle 同款口径）：
  e_lin 对 far 的 ERLE > 10dB；e_lin 对 far 的相干在 far 各音频率上显著低于 mic 对 far
  （mic≈1 → e_lin<0.5）；y_echo 与 echo 相关 > 0.9；
- min_len 截断（mic/far 不等长取 min）与断点续跑（三槽齐即跳过；--force 才覆盖）。

注：脚本本身就是"直接驱动 exp/06_trad_aec 的 MDF"（TradAEC method=mdf, partitions=8,
mu=0.7），故无需另写等价语义测试——参考复算用同一配置保证逐位可比。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import coherence

_REPO = Path(__file__).resolve().parents[1]
_DIR09 = _REPO / "exp" / "09_cascade_aec"
_DIR06 = _REPO / "exp" / "06_trad_aec"

_spec = importlib.util.spec_from_file_location("precompute_cascade", _DIR09 / "precompute_cascade.py")
pc = importlib.util.module_from_spec(_spec)
sys.modules["precompute_cascade"] = pc
_spec.loader.exec_module(pc)               # 副作用：sys.path 注入 + import pipeline（纯 numpy/scipy）

sys.path.insert(0, str(_DIR06))            # 参考实现与脚本同源
from pipeline import TradAEC  # noqa: E402

SR = 16000
STEADY = 0.25                              # 跳过前 25% 收敛瞬态（仓内既有口径）


def _synth_case():
    """(far, echo, near, mic) 1s 合成：echo=far 右移 20ms×0.5，mic=echo+near。"""
    t = np.arange(SR) / SR
    far = (0.6 * np.sin(2 * np.pi * 300 * t) + 0.4 * np.sin(2 * np.pi * 700 * t)
           + 0.3 * np.sin(2 * np.pi * 1300 * t) + 0.2 * np.sin(2 * np.pi * 1900 * t))
    d = int(0.020 * SR)                    # 20ms = 320 样本
    echo = np.concatenate([np.zeros(d), 0.5 * far])[:SR]
    near = 0.03 * np.sin(2 * np.pi * 500 * t) + 0.03 * np.sin(2 * np.pi * 900 * t)
    return far.astype(np.float32), echo, near, (echo + near).astype(np.float32)


def _make_synth_dir(root, clips):
    """clips: [(name, mic, far, near)] → manifest.csv + mic/farend/nearend 三槽。"""
    for sub in ("mic", "farend", "nearend"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    lines = ["name,scenario"]
    for name, mic, far, near in clips:
        sf.write(root / "mic" / f"{name}.wav", mic, SR, subtype="FLOAT")
        sf.write(root / "farend" / f"{name}.wav", far, SR, subtype="FLOAT")
        sf.write(root / "nearend" / f"{name}.wav", near, SR, subtype="FLOAT")
        lines.append(f"{name},dt")
    (root / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root / "manifest.csv"


def _run_main(synth, out, force=False):
    """按 CLI 驱动脚本本体（main 走 argparse，需临时替换 sys.argv）。"""
    argv = ["precompute_cascade.py", "--synth-dir", str(synth), "--out-dir", str(out)]
    if force:
        argv.append("--force")
    old = sys.argv
    sys.argv = argv
    try:
        pc.main()
    finally:
        sys.argv = old


def _reference(mic32, far32):
    """脚本语义的参考复算（同一 MDF 配置、同一 float32 截断/相减次序）。"""
    n = min(len(mic32), len(far32))
    res = TradAEC(method="mdf", partitions=8, mu=0.7).process(
        mic32[:n].astype(np.float64), far32[:n].astype(np.float64))
    y = res.y_echo[:n].astype(np.float32)
    e = (mic32[:n] - y).astype(np.float32)             # = 脚本 e_lin（RES 前误差）
    return e, y


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-30))


def _erle_db(mic, e, k=0):
    return 10 * np.log10(np.sum(mic[k:] ** 2) / (np.sum(e[k:] ** 2) + 1e-20))


@pytest.fixture(scope="module")
def cascade(tmp_path_factory):
    """合成 synth-dir + 跑一次 main() + 参考复算。"""
    synth = tmp_path_factory.mktemp("synth")
    out = tmp_path_factory.mktemp("out")
    far, echo, near, mic = _synth_case()
    _make_synth_dir(synth, [("c01", mic, far, near.astype(np.float32))])
    _run_main(synth, out)
    e_ref, y_ref = _reference(mic, far)
    return dict(synth=synth, out=out, far=far, echo=echo, near=near, mic=mic,
                e_ref=e_ref, y_ref=y_ref)


# ---------------------------------------------------------------- 槽位语义
def test_slots_bitwise_match_reference_mdf(cascade):
    out = cascade["out"]
    e_out, sr = sf.read(out / "mic" / "c01.wav", dtype="float32")
    y_out, sr2 = sf.read(out / "farend" / "c01.wav", dtype="float32")
    assert sr == sr2 == SR
    np.testing.assert_array_equal(e_out, cascade["e_ref"])   # mic 槽 = e_lin，逐位
    np.testing.assert_array_equal(y_out, cascade["y_ref"])   # farend 槽 = y_echo，逐位
    assert (out / "nearend" / "c01.wav").read_bytes() == \
        (cascade["synth"] / "nearend" / "c01.wav").read_bytes()      # 目标逐字节复制
    assert (out / "manifest.csv").read_bytes() == \
        (cascade["synth"] / "manifest.csv").read_bytes()             # manifest 原样复制


def test_e_lin_erle_and_coherence_drop(cascade):
    """e_lin 对 far 的能量比/相干比 mic 显著下降：稳态 ERLE>10dB、far 音频率相干≈1→<0.5。"""
    mic = cascade["mic"].astype(np.float64)
    e = cascade["e_ref"].astype(np.float64)
    far = cascade["far"].astype(np.float64)
    k = int(len(mic) * STEADY)
    assert _erle_db(mic, e, k) > 10.0                        # 稳态 ERLE（实测 ~14dB）
    assert float(np.sum(e ** 2)) < float(np.sum(mic ** 2))   # 全程能量也下降
    f, c_mic = coherence(mic, far, fs=SR, nperseg=1024)
    _, c_e = coherence(e, far, fs=SR, nperseg=1024)
    for f0 in (300, 700, 1300, 1900):                        # far 的音频率
        i = int(np.argmin(np.abs(f - f0)))
        assert c_mic[i] > 0.9
        assert c_e[i] < 0.5                                  # 相干显著下降


def test_y_echo_tracks_echo(cascade):
    """y_echo（farend 槽）与合成 echo 高相关：稳态 >0.9（实测 ~0.99）、整体 >0.85。"""
    y = cascade["y_ref"].astype(np.float64)
    echo = cascade["echo"].astype(np.float64)
    k = int(len(y) * STEADY)
    assert _corr(y[k:], echo[k:]) > 0.9
    assert _corr(y, echo) > 0.85


# ---------------------------------------------------------------- 边界与续跑
def test_min_len_truncation(tmp_path):
    """mic 0.5s / far 1.0s → 双路按 min 截断为 8000，输出仍与参考逐位一致。"""
    far, echo, near, mic = _synth_case()
    synth, out = tmp_path / "s", tmp_path / "o"
    _make_synth_dir(synth, [("c01", mic[:8000], far, near[:8000].astype(np.float32))])
    _run_main(synth, out)
    e_out, _ = sf.read(out / "mic" / "c01.wav", dtype="float32")
    y_out, _ = sf.read(out / "farend" / "c01.wav", dtype="float32")
    e_ref, y_ref = _reference(mic[:8000], far)
    assert len(e_out) == len(y_out) == 8000
    np.testing.assert_array_equal(e_out, e_ref)
    np.testing.assert_array_equal(y_out, y_ref)


def test_skip_resume_and_force(cascade, tmp_path):
    """断点续跑：三槽齐即跳过（残留不被覆盖）；--force 才重算覆盖（结果仍逐位确定）。"""
    out2 = tmp_path / "out2"
    _run_main(cascade["synth"], out2)                        # 首跑：写三槽
    p_near = out2 / "nearend" / "c01.wav"
    p_near.write_bytes(b"junk-not-a-wav")                    # 模拟残留
    _run_main(cascade["synth"], out2)                        # 无 --force：跳过
    assert p_near.read_bytes() == b"junk-not-a-wav"
    _run_main(cascade["synth"], out2, force=True)            # --force：重算覆盖
    assert p_near.read_bytes() == \
        (cascade["synth"] / "nearend" / "c01.wav").read_bytes()
    e_out, _ = sf.read(out2 / "mic" / "c01.wav", dtype="float32")
    np.testing.assert_array_equal(e_out, cascade["e_ref"])   # 重算与参考逐位一致


def test_import_exposes_same_pipeline_mdf():
    """import 安全：模块级完成 sys.path 注入并从 exp/06_trad_aec 导入 TradAEC。"""
    assert callable(pc.main)
    assert pc.TradAEC is TradAEC                             # 与参考实现同源
