"""S2 单测：官方 AECMOS 评估脚本的协议语义（official_aecmos_eval.py，不改本体）。

覆盖：
- trim_for 送评裁剪三分支（官方 proto 口径：FEST 送后半段 / DT 送最后 (len_s-15)/2 s /
  NST 全段、rating<=0 不裁）与 full / t2s 对照口径、±1/±2 样本边界、非法 variant；
- load16 官方 48k→16k 往返：resample_poly(1,3) 长度 = ceil(n48/3)、>7.5kHz 能量≈0、
  与测试内参考重采样逐位一致（sf FLOAT 落盘逐位往返）、双路 min_len 截断；
- import 安全：模块级只有路径常量与 sys.path 注入，import 不触发 D:/data / 网络读取。

被测模块用 importlib 按绝对路径加载（tests/ 不与其构成包依赖）。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

_REPO = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO / "exp" / "07_nn_aec" / "official_aecmos_eval.py"

_spec = importlib.util.spec_from_file_location("official_aecmos_eval", _MODULE_PATH)
oae = importlib.util.module_from_spec(_spec)
sys.modules["official_aecmos_eval"] = oae
_spec.loader.exec_module(oae)              # 副作用仅 sys.path 注入 + 路径常量，无 IO

SR48 = oae.SR_AECMOS                       # 48000


def _sigs(n):
    """三路可区分等长信号（裁剪一致性逐路可见）。"""
    x = np.arange(n, dtype=np.float64)
    return [x, x * 2.0 + 1000.0, -x - 7.0]


# ---------------------------------------------------------------- import 安全
def test_import_side_effects_are_path_only():
    """import 本身成功即证明不依赖 D:/data；常量为纯 Path/数值；副作用仅 sys.path 注入。"""
    assert oae.SR_PROC == 16000 and oae.SR_AECMOS == 48000
    assert set(oae.SUBSETS) == {"fest", "dt", "nst"}
    assert oae.TALK == {"fest": "st", "dt": "dt", "nst": "nst"}
    for p in (oae.TEST_ROOT, oae.AECMOS_DIR, oae.AECMOS_MODEL, oae.STAGE, oae.REPORTS):
        assert isinstance(p, Path)                       # 只是常量，import 期不触盘
    assert str(_MODULE_PATH.parent) in sys.path          # HERE 注入（model.py 用）
    assert str(_REPO / "exp" / "06_trad_aec") in sys.path  # WP-06 管道注入


# ---------------------------------------------------------------- trim_for
def test_trim_proto_fest_last_half():
    """FEST：官方送后半段（last n//2，奇数下取整）；三路同裁；返回裁掉样本数 n-k。"""
    for n in (1001, 48000 * 40):                         # 奇数边界 + 官方真实 40s 规模
        sigs = _sigs(n)
        out, trim, note = oae.trim_for("fest", "proto", sigs)
        k = n // 2
        assert len(out[0]) == k
        for a, b in zip(out, sigs):
            np.testing.assert_array_equal(a, b[-k:])
        assert trim == n - k
        assert "FEST" in note and "后半段" in note


def test_trim_proto_dt_last_len_minus_15_over_2():
    """DT：官方送最后 (len_s-15)/2 s（int 样本数）：40s→12.5s，37s→11s。"""
    for n in (48000 * 40, 48000 * 37):
        sigs = _sigs(n)
        out, trim, note = oae.trim_for("dt", "proto", sigs)
        k = int((n - 15 * SR48) / 2)
        assert len(out[0]) == k
        for a, b in zip(out, sigs):
            np.testing.assert_array_equal(a, b[-k:])
        assert trim == n - k
        assert "(len_s-15)/2" in note


def test_trim_proto_dt_rating_le_zero_no_trim():
    """len<=15s（15s 整 / −1 / +1 / 14s）→ k<=0 不裁，三路原样返回。"""
    for n in (15 * SR48, 15 * SR48 - 1, 15 * SR48 + 1, 14 * SR48):
        sigs = _sigs(n)
        out, trim, note = oae.trim_for("dt", "proto", sigs)
        for a, b in zip(out, sigs):
            np.testing.assert_array_equal(a, b)
        assert trim == 0 and "不裁" in note


def test_trim_proto_dt_first_cut_sample_boundary():
    """首刀边界：+2 样本 → k=1 只留最后 1 样本；+3 样本 int() 截断同为 k=1。"""
    for n in (15 * SR48 + 2, 15 * SR48 + 3):
        sigs = _sigs(n)
        out, trim, note = oae.trim_for("dt", "proto", sigs)
        assert len(out[0]) == 1 and trim == n - 1
        np.testing.assert_array_equal(out[0], sigs[0][-1:])


def test_trim_proto_nst_full_and_full_variant():
    """NST 全段送评 + full 对照口径：一律原样返回（trim=0）。"""
    n = 5000
    sigs = _sigs(n)
    for subset, variant, fragment in (("nst", "proto", "NST"), ("dt", "full", "全段")):
        out, trim, note = oae.trim_for(subset, variant, sigs)
        for a, b in zip(out, sigs):
            np.testing.assert_array_equal(a, b)
        assert trim == 0 and fragment in note


def test_trim_t2s_crops_min_of_2s_and_half():
    """t2s 口径：去前 min(2s, n//2) 个样本；对 subset 不敏感（三子集同行为）。"""
    for n, k in ((250000, 96000), (1000, 500), (191999, 95999)):
        sigs = _sigs(n)
        for subset in ("fest", "dt", "nst"):
            out, trim, _note = oae.trim_for(subset, "t2s", sigs)
            assert k == min(2 * SR48, n // 2)
            assert len(out[0]) == n - k and trim == k
            for a, b in zip(out, sigs):
                np.testing.assert_array_equal(a, b[k:])


def test_trim_invalid_variant_raises():
    with pytest.raises(ValueError, match="bogus"):
        oae.trim_for("nst", "bogus", _sigs(100))


# ---------------------------------------------------------------- load16
@pytest.fixture(scope="module")
def wav48(tmp_path_factory):
    """48k FLOAT 落盘的官方集同构样本：lpb/mic 等长 1.2s，另有 1.0s 短 mic。"""
    root = tmp_path_factory.mktemp("official48")
    n = 57600                                            # 1.2s
    t = np.arange(n) / 48000.0
    base = (0.5 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 2000 * t)
            + 0.2 * np.sin(2 * np.pi * 4000 * t))
    oob = 0.4 * np.sin(2 * np.pi * 10000 * t) + 0.3 * np.sin(2 * np.pi * 14000 * t)
    lpb = (base + oob).astype(np.float32)                # 含 >8kHz 带外分量
    mic = (0.8 * base).astype(np.float32)
    p_lpb, p_mic, p_short = root / "a.wav", root / "b.wav", root / "d.wav"
    sf.write(p_lpb, lpb, 48000, subtype="FLOAT")
    sf.write(p_mic, mic, 48000, subtype="FLOAT")
    sf.write(p_short, lpb[:48000], 48000, subtype="FLOAT")   # 1.0s
    return dict(clip={"lpb": str(p_lpb), "mic": str(p_mic)},
                clip_short={"lpb": str(p_lpb), "mic": str(p_short)},
                lpb=lpb, mic=mic, short=lpb[:48000])


def test_load16_roundtrip_bitwise_and_length(wav48):
    """48k→16k：长度 = ceil(n48/3)；与测试内参考 resample_poly(1,3) 逐位一致。"""
    l16, m16 = oae.load16(wav48["clip"])
    assert l16.dtype == np.float32 and m16.dtype == np.float32
    n48 = len(wav48["lpb"])
    assert len(l16) == len(m16) == -(-n48 // 3)          # ≈ n48/3（上取整）
    ref_l = resample_poly(wav48["lpb"], 1, 3)
    ref_m = resample_poly(wav48["mic"], 1, 3)
    np.testing.assert_array_equal(l16, ref_l)            # 逐位一致
    np.testing.assert_array_equal(m16, ref_m)


def test_load16_high_band_energy_near_zero(wav48):
    """>7.5kHz 残余能量占比≈0（抗混叠后 10k/14k 带外音几乎消失，实测 ~1e-6）。"""
    l16, _ = oae.load16(wav48["clip"])
    spec = np.abs(np.fft.rfft(l16)) ** 2
    f = np.fft.rfftfreq(len(l16), d=1.0 / oae.SR_PROC)
    hi = float(spec[f > 7500].sum() / spec.sum())
    assert hi < 1e-5
    assert float(spec[f <= 7500].sum() / spec.sum()) > 1 - 1e-5


def test_load16_min_len_across_channels(wav48):
    """lpb 1.2s / mic 1.0s → 双路截到 min=16000；长路为前缀截断。"""
    l16, m16 = oae.load16(wav48["clip_short"])
    n_min = min(-(-57600 // 3), -(-48000 // 3))
    assert len(l16) == len(m16) == n_min
    np.testing.assert_array_equal(m16, resample_poly(wav48["short"], 1, 3))
    np.testing.assert_array_equal(l16, resample_poly(wav48["lpb"], 1, 3)[:n_min])
