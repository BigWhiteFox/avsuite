"""T6：`tools/evaluate_full.py` 纯逻辑覆盖（test_gap.md §2 T6）。

`_score_one`（指标计算）/`collect_cases`（分组与配对）/`dnsmos_score_all`（无参考分）
是全仓"最终验收数字"的生产者，此前零覆盖。全部用合成信号 + `tmp_path`，无 D:/data。
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.evaluate import _safe_pesq, _safe_stoi, _si_snr_np  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EF = _load("t6_evaluate_full", "tools/evaluate_full.py")
SR = 16000


def _pseudo_speech(n=SR, seed=0):
    """类语音信号（与 tests/test_evaluate.py::_pseudo_speech 同族口径）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f * t) for f in (140.0, 220.0, 330.0, 460.0))
    env = 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 4.0 * t))
    return (0.2 * x * env + 0.01 * rng.standard_normal(n)).astype(np.float32)


def test_score_one_matches_core_primitives():
    ref = _pseudo_speech(SR)
    enh = 0.9 * ref
    row = EF._score_one(("c1", "g", enh, ref))
    assert row["si_snr"] == _si_snr_np(ref, enh)
    assert row["pesq_wb"] == _safe_pesq(ref, enh)
    assert row["stoi"] == _safe_stoi(ref, enh)
    assert set(row) == {"name", "group", "pesq_wb", "stoi", "si_snr"}


def test_score_one_non_fest_unchanged_by_extra_mic_element():
    """非 fest 场景：多传第 5 项（mic）不得改变任何数值（向后兼容 4 元组）。"""
    ref = _pseudo_speech(SR)
    enh = 0.9 * ref
    mic = _pseudo_speech(SR, seed=3)
    r4 = EF._score_one(("c1", "g", enh, ref))
    r5 = EF._score_one(("c1", "g", enh, ref, mic))
    assert r4 == r5
    assert "erle_db" not in r5


def test_score_one_fest_zero_ref_skips_reference_metrics():
    """fest（nearend 全零）：不算 PESQ/STOI/SI-SNR（记 NaN），只算 ERLE。"""
    ref = np.zeros(SR * 3, dtype=np.float32)
    mic = _pseudo_speech(SR * 3, seed=5)
    enh = 0.3 * mic
    row = EF._score_one(("f1", "fest", enh, ref, mic))
    assert row["pesq_wb"] != row["pesq_wb"]          # NaN
    assert row["stoi"] != row["stoi"]
    assert row["si_snr"] != row["si_snr"]
    assert np.isfinite(row["erle_db"])


def test_score_one_fest_erle_matches_exp07_formula():
    """ERLE 口径与 exp/07_nn_aec/eval.py::erle_db 完全一致（跳 2s + 1e-12 下限）。"""
    n = SR * 4
    mic = _pseudo_speech(n, seed=7)
    enh = 0.25 * mic
    k = int(2.0 * SR)
    expect = 10 * np.log10((np.sum(mic[k:] ** 2) + 1e-12) / (np.sum(enh[k:] ** 2) + 1e-12))
    row = EF._score_one(("f2", "fest", enh, np.zeros(n, dtype=np.float32), mic))
    assert row["erle_db"] == pytest.approx(float(expect))
    assert row["erle_db"] == pytest.approx(10 * np.log10(1 / 0.25 ** 2), abs=1e-6)


def test_score_one_fest_without_mic_yields_nan_erle():
    """fest 但未提供 mic（旧 4 元组调用）：ERLE 记 NaN 而不是崩。"""
    row = EF._score_one(("f3", "fest", np.ones(SR, dtype=np.float32),
                         np.zeros(SR, dtype=np.float32)))
    assert np.isnan(row["erle_db"]) and np.isnan(row["si_snr"])


def test_score_one_truncates_to_shorter():
    ref = _pseudo_speech(SR)
    enh = np.concatenate([0.8 * ref, np.random.default_rng(1).standard_normal(4000)]).astype(np.float32)
    row = EF._score_one(("c2", "g", enh, ref))
    assert row["si_snr"] == _si_snr_np(ref, enh[:SR])
    assert row["pesq_wb"] == _safe_pesq(ref, enh[:SR])


def _write_pairs(tmp_path, names, only_noisy=None):
    clean = tmp_path / "clean"
    noisy = tmp_path / "noisy"
    clean.mkdir(exist_ok=True)
    noisy.mkdir(exist_ok=True)
    x = _pseudo_speech(4000)
    for n in names:
        sf.write(clean / n, x, SR)
        if n != only_noisy:
            sf.write(noisy / n, x, SR)
    return noisy, clean


def test_collect_cases_speaker_grouping(tmp_path):
    noisy, clean = _write_pairs(tmp_path, ["p226_001.wav", "p227_002.wav", "x_003.wav"])
    sf.write(noisy / "only_noisy.wav", _pseudo_speech(4000), SR)      # 只在一侧 -> 排除
    args = SimpleNamespace(model="crn", synth=0, group_by="speaker", max_files=0,
                           noisy_dir=noisy, clean_dir=clean)
    cases = EF.collect_cases(args)
    assert len(cases) == 3
    assert [c[1] for c in cases] == ["p226", "p227", "x"]
    assert all(c[0] != "only_noisy.wav" for c in cases)


def test_collect_cases_group_by_all_and_max_files(tmp_path):
    noisy, clean = _write_pairs(tmp_path, ["p226_001.wav", "p227_002.wav", "x_003.wav"])
    args = SimpleNamespace(model="crn", synth=0, group_by="all", max_files=2,
                           noisy_dir=noisy, clean_dir=clean)
    cases = EF.collect_cases(args)
    assert len(cases) == 2
    assert {c[1] for c in cases} == {"all"}


def test_collect_cases_resamples_non_16k_via_load16k(tmp_path):
    """t24（v3a g29 修复）：collect_cases 走 core.evaluate.load16k——48k 输入被重采样到 16k。

    旧实现 `sf.read(...)[0]` 忽略原生采样率，48k 数据会被当 16k 直接喂模型（数值全错）。
    """
    noisy = tmp_path / "noisy"
    clean = tmp_path / "clean"
    noisy.mkdir(), clean.mkdir()
    x = _pseudo_speech(48000)                                # 48000 采样，按 48k 写出 = 1s
    sf.write(noisy / "p226_048.wav", x, 48000)
    sf.write(clean / "p226_048.wav", x, 48000)
    args = SimpleNamespace(model="crn", synth=0, group_by="speaker", max_files=0,
                           noisy_dir=noisy, clean_dir=clean)
    cases = EF.collect_cases(args)
    assert len(cases) == 1
    _name, _grp, nz, cl, far = cases[0]
    assert len(nz) == SR and len(cl) == SR, (len(nz), len(cl))   # 48k -> 16k
    assert nz.dtype == np.float32 and far is None


def test_dnsmos_score_all_skips_unavailable_scores():
    class FakeDNS:
        def score(self, w):
            return None if len(w) < SR else {"ovr": 1.0, "sig": 2.0, "bak": 3.0}

    short = np.zeros(SR // 2, dtype=np.float32)
    long = np.zeros(SR, dtype=np.float32)
    out = EF.dnsmos_score_all([("a", None, short, None), ("b", None, long, None)], FakeDNS())
    assert set(out) == {"b"}
    assert out["b"]["ovr"] == 1.0


def test_load_model_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        EF.load_model(SimpleNamespace(model="bogus"))


def test_load_module_registers_in_sys_modules(monkeypatch):
    mod = EF._load_module("evfull_probe", ROOT / "exp/04_dccrn/dynamic_mixing.py")
    assert sys.modules["evfull_probe"] is mod
    assert hasattr(mod, "DNSDynamicMix")
    monkeypatch.delitem(sys.modules, "evfull_probe", raising=False)
