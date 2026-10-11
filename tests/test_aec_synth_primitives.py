"""T8：`scripts/aec_synth.py` 基元补口（test_gap.md §2 T8）。

`fit_len`/`db_scale`/`rir_exponential`/`get_rir`/`load_rir_dir`/`load_wav_16k` 此前零覆盖，
而它们"错了会让整个数据集静默偏掉"。全部用合成数组 + `tmp_path`，无 D:/data、无网络。

注：`get_rir` 的走廊模拟分支（pyroomacoustics）用 monkeypatch 强制 ImportError，
以走 `rir_exponential` 兜底路径 —— 既快又确定。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AS = _load("t8_aec_synth", "scripts/aec_synth.py")
SR = AS.SR


def test_fit_len_short_input_tiles_exactly():
    x = np.arange(5)
    out = AS.fit_len(x, 17, np.random.default_rng(0))
    assert len(out) == 17
    assert np.array_equal(out, np.tile(x, 4)[:17])


def test_fit_len_long_input_random_crop_is_deterministic():
    x = np.arange(100)
    a = AS.fit_len(x, 20, np.random.default_rng(7))
    b = AS.fit_len(x, 20, np.random.default_rng(7))
    assert len(a) == 20 and np.array_equal(a, b)
    assert np.all(np.diff(a) == 1)          # 连续切片，不是首段


def test_db_scale_targets_exact_db():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR)
    for target in (-6.0, 0.0, 6.0):
        out = AS.db_scale(x, 1.0, target)
        got = 10 * np.log10(np.mean(out ** 2) / 1.0)
        assert abs(got - target) < 1e-9


def test_db_scale_zero_reference_passthrough():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1000)
    out = AS.db_scale(x, 0.0, 6.0)
    assert np.array_equal(out, x)           # FEST 零噪声路径依赖"跳过定标"


def test_rir_exponential_shape_and_direct_path_delay():
    rng = np.random.default_rng(3)
    h = AS.rir_exponential(rng)
    assert h[0] == 0.0
    assert abs(np.linalg.norm(h) - 10.0) < 1e-6
    n_max = int(SR * 0.6) + int(0.03 * SR)
    assert 0 < len(h) <= n_max


def test_get_rir_truncates_with_max_ms(monkeypatch):
    monkeypatch.setattr(AS, "rir_simulate_pra",
                        lambda *a, **k: (_ for _ in ()).throw(ImportError("no pra")))
    rng = np.random.default_rng(1)
    h50 = AS.get_rir(None, rng, max_ms=50)
    assert len(h50) == int(SR * 0.05)
    h_full = AS.get_rir(None, np.random.default_rng(1))
    assert len(h_full) >= len(h50)


def test_load_rir_dir_empty_raises(tmp_path):
    empty = tmp_path / "rir_empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="无 wav"):
        AS.load_rir_dir(empty)


def test_get_rir_from_dir_averages_multichannel(tmp_path):
    d = tmp_path / "rir"
    d.mkdir()
    stereo = np.stack([np.linspace(0, 1, 800), np.linspace(1, 0, 800)], axis=1)
    sf.write(d / "a_stereo.wav", stereo, 48000)
    sf.write(d / "b_mono.wav", np.linspace(0, 1, 500).astype(np.float32), 48000)
    files = AS.load_rir_dir(d)
    assert [p.name for p in files] == ["a_stereo.wav", "b_mono.wav"]
    h = AS.get_rir(files, np.random.default_rng(0))
    assert h.ndim == 1 and len(h) > 0


def test_load_wav_16k_rejects_wrong_samplerate(tmp_path):
    bad = tmp_path / "bad8k.wav"
    sf.write(bad, np.zeros(800), 8000)
    with pytest.raises(AssertionError):
        AS.load_wav_16k(bad)
    ok = tmp_path / "ok16k.wav"
    stereo = np.stack([np.ones(1600), -np.ones(1600)], axis=1)
    sf.write(ok, stereo, SR)
    y = AS.load_wav_16k(ok)
    assert y.ndim == 1 and len(y) == 1600
    # 左右反相降混 -> ≈0；用 atol 吸收 WAV 默认 PCM_16 的量化残差（±1 LSB）
    assert np.allclose(y, 0.0, atol=1e-4)
