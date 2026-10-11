"""T11：`exp/07_nn_aec/dataset.py` 覆盖（test_gap.md §2 T11）。

原唯一测试（test_nn_aec.py::test_dataset_jitter_and_shift）依赖 `D:/data/aec-synth/eval_v3`，
CI 必跳 → CI 语境下 dataset.py 零覆盖。本文件用 `tmp_path` 造 manifest + 三路合成 wav，
把 `_shift` 语义、jitter 的 `seed*1000003+idx` 确定性、scenarios 过滤、空 manifest 报错
全部锁死（零 D:/data）。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DS = _load("t11_nnaec_dataset", "exp/07_nn_aec/dataset.py")
AecDataset, _shift = DS.AecDataset, DS._shift
SR = DS.SR


def test_shift_positive_and_negative():
    x = np.arange(10, dtype=np.float32)
    assert np.array_equal(_shift(x, 3), np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6], dtype=np.float32))
    assert np.array_equal(_shift(x, -2), np.array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0], dtype=np.float32))
    assert np.array_equal(_shift(x, 0), x)
    assert np.all(_shift(x, 20) == 0)


def _make_synth(tmp_path, specs):
    """specs: [(name, scenario, n_mic, n_far, n_near)]"""
    for sub in ("mic", "farend", "nearend"):
        (tmp_path / sub).mkdir(exist_ok=True)
    rows = ["name,scenario"]
    rng = np.random.default_rng(0)
    for name, sc, nm, nf, nn in specs:
        rows.append(f"{name},{sc}")
        for sub, n in (("mic", nm), ("farend", nf), ("nearend", nn)):
            sf.write(tmp_path / sub / f"{name}.wav",
                     (0.1 * rng.standard_normal(n)).astype(np.float32), SR)
    (tmp_path / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path


def test_dataset_reads_triples_and_truncates_to_min_len(tmp_path):
    d = _make_synth(tmp_path, [("a", "fest", 4000, 4000, 4000), ("b", "dt", 3500, 4000, 4000)])
    ds = AecDataset(d, jitter_ms=0)
    assert len(ds) == 2
    item = ds[1]
    assert item["mic"].shape == item["far"].shape == item["near"].shape
    assert item["mic"].shape[0] == 3500
    assert item["mic"].dtype == torch.float32
    assert item["name"] == "b" and item["scenario"] == "dt"


def test_jitter_is_deterministic_per_index_and_seed(tmp_path):
    d = _make_synth(tmp_path, [("a", "dt", 8000, 8000, 8000)])
    a1 = AecDataset(d, jitter_ms=10, seed=1)[0]["far"]
    a2 = AecDataset(d, jitter_ms=10, seed=1)[0]["far"]
    b1 = AecDataset(d, jitter_ms=10, seed=2)[0]["far"]
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, b1)


def test_jitter_moves_only_far_and_disabled_at_zero(tmp_path):
    d = _make_synth(tmp_path, [("a", "dt", 8000, 8000, 8000)])
    ds10 = AecDataset(d, jitter_ms=10, seed=1)
    ds0 = AecDataset(d, jitter_ms=0)
    it10, it0 = ds10[0], ds0[0]
    assert not torch.allclose(it10["far"], it0["far"])
    assert torch.allclose(it10["mic"], it0["mic"])
    assert torch.allclose(it10["near"], it0["near"])


def test_jitter_skipped_when_signal_too_short(tmp_path):
    """n <= 2*jitter 时不平移（jitter=160 样本时 n=320 走保护分支）。"""
    d = _make_synth(tmp_path, [("a", "dt", 320, 320, 320)])
    a = AecDataset(d, jitter_ms=10, seed=1)[0]["far"]
    b = AecDataset(d, jitter_ms=0)[0]["far"]
    assert torch.equal(a, b)


def test_scenarios_filter(tmp_path):
    d = _make_synth(tmp_path, [("f", "fest", 2000, 2000, 2000), ("d", "dt", 2000, 2000, 2000),
                               ("n", "nst", 2000, 2000, 2000)])
    ds = AecDataset(d, jitter_ms=0, scenarios=["dt"])
    assert len(ds) == 1 and ds[0]["scenario"] == "dt"


def test_empty_or_missing_manifest_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "manifest.csv").write_text("name,scenario\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="manifest 空或缺失"):
        AecDataset(d)
    with pytest.raises(FileNotFoundError):
        AecDataset(tmp_path / "does_not_exist")
