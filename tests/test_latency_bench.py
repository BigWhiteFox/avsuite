"""T5：`tools/latency_bench.py` 纯函数覆盖（test_gap.md §2 T5）。

`stats_ms` / `cpu_name` / `mdf_blockwise_equivalence` 是"时延叙事（AEC 32ms + NR 32ms）
实测底账"的依据，此前零覆盖。`bench_mdf_block()`（~1000 块，5–10s）按 test_gap 的
"可选"处理：本文件只做轻量键位检查的默认关闭（见文件末说明），保证 <30s。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str, extra_path: str | None = None):
    if extra_path:
        sys.path.insert(0, str(ROOT / extra_path))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LB = _load("t5_latency_bench", "tools/latency_bench.py", "exp/06_trad_aec")


def test_stats_ms_hand_computed():
    s = LB.stats_ms([1.0, 2.0, 3.0, 4.0])
    assert s["mean_ms"] == 2.5
    assert s["p50_ms"] == 2.5
    assert s["min_ms"] == 1.0
    assert s["max_ms"] == 4.0
    for k in ("mean_ms", "p50_ms", "min_ms", "max_ms", "p95_ms", "p99_ms"):
        assert isinstance(s[k], float)
        assert round(s[k], 4) == s[k]
        assert s["min_ms"] <= s[k] <= s["max_ms"], k


def test_stats_ms_single_sample():
    s = LB.stats_ms([7.5])
    for k, v in s.items():
        assert v == 7.5, f"{k}={v}"


def test_stats_ms_accepts_list_and_ndarray():
    data = [1.0, 3.0, 2.0, 5.0]
    a = LB.stats_ms(data)
    b = LB.stats_ms(np.asarray(data))
    assert a == b


def test_cpu_name_returns_nonempty_str():
    name = LB.cpu_name()
    assert isinstance(name, str) and len(name) > 0


def test_mdf_blockwise_equivalence_is_bitwise():
    r = LB.mdf_blockwise_equivalence(n_blocks=20)
    assert r["bitwise_equal"] is True
    assert r["max_abs_diff"] == 0.0
    assert r["samples"] == 20 * 256


def test_bench_mdf_block_keys_and_rtf_relation():
    """轻量断言（~5–10s）：键位齐全 + rtf ≈ mean_ms/16 + 分位单调。

    若嫌慢可用 `pytest -k "not bench_mdf_block"` 跳过；本文件其余用例 <2s。
    """
    r = LB.bench_mdf_block()
    need = {"mean_ms", "p50_ms", "p95_ms", "p99_ms", "min_ms", "max_ms",
            "rtf_block_mean", "block_samples", "warmup_chunks", "timed_chunks"}
    assert need <= set(r)
    assert r["block_samples"] == 256
    assert r["p50_ms"] <= r["p95_ms"] <= r["p99_ms"]
    assert r["rtf_block_mean"] == pytest.approx(r["mean_ms"] / 16.0, abs=1e-4)
