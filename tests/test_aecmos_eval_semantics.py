"""T10：`exp/07_nn_aec/aecmos_eval.py` 聚合语义（test_gap.md §2 T10）。

`score_all()` 的分场景聚合 + `round(...,3)` + 缺场景跳过，与 `load_clips()` 的读取口径
此前零覆盖；AECMOS 是 WP-07 的 DoD 判据，对外数字精度必须有锁。

**口径缺陷另见**（t19 分诊 P0-1）：本文件的模型是 48kHz 版，而 16k 波形未重采样直送
→ 绝对值不可引用。本测试只锁"聚合/映射/精度"这些**与采样率无关**的语义。
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


AE = _load("t10_aecmos_eval", "exp/07_nn_aec/aecmos_eval.py")


class FakeEst:
    def __init__(self, ret=None):
        self.talks = []
        self.ret = ret or {}

    def run(self, talk, lpb, mic, enh):
        self.talks.append(talk)
        return self.ret.get(talk, (4.0, 3.0))


def _pair(name, sc, n=1600):
    x = np.zeros(n, dtype=np.float32)
    return (name, sc, x, x, x)


def test_score_all_maps_scenario_to_talk_type():
    est = FakeEst()
    pairs = [_pair("f", "fest"), _pair("d", "dt"), _pair("n", "nst")]
    rows, _ = AE.score_all(pairs, est)
    assert est.talks == ["st", "dt", "nst"]
    assert [r["scenario"] for r in rows] == ["fest", "dt", "nst"]


def test_score_all_aggregates_and_rounds_per_scenario():
    est = FakeEst({"dt": (4.0, 3.0), "st": (2.0, 1.0)})
    pairs = [_pair("d1", "dt"), _pair("d2", "dt"), _pair("f1", "fest")]
    est.ret["dt"] = (4.0, 3.0)
    # 两条 dt 分别返回不同值：用可变的假 estimator
    class Est2:
        def __init__(self):
            self.k = 0

        def run(self, talk, lpb, mic, enh):
            if talk == "dt":
                self.k += 1
                return (4.0, 3.0) if self.k == 1 else (4.5, 3.5)
            return (2.0, 1.0)

    _, summary = AE.score_all(pairs, Est2())
    assert summary["dt"]["n"] == 2
    assert summary["dt"]["echo_mos"] == 4.25
    assert abs(summary["dt"]["deg_mos"] - 3.25) < 1e-9
    assert summary["fest"]["n"] == 1
    assert "nst" not in summary


def test_score_all_rounds_to_three_decimals():
    class Est3:
        def __init__(self):
            self.k = 0

        def run(self, talk, lpb, mic, enh):
            self.k += 1
            return (1.0004 if self.k < 3 else 1.0005, 2.0)

    pairs = [_pair(f"d{i}", "dt") for i in range(3)]
    _, summary = AE.score_all(pairs, Est3())
    assert summary["dt"]["echo_mos"] == 1.0


def test_score_all_rows_order_and_keys():
    est = FakeEst()
    pairs = [_pair("a", "fest"), _pair("b", "dt")]
    rows, _ = AE.score_all(pairs, est)
    assert len(rows) == 2
    assert [r["name"] for r in rows] == ["a", "b"]
    assert set(rows[0]) == {"name", "scenario", "echo_mos", "deg_mos"}


def test_load_clips_reads_manifest_and_pairs_wavs(tmp_path):
    (tmp_path / "farend").mkdir()
    (tmp_path / "mic").mkdir()
    (tmp_path / "manifest.csv").write_text(
        "name,scenario\nfileid_0,fest\nfileid_1,dt\nfileid_2,nst\n", encoding="utf-8")
    for name, val in (("fileid_0", 0.5), ("fileid_1", 0.25), ("fileid_2", 0.1)):
        sf.write(tmp_path / "farend" / f"{name}.wav", np.full(400, val, dtype=np.float32), 16000)
        sf.write(tmp_path / "mic" / f"{name}.wav", np.full(400, val * 2, dtype=np.float32), 16000)

    clips = AE.load_clips(tmp_path)
    assert len(clips) == 3
    name, sc, lpb, mic = clips[0]
    assert (name, sc) == ("fileid_0", "fest")
    assert lpb.dtype == np.float32 and mic.dtype == np.float32
    # 命名易错点：lpb 取自 farend/（0.5），mic 取自 mic/（1.0）
    assert pytest.approx(lpb[0], rel=1e-3) == 0.5
    assert pytest.approx(mic[0], rel=1e-3) == 1.0


def test_load_clips_empty_manifest_returns_empty(tmp_path):
    (tmp_path / "manifest.csv").write_text("name,scenario\n", encoding="utf-8")
    assert AE.load_clips(tmp_path) == []
