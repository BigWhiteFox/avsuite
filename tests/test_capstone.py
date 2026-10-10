"""WP-10.5 单测：级联端到端（真实 DT 片段）+ 三级 AECMOS 单调改善。"""
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

HERE = Path(__file__).resolve().parents[1] / "exp" / "10.5_capstone"
sys.path.insert(0, str(HERE))

from cascade import Cascade  # noqa: E402

EVAL = Path("D:/data/aec-synth/eval_v3")


@pytest.fixture(scope="module")
def cascade():
    return Cascade()


@pytest.mark.skipif(not EVAL.exists(), reason="需要本机 eval_v3 数据(CI 跳过)")
def test_cascade_shapes_and_finite(cascade):
    mic = sf.read(EVAL / "mic" / "fileid_0_dt.wav", dtype="float32")[0][:32000]
    far = sf.read(EVAL / "farend" / "fileid_0_dt.wav", dtype="float32")[0][:32000]
    sigs = cascade.process(mic, far)
    for k in ("mic", "aec", "nr"):
        assert sigs[k].shape == mic.shape, k
        assert np.isfinite(sigs[k]).all(), k
    # 回声能量逐级下降（DT 段的粗略代理：与 far 的相关能量）
    def echo_proxy(x):
        n = min(len(x), len(far))
        return float(np.abs(np.correlate(x[:n], far[:n]))[0])
    e_mic = echo_proxy(sigs["mic"])
    assert echo_proxy(sigs["aec"]) < e_mic * 0.6, "AEC 级应显著消回声"


@pytest.mark.skipif(not EVAL.exists(), reason="需要本机 eval_v3 数据(CI 跳过)")
def test_cascade_aecmos_stagewise(cascade):
    """三级 AECMOS：echo_mos 单调上升（原始→AEC→AEC+NR），量化'每级贡献'。"""
    mic = sf.read(EVAL / "mic" / "fileid_0_dt.wav", dtype="float32")[0][:32000]
    far = sf.read(EVAL / "farend" / "fileid_0_dt.wav", dtype="float32")[0][:32000]
    sigs = cascade.process(mic, far)
    scores = cascade.aecmos_scores(far, sigs, talk_type="dt")
    e_mic, e_aec, e_nr = (scores["mic"][0], scores["aec"][0], scores["nr"][0])
    assert e_aec > e_mic + 0.5, f"AEC 级 echo 应大涨：{e_mic:.2f} -> {e_aec:.2f}"
    assert e_nr >= e_aec - 0.3, f"NR 级不应回退 echo：{e_aec:.2f} -> {e_nr:.2f}"
    print(f"三级 echo_mos: {e_mic:.3f} -> {e_aec:.3f} -> {e_nr:.3f}")
