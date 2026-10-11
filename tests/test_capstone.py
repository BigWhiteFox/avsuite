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
    # 回声抑制用**引导段真 ERLE** 度量。DT 片段前 2.0s（= aec_synth 默认
    # --dt-leadin-sec）近端恒 0，mic 段只剩回声+噪声 ⇒ Σmic²/Σaec² 是 AEC
    # 抑制能力的直接读数。
    # 历史写法 np.correlate(x[:n], far[:n])[0] 是**零滞后未归一点积**且取样段恰为
    # 引导段：它只反映整体能量量级，AEC 只要把输出整体缩小（哪怕只是加个增益）
    # 就"通过"，对"是否真的消了回声"零信息，断言形同虚设。
    lead = int(2.0 * 16000)

    def erle_db(ref, out):
        r = float(np.sum(ref[:lead] ** 2))
        o = float(np.sum(out[:lead] ** 2))
        return 10.0 * np.log10((r + 1e-20) / (o + 1e-20))

    erle_aec = erle_db(sigs["mic"], sigs["aec"])
    assert erle_aec > 20.0, f"AEC 级引导段 ERLE {erle_aec:.1f}dB 应 >20dB（实测 67.6）"
    assert erle_db(sigs["mic"], sigs["nr"]) > erle_aec - 1e-9, "NR 级不应回退抑制"


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


# 已文档化的过度抑制水平带（t5 实测 + docs/NUMBERS.md 机理节）：
# Cascade 的 AEC 级在 DT 片段把输出压到 rms ≈ 1.5e-5，而 mic 级 rms ≈ 3.7e-2，
# 相当于引导段 ERLE ≈ 67dB —— 远超正常 AEC 所需的 20-30dB，说明近端成分被
# 连同回声一起压掉（不是"只消回声"）。这不是验收正确行为，而是**墓碑**：
_TOMBSTONE_AEC_RMS_BAND = (1e-6, 1e-4)      # 实测 1.5295e-05


@pytest.mark.skipif(not EVAL.exists(), reason="需要本机 eval_v3 数据(CI 跳过)")
def test_cascade_aec_over_suppression_tombstone(cascade):
    """墓碑：钉住 DT 片段 AEC 级的过度抑制水平（已知缺陷，勿当正确行为读）。

    读法：
      * 若 AEC 被修好（保留近端、ERLE 回到 20-40dB 合理区），aec rms 会显著变大
        ⇒ 上界失配，本用例必须被**显式**更新（这正是墓碑的用途：让"修好了"这件事
        无法悄悄发生，也提醒同步更新 docs/NUMBERS.md 的过度抑制记述）；
      * 若进一步恶化（输出更接近全零、近端被压得更狠），下界失配 ⇒ 抓回归。
    机理依据：NUMBERS deg 机理节（近端 0/回声延迟 618 样本的引导段占比）+ t5 实测。
    """
    mic = sf.read(EVAL / "mic" / "fileid_0_dt.wav", dtype="float32")[0][:32000]
    far = sf.read(EVAL / "farend" / "fileid_0_dt.wav", dtype="float32")[0][:32000]
    sigs = cascade.process(mic, far)
    lo, hi = _TOMBSTONE_AEC_RMS_BAND
    aec_rms = float(np.sqrt(np.mean(sigs["aec"] ** 2)))
    nr_rms = float(np.sqrt(np.mean(sigs["nr"] ** 2)))
    mic_rms = float(np.sqrt(np.mean(sigs["mic"] ** 2)))
    assert lo <= aec_rms <= hi, (
        f"AEC 级 rms {aec_rms:.3e} 超出已文档化的过度抑制带 [{lo:.0e}, {hi:.0e}]"
        f"（mic 级 rms {mic_rms:.3e}）；若这是修复所致，请更新本墓碑与 NUMBERS.md")
    assert nr_rms <= aec_rms, f"NR 级 rms {nr_rms:.3e} 不应大于 AEC 级 {aec_rms:.3e}"
