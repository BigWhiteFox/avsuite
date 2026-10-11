"""T2：`exp/06_trad_aec/kalman_eval.py` 纯函数语义（test_gap.md §2 T2）。

`far_gate`/`erle_curve`/`window_erle`/`t90_of`/`summarize`/`build_conclusion` 零覆盖，
而它们承载 V5 要"严谨化"的结论口径（t90 vs t(15dB) 退化、DT 段读数只作趋势）。

**与 test_gap §2 T2 的两处偏差（以落定态为准）**：
- `summarize` 的字段名是 `dt_seg_erle_median` / `nst_steady_median` / `fest_steady_median`
  与 `curve_all`/`curve_fest`（不是 spec 写的 `dt_seg_erle_median is None` + 嵌套 `curve`）；
- spec 预言 `build_conclusion` 在 DT 全缺失时抛 `TypeError`，实测（本机）是
  `ValueError: max() arg is an empty sequence` —— 本文件按实测值 pin，并在 docstring 标注
  为**待修的真实健壮性缺口**（t14 不修本体，仅固化现状 + 上报）。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def _load(name: str, rel: str, extra_path: str | None = None):
    if extra_path:
        sys.path.insert(0, str(ROOT / extra_path))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


KE = _load("t2_kalman_eval", "exp/06_trad_aec/kalman_eval.py", "exp/06_trad_aec")
PIPE = _load("t2_pipeline", "exp/06_trad_aec/pipeline.py", "exp/06_trad_aec")

CFG = "MDF(mu0.7,P8)"


def _rec(name, scenario, steady, **kw):
    d = {"name": name, "scenario": scenario, "config": CFG, "steady2s_erle_db": steady,
         "t90_s": "", "dt_seg_erle_db": "", "dt_pesq": "", "dt_corr_near": "", "wall_s": 1.0,
         "erle_t0.5": steady - 1.0, "erle_t1": steady}
    d.update(kw)
    return d


def test_far_gate_matches_pipeline_semantics():
    """`far_gate` 必须与 `pipeline.TradAEC._far_gate` 逐点一致（防两处实现漂移）。"""
    far_al = np.concatenate([np.full(100 * 256, 0.1), np.zeros(20 * 256)])
    gate = KE.far_gate(far_al, 120)
    assert np.allclose(gate[:100], 1.0)
    assert np.allclose(gate[100:], KE.IDLE)
    ref = PIPE.TradAEC()._far_gate(far_al, 120)
    assert np.array_equal(gate, ref)


def test_erle_curve_and_window_erle_consistency():
    rng = np.random.default_rng(0)
    mic = rng.standard_normal(4 * SR)
    e = mic * 0.1
    xs, ys = KE.erle_curve(mic, e)
    assert np.allclose(ys, 20.0, atol=0.1)
    assert abs(KE.window_erle(mic, e, 2.0, 4.0) - 20.0) < 0.1
    assert xs[0] == KE.CURVE_STEP and np.allclose(np.diff(xs), KE.CURVE_STEP)


def test_window_erle_clamps_t1_to_len():
    rng = np.random.default_rng(1)
    mic = rng.standard_normal(2 * SR)
    e = mic * 0.5
    assert KE.window_erle(mic, e, 0.0, 99.0) == KE.window_erle(mic, e, 0.0, 2.0)


def test_t90_of_threshold_and_nan():
    xs = np.array([0.5, 1.0, 1.5, 2.0])
    assert KE.t90_of(xs, np.array([1.0, 5.0, 9.0, 10.0]), 10.0) == 1.5
    assert np.isnan(KE.t90_of(xs, np.array([1.0, 2.0, 3.0, 4.0]), 10.0))


def test_summarize_handles_empty_and_partial_groups():
    """只有 2 条 fest：dt/nst 相关字段应为 None、dt 分离散点计数为 0、curve_dt 不存在。"""
    recs = [_rec("a", "fest", 11.0), _rec("b", "fest", 15.0)]
    s = KE.summarize(recs, [0.5, 1.0])[CFG]
    assert s["n"] == 2
    assert s["fest_steady_median"] == 13.0
    assert s["dt_seg_erle_median"] is None
    assert s["nst_steady_median"] is None
    assert s["dt_n_diverge"] == 0
    assert "curve_fest" in s and "curve_dt" not in s


def test_build_conclusion_tolerates_missing_dt_and_diag_groups():
    """t23 健壮化：DT 组/诊断组整体缺失时不再抛 `max()` 空序列 ValueError。

    原实现（`dt_best = max(dte)`、`ste_diag = max(...)`）在"只跑 FEST 子集"或
    "分批跑网格"时直接崩溃；现改为按 0 条/缺失处理并在第 3 行显式标注。数据齐全时
    输出与旧实现逐字一致（见下一条用例）。
    """
    recs = [_rec("a", "fest", 11.0), _rec("b", "fest", 15.0)]
    s = KE.summarize(recs, [0.5, 1.0])
    lines = KE.build_conclusion(s)                       # 不应抛异常
    assert len(lines) == 3
    assert "缺失" in lines[2] and "跳过" in lines[2]


def test_build_conclusion_missing_dt_reports_zero_count():
    """缺 DT 时第 3 行必须写明"0 条"，避免读者把空对比误读成"无差异"。"""
    summary = {}
    for cfg in KE.CONFIGS:
        summary[cfg["label"]] = {
            "n": 0, "fest_steady_median": 12.0, "fest_steady_mean": 12.0,
            "fest_t90_median_s": 2.0, "fest_t90_n": 3, "fest_t15_median_s": 3.0,
            "fest_t15_n": 12, "dt_seg_erle_median": None, "dt_pesq_median": None,
            "dt_corr_near_median": None, "dt_n_diverge": 0, "nst_steady_median": None,
            "full_steady_median": 12.0, "diag": bool(cfg.get("diag")), "wall_s_total": 0.0,
        }
    lines = KE.build_conclusion(summary)
    assert "0 条" in lines[2] and "缺失" in lines[2]


def test_build_conclusion_three_lines_content():
    """完整合法 summary（6 个 CONFIGS 标签齐全、字段齐备）时返回 3 行。"""
    summary = {}
    for cfg in KE.CONFIGS:
        lab = cfg["label"]
        summary[lab] = {
            "n": 4, "fest_steady_median": 12.0 + len(summary), "fest_steady_mean": 12.0,
            "fest_t90_median_s": 2.0, "fest_t90_n": 3, "fest_t15_median_s": 3.0,
            "fest_t15_n": 12, "dt_seg_erle_median": 10.0 + len(summary),
            "dt_pesq_median": 1.5, "dt_corr_near_median": 0.9, "dt_n_diverge": 0,
            "nst_steady_median": 5.0, "full_steady_median": 11.0, "diag": bool(cfg.get("diag")),
            "wall_s_total": 3.0,
        }
    lines = KE.build_conclusion(summary)
    assert len(lines) == 3
    assert "t(15dB)" in lines[0] and "/12 条" in lines[0]
    assert "MDF" in lines[1] and "dB" in lines[1]
    assert "DT" in lines[2] and "PESQ" in lines[2]
