"""t13 单测：评测矩阵驱动与汇总的**口径**部分（不依赖真实音频/GPU）。

覆盖：
(a) `eval_matrix.erle_split`：整段/瞬态(0–2 s)/稳态(>2 s) 三分报的数值口径——
    造一个"前 2 s 无抑制、之后抑制 20 dB"的确定性例子，断言三列分别等于解析值；
(b) `eval_matrix_report` 的统计层：配对 bootstrap 在常数差上必得常数 CI、在均值上有界；
    两样本 bootstrap 的区间包含均值差；`boot_paired`/`boot_two` 确定性（同 seed 同结果）；
(c) `eval_matrix_report.build`：用 6 行合成读数走通 绝对分 / 集内配对差 / 跨集差（含配对标记）/ DiD，
    并检查饱和守卫（FEST deg≈5.0 且 sd≈0 ⇒ saturated=True）。

用法：`pytest tests/test_eval_matrix.py -q`
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp" / "07_nn_aec"))
sys.path.insert(0, str(ROOT / "exp" / "06_trad_aec"))
sys.path.insert(0, str(ROOT))

import eval_matrix as EM  # noqa: E402
import eval_matrix_report as ER  # noqa: E402


# ---------------------------------------------------------------- (a) ERLE 三分报
def test_erle_split_three_band_convention():
    sr = EM.SR
    n = 8 * sr
    ref = np.ones(n, dtype=np.float64) * 0.5          # 参考（mic）
    est = ref.copy()
    est[2 * sr:] = ref[2 * sr:] * 0.1                 # 2 s 之后抑制 20 dB
    got = EM.erle_split(est, ref)
    assert got["erle_transient"] == pytest.approx(0.0, abs=1e-6)      # 0–2 s 无抑制
    assert got["erle_steady"] == pytest.approx(20.0, abs=1e-3)        # >2 s = 20 dB
    assert got["erle_full"] == pytest.approx(10 * math.log10(8 / (2 + 6 * 0.01)), abs=1e-3)


def test_erle_split_short_clip_has_none_bands():
    sr = EM.SR
    ref = np.ones(sr, dtype=np.float64)               # 1 s < SKIP_S
    est = ref * 0.5
    got = EM.erle_split(est, ref)
    assert got["erle_full"] == pytest.approx(6.0206, abs=1e-3)
    assert got["erle_steady"] is None                 # 没有 >2 s 的段
    assert got["erle_transient"] is not None


# ---------------------------------------------------------------- (b) 统计层
def test_bootstrap_deterministic_and_constant():
    d = [1.0] * 20
    lo1, hi1 = ER.boot_paired(d)
    lo2, hi2 = ER.boot_paired(d)
    assert (lo1, hi1) == (lo2, hi2)                   # 同 seed 确定
    assert lo1 == pytest.approx(1.0) and hi1 == pytest.approx(1.0)
    a = [2.0] * 30
    b = [1.0] * 30
    lo, hi = ER.boot_two(a, b)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_cell_saturation_guard_and_ci():
    sat = ER._cell([5.0, 5.0, 5.0, 5.0, 5.0])
    assert sat["saturated"] is True and sat["sd"] == 0.0
    live = ER._cell([3.0, 3.2, 2.8, 3.1, 2.9])
    assert "saturated" not in live
    assert live["ci_half"] > 0 and live["n"] == 5


# ---------------------------------------------------------------- (c) 汇总 build
def _write(out: Path, tag: str, set_name: str, rows: list[dict]):
    d = out / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{set_name}.json").write_text(
        json.dumps({"key": f"{tag}:{set_name}", "rows": rows}, ensure_ascii=False),
        encoding="utf-8")


def _rows(n: int, echo: float, deg: float, sc: str = "dt") -> list[dict]:
    out = []
    for i in range(n):
        out.append({"name": f"f{i}_{sc}", "scenario": sc, "echo_mos": echo, "deg_mos": deg,
                    "pesq": 1.5, "stoi": 0.9, "sisnr": 5.0, "pesq_mic": 1.2})
    return out


# ---------------------------------------------------------------- (d) 级联口径
def test_cascade_enhance_uses_lin_residual_and_echo_estimate():
    """级联 = 同一个 NN 换一对输入 `(e_lin, y_echo)`（口径见 exp/09_cascade_aec/train.py 头注）。

    用桩对象断言参数顺序与线性级公式：`e_lin = mic - y_echo`，且 NN 的第二路输入是
    `y_echo`（**不是**原始 far）——写错会让 B 档读数整体失真。
    """
    import numpy as np

    class _Aec:
        def process(self, mic, far):
            return type("R", (), {"y_echo": 0.25 * far})()

    class _NN:
        def __init__(self):
            self.args = None

        def enhance(self, a, b):
            self.args = (a.cpu().numpy().copy(), b.cpu().numpy().copy())
            return a * 0.0      # 形状 [:1,N]

    mic = np.full(64, 1.0, dtype=np.float32)
    far = np.array([0.0, 0.5, 1.0, 1.5] * 16, dtype=np.float32)[:64]
    nn = _NN()
    out = EM.enhance("cascade", nn, mic, far, "cpu", aec=_Aec())
    e_lin, y_echo = nn.args
    assert np.allclose(y_echo, 0.25 * far, atol=1e-6)          # 第二路 = 线性回声估计
    assert np.allclose(e_lin, mic - 0.25 * far, atol=1e-6)     # 第一路 = 线性残差
    assert out.shape == mic.shape


def test_linear_kind_is_mdf_residual_without_nn():
    """MDF-only 消融（t28 附加）：`linear` 必须**完全不调用 NN**，输出 = `mic − y_echo`。"""
    import numpy as np

    class _Aec:
        def process(self, mic, far):
            return type("R", (), {"y_echo": 0.25 * far})()

    class _NNMustNotBeCalled:
        def enhance(self, *a, **kw):        # pragma: no cover - 调用即失败
            raise AssertionError("linear 路径不应调用 NN")

        def enhance_wav(self, *a, **kw):    # pragma: no cover
            raise AssertionError("linear 路径不应调用 NN")

    mic = np.full(32, 1.0, dtype=np.float32)
    far = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    out = EM.enhance("linear", _NNMustNotBeCalled(), mic, far, "cpu", aec=_Aec())
    assert np.allclose(out, mic - 0.25 * far, atol=1e-6)
    assert out.shape == mic.shape


def test_cascade_kind_parsed_from_ckpt_spec(monkeypatch, tmp_path):
    """`--ckpt name=path:cascade` 必须被解析成 kind=cascade（否则会退化成纯 NN 口径）。"""
    ck = tmp_path / "fake.ckpt"
    ck.write_bytes(b"x")
    seen = {}

    def _fake_process_key(key, ckpts, *a, **kw):
        seen.update(ckpts)
        return {"key": key, "rows": []}

    monkeypatch.setattr(EM, "process_key", _fake_process_key)
    monkeypatch.setattr(EM, "make_aecmos", lambda: None)
    monkeypatch.setattr(sys, "argv", ["eval_matrix.py", "--ckpt", f"casc={ck}:cascade",
                                      "--keys", "casc:eval_v8"])
    assert EM.main() == 0
    assert seen["casc"]["kind"] == "cascade" and seen["casc"]["path"] == ck


def test_build_tables_end_to_end(tmp_path, monkeypatch):
    for tag in ("trad", "v4"):
        _write(tmp_path, tag, "eval_v8", _rows(6, 3.5 if tag == "trad" else 4.0,
                                               3.8 if tag == "trad" else 3.7))
        _write(tmp_path, tag, "eval_v5", _rows(6, 3.4 if tag == "trad" else 3.9, 3.8))
        _write(tmp_path, tag, "eval_rir", _rows(6, 3.6 if tag == "trad" else 4.1, 3.7))
        _write(tmp_path, tag, "eval_rir_sim", _rows(6, 3.5 if tag == "trad" else 4.0, 3.7))
    res = ER.build(ER.load(tmp_path))
    # 绝对分
    assert res["absolute"]["v4"]["eval_v8"]["dt.echo_mos"]["mean"] == pytest.approx(4.0)
    # 集内配对差 tag−trad
    p = res["paired_vs_trad"]["v4"]["eval_v8"]["dt.echo_mos"]
    assert p["delta"] == pytest.approx(0.5) and p["ci"] == [0.5, 0.5]
    # 跨集差：eval_v8 vs eval_v5 非配对；eval_rir vs eval_rir_sim 逐样本配对
    cs = res["cross_set"]["v4"]
    assert cs["eval_v8-eval_v5.dt.echo_mos"]["paired"] is False
    assert cs["eval_rir-eval_rir_sim.dt.echo_mos"]["paired"] is True
    assert cs["eval_rir-eval_rir_sim.dt.echo_mos"]["delta"] == pytest.approx(0.1)
    # DiD：(v4−trad)@rir − (v4−trad)@rir_sim = 0.5 − 0.5 = 0
    did = res["did"]["v4|eval_rir-eval_rir_sim.dt.echo_mos"]
    assert did["did"] == pytest.approx(0.0, abs=1e-9) and did["sig"] is False
    md = ER.to_md(res)
    assert "eval_rir-eval_rir_sim" in md and "逐样本配对" in md


def test_load_skips_missing(tmp_path):
    _write(tmp_path, "trad", "eval_v8", _rows(6, 3.5, 3.8))
    data = ER.load(tmp_path)
    assert ("trad", "eval_v8") in data and ("v4", "eval_v8") not in data
