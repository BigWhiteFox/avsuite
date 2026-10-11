"""T7：`tools/dnsmos.py` 官方口径（V5 wave-C t24 校准后）。

用假 `sess` 直接注入，绕开 `_ensure_model()/_download()`：零网络、零 D:/data。

**口径（t24 已对齐官方 dnsmos_local.py；t19 分诊 P0-2 修复）**：
- 校准：raw 三段经官方三条 polyfit 映射为 SIG/BAK/OVR（raw 以 `*_raw` 保留）；
- 短片段：**重复自身**补足到 9.01s（不是零补）；
- 长音频：hop = fs（1s），段数 `int(floor(len/fs) − 9.01) + 1`。
实测对齐：同一批 clip 与官方算法 max|Δ| = 0.000e+00（见 t24 回报与 triage_v3 §P0-2）。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DM = _load("t7_dnsmos", "tools/dnsmos.py")
SR, FIXED = DM.SR, DM.FIXED_LEN

KEYS = {"sig", "bak", "ovr", "sig_raw", "bak_raw", "ovr_raw"}


class FakeSess:
    """记录调用、输入形状与**实际喂入的数组**，按调用序返回递增的三段 raw 分。"""

    def __init__(self):
        self.calls = []
        self.inputs = []
        self._k = 0

    def run(self, _outs, feed):
        x = feed["input_1"]
        self.calls.append(x.shape)
        self.inputs.append(np.array(x, copy=True))
        self._k += 1
        # 与 tools/dnsmos.py 的 `...[0][0]` 解包对齐：返回形状 (1,3) 的批次
        return [np.array([[float(self._k), float(self._k) + 0.5, float(self._k) + 0.25]])]


def _mk(sess=None):
    d = DM.DNSMOS.__new__(DM.DNSMOS)
    d.sess = sess
    d.input_name = "input_1" if sess is not None else None
    d.model_path = "fake.onnx" if sess is not None else None
    return d


def test_score_returns_none_when_model_unavailable():
    d = _mk(None)
    assert d.available is False
    assert d.score(np.zeros(SR)) is None


def test_score_returns_none_for_short_wav():
    fake = FakeSess()
    d = _mk(fake)
    assert d.score(np.zeros(SR // 2 - 1)) is None
    assert fake.calls == []


def test_score_repeat_pads_short_segment_to_fixed_len():
    """短片段按官方"重复自身"补齐（非零补）：8000 -> 256000 采样 = 7 段。"""
    fake = FakeSess()
    d = _mk(fake)
    m = d.score(np.full(8000, 2.0, dtype=np.float32))
    assert len(fake.calls) == 7, fake.calls
    assert all(shape == (1, FIXED) for shape in fake.calls)
    assert set(m) == KEYS
    assert all(isinstance(v, float) for v in m.values())
    # 判别"重复自身" vs "零补"：第 8000 个样本在零补下是 0.0，重复下仍是 2.0
    fed = fake.inputs[0][0]
    assert fed[8000] == 2.0, "应为重复自身填充（零补会在此处得 0.0）"
    assert fed[16000] == 2.0


def test_score_slides_window_with_official_1s_hop():
    """官方 hop = fs（1s）滑窗，且复刻官方"短 1 样本即跳过"的截断行为。

    官方 `int((idx + 9.01) * fs)` 因浮点表示会让约一半窗口短 1 个样本、被
    `continue` 跳过——本实现逐位复刻（与 auditor 的官方算法对拍 max|Δ| = 0，
    见 t24 回报）。此处用"样本序号当信号"验证窗起点 = 1s 整数倍 + 实测段数。
    """
    fake = FakeSess()
    d = _mk(fake)
    n = int(FIXED * 2.5)
    x = np.arange(n, dtype=np.float32)          # float32 可精确表示到 2^24，序号可验窗起点
    _ = d.score(x)
    assert len(fake.calls) == 7, f"实测 7 段（官方截断跳过后）: {fake.calls}"
    assert all(shape == (1, FIXED) for shape in fake.calls)
    starts = [int(inp[0, 0]) for inp in fake.inputs]
    assert starts == [0, 16000, 32000, 48000, 64000, 80000, 96000], starts   # hop = 1s
    m = d.score(np.zeros(n, dtype=np.float32))
    assert set(m) == KEYS


def test_score_applies_official_polyfit_calibration():
    """raw 三段经官方 polyfit 映射；raw 值以 *_raw 保留。"""
    fake = FakeSess()
    d = _mk(fake)
    m = d.score(np.zeros(FIXED, dtype=np.float32))       # 恰好 1 段，k=1
    assert m["sig_raw"] == 1.0 and m["bak_raw"] == 1.5 and m["ovr_raw"] == 1.25
    assert m["ovr"] == pytest.approx(float(DM._POLY_OVR(1.25)))
    assert m["sig"] == pytest.approx(float(DM._POLY_SIG(1.0)))
    assert m["bak"] == pytest.approx(float(DM._POLY_BAK(1.5)))
    # 校准必须真的改变数值（否则等于没校准）
    assert m["ovr"] != m["ovr_raw"]


def test_run_truncates_and_returns_calibrated_plus_raw():
    fake = FakeSess()
    d = _mk(fake)
    out = d._run(np.zeros(FIXED + 500, dtype=np.float32))
    assert fake.calls[-1] == (1, FIXED)
    assert set(out) == KEYS and all(isinstance(v, float) for v in out.values())


def test_run_repeat_pads_when_called_directly():
    fake = FakeSess()
    d = _mk(fake)
    _ = d._run(np.full(1000, 3.0, dtype=np.float32))
    assert fake.calls[-1] == (1, FIXED)
    assert fake.inputs[-1][0][1000] == 3.0            # 重复自身，非零补


def test_repeat_pad_handles_empty_array_without_hanging():
    out = DM._repeat_pad(np.zeros(0, dtype=np.float32), 100)
    assert out.shape == (100,) and not out.any()


def test_ensure_model_prefers_existing_cache(tmp_path, monkeypatch):
    cache = tmp_path / "m.onnx"
    cache.write_bytes(b"x" * 200_000)
    monkeypatch.setattr(DM, "CACHE", cache)
    monkeypatch.setattr(DM, "_download", lambda: pytest.fail("不应联网下载"))
    assert DM._ensure_model() == str(cache)


def test_ensure_model_returns_none_when_download_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(DM, "CACHE", tmp_path / "missing.onnx")
    monkeypatch.setattr(DM, "_download", lambda: False)
    assert DM._ensure_model() is None
