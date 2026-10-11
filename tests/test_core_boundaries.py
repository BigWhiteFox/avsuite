"""T4：`core/` 边界条件补测（test_gap.md §2 T4）。

core 覆盖最好，但既有测试几乎只覆盖"约定内"输入；V5 正在改 core（t11：istft 的
`flush()`/`hop==n_fft` 告警/`length<0` 校验），以下边界必须有断言兜底。

**与 test_gap §2 T4 的偏差（以 t11 落定态为准）**：
- `ISTFT.inverse(length=-1)` 现在**显式抛 ValueError**（t11 修复；spec 记录的是旧行为
  "静默截断"）→ 本文件 pin 新行为，旧行为不再复现；
- `core.__all__` 与实际导出仍漂移（v3a-010）→ 用 xfail 固化现状并标注。
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.evaluate import _safe_pesq, _safe_stoi, _si_snr_np  # noqa: E402
from core.istft import ISTFT, StreamingISTFT  # noqa: E402
from core.losses import multi_scale_stft_loss, si_snr, si_snr_loss  # noqa: E402
from core.stft import STFT, StreamingSTFT  # noqa: E402


def _torch_stft_zeros_pad(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    # 注意：torch.stft 的 pad_mode 不接受 "zeros"（本仓 core.STFT 用自有字符串约定），
    # 等价写法是 pad_mode="constant"（默认 value=0）。
    return torch.stft(x, n_fft, hop_length=hop, window=torch.hann_window(n_fft),
                      center=True, pad_mode="constant", return_complex=True)


def test_stft_invalid_window_name_raises():
    with pytest.raises(ValueError, match="unsupported window"):
        STFT(window="bogus").transform(torch.randn(1, 1600))
    for w in ("hann", "hamming", "blackman"):
        STFT(window=w).transform(torch.randn(1, 1600))


def test_stft_reflect_pad_requires_long_signal():
    for n in (100, 1):
        with pytest.raises(ValueError, match="pad_mode"):
            STFT().transform(torch.randn(1, n))
        spec = STFT(pad_mode="zeros").transform(torch.randn(1, n))
        assert spec.shape == (1, 257, 1)


def test_stft_odd_and_tiny_lengths():
    for n in (1, 511, 512, 513):
        x = torch.randn(1, n)
        spec = STFT(n_fft=512, hop=256, pad_mode="zeros").transform(x)
        assert spec.shape == (1, 257, n // 256 + 1)
        assert torch.isfinite(spec).all()
        ref = _torch_stft_zeros_pad(x, 512, 256)
        torch.testing.assert_close(spec, ref, rtol=0, atol=1e-5)


def test_istft_negative_length_is_rejected():
    """t11 修复固化：负 length 显式拒绝（旧行为是按负索引静默截尾）。"""
    spec = STFT().transform(torch.randn(1, 1600))
    assert ISTFT().inverse(spec, length=None).shape[-1] == 1536
    for bad in (-1, -2):
        with pytest.raises(ValueError, match="length must be"):
            ISTFT().inverse(spec, length=bad)
    assert ISTFT().inverse(spec, length=0).shape[-1] == 0
    assert ISTFT().inverse(spec, length=20000).shape[-1] == 20000


def test_istft_center_false_zero_overlap_point_is_zeroed():
    n_fft, hop = 256, 64
    wav = torch.randn(2, 16000, dtype=torch.float64)
    spec = STFT(n_fft=n_fft, hop=hop, window="hann", center=False,
                pad_mode="zeros").transform(wav)
    out = ISTFT(n_fft=n_fft, hop=hop, window="hann", center=False).inverse(spec)
    assert out.shape == (2, 16000)
    assert out[0, 0].item() == 0.0 and out[1, 0].item() == 0.0
    assert torch.isfinite(out).all()
    # torch.istft 对同配置直接拒绝（本实现能算）——差分断言
    with pytest.raises(RuntimeError):
        torch.istft(spec, n_fft, hop_length=hop, win_length=n_fft, center=False,
                    window=torch.hann_window(n_fft, periodic=True))


def test_streaming_istft_hop_validation_surface():
    """t11：hop==n_fft + hann/blackman 有一次性告警；hop 非法仍应被拒。"""
    with pytest.raises(ValueError):
        ISTFT(n_fft=512, hop=0)
    st = StreamingISTFT(n_fft=512, hop=256)
    assert st.add_frame(np.zeros(512)).shape == (256,)


def test_streaming_stft_flush_without_push_returns_empty():
    """t24 修复固化（v3a 高危）：从未 push 的空流 flush() 返回空列表，不抛异常。

    旧实现落到 `_cat()` 的 `torch.cat([])` → RuntimeError；center=False
    （_pad=0）与 center=True 两条路径都要覆盖。
    """
    for center in (True, False):
        st = StreamingSTFT(STFT(512, 256, center=center, pad_mode="zeros"))
        assert st.flush() == []
        assert st.cache_size() == 0
    # 空流 flush 后仍按纪律拒绝再 push
    st = StreamingSTFT(STFT())
    st.flush()
    with pytest.raises(RuntimeError):
        st.push(torch.randn(1, 100))
    # 正常路径不受影响：push 后 flush 仍产出尾帧
    st2 = StreamingSTFT(STFT(512, 256, pad_mode="zeros"))
    st2.push(torch.randn(1, 1000))
    tail = st2.flush()
    assert isinstance(tail, list) and len(tail) > 0


def test_losses_si_snr_batch_shapes_and_dtype():
    a = torch.randn(3, 7, 1000)
    b = torch.randn(3, 7, 1000)
    v = si_snr(a, b)
    assert v.shape == (3, 7)
    assert si_snr(a.to(torch.float32), b.to(torch.float32)).dtype == torch.float32
    assert si_snr_loss(a, b).ndim == 0


def test_losses_multi_scale_stft_short_input_boundary():
    for n in (2047, 2049):
        v = multi_scale_stft_loss(torch.zeros(1, n), torch.zeros(1, n))
        assert torch.isfinite(v)


def test_evaluate_si_snr_np_degenerate_is_nan():
    """t24 修复固化：静音/塌陷（全零 ref 或 est）显式 NaN，不再返回 0.0dB 假满分。"""
    assert np.isnan(_si_snr_np(np.zeros(100), np.zeros(100)))
    assert np.isnan(_si_snr_np(np.ones(100), np.zeros(100)))      # est 塌陷
    assert np.isnan(_si_snr_np(np.zeros(100), np.ones(100)))      # ref 静音
    assert np.isnan(_si_snr_np(np.zeros(0), np.zeros(0)))         # 空输入


def test_evaluate_si_snr_np_aligns_to_shorter():
    """t24 修复固化：入口按较短者截断（旧行为是 numpy 广播 ValueError）。"""
    a = _si_snr_np(np.ones(100), np.ones(50))
    b = _si_snr_np(np.ones(50), np.ones(50))
    assert a == b and np.isfinite(a)
    # 截断语义：与显式先截断再调用一致
    ref = np.arange(200, dtype=np.float64)
    est = ref * 0.9
    assert _si_snr_np(ref, est[:120]) == _si_snr_np(ref[:120], est[:120])
    # 正常路径数值不变：等长调用与旧公式一致
    x, y = np.random.default_rng(0).standard_normal(500), \
        np.random.default_rng(1).standard_normal(500)
    xd, yd = x - x.mean(), y - y.mean()
    proj = np.sum(yd * xd) / max(np.sum(xd ** 2), 1e-8) * xd
    expect = float(10 * np.log10(max(np.sum(proj ** 2), 1e-8)
                                 / max(np.sum((yd - proj) ** 2), 1e-8)))
    assert _si_snr_np(x, y) == expect


def test_evaluate_safe_pesq_stoi_degenerate_returns_nan():
    with pytest.warns(RuntimeWarning):
        assert np.isnan(_safe_pesq(np.zeros(100), np.zeros(100)))
    assert np.isnan(_safe_stoi(np.zeros(100), np.zeros(100)))


def test_core_reexports_stft_istft():
    import core
    assert core.STFT is core.stft.STFT
    assert core.ISTFT is core.istft.ISTFT


def _core_all_probe() -> tuple[bool, bool, list[str]]:
    """独立子进程里 `import core` 后探测 losses/evaluate 是否真的可用。

    必须起子进程：本测试进程内其它用例早已 `from core.losses import ...`，
    属性会被子模块导入副作用带进来，进程内 hasattr 恒为真（测不出漂移）。
    """
    import json
    import subprocess
    code = ("import json,core;print(json.dumps([hasattr(core,'losses'),"
            "hasattr(core,'evaluate'),list(core.__all__)]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    has_l, has_e, names = json.loads(out.stdout.strip())
    return has_l, has_e, names


@pytest.mark.xfail(strict=False, reason="v3a-010：core.__all__ 声明 losses/evaluate 但 "
                                        "__init__.py 未导入 → 裸 `import core` 后属性缺失"
                                        "（已知缺口，t14 只记录不改本体）")
def test_core_all_is_importable_in_fresh_interpreter():
    has_l, has_e, names = _core_all_probe()
    assert "losses" in names and "evaluate" in names
    assert has_l and has_e, f"__all__={names} 但 losses/evaluate 不可用"


def test_core_all_declared_names_exist_in_fresh_interpreter():
    """无条件契约：`import core` 之后 `core.__all__` 里的名字都应可用（缺口项除外）。"""
    import json
    import subprocess
    gap = {"losses", "evaluate"}          # v3a-010 已知漂移项（由上面 xfail 用例表达）
    code = ("import json,core;print(json.dumps({n: hasattr(core, n) for n in core.__all__}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    present = json.loads(out.stdout.strip())
    missing = [n for n, ok in present.items() if not ok and n not in gap]
    assert not missing, f"__all__ 声明但不可用: {missing}"
    assert "STFT" in present and "ISTFT" in present
