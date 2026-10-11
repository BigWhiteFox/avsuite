"""WP-05b 单测：流式 vs 非流式逐帧对齐（<1e-5）+ cache 有界性。"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nnaec_shim as X  # noqa: E402

NNAEC = X.NNAEC

# 本机合成数据根目录（CI 无此数据，相关用例按存在性跳过）
EVAL_V3 = Path("D:/data/aec-synth/eval_v3")


def _load_model():
    """确定性权重的 NNAEC（v3a-023）。

    原实现 `NNAEC().eval()` 未播种 ⇒ 权重随全局 RNG 漂移，同一用例在不同
    收集顺序/不同 pytest 版本下拿到不同网络，回归不可复现。这里固定种子。
    注意：本函数**不**改 BN running stats——保持 bn(0)=0，使既有 parity 契约
    继续成立；非平凡 BN 的 warmup 覆盖见
    test_warmup_frames_match_offline_with_nontrivial_bn。
    """
    torch.manual_seed(0)
    return NNAEC().eval()


def _set_nontrivial_bn(model, mean=0.5, var=4.0):
    """把 BatchNorm2d 的 running stats 设成非平凡值（v3a-020/R14 的判别力来源）。

    未训练模型 running_mean=0/var=1 ⇒ bn(0)=0（PReLU beta 也为 0），于是流式
    warmup 分支的"直接返回零"与离线 act(bn(0))=0 数值相同，缺陷不可见。
    置为非平凡值后 bn(0) = gamma·(0-mean)/sqrt(var+eps) + beta ≠ 0。
    """
    n = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            with torch.no_grad():
                mod.running_mean.fill_(mean)
                mod.running_var.fill_(var)
            n += 1
    assert n > 0, "模型内无 BatchNorm2d，本用例失去判别力"
    return n


def _offline_ref(model, mic, far):
    """离线参考：zeros-pad STFT（与 StreamingSTFT 同约定）-> forward_spec -> ISTFT。"""
    from core.istft import ISTFT
    from core.stft import STFT
    stft = STFT(512, 320, pad_mode="zeros")
    ms = stft.transform(torch.from_numpy(mic.astype(np.float32)).unsqueeze(0))
    fs = stft.transform(torch.from_numpy(far.astype(np.float32)).unsqueeze(0))
    with torch.no_grad():
        mask, enh = model.forward_spec(ms, fs)
    return ISTFT(512, 320).inverse(enh, length=None)[0].numpy()


def _run_streaming(model, mic, far, chunk=320):
    from streaming import NNAECStreaming
    st = NNAECStreaming(model, device="cpu")
    outs = []
    for i in range(0, len(mic), chunk):
        outs.append(st.process_chunk(mic[i:i + chunk], far[i:i + chunk]))
    outs.append(st.flush())
    return np.concatenate(outs)


def _align(stream_out, ref):
    """流式输出 out[j] = 离线 y_full[j]（帧网格从 0 起，pad 区被窗自然乘零）；
    离线 center 裁剪 = y_full[256 : 256+(T-1)·hop]——对齐取同区间。

    注意（t18 复核，原口径的盲区）：本裁剪的终点 256+(T-1)·hop 落在**丢尾区之前**。
    以 3s（n=48000, T=151）为例：全网格长 (T-1)·hop+n_fft = 48512，尾段为最后
    n_fft-hop = 192 个样本；而旧实现（未透传 core ISTFT flush）流式总长 = T·hop
    = 48320 ≥ 256+(T-1)·hop = 48256，裁剪区间 [256, 48256) 完全落在已发射区内
    → **旧测试全绿也发现不了丢尾**。全网格不丢尾由下面两个用例专门断言。
    """
    assert len(stream_out) >= 256 + len(ref), (len(stream_out), len(ref))
    return stream_out[256: 256 + len(ref)]


def _offline_full_ref(model, mic, far):
    """离线**全网格**参考（center=False，不裁剪）：用于校验流式不丢尾样点。

    与 _offline_ref 的区别：_offline_ref 返回 center=True 裁剪后的 (T-1)·hop 点，
    而流式输出 out[j] = y_full[j]、全网格长 (T-1)·hop + n_fft。
    """
    from core.istft import ISTFT
    from core.stft import STFT
    stft = STFT(512, 320, pad_mode="zeros")
    ms = stft.transform(torch.from_numpy(mic.astype(np.float32)).unsqueeze(0))
    fs = stft.transform(torch.from_numpy(far.astype(np.float32)).unsqueeze(0))
    with torch.no_grad():
        mask, enh = model.forward_spec(ms, fs)
    return ISTFT(512, 320, center=False).inverse(enh, length=None)[0].numpy()


def test_streaming_matches_offline_320chunks():
    """320 定块：流式输出 [256:256+(T-1)hop] 与离线一致（<1e-5，实测应 ~1e-10）。"""
    rng = np.random.default_rng(0)
    n = 16000 * 3                                   # 3s -> T=151 帧
    mic = (0.3 * rng.standard_normal(n)).astype(np.float32)
    far = (0.3 * rng.standard_normal(n)).astype(np.float32)
    model = _load_model()
    ref = _offline_ref(model, mic, far)            # 长度 (T-1)*hop = 48000
    out = _run_streaming(model, mic, far, 320)
    got = _align(out, ref)
    assert got.shape == ref.shape
    err = np.abs(got - ref).max()
    assert err < 1e-5, f"逐帧对齐误差 {err:.2e} >= 1e-5"


def test_streaming_matches_offline_irregular_chunks():
    """不规则块（800/640/320 混合）：块边界不影响逐帧等价性。"""
    rng = np.random.default_rng(1)
    n = 16000 * 3
    mic = (0.3 * rng.standard_normal(n)).astype(np.float32)
    far = (0.3 * rng.standard_normal(n)).astype(np.float32)
    model = _load_model()
    ref = _offline_ref(model, mic, far)
    from streaming import NNAECStreaming
    st = NNAECStreaming(model, device="cpu")
    outs = []
    i = 0
    pattern = [800, 640, 320, 960, 320]
    while i < n:
        c = pattern[len(outs) % len(pattern)]
        outs.append(st.process_chunk(mic[i:i + c], far[i:i + c]))
        i += c
    outs.append(st.flush())
    out = np.concatenate(outs)
    got = _align(out, ref)
    err = np.abs(got - ref).max()
    assert err < 1e-5, f"不规则块对齐误差 {err:.2e} >= 1e-5"


def test_flush_recovers_istft_tail_full_grid():
    """flush() 透传 core ISTFT flush：流式总长 == 离线全网格 (T-1)·hop + n_fft。

    回归点（t18）：旧实现 flush() 只结算 STFT 侧，core ISTFT 缓冲里残留
    n_fft-hop = 192 样点（12ms）永不发射，流式总长停在 T·hop = 48320 而非全网格
    48512。旧 _align 的裁剪终点 48256 落在丢尾区之前，故旧测试全绿也发现不了
    （详见 _align docstring）。

    容差口径：主体区（= 离线 center 裁剪区）保持既有的 <1e-5 契约；尾段位于窗
    边缘、单帧覆盖，重建需除以 w²（w[511]=3.765e-5）而把 float32 帧误差放大
    1/w ≈ 2.7e4，故绝对 <1e-5 在该区不可达（实测绝对差 ~1.5e-4）——改用
    **相对峰值**判据（实测 9.1e-8，属 float32 级一致）。
    """
    rng = np.random.default_rng(3)
    n = 16000 * 3                                    # 3s -> T=151 帧，全网格 48512
    mic = (0.3 * rng.standard_normal(n)).astype(np.float32)
    far = (0.3 * rng.standard_normal(n)).astype(np.float32)
    model = _load_model()
    full = _offline_full_ref(model, mic, far)
    out = _run_streaming(model, mic, far, 320)

    # 1) 不丢尾：流式总长 == 离线全网格长（旧实现差 192）
    assert len(out) == len(full), (
        f"流式总长 {len(out)} != 离线全网格 {len(full)}"
        f"（差 {len(full) - len(out)} = n_fft-hop 尾样点丢失）")

    # 2) 主体区（= 离线 center 裁剪区）保持 <1e-5 契约
    crop_end = 256 + (full.shape[0] - 512)
    err_crop = np.abs(out[256:crop_end] - full[256:crop_end]).max()
    assert err_crop < 1e-5, f"主体区对齐误差 {err_crop:.2e} >= 1e-5"

    # 3) 尾段：非零（排除"补零"式假通过）+ 相对峰值一致
    tail = slice(-(512 - 320), None)
    assert np.abs(full[tail]).max() > 0, "参考尾段全零，本用例失去判别力"
    assert np.abs(out[tail]).max() > 0, "流式尾段全零（疑似补零而非真实结算）"
    rel = np.abs(out[tail] - full[tail]).max() / np.abs(full[tail]).max()
    assert rel < 1e-5, f"尾段相对峰值误差 {rel:.2e} >= 1e-5"


def test_flush_worst_case_stft_side_yields_no_frame():
    """最坏情形：L 使 STFT 侧 flush 不产出新帧（(L+64) mod 320 < 64，如 L=3136）。

    此时旧实现 flush() 返回**空数组**，末尾 192 样点整段丢失；透传 core ISTFT
    flush 后返回值全部来自 ISTFT 尾段（恰 192 个），总长达离线全网格。
    """
    rng = np.random.default_rng(4)
    n = 3136                                         # (3136+64) % 320 == 0
    mic = (0.3 * rng.standard_normal(n)).astype(np.float32)
    far = (0.3 * rng.standard_normal(n)).astype(np.float32)
    model = _load_model()
    full = _offline_full_ref(model, mic, far)
    from streaming import NNAECStreaming
    st = NNAECStreaming(model, device="cpu")
    outs = [st.process_chunk(mic[i:i + 320], far[i:i + 320])
            for i in range(0, n, 320)]
    tail = st.flush()
    assert len(tail) == 512 - 320, (
        f"该 L 下 STFT 侧无尾帧，flush 应完全由 ISTFT 尾段构成（{512 - 320} 个），"
        f"实得 {len(tail)}")
    out = np.concatenate([*outs, tail])
    assert len(out) == len(full), f"流式总长 {len(out)} != 全网格 {len(full)}"
    crop_end = 256 + (full.shape[0] - 512)
    assert np.abs(out[256:crop_end] - full[256:crop_end]).max() < 1e-5
    t = slice(-(512 - 320), None)
    assert np.abs(full[t]).max() > 0 and np.abs(out[t]).max() > 0
    assert np.abs(out[t] - full[t]).max() / np.abs(full[t]).max() < 1e-5


def test_cache_bounded_and_accounted():
    """cache 内存与信号时长无关；账本 ≈ 270KB + LSTM state。"""
    from streaming import NNAECStreaming
    model = _load_model()
    st = NNAECStreaming(model, device="cpu")
    rng = np.random.default_rng(2)
    s1 = st.cache_size_bytes()
    for _ in range(50):                             # 1s
        st.process_chunk(rng.standard_normal(320).astype(np.float32),
                         rng.standard_normal(320).astype(np.float32))
    s2 = st.cache_size_bytes()
    assert s1 == s2, "cache 应随帧数滚动而非增长"
    assert 200_000 < s1 < 400_000, f"cache 账本 {s1} B 超出预期带（~270KB）"
    # 层内 5 帧 × 通道 × 频点 手算对照（保守带内即可）
    assert st.enc[0].cache.shape == (1, 4, 257, 5)
    assert st.dec[0].cache.shape == (1, 64, 33, 5)


@pytest.mark.xfail(strict=True, reason=(
    "R14 代码侧未修：exp/07_nn_aec/streaming.py:81 的 warmup 分支（t<2）直接 "
    "new_zeros 返回，跳过 act(bn(conv(...)))，而离线路径是 act(bn(...))。"
    "非平凡 BN（已训练 ckpt）下 bn(0)≠0，误差还经 LSTM 隐状态传播到全段。"
    "实测：前 640 样点 max|Δ|=5.54、其余 max|Δ|=0.069。"
    "修好 streaming.py 的 warmup 分支后本用例转 xpass，届时删除本标记。"))
def test_warmup_frames_match_offline_with_nontrivial_bn():
    """非平凡 BN 下，warmup 前 2 帧也必须与离线一致（v3a-020 / R14 的可观测化）。

    判别力来源见 _set_nontrivial_bn：未训练 BN 让 bn(0)=0，使"跳过 bn/act"与
    "act(bn(0))"数值相同，缺陷在既有 parity 用例里完全不可见。
    """
    rng = np.random.default_rng(5)
    n = 16000
    mic = (0.3 * rng.standard_normal(n)).astype(np.float32)
    far = (0.3 * rng.standard_normal(n)).astype(np.float32)
    model = _load_model()
    _set_nontrivial_bn(model)
    full = _offline_full_ref(model, mic, far)
    out = _run_streaming(model, mic, far, 320)
    assert len(out) == len(full)
    warm = 2 * 320                                    # deconv warmup = 前 2 帧
    err_warm = float(np.abs(out[:warm] - full[:warm]).max())
    assert err_warm < 1e-5, (
        f"warmup 前 2 帧与离线不一致：max|Δ|={err_warm:.3e}（实测 5.54）——"
        f"streaming.py warmup 分支跳过了 act(bn(conv(...)))")
    err_all = float(np.abs(out - full).max())
    assert err_all < 1e-5, f"非平凡 BN 下全段应与离线一致，实测 max|Δ|={err_all:.3e}"


@pytest.mark.skipif(not EVAL_V3.exists(), reason="需要本机 eval_v3 数据(CI 跳过)")
def test_real_data_end_to_end():
    """真实数据（eval_v3 2s 段）：流式输出与离线一致 + 全零静音段输出非 NaN。"""
    import soundfile as sf
    mic, _ = sf.read("D:/data/aec-synth/eval_v3/mic/fileid_0_dt.wav", dtype="float32")
    far, _ = sf.read("D:/data/aec-synth/eval_v3/farend/fileid_0_dt.wav", dtype="float32")
    n = 32000
    mic, far = mic[:n], far[:n]
    model = _load_model()
    ref = _offline_ref(model, mic, far)
    out = _run_streaming(model, mic, far, 320)
    got = _align(out, ref)
    err = np.abs(got - ref).max()
    assert err < 1e-5, f"真实数据对齐误差 {err:.2e}"
    assert np.isfinite(got).all()
