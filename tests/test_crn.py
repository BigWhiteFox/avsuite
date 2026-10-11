"""WP-03 单测：CRN 模型形状与开关 / 单 batch 过拟合 / train 冒烟 / 断点续训 / 数据切分。"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exp03_shim  # noqa: E402

CRN = exp03_shim.CRN
dataset = exp03_shim.dataset
train_mod = exp03_shim.train
STFT = exp03_shim.STFT


def test_model_shapes_all_configs():
    """4 组开关配置：输出波形长度 == 输入，中间谱 [B,257,T]；irm 掩码在 (0,1)。"""
    torch.manual_seed(0)
    wav = torch.randn(2, 16000)
    for mask_type in ("crm", "irm"):
        for rnn_type in ("blstm", "lstm"):
            m = CRN(hidden=16, rnn_type=rnn_type, mask_type=mask_type)
            enh, mask, noisy_spec, enh_spec = m(wav)
            assert enh.shape == wav.shape
            assert noisy_spec.shape == (2, 257, 63) and enh_spec.shape == (2, 257, 63)
            assert mask.shape[0] == 2
            if mask_type == "irm":
                assert bool((mask >= 0).all() and (mask <= 1).all())
    # 模型内 STFT 与 torch.stft **数值**对拍：同输入同谱。
    # 只断言 shape（历史写法）会漏掉布局/窗约定被换掉的回归——例如 Re/Im 通道
    # 互换、periodic 与非周期窗混用后，shape 依然正确但数值全错。
    spec = STFT().transform(wav)
    assert spec.shape == (2, 257, 63)
    ref = torch.stft(wav, 512, hop_length=256, win_length=512,
                     window=torch.hann_window(512, periodic=True, dtype=wav.dtype),
                     center=True, pad_mode="reflect", onesided=True, return_complex=True)
    torch.testing.assert_close(spec, ref, rtol=1e-5, atol=1e-5)


def test_bounded_mask_range():
    """bounded cRM 输出 Re/Im 每分量在 [-1.55, 1.55]（估计空间减半的设计点）。"""
    torch.manual_seed(0)
    m = CRN(hidden=8, bounded_mask=True)
    _, mask, _, enh = m(torch.randn(1, 8000))
    assert bool((mask.real.abs() <= 1.55 + 1e-6).all())
    assert bool((mask.imag.abs() <= 1.55 + 1e-6).all())
    assert enh.is_complex()


def test_skip_vs_noskip_and_param_count():
    """跳连开/关都前向通过；hidden 增大参数量单调涨（给 README 的参数表兜底）。"""
    wav = torch.randn(1, 8000)
    p_skip = sum(p.numel() for p in CRN(hidden=16, skip=True).parameters())
    p_noskip = sum(p.numel() for p in CRN(hidden=16, skip=False).parameters())
    assert p_skip > p_noskip
    p_small = sum(p.numel() for p in CRN(hidden=16).parameters())
    p_big = sum(p.numel() for p in CRN(hidden=64).parameters())
    assert p_big > p_small
    CRN(hidden=16, skip=False)(wav)


def test_gradient_flow():
    torch.manual_seed(0)
    m = CRN(hidden=8)
    clean = 0.3 * torch.randn(1, 8000)
    noisy = clean + 0.05 * torch.randn(1, 8000)
    loss = train_mod.make_loss_fn("crm")(m, {"noisy": noisy, "clean": clean})
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)


def test_overfit_single_batch():
    """单 batch 过拟合 sanity：小模型 + 固定合成对，loss 应显著下降（>40%）。"""
    torch.manual_seed(0)
    t = torch.arange(16000) / 16000
    clean = (0.3 * torch.sin(2 * torch.pi * 200 * t) +
             0.15 * torch.sin(2 * torch.pi * 600 * t)).repeat(1, 1)
    noise = 0.05 * torch.randn(1, 16000)
    batch = {"noisy": clean + noise, "clean": clean}
    m = CRN(hidden=16)
    loss_fn = train_mod.make_loss_fn("crm")
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for _i in range(60):
        loss = loss_fn(m, batch)
        if first is None:
            first = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.6 * first, f"overfit failed: {first:.4f} -> {float(loss):.4f}"


def test_dataset_and_speaker_split(tmp_path):
    """合成 wav fixture：配对、说话人不相交切分、Dataset 定长输出。"""
    clean_dir, noisy_dir = tmp_path / "clean", tmp_path / "noisy"
    clean_dir.mkdir(), noisy_dir.mkdir()
    rng = np.random.default_rng(0)
    names = [f"p226_{i:03d}.wav" for i in range(6)] + [f"p227_{i:03d}.wav" for i in range(6)]
    for n in names:
        y = (0.1 * rng.standard_normal(16000)).astype(np.float32)
        sf.write(clean_dir / n, y, 16000)
        sf.write(noisy_dir / n, (y + 0.01 * rng.standard_normal(16000)).astype(np.float32), 16000)
    pairs = dataset.list_pairs(clean_dir, noisy_dir)
    assert len(pairs) == 12
    tr, va = dataset.speaker_split(pairs, val_speakers=1)
    tr_spk = {dataset.speaker_of(n.name) for n, _ in tr}
    va_spk = {dataset.speaker_of(n.name) for n, _ in va}
    assert not (tr_spk & va_spk) and va_spk, "val 应非空且与 train 说话人不相交"
    ds = dataset.VoiceBankDataset(clean_dir, noisy_dir, segment_sec=1.0)
    item = ds[0]
    assert item["noisy"].shape == (16000,) and item["clean"].shape == (16000,)


def test_train_smoke_and_resume(tmp_path):
    """train.py --smoke 全链路（合成数据 1 epoch）+ --resume 续训从 epoch 2。"""
    out = tmp_path / "runs"
    base = [sys.executable, "-s", str(Path(train_mod.__file__)),
            "--smoke", "--hidden", "16", "--out", str(out)]
    r1 = subprocess.run(base, capture_output=True, text=True, timeout=600)
    assert r1.returncode == 0, r1.stderr[-2000:]
    ckpt = out / "crn_crm_blstm" / "last.ckpt"
    assert ckpt.exists()
    assert torch.load(ckpt, weights_only=False)["epoch"] == 1

    # 续训：不用 --smoke（它钉死 epochs=1，range(2,2) 为空跑不了第二轮）
    r2 = subprocess.run(
        [sys.executable, "-s", str(Path(train_mod.__file__)),
         "--synthetic", "--epochs", "2", "--max-train-batches", "2",
         "--num-workers", "0", "--hidden", "16", "--out", str(out),
         "--resume", str(ckpt)],
        capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0, r2.stderr[-2000:]
    assert torch.load(ckpt, weights_only=False)["epoch"] == 2, "resume 应从 epoch 2 继续"
    # 续训分支真的走到的**直接证据**（v3a-022 建议）：训练脚本在 resume 分支打印
    #   "[resume] epoch {start_ep}, best=..."
    # 注意：不能用"续训 vs 全新训练权重不同"作判据——smoke(1 epoch) 与 fresh(2 epochs)
    # 共享同一 seed/数据/超参，续训(=smoke 的 epoch1 + epoch2) 与 fresh(epoch1+epoch2)
    # 在确定性训练下**必然逐位相同**，该判据会把"resume 正常工作"误判为失败（实测 max|Δ|=0.0）。
    assert "[resume] epoch 2" in r2.stdout, (
        f"未见 resume 分支打印；stdout 尾部：{r2.stdout[-500:]}")

    # 补充的判别力：全新训练（不带 --resume）不应打印 resume 行
    fresh_out = tmp_path / "runs_fresh"
    r3 = subprocess.run(
        [sys.executable, "-s", str(Path(train_mod.__file__)),
         "--synthetic", "--epochs", "2", "--max-train-batches", "2",
         "--num-workers", "0", "--hidden", "16", "--out", str(fresh_out)],
        capture_output=True, text=True, timeout=600)
    assert r3.returncode == 0, r3.stderr[-2000:]
    assert "[resume]" not in r3.stdout
    resumed_sd = torch.load(ckpt, weights_only=False)["model"]
    fresh_sd = torch.load(fresh_out / "crn_crm_blstm" / "last.ckpt",
                          weights_only=False)["model"]
    assert set(resumed_sd) == set(fresh_sd), "两次训练的 state_dict 键应一致"
