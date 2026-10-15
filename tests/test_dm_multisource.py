"""P1X-2 单测：DM 多源池扩展（far 多语料 + 噪声多源加权 + flac/重采样 + 口径留痕）。

覆盖：
(a) **v5 单源口径不变**：与改动前固化的黄金基准（`tests/golden/dm_single_source_v5.json`）
    逐字段 + 逐位（三个张量的 sha256）一致 ⇒ 单源路径不多抽任何随机数；
(b) **多源加权采样**：按权重抽源（经验占比对账）+ 源内均匀；文件只来自各自源的清单；
(c) **噪声多源 CLI**：`--noise-source` 重复给 + `--noise-source-weights`，manifest 记
    权重（raw+归一）、`sampling_scheme`、每源 n/清单 sha256；
(d) **多语料格式**：`.flac` 与**相对路径**清单可加载；44.1k 源被重采样到 16k 并在
    `manifest.resample.applied_to` 留痕；
(e) **跨源互斥**：near/far 跨源同名 → 拒绝；noise 源与语音源重叠 → 拒绝；
(f) **口径单变量**：`scene_param_policy` 声明"只扩数据池"，场景参数（SER/SNR/引导/时延/
    soft-clip/jitter）与 v5 一字不改；
(g) **CLI fail fast**：权重个数不符 / 同时给 dir 与 source / 源规格缺字段 / 标签重复；
(h) **扩展池训练侧文件清单生成器**（`build_extension_file_lists.py`）：合成 LS/AISHELL
    目录树 → 清单行序 = reader 清单序 × 名字节序、只含清单 reader、holdout 零交集、
    `--verify` 复算逐行一致。
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
EXP07 = ROOT / "exp" / "07_nn_aec"
TRAIN07 = EXP07 / "train.py"
DM_SYNTH = ROOT / "scripts" / "dm_aec_synth.py"
EXT_LISTS = ROOT / "scripts" / "build_extension_file_lists.py"
GOLDEN = ROOT / "tests" / "golden" / "dm_single_source_v5.json"
SR = 16000

import dm_shim as SH  # noqa: E402

DM = SH.dm_module()
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------- fixtures
def _audio(path, n=20000, seed=0, sr=SR, ext=None, level=0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    y = (level * rng.standard_normal(int(n * sr / SR))).astype(np.float32)
    sf.write(path, y, sr, format=None if ext is None else ext.upper())
    return path


def _pool(root, names, seed0, sr=SR, ext="wav"):
    root.mkdir(parents=True, exist_ok=True)
    for i, nm in enumerate(names):
        _audio(root / nm, seed=seed0 + i, sr=sr, ext=ext)
    lst = root.parent / f"{root.name}.txt"
    lst.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
    return root, lst


def _multi_fixture(root):
    """三侧多源：near 单源；far 两源（wav + flac，名字可区分）；noise 两源。"""
    near, near_l = _pool(root / "near_dns", ("dnsA_1.wav", "dnsA_2.wav"), 10)
    far1, far1_l = _pool(root / "far_dns", ("dnsB_1.wav", "dnsB_2.wav"), 20)
    far2, far2_l = _pool(root / "far_ls", ("lsb_0001.flac", "lsb_0002.flac"), 30, ext="flac")
    noise1, noise1_l = _pool(root / "noise_dns", ("dnsq_1.wav", "dnsq_2.wav"), 40)
    noise2, noise2_l = _pool(root / "noise_musan", ("mus_0001.wav", "mus_0002.wav"), 50)
    rir = root / "rir"
    rir.mkdir(parents=True, exist_ok=True)
    h = np.zeros(512, dtype=np.float32)
    h[0] = 1.0
    sf.write(rir / "rir_0.wav", h, SR)
    return {"near": (near, near_l), "far": [(far1, far1_l), (far2, far2_l)],
            "noise": [(noise1, noise1_l), (noise2, noise2_l)], "rir": rir}


def _ds(fx, *, far_weights=("0.5", "0.5"), noise_weights=("0.5", "0.5"),
        epoch_len=6, seg_sec=2.0, seed=7, near_extra=(), far_extra=(), noise_extra=()):
    """按多源装配数据集（直调 build_pools_multi，与 CLI 同一实现）。"""
    def src(tag, extra=()):
        out = []
        if tag == "near":
            d, lst = fx["near"]
            out.append({"dir": d, "list": lst, "weight": 1.0, "label": d.name})
            w = ["1.0"]
        else:
            entries, w = fx[tag], list(extra or ())
            for (d, lst), wt in zip(entries, w):
                out.append({"dir": d, "list": lst, "weight": float(wt), "label": d.name})
        return out
    near_s = src("near", near_extra)
    far_s = src("far", far_weights)
    noise_s = src("noise", noise_weights)
    preset, plan_info = DM.load_plan(None, "train_v1_repro")
    near, far, noise, info = DM.build_pools_multi(near_s, far_s, noise_s)
    ds = DM.AecDynamicMix(near, far, noise, preset, plan_info=plan_info, pool_info=info,
                          epoch_len=epoch_len, seg_sec=seg_sec,
                          rir_files=sorted(fx["rir"].glob("*.wav")), seed=seed)
    return ds, info


def _run(argv, timeout=900):
    return subprocess.run(argv, capture_output=True, timeout=timeout,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


# ---------------------------------------------------------------- (a) v5 单源不变
def test_single_source_matches_v5_golden(tmp_path):
    """单源（v5 形态）与改动前固化基准逐字段 + 逐位一致（不多抽随机数）。"""
    g = json.loads(GOLDEN.read_text(encoding="utf-8"))
    fx = {"near": _pool(tmp_path / "near", ("readerA_1.wav", "readerA_2.wav", "readerA_3.wav"), 10),
          "far": [_pool(tmp_path / "far", ("readerB_1.wav", "readerB_2.wav"), 20)],
          "noise": [_pool(tmp_path / "noise", ("q_1.wav", "q_2.wav"), 30)], "rir": None}
    rir = tmp_path / "rir"
    rir.mkdir()
    h = np.zeros(512, dtype=np.float32)
    h[0] = 1.0
    sf.write(rir / "rir_0.wav", h, SR)
    fx["rir"] = rir
    preset, plan_info = DM.load_plan(None, "train_v1_repro")
    near, far, noise, info = DM.build_pools(
        fx["near"][0], fx["near"][1], fx["far"][0][0], fx["far"][0][1],
        fx["noise"][0][0], fx["noise"][0][1])
    ds = DM.AecDynamicMix(near, far, noise, preset, plan_info=plan_info, pool_info=info,
                          epoch_len=g["epoch_len"], seg_sec=g["seg_sec"],
                          rir_files=sorted(rir.glob("*.wav")), seed=g["seed"])
    assert len(ds) == len(g["items"])
    for ref in g["items"]:
        it = ds[ref["idx"]]
        audit = {k: it[k] for k in ("scenario", "ser_db", "snr_db", "leadin_sec",
                                    "soft_clip", "delay_ms", "name")}
        assert audit == ref["audit"], f"idx {ref['idx']} 审计字段漂移（单源 rng 流被改动）"
        if sys.platform == "win32":
            # 逐位指纹只在同平台有效（libm/SIMD ULP 差异跨平台必然漂移，见
            # test_delay_drift 的同款注释）；Windows 开发机保留最强断言
            for key in ("mic", "far", "near"):
                got = hashlib.sha256(it[key].numpy().tobytes()).hexdigest()
                assert got == ref["sha256"][key], f"idx {ref['idx']}/{key} 波形指纹漂移"
        else:
            # 非 Windows（CI）：波形数值统计特征断言（防整体内容级漂移）；
            # rng 流正确性已由上方 audit 字段逐字段锁定
            import math
            for key in ("mic", "far", "near"):
                w = it[key].numpy()
                rms = math.sqrt(float((w * w).mean()))
                assert rms > 1e-6, f"idx {ref['idx']}/{key} 波形塌陷"
                assert math.isfinite(float(np.abs(w).max())), f"idx {ref['idx']}/{key} 非有限值"
    prov = ds.provenance()
    assert prov["sampling_scheme"] == "single_source"
    assert prov["split_lists"]["near"]["n"] == 3            # v5 形状保持
    assert all(v is None for v in prov["source_groups"].values())


# ---------------------------------------------------------------- (b)(c) 多源加权
def test_multi_source_weighted_sampling_and_manifest(tmp_path):
    """多源 far：经验占比对账权重 + 文件只来自各自源 + manifest 留痕齐全。"""
    fx = _multi_fixture(tmp_path)
    ds, info = _ds(fx, far_weights=("0.75", "0.25"), epoch_len=600, seg_sec=0.5, seed=11)
    prov = ds.provenance()
    assert prov["sampling_scheme"] == "weighted_source_then_uniform_intra"
    srcs = {e["label"]: e for e in prov["sources"]["far"]}
    assert set(srcs) == {"far_dns", "far_ls"}
    assert (srcs["far_dns"]["weight_raw"], srcs["far_dns"]["weight_norm"]) == (0.75, 0.75)
    assert abs(srcs["far_ls"]["weight_norm"] - 0.25) < 1e-12
    assert srcs["far_ls"]["n_files"] == 2 and len(srcs["far_ls"]["list_sha256"]) == 64
    assert prov["split_lists"]["far"]["n"] == 4 and "sha256_combined" in prov["split_lists"]["far"]
    assert prov["scene_param_policy"].startswith("inherit_v5_dt_plan_v1")
    # 源分布对账：直接对采样函数做统计（条目里不含源信息）
    dns, ls = srcs["far_dns"], srcs["far_ls"]
    rng = np.random.default_rng(3)
    files = ds.far_files
    labels = []
    for _ in range(2000):
        f = ds._pick_file(rng, "far", files)
        labels.append("far_dns" if "dnsB" in f.name else "far_ls")
    share = labels.count("far_dns") / len(labels)
    assert abs(share - 0.75) < 0.05, f"源权重未生效：far_dns 占比 {share:.3f}"
    # 源内均匀（两家各 2 个文件都应出现）
    got = {f.name for f in (ds._pick_file(rng, "far", files) for _ in range(50))}
    assert got == {"dnsB_1.wav", "dnsB_2.wav", "lsb_0001.flac", "lsb_0002.flac"}
    assert dns["probe_sr"] == 16000 and ls["probe_sr"] == 16000


def test_multi_source_noise_cli_weights_and_resample(tmp_path):
    """噪声多源 + `--noise-source-weights`（CLI）+ 44.1k 源重采样留痕。"""
    fx = _multi_fixture(tmp_path)
    # 把第二个噪声源换成 44.1k，验证重采样路径
    fx["noise"][1] = _pool(tmp_path / "noise_musan44", ("mus44_1.wav", "mus44_2.wav"), 60,
                           sr=44100)
    out = tmp_path / "out"
    argv = [sys.executable, "-s", str(TRAIN07), "--dynamic-mixing", "--device", "cpu",
            "--num-workers", "0", "--batch-size", "2", "--dm-epoch-len", "4",
            "--dm-seg-sec", "0.5", "--epochs", "1",
            "--near-dir", str(fx["near"][0]), "--near-list", str(fx["near"][1]),
            "--far-source", f"{fx['far'][0][0]}|{fx['far'][0][1]}|0.6|dns_far",
            "--far-source", f"{fx['far'][1][0]}|{fx['far'][1][1]}|0.4|ls_far",
            "--noise-source", f"{fx['noise'][0][0]}|{fx['noise'][0][1]}||dns_noise",
            "--noise-source", f"{fx['noise'][1][0]}|{fx['noise'][1][1]}||musan44",
            "--noise-source-weights", "0.7,0.3",
            "--rir-dir", str(fx["rir"]), "--val", str(_val_dir(tmp_path / "val")),
            "--out", str(out)]
    rr = _run(argv)
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-3000:]
    man = json.loads((out / "nnaec_g0.5_j10_dm" / "train_manifest.json")
                     .read_text(encoding="utf-8"))
    ns = {e["label"]: e for e in man["data"]["sources"]["noise"]}
    assert abs(ns["dns_noise"]["weight_norm"] - 0.7) < 1e-9
    assert abs(ns["musan44"]["weight_norm"] - 0.3) < 1e-9
    assert ns["musan44"]["probe_sr"] == 44100
    assert man["data"]["resample"]["target_sr"] == 16000
    assert man["data"]["resample"]["applied_to"] == ["musan44"]
    assert man["data"]["sampling_scheme"] == "weighted_source_then_uniform_intra"
    fs = {e["label"]: e for e in man["data"]["sources"]["far"]}
    assert abs(fs["dns_far"]["weight_norm"] - 0.6) < 1e-9
    hist = json.loads((out / "nnaec_g0.5_j10_dm" / "history.json").read_text(encoding="utf-8"))
    assert (hist["epoch_len"], hist["steps_per_epoch"], hist["step"]) == (4, 2, 2)


def _val_dir(root, n=2, seg_sec=0.5, seed=5):
    """最小静态 val（aec_synth 结构）。"""
    rng = np.random.default_rng(seed)
    for sub in ("farend", "nearend", "mic"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        far = (0.3 * rng.standard_normal(int(seg_sec * SR))).astype(np.float32)
        near = (0.05 * rng.standard_normal(int(seg_sec * SR))).astype(np.float32)
        sf.write(root / "farend" / f"v{i}.wav", far, SR)
        sf.write(root / "nearend" / f"v{i}.wav", near, SR)
        sf.write(root / "mic" / f"v{i}.wav", near + 0.7 * far, SR)
        rows.append({"name": f"v{i}", "scenario": "dt", "rir_src": "dir", "near_file": "n",
                     "far_file": "f", "noise_file": "q", "delay_ms": "10.0", "ser_db": "5.0",
                     "snr_db": "20.0", "soft_clip": "0", "leadin_sec": "0.0"})
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root


def test_resample_44k_source_to_16k(tmp_path):
    """44.1k 源读入后被重采样到 16k（长度按比例、频谱峰值对应原频率）。"""
    p = tmp_path / "tone44.wav"
    sr, freq, sec = 44100, 1000.0, 0.5
    t = np.arange(int(sr * sec)) / sr
    sf.write(p, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    y = DM.load_wav_sr(p, SR)
    assert abs(len(y) / SR - sec) < 0.01
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    peak_hz = np.fft.rfftfreq(len(y), 1 / SR)[int(np.argmax(spec))]
    assert abs(peak_hz - freq) < 20, f"重采样后频率漂移到 {peak_hz:.0f} Hz"
    y16 = DM.load_wav_sr(_audio(tmp_path / "t16.wav", n=8000, seed=1), SR)
    assert len(y16) == 8000                                   # 16k 源原样返回


# ---------------------------------------------------------------- (e)(g) 互斥与 fail fast
def test_cross_source_mutex_and_cli_fail_fast(tmp_path):
    """跨源同名 → 拒绝；噪声源与语音源重叠 → 拒绝；权重/规格/标签问题 → fail fast。"""
    fx = _multi_fixture(tmp_path)
    preset, plan_info = DM.load_plan(None, "train_v1_repro")
    # near 源与 far 源出现同名文件（跨目录复制）→ 跨源互斥拦截
    (fx["far"][0][0] / "dnsA_1.wav").write_bytes((fx["near"][0] / "dnsA_1.wav").read_bytes())
    with open(fx["far"][0][1], "a", encoding="utf-8") as f:
        f.write("dnsA_1.wav\n")
    with pytest.raises(Exception, match="utterance 主名|复用同一文件"):
        _ds(fx)
    # 噪声源指到语音池 → 拒绝
    d, lst = fx["near"]
    with pytest.raises(Exception, match="语音清单与噪声清单|复用同一文件"):
        DM.build_pools_multi(
            [{"dir": fx["near"][0], "list": fx["near"][1], "weight": 1.0, "label": "n"}],
            [{"dir": fx["far"][0][0], "list": fx["far"][0][1], "weight": 1.0, "label": "f"}],
            [{"dir": d, "list": lst, "weight": 1.0, "label": "q"}])
    # 源规格缺字段 / 权重 <= 0
    with pytest.raises(DM.DMPoolError, match="源规格需形如"):
        DM.parse_source_spec("only-one-field")
    with pytest.raises(DM.DMPoolError, match="权重需 > 0"):
        DM.parse_source_spec("D:/x|L.txt|0|lab")
    # 权重个数不符 / dir 与 source 同时给 / 标签重复（CLI 层）
    fx2 = _multi_fixture(tmp_path / "f2")
    base = [sys.executable, "-s", str(TRAIN07), "--dynamic-mixing", "--device", "cpu",
            "--epochs", "1", "--dm-epoch-len", "2", "--batch-size", "2",
            "--near-dir", str(fx2["near"][0]), "--near-list", str(fx2["near"][1]),
            "--far-source", f"{fx2['far'][0][0]}|{fx2['far'][0][1]}|0.5|a",
            "--far-source", f"{fx2['far'][1][0]}|{fx2['far'][1][1]}|0.5|b",
            "--noise-dir", str(fx2["noise"][0][0]), "--noise-list", str(fx2["noise"][0][1]),
            "--rir-dir", str(fx2["rir"]), "--val", str(_val_dir(tmp_path / "val2")),
            "--out", str(tmp_path / "o")]
    rr = _run(base + ["--far-source-weights", "0.5"])
    assert rr.returncode != 0 and "一一对应" in rr.stderr.decode("utf-8", "replace")
    rr = _run(base + ["--far-dir", str(fx2["far"][0][0]), "--far-list",
                      str(fx2["far"][0][1])])
    assert rr.returncode != 0 and "不能同时给" in rr.stderr.decode("utf-8", "replace")
    rr = _run([(a[:-1] + "same") if (a.endswith("|a") or a.endswith("|b")) else a
               for a in base])
    assert rr.returncode != 0 and "标签重复" in rr.stderr.decode("utf-8", "replace")


def test_dm_aec_synth_multisource_dryrun(tmp_path):
    """冒烟/吞吐预检 CLI 也支持多源（D1 预检要按 v5x 池口径测）。"""
    fx = _multi_fixture(tmp_path)
    rr = _run([sys.executable, "-s", str(DM_SYNTH), "--num", "4", "--dry-run",
               "--seg-sec", "0.5", "--seed", "3",
               "--near-dir", str(fx["near"][0]), "--near-list", str(fx["near"][1]),
               "--far-source", f"{fx['far'][0][0]}|{fx['far'][0][1]}|0.7|dns",
               "--far-source", f"{fx['far'][1][0]}|{fx['far'][1][1]}|0.3|ls",
               "--noise-dir", str(fx["noise"][0][0]), "--noise-list", str(fx["noise"][0][1]),
               "--rir-dir", str(fx["rir"])])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    assert "样本/秒" in rr.stdout.decode("utf-8")


# ---------------------------------------------------------------- (h) 扩展池文件清单生成器
def _ls_tree(root):
    """合成 LibriSpeech 式目录树：<subset>/<reader>/<chapter>/<file>.flac。

    特意含"chapter 目录名包含 holdout reader 数字"的坑：`100/121669/...` 里的 `121669`
    含子串 `669`（train-eng 实测的误报源）。reader 字段是第二段（`100`），不是 chapter。
    """
    for sub, reader, ch, name in (("train-clean-100", "100", "100-1", "100-1-0000.flac"),
                                  ("train-clean-100", "100", "100-1", "100-1-0001.flac"),
                                  ("train-clean-360", "100", "121669", "100-121669-0000.flac"),
                                  ("train-clean-360", "1025", "1025-1", "1025-1-0000.flac"),
                                  ("train-clean-360", "9999", "9999-1", "9999-1-0000.flac"),
                                  # 真 holdout reader 的文件（**在池里存在**，用于真泄漏用例）
                                  ("train-clean-360", "669", "669-1", "669-1-0000.flac")):
        _audio(root / sub / reader / ch / name, seed=abs(hash(name)) % 1000, ext="flac")
    return root


def _ext_lists_module():
    """把生成器当模块加载（测 reader_of_rel / LAYOUT，不必起子进程）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_extension_file_lists", EXT_LISTS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_extension_file_lists"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_extension_file_lists_generator_and_verify(tmp_path):
    """生成器：行序 = reader 清单序 × 名字节序；只含清单 reader；--verify 复算一致。"""
    ext = _ext_lists_module()
    root = _ls_tree(tmp_path / "librispeech")
    readers = tmp_path / "readers_train70.txt"
    readers.write_text("1025\n100\n", encoding="utf-8")           # 故意非字典序
    holdout = tmp_path / "readers_holdout15.txt"
    holdout.write_text("669\n9999\n", encoding="utf-8")           # 669 是 chapter "121669" 的子串
    out = tmp_path / "ext_ls_far_train70_files.txt"
    rr = _run([sys.executable, "-s", str(EXT_LISTS), "--corpus", "librispeech",
               "--root", str(root), "--readers", str(readers),
               "--holdout-readers", str(holdout), "--out", str(out)])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln]
    assert lines == ["train-clean-360/1025/1025-1/1025-1-0000.flac",
                     "train-clean-100/100/100-1/100-1-0000.flac",
                     "train-clean-100/100/100-1/100-1-0001.flac",
                     "train-clean-360/100/121669/100-121669-0000.flac"], \
        "行序 = reader 清单序 × 名字节序"
    assert not any("9999" in ln for ln in lines), "holdout reader 的文件不得入训练清单"
    # **回归（train-eng 实锤的误报）**：chapter 名含 holdout 数字 669 ≠ 泄漏，不得拦下生成
    assert any("121669" in ln for ln in lines)
    assert "隔离断言通过" in rr.stdout.decode("utf-8")
    # 判据必须是**路径字段**（不是子串）
    assert ext.LAYOUT["librispeech"]["reader_index"] == 1
    assert ext.reader_of_rel("train-clean-360/100/121669/100-121669-0000.flac",
                             ext.LAYOUT["librispeech"]) == "100"
    assert ext.reader_of_rel("train-clean-360/9999/9999-1/9999-1-0000.flac",
                             ext.LAYOUT["librispeech"]) == "9999"
    # --verify 复算
    rr2 = _run([sys.executable, "-s", str(EXT_LISTS), "--corpus", "librispeech",
                "--root", str(root), "--readers", str(readers), "--out", str(out),
                "--verify"])
    assert rr2.returncode == 0 and "verify PASS" in rr2.stdout.decode("utf-8")
    # 篡改一行 → verify 必须拦住
    out.write_text(out.read_text(encoding="utf-8").replace("100-1-0001", "100-1-0002"),
                   encoding="utf-8")
    rr3 = _run([sys.executable, "-s", str(EXT_LISTS), "--corpus", "librispeech",
                "--root", str(root), "--readers", str(readers), "--out", str(out),
                "--verify"])
    assert rr3.returncode != 0 and "不一致" in rr3.stderr.decode("utf-8", "replace")


def test_extension_file_lists_leak_check_is_reader_field_not_substring(tmp_path):
    """t25 回归（train-eng 实锤）：holdout id `669` ⊂ chapter 名 `121669` ⇒ **不得误报**，
    同时**真泄漏仍要抓住**，且 `--root` 给错层要有候选提示。"""
    ext = _ext_lists_module()
    layout = ext.LAYOUT["librispeech"]
    files = ["train-clean-360/100/121669/100-121669-0000.flac",     # reader=100（正常文件）
             "train-clean-100/100/100-1/100-1-0000.flac",
             "train-clean-360/669/669-1/669-1-0000.flac"]            # 真 holdout 行
    # 真泄漏仍能抓（按 reader 字段相等）
    assert ext.leaked_files(files, {"669"}, layout) == \
        ["train-clean-360/669/669-1/669-1-0000.flac"]
    # 不误报：子串式命中（121669 含 669）不再算泄漏
    assert ext.leaked_files(files, {"9999", "1216"}, layout) == []
    # 用例有牙：旧子串判据在本 fixture 上确实会误报
    old = [f for f in files if any(r in f for r in {"669"})]
    assert old != ext.leaked_files(files, {"669"}, layout) and len(old) == 2
    # CLI 层真泄漏（reader 清单 ∩ holdout 非空）必须非零退出
    root = _ls_tree(tmp_path / "ls")
    readers = tmp_path / "r.txt"
    readers.write_text("100\n669\n", encoding="utf-8")
    hold = tmp_path / "h.txt"
    hold.write_text("669\n", encoding="utf-8")
    rr = _run([sys.executable, "-s", str(EXT_LISTS), "--corpus", "librispeech",
               "--root", str(root), "--readers", str(readers), "--holdout-readers",
               str(hold), "--out", str(tmp_path / "o.txt")])
    assert rr.returncode != 0 and "有交集" in rr.stderr.decode("utf-8", "replace")
    # ④ --root 给错层（多一层 LibriSpeech/）：错误信息给出候选目录
    shallow = tmp_path / "shallow"
    _ls_tree(shallow / "LibriSpeech")
    r2 = tmp_path / "r2.txt"
    r2.write_text("100\n1025\n", encoding="utf-8")
    rr2 = _run([sys.executable, "-s", str(EXT_LISTS), "--corpus", "librispeech",
                "--root", str(shallow), "--readers", str(r2),
                "--out", str(tmp_path / "o2.txt")])
    err = rr2.stderr.decode("utf-8", "replace")
    assert rr2.returncode != 0 and "提示" in err and "LibriSpeech" in err, err
    assert "命中 2/2" in err, f"候选提示要给出命中比例：{err}"
