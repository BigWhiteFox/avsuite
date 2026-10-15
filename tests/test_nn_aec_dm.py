"""P1-3 单测：DM（动态混合）训练管线 —— 采样对账 / 清单互斥 / 断点续训三件套 / 口径换算。

覆盖（全部 CPU 可跑，无 D:/data、无网络）：

(a) **DM 采样对账清单**（0-5 接入）：场景配比 ≈ preset、DT 的 SER/SNR 落在清单区间且均值
    接近区间中点、引导时长等于清单点分布、DT 引导段近端恒零（不是"整段被清零"）；
(b) **清单互斥**（0-2/0-3 纪律）：near/far 同名或同文件 → 报错；噪声清单与语音清单重叠
    → 报错；清单成员不在池内 / 池内同名歧义 / 空清单 → 报错；同一池目录 + 两份互斥名单
    （合法切分）→ 通过；
(c) **口径换算**（附录B 方案 a）：240ep/bs16 → 170 步/ep、epoch_len 2720、总 40800（−30 步）；
    v4 参照 40830 与本地 v4 产物 ckpt 的 step 一致（无产物则 skip）；LR/早停缺省在 DM 下
    为 patience 20 / 关早停、非 DM 下保持现状 3 / 8；
(d) **断点续训三件套兼容**：best.ckpt 与 last.ckpt 键集相同（model/opt/sched/epoch/best/bad/
    step/val/cfg）、`--resume` 从 last/best 都能继续、epoch/step 递增、history.json 记录
    实际 epoch_len 与总步数；**从非 DM 谱系 resume 被拒**（阶段0 §2.3a：对照必须从零起）；
(e) **默认行为不变**：不开 `--dynamic-mixing` 时 ckpt cfg 不含任何新键（NEW_ARG_KEYS /
    train_manifest）、不产出 train_manifest.json、early-stop 仍为 8；
(f) **端到端冒烟**（epoch_len=10）：`train.py --dynamic-mixing` 与 `scripts/dm_aec_synth.py`
    两条 CLI 都跑通并落盘留痕。

注：train.py 的 DM 分支用子进程跑（与 test_nn_aec.py 的既有做法一致）；纯函数与数据集
在进程内直调。dm_dataset 用 importlib 定点加载（避免把 exp/07_nn_aec 永久挂上 sys.path，
那会遮蔽 03/04 的裸名 model/train/dataset —— 见 tests/conftest.py §3.5）。
"""
import csv
import importlib.util
import json
import locale
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp" / "07_nn_aec"
TRAIN = EXP / "train.py"
DM_SYNTH = ROOT / "scripts" / "dm_aec_synth.py"
SR = 16000

scripts_dir = str(ROOT / "scripts")


def _load(name: str, path: Path):
    """定点加载（临时把 exp 目录放上 sys.path，加载后立刻摘掉）。"""
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        if str(path.parent) in sys.path:
            sys.path.remove(str(path.parent))
    return mod


import dm_shim as SH  # noqa: E402

DM = SH.dm_module()
DS = _load("nnaec_dataset_cmp", EXP / "dataset.py")     # 仅用于 jitter 实现对账
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)


# ---------------------------------------------------------------- fixture
def _wav(path, n=20000, seed=0, level=0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sf.write(path, (level * rng.standard_normal(n)).astype(np.float32), SR)


NEAR_NAMES = ("readerA_1.wav", "readerA_2.wav", "readerA_3.wav")
FAR_NAMES = ("readerB_1.wav", "readerB_2.wav")
NOISE_NAMES = ("q_1.wav", "q_2.wav")


def _fixture(root, *, near_dir=None, far_dir=None, noise_dir=None):
    """三池 + 三清单 + RIR；同名清单成员保证 near/far 主名互斥（合法切分）。"""
    nd = Path(near_dir) if near_dir else root / "near"
    fd = Path(far_dir) if far_dir else root / "far"
    qd = Path(noise_dir) if noise_dir else root / "noise"
    for d, names, base in ((nd, NEAR_NAMES, 1), (fd, FAR_NAMES, 2), (qd, NOISE_NAMES, 3)):
        for i, nm in enumerate(names):
            _wav(d / nm, seed=base * 10 + i)
    nl, fl, ql = root / "near.txt", root / "far.txt", root / "noise.txt"
    nl.write_text("\n".join(NEAR_NAMES) + "\n", encoding="utf-8")
    fl.write_text("\n".join(FAR_NAMES) + "\n", encoding="utf-8")
    ql.write_text("\n".join(NOISE_NAMES) + "\n", encoding="utf-8")
    rir = root / "rir"
    rir.mkdir(parents=True, exist_ok=True)
    h = np.zeros(512, dtype=np.float32)
    h[0] = 1.0
    sf.write(rir / "rir_0.wav", h, SR)
    return {"near_dir": nd, "far_dir": fd, "noise_dir": qd, "near_list": nl,
            "far_list": fl, "noise_list": ql, "rir": rir}


def _make_val(root, n=3, seg_sec=1.0, seed=5):
    """最小静态 val 集（AecDataset 只吃 manifest 的 name/scenario）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for sub in ("farend", "nearend", "mic"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        y = (0.2 * rng.standard_normal(int(seg_sec * SR))).astype(np.float32)
        for sub in ("farend", "nearend", "mic"):
            sf.write(root / sub / f"v{i}.wav", y, SR)
        rows.append({"name": f"v{i}", "scenario": "dt", "rir_src": "dir", "near_file": "n",
                     "far_file": "f", "noise_file": "q", "delay_ms": "10.0", "ser_db": "5.0",
                     "snr_db": "20.0", "soft_clip": "0", "leadin_sec": "0.0"})
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root


def _dm_dataset(fx, *, epoch_len=8, seg_sec=2.0, seed=7, plan=None, preset="train_v1_repro"):
    preset_d, plan_info = DM.load_plan(plan, preset)
    near, far, noise, pool_info = DM.build_pools(
        fx["near_dir"], fx["near_list"], fx["far_dir"], fx["far_list"],
        fx["noise_dir"], fx["noise_list"])
    return DM.AecDynamicMix(near, far, noise, preset_d, plan_info=plan_info,
                            pool_info=pool_info, epoch_len=epoch_len, seg_sec=seg_sec,
                            rir_files=sorted(fx["rir"].glob("*.wav")), seed=seed)


def _run(argv, timeout=900):
    return subprocess.run(argv, capture_output=True, timeout=timeout,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def _train_dm_argv(fx, out, val, extra=()):
    return [sys.executable, "-s", str(TRAIN), "--dynamic-mixing", "--device", "cpu",
            "--num-workers", "0", "--batch-size", "4", "--dm-epoch-len", "10",
            "--near-dir", str(fx["near_dir"]), "--near-list", str(fx["near_list"]),
            "--far-dir", str(fx["far_dir"]), "--far-list", str(fx["far_list"]),
            "--noise-dir", str(fx["noise_dir"]), "--noise-list", str(fx["noise_list"]),
            "--rir-dir", str(fx["rir"]), "--val", str(val), "--out", str(out), *extra]


# ---------------------------------------------------------------- (a) 采样对账
def test_dm_scene_ratio_and_dt_distribution_match_plan(tmp_path):
    """对账清单：场景配比、DT 的 SER/SNR 区间与均值、引导点分布、DT 引导段语义。"""
    fx = _fixture(tmp_path)
    ds = _dm_dataset(fx, epoch_len=400, seg_sec=2.0, seed=11)
    items = [ds[i] for i in range(len(ds))]
    from collections import Counter
    frac = {k: v / len(items) for k, v in Counter(it["scenario"] for it in items).items()}
    for scene, target in ds.preset["scenarios"].items():
        assert abs(frac.get(scene, 0.0) - target) < 0.08, (scene, frac, target)
    dt = [it for it in items if it["scenario"] == "dt"]
    assert len(dt) > 80, f"DT 样本过少（{len(dt)}），分布断言不可靠"
    ser = np.array([it["ser_db"] for it in dt])
    snr = np.array([it["snr_db"] for it in dt])
    # 清单 train_v1_repro：SER 0..10、SNR 15..30、引导 1.5s（点分布）
    assert ser.min() >= 0.0 and ser.max() <= 10.0
    assert snr.min() >= 15.0 and snr.max() <= 30.0
    assert abs(ser.mean() - 5.0) < 0.6 and abs(snr.mean() - 22.5) < 0.8
    assert {it["leadin_sec"] for it in dt} == {1.5}
    assert all(it["leadin_sec"] == 0.0 for it in items if it["scenario"] != "dt")
    # 引导段纯远端：引导段恒 0、引导段之后必须有近端（防"整段清零退化 FEST"）
    lead = int(1.5 * SR)
    for it in dt[:20]:
        near = it["near"].numpy()
        assert len(near) == 2 * SR and np.abs(near[:lead]).max() == 0
        assert np.abs(near[lead:]).max() > 1e-5
    # 池选择落在清单成员内（防取到池外文件）
    assert all(it["mic"].shape[-1] == 2 * SR for it in items)


def test_dm_plan_provenance_and_segments(tmp_path):
    """清单版本号/sha256/池与清单计数进 provenance；采样参数与 v4 口径一致。"""
    fx = _fixture(tmp_path)
    ds = _dm_dataset(fx, epoch_len=4)
    p = ds.provenance()
    assert p["dt_plan"]["version"] == "dt_plan_v1"
    assert p["dt_plan"]["preset"] == "train_v1_repro"
    assert len(p["dt_plan"]["sha256"]) == 64
    assert p["mode"] == "dynamic_mixing" and p["seed_mode"] == "per_idx_deterministic"
    assert p["n_files"] == {"near": 3, "far": 2, "noise": 2, "rir": 1}
    assert p["mutex"]["checked"] is True
    assert p["split_lists"]["near"]["n"] == 3
    assert p["split_lists"]["near"]["sha256"] == DM._file_sha256(fx["near_list"])
    assert p["rir_src"] == "dir" and p["rir_max_ms"] == 100.0
    assert p["delay_ms_range"] == [10.0, 80.0] and p["soft_clip_prob"] == 0.15
    assert p["jitter_ms"] == 10.0
    # val 侧不在 DM 数据集里（口径由 train.py 的静态 --val 负责）——此处只钉住供给侧字段
    assert set(p) >= {"epoch_len", "seg_sec", "pools", "split_lists", "mutex"}


def test_dm_determinism_by_seed_and_entropy_mode(tmp_path):
    """固定 seed：逐 (seed, idx) 确定（位级）；seed=None：每次取样不同（训练期多样性）。"""
    fx = _fixture(tmp_path)
    ds = _dm_dataset(fx, epoch_len=6, seed=99)
    a, b = ds[3], ds[3]
    assert np.array_equal(a["mic"].numpy(), b["mic"].numpy())
    assert np.array_equal(a["far"].numpy(), b["far"].numpy())
    assert not np.array_equal(a["mic"].numpy(), ds[4]["mic"].numpy())
    ds2 = _dm_dataset(fx, epoch_len=6, seed=99)
    assert np.array_equal(ds[3]["mic"].numpy(), ds2[3]["mic"].numpy()), "同 seed 不同实例应逐位一致"
    ent = _dm_dataset(fx, epoch_len=2, seed=None)
    assert not np.array_equal(ent[0]["mic"].numpy(), ent[0]["mic"].numpy())
    assert ent.provenance()["seed_mode"] == "entropy"


def test_dm_item_contract_and_picklable(tmp_path):
    """DataLoader 契约（collate 需要的键）与 worker 可 pickle（Windows spawn 前提）。"""
    fx = _fixture(tmp_path)
    ds = _dm_dataset(fx, epoch_len=4)
    it = ds[0]
    assert set(it) >= {"mic", "far", "near", "name", "scenario"}
    assert it["mic"].dtype == torch.float32
    assert it["far"].shape == it["near"].shape == it["mic"].shape
    blob = pickle.dumps(ds)
    sys.path.insert(0, str(EXP))
    try:
        ds_back = pickle.loads(blob)
    finally:
        sys.path.remove(str(EXP))
    assert ds_back.epoch_len == ds.epoch_len and ds_back.seed == ds.seed
    assert np.array_equal(ds_back[2]["near"].numpy(), ds[2]["near"].numpy())


def test_dm_jitter_matches_static_dataset_shift():
    """DM 的 far 平移与 dataset.AecDataset 的 _shift 同实现（防两处漂移）。"""
    x = np.arange(20, dtype=np.float32)
    for k in (-7, 0, 3):
        assert np.array_equal(DM._shift(x, k), DS._shift(x, k))


# ---------------------------------------------------------------- (b) 清单互斥
def test_dm_build_pools_ok_and_same_dir_disjoint_lists(tmp_path):
    """合法：主名互斥；且**同一池目录 + 两份互斥名单**是合法切分（不该因目录相同而误报）。"""
    fx = _fixture(tmp_path)
    near, far, noise, info = DM.build_pools(
        fx["near_dir"], fx["near_list"], fx["far_dir"], fx["far_list"],
        fx["noise_dir"], fx["noise_list"])
    assert (len(near), len(far), len(noise)) == (3, 2, 2)
    assert info["mutex"]["near_far_conflicts"] == []
    # 极简变体：near/far 同一目录、名单互斥
    same = tmp_path / "same"
    for i, nm in enumerate(NEAR_NAMES + FAR_NAMES):
        _wav(same / nm, seed=40 + i)
    nl, fl = tmp_path / "s_near.txt", tmp_path / "s_far.txt"
    nl.write_text("\n".join(NEAR_NAMES) + "\n", encoding="utf-8")
    fl.write_text("\n".join(FAR_NAMES) + "\n", encoding="utf-8")
    near2, far2, _, _ = DM.build_pools(same, nl, same, fl, fx["noise_dir"], fx["noise_list"])
    assert len(near2) == 3 and len(far2) == 2


def test_dm_build_pools_rejects_same_stem(tmp_path):
    """near/far 名单出现同一 utterance 主名（跨目录复制）→ 入口拒绝（0-3 判据）。"""
    fx = _fixture(tmp_path)
    (fx["far_dir"] / NEAR_NAMES[0]).write_bytes((fx["near_dir"] / NEAR_NAMES[0]).read_bytes())
    with open(fx["far_list"], "a", encoding="utf-8") as f:
        f.write(NEAR_NAMES[0] + "\n")
    with pytest.raises(Exception, match="utterance 主名"):
        DM.build_pools(fx["near_dir"], fx["near_list"], fx["far_dir"], fx["far_list"],
                       fx["noise_dir"], fx["noise_list"])


def test_dm_build_pools_rejects_same_file_identity(tmp_path):
    """同一文件（硬链接）出现在两份名单 → 入口拒绝（文件同一性判据）。"""
    fx = _fixture(tmp_path)
    link = fx["far_dir"] / "link_to_near.wav"
    try:
        os.link(fx["near_dir"] / NEAR_NAMES[1], link)
    except (OSError, NotImplementedError):
        pytest.skip("本文件系统不支持硬链接")
    with open(fx["far_list"], "a", encoding="utf-8") as f:
        f.write("link_to_near.wav\n")
    with pytest.raises(Exception, match="复用同一文件"):
        DM.build_pools(fx["near_dir"], fx["near_list"], fx["far_dir"], fx["far_list"],
                       fx["noise_dir"], fx["noise_list"])


def test_dm_build_pools_rejects_noise_overlapping_speech(tmp_path):
    """噪声清单里混进语音池的**同一个文件** → 入口拒绝（噪声池指错 = 训练侧双用）。"""
    fx = _fixture(tmp_path)
    link = fx["noise_dir"] / "speech_copy.wav"
    try:
        os.link(fx["near_dir"] / NEAR_NAMES[0], link)
    except (OSError, NotImplementedError):
        pytest.skip("本文件系统不支持硬链接")
    with open(fx["noise_list"], "a", encoding="utf-8") as f:
        f.write("speech_copy.wav\n")
    with pytest.raises(Exception, match="语音清单与噪声清单"):
        DM.build_pools(fx["near_dir"], fx["near_list"], fx["far_dir"], fx["far_list"],
                       fx["noise_dir"], fx["noise_list"])
    assert link.exists()


def test_dm_pool_loading_fail_fast(tmp_path):
    """清单缺成员 / 空清单 / 池内同名歧义 / 池目录不存在：全部 fail fast（不静默取子集）。"""
    fx = _fixture(tmp_path)
    miss = tmp_path / "miss.txt"
    miss.write_text("not_in_pool.wav\n", encoding="utf-8")
    with pytest.raises(DM.DMPoolError, match="不在池"):
        DM.load_pool(fx["near_dir"], miss)
    empty = tmp_path / "empty.txt"
    empty.write_text("\n# 只有注释\n", encoding="utf-8")
    with pytest.raises(DM.DMPoolError, match="为空"):
        DM.read_split_list(empty)
    dup = tmp_path / "dup"
    _wav(dup / "a" / "same.wav", seed=1)
    _wav(dup / "b" / "same.wav", seed=2)
    lst = tmp_path / "dup.txt"
    lst.write_text("same.wav\n", encoding="utf-8")
    with pytest.raises(DM.DMPoolError, match="同名音频"):   # P1X-2：扩展名含 .flac，措辞改"音频"
        DM.load_pool(dup, lst)
    with pytest.raises(DM.DMPoolError, match="池目录不存在"):
        DM.load_pool(tmp_path / "nope", fx["near_list"])


# ---------------------------------------------------------------- (c) 口径换算
def test_resolve_epoch_scheme_matches_v4_budget():
    """方案(a)：240ep/bs16 → 170 步/ep、epoch_len 2720、总 40800 步（v4 40830，−0.073%）。"""
    sch = DM.resolve_epoch_scheme(240, 16)
    assert sch["scheme"] == "a" and sch["epoch_len_source"] == "derived_scheme_a"
    assert (sch["steps_per_epoch"], sch["epoch_len"], sch["achieved_steps"]) == (170, 2720, 40800)
    assert sch["target_steps"] == DM.V4_REF_STEPS == 40830
    assert sch["delta_steps"] == -30 and abs(sch["delta_frac"]) < 1e-3
    assert sch["v4_ref"]["source"].startswith("exp/07_nn_aec/outputs_v4")
    # 换 epochs 仍对齐总步数（等效更新步数口径与 epoch 数解耦）
    sch100 = DM.resolve_epoch_scheme(100, 16)
    assert sch100["steps_per_epoch"] == 408 and sch100["achieved_steps"] == 40800
    # CLI 覆盖：如实标记来源，不假装是方案(a)
    ov = DM.resolve_epoch_scheme(240, 16, epoch_len_override=10)
    assert ov["epoch_len"] == 10 and ov["steps_per_epoch"] == 0 and \
        ov["epoch_len_source"] == "cli_override"
    with pytest.raises(ValueError):
        DM.resolve_epoch_scheme(0, 16)
    with pytest.raises(ValueError):
        DM.resolve_epoch_scheme(240, 0)


def test_v4_ref_steps_matches_local_artifact():
    """v4 参照步数不是手抄的：与本地 v4 产物 ckpt 的 step 对账（无产物则 skip）。"""
    ckpt = EXP / "outputs_v4" / "nnaec_g0.5_j10" / "last.ckpt"
    if not ckpt.is_file():
        pytest.skip("本机无 outputs_v4 产物（服务器 code-only 仓库属正常）")
    import torch
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert st["step"] == DM.V4_REF_STEPS, f"v4 参照步数漂移：{st['step']} != {DM.V4_REF_STEPS}"


def test_resolve_schedule_defaults_and_override():
    """非 DM 缺省 = 现状（3/8）；DM 缺省 = 20/关早停；显式给值优先并标记。"""
    base = _train_module().resolve_schedule(False)
    assert (base["patience"], base["early_stop"]) == (3, 8)
    assert base["patience_explicit"] is False and base["early_stop_explicit"] is False
    dm = _train_module().resolve_schedule(True)
    assert (dm["patience"], dm["early_stop"]) == (20, 0)
    assert "§2.3b" in dm["basis"]
    exp = _train_module().resolve_schedule(True, lr_patience=7, early_stop=4)
    assert (exp["patience"], exp["early_stop"]) == (7, 4)
    assert exp["patience_explicit"] is True and exp["early_stop_explicit"] is True


_TRAIN_MOD = None


def _train_module():
    """train.py 定点加载（其顶层 import model/dataset 需 exp 目录在 sys.path 上）。"""
    global _TRAIN_MOD
    if _TRAIN_MOD is None:
        _TRAIN_MOD = _load("nnaec_train_dm", TRAIN)
    return _TRAIN_MOD


# ---------------------------------------------------------------- (e) 默认行为不变
def test_train_default_off_artifact_unchanged(tmp_path):
    """不开 --dynamic-mixing：cfg 无任何新键、无 train_manifest.json、early-stop 仍 8。"""
    val = _make_val(tmp_path / "val")
    out = tmp_path / "out"
    rr = _run([sys.executable, "-s", str(TRAIN), "--synthetic", "--epochs", "1",
               "--num-workers", "0", "--device", "cpu", "--val", str(val),
               "--out", str(out)])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    tag_dir = out / "nnaec_g0.5_j10"
    assert tag_dir.is_dir() and not (tag_dir / "train_manifest.json").exists()
    import torch
    cfg = torch.load(tag_dir / "last.ckpt", map_location="cpu",
                     weights_only=False)["cfg"]
    for key in _train_module().NEW_ARG_KEYS:
        assert key not in cfg, f"非 DM 的 cfg 不应出现新键 {key}"
    assert "train_manifest" not in cfg
    assert cfg["early_stop"] == 8


# ---------------------------------------------------------------- (d)(f) 端到端
def test_train_dm_smoke_manifest_and_resume_three_piece(tmp_path):
    """DM 端到端（epoch_len=10）：留痕齐全 + 断点续训三件套 + 非 DM 谱系 resume 被拒。"""
    fx = _fixture(tmp_path)
    val = _make_val(tmp_path / "val")
    out = tmp_path / "out"
    rr = _run(_train_dm_argv(fx, out, val, ["--epochs", "1", "--probe-mem", "2"]))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-3000:]
    d = out / "nnaec_g0.5_j10_dm"
    man = json.loads((d / "train_manifest.json").read_text(encoding="utf-8"))
    assert man["schema"] == "train_manifest/1"
    assert man["data"]["dt_plan"]["version"] == "dt_plan_v1"
    assert man["data"]["dt_plan"]["preset"] == "train_v1_repro"
    assert len(man["data"]["dt_plan"]["sha256"]) == 64
    assert man["data"]["mutex"]["checked"] and man["data"]["mutex"]["near_far_conflicts"] == []
    assert man["epoch_scheme"]["epoch_len_source"] == "cli_override"
    assert man["lr_sched"]["patience"] == 20 and man["lr_sched"]["early_stop"] == 0
    assert man["run"]["from_scratch"] is True
    assert man["probe_mem"]["available"] is False and man["probe_mem"]["step"] == 2
    hist = json.loads((d / "history.json").read_text(encoding="utf-8"))
    assert hist["epoch"] == 1 and hist["step"] == 2 and hist["epoch_len"] == 10
    assert hist["patience"] == 20 and hist["early_stop"] == 0 and hist["target_steps"] == 40830

    import torch
    best = torch.load(d / "best.ckpt", map_location="cpu", weights_only=False)
    last = torch.load(d / "last.ckpt", map_location="cpu", weights_only=False)
    assert set(best) == set(last) >= {"model", "opt", "sched", "epoch", "best", "bad",
                                      "step", "val", "cfg"}
    assert best["cfg"]["train_manifest"]["epoch_scheme"]["epoch_len"] == 10

    # 续训：同一口径继续练 → epoch/step 递增，from_scratch=False，口径不触发重置
    rr2 = _run(_train_dm_argv(fx, out, val, ["--epochs", "2",
                                             "--resume", str(d / "last.ckpt")]))
    assert rr2.returncode == 0, rr2.stderr.decode("utf-8", "replace")[-3000:]
    hist2 = json.loads((d / "history.json").read_text(encoding="utf-8"))
    assert hist2["epoch"] == 2 and hist2["step"] == 4
    man2 = json.loads((d / "train_manifest.json").read_text(encoding="utf-8"))
    assert man2["run"]["from_scratch"] is False
    # 也能从 best.ckpt 续（V5 v3b-016 的键集一致性）
    rr3 = _run(_train_dm_argv(fx, tmp_path / "out2", val, ["--epochs", "3",
                                                          "--resume", str(d / "best.ckpt")]))
    assert rr3.returncode == 0, rr3.stderr.decode("utf-8", "replace")[-3000:]

    # 非 DM 谱系 ckpt：DM 续训必须被拒（阶段0 §2.3a 对照从零起）
    stripped = tmp_path / "nondm.ckpt"
    st = torch.load(d / "last.ckpt", map_location="cpu", weights_only=False)
    st["cfg"].pop("train_manifest")
    torch.save(st, stripped)
    rr4 = _run(_train_dm_argv(fx, tmp_path / "out3", val,
                              ["--epochs", "2", "--resume", str(stripped)]))
    assert rr4.returncode != 0
    err = rr4.stderr.decode("utf-8", "replace")
    assert "从零起" in err and "train_manifest" in err


def test_dm_scenario_ratio_order_is_fest_nst_dt_and_explicit_key(tmp_path):
    """`--dm-scenarios` 顺序 = **FEST,NST,DT**（用非对称值钉死，防 nst/dt 交换漏网）。

    踩过的坑（2026-10-13）：v4 训练配方 nst 0.5086 / dt 0.4914 曾被按 `dt,nst` 的书写
    顺序填进本参数 ⇒ 两个值互换了。对称值（0.5,0.5）测不出来，故这里用**非对称**值断言
    具名映射，并同时钉住 `scenarios_explicit` 字面键（T14 收档按具名字典核对）。
    """
    preset, _ = DM.load_plan(None, "train_v1_repro")
    rp = DM.with_scenario_ratio(preset, ["0", "0.5086", "0.4914"])
    assert rp["scenarios"] == {"fest": 0.0, "nst": 0.5086, "dt": 0.4914}
    # 反序必须得到**不同**的具名映射（否则本用例失去意义）
    rp_swapped = DM.with_scenario_ratio(preset, ["0", "0.4914", "0.5086"])
    assert rp_swapped["scenarios"] != rp["scenarios"]
    assert rp_swapped["scenarios"] == {"fest": 0.0, "nst": 0.4914, "dt": 0.5086}
    # 清单 preset 本身 = 数据集配比 0.4/0.3/0.3（未覆盖时 scenarios_explicit=False）
    fx = _fixture(tmp_path)
    val = _make_val(tmp_path / "val")
    out_default = tmp_path / "o_default"
    rr = _run(_train_dm_argv(fx, out_default, val, ["--epochs", "1"]))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-3000:]
    man = json.loads((out_default / "nnaec_g0.5_j10_dm" / "train_manifest.json")
                     .read_text(encoding="utf-8"))
    assert man["data"]["scenario_ratio"] == {"fest": 0.4, "nst": 0.3, "dt": 0.3}
    assert man["data"]["scenarios_explicit"] is False
    assert man["data"]["scenario_ratio_source"] == "preset"


def test_train_dm_requires_split_lists(tmp_path):
    """DM 必须显式给池目录 + 分割清单（防泄漏）；缺参数 fail fast。"""
    val = _make_val(tmp_path / "val")
    rr = _run([sys.executable, "-s", str(TRAIN), "--dynamic-mixing", "--device", "cpu",
               "--epochs", "1", "--batch-size", "4", "--dm-epoch-len", "10",
               "--val", str(val), "--out", str(tmp_path / "out")])
    assert rr.returncode != 0
    err = rr.stderr.decode("utf-8", "replace")
    assert "分割清单" in err and "near_list" in err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="无 CUDA：显存探针分支不适用")
def test_probe_mem_records_peak_on_cuda_with_console_encoding(tmp_path):
    """`--probe-mem`：CUDA 分支真记录峰值显存，且**不覆盖 PYTHONIOENCODING** 也能跑。

    不带 PYTHONIOENCODING 是刻意的：Windows GBK 控制台下，日志里若出现 GBK 不可编码的
    字符（实测 "≲"）会让训练直接 UnicodeEncodeError 崩掉——本用例把"日志只写本机能写的
    字符"钉成回归（同一 GPU 分支同时验证 max_memory_allocated 被记录）。
    """
    val = _make_val(tmp_path / "val")
    out = tmp_path / "out"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    rr = subprocess.run([sys.executable, "-s", str(TRAIN), "--synthetic", "--epochs", "1",
                         "--batch-size", "2", "--num-workers", "0", "--device", "cuda",
                         "--probe-mem", "2", "--val", str(val), "--out", str(out)],
                        capture_output=True, timeout=900, env=env)
    enc = locale.getpreferredencoding(False)
    assert rr.returncode == 0, rr.stderr.decode(enc, "replace")[-3000:]
    assert "max_memory_allocated" in rr.stdout.decode(enc, "replace")
    cfg = torch.load(out / "nnaec_g0.5_j10" / "last.ckpt", map_location="cpu",
                     weights_only=False)["cfg"]
    rec = cfg["probe_mem"]
    assert rec["available"] is True and rec["max_memory_allocated"] > 0
    assert rec["step"] == 2 and rec["max_memory_reserved"] >= rec["max_memory_allocated"]
    assert rec["device_total"] > 0 and rec["suggest_batch"] > 0


def test_dm_scenario_ratio_override_is_explicit(tmp_path):
    """场景配比可用 --dm-scenarios 显式覆盖（留痕），非法值 fail fast。

    动因（v4 口径）：清单 preset 描述**数据集**配比 0.4/0.3/0.3，而 v4 训练叠了
    `--scenarios dt,nst`（0.4914/0.5086，没见过 FEST）——复刻 v4 训练配比必须显式选择。
    """
    fx = _fixture(tmp_path)
    val = _make_val(tmp_path / "val")
    out = tmp_path / "out"
    rr = _run(_train_dm_argv(fx, out, val, ["--epochs", "1", "--dm-scenarios", "0,0.5,0.5"]))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-3000:]
    man = json.loads((out / "nnaec_g0.5_j10_dm" / "train_manifest.json")
                     .read_text(encoding="utf-8"))
    assert man["data"]["scenario_ratio"] == {"fest": 0.0, "nst": 0.5, "dt": 0.5}
    assert man["data"]["scenario_ratio_source"] == "cli_override"
    # 其余 DT 分布仍来自清单（覆盖只动配比）
    assert man["data"]["dt_plan"]["preset"] == "train_v1_repro"
    assert man["data"]["dt_plan"]["scenarios"] == {"fest": 0.4, "nst": 0.3, "dt": 0.3}
    # 非法配比：拒绝
    rr2 = _run(_train_dm_argv(fx, tmp_path / "o2", val,
                              ["--epochs", "1", "--dm-scenarios", "0.5,0.5,0.5"]))
    assert rr2.returncode != 0
    assert "和为 1" in rr2.stderr.decode("utf-8", "replace")
    # 缺省（不给 --dm-scenarios）= 清单 preset 配比
    rr3 = _run(_train_dm_argv(fx, tmp_path / "o3", val, ["--epochs", "1"]))
    assert rr3.returncode == 0
    man3 = json.loads((tmp_path / "o3" / "nnaec_g0.5_j10_dm" / "train_manifest.json")
                      .read_text(encoding="utf-8"))
    assert man3["data"]["scenario_ratio"] == {"fest": 0.4, "nst": 0.3, "dt": 0.3}
    assert man3["data"]["scenario_ratio_source"] == "preset"
    # 纯函数层面的校验
    preset, _ = DM.load_plan(None, "train_v1_repro")
    assert DM.with_scenario_ratio(preset, [0, 0.4914, 0.5086])["scenarios"]["nst"] == 0.4914
    with pytest.raises(DM.DMPoolError):
        DM.with_scenario_ratio(preset, [1, 1, 1])


def test_dm_aec_synth_cli_dryrun_and_write(tmp_path):
    """附录B 第 2 步 CLI：--dry-run 打印吞吐不落盘；给 --out-dir 则落盘 + 留痕。"""
    fx = _fixture(tmp_path)
    common = ["--near-dir", str(fx["near_dir"]), "--near-list", str(fx["near_list"]),
              "--far-dir", str(fx["far_dir"]), "--far-list", str(fx["far_list"]),
              "--noise-dir", str(fx["noise_dir"]), "--noise-list", str(fx["noise_list"]),
              "--rir-dir", str(fx["rir"]), "--seed", "7"]
    rr = _run([sys.executable, "-s", str(DM_SYNTH), "--num", "6", "--dry-run", *common])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    out = rr.stdout.decode("utf-8")
    assert "样本/秒" in out and "dt_plan_v1/train_v1_repro" in out and "不落盘" in out
    dst = tmp_path / "dm_set"
    rr2 = _run([sys.executable, "-s", str(DM_SYNTH), "--num", "3", "--out-dir", str(dst),
                *common])
    assert rr2.returncode == 0, rr2.stderr.decode("utf-8", "replace")[-2000:]
    rows = list(csv.DictReader(open(dst / "manifest.csv", encoding="utf-8")))
    assert len(rows) == 3
    assert all(r["dt_plan_version"] == "dt_plan_v1"
               and r["dt_plan_preset"] == "train_v1_repro" for r in rows)
    assert all(r["nearfar_mutex"] == "utterance" for r in rows)
    for sub in ("farend", "nearend", "mic"):
        assert len(list((dst / sub).glob("*.wav"))) == 3
    prof = json.loads((dst / "dm_profile.json").read_text(encoding="utf-8"))
    assert prof["dt_plan"]["sha256"] == DM.load_plan(None, None)[1]["sha256"]
    assert prof["mutex"]["checked"] and prof["sampling"]["seed"] == 7
    # 既不给 --out-dir 也不给 --dry-run：拒绝静默什么都不做
    rr3 = _run([sys.executable, "-s", str(DM_SYNTH), "--num", "2", *common])
    assert rr3.returncode != 0
