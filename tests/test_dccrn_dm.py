"""P1-5a 单测：DCCRN 接入共享 DM（清单圈定 + 口径同构 + 防泄漏）。

覆盖：
(a) **键映射与同供给**：`DccrnDynamicMix` 的 `noisy/clean` 逐位等于共享 DM 的 `mic/near`
    （同 seed 同一条 rng 流 ⇒ 与 NN-AEC/级联可配对对照）；
(b) **清单圈定（防泄漏）**：缺池目录/清单 → 拒绝；near/far 同名或同文件 → 入口拒绝；
    缺 `--val` → 拒绝；`--dynamic-mixing` 与 `--no-dm/--synthetic` 互斥；
(c) **口径同构**：manifest 的 `epoch_scheme`（方案 a，参照 = DCCRN 历史 18.7K 步）、
    `lr_sched`（DM 缺省 patience 20 / 关早停）、`scenarios_explicit` +
    `scenario_ratio`（具名）、`from_scratch`、`train_manifest` 进 ckpt cfg；
(d) **旧路径回归**：不开 `--dynamic-mixing` 时 ckpt cfg 仍是历史键集（无新增键）、
    `early_stop` 仍 10、无 `train_manifest.json`、tag 不带 `_dm`；
(e) **断点续训**：同口径续训 epoch/step 递增；非 DM 谱系 ckpt 续训被拒（阶段0 §2.3a）；
(f) **共享口径单一实现**：`core/train_protocol.resolve_schedule` 同时服务 07/04，
    两边 legacy 缺省各自保持（07=8 / 04=10）。
"""
import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP04 = ROOT / "exp" / "04_dccrn"
EXP07 = ROOT / "exp" / "07_nn_aec"
TRAIN04 = EXP04 / "train.py"
SR = 16000
NEAR_NAMES = ("readerA_1.wav", "readerA_2.wav", "readerA_3.wav")
FAR_NAMES = ("readerB_1.wav", "readerB_2.wav")
NOISE_NAMES = ("q_1.wav", "q_2.wav")


def _load(name: str, path: Path):
    """定点加载（临时挂目录上 sys.path，加载后摘掉）。"""
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
DCCRN_DM = SH.dccrn_dm_module()
if str(ROOT) not in sys.path:                           # core.* 的规范导入路径
    sys.path.insert(0, str(ROOT))
import core.train_protocol as CORE_TP  # noqa: E402

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _train04_module():
    """04 的 train 模块——**按需加载、不在模块级加载**。

    它 import 04 自己的 `model`/`dynamic_mixing`（裸名），一旦留在 sys.modules 里，
    后续用例（如 test_nn_aec_dm 里加载 07 的 train）会 `from model import HOP` 命中
    04 的 model 而崩——conftest 的 per-test 隔离只在**用例结束时**清理，所以必须把
    加载放进用例内（实测踩过：全仓跑 2 failed / 单文件跑全绿）。
    """
    return _load("dccrn_train_t18", TRAIN04)


def _wav(path, n=20000, seed=0, level=0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sf.write(path, (level * rng.standard_normal(n)).astype(np.float32), SR)


def _pools(root):
    nd, fd, qd = root / "near", root / "far", root / "noise"
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


def _val_set(root, n=2, seg_sec=1.0, seed=5):
    """aec_synth 结构的静态 val（DM 的 --val 输入）。"""
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


def _mk_ds(fx, cls, *, epoch_len=6, seg_sec=2.0, seed=7):
    preset, plan_info = DM.load_plan(None, "train_v1_repro")
    near, far, noise, pool_info = DM.build_pools(
        fx["near_dir"], fx["near_list"], fx["far_dir"], fx["far_list"],
        fx["noise_dir"], fx["noise_list"])
    kwargs = dict(plan_info=plan_info, pool_info=pool_info, epoch_len=epoch_len,
                  seg_sec=seg_sec, rir_files=sorted(fx["rir"].glob("*.wav")), seed=seed)
    return cls(near, far, noise, preset, **kwargs)


def _argv(fx, val, out, extra=(), epoch_len=10):
    """DM 命令行。``epoch_len=None`` 表示**不给** `--epoch-len`（走方案(a) 换算）。"""
    argv = [sys.executable, "-s", str(TRAIN04), "--dynamic-mixing", "--device", "cpu",
            "--num-workers", "0", "--batch-size", "4",
            "--dm-seg-sec", "1.0", "--dm-scenarios", "0,0.5086,0.4914",
            "--near-dir", str(fx["near_dir"]), "--near-list", str(fx["near_list"]),
            "--far-dir", str(fx["far_dir"]), "--far-list", str(fx["far_list"]),
            "--noise-dir", str(fx["noise_dir"]), "--noise-list", str(fx["noise_list"]),
            "--rir-dir", str(fx["rir"]), "--val", str(val), "--out", str(out), *extra]
    if epoch_len is not None:
        argv += ["--epoch-len", str(epoch_len)]
    return argv


def _run(argv, timeout=900):
    return subprocess.run(argv, capture_output=True, timeout=timeout,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


# ---------------------------------------------------------------- (a) 键映射与同供给
def test_dccrn_key_map_equals_shared_dm(tmp_path):
    """noisy/clean 逐位等于共享 DM 的 mic/near；far 原样透传；固定 seed 可复现。"""
    fx = _pools(tmp_path)
    plain = _mk_ds(fx, DM.AecDynamicMix, seed=13)
    dccrn = _mk_ds(fx, DCCRN_DM.DccrnDynamicMix, seed=13)
    for idx in range(len(plain)):
        a, b = plain[idx], dccrn[idx]
        assert np.array_equal(b["noisy"].numpy(), a["mic"].numpy())
        assert np.array_equal(b["clean"].numpy(), a["near"].numpy())
        assert np.array_equal(b["far"].numpy(), a["far"].numpy())
        for key in ("scenario", "ser_db", "snr_db", "delay_ms", "soft_clip", "name"):
            assert a[key] == b[key], f"同 seed 应共享同一条 rng 流（字段 {key} 不一致）"
    again = _mk_ds(fx, DCCRN_DM.DccrnDynamicMix, seed=13)
    assert np.array_equal(dccrn[2]["noisy"].numpy(), again[2]["noisy"].numpy())
    p = dccrn.provenance()
    assert p["mode"] == "dccrn_aec_dynamic_mixing"
    assert p["key_map"] == {"noisy": "mic", "clean": "near", "far": "far"}
    assert "2.1811" in p["note"] and "混比" in p["note"]


def test_dccrn_dm_without_epoch_len_uses_scheme_a(tmp_path):
    """回归：DM **不给** `--epoch-len` 时必须走方案(a)（历史事故：argparse 的 default=2000
    被 `resolve_epoch_len` 当成"显式值"⇒ `epoch_len_source=cli_override`、每 epoch 500 步
    而非 187，实际只跑到目标 16.6%，`delta_frac=-0.834`，且**没有任何提示**）。

    这里用很小的目标步数（8）让方案(a) 在 CPU 上真跑得完：steps_per_epoch=8、epoch_len=32
    ⇒ 一个 epoch 恰好 8 步；`history.json` 的 `step` 同时证明**训练真的按该口径跑了**。
    若回归（default 又变成 2000），manifest 会写 cli_override/500 步，本用例立刻失败；
    timeout 兜住"回归时白跑一个 500 步的 epoch"的最坏代价。
    """
    fx = _pools(tmp_path)
    val = _val_set(tmp_path / "val")
    out = tmp_path / "out_scheme_a"
    rr = _run(_argv(fx, val, out, ["--epochs", "1", "--dm-target-steps", "8"],
                    epoch_len=None), timeout=300)
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-3000:]
    man = json.loads((out / "dccrn_dm" / "train_manifest.json").read_text(encoding="utf-8"))
    sch = man["epoch_scheme"]
    assert sch["epoch_len_source"] == "derived_scheme_a", \
        f"DM 未显式给 --epoch-len 却被当成覆盖：{sch['epoch_len_source']}（实际 epoch_len " \
        f"{sch['epoch_len']}）"
    assert (sch["target_steps"], sch["steps_per_epoch"], sch["epoch_len"],
            sch["achieved_steps"], sch["delta_steps"]) == (8, 8, 32, 8, 0)
    assert sch["reference"]["name"] == "dccrn_hist" and "v4_ref" not in sch
    hist = json.loads((out / "dccrn_dm" / "history.json").read_text(encoding="utf-8"))
    assert hist["epoch_len"] == 32 and hist["steps_per_epoch"] == 8
    assert hist["step"] == 8, f"训练实际步数 {hist['step']} != 方案(a) 的 8 步"
    # 反向对照：显式给 --epoch-len 才允许进入 cli_override，并且打印偏差告警
    out2 = tmp_path / "out_override"
    rr2 = _run(_argv(fx, val, out2, ["--epochs", "1", "--dm-target-steps", "8"],
                     epoch_len=10), timeout=300)
    assert rr2.returncode == 0
    man2 = json.loads((out2 / "dccrn_dm" / "train_manifest.json").read_text(encoding="utf-8"))
    assert man2["epoch_scheme"]["epoch_len_source"] == "cli_override"
    assert "[warn]" in rr2.stdout.decode("utf-8") and "覆盖了方案(a)" in rr2.stdout.decode("utf-8")


def test_dccrn_val_adapter_reads_aec_synth(tmp_path):
    """静态 val 适配：noisy=mic、clean=nearend、far=farend（与 07 同目录同列）。"""
    val = _val_set(tmp_path / "val")
    ds = DCCRN_DM.AecValAsDccrn(val)
    assert len(ds) == 2
    it = ds[0]
    mic = sf.read(val / "mic" / "v0.wav", dtype="float32")[0]
    near = sf.read(val / "nearend" / "v0.wav", dtype="float32")[0]
    far = sf.read(val / "farend" / "v0.wav", dtype="float32")[0]
    assert np.array_equal(it["noisy"].numpy(), mic)
    assert np.array_equal(it["clean"].numpy(), near)
    assert np.array_equal(it["far"].numpy(), far)
    assert it["scenario"] == "dt" and it["name"] == "v0"
    with pytest.raises(FileNotFoundError):
        DCCRN_DM.AecValAsDccrn(tmp_path / "nope")


# ---------------------------------------------------------------- (c) 口径同构
def test_dccrn_epoch_scheme_reference_is_own_history():
    """方案(a) 用 DCCRN 自己的历史口径：100ep/bs64 → 187 步/ep、epoch_len 11968、总 18700。"""
    ref = DCCRN_DM.DCCRN_REF
    assert (ref["name"], ref["steps"], ref["steps_per_epoch"]) == ("dccrn_hist", 18700, 187)
    sch = DM.resolve_epoch_scheme(100, 64, ref["steps"], reference=ref)
    assert (sch["steps_per_epoch"], sch["epoch_len"], sch["achieved_steps"]) == (187, 11968, 18700)
    assert sch["delta_steps"] == 0 and sch["reference"]["name"] == "dccrn_hist"
    assert "v4_ref" not in sch, "非缺省参照时不应带 v4 兼容键"
    # 缺省参照（NN-AEC 用法）仍带 v4_ref（兼容键，07/09 单测依赖）
    assert "v4_ref" in DM.resolve_epoch_scheme(240, 16)
    assert "README" in ref["source"]


def test_core_train_protocol_shared_by_both_entries():
    """LR/早停口径单一实现：core 一份，07/04 各自 legacy 缺省（8/10）。"""
    assert _train04_module()._shared_resolve_schedule is CORE_TP.resolve_schedule
    assert _train04_module().probe_memory is CORE_TP.probe_memory
    for legacy_stop, expected in ((8, 8), (10, 10)):
        base = CORE_TP.resolve_schedule(False, legacy_early_stop=legacy_stop)
        assert (base["patience"], base["early_stop"]) == (3, expected)
        assert f"patience 3 + early-stop {expected}" in base["basis"]
    dm = CORE_TP.resolve_schedule(True, legacy_early_stop=10)
    assert (dm["patience"], dm["early_stop"]) == (20, 0) and "§2.3b" in dm["basis"]
    exp = CORE_TP.resolve_schedule(True, lr_patience=7, early_stop=3, legacy_early_stop=10)
    assert (exp["patience"], exp["early_stop"]) == (7, 3)
    # 04 的 legacy 缺省解析：epoch_len 2000 / DM 交给方案(a) / 显式值优先
    assert _train04_module().resolve_epoch_len(False, None) == 2000
    assert _train04_module().resolve_epoch_len(True, None) is None
    assert _train04_module().resolve_epoch_len(True, 10) == 10
    assert _train04_module().resolve_epoch_len(False, 1500) == 1500


# ---------------------------------------------------------------- (b)(d)(e) CLI
def test_dccrn_dm_smoke_manifest_and_resume(tmp_path):
    """DM 端到端（epoch_len=10，CPU）：留痕齐全 + 续训 + 非 DM 谱系被拒。"""
    fx = _pools(tmp_path)
    val = _val_set(tmp_path / "val")
    out = tmp_path / "out"
    rr = _run(_argv(fx, val, out, ["--epochs", "1", "--probe-mem", "2"]))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-3000:]
    d = out / "dccrn_dm"
    man = json.loads((d / "train_manifest.json").read_text(encoding="utf-8"))
    assert man["data"]["train_mode"] == "dccrn_aec_dynamic_mixing"
    assert man["data"]["key_map"] == {"noisy": "mic", "clean": "near", "far": "far"}
    assert man["data"]["dt_plan"]["version"] == "dt_plan_v1"
    assert len(man["data"]["dt_plan"]["sha256"]) == 64
    assert man["data"]["mutex"]["checked"] and man["data"]["mutex"]["near_far_conflicts"] == []
    assert man["data"]["scenario_ratio"] == {"fest": 0.0, "nst": 0.5086, "dt": 0.4914}
    assert man["data"]["scenarios_explicit"] is True
    assert man["epoch_scheme"]["reference"]["name"] == "dccrn_hist"
    assert man["epoch_scheme"]["epoch_len"] == 10
    assert man["lr_sched"]["patience"] == 20 and man["lr_sched"]["early_stop"] == 0
    assert man["run"]["from_scratch"] is True
    assert man["probe_mem"]["available"] is False and man["probe_mem"]["step"] == 2
    hist = json.loads((d / "history.json").read_text(encoding="utf-8"))
    assert hist["epoch"] == 1 and hist["step"] == 2 and hist["epoch_len"] == 10
    assert hist["target_steps"] == 18700 and hist["patience"] == 20 and hist["early_stop"] == 0
    st = torch.load(d / "last.ckpt", map_location="cpu", weights_only=False)
    assert set(st) >= {"model", "opt", "sched", "epoch", "best", "bad", "step", "val", "cfg"}
    assert st["cfg"]["train_manifest"]["epoch_scheme"]["epoch_len"] == 10

    # 同口径续训：epoch/step 递增、from_scratch=False
    rr2 = _run(_argv(fx, val, out, ["--epochs", "2", "--resume", str(d / "last.ckpt")]))
    assert rr2.returncode == 0, rr2.stderr.decode("utf-8", "replace")[-3000:]
    hist2 = json.loads((d / "history.json").read_text(encoding="utf-8"))
    assert hist2["epoch"] == 2 and hist2["step"] == 4
    man2 = json.loads((d / "train_manifest.json").read_text(encoding="utf-8"))
    assert man2["run"]["from_scratch"] is False

    # 非 DM 谱系 ckpt（cfg 无 train_manifest）：DM 续训必须被拒
    stripped = tmp_path / "dns.ckpt"
    st2 = torch.load(d / "last.ckpt", map_location="cpu", weights_only=False)
    st2["cfg"].pop("train_manifest")
    torch.save(st2, stripped)
    rr3 = _run(_argv(fx, val, tmp_path / "out2",
                     ["--epochs", "2", "--resume", str(stripped)]))
    assert rr3.returncode != 0
    err = rr3.stderr.decode("utf-8", "replace")
    assert "从零起" in err and "train_manifest" in err


def test_dccrn_legacy_path_cfg_unchanged(tmp_path):
    """不开 --dynamic-mixing：cfg 仍是历史键集、early_stop 10、无 train_manifest.json。"""
    out = tmp_path / "out_legacy"
    rr = _run([sys.executable, "-s", str(TRAIN04), "--smoke", "--device", "cpu",
               "--batch-size", "2", "--out", str(out)])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-3000:]
    d = out / "dccrn"
    assert d.is_dir() and not (d / "train_manifest.json").exists()
    cfg = torch.load(d / "last.ckpt", map_location="cpu", weights_only=False)["cfg"]
    for key in _train04_module().NEW_ARG_KEYS:
        assert key not in cfg, f"非 DM 的 cfg 不应出现新键 {key}"
    assert cfg["early_stop"] == 10 and cfg["epoch_len"] == 8      # smoke 显式给 8
    hist = json.loads((d / "history.json").read_text(encoding="utf-8"))
    assert set(hist) == {"epoch", "val", "best", "bad", "step"}, "旧 history 形状不变"


def test_dccrn_dm_fail_fast_paths(tmp_path):
    """缺清单/缺 val/与 --no-dm 冲突/near-far 同名：全部 fail fast 且信息可读。"""
    fx = _pools(tmp_path)
    val = _val_set(tmp_path / "val")
    # 缺清单
    rr = _run([sys.executable, "-s", str(TRAIN04), "--dynamic-mixing", "--device", "cpu",
               "--epochs", "1", "--batch-size", "4", "--epoch-len", "10",
               "--val", str(val), "--out", str(tmp_path / "o1")])
    assert rr.returncode != 0
    err = rr.stderr.decode("utf-8", "replace")
    assert "分割清单" in err and "near_list" in err
    # 缺 --val
    rr = _run([sys.executable, "-s", str(TRAIN04), "--dynamic-mixing", "--device", "cpu",
               "--epochs", "1", "--near-dir", str(fx["near_dir"]),
               "--near-list", str(fx["near_list"]), "--far-dir", str(fx["far_dir"]),
               "--far-list", str(fx["far_list"]), "--noise-dir", str(fx["noise_dir"]),
               "--noise-list", str(fx["noise_list"]), "--out", str(tmp_path / "o2")])
    assert rr.returncode != 0 and "--val" in rr.stderr.decode("utf-8", "replace")
    # 与 --no-dm 互斥
    rr = _run(_argv(fx, val, tmp_path / "o3", ["--epochs", "1", "--no-dm"]))
    assert rr.returncode != 0 and "互斥" in rr.stderr.decode("utf-8", "replace")
    # near/far 同名（跨目录复制）→ 共享 build_pools 的互斥判据拒绝
    (fx["far_dir"] / NEAR_NAMES[0]).write_bytes((fx["near_dir"] / NEAR_NAMES[0]).read_bytes())
    with open(fx["far_list"], "a", encoding="utf-8") as f:
        f.write(NEAR_NAMES[0] + "\n")
    rr = _run(_argv(fx, val, tmp_path / "o4", ["--epochs", "1"]))
    assert rr.returncode != 0
    assert "utterance 主名" in rr.stderr.decode("utf-8", "replace")
