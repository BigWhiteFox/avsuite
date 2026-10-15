"""P1-3b 单测：级联 DM（AecCascadeDynamicMix + 09 训练入口 + val 侧预计算口径）。

覆盖：
(a) **线性级正确性**：`AecCascadeDynamicMix` 的条目 = 「DM 合成 → TradAEC(MDF)」逐位复算
    （e_lin = mic − y_echo）；far 平移作用在 **y_echo** 上而不是原始 far（与静态级联
    训练同序：静态集是先在 `precompute_cascade.py` 跑 MDF，再由 AecDataset 对 `farend/`
    做 jitter）；
(b) **rng 流对齐**：同 seed 下级联版与纯 DM 版看到同一条随机流（场景/文件/SER/SNR/
    时延/soft-clip/平移量一致）——配对对照的前提；
(c) **口径留痕**：manifest 的 `linear_stage`（TradAEC 全部生效参数 + `fresh_per_clip`
    + `equivalent_to_static_precompute=False`）与 `scenarios_explicit`；
(d) **09 训练入口**：`exp/09_cascade_aec/train.py` 薄壳透传到 07（补 `--cascade-linear`），
    CPU 端到端跑通（epoch_len=10，含 MDF 路径）；`--cascade-linear` 缺 `--dynamic-mixing`
    时拒绝；
(e) **val 侧预计算口径**：`precompute_cascade.py --fresh-instance-per-clip` 的产物与
    `AecCascadeDynamicMix.run_linear` 逐位一致（训练/val 同约定）；不带该开关时 profile
    如实标注 `shared_across_clips` / `equals_dm_training_policy=False`（与历史静态集
    不等价、不可直接混比）；`--measure N` 只测耗时、不落盘。
"""
import csv
import importlib.util
import json
import locale
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp"
TRAIN_09 = EXP / "09_cascade_aec" / "train.py"
PRECOMPUTE = EXP / "09_cascade_aec" / "precompute_cascade.py"
DM_SYNTH = ROOT / "scripts" / "dm_aec_synth.py"
SR = 16000


def _load(name: str, path: Path):
    """定点加载（临时把所在目录挂上 sys.path，加载后摘掉）。"""
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
CM = SH.cascade_module()
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

NEAR_NAMES = ("readerA_1.wav", "readerA_2.wav", "readerA_3.wav")
FAR_NAMES = ("readerB_1.wav", "readerB_2.wav")
NOISE_NAMES = ("q_1.wav",)


def _wav(path, n=20000, seed=0, level=0.2):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sf.write(path, (level * rng.standard_normal(n)).astype(np.float32), SR)


def _fixture(root):
    """三池 + 三清单 + RIR（near/far 主名互斥）。"""
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


def _make_synth_dir(root, n=3, seg_sec=1.0, seed=5):
    """最小 aec_synth 结构（precompute/train 的 --val 输入）：mic/farend/nearend + manifest。"""
    rng = np.random.default_rng(seed)
    for sub in ("farend", "nearend", "mic"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        far = (0.3 * rng.standard_normal(int(seg_sec * SR))).astype(np.float32)
        near = (0.05 * rng.standard_normal(int(seg_sec * SR))).astype(np.float32)
        mic = far * 0.7 + near
        sf.write(root / "farend" / f"v{i}.wav", far, SR)
        sf.write(root / "nearend" / f"v{i}.wav", near, SR)
        sf.write(root / "mic" / f"v{i}.wav", mic, SR)
        rows.append({"name": f"v{i}", "scenario": "dt", "rir_src": "dir", "near_file": "n",
                     "far_file": "f", "noise_file": "q", "delay_ms": "10.0", "ser_db": "5.0",
                     "snr_db": "20.0", "soft_clip": "0", "leadin_sec": "0.0"})
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root


def _mk_ds(fx, cls, *, epoch_len=6, seg_sec=2.0, seed=7, jitter_ms=10.0, trad=None):
    preset, plan_info = DM.load_plan(None, "train_v1_repro")
    near, far, noise, pool_info = DM.build_pools(
        fx["near_dir"], fx["near_list"], fx["far_dir"], fx["far_list"],
        fx["noise_dir"], fx["noise_list"])
    kwargs = dict(plan_info=plan_info, pool_info=pool_info, epoch_len=epoch_len,
                  seg_sec=seg_sec, rir_files=sorted(fx["rir"].glob("*.wav")), seed=seed,
                  jitter_ms=jitter_ms)
    if cls is CM.AecCascadeDynamicMix:
        kwargs["trad_kwargs"] = trad or {"method": "mdf", "partitions": 8, "mu": 0.7}
    return cls(near, far, noise, preset, **kwargs)


def _run(argv, timeout=900):
    return subprocess.run(argv, capture_output=True, timeout=timeout,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


# ---------------------------------------------------------------- (a)(b) 线性级正确性
def test_cascade_item_matches_direct_trad_aec(tmp_path):
    """级联条目 = DM 合成 → TradAEC(MDF)：e_lin/y_echo 与逐位复算一致。"""
    fx = _fixture(tmp_path)
    plain = _mk_ds(fx, DM.AecDynamicMix, seed=11, jitter_ms=0.0)
    casc = _mk_ds(fx, CM.AecCascadeDynamicMix, seed=11, jitter_ms=0.0)
    for idx in (0, 3, 5):
        raw = plain[idx]                         # jitter=0：far 即 MDF 的输入
        y_echo, e_lin = casc.run_linear(raw["mic"].numpy(), raw["far"].numpy())
        it = casc[idx]
        assert np.array_equal(it["mic"].numpy(), e_lin), f"idx {idx}: e_lin 不一致"
        assert np.array_equal(it["far"].numpy(), y_echo), f"idx {idx}: y_echo 不一致"
        assert np.array_equal(it["near"].numpy(), raw["near"].numpy())
        # 公式复核：e_lin = mic − y_echo（float32）
        assert np.array_equal(it["mic"].numpy(),
                              (raw["mic"].numpy() - it["far"].numpy()).astype(np.float32))
    assert casc.linear_calls == 6, "3 次直调 + 3 次取条目，各跑一次线性级"


def test_cascade_jitter_applied_to_y_echo_not_raw_far(tmp_path):
    """far 平移作用在 y_echo 上（静态级联口径），不是平移后再过 MDF。"""
    fx = _fixture(tmp_path)
    jit = 10
    plain0 = _mk_ds(fx, DM.AecDynamicMix, seed=13, jitter_ms=0.0)
    plain1 = _mk_ds(fx, DM.AecDynamicMix, seed=13, jitter_ms=jit)
    casc1 = _mk_ds(fx, CM.AecCascadeDynamicMix, seed=13, jitter_ms=jit)
    for idx in (1, 4):
        raw = plain0[idx]
        base = plain1[idx]
        it = casc1[idx]
        # 从"纯 DM 平移"里反解出本次的整数平移量 k（两条流同 seed ⇒ 同一个 k）
        k = None
        for cand in range(-int(jit * SR / 1000), int(jit * SR / 1000) + 1):
            if np.array_equal(DM._shift(raw["far"].numpy(), cand), base["far"].numpy()):
                k = cand
                break
        assert k is not None, f"idx {idx}: 未能在 jitter 范围内反解出平移量"
        y_echo, e_lin = casc1.run_linear(raw["mic"].numpy(), raw["far"].numpy())
        assert np.array_equal(it["far"].numpy(), DM._shift(y_echo, k))
        assert np.array_equal(it["mic"].numpy(), e_lin), "线性级输出不应被平移"


def test_cascade_shares_rng_stream_with_plain_dm(tmp_path):
    """同 seed：级联版与纯 DM 版看到同一条随机流（配对对照前提）。"""
    fx = _fixture(tmp_path)
    plain = _mk_ds(fx, DM.AecDynamicMix, seed=23)
    casc = _mk_ds(fx, CM.AecCascadeDynamicMix, seed=23)
    for idx in range(len(plain)):
        a, b = plain[idx], casc[idx]
        for key in ("scenario", "ser_db", "snr_db", "leadin_sec", "soft_clip", "delay_ms"):
            assert a[key] == b[key], f"idx {idx} 字段 {key} 不一致（rng 流漂移）"
        assert a["name"] == b["name"]


def test_cascade_determinism_and_linear_stage_info(tmp_path):
    """固定 seed 逐位可复现；linear_stage 口径字段齐全且标明不等价于静态预计算。"""
    fx = _fixture(tmp_path)
    casc = _mk_ds(fx, CM.AecCascadeDynamicMix, seed=31)
    assert np.array_equal(casc[2]["mic"].numpy(), casc[2]["mic"].numpy())
    again = _mk_ds(fx, CM.AecCascadeDynamicMix, seed=31)
    assert np.array_equal(casc[2]["far"].numpy(), again[2]["far"].numpy())
    info = casc.linear_stage_info()
    p = info["trad_params"]
    assert info["kind"] == "cascade_mdf"
    assert (p["method"], p["partitions"], p["mu"]) == ("mdf", 8, 0.7)
    for key in ("block", "use_dtd", "use_res", "align_margin_ms", "reestimate_sec",
                "realign_min_delta_ms", "mdf_w_abs_cap", "res_alpha", "res_gmin", "sr"):
        assert key in p, f"linear_stage 缺 TradAEC 参数 {key}"
    assert info["trad_instance_policy"] == "fresh_per_clip"
    assert info["equivalent_to_static_precompute"] is False
    assert "不等价" in info["note"] and "混比" in info["note"]
    assert casc.provenance()["mode"] == "cascade_dynamic_mixing"
    with pytest.raises(ValueError, match="不认识的字段"):
        _mk_ds(fx, CM.AecCascadeDynamicMix, trad={"method": "mdf", "partitons": 8})


# ---------------------------------------------------------------- (d) 09 训练入口
def _train09_argv(fx, val, out, extra=()):
    return [sys.executable, "-s", str(TRAIN_09), "--dynamic-mixing", "--device", "cpu",
            "--num-workers", "0", "--batch-size", "4", "--dm-epoch-len", "10",
            "--dm-scenarios", "0,0.5086,0.4914",
            "--near-dir", str(fx["near_dir"]), "--near-list", str(fx["near_list"]),
            "--far-dir", str(fx["far_dir"]), "--far-list", str(fx["far_list"]),
            "--noise-dir", str(fx["noise_dir"]), "--noise-list", str(fx["noise_list"]),
            "--rir-dir", str(fx["rir"]), "--val", str(val), "--out", str(out), *extra]


def test_train09_cascade_cpu_smoke_and_manifest(tmp_path):
    """09 入口 CPU 端到端（epoch_len=10，含 MDF）：口径齐全 + 目录带 _dm_casc。"""
    fx = _fixture(tmp_path)
    val = _make_synth_dir(tmp_path / "val")
    out = tmp_path / "out"
    rr = _run(_train09_argv(fx, val, out, ["--epochs", "1"]))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-3000:]
    assert "[cascade] 转交" in rr.stdout.decode("utf-8")
    d = out / "nnaec_g0.5_j10_dm_casc"
    man = json.loads((d / "train_manifest.json").read_text(encoding="utf-8"))
    assert man["data"]["train_mode"] == "cascade_dynamic_mixing"
    assert man["data"]["linear_stage"]["trad_instance_policy"] == "fresh_per_clip"
    assert man["data"]["linear_stage"]["trad_params"]["partitions"] == 8
    assert man["data"]["linear_stage"]["trad_params"]["mu"] == 0.7
    # ⑤：场景配比具名留痕 + scenarios_explicit 字面键（顺序 = FEST,NST,DT）
    assert man["data"]["scenario_ratio"] == {"fest": 0.0, "nst": 0.5086, "dt": 0.4914}
    assert man["data"]["scenarios_explicit"] is True
    assert man["data"]["scenario_ratio_source"] == "cli_override"
    assert man["epoch_scheme"]["epoch_len"] == 10
    assert man["lr_sched"]["patience"] == 20 and man["lr_sched"]["early_stop"] == 0
    # t26：级联 run 的 manifest 同源带 RIR 池留痕（09 是 07 的薄壳 ⇒ 自动受益）
    rir = man["data"]["rir"]
    assert rir["src"] == "dir" and rir["dir"] == str(fx["rir"])
    assert rir["n_files"] == 1 and len(rir["set_sha256"]) == 64
    assert rir["max_ms"] == 100.0 and rir["list"] is None
    st = torch.load(d / "last.ckpt", map_location="cpu", weights_only=False)
    assert "linear_stage" in st["cfg"]["train_manifest"]["data"]  # 口径钉在权重旁
    assert st["cfg"]["train_manifest"]["data"]["rir"]["set_sha256"] == rir["set_sha256"]
    assert json.loads((d / "history.json").read_text(encoding="utf-8"))["epoch"] == 1


def test_train09_requires_dynamic_mixing(tmp_path):
    """`--cascade-linear` 缺 `--dynamic-mixing` → 拒绝（级联是 DM 合成后接线性级）。"""
    rr = _run([sys.executable, "-s", str(TRAIN_09), "--epochs", "1", "--device", "cpu",
               "--val", str(tmp_path), "--out", str(tmp_path / "o")])
    assert rr.returncode != 0
    assert "--dynamic-mixing" in rr.stderr.decode("utf-8", "replace")


# ---------------------------------------------------------------- (e) val 侧预计算口径
def test_precompute_fresh_instance_parity_with_dm(tmp_path):
    """val 侧 `--fresh-instance-per-clip` 与训练侧 run_linear 逐位一致（同约定）。"""
    root = _make_synth_dir(tmp_path / "src", n=2, seg_sec=1.0, seed=9)
    out = tmp_path / "src_lin"
    rr = _run([sys.executable, "-s", str(PRECOMPUTE), "--synth-dir", str(root),
               "--out-dir", str(out), "--fresh-instance-per-clip"])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    prof = json.loads((out / "cascade_lin_profile.json").read_text(encoding="utf-8"))
    assert prof["trad_instance_policy"] == "fresh_per_clip"
    assert prof["equals_dm_training_policy"] is True
    assert prof["trad_params"]["partitions"] == 8 and prof["trad_params"]["mu"] == 0.7
    assert (out / "manifest.csv").read_bytes() == (root / "manifest.csv").read_bytes()
    fx = _fixture(tmp_path)
    ds = _mk_ds(fx, CM.AecCascadeDynamicMix, seed=1)
    for name in ("v0", "v1"):
        mic = sf.read(root / "mic" / f"{name}.wav", dtype="float32")[0]
        far = sf.read(root / "farend" / f"{name}.wav", dtype="float32")[0]
        y_echo, e_lin = ds.run_linear(mic, far)
        got_far = sf.read(out / "farend" / f"{name}.wav", dtype="float32")[0]
        got_mic = sf.read(out / "mic" / f"{name}.wav", dtype="float32")[0]
        assert np.array_equal(got_far, y_echo), f"{name}: y_echo 与训练侧不一致"
        assert np.array_equal(got_mic, e_lin), f"{name}: e_lin 与训练侧不一致"
        # 目标逐字节复制（不重编码）
        assert (out / "nearend" / f"{name}.wav").read_bytes() == \
            (root / "nearend" / f"{name}.wav").read_bytes()


def test_precompute_shared_instance_policy_is_flagged(tmp_path):
    """缺省（历史口径）跨 clip 复用：profile 必须标明与 DM 训练**不等价**。"""
    root = _make_synth_dir(tmp_path / "src", n=2, seg_sec=1.0, seed=3)
    out = tmp_path / "src_lin_shared"
    rr = _run([sys.executable, "-s", str(PRECOMPUTE), "--synth-dir", str(root),
               "--out-dir", str(out)])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    prof = json.loads((out / "cascade_lin_profile.json").read_text(encoding="utf-8"))
    assert prof["trad_instance_policy"] == "shared_across_clips"
    assert prof["equals_dm_training_policy"] is False
    assert "不等价" in prof["note"]


def test_precompute_measure_mode_writes_nothing(tmp_path):
    """`--measure N`：只测 ms/clip（含实例创建）、不落盘、不需要 --out-dir。"""
    root = _make_synth_dir(tmp_path / "src", n=3, seg_sec=1.0, seed=7)
    before = sorted(p.name for p in tmp_path.iterdir())
    rr = _run([sys.executable, "-s", str(PRECOMPUTE), "--synth-dir", str(root),
               "--measure", "2"])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    out = rr.stdout.decode("utf-8")
    assert "ms/clip" in out and "fresh_per_clip" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_cli_help_is_console_encoding_safe(tmp_path):
    """四个入口的 `--help` 与告警分支必须在**本机控制台编码**下可打印（GBK 不崩）。

    规则：只有会被写到控制台的文本受此约束——argparse 的 ``description=__doc__``、
    ``print``、异常消息；注释与普通 docstring 不在此列（不写控制台）。
    实测踩过两次：docstring 里的 "⇒"（`--help` 直接 UnicodeEncodeError）与 print 里的
    "−"（`--dm-seg-sec` 与清单不一致的告警分支崩训练）。本用例把该契约钉住。
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    enc = locale.getpreferredencoding(False)
    for cmd, must_have in (
            (str(TRAIN_09), "--cascade-linear"),
            (str(EXP / "07_nn_aec" / "train.py"), "--dynamic-mixing"),
            (str(DM_SYNTH), "--dry-run"),
            (str(PRECOMPUTE), "--fresh-instance-per-clip")):
        rr = subprocess.run([sys.executable, "-s", cmd, "--help"], capture_output=True,
                            timeout=300, env=env)
        assert rr.returncode == 0, \
            f"{cmd} --help 在 {enc} 控制台下失败：{rr.stderr.decode(enc, 'replace')[-800:]}"
        out = rr.stdout.decode(enc, "replace")
        assert must_have in out, f"{cmd} --help 缺 {must_have}（参数没接上？）"
    # 告警分支（--dm-seg-sec 与清单不一致）在控制台编码下也要能打印
    fx = _fixture(tmp_path)
    val = _make_synth_dir(tmp_path / "val")
    argv = _train09_argv(fx, val, tmp_path / "out_seg", ["--epochs", "1", "--dm-seg-sec", "3"])
    rr = subprocess.run(argv, capture_output=True, timeout=900, env=env)
    assert rr.returncode == 0, rr.stderr.decode(enc, "replace")[-2000:]
    assert "dur_sec" in rr.stdout.decode(enc, "replace")


def test_dm_aec_synth_cli_still_works_after_cascade_changes(tmp_path):
    """冒烟：dm_aec_synth 的 CLI 契约未被 P1-3b 改动破坏（t10 回归）。"""
    fx = _fixture(tmp_path)
    rr = _run([sys.executable, "-s", str(DM_SYNTH), "--num", "3", "--dry-run",
               "--near-dir", str(fx["near_dir"]), "--near-list", str(fx["near_list"]),
               "--far-dir", str(fx["far_dir"]), "--far-list", str(fx["far_list"]),
               "--noise-dir", str(fx["noise_dir"]), "--noise-list", str(fx["noise_list"]),
               "--rir-dir", str(fx["rir"]), "--seed", "5"])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    assert "样本/秒" in rr.stdout.decode("utf-8")
