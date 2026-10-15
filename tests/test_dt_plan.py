"""阶段0 / 工作项 0-5 单测：DT 合成比例清单（`scripts/dt_plan.py` + `--dt-plan`）。

覆盖四件事：
(a) **清单本身可读可复现**：`build_plan` 生成的结果与随仓的 `configs/dt_plan_v1.json` 语义一致
    （presets/分布/版本号），`--write` 幂等；`load_plan` 对坏 schema/坏配比/坏分布一律拒绝。
(b) **默认路径逐比特不变（0-5 的硬约束）**：不给 `--dt-plan` 时 manifest 的**列集合与数值**
    与改动前完全一致（列白名单 + 与 `tests/test_aec_synth.py` 同法的数值复算），
    且不产出 `dt_plan_profile.json`。
(c) **清单路径真的按清单采样**：manifest 追加 `dt_plan_version`/`dt_plan_preset` 两列、
    DT 的 leadin/ser/snr 落在清单区间内、`dt_plan_profile.json` 记录了清单 sha 与 CLI 留痕；
    同 seed 重跑逐字节一致（"可复现同分布"）。
(d) **显式冲突要被拦住**：`--scenarios` 显式填了与清单不同的配比 → 非零退出且不产出；
    只吃默认值时由清单覆盖（不逼用户重复声明）；`--dt-plan-preset` 缺 `--dt-plan` 报错。

用法：`pytest tests/test_dt_plan.py -q`
"""
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import aec_synth as S  # noqa: E402
import dt_plan as DP  # noqa: E402

SR = 16000
SCRIPT = Path(S.__file__)
PLAN = ROOT / "configs" / "dt_plan_v1.json"
LEGACY_COLUMNS = ["name", "scenario", "rir_src", "near_file", "far_file", "noise_file",
                  "delay_ms", "ser_db", "snr_db", "soft_clip", "leadin_sec"]


# ---------------------------------------------------------------- fixture
def _wav(path, n=24000, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sf.write(path, (0.2 * rng.standard_normal(n)).astype(np.float32), SR)


def _rir_dir(root):
    """确定性自备 RIR：避开 pyroomacoustics 的环境依赖（RIR 与本用例无关）。

    长度取 1s（> 本文件任何 --dur-sec），使 RIR 文件走 fit_len 的**裁剪**分支；短文件会走
    循环补、不抽裁剪偏移，随机流消耗序随之改变（C/G 对账按真分支走，故 fixture 要对齐现实）。
    """
    d = root / "rir"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4242)
    h = rng.standard_normal(SR) * np.exp(-np.arange(SR) / (0.02 * SR))
    h[0] = 1.0
    sf.write(d / "rir.wav", (h / np.linalg.norm(h) * 0.7).astype(np.float32), SR)
    return d


def _fixture(root):
    """互斥 near/far 池 + 噪声池 + RIR；DT 样本足够多以便核对区间。"""
    nd, fd, qd = root / "near", root / "far", root / "noise"
    for i in range(4):
        _wav(nd / f"n_{i:04d}.wav", seed=100 + i)
        _wav(fd / f"f_{i:04d}.wav", seed=200 + i)
    for i in range(3):
        _wav(qd / f"q_{i:04d}.wav", seed=300 + i)
    return nd, fd, qd, _rir_dir(root)


def _run(nd, fd, qd, out, rir, extra=(), num=40, seed=7, dur="2.0"):
    argv = [sys.executable, "-s", str(SCRIPT), "--num", str(num), "--dur-sec", dur,
            "--seed", str(seed), "--near-dir", str(nd), "--far-dir", str(fd),
            "--noise-dir", str(qd), "--out-dir", str(out)]
    if rir is not None:                    # None 时不传 --rir-dir -> 走 RIR 仿真/兜底分支（sim 档）
        argv += ["--rir-dir", str(rir)]
    return subprocess.run(argv + list(extra), capture_output=True, timeout=600,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def _rows(out):
    with open(out / "manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- (a) 清单本体
def test_shipped_plan_matches_builder(tmp_path):
    """随仓清单与 `build_plan()` 的语义一致（手改清单会被本用例抓住）。"""
    plan, sha = DP.load_plan(PLAN)
    built = DP.build_plan()
    assert plan["schema"] == built["schema"] == DP.SCHEMA
    assert plan["version"] == built["version"] == "dt_plan_v1"
    assert set(plan["presets"]) == set(built["presets"])
    for name, want in built["presets"].items():
        got = plan["presets"][name]
        assert got["scenarios"] == want["scenarios"], name
        for field in ("dt_leadin_sec", "dt_ser_db", "dt_snr_db"):
            assert DP._dist(got[field], field) == DP._dist(want[field], field), (name, field)
    assert len(sha) == 64 and sha == hashlib.sha256(PLAN.read_bytes()).hexdigest()


def test_plan_write_is_idempotent(tmp_path):
    """`--write` 幂等：两次生成逐字节一致（清单可复现）。"""
    outs = []
    for i in range(2):
        out = tmp_path / f"plan_{i}.json"
        rr = subprocess.run([sys.executable, "-s", str(ROOT / "scripts" / "dt_plan.py"),
                             "--write", str(out)],
                            capture_output=True, timeout=120,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]


@pytest.mark.parametrize("mutate,expect", [
    (lambda p: p.update(schema="nope/9"), "schema"),
    (lambda p: p["presets"]["eval_v5_repro"]["scenarios"].update({"fest": 0.9}), "和为 1"),
    (lambda p: p["presets"]["eval_v5_repro"].update(dt_ser_db={"kind": "normal", "lo": 0, "hi": 1}),
     "kind"),
    (lambda p: p["presets"]["eval_v5_repro"].update(dt_snr_db={"lo": 30, "hi": 20}), "lo"),
    (lambda p: p.update(version="v1"), "version"),
])
def test_load_plan_rejects_bad_manifest(tmp_path, mutate, expect):
    """坏清单必须 fail fast（schema/配比/分布 kind/区间/版本号）。"""
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    mutate(plan)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DP.DtPlanError, match=expect):
        DP.load_plan(bad)


def test_sample_dt_order_and_ranges():
    """采样顺序固定 ser -> snr -> leadin，且值落在清单区间（点分布取常数）。"""
    plan, sha = DP.load_plan(PLAN)
    rp = DP.resolve_preset(plan, "train_v1_repro", sha)
    rng = np.random.default_rng(0)
    s = DP.sample_dt(rng, rp)
    assert set(s) == {"ser_db", "snr_db", "leadin_sec"}
    assert 0.0 <= s["ser_db"] <= 10.0
    assert 15.0 <= s["snr_db"] <= 30.0
    assert s["leadin_sec"] == 1.5                    # v1 是点分布（精确复现历史口径）
    # 同 seed 同顺序 -> 同值
    rng2 = np.random.default_rng(0)
    assert DP.sample_dt(rng2, rp) == s
    # sample_ratio 与 aec_synth.pick_scene 同型
    rng3, rng4 = np.random.default_rng(5), np.random.default_rng(5)
    ratio = [rp["scenarios"][k] for k in DP.SCENARIOS]
    for _ in range(20):
        assert DP.sample_ratio(rng3, rp) == S.pick_scene(rng4, ratio)


# ---------------------------------------------------------------- (b) 默认路径不变
def test_default_path_schema_unchanged(tmp_path):
    """不给 --dt-plan：列集合与改动前逐字一致，且不产出 profile（0-5 硬约束）。"""
    nd, fd, qd, rir = _fixture(tmp_path)
    out = tmp_path / "out_legacy"
    rr = _run(nd, fd, qd, out, rir)
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    rows = _rows(out)
    assert list(rows[0].keys()) == LEGACY_COLUMNS
    assert not (out / "dt_plan_profile.json").exists()


def test_default_path_values_match_legacy_formula(tmp_path):
    """默认路径的 DT 参数仍按老公式抽（ser/snr 各一次 uniform；leadin 用 CLI 值）。

    复算随机流时必须照抄 aec_synth.main 的真实消耗序（含 `fit_len` 的**条件**抽样：源长于目标
    才抽裁剪偏移、短则循环补不抽）。本用例按源 wav 的实际帧数判定该分支，不靠"假设"。
    """
    nd, fd, qd, rir = _fixture(tmp_path)
    out = tmp_path / "out_legacy2"
    seed, num, dur = 11, 30, 2.0
    rr = _run(nd, fd, qd, out, rir, num=num, seed=seed, dur=str(dur),
              extra=["--dt-leadin-sec", "1.5"])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    rows = _rows(out)
    n = int(dur * SR)
    lens = {}
    for d in (nd, fd, qd):
        for p in d.glob("*.wav"):
            lens[p.name] = sf.info(str(p)).frames
    rng = np.random.default_rng(seed)
    for row in rows:
        rng.random()                                              # pick_scene
        rng.integers(len(list(nd.glob("*.wav"))))
        rng.integers(len(list(fd.glob("*.wav"))))
        rng.integers(len(list(qd.glob("*.wav"))))
        for key in ("near_file", "far_file", "noise_file"):        # fit_len 条件抽样
            if lens[row[key]] >= n:
                rng.integers(0, n + 1)
        rng.integers(1)                                           # get_rir（fixture 用 --rir-dir）
        ser, snr = rng.uniform(0.0, 10.0), rng.uniform(5.0, 20.0)  # CLI 默认区间
        rng.uniform(10.0, 80.0)                                    # delay
        rng.random()                                               # soft_clip
        assert float(row["ser_db"]) == pytest.approx(ser, abs=1e-12)
        assert float(row["snr_db"]) == pytest.approx(snr, abs=1e-12)
        if row["scenario"] == "dt":
            assert float(row["leadin_sec"]) == 1.5                 # CLI 值，非清单点分布


# ---------------------------------------------------------------- (c) 清单路径
def test_plan_path_columns_and_ranges(tmp_path):
    """清单路径：追加两列 + profile + DT 参数落在清单区间（serve 0-4 的留痕需求）。"""
    nd, fd, qd, rir = _fixture(tmp_path)
    out = tmp_path / "out_plan"
    rr = _run(nd, fd, qd, out, rir, num=60,
              extra=["--dt-plan", str(PLAN), "--dt-plan-preset", "eval_v5_repro"])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    rows = _rows(out)
    assert list(rows[0].keys()) == LEGACY_COLUMNS + ["dt_plan_version", "dt_plan_preset"]
    assert {r["dt_plan_version"] for r in rows} == {"dt_plan_v1"}
    assert {r["dt_plan_preset"] for r in rows} == {"eval_v5_repro"}
    dt = [r for r in rows if r["scenario"] == "dt"]
    assert dt, "本用例需要 DT 样本（num=60 + DT 占比 0.5）"
    assert {float(r["leadin_sec"]) for r in dt} == {2.0}          # eval_v5_repro 点分布
    assert all(20.0 <= float(r["snr_db"]) <= 30.0 for r in dt)
    assert all(0.0 <= float(r["ser_db"]) <= 10.0 for r in dt)
    prof = json.loads((out / "dt_plan_profile.json").read_text(encoding="utf-8"))
    assert prof["version"] == "dt_plan_v1" and len(prof["sha256"]) == 64
    assert prof["preset"] == "eval_v5_repro" and prof["seed"] == 7
    assert prof["sampled_order"] == ["scene", "near_idx", "far_idx", "noise_idx", "rir_draw",
                                     "ser_db", "snr_db", "leadin_sec", "delay_ms", "soft_clip"]
    assert prof["cli_args"]["soft_clip_prob"] == 0.3             # 不经清单的项也留痕


def test_plan_path_distribution_is_deterministic_and_bounded(tmp_path):
    """清单路径：同 seed 两次生成逐字节一致，且 DT 参数落在清单区间（同分布可复现）。

    逐值复算随机流放在 `scripts/verify_dt_plan.py`（它对的是真实调用序，硬编码进单测会随
    合成器演进而腐化）；单测这里守的是"可复现 + 区间合规"这两条不可回归的性质。
    """
    nd, fd, qd, rir = _fixture(tmp_path)
    plan, sha = DP.load_plan(PLAN)
    rp = DP.resolve_preset(plan, "train_v1_repro", sha)
    outs = []
    for i in range(2):
        out = tmp_path / f"dist_{i}"
        rr = _run(nd, fd, qd, out, rir, num=50, seed=13,
                  extra=["--dt-plan", str(PLAN), "--dt-plan-preset", "train_v1_repro"])
        assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
        outs.append((out / "manifest.csv").read_bytes())
    assert outs[0] == outs[1]                                     # 同 seed -> 同结果
    rows = _rows(tmp_path / "dist_0")
    dt = [r for r in rows if r["scenario"] == "dt"]
    assert dt
    assert all(rp["dt_ser_db"]["lo"] <= float(r["ser_db"]) <= rp["dt_ser_db"]["hi"] for r in dt)
    assert all(rp["dt_snr_db"]["lo"] <= float(r["snr_db"]) <= rp["dt_snr_db"]["hi"] for r in dt)
    assert {float(r["leadin_sec"]) for r in dt} == {rp["dt_leadin_sec"]["lo"]}


def test_plan_path_is_reproducible(tmp_path):
    """同 seed + 同清单 -> manifest 逐字节一致（0-5 的"可复现同分布"）。"""
    nd, fd, qd, rir = _fixture(tmp_path)
    outs = []
    for i in range(2):
        out = tmp_path / f"repro_{i}"
        rr = _run(nd, fd, qd, out, rir, num=25, seed=99,
                  extra=["--dt-plan", str(PLAN), "--dt-plan-preset", "eval_v5_repro"])
        assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
        outs.append((out / "manifest.csv").read_bytes())
    assert outs[0] == outs[1]


def test_dt_plan_list_prints_presets(tmp_path):
    """--dt-plan-list 只列 preset 不合成（下游选口径的入口）。"""
    nd, fd, qd, rir = _fixture(tmp_path)
    out = tmp_path / "out_list"
    rr = _run(nd, fd, qd, out, rir, extra=["--dt-plan", str(PLAN), "--dt-plan-list"])
    assert rr.returncode == 0
    text = rr.stdout.decode("utf-8", "replace")
    assert "train_v1_repro" in text and "eval_v5_repro" in text
    assert not out.exists()


# ---------------------------------------------------------------- (e) sim-RIR 适用性（t7）
VERIFY = ROOT / "scripts" / "verify_dt_plan.py"


def _load_verify():
    spec = importlib.util.spec_from_file_location("verify_dt_plan_t", VERIFY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_logrng_forwards_size_kwarg():
    """t7 修复：sim-RIR 路径会用 `uniform(lo, hi, size=n)`（pyroomacoustics 数组抽样），
    LogRng 必须完整转发位置/关键字参数——早期版本一遇 `size=` 就 TypeError、回放整条崩。"""
    V = _load_verify()
    rng = V.LogRng(np.random.default_rng, 0)
    vec = rng.uniform(3.0, 8.0, size=3)
    assert np.asarray(vec).shape == (3,)
    scal = rng.uniform(0.1, 0.45)                 # 标量调用照常
    assert isinstance(float(scal), float)
    assert [c[0] for c in rng.log] == ["uniform", "uniform"]
    assert all(isinstance(c[1], float) for c in rng.log)
    # 数组抽样后的流位置 == 3 次标量抽样（对账口径一致）
    a = np.random.default_rng(0)
    a.uniform(3.0, 8.0, size=3)
    b = np.random.default_rng(0)
    b.uniform(3.0, 8.0), b.uniform(3.0, 8.0), b.uniform(3.0, 8.0)
    assert a.random() == b.random()


def test_rir_mode_detected_from_profile():
    """g2：auto 判定按 profile 的 pool_sizes/pool_dirs 走（rir 有值=dir，否则 sim）。"""
    V = _load_verify()
    assert V.detect_rir_mode({"pool_sizes": {"rir": 60}, "pool_dirs": {"rir": "x"}}) == "dir"
    assert V.detect_rir_mode({"pool_sizes": {"rir": 0}, "pool_dirs": {"rir": None}}) == "sim"
    assert V.detect_rir_mode({"pool_sizes": {"rir": 0}}) == "sim"
    assert V.detect_rir_mode(None) == "sim"                     # 无从判定时保守按 sim
    assert V.detect_rir_mode({"pool_sizes": {"rir": 0}}, "dir") == "dir"   # 显式覆盖


def test_sim_rir_set_skips_offline_replay_with_note(tmp_path):
    """g1/g4：sim-RIR 集不跑离线重放（C 改走实测流核对），且不把"缺源目录"当数据失败。
    这里只验证"跳过 + 说明"这条路径，不真跑 pyroomacoustics。"""
    V = _load_verify()
    out = tmp_path / "sim_set"
    out.mkdir()
    (out / "manifest.csv").write_text(
        "name,scenario,rir_src,near_file,far_file,noise_file,delay_ms,ser_db,snr_db,"
        "soft_clip,leadin_sec,dt_plan_version,dt_plan_preset\n"
        "fileid_0_dt,dt,sim,n.wav,f.wav,q.wav,10.0,5.0,25.0,0,2.0,dt_plan_v1,eval_v5_repro\n",
        encoding="utf-8", newline="\n")
    (out / "dt_plan_profile.json").write_text(json.dumps({
        "version": "dt_plan_v1", "preset": "eval_v5_repro", "seed": 1, "sha256": "0" * 64,
        "pool_sizes": {"near": 3, "far": 3, "noise": 2, "rir": 0},   # rir=0 -> sim
        "cli_args": {"dur_sec": 8.0}}), encoding="utf-8", newline="\n")
    # 源目录环境变量故意不设：sim 核实跑不了 -> 只应报"需要环境变量"，而不是数据/清单失败
    for k in ("NEAR_DIR", "FAR_DIR", "NOISE_DIR"):
        os.environ.pop(f"DT_PLAN_{k}", None)
    prof = V.profile_of(out / "manifest.csv")
    assert V.detect_rir_mode(prof) == "sim"
    problems = V.check_evidence(
        V.read_manifest(out / "manifest.csv"),
        {"scenarios": {"fest": 0.25, "nst": 0.25, "dt": 0.5}}, prof, PLAN)
    assert len(problems) == 1 and "DT_PLAN_NEAR_DIR" in problems[0]


def test_dir_rir_set_runs_offline_replay(tmp_path):
    """g3：dir-RIR 集同样走 C 的实测流核对（两种 RIR 来源共用同一条检查路径）。"""
    V = _load_verify()
    nd, fd, qd, rir = _fixture(tmp_path)
    out = tmp_path / "dir_set"
    rr = _run(nd, fd, qd, out, rir, num=20, seed=5,
              extra=["--dt-plan", str(PLAN), "--dt-plan-preset", "train_v1_repro"])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    prof = V.profile_of(out / "manifest.csv")
    assert V.detect_rir_mode(prof) == "dir"
    plan, sha = DP.load_plan(PLAN)
    rp = DP.resolve_preset(plan, "train_v1_repro", sha)
    rows = V.read_manifest(out / "manifest.csv")
    # dir-RIR 集：源目录环境变量指向 fixture 的池（与 --rir-dir 由 DT_PLAN_RIR_DIR 提供）
    old = {k: os.environ.get(f"DT_PLAN_{k}") for k in ("NEAR_DIR", "FAR_DIR", "NOISE_DIR",
                                                       "RIR_DIR")}
    os.environ.update(DT_PLAN_NEAR_DIR=str(nd), DT_PLAN_FAR_DIR=str(fd),
                      DT_PLAN_NOISE_DIR=str(qd), DT_PLAN_RIR_DIR=str(rir))
    try:
        problems = V.check_evidence(rows, rp, prof, PLAN)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(f"DT_PLAN_{k}", None)
            else:
                os.environ[f"DT_PLAN_{k}"] = v
    assert problems == []


def test_sim_rir_end_to_end_via_fallback(tmp_path):
    """g1 端到端：不给 --rir-dir（sim 档）也能过 C——用 aec_synth 的无 pyroomacoustics
    兜底（rir_exponential，同样在共享 rng 流上抽数）复刻 sim 档结构，不必装依赖。"""
    V = _load_verify()
    nd, fd, qd, _ = _fixture(tmp_path)
    out = tmp_path / "sim_e2e"
    rr = _run(nd, fd, qd, out, None, num=20, seed=17, dur="1.0",
              extra=["--dt-plan", str(PLAN), "--dt-plan-preset", "train_v1_repro"])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    prof = V.profile_of(out / "manifest.csv")
    assert V.detect_rir_mode(prof) == "sim"            # 无 --rir-dir -> rir 池为 0
    assert (prof.get("pool_sizes") or {}).get("rir") == 0
    plan, sha = DP.load_plan(PLAN)
    rp = DP.resolve_preset(plan, "train_v1_repro", sha)
    rows = V.read_manifest(out / "manifest.csv")
    old = {k: os.environ.get(f"DT_PLAN_{k}") for k in ("NEAR_DIR", "FAR_DIR", "NOISE_DIR")}
    os.environ.update(DT_PLAN_NEAR_DIR=str(nd), DT_PLAN_FAR_DIR=str(fd),
                      DT_PLAN_NOISE_DIR=str(qd))
    try:
        problems = V.check_evidence(rows, rp, prof, PLAN)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(f"DT_PLAN_{k}", None)
            else:
                os.environ[f"DT_PLAN_{k}"] = v
    assert problems == [], problems


def test_legacy_manifest_has_no_plan_columns(tmp_path):
    """g5：非清单集（无 dt_plan_preset 列/无 profile）应被判为"不是按清单生成"。"""
    V = _load_verify()
    out = tmp_path / "legacy"
    out.mkdir()
    (out / "manifest.csv").write_text(
        "name,scenario,rir_src,near_file,far_file,noise_file,delay_ms,ser_db,snr_db,"
        "soft_clip,leadin_sec\n"
        "f0,dt,sim,n.wav,f.wav,q.wav,10.0,5.0,25.0,0,2.0\n", encoding="utf-8", newline="\n")
    rows = V.read_manifest(out / "manifest.csv")
    assert not rows[0].get("dt_plan_preset")
    assert V.profile_of(out / "manifest.csv") is None
    assert V.detect_rir_mode(None) == "sim"


# ---------------------------------------------------------------- (d) 冲突与坏输入
def test_explicit_conflicting_scenarios_is_rejected(tmp_path):
    """显式填了与清单不同的配比 -> 非零退出且不产出（不静默二选一）。"""
    nd, fd, qd, rir = _fixture(tmp_path)
    out = tmp_path / "out_conflict"
    rr = _run(nd, fd, qd, out, rir, extra=["--dt-plan", str(PLAN),
                                           "--dt-plan-preset", "eval_v5_repro",
                                           "--scenarios", "0.4,0.3,0.3"])
    assert rr.returncode != 0
    assert "冲突" in rr.stderr.decode("utf-8", "replace")
    assert not (out / "manifest.csv").exists()


def test_default_scenarios_are_overridden_by_plan(tmp_path):
    """只吃 CLI 默认配比时不报错（由清单覆盖），避免用户重复声明清单口径。"""
    nd, fd, qd, rir = _fixture(tmp_path)
    out = tmp_path / "out_default_ratio"
    rr = _run(nd, fd, qd, out, rir, num=20,
              extra=["--dt-plan", str(PLAN), "--dt-plan-preset", "eval_v5_repro"])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    assert len(_rows(out)) == 20


def test_preset_without_plan_is_rejected(tmp_path):
    """--dt-plan-preset 缺 --dt-plan：报错退出（不留隐性默认口径）。"""
    nd, fd, qd, rir = _fixture(tmp_path)
    out = tmp_path / "out_no_plan"
    rr = _run(nd, fd, qd, out, rir, extra=["--dt-plan-preset", "eval_v5_repro"])
    assert rr.returncode != 0
    assert "需要同时给" in rr.stderr.decode("utf-8", "replace")


def test_missing_plan_file_is_rejected(tmp_path):
    nd, fd, qd, rir = _fixture(tmp_path)
    out = tmp_path / "out_noplan"
    rr = _run(nd, fd, qd, out, rir, extra=["--dt-plan", str(tmp_path / "nope.json")])
    assert rr.returncode != 0
    assert "清单" in rr.stderr.decode("utf-8", "replace")
