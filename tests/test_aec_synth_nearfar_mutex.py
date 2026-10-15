"""阶段0 / 工作项 0-3 单测：`aec_synth.py` near/far 分池互斥（--nearfar-mutex）。

覆盖三件事：
(a) **互斥开启时 near/far 无同源**——pool 档的同一目录 / 目录嵌套 / 文件复用（硬链接）
    三档判据，utterance 档的同名副本判据；违例**失败且不产出任何 wav/manifest**
    （main 在任何写盘之前 fail fast）。合法（两个互斥池）时通过，且 manifest 逐样本
    near_file != far_file、两池文件名不相交。
    判据的**已知边界**（改名副本内容级不可判）也各有一条用例固化，避免后续误以为已覆盖。
(b) **默认参数下输出与改动前一致（回归）**——与 `tests/golden/aec_synth_drift0/`
    （P2-1 时期固化的默认路径基准，早于本次改动）逐字节对比；并验证显式
    `--nearfar-mutex off` 与缺省完全一致、缺省 manifest 不多列（schema 不变）。
(c) **off 档零侵入**：不做 glob/stat 前置校验，旧用法（near_dir 与 far_dir 同目录）
    在缺省参数下仍可跑通——默认行为不变是 0-3 的硬约束（REMEDIATION_PLAN_V3.1.md §0-3）。

注：本文件用子进程跑 CLI（与 test_aec_synth.py / test_delay_drift.py 同法），
判定函数另用直调覆盖；子进程统一 `PYTHONIOENCODING=utf-8` + `encoding="utf-8"`，
使中文断言不随本机代码页漂移。
"""
import csv
import os
import shutil
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

SR = 16000
SCRIPT = Path(S.__file__)
SUBS = ("farend", "nearend", "mic")


# ---------------------------------------------------------------- fixture / 驱动
def _wav(path, n=24000, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sf.write(path, (0.2 * rng.standard_normal(n)).astype(np.float32), SR)


def _rir_dir(root):
    """确定性自备 RIR：避开 pyroomacoustics 的环境依赖与耗时（RIR 与本用例无关）。"""
    d = root / "rir"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    h = rng.standard_normal(512) * np.exp(-np.arange(512) / (0.02 * SR))
    h[0] = 1.0
    sf.write(d / "rir.wav", (h / np.linalg.norm(h) * 0.7).astype(np.float32), SR)
    return d


def _pools(root):
    """两个**互斥**池（文件名/主名均不重叠）+ 噪声目录 + RIR 目录。"""
    nd, fd, qd = root / "near", root / "far", root / "noise"
    for i, nm in enumerate(("n_0001.wav", "n_0002.wav")):
        _wav(nd / nm, seed=10 + i)
    for i, nm in enumerate(("f_0001.wav", "f_0002.wav")):
        _wav(fd / nm, seed=20 + i)
    _wav(qd / "q_0001.wav", seed=30)
    return nd, fd, qd, _rir_dir(root)


def _run(nd, fd, qd, out, rir=None, extra=(), num=2, dur="1.0", seed=3):
    argv = [sys.executable, "-s", str(SCRIPT), "--num", str(num), "--dur-sec", dur,
            "--seed", str(seed), "--near-dir", str(nd), "--far-dir", str(fd),
            "--noise-dir", str(qd), "--out-dir", str(out)]
    if rir is not None:
        argv += ["--rir-dir", str(rir)]
    rr = subprocess.run(argv + list(extra), capture_output=True, timeout=600,
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return rr


def _rows(out):
    with open(out / "manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _no_output(out):
    """违例必须 fail fast：不建产物目录、不写 manifest。"""
    return not (out / "manifest.csv").exists() and not (out / "mic").exists()


# ---------------------------------------------------------------- (a) 判定函数直调
def test_off_level_is_noop_even_for_shared_dir(tmp_path):
    """off（缺省）档不校验：同目录、嵌套、硬链接都放行（旧行为逐比特保留）。"""
    nd, _, _, _ = _pools(tmp_path)
    for near, far in ((nd, nd), (nd, nd / "x")):
        assert S.check_nearfar_mutex(near, far, "off") is None
        assert S.check_nearfar_mutex(near, far) is None       # 缺省参数 = off


def test_pool_level_same_dir_conflict(tmp_path):
    nd, _, _, _ = _pools(tmp_path)
    reasons = S.nearfar_pool_conflicts(nd, nd)
    assert len(reasons) == 1 and "同一目录" in reasons[0]
    with pytest.raises(S.NearFarMutexError, match="同一目录"):
        S.check_nearfar_mutex(nd, nd, "pool")


def test_pool_level_nested_dir_conflict(tmp_path):
    """目录嵌套 = far 池是 near 池子集（旧文档示例的写法）：说话人双端复用，判违例。"""
    nd, _, _, _ = _pools(tmp_path)
    sub = nd / "far_sub"
    _wav(sub / "f_other.wav", seed=99)
    reasons = S.nearfar_pool_conflicts(nd, sub)
    assert "嵌套" in reasons[0] and "同一目录" not in reasons[0]
    with pytest.raises(S.NearFarMutexError, match="嵌套"):
        S.check_nearfar_mutex(nd, sub, "pool")


def test_pool_level_shared_file_identity_conflict(tmp_path):
    """同一条语音换个文件名放进 far 池（硬链接）：路径不同但文件同一性相同 → 判违例。"""
    nd, fd, _, _ = _pools(tmp_path)
    link = fd / "f_copy_of_n1.wav"
    try:
        os.link(nd / "n_0001.wav", link)
    except (OSError, NotImplementedError):
        pytest.skip("本文件系统不支持硬链接")
    reasons = S.nearfar_pool_conflicts(nd, fd)
    assert len(reasons) == 1 and "复用同一文件" in reasons[0]
    with pytest.raises(S.NearFarMutexError, match="复用同一文件"):
        S.check_nearfar_mutex(nd, fd, "pool")


def test_pool_level_disjoint_pools_pass(tmp_path):
    nd, fd, _, _ = _pools(tmp_path)
    assert S.nearfar_pool_conflicts(nd, fd) == []
    assert S.nearfar_utterance_conflicts(sorted(nd.glob("**/*.wav")),
                                         sorted(fd.glob("**/*.wav"))) == []
    assert S.check_nearfar_mutex(nd, fd, "pool") is None
    assert S.check_nearfar_mutex(nd, fd, "utterance") is None


def test_utterance_level_catches_same_stem_copy(tmp_path):
    """utterance 档：同一 utterance 主名在两池同时出现（跨目录复制的常见形态）。

    分层语义：pool 档只看路径/文件项，看不到副本（不同 inode）→ 通过；
    utterance 档按主名判 → 拦下。
    """
    nd, fd, _, _ = _pools(tmp_path)
    shutil.copy2(nd / "n_0001.wav", fd / "n_0001.wav")
    assert S.nearfar_pool_conflicts(nd, fd) == []              # pool 档看不见副本
    assert S.check_nearfar_mutex(nd, fd, "pool") is None
    assert S.nearfar_utterance_conflicts(sorted(nd.glob("**/*.wav")),
                                         sorted(fd.glob("**/*.wav"))) == ["n_0001"]
    with pytest.raises(S.NearFarMutexError, match="utterance 主名"):
        S.check_nearfar_mutex(nd, fd, "utterance")


def test_renamed_copy_stays_out_of_scope(tmp_path):
    """**已知边界固化**：改名副本（同内容、不同文件名）不在自动判据内。

    判据有意止步于"路径/文件项 + utterance 主名"——全量大池每次启动哈希代价高；
    内容级去重按 0-2 的 sha256 清单法离线做（见 nearfar_pool_conflicts 注释）。
    本用例把该边界写成断言，防止后续把它当"已覆盖"。
    """
    nd, fd, _, _ = _pools(tmp_path)
    shutil.copy2(nd / "n_0001.wav", fd / "f_renamed_copy.wav")
    assert S.nearfar_pool_conflicts(nd, fd) == []
    assert S.check_nearfar_mutex(nd, fd, "utterance") is None


def test_pool_level_alias_forms_of_same_dir(tmp_path):
    """同目录的别名写法：大小写（Windows）+ 尾部分隔符 + 相对片段，全部识破。"""
    nd, _, _, _ = _pools(tmp_path)
    assert S.nearfar_pool_conflicts(nd, Path(str(nd) + os.sep)) == \
        S.nearfar_pool_conflicts(nd, nd)
    assert "同一目录" in S.nearfar_pool_conflicts(nd, nd / ".")[0]
    upper = Path(str(nd).upper())
    if not upper.exists():
        pytest.skip("本文件系统大小写敏感：大小写别名不适用")
    assert "同一目录" in S.nearfar_pool_conflicts(nd, upper)[0]


def test_unknown_level_raises_valueerror(tmp_path):
    nd, fd, _, _ = _pools(tmp_path)
    with pytest.raises(ValueError, match="未知"):
        S.check_nearfar_mutex(nd, fd, "strict")


# ---------------------------------------------------------------- (a) CLI 端到端
def test_cli_pool_same_dir_fails_fast(tmp_path):
    nd, _, qd, rir = _pools(tmp_path)
    out = tmp_path / "out_same"
    rr = _run(nd, nd, qd, out, rir, extra=["--nearfar-mutex", "pool"])
    assert rr.returncode != 0
    err = rr.stderr.decode("utf-8")
    assert "分池互斥违例" in err and "同一目录" in err and "--nearfar-mutex=pool" in err
    assert _no_output(out), "违例必须在写盘前终止（不得产出半套数据）"


def test_cli_utterance_level_rejects_same_stem_copy(tmp_path):
    nd, fd, qd, rir = _pools(tmp_path)
    shutil.copy2(nd / "n_0002.wav", fd / "n_0002.wav")
    out = tmp_path / "out_u"
    rr = _run(nd, fd, qd, out, rir, extra=["--nearfar-mutex", "utterance"])
    assert rr.returncode != 0
    assert "utterance 主名" in rr.stderr.decode("utf-8")
    assert _no_output(out)


def test_cli_disjoint_pools_pass_and_manifest_records_level(tmp_path):
    nd, fd, qd, rir = _pools(tmp_path)
    out = tmp_path / "out_ok"
    rr = _run(nd, fd, qd, out, rir, extra=["--nearfar-mutex", "pool"])
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    stdout = rr.stdout.decode("utf-8")
    assert "near/far 分池互斥：pool" in stdout
    rows = _rows(out)
    assert len(rows) == 2 and all(r["nearfar_mutex"] == "pool" for r in rows)
    # 互斥成立：逐样本不同源，且两池文件名在整表内不相交
    for r in rows:
        assert r["near_file"] != r["far_file"]
    assert {r["near_file"] for r in rows}.isdisjoint({r["far_file"] for r in rows})
    for sub in SUBS:
        assert len(list((out / sub).glob("*.wav"))) == 2


def test_cli_invalid_level_rejected(tmp_path):
    nd, fd, qd, rir = _pools(tmp_path)
    rr = _run(nd, fd, qd, tmp_path / "out_bad", rir, extra=["--nearfar-mutex", "strict"])
    assert rr.returncode == 2                      # argparse choices 拒绝
    assert "nearfar-mutex" in rr.stderr.decode("utf-8")


# ---------------------------------------------------------------- (c) 默认零侵入
def test_cli_default_off_allows_shared_dir_and_keeps_schema(tmp_path):
    """缺省（off）：旧用法（near_dir == far_dir）照跑，且 manifest 不多列。"""
    nd, _, qd, rir = _pools(tmp_path)
    out = tmp_path / "out_default"
    rr = _run(nd, nd, qd, out, rir)
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    rows = _rows(out)
    assert len(rows) == 2
    for r in rows:
        assert "nearfar_mutex" not in r
    # 缺省 manifest 不得因 0-3 变化：原有 11 列名前缀不变、无 nearfar_mutex 列
    # （只做前缀断言，后续工作项追加列——如 0-5 的 DT 比例清单——不误伤本用例）
    header = (out / "manifest.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:11] == ["name", "scenario", "rir_src", "near_file", "far_file",
                           "noise_file", "delay_ms", "ser_db", "snr_db", "soft_clip",
                           "leadin_sec"]
    assert "nearfar_mutex" not in header


# ---------------------------------------------------------------- (b) 回归：默认输出不变
_GOLDEN = Path(__file__).resolve().parent / "golden" / "aec_synth_drift0"
_G_NUM, _G_DUR, _G_SEED = 2, 0.5, 7


def _golden_fixture(root):
    """与 golden 基准生成时同参的确定性输入（见 test_delay_drift._golden_inputs）。"""
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for sub in ("near", "far", "noise"):
        d = root / sub
        d.mkdir()
        for i in range(2):
            y = (0.2 * rng.standard_normal(12000)).astype(np.float32)
            sf.write(d / f"{sub}{i}.wav", y, SR)
    return root


def test_default_matches_prechange_golden_baseline(tmp_path):
    """默认参数（--nearfar-mutex 缺省）输出与**改动前**固化基准逐字节一致。

    基准 `tests/golden/aec_synth_drift0/` 是 P2-1 时期固化的默认路径输出（早于 0-3 改动），
    所以本用例是 0-3 "默认行为完全不变" 的直接证据：manifest + 三路 wav 全部按字节比对。
    """
    fx = _golden_fixture(tmp_path / "fx")
    assert _GOLDEN.is_dir(), f"缺少黄金基准 {_GOLDEN}"
    expect = _GOLDEN / "expected"
    out = tmp_path / "out_g"
    rr = _run(fx / "near", fx / "far", fx / "noise", out, _GOLDEN / "rir",
              num=_G_NUM, dur=str(_G_DUR), seed=_G_SEED)
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    assert (out / "manifest.csv").read_bytes() == (expect / "manifest.csv").read_bytes()
    names = sorted(p.name for p in (expect / "mic").glob("*.wav"))
    assert len(names) == _G_NUM
    for sub in SUBS:
        for name in names:
            assert (out / sub / name).read_bytes() == (expect / sub / name).read_bytes(), \
                f"默认路径输出与改动前基准不一致：{sub}/{name}"


def test_explicit_off_equals_default_bytes(tmp_path):
    """显式 --nearfar-mutex off 与缺省参数逐字节等价（新增参数不改默认路径）。"""
    nd, fd, qd, rir = _pools(tmp_path)
    o1, o2 = tmp_path / "d", tmp_path / "e"
    r1 = _run(nd, fd, qd, o1, rir)
    r2 = _run(nd, fd, qd, o2, rir, extra=["--nearfar-mutex", "off"])
    assert r1.returncode == 0 and r2.returncode == 0
    assert (o1 / "manifest.csv").read_bytes() == (o2 / "manifest.csv").read_bytes()
    for sub in SUBS:
        for f in sorted((o1 / sub).glob("*.wav")):
            assert f.read_bytes() == (o2 / sub / f.name).read_bytes(), f"{sub}/{f.name}"


def test_mutex_on_is_pure_guard_for_disjoint_pools(tmp_path):
    """开启互斥只做校验、不扰动合成：互斥池上 on/off 的 wav 与底座字段逐字节一致。

    0-4 会带 `--nearfar-mutex` 重生成 eval_v8，本用例保证"开了约束"本身不改变数据内容
    （rng 序列、场景/文件/时延/SER/SNR 全部不动，manifest 只多一个留痕列）。
    """
    nd, fd, qd, rir = _pools(tmp_path)
    o1, o2 = tmp_path / "off", tmp_path / "on"
    r1 = _run(nd, fd, qd, o1, rir)
    r2 = _run(nd, fd, qd, o2, rir, extra=["--nearfar-mutex", "utterance"])
    assert r1.returncode == 0 and r2.returncode == 0
    for sub in SUBS:
        for f in sorted((o1 / sub).glob("*.wav")):
            assert f.read_bytes() == (o2 / sub / f.name).read_bytes(), f"{sub}/{f.name}"
    base = [{k: v for k, v in r.items() if k != "nearfar_mutex"} for r in _rows(o2)]
    assert base == _rows(o1)
    assert all(r["nearfar_mutex"] == "utterance" for r in _rows(o2))
