"""t26 单测：`train_manifest` 的 **RIR 池留痕**（rir_dir + 清单名/sha256 + 池内文件数 + 集合指纹）。

背景（t13 发现的 schema 缺口）：near/far/noise 都记了池目录与清单 sha256，RIR 只记
`rir_src`/`rir_max_ms` ⇒ RIR 池污染只能靠命令行 + 人工核对发现。本文件锁死：

(a) **纯函数三态**（`DM.rir_provenance`）：无 RIR（仿真）/ 整目录枚举 / 清单驱动；
    集合指纹 `set_sha256` 与文件顺序无关、`list_sha256` == 清单文件内容 sha256；
(b) **清单驱动 = 隔离手段**（`DM.load_rir_pool`）：给了清单只吃清单内成员（池里多出来的
    RIR 不参与采样）+ 顺序 = 清单序；
(c) **07 集成**：无 `--rir-list` 时 manifest.data.rir 记 dir/枚举口径/文件数/集合指纹；
    给了 `--rir-list` 时额外记 清单名 + sha256 + `enumeration=list_order`，且 `n_files`
    等于**清单内**条数（不是池内总数）——这正是"事后能发现污染"的判据；
(d) **04 集成**：同一块（共享 `dm_dataset` 实现）；
(e) **09（级联）**：manifest 同源（断言放在 `tests/test_cascade_dm.py` 的冒烟里）。
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
TRAIN07 = ROOT / "exp" / "07_nn_aec" / "train.py"
TRAIN04 = ROOT / "exp" / "04_dccrn" / "train.py"
SR = 16000

import dm_shim as SH  # noqa: E402

DM = SH.dm_module()


def _wav(path, *, seed=0, n=4000, sr=SR, imp=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float32) if imp else (0.2 * rng.standard_normal(n)).astype(np.float32)
    if imp:
        y[0] = 1.0                                   # 冲激响应：RIR 用
    sf.write(path, y, sr)
    return path


def _fixture(root, *, n_rir=3, listed=None):
    """near/far/noise 单源池 + RIR 池（可选清单）。"""
    fx = {}
    for side, names in (("near", ("na_1.wav", "na_2.wav")), ("far", ("fb_1.wav", "fb_2.wav")),
                        ("noise", ("qc_1.wav", "qc_2.wav"))):
        d = root / side
        for i, nm in enumerate(names):
            _wav(d / nm, seed=10 + i + len(side))
        lst = root / f"{side}.txt"
        lst.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
        fx[f"{side}_dir"], fx[f"{side}_list"] = d, lst
    rir = root / "rir"
    rir_names = [f"rir_{i}.wav" for i in range(n_rir)]
    for i, nm in enumerate(rir_names):
        _wav(rir / nm, seed=i, n=512, imp=True)
    fx["rir"], fx["rir_names"] = rir, rir_names
    if listed is not None:
        fx["rir_list"] = root / "rir_list.txt"
        fx["rir_list"].write_text("".join(f"{rir_names[i]}\n" for i in listed), encoding="utf-8")
    return fx


def _val(root, n=2, seg=0.5, seed=5):
    rng = np.random.default_rng(seed)
    for sub in ("farend", "nearend", "mic"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        far = (0.3 * rng.standard_normal(int(seg * SR))).astype(np.float32)
        near = (0.05 * rng.standard_normal(int(seg * SR))).astype(np.float32)
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


def _argv07(fx, out, tmp, extra=()):
    a = [sys.executable, "-s", str(TRAIN07), "--dynamic-mixing", "--device", "cpu",
         "--num-workers", "0", "--batch-size", "2", "--dm-epoch-len", "4",
         "--dm-seg-sec", "0.5", "--epochs", "1",
         "--near-dir", str(fx["near_dir"]), "--near-list", str(fx["near_list"]),
         "--far-dir", str(fx["far_dir"]), "--far-list", str(fx["far_list"]),
         "--noise-dir", str(fx["noise_dir"]), "--noise-list", str(fx["noise_list"]),
         "--rir-dir", str(fx["rir"]), "--val", str(_val(tmp / "val")), "--out", str(out)]
    return a + list(extra)


def _argv04(fx, out, tmp, extra=()):
    a = [sys.executable, "-s", str(TRAIN04), "--dynamic-mixing", "--device", "cpu",
         "--num-workers", "0", "--batch-size", "2", "--epoch-len", "4",
         "--dm-seg-sec", "0.5", "--epochs", "1",
         "--near-dir", str(fx["near_dir"]), "--near-list", str(fx["near_list"]),
         "--far-dir", str(fx["far_dir"]), "--far-list", str(fx["far_list"]),
         "--noise-dir", str(fx["noise_dir"]), "--noise-list", str(fx["noise_list"]),
         "--rir-dir", str(fx["rir"]), "--val", str(_val(tmp / "val")), "--out", str(out)]
    return a + list(extra)


def _run(argv):
    return subprocess.run(argv, capture_output=True, timeout=900,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def _manifest(out, tag):
    return json.loads((out / tag / "train_manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- (a) 纯函数三态
def test_rir_provenance_three_states(tmp_path):
    files = [_wav(tmp_path / "r" / "b.wav", n=512, imp=True),
             _wav(tmp_path / "r" / "a.wav", n=512, imp=True)]
    # 仿真（无 RIR 目录/文件）
    sim = DM.rir_provenance(None, None, None, max_ms=100.0)
    assert (sim["src"], sim["dir"], sim["list"], sim["n_files"], sim["set_sha256"]) == \
        ("simulated", None, None, 0, None)
    assert sim["enumeration"] == "n/a(simulated)" and sim["max_ms"] == 100.0
    # 整目录枚举
    d = DM.rir_provenance(tmp_path / "r", None, files, max_ms=100.0)
    assert (d["src"], d["enumeration"]) == ("dir", "list_pool_byte_order")
    assert d["dir"] == str(tmp_path / "r") and d["list"] is None and d["list_sha256"] is None
    assert d["n_files"] == 2
    # 集合指纹与顺序无关（跨平台稳定 ⇒ 可 diff 两次 run 的 RIR 池）
    assert d["set_sha256"] == DM.rir_provenance(tmp_path / "r", None, files[::-1],
                                                max_ms=100.0)["set_sha256"]
    assert len(d["set_sha256"]) == 64
    # 只有文件列表、没给目录 → 从文件推目录
    assert DM.rir_provenance(None, None, files)["dir"] == str(tmp_path / "r")
    # 清单驱动：清单名 + sha256（== 清单文件内容 sha256）
    lst = tmp_path / "rir_list.txt"
    lst.write_text("a.wav\nb.wav\n", encoding="utf-8")
    lp = DM.rir_provenance(tmp_path / "r", lst, files, max_ms=100.0)
    assert (lp["src"], lp["enumeration"]) == ("list", "list_order")
    assert lp["list"] == str(lst)
    assert lp["list_sha256"] == hashlib.sha256(lst.read_bytes()).hexdigest()
    # 指纹按**池内相对名**算：跟着实际使用的集合走，少一条就变
    # （换挂载点的不变性由 test_rir_set_sha256_is_mount_point_independent 专测）
    assert DM.rir_provenance(tmp_path / "r", None, files[:1])["set_sha256"] != d["set_sha256"]
    assert d["set_sha256_basis"].startswith("池内相对路径")


def test_rir_set_sha256_is_mount_point_independent(tmp_path):
    """**换服务器/换挂载点**：同一套 RIR 放在不同绝对路径下，指纹必须一致（否则假警报）。

    本轮的"新服务器续跑"正是这个场景：池可能 rsync 到别的根目录，若指纹含绝对路径，
    `set_sha256` 会变，把"池动过"误报成噪音。
    """
    rel = ("b.wav", "a.wav")                      # 故意非字节序：只比集合指纹
    fa, fb = [], []
    for base, out in ((tmp_path / "serverA" / "rirs_train70", fa),
                      (tmp_path / "serverB" / "deep" / "rirs_train70", fb)):
        for nm in rel:
            out.append(_wav(base / nm, n=512, imp=True))
    sha_a = DM.rir_provenance(tmp_path / "serverA" / "rirs_train70", None, fa)["set_sha256"]
    sha_b = DM.rir_provenance(tmp_path / "serverB" / "deep" / "rirs_train70", None, fb)["set_sha256"]
    assert sha_a == sha_b, "指纹不应依赖绝对路径（换挂载点不该报警）"


# ---------------------------------------------------------------- (c2) t27 顺序护栏三结局
def test_rir_list_order_guard_three_outcomes(tmp_path, capsys):
    """① 双一致 → 静默；② 集合同顺序异 → warn（映射变了）；③ 集合异 → 既有行为（不挡开跑）。"""
    fx = _fixture(tmp_path, n_rir=3)                  # rir_0/1/2.wav（名字天然字节序）
    pool_dir = fx["rir"]
    pool = [p.name for p in DM.list_pool(pool_dir)]
    # ① 清单序 == 目录字节序 → 无告警、order_matches_pool=True
    same = tmp_path / "same.txt"
    same.write_text("".join(f"{n}\n" for n in pool), encoding="utf-8")
    files = DM.load_rir_pool(pool_dir, same)          # 加载期静默
    out = capsys.readouterr().out
    assert "[warn]" not in out
    got = DM.rir_provenance(pool_dir, same, files)
    assert got["order_matches_pool"] is True and "warnings" not in got
    assert got["n_in_pool"] == 3 and got["extra_pool_files"] == 0
    # ② 集合同、顺序不同（逆序）→ 当场 warn + manifest 记录 False
    rev = tmp_path / "rev.txt"
    rev.write_text("".join(f"{n}\n" for n in pool[::-1]), encoding="utf-8")
    files2 = DM.load_rir_pool(pool_dir, rev)
    out2 = capsys.readouterr().out
    assert "[dm] [warn]" in out2 and "顺序" in out2 and "index->文件 映射" in out2
    got2 = DM.rir_provenance(pool_dir, rev, files2)
    assert got2["order_matches_pool"] is False
    assert got2["n_files"] == got2["n_in_pool"] == 3 and got2["extra_pool_files"] == 0
    assert any("顺序" in w for w in got2["warnings"])
    assert got2["set_sha256"] == got["set_sha256"], "集合同 ⇒ 集合指纹必须相同"
    # ③ 集合不同（池里多一条）→ 不报错（清单驱动的隔离用法），但提示"只用了 n/m 条"
    sub = tmp_path / "sub.txt"
    sub.write_text("".join(f"{n}\n" for n in pool[:2]), encoding="utf-8")
    files3 = DM.load_rir_pool(pool_dir, sub)
    out3 = capsys.readouterr().out
    assert "[dm] [warn]" in out3 and "只使用清单内 2/3 条" in out3
    assert [p.name for p in files3] == pool[:2]
    got3 = DM.rir_provenance(pool_dir, sub, files3)
    assert got3["order_matches_pool"] is None and got3["extra_pool_files"] == 1
    assert got3["n_files"] == 2 and got3["n_in_pool"] == 3
    # 清单里有池外成员 → 既有 fail fast 不变（load_pool 拦下，不进护栏）
    bad = tmp_path / "bad2.txt"
    bad.write_text("nope.wav\n", encoding="utf-8")
    with pytest.raises(DM.DMPoolError, match="不在池"):
        DM.load_rir_pool(pool_dir, bad)
    # warn=False 可静音（给脚本/一次性核对用）
    files4 = DM.load_rir_pool(pool_dir, rev, warn=False)
    assert "[warn]" not in capsys.readouterr().out and len(files4) == 3


def test_load_rir_pool_list_driven_isolation(tmp_path):
    """清单驱动 = 隔离手段：池里多出来的 RIR 不参与采样，顺序 = 清单序。"""
    fx = _fixture(tmp_path, n_rir=3, listed=(2, 0))
    got = DM.load_rir_pool(fx["rir"], fx["rir_list"])
    assert [p.name for p in got] == ["rir_2.wav", "rir_0.wav"], "顺序 = 清单序（= 采样序）"
    assert len(DM.load_rir_pool(fx["rir"])) == 3, "不给清单 = 整目录枚举"
    assert DM.load_rir_pool(None) is None
    # 清单里有池内不存在的成员 → fail fast（不静默漏用）
    bad = tmp_path / "bad.txt"
    bad.write_text("nope.wav\n", encoding="utf-8")
    with pytest.raises(DM.DMPoolError, match="不在池"):
        DM.load_rir_pool(fx["rir"], bad)


# ---------------------------------------------------------------- (c) 07 集成
def test_nnaec_manifest_records_rir_dir_when_no_list(tmp_path):
    fx = _fixture(tmp_path, n_rir=3)
    out = tmp_path / "out"
    rr = _run(_argv07(fx, out, tmp_path))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    man = _manifest(out, "nnaec_g0.5_j10_dm")
    rir = man["data"]["rir"]
    assert rir["src"] == "dir" and rir["dir"] == str(fx["rir"])
    assert rir["list"] is None and rir["list_sha256"] is None
    assert rir["n_files"] == 3 and rir["enumeration"] == "list_pool_byte_order"
    assert rir["max_ms"] == 100.0 and len(rir["set_sha256"]) == 64
    assert man["data"]["n_files"]["rir"] == 3
    # 兼容键保留（既有审计脚本读 rir_src/rir_max_ms）
    assert man["data"]["rir_src"] == "dir" and man["data"]["rir_max_ms"] == 100.0
    # 同池同参数两次 run：集合指纹一致（可作"RIR 池没变过"的判据）
    out2 = tmp_path / "out2"
    rr2 = _run(_argv07(fx, out2, tmp_path))
    assert rr2.returncode == 0
    assert _manifest(out2, "nnaec_g0.5_j10_dm")["data"]["rir"]["set_sha256"] == rir["set_sha256"]


def test_nnaec_manifest_records_rir_list_and_isolation(tmp_path):
    """给了 --rir-list：manifest 记清单名 + sha256；n_files = **清单内**条数（污染的判据）。"""
    fx = _fixture(tmp_path, n_rir=3, listed=(0, 1))
    out = tmp_path / "out"
    rr = _run(_argv07(fx, out, tmp_path, ["--rir-list", str(fx["rir_list"])]))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    rir = _manifest(out, "nnaec_g0.5_j10_dm")["data"]["rir"]
    assert rir["src"] == "list" and rir["enumeration"] == "list_order"
    assert rir["list"] == str(fx["rir_list"])
    assert rir["list_sha256"] == hashlib.sha256(fx["rir_list"].read_bytes()).hexdigest()
    assert rir["n_files"] == 2, "只算清单内成员（池内 3 条 ⇒ 能看出池被裁过/污染过）"
    assert rir["set_sha256"] == DM.rir_provenance(
        fx["rir"], fx["rir_list"],
        [fx["rir"] / "rir_0.wav", fx["rir"] / "rir_1.wav"], max_ms=100.0)["set_sha256"]
    # 清单里塞一个池外/污染的 RIR → 立刻 fail fast（这正是 t13 想要的能力）
    bad = tmp_path / "polluted.txt"
    bad.write_text("rir_0.wav\n../eval_axis/rir_evil.wav\n", encoding="utf-8")
    rr_bad = _run(_argv07(fx, tmp_path / "out_bad", tmp_path, ["--rir-list", str(bad)]))
    assert rr_bad.returncode != 0 and "不在池" in rr_bad.stderr.decode("utf-8", "replace")


def test_nnaec_rir_order_guard_fires_through_cli(tmp_path):
    """t27 端到端：清单序 ≠ 目录字节序时，**开跑横幅就 warn**，且 manifest 记 False。

    这是本轮 casc 续跑依赖的那条不变式（清单序 ≡ 目录枚举序 ⇒ 逐样本对齐）：一旦有人把清单
    重排/重生成成别的顺序，run 必须当场说出来，而不是静默换一组 RIR。
    """
    fx = _fixture(tmp_path, n_rir=3)
    pool = [p.name for p in DM.list_pool(fx["rir"])]
    rev = tmp_path / "rev.txt"
    rev.write_text("".join(f"{n}\n" for n in pool[::-1]), encoding="utf-8")
    out = tmp_path / "out_rev"
    rr = _run(_argv07(fx, out, tmp_path, ["--rir-list", str(rev)]))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    sout = rr.stdout.decode("utf-8", "replace")
    assert "[dm] [warn]" in sout and "顺序" in sout and "index->文件 映射" in sout
    rir = _manifest(out, "nnaec_g0.5_j10_dm")["data"]["rir"]
    assert rir["order_matches_pool"] is False
    assert rir["n_files"] == rir["n_in_pool"] == 3 and rir["extra_pool_files"] == 0
    assert any("顺序" in w for w in rir["warnings"])
    # 反例（清单恰好 = 目录字节序）→ 静默（只按 RIR 护栏的特征文本断言，避免误伤
    # 其它既有 warn，如 --dm-seg-sec 与清单 dur_sec 不一致）+ manifest True
    fwd = tmp_path / "fwd.txt"
    fwd.write_text("".join(f"{n}\n" for n in pool), encoding="utf-8")
    out_ok = tmp_path / "out_ok"
    rr2 = _run(_argv07(fx, out_ok, tmp_path, ["--rir-list", str(fwd)]))
    assert rr2.returncode == 0
    sout_ok = rr2.stdout.decode("utf-8", "replace")
    assert "index->文件 映射" not in sout_ok and "RIR 清单" not in sout_ok
    rir_ok = _manifest(out_ok, "nnaec_g0.5_j10_dm")["data"]["rir"]
    assert rir_ok["order_matches_pool"] is True and "warnings" not in rir_ok


# ---------------------------------------------------------------- (d) 04 集成
def test_dccrn_manifest_records_rir_block(tmp_path):
    fx = _fixture(tmp_path, n_rir=3, listed=(0, 1, 2))
    out = tmp_path / "out"
    rr = _run(_argv04(fx, out, tmp_path, ["--rir-list", str(fx["rir_list"])]))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")[-2000:]
    rir = _manifest(out, "dccrn_dm")["data"]["rir"]
    assert rir["src"] == "list" and rir["n_files"] == 3
    assert rir["list_sha256"] == hashlib.sha256(fx["rir_list"].read_bytes()).hexdigest()
    assert rir["dir"] == str(fx["rir"]) and rir["max_ms"] == 100.0
