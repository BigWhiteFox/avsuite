"""0-4 单测：语音池 reader 三分割 + eval 侧 near/far 互斥池（`scripts/split_speech_pool.py`）。

覆盖五件事：
(a) **哈希切分是纯函数**：同 reader 恒定落桶，桶划分穷尽且互斥（构造性零重叠）；
(b) **名单可复现**：同输入两次 `--write` 逐字节一致（UTF-8/LF/无 BOM），
    near∪far 恰好覆盖 holdout reader、near∩far=∅、成员都在源池；
(c) **验收能抓坏名单**：手改（把 far 名单塞进 near）→ 退出码 1，并报出重叠/归属问题；
(d) **输入错误与不通过分开**：源池不存在 / 名单缺失 → 退出码 2；
(e) **物化池目录**：`--materialize-root` 后池目录内容与名单一致、验收 0（硬链接或复制均可）。

用法：`pytest tests/test_split_speech_pool.py -q`
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import split_speech_pool as SP  # noqa: E402

SCRIPT = Path(SP.__file__)
PY = sys.executable


def _mk_pool(root: Path, n_readers: int = 60, per_reader: int = 3) -> Path:
    """合成 clean 池：reader id 用 5 位数（与 DNS 命名一致），文件内容不参与切分。"""
    d = root / "clean"
    d.mkdir(parents=True, exist_ok=True)
    for r in range(1, n_readers + 1):
        for i in range(per_reader):
            p = d / f"book_000{r:02d}_chp_0001_reader_{r:05d}_{i}.wav"
            p.write_bytes(b"RIFF")
    return d


def _run(*argv: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-s", str(SCRIPT), *argv], capture_output=True,
                          timeout=timeout,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def _write_lists(tmp_path: Path, clean: Path) -> Path:
    out = tmp_path / "lists"
    rr = _run("--clean-dir", str(clean), "--out-dir", str(out))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    return out


def _read(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]


# ---------------------------------------------------------------- (a) 切分纯函数
def test_bucket_split_is_deterministic_and_exhaustive():
    ids = [f"{i:05d}" for i in range(1, 401)]
    assert all(SP.reader_split(r) == SP.reader_split(r) for r in ids)
    parts = {s: [r for r in ids if SP.reader_split(r) == s] for s in SP.SPLIT_NAMES}
    assert sorted(sum(parts.values(), [])) == ids          # 穷尽且互斥
    # holdout reader 的 near/far 侧同样是划分
    hold = parts[SP.HOLDOUT_NAME]
    sides = {s: [r for r in hold if SP.reader_side(r) == s] for s in (SP.SIDE_NEAR, SP.SIDE_FAR)}
    assert sorted(sum(sides.values(), [])) == sorted(hold)
    assert sides[SP.SIDE_NEAR] and sides[SP.SIDE_FAR]      # 两侧非空


def test_reader_of_requires_named_reader():
    assert SP.reader_of("book_00000_chp_0009_reader_06709_3.wav") == "06709"
    assert SP.reader_of("no_reader_marker.wav") is None


# ---------------------------------------------------------------- (b) 名单可复现
def test_write_lists_disjoint_and_reproducible(tmp_path):
    clean = _mk_pool(tmp_path)
    out_a, out_b = _write_lists(tmp_path, clean), tmp_path / "lists2"
    assert _run("--clean-dir", str(clean), "--out-dir", str(out_b)).returncode == 0
    for f in list(SP.READER_LISTS.values()) + list(SP.SIDE_LISTS.values()) + [SP.META_NAME]:
        assert (out_a / f).read_bytes() == (out_b / f).read_bytes(), f     # 幂等
        assert not (out_a / f).read_bytes().startswith(b"\xef\xbb\xbf")
        assert b"\r" not in (out_a / f).read_bytes()

    near = _read(out_a / SP.SIDE_LISTS[SP.SIDE_NEAR])
    far = _read(out_a / SP.SIDE_LISTS[SP.SIDE_FAR])
    assert near and far and not (set(near) & set(far))
    pool = {p.name for p in clean.glob("*.wav")}
    assert set(near) | set(far) <= pool
    nr = {SP.reader_of(n) for n in near}
    fr = {SP.reader_of(n) for n in far}
    assert not (nr & fr)
    hold = set(_read(out_a / SP.READER_LISTS[SP.HOLDOUT_NAME]))
    assert nr | fr == hold


def test_verify_passes_on_generated_lists(tmp_path):
    clean = _mk_pool(tmp_path)
    out = _write_lists(tmp_path, clean)
    rr = _run("--clean-dir", str(clean), "--out-dir", str(out), "--verify")
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")


# ---------------------------------------------------------------- (c) 验收抓坏名单
def test_verify_rejects_tampered_side_list(tmp_path):
    clean = _mk_pool(tmp_path)
    out = _write_lists(tmp_path, clean)
    near_p, far_p = out / SP.SIDE_LISTS[SP.SIDE_NEAR], out / SP.SIDE_LISTS[SP.SIDE_FAR]
    far = _read(far_p)
    near_p.write_text("".join(f"{n}\n" for n in _read(near_p) + [far[0]]), encoding="utf-8", newline="\n")
    rr = _run("--clean-dir", str(clean), "--out-dir", str(out), "--verify")
    assert rr.returncode == 1
    msg = rr.stderr.decode("utf-8", "replace")
    assert "near/far" in msg and "不一致" in msg


def test_verify_rejects_reader_manifest_edit(tmp_path):
    clean = _mk_pool(tmp_path)
    out = _write_lists(tmp_path, clean)
    man = out / SP.READER_MANIFEST
    text = man.read_text(encoding="utf-8").splitlines()
    head = text[0].split(",")
    row = text[1].split(",")
    row[head.index("split")] = SP.HOLDOUT_NAME if row[head.index("split")] != SP.HOLDOUT_NAME \
        else SP.TRAIN_NAME
    man.write_text("\n".join([text[0], ",".join(row)] + text[2:]) + "\n", encoding="utf-8", newline="\n")
    rr = _run("--clean-dir", str(clean), "--out-dir", str(out), "--verify")
    assert rr.returncode == 1


# ---------------------------------------------------------------- (d) 退出码 2
@pytest.mark.parametrize("argv", [
    ("--clean-dir", "{missing_clean}", "--out-dir", "{out}", "--verify"),
    ("--clean-dir", "{clean}", "--out-dir", "{empty}", "--verify"),
])
def test_input_errors_exit_2(tmp_path, argv):
    clean = _mk_pool(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    filled = [a.format(clean=clean, out=_write_lists(tmp_path, clean),
                       empty=empty, missing_clean=tmp_path / "nope") for a in argv]
    rr = _run(*filled)
    assert rr.returncode == 2, rr.stderr.decode("utf-8", "replace")


# ---------------------------------------------------------------- (e) 物化池目录
def test_materialize_and_verify_pool_dirs(tmp_path):
    clean = _mk_pool(tmp_path)
    noise_src = tmp_path / "noisesrc"
    noise_src.mkdir()
    noise_names = [f"noise_{i:03d}.wav" for i in range(5)]
    for n in noise_names:
        (noise_src / n).write_bytes(b"RIFF")
    noise_list = tmp_path / "noise_eval15.txt"
    noise_list.write_text("".join(f"{n}\n" for n in noise_names), encoding="utf-8", newline="\n")

    out = _write_lists(tmp_path, clean)
    root = tmp_path / "pools"
    rr = _run("--clean-dir", str(clean), "--out-dir", str(out),
              "--materialize-root", str(root), "--noise-src", str(noise_src),
              "--noise-list", str(noise_list))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    assert sorted(p.name for p in (root / "near").glob("*.wav")) == \
        sorted(_read(out / SP.SIDE_LISTS[SP.SIDE_NEAR]))
    assert sorted(p.name for p in (root / "far").glob("*.wav")) == \
        sorted(_read(out / SP.SIDE_LISTS[SP.SIDE_FAR]))
    assert sorted(p.name for p in (root / "noise").glob("*.wav")) == sorted(noise_names)
    rr = _run("--clean-dir", str(clean), "--out-dir", str(out), "--verify",
              "--materialize-root", str(root), "--noise-src", str(noise_src),
              "--noise-list", str(noise_list))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")


def test_meta_counts_match_lists(tmp_path):
    clean = _mk_pool(tmp_path)
    out = _write_lists(tmp_path, clean)
    meta = json.loads((out / SP.META_NAME).read_text(encoding="utf-8"))
    assert meta["version"] == "speech_split_v1"
    assert meta["counts"]["files"] == 60 * 3
    assert meta["counts"]["files_" + SP.SIDE_NEAR] == len(_read(out / SP.SIDE_LISTS[SP.SIDE_NEAR]))
    assert meta["counts"]["files_" + SP.SIDE_FAR] == len(_read(out / SP.SIDE_LISTS[SP.SIDE_FAR]))
    assert meta["counts"]["readers"] == 60
