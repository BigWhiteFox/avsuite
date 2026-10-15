"""t17 小修单测：① `_files` 清单正式化 ② noise meta 平台无关化。

覆盖：
(a) `scripts/build_speech_file_lists.py` 四条断言（near∩far=∅ / 并集==train70 /
    与 holdout 文件零交集 / reader 级并集==train70）在合成池上成立且可复现；
    篡改 reader 级清单后必须**失败**（退出码 1）而不是静默产出。
(b) `split_noise_pool` 产出的 meta **不含任何绝对路径**，只留 `pool_dir_hint`；
    `verify_split` 在 `--noise-dir` 与 `NOISE_POOL_DIR` 两条路径下都能跑 `--reproduce`
    通过（平台无关：不依赖 meta 里写死的本机路径）。

用法：`pytest tests/test_split_lists_t17.py -q`
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
BUILD = SCRIPTS / "build_speech_file_lists.py"
NOISE = SCRIPTS / "split_noise_pool.py"
VERIFY = SCRIPTS / "verify_split.py"
PY = sys.executable


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_clean(root: Path, readers: list[str], per: int = 3) -> Path:
    d = root / "clean"
    d.mkdir(parents=True, exist_ok=True)
    for r in readers:
        for i in range(per):
            (d / f"book_00001_chp_0002_reader_{r}_{i}.wav").write_bytes(b"RIFF")
    return d


def _run(script: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-s", str(script), *argv], capture_output=True,
                          timeout=180, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def _write_list(p: Path, names: list[str]) -> None:
    p.write_text("".join(f"{n}\n" for n in names), encoding="utf-8", newline="\n")


def _sha(names: list[str]) -> str:
    return hashlib.sha256("".join(f"{n}\n" for n in names).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- (a) build_speech_file_lists
def _fixture_lists(tmp_path: Path, near=("00001", "00002"), far=("00003",),
                   hold=("00004",), per=3):
    """合成池 + reader 级清单（train70 = near ∪ far）。"""
    clean = _mk_clean(tmp_path, list(near) + list(far) + list(hold), per=per)
    out = tmp_path / "lists"
    out.mkdir(parents=True, exist_ok=True)
    _write_list(out / "speech_near_train70.txt", list(near))
    _write_list(out / "speech_far_train70.txt", list(far))
    _write_list(out / "speech_readers_train70.txt", list(near) + list(far))
    _write_list(out / "speech_readers_holdout15.txt", list(hold))
    return clean, out


def _expected(clean: Path, readers) -> list[str]:
    names = []
    for r in readers:
        names += sorted(p.name for p in clean.glob(f"*reader_{r}_*"))
    return names


def test_file_lists_generated_with_assertions(tmp_path):
    """四断言成立：near/far 零交集、并集==train70、与 holdout 零交集，且内容=规则重算。"""
    clean, out = _fixture_lists(tmp_path)
    rr = _run(BUILD, "--clean-dir", str(clean), "--out-dir", str(out))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")

    near = [ln for ln in (out / "speech_near_train70_files.txt").read_text().splitlines() if ln]
    far = [ln for ln in (out / "speech_far_train70_files.txt").read_text().splitlines() if ln]
    assert near == _expected(clean, ["00001", "00002"])
    assert far == _expected(clean, ["00003"])
    assert not set(near) & set(far)
    assert set(near) | set(far) == set(_expected(clean, ["00001", "00002", "00003"]))
    assert not (set(near) | set(far)) & set(_expected(clean, ["00004"]))

    meta = json.loads((out / "speech_nearfar_train_meta.json").read_text(encoding="utf-8-sig"))
    fl = meta["file_lists"]
    assert fl["counts"] == {"near_files": len(near), "far_files": len(far)}
    assert fl["sha256"]["speech_near_train70_files.txt"] == _sha(near)
    assert fl["sha256"]["speech_far_train70_files.txt"] == _sha(far)
    assert fl["assertions"]["A_near_inter_far_files"] == 0

    # 无 BOM（跨平台可读）
    raw = (out / "speech_near_train70_files.txt").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw


def test_verify_mode_passes_then_detects_tamper(tmp_path):
    """--verify：先与既有 _files 逐行一致通过；篡改 reader 级清单致交集非空 -> 退出码 1。"""
    clean, out = _fixture_lists(tmp_path)
    assert _run(BUILD, "--clean-dir", str(clean), "--out-dir", str(out)).returncode == 0
    rr = _run(BUILD, "--clean-dir", str(clean), "--out-dir", str(out), "--verify")
    assert rr.returncode == 0, rr.stdout.decode("utf-8", "replace")

    # 篡改：把 far reader 也塞进 near 名单 -> A 断言（交集）必须失败
    _write_list(out / "speech_near_train70.txt", ["00001", "00002", "00003"])
    rr = _run(BUILD, "--clean-dir", str(clean), "--out-dir", str(out))
    assert rr.returncode == 1
    assert "near" in rr.stdout.decode("utf-8", "replace")


def test_file_lists_detect_holdout_overlap(tmp_path):
    """C 断言：把 holdout reader 混进 far 训练清单 -> 必须失败（评测轴互斥是硬约束）。"""
    clean, out = _fixture_lists(tmp_path)
    _write_list(out / "speech_far_train70.txt", ["00003", "00004"])
    _write_list(out / "speech_readers_train70.txt", ["00001", "00002", "00003", "00004"])
    rr = _run(BUILD, "--clean-dir", str(clean), "--out-dir", str(out))
    assert rr.returncode == 1
    assert "holdout" in rr.stdout.decode("utf-8", "replace")


# ---------------------------------------------------------------- (b) noise meta 平台无关
def _mk_noise_pool(root: Path, n: int | None = None) -> Path:
    """合成噪声池：足够大且**三桶都非空**（验收脚本拒绝空组，故不能只放几个文件）。

    n 缺省时自动扩张到 `split_noise_pool` 的三段都非空为止（规则与生产一致：文件名哈希分桶）。
    """
    P = _load(NOISE, "split_noise_pool_t17")
    d = root / "noise"

    def counts(size):
        names = [f"clip_{i:04d}.wav" for i in range(size)]
        c = {"train70": 0, "val15": 0, "eval15": 0}
        for nm in names:
            c[P.split_of(nm)] += 1
        return c

    if n is None:
        n = 60
        while min(counts(n).values()) == 0:
            n += 20
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"clip_{i:04d}.wav").write_bytes(b"RIFF")
    return d


def test_noise_meta_has_no_absolute_path(tmp_path):
    """meta 不得出现任何绝对/Windows 路径；只留文档性 pool_dir_hint。"""
    pool = _mk_noise_pool(tmp_path)
    out = tmp_path / "lists"
    rr = _run(NOISE, "--noise-dir", str(pool), "--out-dir", str(out))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    meta = json.loads((out / "noise_split_meta.json").read_text(encoding="utf-8-sig"))
    assert "noise_dir" not in meta
    assert meta["pool_dir_hint"] == "<noise-pool>/noise"
    blob = json.dumps(meta, ensure_ascii=False)
    assert str(tmp_path) not in blob and ":\\" not in blob and ":/" not in blob


def test_noise_meta_hash_sentinel_unchanged(tmp_path):
    """平台无关化后，名单内容 sha256 哨兵与计数不变（仅 meta 的路径字段变化）。"""
    pool = _mk_noise_pool(tmp_path)
    out = tmp_path / "lists"
    assert _run(NOISE, "--noise-dir", str(pool), "--out-dir", str(out)).returncode == 0
    meta = json.loads((out / "noise_split_meta.json").read_text(encoding="utf-8-sig"))
    names = {g: [ln for ln in (out / meta["list_files"][g]).read_text().splitlines() if ln]
             for g in ("train70", "val15", "eval15")}
    for g, lines in names.items():
        assert meta["list_sha256"][g] == _sha(lines)
    assert meta["counts"] == {g: len(v) for g, v in names.items()}
    assert sum(meta["counts"].values()) == meta["pool_files"]


def test_verify_split_resolves_pool_without_meta_path(tmp_path):
    """平台无关验收：--noise-dir 与 NOISE_POOL_DIR 两条路径都能跑 --reproduce 通过。"""
    pool = _mk_noise_pool(tmp_path)
    out = tmp_path / "lists"
    assert _run(NOISE, "--noise-dir", str(pool), "--out-dir", str(out)).returncode == 0

    # 1) 显式 --noise-dir
    rr = _run(VERIFY, "--dir", str(out), "--noise-dir", str(pool), "--reproduce")
    assert rr.returncode == 0, rr.stdout.decode("utf-8", "replace") + rr.stderr.decode("utf-8", "replace")

    # 2) 环境变量（不给 --noise-dir）
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "NOISE_POOL_DIR": str(pool)}
    rr = subprocess.run([PY, "-s", str(VERIFY), "--dir", str(out), "--reproduce"],
                        capture_output=True, timeout=180, env=env)
    assert rr.returncode == 0, rr.stdout.decode("utf-8", "replace")

    # 3) 池不可知时应给出可操作提示（非崩溃）；--skip-pool-check 明确跳过
    env2 = {k: v for k, v in env.items() if k != "NOISE_POOL_DIR"}
    rr = subprocess.run([PY, "-s", str(VERIFY), "--dir", str(out), "--skip-pool-check"],
                        capture_output=True, timeout=180, env=env2)
    assert rr.returncode == 0, rr.stdout.decode("utf-8", "replace")


def test_pool_dir_from_meta_prefers_env_and_hint(tmp_path, monkeypatch):
    """resolve 顺序：NOISE_POOL_DIR > hint（相对解析）> 旧 noise_dir；都不成立返回 None。"""
    V = _load(VERIFY, "verify_split_t17")
    P = _load(NOISE, "split_noise_pool_t17b")
    pool = _mk_noise_pool(tmp_path)
    out = tmp_path / "lists"
    out.mkdir()

    # hint 相对当前工作目录可解析
    monkeypatch.chdir(tmp_path)
    (out / P.META_NAME).write_text(
        json.dumps({"pool_dir_hint": "<noise-pool>/noise"}), encoding="utf-8")
    assert V.pool_dir_from_meta(out, P) == pool.resolve()

    # 环境变量优先
    other = tmp_path / "other_pool"
    other.mkdir()
    monkeypatch.setenv("NOISE_POOL_DIR", str(other))
    assert V.pool_dir_from_meta(out, P) == other

    # 两者都不成立 -> None（调用方给提示，不猜）
    monkeypatch.delenv("NOISE_POOL_DIR", raising=False)
    (out / P.META_NAME).write_text(json.dumps({"pool_dir_hint": "<noise-pool>/nope"}),
                                   encoding="utf-8")
    assert V.pool_dir_from_meta(out, P) is None
