"""t16 单测：真实 RIR 按位置组三分割（`scripts/split_rir_pool.py`）。

覆盖五件事：
(a) **分组规则**：IR64CAR_<angle> 命中位置组、同角度不同胶囊同组；无角度命名退化为上级目录；
(b) **组级互斥 / 覆盖完整**：train/val/holdout 两两不交、并集 = 全池、同组不跨 split（泄漏红线）；
(c) **名单可复现**：同池两次 `--write` 逐字节一致（UTF-8/LF/无 BOM），meta 计数与内容 sha256 自洽；
(d) **验收能抓坏名单**：手改（把 train 文件名塞进 holdout）→ 退出码 1；池不存在 → 退出码 2；
(e) **物化**：`--materialize-root` 后目录内容 = holdout15 名单，验收 0。

用法：`pytest tests/test_split_rir_pool.py -q`
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import split_rir_pool as RP  # noqa: E402

SCRIPT = Path(RP.__file__)
PY = sys.executable


def _mk_pool(root: Path, angles=(0, 45, 90, 135, 180, 225, 270, 315), capsules=(0, 8, 16, 24)):
    """合成 IR64-CAR 形态的池：角度 × 胶囊，文件名与真实池同构。"""
    d = root / "pool_16k_mono"
    d.mkdir(parents=True, exist_ok=True)
    for a in angles:
        for c in capsules:
            (d / f"car_IR64CAR_{a}_c{c:02d}.wav").write_bytes(b"RIFF" + bytes([a % 256, c]))
    return d


def _run(*argv: str, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-s", str(SCRIPT), *argv], capture_output=True, timeout=timeout,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def _read(p: Path) -> list[str]:
    return [ln for ln in p.read_text(encoding="utf-8").split("\n") if ln]


def _write(tmp_path: Path, pool: Path) -> Path:
    out = tmp_path / "lists"
    rr = _run("--rir-dir", str(pool), "--out-dir", str(out))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    return out


# ---------------------------------------------------------------- (a) 分组规则
def test_group_rule_position_and_fallback(tmp_path):
    pool = _mk_pool(tmp_path)
    g = RP.group_of(pool / "car_IR64CAR_135_c16.wav", pool)
    assert g == "ir64car_pos135"                                   # 同角度同组
    assert RP.group_of(pool / "car_IR64CAR_135_c00.wav", pool) == g
    assert RP.group_of(pool / "car_IR64CAR_0_c24.wav", pool) != g
    other = tmp_path / "other" / "sessionA"
    other.mkdir(parents=True)
    assert RP.group_of(other / "rir_x.wav", tmp_path / "other") == "dir:sessionA"
    (tmp_path / "other" / "top.wav").write_bytes(b"RIFF")
    assert RP.group_of(tmp_path / "other" / "top.wav", tmp_path / "other") == "dir:."
    assert RP.session_of(pool / "car_IR64CAR_0_c00.wav") == "ir64car"


def test_allocate_keeps_train_nonempty():
    for g in range(1, 40):
        alloc = RP.allocate(g)
        assert sum(alloc.values()) == g
        assert alloc[RP.TRAIN_NAME] >= 1 or g <= 2
        if g >= 3:
            assert alloc[RP.HOLDOUT_NAME] >= 1 and alloc[RP.VAL_NAME] >= 1


# ---------------------------------------------------------------- (b)(c) 名单
def test_write_lists_group_disjoint_and_reproducible(tmp_path):
    pool = _mk_pool(tmp_path)
    out_a = _write(tmp_path, pool)
    out_b = tmp_path / "lists2"
    assert _run("--rir-dir", str(pool), "--out-dir", str(out_b)).returncode == 0
    for f in list(RP.LIST_FILES.values()) + [RP.MANIFEST_NAME, RP.META_NAME]:
        assert (out_a / f).read_bytes() == (out_b / f).read_bytes(), f
        assert not (out_a / f).read_bytes().startswith(b"\xef\xbb\xbf")
    for f in list(RP.LIST_FILES.values()) + [RP.META_NAME]:     # 名单/meta 强制 LF（manifest 随仓惯例 CRLF）
        assert b"\r" not in (out_a / f).read_bytes(), f

    train = set(_read(out_a / RP.LIST_FILES[RP.TRAIN_NAME]))
    val = set(_read(out_a / RP.LIST_FILES[RP.VAL_NAME]))
    hold = set(_read(out_a / RP.LIST_FILES[RP.HOLDOUT_NAME]))
    pool_names = {p.name for p in pool.glob("*.wav")}
    assert train | val | hold == pool_names
    assert not (train & hold) and not (train & val) and not (val & hold)
    assert train and val and hold

    # 同组不跨 split（位置级互斥）
    rows = list(__import__("csv").DictReader(open(out_a / RP.MANIFEST_NAME, encoding="utf-8")))
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_group.values())
    # 8 组 -> 2 组 holdout（§7「RIR 优先入 eval 轴」）；实测 4 条/组
    assert len(hold) == 8


def test_meta_self_consistent(tmp_path):
    pool = _mk_pool(tmp_path)
    out = _write(tmp_path, pool)
    meta = json.loads((out / RP.META_NAME).read_text(encoding="utf-8"))
    assert meta["version"] == "rir_split_v1"
    assert meta["counts"]["files"] == 32 and meta["counts"]["groups"] == 8
    assert meta["counts"]["files_" + RP.HOLDOUT_NAME] == 8
    assert meta["counts"]["groups_" + RP.HOLDOUT_NAME] == 2
    for fname, sha in meta["lists_sha256"].items():
        assert sha == RP.content_sha256(_read(out / fname))


# ---------------------------------------------------------------- (d) 退出码
def test_verify_passes(tmp_path):
    pool = _mk_pool(tmp_path)
    out = _write(tmp_path, pool)
    assert _run("--rir-dir", str(pool), "--out-dir", str(out), "--verify").returncode == 0


def test_verify_rejects_train_file_in_holdout(tmp_path):
    pool = _mk_pool(tmp_path)
    out = _write(tmp_path, pool)
    hold_p = out / RP.LIST_FILES[RP.HOLDOUT_NAME]
    train_names = _read(out / RP.LIST_FILES[RP.TRAIN_NAME])
    hold_p.write_text("".join(f"{n}\n" for n in _read(hold_p) + [train_names[0]]),
                      encoding="utf-8", newline="\n")
    rr = _run("--rir-dir", str(pool), "--out-dir", str(out), "--verify")
    assert rr.returncode == 1
    assert "不一致" in rr.stderr.decode("utf-8", "replace")


def test_verify_rejects_group_split_edit(tmp_path):
    """权威表把某条改成别的 split（模拟"同组跨 split"）→ 必须被抓住。"""
    pool = _mk_pool(tmp_path)
    out = _write(tmp_path, pool)
    man = out / RP.MANIFEST_NAME
    lines = man.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    cols = head.index("split")
    row = lines[1].split(",")
    row[cols] = RP.HOLDOUT_NAME if row[cols] == RP.TRAIN_NAME else RP.TRAIN_NAME
    man.write_text("\n".join([lines[0], ",".join(row)] + lines[2:]) + "\n",
                   encoding="utf-8", newline="\n")
    rr = _run("--rir-dir", str(pool), "--out-dir", str(out), "--verify")
    assert rr.returncode == 1


def test_input_errors_exit_2(tmp_path):
    pool = _mk_pool(tmp_path)
    out = _write(tmp_path, pool)
    assert _run("--rir-dir", str(tmp_path / "nope"), "--out-dir", str(out),
                "--verify").returncode == 2
    empty = tmp_path / "emptylists"
    empty.mkdir()
    assert _run("--rir-dir", str(pool), "--out-dir", str(empty), "--verify").returncode == 2


# ---------------------------------------------------------------- (e) 物化
def test_materialize_and_verify(tmp_path):
    pool = _mk_pool(tmp_path)
    out = _write(tmp_path, pool)
    root = tmp_path / "eval_rir_pool"
    rr = _run("--rir-dir", str(pool), "--out-dir", str(out), "--materialize-root", str(root))
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")
    assert sorted(p.name for p in root.glob("*.wav")) == _read(out / RP.LIST_FILES[RP.HOLDOUT_NAME])
    assert _run("--rir-dir", str(pool), "--out-dir", str(out), "--verify",
                "--materialize-root", str(root)).returncode == 0
