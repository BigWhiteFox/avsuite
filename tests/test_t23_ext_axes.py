"""t23 单测：多轴评测集的准备机具（DT 清单 v2 派生 / 服务器只读子集抽样 / 扩量池转码）。

覆盖三件事：
(a) **DT 清单 v2 是 v1 的纯派生**：`scripts/make_dt_plan_v2.py` 的 `build()` 结果与随仓
    `configs/dt_plan_v2.json` 逐字节一致；v2 的 `eval_v5_repro` 与 v1 逐字段相同；v2 多出的
    两档只改 `dt_snr_db`；v1 文件本身不含新 preset（派生不回溯改 v1）；
(b) **子集抽样规则**：确定性、长度=target、保持升序、等间隔（跨说话人分散）；
(c) **扩量池转码**：44.1k/16k/flac 输入 → 16k mono PCM_16 `.wav`（扩展名统一成 `.wav`，
    同名冲突加前缀），并落 `_manifests/<pool>.csv`。

用法：`pytest tests/test_t23_ext_axes.py -q`
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import build_ext_eval_pools as BP  # noqa: E402
import dt_plan as DP  # noqa: E402
import make_dt_plan_v2 as MP  # noqa: E402
import pull_ext_subsets as P  # noqa: E402


# ---------------------------------------------------------------- (a) v2 派生
def test_v2_is_pure_derivation_of_v1():
    v1, v1_sha = DP.load_plan(MP.V1)
    plan = MP.build(v1, v1_sha)
    DP.validate_plan(plan)
    on_disk = MP.V2.read_text(encoding="utf-8")
    assert on_disk == json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    # v1 不含新 preset（派生不回溯改 v1）
    assert set(v1["presets"]) == {"train_v1_repro", "eval_v5_repro"}
    assert set(plan["presets"]) == {"train_v1_repro", "eval_v5_repro",
                                    "eval_snr_low", "eval_snr_high"}
    # eval_v5_repro 逐字段继承
    assert plan["presets"]["eval_v5_repro"] == v1["presets"]["eval_v5_repro"]
    # 两档只改 dt_snr_db
    for name, (lo, hi) in (("eval_snr_low", (0.0, 10.0)), ("eval_snr_high", (30.0, 40.0))):
        base = dict(v1["presets"]["eval_v5_repro"])
        got = dict(plan["presets"][name])
        assert (got.pop("dt_snr_db"), base.pop("dt_snr_db")) == (
            {"kind": "uniform", "lo": lo, "hi": hi},
            v1["presets"]["eval_v5_repro"]["dt_snr_db"])
        got.pop("used_by"), got.pop("source")
        base.pop("used_by"), base.pop("source")
        assert got == base
    # 版本号符合 schema 的 dt_plan_vN 规则
    assert plan["version"] == "dt_plan_v2"
    assert plan["derived_from"]["sha256"] == v1_sha


def test_v2_check_mode_passes():
    rr = subprocess.run([sys.executable, "-s", str(ROOT / "scripts" / "make_dt_plan_v2.py"),
                         "--check"], capture_output=True, timeout=120,
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert rr.returncode == 0, rr.stderr.decode("utf-8", "replace")


# ---------------------------------------------------------------- (b) 子集抽样
def test_subset_rule_is_deterministic_and_spread():
    names = [f"spk{i // 10:03d}-{i:04d}" for i in range(1000)]
    picked = P.subset(names, 100)
    assert picked == P.subset(list(reversed(names)), 100)      # 顺序无关
    assert len(picked) == 100 and picked == sorted(picked)
    assert picked[0] == sorted(names)[0]                        # 等间隔从头部起
    assert len(set(picked)) == 100
    srcs = {p.split("-")[0] for p in picked}
    assert len(srcs) >= 50                                      # 跨说话人分散（非聚簇）
    assert P.subset(names[:30], 100) == sorted(names[:30])      # 少于目标 -> 全取


# ---------------------------------------------------------------- (c) 池转码
def _mk_pull(tmp_path: Path) -> Path:
    """合成 pull 目录：44.1k wav + 16k flac + 同名冲突两个文件。"""
    pull = tmp_path / "pull"
    d = pull / "corp" / "train" / "spkA"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    x = (0.2 * rng.standard_normal(44100)).astype(np.float32)   # 1 s @44.1k
    sf.write(str(d / "a.wav"), x, 44100, subtype="PCM_16")
    f = pull / "corp2" / "sub1"
    f.mkdir(parents=True)
    sf.write(str(f / "b.flac"), x[:16000], 16000)
    g = pull / "corp2" / "sub2"
    g.mkdir(parents=True)
    sf.write(str(g / "b.flac"), x[:8000], 16000)                # 同名不同目录（冲突）
    return pull


def test_build_pool_converts_and_handles_duplicates(tmp_path):
    pull = _mk_pull(tmp_path)
    outs = []
    for sub in ("corp", "corp2"):
        info = BP.build_pool(pull, tmp_path / "pools", f"p_{sub}", sub, (".wav", ".flac"))
        outs.append(info)
        assert info["sr_in"] in ({44100: 1}, {16000: 2})
        for p in Path(info["out_dir"]).glob("*.wav"):
            mi = sf.info(str(p))
            assert mi.samplerate == BP.SR and mi.channels == 1
    # 同名冲突 -> 加顶层目录前缀，两个都在
    names = sorted(p.name for p in Path(outs[1]["out_dir"]).glob("*.wav"))
    assert len(outs[1]["dup_names"]) == 1 and names == ["sub1_b.wav", "sub2_b.wav"]
    man = Path(outs[1]["manifest"])
    assert man.is_file()
    rows = list(csv.DictReader(open(man, encoding="utf-8")))
    assert {r["out_name"] for r in rows} == set(names)
    assert all(len(r["sha256"]) == 64 for r in rows)
    # 清单里的 sha256 与实际文件一致
    for r in rows:
        got = hashlib.sha256((Path(outs[1]["out_dir"]) / r["out_name"]).read_bytes()).hexdigest()
        assert got == r["sha256"]


def test_build_pool_missing_input_is_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        BP.build_pool(tmp_path / "nope", tmp_path / "pools", "x", "corp", (".wav",))
