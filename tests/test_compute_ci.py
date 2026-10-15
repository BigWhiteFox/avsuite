"""S2 单测：tools/compute_ci.py 的解析 CI / 配对 bootstrap 语义（不改本体）。

- fmt_ci / stats_table 手工算例：95%CI = mean ± 1.96·sd/√n（sd 为样本标准差 n−1）；
  n=1、全同值、空表、缺指标等边界；
- find_json / load_all：monkeypatch 模块级 REPO/OUT_DIRS 到临时目录，不读真实 outputs；
- paired_bootstrap：固定 seed=777、B=2000 的小样本合成配对，断言 lo ≤ Δmean ≤ hi、
  结论列显著性、两次调用逐位可复现（与测试内同 seed 手工复算逐位一致）。
  合成值全部取二进制可精确表示的数（0.25 步进），保证脚本内 diff 重算与测试内
  diffs 逐位相等，bootstrap 序列完全可复现。

被测模块用 importlib 按绝对路径加载。
发现不修（记录）：paired_bootstrap(data) 的 data 形参未被使用——函数经 PAIRS/find_json
直接重读磁盘（模块级 REPO/OUT_DIRS），故测试以 monkeypatch 注入临时仓结构。
另：main() 会写真实仓 reports/DATA_CI.md，本单测不调用 main()。
"""
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("compute_ci", _REPO / "tools" / "compute_ci.py")
ci = importlib.util.module_from_spec(_spec)
sys.modules["compute_ci"] = ci
_spec.loader.exec_module(ci)


# ---------------------------------------------------------------- 解析 CI
def test_fmt_ci_hand_examples():
    assert ci.fmt_ci(4.0, 0.5, 100) == "4.000±0.10"     # 1.96·0.5/√100 = 0.098
    assert ci.fmt_ci(2.0, 1.0, 3) == "2.000±1.13"       # 1.96/√3 = 1.1316
    assert ci.fmt_ci(-1.2, 0.0, 1) == "-1.200±0.00"     # n=1 → sd 0
    assert ci.fmt_ci(3.5, 0.0, 7) == "3.500±0.00"       # 全同值 → sd 0


def test_stats_table_hand_example():
    data = {"a/aecmos_x.json": {"dt": {"echo_mos": [1.0, 2.0, 3.0], "deg_mos": [4.0]}}}
    lines = ci.stats_table(data)
    assert lines[0].startswith("## 一、")
    assert lines[2] == "| 文件 | 场景 | 指标 | n | mean | sd | 95%CI |"
    assert lines[3] == "|---|---|---|---|---|---|---|"
    # 样本标准差（n−1）：[1,2,3] → sd=1；半宽 = 1.96/√3 = 1.13
    assert lines[4] == "| a/aecmos_x.json | dt | echo_mos | 3 | 2.000 | 1.000 | 2.000±1.13 |"
    # n=1 → sd=0 → ±0.00；METRICS 顺序 echo_mos 在前
    assert lines[5] == "| a/aecmos_x.json | dt | deg_mos | 1 | 4.000 | 0.000 | 4.000±0.00 |"
    assert len(lines) == 6


def test_stats_table_n1_identical_and_empty():
    # n=1：sd 记 0
    lines = ci.stats_table({"f.json": {"nst": {"echo_mos": [4.2]}}})
    assert lines[4] == "| f.json | nst | echo_mos | 1 | 4.200 | 0.000 | 4.200±0.00 |"
    # 全同值：sd=0 → ±0.00
    lines = ci.stats_table({"f.json": {"dt": {"echo_mos": [3.0] * 5}}})
    assert lines[4] == "| f.json | dt | echo_mos | 5 | 3.000 | 0.000 | 3.000±0.00 |"
    # 空表：只有表头块
    lines = ci.stats_table({})
    assert len(lines) == 4 and lines[2].startswith("| 文件 |")
    # 场景无该指标 / 缺指标键：不产出该行
    lines = ci.stats_table({"f.json": {"dt": {}}})
    assert len(lines) == 4
    lines = ci.stats_table({"f.json": {"dt": {"deg_mos": [1.0]}}})
    assert len(lines) == 5 and "deg_mos" in lines[4] and "echo_mos" not in lines[4]


# ---------------------------------------------------------------- 文件发现/聚合
def test_find_json_resolution(monkeypatch, tmp_path):
    outs = tmp_path / "outputs"
    outs.mkdir()
    monkeypatch.setattr(ci, "REPO", tmp_path)
    monkeypatch.setattr(ci, "OUT_DIRS", [outs])
    (tmp_path / "root_key.json").write_text("{}", encoding="utf-8")
    (outs / "outs_key.json").write_text("{}", encoding="utf-8")
    assert ci.find_json("root_key") == (tmp_path / "root_key.json").resolve()
    assert ci.find_json("outs_key") == (outs / "outs_key.json").resolve()
    assert ci.find_json("missing") is None
    # 同名时仓库根优先
    (tmp_path / "both.json").write_text("{}", encoding="utf-8")
    (outs / "both.json").write_text("{}", encoding="utf-8")
    assert ci.find_json("both") == (tmp_path / "both.json").resolve()


def test_load_all_aggregates_rows_and_skips_empty(monkeypatch, tmp_path):
    outs = tmp_path / "outputs"
    (outs / "sub").mkdir(parents=True)
    rows = [
        {"name": "c1", "scenario": "dt", "echo_mos": 4.0, "deg_mos": None},
        {"name": "c2", "scenario": "dt", "echo_mos": 4.25},
        {"name": "c3", "scenario": "nst", "echo_mos": 3.0, "deg_mos": 2.5},
    ]
    (outs / "sub" / "aecmos_x.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    (outs / "aecmos_empty.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    (outs / "aecmos_norows.json").write_text(json.dumps({"meta": 1}), encoding="utf-8")
    monkeypatch.setattr(ci, "REPO", tmp_path)
    monkeypatch.setattr(ci, "OUT_DIRS", [outs])
    data = ci.load_all()
    # None 指标跳过、无 rows / 空 rows 文件跳过、label 为 REPO 相对 posix 路径
    assert data == {"outputs/sub/aecmos_x.json": {
        "dt": {"echo_mos": [4.0, 4.25]},
        "nst": {"echo_mos": [3.0], "deg_mos": [2.5]},
    }}


# ---------------------------------------------------------------- 配对 bootstrap
_TRAD_ECHO = [4.00, 3.75, 4.25, 4.00, 4.50, 3.75, 4.25, 4.00]
_DIFFS = [0.50, 0.25, 0.75, 0.00, 0.25, -0.25, 0.50, 0.75]     # Δmean = 0.34375
_NN_ECHO = [round(t + d, 10) for t, d in zip(_TRAD_ECHO, _DIFFS)]  # 0.25 步进，二进制精确


def _write_pair(tmp_path):
    rows_nn, rows_trad = [], []
    for i in range(8):                                   # c01..c08：dt 配对样本
        rows_nn.append({"name": f"c0{i + 1}", "scenario": "dt",
                        "echo_mos": _NN_ECHO[i], "deg_mos": 3.75})
        rows_trad.append({"name": f"c0{i + 1}", "scenario": "dt",
                          "echo_mos": _TRAD_ECHO[i], "deg_mos": 3.50})
    rows_nn.append({"name": "c09", "scenario": "dt",     # None 指标行：应被剔除
                    "echo_mos": None, "deg_mos": None})
    rows_trad.append({"name": "c09", "scenario": "dt", "echo_mos": 4.0, "deg_mos": 3.5})
    for r_s, v in ((rows_nn, 5.0), (rows_trad, 4.0)):    # nst 场景：不在统计范围
        r_s.append({"name": "c01", "scenario": "nst", "echo_mos": v, "deg_mos": v})
    (tmp_path / "aecmos_nn_v4.json").write_text(json.dumps({"rows": rows_nn}), encoding="utf-8")
    (tmp_path / "aecmos_trad_wp06.json").write_text(json.dumps({"rows": rows_trad}),
                                                    encoding="utf-8")


def _boot(diffs, rng):
    """与被测脚本同款循环的手工复算（供逐位比对）。"""
    n = len(diffs)
    boots = []
    for _ in range(ci.B_BOOT):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(s) / n)
    boots.sort()
    return boots[int(0.025 * ci.B_BOOT)], boots[int(0.975 * ci.B_BOOT) - 1]


def _cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def test_paired_bootstrap_fixed_seed_small_sample(monkeypatch, tmp_path):
    monkeypatch.setattr(ci, "REPO", tmp_path)
    monkeypatch.setattr(ci, "OUT_DIRS", [tmp_path / "outputs"])     # 不存在 → 不干扰
    monkeypatch.setattr(ci, "PAIRS", [("aecmos_nn_v4", "aecmos_trad_wp06")])
    _write_pair(tmp_path)

    lines = ci.paired_bootstrap({})
    rows = [ln for ln in lines if ln.startswith("| aecmos_nn_v4.json vs aecmos_trad_wp06.json |")]
    assert len(rows) == 2                                           # echo_mos + deg_mos
    r_e, r_d = _cells(rows[0]), _cells(rows[1])

    # echo_mos：n=8（None 行剔除）、Δmean、CI 含 Δmean 且显著、与同 seed 手工复算逐位一致
    dmean = sum(_DIFFS) / 8
    assert r_e[2] == "echo_mos" and r_e[3] == "8"
    assert r_e[4] == f"{dmean:+.3f}"                                # "+0.344"
    m = re.fullmatch(r"\[([+-][\d.]+), ([+-][\d.]+)\]", r_e[5])
    lo, hi = float(m.group(1)), float(m.group(2))
    assert lo <= dmean <= hi                                        # 合理近似
    assert lo > 0 and r_e[6].startswith("显著")                     # CI 不含 0
    rng = random.Random(ci.SEED)
    lo_e, hi_e = _boot(_DIFFS, rng)
    lo_d, hi_d = _boot([0.25] * 8, rng)                             # rng 消耗顺序同脚本
    assert r_e[5] == f"[{lo_e:+.3f}, {hi_e:+.3f}]"

    # deg_mos：全同差值 → 退化 CI [Δ, Δ]
    assert r_d[2] == "deg_mos" and r_d[3] == "8"
    assert r_d[4] == "+0.250" and r_d[5] == f"[{lo_d:+.3f}, {hi_d:+.3f}]" == "[+0.250, +0.250]"
    assert r_d[6].startswith("显著")

    # 两次调用逐位可复现（每次调用重置 seed=777）
    assert ci.paired_bootstrap({}) == lines


def test_paired_bootstrap_missing_files(monkeypatch, tmp_path):
    monkeypatch.setattr(ci, "REPO", tmp_path)
    monkeypatch.setattr(ci, "OUT_DIRS", [tmp_path / "outputs"])
    monkeypatch.setattr(ci, "PAIRS", [("nope_nn", "nope_trad")])
    lines = ci.paired_bootstrap({})
    assert any("文件缺失" in ln for ln in lines)                      # 记缺不抛异常
