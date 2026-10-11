"""T9：`scripts/verify_data.py` 数据就绪度检查覆盖（test_gap.md §2 T9）。

全部用 `monkeypatch` 把模块级路径常量指向 `tmp_path`：零 D:/data 依赖。
最后一个用例起子进程跑真实脚本，验证"report-only、退出码恒 0"这一自我声明。
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


VD = _load("t9_verify_data", "scripts/verify_data.py")


def test_mb_computes_megabytes(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"0" * (2 * 1024 * 1024))
    assert mb_ok(VD.mb(p), 2.0)


def mb_ok(got, want):
    return abs(got - want) < 1e-6


def _mk_vb(tmp_path, n_ok=1, extra=False):
    vb = tmp_path / "vb"
    vb.mkdir()
    for name in VD.VB_EXPECTED[:n_ok]:
        (vb / name).write_bytes(b"0" * (2 * 1024 * 1024))
    # VD.VB_EXPECTED[n_ok:] 各项缺失 -> MISS 分支（刻意不创建文件）
    if extra:
        (vb / "extra.parquet").write_bytes(b"0" * 100)
    return vb


def test_check_voicebank_reports_ok_and_miss(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(VD, "VB", _mk_vb(tmp_path))
    VD.check_voicebank()
    out = capsys.readouterr().out
    assert "OK " in out and "MISS" in out


def test_check_voicebank_missing_dir_returns_early(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(VD, "VB", tmp_path / "nope")
    VD.check_voicebank()
    assert "MISS 目录不存在" in capsys.readouterr().out


def test_check_voicebank_warns_extra_parquet(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(VD, "VB", _mk_vb(tmp_path, extra=True))
    VD.check_voicebank()
    out = capsys.readouterr().out
    assert "WARN 额外文件" in out and "extra.parquet" in out


def test_check_dns_counts_files_per_subdir(tmp_path, monkeypatch, capsys):
    dns = tmp_path / "dns"
    (dns / "clean").mkdir(parents=True)
    (dns / "noise").mkdir()
    for i in range(3):
        (dns / "clean" / f"c{i}.wav").write_bytes(b"0")
    for i in range(2):
        (dns / "noise" / f"n{i}.wav").write_bytes(b"0")
    monkeypatch.setattr(VD, "DNS", dns)
    VD.check_dns()
    out = capsys.readouterr().out
    assert "clean" in out and "3" in out
    assert "noise" in out and "2" in out


def test_check_aec_marks_full_sets_ok(tmp_path, monkeypatch, capsys):
    root = tmp_path / "repo"
    tests = root / "datasets" / "test_set_icassp2022"
    for sub, n in (("doubletalk", 600), ("farend-singletalk", 400), ("nearend-singletalk", 400)):
        d = tests / sub
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"f{i}.wav").touch()
    rirs = root / "datasets" / "RIRs"
    rirs.mkdir(parents=True)
    (root / "AECMOS").mkdir(parents=True, exist_ok=True)      # 官方脚本目录也需存在
    monkeypatch.setattr(VD, "AEC_TESTSET", tests)
    monkeypatch.setattr(VD, "RIRS", rirs)
    monkeypatch.setattr(VD, "AEC_REPO", root)
    VD.check_aec()
    out = capsys.readouterr().out
    assert out.count("OK ") >= 4          # 三个场景 + RIR
    assert "MISS" not in out


def test_script_runs_report_only_exit_zero():
    """脚本自我声明"report-only、退出码恒 0"——起子进程验证（无 D:/data 时也应为 0）。

    V5 终审 F12：显式 `encoding="utf-8"`。否则当环境设了 `PYTHONIOENCODING=utf-8`
    （子进程按 UTF-8 输出、父进程 `text=True` 按 locale cp936 解码）时，reader 线程会抛
    UnicodeDecodeError → `r.stdout is None` → 本用例以 TypeError 失败。
    """
    r = subprocess.run([sys.executable, "-s", str(ROOT / "scripts" / "verify_data.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    assert r.stdout is not None, "stdout 解码失败（应为 str）"
    for tag in ("[1]", "[2]", "[3]"):
        assert tag in r.stdout
