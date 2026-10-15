"""P1-5c 单测：池枚举排序键平台无关化（防跨平台"下标→文件"映射分叉）。

背景（t12 实测）：`sorted(Path)` 比较的是 **normcase 之后的部件**（`PurePath.__lt__` 比
`_cparts`）——Windows 折叠大小写、POSIX 不折叠 ⇒ 池内文件名大小写混合时，两平台的枚
举序不同，同 seed 下"下标 → 文件名"映射随之分叉：eval_v8 在服务器重放本地冻结集时
12/13 列逐位一致、唯 `noise_file` 列 1234/1400 行映射不同（**集合相等、排序不同**）。

覆盖：
(a) **键语义**：`pool_sort_key` 的输出 == 池内相对路径（`/` 归一）的 UTF-8 字节序；
    与 POSIX 语义（`PurePosixPath` 比较，不折叠）一致；与 Windows 语义（`casefold`，
    = `PureWindowsPath._cparts`）**在本 fixture 上确实不同**（否则用例没牙）；
(b) **输入序无关**：把同一批文件以任意顺序喂进去，输出恒为字节序（这才是"枚举序确定"
    的真正含义——枚举结果不依赖文件系统返回顺序）；
(c) **接入点**：`list_pool`（near/far/noise/rir 的唯一入口）、`load_rir_dir`、
    互斥判据的内部枚举、`dm_dataset` 的池索引全部走同一键；
(d) **接口不变式**：dm_dataset 的**采样序由清单文件决定**（清单序 = 采样序），
    池枚举序只影响诊断信息——显式钉死，防日后有人"顺手排序"改动清单语义；
(e) 码点序 ≡ UTF-8 字节序（含非 ASCII），故 POSIX 既有行为不变。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aec_synth as S  # noqa: E402


def _load(name: str, path: Path):
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

#: 大小写混合 + 非 ASCII 的池成员（Windows 折叠序与字节序必然不同：'B' < 'a' 仅后者成立）
NAMES = ("B_2.wav", "a_1.wav", "A_1.wav", "b_2.wav", "Zz.wav", "zz.wav", "é_1.wav", "中_1.wav")


def _make_pool(root, names=NAMES, nested=True):
    """建池：一半放根、一半放 reader 子目录（覆盖相对路径含分隔符的情形）。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "readerA").mkdir(exist_ok=True)
    for i, nm in enumerate(names):
        d = root / "readerA" if (nested and i % 2) else root
        (d / nm).write_bytes(b"RIFF")          # 内容无关：只测枚举序
    return root


def _rel(root, p):
    return str(Path(p).relative_to(root)).replace("\\", "/")


def _byte_order(root, paths):
    return sorted(paths, key=lambda p: _rel(root, p).encode("utf-8"))


def _win_order(root, paths):
    """Windows 语义（大小写折叠）——显式构造，任何平台上都同义。"""
    return sorted(paths, key=lambda p: _rel(root, p).casefold())


def test_pool_sort_key_is_utf8_byte_order_of_relative_path(tmp_path):
    """(a) 键 = 相对路径（`/` 归一）的 UTF-8 字节序；与 POSIX 语义一致、与 Windows 语义不同。"""
    pool = _make_pool(tmp_path / "pool")
    paths = list(pool.glob("**/*.wav"))
    got = S.sort_pool_paths(pool, paths)
    expected = _byte_order(pool, paths)
    assert [_rel(pool, p) for p in got] == [_rel(pool, p) for p in expected]
    assert [S.pool_sort_key(pool, p) for p in got] == sorted(S.pool_sort_key(pool, p) for p in paths)
    # 相对路径含分隔符（'readerA/...'）也要参与比较，故分隔符必须归一为 '/'
    assert all(isinstance(S.pool_sort_key(pool, p), bytes) for p in paths)
    assert b"\\" not in b"".join(S.pool_sort_key(pool, p) for p in paths)
    # 本 fixture 上两种平台语义确实不同（用例有牙）
    assert [_rel(pool, p) for p in _win_order(pool, paths)] != [_rel(pool, p) for p in expected]
    assert [_rel(pool, p) for p in got] != [_rel(pool, p) for p in _win_order(pool, paths)]
    # POSIX 语义（PurePosixPath 比较 = 不折叠）与我们一致
    from pathlib import PurePosixPath
    assert [str(PurePosixPath(_rel(pool, p))) for p in got] == \
        sorted(str(PurePosixPath(_rel(pool, p))) for p in paths)
    # 码点序 ≡ UTF-8 字节序（含非 ASCII ⇒ POSIX 既有行为不变）
    names = [_rel(pool, p) for p in paths]
    assert sorted(names) == sorted(names, key=lambda s: s.encode("utf-8"))
    # Windows 语义的具体体现（回归哨兵）：'B_2.wav' 折叠后排在 'a_1.wav' 之前
    assert _win_order(pool, paths)[0].name in ("B_2.wav", "b_2.wav", "A_1.wav", "a_1.wav")
    assert got[0].name.startswith("A")


def test_pool_sort_is_input_order_independent(tmp_path):
    """(b) 枚举序只由文件集合决定，与喂入顺序（= 文件系统返回顺序）无关。"""
    pool = _make_pool(tmp_path / "pool")
    paths = list(pool.glob("**/*.wav"))
    base = [_rel(pool, p) for p in S.sort_pool_paths(pool, paths)]
    for shifted in (paths[::-1], paths[3:] + paths[:3], paths[1::2] + paths[::2]):
        assert [_rel(pool, p) for p in S.sort_pool_paths(pool, shifted)] == base
    assert len(base) == len(set(base)) == len(NAMES)


def test_list_pool_and_load_rir_dir_use_same_key(tmp_path):
    """(c) 接入点：list_pool / load_rir_dir / 互斥判据内部枚举 / dm_dataset 池索引同一键。"""
    pool = _make_pool(tmp_path / "noise")
    paths = list(pool.glob("**/*.wav"))
    assert [_rel(pool, p) for p in S.list_pool(pool)] == \
        [_rel(pool, p) for p in _byte_order(pool, paths)]
    rir = _make_pool(tmp_path / "rir", names=("B.wav", "a.wav", "A.wav"))
    assert [p.name for p in S.load_rir_dir(rir)] == ["A.wav", "B.wav", "a.wav"]
    with pytest.raises(FileNotFoundError):
        S.load_rir_dir(tmp_path / "empty_rir")
    # 互斥判据内部枚举（near/far 传 None 时）：顺序同键（影响"示例路径"的确定性）
    reasons = S.nearfar_pool_conflicts(tmp_path / "noise", tmp_path / "noise")
    assert reasons and "同一目录" in reasons[0]
    # dm_dataset 用的是**同一个函数对象**（单一实现，避免两处排序键再分叉）
    assert DM.sort_pool_paths is S.sort_pool_paths and DM.list_pool is S.list_pool


def test_dm_pool_index_platform_order_but_list_order_is_interface(tmp_path):
    """(d) 采样序 = 清单序（接口）；池索引（诊断用）走平台无关序，两者互不越界。"""
    pool = _make_pool(tmp_path / "near")
    names = list(NAMES)
    lst = tmp_path / "near.txt"
    lst.write_text("\n".join(names[::-1]) + "\n", encoding="utf-8")   # 故意逆序写清单
    got = DM.load_pool(pool, lst)
    assert [p.name for p in got] == names[::-1], "清单序 = 采样序，不得被排序改写"
    # 池内同名歧义：先按平台无关序枚举 ⇒ 诊断信息与平台无关
    dup = tmp_path / "dup"
    (dup / "sub1").mkdir(parents=True)
    (dup / "sub2").mkdir(parents=True)
    for d in ("sub1", "sub2"):
        (dup / d / "same.wav").write_bytes(b"RIFF")
    with pytest.raises(DM.DMPoolError, match="同名音频"):   # P1X-2：扩展名含 .flac，措辞改"音频"
        DM.load_pool(dup, lst)


def test_sorted_path_default_differs_from_fix_on_case_mixed_pool(tmp_path):
    """缺口复现（回归哨兵）：`sorted(Path)` 在本机（Windows）确实给出与字节序不同的序。"""
    pool = _make_pool(tmp_path / "pool")
    paths = list(pool.glob("**/*.wav"))
    old = [p.name for p in sorted(paths)]                    # 旧口径：Path 比较
    new = [p.name for p in S.list_pool(pool)]                # 新口径：字节序
    if sys.platform == "win32":
        assert old != new, "Windows 上旧口径应与字节序不同（大小写折叠）"
    else:
        assert old == new, "POSIX 上旧口径本就等于字节序（本改动不改变服务器行为）"
