"""NN-AEC 动态混合（DM）数据集：训练时 DataLoader worker **现合成** (mic, far, near)。

来源与定位（REMEDIATION_PLAN_V3.1 §3.2「DM 化」/ 工作项 P1-3）
-------------------------------------------------------------
沿用 `exp/04_dccrn/dynamic_mixing.py` 的 DM 框架（虚拟长度数据集：``__len__ = epoch_len``，
每次取样现场配对/合成），但**合成基元直接复用 `scripts/aec_synth.py`** 的
``load_wav_16k`` / ``fit_len`` / ``get_rir`` / ``generate_one`` 与 ``scripts/dt_plan.py``
的清单采样——训练与评测（0-4 eval_v8）走同一套物理链路，"train/eval 同源"可追溯。

三条硬口径
----------
1. **清单驱动**（0-5）：场景配比、DT 的 SER/SNR/引导时长全部来自
   ``configs/dt_plan_v1.json`` 的 preset（缺省 ``train_v1_repro``）；清单路径、版本号、
   preset、sha256 随训练 manifest 落盘（见 ``provenance()``）。
2. **池隔离**（0-2/0-3）：near/far/noise 三份**分割清单显式传入**，实例化时做互斥校验
   （文件同一性 + utterance 主名，``aec_synth.nearfar_file_conflicts``）。DM 现合成没有
   manifest 兜底，泄漏只能在入口拦——不通过直接抛错，绝不静默放行。
3. **可复现**：``seed`` 固定时 ``(seed, idx)`` → 逐位确定的样本（单测/审计/对照用）；
   ``seed=None`` 时每次取样吃系统熵（训练用，epoch 之间不重复 ⇒ DM 的组合多样性）。
   两种模式都不碰全局 numpy/torch RNG，num_workers>0（Windows spawn）下可 pickle。

与 v4 训练口径的对应（均可在 ``provenance()`` 里回读）：4s 片段、delay 10–80ms、
soft-clip 概率 0.15、RIR 截断 100ms、far 相对 mic 的 ±jitter_ms 平移（与
``dataset.AecDataset`` 的训练期抖动同实现）。**val 不动**：仍用静态 ``AecDataset``，
口径与 v4 可比（DM 只改训练侧供给）。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# scripts/ 下的合成基元与清单模块（与 aec_synth.py / dt_plan.py 单实例共享）
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import aec_synth as S  # noqa: E402
import dt_plan as DP  # noqa: E402

SR = S.SR
HERE = Path(__file__).resolve().parent
DEFAULT_PLAN = Path(__file__).resolve().parents[2] / "configs" / "dt_plan_v1.json"

#: 池枚举的平台无关排序（唯一实现在 scripts/aec_synth.py；DM 与静态合成共用同一口径，
#: 避免"两处排序键"再次分叉——见 aec_synth.pool_sort_key 的实测依据）
sort_pool_paths = S.sort_pool_paths
list_pool = S.list_pool

#: epoch 口径方案（附录B）：(a) 等效更新步数对齐 v4。方案 (b)（按数据量同比例放大）
#: 未实现——DM 的池是无限组合，没有"数据量"分母可比；本轮定死 (a) 并写进 manifest。
DM_EPOCH_SCHEME = "a"

#: v4 口径参照（方案 (a) 的对齐目标）——**从 v4 产物回读**，不是"240ep"这种在新数据下
#: 不可比的表述：``outputs_v4/nnaec_g0.5_j10/last.ckpt`` 的 ``step=40830``，构成为
#: v3 续训谱系 186ep×187 步（train_v1 3000 条 / bs16）+ 54ep×112 步（dt,nst 1795 条）。
V4_REF_STEPS = 40830
V4_REF_EPOCHS = 240
V4_REF_BATCH = 16
V4_REF_SOURCE = ("exp/07_nn_aec/outputs_v4/nnaec_g0.5_j10/last.ckpt::step "
                 "（186ep×187 + 54ep×112 = 40830）")


class DMPoolError(ValueError):
    """DM 池/清单不合法（缺文件、同名歧义、清单互斥违例）——入口即失败。"""


# ---------------------------------------------------------------- 清单与池
def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_split_list(list_path):
    """读分割清单（每行一个**文件名**；忽略空行与 `#` 注释）→ 名字列表（保序去重）。"""
    p = Path(list_path)
    if not p.is_file():
        raise DMPoolError(f"分割清单不存在：{p}")
    names, seen = [], set()
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        if name in seen:                       # 清单内重复：去重但记录（不静默放大权重）
            continue
        seen.add(name)
        names.append(name)
    if not names:
        raise DMPoolError(f"分割清单为空：{p}")
    return names


#: 池内可枚举的音频扩展名（**仅** DM 的清单驱动加载用；`aec_synth.list_pool` 仍只收 .wav，
#: 保持静态合成路径的既有行为不变）
AUDIO_EXTS = (".wav", ".flac")
#: 源规格里各字段的分隔符：**不能用 `:`**——Windows 盘符自带冒号（D:\...）
SOURCE_SEP = "|"


def parse_source_spec(spec, default_weight=1.0):
    """`DIR|LIST[|WEIGHT[|LABEL]]` → 源 dict（P1X-2 多源池）。

    一个源 = 一个池目录 + 一份文件级清单 + 权重 + 标签。选 `|` 而不是 `:` 是因为
    Windows 路径形如 `D:\\data\\far`（含冒号）；标签缺省取目录名（manifest 可读）。
    """
    parts = [p.strip() for p in str(spec).split(SOURCE_SEP)]
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise DMPoolError(
            f"源规格需形如 `DIR{SOURCE_SEP}LIST[{SOURCE_SEP}WEIGHT[{SOURCE_SEP}LABEL]]`，"
            f"实为 {spec!r}（分隔符是 '{SOURCE_SEP}'，不是 ':'——Windows 盘符含冒号）")
    if len(parts) > 4:
        raise DMPoolError(f"源规格字段过多（最多 4 个）：{spec!r}")
    d, lst = Path(parts[0]), Path(parts[1])
    weight = float(parts[2]) if len(parts) > 2 and parts[2] else float(default_weight)
    if not (weight > 0):
        raise DMPoolError(f"源权重需 > 0，实为 {weight}（{spec!r}）")
    label = parts[3] if len(parts) > 3 and parts[3] else d.name
    return {"dir": d, "list": lst, "weight": weight, "label": label}


def resolve_side_sources(args, side, *, required=True, weights=None):
    """把一个侧的 CLI 取值解析成源列表（`--{side}-source` 优先，兼容 v5 的 dir/list）。

    - `--{side}-source`（可重复）给出全部源；此时再给 `--{side}-dir/--{side}-list` 是
      **歧义**（不静默择一，直接报错）；
    - 不给 `--{side}-source` 时退化为 v5 的单源形态（dir + list，权重 1.0）；
    - `weights`（如 `--noise-source-weights`）覆盖内联权重，长度必须与源数相等。
    """
    specs = getattr(args, f"{side}_source", None) or []
    d, lst = getattr(args, f"{side}_dir", None), getattr(args, f"{side}_list", None)
    if specs and (d is not None or lst is not None):
        raise SystemExit(
            f"--{side}-source 与 --{side}-dir/--{side}-list 不能同时给（口径歧义）："
            f"多源请全部用 --{side}-source，单源请用 --{side}-dir/--{side}-list")
    sources = [parse_source_spec(s) for s in specs]
    if not sources:
        if d is None or lst is None:
            if required:
                raise SystemExit(
                    f"缺 {side} 池目录与显式分割清单（参数名 {side}_dir/{side}_list）。\n"
                    f"  单源：--{side}-dir <池> --{side}-list <清单>\n"
                    f"  多源：重复 --{side}-source "
                    f"\"DIR{SOURCE_SEP}LIST[{SOURCE_SEP}WEIGHT[{SOURCE_SEP}LABEL]]\"\n"
                    f"  清单格式见 split_lists/README.md；DM 现合成没有 manifest 兜底，"
                    f"不给清单等于放弃 0-2/0-3 的隔离纪律")
            return []
        sources = [{"dir": Path(d), "list": Path(lst), "weight": 1.0, "label": Path(d).name}]
    if weights:
        vals = [float(x) for x in weights]
        if len(vals) != len(sources):
            raise SystemExit(f"--{side}-source-weights 有 {len(vals)} 个值，但有 "
                             f"{len(sources)} 个 {side} 源（必须一一对应）")
        for s, w in zip(sources, vals):
            if not (w > 0):
                raise SystemExit(f"--{side}-source-weights 需全 > 0，实为 {vals}")
            s["weight"] = w
    # 标签唯一（manifest 里按标签索引，重名会掩盖源）
    labels = [s["label"] for s in sources]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"{side} 源标签重复：{labels}（标签用于 manifest 留痕，必须唯一）")
    return [dict(s, weight_norm=0.0) for s in sources]      # weight_norm 由 build_pools_multi 填


def load_pool(pool_dir, list_path, *, exts=AUDIO_EXTS):
    """池目录 + 清单 → 文件绝对路径列表（校验成员都在池内、名字唯一）。

    清单成员支持两种写法（P1X-2 扩展，向后兼容）：
      1. **相对路径**（推荐给多语料：`train-clean-360/1079/128631/1079-128631-0000.flac`、
         `music/fma-western-art/music-fma-wa-0000.wav`）—— 按池目录拼接后必须存在；
      2. **裸文件名**（`split_lists/` 的 0-2/0-3 老惯例，如 `book_..._reader_08994_0.wav`）
         —— 在递归索引里查名；**同名歧义**（池里两个不同文件同 basename）直接报错，
         否则清单指向哪个文件是不确定的。

    只对第 2 种写法才建递归索引（LibriSpeech 那种十几万文件的池子，纯相对路径清单
    不必扫盘）。返回列表的**顺序 = 清单顺序**（清单序 = 采样序，是接口的一部分）；
    索引枚举序用 `sort_pool_paths`（平台无关，见 `aec_synth.pool_sort_key`）。
    """
    pool_dir = Path(pool_dir)
    if not pool_dir.is_dir():
        raise DMPoolError(f"池目录不存在：{pool_dir}")
    names = read_split_list(list_path)
    index: dict = {}
    need_index = [n for n in names if not (pool_dir / n).is_file()]
    if need_index:
        dup = []
        for p in sort_pool_paths(pool_dir, _iter_audio(pool_dir, exts)):
            if p.name in index:
                dup.append(p.name)
            else:
                index[p.name] = p
        if dup:
            raise DMPoolError(f"池内存在同名音频（清单只带文件名时无法消歧）：{pool_dir} "
                              f"例：{sorted(set(dup))[:3]}")
        missing = [n for n in need_index if n not in index]
        if missing:
            raise DMPoolError(
                f"清单 {list_path} 有 {len(missing)} 个成员不在池 {pool_dir} 内（例："
                f"{missing[:3]}）；成员需是相对路径或池内唯一的裸文件名")
    # 保序解析：清单序 = 采样序
    return [(pool_dir / n) if (pool_dir / n).is_file() else index[n] for n in names]


def _iter_audio(root, exts):
    for ext in exts:
        yield from root.rglob(f"*{ext}")


def probe_sample_rate(path):
    """读一条文件的采样率（不读波形）：用于 manifest 记录"哪个源需要重采样"。"""
    import soundfile as sf
    try:
        return int(sf.info(str(path)).samplerate)
    except Exception:                                    # pragma: no cover - 罕见坏文件
        return 0


def load_wav_sr(path, target_sr=SR):
    """读单声道波形并**按需重采样**到目标采样率（P1X-2：AISHELL-3 等 44.1k 语料）。

    与 `aec_synth.load_wav_16k` 的分工：后者是**静态合成路径的硬约束**（非 16k 直接断言
    报错，行为不变）；DM 现合成要在 worker 里吃多语料，44.1k 必须能进来，故这里做显式
    重采样（`scipy.signal.resample_poly` + gcd 约简，与仓内 aecmos_eval 同法）。哪些源
    被重采样由 `probe_sample_rate` 记录进 manifest（`resample.applied_to`）。
    """
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr == target_sr:
        return y
    g = gcd(int(sr), int(target_sr))
    return resample_poly(y, target_sr // g, sr // g).astype(np.float32)


def build_pools_multi(near_sources, far_sources, noise_sources, *, mutex_level="utterance",
                      target_sr=SR):
    """多源版的 `build_pools`：每侧可含多个（目录+清单+权重+标签）源。

    返回 `(near_files, far_files, noise_files, info)`：
      - 文件列表 = 各源清单**按源内清单序拼接**（采样序仍由清单决定）；
      - `info["sources"]` = 每源的留痕表（路径/清单 sha256/文件数/权重 raw+归一/探针采样率）；
      - `info["split_lists"]`：单源时与 v5 完全同形（`{path,n,sha256}`）；多源时给
        `{n, sha256_combined, sources}`（n = 总文件数）；
      - 互斥校验在**并集**上做（跨源泄漏同样拦）：near vs far 文件同一性 + utterance 主名，
        noise vs（near ∪ far）。
    """
    info_sources, all_files = {}, {}
    for side, sources in (("near", near_sources), ("far", far_sources),
                          ("noise", noise_sources)):
        if not sources:
            raise DMPoolError(f"{side} 至少需要一个源")
        total_w = float(sum(s["weight"] for s in sources))
        entries, files = [], []
        for s in sources:
            fs = load_pool(s["dir"], s["list"])
            entries.append({
                "label": s["label"], "dir": str(s["dir"]), "list": str(s["list"]),
                "list_sha256": _file_sha256(s["list"]), "n_files": len(fs),
                "weight_raw": float(s["weight"]),
                "weight_norm": float(s["weight"]) / total_w,
                "probe_sr": probe_sample_rate(fs[0]) if fs else 0,
            })
            files.extend(fs)
        all_files[side] = files
        info_sources[side] = entries

    near_files, far_files, noise_files = (all_files["near"], all_files["far"],
                                          all_files["noise"])
    reasons = S.nearfar_file_conflicts(near_files, far_files, level=mutex_level)
    n_sn, example = S.file_id_overlap(noise_files, near_files + far_files)
    if n_sn:
        reasons.append(f"{n_sn} 个 wav 同时出现在语音清单与噪声清单里（例：{example}）")
    if reasons:
        raise S.NearFarMutexError(
            "；".join(reasons)
            + "。修法：near/far 必须来自互斥 reader 切分（split_lists/speech_* 与 ext_*），"
              "噪声只能用 *_train70.txt（*_eval*.txt 对训练侧是禁区）；"
              "见 REMEDIATION_PLAN_V3.1.md §3.1/§0-2")
    split_lists = {}
    for side in ("near", "far", "noise"):
        ents = info_sources[side]
        if len(ents) == 1:
            e = ents[0]
            split_lists[side] = {"path": e["list"], "n": e["n_files"],
                                 "sha256": e["list_sha256"]}          # v5 同形
        else:
            h = hashlib.sha256()
            for e in ents:
                h.update(e["list_sha256"].encode("utf-8"))
            split_lists[side] = {"n": len(all_files[side]),
                                 "sha256_combined": h.hexdigest(), "sources": ents}
    multi = any(len(info_sources[s]) > 1 for s in ("near", "far", "noise"))
    resampled = sorted({e["label"] for s in ("near", "far", "noise")
                        for e in info_sources[s]
                        if e["probe_sr"] and e["probe_sr"] != int(target_sr)})
    info = {
        "pools": {side: (info_sources[side][0]["dir"] if len(info_sources[side]) == 1
                         else [e["dir"] for e in info_sources[side]])
                  for side in ("near", "far", "noise")},
        "split_lists": split_lists,
        "sources": info_sources,
        "sampling_scheme": ("weighted_source_then_uniform_intra" if multi
                            else "single_source"),
        "sampling_note": ("多源：每样本每侧先按归一权重抽源（1 次额外抽样）、再在源内均匀抽"
                          "文件 => 与 v5 单源**不逐样本对齐**（配方/参数域一致，仅数据池不同）"
                          if multi else
                          "单源：与 v5 完全同口径（无额外抽样，rng 流不变）"),
        "resample": {"target_sr": int(target_sr),
                     "policy": "soundfile 读原生采样率；非目标采样率用 scipy.signal."
                               "resample_poly（gcd 约简）重采样",
                     "applied_to": resampled},
        "scene_param_policy": ("inherit_v5_dt_plan_v1（单变量：v5x 只扩数据池，场景参数域"
                               "（SER/SNR/引导/时延/soft-clip/jitter）一字不改）"),
        "mutex": {"level": mutex_level, "near_far_conflicts": [], "noise_speech_overlap": n_sn,
                  "checked": True},
    }
    return near_files, far_files, noise_files, info


def build_pools(near_dir, near_list, far_dir, far_list, noise_dir, noise_list, *,
                mutex_level="utterance"):
    """v5 单源 API（兼容保留）：三份清单 + 三个池目录 → (near_files, far_files,
    noise_files, info)。实现走 `build_pools_multi`（单源 ⇒ 与历史逐字段/逐位一致）。

    互斥校验（0-2/0-3）：near vs far 文件同一性 + utterance 主名；noise vs（near ∪ far）。
    """
    def mk(d, lst):
        return [{"dir": Path(d), "list": Path(lst), "weight": 1.0, "label": Path(d).name}]

    return build_pools_multi(mk(near_dir, near_list), mk(far_dir, far_list),
                             mk(noise_dir, noise_list), mutex_level=mutex_level)


# ---------------------------------------------------------------- RIR 池留痕（t26）
def rir_list_order_check(rir_dir, rir_list, rir_files):
    """RIR 清单 vs 目录枚举的**顺序护栏**（t27）——单一实现，加载期与 manifest 共用。

    返回 `{"order_matches_pool", "n_in_list", "n_in_pool", "extra_pool_files", "warnings"}`。

    为什么需要：RIR 抽样**按下标**取（`rng.integers(len(rir_files))`）⇒ **顺序就是接口的一部分**：
    集合相同而顺序不同时，"同 seed 复现"会静默换成另一组 RIR（本轮 casc 续跑正依赖
    "清单序 ≡ 目录枚举序"，故这条护栏必须在开跑时就喊出来）。集合不同则是清单驱动的
    **隔离用法**（正常，不报错），但要说清"本次只用了清单内 n/m 条"，否则"切清单 = 零变化"
    的假设会被静默打破。
    """
    files = list(rir_files or [])
    if rir_list is None or rir_dir is None:
        return {"order_matches_pool": None, "n_in_list": len(files), "n_in_pool": None,
                "extra_pool_files": None, "warnings": []}

    def rel(p):
        p = Path(p)
        try:
            return p.relative_to(Path(rir_dir)).as_posix()
        except ValueError:                               # 不在池内（正常不会发生）
            return p.name

    got = [rel(p) for p in files]
    try:
        pool = [rel(p) for p in list_pool(rir_dir)]
    except OSError as exc:                               # 池枚举失败不挡训练，只记一笔
        return {"order_matches_pool": None, "n_in_list": len(got), "n_in_pool": None,
                "extra_pool_files": None,
                "warnings": [f"RIR 目录不可枚举（{exc}）：无法核对清单序与目录序"]}
    out = {"order_matches_pool": None, "n_in_list": len(got), "n_in_pool": len(pool),
           "extra_pool_files": None, "warnings": []}
    if set(got) == set(pool):
        out["order_matches_pool"] = (got == pool)
        out["extra_pool_files"] = 0
        if got != pool:
            i = next(k for k, (a, b) in enumerate(zip(got, pool)) if a != b)
            out["warnings"].append(
                f"RIR 清单 {Path(rir_list).name} 的顺序 != 目录字节序（首个差异第 {i + 1} 条："
                f"清单 {got[i]!r} vs 目录 {pool[i]!r}）：集合相同但 index->文件 映射变了 => "
                f"同 seed 不再复现历史 RIR 序列（要逐样本对齐，就把清单按目录字节序重排）")
    else:
        extra = sorted(set(pool) - set(got))
        missing = sorted(set(got) - set(pool))
        out["order_matches_pool"] = None                  # 集合不同时"顺序"无意义
        out["extra_pool_files"] = len(extra)
        if extra:
            out["warnings"].append(
                f"RIR 池内有清单外成员 {len(extra)} 条（例：{extra[:3]}）：本次只使用清单内 "
                f"{len(got)}/{len(pool)} 条（清单驱动的隔离用法；若要与此前的目录枚举对齐，"
                f"请先核对池是否被污染）")
        if missing:                                       # 通常已被 load_pool 拦在前面
            out["warnings"].append(f"RIR 清单有 {len(missing)} 条不在池内（例：{missing[:3]}）")
    return out


def load_rir_pool(rir_dir, rir_list=None, *, warn=True):
    """RIR 池 → 文件列表。给了清单 ⇒ **清单驱动**（隔离纪律：训练侧只吃 rir_train70.txt）；
    否则整目录枚举（`list_pool`，平台无关序）。

    为什么要清单驱动：RIR 池与本轮评测用的 RIR 若混在一起，靠命令行核对发现不了污染；
    清单化后 manifest 会记下清单名 + sha256（见 `rir_provenance`），可事后比对。

    t27 护栏：清单驱动时当场核对"清单序 vs 目录字节序"（集合相同顺序不同 ⇒ **warn**，
    因为 index→文件 映射变了；集合不同 ⇒ 既有行为不变，只在池有多余成员时提示
    "本次只用清单内 n/m 条"）。告警同时写入 `provenance()["rir"]`（`order_matches_pool`/
    `extra_pool_files`/`warnings`），落进 train_manifest 可事后审计。
    """
    if rir_dir is None:
        return None
    if rir_list is None:
        return list_pool(rir_dir)
    files = load_pool(rir_dir, rir_list, exts=(".wav",))
    if warn:
        for w in rir_list_order_check(rir_dir, rir_list, files)["warnings"]:
            print(f"[dm] [warn] {w}", flush=True)
    return files


def _names_sha256(names):
    """一组名字的集合指纹（排序后拼接，跨平台稳定）。"""
    return hashlib.sha256(("\n".join(sorted(str(n) for n in names)) + "\n")
                          .encode("utf-8")).hexdigest()


def _pool_rel_name(rir_dir, p):
    """池内相对名（POSIX 分隔符）；不在池内则退回文件名。

    **指纹口径**：集合指纹按"池内相对名"算，而不是绝对路径——换服务器/换挂载点（同一套 RIR
    放在别的目录）时指纹必须不变，否则"池动过"会是假警报（本轮新服务器续跑正踩这个场景）。
    """
    p = Path(p)
    if rir_dir:
        try:
            return p.relative_to(Path(rir_dir)).as_posix()
        except ValueError:
            pass
    return p.name


def rir_provenance(rir_dir=None, rir_list=None, rir_files=None, *, max_ms=None):
    """RIR 池留痕块（进 `train_manifest.data.rir`）。

    背景（t13 发现的 schema 缺口）：near/far/noise 都记了池目录 + 清单 sha256，RIR 却只有
    `rir_src`/`rir_max_ms` ⇒ 本轮 RIR 污染只能靠**命令行 + 人工核对**发现。本块补齐：
    目录、清单文件名 + sha256、实际使用文件数、**使用集合指纹**（可 diff 两次 run 的 RIR 池）、
    枚举口径、截断长度，以及 t27 的顺序护栏结果（`order_matches_pool`/`warnings`）。

    `set_sha256` 之所以足够（不必再记有序列表）：无清单时枚举序是 `list_pool` 的
    **平台无关字节序**（t19），有清单时枚举序就是清单序（清单自身 sha256 已记）——
    两种情况"集合 + 枚举口径"都能唯一确定 index→文件 映射。指纹按**池内相对名**算
    （见 `_pool_rel_name`）：换机器/换挂载点不变，且当清单恰好按字节序枚举整池时
    `set_sha256 == list_sha256`（可直接与 teammate 已留档的基线指纹对账）。
    """
    files = [Path(p) for p in rir_files] if rir_files else []
    if rir_list is not None:
        src, enum = "list", "list_order"
    elif files:
        src, enum = "dir", "list_pool_byte_order"
    else:
        src, enum = "simulated", "n/a(simulated)"
    names = [_pool_rel_name(rir_dir or (files[0].parent if files else None), p) for p in files]
    out = {
        "src": src,
        "dir": str(rir_dir) if rir_dir else None,
        "list": str(rir_list) if rir_list else None,
        "list_sha256": _file_sha256(rir_list) if rir_list else None,
        "n_files": len(files),
        "set_sha256": _names_sha256(names) if names else None,
        "set_sha256_basis": "池内相对路径（POSIX，排序后）",
        "enumeration": enum,
        "max_ms": float(max_ms) if max_ms is not None else None,
        "note": ("RIR 池身份留痕：dir/list/list_sha256/set_sha256 用于事后发现池污染；"
                 "无清单时枚举序 = list_pool 的平台无关字节序（t19）"),
    }
    if rir_dir is None and files:
        out["dir"] = str(Path(files[0]).parent)      # 兜底：调用方只给了文件列表
    if rir_list is not None:
        # t27 顺序护栏：结果入 manifest（加载期另外会当场 warn）
        chk = rir_list_order_check(out["dir"], rir_list, files)
        out["order_matches_pool"] = chk["order_matches_pool"]
        out["n_in_pool"] = chk["n_in_pool"]
        out["extra_pool_files"] = chk["extra_pool_files"]
        if chk["warnings"]:
            out["warnings"] = chk["warnings"]
    return out


# ---------------------------------------------------------------- CLI 装配（07/04/dm_aec_synth 共用）
def add_dm_source_args(ap):
    """给训练/冒烟入口补上**多源池**参数（P1X-2）——一套实现，三个入口共用。

    每个侧两个旗标：
      `--{side}-source "DIR|LIST[|WEIGHT[|LABEL]]"`（可重复）= 多源；
      `--{side}-source-weights W1,W2,...`（可选）= 覆盖内联权重（长度须等于源数）。
    与 v5 的 `--{side}-dir/--{side}-list` 互斥（同时给会报错，不静默择一）。
    """
    for side, cn in (("near", "近端/目标语音"), ("far", "远端/回声源"),
                     ("noise", "噪声")):
        ap.add_argument(f"--{side}-source", action="append", default=None,
                        metavar="DIR|LIST[|W[|LABEL]]",
                        help=f"{cn}源（可重复；字段分隔符 '{SOURCE_SEP}'，**不是** ':'，"
                             f"因为 Windows 盘符含冒号）。权重列可留空（如 DIR|LIST||label，"
                             f"权重走 --{side}-source-weights 或缺省 1.0）；"
                             f"与 --{side}-dir/--{side}-list 互斥")
        ap.add_argument(f"--{side}-source-weights", default=None,
                        help=f"{cn}各源权重（逗号分隔，长度 = 源数；覆盖内联权重）")


def build_pools_from_args(args, *, mutex_level="utterance"):
    """从 CLI 参数装配三侧池（v5 单源与 P1X-2 多源走同一入口，口径一处实现）。"""
    def side(name):
        w = getattr(args, f"{name}_source_weights", None)
        return resolve_side_sources(args, name,
                                    weights=[x for x in str(w).split(",")] if w else None)
    return build_pools_multi(side("near"), side("far"), side("noise"),
                             mutex_level=mutex_level)


def sources_summary(info):
    """给日志/摘要用的源表（每侧 `label×权重(n)`）。"""
    out = {}
    for side, ents in (info.get("sources") or {}).items():
        out[side] = [f"{e['label']}×{e['weight_norm']:.2f}({e['n_files']})" for e in ents]
    return out


def load_plan(path=None, preset=None):
    """载入 dt_plan 清单并解析 preset → (preset_dict, plan_info)。"""
    p = Path(path) if path else DEFAULT_PLAN
    plan, sha = DP.load_plan(p)
    rp = DP.resolve_preset(plan, preset, sha)
    info = {"path": str(p), "version": rp["version"], "preset": rp["preset"],
            "sha256": rp["plan_sha256"], "schema": plan.get("schema", ""),
            "scenarios": rp["scenarios"], "dur_sec": rp["dur_sec"]}
    return rp, info


def with_scenario_ratio(preset, ratios):
    """显式覆盖场景配比（FEST,NST,DT 三个非负数且和为 1），其余分布仍取 preset。

    为什么要这个口子：清单 preset 描述的是**数据集**配比（`train_v1_repro` = 0.4/0.3/0.3），
    而 v4 的**训练**还叠了 `--scenarios dt,nst` 过滤（cfg 实证：没见过 FEST 场景）——
    两者不是一回事。阶段1 要不要复刻 v4 训练配比是个口径决定，调用方显式给、并留痕，
    比在代码里替他选一个更好。此处只做数值校验 + 生成新 preset 副本（不改清单）。
    """
    vals = [float(x) for x in ratios]
    if len(vals) != 3 or min(vals) < 0 or abs(sum(vals) - 1.0) > 1e-6:
        raise DMPoolError(f"场景配比需 3 个非负且和为 1 的数（FEST,NST,DT），实为 {ratios}")
    return dict(preset, scenarios=dict(zip(DP.SCENARIOS, vals)))


# ---------------------------------------------------------------- epoch 口径（方案 a）
def resolve_epoch_scheme(epochs, batch_size, target_steps=V4_REF_STEPS,
                         epoch_len_override=None, reference=None):
    """方案 (a)：把「等效更新步数」换算成 DM 的 ``epoch_len``（样本数）。

    ``steps_per_epoch = round(target_steps / epochs)``、``epoch_len = steps_per_epoch ×
    batch_size``；配合 DataLoader ``drop_last=True``，每 epoch 恰好 steps_per_epoch 步，
    总步数 = steps_per_epoch × epochs。取整不可避免地带 ≤0.5 步/epoch 的残差，**如实
    记进 manifest**（``delta_steps``/``delta_frac``），不假装精确相等。

    240ep / bs16 下（缺省目标 = v4 的 40830 步）：40830/240 = 170.125 → 170 步/ep
    （epoch_len 2720），总 40800 步（比 v4 少 30 步 = −0.073%）。

    ``reference``：目标步数的出处；别的模型要用自己的历史口径时传入（如 DCCRN 的
    18.7K 步），缺省 = v4 参照（此时返回里额外带 ``v4_ref`` 键，既有调用/单测不变）。

    ``epoch_len_override``（CLI 显式 epoch_len，冒烟/调试用）优先于换算：此时
    ``steps_per_epoch = epoch_len // batch_size``（drop_last），``epoch_len_source``
    记为 ``cli_override``——覆盖事实与由此产生的偏差一并留痕，不假装是方案 (a) 的结果。
    """
    if epochs <= 0:
        raise ValueError(f"epochs 需 > 0，实为 {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size 需 > 0，实为 {batch_size}")
    ref = dict(reference) if reference else {
        "name": "v4", "steps": V4_REF_STEPS, "epochs": V4_REF_EPOCHS,
        "batch_size": V4_REF_BATCH, "source": V4_REF_SOURCE}
    if epoch_len_override is not None:
        epoch_len = int(epoch_len_override)
        if epoch_len <= 0:
            raise ValueError(f"epoch_len 需 > 0，实为 {epoch_len}")
        steps_per_epoch = epoch_len // batch_size          # drop_last=True
        source = "cli_override"
    else:
        steps_per_epoch = max(1, int(round(target_steps / epochs)))
        epoch_len = steps_per_epoch * batch_size
        source = "derived_scheme_a"
    achieved = steps_per_epoch * epochs
    out = {
        "scheme": DM_EPOCH_SCHEME,
        "target_steps": int(target_steps),
        "steps_per_epoch": steps_per_epoch,
        "epoch_len": epoch_len,
        "epoch_len_source": source,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "achieved_steps": achieved,
        "delta_steps": achieved - int(target_steps),
        "delta_frac": (achieved - int(target_steps)) / float(target_steps),
        "reference": ref,
        "note": "附录B 方案(a)：等效更新步数对齐历史口径；假设 DataLoader drop_last=True",
    }
    if reference is None:
        out["v4_ref"] = ref            # 兼容键（缺省目标 = v4；既有单测依赖）
    return out


# ---------------------------------------------------------------- 数据集
def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """far 相对 mic 的整数样本平移（正 k 右移），越界补零。

    与 `dataset._shift` 同实现，但**有意本地复制**：本仓 exp 目录之间存在同名模块
    （model/train/dataset）作用域冲突的历史（见 tests/conftest.py §3.5），
    dm_dataset 又会被测试 shim 与 scripts 两处加载——本地实现可避免"取到别的 exp 的
    dataset"。两者的等价性由 tests/test_nn_aec_dm.py 的用例对账。
    """
    out = np.zeros_like(x)
    if k >= 0:
        out[k:] = x[:len(x) - k]
    else:
        out[:len(x) + k] = x[-k:]
    return out


class AecDynamicMix(Dataset):
    """虚拟长度数据集：``__len__ = epoch_len``，``__getitem__`` 现场合成一条三元组。

    参数
    ----
    near_files/far_files/noise_files : 三个**互斥**池的文件路径（见 build_pools）
    preset            : `dt_plan.resolve_preset` 解析后的 preset（场景配比 + DT 分布）
    plan_info         : 清单留痕（进 manifest / 日志）
    epoch_len         : 每 epoch 的样本数（方案 (a) 由 resolve_epoch_scheme 给出）
    seg_sec           : 片段时长（应与 preset.dur_sec 一致，由调用方核对）
    rir_files         : 自备 RIR 列表；None → pyroomacoustics 仿真（缺则指数衰减兜底）
    rir_max_ms        : RIR 截断（v4 口径 100ms）
    jitter_ms         : far 随机平移 ±（训练期时延鲁棒，v4 口径 10ms）
    soft_clip_prob    : 功放软剪切样本比例（v4 口径 0.15）
    seed              : None=每次取样吃熵（训练）；整数=逐 (seed, idx) 确定（审计/对照）
    """

    def __init__(self, near_files, far_files, noise_files, preset, *, plan_info=None,
                 pool_info=None, epoch_len=2000, seg_sec=4.0, rir_files=None,
                 rir_max_ms=100.0, delay_ms_range=(10.0, 80.0), soft_clip_prob=0.15,
                 jitter_ms=10.0, seed=None, rir_dir=None, rir_list=None):
        if not near_files or not far_files or not noise_files:
            raise DMPoolError("near/far/noise 三个池都不能为空（DM 入口即校验）")
        self.near_files = [Path(p) for p in near_files]
        self.far_files = [Path(p) for p in far_files]
        self.noise_files = [Path(p) for p in noise_files]
        self.rir_files = [Path(p) for p in rir_files] if rir_files else None
        self.preset = preset
        self.plan_info = dict(plan_info or {})
        self.pool_info = dict(pool_info or {})
        self.epoch_len = int(epoch_len)
        if self.epoch_len <= 0:
            raise DMPoolError(f"epoch_len 需 > 0，实为 {epoch_len}")
        self.seg_sec = float(seg_sec)
        self.n = int(seg_sec * SR)
        self.rir_max_ms = float(rir_max_ms)
        # RIR 池留痕（t26）：目录 + 清单（若给了）+ 实际使用集合的指纹，进 manifest。
        # 只存"从哪来"，块本身由 rir_provenance() 现算（单一实现，避免两处口径漂移）。
        self.rir_dir = Path(rir_dir) if rir_dir else None
        self.rir_list = Path(rir_list) if rir_list else None
        self.delay_ms_range = (float(delay_ms_range[0]), float(delay_ms_range[1]))
        self.soft_clip_prob = float(soft_clip_prob)
        self.jitter = int(jitter_ms * SR / 1000)
        self.seed = seed
        # 多源（P1X-2）：每侧的源分组（标签/文件/归一权重）。缺省 = 单源（= 整池一源），
        # 采样路径与 v5 **逐位一致**（不额外抽随机数，见 _pick_file）。
        self.source_groups = self._make_source_groups()

    def __len__(self):
        return self.epoch_len

    # ---------------------------------------------------------- 多源（P1X-2）
    def _make_source_groups(self):
        """从 pool_info["sources"] 还原每侧的源分组；缺省整池一源（v5 单源）。"""
        groups = {}
        for side, files in (("near", self.near_files), ("far", self.far_files),
                            ("noise", self.noise_files)):
            ents = (self.pool_info.get("sources") or {}).get(side)
            if not ents or len(ents) == 1:
                groups[side] = None                    # 单源：走 v5 路径（无额外抽样）
                continue
            off, gs = 0, []
            for e in ents:
                n = int(e["n_files"])
                gs.append({"label": e["label"], "files": files[off:off + n],
                           "weight": float(e.get("weight_norm", 0.0))})
                off += n
            if off != len(files):
                raise DMPoolError(
                    f"{side} 源分组与文件列表不一致（分组 {off} vs 列表 {len(files)}）——"
                    f"源表须来自 build_pools_multi")
            tot = sum(g["weight"] for g in gs) or 1.0
            for g in gs:
                g["weight"] = g["weight"] / tot
            groups[side] = gs
        return groups

    def _pick_file(self, rng, side, files):
        """抽一个文件：单源 = 与 v5 完全一致的 `rng.integers(n)`；多源 = 先按权重抽源
        （1 次额外抽样）再在源内均匀抽。

        ⚠️ 单源分支**不加任何抽样**，这是 "v5 单源行为逐位不变" 的硬保证
        （tests/golden/dm_single_source_v5.json 固化了对账基准）。
        """
        groups = self.source_groups.get(side)
        if not groups:
            return files[int(rng.integers(len(files)))]
        ws = np.array([g["weight"] for g in groups], dtype=np.float64)
        ws = ws / ws.sum()
        g = groups[int(rng.choice(len(groups), p=ws))]
        return g["files"][int(rng.integers(len(g["files"])))]

    @property
    def jitter_ms(self):
        """训练期 far 相对 mic 的平移幅度（ms，整数样本换算后的实际值）。"""
        return self.jitter * 1000.0 / SR

    def _rng(self, idx):
        # 不碰全局 RNG：固定 seed 时逐 (seed, idx) 确定；否则系统熵（训练期多样性）
        return (np.random.default_rng([int(self.seed), int(idx)])
                if self.seed is not None else np.random.default_rng())

    def _augment_far(self, far_r, rng):
        """far 相对 mic 的训练期平移（v4 口径 ±jitter_ms）；抽成方法供级联子类复用。"""
        if self.jitter and len(far_r) > 2 * self.jitter:
            return _shift(far_r, int(rng.integers(-self.jitter, self.jitter + 1)))
        return far_r

    def _post_synth(self, mic_r, far_r, near_r, rng):
        """合成后的收尾（默认）：只做 far 平移。

        抽成方法是为了级联（P1-3b）能在"合成"与"far 平移"之间插入线性级：静态级联集
        是**先跑完 MDF、再对 `farend/`(= y_echo) 做 jitter**（见 precompute_cascade.py +
        AecDataset 的 jitter），DM 侧必须同序；这样级联版与纯 DM 版在**同 seed 下消耗同一条
        rng 流**（场景/文件/SER/SNR/时延/soft-clip 与平移量全部对齐），可做配对对照。
        """
        return mic_r, self._augment_far(far_r, rng), near_r

    def __getitem__(self, idx):
        rng = self._rng(idx)
        ps = self.preset
        scene = DP.sample_ratio(rng, ps)                      # 1 random（清单配比）
        near_f = self._pick_file(rng, "near", self.near_files)
        far_f = self._pick_file(rng, "far", self.far_files)
        noise_f = self._pick_file(rng, "noise", self.noise_files)
        near = S.fit_len(load_wav_sr(near_f, SR), self.n, rng)
        far = S.fit_len(load_wav_sr(far_f, SR), self.n, rng)
        noise = S.fit_len(load_wav_sr(noise_f, SR), self.n, rng)
        rir = S.get_rir(self.rir_files, rng, max_ms=self.rir_max_ms)
        s = DP.sample_dt(rng, ps)                             # 3 uniform：ser->snr->leadin
        ser_db, snr_db, leadin_sec = s["ser_db"], s["snr_db"], s["leadin_sec"]
        if scene == "dt" and leadin_sec > 0:
            near[:int(leadin_sec * SR)] = 0.0                 # 引导段纯远端（同 aec_synth）
        far_r, near_r, mic_r, _noise_r, par = S.generate_one(
            near, far, noise, rir, scene, ser_db=ser_db, snr_db=snr_db,
            delay_ms=float(rng.uniform(*self.delay_ms_range)),
            use_soft_clip=bool(rng.random() < self.soft_clip_prob))
        # 合成后的收尾（默认只做 far 平移）：级联子类在这里插入线性级（见 _post_synth）
        mic_r, far_r, near_r = self._post_synth(mic_r, far_r, near_r, rng)
        return {
            "mic": torch.from_numpy(np.ascontiguousarray(mic_r)),
            "far": torch.from_numpy(np.ascontiguousarray(far_r)),
            "near": torch.from_numpy(np.ascontiguousarray(near_r)),
            "name": f"dm_{idx}_{scene}",
            "scenario": scene,
            # 审计字段（collate 会丢弃；DM 冒烟/单测用来对账清单分布）
            "ser_db": float(par["ser_db"]),
            "snr_db": float(par["snr_db"]),
            "leadin_sec": float(leadin_sec) if scene == "dt" else 0.0,
            "soft_clip": int(par["soft_clip"]),
            "delay_ms": float(par["delay_ms"]),
        }

    # ---------------------------------------------------------- 留痕
    def provenance(self):
        """本次 DM 供给口径的留痕（进 train_manifest.json / 日志）。

        P1X-2 多源：额外带 `sources`（每侧每源的池/清单/sha256/文件数/权重 raw+归一/
        探针采样率）、`sampling_scheme`、`resample`、`scene_param_policy`。单源时
        `pools`/`split_lists` 与 v5 完全同形（既有 manifest 读法不变）。
        """
        info = self.pool_info
        return {
            "mode": "dynamic_mixing",
            "epoch_len": self.epoch_len,
            "seg_sec": self.seg_sec,
            "seed": self.seed,
            "seed_mode": "per_idx_deterministic" if self.seed is not None else "entropy",
            "dt_plan": self.plan_info,
            "pools": info.get("pools", {}),
            "split_lists": info.get("split_lists", {}),
            "sources": info.get("sources", {}),
            "sampling_scheme": info.get("sampling_scheme", "single_source"),
            "sampling_note": info.get("sampling_note", ""),
            "resample": info.get("resample", {}),
            "scene_param_policy": info.get("scene_param_policy", ""),
            "mutex": info.get("mutex", {}),
            "source_groups": {side: (None if not gs else
                                     [{"label": g["label"], "n": len(g["files"]),
                                       "weight_norm": round(g["weight"], 6)} for g in gs])
                              for side, gs in self.source_groups.items()},
            "n_files": {"near": len(self.near_files), "far": len(self.far_files),
                        "noise": len(self.noise_files),
                        "rir": len(self.rir_files) if self.rir_files else 0},
            # RIR 池留痕（t26）：目录/清单+sha256/使用集合指纹/枚举口径（单一实现）
            "rir": rir_provenance(self.rir_dir, self.rir_list, self.rir_files,
                                  max_ms=self.rir_max_ms),
            "rir_src": "dir" if self.rir_files else _rir_src(),
            "rir_max_ms": self.rir_max_ms,
            "delay_ms_range": list(self.delay_ms_range),
            "soft_clip_prob": self.soft_clip_prob,
            "jitter_ms": self.jitter_ms,
        }


def _rir_src():
    """RIR 来源标签（与 aec_synth.main 的判定一致：sim / exp 兜底）。"""
    try:
        import pyroomacoustics  # noqa: F401
        return "sim"
    except ImportError:
        return "exp"
