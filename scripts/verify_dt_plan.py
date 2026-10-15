#!/usr/bin/env python
"""0-5 验收脚本：DT 比例清单可读、可比、可复现（清单 <-> manifest 双向核对）。

判定项（任一不满足即 **退出码非零**）
-------------------------------------
A. **清单结构**：schema / version（清单版本号）/ preset 齐全；scenarios 非负且和为 1；
   三个 DT 分布字段合法（lo<=hi，kind 受支持）。
B. **留痕闭环**：manifest 里 `dt_plan_version` == 清单 version，`dt_plan_preset` 落在清单内；
   同目录 `dt_plan_profile.json` 记录的清单 sha256 / version 与实际文件一致。
C. **可复现同分布（核心）**：实跑一次当前 `aec_synth.py`（同 seed、同清单、同池），按
   "每样本以 `random` 收尾、收尾前恰好 6 次 `uniform`"的尾结构锚定每样本的清单采样块，
   要求 DT 的 ser/snr/leadin 与该块中的真实抽样值**逐行完全相等**；另用块首那次
   `random`（pick_scene 的抽值）按清单配比反推场景、与 manifest 的 scenario 列对账——
   这条独立证据证明"切流对齐"正确，而不是靠自身结构自证。
   **两种 RIR 来源共用这一条路径**（`check_evidence`）：既然 RIR 段长度取决于来源
   （dir=固定一次 `integers`；sim=pyroomacoustics 自取若干个），就不去离线复刻这两种
   内部细节，直接观察真实现场流——证据强度与离线重放等价（同样逐行、同样精确相等）。
   这条把"清单 = 声明口径"钉死：改一行 manifest 或换一次抽样都会失败。
   需要源目录（`DT_PLAN_NEAR_DIR` / `DT_PLAN_FAR_DIR` / `DT_PLAN_NOISE_DIR`，可选
   `DT_PLAN_RIR_DIR`）才能实跑；大集建议配 `--limit N` 只核前 N 行。
D. **区间合规**：所有 DT 行的 ser/snr/leadin 落在清单声明区间内（容许浮点末位误差）。
E. **场景占比**：实测 DT 占比与清单声明概率的偏差 ≤ 3σ 二项界（防"清单写 0.5、实测 0.25"）。
F. **预设专项**（--strict-preset）：`train_v1_repro` 要求 DT leadin 恒 1.5s、SNR ⊂ [15,30]；
   `eval_v5_repro` 要求 DT leadin 恒 2.0s、SNR ⊂ [20,30]；两者 SER ⊂ [0,10]。
G. **前缀回放（--replay-prefix N）**：现场重生成**前 N 条**（default_rng 的前缀确定性见
   `reports/refactor_wave5/data_inventory.md` §5.2），与既有 manifest 前 N 行逐字节一致，
   并对抽到的 wav 做逐字节抽查 —— 证明"同命令同 seed 产出同一批文件"。
   **sim-RIR 集默认跳过**（需本机装 pyroomacoustics 且版本一致；实测本机 20 条前缀下
   单次约 1.2s 量级），要跑就显式加 `--replay-sim-rir`（建议小前缀，如 3-5）。

适用边界（**务必先看这一条**）
------------------------------
- A/B/D/E/F：与 RIR 来源无关，永远适用。
- C：dir 与 sim **都适用**，实现同为"实跑 + 收尾结构锚定 + 逐行对账"（见上）。报告里建议
  写清 `rir=` 值（脚本会打印 `rir=dir|sim`）以便复现者知道本集的 RIR 来源。
- G：dir 模式直接可用；sim 模式需 `--replay-sim-rir` **且**本机 pyroomacoustics 可用。
  **不要在缺该依赖的机器上把 G 的跳过当成失败**——那是环境限制，不是数据/清单问题。
- 模式判定：`--rir-mode auto`（默认）按 profile 的 `pool_sizes.rir` / `pool_dirs.rir` 判定
  （rir > 0 或非空 ⇒ dir）；也可用 `--rir-mode dir|sim` 显式覆盖。
- 存量的历史集（eval_v5/v7 等）由**旧版** aec_synth 生成：没有 `dt_plan_version` 列与
  `dt_plan_profile.json`，A/B/C/E/F 都无从核对（脚本会明确报"未按清单采样"），这不是回归。

退出码：0 = 全部通过；1 = 有不通过项；2 = 输入/环境错误（清单或 manifest 缺失/非法）。

用法
----
    python scripts/verify_dt_plan.py --plan configs/dt_plan_v1.json
    python scripts/verify_dt_plan.py --plan configs/dt_plan_v1.json \\
        --manifest D:/data/aec-synth/eval_v5/manifest.csv --strict-preset
    # sim-RIR 集（eval_v5/v7/v8）：C 走实测流核对（需源目录），G 默认跳过
    DT_PLAN_NEAR_DIR=... DT_PLAN_FAR_DIR=... DT_PLAN_NOISE_DIR=... \\
    python scripts/verify_dt_plan.py --plan configs/dt_plan_v1.json \\
        --manifest D:/data/aec-synth/eval_v8/manifest.csv --strict-preset --limit 20
    # 产物逐字节复现（sim 集需 --replay-sim-rir；dir 集直接可用）
    python scripts/verify_dt_plan.py --plan configs/dt_plan_v1.json \\
        --manifest <manifest.csv> --set-dir <输出目录> --replay-prefix 50
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SCEN = ("fest", "nst", "dt")
HIST_PRESETS = ("train_v1_repro", "eval_v5_repro")
#: 预设专项期望（F 项）：preset -> (leadin 唯一值, snr 区间, ser 区间)
PRESET_EXPECT = {
    "train_v1_repro": (1.5, (15.0, 30.0), (0.0, 10.0)),
    "eval_v5_repro": (2.0, (20.0, 30.0), (0.0, 10.0)),
}
#: 回放前缀需要的源目录（来自既有合成配方；用 DT_PLAN_<KEY> 环境变量提供）
REPLAY_SOURCES = ("NEAR_DIR", "FAR_DIR", "NOISE_DIR")
SIGMA_K = 3.0                 # 场景占比容许倍数（二项 3σ）
ABS_TOL = 1e-9                # 逐行数值比对容差


def _load_dt_plan():
    sys.path.insert(0, str(HERE))
    import dt_plan as DP  # noqa: PLC0415
    return DP


def read_manifest(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest 不存在：{path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"manifest 为空：{path}")
    return rows


def profile_of(manifest: Path) -> dict | None:
    p = manifest.parent / "dt_plan_profile.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def detect_rir_mode(prof: dict | None, override: str = "auto") -> str:
    """判定该集的 RIR 来源：`dir`（--rir-dir 自备，C/G 可离线重放）或 `sim`（pyroomacoustics）。

    判据取自 profile：`pool_sizes.rir == 0` 或 `pool_dirs.rir is None` ⇒ sim。
    这是 C（随机流重放）适用性的分水岭：dir 时每样本的 RIR 抽取是固定一次
    `rng.integers(len(rir_files))`；sim 时 pyroomacoustics 在**同一条 rng 流**上按其内部
    算法自取若干个数（含 `uniform(size=...)` 数组抽样），离线重放需复刻其实现——不做。
    """
    if override != "auto":
        return override
    if not prof:
        return "sim"            # 无 profile 时无法判定；保守按 sim（跳过离线重放）
    sizes = prof.get("pool_sizes") or {}
    dirs = prof.get("pool_dirs") or {}
    if sizes.get("rir") or dirs.get("rir"):
        return "dir"
    return "sim"


def check_evidence(rows: list[dict], rp, prof: dict, plan_path: Path) -> list[str]:
    """C 的实现：实跑一次，核对"样本收尾结构 + 场景标签 + DT 采样值"（两种 RIR 来源通用）。

    证据强度与"离线重放"等价：都要求 DT 的 ser/snr/leadin 与随机流中该样本的真实抽样值
    逐行相等（改一行 manifest、或换一次抽样，都会失败）。区别只为：RIR 来源不同导致 RIR 段
    在流里的长度不同（dir=固定一次 `integers`；sim=pyroomacoustics 自取若干个），
    本实现不去离线复刻这两种内部细节，而是直接观察真实现场流。
    额外的**独立**证据：每样本首调用（pick_scene 的 random 值）按清单配比映射出的场景
    必须与该行 manifest 的 scenario 列一致——这证明"切流对齐"正确，而不是靠自身结构自证。
    """
    cli = prof.get("cli_args", {})
    src = {k: os.environ.get(f"DT_PLAN_{k}") for k in REPLAY_SOURCES}
    missing = [f"DT_PLAN_{k}" for k, v in src.items() if not v]
    if missing:
        return [f"C（同分布核对）需要源目录环境变量：{missing}"
                f"（否则无法实跑；适用边界见模块 docstring）"]
    argv = ["aec_synth.py", "--out-dir", tempfile.mkdtemp(),
            "--near-dir", src["NEAR_DIR"], "--far-dir", src["FAR_DIR"],
            "--noise-dir", src["NOISE_DIR"],
            "--dt-plan", str(plan_path), "--dt-plan-preset", prof.get("preset"),
            "--num", str(len(rows)), "--seed", str(prof.get("seed"))] + _replay_flags(cli)
    try:
        logged = _run_main_capture(argv)
    except Exception as e:                                   # noqa: BLE001 - 失败即 findings
        return [f"C（同分布核对）执行失败：{type(e).__name__}: {e}"]
    if not logged:
        return ["C（同分布核对）未捕获到 rng 实例"]
    log = logged[0].log
    problems = check_observed_stream(rows, log)
    problems += _check_scene_labels(rows, log, rp)
    return problems


def _check_scene_labels(rows: list[dict], log: list[tuple], rp) -> list[str]:
    """独立核对：实测流每样本首调用按清单配比映射的场景 == manifest 的 scenario 列。"""
    kinds = [c[0] for c in log]
    pos, scenes = 0, []
    for _ in range(len(rows)):
        if kinds[pos:pos + 1] != ["random"]:
            return [f"第 {len(scenes)} 行：样本未以 random 起始（实测 {kinds[pos:pos + 3]}）"]
        scenes.append(float(log[pos][1]))
        block = _find_dt_block(kinds, pos + 1)
        if block is None:
            return []
        pos = block[1]
    problems = []
    for i, (row, r) in enumerate(zip(rows, scenes)):
        want = row["scenario"]
        got = _scene_of(r, rp)
        if want != got:
            problems.append(f"第 {i} 行（{row['name']}）场景标签 manifest={want} "
                            f"实测流（pick_scene 抽值 {r:.6f}）={got}")
            if len(problems) >= 3:
                break
    return problems


def _scene_of(r: float, rp) -> str:
    """按清单累积配比把 pick_scene 的 uniform 抽值映射回场景名。"""
    acc = 0.0
    for s in SCEN:
        acc += float(rp["scenarios"][s])
        if r < acc:
            return s
    return SCEN[-1]


# ------------------------------------------------------------------ A
def check_structure(DP, plan, sha) -> list[str]:
    problems = []
    for name in HIST_PRESETS:
        if name not in plan["presets"]:
            problems.append(f"清单缺 preset {name}（历史口径必须可复现）")
    for name in plan["presets"]:
        rp = DP.resolve_preset(plan, name, sha)
        tot = sum(rp["scenarios"].values())
        if abs(tot - 1.0) > 1e-9:
            problems.append(f"preset {name}: scenarios 和 = {tot}（应 =1）")
        if rp["scenarios"]["dt"] <= 0:
            problems.append(f"preset {name}: DT 占比非正（清单未显式化 DT 口径）")
    return problems


# ------------------------------------------------------------------ B
def check_provenance(manifest: Path, rows: list[dict], plan, sha) -> list[str]:
    problems = []
    if "dt_plan_version" not in rows[0]:
        return [f"manifest 无 dt_plan_version 列（{manifest} 未按清单采样，0-5 无从核对）"]
    vers = {r["dt_plan_version"] for r in rows}
    if vers != {plan["version"]}:
        problems.append(f"dt_plan_version={sorted(vers)} != 清单 {plan['version']}")
    presets = {r.get("dt_plan_preset", "") for r in rows}
    if len(presets) != 1 or not presets <= set(plan["presets"]):
        problems.append(f"dt_plan_preset 列异常：{sorted(presets)}（应恰为清单中一个 preset）")
    prof = profile_of(manifest)
    if prof is None:
        problems.append(f"缺 dt_plan_profile.json（同目录）：{manifest.parent}")
    else:
        if prof.get("sha256") and prof["sha256"] != sha:
            problems.append("dt_plan_profile.json 记录的清单 sha256 与实际文件不符（清单被改过）")
        if prof.get("version") != plan["version"]:
            problems.append("dt_plan_profile.json 的 version 与清单不一致")
        if prof.get("preset") not in set(plan["presets"]):
            problems.append(f"dt_plan_profile.json 的 preset={prof.get('preset')!r} 不在清单内")
    return problems


# ------------------------------------------------------------------ C
def check_observed_stream(rows: list[dict], log: list[tuple]) -> list[str]:
    """用**现场实跑到的那条随机流**核对 DT 参数（C 的唯一实现，两种 RIR 来源通用）。

    为什么不做"离线重放"：RIR 段在流里的长度取决于 RIR 来源（dir=固定 1 次
    `integers(len(rir_files))`；sim=pyroomacoustics 自取若干个数），而 `fit_len` 的抽样
    次数又取决于源 wav 的**原始长度**——离线复刻这两件事会与实现细节耦合，且已实测容易
    静默错位。改为直接观察真实流：每样本以 `random` 收尾、收尾前恰好 6 次 `uniform`
    （前 2 次=被清单覆盖的 ser/snr，随后 3 次=清单 ser/snr/leadin，第 6 次=delay），
    据此锚定块首并逐行比对三列采样值。

    独立证据：块首那一次 `random` 是 `pick_scene` 的抽值，按清单配比映射出的场景必须等于
    该行的 scenario 列（`_check_scene_labels`）——它证明切流对齐正确，而不是靠自身结构自证。
    """
    problems: list[str] = []
    kinds = [c[0] for c in log]
    pos = 0
    n_dt = 0
    for i, row in enumerate(rows):
        if kinds[pos:pos + 1] != ["random"]:
            problems.append(f"第 {i} 行：样本未以 random 起始（实测 {kinds[pos:pos + 4]}）")
            break
        block = _find_dt_block(kinds, pos + 1)
        if block is None:
            problems.append(f"第 {i} 行（{row['name']}）：在流中找不到清单采样块"
                            f"（收尾 random 前应有 6 次 uniform）；该集可能不是按当前 "
                            f"aec_synth 生成")
            break
        uniform_idx, nxt = block
        if row["scenario"] == "dt":
            n_dt += 1
            vals = [log[uniform_idx + 2 + k][1] for k in range(3)]   # 跳过被覆盖的 ser/snr
            for key, got_v in zip(("ser_db", "snr_db", "leadin_sec"), vals):
                got, exp = float(row[key]), float(got_v)
                if not math.isclose(got, exp, rel_tol=0.0, abs_tol=ABS_TOL):
                    problems.append(f"第 {i} 行（{row['name']}）{key}: manifest={got!r} "
                                    f"实测流中该样本的采样值={exp!r}")
                    break
        pos = nxt
    if not problems and pos != len(kinds):
        problems.append(f"流未被完整消费（剩 {len(kinds) - pos} 次调用）——样本结构与模型不符")
    if n_dt == 0 and not problems:
        problems.append("manifest 无 DT 行（无法核对 DT 口径）")
    return problems


def _find_dt_block(kinds: list[str], start: int):
    """找"清单采样块"：该样本结束的那个 random 之前，恰好是 6 次 uniform。

    返回 (块首下标, 该样本结束后的下标)；找不到返回 None。RIR 段长度未知（sim-RIR 可变），
    故用尾结构回锚而不是从头数。
    """
    for p in range(start, len(kinds)):
        if kinds[p] == "random" and kinds[max(0, p - 6):p] == ["uniform"] * 6:
            return p - 6, p + 1
    return None


# ------------------------------------------------------------------ D/E
def check_ranges(DP, rows: list[dict], rp) -> list[str]:
    problems = []
    dt = [r for r in rows if r["scenario"] == "dt"]
    for field, key, label in (("ser_db", "dt_ser_db", "SER"),
                              ("snr_db", "dt_snr_db", "SNR"),
                              ("leadin_sec", "dt_leadin_sec", "leadin")):
        vals = [float(r[field]) for r in dt if r.get(field) not in (None, "")]
        if not vals:
            problems.append(f"DT 行缺 {field} 列")
            continue
        dist = rp[key]
        if min(vals) < dist["lo"] - 1e-6 or max(vals) > dist["hi"] + 1e-6:
            problems.append(f"DT {label} 实测 [{min(vals)}, {max(vals)}] 越出清单区间 "
                            f"[{dist['lo']}, {dist['hi']}]")
    n = len(rows)
    frac, p = len(dt) / n, rp["scenarios"]["dt"]
    tol = SIGMA_K * math.sqrt(max(p * (1 - p), 1e-12) / n) + 1e-9
    if abs(frac - p) > tol:
        problems.append(f"DT 占比实测 {frac:.4f} 偏离清单 {p}（{SIGMA_K}σ 容许 ±{tol:.4f}）")
    return problems


# ------------------------------------------------------------------ F
def check_preset_expect(rows: list[dict], preset_name: str) -> list[str]:
    exp = PRESET_EXPECT.get(preset_name)
    if not exp:
        return [f"--strict-preset 无 {preset_name!r} 的期望表（新增 preset 请同步登记）"]
    leadin, snr_rng, ser_rng = exp
    problems = []
    dt = [r for r in rows if r["scenario"] == "dt"]
    leads = sorted({round(float(r["leadin_sec"]), 6) for r in dt if r.get("leadin_sec")})
    if leads and leads != [leadin]:
        problems.append(f"preset {preset_name} 要求 DT leadin 恒为 {leadin}s，实测 {leads}")
    for field, (lo, hi), label in (("snr_db", snr_rng, "SNR"), ("ser_db", ser_rng, "SER")):
        vals = [float(r[field]) for r in dt if r.get(field) not in (None, "")]
        if vals and (min(vals) < lo - 1e-6 or max(vals) > hi + 1e-6):
            problems.append(f"preset {preset_name} 要求 {label} ⊂ [{lo},{hi}]，"
                            f"实测 [{min(vals)}, {max(vals)}]")
    return problems


# ------------------------------------------------------------------ G
def check_replay_prefix(DP, rp, plan_path: Path, manifest: Path, set_dir: Path,
                        n_prefix: int) -> list[str]:
    """用当前 aec_synth.py 重生成前 n_prefix 条，与既有 manifest 前缀逐字节比对。

    实测要拦两件事：① 生成的 manifest/波形是否与既有逐字节一致（可复现）；
    ② **实测的随机流调用序列**是否与本模块模型（`_rng_stream`）一致——只有两者对得上，
    "用清单 + seed 就能复算参数"这句话才成立。故这里在同进程里跑 `main()`
    （importlib 按路径加载，天然用已导入的 numpy），并临时替换 rng 与 `sf.write`：
    既拿到真实调用序列，又不真写 wav。
    """
    prof = profile_of(manifest)
    if prof is None:
        return [f"--replay-prefix 需要 {manifest.parent / 'dt_plan_profile.json'}"]
    cli = prof.get("cli_args", {})
    seed, preset = prof.get("seed"), prof.get("preset")
    if seed is None or not preset:
        return ["profile 缺 seed/preset，无法回放"]
    src = {k: os.environ.get(f"DT_PLAN_{k}") for k in REPLAY_SOURCES}
    missing = [f"DT_PLAN_{k}" for k, v in src.items() if not v]
    if missing:
        return [f"--replay-prefix 需环境变量提供源目录：{missing}"]
    problems: list[str] = []
    want = read_manifest(manifest)[:n_prefix]
    with tempfile.TemporaryDirectory() as td:
        argv = ["aec_synth.py", "--out-dir", td,
                "--near-dir", src["NEAR_DIR"], "--far-dir", src["FAR_DIR"],
                "--noise-dir", src["NOISE_DIR"],
                "--dt-plan", str(plan_path), "--dt-plan-preset", preset,
                "--num", str(n_prefix), "--seed", str(seed)] + _replay_flags(cli)
        try:
            logged = _run_main_capture(argv)
        except Exception as e:                                   # noqa: BLE001 - 回放失败即 findings
            return [f"回放执行失败：{type(e).__name__}: {e}"]
        if not logged:
            return ["回放未捕获到 rng 实例（main 未跑到建 rng？）"]
        # ① 产物逐字节（流结构/取值已由 C 的 check_observed_stream 核对）
        new_rows = read_manifest(Path(td) / "manifest.csv")
        if len(new_rows) < n_prefix:
            problems.append(f"回放只产出 {len(new_rows)} 行（要求 {n_prefix}）")
        else:
            diff = [i for i, (a, b) in enumerate(zip(new_rows, want)) if a != b]
            if diff:
                problems.append(f"回放 manifest 与既有前 {n_prefix} 行有 {len(diff)} 行不一致"
                                f"（首个：第 {diff[0]} 行）")
        for sub in ("mic", "nearend", "farend"):
            for f in sorted((Path(td) / sub).glob("*.wav"))[:3]:
                ref = set_dir / sub / f.name
                if ref.is_file() and f.read_bytes() != ref.read_bytes():
                    problems.append(f"回放波形与既有不一致（抽查）：{sub}/{f.name}")
                    break
    return problems


def _run_main_capture(argv: list[str]):
    """同进程跑 aec_synth.main（替换 rng 与 sf.write），返回捕获到的 LogRng 列表。"""
    spec = importlib.util.spec_from_file_location("aec_synth_verify", HERE / "aec_synth.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    real = np.random.default_rng           # 替换前抓住真身（避免自递归）
    logged: list[LogRng] = []

    def factory(seed, *a, **k):
        r = LogRng(real, seed)
        logged.append(r)
        return r

    saved_rng = mod.np.random.default_rng
    saved_write = mod.sf.write
    saved_argv = sys.argv
    mod.np.random.default_rng = factory
    mod.sf.write = lambda *a, **k: None    # 只拦写盘：合成数学全部真实执行
    sys.argv = argv
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            mod.main()
    finally:
        mod.np.random.default_rng = saved_rng
        mod.sf.write = saved_write
        sys.argv = saved_argv
    return logged


class LogRng:
    """记录 rng 调用序与**返回值**（供"实测流 vs manifest"对账）。

    必须完整转发 numpy 的位置/关键字参数：sim-RIR 路径会经 pyroomacoustics 调用
    `rng.uniform(lo, hi, size=n)`（数组抽样）——早期版本只接 `(a, b)` 两参，一遇到
    `size=` 就抛 TypeError，整条回放崩掉（t7 修的正是这个）。
    """

    def __init__(self, real_factory, seed):
        self._r = real_factory(seed)
        self.log: list[tuple] = []

    def random(self, *args, **kwargs):
        v = self._r.random(*args, **kwargs)
        self.log.append(("random", _scalar(v)))
        return v

    def uniform(self, *args, **kwargs):
        v = self._r.uniform(*args, **kwargs)
        self.log.append(("uniform", _scalar(v)))
        return v

    def integers(self, *args, **kwargs):
        v = self._r.integers(*args, **kwargs)
        self.log.append(("integers", _scalar(v)))
        return v


def _scalar(v):
    """数组抽样取首个元素（对账只看标量口径）；标量原样返回。"""
    arr = np.asarray(v)
    return float(arr.reshape(-1)[0]) if arr.size else float("nan")


def _replay_flags(cli: dict) -> list[str]:
    """按 profile 的 cli_args 还原**不经清单**的参数。

    必须还原的（漏一个就会静默换口径——本函数最初漏了 `--dur-sec`，导致回放用了 CLI 默认 4s
    而记录里是 8s，`fit_len` 抽样次数与全部样本随之改变，回放 49/50 行不符）：
      dur_sec / rir_max_ms / soft_clip_prob / delay_min_ms / delay_max_ms / delay_drift /
      nearfar_mutex。由清单决定的 ser/snr 区间与 scenarios **不**还原（清单是它们的来源）。
    """
    flags: list[str] = []
    for key, flag in (("dur_sec", "--dur-sec"),
                      ("rir_max_ms", "--rir-max-ms"),
                      ("soft_clip_prob", "--soft-clip-prob"),
                      ("delay_min_ms", "--delay-min-ms"),
                      ("delay_max_ms", "--delay-max-ms")):
        if cli.get(key) is not None:
            flags += [flag, str(cli[key])]
    if cli.get("delay_drift"):
        flags += ["--delay-drift", str(cli["delay_drift"])]
    if cli.get("nearfar_mutex") and cli["nearfar_mutex"] != "off":
        flags += ["--nearfar-mutex", str(cli["nearfar_mutex"])]
    if os.environ.get("DT_PLAN_RIR_DIR"):
        flags += ["--rir-dir", os.environ["DT_PLAN_RIR_DIR"]]
    return flags


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows 默认 GBK 控制台
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, required=True, help="DT 比例清单 JSON")
    ap.add_argument("--manifest", type=Path, default=None, help="待核对的合成 manifest.csv")
    ap.add_argument("--set-dir", type=Path, default=None, help="--replay-prefix 用：既有输出目录")
    ap.add_argument("--strict-preset", action="store_true", help="启用 F 项（preset 专项期望）")
    ap.add_argument("--replay-prefix", type=int, default=0, metavar="N",
                    help="启用 G 项：重生成前 N 条并与既有 manifest 前缀逐字节比对")
    ap.add_argument("--rir-mode", choices=("auto", "dir", "sim"), default="auto",
                    help="RIR 来源（决定 C/G 的适用性）：auto=按 profile 的 pool_sizes/pool_dirs "
                         "判定；dir=--rir-dir 自备 RIR（C/G 可离线重放）；sim=pyroomacoustics "
                         "仿真（C 改走实测流核对，G 默认跳过）")
    ap.add_argument("--replay-sim-rir", action="store_true",
                    help="sim-RIR 集也跑 G（需本机 pyroomacoustics，每条 ~1-2s；建议小前缀）")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="只核对 manifest 前 N 行（0=全部）。sim-RIR 的实测流核对要跑仿真，"
                         "大集建议先取前缀（前缀确定性：同 seed 下 --num N 的输出 = 全量的前 N 行）")
    args = ap.parse_args()

    DP = _load_dt_plan()
    try:
        plan, sha = DP.load_plan(args.plan)
    except DP.DtPlanError as e:
        print(f"[verify_dt_plan] ERROR 清单不可用：{e}")
        return 2

    print(f"[verify_dt_plan] plan={args.plan} version={plan['version']} sha256={sha[:12]}…")
    print(f"  presets={sorted(plan['presets'])}")
    problems = check_structure(DP, plan, sha)

    if args.manifest:
        try:
            rows = read_manifest(args.manifest)
        except (FileNotFoundError, ValueError) as e:
            print(f"[verify_dt_plan] ERROR {e}")
            return 2
        preset_name = rows[0].get("dt_plan_preset") or ""
        if args.limit:
            rows = rows[:args.limit]
        # 非清单集（旧版 aec_synth 生成的历史集）没有 dt_plan_preset 列：此时无法确定该用哪个
        # preset 比对，**不要**退回默认 preset 去报一堆"区间越界/占比偏离"的误导性 findings
        # （那是拿 eval 集去对训练口径）。直接给一条准确原因 + 处置建议后收工。
        if not preset_name:
            n_dt = sum(1 for r in rows if r["scenario"] == "dt")
            print(f"  manifest={args.manifest} rows={len(rows)} DT={n_dt} preset=(缺列)")
            problems.append(f"{args.manifest} 无 dt_plan_preset 列：该集不是按 DT 比例清单"
                            f"生成的（历史集由旧版 aec_synth 产出），C/D/E/F 无从核对。"
                            f"隔离评估请用 0-4 按清单重生成的集（如 eval_v8）")
            print(f"  FAIL 共 {len(problems)} 项：")
            for p in problems:
                print(f"   - {p}")
            return 1
        try:
            rp = DP.resolve_preset(plan, preset_name or None, sha)
        except DP.DtPlanError as e:
            print(f"[verify_dt_plan] ERROR {e}")
            return 2
        prof = profile_of(args.manifest)
        problems += check_provenance(args.manifest, rows, plan, sha)
        rir_mode = detect_rir_mode(prof, args.rir_mode)
        if prof is None or prof.get("seed") is None:
            problems += [f"{args.manifest.parent / 'dt_plan_profile.json'} 缺 seed："
                         f"该集不是按清单生成的，无法做同分布核对"]
        elif not prof.get("pool_sizes"):
            problems += ["dt_plan_profile.json 缺 pool_sizes（无法核对随机流）："
                         "该集由旧版 aec_synth 生成，请用当前版本重生成后再核"]
        else:
            problems += check_evidence(rows, rp, prof, args.plan)
        problems += check_ranges(DP, rows, rp)
        if args.strict_preset:
            problems += check_preset_expect(rows, preset_name)
        if args.replay_prefix:
            if not args.set_dir:
                problems.append("--replay-prefix 需同时给 --set-dir")
            elif rir_mode == "sim" and not args.replay_sim_rir:
                print("  [note] sim-RIR 集：G 组（现场重生成逐字节比对）默认跳过——")
                print("         它需要本机装 pyroomacoustics 且版本一致、每条 ~1-2s；")
                print("         要跑就显式加 --replay-sim-rir（建议小前缀，如 5）。")
            else:
                problems += check_replay_prefix(DP, rp, args.plan, args.manifest, args.set_dir,
                                                args.replay_prefix)
        n_dt = sum(1 for r in rows if r["scenario"] == "dt")
        seed_txt = prof.get("seed") if prof else "(缺 profile)"
        print(f"  manifest={args.manifest} rows={len(rows)} DT={n_dt} "
              f"preset={preset_name or '(缺列)'} seed={seed_txt} rir={rir_mode}")
        if prof and prof.get("pool_sizes"):
            print(f"  pool_sizes={prof['pool_sizes']}（回放随机流所需）")
        if rir_mode == "sim":
            print("  sim-RIR：C 走「实跑 + 收尾结构锚定 + 逐行对账」（见 docstring 适用边界）；")
            print("  G（产物逐字节复现）默认跳过——需本机 pyroomacoustics 且版本一致，"
                  "要跑请加 --replay-sim-rir。")
        if args.replay_prefix:
            print(f"  回放前缀 {args.replay_prefix} 条（源目录取 DT_PLAN_* 环境变量）")
    else:
        print("  （未给 --manifest：只做清单结构检查）")
        for name in sorted(plan["presets"]):
            rp = DP.resolve_preset(plan, name, sha)
            print(f"    {name}: DT 占比 {rp['scenarios']['dt']} | leadin "
                  f"{DP.dist_repr(rp['dt_leadin_sec'])}s | SER "
                  f"{DP.dist_repr(rp['dt_ser_db'])}dB | SNR {DP.dist_repr(rp['dt_snr_db'])}dB")

    if problems:
        print(f"  FAIL 共 {len(problems)} 项：")
        for p in problems:
            print(f"   - {p}")
        return 1
    print("  PASS 清单结构 / 留痕闭环 / 同分布核对（C）/ 区间合规（D/E/F） 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
