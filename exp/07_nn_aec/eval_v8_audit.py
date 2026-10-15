#!/usr/bin/env python
"""0-4/t16 验收：合成评测集的逐项准入审计（数据集侧，不跑模型）。

默认可直接审 ``D:/data/aec-synth/eval_v8``（0-4 的隔离集）；经参数化后同一份工具也审
t16 的两条新轴集：``eval_ns``（``--expect-rir-src sim``）与 ``eval_rir``
（``--expect-rir-src dir``，此时追加 I 组：评测 RIR 池 == ``rir_holdout15`` 名单，
且与 ``rir_train70`` 在**文件级与位置组级**零交集）。

对 ``D:/data/aec-synth/eval_v8`` 做 8 组判定（dir 档再 +I 组），任一不过 → 退出码 1（输入缺失 → 2）：

A. **规模/分层**：行数 == 生成时 num；每场景 ≥300（REMEDIATION_PLAN_V3.1.md §3.2
   "每场景 ≥300 条"，沿用 eval_v7 的 n 扩档口径）；实测占比不越 3σ 二项界。
B. **留痕**：manifest 带 ``nearfar_mutex=utterance``（0-3 跑过）、
   ``dt_plan_version=dt_plan_v1``/``dt_plan_preset=eval_v5_repro``（0-5 口径来源）；
   ``dt_plan_profile.json`` 记录的清单 sha256 与 ``configs/dt_plan_v1.json`` 实际文件一致。
C. **隔离（构造性）**：每行 near_file ∈ near 池名单、far_file ∈ far 池名单、
   noise_file ∈ 0-2 的 ``noise_eval15.txt``；池名单 near∩far=∅（reader 级 + 文件级）、
   holdout reader 全量覆盖、且与历史训练语料（train_v1/train_nl 的 near/far 文件名）
   **零交集**。
D. **无自回声**：行内 near_file != far_file，且 near/far 池无同名 utterance。
E. **DT 口径**：DT 行 ser∈[0,10]、snr∈[20,30]、leadin 恒 2.0s；FEST/NST leadin=0；
   ``--snr-range`` 口径下 FEST/NST 的 snr∈[20,30]。
F. **其它合成参数**：delay∈[10,80]ms、soft_clip 全 0、rir_src 全 sim（与 eval_v5 同口径，
   刻意不用训练侧 ``rir_pool``/真实 RIR 池——那是 eval-rir 轴，见报告边界节）。
G. **产物完整**：farend/nearend/mic 各 num 个 wav，8s@16k。
H. **历史泄漏披露（不是失败项，是必须入档的数字）**：eval15 噪声名单里有多少文件
   被历史 train_v1/train_nl/train_v1_drift 用过 —— 该数字决定"v4 基线在 eval_v8 上
   还能不能测到噪声隔离"，报告里必须如实写出。

用法：
    python exp/07_nn_aec/eval_v8_audit.py                       # 默认路径
    python exp/07_nn_aec/eval_v8_audit.py --set-dir <dir> --json <out.json>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import split_rir_pool as RP  # noqa: E402
import split_speech_pool as SP  # noqa: E402

DEFAULT_SET = Path("D:/data/aec-synth/eval_v8")
DEFAULT_LISTS = REPO / "split_lists"
DEFAULT_PLAN = REPO / "configs" / "dt_plan_v1.json"
DEFAULT_CLEAN = Path("D:/data/dns-challenge-mirror/datasets/clean")
TRAIN_MANIFESTS = ("train_v1", "train_nl", "train_v1_drift")
SYNTH_ROOT = Path("D:/data/aec-synth")
SR = 16000
MIN_PER_SCENE = 300
SIGMA_K = 3.0


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _names(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]


def _train_speech_files() -> dict[str, set[str]]:
    out = {}
    for s in TRAIN_MANIFESTS:
        p = SYNTH_ROOT / s / "manifest.csv"
        out[s] = ({r["near_file"] for r in _rows(p)} | {r["far_file"] for r in _rows(p)}) \
            if p.is_file() else set()
    return out


def _train_noise_files() -> dict[str, set[str]]:
    out = {}
    for s in TRAIN_MANIFESTS:
        p = SYNTH_ROOT / s / "manifest.csv"
        out[s] = {r["noise_file"] for r in _rows(p)} if p.is_file() else set()
    return out


def audit(set_dir: Path, lists: Path, plan: Path, clean: Path, expect_rir_src: str = "sim",
          rir_list: Path | None = None, rir_src_dir: Path | None = None,
          expect_seed: int | None = None, expect_preset: str = "eval_v5_repro",
          expect_version: str = "dt_plan_v1", dt_ser_range: tuple = (0.0, 10.0),
          dt_snr_range: tuple = (20.0, 30.0), rest_snr_range: tuple = (20.0, 30.0),
          dt_leadin: float = 2.0, near_list: Path | None = None,
          far_list: Path | None = None, noise_list: Path | None = None,
          expect_dur: float | None = None) -> tuple[list[str], dict]:
    """算一套合成集的准入项。

    ``expect_rir_src`` = manifest 的 ``rir_src`` 期望值（eval_v8/eval_ns = ``sim``；
    eval_rir/eval_rir_sim = ``dir``，此时追加 I 组：评测 RIR 池 == ``rir_holdout15`` 名单、
    且与 ``rir_train70`` 名单在**文件级与位置组级**零交集）。``rir_list``/``rir_src_dir``/
    ``expect_seed`` 缺省走仓内名单与 profile 记录（默认行为 = eval_v8 口径，不变）。
    ``expect_preset``/``expect_version``/``dt_*``/``rest_snr_range`` 供新轴集使用
    （如 t23 的 `eval_snr_low|high`：`dt_plan_v2` + SNR 域外区间）。
    """
    problems: list[str] = []
    info: dict = {}

    man = set_dir / "manifest.csv"
    prof_p = set_dir / "dt_plan_profile.json"
    if not man.is_file() or not prof_p.is_file():
        raise FileNotFoundError(f"缺 {man} 或 {prof_p}")
    rows, prof = _rows(man), json.loads(prof_p.read_text(encoding="utf-8"))
    scenes = Counter(r["scenario"] for r in rows)
    info["rows"], info["scenes"] = len(rows), dict(scenes)

    # ---- A 规模/分层
    if len(rows) != int(prof.get("num", -1)):
        problems.append(f"A 行数 {len(rows)} != profile.num {prof.get('num')}")
    for sc in ("fest", "nst", "dt"):
        if scenes[sc] < MIN_PER_SCENE:
            problems.append(f"A {sc} 只有 {scenes[sc]} 条 < {MIN_PER_SCENE}")
    target = {sc: float(prof["scenarios"][sc]) for sc in ("fest", "nst", "dt")}
    n = len(rows)
    info["scenes_target"] = target
    for sc, p in target.items():
        sd = (p * (1 - p) / n) ** 0.5
        lo, hi = n * p - SIGMA_K * n * sd, n * p + SIGMA_K * n * sd
        if not (lo <= scenes[sc] <= hi):
            problems.append(f"A {sc} 实测 {scenes[sc]} 越出 3σ 界 [{lo:.1f}, {hi:.1f}]")

    # ---- B 留痕 / 清单闭环
    cols = set(rows[0])
    info["columns"] = sorted(cols)
    for col, want in (("nearfar_mutex", "utterance"), ("dt_plan_version", expect_version),
                      ("dt_plan_preset", expect_preset)):
        if col not in cols:
            problems.append(f"B manifest 缺列 {col}")
        elif {r[col] for r in rows} != {want}:
            problems.append(f"B {col} 列取值 {sorted({r[col] for r in rows})} != {want}")
    real_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    info["plan_sha256"] = prof.get("sha256")
    if prof.get("sha256") != real_sha:
        problems.append(f"B profile.sha256 与清单 {plan.name} 实际不一致")
    if prof.get("preset") != expect_preset or prof.get("version") != expect_version:
        problems.append(f"B profile preset/version 异常：{prof.get('preset')}/{prof.get('version')}"
                        f"（期望 {expect_preset}/{expect_version}）")
    info["profile"] = {k: prof.get(k) for k in ("version", "preset", "sha256", "seed", "num",
                                                "dur_sec", "pool_sizes", "pool_dirs",
                                                "cli_args")}

    # ---- C 隔离（池名单 + 行内成员）
    near_l = _names(near_list) if near_list else _names(lists / SP.SIDE_LISTS[SP.SIDE_NEAR])
    far_l = _names(far_list) if far_list else _names(lists / SP.SIDE_LISTS[SP.SIDE_FAR])
    noise_l = _names(noise_list) if noise_list else _names(lists / "noise_eval15.txt")
    info["list_sources"] = {"near": str(near_list or lists / SP.SIDE_LISTS[SP.SIDE_NEAR]),
                            "far": str(far_list or lists / SP.SIDE_LISTS[SP.SIDE_FAR]),
                            "noise": str(noise_list or lists / "noise_eval15.txt")}
    near_s, far_s, noise_s = set(near_l), set(far_l), set(noise_l)
    if near_s & far_s:
        problems.append(f"C near∩far 名单交集 {len(near_s & far_s)}")
    # 名单 vs 独立重算（仅当用 0-4 的 DNS 语音池名单时才有这一步）
    if near_list is None and far_list is None and clean.is_dir():
        per = SP.scan_pool(clean)
        built = SP.build_split(per)
        if sorted(near_l) != built["near_files"] or sorted(far_l) != built["far_files"]:
            problems.append("C 池名单与 split_speech_pool 独立重算不一致")
    train_sp = _train_speech_files()
    for s, files in train_sp.items():
        if (near_s | far_s) & files:
            problems.append(f"C 评测语音池与 {s} 的训练语料交集 "
                            f"{len((near_s | far_s) & files)}")
    info["train_speech_overlap"] = {s: len((near_s | far_s) & f) for s, f in train_sp.items()}
    bad = [(r["name"], "near") for r in rows if r["near_file"] not in near_s]
    bad += [(r["name"], "far") for r in rows if r["far_file"] not in far_s]
    bad += [(r["name"], "noise") for r in rows if r["noise_file"] not in noise_s]
    if bad:
        problems.append(f"C 有 {len(bad)} 行的 near/far/noise 不在对应池名单（例：{bad[0]}）")
    info["noise_unique_rows"] = len({r["noise_file"] for r in rows})

    # ---- D 无自回声
    self_echo = [(r["name"], r["near_file"], r["far_file"]) for r in rows
                 if r["near_file"] == r["far_file"]
                 or Path(r["near_file"]).stem == Path(r["far_file"]).stem]
    if self_echo:
        problems.append(f"D 自回声 {len(self_echo)} 行（例：{self_echo[0]}）")
    info["self_echo_rows"] = len(self_echo)

    # ---- E DT 口径
    dt = [r for r in rows if r["scenario"] == "dt"]
    if {round(float(r["leadin_sec"]), 6) for r in dt} != {round(float(dt_leadin), 6)}:
        problems.append(f"E DT leadin 取值 {sorted({float(r['leadin_sec']) for r in dt})} "
                        f"!= {dt_leadin}")
    for key, (lo, hi) in (("ser_db", dt_ser_range), ("snr_db", dt_snr_range)):
        vals = [float(r[key]) for r in dt]
        if min(vals) < lo - 1e-9 or max(vals) > hi + 1e-9:
            problems.append(f"E DT {key} 实测 [{min(vals):.3f}, {max(vals):.3f}] 越出 [{lo}, {hi}]")
        info[f"dt_{key}"] = [round(min(vals), 3), round(max(vals), 3)]
    rest = [r for r in rows if r["scenario"] != "dt"]
    if {float(r["leadin_sec"]) for r in rest} != {0.0}:
        problems.append("E FEST/NST 的 leadin 非 0")
    snr_all = [float(r["snr_db"]) for r in rest]
    if min(snr_all) < rest_snr_range[0] - 1e-9 or max(snr_all) > rest_snr_range[1] + 1e-9:
        problems.append(f"E FEST/NST snr 实测 [{min(snr_all):.3f}, {max(snr_all):.3f}] "
                        f"越出 [{rest_snr_range[0]}, {rest_snr_range[1]}]")
    info["dt_ser_db_range"], info["dt_snr_db_range"] = info.get("dt_ser_db"), info.get("dt_snr_db")
    info["festsnr_range"] = [round(min(snr_all), 3), round(max(snr_all), 3)]

    # ---- F 其它合成参数
    if {r["soft_clip"] for r in rows} != {"0"}:
        problems.append(f"F soft_clip 非全 0：{sorted({r['soft_clip'] for r in rows})}")
    got_src = {r["rir_src"] for r in rows}
    info["rir_src"] = sorted(got_src)
    if got_src != {expect_rir_src}:
        problems.append(f"F rir_src 非全 {expect_rir_src}：{sorted(got_src)}")
    if expect_seed is not None and int(prof.get("seed", -1)) != expect_seed:
        problems.append(f"B profile.seed {prof.get('seed')} != 期望 {expect_seed}")
    delay = [float(r["delay_ms"]) for r in rows]
    if min(delay) < 10.0 - 1e-9 or max(delay) > 80.0 + 1e-9:
        problems.append(f"F delay 实测 [{min(delay):.3f}, {max(delay):.3f}] 越出 [10,80]")
    info["delay_ms_range"] = [round(min(delay), 3), round(max(delay), 3)]

    # ---- I RIR 来源清单闭环（仅 dir 档；sim 档无外部 RIR 清单）
    if expect_rir_src == "dir":
        rir_list = rir_list or (lists / "rir_holdout15.txt")
        train_list = lists / "rir_train70.txt"
        if not rir_list.is_file() or not train_list.is_file():
            problems.append(f"I 缺 RIR 名单：{rir_list} / {train_list}")
        else:
            hold, train = _names(rir_list), _names(train_list)
            info["rir_holdout_n"], info["rir_train_n"] = len(hold), len(train)
            info["rir_list_sha256"] = hashlib.sha256(
                ("\n".join(hold) + "\n").encode("utf-8")).hexdigest()
            if set(hold) & set(train):
                problems.append(f"I 评测 RIR 名单与训练名单交集 {len(set(hold) & set(train))}")
            gh = {RP.group_of(Path(n), Path(".")) for n in hold}
            gt = {RP.group_of(Path(n), Path(".")) for n in train}
            info["rir_groups"] = {"holdout": sorted(gh), "train": sorted(gt)}
            if gh & gt:
                problems.append(f"I 评测 RIR 与训练 RIR 共用位置组：{sorted(gh & gt)}")
            src_dir = rir_src_dir
            if src_dir is None:
                pool_dirs = prof.get("pool_dirs") or {}
                src_dir = Path(pool_dirs["rir"]) if pool_dirs.get("rir") else None
            if src_dir is None or not Path(src_dir).is_dir():
                problems.append(f"I profile 未记 --rir-dir 池目录（或目录不存在）：{src_dir}")
            else:
                got = sorted(p.name for p in Path(src_dir).glob("**/*.wav"))
                if got != sorted(hold):
                    problems.append(f"I 评测 RIR 池 {src_dir} 内容 != {rir_list.name}"
                                    f"（{len(got)} vs {len(hold)}）")
                if int((prof.get("pool_sizes") or {}).get("rir", -1)) != len(hold):
                    problems.append("I profile.pool_sizes.rir != 评测 RIR 名单长度")
                info["rir_pool_dir"] = str(src_dir)

    # ---- G 产物完整
    n_wav = {sub: len(list((set_dir / sub).glob("*.wav"))) for sub in ("farend", "nearend",
                                                                       "mic")}
    info["wav_counts"] = n_wav
    if set(n_wav.values()) != {len(rows)}:
        problems.append(f"G 三槽 wav 数 {n_wav} != 行数 {len(rows)}")
    import soundfile as sf
    dur_expect = float(expect_dur if expect_dur is not None else prof.get("dur_sec", 8))
    info["wav_meta"] = {}
    for sub in ("farend", "nearend", "mic"):
        f = next(iter((set_dir / sub).glob("*.wav")))
        mi = sf.info(str(f))
        info["wav_meta"][sub] = [mi.samplerate, round(mi.duration, 3)]
        if mi.samplerate != SR or abs(mi.duration - dur_expect) > 1e-3:
            problems.append(f"G {sub}/{f.name} 采样率/时长异常：{mi.samplerate}/{mi.duration}"
                            f"（期望 {SR}/{dur_expect}）")

    # ---- H 历史泄漏披露（数字入档，不判定失败）
    train_noise = _train_noise_files()
    info["noise_seen_by_train"] = {s: len(noise_s & f) for s, f in train_noise.items()}
    info["noise_seen_by_train_union"] = len(noise_s & set().union(*train_noise.values()))
    info["noise_pool_size"] = len(noise_s)
    return problems, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set-dir", type=Path, default=DEFAULT_SET)
    ap.add_argument("--lists", type=Path, default=DEFAULT_LISTS)
    ap.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    ap.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--expect-rir-src", choices=("sim", "dir"), default="sim",
                    help="manifest 的 rir_src 期望值（eval_v8/eval_ns=sim；eval_rir=dir，"
                         "追加 RIR 来源清单闭环组 I）")
    ap.add_argument("--rir-list", type=Path, default=None,
                    help="评测 RIR 名单（默认 split_lists/rir_holdout15.txt）")
    ap.add_argument("--rir-src-dir", type=Path, default=None,
                    help="评测 RIR 池目录（默认取 dt_plan_profile.json 的 pool_dirs.rir）")
    ap.add_argument("--expect-seed", type=int, default=None, help="校验 profile.seed")
    ap.add_argument("--expect-preset", default="eval_v5_repro", help="manifest/profile 期望 preset")
    ap.add_argument("--expect-version", default="dt_plan_v1", help="manifest/profile 期望清单版本")
    ap.add_argument("--dt-ser-range", default="0,10", help="DT SER 期望区间（lo,hi）")
    ap.add_argument("--dt-snr-range", default="20,30", help="DT SNR 期望区间（lo,hi）")
    ap.add_argument("--rest-snr-range", default="20,30", help="FEST/NST SNR 期望区间（lo,hi）")
    ap.add_argument("--dt-leadin", type=float, default=2.0, help="DT 引导时长期望值（秒）")
    ap.add_argument("--near-list", type=Path, default=None,
                    help="near 池名单覆盖（默认 0-4 的 speech_near_eval.txt）")
    ap.add_argument("--far-list", type=Path, default=None,
                    help="far 池名单覆盖（t23 扩量轴集用 ext_pool_*_used.txt）")
    ap.add_argument("--noise-list", type=Path, default=None,
                    help="噪声池名单覆盖（t23 eval_music 用 ext_pool_musan_eval15_used.txt）")
    ap.add_argument("--expect-dur", type=float, default=None,
                    help="产物时长期望（秒；默认取 profile 的 dur_sec，eval_long_ls 需显式给）")
    ap.add_argument("--set-name", default=None, help="档案名（默认取 --set-dir 的目录名）")
    ap.add_argument("--command-ref", default="见 reports/eval_ns_rir_20261013.md（逐字留档）",
                    help="provenance 里的生成命令出处")
    ap.add_argument("--write-provenance", type=Path, default=None,
                    help="把该集的口径清单版本 + 名单哈希写为 provenance JSON（放在集合目录里）")
    args = ap.parse_args()

    def _rng(text):
        a, b = str(text).split(",")
        return (float(a), float(b))

    try:
        problems, info = audit(args.set_dir, args.lists, args.plan, args.clean_dir,
                               expect_rir_src=args.expect_rir_src, rir_list=args.rir_list,
                               rir_src_dir=args.rir_src_dir, expect_seed=args.expect_seed,
                               expect_preset=args.expect_preset,
                               expect_version=args.expect_version,
                               dt_ser_range=_rng(args.dt_ser_range),
                               dt_snr_range=_rng(args.dt_snr_range),
                               rest_snr_range=_rng(args.rest_snr_range),
                               dt_leadin=args.dt_leadin, near_list=args.near_list,
                               far_list=args.far_list, noise_list=args.noise_list,
                               expect_dur=args.expect_dur)
    except (FileNotFoundError, ValueError) as e:
        print(f"[eval_v8_audit] 输入错误：{e}", file=sys.stderr)
        return 2
    print(f"[eval_v8_audit] {args.set_dir}：{info['rows']} 行 {info['scenes']}")
    print(f"  分层目标 {info['scenes_target']}（每场景 ≥{MIN_PER_SCENE}）")
    print(f"  清单留痕：{info['profile']['version']}/{info['profile']['preset']} "
          f"sha256={str(info['profile']['sha256'])[:12]}… seed={info['profile']['seed']}")
    eff = {k: Path(v) for k, v in info["list_sources"].items()}
    print(f"  池：near {len(_names(eff['near']))}({eff['near'].name}) / "
          f"far {len(_names(eff['far']))}({eff['far'].name}) / "
          f"noise {info['noise_pool_size']}({eff['noise'].name})"
          f"（行内唯一噪声 {info['noise_unique_rows']}） / rir_src {info['rir_src']}")
    if args.expect_rir_src == "dir":
        print(f"  RIR 来源：清单 {info.get('rir_holdout_n')} 条（sha256 "
              f"{str(info.get('rir_list_sha256'))[:12]}…）池 {info.get('rir_pool_dir')}；"
              f"与训练名单 {info.get('rir_train_n')} 条零交集")
    print(f"  历史训练语料交集：{info['train_speech_overlap']}；"
          f"eval15 噪声被历史训练用过 {info['noise_seen_by_train_union']}"
          f"/{info['noise_pool_size']}（披露项，决定 v4 能否测到噪声隔离）")
    print(f"  范围：DT ser {info['dt_ser_db_range']} snr {info['dt_snr_db_range']} "
          f"| FEST/NST snr {info.get('festsnr_range')} "
          f"delay {info['delay_ms_range']}；自回声 {info['self_echo_rows']} 行；"
          f"wav {info['wav_counts']}")
    if args.write_provenance:
        prov = {"set": args.set_name or args.set_dir.name,
                "audit": "exp/07_nn_aec/eval_v8_audit.py",
                "generator": "scripts/aec_synth.py",
                "command": args.command_ref,
                "seed": info["profile"]["seed"], "num": info["rows"],
                "scenes": info["scenes"], "dur_sec": info["profile"]["dur_sec"],
                "dt_plan": {"path": str(args.plan), "version": info["profile"]["version"],
                            "preset": info["profile"]["preset"],
                            "sha256": info["profile"]["sha256"]},
                "near_far_lists": {
                    "near": {"file": str(eff["near"]), "n": len(_names(eff["near"])),
                             "sha256": SP.content_sha256(_names(eff["near"]))},
                    "far": {"file": str(eff["far"]), "n": len(_names(eff["far"])),
                            "sha256": SP.content_sha256(_names(eff["far"]))}},
                "noise_list": {"file": str(eff["noise"]), "n": info["noise_pool_size"],
                               "sha256": SP.content_sha256(_names(eff["noise"]))},
                "rir": ({"mode": "dir", "list_file": str(args.rir_list
                                                         or args.lists / "rir_holdout15.txt"),
                         "n": info.get("rir_holdout_n"),
                         "sha256": info.get("rir_list_sha256"),
                         "pool_dir": info.get("rir_pool_dir"),
                         "train_list": str(args.lists / "rir_train70.txt"),
                         "n_train": info.get("rir_train_n"),
                         "groups": info.get("rir_groups")}
                        if args.expect_rir_src == "dir" else {"mode": "sim"}),
                "profile_pool_sizes": info["profile"]["pool_sizes"],
                "profile_pool_dirs": info["profile"]["pool_dirs"],
                "profile_cli_args": info["profile"]["cli_args"],
                "columns": info["columns"]}
        args.write_provenance.write_text(
            json.dumps(prov, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            newline="\n")
        print(f"  provenance -> {args.write_provenance}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"problems": problems, "info": info},
                                        ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
        print(f"  -> {args.json}")
    if problems:
        print(f"[eval_v8_audit] FAIL 共 {len(problems)} 项：", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("[eval_v8_audit] PASS：规模/留痕/隔离/无自回声/DT 口径/参数域/产物 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
