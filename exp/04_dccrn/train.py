"""DCCRN 训练脚本（WP-04 / 模块4）。

与 CRN train.py 同骨架（TensorBoard / Plateau / early stop / resume），差异：
- 数据：DNSDynamicMix 在线混合（epoch_len 虚拟长度），val 用固定 seed 的可复现子集
- 损失：复数谱 MSE + si_snr_weight × SI-SNR（官方是 SI-SNR + 多分辨率复数谱损，
  我们简化并文档化）
- 默认 4s 片段 / hop 200 / batch 8（8GB 显存）

`--dynamic-mixing`（P1-5a，阶段1 隔离性必需）：把数据源换成**清单圈定**的共享 DM
（`exp/07_nn_aec/dm_dataset.AecDynamicMix` 经 `dccrn_dm.DccrnDynamicMix` 适配）——
`--near-dir/--far-dir/--noise-dir` + `--near-list/--far-list/--noise-list` **必填**，
入口做 near/far 互斥与 noise∩speech 校验，清单版本号/池规模/配比进 `train_manifest.json`。
数据 = AEC 三元组投影（noisy=mic、clean=near），与 NN-AEC/级联同一条供给、同一 rng 流；
epoch 口径按附录B 方案(a) 对齐 04 历史 18.7K 步，LR/早停按阶段0 §2.3b（patience 20/关早停），
训练对照从零起。**不开本开关时旧 DNSDynamicMix/StaticPairs 路径逐键不变**。

用法：
    python train.py --device cuda --epochs 100 --batch-size 8        # 长训练
    python train.py --smoke                                          # 合成数据 1ep×2batch
    python train.py --device cuda --no-dm                            # 静态混合消融
    python train.py --device cuda --dynamic-mixing --epochs 100 --batch-size 64 \\
        --near-dir <pool>/near --near-list ../split_lists/speech_near_train70_files.txt \\
        --far-dir <pool>/far --far-list ../split_lists/speech_far_train70_files.txt \\
        --noise-dir <pool>/noise --noise-list ../split_lists/noise_train70.txt \\
        --rir-dir <rir-pool> --val D:/data/aec-synth/eval_v5 \\
        --dm-scenarios 0,0.5086,0.4914                                # 阶段1（P1-5a）
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynamic_mixing import DNSDynamicMix, StaticPairs  # noqa: E402
from model import DCCRN  # noqa: E402

from core.losses import complex_mse_loss, si_snr_loss  # noqa: E402
from core.stft import STFT  # noqa: E402
from core.train_protocol import probe_memory  # noqa: E402
from core.train_protocol import resolve_schedule as _shared_resolve_schedule  # noqa: E402

# P1X-2：池装配（多源）与 CLI 旗标与 07 共用同一实现（口径一处实现）。`dm_dataset` 在 07 的
# 目录里，而 04/07 有同名模块（model/dynamic_mixing）⇒ **只在导入期**临时补路径并立刻摘掉；
# 导入期补路径不会遮蔽 04 自己的模块（它们在本文件更早的 import 里已缓存）。
_EXP07 = str(Path(__file__).resolve().parents[1] / "07_nn_aec")
_TEMP_PATH = _EXP07 not in sys.path
if _TEMP_PATH:
    sys.path.insert(0, _EXP07)
try:
    import dm_dataset as DM  # noqa: E402
finally:
    if _TEMP_PATH and _EXP07 in sys.path:
        sys.path.remove(_EXP07)

#: 04 的历史缺省（不开 --dynamic-mixing 时逐键不变）：patience 3 / early-stop 10 / epoch_len 2000
LEGACY_EARLY_STOP = 10
LEGACY_PATIENCE = 3
LEGACY_EPOCH_LEN = 2000
#: 新增参数（P1-5a）——**不进**非 DM 的 ckpt cfg（`vars(args)` 全量存 cfg 是 04 的历史做法，
#: 所以这里显式过滤，保证不开 --dynamic-mixing 时既有产物逐键不变）
NEW_ARG_KEYS = ("dynamic_mixing", "near_dir", "far_dir", "noise_dir", "near_list",
                "far_list", "noise_list", "rir_dir", "rir_max_ms", "soft_clip_prob",
                "delay_min_ms", "delay_max_ms", "dm_plan", "dm_preset", "dm_scenarios",
                "dm_target_steps", "dm_seg_sec", "val", "probe_mem", "lr_patience")


def resolve_epoch_len(dynamic_mixing, explicit):
    """epoch_len 解析：显式值优先；非 DM 缺省 = 历史 2000（逐键兼容）。

    DM 路径下 None 表示"交给方案(a) 换算"（由 dm_dataset.resolve_epoch_scheme 定）。
    """
    if explicit is not None:
        return int(explicit)
    return None if dynamic_mixing else LEGACY_EPOCH_LEN


def legacy_cfg(args):
    """非 DM 的 ckpt cfg（= 历史 `vars(args)`，剔除 P1-5a 新增键）。"""
    return {k: v for k, v in vars(args).items() if k not in NEW_ARG_KEYS}


def build_dccrn_dm_set(args, out, tag, sched_cfg):
    """DM 训练集 + val + 留痕（P1-5a）；缺参数/清单不合法一律 fail fast。"""
    from dccrn_dm import DCCRN_REF, AecValAsDccrn, DccrnDynamicMix
    if args.no_dm or args.synthetic:
        raise SystemExit("--dynamic-mixing 与 --no-dm/--synthetic 互斥（三种数据源只能选一）")
    if args.val is None:
        raise SystemExit("--dynamic-mixing 需要 --val <aec_synth 目录>：val 保持静态"
                         "（与 NN-AEC 同口径），否则模型选择不可比")

    preset, plan_info = DM.load_plan(args.dm_plan, args.dm_preset)
    ratio_src = "preset"
    if args.dm_scenarios:
        preset = DM.with_scenario_ratio(preset, args.dm_scenarios.split(","))
        ratio_src = "cli_override"
    # 池装配：v5 单源（--{side}-dir/--{side}-list）与 P1X-2 多源（--{side}-source）同一入口；
    # 缺清单/歧义由 resolve_side_sources fail fast（消息含参数名与隔离纪律）
    near_files, far_files, noise_files, pool_info = DM.build_pools_from_args(args)
    seg = float(args.dm_seg_sec or preset["dur_sec"] or args.seg_sec)
    sch = DM.resolve_epoch_scheme(args.epochs, args.batch_size,
                                  args.dm_target_steps or DCCRN_REF["steps"],
                                  epoch_len_override=resolve_epoch_len(True, args.epoch_len),
                                  reference=DCCRN_REF)
    if sch["steps_per_epoch"] < 1:
        raise SystemExit(f"epoch_len={sch['epoch_len']} < batch_size={args.batch_size}："
                         f"drop_last 下每 epoch 0 步")
    # RIR 池：有 --rir-list 走清单驱动（隔离纪律），否则整目录枚举（平台无关序，t19）
    rir_files = DM.load_rir_pool(args.rir_dir, args.rir_list)
    if args.rir_dir and not rir_files:
        raise SystemExit(f"--rir-dir 内无 wav：{args.rir_dir}")
    tr = DccrnDynamicMix(near_files, far_files, noise_files, preset, plan_info=plan_info,
                         pool_info=pool_info, epoch_len=sch["epoch_len"], seg_sec=seg,
                         rir_files=rir_files, rir_max_ms=args.rir_max_ms,
                         rir_dir=args.rir_dir, rir_list=args.rir_list,
                         delay_ms_range=(args.delay_min_ms, args.delay_max_ms),
                         soft_clip_prob=args.soft_clip_prob, seed=None)
    va = AecValAsDccrn(args.val)
    manifest = {
        "schema": "train_manifest/1",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": {"out": str(out), "tag": tag, "device": args.device, "seed": args.seed,
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "num_workers": args.num_workers,
                "from_scratch": args.resume is None,
                "resume": str(args.resume) if args.resume else None},
        "loss": {"kind": "complex_mse+si_snr", "si_snr_weight": args.si_snr_weight},
        "data": dict(tr.provenance(), val=str(args.val), val_n=len(va),
                     train_mode="dccrn_aec_dynamic_mixing",
                     scenario_ratio=preset["scenarios"],
                     scenarios_explicit=bool(args.dm_scenarios),
                     scenario_ratio_source=ratio_src),
        "epoch_scheme": sch,
        "lr_sched": sched_cfg,
        "note": ("P1-5a：DCCRN 走与 NN-AEC/级联同一条 DM 供给（noisy=mic、clean=near，"
                 "见 data.key_map）；epoch 方案(a) 对齐 04 历史 18.7K 步；LR/早停按阶段0 "
                 "§2.3b（DM 缺省 patience 20/关早停）；训练对照从零起（§2.3a）"),
    }
    return manifest, tr, va


def print_dm_summary(manifest):
    """把 DM 口径打到 stdout（tmux 日志/阶段1 核对用）。"""
    d, sch, lr = manifest["data"], manifest["epoch_scheme"], manifest["lr_sched"]
    plan, run = d.get("dt_plan", {}), manifest["run"]
    print(f"[dm] 清单 {plan.get('version')}/{plan.get('preset')} "
          f"sha256={str(plan.get('sha256'))[:12]}…  池 near/far/noise="
          f"{d['n_files']['near']}/{d['n_files']['far']}/{d['n_files']['noise']} "
          f"rir={d['n_files']['rir']}({d['rir_src']})", flush=True)
    print(f"[dm] 互斥校验通过：{d['mutex'].get('level')}（near/far 名单 + noise∩speech）"
          f"；键映射 {d.get('key_map')}；场景配比 {d.get('scenario_ratio')}"
          f"（来源 {d.get('scenario_ratio_source')}，scenarios_explicit="
          f"{d.get('scenarios_explicit')}）", flush=True)
    print(f"[dm] epoch 口径 方案({sch['scheme']})：目标 {sch['target_steps']} 步"
          f"（参照 {sch['reference'].get('name')}）→ {sch['steps_per_epoch']} 步/epoch × "
          f"{sch['epochs']}ep = {sch['achieved_steps']} 步"
          f"（Δ{sch['delta_steps']:+d}，{sch['delta_frac'] * 100:+.3f}%）；"
          f"epoch_len={sch['epoch_len']}（来源 {sch['epoch_len_source']}）", flush=True)
    if sch["epoch_len_source"] == "cli_override":
        # 显式 --epoch-len 覆盖方案(a)：把偏差喊出来（历史事故：默认值 2000 被当成显式值，
        # 只跑到目标 16.6% 而无任何提示 —— 见 tests/test_dccrn_dm.py 的集成用例）
        print(f"[dm] [warn] --epoch-len={sch['epoch_len']} 覆盖了方案(a) 的换算："
              f"实际 {sch['achieved_steps']} 步 vs 目标 {sch['target_steps']} 步"
              f"（{sch['delta_frac'] * 100:+.2f}%）。正式训练请**不要**给 --epoch-len，"
              f"让等效更新步数对齐历史口径；本行偏差已记入 manifest", flush=True)
    print(f"[dm] LR/早停口径：{lr['kind']} factor={lr['factor']} patience={lr['patience']}"
          f"{'（显式）' if lr['patience_explicit'] else '（DM 缺省）'}，early-stop="
          f"{lr['early_stop']}{'（显式）' if lr['early_stop_explicit'] else '（DM 缺省=关闭）'}"
          f"；依据 {lr['basis']}", flush=True)
    print(f"[dm] {'从零训练' if run['from_scratch'] else '续训（resume ' + str(run['resume']) + '）'}"
          f"——阶段0 §2.3(a)：损失/配方/数据对照须从零起", flush=True)


def make_loss_fn(si_snr_weight, device):
    stft = STFT(n_fft=512, hop=200).to(device)

    def loss_fn(model, batch):
        enh_wav, mask, noisy_spec, enh_spec = model(batch["noisy"])
        clean_spec = stft.transform(batch["clean"])
        return complex_mse_loss(enh_spec, clean_spec) \
            + si_snr_weight * si_snr_loss(enh_wav, batch["clean"])

    return loss_fn


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        total += float(loss_fn(model, batch)) * len(batch["noisy"])
        n += len(batch["noisy"])
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--epoch-len", type=int, default=None,
                    help="DM 虚拟 epoch 样本数。缺省：非 DM=2000（历史口径）；**DM=按附录B "
                         "方案(a) 换算**（= round(目标步数/epochs)×batch_size）。"
                         "显式给值则记 epoch_len_source=cli_override 并打印偏差告警")
    ap.add_argument("--val-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seg-sec", type=float, default=4.0)
    ap.add_argument("--si-snr-weight", type=float, default=0.1)
    ap.add_argument("--clstm-hidden", type=int, default=64)
    ap.add_argument("--real-conv", action="store_true", help="消融：实数卷积")
    ap.add_argument("--no-dm", action="store_true", help="消融：静态混合对")
    ap.add_argument("--early-stop", type=int, default=None,
                    help="早停的连续无改善 epoch 数；0=关闭。缺省：非 DM=10（现状）、DM=0")
    ap.add_argument("--lr-patience", type=int, default=None,
                    help="ReduceLROnPlateau patience；缺省：非 DM=3（现状）、DM=20")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true", help="随机张量（无 DNS 冒烟）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    # ---------------- P1-5a：共享 DM（清单圈定，防泄漏） ----------------
    ap.add_argument("--dynamic-mixing", action="store_true",
                    help="用清单圈定的共享 DM（dm_dataset.AecDynamicMix → dccrn_dm 适配）："
                         "需三池+三清单+--val；口径与 NN-AEC 同构（方案 a/patience 20/关早停/"
                         "从零/manifest 留痕）。默认关闭 = 旧 DNSDynamicMix 行为不变")
    ap.add_argument("--near-dir", type=Path, default=None, help="DM：近端语音池目录")
    ap.add_argument("--far-dir", type=Path, default=None, help="DM：远端语音池目录")
    ap.add_argument("--noise-dir", type=Path, default=None, help="DM：噪声池目录")
    ap.add_argument("--near-list", type=Path, default=None,
                    help="DM：近端池分割清单（每行一个文件名；见 split_lists/）")
    ap.add_argument("--far-list", type=Path, default=None, help="DM：远端池分割清单")
    ap.add_argument("--noise-list", type=Path, default=None,
                    help="DM：噪声池分割清单（训练侧只能用 noise_train70.txt）")
    ap.add_argument("--rir-dir", type=Path, default=None,
                    help="DM：自备 RIR 目录；缺省用 pyroomacoustics 仿真（吞吐低）")
    ap.add_argument("--rir-max-ms", type=float, default=100.0, help="DM：RIR 截断（100ms）")
    ap.add_argument("--rir-list", type=Path, default=None,
                    help="DM：RIR 池清单（训练侧只应吃 split_lists/rir_train70.txt）；"
                         "给了 => 清单驱动并把清单名+sha256 记入 manifest.data.rir（t26）")
    ap.add_argument("--soft-clip-prob", type=float, default=0.15, help="DM：软剪切样本比例")
    ap.add_argument("--delay-min-ms", type=float, default=10.0, help="DM：回声时延下界")
    ap.add_argument("--delay-max-ms", type=float, default=80.0, help="DM：回声时延上界")
    ap.add_argument("--dm-plan", type=Path, default=None,
                    help="DM：DT 比例清单 JSON（缺省 configs/dt_plan_v1.json）")
    ap.add_argument("--dm-preset", default=None, help="DM：清单 preset（缺省 train_v1_repro）")
    ap.add_argument("--dm-scenarios", default=None,
                    help="DM：显式覆盖场景配比 FEST,NST,DT（和=1）。v4 实际训练配方 = "
                         "0,0.5086,0.4914（nst 913/1795、dt 882/1795，无 FEST）")
    ap.add_argument("--dm-target-steps", type=int, default=None,
                    help="DM：等效更新步数目标（方案 a）；缺省 = 04 历史 18.7K 步就位")
    ap.add_argument("--dm-seg-sec", type=float, default=None,
                    help="DM：片段时长；缺省取清单 preset 的 dur_sec（train_v1_repro=4s）")
    ap.add_argument("--val", type=Path, default=None,
                    help="DM：静态验证集（aec_synth 目录，模型选择用；与 NN-AEC 同集可比）")
    DM.add_dm_source_args(ap)      # P1X-2：多源池（--{near,far,noise}-source 可重复）
    ap.add_argument("--probe-mem", type=int, default=None, metavar="N",
                    help="第 N 个优化步后打印峰值显存（阶段1 定 batch 用；CPU 下提示无意义）")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.epoch_len, args.val_len, args.num_workers = 1, 8, 4, 0
        args.synthetic = True

    # 口径解析（阶段0 §2.3b + 方案 a）：LR/早停与 epoch_len 都按 DM/非 DM 分别定死
    sched_cfg = _shared_resolve_schedule(args.dynamic_mixing, args.lr_patience,
                                        args.early_stop,
                                        legacy_early_stop=LEGACY_EARLY_STOP,
                                        legacy_patience=LEGACY_PATIENCE)
    args.early_stop = sched_cfg["early_stop"]
    explicit_epoch_len = args.epoch_len
    resolved_epoch_len = resolve_epoch_len(args.dynamic_mixing, explicit_epoch_len)
    if resolved_epoch_len is not None:
        # 非 DM：历史缺省 2000（或缺省显式值）⇒ 旧路径逐键不变；
        # DM：None 表示交给方案(a) 换算，build_dccrn_dm_set 返回后再回填生效值。
        args.epoch_len = resolved_epoch_len

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    tag = "dccrn" + ("_realconv" if args.real_conv else "") + ("_nodm" if args.no_dm else "")
    if args.dynamic_mixing:
        tag += "_dm"          # DM 单独输出目录：不覆盖历史静态/DNS 产物
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)

    train_manifest = None
    if args.dynamic_mixing:
        train_manifest, tr, va = build_dccrn_dm_set(args, out, tag, sched_cfg)
        (out / "train_manifest.json").write_text(
            json.dumps(train_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_dm_summary(train_manifest)
        print(f"[data] DM train(epoch_len {len(tr)}) / val {len(va)} (静态 val，与 NN-AEC 同口径)",
              flush=True)
    elif args.synthetic:
        class _Syn(torch.utils.data.Dataset):
            def __init__(self, n, seed):
                self.n, self.seed = n, seed

            def __len__(self):
                return self.n

            def __getitem__(self, i):
                g = torch.Generator().manual_seed(self.seed * 100003 + i)
                clean = 0.1 * torch.randn(64000, generator=g)
                noisy = clean + 0.05 * torch.randn(64000, generator=g)
                return {"noisy": noisy, "clean": clean, "name": f"s{i}"}
        tr, va = _Syn(args.epoch_len, 1), _Syn(args.val_len, 999)
        # V5 v3a-045 族：_Syn 是**局部类**，num_workers>0 时 DataLoader 需 pickle 数据集
        # → 局部类不可 pickle，直接崩。--synthetic 一律退 0 worker。
        if args.num_workers > 0:
            print(f"[data] --synthetic 用局部类 _Syn（不可 pickle）→ num_workers "
                  f"{args.num_workers}->0", flush=True)
            args.num_workers = 0
        print("[data] SYNTHETIC")
    elif args.no_dm:
        tr = StaticPairs(n_pairs=args.epoch_len, seg_sec=args.seg_sec, seed=args.seed)
        va = StaticPairs(n_pairs=args.val_len, seg_sec=args.seg_sec, seed=12345)
        print(f"[data] STATIC pairs train {len(tr)} / val {len(va)}")
    else:
        tr = DNSDynamicMix(epoch_len=args.epoch_len, seg_sec=args.seg_sec)
        va = DNSDynamicMix(epoch_len=args.val_len, seg_sec=args.seg_sec, seed=123)
        print(f"[data] DYNAMIC MIXING train {len(tr)}/ep / val {len(va)} (seed 固定)")
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0)
    vl = DataLoader(va, batch_size=args.batch_size, num_workers=0)

    model = DCCRN(clstm_hidden=args.clstm_hidden, real_conv=args.real_conv).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] DCCRN{'(real-conv)' if args.real_conv else ''} {n_par/1e6:.2f}M -> {device}",
          flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # LR 调度口径由 core.train_protocol.resolve_schedule 定死（非 DM = 现状 patience 3；
    # DM 缺省 patience 20 + 关早停，阶段0 §2.3b）
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=sched_cfg["factor"], patience=sched_cfg["patience"])
    print(f"[sched] {sched_cfg['kind']} factor={sched_cfg['factor']} "
          f"patience={sched_cfg['patience']} early-stop={sched_cfg['early_stop']}"
          f"（{sched_cfg['basis']}）", flush=True)
    loss_fn = make_loss_fn(args.si_snr_weight, device)
    tb = SummaryWriter(out / "tb")

    # ckpt cfg：非 DM = 历史 `vars(args)`（P1-5a 新增键已剔除，逐键不变）；
    # DM 另挂 train_manifest（口径钉在权重旁边）
    if train_manifest is not None:
        args.epoch_len = train_manifest["epoch_scheme"]["epoch_len"]
    meta = legacy_cfg(args)
    if train_manifest is not None:
        meta["train_manifest"] = train_manifest
        if args.lr_patience is not None:
            meta["lr_patience"] = sched_cfg["patience"]

    start_ep, best, bad, step = 1, float("inf"), 0, 0
    ckpt_path, best_path = out / "last.ckpt", out / "best.ckpt"
    if args.resume:
        # V5 v3b-015 族：--resume 指向不存在的 ckpt 时显式报错（旧实现静默从头训练
        # 并覆盖同名输出）
        if not Path(args.resume).exists():
            raise SystemExit(f"[resume] 指定的 ckpt 不存在：{args.resume}"
                             "（拒绝静默从头训练并覆盖输出）")
        st = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_ep, best, bad, step = st["epoch"] + 1, st["best"], st["bad"], st.get("step", 0)
        prev = st.get("cfg", {}).get("train_manifest")
        if args.dynamic_mixing:
            # 阶段0 §2.3(a)：训练对照必须从零起。DCCRN 的历史 ckpt 是 DNS 数据轴 +
            # 旧 LR 口径的产物，续上去既无法检验数据变量、又会继承旧口径 → 直接拒绝。
            if prev is None:
                raise SystemExit(
                    f"[resume] {args.resume} 不是 DM 训练的 ckpt（cfg 无 train_manifest）：\n"
                    "  DM 对照必须从零起（REMEDIATION_PLAN_V3.1 附录B + 阶段0 §2.3a）；"
                    "历史 DCCRN 是 DNS 数据轴 + 旧 LR 口径，续训无法检验数据变量。\n"
                    "  若只是调试，请换独立 --out 目录从零起（不要复用/覆盖正式产物）")
            now = train_manifest
            drift = []
            if prev.get("epoch_scheme", {}).get("epoch_len") != now["epoch_scheme"]["epoch_len"]:
                drift.append("epoch_len")
            if (prev.get("data", {}).get("dt_plan", {}).get("sha256")
                    != now["data"]["dt_plan"]["sha256"]):
                drift.append("dt_plan_sha256")
            if (prev.get("data", {}).get("dt_plan", {}).get("preset")
                    != now["data"]["dt_plan"]["preset"]):
                drift.append("dt_plan_preset")
            if drift:
                best, bad = float("inf"), 0
                print(f"[resume] DM 口径漂移 {drift}：best/bad 已重置（旧 best 与当前 val "
                      f"口径不可比）。注意：口径漂移后的续训**不再等价**于从零起的对照。",
                      flush=True)
        elif prev is not None:
            print("[resume] [warn] 该 ckpt 来自 DM 训练，当前是非 DM（DNS/静态）训练："
                  "谱系/数据口径已变，不构成有效对照（阶段0 §2.3a）", flush=True)
        print(f"[resume] epoch {start_ep} best {best:.5f} step {step}", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        for batch in tl:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            loss = loss_fn(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tb.add_scalar("train/loss", float(loss), step)
            step += 1
            if args.probe_mem and step == args.probe_mem:
                # P1-5a / 附录B 第 3 步：定 batch 的显存实测（投影到 manifest 与 cfg）
                rec = probe_memory(args.batch_size, device, step, tb)
                meta["probe_mem"] = rec
                if train_manifest is not None:
                    train_manifest["probe_mem"] = rec
                    (out / "train_manifest.json").write_text(
                        json.dumps(train_manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        val_loss = validate(model, vl, loss_fn, device)
        sched.step(val_loss)
        tb.add_scalar("val/loss", val_loss, ep)
        tb.add_scalar("lr", opt.param_groups[0]["lr"], ep)
        mark = ""
        if val_loss < best:
            best, bad, mark = val_loss, 0, " *best*"
            # V5 v3b-016 族：best.ckpt 与 last.ckpt 存**同一组键**（旧实现 best.ckpt
            # 缺 opt/sched/epoch/bad/step → `--resume best.ckpt` 直接 KeyError）
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": ep, "best": best, "bad": bad,
                        "step": step, "val": val_loss, "cfg": meta}, best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best, "bad": bad,
                    "step": step, "val": val_loss, "cfg": meta}, ckpt_path)
        hist = {"epoch": ep, "val": val_loss, "best": best, "bad": bad, "step": step}
        if train_manifest is not None:
            # 附录B：history.json 记录**实际** epoch_len / 总步数，防 "100ep" 被误读
            hist.update(epoch_len=len(tr),
                        steps_per_epoch=train_manifest["epoch_scheme"]["steps_per_epoch"],
                        target_steps=train_manifest["epoch_scheme"]["target_steps"],
                        schedule=train_manifest["lr_sched"]["kind"],
                        patience=train_manifest["lr_sched"]["patience"],
                        early_stop=train_manifest["lr_sched"]["early_stop"])
            train_manifest["history_tail"] = hist
            (out / "train_manifest.json").write_text(
                json.dumps(train_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        (out / "history.json").write_text(json.dumps(hist))
        print(f"epoch {ep:3d}  val {val_loss:.5f}  ({time.perf_counter()-t0:.0f}s){mark}",
              flush=True)
        if args.early_stop > 0 and bad >= args.early_stop:
            # 0 = 关闭早停（DM 缺省；见 resolve_schedule 与阶段0 §2.3b）
            print(f"[early-stop] best {best:.5f}")
            break
    print(f"[done] best {best:.5f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
