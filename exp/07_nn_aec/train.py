"""NN-AEC 训练（WP-07 / 模块7，简历核心）。

损失（复用 core/losses，WORKLIST：加权失真 γ 消融）：
    L = weighted_distortion(enh谱, near谱, γ) + w_sisnr × SI-SNR(enh波形, near波形)
    γ 默认 0.5（AEC 式响度加权）；γ=0 退化为复数谱 MSE（消融开关 --gamma 0）

`--loss psycho`（P0-5 损失矩阵列②）：把上式第一项换成心理声学损失
（`core.losses.psychoacoustic_loss`，PAQM 派生的耳蜗压缩 + 临界带可听度加权），
共同项 SI-SNR 与其它超参一字不改 => 与 `--loss weighted`（v4 原列）构成单变量对照。
缩放常数 c 在启动时按**同批训练数据**实测（L_WD/L_psy），并连同探针数字写入
ckpt 的 cfg 与日志，保证"C 从哪来"可追溯（详见 `exp/07_nn_aec/loss_scale_probe.py`）。

DM（动态混合）训练（P1-3，REMEDIATION_PLAN_V3.1 §3.2）
----------------------------------------------------
`--dynamic-mixing` 打开训练时现合成：训练集换成 `dm_dataset.AecDynamicMix`
（DataLoader worker 现场跑 `aec_synth` 合成基元），val 仍是静态 `--val` 集
（口径与 v4 可比）。三条硬口径随 `train_manifest.json` 落盘：

1. **清单驱动**：场景配比 / DT SER / DT SNR / DT 引导全部来自
   `configs/dt_plan_v1.json` 的 preset（缺省 `train_v1_repro`），版本号 + sha256 入 manifest；
2. **池隔离**：`--near-list/--far-list/--noise-list` + `--near-dir/--far-dir/--noise-dir`
   **必须显式给**，实例化时做 near/far 互斥与 noise∩speech 校验（DM 没有 manifest 兜底）；
3. **epoch/LR 口径**（阶段0 §2.3b）：`epoch_len` 按附录B 方案(a) 换算（等效更新步数对齐
   v4 的 40830 步），LR/早停判据一并登记——DM 缺省 `patience=20` + **关闭早停**，
   修掉"patience=3 在 20–30ep 折半、ep55–71 早停"截断 240ep 预算的问题。

**训练对照必须从零起**（阶段0 §2.3a）：DM 训练拒绝从非 DM 谱系 ckpt 续训（口径不同、
无法检验数据/损失变量）；从零训练的标志 `from_scratch` 写入 manifest。

用法：
    python train.py --data D:/data/aec-synth/train_v1 --val D:/data/aec-synth/eval_v3 \
        --device cuda --epochs 60
    python train.py --smoke            # 合成张量链路验证
    python train.py ... --gamma 0.3    # γ 消融
    python train.py ... --loss psycho  # 心理声学损失列（P0-5）
    python train.py --device cuda --dynamic-mixing --epochs 240 \
        --near-dir <pool> --far-dir <pool> --noise-dir <pool> \
        --near-list split_lists/...txt --far-list split_lists/...txt \
        --noise-list split_lists/noise_train70.txt --rir-dir <rir-pool>   # DM（P1-3）
    python train.py --device cuda --probe-mem 20 --batch-size 16          # 显存探针
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

# P1X-2：池装配（多源）与 CLI 旗标与 04/dm_aec_synth 共用同一实现（口径一处实现）。
# 07 自己的目录已在上面挂进 sys.path，故这里可直接 import（模块级，parser 组装时就要用）。
import dm_dataset as DM  # noqa: E402
from dataset import AecDataset  # noqa: E402
from model import HOP, N_FFT, NNAEC  # noqa: E402

from core.losses import (  # noqa: E402
    psychoacoustic_loss,
    si_snr_loss,
    weighted_distortion_loss,
)
from core.stft import STFT  # noqa: E402

# 训练口径（LR/早停 + 显存探针）放 core/train_protocol.py：与 exp/04_dccrn/train.py 同源，
# 避免"两处实现漂移"（阶段0 教训）。常量与函数从该模块取值，本文件不再留副本。
from core.train_protocol import probe_memory  # noqa: E402
from core.train_protocol import resolve_schedule as _shared_resolve_schedule  # noqa: E402

#: 新增参数（DM/P1-3 引入）——**不进**非 DM 的 ckpt cfg，保证不开 --dynamic-mixing 时
#: 既有产物逐字节不变（口径信息只在 DM 的 train_manifest / cfg["train_manifest"] 里）
NEW_ARG_KEYS = ("dynamic_mixing", "probe_mem", "near_dir", "far_dir", "noise_dir",
                "near_list", "far_list", "noise_list", "dm_plan", "dm_preset",
                "dm_epoch_len", "dm_target_steps", "dm_seg_sec", "rir_dir", "rir_max_ms",
                "soft_clip_prob", "delay_min_ms", "delay_max_ms", "lr_patience",
                "cascade_linear", "trad_partitions", "trad_mu")


def resolve_schedule(dynamic_mixing, lr_patience=None, early_stop=None):
    """07 的 LR/早停口径入口：非 DM 缺省 = 现状（patience 3 / early-stop 8）。

    实现在 `core/train_protocol.resolve_schedule`（与 exp/04_dccrn/train.py 同源）。
    """
    return _shared_resolve_schedule(dynamic_mixing, lr_patience, early_stop,
                                    legacy_early_stop=8, legacy_patience=3)


def collate(batch):
    """变长保护：批内截到最短（aec_synth 定长生成时恒等长）。"""
    n = min(b["mic"].shape[-1] for b in batch)
    out = {k: torch.stack([b[k][..., :n] for b in batch]) for k in ("mic", "far", "near")}
    out["name"] = [b["name"] for b in batch]
    out["scenario"] = [b["scenario"] for b in batch]
    return out


def make_loss_fn(gamma, w_sisnr, device, kind="weighted", psycho_c=None):
    """构造损失函数。kind='weighted' = v4 原列；kind='psycho' 需给定缩放常数 psycho_c。"""
    stft = STFT(N_FFT, HOP).to(device)

    def loss_fn(model, batch):
        enh_wav, mask, mic_spec, enh_spec = model(batch["mic"], batch["far"])
        near_spec = stft.transform(batch["near"])
        if kind == "psycho":
            l_psy, _scale = psychoacoustic_loss(enh_spec, near_spec)
            loss = psycho_c * l_psy
        else:
            loss = weighted_distortion_loss(enh_spec, near_spec, gamma=gamma)
        if w_sisnr > 0:
            loss = loss + w_sisnr * si_snr_loss(enh_wav, batch["near"])
        return loss

    return loss_fn


@torch.no_grad()
def calibrate_psycho_scale(model, loader, device, gamma, w_sisnr, n_batches=4):
    """用训练流的最前 n_batches 批（未更新权重前）实测 c = mean(L_WD)/mean(L_psy)。

    在**同批数据、同一模型状态**下同时量两项谱误差 ⇒ 两列的谱项损失值天然可比，
    常数写入 ckpt cfg 与日志。返回 (c, 探针明细)。
    """
    stft = STFT(N_FFT, HOP).to(device)
    model.eval()
    wd_list, psy_list = [], []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        _enh, _mask, _mic_spec, enh_spec = model(batch["mic"], batch["far"])
        near_spec = stft.transform(batch["near"])
        wd_list.append(float(weighted_distortion_loss(enh_spec, near_spec, gamma=gamma)))
        psy_list.append(float(psychoacoustic_loss(enh_spec, near_spec)[0]))
    model.train()
    wd, psy = float(np.mean(wd_list)), float(np.mean(psy_list))
    return wd / max(psy, 1e-12), {"wd_probe": wd, "psy_probe": psy, "n_batches": len(wd_list),
                                  "w_sisnr": w_sisnr}


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        total += float(loss_fn(model, batch)) * len(batch["mic"])
        n += len(batch["mic"])
    return total / max(n, 1)


def build_dm_train_set(args, out, tag, sched_cfg):
    """按 DM 口径构建训练集与其留痕 manifest（P1-3）；缺参数/清单不合法一律 fail fast。"""
    if args.synthetic:
        raise SystemExit("--synthetic 与 --dynamic-mixing 互斥：DM 需要真实池 + 分割清单")
    if args.scenarios:
        raise SystemExit("--dynamic-mixing 下场景配比由清单 preset 决定，请去掉 --scenarios"
                         "（它只对静态数据集做场景过滤）")
    preset, plan_info = DM.load_plan(args.dm_plan, args.dm_preset)
    ratio_src = "preset"
    if args.dm_scenarios:
        preset = DM.with_scenario_ratio(preset, args.dm_scenarios.split(","))
        ratio_src = "cli_override"
        print(f"[dm] 场景配比显式覆盖 -> {preset['scenarios']}"
              f"（清单 preset 原值 {plan_info['scenarios']}；选择已记入 manifest）", flush=True)
    # 池装配：v5 单源（--{side}-dir/--{side}-list）与 P1X-2 多源
    # （--{side}-source 可重复）走同一入口；缺清单/歧义由 resolve_side_sources fail fast
    near_files, far_files, noise_files, pool_info = DM.build_pools_from_args(args)
    seg = float(args.dm_seg_sec or preset["dur_sec"] or 4.0)
    if preset["dur_sec"] and abs(seg - preset["dur_sec"]) > 1e-9:
        print(f"[dm] [warn] --dm-seg-sec={seg}s 与清单 dur_sec={preset['dur_sec']}s 不同"
              f"（DT 重叠段 = dur - leadin 随之变；口径已记入 manifest）", flush=True)
    sch = DM.resolve_epoch_scheme(args.epochs, args.batch_size,
                                  args.dm_target_steps or DM.V4_REF_STEPS,
                                  epoch_len_override=args.dm_epoch_len)
    if sch["steps_per_epoch"] < 1:
        raise SystemExit(f"--dm-epoch-len={sch['epoch_len']} < batch_size="
                         f"{args.batch_size}：drop_last 下每 epoch 0 步")
    # RIR 池：有 --rir-list 走清单驱动（隔离纪律），否则整目录枚举（平台无关序，t19）
    rir_files = DM.load_rir_pool(args.rir_dir, args.rir_list)
    if args.rir_dir and not rir_files:
        raise SystemExit(f"--rir-dir 内无 wav：{args.rir_dir}")
    ds_kwargs = dict(plan_info=plan_info, pool_info=pool_info, epoch_len=sch["epoch_len"],
                     seg_sec=seg, rir_files=rir_files, rir_max_ms=args.rir_max_ms,
                     rir_dir=args.rir_dir, rir_list=args.rir_list,
                     delay_ms_range=(args.delay_min_ms, args.delay_max_ms),
                     soft_clip_prob=args.soft_clip_prob, jitter_ms=args.jitter_ms,
                     seed=None)   # 训练取样吃熵：epoch 之间不重复（DM 的多样性来源）
    if args.cascade_linear:
        # P1-3b：级联 = 同一个 NN 换一对输入 (e_lin, y_echo)；线性级实现在 09（复用 06 的 TradAEC）
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "09_cascade_aec"))
        import cascade_dm
        tr = cascade_dm.AecCascadeDynamicMix(
            near_files, far_files, noise_files, preset,
            trad_kwargs={"method": "mdf", "partitions": args.trad_partitions,
                         "mu": args.trad_mu}, **ds_kwargs)
    else:
        tr = DM.AecDynamicMix(near_files, far_files, noise_files, preset, **ds_kwargs)
    manifest = {
        "schema": "train_manifest/1",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": {"out": str(out), "tag": tag, "device": args.device, "seed": args.seed,
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "num_workers": args.num_workers,
                "from_scratch": args.resume is None,
                "resume": str(args.resume) if args.resume else None},
        "loss": {"kind": args.loss, "gamma": args.gamma, "w_sisnr": args.w_sisnr},
        "data": dict(tr.provenance(), val=str(args.val),
                     val_jitter_ms=args.val_jitter_ms, train_mode=(
                         "cascade_dynamic_mixing" if args.cascade_linear
                         else "dynamic_mixing"),
                     scenario_ratio=preset["scenarios"],
                     scenarios_explicit=bool(args.dm_scenarios),
                     scenario_ratio_source=ratio_src),
        "epoch_scheme": sch,
        "lr_sched": sched_cfg,
        "note": ("附录B 方案(a)：等效更新步数对齐 v4；epoch_len 换算 + LR 调度/早停判据"
                 "一并登记（阶段0 §2.3b）；损失/数据对照必须从零起（§2.3a）"),
    }
    return manifest, tr


def print_dm_summary(manifest):
    """把 DM 口径打到 stdout（tmux 日志/阶段1 核对用）。"""
    d, sch, lr = manifest["data"], manifest["epoch_scheme"], manifest["lr_sched"]
    plan = d.get("dt_plan", {})
    run = manifest["run"]
    print(f"[dm] 清单 {plan.get('version')}/{plan.get('preset')} "
          f"sha256={str(plan.get('sha256'))[:12]}…  池 near/far/noise="
          f"{d['n_files']['near']}/{d['n_files']['far']}/{d['n_files']['noise']} "
          f"rir={d['n_files']['rir']}({d['rir_src']})", flush=True)
    print(f"[dm] 互斥校验通过：{d['mutex'].get('level')}（near/far 名单 + noise∩speech）"
          f"；场景配比 {d.get('scenario_ratio')}（来源 {d.get('scenario_ratio_source')}，"
          f"scenarios_explicit={d.get('scenarios_explicit')}）"
          f"；片段 {d['seg_sec']}s，delay {d['delay_ms_range']}ms，soft-clip "
          f"{d['soft_clip_prob']}，jitter {d['jitter_ms']:.0f}ms", flush=True)
    print(f"[dm] epoch 口径 方案({sch['scheme']})：目标 {sch['target_steps']} 步（v4 参照"
          f"）→ {sch['steps_per_epoch']} 步/epoch × {sch['epochs']}ep = "
          f"{sch['achieved_steps']} 步（Δ{sch['delta_steps']:+d}，"
          f"{sch['delta_frac'] * 100:+.3f}%）；epoch_len={sch['epoch_len']}"
          f"（来源 {sch['epoch_len_source']}）", flush=True)
    if sch["epoch_len_source"] == "cli_override":
        print("[dm] 注意：--dm-epoch-len 覆盖了方案(a) 的换算（冒烟/调试用）；正式训练请"
              "去掉它，让等效更新步数真正对齐 v4", flush=True)
    print("[dm] 注意：--data 不参与 DM 训练（训练集现合成）；val 用 --val 静态集", flush=True)
    lin = d.get("linear_stage")
    if lin:
        p = lin["trad_params"]
        print(f"[casc] 线性级 {p['method']}：partitions={p['partitions']} mu={p['mu']} "
              f"block={p['block']} use_dtd={p['use_dtd']} use_res={p['use_res']} "
              f"align_margin_ms={p['align_margin_ms']} reestimate_sec={p['reestimate_sec']}；"
              f"实例策略 {lin['trad_instance_policy']}（等效静态预计算="
              f"{lin['equivalent_to_static_precompute']}）；far 平移在 MDF 之后作用于 y_echo",
              flush=True)
        print(f"[casc] val 侧须用**同 per-clip 约定**预计算：python "
              f"exp/09_cascade_aec/precompute_cascade.py --synth-dir <eval集> "
              f"--out-dir <eval集>_lin --fresh-instance-per-clip "
              f"--partitions {p['partitions']} --mu {p['mu']}", flush=True)
    print(f"[dm] LR/早停口径：{lr['kind']} factor={lr['factor']} patience={lr['patience']}"
          f"{'（显式）' if lr['patience_explicit'] else '（DM 缺省）'}，early-stop="
          f"{lr['early_stop']}{'（显式）' if lr['early_stop_explicit'] else '（DM 缺省=关闭）'}"
          f"；依据 {lr['basis']}", flush=True)
    print(f"[dm] {'从零训练' if run['from_scratch'] else '续训（resume ' + str(run['resume']) + '）'}"
          f"——阶段0 §2.3(a)：损失/配方/数据对照须从零起", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("D:/data/aec-synth/train_v1"))
    ap.add_argument("--val", type=Path, default=Path("D:/data/aec-synth/eval_v3"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "outputs")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--w-sisnr", type=float, default=0.1)
    ap.add_argument("--loss", choices=["weighted", "psycho"], default="weighted",
                    help="谱误差项：weighted=v4 原列（能量幂律加权）；psycho=心理声学列（P0-5）")
    ap.add_argument("--psycho-c", type=float, default=None,
                    help="psycho 列的缩放常数；缺省时启动时按同批训练数据实测 L_WD/L_psy")
    ap.add_argument("--psycho-c-probe-batches", type=int, default=4,
                    help="实测缩放常数的探针批数（仅 --loss psycho 且未显式给 --psycho-c 时用）")
    ap.add_argument("--jitter-ms", type=float, default=10.0)
    ap.add_argument("--val-jitter-ms", type=float, default=0.0)
    ap.add_argument("--scenarios", default=None,
                    help="训练场景过滤（逗号分隔，如 dt,nst；默认全场景）")
    ap.add_argument("--early-stop", type=int, default=None,
                    help="早停的连续无改善 epoch 数；0=关闭。缺省：非 DM=8（现状）、"
                         "DM=0（阶段0 §2.3b：patience=3 的默认调度会截断 240ep 预算）")
    ap.add_argument("--lr-patience", type=int, default=None,
                    help="ReduceLROnPlateau patience；缺省：非 DM=3（现状）、DM=20。"
                         "口径与实际取值写入 ckpt cfg / train_manifest.json")
    # ---------------- P1-3：DM（动态混合）训练 + 显存探针 ----------------
    ap.add_argument("--dynamic-mixing", action="store_true",
                    help="训练时现合成（dm_dataset.AecDynamicMix）：需同时给三个池目录与三份"
                         "分割清单；val 仍是静态 --val 集。默认关闭 = 既有行为不变")
    ap.add_argument("--probe-mem", type=int, default=None, metavar="N",
                    help="第 N 个优化步后打印 torch.cuda.max_memory_allocated()/"
                         "max_memory_reserved()（阶段1 定 batch 用；CPU 下提示无意义）")
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
    ap.add_argument("--rir-list", type=Path, default=None,
                    help="DM：RIR 池清单（训练侧只应吃 split_lists/rir_train70.txt）。"
                         "给了 => 清单驱动并把清单名+sha256 记入 manifest.data.rir（t26）")
    ap.add_argument("--rir-max-ms", type=float, default=100.0, help="DM：RIR 截断（v4 口径 100ms）")
    ap.add_argument("--soft-clip-prob", type=float, default=0.15,
                    help="DM：功放软剪切样本比例（v4 口径 0.15）")
    ap.add_argument("--delay-min-ms", type=float, default=10.0, help="DM：回声时延下界")
    ap.add_argument("--delay-max-ms", type=float, default=80.0, help="DM：回声时延上界")
    ap.add_argument("--dm-plan", type=Path, default=None,
                    help="DM：DT 比例清单 JSON（缺省 configs/dt_plan_v1.json）")
    ap.add_argument("--dm-preset", default=None,
                    help="DM：清单 preset（缺省清单内默认 preset = train_v1_repro）")
    ap.add_argument("--dm-scenarios", default=None,
                    help="DM：显式覆盖场景配比 FEST,NST,DT（和=1），其余分布仍取 preset。"
                         "清单 preset 描述数据集配比，v4 训练还叠过 --scenarios dt,nst"
                         "（0.4914,0.5086）——要复刻 v4 训练配比就显式给并留痕")
    ap.add_argument("--dm-target-steps", type=int, default=None,
                    help="DM：等效更新步数目标（附录B 方案 a；缺省对齐 v4 的 40830 步）")
    ap.add_argument("--dm-epoch-len", type=int, default=None,
                    help="DM：显式 epoch_len（样本数，覆盖方案 a 的换算；冒烟/调试用，"
                         "覆盖事实与偏差一并写入 manifest）")
    ap.add_argument("--dm-seg-sec", type=float, default=None,
                    help="DM：片段时长；缺省取清单 preset 的 dur_sec")
    DM.add_dm_source_args(ap)      # P1X-2：多源池（--{near,far,noise}-source 可重复）
    ap.add_argument("--num-workers", type=int, default=4)
    # ---------------- P1-3b：级联 DM（线性级 MDF 进训练循环） ----------------
    ap.add_argument("--cascade-linear", action="store_true",
                    help="级联 DM：合成后跑 TradAEC(MDF) 线性级，NN 吃 (e_lin, y_echo)。"
                         "需与 --dynamic-mixing 同用；线性级口径进 manifest.linear_stage")
    ap.add_argument("--trad-partitions", type=int, default=8,
                    help="级联线性级 MDF 分块数（P1-1 三方对照口径 = 8）")
    ap.add_argument("--trad-mu", type=float, default=0.7,
                    help="级联线性级步长（P1-1 三方对照口径 = 0.7）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.num_workers, args.synthetic = 1, 0, True
    if args.cascade_linear and not args.dynamic_mixing:
        raise SystemExit("--cascade-linear 需要同时给 --dynamic-mixing：级联是 DM 合成后"
                         "接线性级（静态路径请沿用 precompute_cascade.py 预计算 + --data）")

    # 口径解析（阶段0 §2.3b）：LR/早停 + DM 的 epoch_len 换算，全部登记进 manifest
    sched_cfg = resolve_schedule(args.dynamic_mixing, args.lr_patience, args.early_stop)
    args.early_stop = sched_cfg["early_stop"]      # 既有早停判据用解析值（0 = 关闭）

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    tag = f"nnaec_g{args.gamma}_j{int(args.jitter_ms)}"
    if args.loss != "weighted":
        tag += f"_{args.loss}"
    if args.dynamic_mixing:
        tag += "_dm"        # DM 单独输出目录：不覆盖静态数据训练的既有产物（v4 谱系）
        if args.cascade_linear:
            tag += "_casc"  # 级联 DM 再分一个目录：不与纯 DM 的 run 混在一起
    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)

    train_manifest = None
    if args.dynamic_mixing:
        train_manifest, tr = build_dm_train_set(args, out, tag, sched_cfg)
        va = AecDataset(args.val, jitter_ms=args.val_jitter_ms, seed=123)
        print(f"[data] DM train(epoch_len {len(tr)}) / val {len(va)} "
              f"(val jitter {args.val_jitter_ms}ms)", flush=True)
        (out / "train_manifest.json").write_text(
            json.dumps(train_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_dm_summary(train_manifest)
    elif args.synthetic:
        class _Syn(torch.utils.data.Dataset):
            def __init__(self, n, seed):
                self.n, self.seed = n, seed

            def __len__(self):
                return self.n

            def __getitem__(self, i):
                g = torch.Generator().manual_seed(self.seed * 100003 + i)
                far = 0.2 * torch.randn(32000, generator=g)
                near = 0.1 * torch.randn(32000, generator=torch.Generator().manual_seed(i + 7))
                mic = far * 0.5 + near
                return {"mic": mic, "far": far, "near": near, "name": f"s{i}",
                        "scenario": "dt"}
        tr, va = _Syn(16, 1), _Syn(8, 999)
        # V5 v3b-013：_Syn 是**局部类**，num_workers>0 时 DataLoader 需要 pickle 数据集
        # （spawn/fork 子进程）→ 局部类不可 pickle，直接崩。--synthetic 一律退 0 worker。
        if args.num_workers > 0:
            print(f"[data] --synthetic 用局部类 _Syn（不可 pickle）→ num_workers "
                  f"{args.num_workers}->0", flush=True)
            args.num_workers = 0
        print("[data] SYNTHETIC")
    else:
        sc_filter = args.scenarios.split(",") if args.scenarios else None
        # V5 v3b-014：训练集也传 seed（旧实现只给验证集 seed=123 → 训练期增强不可复现）
        tr = AecDataset(args.data, jitter_ms=args.jitter_ms, scenarios=sc_filter,
                        seed=args.seed)
        va = AecDataset(args.val, jitter_ms=args.val_jitter_ms, seed=123)
        from collections import Counter
        print(f"[data] train {len(tr)} {dict(Counter(r['scenario'] for r in tr.rows))} "
              f"/ val {len(va)} (jitter {args.jitter_ms}/{args.val_jitter_ms}ms)")
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate, drop_last=True,
                    persistent_workers=args.num_workers > 0)
    vl = DataLoader(va, batch_size=args.batch_size, num_workers=0, collate_fn=collate)

    model = NNAEC().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] NNAEC {n_par/1e6:.3f}M params -> {device}", flush=True)
    assert n_par < 1e6, f"参数量 {n_par/1e6:.2f}M 超出 <1M 约束"
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # LR 调度口径由 resolve_schedule 定死（非 DM = 现状 patience 3；DM 缺省 patience 20）
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=sched_cfg["factor"], patience=sched_cfg["patience"])
    print(f"[sched] {sched_cfg['kind']} factor={sched_cfg['factor']} "
          f"patience={sched_cfg['patience']} early-stop={sched_cfg['early_stop']}"
          f"（{sched_cfg['basis']}）", flush=True)
    psycho_c = None
    probe: dict = {}
    if args.loss == "psycho":
        if args.psycho_c is not None:
            psycho_c = float(args.psycho_c)
            print(f"[loss] psycho 列，缩放常数由命令行给定 c={psycho_c:.6f}", flush=True)
        else:
            # 缩放常数标定（P0-5）：同批训练数据上实测 L_WD/L_psy ⇒ 两列谱误差项量级可比。
            # 用训练集**头部子集**建临时 loader（不动训练流的 shuffle 顺序，不消耗真实
            # epoch），模型状态=初始；常数写入 cfg 与日志，可回读复现。
            probe_n = args.batch_size * args.psycho_c_probe_batches * 2
            sub = torch.utils.data.Subset(tr, list(range(min(probe_n, len(tr)))))
            pl = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate, drop_last=True)
            psycho_c, probe = calibrate_psycho_scale(model, pl, device, args.gamma,
                                                     args.w_sisnr,
                                                     args.psycho_c_probe_batches)
            del pl, sub
            print(f"[loss] psycho 列，缩放常数实测 c={psycho_c:.6f} "
                  f"(WD={probe['wd_probe']:.4f} / PSY={probe['psy_probe']:.5f}，"
                  f"{probe['n_batches']} 批，模型状态=初始)", flush=True)
    loss_fn = make_loss_fn(args.gamma, args.w_sisnr, device, kind=args.loss, psycho_c=psycho_c)
    tb = SummaryWriter(out / "tb")
    loss_sig = (args.gamma, args.w_sisnr, args.scenarios, args.loss, psycho_c)
    # ckpt 元数据：把本列的口径钉死在权重旁边（P0-5 归因要求"回读 cfg 可复现"）。
    # P1-3：新增参数不进非 DM 的 cfg（不开 --dynamic-mixing 时 cfg 与现状逐键一致）；
    # DM 的 epoch/LR/池/清单口径走 meta["train_manifest"]（与 train_manifest.json 同源）。
    meta = {k: v for k, v in vars(args).items() if k not in NEW_ARG_KEYS}
    meta.update(loss_sig=loss_sig, loss_kind=args.loss, psycho_scale=psycho_c)
    if args.lr_patience is not None:            # 显式覆盖才落 cfg（缺省 = 现状，不动 cfg）
        meta["lr_patience"] = sched_cfg["patience"]
    if train_manifest is not None:
        meta["train_manifest"] = train_manifest
    if args.loss == "psycho":
        meta["psycho_probe"] = probe
        meta["psycho_spec"] = {"n_bands": 28, "alpha": 0.3, "tau": 1.5, "w_tilt": 0.1,
                               "tilt_band_hz": [4000.0, 8000.0],
                               "citation": "arXiv:2608.02918 (JAES 74(6) 430-444, 2026)",
                               "citation_retrieved": "2026-10-13"}

    start_ep, best, bad, step = 1, float("inf"), 0, 0
    ckpt_path, best_path = out / "last.ckpt", out / "best.ckpt"
    if args.resume:
        # V5 v3b-015：--resume 指向不存在的 ckpt 时**显式报错**。旧实现静默跳过 resume
        # 从头训练并覆盖同名输出（长训任务被悄悄清零，且无任何提示）。
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
            # 阶段0 §2.3(a)：训练对照必须从零起（或从同口径 DM 谱系续训）。非 DM 谱系的
            # ckpt 是用**别的数据分布与别的 LR 口径**训出来的，续上去既检验不了数据变量，
            # 又会把旧 best/lr 口径带进来——直接拒绝，别让"看起来在跑"掩盖口径失效。
            if prev is None:
                raise SystemExit(
                    f"[resume] {args.resume} 不是 DM 训练的 ckpt（cfg 无 train_manifest）：\n"
                    "  DM 对照必须从零起（REMEDIATION_PLAN_V3.1 附录B + 阶段0 §2.3a）；"
                    "非 DM 谱系续训无法检验数据/损失变量，且会继承旧 LR 口径。\n"
                    "  若只是调试，请换独立 --out 目录从零起（不要复用/覆盖正式产物）")
            drift = []
            if prev.get("epoch_scheme", {}).get("epoch_len") != \
                    train_manifest["epoch_scheme"]["epoch_len"]:
                drift.append("epoch_len")
            if (prev.get("data", {}).get("dt_plan", {}).get("sha256")
                    != train_manifest["data"]["dt_plan"]["sha256"]):
                drift.append("dt_plan_sha256")
            if prev.get("data", {}).get("dt_plan", {}).get("preset") != \
                    train_manifest["data"]["dt_plan"]["preset"]:
                drift.append("dt_plan_preset")
            if drift:
                best, bad = float("inf"), 0
                print(f"[resume] DM 口径漂移 {drift}：best/bad 已重置（旧 best 与当前 val "
                      f"口径不可比）。注意：口径漂移后的续训**不再等价**于从零起的对照。",
                      flush=True)
        elif prev is not None:
            # 非 DM 是历史路径：不阻断，但必须显式提示谱系混用（阶段0 §2.3a）
            print("[resume] [warn] 该 ckpt 来自 DM 训练，当前是非 DM（静态数据）训练："
                  "谱系/数据口径已变，不构成有效对照（阶段0 §2.3a）", flush=True)
        if st.get("cfg", {}).get("loss_sig") != loss_sig:
            # 损失函数配置变了（如 w-sisnr 0.1→0.5）：旧 best 口径失效，重置
            # （否则 val 尺度不同，best.ckpt 永远不会被新损失下的更优模型覆盖）
            best, bad = float("inf"), 0
            print("[resume] 检测到损失配置变化，best/bad 已重置", flush=True)
        print(f"[resume] epoch {start_ep} best {best:.5f}", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        for batch in tl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            loss = loss_fn(model, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tb.add_scalar("train/loss", float(loss), step)
            step += 1
            if args.probe_mem and step == args.probe_mem:
                # P1-3 / 附录B 第 3 步：定 batch 的显存实测（投影到 manifest 与 cfg）
                rec = probe_memory(args.batch_size, device, step, tb)
                meta["probe_mem"] = rec
                if train_manifest is not None:
                    train_manifest["probe_mem"] = rec
                    (out / "train_manifest.json").write_text(
                        json.dumps(train_manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        val_loss = validate(model, vl, loss_fn, device)
        sched.step(val_loss)
        cur_lr = float(opt.param_groups[0]["lr"])
        tb.add_scalar("val/loss", val_loss, ep)
        tb.add_scalar("val/lr", cur_lr, ep)
        mark = ""
        if val_loss < best:
            best, bad, mark = val_loss, 0, " *best*"
            # V5 v3b-016：best.ckpt 与 last.ckpt 存**同一组键**。旧实现 best.ckpt 缺
            # opt/sched/epoch/bad/step → `--resume best.ckpt` 直接 KeyError。
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": ep, "best": best, "bad": bad,
                        "step": step, "val": val_loss, "cfg": meta}, best_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best, "bad": bad,
                    "step": step, "val": val_loss, "cfg": meta}, ckpt_path)
        hist = {"epoch": ep, "val": val_loss, "best": best, "bad": bad, "step": step,
                "lr": cur_lr}
        if train_manifest is not None:
            # 附录B：history.json 记录**实际** epoch_len / 总步数，防 "240ep" 被误读为等价训练量
            hist.update(epoch_len=len(tr), steps_per_epoch=train_manifest["epoch_scheme"][
                "steps_per_epoch"], target_steps=train_manifest["epoch_scheme"]["target_steps"],
                schedule=train_manifest["lr_sched"]["kind"],
                patience=train_manifest["lr_sched"]["patience"],
                early_stop=train_manifest["lr_sched"]["early_stop"])
            train_manifest["history_tail"] = hist
            (out / "train_manifest.json").write_text(
                json.dumps(train_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
        (out / "history.json").write_text(json.dumps(hist))
        print(f"epoch {ep:3d}  val {val_loss:.5f}  lr {cur_lr:.2e}  "
              f"({time.perf_counter()-t0:.0f}s){mark}", flush=True)
        if args.early_stop > 0 and bad >= args.early_stop:
            # 0 = 关闭早停（DM 长训缺省；见 resolve_schedule 与阶段0 §2.3b）
            print(f"[early-stop] best {best:.5f}")
            break
    print(f"[done] best {best:.5f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
