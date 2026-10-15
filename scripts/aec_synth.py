"""AEC 数据合成管道（WP-07D / 模块7 前置）：造 (farend, nearend, mic) 三元组 + manifest。

物理链路（每条样本）：
    far  = 远端语音（电信号参考，AEC 的输入之一）
    echo = soft_clip( conv(far, RIR) )[右移 delay 样本] × echo_gain   # 扬声器->麦克风
    mic  = near + echo + noise                                       # 麦克风信号

三场景（配比 --scenarios，默认 FEST:0.5 / NST:0.25 / DT:0.25）：
    FEST 远端单讲: near=0,        mic = echo + noise（ERLE 统计段）
    NST 近端单讲: echo=0,        mic = near + noise（近端保真段）
    DT  双讲:     都有,          mic = near + echo + noise（SER 可配）

RIR 来源（三选一，自动降级）：
    1. --rir-dir  用户自备 RIR wav 目录（官方 AEC-Challenge RIR / 自采）
    2. pyroomacoustics 随机 shoe-box 房间仿真（默认；尺寸/吸声随机）
    3. 指数衰减噪声近似（无依赖兜底，仅冒烟用）

时延扰动：echo 相对 far 参考随机右移 delay∈[--delay-min-ms,--delay-max-ms]，
模拟 loopback 时延不确定性，训练 AEC 的时延鲁棒性（WP-07 用）。

时延漂移（--delay-drift，P2-1）：>0 时每条样本在基础时延上叠加慢随机游走
delay(t) = delay_ms + w(t)：w 每整秒步进 N(0, σ)、限幅 ±2σ（σ=--delay-drift ms），
节点间线性过渡；分数时延用线性插值实现（近似口径见 apply_delay_drift 注释）。
游走从**独立** Generator(default_rng([seed, 0x9E3779B9])) 抽样，不占主 rng——
同 seed 下开/关漂移的底座内容（场景/文件/时延/SER/SNR 序列）完全一致，可配对对比。
0（默认）= 关闭：不走漂移分支、不建不抽任何游走流，输出与固定时延版逐比特一致。

near/far 分池互斥（--nearfar-mutex，阶段0 工作项 0-3）：
    near_dir 与 far_dir 必须**按 reader/说话人切分且互斥**（REMEDIATION_PLAN_V3.1.md
    §3.1：near 池 70% / far 70%（与 near 互斥）/ held-out 15% 评测）。缺省 off
    不校验，行为与旧版逐比特一致；pool/utterance 两档在写盘前（fail fast）校验，
    违例直接退出、不产出任何 wav。判据与已知边界见 check_nearfar_mutex 注释。

DT 合成比例清单（--dt-plan / --dt-plan-preset，阶段0 工作项 0-5）：
    双讲参数（场景配比中的 DT 占比、DT 近端引导时长=重叠段补、DT 的 SER 与 SNR 分布）
    改由带版本号的 JSON 清单显式给出（默认 configs/dt_plan_v1.json，模块 scripts/dt_plan.py）。
    缺省不给 --dt-plan 时**不读清单、不追加列、不额外抽随机数**，行为与旧版逐比特一致；
    给了清单则 manifest 追加 dt_plan_version / dt_plan_preset 两列，并额外写出
    dt_plan_profile.json（清单路径 + sha256 + 解析后的分布），供 0-4 复现与收档引用。
    采样口径见 scripts/dt_plan.py docstring（单区间均匀、顺序 ser->snr->leadin）。

池枚举序（平台无关，P1-5c）：
    **所有**池枚举（near/far/noise/rir）都走 `list_pool` / `sort_pool_paths`——按池内
    相对路径的 **UTF-8 字节序**排序，而不是 `sorted(Path)`。后者比较的是 normcase 之后的
    部件（Windows 大小写折叠 / POSIX 不折叠）⇒ 池内文件名大小写混合时，"下标→文件"映射
    两平台分叉（实测：DNS 噪声池 637/659、eval_v8 噪声池 82/95 个位置不同；语音与 RIR 池
    恰好相同）。UTF-8 字节序 ≡ 码点序 ⇒ POSIX 既有行为不变，Windows 归一到 POSIX。
    详见 `pool_sort_key`。

用法：
    python aec_synth.py --num 100 --out-dir D:/data/aec-synth/v1 \
        --near-dir D:/data/dns-clean/near-pool \
        --far-dir  D:/data/dns-clean/far-pool \
        --noise-dir D:/data/dns-challenge-mirror/datasets/noise \
        --nearfar-mutex utterance \
        --dt-plan configs/dt_plan_v1.json --dt-plan-preset eval_v5_repro
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

try:                       # 0-5：DT 比例清单（脚本目录被调用方/测试插入 sys.path 时直接可导）
    import dt_plan as DP
except ImportError:        # pragma: no cover - 仅在 caller 未把 scripts/ 放进 sys.path 时
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "dt_plan", Path(__file__).resolve().with_name("dt_plan.py"))
    DP = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(DP)

SR = 16000
SCENARIOS = ("fest", "nst", "dt")
DT_PLAN_RATIO_TOL = 1e-6   # 清单配比与 --scenarios 的一致性容差（显式冲突即报错）


# ---------------------------------------------------------------- RIR 来源
def rir_simulate_pra(rng, sr=SR):
    """pyroomacoustics 随机房间：尺寸 3-8m、吸声 0.10-0.45、源/麦距>=0.5m。"""
    import pyroomacoustics as pra
    dim = rng.uniform(3, 8, size=3)
    mic = dim / 2 + rng.uniform(-0.5, 0.5, size=3)
    while True:
        src = rng.uniform(0.2, 1.0, size=3) * dim
        if np.linalg.norm(src - mic) >= 0.5:
            break
    room = pra.ShoeBox(dim, fs=sr, materials=pra.Material(rng.uniform(0.10, 0.45)),
                       max_order=8)
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    return room.rir[0][0].astype(np.float64)


def rir_exponential(rng, sr=SR):
    """兜底：直接路径脉冲 + 指数衰减尾（RT60 0.2-0.6s），仅冒烟/单测用。"""
    n = int(sr * rng.uniform(0.2, 0.6))
    t = np.arange(n) / sr
    tau = n / sr / 3
    h = rng.standard_normal(n) * np.exp(-t / tau)
    h[0] = 0.0
    d0 = rng.integers(0, int(0.03 * sr))       # 直接路径时延 0-30ms
    h = np.concatenate([np.zeros(d0), h])[: n + int(0.03 * sr)]
    return h / (np.linalg.norm(h) + 1e-12) * 10


# ---------------------------------------------------------------- 池枚举序（平台无关）
def pool_sort_key(root, p):
    """池枚举的**平台无关**排序键：池内相对路径（分隔符归一为 ``/``）的 UTF-8 字节序。

    为什么不能直接 ``sorted(Path)``：``PurePath.__lt__`` 比较的是 ``_cparts``（**normcase
    之后的部件**）——Windows 上 ``os.path.normcase`` 会**大小写折叠**并把 ``/`` 转成 ``\\``，
    POSIX 上则是恒等。于是同一份池在两平台上的枚举序不同：含大小写混合文件名时，
    "下标 → 文件名"的映射随之分叉，同 seed 下样本逐行不同（集合相等、排序不同）。

    实测（本机 Windows，池内相对路径）：DNS 噪声池 659 个文件里 **637 个位置**与字节序不同、
    eval_v8 噪声池 95 个里 **82 个**不同；而语音池（993/599）与 RIR 池（60）恰好相同
    —— 这正是 t12 在服务器（Linux）重放本地冻结集时"仅 noise_file 列 1234/1400 行映射
    不同、集合相等"的根因。

    选 UTF-8 字节序而不是别的：**UTF-8 的字节序与码点序一致**，而 POSIX 的 ``sorted(str)``
    就是码点序 ⇒ 服务器既有行为不变，Windows 归一到 POSIX 口径。
    """
    root = Path(root)
    try:
        rel = Path(p).relative_to(root)
    except ValueError:                       # 符号链接/跨挂载点等罕见情形：退化用原路径
        rel = Path(p)
    return str(rel).replace("\\", "/").encode("utf-8")


def sort_pool_paths(root, paths):
    """按池内相对路径的字节序排序（池枚举的**唯一**排序口径，见 `pool_sort_key`）。"""
    return sorted(paths, key=lambda p: pool_sort_key(root, p))


def list_pool(root):
    """枚举池内所有 wav（``**/*.wav``）——**唯一**入口，平台无关序。

    aec_synth / dm_dataset / 各训练入口的池枚举都必须走这里（或 `sort_pool_paths`），
    否则同 seed 的"下标→文件"映射会随操作系统变化（见 `pool_sort_key`）。
    """
    return sort_pool_paths(root, Path(root).glob("**/*.wav"))


def load_rir_dir(rir_dir):
    files = list_pool(rir_dir)               # 平台无关序（池下标→RIR 文件）
    if not files:
        raise FileNotFoundError(f"--rir-dir 无 wav: {rir_dir}")
    return files


def get_rir(rir_files, rng, max_ms=None):
    """按可用来源取一条 RIR；max_ms 截断（车厢 RIR 短，RT60>300ms 的部分截掉）。"""
    if rir_files is not None:
        h, _ = sf.read(str(rir_files[rng.integers(len(rir_files))]), dtype="float64",
                       always_2d=True)
        h = h[:, 0] if h.shape[1] == 1 else h.mean(1)
    else:
        try:
            h = rir_simulate_pra(rng)
        except ImportError:
            h = rir_exponential(rng)
    if max_ms:
        h = h[: int(SR * max_ms / 1000)]
    return h


# ---------------------------------------------------------------- 合成基元
def soft_clip(x, ceiling=0.5):
    """功放饱和近似（WP-08 非线性 AEC 的失真源开关）：y = c·tanh(x/c)。

    ceiling 以下信号近似线性，接近/超过 ceiling 处压缩——模拟小扬声器满幅失真。
    """
    return ceiling * np.tanh(x / ceiling)


def fit_len(x, n, rng):
    """裁到 n：长了随机裁剪，短了循环补。"""
    if len(x) >= n:
        off = rng.integers(0, len(x) - n + 1)
        return x[off:off + n].copy()
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def db_scale(x, ref_power, target_db, eps=1e-12):
    """把 x 缩放到 相对 ref_power 的目标 dB（SER/SNR 通用）。

    ref_power≈0（无参考，如 FEST 中噪声为零）时跳过定标、保持原电平。
    """
    if ref_power <= eps:
        return x
    p = np.mean(x ** 2) + eps
    g = 10 ** (target_db / 10) * ref_power / p
    return x * np.sqrt(g)


# ---------------------------------------------------------------- 时延漂移（P2-1）
def delay_drift_walk(n, delay_ms, drift_ms, rng):
    """慢随机游走节点序列（P2-1 --delay-drift，漂移幅度 σ=drift_ms）。

    w_0 = 0（基础时延处出发）；每整秒一步：w_k = clip(w_{k-1} + N(0, σ), -2σ, +2σ)，
    节点时刻 t_k = k 秒，覆盖 [0, ceil(n/SR)]。w 是相对基础时延 delay_ms 的偏移（ms），
    与固定时延口径一致。返回 (node_t_sec, node_w_ms)。
    """
    n_steps = int(np.ceil(n / SR))
    lim = 2.0 * drift_ms
    w = np.empty(n_steps + 1, dtype=np.float64)
    w[0] = 0.0
    for k in range(1, n_steps + 1):
        w[k] = np.clip(w[k - 1] + rng.normal(0.0, drift_ms), -lim, lim)
    return np.arange(n_steps + 1, dtype=np.float64), w


def apply_delay_drift(echo, n, delay_ms, drift_ms, rng):
    """把固定时延 echo 换成时变分数时延版本（仅 --delay-drift>0 时进入）。

    delay(t) = delay_ms + w(t)，w 为每秒步进 σ 的限幅随机游走（见 delay_drift_walk）。
    近似口径（二选一之「线性插值」，未采用 resample_poly 变比率重采样）：
      ① 游走在整秒节点间线性过渡（分段线性时延轨迹，慢漂移）；
      ② 输出样本 m 在 echo 上的分数读位置 pos = m - delay(t_m)·SR/1000，
         用一阶线性插值取值——相对理想带限分数时延有高频镜像/陷波误差，
         对 16k 语音的 AECMOS 评分影响可忽略（本开关关注时延失配本身）；
      ③ 读位置为负（段首大时延）与旧口径一致补零；far 参考不移；
         pos 越出 echo 右界的极端配置做边界夹取（正常参数到不了）。
    """
    node_t, node_w = delay_drift_walk(n, delay_ms, drift_ms, rng)
    t = np.arange(n) / SR
    tau_ms = np.interp(t, node_t, node_w) + delay_ms        # 每样本时延（ms）
    pos = np.arange(n) - tau_ms * SR / 1000.0               # echo 上的分数读位置
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    i0c = np.clip(i0, 0, len(echo) - 2)                     # 越界夹取（仅极端配置触顶）
    y = (1.0 - frac) * echo[i0c] + frac * echo[i0c + 1]
    y[pos < 0] = 0.0
    return y


def generate_one(near, far, noise, rir, scenario, ser_db, snr_db,
                 delay_ms, use_soft_clip, drift_ms=0.0, drift_rng=None):
    """单样本合成 -> (far_ref, near_clean, mic, noise, aux)。

    前四条等长 float32 波形（noise 一并返回便于测试对账/后续 ERLE 抽检）；
    aux 为参数 dict（manifest 用）。drift_ms>0 且 drift_rng 给定时，echo 时延
    = delay_ms + 慢随机游走（P2-1 --delay-drift，口径见 apply_delay_drift）；
    默认 drift_ms=0 走原固定时延路径，且不消耗 drift_rng 抽样（逐比特兼容）。

    定标协议（全在函数内完成，输入为原始电平）：
      FEST: echo 相对 noise 定 snr_db；NST: near 相对 noise 定 snr_db；
      DT:   near 相对 noise 定 snr_db，echo 相对 near 定 -ser_db（SER=P_near/P_echo）。
    防削波：mic 峰值超 0.9 时全部成分同缩 k（SER/SNR 口径不变）。
    """
    n = len(near)
    echo = fftconvolve(far, rir)[: n + int(SR * 0.1)]      # RIR 尾部截断缓冲
    if use_soft_clip:
        echo = soft_clip(echo)
    if drift_ms > 0 and drift_rng is not None:
        echo = apply_delay_drift(echo, n, delay_ms, drift_ms, drift_rng)
    else:
        d = int(round(delay_ms * SR / 1000))
        echo = np.concatenate([np.zeros(d), echo])[:n]      # 时延扰动（far 参考不移）
    noise_p = np.mean(noise ** 2)
    if scenario == "fest":
        near = np.zeros(n)
        echo = db_scale(echo, noise_p, snr_db)              # 回声 vs 噪声
    elif scenario == "nst":
        echo = np.zeros(n)
        near = db_scale(near, noise_p, snr_db)              # 近端 vs 噪声
    else:  # dt
        near = db_scale(near, noise_p, snr_db)
        echo = db_scale(echo, np.mean(near ** 2), -ser_db)  # SER=10log(P_near/P_echo)
    mic = near + echo + noise
    peak = np.abs(mic).max()
    if peak > 0.9:                                          # 防削波统一缩放（SER/SNR 不变）
        k = 0.9 / peak
        mic, near, echo, noise, far = mic * k, near * k, echo * k, noise * k, far * k
    aux = dict(delay_ms=float(delay_ms), ser_db=float(ser_db), snr_db=float(snr_db),
               soft_clip=int(bool(use_soft_clip)))
    if drift_ms > 0 and drift_rng is not None:
        aux["delay_drift_ms"] = float(drift_ms)             # 游走 σ；轨迹体现在 wav 内
    return (far.astype(np.float32), near.astype(np.float32), mic.astype(np.float32),
            noise.astype(np.float32), aux)


# ---------------------------------------------------------------- 批量驱动
def pick_scene(rng, scenario_ratio):
    r = rng.random()
    acc = 0.0
    for s, w in zip(SCENARIOS, scenario_ratio):
        acc += w
        if r < acc:
            return s
    return SCENARIOS[-1]


def load_wav_16k(path):
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == SR, f"{path} 采样率 {sr} != 16000（管道统一 16k）"
    return y.mean(1)


# ---------------------------------------------------------------- near/far 分池互斥（0-3）
MUTEX_LEVELS = ("off", "pool", "utterance")


class NearFarMutexError(ValueError):
    """near/far 分池互斥违例（--nearfar-mutex，阶段0 工作项 0-3）。"""


def _pool_path(p):
    """池路径规范化（比较用，不要求路径存在）。"""
    try:
        return Path(p).resolve()
    except OSError:                                          # pragma: no cover - 罕见
        return Path(p).absolute()


def _same_dir(a, b):
    """两池目录是否同一个目录项（含 Windows 大小写别名、目录 Junction/符号链接）。"""
    try:
        return Path(a).samefile(b)      # 路径存在时按 (卷号, 文件号) 比较
    except OSError:
        return _pool_path(a) == _pool_path(b)


def _file_ids(files):
    """文件同一性 -> 首个出现路径 的映射（evidence 用）。

    同一性优先取 (st_dev, st_ino)：硬链接/符号链接自动折叠成同一项；stat 失败或
    文件系统 st_ino 恒 0（部分网络/虚拟 FS）时退化为规范化绝对路径。
    """
    ids = {}
    for f in files:
        try:
            st = os.stat(f)
        except OSError:                                      # pragma: no cover - 罕见
            st = None
        key = ((st.st_dev, st.st_ino) if st is not None and st.st_ino
               else ("path", str(_pool_path(f))))
        ids.setdefault(key, str(f))
    return ids


def file_id_overlap(files_a, files_b):
    """两组文件的"同一个文件"交集（(卷号,文件号) 同一性，硬链接/符号链接折叠）。

    返回 ``(n, 例子字符串)``；``n == 0`` 时例子为空串。与池/清单判据共用同一实现，
    避免"哪个函数判什么"出现分叉。
    """
    ia, ib = _file_ids(files_a), _file_ids(files_b)
    shared = sorted(ia.keys() & ib.keys(), key=str)
    example = f"{ia[shared[0]]} / {ib[shared[0]]}" if shared else ""
    return len(shared), example


def nearfar_file_conflicts(near_files, far_files, level="utterance"):
    """只看**两份文件清单**的同源冲突（不判目录）；空列表 = 清单级互斥成立。

    与 `nearfar_pool_conflicts` 的分工：池路径**可以相同**（同一 clean 目录 + 两份互斥
    名单是合法切分），此时目录级判据会误报；本函数只吃清单本身，供 DM 训练
    （训练时现合成、没有 manifest 兜底，泄漏只能在入口拦）与清单预检复用。

    level：
      pool       只判文件同一性（含硬链接/符号链接折叠，见 `_file_ids`）；
      utterance  追加 utterance 主名（文件名）互斥，拦跨目录复制的同名 utterance。
    """
    reasons = []
    n, example = file_id_overlap(near_files, far_files)
    if n:
        reasons.append(f"{n} 个 wav 复用同一文件（例：near {example}）")
    if level == "utterance":
        stems = nearfar_utterance_conflicts(near_files, far_files)
        if stems:
            reasons.append(f"{len(stems)} 个 utterance 主名两清单同名（例：{stems[0]}）")
    return reasons


def nearfar_pool_conflicts(near_dir, far_dir, near_files=None, far_files=None):
    """列出 near/far 池的**同源**证据（0-3）；空列表 = 池级互斥成立。

    判据（--nearfar-mutex=pool 起生效）：
      1. 同一目录：samefile 折到同一个目录项——同路径、Windows 大小写别名、目录
         Junction/符号链接、尾部分隔符写法差异都会被识破；
      2. 目录嵌套：一者是另一者的祖先。旧文档示例把同一个 clean/ 同时喂给两端即属
         此类：far 池实质是 near 池的子集，同一说话人双端复用 = 说话人泄漏；
      3. 文件复用：两池 wav 的文件同一性有交集（含硬链接——路径不同但同一文件项）。

    不做（有意）：**改名副本**（同一条语音拷成不同文件名）内容级不可判——未做全量
    哈希（大池每次启动哈希代价高）；内容去重请按 0-2 的 sha256 清单法离线做一次。
    utterance 主名判据见 nearfar_utterance_conflicts；只看清单（不判目录）的入口
    见 nearfar_file_conflicts。
    """
    if near_files is None:
        near_files = list_pool(near_dir)
    if far_files is None:
        far_files = list_pool(far_dir)
    reasons = []
    nd, fd = _pool_path(near_dir), _pool_path(far_dir)
    if _same_dir(near_dir, far_dir):
        reasons.append(f"near_dir 与 far_dir 指向同一目录：{nd}")
    elif nd.is_relative_to(fd) or fd.is_relative_to(nd):
        parent, child = (nd, fd) if fd.is_relative_to(nd) else (fd, nd)
        reasons.append(f"目录嵌套（同源）：{child} 位于 {parent} 之下")
    else:
        # 目录级无交集时才查文件同一性（同目录/嵌套时文件必然复用，报告冗余）
        reasons += nearfar_file_conflicts(near_files, far_files, level="pool")
    return reasons


def nearfar_utterance_conflicts(near_files, far_files):
    """utterance 主名（stem）级同源：同一条语音/同一 id 在 near 与 far 双侧出现。

    口径：`Path(p).stem` 视为 utterance 标识（DNS/LibriSpeech/AISHELL 的命名空间都是
    说话人目录 + utterance 文件名）。池按 reader 切分时两端命名空间天然不重叠；出现
    同名即视为同源复用（换目录复制的同一 utterance 也会被拦住）。
    """
    nk = {Path(f).stem for f in near_files}
    fk = {Path(f).stem for f in far_files}
    return sorted(nk & fk)


def check_nearfar_mutex(near_dir, far_dir, level=MUTEX_LEVELS[0], near_files=None,
                        far_files=None):
    """按 level 校验 near/far 池互斥；违例抛 NearFarMutexError（消息含证据与修法）。

    level（--nearfar-mutex，缺省 off）：
      off        不校验、不 glob 不 stat：默认行为与旧版逐比特一致（旧命令照跑）；
      pool       同一目录 / 目录嵌套 / 两池复用同一文件 → 违例；
      utterance  在 pool 之上追加 utterance 主名（文件名）级互斥——拦跨目录复制的
                 同名 utterance（内容级改名副本仍需离线 sha256 清单，见 pool 注释）。

    调用点（main）在做任何写盘之前，违例即退出、不产出半套数据：隔离性违例的代价是
    评测数字失真，宁可不产出也不能静默生成（0-4 eval_v8 依赖本约束成立）。
    """
    if level == "off":
        return
    if level not in MUTEX_LEVELS:                            # pragma: no cover - argparse 已挡
        raise ValueError(f"未知 --nearfar-mutex 级别：{level}")
    reasons = nearfar_pool_conflicts(near_dir, far_dir, near_files, far_files)
    if level == "utterance":
        if near_files is None:
            near_files = list_pool(near_dir)
        if far_files is None:
            far_files = list_pool(far_dir)
        stems = nearfar_utterance_conflicts(near_files, far_files)
        if stems:
            reasons.append(f"{len(stems)} 个 utterance 主名两池同名"
                           f"（例：{stems[0]}）")
    if reasons:
        raise NearFarMutexError(
            "；".join(reasons)
            + "。修法：near 池与 far 池必须按 reader/说话人切分且互斥（near 70% / "
              "far 70% 与 near 互斥 / held-out 15% 评测；见 REMEDIATION_PLAN_V3.1.md §3.1）")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--near-dir", type=Path, required=True)
    ap.add_argument("--far-dir", type=Path, required=True)
    ap.add_argument("--noise-dir", type=Path, required=True)
    ap.add_argument("--rir-dir", type=Path, default=None,
                    help="自备 RIR wav 目录；缺省用 pyroomacoustics 仿真")
    ap.add_argument("--rir-max-ms", type=float, default=None,
                    help="RIR 截断长度（车厢场景 ~300ms；None 不截断）")
    ap.add_argument("--scenarios", default=None, metavar="FEST,NST,DT",
                    help="FEST,NST,DT 配比（逗号分隔）。缺省 0.5,0.25,0.25；给了 "
                         "--dt-plan 时以清单为准——此处**显式**填了与清单不同的配比即报错"
                         "（不静默二选一），缺省值被清单覆盖（0-5）")
    ap.add_argument("--dur-sec", type=float, default=4.0)
    ap.add_argument("--ser-range", default="0,10", help="DT 场景 SER dB 范围")
    ap.add_argument("--snr-range", default="5,20", help="噪声 SNR dB 范围")
    ap.add_argument("--delay-min-ms", type=float, default=10.0)
    ap.add_argument("--delay-max-ms", type=float, default=80.0)
    ap.add_argument("--delay-drift", type=float, default=0.0,
                    help="回声时延漂移 σ（ms，P2-1）：>0 时每条样本时延 = 基础时延 + "
                         "慢随机游走（每整秒步进 N(0,σ)、限幅 ±2σ、节点间线性过渡），"
                         "分数时延用线性插值实现（近似口径见 apply_delay_drift 注释）；"
                         "游走用独立 Generator（不占主 rng，同 seed 开/关漂移底座内容一致，"
                         "可配对对比）；0（默认）关闭，输出与固定时延版逐比特一致")
    ap.add_argument("--soft-clip-prob", type=float, default=0.3,
                    help="启用功放软剪切的样本比例（WP-08 开关）")
    ap.add_argument("--dt-leadin-sec", type=float, default=2.0,
                    help="DT 样本近端加入前的纯远端引导秒数（真实通话时序：先 FEST 收敛再双讲）")
    ap.add_argument("--nearfar-mutex", choices=MUTEX_LEVELS, default="off",
                    help="near/far 分池互斥校验（阶段0 工作项 0-3）。off（默认）=不校验，"
                         "行为与旧版逐比特一致；pool=同一目录/目录嵌套/两池复用同一文件即"
                         "报错；utterance=pool 之上再要求 utterance 主名两端不重叠。"
                         "校验在任何写盘之前，违例直接退出、不产出 wav/manifest")
    ap.add_argument("--dt-plan", type=Path, default=None,
                    help="DT 合成比例清单 JSON（阶段0 工作项 0-5，默认 configs/dt_plan_v1.json）。"
                         "给了清单则按 preset 显式采样 DT 的 SER/SNR/引导时长，manifest 追加 "
                         "dt_plan_version/dt_plan_preset 列并写出 dt_plan_profile.json；"
                         "不给则完全不读清单、行为与旧版逐比特一致")
    ap.add_argument("--dt-plan-preset", default=None,
                    help="--dt-plan 下的 preset 名（缺省取清单内的默认 preset）")
    ap.add_argument("--dt-plan-list", action="store_true",
                    help="列出 --dt-plan 清单里的 preset 后退出（不合成）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # ---- 0-5：清单载入（在任何写盘/抽随机数之前；结构非法即退出）
    plan_ref, preset = None, None
    if args.dt_plan_preset and not args.dt_plan:
        raise SystemExit("[aec_synth] --dt-plan-preset 需要同时给 --dt-plan <清单 JSON>")
    if args.dt_plan:
        try:
            plan, plan_sha = DP.load_plan(args.dt_plan)
            preset = DP.resolve_preset(plan, args.dt_plan_preset, plan_sha)
        except DP.DtPlanError as e:
            raise SystemExit(f"[aec_synth] DT 比例清单不可用：{e}") from None
        plan_ref = {"path": str(args.dt_plan), "sha256": plan_sha,
                    "version": plan["version"], "preset": preset["preset"]}
        if args.dt_plan_list:
            print(f"[aec_synth] {args.dt_plan}（{plan['version']}，sha256={plan_sha[:12]}…）")
            for name in plan["presets"]:
                rp = DP.resolve_preset(plan, name, plan_sha)
                print(f"  {name}: DT 占比 {rp['scenarios']['dt']} | 引导 "
                      f"{DP.dist_repr(rp['dt_leadin_sec'])}s | SER "
                      f"{DP.dist_repr(rp['dt_ser_db'])}dB | SNR "
                      f"{DP.dist_repr(rp['dt_snr_db'])}dB")
            return

    DEFAULT_SCENARIOS = "0.5,0.25,0.25"       # 与旧版默认值一致（逐比特兼容）
    explicit_ratio = args.scenarios is not None
    scenarios_arg = args.scenarios or DEFAULT_SCENARIOS
    ratio = [float(x) for x in scenarios_arg.split(",")]
    assert len(ratio) == 3 and abs(sum(ratio) - 1) < 1e-6, "配比需 3 个数且和为 1"
    ser_lo, ser_hi = map(float, args.ser_range.split(","))
    snr_lo, snr_hi = map(float, args.snr_range.split(","))
    if preset is not None:
        # 清单是 DT 口径的权威来源：显式填了冲突配比即报错（不静默二选一）；
        # 只吃 CLI 默认值时由清单覆盖，无需用户重复声明清单里的配比。
        plan_ratio = [preset["scenarios"][s] for s in SCENARIOS]
        if explicit_ratio and any(abs(a - b) > DT_PLAN_RATIO_TOL
                                  for a, b in zip(ratio, plan_ratio)):
            raise SystemExit(
                f"[aec_synth] --scenarios={args.scenarios} 与清单 preset "
                f"{preset['preset']!r} 的配比 {plan_ratio} 冲突：请只保留清单口径"
                f"（去掉 --scenarios，或用清单里的 preset）")
        ratio = plan_ratio
        print(f"[aec_synth] DT 比例清单 {plan_ref['version']} / preset {preset['preset']}"
              f"（sha256={plan_ref['sha256'][:12]}…）")
        print(f"  DT 占比 {preset['scenarios']['dt']} | 引导 "
              f"{DP.dist_repr(preset['dt_leadin_sec'])}s | SER "
              f"{DP.dist_repr(preset['dt_ser_db'])}dB | SNR "
              f"{DP.dist_repr(preset['dt_snr_db'])}dB")
        if preset["dur_sec"] and abs(args.dur_sec - preset["dur_sec"]) > 1e-9:
            print(f"  [warn] --dur-sec={args.dur_sec} 与清单记录的 {preset['dur_sec']} 不同"
                  f"（清单里 dur_sec 只作记录，重叠段 = dur_sec - 引导时长）")
    rng = np.random.default_rng(args.seed)
    # 游走专用 Generator（P2-1）：与主 rng 分流——drift>0 时游走抽样不扰动主
    # 序列，同 seed 下开/关漂移的底座内容逐项一致（可做配对对照）；drift=0 时
    # 不建不抽，默认路径逐比特兼容。
    walk_rng = (np.random.default_rng([args.seed, 0x9E3779B9])
                if args.delay_drift > 0 else None)

    # 池枚举必须走 list_pool（平台无关序）：下标→文件名 的映射决定同 seed 的逐行结果
    near_files = list_pool(args.near_dir)
    far_files = list_pool(args.far_dir)
    noise_files = list_pool(args.noise_dir)
    assert near_files and far_files and noise_files, "输入目录缺 wav"
    if args.nearfar_mutex != "off":
        # 0-3：写盘前 fail fast（违例不产出任何 wav/manifest）
        try:
            check_nearfar_mutex(args.near_dir, args.far_dir, args.nearfar_mutex,
                                near_files=near_files, far_files=far_files)
        except NearFarMutexError as e:
            raise SystemExit(f"[aec_synth] near/far 分池互斥违例"
                             f"（--nearfar-mutex={args.nearfar_mutex}）：{e}") from None
    rir_files = load_rir_dir(args.rir_dir) if args.rir_dir else None
    if rir_files:
        rir_src = "dir"
    else:
        try:
            import pyroomacoustics  # noqa: F401
            rir_src = "sim"
        except ImportError:
            rir_src = "exp"      # 指数衰减近似（仅冒烟）

    out = args.out_dir
    for sub in ("farend", "nearend", "mic"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    n = int(args.dur_sec * SR)
    rows = []
    for i in range(args.num):
        scene = pick_scene(rng, ratio)
        near_f = near_files[rng.integers(len(near_files))]
        far_f = far_files[rng.integers(len(far_files))]
        noise_f = noise_files[rng.integers(len(noise_files))]
        near = fit_len(load_wav_16k(near_f), n, rng)
        far = fit_len(load_wav_16k(far_f), n, rng)
        noise = fit_len(load_wav_16k(noise_f), n, rng)
        rir = get_rir(rir_files, rng, max_ms=args.rir_max_ms)
        leadin_sec = args.dt_leadin_sec
        ser_db, snr_db = rng.uniform(ser_lo, ser_hi), rng.uniform(snr_lo, snr_hi)
        if preset is not None:
            # 0-5：清单口径采样。顺序固定 ser -> snr -> leadin（见 dt_plan.sample_dt），
            # 各参数仍是一次 rng 调用（单区间均匀），故随机流与旧配方同构、可回放核对。
            s = DP.sample_dt(rng, preset)
            ser_db, snr_db, leadin_sec = s["ser_db"], s["snr_db"], s["leadin_sec"]
        if scene == "dt" and leadin_sec > 0:
            lead = int(leadin_sec * SR)
            near[:lead] = 0.0            # 引导段纯远端：给自适应滤波收敛窗口
        far_r, near_r, mic_r, _, par = generate_one(
            near, far, noise, rir, scene,
            ser_db=ser_db,
            snr_db=snr_db,
            delay_ms=rng.uniform(args.delay_min_ms, args.delay_max_ms),
            use_soft_clip=bool(rng.random() < args.soft_clip_prob),
            drift_ms=args.delay_drift, drift_rng=walk_rng)
        uid = f"fileid_{i}_{scene}"
        sf.write(out / "farend" / f"{uid}.wav", far_r, SR)
        sf.write(out / "nearend" / f"{uid}.wav", near_r, SR)
        sf.write(out / "mic" / f"{uid}.wav", mic_r, SR)
        row = dict(name=uid, scenario=scene, rir_src=rir_src,
                   near_file=near_f.name, far_file=far_f.name,
                   noise_file=noise_f.name,
                   leadin_sec=leadin_sec if scene == "dt" else 0.0, **par)
        if args.nearfar_mutex != "off":
            row["nearfar_mutex"] = args.nearfar_mutex   # 0-3 留痕：本次跑过互斥校验
        if preset is not None:
            row["dt_plan_version"] = preset["version"]      # 0-5 留痕：本次 DT 口径来源
            row["dt_plan_preset"] = preset["preset"]
        rows.append(row)
    fields = ["name", "scenario", "rir_src", "near_file", "far_file", "noise_file",
              "delay_ms", "ser_db", "snr_db", "soft_clip", "leadin_sec"]
    if args.delay_drift > 0:
        fields.append("delay_drift_ms")
    if args.nearfar_mutex != "off":
        fields.append("nearfar_mutex")
    if preset is not None:
        fields += ["dt_plan_version", "dt_plan_preset"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if preset is not None:
        profile = dict(plan_ref, schema=plan.get("schema", ""),
                       created=plan.get("created", ""),
                       scenarios=preset["scenarios"],
                       dt_leadin_sec=preset["dt_leadin_sec"],
                       dt_ser_db=preset["dt_ser_db"],
                       dt_snr_db=preset["dt_snr_db"],
                       dur_sec=preset["dur_sec"] or args.dur_sec,
                       # 复现留痕（0-5）：随机流消耗序 + 各池规模。
                       # 池规模必须记：numpy `integers(n)` 的消耗量依赖 n 是否为 2 的幂
                       # （1 次或 2 次 32/64 位抽样），只有真实规模才能逐位回放随机流。
                       sampled_order=["scene", "near_idx", "far_idx", "noise_idx",
                                      "rir_draw", "ser_db", "snr_db", "leadin_sec",
                                      "delay_ms", "soft_clip"],
                       pool_sizes={"near": len(near_files), "far": len(far_files),
                                   "noise": len(noise_files),
                                   "rir": len(rir_files) if rir_files else 0},
                       pool_dirs={"near": str(args.near_dir), "far": str(args.far_dir),
                                  "noise": str(args.noise_dir),
                                  "rir": str(args.rir_dir) if args.rir_dir else None},
                       num=args.num, seed=args.seed,
                       # 不经清单的参数也如实记下（复现/审计用）
                       cli_args={"num": args.num, "dur_sec": args.dur_sec, "seed": args.seed,
                                 "snr_range": args.snr_range, "ser_range": args.ser_range,
                                 "soft_clip_prob": args.soft_clip_prob,
                                 "delay_min_ms": args.delay_min_ms,
                                 "delay_max_ms": args.delay_max_ms,
                                 "delay_drift": args.delay_drift,
                                 "rir_max_ms": args.rir_max_ms,
                                 "nearfar_mutex": args.nearfar_mutex})
        with open(out / "dt_plan_profile.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
            f.write("\n")
    from collections import Counter
    print(f"[aec_synth] {args.num} 条 -> {out}")
    if args.delay_drift > 0:
        print(f"  delay-drift σ={args.delay_drift} ms（每秒步进，限幅 ±{2 * args.delay_drift} ms）")
    if args.nearfar_mutex != "off":
        print(f"  near/far 分池互斥：{args.nearfar_mutex}（校验通过，0-3）")
    if preset is not None:
        print(f"  DT 比例清单：{preset['version']}/{preset['preset']}"
              f"（profile -> {out / 'dt_plan_profile.json'}，0-5）")
    print(f"  场景分布 {dict(Counter(r['scenario'] for r in rows))}（目标 {ratio}）")
    print(f"  manifest: {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
