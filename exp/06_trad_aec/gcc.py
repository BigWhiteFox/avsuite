"""GCC-PHAT 时延估计（WP-06）。

互相关 + PHAT 白化（每频点单位模长）——语音宽带相关函数会被谐波结构主导，
白化后峰更尖锐、对混响更鲁棒。
"""
import numpy as np


def gcc_phat_stats(mic: np.ndarray, far: np.ndarray, sr: int,
                   max_delay_ms: float = 120.0):
    """GCC-PHAT 时延 + 峰旁瓣比（回声存在性证据）。

    peak_ratio = 相关峰 / 旁瓣中位数：有回声（线性卷积关系）时白化相关峰尖锐
    （实测 FEST 19-84 / DT 15-66）；mic 与 far 独立时相关是平坦噪声
    （实测 NST 4.6-13.4）。阈值 14 可分——比累积相关系数 robust 得多：
    后者被混响稀释（corr ≈ 直达径/总能量比，仅 0.1-0.3），与 NST 噪声带重叠。
    返回 (delay_ms, peak_ratio)。

    **peak_ratio 的尺度依赖性（V5 审计 G7，阈值 14 仅对标定窗长有效）**：
    本统计量 = 相关峰 / |旁瓣| 中位数，**随分析窗长单调增大**，不是尺度不变量。
    同一内容切不同长度实测（本机合成）：0.1s → 404、1s → 1328、2s → 2141、
    4s → 2262、10s → 2423、20s → 2490（有回声型）；无回声型 0.1s → 5.6~7.5、
    ≥1s → 4.2~6.2。因此：
    - 阈值 14 只在"当初标定时的窗长"（整段、秒级~十几秒）下有实测意义；
      换评估窗长（例如统一 10s 或极小片段）后 FEST/DT 的判定余量与
      "实测 FEST 19-84 / DT 15-66 / NST 4.6-13.4" 不再可比；
    - 短段（≤0.1s）无回声侧已实测到 7.5（阈值的一半），存在误判"有回声"的风险。
    调用方若改变分析窗长，应重新标定 EVIDENCE_THR（见 pipeline.py 与
    exp/06_trad_aec/README.md「GCC 证据口径」附记）。

    **非有限输入**：mic/far 含 NaN/Inf 时先按 0 处理（该样本视为缺失），
    避免 FFT 把非有限值扩散到整条频带；`out_valid=False` 表示 far（参考端）
    本身含非有限样本、本次 peak_ratio 不可用于回声存在性判决，调用方必须
    显式降级（见 pipeline.process 的 echo_prior 处理），不得把 nan 当"无回声"。
    """
    n = len(mic)
    if n < 2:
        # V5 审计 G3：n=0 时 log2(0)=-inf -> int() 抛 OverflowError；
        # n=1 时 N=2、lag 表会越界 IndexError。极短输入直接返回不可用证据。
        return 0.0, float("nan"), False
    N = 1 << int(np.ceil(np.log2(2 * n)))
    # 非有限输入的最小防护（触发条件：mic/far 含 NaN 或 inf，例如
    # tests/test_trad_aec_hardening.py::test_mdf_divergence_guard_keeps_output_finite
    # 往 mic 里注入 NaN 突发、或发散信号上溢为 inf）：FFT 会把非有限样本扩散到
    # 整条频带，R 出现 inf/NaN，白化时 inf/inf、NaN/NaN 得到 NaN，经 irfft 污染
    # 整帧相关峰与 peak_ratio（实测 ratio=nan）。先按 0 处理非有限样本；
    # 有限输入下 nan_to_num 是恒等变换，数值逐位不变。
    mic = np.nan_to_num(mic, nan=0.0, posinf=0.0, neginf=0.0)
    far_bad = not np.isfinite(far).all()
    far = np.nan_to_num(far, nan=0.0, posinf=0.0, neginf=0.0)
    M, F = np.fft.rfft(mic, N), np.fft.rfft(far, N)
    R = M * np.conj(F)
    # 上溢防护（触发条件：输入本身有限但幅值极端，M*conj(F) 溢出为 inf）：
    # 非有限频点置 0（该频点不参与白化），避免 inf/inf 与 NaN 进入 irfft。
    # 全部频点有限时本行不做任何写入，白化路径与数值结果保持原样。
    np.copyto(R, 0.0, where=~np.isfinite(R))
    R /= np.abs(R) + 1e-12                       # PHAT 白化
    cc = np.fft.irfft(R, N)
    # V5 审计 G4：maxlag 必须在可搜索范围内（lag 表用 N-maxlag..N-1 覆盖负 lag），
    # 否则 np.arange 越界 IndexError（实测 max_delay_ms=4000 @ n=16000 -> N=32768 抛错）。
    maxlag = min(max(int(max_delay_ms * sr / 1000), 1), N // 2 - 1)
    lags = np.concatenate([np.arange(0, maxlag + 1), np.arange(N - maxlag, N)])
    k = int(lags[np.argmax(cc[lags])])
    d = k if k <= maxlag else k - N              # 负 lag = far 滞后 mic（非物理，截 0）
    seg = np.abs(cc[:maxlag])
    # V5 v3a-072：peak_ratio 的分子必须与"延迟搜索实际找到的峰"同属一个 lag 集。
    # 旧式分子写死 cc[:maxlag].max()（只含正 lag 区），而峰搜索还含负 lag 环回区
    # （np.arange(N-maxlag, N)）——真时延超出搜索窗时峰落在环回区，分子取不到峰，
    # ratio 被系统性低估 ⇒ echo_prior 误判"无回声"⇒ DTD 持续硬冻结。
    # 搜索峰落在正 lag 区内（有限输入的默认路径）时，第一支与旧式**逐位相同**。
    num = cc[:maxlag].max() if 0 <= k < maxlag else abs(cc[k])
    ratio = float(num / (np.median(seg) + 1e-12))
    return max(d, 0) / sr * 1000.0, ratio, not far_bad


def gcc_phat_delay(mic: np.ndarray, far: np.ndarray, sr: int,
                   max_delay_ms: float = 120.0) -> float:
    """估计 mic 中回声相对 far 参考的时延（ms，正数=mic 滞后）。"""
    return gcc_phat_stats(mic, far, sr, max_delay_ms)[0]


def align_far(far: np.ndarray, delay_ms: float, sr: int, n_out: int) -> np.ndarray:
    """把 far 右移 delay_ms（对齐到 mic 坐标系），定长 n_out（不足补零）。

    非有限 delay_ms（GCC 不可用时的 NaN）会被显式拒绝：先前的 int(round(nan))
    抛 `ValueError: cannot convert float NaN to integer`，调用方无法区分"参数错误"
    与"上游数值故障"。调用方应先判 `np.isfinite(delay_ms)` 再决定对齐策略。
    """
    if not np.isfinite(delay_ms):
        raise ValueError(
            f"align_far 的 delay_ms 必须有限，收到 {delay_ms!r}；"
            "这通常表示 GCC-PHAT 上游给出了不可用结果，调用方应先做非有限判别")
    d = max(int(round(delay_ms * sr / 1000)), 0)   # 负对齐无物理意义（far 早于 mic）
    shifted = np.concatenate([np.zeros(d), far])[:n_out]
    if len(shifted) < n_out:
        shifted = np.concatenate([shifted, np.zeros(n_out - len(shifted))])
    return shifted
