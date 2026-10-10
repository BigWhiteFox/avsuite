"""双讲检测 DTD（WP-06）：Geigel 能量比 + 频域相干性 双判据。

判据 1（Geigel 能量）：假设回声路径衰减 ≥ alpha（0.5≈6dB ERL）；
|mic| 峰值明显超过远端最近历史窗峰值的 alpha 倍 -> 存在近端信号。

判据 2（相干性 MSC）：STFT 域 EMA 平滑谱，逐频点
    γ²(f) = |S_xy|² / (S_xx·S_yy)，取语音频段均值。
纯线性回声（任意长 RIR、任意混响）γ²→1——**时域相关会被混响打散而相干性不会**，
这是选它当第二判据的原因（实测教训：时域相关判据在随机 RIR 下纯 FEST 也误报）。
近端语音与远端不相关 -> γ² 明显跌落。

两判据同时满足才判 DT + hangover 保持。判 DT 时 pipeline 把自适应步长缩到
freeze_scale（软冻结：系数停止大幅更新、滤波照常），防止近端被学进回声路径。
"""
import numpy as np


class DTD:
    def __init__(self, frame: int = 256, alpha: float = 0.5, coh_thr: float = 0.5,
                 hangover: int = 8, ema: float = 0.7, sr: int = 16000):
        self.frame, self.alpha, self.coh_thr = frame, alpha, coh_thr
        self.hangover, self.ema = hangover, ema
        self.win = np.hanning(frame)
        freqs = np.fft.rfftfreq(frame, 1 / sr)
        self.bins = (freqs >= 100) & (freqs <= 3800)

    def decide_frames(self, mic: np.ndarray, far_aligned: np.ndarray) -> np.ndarray:
        """逐帧判决 0/1（1=双讲）。far 需已做时延对齐。返回长度 = n_frames*frame。"""
        fr = self.frame
        n_frames = len(mic) // fr
        flags = np.zeros(n_frames)
        sxx = np.zeros(fr // 2 + 1)
        syy = np.zeros_like(sxx)
        sxy = np.zeros_like(sxx, dtype=complex)
        far_peak_hist = 0.0
        hang = 0
        eps = 1e-12
        for i in range(n_frames):
            x = far_aligned[i * fr:(i + 1) * fr] * self.win
            y = mic[i * fr:(i + 1) * fr] * self.win
            X, Y = np.fft.rfft(x), np.fft.rfft(y)
            b = 1.0 - self.ema
            sxx = self.ema * sxx + b * (X * X.conj()).real
            syy = self.ema * syy + b * (Y * Y.conj()).real
            sxy = self.ema * sxy + b * (X * Y.conj())
            coh = float(np.mean((np.abs(sxy) ** 2 /
                                 (sxx * syy + eps))[self.bins]))
            far_peak_hist = max(far_peak_hist, np.abs(far_aligned[i * fr:(i + 1) * fr]).max())
            geigel = np.abs(y).max() > self.alpha * far_peak_hist + 1e-10
            dt = geigel and coh < self.coh_thr
            hang = self.hangover if dt else max(hang - 1, 0)
            flags[i] = dt or hang > 0
        return flags


class DTDStreaming:
    """流式 DTD（pipeline 用）：后置滤波相干 + Geigel + 收敛闩锁 + 远端活跃门。

    与上面离线版 DTD 的本质区别——相干性算在 **mic vs y_echo**（自适应滤波器的
    回声估计）上，而不是 mic vs far：
    - mic-vs-far 的分段相干被"段内泄漏"压低：帧长 256 样本 << RIR 长（1600+ taps），
      帧内只覆盖 RIR 能量的 ~16% -> 纯 FEST 相干也只有 ~0.2 -> 全被误判双讲
      （实测 dt_frac 0.93-1.00，ERLE 被冻结在 2dB）；
    - y_echo 与 mic 逐块时间对齐、同带宽 -> 无线段泄漏：FEST 收敛后 γ²→1，
      双讲时 γ² ≈ P_echo/P_mic 骤降。这是 WebRTC AEC suppressor 的同款思路。

    收敛闩锁：块级 ERLE 的 EMA 连续 ≥ latch_blocks 块超过 erle_thr 才允许判 DT。
    冷启动 y≈0、γ²≈0，不闩锁会把整个收敛期误判成双讲而永久冻结（鸡生蛋问题）。
    远端活跃门：far 静默块自适应缩到 idle_scale——防 NST 下学到虚假相关
    （随机游走系数会往近端注入垃圾，实测 NST PESQ 2.60→2.23 的元凶）。
    """

    def __init__(self, frame: int = 256, sr: int = 16000, alpha: float = 0.5,
                 coh_thr: float = 0.6, hangover: int = 8, ema: float = 0.7,
                 erle_thr: float = 6.0, latch_blocks: int = 10, far_rel_thr: float = 0.03,
                 warmup_corr_thr: float = 0.1, warm_blocks: int = 8,
                 far_abs_floor: float = 1e-3, corr_min_blocks: int = 6,
                 echo_prior: "bool | None" = None):
        self.frame, self.sr, self.alpha = frame, sr, alpha
        self.coh_thr, self.hangover, self.ema = coh_thr, hangover, ema
        self.erle_thr, self.latch_blocks = erle_thr, latch_blocks
        self.far_rel_thr = far_rel_thr
        self.far_abs_floor = far_abs_floor   # -60dBFS 绝对门：安静 far 文件的"停顿"
                                             # 相对门分不清（rms 0.0003 vs peak 0.005），
                                             # 静音块 corr 分母塌缩产生假回声证据
        self.warmup_corr_thr = warmup_corr_thr
        self.warm_blocks = warm_blocks    # 累积相关不可靠期：默认冻结（FEST 只损失
                                          # ~0.2s 收敛起点；NST 零虚假相关垃圾——
                                          # 实测暖机掷硬币期全速学习会让 NST 近端
                                          # PESQ 永久掉 0.3-0.6）
        self.nb = 0
        self.n_active = 0                 # 远端活跃块累积计数（corr 可信度门槛）
        self.corr_min_blocks = corr_min_blocks
        self.last_state = 2               # hangover 保持的状态语义
        # echo_prior：GCC 峰旁瓣比的整段回声存在性判决（pipeline 初始化算好传入）。
        # True=FEST/DT（有回声，暖机后放行自适应）；False=NST（无回声，持续硬冻结）；
        # None=退回累积相关判据。累积相关会被混响稀释（corr ≈ 直达径/总能量比，
        # 实测 FEST 仅 0.1-0.3，与 NST 噪声带重叠），峰旁瓣比不受混响影响
        # （实测 FEST 19-84 / DT 15-66 / NST 4.6-13.4，阈值 14 干净可分）。
        self.echo_prior = echo_prior
        self.win = np.hanning(frame)
        freqs = np.fft.rfftfreq(frame, 1 / sr)
        self.bins = (freqs >= 100) & (freqs <= 3800)
        nb = frame // 2 + 1
        self.sxx = np.zeros(nb)
        self.syy = np.zeros(nb)
        self.sxy = np.zeros(nb, dtype=complex)
        self.erle_ema = None
        self.cmxy = 0.0        # mic·far 累积内积 EMA（暖机判据，标量长统计）
        self.cmxx = 0.0
        self.cmyy = 0.0
        self.run = 0
        self.converged = False
        self.far_rms_max = 1e-10
        self.far_peaks = np.zeros(8)
        self.hang = 0
        self.far_active = True

    def update(self, mic_b: np.ndarray, far_b: np.ndarray, y_b: np.ndarray) -> int:
        """喂一个块，返回三态：0=正常自适应，1=DT 软冻结（步长×FREEZE，允许
        慢速跟踪回声路径漂移），2=近端主导硬冻结（步长=0，完全不学——NST 下
        软冻结的随机游走 8s 也能积累出 40% 量级的虚假系数，必须硬冻结）。"""
        eps = 1e-12
        # 1) 远端活跃度：相对滚动峰值 + 绝对下限（数字静音/低电平文件双保护）
        frms = float(np.sqrt(np.mean(far_b ** 2)))
        self.far_rms_max = max(self.far_rms_max * 0.9995, frms, 1e-10)
        self.far_active = (frms > self.far_rel_thr * self.far_rms_max
                           and frms > self.far_abs_floor)
        self.far_peaks = np.roll(self.far_peaks, 1)
        self.far_peaks[0] = np.abs(far_b).max()
        # 2) mic-vs-far 累积相关（标量 EMA 内积，无分段泄漏）——"有没有回声"的证据。
        # FEST: mic=conv(far,rir) -> 秒级累积相关 0.3+；NST: 独立信号 -> ±1/sqrt(N)≈0.01。
        # 块内相关不可用：100ms RIR 跨 6+ 个 256 块，单块相关被稀释到 0.05-0.7。
        # 仅远端活跃块累积：静默块 far@far≈0 会把 cmyy 拉塌，corr 变数值垃圾
        # （实测安静文件停顿处 corr 冲到 0.2 -> 14 个块误开全速自适应 ->
        # 垃圾系数毒化整段近端，冻结只能保留垃圾不能清除）。
        if self.far_active:
            cw = 0.995
            self.cmxy = cw * self.cmxy + float(mic_b @ far_b)
            self.cmxx = cw * self.cmxx + float(mic_b @ mic_b)
            self.cmyy = cw * self.cmyy + float(far_b @ far_b)
            self.n_active += 1
        corr_cum = abs(self.cmxy) / (np.sqrt(self.cmxx * self.cmyy) + eps)
        # 累积计数不足时 corr 是少数块的随机数（安静文件可抖到 0.2+）：不可信 -> 冻结
        corr_reliable = self.n_active >= self.corr_min_blocks
        # 3) 收敛闩锁：块级 ERLE EMA + 回声证据门槛。
        # 门槛必不可少：NST 下虚假相关会"消掉"部分近端 -> ERLE 假性 >6dB ->
        # 假收敛 -> 相干判据被绕过 -> 近端被全速吃掉（实测 NST PESQ 2.45->1.71）。
        e_b = mic_b - y_b
        erle_inst = 10 * np.log10((np.mean(mic_b ** 2) + eps) / (np.mean(e_b ** 2) + eps))
        if self.erle_ema is None:
            self.erle_ema = erle_inst
        else:
            self.erle_ema = 0.9 * self.erle_ema + 0.1 * erle_inst
        if self.erle_ema > self.erle_thr:
            self.run += 1
        else:
            self.run = max(self.run - 1, 0)
        if self.run >= self.latch_blocks and (self.echo_prior or
                                              corr_cum >= self.warmup_corr_thr):
            self.converged = True
        # 4) 远端静默 -> 硬冻结：无可学习，且谱队列 X[p] 里的旧 far 会与近端误差
        # 形成虚假梯度（也消除与 pipeline 远端门窗口不一致造成的漏判）
        if not self.far_active:
            self.nb += 1
            return 2
        # 5) mic-vs-y 相干（EMA 谱平滑）
        X = np.fft.rfft(mic_b * self.win)
        Y = np.fft.rfft(y_b * self.win)
        b = 1.0 - self.ema
        self.sxx = self.ema * self.sxx + b * (X * X.conj()).real
        self.syy = self.ema * self.syy + b * (Y * Y.conj()).real
        self.sxy = self.ema * self.sxy + b * (X * Y.conj())
        coh = float(np.mean((np.abs(self.sxy) ** 2 / (self.sxx * self.syy + eps))[self.bins]))
        # 6) Geigel：mic 峰值 vs 滚动远端峰值窗
        geigel = np.abs(mic_b).max() > self.alpha * self.far_peaks.max() + 1e-10
        # 7) 状态机（hangover 保持原状态语义：硬冻结不被软化为 DT 软冻结——
        # 实测软化后 0.02 步长仍会经深谷梯度注入垃圾毒化近端）
        if self.converged:
            state = 1 if (self.far_active and geigel and coh < self.coh_thr) else 0
        elif self.nb < self.warm_blocks:
            state = 2
        elif self.echo_prior is True:
            state = 0                       # 整段判定有回声：暖机后放行自适应
        elif self.echo_prior is False:
            state = 2                       # 整段判定无回声（NST）：持续硬冻结
        else:
            state = 2 if (not corr_reliable or
                          (self.far_active and corr_cum < self.warmup_corr_thr)) else 0
        self.nb += 1
        if state:
            self.hang = self.hangover
            self.last_state = state
        else:
            self.hang = max(self.hang - 1, 0)
        return state if not self.hang or state else self.last_state
