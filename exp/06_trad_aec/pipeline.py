"""传统 AEC 全链路 pipeline（WP-06）。

    far ─GCC-PHAT 时延估计─> 对齐 ─┐
                                   ├─自适应滤波(MDF/NLMS)─> e ─RES(DTD门控)─> out
    mic ──────────────────────────┘

单 pass 在线结构（WebRTC AEC 同款状态机思路）：
- MDF 逐块回调 block_hook(bi, far块, mic块, y块)：DTDStreaming 拿到**本块回声估计 y**
  算 mic-vs-y 相干（无分段泄漏问题）+ Geigel + 收敛闩锁，返回该块步长缩放；
- 远端活跃门（静态可算，MDF/NLMS 通用）：far 静默块步长×0.05——NST 下防止
  滤波器对不相关近端做随机游走（系数泄漏 leak 双保险）；
- 冻结只冻"学习"不冻"滤波"：已有系数继续消回声，近端不被学掉；
- RES 用 DTD 帧门控：双讲/近端帧直通（验收红线），远端单讲帧做 Wiener 残余抑制。

NLMS 分支为基线/消融用（无在线 DTD 冻结，use_dtd 时告警降级）——完整链路走 MDF。

V4 整改新增（2026-10-11，默认参数逐位兼容，见 reports/v4_rectification/）：
- align_margin_ms：对齐余量——对齐右移量 = max(估计时延 − margin, 0)，使残余直达径
  落在 MDF 滤波器窗口内部而非 lag 0。修复"负向时延漂移/路径移动 → 残余路径非因果
  → 任何 FIR 不可表示 → 回声泄漏崩塌"的实现缺陷（原实现把直达径放在窗口边缘，
  漂移集 FEST −1.89、官方集 with-movement 条发散的同族根因）；
- reestimate_sec：周期性窗ed GCC 重估 + 再对齐（超 realign_min_delta_ms 才动作，
  动作时复位 MDF 与 DTD 自适应状态）；配合 realign_search_pad_ms 的搜索余量；
- mdf_w_abs_cap / 发散防护在 mdf.MDF 内（默认开）：输出恒有限 + n_divergence_resets
  计数；DTDStreaming.reset() 供再对齐时清闩锁/EMA。
- AECResult 新增 n_realigns / n_divergence_resets / delay_track / align_ms 字段。
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))               # 同目录模块

from dtd import DTDStreaming  # noqa: E402
from gcc import align_far, gcc_phat_stats  # noqa: E402
from mdf import MDF  # noqa: E402
from nlms import NLMS  # noqa: E402
from res import residual_suppress  # noqa: E402

FREEZE = 0.02          # 双讲块步长缩放（软冻结，允许慢速跟踪路径变化）
IDLE = 0.0             # 远端静默块硬冻结：谱队列 X[p] 还留着旧 far 数据，
                       # 慢速自适应会把近端学成虚假相关（实测 NST y/mic=0.17、
                       # PESQ 2.30->1.73 的元凶）；静默期本来就无可学习
FAR_REL_THR = 0.03     # 远端活跃判限（相对滚动峰值 rms，≈-30dB：语音音素级动态 20dB+，0.15 会把清音全冻结）
FAR_ABS_THR = 1e-3     # 远端活跃绝对下限（-60dBFS）
EVIDENCE_THR = 14.0    # GCC 峰旁瓣比阈值：回声存在性整段判决
                       # （实测 FEST 19-84 / DT 15-66 / NST 4.6-13.4）


@dataclass
class AECResult:
    error: np.ndarray          # 自适应滤波输出（线性 AEC 后）
    out: np.ndarray            # 最终输出（use_res=False 时 == error）
    y_echo: np.ndarray         # 自适应滤波器的回声估计
    delay_ms: float
    dt_frac: float             # 双讲块占比
    method: str
    evidence: float = 0.0      # GCC 峰旁瓣比（回声存在性证据，阈值 14）
    align_ms: float = 0.0      # V4：实际对齐右移量（= max(delay−margin, 0)）
    n_realigns: int = 0        # V4：周期重估触发的再对齐次数
    n_divergence_resets: int = 0  # V4：MDF 发散防护复位次数
    delay_track: "list" = field(default_factory=list)  # V4：(t_s, est_ms, realigned) 记录


@dataclass
class TradAEC:
    method: str = "mdf"                # 'mdf' | 'nlms'
    mu: float = 0.7
    block: int = 256
    partitions: int = 12               # mdf: 滤波器长 = 12*256 = 3072 taps (~192ms)
    nlms_len: int = 3072
    use_dtd: bool = True
    use_res: bool = True
    res_alpha: float = 1.5
    res_gmin: float = 0.1
    sr: int = 16000
    # ---- V4 整改新增（默认值保持原行为逐位兼容）----
    align_margin_ms: float = 0.0       # 对齐余量：把直达径放进滤波器窗口内部
    reestimate_sec: "float | None" = None  # 周期重估周期（None=关闭，原单次估计行为）
    realign_min_delta_ms: float = 5.0  # 重估变化小于此值不动（防语音假峰抖动）
    reestimate_win_s: float = 2.0      # 重估窗口（最近 N 秒原始 mic/far）
    realign_search_pad_ms: float = 150.0   # 重估搜索上限 = 当前基准 + pad（容大跳变）
    mdf_w_abs_cap: float = 1e3         # MDF 系数谱域绝对界（发散防护，见 mdf.py）

    def _far_gate(self, far_al: np.ndarray, n_blocks: int) -> np.ndarray:
        """远端活跃门（逐块 1.0/IDLE）：rms 相对最近 1s 滚动峰值。"""
        B = self.block
        rms = np.array([np.sqrt(np.mean(far_al[i * B:(i + 1) * B] ** 2) + 1e-20)
                        for i in range(n_blocks)])
        W = max(int(1.0 * self.sr / B), 1)          # 1 秒窗
        gate = np.ones(n_blocks)
        for i in range(n_blocks):
            peak = rms[max(0, i - W):i + 1].max()
            if rms[i] <= max(FAR_REL_THR * peak, FAR_ABS_THR):
                gate[i] = IDLE
        return gate

    def process(self, mic: np.ndarray, far: np.ndarray) -> AECResult:
        mic = np.asarray(mic, dtype=np.float64)
        far = np.asarray(far, dtype=np.float64)
        n = len(mic)
        # 初始化判决（在线系统对应呼叫建立期）：时延 + 回声存在性证据
        delay_ms, evidence = gcc_phat_stats(mic, far, self.sr)
        echo_prior = evidence >= EVIDENCE_THR
        # V4：对齐右移量留出余量，使残余直达径落在滤波器窗口内部（margin 处）
        align_ms = max(delay_ms - self.align_margin_ms, 0.0)
        far_al = align_far(far, align_ms, self.sr, n)
        n_blocks = int(np.ceil(n / self.block))
        flags = np.zeros(n_blocks)
        n_realigns = 0
        delay_track = [(0.0, float(delay_ms), False)]

        if self.method == "mdf":
            af = MDF(block=self.block, partitions=self.partitions, mu=self.mu,
                     w_abs_cap=self.mdf_w_abs_cap)
            dtds = DTDStreaming(frame=self.block, sr=self.sr,
                                echo_prior=echo_prior) if self.use_dtd else None
            if not self.reestimate_sec:                       # 原路径（逐位兼容）
                base_scale = self._far_gate(far_al, n_blocks)
                if self.use_dtd:
                    def hook(bi, fb, db, y):
                        st = dtds.update(db, fb, y)      # 0=自适应 1=DT软冻结 2=近端硬冻结
                        flags[bi] = float(st > 0)
                        if st == 0:
                            return base_scale[bi]
                        return FREEZE if st == 1 else 0.0

                    e, y = af.process(far_al, mic, block_hook=hook, return_y=True)
                else:
                    e, y = af.process(far_al, mic, adapt_scale=base_scale, return_y=True)
            else:                                             # V4：分段 + 周期重估再对齐
                seg = max(int(round(self.reestimate_sec * self.sr)), self.block)
                e = np.empty(n)
                y = np.empty(n)
                base = float(delay_ms)          # 当前对齐基准（未减 margin 的估计）
                b_off = 0                       # 全局块偏移（flags/dt_frac 用）
                pos = 0
                while pos < n:
                    end = min(pos + seg, n)
                    mic_seg = mic[pos:end]
                    if pos > 0:
                        # 窗口重估：最近 reestimate_win_s 的原始 mic/far（未对齐坐标）
                        w0 = max(0, pos - int(self.reestimate_win_s * self.sr))
                        est, _ = gcc_phat_stats(mic[w0:pos], far[w0:pos], self.sr,
                                                max_delay_ms=base + self.realign_search_pad_ms)
                        realigned = abs(est - base) >= self.realign_min_delta_ms
                        if realigned:
                            base = float(est)
                            far_al = np.concatenate([
                                far_al[:pos],
                                align_far(far[pos:], max(base - self.align_margin_ms, 0.0),
                                          self.sr, n - pos)])
                            af.reset()            # 系数/谱队列与新对齐坐标系不一致，必须复位
                            if dtds is not None:
                                dtds.reset()      # 清闩锁/EMA，暖机期重新冻结
                            n_realigns += 1
                        delay_track.append((round(pos / self.sr, 3), float(est), realigned))
                    L = end - pos
                    nb_seg = int(np.ceil(L / self.block))
                    g = self._far_gate(far_al[pos:pos + L], nb_seg)
                    if self.use_dtd:
                        def hook(bi, fb, db, yb, _g=g, _off=b_off):
                            st = dtds.update(db, fb, yb)
                            flags[_off + bi] = float(st > 0)
                            if st == 0:
                                return _g[bi]
                            return FREEZE if st == 1 else 0.0

                        e_seg, y_seg = af.process(far_al[pos:pos + L], mic_seg,
                                                  block_hook=hook, return_y=True)
                    else:
                        e_seg, y_seg = af.process(far_al[pos:pos + L], mic_seg,
                                                  adapt_scale=g, return_y=True)
                    e[pos:pos + L] = e_seg[:L]
                    y[pos:pos + L] = y_seg[:L]
                    b_off += nb_seg
                    pos = end
        elif self.method == "nlms":
            if self.use_dtd:
                import warnings
                # 不补 stacklevel（会改变警告归属的调用栈行号），保持基线行为，豁免 B028
                warnings.warn("NLMS 分支不支持在线 DTD 冻结（基线用途），按静态远端门运行")  # noqa: B028
            af = NLMS(length=self.nlms_len, mu=self.mu)
            scale_ps = np.repeat(self._far_gate(far_al, n_blocks), self.block)[:n]
            e, y = af.process(far_al, mic, adapt_scale=scale_ps, return_y=True)
        else:
            raise ValueError(f"unknown method: {self.method}")

        gate = 1.0 - np.repeat(flags, self.block)[:n] if self.use_dtd else None
        out = residual_suppress(e, y, gate=gate, sr=self.sr,
                                alpha=self.res_alpha, g_min=self.res_gmin) \
            if self.use_res else e
        n_div = int(getattr(af, "n_divergence_resets", 0))
        return AECResult(error=e[:n], out=out[:n], y_echo=y[:n], delay_ms=delay_ms,
                         dt_frac=float(flags.mean()), method=self.method,
                         evidence=evidence, align_ms=float(align_ms),
                         n_realigns=n_realigns, n_divergence_resets=n_div,
                         delay_track=delay_track)
