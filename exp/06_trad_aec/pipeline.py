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
"""
import sys
from dataclasses import dataclass
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
        far_al = align_far(far, delay_ms, self.sr, n)
        n_blocks = int(np.ceil(n / self.block))
        base_scale = self._far_gate(far_al, n_blocks)

        flags = np.zeros(n_blocks)
        if self.method == "mdf":
            af = MDF(block=self.block, partitions=self.partitions, mu=self.mu)
            if self.use_dtd:
                dtds = DTDStreaming(frame=self.block, sr=self.sr, echo_prior=echo_prior)

                def hook(bi, fb, db, y):
                    st = dtds.update(db, fb, y)      # 0=自适应 1=DT软冻结 2=近端硬冻结
                    flags[bi] = float(st > 0)
                    if st == 0:
                        return base_scale[bi]
                    return FREEZE if st == 1 else 0.0

                e, y = af.process(far_al, mic, block_hook=hook, return_y=True)
            else:
                e, y = af.process(far_al, mic, adapt_scale=base_scale, return_y=True)
        elif self.method == "nlms":
            if self.use_dtd:
                import warnings
                # 不补 stacklevel（会改变警告归属的调用栈行号），保持基线行为，豁免 B028
                warnings.warn("NLMS 分支不支持在线 DTD 冻结（基线用途），按静态远端门运行")  # noqa: B028
            af = NLMS(length=self.nlms_len, mu=self.mu)
            scale_ps = np.repeat(base_scale, self.block)[:n]
            e, y = af.process(far_al, mic, adapt_scale=scale_ps, return_y=True)
        else:
            raise ValueError(f"unknown method {self.method}")

        gate = 1.0 - np.repeat(flags, self.block)[:n] if self.use_dtd else None
        out = residual_suppress(e, y, gate=gate, sr=self.sr,
                                alpha=self.res_alpha, g_min=self.res_gmin) \
            if self.use_res else e
        return AECResult(error=e[:n], out=out[:n], y_echo=y[:n], delay_ms=delay_ms,
                         dt_frac=float(flags.mean()), method=self.method,
                         evidence=evidence)
