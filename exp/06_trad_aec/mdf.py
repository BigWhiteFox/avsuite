"""MDF / PBFDAF 分块频域自适应滤波（WP-06）。

Overlap-save 分块卷积实现（Valin, "On Adjusting the Learning Rate in
Frequency Domain Echo Cancellation With Double-Talk" 的基础版）：

- 滤波器总长 L = partitions × B（B=块长）；每块处理 B 个新样本
- 频域保存最近 P 个 2B 点频谱 X[p]（X[0] 最新），滤波输出
  y = Σ_p IFFT(W[p]·X[p])[B:]                     （丢前 B 个循环卷绕样本）
- 误差频谱 E = FFT([0_B, e_B])，梯度/更新
  W[p] += μ·conj(X[p])·E / (P·ph + ε)             （ph=当前帧平滑功率谱）
- 约束步：更新后 IFFT 取前 B 个 tap 再补零 FFT——保证时域滤波器长度不越界

对照 NLMS 的收益：FFT 批处理 O(L log L)/块（NLMS 是 O(L)/样本）+
逐频点功率归一（语音有色谱收敛更均匀）。
"""
import numpy as np


class MDF:
    def __init__(self, block: int = 256, partitions: int = 8, mu: float = 0.5,
                 power_ema: float = 0.97, power_floor: float = 0.5,
                 rel_cap: float = 1.0, sig_cap: float = 0.5, leak: float = 1e-4,
                 loading: float = 0.05):
        self.B, self.P, self.mu = block, partitions, mu
        self.N = 2 * block
        self.W = np.zeros((partitions, self.N // 2 + 1), dtype=complex)
        self.X = np.zeros((partitions, self.N // 2 + 1), dtype=complex)
        self.xhist = np.zeros(self.N)               # 最近 N 个输入样本
        self.span_ph = None                         # 跨度功率慢速 EMA（见 process）
        self.ema = power_ema                        # 0.97 ≈ 0.5s 窗，跨语音停顿
        self.pfloor = power_floor                   # 爆发点下限系数（×瞬时 span）
        self.rel_cap, self.sig_cap = rel_cap, sig_cap   # 信赖域（相对/信号标度）
        self.leak = leak                            # 系数泄漏（界住远端静默期随机游走）
        self.loading = loading                      # 对角加载（深谷 bin 梯度限幅）

    @property
    def taps(self) -> np.ndarray:
        """当前等效时域滤波器（L = P·B taps），测试/可视化用。"""
        h = np.concatenate([np.fft.irfft(self.W[p], self.N)[:self.B]
                            for p in range(self.P)])
        return h

    def process(self, far: np.ndarray, mic: np.ndarray,
                adapt_scale: "np.ndarray | float" = 1.0, return_y: bool = False,
                block_hook=None):
        """整段处理（内部按 B 分块）。far/mic 等长。

        adapt_scale：标量或逐块数组（静态缩放）。
        block_hook：fn(bi, far_block, mic_block, y_block) -> float，逐块动态步长
            （DTDStreaming 在此接入：拿到本块回声估计 y 后决定冻结与否）。
        return_y=True 时同时返回回声估计 y（RES 后滤用）。
        """
        B, N = self.B, self.N
        if len(far) != len(mic):
            raise ValueError("far/mic must be equal length")
        n_blocks = int(np.ceil(len(far) / B))
        scales = (np.broadcast_to(adapt_scale, (n_blocks,)) if isinstance(adapt_scale, np.ndarray)
                  else np.full(n_blocks, float(adapt_scale))).astype(float)
        pad = (-len(far)) % B
        far_p = np.concatenate([far, np.zeros(pad)])
        mic_p = np.concatenate([mic, np.zeros(pad)])
        e, y_all = np.empty_like(far_p), np.empty_like(far_p)
        for bi, i in enumerate(range(0, len(far_p), B)):
            fb, db = far_p[i:i + B], mic_p[i:i + B]
            self.xhist = np.concatenate([self.xhist[B:], fb])      # 滑窗（尾为新）
            X_new = np.fft.rfft(self.xhist)
            self.X[1:] = self.X[:-1]                               # 谱环形队列：X[0] 最新
            self.X[0] = X_new
            y = np.zeros(B)
            for p in range(self.P):
                y += np.fft.irfft(self.W[p] * self.X[p], N)[B:]
            eb = db - y
            e[i:i + B], y_all[i:i + B] = eb, y
            scale = float(scales[bi]) if block_hook is None else float(block_hook(bi, fb, db, y))
            E = np.fft.rfft(np.concatenate([np.zeros(B), eb]))
            # 归一化 = 跨度总功率 Σ_p|X[p]|² 的慢速 EMA（Mansour-Gray PBFDAF 期望功率版）
            # + 0.5·瞬时 span 下限：语音爆发点 span_ph 还停在静音电平，无下限时
            # 步长放大 P 倍直接发散（实测 ERLE -87dB）。
            span = np.sum(np.abs(self.X) ** 2, axis=0)
            self.span_ph = span.copy() if self.span_ph is None else \
                self.ema * self.span_ph + (1 - self.ema) * span
            # 归一化：max(慢EMA跨度功率, 0.5·瞬时span, 5%·平均跨度功率)。
            # 第三项是 max 形式的对角加载：只托底深谷 bin（防 |E|/nrm 爆炸），
            # 不拖累安静但有效的 bin——加性 loading 会把语音高频段（天然低于
            # 均值 20-30dB）的步长压小几十倍，实测收敛比 NLMS 慢 6 倍、稳态低 5dB。
            nrm = np.maximum(np.maximum(self.span_ph, self.pfloor * span),
                             self.loading * self.span_ph.mean()) + 1e-10
            # 信赖域：每块每分区系数变化 ≤ max(rel·||w_p||, sig·sqrt(B)·rms(当前块))
            # 信号标度的绝对上限：冷启动(w=0)不冻结收敛，病态归一化下也不发散
            cap_abs = self.sig_cap * np.sqrt(B) * (np.sqrt(np.mean(db ** 2)) + 1e-10)
            for p in range(self.P):
                G = np.conj(self.X[p]) * E / nrm
                dw = np.fft.irfft(self.mu * scale * G, N)
                dw[B:] = 0                                   # 梯度约束（时域前 B 有效）
                w = np.fft.irfft(self.W[p], N)
                cap = max(self.rel_cap * np.linalg.norm(w[:B]), cap_abs)
                g_norm = np.linalg.norm(dw[:B])
                if g_norm > cap:
                    dw[:B] *= cap / g_norm
                w[:B] = (w[:B] + dw[:B]) * (1 - self.leak)   # 更新 + 泄漏
                w[B:] = 0                                    # 系数约束
                self.W[p] = np.fft.rfft(w)
        e, y_all = e[:len(far)], y_all[:len(far)]
        return (e, y_all) if return_y else e

    def reset(self):
        self.W[:] = 0
        self.X[:] = 0
        self.xhist[:] = 0
        self.span_ph = None
