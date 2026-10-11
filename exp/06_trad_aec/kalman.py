"""频域块 Kalman AEC(P2-2 整改第 1 级,纯 numpy)。

状态 = 回声路径频域响应 H(p,f)(partitions×bin,与 mdf.MDF 的 W 同一坐标系,
重叠保存分块卷积):每块

    y(f)  = Σ_p IFFT(H[p]·X[p])[B:]                 逐分区时域求和(丢循环卷绕段)
    E(f)  = FFT([0_B, e_B])                          新息(观测 = d − x^H·H)
    K(f)  = P(f)·x*(f) / (x(f)ᵀ·P(f)·x*(f) + R(f))   逐 bin 向量卡尔曼增益
    H(f) ← H(f) + s·K(f)·E(f)
    P(f) ← (I − s·K(f)·x(f)^H)·P(f) + s·Q·I          协方差按块演化(Hermitian 保持)

其中 x(f) = [X(0,f),…,X(P−1,f)] 为该 bin 的分区回归量向量,x* 为其共轭
(观测按 y = u^H·H 写,u = x*,故梯度方向与 MDF 的 conj(X) 一致),s 为本块
步长缩放(DTD 三态经 block_hook 接入,与 MDF 同策略:0=全速 / 1=DT 软冻结 /
2=硬冻结,冻结只冻学习不冻滤波;远端静默门由调用方并入 s)。

噪声口径:R(f) = r·nrm(f),Q = q·I。nrm 与 MDF 完全同款(跨度总功率慢 EMA
+ 0.5×瞬时下限 + 5% 均值托底),量纲 = bin 回声功率,故
- 观测噪声 R 随 bin 信号功率配比(近端+噪声 ≈ r×回声功率的先验),r 无量纲;
- 冷启动 P=I 时 denom ≈ span·(1+r),K ≈ x*/span —— 初始即 NLMS 全步,冷启动
  不卡收敛(与 MDF 的归一化收益一致),收敛后 P 收缩、K 自动变小(稳态误调
  由 Q 控制:Q 大跟踪快但稳态低,Q 小反之)。
更新后做与 MDF 相同的时域约束(每分区 tap 截断到 B),保证滤波器长度不越界。
纯 numpy,无第三方依赖。
"""
import numpy as np


class FDKalman:
    def __init__(self, block: int = 256, partitions: int = 8, q: float = 1e-4,
                 r: float = 1e-2, p0: float = 1.0, power_ema: float = 0.97,
                 power_floor: float = 0.5, loading: float = 0.05):
        self.B, self.P = block, partitions
        self.N = 2 * block
        self.nb = self.N // 2 + 1
        self.q, self.r, self.p0 = float(q), float(r), float(p0)
        self.ema, self.pfloor, self.loading = power_ema, power_floor, loading
        self.H = np.zeros((partitions, self.nb), dtype=complex)
        self.X = np.zeros((partitions, self.nb), dtype=complex)
        self.xhist = np.zeros(self.N)               # 最近 N 个输入样本(尾为新)
        self.span_ph = None                         # 跨度功率慢速 EMA(与 MDF 同)
        self.eye = np.eye(partitions, dtype=complex)
        self.Pc = np.repeat(self.p0 * self.eye[None], self.nb, axis=0)  # (nb,P,P)

    @property
    def taps(self) -> np.ndarray:
        """当前等效时域滤波器(L = P·B taps),测试/可视化用(与 MDF.taps 同)。"""
        return np.concatenate([np.fft.irfft(self.H[p], self.N)[:self.B]
                               for p in range(self.P)])

    def process(self, far: np.ndarray, mic: np.ndarray,
                adapt_scale: "np.ndarray | float" = 1.0, return_y: bool = False,
                block_hook=None):
        """整段处理(内部按 B 分块),接口与 mdf.MDF.process 一致。

        block_hook:fn(bi, far_block, mic_block, y_block) -> float,逐块步长缩放
        (DTDStreaming 在此接入,拿到本块回声估计 y 后决定冻结与否)。
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
            self.xhist = np.concatenate([self.xhist[B:], fb])      # 滑窗(尾为新)
            X_new = np.fft.rfft(self.xhist)
            self.X[1:] = self.X[:-1]                               # 谱环形队列:X[0] 最新
            self.X[0] = X_new
            y = np.zeros(B)
            for p in range(self.P):
                y += np.fft.irfft(self.H[p] * self.X[p], N)[B:]
            eb = db - y
            e[i:i + B], y_all[i:i + B] = eb, y
            scale = float(scales[bi]) if block_hook is None else float(block_hook(bi, fb, db, y))
            E = np.fft.rfft(np.concatenate([np.zeros(B), eb]))
            # nrm 与 MDF 同款:max(慢EMA跨度功率, 0.5×瞬时span, 5%均值) —— 见 mdf.py 注
            span = np.sum(np.abs(self.X) ** 2, axis=0)
            self.span_ph = span.copy() if self.span_ph is None else \
                self.ema * self.span_ph + (1 - self.ema) * span
            nrm = np.maximum(np.maximum(self.span_ph, self.pfloor * span),
                             self.loading * self.span_ph.mean()) + 1e-10
            if scale != 0.0:                                       # 硬冻结:H/P 全不动
                # 观测模型 y = u^H·H,u = conj(x)(LMS 梯度方向是 conj(X),见 mdf.py);
                # 增益 K = P·u/(u^H·P·u + R),P=I 时退化为 x*/(|x|²+R) —— 与 MDF 同向
                u = np.conj(self.X).T                              # (nb, P) = x*
                Pu = np.einsum('fij,fj->fi', self.Pc, u)           # P·u
                Rf = self.r * nrm                                  # (nb,) 观测噪声功率
                denom = np.real(np.einsum('fj,fj->f', u.conj(), Pu)) + Rf
                K = Pu / denom[:, None]                            # (nb, P) 卡尔曼增益
                # P ← (I − s·K·u^H)P + s·Q·I   (u^H·P = conj(P·u),P Hermitian)
                self.Pc -= scale * K[:, :, None] * np.conj(Pu)[:, None, :]
                self.Pc += scale * self.q * self.eye[None]
                self.Pc = 0.5 * (self.Pc + self.Pc.conj().transpose(0, 2, 1))
                # H ← H + s·K·E   (逐 bin:标量新息 × 增益向量)
                self.H += scale * (K * E[:, None]).T
                # 时域约束(与 MDF 同):每分区 tap 截断到前 B 个
                ht = np.fft.irfft(self.H, N, axis=1)
                ht[:, B:] = 0.0
                self.H = np.fft.rfft(ht, axis=1)
        e, y_all = e[:len(far)], y_all[:len(far)]
        return (e, y_all) if return_y else e

    def reset(self):
        self.H[:] = 0
        self.X[:] = 0
        self.xhist[:] = 0
        self.span_ph = None
        self.Pc = np.repeat(self.p0 * self.eye[None], self.nb, axis=0)
