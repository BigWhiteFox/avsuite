"""时域 NLMS 自适应滤波器（WP-06）。

w[n+1] = w[n] + μ·e[n]·x[n] / (xᵀx + ε)，每样本归一化步长——对信号功率变化稳健。
echo = wᵀx 消除后输出 e = mic − echo_est。
时间复杂度 O(L) 每样本：长滤波器（>4000 taps）慢，收敛也慢于频域方法
（MDF）——这正是车载 DSP 用 MDF 的原因，消融曲线会展示这个差距。
"""
import numpy as np


class NLMS:
    def __init__(self, length: int = 2048, mu: float = 0.5, leak: float = 1e-5):
        if not 0 < mu <= 2:
            raise ValueError(f"NLMS mu 建议在 (0,2]，got {mu}")
        self.L, self.mu, self.leak = length, mu, leak
        self.w = np.zeros(length)
        self.xbuf = np.zeros(length)
        self.x2_ema = None          # 输入功率慢 EMA（静音期归一化下限，防爆步长）

    def reset(self):
        self.w[:] = 0
        self.xbuf[:] = 0
        self.x2_ema = None

    def process(self, far: np.ndarray, mic: np.ndarray,
                adapt_scale: "np.ndarray | float" = 1.0, return_y: bool = False):
        """逐样本滤波+（可选）自适应。adapt_scale：0~1，DTD/远端静默时压低（软冻结）。

        归一化分母 = max(||x||², 0.25·功率EMA)：远端静默时 ||x||²→0，若无下限
        步长 ∝ 1/||x|| 爆炸（NST 下系数随机游走、往近端注垃圾的元凶）。
        """
        e = np.empty_like(mic)
        y_all = np.empty_like(mic)
        scale = np.broadcast_to(adapt_scale, mic.shape) if isinstance(adapt_scale, np.ndarray) \
            else np.full(len(mic), float(adapt_scale))
        decay = 1.0 - self.leak
        for n in range(len(mic)):
            self.xbuf[1:] = self.xbuf[:-1]           # 环形缓冲（新样本在 xbuf[0]）
            self.xbuf[0] = far[n]
            x2 = float(self.xbuf @ self.xbuf)
            self.x2_ema = x2 if self.x2_ema is None else 0.999 * self.x2_ema + 0.001 * x2
            y = float(self.w @ self.xbuf)
            e[n], y_all[n] = mic[n] - y, y
            p = max(x2, 0.25 * self.x2_ema) + 1e-10
            self.w *= decay
            self.w += (self.mu * scale[n]) * e[n] / p * self.xbuf
        return (e, y_all) if return_y else e
