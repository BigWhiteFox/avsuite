"""频域残余回声抑制 RES（WP-06）：自适应滤波消不干净的部分用谱减兜底。

增益：G(f,t) = clip(1 − α·|Ŷ_echo|/|E|, g_min, 1)，作用在误差谱上、相位不变。
- Ŷ_echo 用自适应滤波器的回声估计 y（pipeline 传入）——它没有收敛的部分
  正是残余的主体，α 过减因子补偿估计误差；
- **DTD 门控**：双讲/近端帧 G 强制 1（近端语音不受损——验收红线），
  只在远端单讲帧启用抑制；
- g_min 是地板（-20dB），留一点底噪避免"死寂"听感。
"""
import numpy as np
from scipy.signal import istft, stft


def residual_suppress(error: np.ndarray, y_echo: np.ndarray, gate: np.ndarray = None,
                      sr: int = 16000, n_fft: int = 512, hop: int = 256,
                      alpha: float = 1.5, g_min: float = 0.1) -> np.ndarray:
    """gate: 与 error 等长的 0/1 样本门（1=允许抑制，即非双讲段）；None=全程启用。

    长度约束（V5 审计 G14 引入；V5 终审 **F4 更正口径**）：真实约束是 scipy 的
    `noverlap < nperseg`。本实现传 `noverlap = n_fft - hop`（默认 256），而 scipy 在
    `len(x) < nperseg` 时会把 nperseg 降到 len(x)，使 noverlap 非法 ⇒ **实测失效边界是
    `len(x) <= n_fft - hop`（默认 256）**，而非早先写的 `len(x) <= n_fft`（512）。
    更正依据（HEAD 版实测扫描，见 tests/test_exp06_wave5.py 的边界用例）：
      - n ≤ 256：`ValueError: noverlap must be less than nperseg`（scipy 侧，真实根因）；
      - 257 ≤ n ≤ 511：HEAD 下报 `operands could not be broadcast`——这是**另一处**
        istft 形状不匹配（scipy 降了 nperseg 而 istft 仍按 n_fft 调），与 scipy 无关；
      - n ≥ 512：成功 ⇒ 旧守卫 `len <= n_fft` 会误拒 HEAD 本可用的 n=512。
    本版取"真正扩大可用面"的修法：守卫放到 `n > n_fft - hop`，并把**短于 n_fft 的输入
    零补到 n_fft** 再走定窗路径（补齐区不抑制），使 257..511 也可用，且输出长度恒等于输入。
    """
    n_in = error.shape[0]
    if n_in <= n_fft - hop:
        raise ValueError(
            f"residual_suppress 需要输入长度 > n_fft - hop = {n_fft - hop}"
            f"（至少 {n_fft - hop + 1} 个样本），收到 {n_in} 个样本："
            f"scipy 定窗 STFT 的 noverlap={n_fft - hop} 必须小于有效 nperseg，"
            "而 len(x) < nperseg 时 scipy 会把 nperseg 降到 len(x)（原始异常看不出根因）；"
            "短音频请先零填充补齐，或用 use_res=False 跳过后置滤波。")
    pad = max(0, n_fft - n_in)
    if pad:                                    # 短输入零补到 n_fft（定窗路径恒定）
        error = np.pad(error, (0, pad))
        y_echo = np.pad(y_echo, (0, pad))
        if gate is not None:
            gate = np.pad(gate, (0, pad))      # 补齐区不抑制（gate=0 → G=1 直通）
    _, _, E = stft(error, sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    _, _, Y = stft(y_echo, sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="zeros")
    G = np.clip(1.0 - alpha * np.abs(Y) / (np.abs(E) + 1e-10), g_min, 1.0)
    if gate is not None:
        fr = hop
        n_frames = min(len(gate) // fr, E.shape[1])
        fgate = np.array([1.0 if gate[i * fr:(i + 1) * fr].mean() >= 0.5 else 0.0
                          for i in range(n_frames)])
        if n_frames < E.shape[1]:                       # 尾帧保守：直通
            fgate = np.concatenate([fgate, np.zeros(E.shape[1] - n_frames)])
        G = fgate[None, :] * G + (1.0 - fgate[None, :])  # 不允许抑制处 G=1（直通）
    _, out = istft(E * G, sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=True)
    return out[:n_in]
