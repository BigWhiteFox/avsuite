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
    """gate: 与 error 等长的 0/1 样本门（1=允许抑制，即非双讲段）；None=全程启用。"""
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
    return out[:len(error)]
