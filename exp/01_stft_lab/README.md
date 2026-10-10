# exp/01_stft_lab — 模块1 实验台（WP-01）

零数据依赖，全部可独立运行（conda env：avsuite）：

```bash
cd exp/01_stft_lab
python visualize.py                 # 合成信号四联图；--wav your.wav 用真实语音
python window_ablation.py           # 窗长(20/32/64ms)×窗型消融表 + 起振拖尾图
python demo_zh.py                   # 共振峰合成 /a-i-u/ -> 加噪 -> 谱减 -> 指标
python demo_zh.py --wav 原文.wav    # 用自己的中文语音跑同一链路
```

## 各脚本看什么

| 脚本 | 产物 | 核心观察点 |
|---|---|---|
| `visualize.py` | `outputs/stft_viz.png` | 幅度/相位/实部/虚部四联图：相位呈卷绕条纹状（±π 跳变），实/虚部是连续干涉纹——这就是"回归 Re/Im 而不回归相位"的直观理由 |
| `window_ablation.py` | stdout 表 + `outputs/window_ablation.png` | 主瓣宽 ∝ 1/窗长；blackman@50% COLA 偏差 0.38 但 WOLA 往返仍 1e-15 精确；1kHz 起振沿拖尾 ≈ 窗长 |
| `demo_zh.py` | 三个 wav + `outputs/demo_spectrograms.png` | 谱减后 SI-SNR +9dB 而 STOI 反降（音乐噪声）——经典方法的玻璃天花板 |

## 与 INTERVIEW 的连接
- 四联图 → 相位卷绕追问的板书素材
- 消融表 → "为什么车载常用 20ms 帧移 + 32ms 窗"的量化依据
- demo 指标 → "指标为什么必须成组看"（SI-SNR/STOI/PESQ 会打架）
