# exp/05c_cpp_demo — C++ 单帧/整段 ONNX 推理 demo（WP-05c 选做）

对标车载 DSP/嵌入式岗的差异化加分项：**零 Python 依赖的纯 C++ 前向**，
GTCRN ONNX 模型（WP-05a 产物）直接吃 wav 出 wav。

## 验证结果（2026-10-09，MSVC 2022 + onnxruntime 1.17.1）

| 项 | 结果 |
|---|---|
| 构建 | cmake + VS2022 生成器通过（onnxruntime C++ 运行时 61MB 下载解压） |
| 端到端一致性 | **C++ vs Python(ONNXRuntime) 增强输出 max diff 3.4e-5**（float32 FFT 舍入量级） |
| RTF（4s 音频整段，单线程） | **0.0094**（37.5ms / 4.0s） |
| 模型 | GTCRN 23.67K 参数，256KB 级权重 |

## 自实现清单（无第三方 DSP 库，全部对齐 torch 约定）

- radix-2 FFT/逆 FFT（float 精度）
- STFT：n_fft=512 / hop=256 / **周期 sqrt-hann 窗** / center=True 两侧各 256
  **reflect 镜像填充**（torch.stft 默认——踩坑：零填充会让首帧错位、全链偏移）
- ISTFT：sqrt-hann WOLA（w² 归一化）+ center 裁剪
- WAV 读写（PCM16 单声道）
- 张量布局：**F-major [1,257,T,2]**（踩坑：T-major 直喂模型 = 转置错误）

## 构建与运行

```bash
# 前置：onnxruntime C++ 运行时
#   https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-win-x64-1.17.1.zip
#   解压到 D:/data/models/onnxruntime-win/（CMakeLists 的 ORT_ROOT 可覆盖）
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
./build/Release/gtcrn_demo.exe <model.onnx> <in.wav> <out.wav>      # 16k PCM16 单声道
```

## 与批次内其他模块的关系
- 模型 = WP-05a `outputs/gtcrn_dns3.sim.onnx`（对拍 6.3e-6 的那份）
- 与 WP-05b（Python 流式 cache）互为补充：本 demo 证明**部署层**（C++ 前向 +
  定点/嵌入式迁移路径），WP-05b 证明**流式层**（逐帧 cache 等价性）
- 面试叙事：「STFT 约定、内存布局、跨实现对拍」三个坑都有实测数据背书
