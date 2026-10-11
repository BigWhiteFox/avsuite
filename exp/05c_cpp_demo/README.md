# exp/05c_cpp_demo — C++ 单帧/整段 ONNX 推理 demo（WP-05c 选做）

对标车载 DSP/嵌入式岗的差异化加分项：**零 Python 依赖的纯 C++ 前向**，
GTCRN ONNX 模型（WP-05a 产物）直接吃 wav 出 wav。

## 验证结果（2026-10-09，MSVC 2022 + onnxruntime 1.17.1；2026-10-12 V5 复跑复核）

| 项 | 结果 |
|---|---|
| 构建 | cmake + VS2022 生成器通过（onnxruntime C++ 运行时 61MB 下载解压） |
| STFT **spec 级**对拍（C++ dump vs torch.stft） | **max abs diff 1.87e-05**（float32 radix-2 FFT 舍入；首/末帧仅 1.3e-06/4.9e-06） |
| 端到端一致性 | **C++ vs Python(ONNXRuntime) 增强输出 max diff 3.43e-05** |
| ↳ 该数字的构成（V5 复核） | PCM16 输出量化步长 = 1/32768 = **3.05e-05**；实测 3.43e-05 = 1.12×步长，且"两边都量化到 PCM16 后"= 恰好 1 个步长 ⇒ **差异被输出量化主导，不是算法/索引误差** |
| RTF（4s 音频整段，单线程） | **0.0085~0.0099**（33-39ms / 4.0s） |
| 模型 | GTCRN 23.67K 参数，256KB 级权重 |

> **V5 复核结论（重要，勿再"修"）**：v3 缺口扫描曾报 `main.cpp` reflect 填充"索引偏移 1"
> （建议左 pad 改 `wav[-idx-1]`、右 pad 改 `wav[2N-1-idx]`）。**实测为误报**：
> torch/numpy 的 `reflect` 不重复边界样本，`padded[i] = x[pad-i]`，故现行
> `wav[-idx]` / `wav[2N-2-idx]` **正确**——spec 级 max|Δ| = **7.3e-07**（float32 舍入）；
> 而按建议改成 `-idx-1`/`2N-1-idx` 后首/末帧 max|Δ| 达 **0.243**（那其实是 `symmetric` 填充）。
> 历史 3.4e-5 声明因此也得到解释：它来自 PCM16 输出量化（见上表），与索引无关。

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
#   解压到任意目录（仓库默认 D:/data/models/onnxruntime-win/onnxruntime-win-x64-1.17.1）
#   V5：ORT_ROOT 可按优先级覆盖 —— -DORT_ROOT=<路径> > 环境变量 ONNXRUNTIME_ROOT
#        > 环境变量 ORT_ROOT > 仓库默认路径；路径下找不到 include/onnxruntime_cxx_api.h 时
#        configure 会 FATAL_ERROR 并打印指引（不再静默用错目录）
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
./build/Release/gtcrn_demo.exe <model.onnx> <in.wav> <out.wav>      # 16k PCM16 单声道
# 只要 spec（对拍用）：额外加 --dump-spec，会把 <out.wav>.spec.bin 写成 F-major [1,257,T,2] float32
./build/Release/gtcrn_demo.exe <model.onnx> <in.wav> out.wav --dump-spec
```

- **dll 部署**：V5 起 CMake 在 Windows 上 POST_BUILD 把 `onnxruntime.dll` 拷到可执行文件目录
  （实测本机 `C:\Windows\System32\onnxruntime.dll` 存在同名 dll，仅靠 PATH 会加载到
  版本不一定匹配的系统副本；应用目录优先级更高，拷贝后即确定性使用编译期同版本）。
- **build/ 目录搬过地方**：`cmake -S . -B build` 会报
  `CMakeCache.txt directory ... different than ...`（cache 里记的是旧源码绝对路径）——
  删掉 `build/` 重新 configure 即可（本目录不入库）。
- **诊断**：模型加载失败返回 2、推理失败返回 3、输出张量 dtype/shape 不符返回 4、
  wav 读写失败返回 1，均带中文/英文诊断行（不再静默成功）。

## 与批次内其他模块的关系
- 模型 = WP-05a `outputs/gtcrn_dns3.sim.onnx`（对拍 6.3e-6 的那份）
- 与 WP-05b（Python 流式 cache）互为补充：本 demo 证明**部署层**（C++ 前向 +
  定点/嵌入式迁移路径），WP-05b 证明**流式层**（逐帧 cache 等价性）
- 面试叙事：「STFT 约定、内存布局、跨实现对拍」三个坑都有实测数据背书

## 附记（V5 加固，2026-10-12）

1. **reflect 索引复核（v3 扫描报"偏移 1"，实测为误报）**：现行
   `wav[-idx]` / `wav[2N-2-idx]` 与 torch `pad_mode='reflect'` 一致
   （spec 级 max|Δ| = 7.3e-07）；按建议改 `-idx-1`/`2N-1-idx` 会变成 symmetric 填充，
   首/末帧 max|Δ| = 0.243。已在 `stft()` 内写下推导注释，防止再次误"修"。
2. **WAV 解析按 RIFF 块遍历**（旧实现固定 `seekg(22/34/40)`）：校验 RIFF/WAVE 魔数、
   fmt 块长度与 PCM16/单声道、data 声明长度与实读长度；自动跳过 LIST/fact/bext 等
   附加块（奇数长度按 RIFF 补 1 字节对齐）。实测：在 `data` 前插入 8 字节 LIST 块的
   wav 解析结果与普通 wav **完全一致**（spec max|Δ| 同为 1.87e-05）；
   float32 wav / 非 RIFF 文件均给出明确报错 + 退出码 1。
3. **Ort 异常保护与张量断言**：`Ort::Session` 构造与 `Run` 均包 try/catch 并打印诊断
   （模型不存在 → 退出码 2；实测报错含 ORT 原始信息 + 提示 dll 版本一致性）；
   输出张量断言 **1 个输出 / dtype=float32 / shape=[1,257,T,2]**，不符退出码 4
   （旧实现直接 `GetTensorData<float>()` 盲取）。
4. **写盘失败不再静默**：`write_wav` 检查 `ofstream` 是否打开成功并检查 flush 后状态。
   实测把输出指到不存在的目录：旧实现打印 `-> path` 且返回 0 但**磁盘上没有文件**；
   现在退出码 1 + `[wav] 无法写入 ...（目录不存在或无权限）`。
5. **CMake 可移植性**：`ORT_ROOT` 支持 `-DORT_ROOT=` / 环境变量 `ONNXRUNTIME_ROOT` /
   `ORT_ROOT` 覆盖，并校验头文件存在（否则 FATAL_ERROR 给指引）；Windows 上
   POST_BUILD 自动部署 `onnxruntime.dll` 到 exe 同目录（消除 System32 同名 dll 的
   版本不确定性，实测已部署 10.9MB dll）。

**对拍数字（V5 复跑，修复后）**：spec 级 max|Δ| = **1.87e-05**（首帧 1.34e-06 / 末帧
4.90e-06）；端到端 max|Δ| = **3.43e-05**（= 1.12 × PCM16 量化步长 3.05e-05，首/中/末
各 ~3.0e-05，无帧边界异常）；RTF = 0.0085~0.0099。修复前后上述数字一致（因为
reflect 公式本就没动，改动只在健壮化路径）。
