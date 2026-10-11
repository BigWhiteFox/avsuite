// GTCRN C++ 单帧/整段 ONNX 推理 demo（WP-05c 选做）
// 对标车载 DSP/嵌入式岗的差异化加分项：零 Python 依赖的纯 C++ 前向。
//
// 约定复刻（与 PyTorch 推理逐位对齐）：
//   spec = torch.stft(wav, 512, 256, 512, hann(512)**0.5)  -> (B,F,T,2) 实虚视图
//   enh  = model(spec) -> istft(同 sqrt-hann 窗)  [w²= hann 的 WOLA]
// 本 demo 自实现：radix-2 FFT + 周期 sqrt-hann 窗 + center padding，
// 输入 16k 单声道 PCM16 wav。
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using std::complex;
using fft_t = complex<float>;

constexpr int kSr = 16000, kNfft = 512, kHop = 256, kFreq = kNfft / 2 + 1;
constexpr float kPi = 3.14159265358979323846f;

// ---------------------------------------------------------------- FFT
static void fft(std::vector<fft_t>& a, bool invert) {
    const int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2 * kPi / len * (invert ? 1.f : -1.f);
        fft_t wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            fft_t w(1, 0);
            for (int j = 0; j < len / 2; ++j) {
                fft_t u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert)
        for (auto& x : a) x /= static_cast<float>(n);
}

static inline float hann_sqrt(int i) { return std::sin(kPi * i / kNfft); }  // hann^0.5 周期窗

// ---------------------------------------------------------------- WAV IO（PCM16 单声道）
// 按 RIFF **块**遍历解析（V5 加固）：旧实现用固定偏移 seekg(22/34/40) 直接读字段，
// 任何带附加块（LIST/fact/bext/cue…）或 fmt 扩展字段的 wav 都会解析错位，
// 且"RIFF/WAVE"魔数、块长、data 实际长度都不校验。
static uint16_t rd_u16(std::istream& f) {
    uint8_t b[2] = {0, 0};
    f.read(reinterpret_cast<char*>(b), 2);
    return static_cast<uint16_t>(b[0] | (b[1] << 8));
}
static uint32_t rd_u32(std::istream& f) {
    uint8_t b[4] = {0, 0, 0, 0};
    f.read(reinterpret_cast<char*>(b), 4);
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
           (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}
static void skip_bytes(std::istream& f, uint32_t n) {
    for (uint32_t k = 0; k < n; ++k) f.get();
}

static std::vector<float> read_wav(const std::string& path, int* sr_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "[wav] 无法打开 %s\n", path.c_str()); std::exit(1); }
    char riff[4], wave[4];
    if (!f.read(riff, 4)) { std::fprintf(stderr, "[wav] 文件过短：%s\n", path.c_str()); std::exit(1); }
    (void)rd_u32(f);                                  // RIFF 块大小（不用于定位）
    if (!f.read(wave, 4)) { std::fprintf(stderr, "[wav] 文件过短：%s\n", path.c_str()); std::exit(1); }
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        std::fprintf(stderr, "[wav] 不是 RIFF/WAVE 文件：%s\n", path.c_str());
        std::exit(1);
    }
    int fmt_seen = 0, audio_fmt = 0, ch = 0, bits = 0;
    uint32_t sr = 0;
    std::vector<float> out;
    while (out.empty()) {
        char id[4];
        if (!f.read(id, 4)) break;                    // 块遍历结束
        uint32_t sz = rd_u32(f);
        if (!f) break;
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (sz < 16) { std::fprintf(stderr, "[wav] fmt 块过短 (%u)\n", sz); std::exit(1); }
            audio_fmt = rd_u16(f);
            ch = rd_u16(f);
            sr = rd_u32(f);
            (void)rd_u32(f);                          // byte rate
            (void)rd_u16(f);                          // block align
            bits = rd_u16(f);
            skip_bytes(f, sz - 16);                   // fmt 扩展字段（cbSize 等）
            fmt_seen = 1;
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (!fmt_seen) { std::fprintf(stderr, "[wav] data 块早于 fmt 块\n"); std::exit(1); }
            if (audio_fmt != 1 || ch != 1 || bits != 16) {
                std::fprintf(stderr, "[wav] 仅支持 PCM16 单声道（fmt=%d ch=%d bits=%d）\n",
                             audio_fmt, ch, bits);
                std::exit(1);
            }
            std::vector<int16_t> raw(sz / 2);
            if (!raw.empty() &&
                !f.read(reinterpret_cast<char*>(raw.data()),
                        static_cast<std::streamsize>(raw.size()) * 2)) {
                std::fprintf(stderr, "[wav] data 块声明 %u 字节但读取不完整\n", sz);
                std::exit(1);
            }
            out.resize(raw.size());
            for (size_t i = 0; i < raw.size(); ++i) out[i] = raw[i] / 32768.0f;
        } else {
            skip_bytes(f, sz + (sz & 1));             // 附加块（奇数长度按 RIFF 补 1 字节对齐）
        }
    }
    if (!fmt_seen || out.empty()) {
        std::fprintf(stderr, "[wav] 缺少 fmt/data 块或音频数据为空：%s\n", path.c_str());
        std::exit(1);
    }
    *sr_out = static_cast<int>(sr);
    return out;
}

static void write_wav(const std::string& path, const std::vector<float>& x, int sr) {
    std::ofstream f(path, std::ios::binary);
    // V5 加固：旧实现不检查 ofstream 是否打开成功——目标目录不存在/无权限时
    // 会"静默成功"（打印 -> path、返回 0，但磁盘上没有任何文件）。
    if (!f) {
        std::fprintf(stderr, "[wav] 无法写入 %s（目录不存在或无权限）\n", path.c_str());
        std::exit(1);
    }
    auto wr16 = [&](uint16_t v) { f.put(v & 0xff); f.put(v >> 8); };
    auto wr32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) f.put((v >> (8 * i)) & 0xff); };
    int dlen = static_cast<int>(x.size()) * 2;
    f.write("RIFF", 4); wr32(36 + dlen); f.write("WAVE", 4);
    f.write("fmt ", 4); wr32(16); wr16(1); wr16(1); wr32(sr); wr32(sr * 2); wr16(2); wr16(16);
    f.write("data", 4); wr32(dlen);
    for (float v : x) {
        int16_t s = static_cast<int16_t>(std::max(-1.f, std::min(1.f, v)) * 32767.f);
        wr16(static_cast<uint16_t>(s));
    }
    f.flush();
    if (!f) {                                          // 写盘失败（磁盘满/句柄失效）也显式报错
        std::fprintf(stderr, "[wav] 写入失败：%s\n", path.c_str());
        std::exit(1);
    }
}

// ---------------------------------------------------------------- STFT（与 torch 约定对齐）
static std::vector<float> stft(const std::vector<float>& wav, int* T_out) {
    const int N = static_cast<int>(wav.size());
    const int T = N / kHop + 1;                       // center=True：两侧各 pad 256
    *T_out = T;
    std::vector<float> spec(static_cast<size_t>(kFreq) * T * 2);   // F-major 布局 [1,F,T,2]
    for (int t = 0; t < T; ++t) {
        std::vector<fft_t> frame(kNfft, 0.f);
        int base = t * kHop - kNfft / 2;
        for (int i = 0; i < kNfft; ++i) {
            int idx = base + i;
            // torch.stft 默认 pad_mode='reflect'（镜像填充，勿用零填充——
            // 首帧 pad 区差异会让模型输入整体错位）。
            // V5 复核（实测，勿再"修"成 -idx-1）：torch/numpy 的 reflect **不重复边界样本**，
            // 左 pad 第 p 个样本 = x[pad-p]（p=0..pad-1）→ 代入 p = pad+idx 得 x[-idx]；
            // 右 pad 第 q 个样本 = x[N-2-q]（q=idx-N）→ x[2N-2-idx]。
            // 对拍：本式 vs torch.stft 的 spec 级 max|Δ|=7.3e-07（float32 FFT 舍入）；
            // 若改成 x[-idx-1]/x[2N-1-idx]（= symmetric 填充）则首/末帧 max|Δ|≈0.24。
            float v;
            if (idx < 0) v = wav[-idx];
            else if (idx >= N) v = wav[2 * N - 2 - idx];
            else v = wav[idx];
            frame[i] = v * hann_sqrt(i);
        }
        fft(frame, false);
        for (int f_ = 0; f_ < kFreq; ++f_) {
            spec[((static_cast<size_t>(f_) * T + t)) * 2] = frame[f_].real();
            spec[((static_cast<size_t>(f_) * T + t)) * 2 + 1] = frame[f_].imag();
        }
    }
    return spec;
}

// ---------------------------------------------------------------- ISTFT（sqrt-hann WOLA）
static std::vector<float> istft(const float* spec, int T, int out_len) {
    std::vector<double> buf((size_t)(T - 1) * kHop + kNfft, 0.0);
    std::vector<double> wsum(buf.size(), 0.0);
    for (int t = 0; t < T; ++t) {
        std::vector<fft_t> fr(kNfft);
        for (int f_ = 0; f_ < kFreq; ++f_)          // F-major 布局（与 STFT 输出一致）
            fr[f_] = fft_t(spec[((static_cast<size_t>(f_) * T + t)) * 2],
                           spec[((static_cast<size_t>(f_) * T + t)) * 2 + 1]);
        for (int f_ = kFreq; f_ < kNfft; ++f_) fr[f_] = std::conj(fr[kNfft - f_]);
        fft(fr, true);
        int base = t * kHop;
        for (int i = 0; i < kNfft; ++i) {
            double w = hann_sqrt(i);
            buf[base + i] += w * fr[i].real();
            wsum[base + i] += w * w;
        }
    }
    std::vector<float> out(out_len);
    for (int i = 0; i < out_len; ++i) {
        int p = i + kNfft / 2;                       // center 裁剪（右侧不足补零）
        if (p < static_cast<int>(buf.size()) && wsum[p] > 1e-12)
            out[i] = static_cast<float>(buf[p] / wsum[p]);
        else
            out[i] = 0.f;
    }
    return out;
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.onnx> <in.wav> <out.wav> [frames]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1], in_path = argv[2], out_path = argv[3];

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gtcrn_demo");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    // 1.17 起 Windows 版 Session 构造吃 wchar 路径（ORT_TSTR）；ASCII 路径安全转换
    std::wstring wpath(model_path.begin(), model_path.end());
    // V5 加固：Session 构造包 try/catch 并给出诊断（旧实现直接抛出 → 无上下文的 abort；
    // 常见原因：模型路径不存在、onnxruntime.dll 版本与头文件不匹配、模型损坏）。
    std::unique_ptr<Ort::Session> session;
    try {
        session = std::make_unique<Ort::Session>(env, wpath.c_str(), opts);
    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[ort] 加载模型失败：%s\n  模型=%s\n"
                             "  提示：确认模型文件存在且 onnxruntime.dll 与编译期头文件同版本\n",
                     e.what(), model_path.c_str());
        return 2;
    }

    int sr = 0;
    auto wav = read_wav(in_path, &sr);
    if (sr != kSr) { std::fprintf(stderr, "输入须 16k（实际 %d）\n", sr); return 1; }

    int T = 0;
    std::vector<float> spec = stft(wav, &T);
    // 注意 dump 模式也在 F-major 布局（与张量一致）
    if (argc > 4 && std::string(argv[4]) == "--dump-spec") {
        std::ofstream d(argv[3] + std::string(".spec.bin"), std::ios::binary);
        d.write(reinterpret_cast<const char*>(spec.data()), spec.size() * sizeof(float));
        std::printf("spec dumped: T=%d\n", T);
        return 0;
    }
    std::vector<int64_t> shape = {1, kFreq, T, 2};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(mem, spec.data(), spec.size(), shape.data(), 4);
    const char* in_names[] = {"spec"};
    const char* out_names[] = {"enh"};

    Ort::RunOptions run_opts;
    auto t0 = std::chrono::steady_clock::now();
    // V5 加固：Run 包 try/catch（输入名/输出名不符、shape 不匹配、ORT 内部错误都会抛）
    std::vector<Ort::Value> outs;
    try {
        outs = session->Run(run_opts, in_names, &input, 1, out_names, 1);
    } catch (const Ort::Exception& e) {
        std::fprintf(stderr, "[ort] 推理失败：%s\n  输入 shape=[1,%d,%d,2] 输入名=spec 输出名=enh\n",
                     e.what(), kFreq, T);
        return 3;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double dur = static_cast<double>(wav.size()) / kSr;

    // V5 加固：输出张量 dtype/shape 断言（旧实现直接 GetTensorData<float> 盲取，
    // 模型换成 fp16/双输出/不同 T 时会静默读错内存）
    if (outs.size() != 1) {
        std::fprintf(stderr, "[ort] 期望 1 个输出，实际 %zu 个\n", outs.size());
        return 4;
    }
    auto info = outs[0].GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        std::fprintf(stderr, "[ort] 输出 dtype 非 float32（element_type=%d）\n",
                     static_cast<int>(info.GetElementType()));
        return 4;
    }
    const auto oshape = info.GetShape();
    if (oshape.size() != 4 || oshape[0] != 1 || oshape[1] != kFreq ||
        oshape[2] != T || oshape[3] != 2) {
        std::fprintf(stderr, "[ort] 输出 shape 不符：期望 [1,%d,%d,2]，实际 [", kFreq, T);
        for (size_t k = 0; k < oshape.size(); ++k)
            std::fprintf(stderr, "%s%lld", k ? "," : "", static_cast<long long>(oshape[k]));
        std::fprintf(stderr, "]\n");
        return 4;
    }
    std::printf("整段推理 %.1f ms (T=%d, %.2fs 音频)  RTF %.4f\n", ms, T, dur, ms / 1000 / dur);

    auto enh = istft(outs[0].GetTensorData<float>(), T, static_cast<int>(wav.size()));
    write_wav(out_path, enh, kSr);
    std::printf("-> %s\n", out_path.c_str());
    return 0;
}
