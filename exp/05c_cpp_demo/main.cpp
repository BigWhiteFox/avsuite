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
static std::vector<float> read_wav(const std::string& path, int* sr_out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    auto rd_u16 = [&] { uint8_t b[2]; f.read(reinterpret_cast<char*>(b), 2); return b[0] | (b[1] << 8); };
    auto rd_u32 = [&] { uint8_t b[4]; f.read(reinterpret_cast<char*>(b), 4);
                        return static_cast<uint32_t>(b[0]) | (b[1] << 8) | (b[2] << 16) | (b[3] << 24); };
    f.seekg(22); int ch = rd_u16(); *sr_out = rd_u32(); f.seekg(34); int bits = rd_u16();
    if (ch != 1 || bits != 16) { std::fprintf(stderr, "仅支持 PCM16 单声道\n"); std::exit(1); }
    f.seekg(40); int dlen = rd_u32();
    std::vector<int16_t> raw(dlen / 2);
    f.read(reinterpret_cast<char*>(raw.data()), dlen);
    std::vector<float> out(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) out[i] = raw[i] / 32768.0f;
    return out;
}

static void write_wav(const std::string& path, const std::vector<float>& x, int sr) {
    std::ofstream f(path, std::ios::binary);
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
            // 首帧 pad 区差异会让模型输入整体错位）
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
    Ort::Session session(env, wpath.c_str(), opts);

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
    auto outs = session.Run(run_opts, in_names, &input, 1, out_names, 1);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double dur = static_cast<double>(wav.size()) / kSr;
    std::printf("整段推理 %.1f ms (T=%d, %.2fs 音频)  RTF %.4f\n", ms, T, dur, ms / 1000 / dur);

    auto enh = istft(outs[0].GetTensorData<float>(), T, static_cast<int>(wav.size()));
    write_wav(out_path, enh, kSr);
    std::printf("-> %s\n", out_path.c_str());
    return 0;
}
