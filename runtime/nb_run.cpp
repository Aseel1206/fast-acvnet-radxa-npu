// Standalone runner for Allwinner NPU (VIPLite) .nb models - no wrapper library needed.
// Loads a .nb, fills the inputs (raw files or random bytes), runs it N times (or for T seconds, optionally paced),
// prints latency statistics and can dump output 0 as float32.
//
//   nb_run -m model.nb [-n 50] [-i0 left.bin] [-i1 right.bin] [-o out.bin] [-hz R] [-t SECONDS]
//   -n N     number of timed runs (default 50)          -t S   run for S seconds instead of N runs
//   -hz R    pace the runs at R per second              -o F   write output 0 as float32 (dequantised if needed)
//   -iK F    raw file for input K (uint8 RGB HWC for the .nb files in this repo)
//
// Build against YOUR copy of the Allwinner VIPLite SDK (header + libNBGlinker.so/libVIPhal.so, e.g. from the AWNPU
// model zoo: common/npuruntime/include and common/npuruntime/lib_linux_aarch64/A733); see build_on_board.sh.
// The Vivante header and libraries are proprietary and are NOT included in this repository.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <vip_lite.h>

static unsigned bytes_of(vip_enum f) {
    switch (f) {
        case VIP_BUFFER_FORMAT_FP32: case VIP_BUFFER_FORMAT_INT32: case VIP_BUFFER_FORMAT_UINT32: return 4;
        case VIP_BUFFER_FORMAT_FP16: case VIP_BUFFER_FORMAT_BFP16: case VIP_BUFFER_FORMAT_INT16: case VIP_BUFFER_FORMAT_UINT16: return 2;
        case VIP_BUFFER_FORMAT_INT64: case VIP_BUFFER_FORMAT_UINT64: case VIP_BUFFER_FORMAT_FP64: return 8;
        default: return 1;
    }
}
static float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v;
    if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexpf((float)(m + 1024), (int)e - 25);
    return s ? -v : v;
}
struct Io { vip_buffer buf = nullptr; vip_buffer_create_params_t p; unsigned elems = 1; unsigned bytes = 0; char name[256] = {0}; };

static bool make_buffers(vip_network net, bool is_input, int count, std::vector<Io>& v) {
    v.resize(count);
    for (int i = 0; i < count; ++i) {
        Io& io = v[i]; memset(&io.p, 0, sizeof(io.p)); io.p.memory_type = VIP_BUFFER_MEMORY_TYPE_DEFAULT;
        auto q = [&](vip_enum prop, void* out) { return is_input ? vip_query_input(net, i, prop, out) : vip_query_output(net, i, prop, out); };
        q(VIP_BUFFER_PROP_DATA_FORMAT, &io.p.data_format); q(VIP_BUFFER_PROP_NUM_OF_DIMENSION, &io.p.num_of_dims);
        q(VIP_BUFFER_PROP_SIZES_OF_DIMENSION, io.p.sizes); q(VIP_BUFFER_PROP_QUANT_FORMAT, &io.p.quant_format); q(VIP_BUFFER_PROP_NAME, io.name);
        if (io.p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) q(VIP_BUFFER_PROP_FIXED_POINT_POS, &io.p.quant_data.dfp.fixed_point_pos);
        if (io.p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM) { q(VIP_BUFFER_PROP_TF_SCALE, &io.p.quant_data.affine.scale); q(VIP_BUFFER_PROP_TF_ZERO_POINT, &io.p.quant_data.affine.zeroPoint); }
        for (unsigned k = 0; k < io.p.num_of_dims; ++k) io.elems *= io.p.sizes[k];
        printf("%s %d: dims", is_input ? "input " : "output", i);
        for (unsigned k = 0; k < io.p.num_of_dims; ++k) printf(" %u", io.p.sizes[k]);
        printf(", data_format=%d, quant_format=%d, name=%s\n", (int)io.p.data_format, (int)io.p.quant_format, io.name);
        if (vip_create_buffer(&io.p, sizeof(io.p), &io.buf) != VIP_SUCCESS) { printf("vip_create_buffer failed (%s %d)\n", is_input ? "input" : "output", i); return false; }
        io.bytes = vip_get_buffer_size(io.buf);
    }
    return true;
}
static std::vector<float> to_float(const Io& o, const uint8_t* d) {
    std::vector<float> r(o.elems); const vip_buffer_create_params_t& p = o.p;
    for (unsigned i = 0; i < o.elems; ++i) {
        switch (p.data_format) {
            case VIP_BUFFER_FORMAT_FP32: r[i] = ((const float*)d)[i]; break;
            case VIP_BUFFER_FORMAT_FP16: r[i] = half_to_float(((const uint16_t*)d)[i]); break;
            case VIP_BUFFER_FORMAT_UINT8: r[i] = p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM ? (d[i] - p.quant_data.affine.zeroPoint) * p.quant_data.affine.scale : d[i]; break;
            case VIP_BUFFER_FORMAT_INT8: r[i] = p.quant_format == VIP_BUFFER_QUANTIZE_TF_ASYMM ? (((const int8_t*)d)[i] - p.quant_data.affine.zeroPoint) * p.quant_data.affine.scale
                                                 : ((const int8_t*)d)[i] * (p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT ? ldexpf(1.f, -p.quant_data.dfp.fixed_point_pos) : 1.f); break;
            case VIP_BUFFER_FORMAT_INT16: r[i] = ((const int16_t*)d)[i] * (p.quant_format == VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT ? ldexpf(1.f, -p.quant_data.dfp.fixed_point_pos) : 1.f); break;
            default: r[i] = NAN;
        }
    }
    return r;
}

int main(int argc, char** argv) {
    std::string model, in[8], out; int n = 50; double hz = 0, tsec = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i]; bool more = i + 1 < argc;
        if (a == "-m" && more) model = argv[++i]; else if (a == "-n" && more) n = atoi(argv[++i]);
        else if (a == "-o" && more) out = argv[++i]; else if (a == "-hz" && more) hz = atof(argv[++i]);
        else if (a == "-t" && more) tsec = atof(argv[++i]);
        else if (a.size() == 3 && a[0] == '-' && a[1] == 'i' && isdigit(a[2]) && more) in[a[2] - '0'] = argv[++i];
    }
    if (model.empty()) { printf("usage: nb_run -m model.nb [-n N] [-i0 f] [-i1 f] [-o out.bin] [-hz R] [-t S]\n"); return 1; }

    std::ifstream f(model, std::ios::binary); if (!f) { printf("cannot open %s\n", model.c_str()); return 2; }
    std::vector<char> nb((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (vip_init() != VIP_SUCCESS) { printf("vip_init failed (is the NPU driver loaded? /dev/vipcore)\n"); return 3; }
    printf("VIPLite driver version 0x%08x\n", vip_get_version());

    auto t0 = std::chrono::steady_clock::now();
    vip_network net = nullptr;
    if (vip_create_network(nb.data(), (vip_uint32_t)nb.size(), VIP_CREATE_NETWORK_FROM_MEMORY, &net) != VIP_SUCCESS) { printf("vip_create_network failed: not a valid .nb for this NPU?\n"); return 4; }
    vip_uint32_t nin = 0, nout = 0;
    vip_query_network(net, VIP_NETWORK_PROP_INPUT_COUNT, &nin); vip_query_network(net, VIP_NETWORK_PROP_OUTPUT_COUNT, &nout);
    std::vector<Io> I, O;
    if (!make_buffers(net, true, nin, I) || !make_buffers(net, false, nout, O)) return 5;
    if (vip_prepare_network(net) != VIP_SUCCESS) { printf("vip_prepare_network failed\n"); return 6; }
    printf("loaded in %.1f ms: %u inputs, %u outputs\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), nin, nout);

    srand(1);
    for (unsigned k = 0; k < nin; ++k) {   // fill inputs once (they do not change between runs)
        uint8_t* p = (uint8_t*)vip_map_buffer(I[k].buf); std::vector<char> d;
        if (k < 8 && !in[k].empty()) { std::ifstream g(in[k], std::ios::binary); d.assign((std::istreambuf_iterator<char>(g)), std::istreambuf_iterator<char>()); }
        if (!d.empty()) { memcpy(p, d.data(), std::min<size_t>(d.size(), I[k].bytes)); printf("input %u: %zu bytes from %s (buffer %u)\n", k, d.size(), in[k].c_str(), I[k].bytes); }
        else { for (unsigned j = 0; j < I[k].bytes; ++j) p[j] = rand() & 0xff; printf("input %u: random bytes (buffer %u)\n", k, I[k].bytes); }
        vip_unmap_buffer(I[k].buf);
    }
    auto run_once = [&]() -> double {
        auto a = std::chrono::steady_clock::now();
        for (auto& io : I) vip_flush_buffer(io.buf, VIP_BUFFER_OPER_TYPE_FLUSH);
        for (unsigned k = 0; k < nin; ++k) vip_set_input(net, k, I[k].buf);
        for (unsigned k = 0; k < nout; ++k) vip_set_output(net, k, O[k].buf);
        vip_status_e s = vip_run_network(net);
        for (auto& io : O) vip_flush_buffer(io.buf, VIP_BUFFER_OPER_TYPE_INVALIDATE);
        if (s != VIP_SUCCESS) { printf("vip_run_network failed, status=%d\n", (int)s); exit(7); }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    };
    for (int i = 0; i < 3; ++i) run_once();   // warm-up
    std::vector<double> ms; auto start = std::chrono::steady_clock::now();
    auto el = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
    for (int i = 0; tsec > 0 ? el() < tsec : i < n; ++i) {
        if (hz > 0) { double due = i / hz, now = el(); if (due > now) std::this_thread::sleep_for(std::chrono::duration<double>(due - now)); }
        ms.push_back(run_once());
    }
    double dur = el(); printf("completed %zu runs in %.1f s = %.2f runs/s\n", ms.size(), dur, ms.size() / dur);
    std::vector<double> v = ms; std::sort(v.begin(), v.end());
    double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    printf("run    mean %.2f ms  p50 %.2f  p95 %.2f  p99 %.2f  min %.2f  max %.2f  (%.2f FPS)\n", mean, v[v.size() / 2], v[(size_t)(v.size() * 0.95)],
           v[std::min(v.size() - 1, (size_t)(v.size() * 0.99))], v.front(), v.back(), 1000.0 / mean);

    for (unsigned k = 0; k < nout; ++k) {
        std::vector<float> r = to_float(O[k], (const uint8_t*)vip_map_buffer(O[k].buf)); vip_unmap_buffer(O[k].buf);
        double mn = 1e30, mx = -1e30, sum = 0; for (float x : r) { mn = std::min<double>(mn, x); mx = std::max<double>(mx, x); sum += x; }
        printf("output %u: %zu values, min %.4f max %.4f mean %.4f\n", k, r.size(), mn, mx, sum / r.size());
        if (k == 0 && !out.empty()) { std::ofstream o(out, std::ios::binary); o.write((const char*)r.data(), r.size() * sizeof(float)); printf("wrote %s\n", out.c_str()); }
    }
    vip_finish_network(net); vip_destroy_network(net);
    for (auto& io : I) vip_destroy_buffer(io.buf); for (auto& io : O) vip_destroy_buffer(io.buf);
    vip_destroy(); return 0;
}
