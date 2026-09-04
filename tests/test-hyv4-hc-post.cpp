#include "ggml.h"
#include "ggml-backend.h"
#include "../src/models/hyv4-hc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Original stream-wise HY4 graph retained as the numerical oracle.
static ggml_tensor * reference(ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * residual, ggml_tensor * post, int64_t n_hc, int64_t n_embd) {
    const int64_t n_tokens = x->ne[1];
    x = ggml_cast(ctx, x, GGML_TYPE_F32);
    post = ggml_cast(ctx, post, GGML_TYPE_F32);
    const auto out_type = residual->type;
    residual = ggml_cast(ctx, residual, GGML_TYPE_F32);
    ggml_tensor * out = nullptr;
    for (int64_t ih = 0; ih < n_hc; ++ih) {
        auto * rh = ggml_view_2d(ctx, residual, n_embd, n_tokens, residual->nb[2], ih * residual->nb[1]);
        auto * ph = ggml_view_2d(ctx, post, 1, n_tokens, post->nb[1], ih * post->nb[0]);
        auto * cur = ggml_add(ctx, rh, ggml_mul(ctx, x, ph));
        cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
        out = out == nullptr ? cur : ggml_concat(ctx, out, cur, 1);
    }
    return ggml_cast(ctx, out, out_type);
}

static void fill(ggml_tensor * tensor, std::mt19937 & rng) {
    std::uniform_real_distribution<float> values(-2.0f, 2.0f);
    const size_t count = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> data(count);
        for (auto & v : data) { v = values(rng); }
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        std::vector<ggml_fp16_t> data(count);
        for (auto & v : data) { v = ggml_fp32_to_fp16(values(rng)); }
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    }
}

static double time_graph(ggml_backend_t backend, ggml_cgraph * graph, int repeats) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) { return -1; }
    }
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats;
}

static bool run_case(ggml_backend_t backend, int64_t width, int64_t streams,
        int64_t tokens, ggml_type type, bool strided, int bench) {
    auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    if (!ctx) { return false; }
    const int64_t pad = strided ? 7 : 0;
    auto * x_storage = ggml_new_tensor_2d(ctx, type, width + pad, tokens);
    auto * r_storage = ggml_new_tensor_3d(ctx, type, width + pad, streams, tokens);
    auto * p_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, streams + pad, tokens);
    auto * x = ggml_view_2d(ctx, x_storage, width, tokens, x_storage->nb[1], 0);
    auto * r = ggml_view_3d(ctx, r_storage, width, streams, tokens, r_storage->nb[1], r_storage->nb[2], 0);
    auto * p = ggml_view_2d(ctx, p_storage, streams, tokens, p_storage->nb[1], 0);
    auto * a = reference(ctx, x, r, p, streams, width);
    auto * b = hyv4_hc_post_broadcast(ctx, x, r, p, streams, width);
    auto * ga = ggml_new_graph(ctx);
    auto * gb = ggml_new_graph(ctx);
    ggml_build_forward_expand(ga, a);
    ggml_build_forward_expand(gb, b);
    constexpr int chain_count = 16;
    ggml_cgraph * ga_chain = nullptr;
    ggml_cgraph * gb_chain = nullptr;
    if (bench > 0) {
        auto * ra = r;
        auto * rb = r;
        for (int i = 0; i < chain_count; ++i) {
            ra = reference(ctx, x, ra, p, streams, width);
            rb = hyv4_hc_post_broadcast(ctx, x, rb, p, streams, width);
        }
        ga_chain = ggml_new_graph(ctx);
        gb_chain = ggml_new_graph(ctx);
        ggml_build_forward_expand(ga_chain, ra);
        ggml_build_forward_expand(gb_chain, rb);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::mt19937 rng(93 + unsigned(width + streams + tokens));
    fill(x_storage, rng); fill(r_storage, rng); fill(p_storage, rng);
    bool ok = ggml_backend_graph_compute(backend, ga) == GGML_STATUS_SUCCESS &&
              ggml_backend_graph_compute(backend, gb) == GGML_STATUS_SUCCESS;
    std::vector<unsigned char> va(ggml_nbytes(a)), vb(ggml_nbytes(b));
    ggml_backend_tensor_get(a, va.data(), 0, va.size());
    ggml_backend_tensor_get(b, vb.data(), 0, vb.size());
    const bool exact = va == vb;
    double max_abs = 0;
    for (size_t i = 0; i < size_t(ggml_nelements(a)); ++i) {
        float av, bv;
        if (type == GGML_TYPE_F32) {
            std::memcpy(&av, va.data() + i * sizeof(float), sizeof(float));
            std::memcpy(&bv, vb.data() + i * sizeof(float), sizeof(float));
        } else {
            ggml_fp16_t ah, bh;
            std::memcpy(&ah, va.data() + i * sizeof(ah), sizeof(ah));
            std::memcpy(&bh, vb.data() + i * sizeof(bh), sizeof(bh));
            av = ggml_fp16_to_fp32(ah); bv = ggml_fp16_to_fp32(bh);
        }
        ok = ok && std::isfinite(av) && std::isfinite(bv);
        max_abs = std::max(max_abs, double(std::abs(av - bv)));
    }
    ok = ok && exact;
    std::printf("%s width=%lld streams=%lld tokens=%lld %s stride=%d nodes=%d/%d max_abs=%.9g bit_exact=%s %s\n",
            ggml_backend_name(backend), (long long) width, (long long) streams, (long long) tokens,
            ggml_type_name(type), strided, ggml_graph_n_nodes(ga), ggml_graph_n_nodes(gb), max_abs,
            exact ? "yes" : "no", ok ? "PASS" : "FAIL");
    if (bench > 0) {
        const double baseline = time_graph(backend, ga, bench);
        const double broadcast = time_graph(backend, gb, bench);
        std::printf("  wall us/op including submit: reference=%.3f broadcast=%.3f\n", baseline, broadcast);
        ok = ok && baseline >= 0 && broadcast >= 0;
        // Compare long chains as well: isolated tiny graphs mostly measure
        // command-buffer submission/wait, not the incremental device work.
        ggml_backend_graph_compute(backend, ga_chain);
        ggml_backend_graph_compute(backend, gb_chain);
        const double baseline_chain = time_graph(backend, ga_chain, bench);
        const double broadcast_chain = time_graph(backend, gb_chain, bench);
        std::printf("  %d-op chain incremental us/op: reference=%.3f broadcast=%.3f (wall slope, not GPU counter)\n",
                chain_count, (baseline_chain - baseline) / (chain_count - 1),
                (broadcast_chain - broadcast) / (chain_count - 1));
        ok = ok && baseline_chain >= 0 && broadcast_chain >= 0;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    int bench = 0;
    bool require_metal = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0 && i + 1 < argc) { bench = std::atoi(argv[++i]); }
        else if (std::strcmp(argv[i], "--require-metal") == 0) { require_metal = true; }
    }
    ggml_backend_load_all();
    int checks = 0, failures = 0, metal_checks = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        const bool cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
        const bool metal = std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "MTL") == 0;
        if (!cpu && !metal) { continue; }
        auto * backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) { continue; }
        for (const auto type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            for (const int64_t streams : {1, 2, 4, 8}) {
                for (const int64_t tokens : {1, 3}) {
                    for (const bool strided : {false, true}) {
                        ++checks; metal_checks += metal;
                        failures += !run_case(backend, 37, streams, tokens, type, strided, 0);
                    }
                }
            }
            ++checks; metal_checks += metal;
            failures += !run_case(backend, 6144, 4, 1, type, false, bench);
        }
        ggml_backend_free(backend);
    }
    std::printf("%d/%d passed; Metal cases=%d\n", checks - failures, checks, metal_checks);
    if (require_metal && metal_checks == 0) {
        std::fprintf(stderr, "FAIL: --require-metal requested but no Metal cases executed\n");
        return 1;
    }
    if (checks == 0) { return 77; }
    return failures ? 1 : 0;
}
