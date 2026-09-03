// Numerical check for the fused --slot8 Metal operator (GGML_OP_FLASHMOE_SLOT8_FFN) on the
// HY4 mixed quant combinations: gate/up in {STQ1_0, IQ2_XXS} x down in {IQ3_XXS, IQ4_XS}.
//
// For every combination (small shape, the real HY4 routed shape, and a non-8 width) this builds
// random quantized expert banks and evaluates the routed FFN four ways:
//   - exact: double-precision CPU evaluation of the dequantized weights (ground truth)
//   - fused: the operator through the Metal backend with the fused Phase A / Phase B kernels
//   - reference: the same operator with LLAMA_FLASH_MOE_SLOT8_REFERENCE=1 (mul_mat/GLU/weighted-sum encoder)
//   - generic: the mul_mat_id + swiglu + mul_mat_id + weighted-sum graph used without --slot8
// and reports max absolute / relative (to max |exact|) error for each pair.
//
// Usage: test-flashmoe-slot8-hyv4 [--bench N]   (N timed iterations per path on the HY4 shape)

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kMaxRelTol = 1e-4; // relative to max |exact|; F32 reorder noise is ~1e-6

struct case_spec {
    std::string name;
    ggml_type   type_gate_up;
    ggml_type   type_down;
    int64_t     n_embd;
    int64_t     n_ff;
    int64_t     n_slots;
    int64_t     n_used;
    uint32_t    seed;
};

struct err_stats {
    double max_abs = 0.0; // max |a - ref|
    double max_rel = 0.0; // max_abs / max |ref|
    double ref_max = 0.0;
};

static err_stats compare_to(const std::vector<float> & a, const std::vector<double> & ref) {
    err_stats st;
    for (double v : ref) {
        st.ref_max = std::max(st.ref_max, std::fabs(v));
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs(double(a[i]) - ref[i]);
        if (!(d <= st.max_abs)) { // also catches NaN
            st.max_abs = std::isnan(d) ? INFINITY : d;
        }
    }
    st.max_rel = st.ref_max > 0.0 ? st.max_abs / st.ref_max : st.max_abs;
    return st;
}

static err_stats compare_ff(const std::vector<float> & a, const std::vector<float> & b) {
    return compare_to(a, std::vector<double>(b.begin(), b.end()));
}

static size_t block_d_offset(ggml_type type) {
    switch (type) {
        case GGML_TYPE_STQ1_0:  return 40; // {uint8 qs[32]; uint8 sign[8]; ggml_half d;}
        case GGML_TYPE_IQ2_XXS:            // {ggml_half d; uint16 qs[32];}
        case GGML_TYPE_IQ3_XXS:            // {ggml_half d; uint8 qs[96];}
        case GGML_TYPE_IQ4_XS:  return 0;  // {ggml_half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128];}
        default:
            fprintf(stderr, "unsupported block type %s\n", ggml_type_name(type));
            exit(1);
    }
}

// Random bit patterns are valid codes for every field of these block types (grid indices, sign
// words, sub-scales, nibbles); only the fp16 block scale needs a sane magnitude. This exercises
// every codebook/sign path uniformly, unlike quantizing smooth data.
static void fill_random_blocks(ggml_type type, uint8_t * data, size_t nbytes, std::mt19937 & rng) {
    const size_t bs    = ggml_type_size(type);
    const size_t d_off = block_d_offset(type);
    if (nbytes % bs != 0 || d_off + sizeof(ggml_fp16_t) > bs) {
        fprintf(stderr, "bad block layout for %s\n", ggml_type_name(type));
        exit(1);
    }

    std::uniform_int_distribution<int>    byte(0, 255);
    std::uniform_real_distribution<float> scale(0.002f, 0.010f);

    for (size_t off = 0; off < nbytes; off += bs) {
        uint8_t * blk = data + off;
        for (size_t i = 0; i < bs; ++i) {
            blk[i] = (uint8_t) byte(rng);
        }
        const ggml_fp16_t d = ggml_fp32_to_fp16(scale(rng));
        memcpy(blk + d_off, &d, sizeof(d));
    }
}

// Exact evaluation: dequantize each selected expert row with the type's CPU to_float and
// accumulate in double. This is the definition the Metal paths are measured against.
static void eval_exact(
        const case_spec & c,
        const std::vector<uint8_t> & gate,
        const std::vector<uint8_t> & up,
        const std::vector<uint8_t> & down,
        const std::vector<float>   & x,
        const std::vector<int32_t> & slots,
        const std::vector<float>   & w,
        std::vector<double>        & out) {
    const ggml_type_traits * tgu = ggml_get_type_traits(c.type_gate_up);
    const ggml_type_traits * tdn = ggml_get_type_traits(c.type_down);

    const size_t gu_row = ggml_row_size(c.type_gate_up, c.n_embd);
    const size_t gu_exp = gu_row * size_t(c.n_ff);
    const size_t dn_row = ggml_row_size(c.type_down, c.n_ff);
    const size_t dn_exp = dn_row * size_t(c.n_embd);

    std::vector<float>  wrow(size_t(std::max(c.n_embd, c.n_ff)));
    std::vector<double> h(size_t(c.n_ff));
    out.assign(size_t(c.n_embd), 0.0);

    for (int64_t e = 0; e < c.n_used; ++e) {
        const size_t slot = size_t(slots[size_t(e)]);

        for (int64_t j = 0; j < c.n_ff; ++j) {
            tgu->to_float(gate.data() + slot*gu_exp + size_t(j)*gu_row, wrow.data(), c.n_embd);
            double g = 0.0;
            for (int64_t i = 0; i < c.n_embd; ++i) {
                g += double(wrow[size_t(i)]) * double(x[size_t(i)]);
            }
            tgu->to_float(up.data() + slot*gu_exp + size_t(j)*gu_row, wrow.data(), c.n_embd);
            double u = 0.0;
            for (int64_t i = 0; i < c.n_embd; ++i) {
                u += double(wrow[size_t(i)]) * double(x[size_t(i)]);
            }
            const double s = g / (1.0 + std::exp(-g));
            h[size_t(j)] = s * u;
        }

        for (int64_t r = 0; r < c.n_embd; ++r) {
            tdn->to_float(down.data() + slot*dn_exp + size_t(r)*dn_row, wrow.data(), c.n_ff);
            double d = 0.0;
            for (int64_t k = 0; k < c.n_ff; ++k) {
                d += double(wrow[size_t(k)]) * h[size_t(k)];
            }
            out[size_t(r)] += double(w[size_t(e)]) * d;
        }
    }
}

struct metal_case {
    ggml_context        * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * x = nullptr, * gate = nullptr, * up = nullptr, * down = nullptr;
    ggml_tensor * ids = nullptr, * w = nullptr;
    ggml_tensor * out_op = nullptr, * out_generic = nullptr;
    ggml_cgraph * gf_op = nullptr, * gf_generic = nullptr;
    // chained graphs (bench only): K dependent copies so per-op GPU time can be separated
    // from the fixed command-buffer submit cost
    int chain_k = 0;
    ggml_tensor * out_chain_op = nullptr, * out_chain_generic = nullptr;
    ggml_cgraph * gf_chain_op = nullptr, * gf_chain_generic = nullptr;

    ~metal_case() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

static ggml_tensor * build_generic_ffn(ggml_context * ctx, const case_spec & c, metal_case & m, ggml_tensor * x) {
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, c.n_embd, 1, 1);
    ggml_tensor * g  = ggml_mul_mat_id(ctx, m.gate, x3, m.ids);  // [n_ff, n_used, 1]
    ggml_tensor * u  = ggml_mul_mat_id(ctx, m.up,   x3, m.ids);
    ggml_tensor * a  = ggml_swiglu_split(ctx, g, u);
    ggml_tensor * d  = ggml_mul_mat_id(ctx, m.down, a, m.ids);   // [n_embd, n_used, 1]
    ggml_tensor * dw = ggml_mul(ctx, d, m.w);
    ggml_tensor * sum = ggml_view_2d(ctx, dw, c.n_embd, 1, dw->nb[2], 0);
    for (int64_t e = 1; e < c.n_used; ++e) {
        sum = ggml_add(ctx, sum, ggml_view_2d(ctx, dw, c.n_embd, 1, dw->nb[2], size_t(e)*dw->nb[1]));
    }
    return sum;
}

static bool build_metal_case(const case_spec & c, ggml_backend_t backend, metal_case & m, int chain_k) {
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*4096 + ggml_graph_overhead()*4,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    m.ctx = ggml_init(ip);
    if (!m.ctx) {
        return false;
    }

    m.x    = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F32, c.n_embd);
    m.gate = ggml_new_tensor_3d(m.ctx, c.type_gate_up, c.n_embd, c.n_ff, c.n_slots);
    m.up   = ggml_new_tensor_3d(m.ctx, c.type_gate_up, c.n_embd, c.n_ff, c.n_slots);
    m.down = ggml_new_tensor_3d(m.ctx, c.type_down,    c.n_ff, c.n_embd, c.n_slots);
    m.ids  = ggml_new_tensor_2d(m.ctx, GGML_TYPE_I32, c.n_used, 1);
    m.w    = ggml_new_tensor_3d(m.ctx, GGML_TYPE_F32, 1, c.n_used, 1);
    ggml_set_name(m.x, "x"); ggml_set_name(m.gate, "gate"); ggml_set_name(m.up, "up");
    ggml_set_name(m.down, "down"); ggml_set_name(m.ids, "ids"); ggml_set_name(m.w, "w");

    // --slot8 operator
    m.out_op = ggml_flashmoe_slot8_ffn(m.ctx, m.x, m.gate, m.up, m.down, m.ids, m.w);
    ggml_set_name(m.out_op, "out_op");
    m.gf_op = ggml_new_graph(m.ctx);
    ggml_build_forward_expand(m.gf_op, m.out_op);

    // generic path: the same graph build_moe_ffn emits without --slot8
    m.out_generic = build_generic_ffn(m.ctx, c, m, m.x);
    ggml_set_name(m.out_generic, "out_generic");
    m.gf_generic = ggml_new_graph(m.ctx);
    ggml_build_forward_expand(m.gf_generic, m.out_generic);

    m.chain_k = chain_k;
    if (chain_k > 1) {
        ggml_tensor * cur = m.x;
        for (int k = 0; k < chain_k; ++k) {
            cur = ggml_flashmoe_slot8_ffn(m.ctx, cur, m.gate, m.up, m.down, m.ids, m.w);
        }
        m.out_chain_op = cur;
        m.gf_chain_op = ggml_new_graph(m.ctx);
        ggml_build_forward_expand(m.gf_chain_op, m.out_chain_op);

        cur = m.x;
        for (int k = 0; k < chain_k; ++k) {
            cur = build_generic_ffn(m.ctx, c, m, cur);
        }
        m.out_chain_generic = cur;
        m.gf_chain_generic = ggml_new_graph(m.ctx);
        ggml_build_forward_expand(m.gf_chain_generic, m.out_chain_generic);
    }

    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, backend);
    return m.buf != nullptr;
}

static bool run_graph(ggml_backend_t backend, ggml_cgraph * gf, ggml_tensor * out, std::vector<float> & res) {
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    res.resize(size_t(ggml_nelements(out)));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
    return true;
}

static double time_graph_ms(ggml_backend_t backend, ggml_cgraph * gf, int iters) {
    ggml_backend_graph_compute(backend, gf); // warm-up (pipeline compile, page-in)
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static void print_stats(const char * label, const err_stats & st) {
    printf("    %-22s max_abs=%.3e max_rel=%.3e (max|exact|=%.3e)\n", label, st.max_abs, st.max_rel, st.ref_max);
}

static bool run_case(const case_spec & c, ggml_backend_t backend, int bench_iters) {
    printf("case %s: gate/up=%s down=%s n_embd=%" PRId64 " n_ff=%" PRId64 " n_slots=%" PRId64 " n_used=%" PRId64 "\n",
            c.name.c_str(), ggml_type_name(c.type_gate_up), ggml_type_name(c.type_down),
            c.n_embd, c.n_ff, c.n_slots, c.n_used);

    std::mt19937 rng(c.seed);

    const size_t gu_bytes = ggml_row_size(c.type_gate_up, c.n_embd) * size_t(c.n_ff) * size_t(c.n_slots);
    const size_t dn_bytes = ggml_row_size(c.type_down, c.n_ff) * size_t(c.n_embd) * size_t(c.n_slots);

    std::vector<uint8_t> gate(gu_bytes), up(gu_bytes), down(dn_bytes);
    fill_random_blocks(c.type_gate_up, gate.data(), gu_bytes, rng);
    fill_random_blocks(c.type_gate_up, up.data(),   gu_bytes, rng);
    fill_random_blocks(c.type_down,    down.data(), dn_bytes, rng);

    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(size_t(c.n_embd));
    for (auto & v : x) {
        v = gauss(rng);
    }

    // distinct resident slots in random order, routing weights normalised to 1
    std::vector<int32_t> slots(size_t(c.n_slots));
    for (int64_t i = 0; i < c.n_slots; ++i) {
        slots[size_t(i)] = int32_t(i);
    }
    std::shuffle(slots.begin(), slots.end(), rng);
    slots.resize(size_t(c.n_used));

    std::uniform_real_distribution<float> uni(0.05f, 1.0f);
    std::vector<float> w(size_t(c.n_used));
    float wsum = 0.0f;
    for (auto & v : w) {
        v = uni(rng);
        wsum += v;
    }
    for (auto & v : w) {
        v /= wsum;
    }

    std::vector<double> exact;
    eval_exact(c, gate, up, down, x, slots, w, exact);

    metal_case m;
    if (!build_metal_case(c, backend, m, bench_iters > 0 ? 16 : 0)) {
        printf("  FAILED: could not build/allocate the Metal case\n");
        return false;
    }

    ggml_backend_tensor_set(m.x,    x.data(),     0, ggml_nbytes(m.x));
    ggml_backend_tensor_set(m.gate, gate.data(),  0, ggml_nbytes(m.gate));
    ggml_backend_tensor_set(m.up,   up.data(),    0, ggml_nbytes(m.up));
    ggml_backend_tensor_set(m.down, down.data(),  0, ggml_nbytes(m.down));
    ggml_backend_tensor_set(m.ids,  slots.data(), 0, ggml_nbytes(m.ids));
    ggml_backend_tensor_set(m.w,    w.data(),     0, ggml_nbytes(m.w));

    std::vector<float> fused, reference, generic;

    unsetenv("LLAMA_FLASH_MOE_SLOT8_REFERENCE");
    if (!run_graph(backend, m.gf_op, m.out_op, fused)) {
        printf("  FAILED: fused graph compute\n");
        return false;
    }

    setenv("LLAMA_FLASH_MOE_SLOT8_REFERENCE", "1", 1);
    const bool ref_ok = run_graph(backend, m.gf_op, m.out_op, reference);
    unsetenv("LLAMA_FLASH_MOE_SLOT8_REFERENCE");
    if (!ref_ok) {
        printf("  FAILED: reference graph compute\n");
        return false;
    }

    if (!run_graph(backend, m.gf_generic, m.out_generic, generic)) {
        printf("  FAILED: generic graph compute\n");
        return false;
    }

    const err_stats st_fused     = compare_to(fused,     exact);
    const err_stats st_reference = compare_to(reference, exact);
    const err_stats st_generic   = compare_to(generic,   exact);
    const err_stats st_fused_ref = compare_ff(fused, reference);
    const err_stats st_fused_gen = compare_ff(fused, generic);

    print_stats("fused vs exact",       st_fused);
    print_stats("reference vs exact",   st_reference);
    print_stats("generic vs exact",     st_generic);
    print_stats("fused vs reference",   st_fused_ref);
    print_stats("fused vs generic",     st_fused_gen);

    bool ok = true;
    ok &= st_fused.max_rel     <= kMaxRelTol;
    ok &= st_reference.max_rel <= kMaxRelTol;
    ok &= st_generic.max_rel   <= kMaxRelTol;
    ok &= st_fused_ref.max_rel <= kMaxRelTol;

    if (bench_iters > 0) {
        const double fused_ms = time_graph_ms(backend, m.gf_op, bench_iters);
        setenv("LLAMA_FLASH_MOE_SLOT8_REFERENCE", "1", 1);
        const double reference_ms = time_graph_ms(backend, m.gf_op, bench_iters);
        unsetenv("LLAMA_FLASH_MOE_SLOT8_REFERENCE");
        const double generic_ms = time_graph_ms(backend, m.gf_generic, bench_iters);
        printf("    timing (%d iters, wall incl. submit): fused=%.3f ms reference=%.3f ms generic=%.3f ms\n",
                bench_iters, fused_ms, reference_ms, generic_ms);

        if (m.chain_k > 1) {
            const double fused_chain_ms = time_graph_ms(backend, m.gf_chain_op, bench_iters);
            setenv("LLAMA_FLASH_MOE_SLOT8_REFERENCE", "1", 1);
            const double reference_chain_ms = time_graph_ms(backend, m.gf_chain_op, bench_iters);
            unsetenv("LLAMA_FLASH_MOE_SLOT8_REFERENCE");
            const double generic_chain_ms = time_graph_ms(backend, m.gf_chain_generic, bench_iters);
            const double k1 = m.chain_k - 1;
            printf("    per-op GPU time from %d chained ops (submit cost removed): fused=%.3f ms reference=%.3f ms generic=%.3f ms\n",
                    m.chain_k,
                    (fused_chain_ms - fused_ms) / k1,
                    (reference_chain_ms - reference_ms) / k1,
                    (generic_chain_ms - generic_ms) / k1);
        }
    }

    printf("  %s\n", ok ? "OK" : "FAILED");
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    int bench_iters = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            bench_iters = atoi(argv[++i]);
        }
    }

    ggml_backend_load_all();

    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        if (strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "MTL") != 0) {
            continue;
        }
        backend = ggml_backend_dev_init(dev, nullptr);
        break;
    }

    if (backend == nullptr) {
        printf("SKIPPED: no usable Metal device (the --slot8 operator is Metal-only)\n");
        return 0;
    }

    const std::vector<case_spec> cases = {
        // small shapes: every HY4 combination
        { "small-stq1_0-iq3_xxs",   GGML_TYPE_STQ1_0,  GGML_TYPE_IQ3_XXS,  512,  768, 12,  8, 101 },
        { "small-stq1_0-iq4_xs",    GGML_TYPE_STQ1_0,  GGML_TYPE_IQ4_XS,   512,  768, 12,  8, 102 },
        { "small-iq2_xxs-iq3_xxs",  GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS,  512,  768, 12,  8, 103 },
        { "small-iq2_xxs-iq4_xs",   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_XS,   512,  768, 12,  8, 104 },
        // width is carried in n_used, not hard-coded
        { "small-stq1_0-iq3_xxs-top10", GGML_TYPE_STQ1_0, GGML_TYPE_IQ3_XXS, 512, 768, 12, 10, 105 },
        { "small-iq2_xxs-iq4_xs-top4",  GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_XS, 512, 768, 12,  4, 106 },
        // real HY4 routed shape: gate/up [6144, 2048], down [2048, 6144], native top-8
        { "hy4-stq1_0-iq3_xxs",     GGML_TYPE_STQ1_0,  GGML_TYPE_IQ3_XXS, 6144, 2048, 10,  8, 201 },
        { "hy4-stq1_0-iq4_xs",      GGML_TYPE_STQ1_0,  GGML_TYPE_IQ4_XS,  6144, 2048, 10,  8, 202 },
        { "hy4-iq2_xxs-iq3_xxs",    GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, 6144, 2048, 10,  8, 203 },
        { "hy4-iq2_xxs-iq4_xs",     GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ4_XS,  6144, 2048, 10,  8, 204 },
    };

    int failures = 0;
    for (const auto & c : cases) {
        const bool is_hy4_shape = c.name.rfind("hy4-", 0) == 0;
        if (!run_case(c, backend, is_hy4_shape ? bench_iters : 0)) {
            ++failures;
        }
    }

    ggml_backend_free(backend);

    printf("%d/%zu cases passed\n", int(cases.size()) - failures, cases.size());
    return failures == 0 ? 0 : 1;
}
