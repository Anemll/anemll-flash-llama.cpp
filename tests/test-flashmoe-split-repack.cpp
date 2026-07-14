#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int32_t kMinVisibleCols = 9;
constexpr float kMaxAbsTol = 3e-3f;
constexpr float kMeanAbsTol = 5e-4f;

struct split_repack_case {
    std::string name;
    bool merged_gate_up = false;
    int64_t n_embd = 0;
    int64_t n_ff = 0;
    int32_t tokens = 0;
    int64_t gate_align = 0;
    int64_t up_align = 0;
    int64_t down_align = 0;
    float output_scale = 1.0f;
    int32_t seed = 0;
};

struct diff_stats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
};

static int64_t gcd_i64(int64_t a, int64_t b) {
    a = std::llabs(a);
    b = std::llabs(b);
    while (b != 0) {
        const int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static int64_t lcm_i64(int64_t a, int64_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return (a / gcd_i64(a, b)) * b;
}

static int32_t pick_split_slices(
        int64_t width,
        int32_t logical_cols,
        int64_t align_elems,
        int32_t min_visible_cols) {
    if (width <= 0 || logical_cols <= 0 || logical_cols > 8 || align_elems <= 0) {
        return 0;
    }

    if (width % align_elems != 0) {
        return 0;
    }

    const int64_t units = width / align_elems;
    int32_t best = 0;
    for (int64_t slices = 2; slices <= units; ++slices) {
        if (units % slices != 0) {
            continue;
        }
        if (logical_cols * slices <= min_visible_cols - 1) {
            continue;
        }
        best = int32_t(slices);
        break;
    }

    return best;
}

static float silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

static float gelu_f32(float x) {
    static constexpr float sq2_over_pi = 0.7978845608028654f;
    static constexpr float coef_a = 0.044715f;
    return 0.5f * x * (1.0f + std::tanh(sq2_over_pi * x * (1.0f + coef_a * x * x)));
}

static void fill_random(std::vector<float> & data, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-0.75f, 0.75f);
    for (float & value : data) {
        value = dist(rng);
    }
}

static float dot_product(const float * a, const float * b, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

static std::vector<float> mul_mat_token_major(
        const std::vector<float> & weights,
        int64_t rows_out,
        int64_t rows_in,
        const std::vector<float> & input,
        int32_t tokens) {
    std::vector<float> out(size_t(rows_out * tokens), 0.0f);

    for (int32_t token = 0; token < tokens; ++token) {
        const float * x = input.data() + size_t(token) * size_t(rows_in);
        float * dst = out.data() + size_t(token) * size_t(rows_out);
        for (int64_t row = 0; row < rows_out; ++row) {
            const float * w = weights.data() + size_t(row) * size_t(rows_in);
            dst[row] = dot_product(w, x, rows_in);
        }
    }

    return out;
}

static void pack_token_slices(
        const std::vector<float> & input,
        int32_t tokens,
        int64_t width,
        int32_t slices,
        std::vector<float> & packed) {
    const int64_t slice_width = width / slices;
    packed.assign(size_t(tokens) * size_t(slices) * size_t(slice_width), 0.0f);

    for (int32_t token = 0; token < tokens; ++token) {
        const float * src_row = input.data() + size_t(token) * size_t(width);
        for (int32_t slice = 0; slice < slices; ++slice) {
            std::copy_n(
                    src_row + size_t(slice) * size_t(slice_width),
                    size_t(slice_width),
                    packed.data() + size_t(token * slices + slice) * size_t(slice_width));
        }
    }
}

static void accumulate_slice(
        const std::vector<float> & weights,
        int64_t rows_out,
        int64_t full_width,
        int64_t slice_width,
        int32_t slice,
        const float * packed_input,
        float * dst) {
    const size_t weight_offset = size_t(slice) * size_t(slice_width);
    for (int64_t row = 0; row < rows_out; ++row) {
        const float * w = weights.data() + size_t(row) * size_t(full_width) + weight_offset;
        dst[row] += dot_product(w, packed_input, slice_width);
    }
}

static std::vector<float> baseline_separate(
        const split_repack_case & cfg,
        const std::vector<float> & input,
        const std::vector<float> & gate_w,
        const std::vector<float> & up_w,
        const std::vector<float> & down_w) {
    std::vector<float> gate = mul_mat_token_major(gate_w, cfg.n_ff, cfg.n_embd, input, cfg.tokens);
    std::vector<float> up = mul_mat_token_major(up_w, cfg.n_ff, cfg.n_embd, input, cfg.tokens);
    std::vector<float> act(size_t(cfg.tokens) * size_t(cfg.n_ff), 0.0f);

    for (int32_t token = 0; token < cfg.tokens; ++token) {
        const float * gate_row = gate.data() + size_t(token) * size_t(cfg.n_ff);
        const float * up_row = up.data() + size_t(token) * size_t(cfg.n_ff);
        float * act_row = act.data() + size_t(token) * size_t(cfg.n_ff);
        for (int64_t row = 0; row < cfg.n_ff; ++row) {
            act_row[row] = silu_f32(gate_row[row]) * up_row[row];
        }
    }

    std::vector<float> out = mul_mat_token_major(down_w, cfg.n_embd, cfg.n_ff, act, cfg.tokens);
    for (float & value : out) {
        value *= cfg.output_scale;
    }
    return out;
}

static std::vector<float> baseline_merged(
        const split_repack_case & cfg,
        const std::vector<float> & input,
        const std::vector<float> & gate_up_w,
        const std::vector<float> & down_w) {
    std::vector<float> gate_up = mul_mat_token_major(gate_up_w, 2 * cfg.n_ff, cfg.n_embd, input, cfg.tokens);
    std::vector<float> act(size_t(cfg.tokens) * size_t(cfg.n_ff), 0.0f);

    for (int32_t token = 0; token < cfg.tokens; ++token) {
        const float * gate_row = gate_up.data() + size_t(token) * size_t(2 * cfg.n_ff);
        const float * up_row = gate_row + cfg.n_ff;
        float * act_row = act.data() + size_t(token) * size_t(cfg.n_ff);
        for (int64_t row = 0; row < cfg.n_ff; ++row) {
            act_row[row] = gelu_f32(gate_row[row]) * up_row[row];
        }
    }

    std::vector<float> out = mul_mat_token_major(down_w, cfg.n_embd, cfg.n_ff, act, cfg.tokens);
    for (float & value : out) {
        value *= cfg.output_scale;
    }
    return out;
}

static std::vector<float> split_repack_separate(
        const split_repack_case & cfg,
        const std::vector<float> & input,
        const std::vector<float> & gate_w,
        const std::vector<float> & up_w,
        const std::vector<float> & down_w) {
    const int64_t gate_align = lcm_i64(cfg.gate_align, cfg.up_align);
    const int32_t gate_slices = pick_split_slices(cfg.n_embd, cfg.tokens, gate_align, kMinVisibleCols);
    const int32_t down_slices = pick_split_slices(cfg.n_ff, cfg.tokens, cfg.down_align, kMinVisibleCols);
    if (gate_slices < 2 || down_slices < 2) {
        throw std::runtime_error("split_repack_separate: expected both stages to split");
    }

    const int64_t gate_slice_width = cfg.n_embd / gate_slices;
    std::vector<float> gate_packed;
    pack_token_slices(input, cfg.tokens, cfg.n_embd, gate_slices, gate_packed);

    std::vector<float> gate_acc(size_t(cfg.tokens) * size_t(cfg.n_ff), 0.0f);
    std::vector<float> up_acc(size_t(cfg.tokens) * size_t(cfg.n_ff), 0.0f);
    for (int32_t slice = 0; slice < gate_slices; ++slice) {
        for (int32_t token = 0; token < cfg.tokens; ++token) {
            const float * packed_input = gate_packed.data() + size_t(token * gate_slices + slice) * size_t(gate_slice_width);
            accumulate_slice(gate_w, cfg.n_ff, cfg.n_embd, gate_slice_width, slice, packed_input,
                    gate_acc.data() + size_t(token) * size_t(cfg.n_ff));
            accumulate_slice(up_w, cfg.n_ff, cfg.n_embd, gate_slice_width, slice, packed_input,
                    up_acc.data() + size_t(token) * size_t(cfg.n_ff));
        }
    }

    std::vector<float> act(size_t(cfg.tokens) * size_t(cfg.n_ff), 0.0f);
    for (int32_t token = 0; token < cfg.tokens; ++token) {
        const float * gate_row = gate_acc.data() + size_t(token) * size_t(cfg.n_ff);
        const float * up_row = up_acc.data() + size_t(token) * size_t(cfg.n_ff);
        float * act_row = act.data() + size_t(token) * size_t(cfg.n_ff);
        for (int64_t row = 0; row < cfg.n_ff; ++row) {
            act_row[row] = silu_f32(gate_row[row]) * up_row[row];
        }
    }

    const int64_t down_slice_width = cfg.n_ff / down_slices;
    std::vector<float> down_packed;
    pack_token_slices(act, cfg.tokens, cfg.n_ff, down_slices, down_packed);

    std::vector<float> out(size_t(cfg.tokens) * size_t(cfg.n_embd), 0.0f);
    for (int32_t slice = 0; slice < down_slices; ++slice) {
        for (int32_t token = 0; token < cfg.tokens; ++token) {
            const float * packed_input = down_packed.data() + size_t(token * down_slices + slice) * size_t(down_slice_width);
            accumulate_slice(down_w, cfg.n_embd, cfg.n_ff, down_slice_width, slice, packed_input,
                    out.data() + size_t(token) * size_t(cfg.n_embd));
        }
    }

    for (float & value : out) {
        value *= cfg.output_scale;
    }
    return out;
}

static std::vector<float> split_repack_separate_stage1_only(
        const split_repack_case & cfg,
        const std::vector<float> & input,
        const std::vector<float> & gate_w,
        const std::vector<float> & up_w,
        const std::vector<float> & down_w) {
    const int64_t gate_align = lcm_i64(cfg.gate_align, cfg.up_align);
    const int32_t gate_slices = pick_split_slices(cfg.n_embd, cfg.tokens, gate_align, kMinVisibleCols);
    if (gate_slices < 2) {
        throw std::runtime_error("split_repack_separate_stage1_only: expected stage1 to split");
    }

    const int32_t down_slices = pick_split_slices(cfg.n_ff, cfg.tokens, cfg.down_align, kMinVisibleCols);
    if (down_slices >= 2) {
        throw std::runtime_error("split_repack_separate_stage1_only: expected stage2 to stay unsplit");
    }

    const int64_t gate_slice_width = cfg.n_embd / gate_slices;
    std::vector<float> gate_packed;
    pack_token_slices(input, cfg.tokens, cfg.n_embd, gate_slices, gate_packed);

    std::vector<float> gate_acc(size_t(cfg.tokens) * size_t(cfg.n_ff), 0.0f);
    std::vector<float> up_acc(size_t(cfg.tokens) * size_t(cfg.n_ff), 0.0f);
    for (int32_t slice = 0; slice < gate_slices; ++slice) {
        for (int32_t token = 0; token < cfg.tokens; ++token) {
            const float * packed_input = gate_packed.data() + size_t(token * gate_slices + slice) * size_t(gate_slice_width);
            accumulate_slice(gate_w, cfg.n_ff, cfg.n_embd, gate_slice_width, slice, packed_input,
                    gate_acc.data() + size_t(token) * size_t(cfg.n_ff));
            accumulate_slice(up_w, cfg.n_ff, cfg.n_embd, gate_slice_width, slice, packed_input,
                    up_acc.data() + size_t(token) * size_t(cfg.n_ff));
        }
    }

    std::vector<float> act(size_t(cfg.tokens) * size_t(cfg.n_ff), 0.0f);
    for (int32_t token = 0; token < cfg.tokens; ++token) {
        const float * gate_row = gate_acc.data() + size_t(token) * size_t(cfg.n_ff);
        const float * up_row = up_acc.data() + size_t(token) * size_t(cfg.n_ff);
        float * act_row = act.data() + size_t(token) * size_t(cfg.n_ff);
        for (int64_t row = 0; row < cfg.n_ff; ++row) {
            act_row[row] = silu_f32(gate_row[row]) * up_row[row];
        }
    }

    std::vector<float> out = mul_mat_token_major(down_w, cfg.n_embd, cfg.n_ff, act, cfg.tokens);
    for (float & value : out) {
        value *= cfg.output_scale;
    }
    return out;
}

static std::vector<float> split_repack_merged(
        const split_repack_case & cfg,
        const std::vector<float> & input,
        const std::vector<float> & gate_up_w,
        const std::vector<float> & down_w) {
    const int32_t gate_slices = pick_split_slices(cfg.n_embd, cfg.tokens, cfg.gate_align, kMinVisibleCols);
    const int32_t down_slices = pick_split_slices(cfg.n_ff, cfg.tokens, cfg.down_align, kMinVisibleCols);
    if (gate_slices < 2 || down_slices < 2) {
        throw std::runtime_error("split_repack_merged: expected both stages to split");
    }

    const int64_t gate_slice_width = cfg.n_embd / gate_slices;
    std::vector<float> gate_packed;
    pack_token_slices(input, cfg.tokens, cfg.n_embd, gate_slices, gate_packed);

    std::vector<float> gate_up_acc(size_t(cfg.tokens) * size_t(2 * cfg.n_ff), 0.0f);
    for (int32_t slice = 0; slice < gate_slices; ++slice) {
        for (int32_t token = 0; token < cfg.tokens; ++token) {
            const float * packed_input = gate_packed.data() + size_t(token * gate_slices + slice) * size_t(gate_slice_width);
            accumulate_slice(gate_up_w, 2 * cfg.n_ff, cfg.n_embd, gate_slice_width, slice, packed_input,
                    gate_up_acc.data() + size_t(token) * size_t(2 * cfg.n_ff));
        }
    }

    std::vector<float> act(size_t(cfg.tokens) * size_t(cfg.n_ff), 0.0f);
    for (int32_t token = 0; token < cfg.tokens; ++token) {
        const float * gate_row = gate_up_acc.data() + size_t(token) * size_t(2 * cfg.n_ff);
        const float * up_row = gate_row + cfg.n_ff;
        float * act_row = act.data() + size_t(token) * size_t(cfg.n_ff);
        for (int64_t row = 0; row < cfg.n_ff; ++row) {
            act_row[row] = gelu_f32(gate_row[row]) * up_row[row];
        }
    }

    const int64_t down_slice_width = cfg.n_ff / down_slices;
    std::vector<float> down_packed;
    pack_token_slices(act, cfg.tokens, cfg.n_ff, down_slices, down_packed);

    std::vector<float> out(size_t(cfg.tokens) * size_t(cfg.n_embd), 0.0f);
    for (int32_t slice = 0; slice < down_slices; ++slice) {
        for (int32_t token = 0; token < cfg.tokens; ++token) {
            const float * packed_input = down_packed.data() + size_t(token * down_slices + slice) * size_t(down_slice_width);
            accumulate_slice(down_w, cfg.n_embd, cfg.n_ff, down_slice_width, slice, packed_input,
                    out.data() + size_t(token) * size_t(cfg.n_embd));
        }
    }

    for (float & value : out) {
        value *= cfg.output_scale;
    }
    return out;
}

static diff_stats compare_outputs(
        const std::vector<float> & baseline,
        const std::vector<float> & repacked) {
    if (baseline.size() != repacked.size()) {
        throw std::runtime_error("compare_outputs: size mismatch");
    }

    diff_stats stats;
    double sum_abs = 0.0;
    for (size_t i = 0; i < baseline.size(); ++i) {
        const float diff = std::fabs(baseline[i] - repacked[i]);
        stats.max_abs = std::max(stats.max_abs, diff);
        sum_abs += diff;
    }
    stats.mean_abs = baseline.empty() ? 0.0f : float(sum_abs / double(baseline.size()));
    return stats;
}

static bool verify_real_alignment_slice_picking() {
    bool ok = true;
    const int64_t qk_align = ggml_blck_size(GGML_TYPE_Q4_K);
    const int64_t down_align = ggml_blck_size(GGML_TYPE_Q6_K);
    const int64_t width = 9 * qk_align;
    const int64_t ff_width = 9 * down_align;

    for (int32_t tokens = 1; tokens <= 8; ++tokens) {
        const int32_t gate_slices = pick_split_slices(width, tokens, qk_align, kMinVisibleCols);
        const int32_t down_slices = pick_split_slices(ff_width, tokens, down_align, kMinVisibleCols);
        if (gate_slices < 2 || tokens * gate_slices < kMinVisibleCols) {
            std::fprintf(stderr,
                    "real-alignment gate split selection failed for tokens=%d slices=%d align=%lld width=%lld\n",
                    tokens, gate_slices, (long long) qk_align, (long long) width);
            ok = false;
        }
        if (down_slices < 2 || tokens * down_slices < kMinVisibleCols) {
            std::fprintf(stderr,
                    "real-alignment down split selection failed for tokens=%d slices=%d align=%lld width=%lld\n",
                    tokens, down_slices, (long long) down_align, (long long) ff_width);
            ok = false;
        }
    }

    return ok;
}

static bool run_split_repack_case(const split_repack_case & cfg) {
    std::mt19937 rng(cfg.seed);

    std::vector<float> input(size_t(cfg.tokens) * size_t(cfg.n_embd));
    std::vector<float> down_w(size_t(cfg.n_embd) * size_t(cfg.n_ff));
    fill_random(input, rng);
    fill_random(down_w, rng);

    diff_stats stats;
    if (cfg.merged_gate_up) {
        std::vector<float> gate_up_w(size_t(2 * cfg.n_ff) * size_t(cfg.n_embd));
        fill_random(gate_up_w, rng);
        const std::vector<float> baseline = baseline_merged(cfg, input, gate_up_w, down_w);
        const std::vector<float> repacked = split_repack_merged(cfg, input, gate_up_w, down_w);
        stats = compare_outputs(baseline, repacked);
    } else {
        std::vector<float> gate_w(size_t(cfg.n_ff) * size_t(cfg.n_embd));
        std::vector<float> up_w(size_t(cfg.n_ff) * size_t(cfg.n_embd));
        fill_random(gate_w, rng);
        fill_random(up_w, rng);
        const std::vector<float> baseline = baseline_separate(cfg, input, gate_w, up_w, down_w);
        const std::vector<float> repacked = split_repack_separate(cfg, input, gate_w, up_w, down_w);
        stats = compare_outputs(baseline, repacked);
    }

    const bool ok = stats.max_abs <= kMaxAbsTol && stats.mean_abs <= kMeanAbsTol;
    if (!ok) {
        std::fprintf(stderr,
                "split-repack mismatch case=%s tokens=%d max_abs=%g mean_abs=%g\n",
                cfg.name.c_str(), cfg.tokens, stats.max_abs, stats.mean_abs);
    }

    return ok;
}

static bool run_emulation_suite() {
    bool ok = true;

    for (int32_t tokens = 1; tokens <= 8; ++tokens) {
        ok = run_split_repack_case({
                /*.name            =*/ "split",
                /*.merged_gate_up  =*/ false,
                /*.n_embd          =*/ 1152,
                /*.n_ff            =*/ 1152,
                /*.tokens          =*/ tokens,
                /*.gate_align      =*/ 64,
                /*.up_align        =*/ 128,
                /*.down_align      =*/ 64,
                /*.output_scale    =*/ 0.875f,
                /*.seed            =*/ 100 + tokens,
        }) && ok;

        ok = run_split_repack_case({
                /*.name            =*/ "merged",
                /*.merged_gate_up  =*/ true,
                /*.n_embd          =*/ 1152,
                /*.n_ff            =*/ 1152,
                /*.tokens          =*/ tokens,
                /*.gate_align      =*/ 128,
                /*.up_align        =*/ 128,
                /*.down_align      =*/ 128,
                /*.output_scale    =*/ 1.125f,
                /*.seed            =*/ 200 + tokens,
        }) && ok;
    }

    {
        const split_repack_case cfg = {
                /*.name            =*/ "stage1_only_minimax_like",
                /*.merged_gate_up  =*/ false,
                /*.n_embd          =*/ 3072,
                /*.n_ff            =*/ 1536,
                /*.tokens          =*/ 1,
                /*.gate_align      =*/ 256,
                /*.up_align        =*/ 256,
                /*.down_align      =*/ 256,
                /*.output_scale    =*/ 1.0f,
                /*.seed            =*/ 301,
        };

        std::mt19937 rng(cfg.seed);
        std::vector<float> input(size_t(cfg.tokens) * size_t(cfg.n_embd));
        std::vector<float> gate_w(size_t(cfg.n_ff) * size_t(cfg.n_embd));
        std::vector<float> up_w(size_t(cfg.n_ff) * size_t(cfg.n_embd));
        std::vector<float> down_w(size_t(cfg.n_embd) * size_t(cfg.n_ff));
        fill_random(input, rng);
        fill_random(gate_w, rng);
        fill_random(up_w, rng);
        fill_random(down_w, rng);

        const std::vector<float> baseline = baseline_separate(cfg, input, gate_w, up_w, down_w);
        const std::vector<float> repacked = split_repack_separate_stage1_only(cfg, input, gate_w, up_w, down_w);
        const diff_stats stats = compare_outputs(baseline, repacked);
        const float stage1_only_max_abs_tol = 1e-2f;
        const float stage1_only_mean_abs_tol = 2e-3f;
        if (!(stats.max_abs <= stage1_only_max_abs_tol && stats.mean_abs <= stage1_only_mean_abs_tol)) {
            std::fprintf(stderr,
                    "split-repack stage1-only mismatch case=%s tokens=%d max_abs=%g mean_abs=%g\n",
                    cfg.name.c_str(), cfg.tokens, stats.max_abs, stats.mean_abs);
            ok = false;
        }
    }

    return ok;
}

}  // namespace

int main() {
    const bool align_ok = verify_real_alignment_slice_picking();
    const bool suite_ok = run_emulation_suite();

    if (!align_ok || !suite_ok) {
        return 1;
    }

    std::printf("split-repack emulation checks passed\n");
    return 0;
}
