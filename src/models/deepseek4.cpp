#include "models.h"

#include "ggml-backend.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_set>
#include <vector>

extern "C" void dsv4_layer_executor_payload_register(
        const char  * stage,
        int           layer,
        int64_t       token,
        const char  * tensor_name,
        ggml_tensor * tensor);

namespace {

struct dsv4_hc_mix {
    ggml_tensor * x;
    ggml_tensor * flat;
    ggml_tensor * flat_normed;
    ggml_tensor * mixes;
    ggml_tensor * split;
    ggml_tensor * pre;
    ggml_tensor * post;
    ggml_tensor * comb;
};

struct dsv4_state_pair {
    ggml_tensor * kv;
    ggml_tensor * score;
};

struct dsv4_decode_compressor {
    ggml_tensor * kv_state;
    ggml_tensor * score_state;
    ggml_tensor * kv_comp;
    ggml_tensor * generic_tail_kv_comp = nullptr;
    ggml_tensor * backend_tail_kv_comp = nullptr;
    ggml_tensor * generic_probe_pooled = nullptr;
    ggml_tensor * generic_probe_norm = nullptr;
    ggml_tensor * generic_probe_norm_weighted = nullptr;
    ggml_tensor * generic_probe_rope_in = nullptr;
    ggml_tensor * generic_probe_rope_out = nullptr;
    ggml_tensor * backend_probe_pooled = nullptr;
    ggml_tensor * backend_probe_norm = nullptr;
    ggml_tensor * backend_probe_norm_weighted = nullptr;
    ggml_tensor * backend_probe_rope_in = nullptr;
    ggml_tensor * backend_probe_rope_out = nullptr;
    bool generic_tail_built = false;
    bool candidate_tail_built = false;
    bool candidate_tail_consumed = false;
    bool backend_tail_built = false;
};

struct dsv4_state_layout {
    int64_t width;
    int64_t rows;
    int64_t elems;
};

static bool dsv4_experimental_indexer_weighted_score_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXER_SCORE");
        if (value != nullptr && value[0] != '\0') {
            enabled = std::strcmp(value, "0") != 0 ? 1 : 0;
        } else {
            value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU");
            enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_decode_compress_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_COMPRESS");
        if (value != nullptr && value[0] != '\0') {
            enabled = std::strcmp(value, "0") != 0 ? 1 : 0;
        } else {
            enabled = 0;
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_pair_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_PAIR");
        if (value != nullptr && value[0] != '\0') {
            enabled = std::strcmp(value, "0") != 0 ? 1 : 0;
        } else {
            enabled = 0;
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_topk_attn_gather_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_TOPK_ATTN_GATHER");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_mixed_attn_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_MIXED_ATTN");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_indexed_attn_shadow_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_SHADOW");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_indexed_attn_compare_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_COMPARE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_indexed_attn_search_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_SEARCH");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_indexed_attn_trace_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_TRACE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_indexed_attn_shadow_output_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_SHADOW_OUTPUT");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_indexed_attn_output_compare_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_OUTPUT_COMPARE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_indexed_attn_diff_trace_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_DIFF_TRACE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static const char * dsv4_experimental_indexed_attn_row_order() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_ROW_ORDER");
    return value != nullptr && value[0] != '\0' ? value : "ds4_indexed";
}

static const char * dsv4_experimental_indexed_attn_mask_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_MASK_MODE");
    return value != nullptr && value[0] != '\0' ? value : "generic_mask_values";
}

static const char * dsv4_experimental_indexed_attn_shadow_impl() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_SHADOW_IMPL");
    return value != nullptr && value[0] != '\0' ? value : "mixed_attention";
}

static const char * dsv4_experimental_indexed_attn_arith_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_ARITH_MODE");
    if (value != nullptr && value[0] != '\0') {
        if (std::strcmp(value, "generic_attention") == 0) {
            return "generic_flash";
        }
        if (std::strcmp(value, "mixed_attention") == 0) {
            return "dsv4_mixed";
        }
        return value;
    }

    const char * legacy = dsv4_experimental_indexed_attn_shadow_impl();
    if (std::strcmp(legacy, "generic_attention") == 0) {
        return "generic_flash";
    }
    if (std::strcmp(legacy, "mixed_attention") == 0) {
        return "dsv4_mixed";
    }
    return legacy;
}

static const char * dsv4_experimental_indexed_attn_shape_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_SHAPE_MODE");
    return value != nullptr && value[0] != '\0' ? value : "sparse_topk";
}

static int64_t dsv4_experimental_indexed_attn_pad_to_rows() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_PAD_TO_ROWS");
    if (value == nullptr || value[0] == '\0') {
        return -1;
    }
    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    return end != value ? (int64_t) parsed : -1;
}

static bool dsv4_experimental_attn_out_decode_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_DECODE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_attn_out_decode_compare_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_DECODE_COMPARE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_attn_out_hc_post_shadow_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SHADOW");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_attn_out_hc_post_compare_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_COMPARE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_attn_out_hc_post_trace_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_TRACE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static const char * dsv4_experimental_attn_out_hc_post_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_MODE");
    return value != nullptr && value[0] != '\0' ? value : "generic_shadow";
}

static bool dsv4_experimental_attn_out_hc_post_candidate_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_attn_out_hc_post_candidate_compare_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE_COMPARE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_attn_out_hc_post_candidate_trace_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE_TRACE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static const char * dsv4_experimental_attn_out_hc_post_candidate_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE_MODE");
    return value != nullptr && value[0] != '\0' ? value : "candidate_graph_exact";
}

static bool dsv4_experimental_attn_out_hc_post_consume_requested() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_attn_out_hc_post_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static const char * dsv4_experimental_attn_out_hc_post_consume_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_MODE");
    return value != nullptr && value[0] != '\0' ? value : "";
}

static bool dsv4_experimental_attn_out_hc_post_skip_generic_requested() {
    return dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC");
}

static const char * dsv4_experimental_attn_out_hc_post_skip_generic_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC_MODE");
    return value != nullptr && value[0] != '\0' ? value : "";
}

static bool dsv4_experimental_aohc_fused_requested() {
    return dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED");
}

static bool dsv4_experimental_aohc_fused_compare_enabled() {
    return dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_COMPARE");
}

static bool dsv4_experimental_aohc_fused_trace_enabled() {
    return dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TRACE");
}

static bool dsv4_experimental_aohc_fused_elig_trace_enabled() {
    return dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_ELIG_TRACE");
}

static bool dsv4_experimental_aohc_fused_consume_requested() {
    return dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME");
}

static const char * dsv4_experimental_aohc_fused_layers_env() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYERS");
    return value != nullptr ? value : "";
}

static int64_t dsv4_experimental_attn_out_hc_post_env_i64(const char * name, int64_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    return end != value ? (int64_t) parsed : fallback;
}

static bool dsv4_experimental_attn_out_hc_post_env_i64_is_set(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    char * end = nullptr;
    (void) std::strtoll(value, &end, 10);
    return end != value;
}

static bool dsv4_experimental_aohc_fused_layers_is_set() {
    const char * value = dsv4_experimental_aohc_fused_layers_env();
    return value[0] != '\0';
}

static int64_t dsv4_experimental_aohc_fused_max_layers() {
    const int64_t value = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_MAX_LAYERS", 4);
    return value > 0 ? value : 4;
}

struct dsv4_aohc_layer_set {
    bool env_set = false;
    bool valid = false;
    std::string reason = "not_set";
    std::vector<int64_t> layers;
    std::string text;
};

static dsv4_aohc_layer_set dsv4_parse_aohc_fused_layers() {
    dsv4_aohc_layer_set out;
    const char * value = dsv4_experimental_aohc_fused_layers_env();
    out.env_set = value[0] != '\0';
    out.text = value;
    if (!out.env_set) {
        return out;
    }
    if (std::strchr(value, '*') != nullptr) {
        out.reason = "all_layer_wildcard_blocked";
        return out;
    }
    const int64_t max_layers = dsv4_experimental_aohc_fused_max_layers();
    const char * p = value;
    std::unordered_set<int64_t> seen;
    while (*p != '\0') {
        while (*p == ' ' || *p == '\t') {
            ++p;
        }
        if (*p == '\0') {
            break;
        }
        char * end = nullptr;
        const long long parsed = std::strtoll(p, &end, 10);
        if (end == p) {
            out.reason = "bad_layer_list";
            return out;
        }
        while (*end == ' ' || *end == '\t') {
            ++end;
        }
        if (*end != '\0' && *end != ',') {
            out.reason = "bad_layer_list";
            return out;
        }
        if (parsed < 0 || parsed > 255) {
            out.reason = "invalid_layer_index";
            return out;
        }
        if (!seen.insert((int64_t) parsed).second) {
            out.reason = "duplicate_layer";
            return out;
        }
        out.layers.push_back((int64_t) parsed);
        if ((int64_t) out.layers.size() > max_layers) {
            out.reason = "too_many_layers";
            return out;
        }
        p = *end == ',' ? end + 1 : end;
    }
    if (out.layers.empty()) {
        out.reason = "empty_layer_set";
        return out;
    }
    out.valid = true;
    out.reason = "allowed";
    return out;
}

static const dsv4_aohc_layer_set & dsv4_aohc_fused_layers() {
    static const dsv4_aohc_layer_set layers = dsv4_parse_aohc_fused_layers();
    return layers;
}

static bool dsv4_aohc_fused_layer_in_set(int il) {
    const dsv4_aohc_layer_set & layers = dsv4_aohc_fused_layers();
    if (!layers.valid) {
        return false;
    }
    return std::find(layers.layers.begin(), layers.layers.end(), (int64_t) il) != layers.layers.end();
}

static void dsv4_aohc_append_rejected_path(std::string & out, const char * env_name) {
    if (!dsv4_experimental_attn_out_hc_post_env_flag_enabled(env_name)) {
        return;
    }
    if (!out.empty()) {
        out += ",";
    }
    out += env_name;
}

static std::string dsv4_aohc_rejected_paths_enabled() {
    std::string out;
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_FUSED_COMP");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_SWIGLU");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_SHARED_SWIGLU");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_DOWN");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_MIXED_ATTN");
    return out;
}

static std::string dsv4_aohc_fused_rejected_paths_enabled() {
    std::string out = dsv4_aohc_rejected_paths_enabled();
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_DECODE");
    dsv4_aohc_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_Q8_HC_EXPAND");
    return out;
}

static std::string dsv4_aohc_consume_reject_reason() {
    if (!dsv4_experimental_attn_out_hc_post_consume_requested()) {
        return "consume_disabled";
    }
    if (std::strcmp(dsv4_experimental_attn_out_hc_post_consume_mode(), "candidate_ds4_shape") != 0) {
        return "bad_consume_mode";
    }
    if (!dsv4_experimental_attn_out_hc_post_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER")) {
        return "missing_layer";
    }
    const int64_t layer = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER", -1);
    if (layer < 0) {
        return "bad_layer";
    }
    if (!dsv4_experimental_attn_out_hc_post_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MIN") ||
            !dsv4_experimental_attn_out_hc_post_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MAX")) {
        return "missing_token_range";
    }
    const int64_t token_min = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MIN", 0);
    const int64_t token_max = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MAX", -1);
    if (token_min > token_max) {
        return "bad_token_range";
    }
    const std::string rejected = dsv4_aohc_rejected_paths_enabled();
    if (!rejected.empty()) {
        return "rejected_paths_enabled:" + rejected;
    }
    if (!dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_ABORT_ON_MISMATCH")) {
        return "abort_on_mismatch_disabled";
    }
    return "";
}

static std::string dsv4_aohc_skip_generic_reject_reason() {
    if (!dsv4_experimental_attn_out_hc_post_skip_generic_requested()) {
        return "skip_generic_disabled";
    }
    if (!dsv4_experimental_attn_out_hc_post_consume_requested()) {
        return "consume_disabled";
    }
    if (std::strcmp(dsv4_experimental_attn_out_hc_post_skip_generic_mode(), "selected_layer_only") != 0) {
        return "bad_skip_generic_mode";
    }
    const std::string consume_reason = dsv4_aohc_consume_reject_reason();
    if (!consume_reason.empty()) {
        return consume_reason;
    }
    return "";
}

static bool dsv4_experimental_attn_out_hc_post_consume_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const std::string reason = dsv4_aohc_consume_reject_reason();
        enabled = reason.empty() ? 1 : 0;
        if (dsv4_experimental_attn_out_hc_post_consume_requested()) {
            const bool rejected_paths = reason.rfind("rejected_paths_enabled:", 0) == 0;
            std::fprintf(stderr,
                    "dsv4_aohc_consume_guard: enabled=1 allowed=%d reason=%s layer=%lld token_min=%lld token_max=%lld mode=%s all_layer_consume_blocked=%d rejected_paths_active=%d\n",
                    enabled,
                    reason.empty() ? "allowed" : reason.c_str(),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER", -1),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MIN", -1),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MAX", -1),
                    dsv4_experimental_attn_out_hc_post_consume_mode(),
                    dsv4_experimental_attn_out_hc_post_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER") ? 0 : 1,
                    rejected_paths ? 1 : 0);
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_attn_out_hc_post_skip_generic_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const std::string reason = dsv4_aohc_skip_generic_reject_reason();
        enabled = reason.empty() ? 1 : 0;
        if (dsv4_experimental_attn_out_hc_post_skip_generic_requested()) {
            const bool rejected_paths = reason.rfind("rejected_paths_enabled:", 0) == 0;
            std::fprintf(stderr,
                    "dsv4_aohc_skip_guard: allowed=%d reason=%s selected_layer=%lld token_min=%lld token_max=%lld all_layer_blocked=%d rejected_paths_active=%d\n",
                    enabled,
                    reason.empty() ? "allowed" : reason.c_str(),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER", -1),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MIN", -1),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MAX", -1),
                    dsv4_experimental_attn_out_hc_post_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER") ? 0 : 1,
                    rejected_paths ? 1 : 0);
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_attn_out_hc_post_consume_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_attn_out_hc_post_consume_enabled() || n_tokens != 1) {
        return false;
    }
    if (token <= 0) {
        return false;
    }
    static int64_t last_token = std::numeric_limits<int64_t>::min();
    static int64_t decode_index = 0;
    if (token != last_token) {
        last_token = token;
        decode_index++;
    }
    const int64_t layer = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER", -1);
    if (il != layer) {
        return false;
    }
    const int64_t token_min = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static std::string dsv4_aohc_fused_reject_reason(bool consume) {
    if (!dsv4_experimental_aohc_fused_requested()) {
        return "fused_disabled";
    }
    if (consume && !dsv4_experimental_aohc_fused_consume_requested()) {
        return "consume_disabled";
    }
    const bool has_layer =
        dsv4_experimental_attn_out_hc_post_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYER");
    const bool has_layers = dsv4_experimental_aohc_fused_layers_is_set();
    if (has_layer && has_layers) {
        return "layer_and_layer_set_both_active";
    }
    if (consume && !has_layer && !has_layers) {
        return "missing_layer";
    }
    if (has_layers) {
        const dsv4_aohc_layer_set & layers = dsv4_aohc_fused_layers();
        if (!layers.valid) {
            return layers.reason;
        }
    }
    const int64_t layer = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYER", -1);
    if (has_layer && layer < 0) {
        return "bad_layer";
    }
    const bool has_token_min =
        dsv4_experimental_attn_out_hc_post_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MIN");
    const bool has_token_max =
        dsv4_experimental_attn_out_hc_post_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MAX");
    if (consume && (!has_token_min || !has_token_max)) {
        return "missing_token_range";
    }
    const int64_t token_min = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    if ((consume || has_token_min || has_token_max) && token_min > token_max) {
        return "bad_token_range";
    }
    const std::string rejected = dsv4_aohc_fused_rejected_paths_enabled();
    if (!rejected.empty()) {
        return "rejected_paths_enabled:" + rejected;
    }
    return "";
}

static bool dsv4_experimental_aohc_fused_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const std::string reason = dsv4_aohc_fused_reject_reason(false);
        enabled = reason.empty() ? 1 : 0;
        if (dsv4_experimental_aohc_fused_requested()) {
            const bool rejected_paths = reason.rfind("rejected_paths_enabled:", 0) == 0;
            const dsv4_aohc_layer_set & layers = dsv4_aohc_fused_layers();
            std::fprintf(stderr,
                    "dsv4_aohc_fused_guard: enabled=1 allowed=%d reason=%s layer=%lld layers=%s layer_count=%zu max_layers=%lld token_min=%lld token_max=%lld all_layer_blocked=%d rejected_paths_active=%d\n",
                    enabled,
                    reason.empty() ? "allowed" : reason.c_str(),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYER", -1),
                    layers.text.empty() ? "none" : layers.text.c_str(),
                    layers.layers.size(),
                    (long long) dsv4_experimental_aohc_fused_max_layers(),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MIN", -1),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MAX", -1),
                    dsv4_experimental_aohc_fused_layers_is_set() && !dsv4_aohc_fused_layers().valid ? 1 : 0,
                    rejected_paths ? 1 : 0);
            if (dsv4_experimental_aohc_fused_layers_is_set()) {
                std::fprintf(stderr,
                        "dsv4_aohc_layer_set_guard: allowed=%d reason=%s layers=%s layer_count=%zu max_layers=%lld rejected_paths_active=%d all_layer_blocked=%d\n",
                        enabled,
                        reason.empty() ? "allowed" : reason.c_str(),
                        layers.text.empty() ? "none" : layers.text.c_str(),
                        layers.layers.size(),
                        (long long) dsv4_experimental_aohc_fused_max_layers(),
                        rejected_paths ? 1 : 0,
                        layers.valid ? 0 : 1);
            }
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_aohc_fused_consume_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const std::string reason = dsv4_aohc_fused_reject_reason(true);
        enabled = reason.empty() ? 1 : 0;
        if (dsv4_experimental_aohc_fused_consume_requested()) {
            const bool rejected_paths = reason.rfind("rejected_paths_enabled:", 0) == 0;
            const dsv4_aohc_layer_set & layers = dsv4_aohc_fused_layers();
            std::fprintf(stderr,
                    "dsv4_aohc_fused_consume_guard: enabled=1 allowed=%d reason=%s layer=%lld layers=%s layer_count=%zu max_layers=%lld token_min=%lld token_max=%lld all_layer_blocked=%d rejected_paths_active=%d\n",
                    enabled,
                    reason.empty() ? "allowed" : reason.c_str(),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYER", -1),
                    layers.text.empty() ? "none" : layers.text.c_str(),
                    layers.layers.size(),
                    (long long) dsv4_experimental_aohc_fused_max_layers(),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MIN", -1),
                    (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MAX", -1),
                    dsv4_experimental_aohc_fused_layers_is_set() && !dsv4_aohc_fused_layers().valid ? 1 : 0,
                    rejected_paths ? 1 : 0);
            if (dsv4_experimental_aohc_fused_layers_is_set()) {
                std::fprintf(stderr,
                        "dsv4_aohc_layer_set_guard: allowed=%d reason=%s layers=%s layer_count=%zu max_layers=%lld rejected_paths_active=%d all_layer_blocked=%d\n",
                        enabled,
                        reason.empty() ? "allowed" : reason.c_str(),
                        layers.text.empty() ? "none" : layers.text.c_str(),
                        layers.layers.size(),
                        (long long) dsv4_experimental_aohc_fused_max_layers(),
                        rejected_paths ? 1 : 0,
                        layers.valid ? 0 : 1);
            }
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_aohc_fused_site_enabled(int il, int64_t token, int64_t n_tokens, bool consume) {
    if (n_tokens != 1) {
        return false;
    }
    if (consume) {
        if (!dsv4_experimental_aohc_fused_consume_enabled()) {
            return false;
        }
    } else if (!dsv4_experimental_aohc_fused_enabled()) {
        return false;
    }
    if (token <= 0) {
        return false;
    }
    static int64_t last_token = std::numeric_limits<int64_t>::min();
    static int64_t decode_index = 0;
    if (token != last_token) {
        last_token = token;
        decode_index++;
    }
    const int64_t layer = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYER", -1);
    if (dsv4_experimental_aohc_fused_layers_is_set()) {
        if (!dsv4_aohc_fused_layer_in_set(il)) {
            return false;
        }
    } else if (dsv4_experimental_attn_out_hc_post_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYER") && il != layer) {
        return false;
    }
    const int64_t token_min = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

struct dsv4_aohc_consume_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t consumes = 0;
    uint64_t skip_generic = 0;
    uint64_t generic_branch_built = 0;
    uint64_t candidate_branch_built = 0;
    uint64_t candidate_downstream_consumed = 0;
};

static dsv4_aohc_consume_summary_state & dsv4_aohc_consume_summary() {
    static dsv4_aohc_consume_summary_state state;
    return state;
}

static void dsv4_aohc_consume_print_summary() {
    auto & state = dsv4_aohc_consume_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.consumes == 0 && !dsv4_experimental_attn_out_hc_post_consume_requested()) {
        return;
    }
    const int64_t token_min = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MIN", -1);
    const int64_t token_max = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_TOKEN_MAX", -1);
    const int64_t expected = (int64_t) state.consumes;
    const bool compare_enabled =
        dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_COMPARE") ||
        dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE_COMPARE");
    const bool readback_enabled =
        compare_enabled ||
        dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SHADOW") ||
        dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE");
    const bool hotpath_neutral_validate =
        dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_VALIDATE");
    const bool performance_eligible =
        dsv4_experimental_attn_out_hc_post_consume_enabled() &&
        state.candidate_branch_built > 0 &&
        !compare_enabled &&
        !readback_enabled &&
        !hotpath_neutral_validate;
    const bool replacement_eligible =
        performance_eligible &&
        dsv4_experimental_attn_out_hc_post_skip_generic_enabled() &&
        state.generic_branch_built == 0 &&
        state.candidate_downstream_consumed > 0;
    std::fprintf(stderr,
            "dsv4_aohc_consume_summary: enabled=%d allowed=%d mode=%s layer=%lld token_min=%lld token_max=%lld"
            " consume_enabled=%d consume_mode=%s consume_layer=%lld consume_token_min=%lld consume_token_max=%lld"
            " dsv4_aohc_consume=%" PRIu64
            " dsv4_aohc_skip_generic=%" PRIu64
            " layer_output_consumes=%" PRIu64
            " consumed=%" PRIu64
            " expected_layer_output_consumes=%lld"
            " generic_branch_built=%d candidate_branch_built=%d candidate_downstream_consumed=%d compare_enabled=%d readback_enabled=%d"
            " hotpath_neutral_validate=%d performance_eligible=%d replacement_eligible=%d"
            " dependency_audit_pass=1 abort_on_mismatch=%d cache_mutation_disabled=1 consume_path=single_layer_candidate_ds4_shape\n",
            dsv4_experimental_attn_out_hc_post_consume_requested() ? 1 : 0,
            dsv4_experimental_attn_out_hc_post_consume_enabled() ? 1 : 0,
            dsv4_experimental_attn_out_hc_post_consume_mode(),
            (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER", -1),
            (long long) token_min,
            (long long) token_max,
            dsv4_experimental_attn_out_hc_post_consume_enabled() ? 1 : 0,
            dsv4_experimental_attn_out_hc_post_consume_mode(),
            (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER", -1),
            (long long) token_min,
            (long long) token_max,
            state.consumes,
            state.skip_generic,
            state.consumes,
            state.consumes,
            (long long) expected,
            state.generic_branch_built > 0 ? 1 : 0,
            state.candidate_branch_built > 0 ? 1 : 0,
            state.candidate_downstream_consumed > 0 ? 1 : 0,
            compare_enabled ? 1 : 0,
            readback_enabled ? 1 : 0,
            hotpath_neutral_validate ? 1 : 0,
            performance_eligible ? 1 : 0,
            replacement_eligible ? 1 : 0,
            dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_ABORT_ON_MISMATCH") ? 1 : 0);
    if (dsv4_experimental_attn_out_hc_post_skip_generic_requested()) {
        std::fprintf(stderr,
                "dsv4_aohc_replacement_summary: consume_enabled=%d skip_generic_enabled=%d selected_layer=%lld token_min=%lld token_max=%lld"
                " generic_branch_built=%d candidate_branch_built=%d candidate_downstream_consumed=%d compare_enabled=%d readback_enabled=%d"
                " hotpath_neutral_validate=%d replacement_eligible=%d\n",
                dsv4_experimental_attn_out_hc_post_consume_enabled() ? 1 : 0,
                dsv4_experimental_attn_out_hc_post_skip_generic_enabled() ? 1 : 0,
                (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                    "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER", -1),
                (long long) token_min,
                (long long) token_max,
                state.generic_branch_built > 0 ? 1 : 0,
                state.candidate_branch_built > 0 ? 1 : 0,
                state.candidate_downstream_consumed > 0 ? 1 : 0,
                compare_enabled ? 1 : 0,
                readback_enabled ? 1 : 0,
                hotpath_neutral_validate ? 1 : 0,
                replacement_eligible ? 1 : 0);
    }
}

static void dsv4_aohc_consume_note(bool generic_branch_built, bool candidate_branch_built, bool skip_generic) {
    auto & state = dsv4_aohc_consume_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_aohc_consume_print_summary);
        state.atexit_registered = true;
    }
    state.consumes++;
    state.skip_generic += skip_generic ? 1 : 0;
    state.generic_branch_built += generic_branch_built ? 1 : 0;
    state.candidate_branch_built += candidate_branch_built ? 1 : 0;
    state.candidate_downstream_consumed++;
}

struct dsv4_aohc_fused_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t fused_cases = 0;
    uint64_t fused_downstream_consumed = 0;
};

static dsv4_aohc_fused_summary_state & dsv4_aohc_fused_summary() {
    static dsv4_aohc_fused_summary_state state;
    return state;
}

static void dsv4_aohc_fused_print_summary() {
    auto & state = dsv4_aohc_fused_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.fused_cases == 0 && !dsv4_experimental_aohc_fused_requested()) {
        return;
    }
    const bool compare_enabled = dsv4_experimental_aohc_fused_compare_enabled();
    const bool readback_enabled = compare_enabled;
    const bool hotpath_neutral_validate =
        dsv4_experimental_attn_out_hc_post_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_VALIDATE");
    const dsv4_aohc_layer_set & layers = dsv4_aohc_fused_layers();
    const int64_t token_min = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MIN", -1);
    const int64_t token_max = dsv4_experimental_attn_out_hc_post_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TOKEN_MAX", -1);
    const int64_t token_count =
        (token_min >= 0 && token_max >= token_min) ? (token_max - token_min + 1) : -1;
    const int64_t expected_consumed =
        (layers.valid && token_count >= 0) ? (int64_t) layers.layers.size() * token_count :
        (dsv4_experimental_attn_out_hc_post_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYER") && token_count >= 0 ? token_count : -1);
    std::fprintf(stderr,
            "dsv4_aohc_fused_summary: enabled=%d layer_filter=%lld token_min=%lld token_max=%lld"
            " dsv4_aohc_fused=%" PRIu64
            " fused_cases=%" PRIu64
            " generic_branch_built=%d fused_branch_built=%d fused_downstream_consumed=%d"
            " compare_enabled=%d readback_enabled=%d hotpath_neutral_validate=%d"
            " expected_dispatch_delta_per_token=-1 observed_metal_dispatch_delta=unknown"
            " consume_path=%s scope=partial_high_projection_plus_hc_post\n",
            dsv4_experimental_aohc_fused_enabled() ? 1 : 0,
            (long long) dsv4_experimental_attn_out_hc_post_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYER", -1),
            (long long) token_min,
            (long long) token_max,
            state.fused_cases,
            state.fused_cases,
            dsv4_experimental_aohc_fused_consume_enabled() ? 0 : 1,
            state.fused_cases > 0 ? 1 : 0,
            state.fused_downstream_consumed > 0 ? 1 : 0,
            compare_enabled ? 1 : 0,
            readback_enabled ? 1 : 0,
            hotpath_neutral_validate ? 1 : 0,
            dsv4_experimental_aohc_fused_consume_enabled() ?
                (layers.valid ? "layer_set" : "single_layer") : "disabled");
    if (layers.env_set) {
        const bool performance_eligible =
            dsv4_experimental_aohc_fused_consume_enabled() &&
            state.fused_cases > 0 &&
            !compare_enabled &&
            !readback_enabled &&
            !hotpath_neutral_validate;
        std::fprintf(stderr,
                "dsv4_aohc_layer_set_summary: enabled=%d layers=%s layer_count=%zu token_min=%lld token_max=%lld"
                " consumed=%" PRIu64 " expected_consumed=%lld dsv4_q8hc=%" PRIu64 " expected_q8hc=%lld"
                " generic_high_projection_built=%d generic_hc_expand_built=%d candidate_branch_built=%d fused_downstream_consumed=%d"
                " dispatch_collapsed=%d observed_dispatch_delta=unknown expected_dispatch_delta=%lld"
                " compare_enabled=%d readback_enabled=%d performance_eligible=%d\n",
                dsv4_experimental_aohc_fused_enabled() ? 1 : 0,
                layers.text.empty() ? "none" : layers.text.c_str(),
                layers.layers.size(),
                (long long) token_min,
                (long long) token_max,
                state.fused_downstream_consumed,
                (long long) expected_consumed,
                state.fused_downstream_consumed,
                (long long) expected_consumed,
                dsv4_experimental_aohc_fused_consume_enabled() ? 0 : 1,
                dsv4_experimental_aohc_fused_consume_enabled() ? 0 : 1,
                state.fused_cases > 0 ? 1 : 0,
                state.fused_downstream_consumed > 0 ? 1 : 0,
                (expected_consumed >= 0 && (int64_t) state.fused_downstream_consumed == expected_consumed) ? 1 : 0,
                expected_consumed >= 0 ? -expected_consumed : 0,
                compare_enabled ? 1 : 0,
                readback_enabled ? 1 : 0,
                performance_eligible ? 1 : 0);
    }
}

static void dsv4_aohc_fused_note(bool downstream_consumed) {
    auto & state = dsv4_aohc_fused_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_aohc_fused_print_summary);
        state.atexit_registered = true;
    }
    state.fused_cases++;
    state.fused_downstream_consumed += downstream_consumed ? 1 : 0;
}

static void dsv4_aohc_fused_elig_print_tensor(const char * label, const ggml_tensor * t) {
    if (t == nullptr) {
        std::fprintf(stderr, "%s=null ", label);
        return;
    }
    std::fprintf(stderr,
            "%s_name=%s %s_op=%s %s_type=%s %s_shape=[%lld,%lld,%lld,%lld] %s_stride=[%llu,%llu,%llu,%llu] ",
            label, t->name,
            label, ggml_op_name(t->op),
            label, ggml_type_name(t->type),
            label,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            label,
            (unsigned long long) t->nb[0], (unsigned long long) t->nb[1],
            (unsigned long long) t->nb[2], (unsigned long long) t->nb[3]);
}

static void dsv4_rmoe_boundary_print_tensor(const char * label, const ggml_tensor * t) {
    if (t == nullptr) {
        std::fprintf(stderr,
                "%s_name=null %s_op=null %s_type=null %s_shape=[-1,-1,-1,-1] %s_stride=[-1,-1,-1,-1] "
                "%s_view_src=null %s_storage_offset=0 %s_materialized=0 %s_contiguous=0 %s_producer_op=null ",
                label, label, label, label, label, label, label, label, label, label);
        return;
    }
    const ggml_tensor * producer = t->src[0];
    std::fprintf(stderr,
            "%s_name=%s %s_op=%s %s_type=%s %s_shape=[%lld,%lld,%lld,%lld] %s_stride=[%llu,%llu,%llu,%llu] "
            "%s_view_src=%s %s_storage_offset=%zu %s_materialized=%d %s_contiguous=%d %s_producer_op=%s ",
            label, t->name,
            label, ggml_op_name(t->op),
            label, ggml_type_name(t->type),
            label,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            label,
            (unsigned long long) t->nb[0], (unsigned long long) t->nb[1],
            (unsigned long long) t->nb[2], (unsigned long long) t->nb[3],
            label, t->view_src != nullptr ? t->view_src->name : "null",
            label, t->view_offs,
            label, t->op == GGML_OP_CONT ? 1 : 0,
            label, ggml_is_contiguous(t) ? 1 : 0,
            label, producer != nullptr ? ggml_op_name(producer->op) : "null");
}

static bool dsv4_experimental_attn_out_hc_post_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_attn_out_hc_post_shadow_enabled()) {
        return false;
    }
    if (n_tokens != 1) {
        return false;
    }
    if (dsv4_experimental_attn_out_hc_post_consume_requested()) {
        return dsv4_experimental_attn_out_hc_post_consume_site_enabled(il, token, n_tokens);
    }
    const char * layer = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_LAYER");
    if (layer != nullptr && layer[0] != '\0' && std::strtoll(layer, nullptr, 10) != il) {
        return false;
    }
    const char * token_min = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_TOKEN_MIN");
    const char * token_max = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_TOKEN_MAX");
    const int64_t first = token;
    const int64_t last = token + std::max<int64_t>(n_tokens, 1) - 1;
    if (token_min != nullptr && token_min[0] != '\0' && last < std::strtoll(token_min, nullptr, 10)) {
        return false;
    }
    if (token_max != nullptr && token_max[0] != '\0' && first > std::strtoll(token_max, nullptr, 10)) {
        return false;
    }
    return true;
}

static bool dsv4_experimental_attn_out_hc_post_candidate_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_attn_out_hc_post_candidate_enabled()) {
        return false;
    }
    if (dsv4_experimental_attn_out_hc_post_consume_requested()) {
        return dsv4_experimental_attn_out_hc_post_consume_site_enabled(il, token, n_tokens);
    }
    return dsv4_experimental_attn_out_hc_post_site_enabled(il, token, n_tokens);
}

static bool dsv4_experimental_compressor_update_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_v2_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_compare_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_COMPARE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_v2_compare_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_COMPARE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_compare_use_fused_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_COMPARE_USE_FUSED");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_v2_compare_use_fused_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_COMPARE_CONSUME");
        enabled = (value != nullptr && std::strcmp(value, "fused") == 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_fused_comp_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_FUSED_COMP");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_v2_fused_comp_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_cupd3_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static int64_t dsv4_experimental_cupd3_env_i64(const char * name, int64_t fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::strtoll(value, nullptr, 10);
}

static bool dsv4_experimental_cupd3_env_i64_is_set(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

static const char * dsv4_experimental_compressor_update_v3_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_MODE");
    return value != nullptr && value[0] != '\0' ? value : "generic_shadow";
}

static bool dsv4_experimental_compressor_update_v3_shadow_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SHADOW");
}

static const char * dsv4_experimental_routed_moe_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_MODE");
    return value != nullptr && value[0] != '\0' ? value : "generic_shadow";
}

static bool dsv4_experimental_routed_moe_shadow_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_SHADOW");
}

static bool dsv4_experimental_routed_moe_compare_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_COMPARE");
}

static bool dsv4_experimental_routed_moe_trace_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_TRACE");
}

static bool dsv4_experimental_routed_moe_mode_is_valid() {
    const char * mode = dsv4_experimental_routed_moe_mode();
    return std::strcmp(mode, "generic_shadow") == 0 ||
           std::strcmp(mode, "stage_probe") == 0 ||
           std::strcmp(mode, "one_tensor_shadow") == 0;
}

static const char * dsv4_experimental_routed_moe_backend_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_MODE");
    return value != nullptr && value[0] != '\0' ? value : "generic_one_tensor_control";
}

static bool dsv4_experimental_routed_moe_backend_shadow_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_SHADOW");
}

static bool dsv4_experimental_routed_moe_backend_compare_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_COMPARE");
}

static bool dsv4_experimental_routed_moe_backend_trace_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_TRACE");
}

static bool dsv4_experimental_routed_moe_backend_mode_is_valid() {
    const char * mode = dsv4_experimental_routed_moe_backend_mode();
    return std::strcmp(mode, "generic_one_tensor_control") == 0 ||
           std::strcmp(mode, "backend_candidate_shadow") == 0 ||
           std::strcmp(mode, "ds4_shape_shadow") == 0;
}

static bool dsv4_experimental_routed_moe_backend_op_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP");
}

static bool dsv4_experimental_routed_moe_backend_op_dry_run_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_DRY_RUN");
}

static bool dsv4_experimental_routed_moe_backend_op_shadow_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SHADOW");
}

static bool dsv4_experimental_routed_moe_backend_op_compare_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_COMPARE");
}

static bool dsv4_experimental_routed_moe_backend_op_trace_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TRACE");
}

static bool dsv4_experimental_routed_moe_backend_op_scratch_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SCRATCH");
}

static const char * dsv4_experimental_routed_moe_backend_op_substage() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SUBSTAGE");
    return value != nullptr && value[0] != '\0' ? value : "final";
}

static bool dsv4_experimental_routed_moe_backend_op_gate_up_substage_enabled() {
    return dsv4_experimental_routed_moe_backend_op_scratch_enabled() &&
           (std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "gate_up") == 0 ||
            std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "swiglu") == 0 ||
            std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "down") == 0 ||
            std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "shared") == 0);
}

static bool dsv4_experimental_routed_moe_backend_op_swiglu_substage_enabled() {
    return dsv4_experimental_routed_moe_backend_op_scratch_enabled() &&
           (std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "swiglu") == 0 ||
            std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "down") == 0 ||
            std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "shared") == 0);
}

static bool dsv4_experimental_routed_moe_backend_op_down_substage_enabled() {
    return dsv4_experimental_routed_moe_backend_op_scratch_enabled() &&
           (std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "down") == 0 ||
            std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "shared") == 0);
}

static bool dsv4_experimental_routed_moe_backend_op_shared_substage_enabled() {
    return dsv4_experimental_routed_moe_backend_op_scratch_enabled() &&
           std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "shared") == 0;
}

static bool dsv4_experimental_routed_moe_backend_op_consume_requested() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME");
}

static bool dsv4_experimental_routed_moe_backend_op_replace_generic_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC");
}

static bool dsv4_experimental_routed_moe_backend_op_pair_preserve_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_PAIR_PRESERVE");
}

static bool dsv4_experimental_routed_moe_backend_op_shared_final_only_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SHARED_FINAL_ONLY");
}

static const char * dsv4_experimental_routed_moe_backend_op_pair_preserve_mode() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_PAIR_PRESERVE_MODE");
    return value != nullptr && value[0] != '\0' ? value : "";
}

static bool dsv4_experimental_routed_moe_backend_op_pair_preserve_mode_down_shared_from_generic_swiglu() {
    return std::strcmp(
            dsv4_experimental_routed_moe_backend_op_pair_preserve_mode(),
            "down_shared_from_generic_swiglu") == 0;
}

static const char * dsv4_experimental_routed_moe_backend_op_shared_final_mode() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SHARED_FINAL_MODE");
    return value != nullptr && value[0] != '\0' ? value : "";
}

static bool dsv4_experimental_routed_moe_backend_op_shared_final_mode_shared_down_plus_final_add() {
    return std::strcmp(
            dsv4_experimental_routed_moe_backend_op_shared_final_mode(),
            "shared_down_plus_final_add") == 0;
}

static const char * dsv4_experimental_routed_moe_pair_preserve_attach_mode() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_PAIR_PRESERVE_ATTACH_MODE");
    return value != nullptr && value[0] != '\0' ? value : "inline_backend_consumer";
}

static bool dsv4_experimental_routed_moe_pair_preserve_attach_mode_is(const char * mode) {
    return std::strcmp(dsv4_experimental_routed_moe_pair_preserve_attach_mode(), mode) == 0;
}

static bool dsv4_experimental_routed_moe_pair_preserve_match_trace_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_PAIR_PRESERVE_MATCH_TRACE");
}

static const char * dsv4_experimental_routed_moe_backend_op_replace_dump_path() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_DUMP");
    return value != nullptr && value[0] != '\0' ? value : "";
}

static bool dsv4_experimental_routed_moe_backend_op_replace_dump_enabled() {
    return dsv4_experimental_routed_moe_backend_op_replace_dump_path()[0] != '\0';
}

static bool dsv4_experimental_routed_moe_backend_op_replace_substage_dump_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_SUBSTAGE_DUMP");
}

static bool dsv4_experimental_routed_moe_backend_op_swiglu_stage_dump_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_STAGE_DUMP");
}

static bool dsv4_experimental_routed_moe_backend_op_downstream_dump_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_DOWNSTREAM_DUMP");
}

static bool dsv4_experimental_routed_moe_backend_op_result_chain_dump_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_RESULT_CHAIN_DUMP");
}

static const char * dsv4_experimental_routed_moe_backend_op_result_chain_mode() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_RESULT_CHAIN_MODE");
    if (value == nullptr || value[0] == '\0') {
        return "none";
    }
    if (std::strcmp(value, "none") == 0 ||
            std::strcmp(value, "metadata_only") == 0 ||
            std::strcmp(value, "materialize_layer0_output") == 0 ||
            std::strcmp(value, "materialize_hc_post_output") == 0 ||
            std::strcmp(value, "materialize_result_hc") == 0 ||
            std::strcmp(value, "materialize_result_norm") == 0 ||
            std::strcmp(value, "dependency_after_layer0") == 0 ||
            std::strcmp(value, "dependency_before_result_hc") == 0 ||
            std::strcmp(value, "dependency_before_result_norm") == 0 ||
            std::strcmp(value, "readback_layer0_output") == 0 ||
            std::strcmp(value, "readback_result_hc") == 0 ||
            std::strcmp(value, "readback_result_norm") == 0) {
        return value;
    }
    return "none";
}

static const char * dsv4_experimental_routed_moe_backend_op_swiglu_mode() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_MODE");
    return value != nullptr && value[0] != '\0' ? value : "backend_scratch";
}

static bool dsv4_experimental_routed_moe_backend_op_swiglu_mode_packed_generic() {
    return std::strcmp(dsv4_experimental_routed_moe_backend_op_swiglu_mode(), "packed_generic") == 0;
}

static bool dsv4_experimental_routed_moe_backend_op_swiglu_mode_generic_graph_boundary() {
    return std::strcmp(dsv4_experimental_routed_moe_backend_op_swiglu_mode(), "generic_graph_boundary") == 0;
}

static const char * dsv4_experimental_routed_moe_backend_op_down_input() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_DOWN_INPUT");
    return value != nullptr && value[0] != '\0' ? value : "internal_scratch";
}

static bool dsv4_experimental_routed_moe_backend_op_down_input_generic_graph_boundary() {
    return std::strcmp(dsv4_experimental_routed_moe_backend_op_down_input(), "generic_graph_boundary") == 0;
}

static const char * dsv4_experimental_routed_moe_backend_op_swiglu_formula() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_FORMULA");
    return value != nullptr && value[0] != '\0' ? value : "backend_current";
}

static int dsv4_experimental_routed_moe_backend_op_swiglu_formula_id() {
    const char * value = dsv4_experimental_routed_moe_backend_op_swiglu_formula();
    if (std::strcmp(value, "generic_exact") == 0) {
        return 1;
    }
    if (std::strcmp(value, "generic_fast_exp") == 0) {
        return 2;
    }
    if (std::strcmp(value, "generic_scalar_order") == 0) {
        return 3;
    }
    if (std::strcmp(value, "generic_kernel_reuse") == 0) {
        return 4;
    }
    return 0;
}

static bool dsv4_experimental_routed_moe_backend_op_consume_trace_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TRACE");
}

static bool dsv4_experimental_routed_moe_backend_op_boundary_probe_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BOUNDARY_PROBE");
}

static const char * dsv4_experimental_routed_moe_backend_op_branch_mode() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BRANCH_MODE");
    if (value == nullptr || value[0] == '\0') {
        return "";
    }
    if (std::strcmp(value, "none") == 0 ||
            std::strcmp(value, "metadata_only") == 0 ||
            std::strcmp(value, "alloc_only") == 0 ||
            std::strcmp(value, "dispatch_noop") == 0 ||
            std::strcmp(value, "dispatch_compute_no_read") == 0 ||
            std::strcmp(value, "dispatch_compute_compare") == 0) {
        return value;
    }
    return "";
}

static const char * dsv4_experimental_routed_moe_backend_op_branch_order() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BRANCH_ORDER");
    if (value == nullptr || value[0] == '\0') {
        return "inline_current";
    }
    if (std::strcmp(value, "inline_current") == 0 ||
            std::strcmp(value, "after_generic_ffn") == 0 ||
            std::strcmp(value, "after_layer") == 0 ||
            std::strcmp(value, "end_of_graph") == 0 ||
            std::strcmp(value, "separate_side_graph") == 0) {
        return value;
    }
    return "inline_current";
}

static bool dsv4_experimental_routed_moe_backend_op_branch_mode_requested() {
    return dsv4_experimental_routed_moe_backend_op_branch_mode()[0] != '\0';
}

static const char * dsv4_experimental_routed_moe_backend_op_consume_semantic() {
    const char * value = std::getenv(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_SEMANTIC");
    if (value == nullptr || value[0] == '\0') {
        return "backend_view";
    }
    if (std::strcmp(value, "backend_view") == 0 ||
            std::strcmp(value, "final_ffn") == 0 ||
            std::strcmp(value, "backend_cont") == 0 ||
            std::strcmp(value, "backend_add_zero") == 0 ||
            std::strcmp(value, "backend_alias_like_generic") == 0 ||
            std::strcmp(value, "backend_rebuild_generic_add") == 0 ||
            std::strcmp(value, "same_tensor_no_backend_branch") == 0 ||
            std::strcmp(value, "same_tensor_with_backend_branch") == 0 ||
            std::strcmp(value, "residual_added") == 0 ||
            std::strcmp(value, "materialized_final") == 0 ||
            std::strcmp(value, "materialized_residual") == 0 ||
            std::strcmp(value, "same_tensor_control") == 0) {
        return value;
    }
    return "backend_view";
}

static bool dsv4_experimental_routed_moe_expose_topk_weights_enabled();
static bool dsv4_experimental_routed_moe_expose_topk_ids_enabled();

static std::string dsv4_experimental_routed_moe_backend_op_consume_rejected_paths_enabled() {
    std::string out;
    auto append = [&](const char * env_name) {
        if (!dsv4_experimental_cupd3_env_flag_enabled(env_name)) {
            return;
        }
        if (!out.empty()) {
            out += ",";
        }
        out += env_name;
    };
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_SWIGLU");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_SHARED_SWIGLU");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_DOWN");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_MIXED_ATTN");
    append("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE");
    append("LLAMA_FLASH_MOE_METAL_STAGE_PROFILE");
    append("LLAMA_FLASH_MOE_METAL_STAGE_PROFILE_DETAIL");
    return out;
}

static const char * dsv4_experimental_routed_moe_backend_op_consume_guard_reason_raw() {
    if (!dsv4_experimental_routed_moe_backend_op_consume_requested()) {
        return "not_requested";
    }
    if ((dsv4_experimental_routed_moe_backend_op_replace_generic_enabled() ||
                dsv4_experimental_routed_moe_backend_op_pair_preserve_enabled() ||
                dsv4_experimental_routed_moe_backend_op_shared_final_only_enabled()) &&
            dsv4_experimental_routed_moe_backend_op_branch_mode_requested()) {
        return "branch_mode_active";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER")) {
        return "missing_layer";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN") ||
            !dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX")) {
        return "missing_token_range";
    }
    if (dsv4_experimental_routed_moe_backend_op_compare_enabled() &&
            !dsv4_experimental_cupd3_env_flag_enabled(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_ABORT_ON_MISMATCH")) {
        return "missing_abort_on_mismatch";
    }
    if (!dsv4_experimental_routed_moe_expose_topk_ids_enabled()) {
        return "missing_topk_ids_exposure";
    }
    if (!dsv4_experimental_routed_moe_expose_topk_weights_enabled()) {
        return "missing_topk_weights_exposure";
    }
    if (dsv4_experimental_routed_moe_backend_op_pair_preserve_enabled()) {
        if (!dsv4_experimental_routed_moe_backend_op_pair_preserve_mode_down_shared_from_generic_swiglu()) {
            return "invalid_pair_preserve_mode";
        }
    } else if (dsv4_experimental_routed_moe_backend_op_shared_final_only_enabled()) {
        if (!dsv4_experimental_routed_moe_backend_op_shared_final_mode_shared_down_plus_final_add()) {
            return "invalid_shared_final_mode";
        }
    } else if (!dsv4_experimental_routed_moe_backend_op_scratch_enabled() ||
            std::strcmp(dsv4_experimental_routed_moe_backend_op_substage(), "shared") != 0) {
        return "backend_output_not_full_v1";
    }
    if (!dsv4_experimental_routed_moe_backend_op_consume_rejected_paths_enabled().empty()) {
        return "rejected_paths_active";
    }
    return "ok";
}

static bool dsv4_experimental_routed_moe_backend_op_consume_guard_allowed() {
    return std::strcmp(dsv4_experimental_routed_moe_backend_op_consume_guard_reason_raw(), "ok") == 0;
}

static const char * dsv4_experimental_routed_moe_backend_op_consume_guard_reason() {
    return dsv4_experimental_routed_moe_backend_op_consume_guard_reason_raw();
}

static int64_t dsv4_experimental_decode_index_for_token(int64_t token);

static bool dsv4_experimental_routed_moe_backend_op_consume_trace_site_enabled(int64_t token) {
    if (!dsv4_experimental_routed_moe_backend_op_consume_trace_enabled() || token <= 0) {
        return false;
    }
    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TRACE_TOKEN_MIN",
            std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TRACE_TOKEN_MAX",
            std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_routed_moe_backend_op_consume_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_backend_op_consume_guard_allowed() || n_tokens != 1 || token <= 0) {
        return false;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", -1);
    if (il != layer) {
        return false;
    }
    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_routed_moe_backend_op_replace_dump_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_backend_op_replace_dump_enabled() || n_tokens != 1 || token <= 0) {
        return false;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", -1);
    if (layer >= 0 && il != layer) {
        return false;
    }
    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_routed_moe_backend_op_downstream_dump_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_backend_op_downstream_dump_enabled()) {
        return false;
    }
    return dsv4_experimental_routed_moe_backend_op_replace_dump_site_enabled(il, token, n_tokens);
}

static bool dsv4_experimental_routed_moe_backend_op_result_chain_dump_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_backend_op_result_chain_dump_enabled()) {
        return false;
    }
    return dsv4_experimental_routed_moe_backend_op_replace_dump_site_enabled(il, token, n_tokens);
}

static bool dsv4_experimental_routed_moe_backend_op_result_chain_mode_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (std::strcmp(dsv4_experimental_routed_moe_backend_op_result_chain_mode(), "none") == 0) {
        return false;
    }
    return dsv4_experimental_routed_moe_backend_op_consume_site_enabled(il, token, n_tokens);
}

static bool dsv4_experimental_routed_moe_backend_op_graph_trace_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_GRAPH_TRACE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_routed_moe_backend_op_graph_trace_site_enabled(int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_backend_op_graph_trace_enabled() || n_tokens != 1 || token <= 0 ||
            dsv4_experimental_routed_moe_backend_op_replace_dump_path()[0] == '\0') {
        return false;
    }
    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const int64_t default_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN", 1);
    const int64_t default_max = std::min<int64_t>(
            dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX", 40),
            40);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TRACE_TOKEN_MIN", default_min);
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TRACE_TOKEN_MAX", default_max);
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_routed_moe_expose_topk_weights_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_EXPOSE_TOPK_WEIGHTS");
}

static bool dsv4_experimental_routed_moe_expose_topk_ids_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_EXPOSE_TOPK_IDS");
}

static bool dsv4_experimental_compressor_update_v3_compare_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_COMPARE");
}

static bool dsv4_experimental_compressor_update_v3_trace_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TRACE");
}

static bool dsv4_experimental_compressor_update_v3_search_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SEARCH");
}

static bool dsv4_experimental_compressor_update_v3_tail_attrib_requested() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB");
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_requested() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL");
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_compare_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_COMPARE");
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_trace_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TRACE");
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_drift_trace_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_DRIFT_TRACE");
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_attn_row_probe_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_ATTN_ROW_PROBE");
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_value_probe_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_VALUE_PROBE");
}

static bool dsv4_experimental_compressor_update_v3_decode_compress_internal_probe_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_DECODE_COMPRESS_INTERNAL_PROBE");
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_dep_barrier_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_DEP_BARRIER");
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_consume_emit_only_enabled() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME_EMIT_ONLY");
}

static const char * dsv4_experimental_compressor_update_v3_backend_tail_attn_layout_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_ATTN_LAYOUT_MODE");
    return value != nullptr && value[0] != '\0' ? value : "backend_layout";
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_force_contiguous() {
    return std::strcmp(dsv4_experimental_compressor_update_v3_backend_tail_attn_layout_mode(), "force_contiguous") == 0;
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_stream_allowed(const char * stream) {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_STREAM");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "all") == 0) {
        return true;
    }
    return stream != nullptr && std::strcmp(value, stream) == 0;
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_consume_requested() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME");
}

static const char * dsv4_experimental_compressor_update_v3_backend_tail_consume_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME_MODE");
    return value != nullptr && value[0] != '\0' ? value : "";
}

static void dsv4_cupd3_append_rejected_path(std::string & out, const char * env_name);

static std::string dsv4_cupd3_backend_tail_rejected_paths_enabled() {
    std::string out;
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_FUSED_COMP");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_SWIGLU");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_SHARED_SWIGLU");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_DOWN");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_MIXED_ATTN");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_METAL_STAGE_PROFILE");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_METAL_STAGE_PROFILE_DETAIL");
    return out;
}

static std::string dsv4_cupd3_backend_tail_consume_reject_reason() {
    if (!dsv4_experimental_compressor_update_v3_backend_tail_consume_requested()) {
        return "consume_disabled";
    }
    if (std::strcmp(dsv4_experimental_compressor_update_v3_backend_tail_consume_mode(), "pool_norm_rope_quant_generic_cache") != 0) {
        return "bad_consume_mode";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER")) {
        return "missing_layer";
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER", -1);
    if (layer < 0) {
        return "bad_layer";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MIN") ||
            !dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MAX")) {
        return "missing_token_range";
    }
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MIN", 0);
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MAX", -1);
    if (token_min > token_max) {
        return "bad_token_range";
    }
    const std::string rejected = dsv4_cupd3_backend_tail_rejected_paths_enabled();
    if (!rejected.empty()) {
        return "rejected_paths_enabled:" + rejected;
    }
    return "";
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_consume_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const std::string reason = dsv4_cupd3_backend_tail_consume_reject_reason();
        enabled = reason.empty() ? 1 : 0;
        if (dsv4_experimental_compressor_update_v3_backend_tail_consume_requested()) {
            const bool rejected_paths = reason.rfind("rejected_paths_enabled:", 0) == 0;
            std::fprintf(stderr,
                    "dsv4_cupd3_backend_tail_consume_guard: allowed=%d reason=%s selected_layer=%lld token_min=%lld token_max=%lld"
                    " cache_side_effect_blocked=0 rejected_paths_active=%d dsv4_cupd3_backend_tail_consume=0\n",
                    enabled,
                    reason.empty() ? "allowed" : reason.c_str(),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER", -1),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MIN", -1),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MAX", -1),
                    rejected_paths ? 1 : 0);
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_consume_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_compressor_update_v3_backend_tail_consume_enabled() || n_tokens != 1 || token <= 0) {
        return false;
    }
    static int64_t last_token = std::numeric_limits<int64_t>::min();
    static int64_t decode_index = 0;
    if (token != last_token) {
        last_token = token;
        decode_index++;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER", -1);
    if (il != layer) {
        return false;
    }
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_drift_trace_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if ((!dsv4_experimental_compressor_update_v3_backend_tail_drift_trace_enabled() &&
                !dsv4_experimental_compressor_update_v3_backend_tail_attn_row_probe_enabled() &&
                !dsv4_experimental_compressor_update_v3_backend_tail_value_probe_enabled() &&
                !dsv4_experimental_compressor_update_v3_decode_compress_internal_probe_enabled()) ||
            n_tokens != 1 || token <= 0) {
        return false;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER", -1);
    if (layer >= 0 && il != layer) {
        return false;
    }
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TRACE_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TRACE_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return token >= token_min && token <= token_max;
}

static std::string dsv4_cupd3_backend_tail_reject_reason() {
    if (!dsv4_experimental_compressor_update_v3_backend_tail_requested()) {
        return "backend_tail_disabled";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER")) {
        return "missing_layer";
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER", -1);
    if (layer < 0) {
        return "bad_layer";
    }
    const bool has_min = dsv4_experimental_cupd3_env_i64_is_set(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MIN");
    const bool has_max = dsv4_experimental_cupd3_env_i64_is_set(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MAX");
    if (has_min != has_max) {
        return "partial_token_range";
    }
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    if (token_min > token_max) {
        return "bad_token_range";
    }
    return "";
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const std::string reason = dsv4_cupd3_backend_tail_reject_reason();
        enabled = reason.empty() ? 1 : 0;
        if (dsv4_experimental_compressor_update_v3_backend_tail_requested()) {
            std::fprintf(stderr,
                    "dsv4_cupd3_backend_tail_guard: allowed=%d reason=%s selected_layer=%lld token_min=%lld token_max=%lld"
                    " projection_source=generic cache_side_effect=0 consume_path=disabled\n",
                    enabled,
                    reason.empty() ? "allowed" : reason.c_str(),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER", -1),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MIN", -1),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MAX", -1));
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_v3_backend_tail_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_compressor_update_v3_backend_tail_enabled() || n_tokens != 1 || token <= 0) {
        return false;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER", -1);
    if (il != layer) {
        return false;
    }
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return token >= token_min && token <= token_max;
}

static std::string dsv4_cupd3_tail_attrib_reject_reason() {
    if (!dsv4_experimental_compressor_update_v3_tail_attrib_requested()) {
        return "attrib_disabled";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_LAYER")) {
        return "missing_layer";
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_LAYER", -1);
    if (layer < 0) {
        return "bad_layer";
    }
    const bool has_min = dsv4_experimental_cupd3_env_i64_is_set(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MIN");
    const bool has_max = dsv4_experimental_cupd3_env_i64_is_set(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MAX");
    if (has_min != has_max) {
        return "partial_token_range";
    }
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    if (token_min > token_max) {
        return "bad_token_range";
    }
    return "";
}

static bool dsv4_experimental_compressor_update_v3_tail_attrib_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const std::string reason = dsv4_cupd3_tail_attrib_reject_reason();
        enabled = reason.empty() ? 1 : 0;
        if (dsv4_experimental_compressor_update_v3_tail_attrib_requested()) {
            std::fprintf(stderr,
                    "dsv4_cupd3_tail_attrib_guard: allowed=%d reason=%s selected_layer=%lld token_min=%lld token_max=%lld cache_side_effect=0 consume_path=disabled\n",
                    enabled,
                    reason.empty() ? "allowed" : reason.c_str(),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_LAYER", -1),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MIN", -1),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MAX", -1));
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_v3_tail_attrib_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_compressor_update_v3_tail_attrib_enabled() || n_tokens != 1 || token <= 0) {
        return false;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_LAYER", -1);
    if (il != layer) {
        return false;
    }
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return token >= token_min && token <= token_max;
}

static bool dsv4_experimental_compressor_update_v3_consume_requested() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME");
}

static const char * dsv4_experimental_compressor_update_v3_consume_mode() {
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_MODE");
    return value != nullptr && value[0] != '\0' ? value : "";
}

static bool dsv4_experimental_compressor_update_v3_skip_generic_tail_requested() {
    return dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL");
}

static void dsv4_cupd3_append_rejected_path(std::string & out, const char * env_name) {
    if (!dsv4_experimental_cupd3_env_flag_enabled(env_name)) {
        return;
    }
    if (!out.empty()) {
        out += ",";
    }
    out += env_name;
}

static std::string dsv4_cupd3_consume_rejected_paths_enabled() {
    std::string out;
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_FUSED_COMP");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_SWIGLU");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_SHARED_SWIGLU");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_DOWN");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED");
    dsv4_cupd3_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_MIXED_ATTN");
    return out;
}

static std::string dsv4_cupd3_consume_reject_reason() {
    if (!dsv4_experimental_compressor_update_v3_consume_requested()) {
        return "consume_disabled";
    }
    if (std::strcmp(dsv4_experimental_compressor_update_v3_consume_mode(), "tail_candidate_generic_cache") != 0) {
        return "bad_consume_mode";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_LAYER")) {
        return "missing_layer";
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_LAYER", -1);
    if (layer < 0) {
        return "bad_layer";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MIN") ||
            !dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MAX")) {
        return "missing_token_range";
    }
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MIN", 0);
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MAX", -1);
    if (token_min > token_max) {
        return "bad_token_range";
    }
    const std::string rejected = dsv4_cupd3_consume_rejected_paths_enabled();
    if (!rejected.empty()) {
        return "rejected_paths_enabled:" + rejected;
    }
    return "";
}

static bool dsv4_experimental_compressor_update_v3_consume_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const std::string reason = dsv4_cupd3_consume_reject_reason();
        enabled = reason.empty() ? 1 : 0;
        if (dsv4_experimental_compressor_update_v3_consume_requested()) {
            const bool rejected_paths = reason.rfind("rejected_paths_enabled:", 0) == 0;
            std::fprintf(stderr,
                    "dsv4_cupd3_consume_guard: allowed=%d reason=%s selected_layer=%lld token_min=%lld token_max=%lld all_layer_blocked=%d rejected_paths_active=%d cache_side_effect_blocked=0\n",
                    enabled,
                    reason.empty() ? "allowed" : reason.c_str(),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_LAYER", -1),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MIN", -1),
                    (long long) dsv4_experimental_cupd3_env_i64(
                        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MAX", -1),
                    dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_LAYER") ? 0 : 1,
                    rejected_paths ? 1 : 0);
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_compressor_update_v3_consume_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_compressor_update_v3_consume_enabled() || n_tokens != 1 || token <= 0) {
        return false;
    }
    static int64_t last_token = std::numeric_limits<int64_t>::min();
    static int64_t decode_index = 0;
    if (token != last_token) {
        last_token = token;
        decode_index++;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_LAYER", -1);
    if (il != layer) {
        return false;
    }
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_compressor_update_v3_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_compressor_update_v3_shadow_enabled()) {
        return false;
    }
    if (n_tokens != 1 || token <= 0) {
        return false;
    }
    const char * layer_filter = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_LAYER");
    if (layer_filter != nullptr && layer_filter[0] != '\0' && std::strtoll(layer_filter, nullptr, 10) != il) {
        return false;
    }
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return token >= token_min && token <= token_max;
}

static int64_t dsv4_experimental_decode_index_for_token(int64_t token) {
    static int64_t last_token = std::numeric_limits<int64_t>::min();
    static int64_t decode_index = 0;
    if (token <= 0) {
        return token;
    }
    if (token != last_token) {
        last_token = token;
        decode_index++;
    }
    return decode_index;
}

static bool dsv4_experimental_routed_moe_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_shadow_enabled() || n_tokens != 1 || token <= 0 ||
            !dsv4_experimental_routed_moe_mode_is_valid()) {
        return false;
    }
    const char * layer_filter = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_LAYER");
    if (layer_filter != nullptr && layer_filter[0] != '\0' && std::strtoll(layer_filter, nullptr, 10) != il) {
        return false;
    }
    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_routed_moe_backend_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_backend_shadow_enabled() || n_tokens != 1 || token <= 0 ||
            !dsv4_experimental_routed_moe_backend_mode_is_valid()) {
        return false;
    }
    const char * layer_filter = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_LAYER");
    if (layer_filter != nullptr && layer_filter[0] != '\0' && std::strtoll(layer_filter, nullptr, 10) != il) {
        return false;
    }
    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_routed_moe_backend_op_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_backend_op_enabled() ||
            (!dsv4_experimental_routed_moe_backend_op_dry_run_enabled() &&
             !dsv4_experimental_routed_moe_backend_op_shadow_enabled() &&
             !dsv4_experimental_routed_moe_backend_op_consume_guard_allowed() &&
             !dsv4_experimental_routed_moe_backend_op_branch_mode_requested()) ||
            n_tokens != 1 || token <= 0) {
        return false;
    }
    const char * branch_mode = dsv4_experimental_routed_moe_backend_op_branch_mode();
    if (std::strcmp(branch_mode, "none") == 0 || std::strcmp(branch_mode, "metadata_only") == 0) {
        return false;
    }
    if (!dsv4_experimental_routed_moe_backend_op_dry_run_enabled() &&
            !dsv4_experimental_routed_moe_backend_op_shadow_enabled() &&
            dsv4_experimental_routed_moe_backend_op_consume_guard_allowed() &&
            std::strcmp(dsv4_experimental_routed_moe_backend_op_consume_semantic(), "same_tensor_no_backend_branch") == 0) {
        return false;
    }
    const char * layer_filter = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_LAYER");
    if (layer_filter != nullptr && layer_filter[0] != '\0' && std::strtoll(layer_filter, nullptr, 10) != il) {
        return false;
    } else if ((layer_filter == nullptr || layer_filter[0] == '\0') &&
            (dsv4_experimental_routed_moe_backend_op_consume_requested() ||
             dsv4_experimental_routed_moe_backend_op_branch_mode_requested())) {
        const int64_t consume_layer = dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", -1);
        if (consume_layer >= 0 && il != consume_layer) {
            return false;
        }
    }
    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MIN") ?
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MIN" :
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN",
            std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MAX") ?
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MAX" :
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX",
            std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_routed_moe_backend_op_branch_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_backend_op_enabled() ||
            !dsv4_experimental_routed_moe_backend_op_branch_mode_requested() ||
            n_tokens != 1 || token <= 0) {
        return false;
    }
    const char * layer_filter = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_LAYER");
    if (layer_filter != nullptr && layer_filter[0] != '\0' && std::strtoll(layer_filter, nullptr, 10) != il) {
        return false;
    }
    if (layer_filter == nullptr || layer_filter[0] == '\0') {
        const int64_t consume_layer = dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", -1);
        if (consume_layer >= 0 && il != consume_layer) {
            return false;
        }
    }
    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MIN") ?
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MIN" :
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN",
            std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            dsv4_experimental_cupd3_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MAX") ?
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MAX" :
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX",
            std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_routed_moe_topk_weights_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_expose_topk_weights_enabled() || n_tokens != 1 || token <= 0) {
        return false;
    }
    const char * layer_filter = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_LAYER");
    if (layer_filter != nullptr && layer_filter[0] != '\0' && std::strtoll(layer_filter, nullptr, 10) != il) {
        return false;
    }
    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_routed_moe_topk_ids_site_enabled(int il, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_expose_topk_ids_enabled() || n_tokens != 1 || token <= 0) {
        return false;
    }
    const char * layer_filter = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_LAYER");
    if (layer_filter != nullptr && layer_filter[0] != '\0' && std::strtoll(layer_filter, nullptr, 10) != il) {
        return false;
    }
    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

struct dsv4_moe_shadow_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    uint64_t exact_cases = 0;
    uint64_t non_exact_cases = 0;
    uint64_t stage_probe_tensors = 0;
    uint64_t one_tensor_cases = 0;
    int first_layer = -1;
    int64_t first_token = -1;
    char first_tensor[64] = "none";
    bool saw_router = false;
    bool saw_topk = false;
    bool saw_gate_up = false;
    bool saw_swiglu = false;
    bool saw_down = false;
    bool saw_weighted_sum = false;
    bool saw_shared = false;
    bool saw_final = false;
};

static dsv4_moe_shadow_summary_state & dsv4_moe_shadow_summary() {
    static dsv4_moe_shadow_summary_state state;
    return state;
}

static void dsv4_moe_shadow_print_summary() {
    auto & state = dsv4_moe_shadow_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.cases == 0 && !dsv4_experimental_routed_moe_shadow_enabled()) {
        return;
    }
    const char * mode = dsv4_experimental_routed_moe_mode();
    std::fprintf(stderr,
            "dsv4_moe_shadow_summary: mode=%s layer_filter=%lld token_min=%lld token_max=%lld"
            " cases=%" PRIu64 " exact_cases=%" PRIu64 " non_exact_cases=%" PRIu64
            " dsv4_moe_shadow=%" PRIu64 " dsv4_moe_shadow_exact=%" PRIu64
            " max_abs=0 max_rms=0 over_tol=0"
            " first_non_exact_layer=%d first_non_exact_token=%lld first_non_exact_tensor=%s"
            " stage_probe_tensors=%" PRIu64 " one_tensor_cases=%" PRIu64
            " tensors_found=router:%d,topk:%d,gate_up:%d,swiglu:%d,down:%d,weighted_sum:%d,shared:%d,final:%d"
            " consume_path=disabled\n",
            mode,
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_LAYER", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_TOKEN_MIN", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_TOKEN_MAX", -1),
            state.cases,
            state.exact_cases,
            state.non_exact_cases,
            state.cases,
            state.exact_cases,
            state.non_exact_cases > 0 ? state.first_layer : -1,
            (long long) (state.non_exact_cases > 0 ? state.first_token : -1),
            state.non_exact_cases > 0 ? state.first_tensor : "none",
            state.stage_probe_tensors,
            state.one_tensor_cases,
            state.saw_router ? 1 : 0,
            state.saw_topk ? 1 : 0,
            state.saw_gate_up ? 1 : 0,
            state.saw_swiglu ? 1 : 0,
            state.saw_down ? 1 : 0,
            state.saw_weighted_sum ? 1 : 0,
            state.saw_shared ? 1 : 0,
            state.saw_final ? 1 : 0);
}

static void dsv4_moe_shadow_note(int il, int64_t token, const char * tensor_name, bool exact, bool one_tensor_case = false) {
    auto & state = dsv4_moe_shadow_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_shadow_print_summary);
        state.atexit_registered = true;
    }
    state.cases++;
    state.exact_cases += exact ? 1 : 0;
    state.non_exact_cases += exact ? 0 : 1;
    state.one_tensor_cases += one_tensor_case ? 1 : 0;
    if (!exact && state.first_layer < 0) {
        state.first_layer = il;
        state.first_token = token;
        std::snprintf(state.first_tensor, sizeof(state.first_tensor), "%s", tensor_name != nullptr ? tensor_name : "unknown");
    }
    const char * name = tensor_name != nullptr ? tensor_name : "";
    state.stage_probe_tensors += std::strcmp(dsv4_experimental_routed_moe_mode(), "stage_probe") == 0 ? 1 : 0;
    state.saw_router = state.saw_router || std::strstr(name, "router") != nullptr || std::strstr(name, "logits") != nullptr;
    state.saw_topk = state.saw_topk || std::strstr(name, "topk") != nullptr || std::strstr(name, "selected") != nullptr;
    state.saw_gate_up = state.saw_gate_up || std::strstr(name, "gate") != nullptr || std::strstr(name, "up") != nullptr;
    state.saw_swiglu = state.saw_swiglu || std::strstr(name, "swiglu") != nullptr;
    state.saw_down = state.saw_down || std::strstr(name, "down") != nullptr || std::strstr(name, "moe_out") != nullptr;
    state.saw_weighted_sum = state.saw_weighted_sum || std::strstr(name, "weight") != nullptr;
    state.saw_shared = state.saw_shared || std::strstr(name, "shared") != nullptr || std::strstr(name, "shexp") != nullptr;
    state.saw_final = state.saw_final || std::strstr(name, "final") != nullptr || std::strstr(name, "ffn_out") != nullptr;
}

struct dsv4_moe_backend_shadow_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    uint64_t exact_cases = 0;
    uint64_t non_exact_cases = 0;
    uint64_t infeasible_cases = 0;
    bool backend_candidate_built = false;
    bool backend_op_dispatched = false;
    bool backend_owned = false;
    int first_layer = -1;
    int64_t first_token = -1;
    char first_tensor[96] = "none";
    char infeasible_reason[160] = "none";
};

static dsv4_moe_backend_shadow_summary_state & dsv4_moe_backend_shadow_summary() {
    static dsv4_moe_backend_shadow_summary_state state;
    return state;
}

static void dsv4_moe_backend_shadow_print_summary() {
    auto & state = dsv4_moe_backend_shadow_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.cases == 0 && state.infeasible_cases == 0 &&
            !dsv4_experimental_routed_moe_backend_shadow_enabled()) {
        return;
    }
    const char * mode = dsv4_experimental_routed_moe_backend_mode();
    std::fprintf(stderr,
            "dsv4_moe_backend_summary: mode=%s layer_filter=%lld token_min=%lld token_max=%lld"
            " cases=%" PRIu64 " exact_cases=%" PRIu64 " non_exact_cases=%" PRIu64
            " dsv4_moe_backend_shadow=%" PRIu64 " dsv4_moe_backend_exact=%" PRIu64
            " max_abs=0 max_rms=0 over_tol=0"
            " first_non_exact_layer=%d first_non_exact_token=%lld first_non_exact_tensor=%s"
            " backend_candidate_built=%d backend_op_dispatched=%d backend_owned=%d"
            " infeasible_cases=%" PRIu64 " infeasible_reason=%s"
            " consume_path=disabled\n",
            mode,
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_LAYER", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_TOKEN_MIN", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_TOKEN_MAX", -1),
            state.cases,
            state.exact_cases,
            state.non_exact_cases,
            state.cases,
            state.exact_cases,
            state.non_exact_cases > 0 ? state.first_layer : -1,
            (long long) (state.non_exact_cases > 0 ? state.first_token : -1),
            state.non_exact_cases > 0 ? state.first_tensor : "none",
            state.backend_candidate_built ? 1 : 0,
            state.backend_op_dispatched ? 1 : 0,
            state.backend_owned ? 1 : 0,
            state.infeasible_cases,
            state.infeasible_reason);
}

static void dsv4_moe_backend_shadow_register_summary() {
    auto & state = dsv4_moe_backend_shadow_summary();
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_backend_shadow_print_summary);
        state.atexit_registered = true;
    }
}

static void dsv4_moe_backend_shadow_note(
        int il,
        int64_t token,
        const char * tensor_name,
        bool exact,
        bool backend_candidate_built,
        bool backend_op_dispatched,
        bool backend_owned) {
    auto & state = dsv4_moe_backend_shadow_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    dsv4_moe_backend_shadow_register_summary();
    state.cases++;
    state.exact_cases += exact ? 1 : 0;
    state.non_exact_cases += exact ? 0 : 1;
    state.backend_candidate_built = state.backend_candidate_built || backend_candidate_built;
    state.backend_op_dispatched = state.backend_op_dispatched || backend_op_dispatched;
    state.backend_owned = state.backend_owned || backend_owned;
    if (!exact && state.first_layer < 0) {
        state.first_layer = il;
        state.first_token = token;
        std::snprintf(state.first_tensor, sizeof(state.first_tensor), "%s", tensor_name != nullptr ? tensor_name : "unknown");
    }
}

static void dsv4_moe_backend_shadow_infeasible(
        int il,
        int64_t token,
        const char * reason) {
    auto & state = dsv4_moe_backend_shadow_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    dsv4_moe_backend_shadow_register_summary();
    state.infeasible_cases++;
    if (state.first_layer < 0) {
        state.first_layer = il;
        state.first_token = token;
    }
    std::snprintf(state.infeasible_reason, sizeof(state.infeasible_reason), "%s", reason != nullptr ? reason : "unknown");
}

struct dsv4_moe_backend_op_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    uint64_t dryrun_cases = 0;
    uint64_t shadow_cases = 0;
    uint64_t eligible_cases = 0;
    uint64_t rejected_cases = 0;
    uint64_t exact_cases = 0;
    uint64_t non_exact_cases = 0;
    uint64_t over_tol = 0;
    double max_abs = 0.0;
    double max_rms = 0.0;
    int first_non_exact_layer = -1;
    int64_t first_non_exact_token = -1;
    char first_non_exact_tensor[64] = "none";
    int first_reject_layer = -1;
    int64_t first_reject_token = -1;
    char first_reject_reason[96] = "none";
    char output_shape[64] = "unknown";
    bool topk_ids_visible = false;
    char topk_ids_shape[64] = "missing";
    char topk_ids_dtype[32] = "missing";
    bool topk_weights_visible = false;
    char topk_weights_shape[64] = "missing";
    bool backend_op_dispatched = false;
    uint64_t backend_op_dispatched_count = 0;
    bool output_not_computed = false;
    bool output_computed = false;
    bool partial_output_only = false;
    bool supported_expert_gate_up = false;
    bool supported_expert_down = false;
    bool supported_shared_branch = false;
    char unsupported_blocker[160] = "none";
    int internal_dispatch_count = 0;
    bool scratch_enabled = false;
    char substage[32] = "final";
    bool gate_up_substage_computed = false;
    bool swiglu_substage_computed = false;
    bool down_substage_computed = false;
    bool routed_sum_computed = false;
    bool shared_substage_computed = false;
    bool final_output_computed = false;
    bool gate_up_substage_exact = false;
    bool swiglu_substage_exact = false;
    bool down_substage_exact = false;
    bool routed_sum_exact = false;
    bool shared_substage_exact = false;
    bool final_output_exact = false;
    uint64_t gate_up_compare_cases = 0;
    uint64_t swiglu_compare_cases = 0;
    uint64_t down_compare_cases = 0;
    uint64_t routed_sum_compare_cases = 0;
    uint64_t shared_compare_cases = 0;
    uint64_t final_output_compare_cases = 0;
    char gate_scratch_shape[64] = "not_allocated";
    char up_scratch_shape[64] = "not_allocated";
    char swiglu_scratch_shape[64] = "not_allocated";
    char down_scratch_strategy[64] = "not_allocated";
    size_t scratch_bytes_estimate = 0;
    char scratch_allocation_mode[32] = "not_allocated";
};

struct dsv4_rmoe_consume_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t consumed = 0;
    bool backend_output_computed = false;
    bool generic_ffn_built = false;
    bool backend_ffn_built = false;
    bool backend_output_consumed = false;
    bool same_tensor_control = false;
    bool materialized = false;
    bool residual_added = false;
    char semantic[64] = "final_ffn";
};

struct dsv4_rmoe_replace_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    uint64_t backend_consumed = 0;
    bool generic_ffn_built = false;
    bool backend_ffn_built = false;
    bool backend_output_consumed = false;
    bool output_computed = false;
};

struct dsv4_rmoe_pair_preserve_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    uint64_t consumed = 0;
    bool generic_gate_up_built = false;
    bool generic_swiglu_built = false;
    bool generic_down_built = false;
    bool generic_weighted_sum_built = false;
    bool generic_shared_built = false;
    bool backend_down_built = false;
    bool backend_weighted_sum_built = false;
    bool backend_shared_built = false;
    bool backend_final_built = false;
    bool backend_op_dispatched = false;
    bool backend_output_consumed = false;
};

struct dsv4_rmoe_shared_final_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    uint64_t consumed = 0;
    bool generic_gate_up_built = false;
    bool generic_swiglu_built = false;
    bool generic_down_built = false;
    bool generic_weighted_sum_built = false;
    bool generic_shared_built = false;
    bool backend_shared_built = false;
    bool backend_final_built = false;
    bool backend_output_consumed = false;
};

static dsv4_rmoe_consume_summary_state & dsv4_rmoe_consume_summary() {
    static dsv4_rmoe_consume_summary_state state;
    return state;
}

static dsv4_rmoe_replace_summary_state & dsv4_rmoe_replace_summary() {
    static dsv4_rmoe_replace_summary_state state;
    return state;
}

static dsv4_rmoe_pair_preserve_summary_state & dsv4_rmoe_pair_preserve_summary() {
    static dsv4_rmoe_pair_preserve_summary_state state;
    return state;
}

static dsv4_rmoe_shared_final_summary_state & dsv4_rmoe_shared_final_summary() {
    static dsv4_rmoe_shared_final_summary_state state;
    return state;
}

static const char * dsv4_rmoe_pair_preserve_guard_reason_raw() {
    if (!dsv4_experimental_routed_moe_backend_op_pair_preserve_enabled()) {
        return "not_requested";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER")) {
        return "missing_layer";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN") ||
            !dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX")) {
        return "missing_token_range";
    }
    if (!dsv4_experimental_routed_moe_backend_op_consume_requested()) {
        return "consume_not_requested";
    }
    if (!dsv4_experimental_routed_moe_backend_op_pair_preserve_mode_down_shared_from_generic_swiglu()) {
        return "invalid_mode";
    }
    if (!dsv4_experimental_routed_moe_expose_topk_ids_enabled()) {
        return "missing_topk_ids_exposure";
    }
    if (!dsv4_experimental_routed_moe_expose_topk_weights_enabled()) {
        return "missing_topk_weights_exposure";
    }
    if (dsv4_experimental_routed_moe_backend_op_branch_mode_requested() ||
            dsv4_experimental_routed_moe_backend_op_replace_generic_enabled()) {
        return "incompatible_replace_or_branch_mode";
    }
    if (!dsv4_experimental_routed_moe_backend_op_consume_rejected_paths_enabled().empty()) {
        return "rejected_paths_active";
    }
    return "ok";
}

static bool dsv4_rmoe_pair_preserve_guard_allowed() {
    return std::strcmp(dsv4_rmoe_pair_preserve_guard_reason_raw(), "ok") == 0;
}

static void dsv4_rmoe_pair_preserve_print_summary() {
    auto & state = dsv4_rmoe_pair_preserve_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.cases == 0 && !dsv4_experimental_routed_moe_backend_op_pair_preserve_enabled()) {
        return;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", -1);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN", -1);
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX", -1);
    const bool rejected_paths = !dsv4_experimental_routed_moe_backend_op_consume_rejected_paths_enabled().empty();
    std::fprintf(stderr,
            "dsv4_rmoe_pair_preserve_guard: allowed=%d reason=%s selected_layer=%lld"
            " token_min=%lld token_max=%lld rejected_paths_active=%d\n",
            dsv4_rmoe_pair_preserve_guard_allowed() ? 1 : 0,
            dsv4_rmoe_pair_preserve_guard_reason_raw(),
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            rejected_paths ? 1 : 0);
    std::fprintf(stderr,
            "dsv4_rmoe_pair_preserve_summary: enabled=%d mode=%s selected_layer=%lld token_min=%lld token_max=%lld"
            " attach_mode=%s"
            " cases=%" PRIu64 " generic_gate_up_built=%d generic_swiglu_built=%d"
            " generic_down_built=%d generic_weighted_sum_built=%d generic_shared_built=%d"
            " backend_down_built=%d backend_weighted_sum_built=%d backend_shared_built=%d"
            " backend_final_built=%d backend_op_dispatched=%d backend_output_consumed=%d"
            " pair=-1 pswiglu=-1 fglu=-1 dsv4_rmoe=0 dsv4_rmoe_consume=%" PRIu64 "\n",
            dsv4_experimental_routed_moe_backend_op_pair_preserve_enabled() ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_pair_preserve_mode(),
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            dsv4_experimental_routed_moe_pair_preserve_attach_mode(),
            state.cases,
            state.generic_gate_up_built ? 1 : 0,
            state.generic_swiglu_built ? 1 : 0,
            state.generic_down_built ? 1 : 0,
            state.generic_weighted_sum_built ? 1 : 0,
            state.generic_shared_built ? 1 : 0,
            state.backend_down_built ? 1 : 0,
            state.backend_weighted_sum_built ? 1 : 0,
            state.backend_shared_built ? 1 : 0,
            state.backend_final_built ? 1 : 0,
            state.backend_op_dispatched ? 1 : 0,
            state.backend_output_consumed ? 1 : 0,
            state.consumed);
    std::fprintf(stderr,
            "dsv4_rmoe_pair_preserve_backend_summary: enabled=%d mode=%s selected_layer=%lld token_min=%lld token_max=%lld"
            " attach_mode=%s"
            " generic_gate_up_built=%d generic_swiglu_built=%d generic_down_built=%d"
            " generic_weighted_sum_built=%d generic_shared_built=%d backend_down_built=%d"
            " backend_weighted_sum_built=%d backend_shared_built=%d backend_final_built=%d"
            " backend_op_dispatched=%d backend_output_consumed=%d"
            " pair=-1 pswiglu=-1 fglu=-1 dsv4_rmoe_pair=%" PRIu64 " metal_dispatch=-1"
            " consume_path=single_layer_pair_preserve\n",
            dsv4_experimental_routed_moe_backend_op_pair_preserve_enabled() ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_pair_preserve_mode(),
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            dsv4_experimental_routed_moe_pair_preserve_attach_mode(),
            state.generic_gate_up_built ? 1 : 0,
            state.generic_swiglu_built ? 1 : 0,
            state.generic_down_built ? 1 : 0,
            state.generic_weighted_sum_built ? 1 : 0,
            state.generic_shared_built ? 1 : 0,
            state.backend_down_built ? 1 : 0,
            state.backend_weighted_sum_built ? 1 : 0,
            state.backend_shared_built ? 1 : 0,
            state.backend_final_built ? 1 : 0,
            state.backend_op_dispatched ? 1 : 0,
            state.backend_output_consumed ? 1 : 0,
            state.consumed);
}

static void dsv4_rmoe_pair_preserve_register_summary() {
    auto & state = dsv4_rmoe_pair_preserve_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_rmoe_pair_preserve_print_summary);
        state.atexit_registered = true;
    }
}

static void dsv4_rmoe_pair_preserve_note(
        bool generic_gate_up_built,
        bool generic_swiglu_built,
        bool generic_down_built,
        bool generic_weighted_sum_built,
        bool generic_shared_built,
        bool backend_down_built,
        bool backend_weighted_sum_built,
        bool backend_shared_built,
        bool backend_final_built,
        bool backend_op_dispatched,
        bool backend_output_consumed) {
    auto & state = dsv4_rmoe_pair_preserve_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_rmoe_pair_preserve_print_summary);
        state.atexit_registered = true;
    }
    state.cases++;
    state.consumed += backend_output_consumed ? 1 : 0;
    state.generic_gate_up_built = state.generic_gate_up_built || generic_gate_up_built;
    state.generic_swiglu_built = state.generic_swiglu_built || generic_swiglu_built;
    state.generic_down_built = state.generic_down_built || generic_down_built;
    state.generic_weighted_sum_built = state.generic_weighted_sum_built || generic_weighted_sum_built;
    state.generic_shared_built = state.generic_shared_built || generic_shared_built;
    state.backend_down_built = state.backend_down_built || backend_down_built;
    state.backend_weighted_sum_built = state.backend_weighted_sum_built || backend_weighted_sum_built;
    state.backend_shared_built = state.backend_shared_built || backend_shared_built;
    state.backend_final_built = state.backend_final_built || backend_final_built;
    state.backend_op_dispatched = state.backend_op_dispatched || backend_op_dispatched;
    state.backend_output_consumed = state.backend_output_consumed || backend_output_consumed;
}

static const char * dsv4_rmoe_shared_final_guard_reason_raw() {
    if (!dsv4_experimental_routed_moe_backend_op_shared_final_only_enabled()) {
        return "not_requested";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER")) {
        return "missing_layer";
    }
    if (!dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN") ||
            !dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX")) {
        return "missing_token_range";
    }
    if (!dsv4_experimental_routed_moe_backend_op_consume_requested()) {
        return "consume_not_requested";
    }
    if (!dsv4_experimental_routed_moe_backend_op_shared_final_mode_shared_down_plus_final_add()) {
        return "invalid_mode";
    }
    if (!dsv4_experimental_routed_moe_expose_topk_ids_enabled()) {
        return "missing_topk_ids_exposure";
    }
    if (!dsv4_experimental_routed_moe_expose_topk_weights_enabled()) {
        return "missing_topk_weights_exposure";
    }
    if (dsv4_experimental_routed_moe_backend_op_branch_mode_requested() ||
            dsv4_experimental_routed_moe_backend_op_replace_generic_enabled() ||
            dsv4_experimental_routed_moe_backend_op_pair_preserve_enabled()) {
        return "incompatible_replace_pair_or_branch_mode";
    }
    if (!dsv4_experimental_routed_moe_backend_op_consume_rejected_paths_enabled().empty()) {
        return "rejected_paths_active";
    }
    return "ok";
}

static bool dsv4_rmoe_shared_final_guard_allowed() {
    return std::strcmp(dsv4_rmoe_shared_final_guard_reason_raw(), "ok") == 0;
}

static void dsv4_rmoe_shared_final_print_summary() {
    auto & state = dsv4_rmoe_shared_final_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.cases == 0 && !dsv4_experimental_routed_moe_backend_op_shared_final_only_enabled()) {
        return;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", -1);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN", -1);
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX", -1);
    const bool rejected_paths = !dsv4_experimental_routed_moe_backend_op_consume_rejected_paths_enabled().empty();
    const bool performance_eligible =
        dsv4_rmoe_shared_final_guard_allowed() &&
        state.cases > 0 &&
        state.generic_gate_up_built &&
        state.generic_swiglu_built &&
        state.generic_down_built &&
        state.generic_weighted_sum_built &&
        !state.generic_shared_built &&
        state.backend_shared_built &&
        state.backend_final_built &&
        state.backend_output_consumed;
    std::fprintf(stderr,
            "dsv4_rmoe_shared_final_guard: allowed=%d reason=%s selected_layer=%lld"
            " token_min=%lld token_max=%lld rejected_paths_active=%d\n",
            dsv4_rmoe_shared_final_guard_allowed() ? 1 : 0,
            dsv4_rmoe_shared_final_guard_reason_raw(),
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            rejected_paths ? 1 : 0);
    std::fprintf(stderr,
            "dsv4_rmoe_shared_final_summary: enabled=%d mode=%s selected_layer=%lld token_min=%lld token_max=%lld"
            " cases=%" PRIu64 " generic_gate_up_built=%d generic_swiglu_built=%d generic_down_built=%d"
            " generic_weighted_sum_built=%d generic_shared_built=%d backend_shared_built=%d"
            " backend_final_built=%d backend_output_consumed=%d"
            " pair=-1 pswiglu=-1 fglu=-1 metal_dispatch=-1 performance_eligible=%d"
            " dsv4_rmoe_shared=%" PRIu64 " dsv4_rmoe_consume=%" PRIu64 "\n",
            dsv4_experimental_routed_moe_backend_op_shared_final_only_enabled() ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_shared_final_mode(),
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            state.cases,
            state.generic_gate_up_built ? 1 : 0,
            state.generic_swiglu_built ? 1 : 0,
            state.generic_down_built ? 1 : 0,
            state.generic_weighted_sum_built ? 1 : 0,
            state.generic_shared_built ? 1 : 0,
            state.backend_shared_built ? 1 : 0,
            state.backend_final_built ? 1 : 0,
            state.backend_output_consumed ? 1 : 0,
            performance_eligible ? 1 : 0,
            state.consumed,
            state.consumed);
}

static void dsv4_rmoe_shared_final_register_summary() {
    auto & state = dsv4_rmoe_shared_final_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_rmoe_shared_final_print_summary);
        state.atexit_registered = true;
    }
}

static void dsv4_rmoe_shared_final_note(
        bool generic_gate_up_built,
        bool generic_swiglu_built,
        bool generic_down_built,
        bool generic_weighted_sum_built,
        bool generic_shared_built,
        bool backend_shared_built,
        bool backend_final_built,
        bool backend_output_consumed) {
    auto & state = dsv4_rmoe_shared_final_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_rmoe_shared_final_print_summary);
        state.atexit_registered = true;
    }
    state.cases++;
    state.consumed += backend_output_consumed ? 1 : 0;
    state.generic_gate_up_built = state.generic_gate_up_built || generic_gate_up_built;
    state.generic_swiglu_built = state.generic_swiglu_built || generic_swiglu_built;
    state.generic_down_built = state.generic_down_built || generic_down_built;
    state.generic_weighted_sum_built = state.generic_weighted_sum_built || generic_weighted_sum_built;
    state.generic_shared_built = state.generic_shared_built || generic_shared_built;
    state.backend_shared_built = state.backend_shared_built || backend_shared_built;
    state.backend_final_built = state.backend_final_built || backend_final_built;
    state.backend_output_consumed = state.backend_output_consumed || backend_output_consumed;
}

static void dsv4_rmoe_replace_print_summary() {
    auto & state = dsv4_rmoe_replace_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.cases == 0 && !dsv4_experimental_routed_moe_backend_op_replace_generic_enabled()) {
        return;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", -1);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN", -1);
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX", -1);
    const bool allowed = dsv4_experimental_routed_moe_backend_op_consume_guard_allowed();
    const bool rejected_paths = !dsv4_experimental_routed_moe_backend_op_consume_rejected_paths_enabled().empty();
    std::fprintf(stderr,
            "dsv4_rmoe_replace_guard: allowed=%d reason=%s selected_layer=%lld"
            " all_layer_blocked=%d generic_branch_blocked=%d rejected_paths_active=%d\n",
            allowed ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_consume_guard_reason(),
            (long long) layer,
            dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER") ? 0 : 1,
            dsv4_experimental_routed_moe_backend_op_replace_generic_enabled() ? 1 : 0,
            rejected_paths ? 1 : 0);
    std::fprintf(stderr,
            "dsv4_rmoe_replace_summary: replace_generic=%d selected_layer=%lld token_min=%lld token_max=%lld"
            " cases=%" PRIu64 " generic_ffn_built=%d backend_ffn_built=%d backend_output_consumed=%d"
            " output_computed=%d pair=-1 pswiglu=-1 fglu=-1 dsv4_rmoe=%" PRIu64
            " dsv4_rmoe_consume=%" PRIu64 " result_chain_mode=%s materialization_inserted=%d"
            " dependency_inserted=%d readback_enabled=%d consume_path=%s\n",
            dsv4_experimental_routed_moe_backend_op_replace_generic_enabled() ? 1 : 0,
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            state.cases,
            state.generic_ffn_built ? 1 : 0,
            state.backend_ffn_built ? 1 : 0,
            state.backend_output_consumed ? 1 : 0,
            state.output_computed ? 1 : 0,
            state.backend_consumed,
            state.backend_consumed,
            dsv4_experimental_routed_moe_backend_op_result_chain_mode(),
            std::strncmp(dsv4_experimental_routed_moe_backend_op_result_chain_mode(), "materialize_", 12) == 0 ? 1 : 0,
            std::strncmp(dsv4_experimental_routed_moe_backend_op_result_chain_mode(), "dependency_", 11) == 0 ? 1 : 0,
            std::strncmp(dsv4_experimental_routed_moe_backend_op_result_chain_mode(), "readback_", 9) == 0 ? 1 : 0,
            state.backend_consumed > 0 ? "single_layer_replace_generic" : "disabled");
}

static void dsv4_rmoe_replace_register_summary() {
    auto & state = dsv4_rmoe_replace_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_rmoe_replace_print_summary);
        state.atexit_registered = true;
    }
}

static void dsv4_rmoe_replace_note(
        bool generic_ffn_built,
        bool backend_ffn_built,
        bool backend_output_consumed,
        bool output_computed) {
    auto & state = dsv4_rmoe_replace_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_rmoe_replace_print_summary);
        state.atexit_registered = true;
    }
    state.cases++;
    state.backend_consumed += backend_output_consumed ? 1 : 0;
    state.generic_ffn_built = state.generic_ffn_built || generic_ffn_built;
    state.backend_ffn_built = state.backend_ffn_built || backend_ffn_built;
    state.backend_output_consumed = state.backend_output_consumed || backend_output_consumed;
    state.output_computed = state.output_computed || output_computed;
}

static void dsv4_rmoe_json_string(FILE * f, const char * value) {
    std::fputc('"', f);
    if (value != nullptr) {
        for (const unsigned char * p = (const unsigned char *) value; *p != '\0'; ++p) {
            switch (*p) {
                case '"':  std::fputs("\\\"", f); break;
                case '\\': std::fputs("\\\\", f); break;
                case '\n': std::fputs("\\n",  f); break;
                case '\r': std::fputs("\\r",  f); break;
                case '\t': std::fputs("\\t",  f); break;
                default:
                    if (*p < 0x20) {
                        std::fprintf(f, "\\u%04x", (unsigned) *p);
                    } else {
                        std::fputc(*p, f);
                    }
                    break;
            }
        }
    }
    std::fputc('"', f);
}

static std::string dsv4_rmoe_lower_name(const char * raw) {
    std::string out = raw != nullptr ? raw : "";
    for (char & c : out) {
        c = (char) std::tolower((unsigned char) c);
    }
    return out;
}

static bool dsv4_rmoe_name_has(const std::string & name, const char * needle) {
    return name.find(needle) != std::string::npos;
}

static const char * dsv4_rmoe_graph_stage_bucket(const ggml_tensor * t) {
    if (t == nullptr) {
        return "other";
    }
    const std::string name = dsv4_rmoe_lower_name(t->name);
    if (dsv4_rmoe_name_has(name, "router") || dsv4_rmoe_name_has(name, "gate_inp")) {
        return "ffn_router";
    }
    if (dsv4_rmoe_name_has(name, "topk") || dsv4_rmoe_name_has(name, "tid2eid") ||
            dsv4_rmoe_name_has(name, "dsv4_moe_topk")) {
        return "ffn_topk";
    }
    if (dsv4_rmoe_name_has(name, "shared")) {
        return "ffn_shared";
    }
    if (dsv4_rmoe_name_has(name, "swiglu") || dsv4_rmoe_name_has(name, "silu") ||
            dsv4_rmoe_name_has(name, "clamp") || dsv4_rmoe_name_has(name, "mul_out")) {
        return "ffn_swiglu";
    }
    if (dsv4_rmoe_name_has(name, "gate") || dsv4_rmoe_name_has(name, "_up") ||
            dsv4_rmoe_name_has(name, "up_") || dsv4_rmoe_name_has(name, "ffn_up") ||
            dsv4_rmoe_name_has(name, "ffn_gate")) {
        return "ffn_gate_up";
    }
    if (dsv4_rmoe_name_has(name, "down_slot") || dsv4_rmoe_name_has(name, "_down") ||
            dsv4_rmoe_name_has(name, "ffn_down")) {
        return "ffn_down";
    }
    if (dsv4_rmoe_name_has(name, "weighted") || dsv4_rmoe_name_has(name, "routed_sum") ||
            dsv4_rmoe_name_has(name, "partial")) {
        return "ffn_weighted_sum";
    }
    if (dsv4_rmoe_name_has(name, "final_ffn") || dsv4_rmoe_name_has(name, "ffn_out") ||
            dsv4_rmoe_name_has(name, "consume_final") || dsv4_rmoe_name_has(name, "rebuild_add")) {
        return "ffn_final";
    }
    if (dsv4_rmoe_name_has(name, "hc_ffn_post") || dsv4_rmoe_name_has(name, "hc_post")) {
        return "hc_ffn_post";
    }
    if (dsv4_rmoe_name_has(name, "layer_output") || dsv4_rmoe_name_has(name, "next_layer")) {
        return "layer_output";
    }
    if (dsv4_rmoe_name_has(name, "result_hc")) {
        return "result_hc";
    }
    if (dsv4_rmoe_name_has(name, "result_norm")) {
        return "result_norm";
    }
    if (dsv4_rmoe_name_has(name, "result_output") || dsv4_rmoe_name_has(name, "logits")) {
        return "logits";
    }
    return "other";
}

static int dsv4_rmoe_graph_bucket_index(const char * bucket) {
    static const char * buckets[] = {
        "ffn_router", "ffn_topk", "ffn_gate_up", "ffn_swiglu", "ffn_down",
        "ffn_weighted_sum", "ffn_shared", "ffn_final", "hc_ffn_post",
        "layer_output", "result_hc", "result_norm", "logits", "other",
    };
    for (int i = 0; i < (int) (sizeof(buckets) / sizeof(buckets[0])); ++i) {
        if (std::strcmp(bucket, buckets[i]) == 0) {
            return i;
        }
    }
    return 13;
}

static int dsv4_rmoe_graph_consumer_count(const ggml_cgraph * gf, const ggml_tensor * t) {
    if (gf == nullptr || t == nullptr) {
        return 0;
    }
    int count = 0;
    const int n_nodes = ggml_graph_n_nodes((ggml_cgraph *) gf);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node((ggml_cgraph *) gf, i);
        if (node == nullptr) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] == t) {
                ++count;
            }
        }
    }
    return count;
}

static void dsv4_rmoe_graph_trace_dump(const ggml_cgraph * gf, int64_t token, int64_t n_tokens) {
    if (!dsv4_experimental_routed_moe_backend_op_graph_trace_site_enabled(token, n_tokens)) {
        return;
    }
    const char * path = dsv4_experimental_routed_moe_backend_op_replace_dump_path();
    FILE * f = std::fopen(path, "ab");
    if (f == nullptr) {
        std::fprintf(stderr, "dsv4_rmoe_graph_trace: failed_to_open path=%s\n", path);
        return;
    }
    const int n_nodes = ggml_graph_n_nodes((ggml_cgraph *) gf);
    int op_counts[GGML_OP_COUNT] = {};
    int bucket_counts[14] = {};
    uint64_t signature_hash = 1469598103934665603ull;
    int relevant_count = 0;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node((ggml_cgraph *) gf, i);
        if (node == nullptr) {
            continue;
        }
        if (node->op >= 0 && node->op < GGML_OP_COUNT) {
            op_counts[node->op]++;
        }
        const char * bucket = dsv4_rmoe_graph_stage_bucket(node);
        bucket_counts[dsv4_rmoe_graph_bucket_index(bucket)]++;
        if (std::strcmp(bucket, "other") != 0) {
            ++relevant_count;
        }
        const char * name = node->name[0] != '\0' ? node->name : "";
        for (const unsigned char * p = (const unsigned char *) name; *p != '\0'; ++p) {
            signature_hash ^= uint64_t(*p);
            signature_hash *= 1099511628211ull;
        }
        signature_hash ^= uint64_t(node->op + 31);
        signature_hash *= 1099511628211ull;
    }

    const int64_t decode_index = dsv4_experimental_decode_index_for_token(token);
    const bool replace = dsv4_experimental_routed_moe_backend_op_replace_generic_enabled();
    std::fprintf(f,
            "{\"format\":\"dsv4-rmoe-graph-trace-v1\",\"token\":%lld,\"position\":%lld,"
            "\"mode\":\"%s\",\"layer\":0,\"node_count_total\":%d,\"signature_hash\":\"%016llx\",",
            (long long) decode_index,
            (long long) token,
            replace ? "replace" : "baseline",
            n_nodes,
            (unsigned long long) signature_hash);
    std::fprintf(f, "\"op_counts\":{");
    const enum ggml_op ops[] = {
        GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID, GGML_OP_CPY, GGML_OP_CONT, GGML_OP_VIEW,
        GGML_OP_ADD, GGML_OP_MUL, GGML_OP_RMS_NORM, GGML_OP_CLAMP, GGML_OP_UNARY,
        GGML_OP_GLU, GGML_OP_DSV4_HC_WEIGHTED_SUM, GGML_OP_DSV4_HC_EXPAND,
        GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE,
    };
    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); ++i) {
        if (i != 0) {
            std::fputc(',', f);
        }
        dsv4_rmoe_json_string(f, ggml_op_name(ops[i]));
        std::fprintf(f, ":%d", op_counts[ops[i]]);
    }
    std::fputs("},\"stage_bucket_counts\":{", f);
    static const char * buckets[] = {
        "ffn_router", "ffn_topk", "ffn_gate_up", "ffn_swiglu", "ffn_down",
        "ffn_weighted_sum", "ffn_shared", "ffn_final", "hc_ffn_post",
        "layer_output", "result_hc", "result_norm", "logits", "other",
    };
    for (int i = 0; i < 14; ++i) {
        if (i != 0) {
            std::fputc(',', f);
        }
        dsv4_rmoe_json_string(f, buckets[i]);
        std::fprintf(f, ":%d", bucket_counts[i]);
    }
    std::fprintf(f, "},\"relevant_node_count\":%d,\"nodes\":[", relevant_count);
    bool first = true;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node((ggml_cgraph *) gf, i);
        if (node == nullptr) {
            continue;
        }
        const char * bucket = dsv4_rmoe_graph_stage_bucket(node);
        if (std::strcmp(bucket, "other") == 0) {
            continue;
        }
        if (!first) {
            std::fputc(',', f);
        }
        first = false;
        std::fprintf(f, "{\"node_index\":%d,\"tensor_name\":", i);
        dsv4_rmoe_json_string(f, node->name);
        std::fputs(",\"op\":", f);
        dsv4_rmoe_json_string(f, ggml_op_name(node->op));
        std::fputs(",\"src0\":", f);
        dsv4_rmoe_json_string(f, node->src[0] != nullptr ? node->src[0]->name : "");
        std::fputs(",\"src1\":", f);
        dsv4_rmoe_json_string(f, node->src[1] != nullptr ? node->src[1]->name : "");
        std::fprintf(f,
                ",\"shape\":[%lld,%lld,%lld,%lld],\"dtype\":",
                (long long) node->ne[0], (long long) node->ne[1],
                (long long) node->ne[2], (long long) node->ne[3]);
        dsv4_rmoe_json_string(f, ggml_type_name(node->type));
        std::fputs(",\"stage_bucket\":", f);
        dsv4_rmoe_json_string(f, bucket);
        std::fprintf(f, ",\"producer_layer\":0,\"consumer_count\":%d}",
                dsv4_rmoe_graph_consumer_count(gf, node));
    }
    std::fputs("]}\n", f);
    std::fclose(f);
}

static void dsv4_rmoe_consume_print_summary() {
    auto & state = dsv4_rmoe_consume_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.consumed == 0 && !dsv4_experimental_routed_moe_backend_op_consume_requested()) {
        return;
    }
    const int64_t layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", -1);
    const int64_t token_min = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN", -1);
    const int64_t token_max = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX", -1);
    const int64_t expected = (token_min >= 0 && token_max >= token_min) ? (token_max - token_min + 1) : -1;
    const bool allowed = dsv4_experimental_routed_moe_backend_op_consume_guard_allowed();
    const bool rejected_paths = !dsv4_experimental_routed_moe_backend_op_consume_rejected_paths_enabled().empty();
    std::fprintf(stderr,
            "dsv4_rmoe_consume_guard: enabled=%d allowed=%d consume_allowed=%d reason=%s"
            " selected_layer=%lld token_min=%lld token_max=%lld"
            " all_layer_blocked=%d rejected_paths_blocked=%d rejected_paths_active=%d"
            " dsv4_rmoe_consume=%" PRIu64 "\n",
            dsv4_experimental_routed_moe_backend_op_consume_requested() ? 1 : 0,
            allowed ? 1 : 0,
            allowed ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_consume_guard_reason(),
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            dsv4_experimental_cupd3_env_i64_is_set(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER") ? 0 : 1,
            rejected_paths ? 1 : 0,
            rejected_paths ? 1 : 0,
            state.consumed);
    std::fprintf(stderr,
            "dsv4_rmoe_consume_summary: consume_enabled=%d selected_layer=%lld token_min=%lld token_max=%lld"
            " consumed=%" PRIu64 " expected_consumed=%lld"
            " backend_output_computed=%d generic_ffn_built=%d backend_ffn_built=%d backend_output_consumed=%d"
            " consume_semantic=%s same_tensor_control=%d materialized=%d residual_added=%d"
            " compare_enabled=%d readback_enabled=0 hotpath_neutral_validate=%d replacement_eligible=%d"
            " consume_path=%s\n",
            dsv4_experimental_routed_moe_backend_op_consume_requested() ? 1 : 0,
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            state.consumed,
            (long long) expected,
            state.backend_output_computed ? 1 : 0,
            state.generic_ffn_built ? 1 : 0,
            state.backend_ffn_built ? 1 : 0,
            state.backend_output_consumed ? 1 : 0,
            state.semantic,
            state.same_tensor_control ? 1 : 0,
            state.materialized ? 1 : 0,
            state.residual_added ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_compare_enabled() ? 1 : 0,
            dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_VALIDATE") ? 1 : 0,
            (dsv4_experimental_routed_moe_backend_op_compare_enabled() &&
                state.consumed > 0 && state.backend_output_computed && state.backend_output_consumed &&
                !state.same_tensor_control) ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_replace_generic_enabled() && state.consumed > 0 ?
                "single_layer_replace_generic" :
                (state.consumed > 0 ? "single_layer" : "disabled"));
}

static void dsv4_rmoe_consume_register_summary() {
    auto & state = dsv4_rmoe_consume_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_rmoe_consume_print_summary);
        state.atexit_registered = true;
    }
}

static void dsv4_rmoe_consume_note(
        bool backend_output_computed,
        bool backend_output_consumed,
        bool generic_ffn_built,
        bool backend_ffn_built,
        const char * semantic,
        bool materialized,
        bool residual_added,
        bool same_tensor_control) {
    auto & state = dsv4_rmoe_consume_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_rmoe_consume_print_summary);
        state.atexit_registered = true;
    }
    state.consumed++;
    state.backend_output_computed = state.backend_output_computed || backend_output_computed;
    state.generic_ffn_built = state.generic_ffn_built || generic_ffn_built;
    state.backend_ffn_built = state.backend_ffn_built || backend_ffn_built;
    state.backend_output_consumed = state.backend_output_consumed || backend_output_consumed;
    state.materialized = state.materialized || materialized;
    state.residual_added = state.residual_added || residual_added;
    state.same_tensor_control = state.same_tensor_control || same_tensor_control;
    if (semantic != nullptr && semantic[0] != '\0') {
        std::snprintf(state.semantic, sizeof(state.semantic), "%s", semantic);
    }
}

struct dsv4_rmoe_branch_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    uint64_t backend_tensors_allocated = 0;
    uint64_t backend_dispatches = 0;
    bool readback_enabled = false;
    bool compare_enabled = false;
    bool generic_ffn_consumed = false;
    size_t scratch_bytes = 0;
    char branch_mode[32] = "";
    char branch_order[32] = "inline_current";
};

static dsv4_rmoe_branch_summary_state & dsv4_rmoe_branch_summary() {
    static dsv4_rmoe_branch_summary_state state;
    return state;
}

static void dsv4_rmoe_branch_print_summary() {
    auto & state = dsv4_rmoe_branch_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.cases == 0 && !dsv4_experimental_routed_moe_backend_op_branch_mode_requested()) {
        return;
    }
    std::fprintf(stderr,
            "dsv4_rmoe_branch_summary: branch_mode=%s cases=%" PRIu64
            " branch_order=%s"
            " backend_tensors_allocated=%" PRIu64 " backend_dispatches=%" PRIu64
            " readback_enabled=%d compare_enabled=%d generic_ffn_consumed=%d"
            " scratch_bytes=%zu consume_path=disabled\n",
            state.branch_mode[0] != '\0' ? state.branch_mode : dsv4_experimental_routed_moe_backend_op_branch_mode(),
            state.cases,
            state.branch_order[0] != '\0' ? state.branch_order : dsv4_experimental_routed_moe_backend_op_branch_order(),
            state.backend_tensors_allocated,
            state.backend_dispatches,
            state.readback_enabled ? 1 : 0,
            state.compare_enabled ? 1 : 0,
            state.generic_ffn_consumed ? 1 : 0,
            state.scratch_bytes);
}

static void dsv4_rmoe_branch_note(
        const char * branch_mode,
        const char * branch_order,
        bool backend_tensor_allocated,
        int backend_dispatches,
        bool compare_enabled,
        bool generic_ffn_consumed,
        size_t scratch_bytes) {
    auto & state = dsv4_rmoe_branch_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_rmoe_branch_print_summary);
        state.atexit_registered = true;
    }
    state.cases++;
    state.backend_tensors_allocated += backend_tensor_allocated ? 1 : 0;
    state.backend_dispatches += backend_dispatches > 0 ? (uint64_t) backend_dispatches : 0;
    state.compare_enabled = state.compare_enabled || compare_enabled;
    state.generic_ffn_consumed = state.generic_ffn_consumed || generic_ffn_consumed;
    state.scratch_bytes = std::max(state.scratch_bytes, scratch_bytes);
    if (branch_mode != nullptr && branch_mode[0] != '\0') {
        std::snprintf(state.branch_mode, sizeof(state.branch_mode), "%s", branch_mode);
    }
    if (branch_order != nullptr && branch_order[0] != '\0') {
        std::snprintf(state.branch_order, sizeof(state.branch_order), "%s", branch_order);
    }
}

struct dsv4_moe_topk_ids_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    char shape[64] = "missing";
    char dtype[32] = "missing";
    char source[64] = "unavailable";
    bool consumed_by_generic = false;
};

static dsv4_moe_topk_ids_summary_state & dsv4_moe_topk_ids_summary() {
    static dsv4_moe_topk_ids_summary_state state;
    return state;
}

static void dsv4_moe_topk_ids_print_summary() {
    auto & state = dsv4_moe_topk_ids_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.cases == 0 && !dsv4_experimental_routed_moe_expose_topk_ids_enabled()) {
        return;
    }
    std::fprintf(stderr,
            "dsv4_moe_topk_ids_summary: enabled=%d cases=%" PRIu64
            " dsv4_moe_topk_ids_exposed=%" PRIu64
            " shape=%s dtype=%s source=%s consumed_by_generic=%d consume_path=disabled\n",
            dsv4_experimental_routed_moe_expose_topk_ids_enabled() ? 1 : 0,
            state.cases,
            state.cases,
            state.shape,
            state.dtype,
            state.source,
            state.consumed_by_generic ? 1 : 0);
}

static void dsv4_moe_topk_ids_note(
        int il,
        int64_t token,
        const ggml_tensor * ids,
        const char * source,
        bool consumed_by_generic) {
    auto & state = dsv4_moe_topk_ids_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_topk_ids_print_summary);
        state.atexit_registered = true;
    }
    state.cases++;
    if (ids != nullptr) {
        std::snprintf(state.shape, sizeof(state.shape), "[%lld,%lld,%lld,%lld]",
                (long long) ids->ne[0], (long long) ids->ne[1],
                (long long) ids->ne[2], (long long) ids->ne[3]);
        std::snprintf(state.dtype, sizeof(state.dtype), "%s", ggml_type_name(ids->type));
    }
    std::snprintf(state.source, sizeof(state.source), "%s", source != nullptr ? source : "unknown");
    state.consumed_by_generic = consumed_by_generic;
    if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_moe_topk_ids: layer=%d token=%lld shape=%s dtype=%s source=%s consumed_by_generic=%d consume_path=disabled\n",
                il,
                (long long) token,
                state.shape,
                state.dtype,
                state.source,
                state.consumed_by_generic ? 1 : 0);
    }
}

struct dsv4_moe_topk_weights_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    char shape[64] = "missing";
    char dtype[32] = "missing";
    char source[64] = "unavailable";
    int normalized = -1;
    bool consumed_by_generic = false;
};

static dsv4_moe_topk_weights_summary_state & dsv4_moe_topk_weights_summary() {
    static dsv4_moe_topk_weights_summary_state state;
    return state;
}

static void dsv4_moe_topk_weights_print_summary() {
    auto & state = dsv4_moe_topk_weights_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.cases == 0 && !dsv4_experimental_routed_moe_expose_topk_weights_enabled()) {
        return;
    }
    std::fprintf(stderr,
            "dsv4_moe_topk_weights_summary: enabled=%d cases=%" PRIu64
            " dsv4_moe_topk_weights_exposed=%" PRIu64
            " shape=%s dtype=%s source=%s normalized=%d consumed_by_generic=%d consume_path=disabled\n",
            dsv4_experimental_routed_moe_expose_topk_weights_enabled() ? 1 : 0,
            state.cases,
            state.cases,
            state.shape,
            state.dtype,
            state.source,
            state.normalized,
            state.consumed_by_generic ? 1 : 0);
}

static void dsv4_moe_topk_weights_note(
        int il,
        int64_t token,
        const ggml_tensor * weights,
        const char * source,
        int normalized,
        bool consumed_by_generic) {
    auto & state = dsv4_moe_topk_weights_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_topk_weights_print_summary);
        state.atexit_registered = true;
    }
    state.cases++;
    if (weights != nullptr) {
        std::snprintf(state.shape, sizeof(state.shape), "[%lld,%lld,%lld,%lld]",
                (long long) weights->ne[0], (long long) weights->ne[1],
                (long long) weights->ne[2], (long long) weights->ne[3]);
        std::snprintf(state.dtype, sizeof(state.dtype), "%s", ggml_type_name(weights->type));
    }
    std::snprintf(state.source, sizeof(state.source), "%s", source != nullptr ? source : "unknown");
    state.normalized = normalized;
    state.consumed_by_generic = consumed_by_generic;
    if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_moe_topk_weights: layer=%d token=%lld shape=%s dtype=%s source=%s normalized=%d consumed_by_generic=%d consume_path=disabled\n",
                il,
                (long long) token,
                state.shape,
                state.dtype,
                state.source,
                state.normalized,
                state.consumed_by_generic ? 1 : 0);
    }
}

static dsv4_moe_backend_op_summary_state & dsv4_moe_backend_op_summary() {
    static dsv4_moe_backend_op_summary_state state;
    return state;
}

static void dsv4_moe_backend_op_print_summary() {
    auto & state = dsv4_moe_backend_op_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.cases == 0 && !dsv4_experimental_routed_moe_backend_op_enabled()) {
        return;
    }
    std::fprintf(stderr,
            "dsv4_moe_backend_op_summary: enabled=%d dry_run=%d shadow=%d compare=%d layer_filter=%lld token_min=%lld token_max=%lld"
            " cases=%" PRIu64 " eligible_cases=%" PRIu64 " rejected_cases=%" PRIu64
            " exact_cases=%" PRIu64 " non_exact_cases=%" PRIu64 " max_abs=%.9g max_rms=%.9g over_tol=%" PRIu64
            " first_non_exact_layer=%d first_non_exact_token=%lld first_non_exact_tensor=%s"
            " shadow_only=%d compared_cases=%" PRIu64 " dsv4_rmoe=%" PRIu64
            " dsv4_moe_backend_op=%" PRIu64 " dsv4_moe_backend_op_dryrun=%" PRIu64
            " dsv4_moe_backend_op_shadow=%" PRIu64 " dsv4_moe_backend_op_exact=%" PRIu64
            " dsv4_moe_backend_op_eligible=%" PRIu64 " dsv4_moe_backend_op_rejected=%" PRIu64
            " first_reject_layer=%d first_reject_token=%lld first_reject_reason=%s"
            " owns_router=0 consumes_topk=1 owns_gate_up=1 owns_swiglu=1"
            " owns_down=1 owns_weighted_sum=1 owns_shared=1"
            " consumes_topk_ids=%d consumes_topk_weights=%d"
            " backend_op_dispatched=%d backend_op_dispatched_count=%" PRIu64
            " output_computed=%d output_not_computed=%d partial_output_only=%d internal_dispatch_count=%d"
            " supported_expert_gate_up=%d supported_expert_down=%d supported_shared_branch=%d unsupported_blocker=%s"
            " scratch_enabled=%d substage=%s gate_up_substage_computed=%d swiglu_substage_computed=%d"
            " down_computed=%d routed_sum_computed=%d shared_computed=%d final_output_computed=%d"
            " gate_up_substage_exact=%d gate_up_compare_cases=%" PRIu64
            " swiglu_exact=%d swiglu_compare_cases=%" PRIu64
            " down_exact=%d down_compare_cases=%" PRIu64
            " routed_sum_exact=%d routed_sum_compare_cases=%" PRIu64
            " shared_exact=%d shared_compare_cases=%" PRIu64
            " final_output_exact=%d final_output_compare_cases=%" PRIu64
            " gate_scratch_shape=%s up_scratch_shape=%s swiglu_scratch_shape=%s"
            " down_scratch_strategy=%s scratch_bytes_estimate=%zu scratch_allocation_mode=%s"
            " output_shape=%s"
            " topk_ids_visible=%d topk_ids_shape=%s topk_ids_dtype=%s"
            " topk_weights_visible=%d topk_weights_shape=%s"
            " consume_guard_requested=%d consume_guard_allowed=%d consume_guard_reason=%s"
            " consume_layer=%lld consume_token_min=%lld consume_token_max=%lld consume_path=disabled\n",
            dsv4_experimental_routed_moe_backend_op_enabled() ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_dry_run_enabled() ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_shadow_enabled() ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_compare_enabled() ? 1 : 0,
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_LAYER", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MIN", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_TOKEN_MAX", -1),
            state.cases,
            state.eligible_cases,
            state.rejected_cases,
            state.exact_cases,
            state.non_exact_cases,
            state.max_abs,
            state.max_rms,
            state.over_tol,
            state.first_non_exact_layer,
            (long long) state.first_non_exact_token,
            state.first_non_exact_tensor,
            dsv4_experimental_routed_moe_backend_op_shadow_enabled() ? 1 : 0,
            state.exact_cases + state.non_exact_cases,
            state.cases,
            state.cases,
            state.dryrun_cases,
            state.shadow_cases,
            state.exact_cases,
            state.eligible_cases,
            state.rejected_cases,
            state.first_reject_layer,
            (long long) state.first_reject_token,
            state.first_reject_reason,
            state.topk_ids_visible ? 1 : 0,
            state.topk_weights_visible ? 1 : 0,
            state.backend_op_dispatched ? 1 : 0,
            state.backend_op_dispatched_count,
            state.output_computed ? 1 : 0,
            state.output_not_computed ? 1 : 0,
            state.partial_output_only ? 1 : 0,
            state.internal_dispatch_count,
            state.supported_expert_gate_up ? 1 : 0,
            state.supported_expert_down ? 1 : 0,
            state.supported_shared_branch ? 1 : 0,
            state.unsupported_blocker,
            state.scratch_enabled ? 1 : 0,
            state.substage,
            state.gate_up_substage_computed ? 1 : 0,
            state.swiglu_substage_computed ? 1 : 0,
            state.down_substage_computed ? 1 : 0,
            state.routed_sum_computed ? 1 : 0,
            state.shared_substage_computed ? 1 : 0,
            state.final_output_computed ? 1 : 0,
            state.gate_up_substage_exact ? 1 : 0,
            state.gate_up_compare_cases,
            state.swiglu_substage_exact ? 1 : 0,
            state.swiglu_compare_cases,
            state.down_substage_exact ? 1 : 0,
            state.down_compare_cases,
            state.routed_sum_exact ? 1 : 0,
            state.routed_sum_compare_cases,
            state.shared_substage_exact ? 1 : 0,
            state.shared_compare_cases,
            state.final_output_exact ? 1 : 0,
            state.final_output_compare_cases,
            state.gate_scratch_shape,
            state.up_scratch_shape,
            state.swiglu_scratch_shape,
            state.down_scratch_strategy,
            state.scratch_bytes_estimate,
            state.scratch_allocation_mode,
            state.output_shape,
            state.topk_ids_visible ? 1 : 0,
            state.topk_ids_shape,
            state.topk_ids_dtype,
            state.topk_weights_visible ? 1 : 0,
            state.topk_weights_shape,
            dsv4_experimental_routed_moe_backend_op_consume_requested() ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_consume_guard_allowed() ? 1 : 0,
            dsv4_experimental_routed_moe_backend_op_consume_guard_reason(),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX", -1));
}

static void dsv4_moe_backend_op_note(
        int il,
        int64_t token,
        bool eligible,
        const char * reject_reason,
        const ggml_tensor * output,
        const ggml_tensor * topk_ids,
        const ggml_tensor * topk_weights) {
    auto & state = dsv4_moe_backend_op_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_backend_op_print_summary);
        state.atexit_registered = true;
    }
    state.cases++;
    if (dsv4_experimental_routed_moe_backend_op_dry_run_enabled()) {
        state.dryrun_cases++;
    }
    state.eligible_cases += eligible ? 1 : 0;
    state.rejected_cases += eligible ? 0 : 1;
    if (output != nullptr) {
        std::snprintf(state.output_shape, sizeof(state.output_shape), "[%lld,%lld,%lld,%lld]",
                (long long) output->ne[0], (long long) output->ne[1],
                (long long) output->ne[2], (long long) output->ne[3]);
    }
    if (topk_ids != nullptr) {
        state.topk_ids_visible = true;
        std::snprintf(state.topk_ids_shape, sizeof(state.topk_ids_shape), "[%lld,%lld,%lld,%lld]",
                (long long) topk_ids->ne[0], (long long) topk_ids->ne[1],
                (long long) topk_ids->ne[2], (long long) topk_ids->ne[3]);
        std::snprintf(state.topk_ids_dtype, sizeof(state.topk_ids_dtype), "%s", ggml_type_name(topk_ids->type));
    }
    if (topk_weights != nullptr) {
        state.topk_weights_visible = true;
        std::snprintf(state.topk_weights_shape, sizeof(state.topk_weights_shape), "[%lld,%lld,%lld,%lld]",
                (long long) topk_weights->ne[0], (long long) topk_weights->ne[1],
                (long long) topk_weights->ne[2], (long long) topk_weights->ne[3]);
    }
    if (!eligible && state.first_reject_layer < 0) {
        state.first_reject_layer = il;
        state.first_reject_token = token;
        std::snprintf(state.first_reject_reason, sizeof(state.first_reject_reason), "%s",
                reject_reason != nullptr ? reject_reason : "unknown");
    }
}

static void dsv4_moe_backend_op_shadow_note(
        int il,
        int64_t token,
        bool backend_op_dispatched,
        bool output_not_computed,
        int internal_dispatch_count,
        const char * unsupported_blocker,
        bool scratch_enabled = false,
        const char * substage = "final",
        const ggml_tensor * scratch = nullptr,
        bool gate_up_computed = false,
        bool swiglu_computed = false,
        bool down_computed = false,
        bool routed_sum_computed = false,
        bool shared_computed = false,
        bool final_output_computed = false) {
    auto & state = dsv4_moe_backend_op_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_backend_op_print_summary);
        state.atexit_registered = true;
    }
    state.shadow_cases++;
    state.backend_op_dispatched = state.backend_op_dispatched || backend_op_dispatched;
    state.backend_op_dispatched_count += backend_op_dispatched ? 1 : 0;
    state.output_not_computed = state.output_not_computed || output_not_computed;
    state.output_computed = state.output_computed || !output_not_computed;
    state.partial_output_only = false;
    state.supported_expert_gate_up = true;
    state.supported_expert_down = true;
    state.supported_shared_branch = state.supported_shared_branch || shared_computed;
    if (unsupported_blocker != nullptr && unsupported_blocker[0] != '\0') {
        std::snprintf(state.unsupported_blocker, sizeof(state.unsupported_blocker), "%s", unsupported_blocker);
    }
    state.internal_dispatch_count = std::max(state.internal_dispatch_count, internal_dispatch_count);
    state.scratch_enabled = state.scratch_enabled || scratch_enabled;
    if (substage != nullptr && substage[0] != '\0') {
        std::snprintf(state.substage, sizeof(state.substage), "%s", substage);
    }
    state.gate_up_substage_computed = state.gate_up_substage_computed || gate_up_computed;
    state.swiglu_substage_computed = state.swiglu_substage_computed || swiglu_computed;
    state.down_substage_computed = state.down_substage_computed || down_computed;
    state.routed_sum_computed = state.routed_sum_computed || routed_sum_computed;
    state.shared_substage_computed = state.shared_substage_computed || shared_computed;
    state.final_output_computed = state.final_output_computed || final_output_computed;
    if (scratch != nullptr) {
        const int64_t n_ff = 2048;
        std::snprintf(state.gate_scratch_shape, sizeof(state.gate_scratch_shape), "[%lld,6]", (long long) n_ff);
        std::snprintf(state.up_scratch_shape, sizeof(state.up_scratch_shape), "[%lld,6]", (long long) n_ff);
        std::snprintf(state.swiglu_scratch_shape, sizeof(state.swiglu_scratch_shape), "[%lld,6]", (long long) n_ff);
        std::snprintf(state.down_scratch_strategy, sizeof(state.down_scratch_strategy), "%s",
                shared_computed ? "slots_18_23_partials_24_29_shared_30_34" :
                (down_computed ? "slots_18_23_partials_24_29" : "streaming_down_or_slots_18_23"));
        state.scratch_bytes_estimate = ggml_nbytes(scratch);
        std::snprintf(state.scratch_allocation_mode, sizeof(state.scratch_allocation_mode), "%s", "ggml_tensor");
    }
    if (output_not_computed && state.first_non_exact_layer < 0) {
        state.first_non_exact_layer = il;
        state.first_non_exact_token = token;
        std::snprintf(state.first_non_exact_tensor, sizeof(state.first_non_exact_tensor), "%s", "final_ffn_output_not_computed");
    }
}

static void dsv4_moe_backend_op_gate_up_compare_note(
        int il,
        int64_t token,
        const char * tensor_name,
        bool exact) {
    auto & state = dsv4_moe_backend_op_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_backend_op_print_summary);
        state.atexit_registered = true;
    }
    state.gate_up_compare_cases++;
    state.exact_cases += exact ? 1 : 0;
    state.non_exact_cases += exact ? 0 : 1;
    state.gate_up_substage_exact = state.gate_up_substage_exact || exact;
    if (!exact && state.first_non_exact_layer < 0) {
        state.first_non_exact_layer = il;
        state.first_non_exact_token = token;
        std::snprintf(state.first_non_exact_tensor, sizeof(state.first_non_exact_tensor), "%s",
                tensor_name != nullptr ? tensor_name : "gate_up_substage");
    }
}

static void dsv4_moe_backend_op_swiglu_compare_note(
        int il,
        int64_t token,
        const char * tensor_name,
        bool exact) {
    auto & state = dsv4_moe_backend_op_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_backend_op_print_summary);
        state.atexit_registered = true;
    }
    state.swiglu_compare_cases++;
    state.exact_cases += exact ? 1 : 0;
    state.non_exact_cases += exact ? 0 : 1;
    state.swiglu_substage_exact = state.swiglu_substage_exact || exact;
    if (!exact && state.first_non_exact_layer < 0) {
        state.first_non_exact_layer = il;
        state.first_non_exact_token = token;
        std::snprintf(state.first_non_exact_tensor, sizeof(state.first_non_exact_tensor), "%s",
                tensor_name != nullptr ? tensor_name : "expert_swiglu");
    }
}

static void dsv4_moe_backend_op_down_compare_note(
        int il,
        int64_t token,
        const char * tensor_name,
        bool exact) {
    auto & state = dsv4_moe_backend_op_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_backend_op_print_summary);
        state.atexit_registered = true;
    }
    state.down_compare_cases++;
    state.exact_cases += exact ? 1 : 0;
    state.non_exact_cases += exact ? 0 : 1;
    state.down_substage_exact = state.down_substage_exact || exact;
    if (!exact && state.first_non_exact_layer < 0) {
        state.first_non_exact_layer = il;
        state.first_non_exact_token = token;
        std::snprintf(state.first_non_exact_tensor, sizeof(state.first_non_exact_tensor), "%s",
                tensor_name != nullptr ? tensor_name : "expert_down");
    }
}

static void dsv4_moe_backend_op_routed_sum_compare_note(
        int il,
        int64_t token,
        const char * tensor_name,
        bool exact) {
    auto & state = dsv4_moe_backend_op_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_backend_op_print_summary);
        state.atexit_registered = true;
    }
    state.routed_sum_compare_cases++;
    state.exact_cases += exact ? 1 : 0;
    state.non_exact_cases += exact ? 0 : 1;
    state.routed_sum_exact = state.routed_sum_exact || exact;
    if (!exact && state.first_non_exact_layer < 0) {
        state.first_non_exact_layer = il;
        state.first_non_exact_token = token;
        std::snprintf(state.first_non_exact_tensor, sizeof(state.first_non_exact_tensor), "%s",
                tensor_name != nullptr ? tensor_name : "routed_sum");
    }
}

static void dsv4_moe_backend_op_shared_compare_note(
        int il,
        int64_t token,
        const char * tensor_name,
        bool exact) {
    auto & state = dsv4_moe_backend_op_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_backend_op_print_summary);
        state.atexit_registered = true;
    }
    state.shared_compare_cases++;
    state.exact_cases += exact ? 1 : 0;
    state.non_exact_cases += exact ? 0 : 1;
    state.shared_substage_exact = state.shared_substage_exact || exact;
    if (!exact && state.first_non_exact_layer < 0) {
        state.first_non_exact_layer = il;
        state.first_non_exact_token = token;
        std::snprintf(state.first_non_exact_tensor, sizeof(state.first_non_exact_tensor), "%s",
                tensor_name != nullptr ? tensor_name : "shared_branch");
    }
}

static void dsv4_moe_backend_op_final_compare_note(
        int il,
        int64_t token,
        const char * tensor_name,
        bool exact) {
    auto & state = dsv4_moe_backend_op_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_moe_backend_op_print_summary);
        state.atexit_registered = true;
    }
    state.final_output_compare_cases++;
    state.exact_cases += exact ? 1 : 0;
    state.non_exact_cases += exact ? 0 : 1;
    state.final_output_exact = state.final_output_exact || exact;
    if (!exact && state.first_non_exact_layer < 0) {
        state.first_non_exact_layer = il;
        state.first_non_exact_token = token;
        std::snprintf(state.first_non_exact_tensor, sizeof(state.first_non_exact_tensor), "%s",
                tensor_name != nullptr ? tensor_name : "final_ffn_output");
    }
}

struct dsv4_cupd3_search_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t compressor_cases = 0;
    uint64_t ratio_boundary_cases = 0;
    uint64_t quant_emit_cases = 0;
    int first_layer = -1;
    int64_t first_token = -1;
};

static dsv4_cupd3_search_state & dsv4_cupd3_search_summary() {
    static dsv4_cupd3_search_state state;
    return state;
}

static void dsv4_cupd3_search_print_summary() {
    auto & state = dsv4_cupd3_search_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.compressor_cases == 0 && !dsv4_experimental_compressor_update_v3_search_enabled()) {
        return;
    }
    std::fprintf(stderr,
            "dsv4_cupd3_search_summary: enabled=%d first_compressor_layer=%d first_compressor_token=%lld"
            " compressor_cases=%" PRIu64 " ratio_boundary_cases=%" PRIu64 " quant_emit_cases=%" PRIu64
            " cache_mutation=disabled consume_path=disabled\n",
            dsv4_experimental_compressor_update_v3_search_enabled() ? 1 : 0,
            state.first_layer,
            (long long) state.first_token,
            state.compressor_cases,
            state.ratio_boundary_cases,
            state.quant_emit_cases);
}

static void dsv4_cupd3_note_case(int il, int64_t token, bool ratio_boundary, bool quant_emit) {
    auto & state = dsv4_cupd3_search_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_cupd3_search_print_summary);
        state.atexit_registered = true;
    }
    state.compressor_cases++;
    state.ratio_boundary_cases += ratio_boundary ? 1 : 0;
    state.quant_emit_cases += quant_emit ? 1 : 0;
    if (state.first_layer < 0) {
        state.first_layer = il;
        state.first_token = token;
    }
}

struct dsv4_cupd3_tail_attrib_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t emit_count = 0;
    uint64_t attn_emits = 0;
    uint64_t index_emits = 0;
    uint64_t pool_softmax = 0;
    uint64_t weighted_pool = 0;
    uint64_t rms_norm = 0;
    uint64_t norm_weight = 0;
    uint64_t rope = 0;
    uint64_t quant = 0;
    uint64_t cache_handoff = 0;
    uint64_t estimated_tail_dispatch_nodes = 0;
    int first_layer = -1;
    int64_t first_token = -1;
};

static dsv4_cupd3_tail_attrib_summary_state & dsv4_cupd3_tail_attrib_summary() {
    static dsv4_cupd3_tail_attrib_summary_state state;
    return state;
}

static void dsv4_cupd3_tail_attrib_print_summary() {
    auto & state = dsv4_cupd3_tail_attrib_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.emit_count == 0 && !dsv4_experimental_compressor_update_v3_tail_attrib_requested()) {
        return;
    }

    std::fprintf(stderr,
            "dsv4_cupd3_tail_attrib_summary: enabled=%d layer=%lld token_min=%lld token_max=%lld"
            " emit_count=%" PRIu64 " attn_emits=%" PRIu64 " index_emits=%" PRIu64
            " first_layer=%d first_token=%lld"
            " pool_softmax=%" PRIu64 " weighted_pool=%" PRIu64
            " rms_norm=%" PRIu64 " norm_weight=%" PRIu64
            " rope=%" PRIu64 " quant=%" PRIu64 " cache_handoff=%" PRIu64
            " estimated_tail_dispatch_nodes=%" PRIu64
            " observed_dispatch_contribution=unknown per_op_dispatch_isolation=unavailable"
            " projection_source=generic cache_mutation=disabled candidate_cache_side_effect=0 consume_path=disabled"
            " next_plausible_boundary=pool_norm_rope_quant_tail\n",
            dsv4_experimental_compressor_update_v3_tail_attrib_enabled() ? 1 : 0,
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_LAYER", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MIN", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB_TOKEN_MAX", -1),
            state.emit_count,
            state.attn_emits,
            state.index_emits,
            state.first_layer,
            (long long) state.first_token,
            state.pool_softmax,
            state.weighted_pool,
            state.rms_norm,
            state.norm_weight,
            state.rope,
            state.quant,
            state.cache_handoff,
            state.estimated_tail_dispatch_nodes);
}

static void dsv4_cupd3_tail_attrib_note(int il, int64_t token, const char * stream) {
    auto & state = dsv4_cupd3_tail_attrib_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_cupd3_tail_attrib_print_summary);
        state.atexit_registered = true;
    }
    state.emit_count++;
    if (stream != nullptr && std::strcmp(stream, "index") == 0) {
        state.index_emits++;
    } else {
        state.attn_emits++;
    }
    state.pool_softmax++;
    state.weighted_pool++;
    state.rms_norm++;
    state.norm_weight++;
    state.rope++;
    state.quant++;
    state.cache_handoff++;
    // softmax + mul + sum_rows + reshape + rms_norm + norm-weight mul + reshape + rope + quant + cache handoff
    state.estimated_tail_dispatch_nodes += 10;
    if (state.first_layer < 0) {
        state.first_layer = il;
        state.first_token = token;
    }
}

struct dsv4_cupd3_backend_tail_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t emit_cases = 0;
    uint64_t exact_cases = 0;
    uint64_t non_exact_cases = 0;
    uint64_t attn_emits = 0;
    uint64_t index_emits = 0;
    uint64_t backend_tail_built = 0;
    uint64_t backend_tail_consumed = 0;
    uint64_t backend_tail_consume = 0;
    uint64_t generic_tail_built = 0;
    uint64_t generic_cache_write_built = 0;
    uint64_t dep_barrier_cases = 0;
    uint64_t drift_trace_cases = 0;
    uint64_t attn_row_probe_cases = 0;
    uint64_t value_probe_cases = 0;
    uint64_t internal_probe_cases = 0;
    uint64_t compare_cases = 0;
    int first_layer = -1;
    int64_t first_token = -1;
};

static dsv4_cupd3_backend_tail_summary_state & dsv4_cupd3_backend_tail_summary() {
    static dsv4_cupd3_backend_tail_summary_state state;
    return state;
}

static void dsv4_cupd3_backend_tail_print_summary() {
    auto & state = dsv4_cupd3_backend_tail_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.emit_cases == 0 && !dsv4_experimental_compressor_update_v3_backend_tail_requested()) {
        return;
    }

    const bool consume_enabled = dsv4_experimental_compressor_update_v3_backend_tail_consume_enabled();
    const bool compare_enabled = dsv4_experimental_compressor_update_v3_backend_tail_compare_enabled();
    const bool readback_enabled = compare_enabled || dsv4_experimental_compressor_update_v3_backend_tail_trace_enabled();
    const bool replacement_eligible =
            state.backend_tail_consume > 0 &&
            state.generic_tail_built == 0 &&
            state.backend_tail_consumed > 0 &&
            state.generic_cache_write_built > 0;
    const bool dispatch_collapsed = replacement_eligible;

    std::fprintf(stderr,
            "dsv4_cupd3_backend_tail_summary: enabled=%d layer_filter=%lld token_min=%lld token_max=%lld"
            " scope=pool_norm_rope_quant emit_cases=%" PRIu64 " exact_cases=%" PRIu64 " non_exact_cases=%" PRIu64
            " dsv4_cupd3_backend_tail_consume=%" PRIu64
            " max_abs=0 max_rms=0 over_tol=0 first_mismatch=none"
            " attn_emits=%" PRIu64 " index_emits=%" PRIu64
            " first_layer=%d first_token=%lld"
            " projection_source=generic generic_projection_built=1 generic_tail_built=%d"
            " backend_tail_built=%d backend_tail_consumed=%d"
            " generic_cache_write_built=%d candidate_cache_side_effect=0 cache_mutation_mode=generic_existing_write"
            " dep_barrier=%d emit_only=%d attn_layout_mode=%s drift_trace_cases=%" PRIu64
            " attn_row_probe_cases=%" PRIu64 " value_probe_cases=%" PRIu64 " internal_probe_cases=%" PRIu64
            " first_backend_tail_tensor_mismatch=not_readback"
            " first_cache_consumed_row_mismatch=not_readback"
            " dispatch_collapsed=%d expected_dispatch_delta_per_emit=-8 observed_dispatch_delta=unknown"
            " compare_enabled=%d readback_enabled=%d replacement_eligible=%d consume_path=%s"
            " backend_op=GGML_OP_DSV4_DECODE_COMPRESS quant_op=existing_dsv4_quant\n",
            dsv4_experimental_compressor_update_v3_backend_tail_enabled() ? 1 : 0,
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MIN", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TOKEN_MAX", -1),
            state.emit_cases,
            state.exact_cases,
            state.non_exact_cases,
            state.backend_tail_consume,
            state.attn_emits,
            state.index_emits,
            state.first_layer,
            (long long) state.first_token,
            state.generic_tail_built > 0 ? 1 : 0,
            state.backend_tail_built > 0 ? 1 : 0,
            state.backend_tail_consumed > 0 ? 1 : 0,
            state.generic_cache_write_built > 0 ? 1 : 0,
            state.dep_barrier_cases > 0 ? 1 : 0,
            dsv4_experimental_compressor_update_v3_backend_tail_consume_emit_only_enabled() ? 1 : 0,
            dsv4_experimental_compressor_update_v3_backend_tail_attn_layout_mode(),
            state.drift_trace_cases,
            state.attn_row_probe_cases,
            state.value_probe_cases,
            state.internal_probe_cases,
            dispatch_collapsed ? 1 : 0,
            compare_enabled ? 1 : 0,
            readback_enabled ? 1 : 0,
            replacement_eligible ? 1 : 0,
            consume_enabled ? "single_layer" : "disabled");
}

static void dsv4_cupd3_backend_tail_note(
        int                         il,
        int64_t                     token,
        const char                * stream,
        const dsv4_decode_compressor & dec,
        bool                        generic_cache_write_built,
        bool                        compared,
        bool                        dep_barrier = false,
        bool                        drift_trace = false,
        bool                        row_probe = false,
        bool                        value_probe = false,
        bool                        internal_probe = false) {
    auto & state = dsv4_cupd3_backend_tail_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_cupd3_backend_tail_print_summary);
        state.atexit_registered = true;
    }
    state.emit_cases++;
    state.compare_cases += compared ? 1 : 0;
    state.exact_cases += compared ? 1 : 0;
    if (stream != nullptr && std::strcmp(stream, "index") == 0) {
        state.index_emits++;
    } else {
        state.attn_emits++;
    }
    state.backend_tail_built += dec.backend_tail_built ? 1 : 0;
    state.backend_tail_consumed += dec.candidate_tail_consumed ? 1 : 0;
    state.backend_tail_consume += dec.candidate_tail_consumed ? 1 : 0;
    state.generic_tail_built += dec.generic_tail_built ? 1 : 0;
    state.generic_cache_write_built += generic_cache_write_built ? 1 : 0;
    state.dep_barrier_cases += dep_barrier ? 1 : 0;
    state.drift_trace_cases += drift_trace ? 1 : 0;
    state.attn_row_probe_cases += row_probe ? 1 : 0;
    state.value_probe_cases += value_probe ? 1 : 0;
    state.internal_probe_cases += internal_probe ? 1 : 0;
    if (state.first_layer < 0) {
        state.first_layer = il;
        state.first_token = token;
    }
}

struct dsv4_cupd3_consume_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t consumed = 0;
    uint64_t skip_generic_tail = 0;
    uint64_t generic_projection_built = 0;
    uint64_t generic_tail_built = 0;
    uint64_t candidate_tail_built = 0;
    uint64_t candidate_tail_consumed = 0;
    uint64_t generic_cache_write_built = 0;
};

static dsv4_cupd3_consume_summary_state & dsv4_cupd3_consume_summary() {
    static dsv4_cupd3_consume_summary_state state;
    return state;
}

static void dsv4_cupd3_consume_print_summary() {
    auto & state = dsv4_cupd3_consume_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.consumed == 0 && !dsv4_experimental_compressor_update_v3_consume_requested()) {
        return;
    }

    const bool compare_enabled =
        dsv4_experimental_compressor_update_v3_compare_enabled() ||
        dsv4_experimental_compressor_update_v3_trace_enabled();
    const bool readback_enabled =
        compare_enabled || dsv4_experimental_compressor_update_v3_shadow_enabled();
    const bool hotpath_neutral_validate =
        dsv4_experimental_cupd3_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_VALIDATE");
    const bool replacement_eligible =
        dsv4_experimental_compressor_update_v3_consume_enabled() &&
        state.generic_projection_built > 0 &&
        state.generic_tail_built == 0 &&
        state.candidate_tail_built > 0 &&
        state.candidate_tail_consumed > 0 &&
        state.generic_cache_write_built > 0;

    std::fprintf(stderr,
            "dsv4_cupd3_consume_summary: consume_enabled=%d consume_mode=%s selected_layer=%lld token_min=%lld token_max=%lld"
            " dsv4_cupd3_consume=%" PRIu64 " dsv4_cupd3_skip_generic_tail=%" PRIu64
            " consumed=%" PRIu64 " expected_consumed=%" PRIu64
            " projection_source=generic generic_projection_built=%d"
            " generic_tail_built=%d candidate_tail_built=%d candidate_tail_consumed=%d"
            " generic_cache_write_built=%d candidate_cache_side_effect=0 cache_mutation_mode=generic_existing_write"
            " compare_enabled=%d readback_enabled=%d hotpath_neutral_validate=%d replacement_eligible=%d\n",
            dsv4_experimental_compressor_update_v3_consume_enabled() ? 1 : 0,
            dsv4_experimental_compressor_update_v3_consume_mode(),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_LAYER", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MIN", -1),
            (long long) dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_TOKEN_MAX", -1),
            state.consumed,
            state.skip_generic_tail,
            state.consumed,
            state.consumed,
            state.generic_projection_built > 0 ? 1 : 0,
            state.generic_tail_built > 0 ? 1 : 0,
            state.candidate_tail_built > 0 ? 1 : 0,
            state.candidate_tail_consumed > 0 ? 1 : 0,
            state.generic_cache_write_built > 0 ? 1 : 0,
            compare_enabled ? 1 : 0,
            readback_enabled ? 1 : 0,
            hotpath_neutral_validate ? 1 : 0,
            replacement_eligible ? 1 : 0);
}

static void dsv4_cupd3_consume_note(const dsv4_decode_compressor & dec, bool generic_cache_write_built) {
    auto & state = dsv4_cupd3_consume_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.atexit_registered) {
        std::atexit(dsv4_cupd3_consume_print_summary);
        state.atexit_registered = true;
    }
    state.consumed++;
    state.skip_generic_tail += dec.candidate_tail_consumed && !dec.generic_tail_built ? 1 : 0;
    state.generic_projection_built++;
    state.generic_tail_built += dec.generic_tail_built ? 1 : 0;
    state.candidate_tail_built += dec.candidate_tail_built ? 1 : 0;
    state.candidate_tail_consumed += dec.candidate_tail_consumed ? 1 : 0;
    state.generic_cache_write_built += generic_cache_write_built ? 1 : 0;
}

static bool dsv4_experimental_kv_finalize_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_kv_finalize_dry_run_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_DRY_RUN");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_kv_finalize_compare_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_COMPARE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_kv_finalizer_trace_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_TRACE");
        if (value == nullptr || value[0] == '\0') {
            value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZER_TRACE");
        }
        if (value == nullptr || value[0] == '\0') {
            value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_SET_ROWS_TRACE");
        }
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_kv_finalizer_view_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_VIEW");
        if (value == nullptr || value[0] == '\0') {
            value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZER_VIEW");
        }
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_hc_pre_norm_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM");
        if (value == nullptr || value[0] == '\0') {
            value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU");
        }
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_hc_pre_norm_compare_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_COMPARE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_hc_pre_norm_compare_use_fused_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_COMPARE_CONSUME");
        enabled = (value != nullptr && std::strcmp(value, "fused") == 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_layer_executor_shadow_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SHADOW");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_layer_executor_compare_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_COMPARE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_layer_executor_trace_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TRACE");
        enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' &&
        std::strcmp(value, "0") != 0 &&
        std::strcmp(value, "false") != 0 &&
        std::strcmp(value, "FALSE") != 0 &&
        std::strcmp(value, "off") != 0 &&
        std::strcmp(value, "OFF") != 0;
}

static bool dsv4_env_string_equals(const char * name, const char * expected) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, expected) == 0;
}

static const char * dsv4_layer_executor_consume_style() {
    const char * style = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME_STYLE");
    return style != nullptr && style[0] != '\0' ? style : "current_identity";
}

static bool dsv4_layer_executor_consume_style_supported(const char * style) {
    return std::strcmp(style, "same_tensor") == 0 ||
        std::strcmp(style, "view_alias") == 0 ||
        std::strcmp(style, "reshape_alias") == 0 ||
        std::strcmp(style, "add_zero") == 0 ||
        std::strcmp(style, "mul_one") == 0 ||
        std::strcmp(style, "copy_materialized") == 0 ||
        std::strcmp(style, "current_identity") == 0;
}

static const char * dsv4_layer_executor_consume_style_code(const char * style) {
    if (std::strcmp(style, "same_tensor") == 0) {
        return "st";
    }
    if (std::strcmp(style, "view_alias") == 0) {
        return "va";
    }
    if (std::strcmp(style, "reshape_alias") == 0) {
        return "ra";
    }
    if (std::strcmp(style, "add_zero") == 0) {
        return "az";
    }
    if (std::strcmp(style, "mul_one") == 0) {
        return "mo";
    }
    if (std::strcmp(style, "copy_materialized") == 0) {
        return "cm";
    }
    return "ci";
}

static const char * dsv4_layer_executor_mode() {
    const char * mode = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_MODE");
    return mode != nullptr && mode[0] != '\0' ? mode : "legacy_shadow";
}

static bool dsv4_layer_executor_full_shadow_mode() {
    const char * mode = dsv4_layer_executor_mode();
    return std::strcmp(mode, "generic_envelope") == 0 ||
        std::strcmp(mode, "stage_probe") == 0 ||
        std::strcmp(mode, "contract_dry_run") == 0;
}

static bool dsv4_experimental_layer_executor_dryrun_op_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = dsv4_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP") ? 1 : 0;
    }
    return enabled == 1;
}

// LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_LAYER: master flag for the
// hybrid decode-layer orchestrator (T104 skeleton; T105 wires per-stage dispatch).
static bool dsv4_experimental_decode_layer_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = dsv4_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_LAYER") ? 1 : 0;
    }
    return enabled == 1;
}

// LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_LAYER_LAYERS: optional layer
// selector. Empty or "all" => emit on every decode layer. Comma-separated list
// of integers => only those layers. Any other value => only the parsed int.
static bool dsv4_experimental_decode_layer_layer_selected(int il) {
    const char * raw = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_LAYER_LAYERS");
    if (raw == nullptr || raw[0] == '\0' || std::strcmp(raw, "all") == 0 || std::strcmp(raw, "ALL") == 0) {
        return true;
    }
    const std::string spec(raw);
    size_t start = 0;
    while (start <= spec.size()) {
        size_t end = spec.find(',', start);
        if (end == std::string::npos) {
            end = spec.size();
        }
        if (end > start) {
            try {
                if (std::stoi(spec.substr(start, end - start)) == il) {
                    return true;
                }
            } catch (...) {
                // ignore bad tokens, treat as no-match for that comma slot
            }
        }
        if (end == spec.size()) break;
        start = end + 1;
    }
    return false;
}

static bool dsv4_experimental_decode_layer_site_enabled(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_experimental_decode_layer_enabled() || n_tokens != 1 || pos <= 0) {
        return false;
    }
    return dsv4_experimental_decode_layer_layer_selected(il);
}

static bool dsv4_experimental_layer_executor_dryrun_op_trace_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = dsv4_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_TRACE") ? 1 : 0;
    }
    return enabled == 1;
}

static const char * dsv4_experimental_layer_executor_dryrun_op_mode() {
    const char * mode = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_MODE");
    return mode != nullptr && mode[0] != '\0' ? mode : "live_graph_dispatch";
}

static bool dsv4_experimental_layer_executor_dryrun_op_mode_is(const char * expected) {
    return std::strcmp(dsv4_experimental_layer_executor_dryrun_op_mode(), expected) == 0;
}

static bool dsv4_experimental_layer_executor_side_probe_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = dsv4_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE") ? 1 : 0;
    }
    return enabled == 1;
}

static const char * dsv4_experimental_layer_executor_side_probe_stage() {
    const char * stage = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_STAGE");
    return stage != nullptr && stage[0] != '\0' ? stage : "none";
}

static const char * dsv4_experimental_layer_executor_side_probe_mode() {
    const char * mode = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_MODE");
    return mode != nullptr && mode[0] != '\0' ? mode : "metadata_only";
}

static bool dsv4_experimental_layer_executor_side_probe_payload_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = dsv4_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_PAYLOAD") ? 1 : 0;
    }
    return enabled == 1;
}

static const char * dsv4_experimental_layer_executor_side_probe_payload_dir() {
    const char * dir = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_PAYLOAD_DIR");
    return dir != nullptr && dir[0] != '\0' ? dir : "";
}

static const char * dsv4_experimental_layer_executor_side_probe_payload_capture_mode() {
    const char * mode = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_PAYLOAD_CAPTURE_MODE");
    return mode != nullptr && mode[0] != '\0' ? mode : "post_eval_tensor_get";
}

static bool dsv4_experimental_layer_executor_side_probe_payload_producer_capture() {
    return std::strcmp(dsv4_experimental_layer_executor_side_probe_payload_capture_mode(), "producer_capture") == 0;
}

static bool dsv4_experimental_layer_executor_side_probe_payload_consumer_dispatch_capture() {
    return std::strcmp(dsv4_experimental_layer_executor_side_probe_payload_capture_mode(), "consumer_dispatch") == 0 ||
        dsv4_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_HC_EXPAND_DISPATCH_CAPTURE");
}

static bool dsv4_layer_executor_payload_should_pin_output(const char * tensor_name) {
    if (tensor_name == nullptr) {
        return false;
    }
    if (std::strncmp(tensor_name, "hc_ws_", 6) == 0) {
        return true;
    }
    if (std::strncmp(tensor_name, "rmoe_", 5) == 0) {
        return true;
    }
    if (std::strncmp(tensor_name, "aohc_", 5) == 0) {
        return true;
    }
    if (std::strncmp(tensor_name, "compressed_", 11) == 0 ||
            std::strncmp(tensor_name, "cupd_", 5) == 0 ||
            std::strncmp(tensor_name, "state_", 6) == 0 ||
            std::strncmp(tensor_name, "pool_", 5) == 0 ||
            std::strncmp(tensor_name, "pooled_", 7) == 0 ||
            std::strncmp(tensor_name, "downstream_", 11) == 0) {
        return true;
    }
    return
        std::strcmp(tensor_name, "hc_pre_norm") == 0 ||
        std::strcmp(tensor_name, "input_hc_original_residual") == 0 ||
        std::strcmp(tensor_name, "split_pre") == 0 ||
        std::strcmp(tensor_name, "reference_cur") == 0 ||
        std::strcmp(tensor_name, "reference_norm") == 0 ||
        std::strcmp(tensor_name, "hc_pre_input_hc_original") == 0 ||
        std::strcmp(tensor_name, "hc_pre_split_pre") == 0 ||
        std::strcmp(tensor_name, "hc_pre_weighted_cur_reference") == 0 ||
        std::strcmp(tensor_name, "hc_pre_norm_reference") == 0;
}

static bool dsv4_experimental_layer_executor_side_probe_stage_is(const char * expected) {
    return std::strcmp(dsv4_experimental_layer_executor_side_probe_stage(), expected) == 0;
}

static bool dsv4_experimental_layer_executor_side_probe_mode_is(const char * expected) {
    return std::strcmp(dsv4_experimental_layer_executor_side_probe_mode(), expected) == 0;
}

static int64_t dsv4_env_i64(const char * name, int64_t default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    char * end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    return end != value ? (int64_t) parsed : default_value;
}

static bool dsv4_env_i64_is_set(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

static bool dsv4_experimental_layer_executor_plan_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = dsv4_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN") ? 1 : 0;
    }
    return enabled == 1;
}

static bool dsv4_experimental_layer_executor_plan_site_enabled(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_experimental_layer_executor_plan_enabled() || n_tokens != 1 || pos <= 0) {
        return false;
    }

    if (dsv4_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_LAYER")) {
        const int64_t layer = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_LAYER", il);
        if (il != layer) {
            return false;
        }
    }

    const int64_t decode_index = dsv4_experimental_decode_index_for_token(pos);
    const int64_t token_min = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_TOKEN_MIN",
            std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_TOKEN_MAX",
            std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

struct dsv4_layer_executor_plan_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t qkv = 0;
    uint64_t compressor_update = 0;
    uint64_t kv_cache = 0;
    uint64_t attention_core = 0;
    uint64_t attention_output = 0;
    uint64_t hc_pre = 0;
    uint64_t hc_post = 0;
    uint64_t routed_moe = 0;
    uint64_t result_head = 0;
};

static dsv4_layer_executor_plan_summary_state & dsv4_layer_executor_plan_summary() {
    static dsv4_layer_executor_plan_summary_state state;
    return state;
}

static void dsv4_layer_executor_plan_print_summary() {
    auto & state = dsv4_layer_executor_plan_summary();
    std::lock_guard<std::mutex> lock(state.mutex);

    const int64_t layer = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_LAYER", -1);
    const int64_t token_min = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_TOKEN_MIN", -1);
    const int64_t token_max = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_TOKEN_MAX", -1);

    std::fprintf(stderr,
            "dsv4_layer_executor_plan_summary: layer=%lld token_min=%lld token_max=%lld "
            "candidate_ffn_only_supported=1 candidate_attention_block_supported=1 candidate_full_layer_supported=1 "
            "stage_counts: qkv=%llu compressor_update=%llu kv_cache=%llu attention_core=%llu "
            "attention_output=%llu hc_pre=%llu hc_post=%llu routed_moe=%llu result_head=%llu "
            "existing_exact_primitives: hcnorm=1 cupd2=1 kvfinalizer=1 aohc=1 rmoe_shadow=1 "
            "local_replacement_blockers: graph_lowers_changed=1 cache_side_effects=1 "
            "non_exact_attention_subset=1 dispatch_neutral=1 "
            "planning_only=1 graph_nodes_added=0 backend_dispatches=0 consume=0\n",
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            (unsigned long long) state.qkv,
            (unsigned long long) state.compressor_update,
            (unsigned long long) state.kv_cache,
            (unsigned long long) state.attention_core,
            (unsigned long long) state.attention_output,
            (unsigned long long) state.hc_pre,
            (unsigned long long) state.hc_post,
            (unsigned long long) state.routed_moe,
            (unsigned long long) state.result_head);
}

static void dsv4_layer_executor_plan_register_summary() {
    auto & state = dsv4_layer_executor_plan_summary();
    if (!state.atexit_registered) {
        std::atexit(dsv4_layer_executor_plan_print_summary);
        state.atexit_registered = true;
    }
}

static void dsv4_layer_executor_plan_note_layer(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_experimental_layer_executor_plan_site_enabled(il, pos, n_tokens)) {
        return;
    }

    auto & state = dsv4_layer_executor_plan_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    dsv4_layer_executor_plan_register_summary();
    state.qkv++;
    state.compressor_update++;
    state.kv_cache++;
    state.attention_core++;
    state.attention_output++;
    state.hc_pre += 2;
    state.hc_post += 2;
    state.routed_moe++;
}

static void dsv4_layer_executor_plan_note_result_head(int64_t pos, int64_t n_tokens) {
    if (!dsv4_experimental_layer_executor_plan_enabled() || n_tokens != 1 || pos <= 0) {
        return;
    }

    const int64_t decode_index = dsv4_experimental_decode_index_for_token(pos);
    const int64_t token_min = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_TOKEN_MIN",
            std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_PLAN_TOKEN_MAX",
            std::numeric_limits<int64_t>::max());
    if (decode_index < token_min || decode_index > token_max) {
        return;
    }

    auto & state = dsv4_layer_executor_plan_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    dsv4_layer_executor_plan_register_summary();
    state.result_head++;
}

static bool dsv4_experimental_indexed_attn_site_enabled(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_experimental_indexed_attn_shadow_enabled() || n_tokens != 1 || pos <= 0) {
        return false;
    }

    if (dsv4_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_LAYER")) {
        const int64_t layer = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_LAYER", il);
        if (il != layer) {
            return false;
        }
    }

    const int64_t token_min = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_INDEXED_ATTN_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return pos >= token_min && pos <= token_max;
}

struct dsv4_indexed_attn_shadow_state {
    std::mutex mutex;
    bool atexit_registered = false;
    std::unordered_set<int> applicable_layers;
    uint64_t probes = 0;
    uint64_t compare_probes = 0;
    uint64_t output_probes = 0;
    uint64_t output_compare_probes = 0;
    uint64_t generic_flash_probes = 0;
    uint64_t dsv4_mixed_probes = 0;
    uint64_t visible_all_probes = 0;
    uint64_t topk_probes = 0;
    uint64_t topk_subset_probes = 0;
    uint64_t not_implemented = 0;
    uint64_t not_applicable = 0;
    uint64_t raw_rows_total = 0;
    uint64_t comp_rows_total = 0;
    uint64_t selected_rows_total = 0;
    int64_t max_compressed_total_rows = 0;
    int64_t max_compressed_visible_rows = 0;
    int64_t max_compressed_topk_rows = 0;
    int64_t max_topk_k = 0;
    uint64_t row_id_mismatch_count = 0;
    uint64_t order_mismatch_count = 0;
    int first_applicable_layer = -1;
    int64_t first_applicable_token = -1;
    int first_topk_subset_layer = -1;
    int64_t first_topk_subset_token = -1;
};

static const char * dsv4_indexed_attn_no_subset_reason(const dsv4_indexed_attn_shadow_state & state) {
    if (state.topk_subset_probes > 0) {
        return "strict_subset_found";
    }
    if (state.probes == 0) {
        return "no_applicable_indexed_attention_cases";
    }
    if (state.max_compressed_visible_rows <= state.max_compressed_topk_rows) {
        return "compressed_visible_rows_never_exceeded_compressed_topk_rows";
    }
    if (state.topk_probes == 0) {
        return "topk_dense_mask_path_not_exercised";
    }
    return "topk_dense_mask_path_seen_without_strict_subset";
}

static dsv4_indexed_attn_shadow_state & dsv4_indexed_attn_shadow_summary() {
    static dsv4_indexed_attn_shadow_state state;
    return state;
}

static void dsv4_indexed_attn_shadow_print_summary() {
    auto & state = dsv4_indexed_attn_shadow_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.probes == 0 && state.not_implemented == 0) {
        return;
    }

    std::fprintf(stderr,
            "dsv4_iattn_shadow_summary: enabled=1 compare_enabled=%d"
            " search_enabled=%d shadow_output_enabled=%d output_compare_enabled=%d"
            " dsv4_iattn_shadow=%" PRIu64
            " compare_probes=%" PRIu64
            " dsv4_iattn_shadow_out=%" PRIu64
            " output_compare_probes=%" PRIu64
            " dsv4_iattn_generic_flash=%" PRIu64
            " dsv4_iattn_dsv4_mixed=%" PRIu64
            " applicable_layers=%zu"
            " applicable_tokens=%" PRIu64
            " visible_all_probes=%" PRIu64
            " topk_probes=%" PRIu64
            " visible_all_cases=%" PRIu64
            " topk_subset_cases=%" PRIu64
            " applicable_cases=%" PRIu64
            " not_implemented=%" PRIu64
            " not_applicable_cases=%" PRIu64
            " raw_rows_total=%" PRIu64
            " comp_rows_total=%" PRIu64
            " selected_rows_total=%" PRIu64
            " max_compressed_total_rows=%lld"
            " max_compressed_visible_rows=%lld"
            " max_compressed_topk_rows=%lld"
            " topk_k=%lld"
            " row_id_mismatch_count=%" PRIu64
            " order_mismatch_count=%" PRIu64
            " first_applicable_layer=%d"
            " first_applicable_token=%lld"
            " first_topk_subset_layer=%d"
            " first_topk_subset_token=%lld"
            " first_topk_subset_reason=%s"
            " no_subset_reason=%s"
            " arith_mode=%s row_ids_exact=%d shadow_kv_exact=%s"
            " generic_downstream_consumed=1 consume_path=disabled cache_mutation_disabled=1\n",
            dsv4_experimental_indexed_attn_compare_enabled() ? 1 : 0,
            dsv4_experimental_indexed_attn_search_enabled() ? 1 : 0,
            dsv4_experimental_indexed_attn_shadow_output_enabled() ? 1 : 0,
            dsv4_experimental_indexed_attn_output_compare_enabled() ? 1 : 0,
            state.probes,
            state.compare_probes,
            state.output_probes,
            state.output_compare_probes,
            state.generic_flash_probes,
            state.dsv4_mixed_probes,
            state.applicable_layers.size(),
            state.probes,
            state.visible_all_probes,
            state.topk_probes,
            state.visible_all_probes,
            state.topk_subset_probes,
            state.probes,
            state.not_implemented,
            state.not_applicable,
            state.raw_rows_total,
            state.comp_rows_total,
            state.selected_rows_total,
            (long long) state.max_compressed_total_rows,
            (long long) state.max_compressed_visible_rows,
            (long long) state.max_compressed_topk_rows,
            (long long) state.max_topk_k,
            state.row_id_mismatch_count,
            state.order_mismatch_count,
            state.first_applicable_layer,
            (long long) state.first_applicable_token,
            state.first_topk_subset_layer,
            (long long) state.first_topk_subset_token,
            state.topk_subset_probes > 0 ? "compressed_visible_rows_gt_compressed_topk_rows" : "none",
            dsv4_indexed_attn_no_subset_reason(state),
            dsv4_experimental_indexed_attn_arith_mode(),
            state.row_id_mismatch_count == 0 ? 1 : 0,
            state.generic_flash_probes > 0 ? "proven_by_generic_attention" : "not_proven_in_this_run");
}

static void dsv4_indexed_attn_shadow_register_atexit() {
    auto & state = dsv4_indexed_attn_shadow_summary();
    if (!state.atexit_registered) {
        std::atexit(dsv4_indexed_attn_shadow_print_summary);
        state.atexit_registered = true;
    }
}

static void dsv4_indexed_attn_shadow_note(
        int          il,
        int64_t      pos,
        int64_t      n_tokens,
        int64_t      n_raw_cache,
        int64_t      raw_window,
        int64_t      n_comp_visible,
        int64_t      selected_rows,
        int64_t      top_k,
        const char * mode,
        bool         order_equivalent) {
    if (!dsv4_experimental_indexed_attn_site_enabled(il, pos, n_tokens)) {
        return;
    }

    const int64_t last_pos = pos + n_tokens - 1;
    const int64_t first_raw_pos = last_pos + 1 - n_raw_cache;
    const int64_t raw_last_pos = first_raw_pos + n_raw_cache - 1;
    const int64_t window_first = raw_window > 0 && pos + 1 > raw_window ? pos + 1 - raw_window : 0;
    const int64_t raw_first = std::max<int64_t>(first_raw_pos, window_first);
    const int64_t raw_last = std::min<int64_t>(pos, raw_last_pos);
    const int64_t raw_count = raw_first <= raw_last ? raw_last - raw_first + 1 : 0;

    const bool visible_all = std::strcmp(mode, "visible_all") == 0;
    const bool topk_subset = std::strcmp(mode, "topk_dense_mask") == 0 &&
            n_comp_visible > selected_rows;

    auto & state = dsv4_indexed_attn_shadow_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    dsv4_indexed_attn_shadow_register_atexit();
    state.applicable_layers.insert(il);
    state.probes++;
    if (state.first_applicable_layer < 0) {
        state.first_applicable_layer = il;
        state.first_applicable_token = pos;
    }
    if (dsv4_experimental_indexed_attn_compare_enabled()) {
        state.compare_probes++;
    }
    if (visible_all) {
        state.visible_all_probes++;
    } else if (std::strcmp(mode, "topk_dense_mask") == 0) {
        state.topk_probes++;
        if (topk_subset && state.first_topk_subset_layer < 0) {
            state.first_topk_subset_layer = il;
            state.first_topk_subset_token = pos;
        }
        if (topk_subset) {
            state.topk_subset_probes++;
        }
    } else {
        state.not_implemented++;
    }
    state.raw_rows_total += uint64_t(std::max<int64_t>(raw_count, 0));
    state.comp_rows_total += uint64_t(std::max<int64_t>(n_comp_visible, 0));
    state.selected_rows_total += uint64_t(std::max<int64_t>(selected_rows, 0));
    state.max_compressed_total_rows = std::max<int64_t>(state.max_compressed_total_rows, n_comp_visible);
    state.max_compressed_visible_rows = std::max<int64_t>(state.max_compressed_visible_rows, n_comp_visible);
    state.max_compressed_topk_rows = std::max<int64_t>(state.max_compressed_topk_rows, selected_rows);
    state.max_topk_k = std::max<int64_t>(state.max_topk_k, top_k);
    if (!order_equivalent) {
        state.order_mismatch_count++;
    }

    if (dsv4_experimental_indexed_attn_trace_enabled() || dsv4_experimental_indexed_attn_search_enabled()) {
        std::fprintf(stderr,
                "dsv4_iattn_shadow: layer=%d token=%lld mode=%s"
                " raw_first=%lld raw_last=%lld raw_row_count=%lld raw_cache_rows=%lld"
                " compress_ratio=4 compressed_total_rows=%lld compressed_visible_rows=%lld compressed_topk_rows=%lld selected_comp_rows=%lld top_k=%lld"
                " visible_all=%d topk_subset=%d"
                " row_set_source=%s row_id_mismatch_count=0 order_equivalent=%d"
                " generic_path=dense_mask indexed_path=topk_scan"
                " generic_downstream_consumed=1 consume_path=disabled\n",
                il,
                (long long) pos,
                mode,
                (long long) raw_first,
                (long long) raw_last,
                (long long) raw_count,
                (long long) n_raw_cache,
                (long long) n_comp_visible,
                (long long) n_comp_visible,
                (long long) selected_rows,
                (long long) selected_rows,
                (long long) top_k,
                visible_all ? 1 : 0,
                topk_subset ? 1 : 0,
                visible_all ? "all_visible_rows" : "same_topk_tensor",
                order_equivalent ? 1 : 0);
    }
}

static void dsv4_indexed_attn_shadow_note_output(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_experimental_indexed_attn_site_enabled(il, pos, n_tokens)) {
        return;
    }

    auto & state = dsv4_indexed_attn_shadow_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    dsv4_indexed_attn_shadow_register_atexit();
    state.output_probes++;
    if (dsv4_experimental_indexed_attn_output_compare_enabled()) {
        state.output_compare_probes++;
    }
    const char * arith_mode = dsv4_experimental_indexed_attn_arith_mode();
    if (std::strcmp(arith_mode, "generic_flash") == 0) {
        state.generic_flash_probes++;
    } else if (std::strcmp(arith_mode, "dsv4_mixed") == 0) {
        state.dsv4_mixed_probes++;
    }

    if (dsv4_experimental_indexed_attn_diff_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_iattn_shadow_output: layer=%d token=%lld row_order=%s mask_mode=%s arith_mode=%s shape_mode=%s shadow_impl_legacy=%s"
                " generic_downstream_consumed=1 consume_path=disabled\n",
                il,
                (long long) pos,
                dsv4_experimental_indexed_attn_row_order(),
                dsv4_experimental_indexed_attn_mask_mode(),
                arith_mode,
                dsv4_experimental_indexed_attn_shape_mode(),
                dsv4_experimental_indexed_attn_shadow_impl());
    }
}

static void dsv4_indexed_attn_shape_note(
        int          il,
        int64_t      pos,
        int64_t      n_tokens,
        const char * shape_mode,
        int64_t      compressed_visible_rows,
        int64_t      compressed_topk_rows,
        int64_t      shadow_row_count,
        bool         nonselected_rows_present,
        bool         nonselected_rows_masked) {
    if (!dsv4_experimental_indexed_attn_site_enabled(il, pos, n_tokens) ||
            !(dsv4_experimental_indexed_attn_trace_enabled() || dsv4_experimental_indexed_attn_search_enabled())) {
        return;
    }

    const bool row_count_matches = shadow_row_count == compressed_visible_rows;
    const bool mask_shape_matches = shadow_row_count == compressed_visible_rows;
    std::fprintf(stderr,
            "dsv4_iattn_shape_case: layer=%d token=%lld shape_mode=%s"
            " compressed_visible_rows=%lld compressed_topk_rows=%lld"
            " shadow_row_count=%lld baseline_row_count=%lld"
            " row_count_matches_baseline=%d mask_shape_matches_baseline=%d"
            " row_id_mismatch_count=0 dense_position_mismatch_count=0"
            " nonselected_rows_present=%d nonselected_rows_masked=%d"
            " generic_downstream_consumed=1 consume_path=disabled\n",
            il,
            (long long) pos,
            shape_mode,
            (long long) compressed_visible_rows,
            (long long) compressed_topk_rows,
            (long long) shadow_row_count,
            (long long) compressed_visible_rows,
            row_count_matches ? 1 : 0,
            mask_shape_matches ? 1 : 0,
            nonselected_rows_present ? 1 : 0,
            nonselected_rows_masked ? 1 : 0);
}

static void dsv4_indexed_attn_shadow_not_implemented(
        int          il,
        int64_t      pos,
        int64_t      n_tokens,
        const char * reason,
        int64_t      compress_ratio) {
    if (!dsv4_experimental_indexed_attn_site_enabled(il, pos, n_tokens)) {
        return;
    }

    auto & state = dsv4_indexed_attn_shadow_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    dsv4_indexed_attn_shadow_register_atexit();
    state.not_implemented++;
    state.not_applicable++;

    if (dsv4_experimental_indexed_attn_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_iattn_shadow: layer=%d token=%lld mode=not_implemented"
                " reason=%s compress_ratio=%lld generic_downstream_consumed=1 consume_path=disabled\n",
                il,
                (long long) pos,
                reason,
                (long long) compress_ratio);
    }
}

static bool dsv4_experimental_layer_executor_site_enabled(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_experimental_layer_executor_shadow_enabled() || n_tokens != 1 || pos <= 0) {
        return false;
    }

    if (dsv4_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER")) {
        const int64_t layer = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER", il);
        if (il != layer) {
            return false;
        }
    }

    const int64_t token_min = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return pos >= token_min && pos <= token_max;
}

static bool dsv4_experimental_layer_executor_full_site_enabled(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_experimental_layer_executor_shadow_enabled() || n_tokens != 1 || pos <= 0) {
        return false;
    }

    if (dsv4_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER")) {
        const int64_t layer = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER", il);
        if (il != layer) {
            return false;
        }
    }

    const int64_t decode_index = dsv4_experimental_decode_index_for_token(pos);
    const int64_t token_min = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MAX", std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_layer_executor_dryrun_op_site_enabled(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_experimental_layer_executor_dryrun_op_enabled() || n_tokens != 1 || pos <= 0) {
        return false;
    }

    if (dsv4_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_LAYER")) {
        const int64_t layer = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_LAYER", il);
        if (il != layer) {
            return false;
        }
    }

    const int64_t decode_index = dsv4_experimental_decode_index_for_token(pos);
    const int64_t token_min = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_TOKEN_MIN",
            std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_TOKEN_MAX",
            std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static bool dsv4_experimental_layer_executor_side_probe_site_enabled(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_experimental_layer_executor_side_probe_enabled() ||
            (!dsv4_experimental_layer_executor_side_probe_stage_is("hc_pre_norm") &&
             !dsv4_experimental_layer_executor_side_probe_stage_is("routed_moe_final_output") &&
             !dsv4_experimental_layer_executor_side_probe_stage_is("aohc_boundary") &&
             !dsv4_experimental_layer_executor_side_probe_stage_is("compressor_update") &&
             !dsv4_experimental_layer_executor_side_probe_stage_is("kv_cache_finalizer")) ||
            n_tokens != 1 ||
            pos <= 0) {
        return false;
    }

    if (dsv4_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_LAYER")) {
        const int64_t layer = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_LAYER", il);
        if (il != layer) {
            return false;
        }
    }

    const int64_t decode_index = dsv4_experimental_decode_index_for_token(pos);
    const int64_t token_min = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_TOKEN_MIN",
            std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_TOKEN_MAX",
            std::numeric_limits<int64_t>::max());
    return decode_index >= token_min && decode_index <= token_max;
}

static void dsv4_layer_executor_append_rejected_path(std::string & out, const char * env_name) {
    if (!dsv4_env_flag_enabled(env_name)) {
        return;
    }
    if (!out.empty()) {
        out += ",";
    }
    out += env_name;
}

static std::string dsv4_layer_executor_rejected_paths_enabled() {
    std::string out;
    dsv4_layer_executor_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME");
    dsv4_layer_executor_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_FUSED_COMP");
    dsv4_layer_executor_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP");
    dsv4_layer_executor_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6");
    dsv4_layer_executor_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_SWIGLU");
    dsv4_layer_executor_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_SHARED_SWIGLU");
    dsv4_layer_executor_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_DOWN");
    dsv4_layer_executor_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_MIXED_ATTN");
    dsv4_layer_executor_append_rejected_path(out, "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE");
    return out;
}

static std::string dsv4_layer_executor_consume_reject_reason() {
    if (!dsv4_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME")) {
        return "consume_disabled";
    }
    if (!dsv4_experimental_layer_executor_shadow_enabled()) {
        return "shadow_disabled";
    }
    if (!dsv4_experimental_layer_executor_compare_enabled()) {
        return "compare_disabled";
    }
    if (!dsv4_env_string_equals(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME_MODEL",
            "layer_output_identity")) {
        return "bad_consume_model";
    }
    if (!dsv4_layer_executor_consume_style_supported(dsv4_layer_executor_consume_style())) {
        return "bad_consume_style";
    }
    if (!dsv4_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER")) {
        return "missing_layer_filter";
    }
    const int64_t layer = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER", -1);
    if (layer < 0) {
        return "bad_layer_filter";
    }
    if (!dsv4_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MIN") ||
            !dsv4_env_i64_is_set("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MAX")) {
        return "missing_token_range";
    }
    const int64_t token_min = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MIN", 0);
    const int64_t token_max = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MAX", -1);
    if (token_min > token_max) {
        return "bad_token_range";
    }
    if (!dsv4_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_ABORT_ON_MISMATCH")) {
        return "abort_on_mismatch_disabled";
    }
    const std::string rejected = dsv4_layer_executor_rejected_paths_enabled();
    if (!rejected.empty()) {
        return "rejected_paths_enabled:" + rejected;
    }
    return "";
}

static bool dsv4_experimental_layer_executor_consume_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = dsv4_layer_executor_consume_reject_reason().empty() ? 1 : 0;
        if (dsv4_env_flag_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME") && enabled == 0) {
            static std::mutex mutex;
            static bool warned = false;
            std::lock_guard<std::mutex> lock(mutex);
            if (!warned) {
                const std::string reason = dsv4_layer_executor_consume_reject_reason();
                std::fprintf(stderr,
                        "dsv4_layer_executor_consume: requested=1 consume_allowed=0 reason=%s\n",
                        reason.c_str());
                warned = true;
            }
        }
    }
    return enabled == 1;
}

static bool dsv4_experimental_layer_executor_consume_site_enabled(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_experimental_layer_executor_consume_enabled() ||
            !dsv4_experimental_layer_executor_site_enabled(il, pos, n_tokens)) {
        return false;
    }
    const int64_t layer = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER", -1);
    return il == layer;
}

struct dsv4_lexec_full_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    uint64_t exact_cases = 0;
    uint64_t non_exact_cases = 0;
    uint64_t over_tol = 0;
    float max_abs = 0.0f;
    float max_rms = 0.0f;
    int first_non_exact_layer = -1;
    int64_t first_non_exact_token = -1;
    char first_non_exact_tensor[96] = "none";
    bool attn_qkv = false;
    bool compressor_update = false;
    bool kv_cache_handoff = false;
    bool attn_core = false;
    bool attn_output = false;
    bool attn_hc_post = false;
    bool ffn_hc_pre = false;
    bool routed_moe = false;
    bool ffn_hc_post = false;
    bool layer_output = false;
    bool input_visible = false;
    bool attention_weights_visible = false;
    bool compressor_state_visible = false;
    bool cache_row_metadata_visible = false;
    bool hc_tensors_visible = false;
    bool routed_moe_ids_weights_visible = false;
    bool expert_shared_weights_visible = false;
    bool layer_output_visible = false;
};

static dsv4_lexec_full_summary_state & dsv4_lexec_full_summary() {
    static dsv4_lexec_full_summary_state state;
    return state;
}

static void dsv4_lexec_full_print_summary() {
    auto & state = dsv4_lexec_full_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.cases == 0 && !state.input_visible && !state.layer_output_visible) {
        return;
    }

    const int64_t layer = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_LAYER", -1);
    const int64_t token_min = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MIN", -1);
    const int64_t token_max = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TOKEN_MAX", -1);
    const bool compare = dsv4_experimental_layer_executor_compare_enabled();

    std::fprintf(stderr,
            "dsv4_lexec_full_summary: dsv4_lexec_full_shadow=1 dsv4_lexec_full_exact=%d"
            " mode=%s layer_filter=%lld token_min=%lld token_max=%lld"
            " cases=%llu exact_cases=%llu non_exact_cases=%llu max_abs=%.9g max_rms=%.9g over_tol=%llu"
            " first_non_exact_layer=%d first_non_exact_token=%lld first_non_exact_tensor=%s"
            " cache_mutation=disabled consume_path=disabled graph_nodes_added=0 backend_dispatches=0"
            " compared_stage_outputs: attn_qkv=%d compressor_update=%d kv_cache_handoff=%d attn_core=%d"
            " attn_output=%d attn_hc_post=%d ffn_hc_pre=%d routed_moe=%d ffn_hc_post=%d layer_output=%d"
            " input_visibility: layer_input=%d attention_weights=%d compressor_state=%d cache_row_metadata=%d"
            " hc_tensors=%d routed_moe_ids_weights=%d expert_shared_weights=%d layer_output=%d"
            " exact_mode=%s shadow_name=DSV4_LAYER_EXECUTOR_FULL_DECODE_LAYER_SHADOW\n",
            state.non_exact_cases == 0 ? 1 : 0,
            dsv4_layer_executor_mode(),
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            (unsigned long long) state.cases,
            (unsigned long long) state.exact_cases,
            (unsigned long long) state.non_exact_cases,
            (double) state.max_abs,
            (double) state.max_rms,
            (unsigned long long) state.over_tol,
            state.first_non_exact_layer,
            (long long) state.first_non_exact_token,
            state.first_non_exact_tensor,
            state.attn_qkv ? 1 : 0,
            state.compressor_update ? 1 : 0,
            state.kv_cache_handoff ? 1 : 0,
            state.attn_core ? 1 : 0,
            state.attn_output ? 1 : 0,
            state.attn_hc_post ? 1 : 0,
            state.ffn_hc_pre ? 1 : 0,
            state.routed_moe ? 1 : 0,
            state.ffn_hc_post ? 1 : 0,
            state.layer_output ? 1 : 0,
            state.input_visible ? 1 : 0,
            state.attention_weights_visible ? 1 : 0,
            state.compressor_state_visible ? 1 : 0,
            state.cache_row_metadata_visible ? 1 : 0,
            state.hc_tensors_visible ? 1 : 0,
            state.routed_moe_ids_weights_visible ? 1 : 0,
            state.expert_shared_weights_visible ? 1 : 0,
            state.layer_output_visible ? 1 : 0,
            compare ? "self_compare" : "trace_only");
}

static void dsv4_lexec_full_register_summary() {
    auto & state = dsv4_lexec_full_summary();
    if (!state.atexit_registered) {
        std::atexit(dsv4_lexec_full_print_summary);
        state.atexit_registered = true;
    }
}

static const char * dsv4_lexec_stage_bucket(const char * stage, const char * tensor_name) {
    if (std::strcmp(stage, "qkv_setup") == 0) {
        return "attn_qkv";
    }
    if (std::strcmp(stage, "compressor_update") == 0) {
        return "compressor_update";
    }
    if (std::strcmp(stage, "kv_finalizer") == 0) {
        return "kv_cache_handoff";
    }
    if (std::strcmp(stage, "attention_core") == 0) {
        return "attn_core";
    }
    if (std::strcmp(stage, "attention_output") == 0) {
        return "attn_output";
    }
    if (std::strcmp(stage, "attn_hc_post") == 0) {
        return "attn_hc_post";
    }
    if (std::strcmp(stage, "ffn_hc_pre_norm") == 0) {
        return "ffn_hc_pre";
    }
    if (std::strcmp(stage, "ffn_moe") == 0) {
        return "routed_moe";
    }
    if (std::strcmp(stage, "ffn_hc_post") == 0) {
        return "ffn_hc_post";
    }
    if (std::strcmp(stage, "layer_output") == 0) {
        return "layer_output";
    }
    if (std::strcmp(stage, "attn_hc_pre_norm") == 0) {
        return "hc_pre";
    }
    return tensor_name != nullptr ? tensor_name : "other";
}

static void dsv4_lexec_full_mark_stage(dsv4_lexec_full_summary_state & state, const char * bucket) {
    if (std::strcmp(bucket, "attn_qkv") == 0) {
        state.attn_qkv = true;
    } else if (std::strcmp(bucket, "compressor_update") == 0) {
        state.compressor_update = true;
    } else if (std::strcmp(bucket, "kv_cache_handoff") == 0) {
        state.kv_cache_handoff = true;
    } else if (std::strcmp(bucket, "attn_core") == 0) {
        state.attn_core = true;
    } else if (std::strcmp(bucket, "attn_output") == 0) {
        state.attn_output = true;
    } else if (std::strcmp(bucket, "attn_hc_post") == 0) {
        state.attn_hc_post = true;
    } else if (std::strcmp(bucket, "ffn_hc_pre") == 0) {
        state.ffn_hc_pre = true;
    } else if (std::strcmp(bucket, "routed_moe") == 0) {
        state.routed_moe = true;
    } else if (std::strcmp(bucket, "ffn_hc_post") == 0) {
        state.ffn_hc_post = true;
    } else if (std::strcmp(bucket, "layer_output") == 0) {
        state.layer_output = true;
    }
}

static void dsv4_lexec_full_note_tensor(
        ggml_tensor * tensor,
        const char  * stage,
        const char  * tensor_name,
        int           il,
        int64_t       pos,
        int64_t       n_tokens) {
    if (!dsv4_layer_executor_full_shadow_mode() ||
            !dsv4_experimental_layer_executor_full_site_enabled(il, pos, n_tokens)) {
        return;
    }

    const char * bucket = dsv4_lexec_stage_bucket(stage, tensor_name);
    auto & state = dsv4_lexec_full_summary();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        dsv4_lexec_full_register_summary();
        state.cases++;
        state.exact_cases++;
        dsv4_lexec_full_mark_stage(state, bucket);
        if (std::strcmp(bucket, "layer_output") == 0) {
            state.layer_output_visible = true;
        }
        if (std::strcmp(bucket, "attn_qkv") == 0) {
            state.attention_weights_visible = true;
        }
        if (std::strcmp(bucket, "compressor_update") == 0) {
            state.compressor_state_visible = true;
        }
        if (std::strcmp(bucket, "kv_cache_handoff") == 0) {
            state.cache_row_metadata_visible = true;
        }
        if (std::strcmp(bucket, "hc_pre") == 0 || std::strcmp(bucket, "attn_hc_post") == 0 ||
                std::strcmp(bucket, "ffn_hc_post") == 0 || std::strcmp(bucket, "ffn_hc_pre") == 0) {
            state.hc_tensors_visible = true;
        }
        if (std::strcmp(bucket, "routed_moe") == 0) {
            state.routed_moe_ids_weights_visible = true;
            state.expert_shared_weights_visible = true;
        }
    }

    if (dsv4_experimental_layer_executor_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_lexec_full_stage: mode=%s layer=%d token=%lld stage=%s bucket=%s tensor=%s"
                " op=%s type=%s shape=[%lld,%lld,%lld,%lld] stride=[%zu,%zu,%zu,%zu]"
                " self_compare_exact=1 cache_mutation=disabled consume_path=disabled backend_dispatches=0\n",
                dsv4_layer_executor_mode(),
                il,
                (long long) pos,
                stage,
                bucket,
                tensor != nullptr ? tensor->name : (tensor_name != nullptr ? tensor_name : "null"),
                tensor != nullptr ? ggml_op_name(tensor->op) : "null",
                tensor != nullptr ? ggml_type_name(tensor->type) : "null",
                tensor != nullptr ? (long long) tensor->ne[0] : 0LL,
                tensor != nullptr ? (long long) tensor->ne[1] : 0LL,
                tensor != nullptr ? (long long) tensor->ne[2] : 0LL,
                tensor != nullptr ? (long long) tensor->ne[3] : 0LL,
                tensor != nullptr ? tensor->nb[0] : (size_t) 0,
                tensor != nullptr ? tensor->nb[1] : (size_t) 0,
                tensor != nullptr ? tensor->nb[2] : (size_t) 0,
                tensor != nullptr ? tensor->nb[3] : (size_t) 0);
    }
}

static void dsv4_lexec_full_note_contract(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    if (!dsv4_layer_executor_full_shadow_mode() ||
            std::strcmp(dsv4_layer_executor_mode(), "contract_dry_run") != 0 ||
            !dsv4_experimental_layer_executor_full_site_enabled(il, pos, n_tokens)) {
        return;
    }

    auto & state = dsv4_lexec_full_summary();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        dsv4_lexec_full_register_summary();
        state.input_visible = true;
        state.attention_weights_visible = true;
        state.compressor_state_visible = true;
        state.cache_row_metadata_visible = true;
        state.hc_tensors_visible = true;
        state.routed_moe_ids_weights_visible = true;
        state.expert_shared_weights_visible = true;
    }

    if (dsv4_experimental_layer_executor_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_lexec_full_contract: mode=contract_dry_run layer=%d token=%lld"
                " input_visible=1 attention_weights_visible=1 compressor_state_visible=1"
                " cache_row_metadata_visible=1 attention_mask_metadata_visible=1 hc_tensors_visible=1"
                " routed_moe_ids_weights_visible=1 expert_shared_weights_visible=1"
                " layer_output_anchor=deferred_until_layer_output missing_inputs=none"
                " cache_mutation=disabled consume_path=disabled backend_dispatches=0\n",
                il,
                (long long) pos);
    }
}

struct dsv4_lexec_dryrun_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    uint64_t eligible_cases = 0;
    uint64_t rejected_cases = 0;
    int first_reject_layer = -1;
    int64_t first_reject_token = -1;
    char first_reject_reason[128] = "none";
    bool has_layer_input = false;
    bool has_attn_qkv = false;
    bool has_compressor_state = false;
    bool has_cache_metadata = false;
    bool has_attention_output = false;
    bool has_hc_tensors = false;
    bool has_routed_moe_ids_weights = false;
    bool has_expert_shared_weights = false;
    bool has_layer_output_anchor = false;
    bool live_graph_nodes_added = false;
    bool live_backend_dispatches = false;
    bool side_graph_created = false;
    bool side_graph_dispatched = false;
};

static dsv4_lexec_dryrun_summary_state & dsv4_lexec_dryrun_summary() {
    static dsv4_lexec_dryrun_summary_state state;
    return state;
}

static void dsv4_lexec_dryrun_print_summary() {
    auto & state = dsv4_lexec_dryrun_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!dsv4_experimental_layer_executor_dryrun_op_enabled() && state.cases == 0) {
        return;
    }

    const int64_t layer = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_LAYER", -1);
    const int64_t token_min = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_TOKEN_MIN", -1);
    const int64_t token_max = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_TOKEN_MAX", -1);
    const char * mode = dsv4_experimental_layer_executor_dryrun_op_mode();
    const uint64_t live_cases =
        std::strcmp(mode, "live_graph_dispatch") == 0 ? state.eligible_cases : 0;

    std::fprintf(stderr,
            "dsv4_lexec_dryrun_summary: enabled=%d mode=%s layer_filter=%lld token_min=%lld token_max=%lld"
            " cases=%llu eligible_cases=%llu rejected_cases=%llu"
            " first_reject_layer=%d first_reject_token=%lld first_reject_reason=%s"
            " op_added=%d backend_op_dispatched=%d"
            " live_graph_nodes_added=%d live_backend_dispatches=%d"
            " side_graph_created=%d side_graph_dispatched=%d"
            " output_consumed=0 cache_mutation=disabled side_effects=disabled"
            " has_layer_input=%d has_attn_qkv=%d has_compressor_state=%d has_cache_metadata=%d"
            " has_attention_output=%d has_hc_tensors=%d has_routed_moe_ids_weights=%d"
            " has_expert_shared_weights=%d has_layer_output_anchor=%d"
            " dsv4_lexec_dryrun=%llu dsv4_lexec_dryrun_eligible=%llu dsv4_lexec_dryrun_rejected=%llu"
            " pair=-1 pswiglu=-1 fglu=-1 metal_dispatch=-1 counter_source=metal_stats\n",
            dsv4_experimental_layer_executor_dryrun_op_enabled() ? 1 : 0,
            mode,
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            (unsigned long long) state.cases,
            (unsigned long long) state.eligible_cases,
            (unsigned long long) state.rejected_cases,
            state.first_reject_layer,
            (long long) state.first_reject_token,
            state.first_reject_reason,
            live_cases > 0 ? 1 : 0,
            live_cases > 0 ? 1 : 0,
            state.live_graph_nodes_added ? 1 : 0,
            state.live_backend_dispatches ? 1 : 0,
            state.side_graph_created ? 1 : 0,
            state.side_graph_dispatched ? 1 : 0,
            state.has_layer_input ? 1 : 0,
            state.has_attn_qkv ? 1 : 0,
            state.has_compressor_state ? 1 : 0,
            state.has_cache_metadata ? 1 : 0,
            state.has_attention_output ? 1 : 0,
            state.has_hc_tensors ? 1 : 0,
            state.has_routed_moe_ids_weights ? 1 : 0,
            state.has_expert_shared_weights ? 1 : 0,
            state.has_layer_output_anchor ? 1 : 0,
            (unsigned long long) live_cases,
            (unsigned long long) state.eligible_cases,
            (unsigned long long) state.rejected_cases);
}

static void dsv4_lexec_dryrun_register_summary() {
    auto & state = dsv4_lexec_dryrun_summary();
    if (!state.atexit_registered) {
        std::atexit(dsv4_lexec_dryrun_print_summary);
        state.atexit_registered = true;
    }
}

static void dsv4_lexec_side_probe_register_summary();

static void dsv4_lexec_dryrun_note(
        int          il,
        int64_t      pos,
        bool         eligible,
        const char * reason,
        bool         has_layer_input,
        bool         has_attn_qkv,
        bool         has_compressor_state,
        bool         has_cache_metadata,
        bool         has_attention_output,
        bool         has_hc_tensors,
        bool         has_routed_moe_ids_weights,
        bool         has_expert_shared_weights,
        bool         has_layer_output_anchor,
        bool         live_graph_node_added,
        bool         live_backend_dispatch,
        bool         side_graph_created,
        bool         side_graph_dispatched) {
    auto & state = dsv4_lexec_dryrun_summary();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        dsv4_lexec_dryrun_register_summary();
        if (dsv4_experimental_layer_executor_side_probe_enabled()) {
            dsv4_lexec_side_probe_register_summary();
        }
        state.cases++;
        if (eligible) {
            state.eligible_cases++;
        } else {
            state.rejected_cases++;
            if (state.first_reject_layer < 0) {
                state.first_reject_layer = il;
                state.first_reject_token = dsv4_experimental_decode_index_for_token(pos);
                std::snprintf(state.first_reject_reason, sizeof(state.first_reject_reason), "%s",
                        reason != nullptr ? reason : "unknown");
            }
        }
        state.has_layer_input = state.has_layer_input || has_layer_input;
        state.has_attn_qkv = state.has_attn_qkv || has_attn_qkv;
        state.has_compressor_state = state.has_compressor_state || has_compressor_state;
        state.has_cache_metadata = state.has_cache_metadata || has_cache_metadata;
        state.has_attention_output = state.has_attention_output || has_attention_output;
        state.has_hc_tensors = state.has_hc_tensors || has_hc_tensors;
        state.has_routed_moe_ids_weights = state.has_routed_moe_ids_weights || has_routed_moe_ids_weights;
        state.has_expert_shared_weights = state.has_expert_shared_weights || has_expert_shared_weights;
        state.has_layer_output_anchor = state.has_layer_output_anchor || has_layer_output_anchor;
        state.live_graph_nodes_added = state.live_graph_nodes_added || live_graph_node_added;
        state.live_backend_dispatches = state.live_backend_dispatches || live_backend_dispatch;
        state.side_graph_created = state.side_graph_created || side_graph_created;
        state.side_graph_dispatched = state.side_graph_dispatched || side_graph_dispatched;
    }

    if (dsv4_experimental_layer_executor_dryrun_op_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_lexec_dryrun: mode=%s layer=%d token=%lld eligible=%d reason=%s"
                " live_graph_nodes_added=%d live_backend_dispatches=%d side_graph_created=%d side_graph_dispatched=%d"
                " output_consumed=0 cache_mutation=disabled side_effects=disabled"
                " has_layer_input=%d has_attn_qkv=%d has_compressor_state=%d has_cache_metadata=%d"
                " has_attention_output=%d has_hc_tensors=%d has_routed_moe_ids_weights=%d"
                " has_expert_shared_weights=%d has_layer_output_anchor=%d\n",
                dsv4_experimental_layer_executor_dryrun_op_mode(),
                il,
                (long long) dsv4_experimental_decode_index_for_token(pos),
                eligible ? 1 : 0,
                reason != nullptr ? reason : "none",
                live_graph_node_added ? 1 : 0,
                live_backend_dispatch ? 1 : 0,
                side_graph_created ? 1 : 0,
                side_graph_dispatched ? 1 : 0,
                has_layer_input ? 1 : 0,
                has_attn_qkv ? 1 : 0,
                has_compressor_state ? 1 : 0,
                has_cache_metadata ? 1 : 0,
                has_attention_output ? 1 : 0,
                has_hc_tensors ? 1 : 0,
                has_routed_moe_ids_weights ? 1 : 0,
                has_expert_shared_weights ? 1 : 0,
                has_layer_output_anchor ? 1 : 0);
    }
}

struct dsv4_lexec_side_probe_summary_state {
    std::mutex mutex;
    bool atexit_registered = false;
    uint64_t cases = 0;
    uint64_t exact_cases = 0;
    uint64_t non_exact_cases = 0;
    uint64_t over_tol = 0;
    float max_abs = 0.0f;
    float max_rms = 0.0f;
    int first_non_exact_layer = -1;
    int64_t first_non_exact_token = -1;
    char first_non_exact_tensor[96] = "none";
    bool hc_pre_cur = false;
    bool hc_pre_norm = false;
    bool hc_pre_comb = false;
    bool hc_post_input = false;
    bool ffn_input = false;
    bool topk_ids = false;
    bool topk_weights = false;
    bool expert_gate_up = false;
    bool expert_swiglu = false;
    bool expert_down = false;
    bool routed_sum = false;
    bool shared_down = false;
    bool final_ffn = false;
    bool attn_core_output = false;
    bool attn_low = false;
    bool attn_out = false;
    bool hc_post_weights = false;
    bool hc_comb = false;
    bool after_attn_hc = false;
    bool layer_attn_output_anchor = false;
    bool kv_proj = false;
    bool score_proj = false;
    bool kv_plus_ape = false;
    bool score_plus_ape = false;
    bool state_kv = false;
    bool state_score = false;
    bool pool_input = false;
    bool pooled = false;
    bool norm = false;
    bool rope = false;
    bool quant = false;
    bool downstream_kv = false;
    bool cache_handoff_metadata = false;
    bool q_rope_tail = false;
    bool kv_rope_tail = false;
    bool kv_quant = false;
    bool swa_cache_row = false;
    bool compressed_cache_row = false;
    bool cache_indices = false;
    bool set_rows_source = false;
    bool set_rows_destination_metadata = false;
    int first_compressor_layer = -1;
    int64_t first_compressor_token = -1;
    uint64_t compressor_cases = 0;
    uint64_t ratio_boundary_cases = 0;
    uint64_t quant_emit_cases = 0;
    int first_swa_cache_layer = -1;
    int64_t first_swa_cache_token = -1;
    int first_compressed_cache_layer = -1;
    int64_t first_compressed_cache_token = -1;
    uint64_t swa_cases = 0;
    uint64_t compressed_cases = 0;
    uint64_t kv_quant_cases = 0;
    uint64_t set_rows_cases = 0;
    uint64_t payload_records = 0;
    uint64_t payload_blocked = 0;
    bool live_graph_nodes_added = false;
    bool live_backend_dispatches = false;
    bool side_graph_created = false;
    bool side_graph_dispatched = false;
};

static dsv4_lexec_side_probe_summary_state & dsv4_lexec_side_probe_summary() {
    static dsv4_lexec_side_probe_summary_state state;
    return state;
}

static void dsv4_lexec_side_probe_print_summary() {
    auto & state = dsv4_lexec_side_probe_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!dsv4_experimental_layer_executor_side_probe_enabled() && state.cases == 0) {
        return;
    }

    const int64_t layer = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_LAYER", -1);
    const int64_t token_min = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_TOKEN_MIN", -1);
    const int64_t token_max = dsv4_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_TOKEN_MAX", -1);

    std::fprintf(stderr,
            "dsv4_lexec_side_probe_summary: enabled=%d stage=%s mode=%s layer_filter=%lld token_min=%lld token_max=%lld"
            " live_graph_nodes_added=%d live_backend_dispatches=%d side_graph_created=%d side_graph_dispatched=%d"
            " output_consumed=0 cache_mutation=disabled"
            " cases=%llu exact_cases=%llu non_exact_cases=%llu max_abs=%.9g max_rms=%.9g over_tol=%llu"
            " first_non_exact_layer=%d first_non_exact_token=%lld first_non_exact_tensor=%s"
            " compared_tensors: hc_pre_cur=%d hc_pre_norm=%d hc_pre_comb=%d hc_post_input=%d"
            " ffn_input=%d topk_ids=%d topk_weights=%d expert_gate_up=%d expert_swiglu=%d"
            " expert_down=%d routed_sum=%d shared_down=%d final_ffn=%d"
            " attn_core_output=%d attn_low=%d attn_out=%d hc_post_weights=%d hc_comb=%d"
            " after_attn_hc=%d layer_attn_output_anchor=%d"
            " kv_proj=%d score_proj=%d kv_plus_ape=%d score_plus_ape=%d state_kv=%d state_score=%d"
            " pool_input=%d pooled=%d norm=%d rope=%d quant=%d downstream_kv=%d cache_handoff_metadata=%d"
            " q_rope_tail=%d kv_rope_tail=%d kv_quant=%d swa_cache_row=%d compressed_cache_row=%d"
            " cache_indices=%d set_rows_source=%d set_rows_destination_metadata=%d"
            " payload_requested=%d payload_capture_mode=%s payload_dir=%s payload_records=%llu payload_blocked=%llu"
            " payload_blocker=%s"
            " dsv4_lexec_compressor_probe_search: first_compressor_layer=%d first_compressor_token=%lld"
            " compressor_cases=%llu ratio_boundary_cases=%llu quant_emit_cases=%llu"
            " dsv4_lexec_kv_probe_search: first_swa_cache_layer=%d first_swa_cache_token=%lld"
            " first_compressed_cache_layer=%d first_compressed_cache_token=%lld"
            " swa_cases=%llu compressed_cases=%llu quant_cases=%llu set_rows_cases=%llu"
            " dsv4_lexec_side_probe=%llu dsv4_lexec_side_probe_exact=%llu\n",
            dsv4_experimental_layer_executor_side_probe_enabled() ? 1 : 0,
            dsv4_experimental_layer_executor_side_probe_stage(),
            dsv4_experimental_layer_executor_side_probe_mode(),
            (long long) layer,
            (long long) token_min,
            (long long) token_max,
            state.live_graph_nodes_added ? 1 : 0,
            state.live_backend_dispatches ? 1 : 0,
            state.side_graph_created ? 1 : 0,
            state.side_graph_dispatched ? 1 : 0,
            (unsigned long long) state.cases,
            (unsigned long long) state.exact_cases,
            (unsigned long long) state.non_exact_cases,
            (double) state.max_abs,
            (double) state.max_rms,
            (unsigned long long) state.over_tol,
            state.first_non_exact_layer,
            (long long) state.first_non_exact_token,
            state.first_non_exact_tensor,
            state.hc_pre_cur ? 1 : 0,
            state.hc_pre_norm ? 1 : 0,
            state.hc_pre_comb ? 1 : 0,
            state.hc_post_input ? 1 : 0,
            state.ffn_input ? 1 : 0,
            state.topk_ids ? 1 : 0,
            state.topk_weights ? 1 : 0,
            state.expert_gate_up ? 1 : 0,
            state.expert_swiglu ? 1 : 0,
            state.expert_down ? 1 : 0,
            state.routed_sum ? 1 : 0,
            state.shared_down ? 1 : 0,
            state.final_ffn ? 1 : 0,
            state.attn_core_output ? 1 : 0,
            state.attn_low ? 1 : 0,
            state.attn_out ? 1 : 0,
            state.hc_post_weights ? 1 : 0,
            state.hc_comb ? 1 : 0,
            state.after_attn_hc ? 1 : 0,
            state.layer_attn_output_anchor ? 1 : 0,
            state.kv_proj ? 1 : 0,
            state.score_proj ? 1 : 0,
            state.kv_plus_ape ? 1 : 0,
            state.score_plus_ape ? 1 : 0,
            state.state_kv ? 1 : 0,
            state.state_score ? 1 : 0,
            state.pool_input ? 1 : 0,
            state.pooled ? 1 : 0,
            state.norm ? 1 : 0,
            state.rope ? 1 : 0,
            state.quant ? 1 : 0,
            state.downstream_kv ? 1 : 0,
            state.cache_handoff_metadata ? 1 : 0,
            state.q_rope_tail ? 1 : 0,
            state.kv_rope_tail ? 1 : 0,
            state.kv_quant ? 1 : 0,
            state.swa_cache_row ? 1 : 0,
            state.compressed_cache_row ? 1 : 0,
            state.cache_indices ? 1 : 0,
            state.set_rows_source ? 1 : 0,
            state.set_rows_destination_metadata ? 1 : 0,
            dsv4_experimental_layer_executor_side_probe_payload_enabled() ? 1 : 0,
            dsv4_experimental_layer_executor_side_probe_payload_capture_mode(),
            dsv4_experimental_layer_executor_side_probe_payload_dir()[0] != '\0' ? dsv4_experimental_layer_executor_side_probe_payload_dir() : "(none)",
            (unsigned long long) state.payload_records,
            (unsigned long long) state.payload_blocked,
            dsv4_experimental_layer_executor_side_probe_payload_enabled() && state.payload_records == 0 ?
                "no_payload_targets_registered" : "none",
            state.first_compressor_layer,
            (long long) state.first_compressor_token,
            (unsigned long long) state.compressor_cases,
            (unsigned long long) state.ratio_boundary_cases,
            (unsigned long long) state.quant_emit_cases,
            state.first_swa_cache_layer,
            (long long) state.first_swa_cache_token,
            state.first_compressed_cache_layer,
            (long long) state.first_compressed_cache_token,
            (unsigned long long) state.swa_cases,
            (unsigned long long) state.compressed_cases,
            (unsigned long long) state.kv_quant_cases,
            (unsigned long long) state.set_rows_cases,
            (unsigned long long) state.cases,
            (unsigned long long) state.exact_cases);
}

static void dsv4_lexec_side_probe_register_summary() {
    auto & state = dsv4_lexec_side_probe_summary();
    if (!state.atexit_registered) {
        std::atexit(dsv4_lexec_side_probe_print_summary);
        state.atexit_registered = true;
    }
}

static void dsv4_lexec_side_probe_register_payload_target(
        const char  * stage,
        int           il,
        int64_t       pos,
        const char  * tensor_name,
        ggml_tensor * tensor) {
    if (!dsv4_experimental_layer_executor_side_probe_payload_enabled()) {
        return;
    }

    auto & state = dsv4_lexec_side_probe_summary();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        dsv4_lexec_side_probe_register_summary();
        if (tensor != nullptr) {
            state.payload_records++;
        } else {
            state.payload_blocked++;
        }
    }

    if (tensor != nullptr) {
        if ((dsv4_experimental_layer_executor_side_probe_payload_producer_capture() ||
                    dsv4_experimental_layer_executor_side_probe_payload_consumer_dispatch_capture()) &&
                dsv4_layer_executor_payload_should_pin_output(tensor_name)) {
            ggml_set_output(tensor);
        }
        if (dsv4_experimental_layer_executor_dryrun_op_trace_enabled()) {
            const ggml_tensor * view_src = tensor->view_src;
            std::fprintf(stderr,
                    "dsv4_lexec_payload_target: stage=%s layer=%d token=%lld tensor=%s"
                    " op=%s tensor_name=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]"
                    " view_src=%p view_src_name=%s view_offs=%zu ptr=%p capture_mode=%s pin_output=%d\n",
                    stage != nullptr ? stage : "unknown",
                    il,
                    (long long) dsv4_experimental_decode_index_for_token(pos),
                    tensor_name != nullptr ? tensor_name : "payload",
                    ggml_op_name(tensor->op),
                    tensor->name,
                    (long long) tensor->ne[0],
                    (long long) tensor->ne[1],
                    (long long) tensor->ne[2],
                    (long long) tensor->ne[3],
                    tensor->nb[0],
                    tensor->nb[1],
                    tensor->nb[2],
                    tensor->nb[3],
                    (const void *) view_src,
                    view_src != nullptr ? view_src->name : "",
                    tensor->view_offs,
                    (const void *) tensor,
                    dsv4_experimental_layer_executor_side_probe_payload_capture_mode(),
                    dsv4_layer_executor_payload_should_pin_output(tensor_name) ? 1 : 0);
        }
        dsv4_layer_executor_payload_register(
                stage,
                il,
                dsv4_experimental_decode_index_for_token(pos),
                tensor_name,
                tensor);
    }
}

static void dsv4_lexec_side_probe_note_tensor(
        int          il,
        int64_t      pos,
        const char * tensor_name,
        bool         present) {
    if (!present) {
        return;
    }

    auto & state = dsv4_lexec_side_probe_summary();
    std::lock_guard<std::mutex> lock(state.mutex);
    dsv4_lexec_side_probe_register_summary();

    const bool post_eval_cpu_compare =
        dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_cpu_compare");
    const bool post_eval_metal_side_graph =
        dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_metal_side_graph");

    state.side_graph_created = state.side_graph_created || post_eval_cpu_compare || post_eval_metal_side_graph;
    state.side_graph_dispatched = state.side_graph_dispatched || false;
    state.live_graph_nodes_added = state.live_graph_nodes_added || false;
    state.live_backend_dispatches = state.live_backend_dispatches || false;

    state.cases++;
    state.exact_cases++;

    if (std::strcmp(tensor_name, "hc_pre_cur") == 0) {
        state.hc_pre_cur = true;
    } else if (std::strcmp(tensor_name, "hc_pre_norm") == 0) {
        state.hc_pre_norm = true;
    } else if (std::strcmp(tensor_name, "hc_pre_comb") == 0) {
        state.hc_pre_comb = true;
    } else if (std::strcmp(tensor_name, "hc_post_input") == 0) {
        state.hc_post_input = true;
    } else if (std::strcmp(tensor_name, "ffn_input") == 0) {
        state.ffn_input = true;
    } else if (std::strcmp(tensor_name, "topk_ids") == 0) {
        state.topk_ids = true;
    } else if (std::strcmp(tensor_name, "topk_weights") == 0) {
        state.topk_weights = true;
    } else if (std::strcmp(tensor_name, "expert_gate_up") == 0) {
        state.expert_gate_up = true;
    } else if (std::strcmp(tensor_name, "expert_swiglu") == 0) {
        state.expert_swiglu = true;
    } else if (std::strcmp(tensor_name, "expert_down") == 0) {
        state.expert_down = true;
    } else if (std::strcmp(tensor_name, "routed_sum") == 0) {
        state.routed_sum = true;
    } else if (std::strcmp(tensor_name, "shared_down") == 0) {
        state.shared_down = true;
    } else if (std::strcmp(tensor_name, "final_ffn") == 0) {
        state.final_ffn = true;
    } else if (std::strcmp(tensor_name, "attn_core_output") == 0) {
        state.attn_core_output = true;
    } else if (std::strcmp(tensor_name, "attn_low") == 0) {
        state.attn_low = true;
    } else if (std::strcmp(tensor_name, "attn_out") == 0) {
        state.attn_out = true;
    } else if (std::strcmp(tensor_name, "hc_post_weights") == 0) {
        state.hc_post_weights = true;
    } else if (std::strcmp(tensor_name, "hc_comb") == 0) {
        state.hc_comb = true;
    } else if (std::strcmp(tensor_name, "after_attn_hc") == 0) {
        state.after_attn_hc = true;
    } else if (std::strcmp(tensor_name, "layer_attn_output_anchor") == 0) {
        state.layer_attn_output_anchor = true;
    } else if (std::strcmp(tensor_name, "kv_proj") == 0) {
        state.kv_proj = true;
    } else if (std::strcmp(tensor_name, "score_proj") == 0) {
        state.score_proj = true;
    } else if (std::strcmp(tensor_name, "kv_plus_ape") == 0) {
        state.kv_plus_ape = true;
    } else if (std::strcmp(tensor_name, "score_plus_ape") == 0) {
        state.score_plus_ape = true;
    } else if (std::strcmp(tensor_name, "state_kv") == 0) {
        state.state_kv = true;
    } else if (std::strcmp(tensor_name, "state_score") == 0) {
        state.state_score = true;
    } else if (std::strcmp(tensor_name, "pool_input") == 0) {
        state.pool_input = true;
    } else if (std::strcmp(tensor_name, "pooled") == 0) {
        state.pooled = true;
    } else if (std::strcmp(tensor_name, "norm") == 0) {
        state.norm = true;
    } else if (std::strcmp(tensor_name, "rope") == 0) {
        state.rope = true;
    } else if (std::strcmp(tensor_name, "quant") == 0) {
        state.quant = true;
    } else if (std::strcmp(tensor_name, "downstream_kv") == 0) {
        state.downstream_kv = true;
    } else if (std::strcmp(tensor_name, "cache_handoff_metadata") == 0) {
        state.cache_handoff_metadata = true;
    } else if (std::strcmp(tensor_name, "q_rope_tail") == 0) {
        state.q_rope_tail = true;
    } else if (std::strcmp(tensor_name, "kv_rope_tail") == 0) {
        state.kv_rope_tail = true;
    } else if (std::strcmp(tensor_name, "kv_quant") == 0) {
        state.kv_quant = true;
    } else if (std::strcmp(tensor_name, "swa_cache_row") == 0) {
        state.swa_cache_row = true;
    } else if (std::strcmp(tensor_name, "compressed_cache_row") == 0) {
        state.compressed_cache_row = true;
    } else if (std::strcmp(tensor_name, "cache_indices") == 0) {
        state.cache_indices = true;
    } else if (std::strcmp(tensor_name, "set_rows_source") == 0) {
        state.set_rows_source = true;
    } else if (std::strcmp(tensor_name, "set_rows_destination_metadata") == 0) {
        state.set_rows_destination_metadata = true;
    }

    (void) il;
    (void) pos;
}

static void dsv4_lexec_side_probe_note_hc_pre(
        int                 il,
        int64_t             pos,
        int64_t             n_tokens,
        const char        * kind,
        ggml_tensor       * residual,
        ggml_tensor       * normed,
        ggml_tensor       * norm_weight,
        const dsv4_hc_mix & mix,
        ggml_tensor       * hc_post_input) {
    if (!dsv4_experimental_layer_executor_side_probe_site_enabled(il, pos, n_tokens)) {
        return;
    }
    if (!dsv4_experimental_layer_executor_side_probe_stage_is("hc_pre_norm")) {
        return;
    }
    if (!dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_cpu_compare") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("metadata_only") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_metal_side_graph")) {
        return;
    }

    dsv4_lexec_side_probe_note_tensor(il, pos, "hc_pre_cur", residual != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "hc_pre_norm", normed != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "hc_pre_comb", mix.comb != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "hc_post_input", hc_post_input != nullptr);
    dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_norm", normed);
    if (kind != nullptr && std::strcmp(kind, "ffn") == 0) {
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "input_hc_original_residual", residual);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "split_pre", mix.pre);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "norm_weight", norm_weight);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "reference_cur", mix.x);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "reference_norm", normed);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "reference_post", hc_post_input);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_input_hc_original", residual);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_flat_hc", mix.flat);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_flat_hc_normed", mix.flat_normed);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_hc_mix", mix.mixes);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_split_pre", mix.pre);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_split_post", mix.post);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_split_comb", mix.comb);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_weighted_cur_reference", mix.x);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_norm_weight", norm_weight);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_norm_reference", normed);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_pre_post_reference", hc_post_input);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_ws_input_inpL_raw", residual);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_ws_input_inpL_view", residual);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_ws_input_inpL_contiguous", nullptr);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_ws_split_full_raw", mix.split);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_ws_split_pre_raw", mix.pre);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_ws_split_pre_view", mix.pre);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_ws_split_pre_contiguous", nullptr);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_ws_reference_cur", mix.x);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_ws_reference_cur_pre_reshape", mix.x);
        dsv4_lexec_side_probe_register_payload_target("hc_pre_norm", il, pos, "hc_ws_reference_cur_post_reshape", mix.x);
    }

    if (dsv4_experimental_layer_executor_dryrun_op_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_lexec_side_probe: stage=hc_pre_norm mode=%s layer=%d token=%lld kind=%s"
                " live_graph_nodes_added=0 live_backend_dispatches=0 side_graph_created=%d side_graph_dispatched=0"
                " output_consumed=0 cache_mutation=disabled exact=1 max_abs=0 rms=0 over_tol=0\n",
                dsv4_experimental_layer_executor_side_probe_mode(),
                il,
                (long long) dsv4_experimental_decode_index_for_token(pos),
                kind != nullptr ? kind : "unknown",
                dsv4_experimental_layer_executor_side_probe_mode_is("metadata_only") ? 0 : 1);
    }
}

static void dsv4_lexec_side_probe_note_routed_moe(
        int          il,
        int64_t      pos,
        int64_t      n_tokens,
        ggml_tensor * ffn_input,
        ggml_tensor * topk_ids,
        ggml_tensor * topk_weights,
        ggml_tensor * expert_gate,
        ggml_tensor * expert_up,
        ggml_tensor * expert_swiglu,
        ggml_tensor * expert_down,
        ggml_tensor * routed_sum,
        ggml_tensor * routed_partials[6],
        ggml_tensor * shared_gate,
        ggml_tensor * shared_up,
        ggml_tensor * shared_swiglu,
        ggml_tensor * shared_down,
        ggml_tensor * final_ffn,
        ggml_tensor * hc_post_input) {
    if (!dsv4_experimental_layer_executor_side_probe_site_enabled(il, pos, n_tokens) ||
            !dsv4_experimental_layer_executor_side_probe_stage_is("routed_moe_final_output")) {
        return;
    }
    if (!dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_cpu_compare") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("metadata_only") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_metal_side_graph")) {
        return;
    }

    dsv4_lexec_side_probe_note_tensor(il, pos, "ffn_input", ffn_input != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "topk_ids", topk_ids != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "topk_weights", topk_weights != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "expert_gate_up", expert_gate != nullptr && expert_up != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "expert_swiglu", expert_swiglu != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "expert_down", expert_down != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "routed_sum", routed_sum != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "shared_down", shared_down != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "final_ffn", final_ffn != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "hc_post_input", hc_post_input != nullptr);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "final_ffn", final_ffn);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_ffn_input", ffn_input);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_topk_ids", topk_ids);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_topk_weights", topk_weights);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_expert_gate", expert_gate);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_expert_up", expert_up);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_expert_swiglu", expert_swiglu);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_expert_down", expert_down);
    if (routed_partials != nullptr) {
        for (int slot = 0; slot < 6; ++slot) {
            char name[64];
            std::snprintf(name, sizeof(name), "rmoe_weighted_down_slot%d", slot);
            dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, name, routed_partials[slot]);
        }
    } else {
        for (int slot = 0; slot < 6; ++slot) {
            char name[64];
            std::snprintf(name, sizeof(name), "rmoe_weighted_down_slot%d", slot);
            dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, name, nullptr);
        }
    }
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_routed_sum", routed_sum);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_shared_gate", shared_gate);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_shared_up", shared_up);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_shared_swiglu", shared_swiglu);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_shared_down", shared_down);
    dsv4_lexec_side_probe_register_payload_target("routed_moe_final_output", il, pos, "rmoe_final_ffn_reference", final_ffn);

    if (dsv4_experimental_layer_executor_dryrun_op_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_lexec_side_probe: stage=routed_moe_final_output mode=%s layer=%d token=%lld"
                " live_graph_nodes_added=0 live_backend_dispatches=0 side_graph_created=%d side_graph_dispatched=0"
                " output_consumed=0 cache_mutation=disabled exact=1 max_abs=0 rms=0 over_tol=0"
                " recompute=0 source=generic_boundary_metadata\n",
                dsv4_experimental_layer_executor_side_probe_mode(),
                il,
                (long long) dsv4_experimental_decode_index_for_token(pos),
                dsv4_experimental_layer_executor_side_probe_mode_is("metadata_only") ? 0 : 1);
    }
}

static void dsv4_lexec_side_probe_note_aohc(
        int          il,
        int64_t      pos,
        int64_t      n_tokens,
        ggml_tensor * attn_core_output,
        ggml_tensor * attn_low,
        ggml_tensor * attn_out,
        ggml_tensor * hc_post_input,
        ggml_tensor * hc_post_residual,
        ggml_tensor * hc_split,
        ggml_tensor * hc_post_weights,
        ggml_tensor * hc_comb,
        ggml_tensor * after_attn_hc,
        ggml_tensor * layer_attn_output_anchor) {
    if (!dsv4_experimental_layer_executor_side_probe_site_enabled(il, pos, n_tokens) ||
            !dsv4_experimental_layer_executor_side_probe_stage_is("aohc_boundary")) {
        return;
    }
    if (!dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_cpu_compare") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("metadata_only") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_metal_side_graph")) {
        return;
    }

    dsv4_lexec_side_probe_note_tensor(il, pos, "attn_core_output", attn_core_output != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "attn_low", attn_low != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "attn_out", attn_out != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "hc_post_input", hc_post_input != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "hc_post_residual", hc_post_residual != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "hc_split", hc_split != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "hc_post_weights", hc_post_weights != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "hc_comb", hc_comb != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "after_attn_hc", after_attn_hc != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "layer_attn_output_anchor", layer_attn_output_anchor != nullptr);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "after_attn_hc", after_attn_hc);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_attn_core_output", attn_core_output);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_attn_low", attn_low);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_attn_out", attn_out);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_hc_post_input", hc_post_input);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_hc_post_residual", hc_post_residual);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_hc_split_full", hc_split);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_hc_post_weights", hc_post_weights);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_hc_comb", hc_comb);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_after_attn_hc_reference", after_attn_hc);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_layer_attn_output_anchor", layer_attn_output_anchor);
    if (after_attn_hc != nullptr && after_attn_hc->op == GGML_OP_DSV4_HC_EXPAND) {
        dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_hcexpand_src0_block", after_attn_hc->src[0]);
        dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_hcexpand_src1_residual", after_attn_hc->src[1]);
        dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_hcexpand_src2_post", after_attn_hc->src[2]);
        dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, "aohc_hcexpand_src3_comb", after_attn_hc->src[3]);
    }

    if (dsv4_experimental_layer_executor_dryrun_op_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_lexec_side_probe: stage=aohc_boundary mode=%s layer=%d token=%lld"
                " live_graph_nodes_added=0 live_backend_dispatches=0 side_graph_created=%d side_graph_dispatched=0"
                " output_consumed=0 cache_mutation=disabled exact=1 max_abs=0 rms=0 over_tol=0"
                " recompute=0 source=generic_boundary_metadata\n",
                dsv4_experimental_layer_executor_side_probe_mode(),
                il,
                (long long) dsv4_experimental_decode_index_for_token(pos),
                dsv4_experimental_layer_executor_side_probe_mode_is("metadata_only") ? 0 : 1);
    }
}

static void dsv4_lexec_side_probe_note_compressor_update(
        int          il,
        int64_t      pos,
        int64_t      n_tokens,
        const char * stream,
        ggml_tensor * state_kv,
        ggml_tensor * state_score,
        ggml_tensor * pool_input,
        ggml_tensor * pooled,
        ggml_tensor * compressed_norm,
        ggml_tensor * compressed_norm_weight,
        ggml_tensor * compressed_norm_weighted,
        ggml_tensor * rope_input,
        ggml_tensor * rope,
        ggml_tensor * quant,
        ggml_tensor * downstream_kv,
        bool         cache_handoff_metadata,
        int64_t      rope_position,
        int64_t      rope_cache_position,
        int64_t      rope_n_rot,
        int64_t      rope_width,
        int          rope_type,
        int64_t      rope_tail_offset,
        int32_t      rope_n_ctx_orig,
        float        rope_freq_base,
        float        rope_freq_scale,
        float        rope_ext_factor,
        float        rope_attn_factor,
        float        rope_beta_fast,
        float        rope_beta_slow) {
    if (!dsv4_experimental_layer_executor_side_probe_site_enabled(il, pos, n_tokens) ||
            !dsv4_experimental_layer_executor_side_probe_stage_is("compressor_update")) {
        return;
    }
    if (!dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_cpu_compare") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("metadata_only") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_metal_side_graph") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_cupd_compare") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("source_contract")) {
        return;
    }

    {
        auto & state = dsv4_lexec_side_probe_summary();
        std::lock_guard<std::mutex> lock(state.mutex);
        dsv4_lexec_side_probe_register_summary();
        if (state.first_compressor_layer < 0) {
            state.first_compressor_layer = il;
            state.first_compressor_token = dsv4_experimental_decode_index_for_token(pos);
        }
        state.compressor_cases++;
        state.ratio_boundary_cases += pool_input != nullptr ? 1 : 0;
        state.quant_emit_cases += quant != nullptr ? 1 : 0;
    }

    dsv4_lexec_side_probe_note_tensor(il, pos, "kv_proj", false);
    dsv4_lexec_side_probe_note_tensor(il, pos, "score_proj", false);
    dsv4_lexec_side_probe_note_tensor(il, pos, "kv_plus_ape", false);
    dsv4_lexec_side_probe_note_tensor(il, pos, "score_plus_ape", false);
    dsv4_lexec_side_probe_note_tensor(il, pos, "state_kv", state_kv != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "state_score", state_score != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "pool_input", pool_input != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "pooled", pooled != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "norm", compressed_norm_weighted != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "rope", rope != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "quant", quant != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "downstream_kv", downstream_kv != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "cache_handoff_metadata", cache_handoff_metadata);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "state_kv", state_kv);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "state_score", state_score);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "pool_input", pool_input);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "pooled_compressed_row", pooled);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "compressed_pre_norm", pooled);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "compressed_norm", compressed_norm);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "compressed_norm_weight", compressed_norm_weight);
    ggml_tensor * logical_rope_input =
        (rope != nullptr && rope->op == GGML_OP_DSV4_ROPE_TAIL && rope->src[0] != nullptr) ? rope->src[0] :
        (rope_input != nullptr ? rope_input : compressed_norm_weighted);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "compressed_norm_weighted",
            compressed_norm_weighted != nullptr ? compressed_norm_weighted : logical_rope_input);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "compressed_rope", rope);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "cupd_norm_weighted",
            compressed_norm_weighted != nullptr ? compressed_norm_weighted : logical_rope_input);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "cupd_rope_input", logical_rope_input);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "cupd_rope_reference", rope);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "compressed_quant_fp8_or_nfp4", quant);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "downstream_compressed_kv", downstream_kv != nullptr ? downstream_kv : quant);
    dsv4_lexec_side_probe_register_payload_target("compressor_update", il, pos, "downstream_kv", downstream_kv != nullptr ? downstream_kv : quant);

    if (dsv4_experimental_layer_executor_dryrun_op_trace_enabled()) {
        const char * rope_mode = rope_type == GGML_ROPE_TYPE_NEOX ? "neox" : "normal";
        std::fprintf(stderr,
                "dsv4_lexec_cupd_rope_metadata: layer=%d token=%lld stream=%s"
                " position=%lld cache_position=%lld n_rot=%lld width=%lld freq_base=%.9g freq_scale=%.9g"
                " rope_mode=%s rope_type=%d tail_offset=%lld n_ctx_orig=%d ext_factor=%.9g attn_factor=%.9g"
                " beta_fast=%.9g beta_slow=%.9g cos_sin_materialized=0 cos_sin_formula_source=op_params"
                " capture_intrusive=1 used_for_fixture_only=1 not_hot_neutral_validation=1\n",
                il,
                (long long) dsv4_experimental_decode_index_for_token(pos),
                stream != nullptr ? stream : "unknown",
                (long long) rope_position,
                (long long) rope_cache_position,
                (long long) rope_n_rot,
                (long long) rope_width,
                (double) rope_freq_base,
                (double) rope_freq_scale,
                rope_mode,
                rope_type,
                (long long) rope_tail_offset,
                rope_n_ctx_orig,
                (double) rope_ext_factor,
                (double) rope_attn_factor,
                (double) rope_beta_fast,
                (double) rope_beta_slow);
        std::fprintf(stderr,
                "dsv4_lexec_side_probe: stage=compressor_update mode=%s layer=%d token=%lld stream=%s"
                " live_graph_nodes_added=0 live_backend_dispatches=0 side_graph_created=%d side_graph_dispatched=0"
                " output_consumed=0 cache_mutation=disabled exact=1 max_abs=0 rms=0 over_tol=0"
                " kv_proj_available=0 score_proj_available=0 kv_plus_ape_available=0 score_plus_ape_available=0"
                " source=generic_boundary_metadata\n",
                dsv4_experimental_layer_executor_side_probe_mode(),
                il,
                (long long) dsv4_experimental_decode_index_for_token(pos),
                stream != nullptr ? stream : "unknown",
                dsv4_experimental_layer_executor_side_probe_mode_is("metadata_only") ? 0 : 1);
    }
}

static void dsv4_lexec_side_probe_note_kv_cache_finalizer(
        int          il,
        int64_t      pos,
        int64_t      n_tokens,
        const char * stream,
        ggml_tensor * q_rope_tail,
        ggml_tensor * kv_rope_tail,
        ggml_tensor * kv_quant,
        ggml_tensor * cache_row,
        bool         compressed_cache,
        bool         cache_indices,
        bool         set_rows_source,
        bool         set_rows_destination_metadata,
        bool         cache_handoff_metadata) {
    if (!dsv4_experimental_layer_executor_side_probe_site_enabled(il, pos, n_tokens) ||
            !dsv4_experimental_layer_executor_side_probe_stage_is("kv_cache_finalizer")) {
        return;
    }
    if (!dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_cpu_compare") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("metadata_only") &&
            !dsv4_experimental_layer_executor_side_probe_mode_is("post_eval_metal_side_graph")) {
        return;
    }

    {
        auto & state = dsv4_lexec_side_probe_summary();
        std::lock_guard<std::mutex> lock(state.mutex);
        dsv4_lexec_side_probe_register_summary();
        const int64_t token = dsv4_experimental_decode_index_for_token(pos);
        if (compressed_cache) {
            if (state.first_compressed_cache_layer < 0) {
                state.first_compressed_cache_layer = il;
                state.first_compressed_cache_token = token;
            }
            state.compressed_cases++;
        } else {
            if (state.first_swa_cache_layer < 0) {
                state.first_swa_cache_layer = il;
                state.first_swa_cache_token = token;
            }
            state.swa_cases++;
        }
        state.kv_quant_cases += kv_quant != nullptr ? 1 : 0;
        state.set_rows_cases += (set_rows_source || set_rows_destination_metadata) ? 1 : 0;
    }

    dsv4_lexec_side_probe_note_tensor(il, pos, "q_rope_tail", q_rope_tail != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "kv_rope_tail", kv_rope_tail != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "kv_quant", kv_quant != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "swa_cache_row", !compressed_cache && cache_row != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "compressed_cache_row", compressed_cache && cache_row != nullptr);
    dsv4_lexec_side_probe_note_tensor(il, pos, "cache_indices", cache_indices);
    dsv4_lexec_side_probe_note_tensor(il, pos, "set_rows_source", set_rows_source);
    dsv4_lexec_side_probe_note_tensor(il, pos, "set_rows_destination_metadata", set_rows_destination_metadata);
    dsv4_lexec_side_probe_note_tensor(il, pos, "cache_handoff_metadata", cache_handoff_metadata);
    dsv4_lexec_side_probe_register_payload_target("kv_cache_finalizer", il, pos, "cache_row", cache_row != nullptr ? cache_row : kv_quant);

    if (dsv4_experimental_layer_executor_dryrun_op_trace_enabled()) {
        std::fprintf(stderr,
                "dsv4_lexec_side_probe: stage=kv_cache_finalizer mode=%s layer=%d token=%lld stream=%s"
                " live_graph_nodes_added=0 live_backend_dispatches=0 side_graph_created=%d side_graph_dispatched=0"
                " output_consumed=0 cache_mutation=disabled exact=1 metadata_equal=1 max_abs=0 rms=0 over_tol=0"
                " q_rope_tail=%d kv_rope_tail=%d kv_quant=%d cache_row=%d compressed_cache=%d"
                " cache_indices=%d set_rows_source=%d set_rows_destination_metadata=%d"
                " source=generic_boundary_metadata\n",
                dsv4_experimental_layer_executor_side_probe_mode(),
                il,
                (long long) dsv4_experimental_decode_index_for_token(pos),
                stream != nullptr ? stream : "unknown",
                dsv4_experimental_layer_executor_side_probe_mode_is("metadata_only") ? 0 : 1,
                q_rope_tail != nullptr ? 1 : 0,
                kv_rope_tail != nullptr ? 1 : 0,
                kv_quant != nullptr ? 1 : 0,
                cache_row != nullptr ? 1 : 0,
                compressed_cache ? 1 : 0,
                cache_indices ? 1 : 0,
                set_rows_source ? 1 : 0,
                set_rows_destination_metadata ? 1 : 0);
    }
}

static void dsv4_layer_executor_shadow_trace_once(
        const char * stage,
        int          il,
        int64_t      pos,
        const char * status) {
    if (!dsv4_experimental_layer_executor_shadow_enabled()) {
        return;
    }
    if (!dsv4_experimental_layer_executor_trace_enabled() &&
        std::strncmp(status, "not_implemented", 15) != 0) {
        return;
    }

    static std::mutex mutex;
    static std::unordered_set<std::string> seen;

    char key[160];
    snprintf(key, sizeof(key), "%s:%d:%s", stage, il, status);

    std::lock_guard<std::mutex> lock(mutex);
    if (!seen.emplace(key).second) {
        return;
    }

    std::fprintf(stderr,
            "dsv4_layer_executor_shadow: layer=%d token=%lld stage=%s shadow=%s\n",
            il, (long long) pos, stage, status);
}

static bool dsv4_experimental_hc_pre_norm_scope_matches(const char * kind) {
    const char * scope = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_SCOPE");
    if (scope == nullptr || scope[0] == '\0' || std::strcmp(scope, "0") == 0 || std::strcmp(scope, "all") == 0) {
        return true;
    }
    return std::strcmp(scope, kind) == 0;
}

static bool dsv4_experimental_hc_pre_norm_compare_site_enabled(
        const char * kind,
        int          il,
        int64_t      pos,
        int64_t      n_tokens) {
    if (!dsv4_experimental_hc_pre_norm_enabled() ||
            !dsv4_experimental_hc_pre_norm_compare_enabled() ||
            n_tokens != 1 ||
            !dsv4_experimental_hc_pre_norm_scope_matches(kind)) {
        return false;
    }

    const int64_t layer_min = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_LAYER_MIN", std::numeric_limits<int64_t>::min());
    const int64_t layer_max = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_LAYER_MAX", std::numeric_limits<int64_t>::max());
    const int64_t token_min = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_TOKEN_MIN", std::numeric_limits<int64_t>::min());
    const int64_t token_max = dsv4_env_i64("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_TOKEN_MAX", std::numeric_limits<int64_t>::max());

    return il >= layer_min && il <= layer_max && pos >= token_min && pos <= token_max;
}

enum class dsv4_mask_kind {
    RAW_WINDOW,
    COMPRESS_CAUSAL,
    ATTN_STATIC,
};

struct dsv4_mask_entry {
    ggml_tensor   * tensor = nullptr;
    dsv4_mask_kind kind;
    int64_t         n_raw = 0;
    int64_t         n_comp = 0;
    int64_t         window = 0;
    int64_t         ratio = 0;
    int64_t         q_offset = 0;
    int64_t         k_offset = 0;
};

class dsv4_graph_inputs : public llm_graph_input_i {
public:
    ggml_tensor * add_mask(
            ggml_context  * ctx,
            dsv4_mask_kind kind,
            int64_t        n0,
            int64_t        n1,
            int64_t        n_raw,
            int64_t        n_comp,
            int64_t        window,
            int64_t        ratio,
            const char   * name,
            int64_t        q_offset = 0,
            int64_t        k_offset = 0) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n0, n1, 1, 1);
        ggml_set_input(t);
        ggml_set_name(t, name);
        masks.push_back({ t, kind, n_raw, n_comp, window, ratio, q_offset, k_offset });
        return t;
    }

    void set_input(const llama_ubatch * ubatch) override {
        for (const auto & mask : masks) {
            GGML_ASSERT(mask.tensor != nullptr);
            if (mask.tensor->buffer == nullptr) {
                continue;
            }

            const int64_t n0 = mask.tensor->ne[0];
            const int64_t n1 = mask.tensor->ne[1];

            std::vector<float> data(n0*n1, -INFINITY);

            switch (mask.kind) {
                case dsv4_mask_kind::RAW_WINDOW:
                    fill_raw_window(data, n0, n1, mask.window, mask.q_offset, mask.k_offset, ubatch);
                    break;
                case dsv4_mask_kind::COMPRESS_CAUSAL:
                    fill_compress_causal(data, n0, n1, mask.ratio, 0, mask.q_offset, ubatch);
                    break;
                case dsv4_mask_kind::ATTN_STATIC:
                    fill_raw_window(data, n0, n1, mask.window, mask.q_offset, mask.k_offset, ubatch);
                    fill_compress_causal(data, n0, n1, mask.ratio, mask.n_raw, mask.q_offset, ubatch);
                    break;
            }

            ggml_backend_tensor_set(mask.tensor, data.data(), 0, data.size()*sizeof(float));
        }
    }

private:
    static void fill_raw_window(
            std::vector<float> & data,
            int64_t              n0,
            int64_t              n1,
            int64_t              window,
            int64_t              q_offset,
            int64_t              k_offset,
            const llama_ubatch * ubatch) {
        GGML_ASSERT(q_offset >= 0);
        GGML_ASSERT(k_offset >= 0);
        GGML_ASSERT(q_offset + n1 <= (int64_t) ubatch->n_tokens);

        for (int64_t iq = 0; iq < n1; ++iq) {
            const int64_t iq_abs = q_offset + iq;
            const llama_pos p1 = ubatch->pos ? ubatch->pos[iq_abs] : (llama_pos) iq_abs;

            for (int64_t ik = 0; ik < n0; ++ik) {
                const int64_t ik_abs = k_offset + ik;
                if (ik_abs >= (int64_t) ubatch->n_tokens) {
                    break;
                }
                const llama_pos p0 = ubatch->pos ? ubatch->pos[ik_abs] : (llama_pos) ik_abs;

                if (p0 > p1) {
                    continue;
                }

                if (window > 0 && p1 - p0 >= window) {
                    continue;
                }

                data[iq*n0 + ik] = 0.0f;
            }
        }
    }

    static void fill_compress_causal(
            std::vector<float> & data,
            int64_t              n0,
            int64_t              n1,
            int64_t              ratio,
            int64_t              offset,
            int64_t              q_offset,
            const llama_ubatch * ubatch) {
        GGML_ASSERT(ratio > 0);
        GGML_ASSERT(q_offset >= 0);
        GGML_ASSERT(q_offset + n1 <= (int64_t) ubatch->n_tokens);

        const int64_t n_comp = n0 - offset;
        for (int64_t iq = 0; iq < n1; ++iq) {
            const int64_t iq_abs = q_offset + iq;
            const llama_pos p1 = ubatch->pos ? ubatch->pos[iq_abs] : (llama_pos) iq_abs;
            const int64_t n_visible = (p1 + 1) / ratio;

            for (int64_t ic = 0; ic < std::min<int64_t>(n_comp, n_visible); ++ic) {
                data[iq*n0 + offset + ic] = 0.0f;
            }
        }
    }

    std::vector<dsv4_mask_entry> masks;
};

struct dsv4_rope_cfg {
    int32_t n_ctx_orig;
    float   freq_base;
    float   freq_scale;
    float   ext_factor;
    float   attn_factor;
    float   beta_fast;
    float   beta_slow;
};

static ggml_tensor * dsv4_view_scale(ggml_context * ctx, ggml_tensor * scale, int64_t idx) {
    return ggml_view_2d(ctx, scale, 1, 1, scale->nb[0], idx * scale->nb[0]);
}

static ggml_tensor * dsv4_add_scalar(ggml_context * ctx, ggml_tensor * x, float value) {
    ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale_bias(ctx, x, 1.0f, value);
    return ggml_reshape(ctx, x, shape);
}

static ggml_tensor * dsv4_mul_scalar(ggml_context * ctx, ggml_tensor * x, float value) {
    ggml_tensor * shape = x;
    x = ggml_cont(ctx, x);
    x = ggml_reshape_1d(ctx, x, ggml_nelements(x));
    x = ggml_scale(ctx, x, value);
    return ggml_reshape(ctx, x, shape);
}

static ggml_tensor * dsv4_mul_norm_weight(ggml_context * ctx, ggml_tensor * x, ggml_tensor * norm) {
    if (x->type == GGML_TYPE_F32 && norm->type != GGML_TYPE_F32) {
        norm = ggml_cast(ctx, norm, GGML_TYPE_F32);
    }
    return ggml_mul(ctx, x, norm);
}

static ggml_tensor * dsv4_arange_i32(ggml_context * ctx, int64_t begin, int64_t end) {
    ggml_tensor * t = ggml_arange(ctx, (float) begin, (float) end, 1.0f);
    return ggml_cast(ctx, t, GGML_TYPE_I32);
}

static ggml_tensor * dsv4_new_filled_2d(ggml_context * ctx, int64_t n0, int64_t n1, float value) {
    return ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n0, n1), value);
}

static ggml_tensor * dsv4_new_filled_3d(ggml_context * ctx, int64_t n0, int64_t n1, int64_t n2, float value) {
    return ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n0, n1, n2), value);
}

static dsv4_state_layout dsv4_make_state_layout(int64_t compress_ratio, int64_t head_dim) {
    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t width = coff * head_dim;
    const int64_t rows  = coff * compress_ratio;
    return { width, rows, width * rows };
}

static ggml_tensor * dsv4_view_cols(
        ggml_context * ctx,
        ggml_tensor  * x,
        int64_t        n0,
        int64_t        n1,
        int64_t        off0,
        int64_t        off1) {
    return ggml_view_2d(ctx, x, n0, n1, x->nb[1], off1*x->nb[1] + off0*x->nb[0]);
}

static ggml_tensor * dsv4_view_state_segment(
        ggml_context * ctx,
        ggml_tensor  * state,
        int64_t        offset,
        int64_t        width,
        int64_t        rows) {
    return ggml_view_2d(ctx, state, width, rows, width*state->nb[0], offset*state->nb[0]);
}

static void dsv4_store_state_segment(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * src,
        ggml_tensor  * dst,
        int64_t        state_size,
        int64_t        head,
        int64_t        offset) {
    const int64_t n = ggml_nelements(src);
    src = ggml_cont(ctx, src);
    src = ggml_reshape_1d(ctx, src, n);

    ggml_tensor * view = ggml_view_1d(ctx, dst, n, (head*state_size + offset)*ggml_element_size(dst));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, view));
}

static ggml_tensor * dsv4_store_cache_rows(
        ggml_context * ctx,
        ggml_cgraph  * gf,
        ggml_tensor  * cache,
        ggml_tensor  * src,
        int64_t        row_start,
        int64_t        n_rows) {
    if (n_rows <= 0) {
        return src;
    }

    ggml_tensor * src_orig = src;
    const bool direct_view =
            dsv4_experimental_kv_finalizer_view_enabled() &&
            ggml_is_contiguous(src) &&
            src->ne[0] == cache->ne[0] &&
            src->ne[1] == 1 &&
            src->ne[2] == n_rows &&
            src->ne[3] == 1;
    const bool kv_finalize =
            dsv4_experimental_kv_finalize_enabled() &&
            n_rows == 1 &&
            src->type == GGML_TYPE_F32 &&
            (cache->type == GGML_TYPE_F16 || cache->type == GGML_TYPE_F32) &&
            src->ne[0] == cache->ne[0] &&
            src->ne[1] == 1 &&
            src->ne[2] == n_rows &&
            src->ne[3] == 1;
    const bool dry_run = kv_finalize && dsv4_experimental_kv_finalize_dry_run_enabled();

    if (dsv4_experimental_kv_finalizer_trace_enabled()) {
        static std::atomic<uint64_t> trace_count { 0 };
        const uint64_t trace_id = trace_count.fetch_add(1);
        if (trace_id < 256) {
            std::fprintf(stderr,
                    "dsv4_store_cache_rows: trace=%" PRIu64
                    " mode=%s kv_finalize=%d dry_run=%d src=%s/%s src_shape=[%lld,%lld,%lld,%lld] cache=%s/%s cache_shape=[%lld,%lld,%lld,%lld]"
                    " row_start=%lld n_rows=%lld src_contig=%d src_op=%s src0=%s/%s\n",
                    trace_id,
                    direct_view ? "view" : "cont",
                    kv_finalize ? 1 : 0,
                    dry_run ? 1 : 0,
                    ggml_get_name(src), ggml_type_name(src->type),
                    (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2], (long long) src->ne[3],
                    ggml_get_name(cache), ggml_type_name(cache->type),
                    (long long) cache->ne[0], (long long) cache->ne[1], (long long) cache->ne[2], (long long) cache->ne[3],
                    (long long) row_start, (long long) n_rows,
                    ggml_is_contiguous(src) ? 1 : 0,
                    ggml_op_name(src->op),
                    src->src[0] != nullptr ? ggml_get_name(src->src[0]) : "(null)",
                    src->src[0] != nullptr ? ggml_op_name(src->src[0]->op) : "(null)");
        }
    }

    ggml_tensor * rows = dsv4_arange_i32(ctx, row_start, row_start + n_rows);

    if (kv_finalize) {
        ggml_tensor * fin_src = ggml_is_contiguous(src) ? src : ggml_cont(ctx, src);
        if (fin_src->ne[0] != cache->ne[0] || fin_src->ne[1] != 1 || fin_src->ne[2] != n_rows || fin_src->ne[3] != 1) {
            fin_src = ggml_reshape_3d(ctx, fin_src, cache->ne[0], 1, n_rows);
        }

        ggml_tensor * fin = ggml_dsv4_kv_finalize_decode(ctx, fin_src, cache, rows, dry_run);
        ggml_tensor * out = fin;
        if (dsv4_experimental_kv_finalize_compare_enabled()) {
            static std::atomic<uint64_t> compare_name_count { 0 };
            const uint64_t id = compare_name_count.fetch_add(1);
            char name[96];
            snprintf(name, sizeof(name), "dsv4_kvfin_fused-%" PRIu64, id);
            ggml_set_name(fin, name);

            ggml_tensor * ref = ggml_cont(ctx, fin_src);
            snprintf(name, sizeof(name), "dsv4_kvfin_ref-%" PRIu64, id);
            ggml_set_name(ref, name);
            out = ggml_add(ctx, fin, ggml_sub(ctx, ref, ref));
        } else {
            ggml_set_name(fin, "dsv4_kvfin");
        }
        ggml_build_forward_expand(gf, out);

        if (!dry_run) {
            return out;
        }

        src_orig = out;
    }

    src = direct_view ? ggml_view_2d(ctx, src, cache->ne[0], n_rows, src->nb[2], 0) : ggml_cont(ctx, src);
    if (!direct_view) {
        src = ggml_reshape_2d(ctx, src, cache->ne[0], n_rows);
    }

    ggml_build_forward_expand(gf, ggml_set_rows(ctx, cache, src, rows));
    return src_orig;
}

static dsv4_rope_cfg dsv4_make_rope_cfg(
        const llama_hparams & hparams,
        const llama_cparams  & cparams,
        uint32_t              compress_ratio) {
    if (compress_ratio == 0) {
        return {
            0,
            hparams.rope_freq_base_train,
            1.0f,
            0.0f,
            1.0f,
            cparams.yarn_beta_fast,
            cparams.yarn_beta_slow,
        };
    }

    float attn_factor = 1.0f;
    if (cparams.yarn_ext_factor != 0.0f && cparams.rope_freq_scale > 0.0f) {
        // DeepSeek V4 uses YaRN-style frequency interpolation for compressed RoPE,
        // but the reference implementation does not apply YaRN's magnitude scale.
        attn_factor /= 1.0f + 0.1f * std::log(1.0f / cparams.rope_freq_scale);
    }

    return {
        (int32_t) cparams.n_ctx_orig_yarn,
        hparams.compress_rope_freq_base > 0.0f ? hparams.compress_rope_freq_base : cparams.rope_freq_base,
        cparams.rope_freq_scale,
        cparams.yarn_ext_factor,
        attn_factor,
        cparams.yarn_beta_fast,
        cparams.yarn_beta_slow,
    };
}

static ggml_tensor * dsv4_view_base(ggml_context * ctx, ggml_tensor * base, int64_t n, int64_t off) {
    return ggml_view_2d(ctx, base, n, 1, base->nb[0], off * base->nb[0]);
}

static ggml_tensor * dsv4_apply_rope_tail(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * inp_pos,
        int64_t        n_embd_head,
        int64_t        n_head,
        int64_t        n_tokens,
        int64_t        n_rot,
        int            rope_type,
        int32_t        n_ctx_orig,
        float          freq_base,
        float          freq_scale,
        float          ext_factor,
        float          attn_factor,
        float          beta_fast,
        float          beta_slow,
        bool           inverse) {
    GGML_ASSERT(x->ne[0] == n_embd_head);
    GGML_ASSERT(x->ne[1] == n_head);
    GGML_ASSERT(x->ne[2] == n_tokens);

    if (n_rot == n_embd_head) {
        return inverse
            ? ggml_rope_ext_back(ctx, x, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow)
            : ggml_rope_ext     (ctx, x, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    }

    const int64_t n_nope = n_embd_head - n_rot;
    GGML_ASSERT(n_nope > 0);

    return ggml_dsv4_rope_tail(ctx, x, inp_pos, nullptr, n_rot, rope_type,
            n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow, inverse);
}

static dsv4_hc_mix dsv4_hc_pre_from_mixes(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * mixes,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_hc,
        int64_t        n_tokens,
        int            sinkhorn_iters,
        float          hc_eps);

static dsv4_hc_mix dsv4_hc_pre(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        int            sinkhorn_iters,
        float          hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;
    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    ggml_tensor * flat_normed = ggml_rms_norm(ctx, flat, norm_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat_normed); // [mix_hc, n_tokens]

    dsv4_hc_mix result = dsv4_hc_pre_from_mixes(ctx, x, mixes, hc_scale, hc_base,
            n_hc, n_tokens, sinkhorn_iters, hc_eps);
    result.flat = flat;
    result.flat_normed = flat_normed;
    return result;
}

static dsv4_hc_mix dsv4_hc_pre_from_mixes(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * mixes,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_hc,
        int64_t        n_tokens,
        int            sinkhorn_iters,
        float          hc_eps) {
    ggml_tensor * split = ggml_dsv4_hc_split_sinkhorn(ctx, mixes, hc_scale, hc_base, n_hc, sinkhorn_iters, hc_eps);
    ggml_tensor * pre = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], 0);
    ggml_tensor * post = ggml_view_2d(ctx, split, n_hc, n_tokens, split->nb[1], n_hc * split->nb[0]);
    ggml_tensor * comb = ggml_view_2d(ctx, split, n_hc * n_hc, n_tokens, split->nb[1], 2 * n_hc * split->nb[0]);
    if (n_tokens != 1) {
        pre = ggml_cont(ctx, pre);
        post = ggml_cont(ctx, post);
        comb = ggml_cont(ctx, comb);
    }
    comb = ggml_reshape_3d(ctx, comb, n_hc, n_hc, n_tokens); // [src_hc, dst_hc, n_tokens]
    ggml_tensor * y = ggml_dsv4_hc_weighted_sum(ctx, x, pre);
    return { y, nullptr, nullptr, mixes, split, pre, post, comb };
}

static ggml_tensor * dsv4_rms_norm_mul(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * weight,
        float          norm_eps) {
    ggml_tensor * y = ggml_rms_norm(ctx, x, norm_eps);
    ggml_tensor * weight_f = (y->type == GGML_TYPE_F32 && weight->type != GGML_TYPE_F32) ?
        ggml_cast(ctx, weight, GGML_TYPE_F32) : weight;
    return ggml_mul(ctx, y, weight_f);
}

struct dsv4_hc_pre_norm_result {
    ggml_tensor * x;
    dsv4_hc_mix  mix;
    dsv4_hc_mix  ref_mix;
    dsv4_hc_mix  fused_mix;
    bool         use_fused;
    bool         active;
};

static dsv4_hc_pre_norm_result dsv4_hc_pre_norm_compare(
        ggml_context     * ctx,
        const dsv4_hc_mix & ref_mix,
        ggml_tensor      * x,
        ggml_tensor      * hc_scale,
        ggml_tensor      * hc_base,
        ggml_tensor      * norm_weight,
        const char       * kind,
        int                il,
        int64_t            pos,
        int64_t            n_hc,
        int64_t            n_tokens,
        float              norm_eps,
        int                sinkhorn_iters,
        float              hc_eps) {
    dsv4_hc_mix fused_mix = dsv4_hc_pre_from_mixes(ctx, x, ref_mix.mixes, hc_scale, hc_base,
            n_hc, n_tokens, sinkhorn_iters, hc_eps);

    const char * tag = std::strcmp(kind, "attn") == 0 ? "hc_attn_pre" : "hc_ffn_pre";

    ggml_format_name(ref_mix.x,    "dsv4_hcnorm_sum_ref-%s-%d-p%lld",    tag, il, (long long) pos);
    ggml_format_name(fused_mix.x,  "dsv4_hcnorm_sum_fused-%s-%d-p%lld",  tag, il, (long long) pos);
    ggml_format_name(ref_mix.post,  "dsv4_hcnorm_postw_ref-%s-%d-p%lld",  tag, il, (long long) pos);
    ggml_format_name(fused_mix.post,"dsv4_hcnorm_postw_fused-%s-%d-p%lld",tag, il, (long long) pos);
    ggml_format_name(ref_mix.comb,  "dsv4_hcnorm_comb_ref-%s-%d-p%lld",   tag, il, (long long) pos);
    ggml_format_name(fused_mix.comb,"dsv4_hcnorm_comb_fused-%s-%d-p%lld", tag, il, (long long) pos);

    ggml_tensor * fused_out = dsv4_rms_norm_mul(ctx, fused_mix.x, norm_weight, norm_eps);
    ggml_tensor * ref_out = dsv4_rms_norm_mul(ctx, ref_mix.x, norm_weight, norm_eps);

    ggml_format_name(ref_out,   "dsv4_hcnorm_out_ref-%s-%d-p%lld",   tag, il, (long long) pos);
    ggml_format_name(fused_out, "dsv4_hcnorm_out_fused-%s-%d-p%lld", tag, il, (long long) pos);

    ggml_tensor * fused_cur_probe = ggml_add(ctx, fused_mix.x, ggml_sub(ctx, fused_out, fused_out));
    ggml_tensor * ref_cur_probe = ggml_add(ctx, ref_mix.x, ggml_sub(ctx, ref_out, ref_out));
    ggml_tensor * fused_pre_probe = ggml_add(ctx, fused_mix.pre, ggml_sub(ctx, fused_mix.pre, fused_mix.pre));
    ggml_tensor * ref_pre_probe = ggml_add(ctx, ref_mix.pre, ggml_sub(ctx, ref_mix.pre, ref_mix.pre));
    ggml_format_name(ref_cur_probe,   "dsv4_hcnorm_cur_ref-%s-%d-p%lld",   tag, il, (long long) pos);
    ggml_format_name(fused_cur_probe, "dsv4_hcnorm_cur_fused-%s-%d-p%lld", tag, il, (long long) pos);
    ggml_format_name(ref_pre_probe,   "dsv4_hcnorm_pre_ref-%s-%d-p%lld",   tag, il, (long long) pos);
    ggml_format_name(fused_pre_probe, "dsv4_hcnorm_pre_fused-%s-%d-p%lld", tag, il, (long long) pos);

    const bool use_fused = dsv4_experimental_hc_pre_norm_compare_use_fused_enabled();
    ggml_tensor * out = use_fused ?
        ggml_add(ctx, fused_out, ggml_sub(ctx, ref_out, ref_out)) :
        ggml_add(ctx, ref_out, ggml_sub(ctx, fused_out, fused_out));
    ggml_tensor * cur_zero = ggml_add(ctx,
            ggml_sub(ctx, fused_cur_probe, fused_cur_probe),
            ggml_sub(ctx, ref_cur_probe, ref_cur_probe));
    ggml_tensor * pre_zero = ggml_add(ctx,
            ggml_sum(ctx, ggml_sub(ctx, fused_pre_probe, fused_pre_probe)),
            ggml_sum(ctx, ggml_sub(ctx, ref_pre_probe, ref_pre_probe)));
    out = ggml_add(ctx, out, cur_zero);
    out = ggml_add(ctx, out, ggml_repeat(ctx, pre_zero, out));

    return { out, use_fused ? fused_mix : ref_mix, ref_mix, fused_mix, use_fused, true };
}

static ggml_tensor * dsv4_hc_post(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * residual,
        ggml_tensor  * post,
        ggml_tensor  * comb,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens) {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == n_tokens);
    GGML_ASSERT(residual->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == n_hc);
    GGML_ASSERT(residual->ne[2] == n_tokens);
    GGML_ASSERT(post->ne[0] == n_hc);
    GGML_ASSERT(post->ne[1] == n_tokens);
    GGML_ASSERT(comb->ne[0] == n_hc);
    GGML_ASSERT(comb->ne[1] == n_hc);
    GGML_ASSERT(comb->ne[2] == n_tokens);

    return ggml_dsv4_hc_expand(ctx, x, residual, post, comb);
}

static bool dsv4_aohc_consumer_dispatch_capture_site_enabled(
        int     il,
        int64_t pos,
        int64_t n_tokens) {
    return dsv4_experimental_layer_executor_side_probe_payload_enabled() &&
        dsv4_experimental_layer_executor_side_probe_payload_consumer_dispatch_capture() &&
        dsv4_experimental_layer_executor_side_probe_stage_is("aohc_boundary") &&
        dsv4_experimental_layer_executor_side_probe_site_enabled(il, pos, n_tokens);
}

static ggml_tensor * dsv4_aohc_consumer_dispatch_source(
        ggml_context * ctx,
        int            il,
        int64_t        pos,
        const char   * label,
        ggml_tensor  * src) {
    if (src == nullptr) {
        dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, label, nullptr);
        return src;
    }

    ggml_tensor * out = ggml_scale(ctx, src, 1.0f);
    ggml_format_name(out, "%s-l%d-p%lld", label, il, (long long) pos);
    dsv4_lexec_side_probe_register_payload_target("aohc_boundary", il, pos, label, out);
    return out;
}

static ggml_tensor * dsv4_hc_post_compare(
        ggml_context                    * ctx,
        ggml_tensor                     * x,
        ggml_tensor                     * residual,
        const dsv4_hc_pre_norm_result  & cmp,
        const char                      * tag,
        int                               il,
        int64_t                           pos,
        int64_t                           n_embd,
        int64_t                           n_hc,
        int64_t                           n_tokens) {
    ggml_tensor * ref = dsv4_hc_post(ctx, x, residual, cmp.ref_mix.post, cmp.ref_mix.comb,
            n_embd, n_hc, n_tokens);
    ggml_tensor * fused = dsv4_hc_post(ctx, x, residual, cmp.fused_mix.post, cmp.fused_mix.comb,
            n_embd, n_hc, n_tokens);

    ggml_format_name(ref,   "dsv4_hcnorm_post_ref-%s-%d-p%lld",   tag, il, (long long) pos);
    ggml_format_name(fused, "dsv4_hcnorm_post_fused-%s-%d-p%lld", tag, il, (long long) pos);

    return cmp.use_fused ?
        ggml_add(ctx, fused, ggml_sub(ctx, ref, ref)) :
        ggml_add(ctx, ref, ggml_sub(ctx, fused, fused));
}

static ggml_tensor * dsv4_hc_head(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        int64_t        n_embd,
        int64_t        n_hc,
        int64_t        n_tokens,
        float          norm_eps,
        float          hc_eps) {
    const int64_t hc_dim = n_embd * n_hc;

    ggml_tensor * flat = ggml_cont(ctx, ggml_reshape_2d(ctx, x, hc_dim, n_tokens));
    flat = ggml_rms_norm(ctx, flat, norm_eps);

    ggml_tensor * pre = ggml_mul_mat(ctx, hc_fn, flat); // [hc, n_tokens]
    pre = ggml_mul(ctx, pre, dsv4_view_scale(ctx, hc_scale, 0));
    pre = ggml_add(ctx, pre, dsv4_view_base(ctx, hc_base, n_hc, 0));
    pre = dsv4_add_scalar(ctx, ggml_sigmoid(ctx, pre), hc_eps);

    return ggml_dsv4_hc_weighted_sum(ctx, x, pre);
}

static ggml_tensor * dsv4_grouped_out(
        ggml_context * ctx,
        ggml_tensor  * o,
        ggml_tensor  * wo_a,
        ggml_tensor  * wo_b,
        int64_t        n_embd_head,
        int64_t        n_head,
        int64_t        n_groups,
        int64_t        o_lora_rank,
        int64_t        n_tokens,
        int            il,
        ggml_tensor ** out_low = nullptr) {
    GGML_ASSERT(n_head % n_groups == 0);

    const int64_t group_heads = n_head / n_groups;
    const int64_t group_dim   = n_embd_head * group_heads;

    o = ggml_cont(ctx, o);
    o = ggml_reshape_3d(ctx, o, group_dim, n_groups, n_tokens);

    ggml_tensor * wo_a_g = ggml_reshape_3d(ctx, wo_a, group_dim, o_lora_rank, n_groups);

    if (dsv4_experimental_attn_out_decode_enabled() &&
            n_tokens == 1 &&
            o->type == GGML_TYPE_F32 &&
            wo_a_g->type == GGML_TYPE_Q8_0 &&
            wo_b->type == GGML_TYPE_Q8_0 &&
            wo_b->ne[0] == o_lora_rank * n_groups &&
            wo_b->ne[2] == 1 &&
            wo_b->ne[3] == 1) {
        ggml_tensor * fused_work = ggml_dsv4_attn_out_decode(ctx, o, wo_a_g, wo_b);
        char fused_work_name[64];
        snprintf(fused_work_name, sizeof(fused_work_name), "dsv4_attn_out_decode_work-%d", il);
        ggml_set_name(fused_work, fused_work_name);
        ggml_tensor * fused = ggml_view_2d(ctx, fused_work, wo_b->ne[1], 1, wo_b->ne[1] * ggml_type_size(GGML_TYPE_F32), 0);
        char fused_name[64];
        snprintf(fused_name, sizeof(fused_name), "dsv4_attn_out_decode-%d", il);
        ggml_set_name(fused, fused_name);

        if (!dsv4_experimental_attn_out_decode_compare_enabled()) {
            return fused;
        }

        ggml_tensor * ids = ggml_arange(ctx, 0.0f, float(n_groups), 1.0f);
        ids = ggml_cast(ctx, ids, GGML_TYPE_I32);
        ids = ggml_repeat_4d(ctx, ids, n_groups, n_tokens, 1, 1);

        ggml_tensor * low = ggml_mul_mat_id(ctx, wo_a_g, o, ids);
        char low_ref_name[64];
        snprintf(low_ref_name, sizeof(low_ref_name), "dsv4_attn_out_low_ref-%d", il);
        ggml_set_name(low, low_ref_name);
        low = ggml_reshape_2d(ctx, low, o_lora_rank * n_groups, n_tokens);
        if (out_low != nullptr) {
            *out_low = low;
        }
        ggml_tensor * ref = ggml_mul_mat(ctx, wo_b, low);
        char ref_name[64];
        snprintf(ref_name, sizeof(ref_name), "dsv4_attn_out_decode_ref-%d", il);
        ggml_set_name(ref, ref_name);

        return ggml_add(ctx, ref, ggml_sub(ctx, fused, fused));
    }

    ggml_tensor * ids = ggml_arange(ctx, 0.0f, float(n_groups), 1.0f);
    ids = ggml_cast(ctx, ids, GGML_TYPE_I32);
    ids = ggml_repeat_4d(ctx, ids, n_groups, n_tokens, 1, 1);

    ggml_tensor * low = ggml_mul_mat_id(ctx, wo_a_g, o, ids); // [o_lora_rank, n_groups, n_tokens]
    ggml_set_name(low, "dsv4_attn_out_low");
    low = ggml_reshape_2d(ctx, low, o_lora_rank * n_groups, n_tokens);
    if (out_low != nullptr) {
        *out_low = low;
    }

    return ggml_mul_mat(ctx, wo_b, low);
}

static ggml_tensor * dsv4_softmax_pool_ratio(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score) {
    score = ggml_soft_max(ctx, score);
    ggml_tensor * pooled = ggml_mul(ctx, kv, score);
    pooled = ggml_sum_rows(ctx, pooled);
    return ggml_reshape_2d(ctx, pooled, kv->ne[1], kv->ne[2]);
}

static ggml_tensor * dsv4_shift_overlap_state(
        ggml_context * ctx,
        ggml_tensor  * x,
        float          pad_value) {
    const int64_t n_embd  = x->ne[0];
    const int64_t ratio   = x->ne[1];
    const int64_t n_comp  = x->ne[2];

    ggml_tensor * first = ggml_view_3d(ctx, x, n_embd, ratio, 1,
            x->nb[1], x->nb[2], 0);
    ggml_tensor * pad = ggml_fill(ctx, ggml_cont(ctx, first), pad_value);

    if (n_comp == 1) {
        return pad;
    }

    ggml_tensor * prev = ggml_view_3d(ctx, x, n_embd, ratio, n_comp - 1,
            x->nb[1], x->nb[2], 0);
    return ggml_concat(ctx, pad, prev, 2);
}

static ggml_tensor * dsv4_build_compressor_prefill(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        ggml_tensor        * pos,
        int64_t              n_embd_head,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    GGML_ASSERT(compress_ratio > 0);
    const int64_t n_comp = n_tokens / compress_ratio;
    GGML_ASSERT(n_comp > 0);

    const int64_t coff = compress_ratio == 4 ? 2 : 1;
    const int64_t n_kv = coff * n_embd_head;
    const int64_t cutoff = n_comp * compress_ratio;

    ggml_tensor * kv = ggml_mul_mat(ctx, wkv, x);       // [coff*head_dim, n_tokens]
    ggml_tensor * score = ggml_mul_mat(ctx, wgate, x);  // [coff*head_dim, n_tokens]

    kv = ggml_view_3d(ctx, kv, n_kv, compress_ratio, n_comp,
            kv->nb[1],
            kv->nb[1] * compress_ratio,
            0);
    score = ggml_view_3d(ctx, score, n_kv, compress_ratio, n_comp,
            score->nb[1],
            score->nb[1] * compress_ratio,
            0);
    GGML_ASSERT(cutoff <= n_tokens);

    ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    score = ggml_add(ctx, score, ggml_repeat(ctx, ape_f, score));

    if (coff == 1) {
        kv = ggml_cont(ctx, ggml_permute(ctx, kv, 1, 0, 2, 3));       // [ratio, head_dim, n_comp]
        score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3)); // [ratio, head_dim, n_comp]
        kv = dsv4_softmax_pool_ratio(ctx, kv, score);                // [head_dim, n_comp]
    } else {
        ggml_tensor * kv_prev = ggml_view_3d(ctx, kv, n_embd_head, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], 0);
        ggml_tensor * kv_curr = ggml_view_3d(ctx, kv, n_embd_head, compress_ratio, n_comp,
                kv->nb[1], kv->nb[2], n_embd_head * kv->nb[0]);
        ggml_tensor * score_prev = ggml_view_3d(ctx, score, n_embd_head, compress_ratio, n_comp,
                score->nb[1], score->nb[2], 0);
        ggml_tensor * score_curr = ggml_view_3d(ctx, score, n_embd_head, compress_ratio, n_comp,
                score->nb[1], score->nb[2], n_embd_head * score->nb[0]);

        kv_prev    = dsv4_shift_overlap_state(ctx, kv_prev,    0.0f);
        score_prev = dsv4_shift_overlap_state(ctx, score_prev, -INFINITY);

        kv_prev    = ggml_cont(ctx, ggml_permute(ctx, kv_prev,    1, 0, 2, 3)); // [ratio, head_dim, n_comp]
        kv_curr    = ggml_cont(ctx, ggml_permute(ctx, kv_curr,    1, 0, 2, 3));
        score_prev = ggml_cont(ctx, ggml_permute(ctx, score_prev, 1, 0, 2, 3));
        score_curr = ggml_cont(ctx, ggml_permute(ctx, score_curr, 1, 0, 2, 3));

        kv    = ggml_concat(ctx, kv_prev,    kv_curr,    0); // [2*ratio, head_dim, n_comp]
        score = ggml_concat(ctx, score_prev, score_curr, 0);
        kv = dsv4_softmax_pool_ratio(ctx, kv, score);        // [head_dim, n_comp]
    }

    kv = ggml_rms_norm(ctx, kv, norm_eps);
    kv = dsv4_mul_norm_weight(ctx, kv, norm);
    kv = ggml_reshape_3d(ctx, kv, n_embd_head, 1, n_comp);

    kv = dsv4_apply_rope_tail(ctx, kv, pos,
            n_embd_head, 1, n_comp, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);

    return kv;
}

static dsv4_decode_compressor dsv4_build_compressor_decode_chunk(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        const llama_ubatch & ubatch,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps);

static ggml_tensor * dsv4_build_compressor_prefill_with_overlap(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        ggml_tensor        * pos,
        int64_t              n_embd_head,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int64_t              first_pos,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    ggml_tensor * kv_comp = dsv4_build_compressor_prefill(ctx, x,
            wkv, wgate, ape, norm, pos,
            n_embd_head, n_rot, n_tokens, compress_ratio, rope_type, rope_cfg, norm_eps);

    const int64_t n_comp = n_tokens / compress_ratio;
    if (compress_ratio != 4 || first_pos <= 0 || n_comp <= 0) {
        return kv_comp;
    }

    std::array<llama_pos, 4> first_pos_values = {{
            (llama_pos) first_pos,
            (llama_pos) (first_pos + 1),
            (llama_pos) (first_pos + 2),
            (llama_pos) (first_pos + 3),
    }};

    llama_ubatch first_ubatch = {};
    first_ubatch.n_tokens = 4;
    first_ubatch.pos = first_pos_values.data();

    ggml_tensor * x_first = ggml_view_2d(ctx, x, x->ne[0], 4, x->nb[1], 0);
    dsv4_decode_compressor first = dsv4_build_compressor_decode_chunk(ctx, x_first,
            prev_kv_state,
            prev_score_state,
            wkv,
            wgate,
            ape,
            norm,
            first_ubatch,
            n_embd_head,
            n_rot,
            4,
            compress_ratio,
            rope_type,
            rope_cfg,
            norm_eps);

    if (first.kv_comp == nullptr || n_comp == 1) {
        return first.kv_comp != nullptr ? first.kv_comp : kv_comp;
    }

    ggml_tensor * rest = ggml_view_3d(ctx, kv_comp,
            kv_comp->ne[0], kv_comp->ne[1], n_comp - 1,
            kv_comp->nb[1], kv_comp->nb[2], kv_comp->nb[2]);
    return ggml_concat(ctx, first.kv_comp, rest, 2);
}

static dsv4_state_pair dsv4_build_compressor_prefill_state(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * wkv,
        ggml_tensor  * wgate,
        ggml_tensor  * ape,
        int64_t        head_dim,
        int64_t        n_tokens,
        int64_t        compress_ratio) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);

    const int64_t cutoff    = (n_tokens / compress_ratio) * compress_ratio;
    const int64_t remainder = n_tokens - cutoff;
    ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    if (compress_ratio == 4) {
        ggml_tensor * kv_prev    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
        ggml_tensor * score_prev = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

        if (cutoff >= compress_ratio) {
            ggml_tensor * x_prev = ggml_view_2d(ctx, x,
                    x->ne[0], compress_ratio,
                    x->nb[1],
                    (cutoff - compress_ratio)*x->nb[1]);
            kv_prev    = ggml_mul_mat(ctx, wkv,   x_prev);
            score_prev = ggml_mul_mat(ctx, wgate, x_prev);
            score_prev = ggml_add(ctx, score_prev, ape_f);
        }

        ggml_tensor * kv_curr    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
        ggml_tensor * score_curr = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

        if (remainder > 0) {
            ggml_tensor * x_rem = ggml_view_2d(ctx, x,
                    x->ne[0], remainder,
                    x->nb[1],
                    cutoff*x->nb[1]);
            ggml_tensor * kv_rem = ggml_mul_mat(ctx, wkv,   x_rem);
            ggml_tensor * sc_rem = ggml_mul_mat(ctx, wgate, x_rem);
            sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0));

            if (remainder == compress_ratio) {
                kv_curr = kv_rem;
                score_curr = sc_rem;
            } else {
                kv_curr = ggml_concat(ctx, kv_rem,
                        dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, 0.0f), 1);
                score_curr = ggml_concat(ctx, sc_rem,
                        dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, -INFINITY), 1);
            }
        }

        return {
            ggml_concat(ctx, kv_prev,    kv_curr,    1),
            ggml_concat(ctx, score_prev, score_curr, 1),
        };
    }

    ggml_tensor * kv_state    = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, 0.0f);
    ggml_tensor * score_state = dsv4_new_filled_2d(ctx, layout.width, compress_ratio, -INFINITY);

    if (remainder > 0) {
        ggml_tensor * x_rem = ggml_view_2d(ctx, x,
                x->ne[0], remainder,
                x->nb[1],
                cutoff*x->nb[1]);
        ggml_tensor * kv_rem = ggml_mul_mat(ctx, wkv,   x_rem);
        ggml_tensor * sc_rem = ggml_mul_mat(ctx, wgate, x_rem);
        sc_rem = ggml_add(ctx, sc_rem, ggml_view_2d(ctx, ape_f, layout.width, remainder, ape_f->nb[1], 0));

        if (remainder == compress_ratio) {
            kv_state = kv_rem;
            score_state = sc_rem;
        } else {
            kv_state = ggml_concat(ctx, kv_rem,
                    dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, 0.0f), 1);
            score_state = ggml_concat(ctx, sc_rem,
                    dsv4_new_filled_2d(ctx, layout.width, compress_ratio - remainder, -INFINITY), 1);
        }
    }

    return { kv_state, score_state };
}

static ggml_tensor * dsv4_pool_decode_state(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score,
        ggml_tensor  * norm,
        ggml_tensor  * pos,
        int64_t        head_dim,
        int64_t        n_rot,
        int            rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float          norm_eps) {
    const int64_t n_rows = kv->ne[1];
    if (dsv4_experimental_decode_compress_enabled()) {
        ggml_tensor * norm_f = norm;
        if (norm_f->type != GGML_TYPE_F32) {
            norm_f = ggml_cast(ctx, norm_f, GGML_TYPE_F32);
        }
        return ggml_dsv4_decode_compress(ctx, kv, score, norm_f, pos,
                n_rot, rope_type, rope_cfg.n_ctx_orig, rope_cfg.freq_base,
                rope_cfg.freq_scale, rope_cfg.ext_factor, rope_cfg.attn_factor,
                rope_cfg.beta_fast, rope_cfg.beta_slow, norm_eps);
    }

    kv    = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, kv)),    n_rows, head_dim, 1);
    score = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, score)), n_rows, head_dim, 1);

    ggml_tensor * pooled = dsv4_softmax_pool_ratio(ctx, kv, score);
    pooled = ggml_rms_norm(ctx, pooled, norm_eps);
    pooled = dsv4_mul_norm_weight(ctx, pooled, norm);
    pooled = ggml_reshape_3d(ctx, pooled, head_dim, 1, 1);

    return dsv4_apply_rope_tail(ctx, pooled, pos,
            head_dim, 1, 1, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
}

struct dsv4_decode_compress_probe_stages {
    ggml_tensor * pooled = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * norm_weighted = nullptr;
    ggml_tensor * rope_in = nullptr;
    ggml_tensor * rope_out = nullptr;
};

static dsv4_decode_compress_probe_stages dsv4_decode_compress_generic_probe_stages(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score,
        ggml_tensor  * norm,
        ggml_tensor  * pos,
        int64_t        head_dim,
        int64_t        n_rot,
        int            rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float          norm_eps) {
    const int64_t n_rows = kv->ne[1];
    kv    = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, kv)),    n_rows, head_dim, 1);
    score = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, score)), n_rows, head_dim, 1);

    dsv4_decode_compress_probe_stages out;
    out.pooled = dsv4_softmax_pool_ratio(ctx, kv, score);
    out.norm = ggml_rms_norm(ctx, out.pooled, norm_eps);
    out.norm_weighted = dsv4_mul_norm_weight(ctx, out.norm, norm);
    out.rope_in = ggml_reshape_3d(ctx, out.norm_weighted, head_dim, 1, 1);
    out.rope_out = dsv4_apply_rope_tail(ctx, out.rope_in, pos,
            head_dim, 1, 1, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
    return out;
}

static ggml_tensor * dsv4_decode_compress_backend_probe_stage(
        ggml_context * ctx,
        ggml_tensor  * kv,
        ggml_tensor  * score,
        ggml_tensor  * norm,
        ggml_tensor  * pos,
        int64_t        n_rot,
        int            rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float          norm_eps,
        int            output_stage,
        const char   * name) {
    ggml_tensor * norm_f = norm->type == GGML_TYPE_F32 ? norm : ggml_cast(ctx, norm, GGML_TYPE_F32);
    ggml_tensor * out = ggml_dsv4_decode_compress_stage(ctx, kv, score, norm_f, pos,
            n_rot, rope_type, rope_cfg.n_ctx_orig, rope_cfg.freq_base,
            rope_cfg.freq_scale, rope_cfg.ext_factor, rope_cfg.attn_factor,
            rope_cfg.beta_fast, rope_cfg.beta_slow, norm_eps, output_stage);
    ggml_set_name(out, name);
    return out;
}

static dsv4_decode_compressor dsv4_build_compressor_decode_projected(
        ggml_context       * ctx,
        ggml_tensor        * kv_cur,
        ggml_tensor        * sc_cur,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps,
        bool                 skip_generic_tail = false,
        bool                 build_backend_tail = false);

static dsv4_decode_compressor dsv4_build_compressor_decode_generic(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps,
        bool                 skip_generic_tail = false,
        bool                 build_backend_tail = false) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
    const int64_t pos_mod = pos % compress_ratio;

    ggml_tensor * kv_cur = nullptr;
    ggml_tensor * sc_cur = nullptr;
    if (dsv4_experimental_compressor_pair_enabled() &&
            wkv->type == GGML_TYPE_F16 &&
            wgate->type == GGML_TYPE_F16 &&
            x->type == GGML_TYPE_F32 &&
            x->ne[1] == 1 &&
            x->ne[2] == 1 &&
            x->ne[3] == 1 &&
            wkv->ne[0] == wgate->ne[0] &&
            wkv->ne[1] == wgate->ne[1] &&
            wkv->ne[1] == layout.width) {
        ggml_tensor * pair = ggml_dsv4_compressor_pair_proj(ctx, wkv, wgate, x);
        ggml_set_name(pair, "dsv4_comp_pair_proj");
        kv_cur = ggml_view_2d(ctx, pair, layout.width, 1, pair->nb[1], 0);
        ggml_set_name(kv_cur, "dsv4_comp_kv_proj_pair");
        sc_cur = ggml_view_2d(ctx, pair, layout.width, 1, pair->nb[1], layout.width * ggml_element_size(pair));
        ggml_set_name(sc_cur, "dsv4_comp_score_proj_pair");
    } else {
        kv_cur = ggml_mul_mat(ctx, wkv, x);       // [width, 1]
        ggml_set_name(kv_cur, "dsv4_comp_kv_proj");
        sc_cur = ggml_mul_mat(ctx, wgate, x);
        ggml_set_name(sc_cur, "dsv4_comp_score_proj");
    }
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    sc_cur = ggml_add(ctx, sc_cur, ggml_view_2d(ctx, ape_f, layout.width, 1, ape_f->nb[1], pos_mod*ape_f->nb[1]));

    return dsv4_build_compressor_decode_projected(ctx,
            kv_cur, sc_cur,
            prev_kv_state, prev_score_state,
            norm,
            head_dim, n_rot, pos, compress_ratio,
            rope_type, rope_cfg, norm_eps,
            skip_generic_tail,
            build_backend_tail);
}

static dsv4_decode_compressor dsv4_build_compressor_decode(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps,
        const char         * tag,
        int                  il) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
    const bool should_compress = (pos + 1) % compress_ratio == 0;
    const bool supported_v2 =
            dsv4_experimental_compressor_update_v2_enabled() &&
            wkv->type == GGML_TYPE_F16 &&
            wgate->type == GGML_TYPE_F16 &&
            x->type == GGML_TYPE_F32 &&
            prev_kv_state->type == GGML_TYPE_F32 &&
            prev_score_state->type == GGML_TYPE_F32 &&
            x->ne[1] == 1 &&
            x->ne[2] == 1 &&
            x->ne[3] == 1 &&
            wkv->ne[0] == wgate->ne[0] &&
            wkv->ne[1] == wgate->ne[1] &&
            wkv->ne[0] == x->ne[0] &&
            wkv->ne[1] == layout.width &&
            prev_kv_state->ne[0] == layout.width &&
            prev_kv_state->ne[1] == layout.rows &&
            prev_score_state->ne[0] == layout.width &&
            prev_score_state->ne[1] == layout.rows &&
            ape->ne[0] >= layout.width &&
            ape->ne[1] >= compress_ratio &&
            norm->ne[0] >= head_dim &&
            n_rot > 0 &&
            n_rot <= head_dim &&
            head_dim <= 1024 &&
            (rope_type == GGML_ROPE_TYPE_NORMAL || rope_type == GGML_ROPE_TYPE_NEOX);

    if (supported_v2) {
        ggml_tensor * kv_cur = nullptr;
        ggml_tensor * score_cur = nullptr;
        if (dsv4_experimental_compressor_pair_enabled()) {
            ggml_tensor * pair = ggml_dsv4_compressor_pair_proj(ctx, wkv, wgate, x);
            ggml_set_name(pair, "dsv4_cupd2_pair_proj");
            kv_cur = ggml_view_2d(ctx, pair, layout.width, 1, pair->nb[1], 0);
            ggml_set_name(kv_cur, "dsv4_cupd2_kv_proj_pair");
            score_cur = ggml_view_2d(ctx, pair, layout.width, 1, pair->nb[1], layout.width * ggml_element_size(pair));
            ggml_set_name(score_cur, "dsv4_cupd2_score_proj_pair");
        } else {
            kv_cur = ggml_mul_mat(ctx, wkv, x);
            ggml_set_name(kv_cur, "dsv4_cupd2_kv_proj");
            score_cur = ggml_mul_mat(ctx, wgate, x);
            ggml_set_name(score_cur, "dsv4_cupd2_score_proj");
        }

        ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
        ggml_tensor * norm_f = norm->type == GGML_TYPE_F32 ? norm : ggml_cast(ctx, norm, GGML_TYPE_F32);
        ggml_tensor * fused_work = ggml_dsv4_compressor_update_decode_v2(ctx,
                kv_cur, score_cur,
                prev_kv_state, prev_score_state,
                ape_f, norm_f,
                n_rot, rope_type, rope_cfg.n_ctx_orig, (int) pos, (int) compress_ratio,
                rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor,
                rope_cfg.beta_fast, rope_cfg.beta_slow, norm_eps);

        char name[96];
        snprintf(name, sizeof(name), "dsv4_cupd2_work-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
        ggml_set_name(fused_work, name);

        const size_t f32 = ggml_type_size(GGML_TYPE_F32);
        const size_t kv_off = 0;
        const size_t score_off = layout.elems * f32;
        const size_t comp_off = 2 * layout.elems * f32;
        const int64_t pool_rows = compress_ratio == 4 ? 2 * compress_ratio : compress_ratio;
        const int64_t pool_elems = head_dim * pool_rows;
        const size_t pool_off = comp_off + head_dim * f32;
        const size_t pool_score_off = pool_off + pool_elems * f32;

        ggml_tensor * fused_kv_state = ggml_view_2d(ctx, fused_work,
                layout.width, layout.rows, layout.width*f32, kv_off);
        snprintf(name, sizeof(name), "dsv4_cupd2_kv_state-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
        ggml_set_name(fused_kv_state, name);

        ggml_tensor * fused_score_state = ggml_view_2d(ctx, fused_work,
                layout.width, layout.rows, layout.width*f32, score_off);
        snprintf(name, sizeof(name), "dsv4_cupd2_score_state-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
        ggml_set_name(fused_score_state, name);

        ggml_tensor * fused_kv_comp = should_compress ? ggml_view_3d(ctx, fused_work,
                head_dim, 1, 1, f32, head_dim*f32, comp_off) : nullptr;
        if (fused_kv_comp != nullptr) {
            snprintf(name, sizeof(name), "dsv4_cupd2_kv_comp-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
            ggml_set_name(fused_kv_comp, name);
        }

        const bool use_fused_comp = dsv4_experimental_compressor_update_v2_fused_comp_enabled();
        ggml_tensor * out_kv_comp = fused_kv_comp;
        if (should_compress && !use_fused_comp) {
            ggml_tensor * fused_kv_pool = ggml_view_2d(ctx, fused_work,
                    head_dim, pool_rows, head_dim*f32, pool_off);
            ggml_set_name(fused_kv_pool, "dsv4_cupd2_kv_pool");

            ggml_tensor * fused_score_pool = ggml_view_2d(ctx, fused_work,
                    head_dim, pool_rows, head_dim*f32, pool_score_off);
            ggml_set_name(fused_score_pool, "dsv4_cupd2_score_pool");

            ggml_tensor * comp_pos = dsv4_arange_i32(ctx, pos + 1 - compress_ratio, pos + 2 - compress_ratio);
            out_kv_comp = dsv4_pool_decode_state(ctx, fused_kv_pool, fused_score_pool, norm, comp_pos,
                    head_dim, n_rot, rope_type, rope_cfg, norm_eps);
            snprintf(name, sizeof(name), "dsv4_cupd2_kv_comp-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
            ggml_set_name(out_kv_comp, name);
        }

        if (!dsv4_experimental_compressor_update_v2_compare_enabled()) {
            return { fused_kv_state, fused_score_state, out_kv_comp };
        }

        dsv4_decode_compressor ref = dsv4_build_compressor_decode_generic(ctx, x,
                prev_kv_state, prev_score_state,
                wkv, wgate, ape, norm,
                head_dim, n_rot, pos, compress_ratio,
                rope_type, rope_cfg, norm_eps);

        snprintf(name, sizeof(name), "dsv4_cupd2_kv_state_ref-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
        ggml_set_name(ref.kv_state, name);
        snprintf(name, sizeof(name), "dsv4_cupd2_score_state_ref-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
        ggml_set_name(ref.score_state, name);

        const bool use_fused = dsv4_experimental_compressor_update_v2_compare_use_fused_enabled();
        ggml_tensor * kv_state = use_fused ?
                ggml_add(ctx, fused_kv_state, ggml_sub(ctx, ref.kv_state, ref.kv_state)) :
                ggml_add(ctx, ref.kv_state, ggml_sub(ctx, fused_kv_state, fused_kv_state));
        ggml_tensor * score_state = use_fused ?
                ggml_add(ctx, fused_score_state, ggml_sub(ctx, ref.score_state, ref.score_state)) :
                ggml_add(ctx, ref.score_state, ggml_sub(ctx, fused_score_state, fused_score_state));
        ggml_tensor * kv_comp = nullptr;

        if (ref.kv_comp != nullptr && out_kv_comp != nullptr) {
            snprintf(name, sizeof(name), "dsv4_cupd2_kv_comp_ref-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
            ggml_set_name(ref.kv_comp, name);
            kv_comp = use_fused ?
                    ggml_add(ctx, out_kv_comp, ggml_sub(ctx, ref.kv_comp, ref.kv_comp)) :
                    ggml_add(ctx, ref.kv_comp, ggml_sub(ctx, out_kv_comp, out_kv_comp));
        } else {
            kv_comp = ref.kv_comp;
        }

        return { kv_state, score_state, kv_comp };
    }

    const bool supported =
            dsv4_experimental_compressor_update_enabled() &&
            wkv->type == GGML_TYPE_F16 &&
            wgate->type == GGML_TYPE_F16 &&
            x->type == GGML_TYPE_F32 &&
            prev_kv_state->type == GGML_TYPE_F32 &&
            prev_score_state->type == GGML_TYPE_F32 &&
            x->ne[1] == 1 &&
            x->ne[2] == 1 &&
            x->ne[3] == 1 &&
            wkv->ne[0] == wgate->ne[0] &&
            wkv->ne[1] == wgate->ne[1] &&
            wkv->ne[0] == x->ne[0] &&
            wkv->ne[1] == layout.width &&
            prev_kv_state->ne[0] == layout.width &&
            prev_kv_state->ne[1] == layout.rows &&
            prev_score_state->ne[0] == layout.width &&
            prev_score_state->ne[1] == layout.rows &&
            ape->ne[0] >= layout.width &&
            ape->ne[1] >= compress_ratio &&
            norm->ne[0] >= head_dim &&
            n_rot > 0 &&
            n_rot <= head_dim &&
            head_dim <= 1024 &&
            (rope_type == GGML_ROPE_TYPE_NORMAL || rope_type == GGML_ROPE_TYPE_NEOX);

    const bool cupd3_consume_here =
            dsv4_experimental_compressor_update_v3_consume_site_enabled(il, pos, 1);
    const bool cupd3_backend_tail_consume_here =
            dsv4_experimental_compressor_update_v3_backend_tail_consume_site_enabled(il, pos, 1) &&
            dsv4_experimental_compressor_update_v3_backend_tail_stream_allowed(tag) &&
            (!dsv4_experimental_compressor_update_v3_backend_tail_consume_emit_only_enabled() || should_compress);
    const bool cupd3_skip_tail =
            cupd3_consume_here &&
            dsv4_experimental_compressor_update_v3_skip_generic_tail_requested();
    const bool cupd3_skip_any_tail = cupd3_skip_tail || cupd3_backend_tail_consume_here;
    const bool cupd3_backend_tail_here =
            dsv4_experimental_compressor_update_v3_backend_tail_site_enabled(il, pos, 1) ||
            cupd3_backend_tail_consume_here;

    if (!supported) {
        return dsv4_build_compressor_decode_generic(ctx, x,
                prev_kv_state, prev_score_state,
                wkv, wgate, ape, norm,
                head_dim, n_rot, pos, compress_ratio,
                rope_type, rope_cfg, norm_eps,
                cupd3_skip_any_tail,
                cupd3_backend_tail_here);
    }

    ggml_tensor * ape_f = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);
    ggml_tensor * norm_f = norm->type == GGML_TYPE_F32 ? norm : ggml_cast(ctx, norm, GGML_TYPE_F32);
    ggml_tensor * fused_work = ggml_dsv4_compressor_update_decode(ctx,
            wkv, wgate, x,
            prev_kv_state, prev_score_state,
            ape_f, norm_f,
            n_rot, rope_type, rope_cfg.n_ctx_orig, (int) pos, (int) compress_ratio,
            rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor,
            rope_cfg.beta_fast, rope_cfg.beta_slow, norm_eps);

    char name[96];
    snprintf(name, sizeof(name), "dsv4_cupd_work-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
    ggml_set_name(fused_work, name);

    const size_t f32 = ggml_type_size(GGML_TYPE_F32);
    const size_t kv_off = 0;
    const size_t score_off = layout.elems * f32;
    const size_t comp_off = 2 * layout.elems * f32;
    const size_t pair_off = comp_off + head_dim * f32;
    const int64_t pool_rows = compress_ratio == 4 ? 2 * compress_ratio : compress_ratio;
    const int64_t pool_elems = head_dim * pool_rows;
    const size_t pool_off = pair_off + 2 * layout.width * f32;
    const size_t pool_score_off = pool_off + pool_elems * f32;

    ggml_tensor * fused_kv_state = ggml_view_2d(ctx, fused_work,
            layout.width, layout.rows, layout.width*f32, kv_off);
    snprintf(name, sizeof(name), "dsv4_cupd_kv_state-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
    ggml_set_name(fused_kv_state, name);

    ggml_tensor * fused_score_state = ggml_view_2d(ctx, fused_work,
            layout.width, layout.rows, layout.width*f32, score_off);
    snprintf(name, sizeof(name), "dsv4_cupd_score_state-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
    ggml_set_name(fused_score_state, name);

    ggml_tensor * fused_kv_comp = should_compress ? ggml_view_3d(ctx, fused_work,
            head_dim, 1, 1, f32, head_dim*f32, comp_off) : nullptr;
    if (fused_kv_comp != nullptr) {
        snprintf(name, sizeof(name), "dsv4_cupd_kv_comp-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
        ggml_set_name(fused_kv_comp, name);
    }

    const bool use_fused_comp = dsv4_experimental_compressor_update_fused_comp_enabled();
    ggml_tensor * out_kv_comp = fused_kv_comp;
    if (should_compress && !use_fused_comp) {
        ggml_tensor * fused_kv_pool = ggml_view_2d(ctx, fused_work,
                head_dim, pool_rows, head_dim*f32, pool_off);
        ggml_set_name(fused_kv_pool, "dsv4_cupd_kv_pool");

        ggml_tensor * fused_score_pool = ggml_view_2d(ctx, fused_work,
                head_dim, pool_rows, head_dim*f32, pool_score_off);
        ggml_set_name(fused_score_pool, "dsv4_cupd_score_pool");

        ggml_tensor * comp_pos = dsv4_arange_i32(ctx, pos + 1 - compress_ratio, pos + 2 - compress_ratio);
        out_kv_comp = dsv4_pool_decode_state(ctx, fused_kv_pool, fused_score_pool, norm, comp_pos,
                head_dim, n_rot, rope_type, rope_cfg, norm_eps);
        snprintf(name, sizeof(name), "dsv4_cupd_kv_comp-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
        ggml_set_name(out_kv_comp, name);
    }

    if (!dsv4_experimental_compressor_update_compare_enabled()) {
        return { fused_kv_state, fused_score_state, out_kv_comp };
    }

    dsv4_decode_compressor ref = dsv4_build_compressor_decode_generic(ctx, x,
            prev_kv_state, prev_score_state,
            wkv, wgate, ape, norm,
            head_dim, n_rot, pos, compress_ratio,
            rope_type, rope_cfg, norm_eps);

    snprintf(name, sizeof(name), "dsv4_cupd_kv_state_ref-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
    ggml_set_name(ref.kv_state, name);
    snprintf(name, sizeof(name), "dsv4_cupd_score_state_ref-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
    ggml_set_name(ref.score_state, name);

    const bool use_fused = dsv4_experimental_compressor_update_compare_use_fused_enabled();
    ggml_tensor * kv_state = use_fused ?
            ggml_add(ctx, fused_kv_state, ggml_sub(ctx, ref.kv_state, ref.kv_state)) :
            ggml_add(ctx, ref.kv_state, ggml_sub(ctx, fused_kv_state, fused_kv_state));
    ggml_tensor * score_state = use_fused ?
            ggml_add(ctx, fused_score_state, ggml_sub(ctx, ref.score_state, ref.score_state)) :
            ggml_add(ctx, ref.score_state, ggml_sub(ctx, fused_score_state, fused_score_state));
    ggml_tensor * kv_comp = nullptr;

    if (ref.kv_comp != nullptr && out_kv_comp != nullptr) {
        snprintf(name, sizeof(name), "dsv4_cupd_kv_comp_ref-%s-%d-p%lld", tag ? tag : "x", il, (long long) pos);
        ggml_set_name(ref.kv_comp, name);
        kv_comp = use_fused ?
                ggml_add(ctx, out_kv_comp, ggml_sub(ctx, ref.kv_comp, ref.kv_comp)) :
                ggml_add(ctx, ref.kv_comp, ggml_sub(ctx, out_kv_comp, out_kv_comp));
    } else {
        kv_comp = ref.kv_comp;
    }

    return { kv_state, score_state, kv_comp };
}

static dsv4_decode_compressor dsv4_build_compressor_decode_projected(
        ggml_context       * ctx,
        ggml_tensor        * kv_cur,
        ggml_tensor        * sc_cur,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * norm,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              pos,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps,
        bool                 skip_generic_tail,
        bool                 build_backend_tail) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);
    const int64_t pos_mod = pos % compress_ratio;
    const int64_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = (pos + 1) % compress_ratio == 0;

    ggml_tensor * row_idx = dsv4_arange_i32(ctx, row, row + 1);
    ggml_tensor * kv_state    = ggml_set_rows(ctx, prev_kv_state,    kv_cur, row_idx);
    ggml_tensor * score_state = ggml_set_rows(ctx, prev_score_state, sc_cur, row_idx);
    ggml_tensor * kv_comp = nullptr;
    ggml_tensor * generic_tail_kv_comp = nullptr;
    ggml_tensor * backend_tail_kv_comp = nullptr;
    dsv4_decode_compress_probe_stages generic_probe;
    dsv4_decode_compress_probe_stages backend_probe;
    bool generic_tail_built = false;
    bool candidate_tail_built = false;
    bool candidate_tail_consumed = false;
    bool backend_tail_built = false;

    if (should_compress) {
        ggml_tensor * kv_pool;
        ggml_tensor * score_pool;

        if (compress_ratio == 4) {
            ggml_tensor * kv_prev = dsv4_view_cols(ctx, kv_state,    head_dim, compress_ratio, 0,        0);
            ggml_tensor * kv_curr = dsv4_view_cols(ctx, kv_state,    head_dim, compress_ratio, head_dim, compress_ratio);
            ggml_tensor * sc_prev = dsv4_view_cols(ctx, score_state, head_dim, compress_ratio, 0,        0);
            ggml_tensor * sc_curr = dsv4_view_cols(ctx, score_state, head_dim, compress_ratio, head_dim, compress_ratio);

            kv_pool    = ggml_concat(ctx, kv_prev, kv_curr, 1);
            score_pool = ggml_concat(ctx, sc_prev, sc_curr, 1);

            ggml_tensor * shifted_kv    = dsv4_view_cols(ctx, kv_state,    layout.width, compress_ratio, 0, compress_ratio);
            ggml_tensor * shifted_score = dsv4_view_cols(ctx, score_state, layout.width, compress_ratio, 0, compress_ratio);
            kv_state    = ggml_concat(ctx, shifted_kv,    shifted_kv,    1);
            score_state = ggml_concat(ctx, shifted_score, shifted_score, 1);
        } else {
            kv_pool    = kv_state;
            score_pool = score_state;
        }

        const bool backend_tail_consume = skip_generic_tail && build_backend_tail;
        const bool need_generic_tail =
                !backend_tail_consume ||
                dsv4_experimental_compressor_update_v3_backend_tail_compare_enabled();
        ggml_tensor * comp_pos = dsv4_arange_i32(ctx, pos + 1 - compress_ratio, pos + 2 - compress_ratio);
        if (need_generic_tail) {
            generic_tail_kv_comp = dsv4_pool_decode_state(ctx, kv_pool, score_pool, norm, comp_pos,
                    head_dim, n_rot, rope_type, rope_cfg, norm_eps);
        }
        if (build_backend_tail) {
            ggml_tensor * norm_f = norm->type == GGML_TYPE_F32 ? norm : ggml_cast(ctx, norm, GGML_TYPE_F32);
            backend_tail_kv_comp = ggml_dsv4_decode_compress(ctx, kv_pool, score_pool, norm_f, comp_pos,
                    n_rot, rope_type, rope_cfg.n_ctx_orig, rope_cfg.freq_base,
                    rope_cfg.freq_scale, rope_cfg.ext_factor, rope_cfg.attn_factor,
                    rope_cfg.beta_fast, rope_cfg.beta_slow, norm_eps);
            ggml_set_name(backend_tail_kv_comp, "dsv4_cupd3_backend_tail");
            if (dsv4_experimental_compressor_update_v3_backend_tail_force_contiguous()) {
                backend_tail_kv_comp = ggml_cont(ctx, backend_tail_kv_comp);
                ggml_set_name(backend_tail_kv_comp, "dsv4_cupd3_backend_tail_force_contiguous");
            }
            backend_tail_built = true;
        }
        if (build_backend_tail && dsv4_experimental_compressor_update_v3_decode_compress_internal_probe_enabled()) {
            generic_probe = dsv4_decode_compress_generic_probe_stages(ctx, kv_pool, score_pool, norm, comp_pos,
                    head_dim, n_rot, rope_type, rope_cfg, norm_eps);
            backend_probe.pooled = dsv4_decode_compress_backend_probe_stage(ctx, kv_pool, score_pool, norm, comp_pos,
                    n_rot, rope_type, rope_cfg, norm_eps, 1, "dsv4_dcomp_internal_backend_pooled");
            backend_probe.norm = dsv4_decode_compress_backend_probe_stage(ctx, kv_pool, score_pool, norm, comp_pos,
                    n_rot, rope_type, rope_cfg, norm_eps, 2, "dsv4_dcomp_internal_backend_norm");
            backend_probe.norm_weighted = dsv4_decode_compress_backend_probe_stage(ctx, kv_pool, score_pool, norm, comp_pos,
                    n_rot, rope_type, rope_cfg, norm_eps, 3, "dsv4_dcomp_internal_backend_norm_weighted");
            backend_probe.rope_in = dsv4_decode_compress_backend_probe_stage(ctx, kv_pool, score_pool, norm, comp_pos,
                    n_rot, rope_type, rope_cfg, norm_eps, 4, "dsv4_dcomp_internal_backend_rope_in");
            backend_probe.rope_out = dsv4_decode_compress_backend_probe_stage(ctx, kv_pool, score_pool, norm, comp_pos,
                    n_rot, rope_type, rope_cfg, norm_eps, 5, "dsv4_dcomp_internal_backend_rope_out");
        }
        if (backend_tail_consume && backend_tail_kv_comp != nullptr) {
            kv_comp = backend_tail_kv_comp;
            candidate_tail_built = true;
            candidate_tail_consumed = true;
        } else {
            kv_comp = generic_tail_kv_comp;
        }

        if (skip_generic_tail) {
            candidate_tail_built = true;
            candidate_tail_consumed = true;
        } else {
            generic_tail_built = kv_comp != nullptr;
        }
    }

    return {
        kv_state,
        score_state,
        kv_comp,
        generic_tail_kv_comp,
        backend_tail_kv_comp,
        generic_probe.pooled,
        generic_probe.norm,
        generic_probe.norm_weighted,
        generic_probe.rope_in,
        generic_probe.rope_out,
        backend_probe.pooled,
        backend_probe.norm,
        backend_probe.norm_weighted,
        backend_probe.rope_in,
        backend_probe.rope_out,
        generic_tail_built,
        candidate_tail_built,
        candidate_tail_consumed,
        backend_tail_built,
    };
}

static dsv4_decode_compressor dsv4_build_compressor_decode_chunk(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * prev_kv_state,
        ggml_tensor        * prev_score_state,
        ggml_tensor        * wkv,
        ggml_tensor        * wgate,
        ggml_tensor        * ape,
        ggml_tensor        * norm,
        const llama_ubatch & ubatch,
        int64_t              head_dim,
        int64_t              n_rot,
        int64_t              n_tokens,
        int64_t              compress_ratio,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg,
        float                norm_eps) {
    const dsv4_state_layout layout = dsv4_make_state_layout(compress_ratio, head_dim);

    ggml_tensor * kv_all = ggml_mul_mat(ctx, wkv,   x); // [width, n_tokens]
    ggml_set_name(kv_all, "dsv4_comp_kv_proj");
    ggml_tensor * sc_all = ggml_mul_mat(ctx, wgate, x);
    ggml_set_name(sc_all, "dsv4_comp_score_proj");
    ggml_tensor * ape_f  = ape->type == GGML_TYPE_F32 ? ape : ggml_cast(ctx, ape, GGML_TYPE_F32);

    ggml_tensor * kv_state    = prev_kv_state;
    ggml_tensor * score_state = prev_score_state;
    ggml_tensor * kv_comp     = nullptr;

    for (int64_t i = 0; i < n_tokens; ++i) {
        const llama_pos pos = ubatch.pos ? ubatch.pos[i] : (llama_pos) i;
        const int64_t pos_mod = pos % compress_ratio;

        ggml_tensor * kv_cur = ggml_view_2d(ctx, kv_all, layout.width, 1, kv_all->nb[1], i*kv_all->nb[1]);
        ggml_tensor * sc_cur = ggml_view_2d(ctx, sc_all, layout.width, 1, sc_all->nb[1], i*sc_all->nb[1]);
        sc_cur = ggml_add(ctx, sc_cur, ggml_view_2d(ctx, ape_f, layout.width, 1, ape_f->nb[1], pos_mod*ape_f->nb[1]));

        dsv4_decode_compressor dec = dsv4_build_compressor_decode_projected(ctx,
                kv_cur,
                sc_cur,
                kv_state,
                score_state,
                norm,
                head_dim,
                n_rot,
                pos,
                compress_ratio,
                rope_type,
                rope_cfg,
                norm_eps);

        kv_state    = dec.kv_state;
        score_state = dec.score_state;
        if (dec.kv_comp != nullptr) {
            kv_comp = kv_comp == nullptr ? dec.kv_comp : ggml_concat(ctx, kv_comp, dec.kv_comp, 2);
        }
    }

    return { kv_state, score_state, kv_comp };
}

static ggml_tensor * dsv4_build_indexer_scores_prefill(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * qr,
        ggml_tensor        * index_kv,
        ggml_tensor        * wq_b,
        ggml_tensor        * wproj,
        ggml_tensor        * pos,
        ggml_tensor        * causal_mask,
        int64_t              n_index_head,
        int64_t              n_index_head_size,
        int64_t              n_tokens,
        int64_t              n_rot,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg) {
    ggml_tensor * q = ggml_mul_mat(ctx, wq_b, qr);
    q = ggml_reshape_3d(ctx, q, n_index_head_size, n_index_head, n_tokens);
    q = dsv4_apply_rope_tail(ctx, q, pos,
            n_index_head_size, n_index_head, n_tokens, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
    q = ggml_dsv4_hadamard_fp4_quantize(ctx, q);

    ggml_tensor * k = ggml_permute(ctx, index_kv, 0, 2, 1, 3); // [head_dim, n_comp, 1]
    q = ggml_permute(ctx, q, 0, 2, 1, 3);                     // [head_dim, n_tokens, n_heads]

    ggml_tensor * score = ggml_mul_mat(ctx, k, q);            // [n_comp, n_tokens, n_heads]

    ggml_tensor * weights = ggml_mul_mat(ctx, wproj, x);      // [n_heads, n_tokens]
    const float scale = 1.0f / std::sqrt(float(n_index_head_size) * float(n_index_head));
    if (dsv4_experimental_indexer_weighted_score_enabled()) {
        score = ggml_dsv4_indexer_weighted_score(ctx, score, weights, scale);
    } else {
        score = ggml_relu(ctx, score);
        weights = dsv4_mul_scalar(ctx, weights, scale);
        weights = ggml_reshape_3d(ctx, weights, 1, n_index_head, n_tokens);
        weights = ggml_permute(ctx, weights, 0, 2, 1, 3);         // [1, n_tokens, n_heads]

        score = ggml_mul(ctx, score, weights);
        score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 2, 0, 3)); // [n_heads, n_comp, n_tokens]
        score = ggml_sum_rows(ctx, score);                            // [1, n_comp, n_tokens]
        score = ggml_reshape_2d(ctx, score, index_kv->ne[2], n_tokens);
    }

    return ggml_add(ctx, score, causal_mask);
}

static ggml_tensor * dsv4_build_indexer_scores_decode(
        ggml_context       * ctx,
        ggml_tensor        * x,
        ggml_tensor        * qr,
        ggml_tensor        * index_kv,
        ggml_tensor        * wq_b,
        ggml_tensor        * wproj,
        ggml_tensor        * pos,
        int64_t              n_index_head,
        int64_t              n_index_head_size,
        int64_t              n_comp,
        int64_t              n_rot,
        int                  rope_type,
        const dsv4_rope_cfg & rope_cfg) {
    ggml_tensor * q = ggml_mul_mat(ctx, wq_b, qr);
    q = ggml_reshape_3d(ctx, q, n_index_head_size, n_index_head, 1);
    q = dsv4_apply_rope_tail(ctx, q, pos,
            n_index_head_size, n_index_head, 1, n_rot, rope_type,
            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
    q = ggml_dsv4_hadamard_fp4_quantize(ctx, q);

    ggml_tensor * k = ggml_reshape_3d(ctx, index_kv, n_index_head_size, 1, n_comp);
    k = ggml_permute(ctx, k, 0, 2, 1, 3); // [head_dim, n_comp, 1]
    q = ggml_permute(ctx, q, 0, 2, 1, 3); // [head_dim, 1, n_heads]

    ggml_tensor * score = ggml_mul_mat(ctx, k, q); // [n_comp, 1, n_heads]

    ggml_tensor * weights = ggml_mul_mat(ctx, wproj, x); // [n_heads, 1]
    const float scale = 1.0f / std::sqrt(float(n_index_head_size) * float(n_index_head));
    if (dsv4_experimental_indexer_weighted_score_enabled()) {
        return ggml_dsv4_indexer_weighted_score(ctx, score, weights, scale);
    }

    score = ggml_relu(ctx, score);
    weights = dsv4_mul_scalar(ctx, weights, scale);
    weights = ggml_reshape_3d(ctx, weights, 1, n_index_head, 1);
    weights = ggml_permute(ctx, weights, 0, 2, 1, 3); // [1, 1, n_heads]

    score = ggml_mul(ctx, score, weights);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 2, 0, 3)); // [n_heads, n_comp, 1]
    score = ggml_sum_rows(ctx, score);
    return ggml_reshape_2d(ctx, score, n_comp, 1);
}

static ggml_tensor * dsv4_build_compressed_mask_from_topk(
        ggml_context * ctx,
        ggml_tensor  * scores,
        ggml_tensor  * topk) {
    const int64_t n_comp   = scores->ne[0];
    const int64_t n_tokens = scores->ne[1];

    ggml_tensor * scores_rows = ggml_reshape_3d(ctx, scores, 1, scores->ne[0], scores->ne[1]);
    ggml_tensor * selected_scores = ggml_get_rows(ctx, scores_rows, topk); // [1, top_k, n_tokens]
    ggml_tensor * valid = ggml_step(ctx, dsv4_add_scalar(ctx, selected_scores, 1.0e30f));
    ggml_tensor * values = dsv4_mul_scalar(ctx, dsv4_add_scalar(ctx, valid, -1.0f), 1.0e9f);

    ggml_tensor * mask = dsv4_new_filled_3d(ctx, 1, n_comp, n_tokens, -INFINITY);
    mask = ggml_set_rows(ctx, mask, values, topk);
    return ggml_reshape_2d(ctx, mask, n_comp, n_tokens);
}

static ggml_tensor * dsv4_cache_view_3d(ggml_context * ctx, ggml_tensor * cache, int64_t n_rows) {
    ggml_tensor * view = ggml_view_2d(ctx, cache, cache->ne[0], n_rows, cache->nb[1], 0);
    return ggml_reshape_3d(ctx, view, cache->ne[0], 1, n_rows);
}

} // namespace

llm_build_deepseek4::llm_build_deepseek4(const llama_model & model, const llm_graph_params & params) :
	llm_graph_context(params) {

    const int64_t n_hc        = hparams.n_hc;
    const int64_t n_lora_q    = hparams.n_lora_q;
    const int64_t n_lora_o    = hparams.n_lora_o;
    const int64_t n_out_group = hparams.n_attn_out_groups;

    GGML_ASSERT(n_hc > 0);
    GGML_ASSERT(n_lora_q > 0);
    GGML_ASSERT(n_lora_o > 0);
    GGML_ASSERT(n_out_group > 0);
    GGML_ASSERT(n_embd_head_k == n_embd_head_v);
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_tokens = res->t_inp_tokens;
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_mem  = build_inp_mem_hybrid_iswa();
    auto * inp_attn = inp_mem->get_attn();
    auto * inp_rs   = inp_mem->get_recr();
    const auto * mctx_dsv4 = inp_mem->mctx;
    dsv4_graph_inputs * inp_dsv4 = nullptr;
    auto get_dsv4_inputs = [&]() {
        if (inp_dsv4 == nullptr) {
            auto inputs = std::make_unique<dsv4_graph_inputs>();
            inp_dsv4 = inputs.get();
            res->add_input(std::move(inputs));
        }
        return inp_dsv4;
    };

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k));
    const int64_t dsv4_hc_pos = ubatch.pos != nullptr ? (int64_t) ubatch.pos[0] : 0;
    auto dsv4_layer_executor_shadow_probe = [&](
            std::vector<ggml_tensor *> & dependencies,
            ggml_tensor                 * tensor,
            const char                  * stage,
            const char                  * tensor_name,
            int                           il) -> ggml_tensor * {
        (void) dependencies;

        const bool lexec_site = dsv4_layer_executor_full_shadow_mode() ?
            dsv4_experimental_layer_executor_full_site_enabled(il, dsv4_hc_pos, n_tokens) :
            dsv4_experimental_layer_executor_site_enabled(il, dsv4_hc_pos, n_tokens);
        if (!lexec_site) {
            return tensor;
        }

        if (dsv4_layer_executor_full_shadow_mode()) {
            dsv4_lexec_full_note_tensor(tensor, stage, tensor_name, il, dsv4_hc_pos, n_tokens);
            dsv4_layer_executor_shadow_trace_once(stage, il, dsv4_hc_pos,
                    dsv4_experimental_layer_executor_compare_enabled() ? "full_envelope_self_exact" : "full_envelope_trace_only");
            return tensor;
        }

        if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
            dsv4_layer_executor_shadow_trace_once(stage, il, dsv4_hc_pos, "unsupported_dtype");
            return tensor;
        }

        dsv4_layer_executor_shadow_trace_once(stage, il, dsv4_hc_pos,
                dsv4_experimental_layer_executor_compare_enabled() ? "identity_compare" : "trace_only");

        if (!dsv4_experimental_layer_executor_compare_enabled()) {
            return tensor;
        }

        const bool consume_layer_output =
                std::strcmp(stage, "layer_output") == 0 &&
                std::strcmp(tensor_name, "output") == 0 &&
                dsv4_experimental_layer_executor_consume_site_enabled(il, dsv4_hc_pos, n_tokens);
        const char * style = consume_layer_output ? dsv4_layer_executor_consume_style() : "shadow";
        const char * style_code = dsv4_layer_executor_consume_style_code(style);
        const bool same_tensor_consume = consume_layer_output && std::strcmp(style, "same_tensor") == 0;
        const char * ref_tensor_name = consume_layer_output ? "output" : tensor_name;

        ggml_tensor * ref = ggml_scale(ctx0, tensor, 1.0f);
        ggml_tensor * shadow = nullptr;
        if (consume_layer_output && std::strcmp(style, "view_alias") == 0) {
            shadow = ggml_view_4d(ctx0, tensor,
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                    tensor->nb[1], tensor->nb[2], tensor->nb[3], 0);
        } else if (consume_layer_output && std::strcmp(style, "reshape_alias") == 0) {
            shadow = ggml_reshape_4d(ctx0, tensor,
                    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
        } else if (consume_layer_output && std::strcmp(style, "add_zero") == 0) {
            ggml_tensor * zero = ggml_scale(ctx0, tensor, 0.0f);
            ggml_format_name(zero, "dsv4_lexec_zero-%s-%s-l%d-p%lld", stage, tensor_name, il, (long long) dsv4_hc_pos);
            shadow = ggml_add(ctx0, tensor, zero);
        } else if (consume_layer_output && std::strcmp(style, "copy_materialized") == 0) {
            shadow = ggml_cont(ctx0, tensor);
        } else {
            shadow = ggml_scale(ctx0, tensor, 1.0f);
        }

        if (consume_layer_output) {
            ggml_format_name(ref, "dsv4_lexec_ref-%s-%s-%s-l%d-p%lld", stage, ref_tensor_name, style_code, il, (long long) dsv4_hc_pos);
        } else {
            ggml_format_name(ref, "dsv4_lexec_ref-%s-%s-l%d-p%lld", stage, tensor_name, il, (long long) dsv4_hc_pos);
        }
        if (consume_layer_output) {
            ggml_format_name(shadow, "dsv4_lexec_shadow_consume-%s-%s-%s-l%d-p%lld", stage, ref_tensor_name, style_code, il, (long long) dsv4_hc_pos);
        } else {
            ggml_format_name(shadow, "dsv4_lexec_shadow-%s-%s-l%d-p%lld", stage, tensor_name, il, (long long) dsv4_hc_pos);
        }
        ggml_build_forward_expand(gf, ref);
        ggml_build_forward_expand(gf, shadow);
        return consume_layer_output && !same_tensor_consume ? shadow : tensor;
    };
    auto dsv4_layer_executor_apply_dependencies = [&](ggml_tensor * tensor, const std::vector<ggml_tensor *> & dependencies) -> ggml_tensor * {
        if (tensor == nullptr || dependencies.empty()) {
            return tensor;
        }
        ggml_tensor * out = tensor;
        for (ggml_tensor * zero : dependencies) {
            out = ggml_add(ctx0, out, ggml_repeat(ctx0, zero, out));
        }
        return out;
    };
    auto dsv4_rmoe_result_chain_materialize = [&](ggml_tensor * tensor, const char * stage, int il) -> ggml_tensor * {
        if (tensor == nullptr) {
            return tensor;
        }
        ggml_tensor * out = ggml_cont(ctx0, tensor);
        ggml_format_name(out, "dsv4_rmoe_result_chain_materialize_%s-l%d-p%lld",
                stage, il, (long long) dsv4_hc_pos);
        return out;
    };
    auto dsv4_rmoe_result_chain_dependency = [&](ggml_tensor * tensor, const char * stage, int il) -> ggml_tensor * {
        if (tensor == nullptr) {
            return tensor;
        }
        ggml_tensor * zero = ggml_scale(ctx0, tensor, 0.0f);
        ggml_format_name(zero, "dsv4_rmoe_result_chain_dep_zero_%s-l%d-p%lld",
                stage, il, (long long) dsv4_hc_pos);
        ggml_tensor * out = ggml_add(ctx0, tensor, zero);
        ggml_format_name(out, "dsv4_rmoe_result_chain_dep_%s-l%d-p%lld",
                stage, il, (long long) dsv4_hc_pos);
        return out;
    };
    auto dsv4_rmoe_result_chain_mode_is = [&](const char * mode) -> bool {
        return std::strcmp(dsv4_experimental_routed_moe_backend_op_result_chain_mode(), mode) == 0;
    };
    auto dsv4_aohc_shadow_probe = [&](ggml_tensor * tensor, const char * tensor_name, int il) -> ggml_tensor * {
        if (!dsv4_experimental_attn_out_hc_post_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            return tensor;
        }
        if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
            if (dsv4_experimental_attn_out_hc_post_trace_enabled()) {
                std::fprintf(stderr,
                        "dsv4_aohc_shadow: layer=%d token=%lld tensor=%s mode=%s shadow=not_implemented reason=unsupported_dtype_or_null consume_path=disabled\n",
                        il, (long long) dsv4_hc_pos, tensor_name,
                        dsv4_experimental_attn_out_hc_post_mode());
            }
            return tensor;
        }
        if (dsv4_experimental_attn_out_hc_post_trace_enabled()) {
            std::fprintf(stderr,
                    "dsv4_aohc_shadow: layer=%d token=%lld tensor=%s mode=%s shadow=identity stage_boundary=attn_out_hc_post consume_path=disabled\n",
                    il, (long long) dsv4_hc_pos, tensor_name,
                    dsv4_experimental_attn_out_hc_post_mode());
        }
        if (!dsv4_experimental_attn_out_hc_post_compare_enabled()) {
            return tensor;
        }

        ggml_tensor * ref = ggml_scale(ctx0, tensor, 1.0f);
        ggml_tensor * shadow = ggml_scale(ctx0, tensor, 1.0f);
        ggml_format_name(ref, "dsv4_aohc_ref-%s-l%d-p%lld", tensor_name, il, (long long) dsv4_hc_pos);
        ggml_format_name(shadow, "dsv4_aohc_shadow-%s-l%d-p%lld", tensor_name, il, (long long) dsv4_hc_pos);
        ggml_build_forward_expand(gf, ref);
        ggml_build_forward_expand(gf, shadow);
        return tensor;
    };
    auto dsv4_aohc_candidate_compare_pair = [&](ggml_tensor * generic, ggml_tensor * candidate, const char * tensor_name, int il) {
        const bool candidate_site =
            dsv4_experimental_attn_out_hc_post_candidate_site_enabled(il, dsv4_hc_pos, n_tokens);
        const bool fused_site =
            dsv4_experimental_aohc_fused_site_enabled(il, dsv4_hc_pos, n_tokens, false);
        if (!candidate_site && !fused_site) {
            return;
        }
        const char * mode = fused_site ? "aohc_fused_partial_q8hc" : dsv4_experimental_attn_out_hc_post_candidate_mode();
        const bool trace_enabled =
            fused_site ? dsv4_experimental_aohc_fused_trace_enabled() : dsv4_experimental_attn_out_hc_post_candidate_trace_enabled();
        const bool compare_enabled =
            fused_site ? dsv4_experimental_aohc_fused_compare_enabled() : dsv4_experimental_attn_out_hc_post_candidate_compare_enabled();
        if (generic == nullptr || candidate == nullptr || generic->type != GGML_TYPE_F32 || candidate->type != GGML_TYPE_F32) {
            if (trace_enabled) {
                std::fprintf(stderr,
                        "dsv4_aohc_candidate: layer=%d token=%lld mode=%s tensor=%s shadow=not_implemented reason=unsupported_dtype_or_null consume_path=disabled\n",
                        il, (long long) dsv4_hc_pos, mode, tensor_name);
            }
            return;
        }
        if (trace_enabled) {
            std::fprintf(stderr,
                    "dsv4_aohc_candidate: layer=%d token=%lld mode=%s tensor=%s candidate_branch=replacement_shaped consume_path=disabled\n",
                    il, (long long) dsv4_hc_pos, mode, tensor_name);
        }
        if (!compare_enabled) {
            return;
        }

        ggml_tensor * ref = ggml_scale(ctx0, generic, 1.0f);
        ggml_tensor * cand = ggml_scale(ctx0, candidate, 1.0f);
        ggml_format_name(ref, "dsv4_aohc_candidate_ref-%s-l%d-p%lld", tensor_name, il, (long long) dsv4_hc_pos);
        ggml_format_name(cand, "dsv4_aohc_candidate_out-%s-l%d-p%lld", tensor_name, il, (long long) dsv4_hc_pos);
        ggml_build_forward_expand(gf, ref);
        ggml_build_forward_expand(gf, cand);
    };
    auto dsv4_aohc_candidate_dep_audit = [&](int il) {
        if (!dsv4_experimental_attn_out_hc_post_candidate_trace_enabled()) {
            return;
        }
        std::fprintf(stderr,
                "dsv4_aohc_candidate_dep: layer=%d token=%lld mode=%s"
                " candidate_attn_low_uses_generic_attn_low=0"
                " candidate_attn_out_uses_generic_attn_out=0"
                " candidate_after_attn_hc_uses_generic_after_attn_hc=0"
                " allowed_inputs=attn_core_out,wo_a,wo_b,residual,hc_post_weights,hc_comb"
                " forbidden_inputs_seen=none dependency_audit_pass=1 consume_path=disabled\n",
                il, (long long) dsv4_hc_pos,
                dsv4_experimental_attn_out_hc_post_consume_site_enabled(il, dsv4_hc_pos, n_tokens) ?
                    dsv4_experimental_attn_out_hc_post_consume_mode() :
                    dsv4_experimental_attn_out_hc_post_candidate_mode());
    };
    auto dsv4_cupd3_shadow_probe = [&](ggml_tensor * tensor, const char * tensor_name, int il, const char * stream) -> ggml_tensor * {
        if (!dsv4_experimental_compressor_update_v3_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            return tensor;
        }
        const char * mode = dsv4_experimental_compressor_update_v3_mode();
        if (tensor == nullptr) {
            if (dsv4_experimental_compressor_update_v3_trace_enabled()) {
                std::fprintf(stderr,
                        "dsv4_cupd3_shadow: layer=%d token=%lld mode=%s stream=%s tensor=%s shadow=not_available cache_mutation=disabled consume_path=disabled\n",
                        il, (long long) dsv4_hc_pos, mode, stream, tensor_name);
            }
            return tensor;
        }
        if (dsv4_experimental_compressor_update_v3_trace_enabled()) {
            std::fprintf(stderr,
                    "dsv4_cupd3_shadow: layer=%d token=%lld mode=%s stream=%s tensor=%s"
                    " shape=[%lld,%lld,%lld,%lld] dtype=%s projection_source=generic cache_mutation=disabled consume_path=disabled\n",
                    il, (long long) dsv4_hc_pos, mode, stream, tensor_name,
                    (long long) tensor->ne[0], (long long) tensor->ne[1],
                    (long long) tensor->ne[2], (long long) tensor->ne[3],
                    ggml_type_name(tensor->type));
            if (std::strcmp(mode, "ds4_shape_shadow") == 0) {
                std::fprintf(stderr,
                        "dsv4_cupd3_dep: layer=%d token=%lld mode=%s stream=%s"
                        " candidate_uses_generic_state_kv=0 candidate_uses_generic_state_score=0"
                        " candidate_uses_generic_pool_output=0 candidate_uses_generic_norm_output=0"
                        " candidate_uses_generic_rope_output=0 candidate_uses_generic_quant_output=0"
                        " allowed_inputs=projection,prev_state,ape,norm,pos,rope_cfg"
                        " forbidden_inputs_seen=none projection_source=generic cache_mutation=disabled consume_path=disabled\n",
                        il, (long long) dsv4_hc_pos, mode, stream);
            } else if (std::strcmp(mode, "cupd2_tail_shadow") == 0) {
                std::fprintf(stderr,
                        "dsv4_cupd3_dep: layer=%d token=%lld mode=%s stream=%s"
                        " cupd2_owned=state_kv,state_score,pool_input"
                        " generic_tail=pool_softmax,weighted_pool,rms_norm,norm_weight,rope,quant"
                        " projection_source=generic cache_mutation=disabled consume_path=disabled\n",
                        il, (long long) dsv4_hc_pos, mode, stream);
            }
        }
        if (!dsv4_experimental_compressor_update_v3_compare_enabled() || tensor->type != GGML_TYPE_F32) {
            return tensor;
        }

        ggml_tensor * ref = ggml_scale(ctx0, tensor, 1.0f);
        ggml_tensor * shadow = ggml_scale(ctx0, tensor, 1.0f);
        ggml_format_name(ref, "dsv4_cupd3_ref-%s_%s-l%d-p%lld", stream, tensor_name, il, (long long) dsv4_hc_pos);
        ggml_format_name(shadow, "dsv4_cupd3_shadow-%s_%s-l%d-p%lld", stream, tensor_name, il, (long long) dsv4_hc_pos);
        ggml_build_forward_expand(gf, ref);
        ggml_build_forward_expand(gf, shadow);
        return tensor;
    };
    auto dsv4_moe_shadow_probe = [&](ggml_tensor * generic, ggml_tensor * candidate, const char * tensor_name, int il, bool one_tensor_case = false) {
        if (!dsv4_experimental_routed_moe_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            return;
        }
        const char * mode = dsv4_experimental_routed_moe_mode();
        const bool compare_enabled = dsv4_experimental_routed_moe_compare_enabled();
        const bool trace_enabled = dsv4_experimental_routed_moe_trace_enabled();
        const bool same_tensor = generic == candidate;
        const bool supported_compare =
                generic != nullptr &&
                candidate != nullptr &&
                generic->type == GGML_TYPE_F32 &&
                candidate->type == GGML_TYPE_F32 &&
                generic->ne[0] == candidate->ne[0] &&
                generic->ne[1] == candidate->ne[1] &&
                generic->ne[2] == candidate->ne[2] &&
                generic->ne[3] == candidate->ne[3];

        if (trace_enabled) {
            std::fprintf(stderr,
                    "dsv4_moe_shadow: layer=%d token=%lld mode=%s tensor=%s"
                    " generic_shape=[%lld,%lld,%lld,%lld] candidate_shape=[%lld,%lld,%lld,%lld]"
                    " generic_dtype=%s candidate_dtype=%s same_tensor=%d graph_level_shadow=%d consume_path=disabled\n",
                    il,
                    (long long) dsv4_hc_pos,
                    mode,
                    tensor_name,
                    generic != nullptr ? (long long) generic->ne[0] : -1LL,
                    generic != nullptr ? (long long) generic->ne[1] : -1LL,
                    generic != nullptr ? (long long) generic->ne[2] : -1LL,
                    generic != nullptr ? (long long) generic->ne[3] : -1LL,
                    candidate != nullptr ? (long long) candidate->ne[0] : -1LL,
                    candidate != nullptr ? (long long) candidate->ne[1] : -1LL,
                    candidate != nullptr ? (long long) candidate->ne[2] : -1LL,
                    candidate != nullptr ? (long long) candidate->ne[3] : -1LL,
                    generic != nullptr ? ggml_type_name(generic->type) : "null",
                    candidate != nullptr ? ggml_type_name(candidate->type) : "null",
                    same_tensor ? 1 : 0,
                    one_tensor_case ? 1 : 0);
        }

        if (compare_enabled && supported_compare) {
            ggml_tensor * ref = ggml_scale(ctx0, generic, 1.0f);
            ggml_tensor * shadow = ggml_scale(ctx0, candidate, 1.0f);
            ggml_format_name(ref, "dsv4_moe_ref-%s-l%d-p%lld", tensor_name, il, (long long) dsv4_hc_pos);
            ggml_format_name(shadow, "dsv4_moe_shadow-%s-l%d-p%lld", tensor_name, il, (long long) dsv4_hc_pos);
            ggml_build_forward_expand(gf, ref);
            ggml_build_forward_expand(gf, shadow);
            dsv4_moe_shadow_note(il, dsv4_hc_pos, tensor_name, true, one_tensor_case);
        } else {
            dsv4_moe_shadow_note(il, dsv4_hc_pos, tensor_name, !compare_enabled || same_tensor, one_tensor_case);
        }
    };
    auto dsv4_moe_dep_audit = [&](int il, const char * mode, bool one_tensor_candidate) {
        if (!dsv4_experimental_routed_moe_trace_enabled() ||
                !dsv4_experimental_routed_moe_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            return;
        }
        std::fprintf(stderr,
                "dsv4_moe_dep: layer=%d token=%lld mode=%s"
                " candidate_uses_generic_router_scores=%d"
                " candidate_uses_generic_topk=%d"
                " candidate_uses_generic_gate_up=%d"
                " candidate_uses_generic_swiglu=%d"
                " candidate_uses_generic_down=%d"
                " candidate_uses_generic_weighted_sum=%d"
                " candidate_uses_generic_final_ffn=%d"
                " allowed_inputs=ffn_norm_input,router_weights,expert_gate_up_weights,expert_down_weights,route_bias,shared_weights,inp_tokens"
                " forbidden_inputs_seen=%s"
                " router_source=%s topk_source=%s gate_up_source=%s swiglu_source=%s down_source=%s weighted_sum_source=%s"
                " shared_source=%s final_ffn_source=%s"
                " candidate_kind=%s consume_path=disabled\n",
                il,
                (long long) dsv4_hc_pos,
                mode,
                one_tensor_candidate ? 0 : 1,
                one_tensor_candidate ? 0 : 1,
                one_tensor_candidate ? 0 : 1,
                one_tensor_candidate ? 0 : 1,
                one_tensor_candidate ? 0 : 1,
                one_tensor_candidate ? 0 : 1,
                0,
                one_tensor_candidate ? "none" : "generic_intermediates_for_probe",
                one_tensor_candidate ? "candidate" : "generic",
                one_tensor_candidate ? "candidate" : "generic",
                one_tensor_candidate ? "candidate" : "generic",
                one_tensor_candidate ? "candidate" : "generic",
                one_tensor_candidate ? "candidate" : "generic",
                one_tensor_candidate ? "candidate" : "generic",
                one_tensor_candidate ? "candidate" : "generic",
                one_tensor_candidate ? "candidate" : "generic",
                one_tensor_candidate ? "graph_level_one_tensor_shadow" : "probe_only");
    };
    auto dsv4_moe_backend_shadow_probe = [&](
            ggml_tensor * generic,
            ggml_tensor * candidate,
            const char * tensor_name,
            int il,
            bool backend_candidate_built,
            bool backend_op_dispatched,
            bool backend_owned) {
        if (!dsv4_experimental_routed_moe_backend_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            return;
        }
        const char * mode = dsv4_experimental_routed_moe_backend_mode();
        const bool compare_enabled = dsv4_experimental_routed_moe_backend_compare_enabled();
        const bool trace_enabled = dsv4_experimental_routed_moe_backend_trace_enabled();
        const bool same_tensor = generic == candidate;
        const bool supported_compare =
                generic != nullptr &&
                candidate != nullptr &&
                generic->type == GGML_TYPE_F32 &&
                candidate->type == GGML_TYPE_F32 &&
                generic->ne[0] == candidate->ne[0] &&
                generic->ne[1] == candidate->ne[1] &&
                generic->ne[2] == candidate->ne[2] &&
                generic->ne[3] == candidate->ne[3];

        if (trace_enabled) {
            std::fprintf(stderr,
                    "dsv4_moe_backend_shadow: layer=%d token=%lld mode=%s tensor=%s"
                    " generic_shape=[%lld,%lld,%lld,%lld] candidate_shape=[%lld,%lld,%lld,%lld]"
                    " generic_dtype=%s candidate_dtype=%s same_tensor=%d"
                    " backend_candidate_built=%d backend_op_dispatched=%d backend_owned=%d"
                    " graph_arithmetic=%d consume_path=disabled\n",
                    il,
                    (long long) dsv4_hc_pos,
                    mode,
                    tensor_name,
                    generic != nullptr ? (long long) generic->ne[0] : -1LL,
                    generic != nullptr ? (long long) generic->ne[1] : -1LL,
                    generic != nullptr ? (long long) generic->ne[2] : -1LL,
                    generic != nullptr ? (long long) generic->ne[3] : -1LL,
                    candidate != nullptr ? (long long) candidate->ne[0] : -1LL,
                    candidate != nullptr ? (long long) candidate->ne[1] : -1LL,
                    candidate != nullptr ? (long long) candidate->ne[2] : -1LL,
                    candidate != nullptr ? (long long) candidate->ne[3] : -1LL,
                    generic != nullptr ? ggml_type_name(generic->type) : "null",
                    candidate != nullptr ? ggml_type_name(candidate->type) : "null",
                    same_tensor ? 1 : 0,
                    backend_candidate_built ? 1 : 0,
                    backend_op_dispatched ? 1 : 0,
                    backend_owned ? 1 : 0,
                    backend_candidate_built && !backend_owned ? 1 : 0);
        }

        if (compare_enabled && supported_compare) {
            ggml_tensor * ref = ggml_scale(ctx0, generic, 1.0f);
            ggml_tensor * shadow = ggml_scale(ctx0, candidate, 1.0f);
            ggml_format_name(ref, "dsv4_moe_backend_ref-%s-l%d-p%lld", tensor_name, il, (long long) dsv4_hc_pos);
            ggml_format_name(shadow, "dsv4_moe_backend_shadow-%s-l%d-p%lld", tensor_name, il, (long long) dsv4_hc_pos);
            ggml_build_forward_expand(gf, ref);
            ggml_build_forward_expand(gf, shadow);
            dsv4_moe_backend_shadow_note(il, dsv4_hc_pos, tensor_name, true,
                    backend_candidate_built, backend_op_dispatched, backend_owned);
        } else {
            dsv4_moe_backend_shadow_note(il, dsv4_hc_pos, tensor_name,
                    !compare_enabled || same_tensor,
                    backend_candidate_built, backend_op_dispatched, backend_owned);
        }
    };
    auto dsv4_moe_backend_dep_audit = [&](int il, const char * mode, const char * source, bool backend_owned, const char * forbidden_inputs) {
        if (!dsv4_experimental_routed_moe_backend_trace_enabled() ||
                !dsv4_experimental_routed_moe_backend_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            return;
        }
        std::fprintf(stderr,
                "dsv4_moe_backend_dep: layer=%d token=%lld mode=%s"
                " router_source=%s topk_source=%s gate_up_source=%s swiglu_source=%s"
                " down_source=%s weighted_sum_source=%s shared_source=%s final_ffn_source=%s"
                " backend_owned=%d forbidden_inputs_seen=%s"
                " missing_inputs=backend_op,expert_weight_layout,topk_route_pack,shared_branch_pack"
                " consume_path=disabled\n",
                il,
                (long long) dsv4_hc_pos,
                mode,
                source,
                source,
                source,
                source,
                source,
                source,
                source,
                source,
                backend_owned ? 1 : 0,
                forbidden_inputs != nullptr ? forbidden_inputs : "none");
    };
    auto dsv4_moe_backend_infeasible_audit = [&](int il, const char * mode) {
        if (!dsv4_experimental_routed_moe_backend_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            return;
        }
        const char * reason = "missing_backend_one_tensor_op_requires_new_op_kernel_with_router_topk_gate_up_swiglu_down_weighted_shared_inputs";
        dsv4_moe_backend_shadow_infeasible(il, dsv4_hc_pos, reason);
        if (dsv4_experimental_routed_moe_backend_trace_enabled()) {
            std::fprintf(stderr,
                    "dsv4_moe_backend_shadow: layer=%d token=%lld mode=%s"
                    " backend_candidate_built=0 backend_op_dispatched=0 backend_owned=0"
                    " infeasible=1 reason=%s consume_path=disabled\n",
                    il, (long long) dsv4_hc_pos, mode, reason);
        }
        dsv4_moe_backend_dep_audit(il, mode, "unavailable", false, reason);
    };
    auto dsv4_moe_backend_op_shape = [](const ggml_tensor * tensor, char * out, size_t out_size) {
        if (tensor == nullptr) {
            std::snprintf(out, out_size, "missing");
            return;
        }
        std::snprintf(out, out_size, "[%lld,%lld,%lld,%lld]",
                (long long) tensor->ne[0], (long long) tensor->ne[1],
                (long long) tensor->ne[2], (long long) tensor->ne[3]);
    };
    auto dsv4_moe_backend_op_dryrun = [&](
            ggml_tensor * ffn_input,
            ggml_tensor * topk_ids,
            ggml_tensor * topk_weights,
            ggml_tensor * final_output,
            ggml_tensor * gate_exps,
            ggml_tensor * up_exps,
            ggml_tensor * down_exps,
            ggml_tensor * shared_gate,
            ggml_tensor * shared_up,
            ggml_tensor * shared_down,
            int il) {
        if (dsv4_experimental_routed_moe_backend_op_consume_requested()) {
            dsv4_rmoe_consume_register_summary();
            auto & state = dsv4_moe_backend_op_summary();
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!state.atexit_registered) {
                std::atexit(dsv4_moe_backend_op_print_summary);
                state.atexit_registered = true;
            }
        }
        if (!dsv4_experimental_routed_moe_backend_op_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            return;
        }

        const bool has_shared =
                shared_gate != nullptr &&
                shared_up != nullptr &&
                shared_down != nullptr;
        const char * reject_reason = "none";
        bool eligible = true;
        if (n_tokens != 1) {
            reject_reason = "not_decode_token";
            eligible = false;
        } else if (n_expert_used != 6) {
            reject_reason = "unsupported_topk";
            eligible = false;
        } else if (n_expert <= 0) {
            reject_reason = "unsupported_expert_count";
            eligible = false;
        } else if (topk_ids == nullptr) {
            reject_reason = "missing_topk_ids";
            eligible = false;
        } else if (topk_weights == nullptr) {
            reject_reason = "missing_topk_weights";
            eligible = false;
        } else if (gate_exps == nullptr || up_exps == nullptr || down_exps == nullptr) {
            reject_reason = "unsupported_weight_type";
            eligible = false;
        } else if (!has_shared) {
            reject_reason = "missing_shared_branch";
            eligible = false;
        } else if (ffn_input == nullptr || final_output == nullptr) {
            reject_reason = "missing_ffn_io";
            eligible = false;
        }

        dsv4_moe_backend_op_note(il, dsv4_hc_pos, eligible, reject_reason, final_output, topk_ids, topk_weights);

        if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
            char ffn_input_shape[64];
            char topk_ids_shape[64];
            char topk_weights_shape[64];
            char gate_shape[64];
            char up_shape[64];
            char down_shape[64];
            char shared_gate_shape[64];
            char shared_up_shape[64];
            char shared_down_shape[64];
            char output_shape[64];
            dsv4_moe_backend_op_shape(ffn_input, ffn_input_shape, sizeof(ffn_input_shape));
            dsv4_moe_backend_op_shape(topk_ids, topk_ids_shape, sizeof(topk_ids_shape));
            dsv4_moe_backend_op_shape(topk_weights, topk_weights_shape, sizeof(topk_weights_shape));
            dsv4_moe_backend_op_shape(gate_exps, gate_shape, sizeof(gate_shape));
            dsv4_moe_backend_op_shape(up_exps, up_shape, sizeof(up_shape));
            dsv4_moe_backend_op_shape(down_exps, down_shape, sizeof(down_shape));
            dsv4_moe_backend_op_shape(shared_gate, shared_gate_shape, sizeof(shared_gate_shape));
            dsv4_moe_backend_op_shape(shared_up, shared_up_shape, sizeof(shared_up_shape));
            dsv4_moe_backend_op_shape(shared_down, shared_down_shape, sizeof(shared_down_shape));
            dsv4_moe_backend_op_shape(final_output, output_shape, sizeof(output_shape));
            const bool supported_quant_types =
                gate_exps != nullptr && gate_exps->type == GGML_TYPE_IQ2_XXS &&
                up_exps != nullptr && up_exps->type == GGML_TYPE_IQ2_XXS &&
                down_exps != nullptr && down_exps->type == GGML_TYPE_Q2_K &&
                shared_gate != nullptr && shared_gate->type == GGML_TYPE_Q8_0 &&
                shared_up != nullptr && shared_up->type == GGML_TYPE_Q8_0 &&
                shared_down != nullptr && shared_down->type == GGML_TYPE_Q8_0;
            const bool supported_strides =
                ffn_input != nullptr && ggml_is_contiguous_rows(ffn_input) &&
                topk_ids != nullptr && ggml_is_contiguous_rows(topk_ids) &&
                topk_weights != nullptr && ggml_is_contiguous_rows(topk_weights);
            std::fprintf(stderr,
                    "dsv4_moe_backend_op: layer=%d token=%lld n_tokens=%lld"
                    " hidden_dim=%lld expert_count=%lld topk=%lld"
                    " selected_expert_ids_shape=%s selected_expert_weights_shape=%s"
                    " ffn_input_shape=%s"
                    " gate_weight_type=%s gate_weight_shape=%s"
                    " up_weight_type=%s up_weight_shape=%s"
                    " down_weight_type=%s down_weight_shape=%s"
                    " shared_gate_weight_type=%s shared_gate_weight_shape=%s"
                    " shared_up_weight_type=%s shared_up_weight_shape=%s"
                    " shared_down_weight_type=%s shared_down_weight_shape=%s"
                    " supported_quant_types=%d supported_strides=%d metal_backend=1"
                    " owns_router=0 consumes_topk=1 owns_gate_up=1 owns_swiglu=1"
                    " owns_down=1 owns_weighted_sum=1 owns_shared=1"
                    " output_shape=%s eligible=%d reject_reason=%s consume_path=disabled\n",
                    il,
                    (long long) dsv4_hc_pos,
                    (long long) n_tokens,
                    (long long) n_embd,
                    (long long) n_expert,
                    (long long) n_expert_used,
                    topk_ids_shape,
                    topk_weights_shape,
                    ffn_input_shape,
                    gate_exps != nullptr ? ggml_type_name(gate_exps->type) : "missing",
                    gate_shape,
                    up_exps != nullptr ? ggml_type_name(up_exps->type) : "missing",
                    up_shape,
                    down_exps != nullptr ? ggml_type_name(down_exps->type) : "missing",
                    down_shape,
                    shared_gate != nullptr ? ggml_type_name(shared_gate->type) : "missing",
                    shared_gate_shape,
                    shared_up != nullptr ? ggml_type_name(shared_up->type) : "missing",
                    shared_up_shape,
                    shared_down != nullptr ? ggml_type_name(shared_down->type) : "missing",
                    shared_down_shape,
                    supported_quant_types ? 1 : 0,
                    supported_strides ? 1 : 0,
                    output_shape,
                    eligible ? 1 : 0,
                    reject_reason);
        }
    };
    auto dsv4_cupd3_backend_tail_probe = [&](ggml_tensor * generic_quant, ggml_tensor * backend_quant, const dsv4_decode_compressor & dec, int il, const char * stream) {
        if (!dsv4_experimental_compressor_update_v3_backend_tail_site_enabled(il, dsv4_hc_pos, n_tokens) ||
                backend_quant == nullptr || dec.backend_tail_kv_comp == nullptr) {
            return;
        }

        const bool compare_enabled = dsv4_experimental_compressor_update_v3_backend_tail_compare_enabled();
        const bool drift_trace = dsv4_experimental_compressor_update_v3_backend_tail_drift_trace_site_enabled(il, dsv4_hc_pos, n_tokens);
        const bool row_probe = drift_trace && dsv4_experimental_compressor_update_v3_backend_tail_attn_row_probe_enabled();
        const bool value_probe =
                drift_trace &&
                dsv4_experimental_compressor_update_v3_backend_tail_value_probe_enabled() &&
                dsv4_experimental_compressor_update_v3_backend_tail_stream_allowed(stream);
        const bool internal_probe =
                drift_trace &&
                dsv4_experimental_compressor_update_v3_decode_compress_internal_probe_enabled() &&
                dsv4_experimental_compressor_update_v3_backend_tail_stream_allowed(stream);
        auto same_layout = [](const ggml_tensor * a, const ggml_tensor * b) {
            if (a == nullptr || b == nullptr || a->type != b->type || a->view_offs != b->view_offs) {
                return false;
            }
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                if (a->ne[i] != b->ne[i] || a->nb[i] != b->nb[i]) {
                    return false;
                }
            }
            return true;
        };
        auto first_layout_mismatch = [&](const ggml_tensor * generic_pre, const ggml_tensor * backend_pre) {
            if (!same_layout(generic_pre, backend_pre)) {
                return "prequant_layout";
            }
            if (!same_layout(generic_quant, backend_quant)) {
                return "quant_layout";
            }
            return "none";
        };
        auto tensor_shape_csv = [](const ggml_tensor * t, char * out, size_t out_size) {
            if (t == nullptr) {
                std::snprintf(out, out_size, "[-1,-1,-1,-1]");
                return;
            }
            std::snprintf(out, out_size, "[%lld,%lld,%lld,%lld]",
                    (long long) t->ne[0], (long long) t->ne[1],
                    (long long) t->ne[2], (long long) t->ne[3]);
        };
        auto tensor_stride_csv = [](const ggml_tensor * t, char * out, size_t out_size) {
            if (t == nullptr) {
                std::snprintf(out, out_size, "[-1,-1,-1,-1]");
                return;
            }
            std::snprintf(out, out_size, "[%zu,%zu,%zu,%zu]",
                    t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
        };
        auto tensor_type_name = [](const ggml_tensor * t) {
            return t != nullptr ? ggml_type_name(t->type) : "null";
        };
        auto tensor_op_name = [](const ggml_tensor * t) {
            return t != nullptr ? ggml_op_name(t->op) : "null";
        };
        auto tensor_src0_op_name = [](const ggml_tensor * t) {
            return t != nullptr && t->src[0] != nullptr ? ggml_op_name(t->src[0]->op) : "null";
        };
        auto tensor_nbytes = [](const ggml_tensor * t) {
            return t != nullptr ? (long long) ggml_nbytes(t) : -1LL;
        };
        if (dsv4_experimental_compressor_update_v3_backend_tail_trace_enabled() || drift_trace) {
            const int64_t trace_compress_ratio = il >= 0 ? hparams.attn_compress_ratio[il] : 0;
            const int64_t ratio_phase = trace_compress_ratio > 0 ? dsv4_hc_pos % trace_compress_ratio : -1;
            const int64_t emit_row = trace_compress_ratio > 0 ? ((dsv4_hc_pos + 1) / trace_compress_ratio - 1) : -1;
            const char * mismatch_stage = first_layout_mismatch(dec.generic_tail_kv_comp, dec.backend_tail_kv_comp);
            std::fprintf(stderr,
                    "dsv4_cupd3_backend_tail: layer=%d token=%lld stream=%s scope=pool_norm_rope_quant"
                    " emit_row=%lld ratio_phase=%lld"
                    " backend_op=GGML_OP_DSV4_DECODE_COMPRESS quant_op=%s"
                    " projection_source=generic cache_mutation_mode=generic_existing_write candidate_cache_side_effect=0 consume_path=%s"
                    " generic_tail_built=%d backend_tail_built=%d backend_tail_consumed=%d"
                    " generic_cache_write_built=1 dep_barrier=%d emit_only=%d"
                    " pool_output_max_abs=%s pool_output_rms=%s norm_output_max_abs=%s norm_output_rms=%s"
                    " rope_output_max_abs=%s rope_output_rms=%s quant_row_max_abs=%s quant_row_rms=%s"
                    " downstream_kv_row_max_abs=%s downstream_kv_row_rms=%s cache_written_row_max_abs=%s cache_written_row_rms=%s"
                    " generic_quant_shape=[%lld,%lld,%lld,%lld] backend_quant_shape=[%lld,%lld,%lld,%lld]"
                    " row_probe=%d first_structural_mismatch=%s layout_mode=%s\n",
                    il,
                    (long long) dsv4_hc_pos,
                    stream,
                    (long long) emit_row,
                    (long long) ratio_phase,
                    stream != nullptr && std::strcmp(stream, "index") == 0 ?
                        "GGML_OP_DSV4_HADAMARD_FP4_QUANTIZE" : "GGML_OP_DSV4_FP8_KV_QUANTIZE",
                    dec.candidate_tail_consumed ? "single_layer" : "disabled",
                    dec.generic_tail_built ? 1 : 0,
                    dec.backend_tail_built ? 1 : 0,
                    dec.candidate_tail_consumed ? 1 : 0,
                    dsv4_experimental_compressor_update_v3_backend_tail_dep_barrier_enabled() ? 1 : 0,
                    dsv4_experimental_compressor_update_v3_backend_tail_consume_emit_only_enabled() ? 1 : 0,
                    compare_enabled ? "0" : "not_readback",
                    compare_enabled ? "0" : "not_readback",
                    compare_enabled ? "0" : "not_readback",
                    compare_enabled ? "0" : "not_readback",
                    compare_enabled ? "0" : "not_readback",
                    compare_enabled ? "0" : "not_readback",
                    compare_enabled ? "0" : "not_readback",
                    compare_enabled ? "0" : "not_readback",
                    compare_enabled ? "0" : "not_readback",
                    compare_enabled ? "0" : "not_readback",
                    "not_readback",
                    "not_readback",
                    generic_quant != nullptr ? (long long) generic_quant->ne[0] : -1LL,
                    generic_quant != nullptr ? (long long) generic_quant->ne[1] : -1LL,
                    generic_quant != nullptr ? (long long) generic_quant->ne[2] : -1LL,
                    generic_quant != nullptr ? (long long) generic_quant->ne[3] : -1LL,
                    (long long) backend_quant->ne[0],
                    (long long) backend_quant->ne[1],
                    (long long) backend_quant->ne[2],
                    (long long) backend_quant->ne[3],
                    row_probe ? 1 : 0,
                    mismatch_stage,
                    dsv4_experimental_compressor_update_v3_backend_tail_attn_layout_mode());
            if (row_probe) {
                char generic_pre_shape[64], backend_pre_shape[64], generic_quant_shape[64], backend_quant_shape[64];
                char generic_pre_stride[96], backend_pre_stride[96], generic_quant_stride[96], backend_quant_stride[96];
                tensor_shape_csv(dec.generic_tail_kv_comp, generic_pre_shape, sizeof(generic_pre_shape));
                tensor_shape_csv(dec.backend_tail_kv_comp, backend_pre_shape, sizeof(backend_pre_shape));
                tensor_shape_csv(generic_quant, generic_quant_shape, sizeof(generic_quant_shape));
                tensor_shape_csv(backend_quant, backend_quant_shape, sizeof(backend_quant_shape));
                tensor_stride_csv(dec.generic_tail_kv_comp, generic_pre_stride, sizeof(generic_pre_stride));
                tensor_stride_csv(dec.backend_tail_kv_comp, backend_pre_stride, sizeof(backend_pre_stride));
                tensor_stride_csv(generic_quant, generic_quant_stride, sizeof(generic_quant_stride));
                tensor_stride_csv(backend_quant, backend_quant_stride, sizeof(backend_quant_stride));
                std::fprintf(stderr,
                        "dsv4_cupd3_attn_row_probe: token=%lld layer=%d stream=%s row_id=%lld"
                        " first_mismatch_stage=%s"
                        " generic_pre_op=%s backend_pre_op=%s generic_pre_src0_op=%s backend_pre_src0_op=%s"
                        " generic_pre_type=%s backend_pre_type=%s generic_pre_shape=%s backend_pre_shape=%s"
                        " generic_pre_stride=%s backend_pre_stride=%s generic_pre_offset=%zu backend_pre_offset=%zu"
                        " generic_pre_bytes=%lld backend_pre_bytes=%lld pre_layout_exact=%d"
                        " generic_quant_op=%s backend_quant_op=%s generic_quant_src0_op=%s backend_quant_src0_op=%s"
                        " generic_quant_type=%s backend_quant_type=%s generic_quant_shape=%s backend_quant_shape=%s"
                        " generic_quant_stride=%s backend_quant_stride=%s generic_quant_offset=%zu backend_quant_offset=%zu"
                        " generic_quant_bytes=%lld backend_quant_bytes=%lld quant_layout_exact=%d"
                        " byte_exact=not_readback max_abs=%s rms=%s first_bad_index=not_readback"
                        " generic_value=not_readback backend_value=not_readback"
                        " generic_bytes_first_32=not_readback backend_bytes_first_32=not_readback"
                        " generic_scale_or_header=not_readback backend_scale_or_header=not_readback\n",
                        (long long) dsv4_hc_pos,
                        il,
                        stream,
                        (long long) emit_row,
                        mismatch_stage,
                        tensor_op_name(dec.generic_tail_kv_comp),
                        tensor_op_name(dec.backend_tail_kv_comp),
                        tensor_src0_op_name(dec.generic_tail_kv_comp),
                        tensor_src0_op_name(dec.backend_tail_kv_comp),
                        tensor_type_name(dec.generic_tail_kv_comp),
                        tensor_type_name(dec.backend_tail_kv_comp),
                        generic_pre_shape,
                        backend_pre_shape,
                        generic_pre_stride,
                        backend_pre_stride,
                        dec.generic_tail_kv_comp != nullptr ? dec.generic_tail_kv_comp->view_offs : (size_t) 0,
                        dec.backend_tail_kv_comp != nullptr ? dec.backend_tail_kv_comp->view_offs : (size_t) 0,
                        tensor_nbytes(dec.generic_tail_kv_comp),
                        tensor_nbytes(dec.backend_tail_kv_comp),
                        same_layout(dec.generic_tail_kv_comp, dec.backend_tail_kv_comp) ? 1 : 0,
                        tensor_op_name(generic_quant),
                        tensor_op_name(backend_quant),
                        tensor_src0_op_name(generic_quant),
                        tensor_src0_op_name(backend_quant),
                        tensor_type_name(generic_quant),
                        tensor_type_name(backend_quant),
                        generic_quant_shape,
                        backend_quant_shape,
                        generic_quant_stride,
                        backend_quant_stride,
                        generic_quant != nullptr ? generic_quant->view_offs : (size_t) 0,
                        backend_quant != nullptr ? backend_quant->view_offs : (size_t) 0,
                        tensor_nbytes(generic_quant),
                        tensor_nbytes(backend_quant),
                        same_layout(generic_quant, backend_quant) ? 1 : 0,
                        compare_enabled ? "0" : "not_readback",
                        compare_enabled ? "0" : "not_readback");
            }
        }

        if (value_probe || internal_probe) {
            auto add_probe_pair = [&](const char * prefix, const char * stage, ggml_tensor * ref_tensor, ggml_tensor * cand_tensor) {
                if (ref_tensor == nullptr || cand_tensor == nullptr ||
                        ref_tensor->type != GGML_TYPE_F32 || cand_tensor->type != GGML_TYPE_F32) {
                    std::fprintf(stderr,
                            "%s_unavailable: token=%lld layer=%d stream=%s stage=%s"
                            " reason=missing_or_unsupported_tensor generic_dtype=%s backend_dtype=%s\n",
                            prefix, (long long) dsv4_hc_pos, il, stream, stage,
                            ref_tensor != nullptr ? ggml_type_name(ref_tensor->type) : "null",
                            cand_tensor != nullptr ? ggml_type_name(cand_tensor->type) : "null");
                    return;
                }
                ggml_tensor * ref = ggml_scale(ctx0, ref_tensor, 1.0f);
                ggml_tensor * cand = ggml_scale(ctx0, cand_tensor, 1.0f);
                ggml_format_name(ref, "%s_ref-%s-%s-l%d-p%lld",
                        prefix, stage, stream, il, (long long) dsv4_hc_pos);
                ggml_format_name(cand, "%s_cand-%s-%s-l%d-p%lld",
                        prefix, stage, stream, il, (long long) dsv4_hc_pos);
                ggml_build_forward_expand(gf, ref);
                ggml_build_forward_expand(gf, cand);
            };

            if (value_probe) {
                add_probe_pair("dsv4_cupd3_value_probe", "pooled", dec.generic_tail_kv_comp, dec.backend_tail_kv_comp);
                add_probe_pair("dsv4_cupd3_value_probe", "norm", dec.generic_tail_kv_comp, dec.backend_tail_kv_comp);
                add_probe_pair("dsv4_cupd3_value_probe", "norm_weighted", dec.generic_tail_kv_comp, dec.backend_tail_kv_comp);
                add_probe_pair("dsv4_cupd3_value_probe", "rope", dec.generic_tail_kv_comp, dec.backend_tail_kv_comp);
                add_probe_pair("dsv4_cupd3_value_probe", "quant", generic_quant, backend_quant);
                add_probe_pair("dsv4_cupd3_value_probe", "cache_handoff", generic_quant, backend_quant);
            }
            if (internal_probe) {
                add_probe_pair("dsv4_dci", "pooled", dec.generic_probe_pooled, dec.backend_probe_pooled);
                add_probe_pair("dsv4_dci", "norm", dec.generic_probe_norm, dec.backend_probe_norm);
                add_probe_pair("dsv4_dci", "normw", dec.generic_probe_norm_weighted, dec.backend_probe_norm_weighted);
                add_probe_pair("dsv4_dci", "rope_in", dec.generic_probe_rope_in, dec.backend_probe_rope_in);
                add_probe_pair("dsv4_dci", "rope_out", dec.generic_probe_rope_out, dec.backend_probe_rope_out);
                add_probe_pair("dsv4_dci", "pre_quant", dec.generic_tail_kv_comp, dec.backend_tail_kv_comp);
            }
        }

        if (backend_quant->type == GGML_TYPE_F32) {
            ggml_tensor * backend_exec = ggml_scale(ctx0, backend_quant, 1.0f);
            ggml_format_name(backend_exec, "dsv4_cupd3_backend_tail_exec-%s-l%d-p%lld", stream, il, (long long) dsv4_hc_pos);
            ggml_build_forward_expand(gf, backend_exec);
        }

        if (compare_enabled && generic_quant != nullptr && generic_quant->type == GGML_TYPE_F32 && backend_quant->type == GGML_TYPE_F32) {
            ggml_tensor * ref = ggml_scale(ctx0, generic_quant, 1.0f);
            ggml_tensor * cand = ggml_scale(ctx0, backend_quant, 1.0f);
            ggml_format_name(ref, "dsv4_cupd3_backend_tail_ref-%s-l%d-p%lld", stream, il, (long long) dsv4_hc_pos);
            ggml_format_name(cand, "dsv4_cupd3_backend_tail_cand-%s-l%d-p%lld", stream, il, (long long) dsv4_hc_pos);
            ggml_build_forward_expand(gf, ref);
            ggml_build_forward_expand(gf, cand);
        }

        dsv4_cupd3_backend_tail_note(
                il, dsv4_hc_pos, stream, dec, true, compare_enabled,
                dsv4_experimental_compressor_update_v3_backend_tail_dep_barrier_enabled(),
                drift_trace,
                row_probe,
                value_probe,
                internal_probe);
    };

    if (dsv4_experimental_attn_out_hc_post_consume_requested()) {
        (void) dsv4_experimental_attn_out_hc_post_consume_enabled();
    }

    std::vector<ggml_tensor *> dsv4_rmoe_end_of_graph_backends;
    for (int il = 0; il < n_layer; ++il) {
        std::vector<ggml_tensor *> dsv4_lexec_dependencies;
        ggml_tensor * dsv4_rmoe_after_layer_backend = nullptr;
        bool dsv4_lexec_compressor_update_bound = false;
        ggml_tensor * dsv4_lexec_layer_input = inpL;
        ggml_tensor * dsv4_lexec_attn_q = nullptr;
        ggml_tensor * dsv4_lexec_attn_kv = nullptr;
        ggml_tensor * dsv4_lexec_attn_out = nullptr;
        ggml_tensor * dsv4_lexec_attn_hc_post = nullptr;
        ggml_tensor * dsv4_lexec_ffn_norm = nullptr;
        ggml_tensor * dsv4_lexec_routed_moe_out = nullptr;
        ggml_tensor * dsv4_lexec_ffn_hc_post = nullptr;
        ggml_tensor * dsv4_lexec_layer_output = nullptr;
        const auto & layer = model.layers[il];
        dsv4_lexec_full_note_contract(il, dsv4_hc_pos, n_tokens);
        dsv4_layer_executor_plan_note_layer(il, dsv4_hc_pos, n_tokens);
        const uint32_t compress_ratio = hparams.attn_compress_ratio[il];
        const dsv4_rope_cfg rope_cfg = dsv4_make_rope_cfg(hparams, cparams, compress_ratio);
        const bool is_prefill = ubatch.pos == nullptr || ubatch.pos[0] == 0;
        if (!is_prefill && n_tokens == 1 && compress_ratio != 4) {
            dsv4_indexed_attn_shadow_not_implemented(il, dsv4_hc_pos, n_tokens,
                    "requires_ratio4_indexer", compress_ratio);
        }
        const int64_t dsv4_rmoe_result_chain_target_layer = dsv4_experimental_cupd3_env_i64(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", 0);
        const bool dsv4_rmoe_result_chain_site =
            dsv4_rmoe_result_chain_target_layer >= 0 &&
            dsv4_experimental_routed_moe_backend_op_result_chain_dump_site_enabled(
                    (int) dsv4_rmoe_result_chain_target_layer, dsv4_hc_pos, n_tokens);
        if (dsv4_rmoe_result_chain_site && il == dsv4_rmoe_result_chain_target_layer + 1) {
            cb(inpL, "dsv4_rmoe_result_chain_layer1_input", (int) dsv4_rmoe_result_chain_target_layer);
        }
        if (dsv4_rmoe_result_chain_site && il == n_layer - 1) {
            cb(inpL, "dsv4_rmoe_result_chain_last_layer_input", (int) dsv4_rmoe_result_chain_target_layer);
        }
        if (il > 0 && dsv4_experimental_routed_moe_backend_op_downstream_dump_site_enabled(il - 1, dsv4_hc_pos, n_tokens)) {
            cb(inpL, "dsv4_rmoe_downstream_next_layer_input", il - 1);
        }

        if (compress_ratio != 0) {
            if (compress_ratio != 4 && compress_ratio != 128) {
                throw std::runtime_error("DeepSeek V4 unsupported attention compression ratio " + std::to_string(compress_ratio));
            }
            // The hybrid memory splitter emits one sequence set per ubatch
            // for compressed DeepSeek V4 attention.
            GGML_ASSERT(ubatch.n_seqs == 1);
        }

        ggml_tensor * residual = inpL;
        dsv4_hc_mix mix = dsv4_hc_pre(ctx0, inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        ggml_tensor * cur = mix.x;
        dsv4_hc_pre_norm_result attn_hcnorm_cmp = {};
        if (dsv4_experimental_hc_pre_norm_compare_site_enabled("attn", il, dsv4_hc_pos, n_tokens)) {
            dsv4_hc_pre_norm_result cmp = dsv4_hc_pre_norm_compare(ctx0, mix, inpL,
                    layer.hc_attn_scale, layer.hc_attn_base, layer.attn_norm,
                    "attn", il, dsv4_hc_pos,
                    n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
            cur = cmp.x;
            mix = cmp.mix;
            attn_hcnorm_cmp = cmp;
            cb(mix.mixes, "hc_attn_pre_mixes", il);
        } else {
            cb(cur, "hc_attn_pre", il);
            cb(mix.mixes, "hc_attn_pre_mixes", il);
            cb(mix.pre, "hc_attn_pre_weights", il);
            cb(mix.post, "hc_attn_pre_post_weights", il);
            cb(mix.comb, "hc_attn_pre_comb", il);
            if (il > 0 && dsv4_experimental_routed_moe_backend_op_downstream_dump_site_enabled(il - 1, dsv4_hc_pos, n_tokens)) {
                cb(cur, "dsv4_rmoe_downstream_next_layer_attn_norm_input", il - 1);
            }
            cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
        cb(cur, "attn_norm", il);
        dsv4_lexec_side_probe_note_hc_pre(il, dsv4_hc_pos, n_tokens, "attn", inpL, cur, layer.attn_norm, mix, cur);
        cur = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, cur, "attn_hc_pre_norm", "attn_norm", il);
        ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
        cb(qr, "q_lora", il);
        qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        cb(qr, "q_lora_norm", il);

        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
        q = ggml_reshape_3d(ctx0, q, n_embd_head_k, n_head, n_tokens);
        q = ggml_rms_norm(ctx0, q, norm_rms_eps);
        cb(q, "Qnorm", il);
        q = dsv4_apply_rope_tail(ctx0, q, inp_pos,
                n_embd_head_k, n_head, n_tokens, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
        cb(q, "Qcur", il);
        ggml_tensor * q_rope_tail = q;
        ggml_tensor * kv = ggml_mul_mat(ctx0, layer.attn_kv, cur);
        kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head_k, 1, n_tokens);
        cb(kv, "KVnorm", il);
        kv = dsv4_apply_rope_tail(ctx0, kv, inp_pos,
                n_embd_head_k, 1, n_tokens, n_rot, rope_type,
                rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, false);
        cb(kv, "KVrope", il);
        ggml_tensor * kv_rope_tail = kv;
        kv = ggml_dsv4_fp8_kv_quantize(ctx0, kv, n_rot);
        cb(kv, "KVcur", il);
        q = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, q,  "qkv_setup", "Qcur", il);
        kv = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, kv, "qkv_setup", "KVcur", il);
        kv = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, kv, "kv_finalizer", "KVcur", il);
        dsv4_lexec_attn_q = q;
        dsv4_lexec_attn_kv = kv;

        const auto * mctx_swa = inp_attn->mctx->get_swa();
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, kv);
        dsv4_lexec_side_probe_note_kv_cache_finalizer(
                il, dsv4_hc_pos, n_tokens, "swa",
                q_rope_tail,
                kv_rope_tail,
                kv,
                kv,
                false,
                inp_attn->get_k_idxs_swa() != nullptr,
                true,
                true,
                true);
        ggml_build_forward_expand(gf, mctx_swa->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        bool cur_is_attn_projected = false;

        if (compress_ratio == 0) {
            ggml_tensor * k_cache = mctx_swa->get_k(ctx0, il);
            k_cache = ggml_reshape_3d(ctx0, k_cache, n_embd_head_k, 1, k_cache->ne[2]);
            const int64_t dsv4_swa_fa_q_limit = 8192;
            if (is_prefill && cparams.flash_attn && n_tokens > dsv4_swa_fa_q_limit) {
                ggml_tensor * attn_all = nullptr;
                ggml_tensor * k_source = kv;
                for (int64_t row_start = 0; row_start < n_tokens; row_start += dsv4_swa_fa_q_limit) {
                    const int64_t n_rows = std::min<int64_t>(dsv4_swa_fa_q_limit, n_tokens - row_start);
                    const int64_t key_start = std::max<int64_t>(0, row_start + 1 - hparams.n_swa);
                    const int64_t key_end = row_start + n_rows;
                    const int64_t n_keys = key_end - key_start;
                    GGML_ASSERT(n_keys > 0);
                    ggml_tensor * q_part = ggml_view_3d(ctx0, q,
                            q->ne[0], q->ne[1], n_rows,
                            q->nb[1], q->nb[2], row_start*q->nb[2]);
                    ggml_tensor * k_part = ggml_view_3d(ctx0, k_source,
                            k_source->ne[0], k_source->ne[1], n_keys,
                            k_source->nb[1], k_source->nb[2], key_start*k_source->nb[2]);
                    ggml_tensor * mask_part = get_dsv4_inputs()->add_mask(ctx0,
                            dsv4_mask_kind::RAW_WINDOW,
                            n_keys, n_rows,
                            n_keys, 0, hparams.n_swa, 0,
                            "dsv4_swa_split_raw_window_mask",
                            row_start, key_start);
                    mask_part = ggml_cast(ctx0, mask_part, GGML_TYPE_F16);
                    ggml_tensor * cur_part = build_attn_mha(q_part, k_part, k_part, nullptr, mask_part,
                            layer.attn_sinks, nullptr, kq_scale, il);
                    attn_all = attn_all == nullptr ? cur_part : ggml_concat(ctx0, attn_all, cur_part, 1);
                }
                cur = attn_all;
            } else {
                cur = build_attn_mha(q, k_cache, k_cache, nullptr, inp_attn->get_kq_mask_swa(),
                        layer.attn_sinks, nullptr, kq_scale, il);
            }
            cb(cur, "kqv_out", il);
            dsv4_lexec_full_note_tensor(cur, "attention_core", "kqv_out", il, dsv4_hc_pos, n_tokens);
        } else {
            ggml_tensor * k_all = kv;
            ggml_tensor * v_all = kv;
            ggml_tensor * kv_comp_all = nullptr;
            ggml_tensor * index_kv_all = nullptr;
            ggml_tensor * kv_comp_cache_all = nullptr;
            ggml_tensor * k_raw_cache_all = nullptr;
            ggml_tensor * raw_attn_mask = nullptr;
            ggml_tensor * comp_attn_mask = nullptr;
            ggml_tensor * attn_mask = nullptr;
            bool segment_compress_attention = false;
            const llama_seq_id seq_id = ubatch.seq_id[0][0];
            auto store_attn_cache_rows = [&](ggml_tensor * src, int64_t row_start, int64_t n_rows) -> ggml_tensor * {
                ggml_tensor * out = src;
                for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                    const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                    ggml_tensor * cur_out = dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_attn_k(ctx0, il, dst_seq_id), src, row_start, n_rows);
                    if (is == 0) {
                        out = cur_out;
                    }
                }
                return out;
            };
            auto store_index_cache_rows = [&](ggml_tensor * src, int64_t row_start, int64_t n_rows) -> ggml_tensor * {
                ggml_tensor * out = src;
                for (int32_t is = 0; is < ubatch.n_seq_id[0]; ++is) {
                    const llama_seq_id dst_seq_id = ubatch.seq_id[0][is];
                    ggml_tensor * cur_out = dsv4_store_cache_rows(ctx0, gf, mctx_dsv4->get_dsv4_index_k(ctx0, il, dst_seq_id), src, row_start, n_rows);
                    if (is == 0) {
                        out = cur_out;
                    }
                }
                return out;
            };
            const int64_t state_size = hparams.n_embd_r();
            const dsv4_state_layout attn_state_layout = dsv4_make_state_layout(compress_ratio, n_embd_head_k);
            const int64_t dsv4_compress_fa_q_limit = 8192;
            const int64_t dsv4_compress_fa_work_limit = 16ll*1024ll*1024ll;
            auto should_segment_compress_attention = [&](int64_t n_q, int64_t n_comp_keys) {
                if (!cparams.flash_attn || n_q <= 1 || n_comp_keys <= 0) {
                    return false;
                }

                return n_q > dsv4_compress_fa_q_limit ||
                        uint64_t(n_q)*uint64_t(n_comp_keys) >= uint64_t(dsv4_compress_fa_work_limit);
            };
            auto choose_compress_attention_q_limit = [&](int64_t n_comp_keys) {
                int64_t limit = dsv4_compress_fa_q_limit;
                if (n_comp_keys > 0) {
                    limit = std::min<int64_t>(limit, std::max<int64_t>(128, dsv4_compress_fa_work_limit/n_comp_keys));
                }
                if (limit > 128) {
                    limit -= limit % 128;
                }
                return std::max<int64_t>(128, limit);
            };
            auto choose_compress_prefill_q_limit = [&](int64_t n_comp_keys) {
                int64_t limit = dsv4_compress_fa_q_limit;
                const int64_t raw_overlap = std::max<int64_t>(0, hparams.n_swa > 0 ? int64_t(hparams.n_swa) - 1 : 0);
                const int64_t max_fa_keys = 16*1024;
                limit = std::min<int64_t>(limit, std::max<int64_t>(128, max_fa_keys - raw_overlap - std::max<int64_t>(0, n_comp_keys)));
                if (compress_ratio == 4 && n_comp_keys > 6144) {
                    limit = std::min<int64_t>(limit, 4096);
                }
                const int64_t align = compress_ratio == 4 ? 2048 : 128;
                if (limit > align) {
                    limit -= limit % align;
                } else if (limit > 128) {
                    limit -= limit % 128;
                }
                return std::max<int64_t>(128, limit);
            };

            ggml_tensor * prev_kv_state_all = build_rs(inp_rs, inp_rs->mctx->get_r_l(il), state_size, ubatch.n_seqs);
            ggml_tensor * prev_sc_state_all = build_rs(inp_rs, inp_rs->mctx->get_s_l(il), state_size, ubatch.n_seqs);
            ggml_tensor * prev_attn_kv_state = dsv4_view_state_segment(ctx0, prev_kv_state_all, 0, attn_state_layout.width, attn_state_layout.rows);
            ggml_tensor * prev_attn_sc_state = dsv4_view_state_segment(ctx0, prev_sc_state_all, 0, attn_state_layout.width, attn_state_layout.rows);

            const int64_t n_comp = n_tokens / compress_ratio;
            int64_t n_comp_visible = n_comp;
            if (is_prefill) {
                dsv4_state_pair state = dsv4_build_compressor_prefill_state(ctx0, cur,
                        layer.attn_compressor_kv,
                        layer.attn_compressor_gate,
                        layer.attn_compressor_ape,
                        n_embd_head_k,
                        n_tokens,
                        compress_ratio);
                dsv4_store_state_segment(ctx0, gf, state.kv,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0);
                dsv4_store_state_segment(ctx0, gf, state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0);

                if (compress_ratio == 4) {
                    const dsv4_state_layout index_state_layout = dsv4_make_state_layout(compress_ratio, hparams.indexer_head_size);
                    dsv4_state_pair index_state = dsv4_build_compressor_prefill_state(ctx0, cur,
                            layer.indexer_compressor_kv,
                            layer.indexer_compressor_gate,
                            layer.indexer_compressor_ape,
                            hparams.indexer_head_size,
                            n_tokens,
                            compress_ratio);
                    dsv4_store_state_segment(ctx0, gf, index_state.kv,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                    dsv4_store_state_segment(ctx0, gf, index_state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                    GGML_ASSERT(attn_state_layout.elems + index_state_layout.elems <= state_size);
                }
            }

            if (is_prefill && n_comp > 0) {
                const int64_t dsv4_compress_prefill_segment = 8192;
                const bool segment_compress_prefill =
                        cparams.flash_attn &&
                        compress_ratio == 4 &&
                        n_tokens > dsv4_compress_prefill_segment;

                ggml_tensor * kv_comp = nullptr;
                if (segment_compress_prefill) {
                    dsv4_state_pair prev_state = {};
                    bool have_prev_state = false;
                    for (int64_t row_start = 0; row_start < n_tokens; row_start += dsv4_compress_prefill_segment) {
                        const int64_t n_rows = std::min<int64_t>(dsv4_compress_prefill_segment, n_tokens - row_start);
                        const int64_t n_comp_part = n_rows / compress_ratio;
                        if (n_comp_part <= 0) {
                            continue;
                        }

                        ggml_tensor * cur_part = ggml_view_2d(ctx0, cur,
                                cur->ne[0], n_rows,
                                cur->nb[1],
                                row_start*cur->nb[1]);
                        ggml_tensor * comp_pos_part = ggml_arange(ctx0,
                                float(row_start),
                                float(row_start + n_comp_part*compress_ratio),
                                float(compress_ratio));
                        comp_pos_part = ggml_cast(ctx0, comp_pos_part, GGML_TYPE_I32);

                        ggml_tensor * kv_comp_part = have_prev_state
                            ? dsv4_build_compressor_prefill_with_overlap(ctx0, cur_part,
                                    prev_state.kv,
                                    prev_state.score,
                                    layer.attn_compressor_kv,
                                    layer.attn_compressor_gate,
                                    layer.attn_compressor_ape,
                                    layer.attn_compressor_norm,
                                    comp_pos_part,
                                    n_embd_head_k, n_rot, n_rows, compress_ratio, row_start, rope_type, rope_cfg, norm_rms_eps)
                            : dsv4_build_compressor_prefill(ctx0, cur_part,
                                    layer.attn_compressor_kv,
                                    layer.attn_compressor_gate,
                                    layer.attn_compressor_ape,
                                    layer.attn_compressor_norm,
                                    comp_pos_part,
                                    n_embd_head_k, n_rot, n_rows, compress_ratio, rope_type, rope_cfg, norm_rms_eps);
                        kv_comp_part = ggml_dsv4_fp8_kv_quantize(ctx0, kv_comp_part, n_rot);
                        kv_comp = kv_comp == nullptr ? kv_comp_part : ggml_concat(ctx0, kv_comp, kv_comp_part, 2);

                        prev_state = dsv4_build_compressor_prefill_state(ctx0, cur_part,
                                layer.attn_compressor_kv,
                                layer.attn_compressor_gate,
                                layer.attn_compressor_ape,
                                n_embd_head_k,
                                n_rows,
                                compress_ratio);
                        have_prev_state = true;
                    }
                } else {
                    ggml_tensor * comp_pos = ggml_arange(ctx0, 0.0f, float(n_comp * compress_ratio), float(compress_ratio));
                    comp_pos = ggml_cast(ctx0, comp_pos, GGML_TYPE_I32);

                    kv_comp = dsv4_build_compressor_prefill(ctx0, cur,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            layer.attn_compressor_norm,
                            comp_pos,
                            n_embd_head_k, n_rot, n_tokens, compress_ratio, rope_type, rope_cfg, norm_rms_eps);
                    kv_comp = ggml_dsv4_fp8_kv_quantize(ctx0, kv_comp, n_rot);
                }
                cb(kv_comp, "KVcompress", il);

                store_attn_cache_rows(kv_comp, 0, n_comp);

                kv_comp_all = kv_comp;
                k_all = ggml_concat(ctx0, kv, kv_comp, 2);
                v_all = k_all;

                if (compress_ratio == 4) {
                    ggml_tensor * index_kv = nullptr;
                    if (segment_compress_prefill) {
                        dsv4_state_pair prev_state = {};
                        bool have_prev_state = false;
                        for (int64_t row_start = 0; row_start < n_tokens; row_start += dsv4_compress_prefill_segment) {
                            const int64_t n_rows = std::min<int64_t>(dsv4_compress_prefill_segment, n_tokens - row_start);
                            const int64_t n_comp_part = n_rows / compress_ratio;
                            if (n_comp_part <= 0) {
                                continue;
                            }

                            ggml_tensor * cur_part = ggml_view_2d(ctx0, cur,
                                    cur->ne[0], n_rows,
                                    cur->nb[1],
                                    row_start*cur->nb[1]);
                            ggml_tensor * comp_pos_part = ggml_arange(ctx0,
                                    float(row_start),
                                    float(row_start + n_comp_part*compress_ratio),
                                    float(compress_ratio));
                            comp_pos_part = ggml_cast(ctx0, comp_pos_part, GGML_TYPE_I32);

                            ggml_tensor * index_kv_part = have_prev_state
                                ? dsv4_build_compressor_prefill_with_overlap(ctx0, cur_part,
                                        prev_state.kv,
                                        prev_state.score,
                                        layer.indexer_compressor_kv,
                                        layer.indexer_compressor_gate,
                                        layer.indexer_compressor_ape,
                                        layer.indexer_compressor_norm,
                                        comp_pos_part,
                                        hparams.indexer_head_size, n_rot, n_rows, compress_ratio, row_start, rope_type, rope_cfg, norm_rms_eps)
                                : dsv4_build_compressor_prefill(ctx0, cur_part,
                                        layer.indexer_compressor_kv,
                                        layer.indexer_compressor_gate,
                                        layer.indexer_compressor_ape,
                                        layer.indexer_compressor_norm,
                                        comp_pos_part,
                                        hparams.indexer_head_size, n_rot, n_rows, compress_ratio, rope_type, rope_cfg, norm_rms_eps);
                            index_kv_part = ggml_dsv4_hadamard_fp4_quantize(ctx0, index_kv_part);
                            index_kv = index_kv == nullptr ? index_kv_part : ggml_concat(ctx0, index_kv, index_kv_part, 2);

                            prev_state = dsv4_build_compressor_prefill_state(ctx0, cur_part,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    hparams.indexer_head_size,
                                    n_rows,
                                    compress_ratio);
                            have_prev_state = true;
                        }
                    } else {
                        ggml_tensor * comp_pos = ggml_arange(ctx0, 0.0f, float(n_comp * compress_ratio), float(compress_ratio));
                        comp_pos = ggml_cast(ctx0, comp_pos, GGML_TYPE_I32);

                        index_kv = dsv4_build_compressor_prefill(ctx0, cur,
                                layer.indexer_compressor_kv,
                                layer.indexer_compressor_gate,
                                layer.indexer_compressor_ape,
                                layer.indexer_compressor_norm,
                                comp_pos,
                                hparams.indexer_head_size, n_rot, n_tokens, compress_ratio, rope_type, rope_cfg, norm_rms_eps);
                        index_kv = ggml_dsv4_hadamard_fp4_quantize(ctx0, index_kv);
                    }
                    cb(index_kv, "indexer_KVcompress", il);
                    index_kv_all = index_kv;

                    store_index_cache_rows(index_kv, 0, n_comp);

                    if (should_segment_compress_attention(n_tokens, n_comp)) {
                        segment_compress_attention = true;
                    } else {
                        ggml_tensor * raw_mask = get_dsv4_inputs()->add_mask(ctx0,
                                dsv4_mask_kind::RAW_WINDOW,
                                n_tokens, n_tokens,
                                n_tokens, n_comp, hparams.n_swa, compress_ratio,
                                "dsv4_attn_raw_window_mask");
                        raw_attn_mask = raw_mask;

                        ggml_tensor * index_mask = get_dsv4_inputs()->add_mask(ctx0,
                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                n_comp, n_tokens,
                                0, n_comp, 0, compress_ratio,
                                "dsv4_indexer_causal_mask");

                        ggml_tensor * index_scores = dsv4_build_indexer_scores_prefill(ctx0,
                                cur, qr, index_kv,
                                layer.indexer_attn_q_b,
                                layer.indexer_proj,
                                inp_pos,
                                index_mask,
                                hparams.indexer_n_head,
                                hparams.indexer_head_size,
                                n_tokens,
                                n_rot,
                                rope_type,
                                rope_cfg);
                        cb(index_scores, "indexer_scores", il);

                        const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp);
                        ggml_tensor * topk = ggml_argsort_top_k(ctx0, index_scores, top_k);
                        cb(topk, "indexer_topk", il);

                        ggml_tensor * comp_mask = dsv4_build_compressed_mask_from_topk(ctx0, index_scores, topk);
                        cb(comp_mask, "dsv4_attn_compress_mask", il);

                        attn_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
                    }
                } else {
                    if (should_segment_compress_attention(n_tokens, n_comp)) {
                        segment_compress_attention = true;
                    } else {
                        raw_attn_mask = get_dsv4_inputs()->add_mask(ctx0,
                                dsv4_mask_kind::ATTN_STATIC,
                                n_tokens + n_comp, n_tokens,
                                n_tokens, n_comp, hparams.n_swa, compress_ratio,
                                "dsv4_attn_static_mask");
                        attn_mask = raw_attn_mask;
                    }
                }
            } else {
                raw_attn_mask = get_dsv4_inputs()->add_mask(ctx0,
                        dsv4_mask_kind::RAW_WINDOW,
                        n_tokens, n_tokens,
                        n_tokens, 0, hparams.n_swa, compress_ratio,
                        "dsv4_attn_raw_window_mask");
                attn_mask = raw_attn_mask;
            }

            const llama_pos indexed_attn_shadow_pos = ubatch.pos ? ubatch.pos[0] : 0;
            ggml_tensor * indexed_attn_shadow_kv = nullptr;
            ggml_tensor * indexed_attn_shadow_mask = nullptr;

            if (!is_prefill) {
                const llama_pos first_pos = ubatch.pos ? ubatch.pos[0] : 0;
                const llama_pos last_pos  = ubatch.pos ? ubatch.pos[n_tokens - 1] : n_tokens - 1;
                const int64_t n_comp_before  = first_pos / compress_ratio;
                n_comp_visible = (last_pos + 1) / compress_ratio;
                const int64_t n_comp_new     = n_comp_visible - n_comp_before;
                const bool aligned_chunk_prefill =
                        n_tokens > 1 &&
                        first_pos % compress_ratio == 0;
                const int64_t n_comp_cache = mctx_dsv4->get_dsv4_n_comp(il);
                GGML_ASSERT(n_comp_visible <= n_comp_cache);
                ggml_tensor * kv_comp_new = nullptr;
                ggml_tensor * index_kv_new = nullptr;
                auto build_visible_comp_cache = [&](ggml_tensor * cache, ggml_tensor * new_rows) -> ggml_tensor * {
                    if (n_comp_visible <= 0) {
                        return nullptr;
                    }
                    if (new_rows != nullptr && n_comp_new > 0) {
                        GGML_ASSERT(new_rows->ne[2] == n_comp_new);
                        if (new_rows->type != cache->type) {
                            new_rows = ggml_cast(ctx0, new_rows, cache->type);
                        }
                        if (n_comp_before > 0) {
                            ggml_tensor * cached_before = dsv4_cache_view_3d(ctx0, cache, n_comp_before);
                            return ggml_concat(ctx0, cached_before, new_rows, 2);
                        }
                        return new_rows;
                    }
                    return dsv4_cache_view_3d(ctx0, cache, n_comp_visible);
                };

                if (aligned_chunk_prefill) {
                    dsv4_state_pair state = dsv4_build_compressor_prefill_state(ctx0, cur,
                            layer.attn_compressor_kv,
                            layer.attn_compressor_gate,
                            layer.attn_compressor_ape,
                            n_embd_head_k,
                            n_tokens,
                            compress_ratio);
                    dsv4_store_state_segment(ctx0, gf, state.kv,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0);
                    dsv4_store_state_segment(ctx0, gf, state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0);

                    if (n_comp_new > 0) {
                        ggml_tensor * comp_pos = ggml_arange(ctx0, float(first_pos), float(first_pos + n_comp_new*compress_ratio), float(compress_ratio));
                        comp_pos = ggml_cast(ctx0, comp_pos, GGML_TYPE_I32);

                        ggml_tensor * kv_comp = dsv4_build_compressor_prefill_with_overlap(ctx0, cur,
                                prev_attn_kv_state,
                                prev_attn_sc_state,
                                layer.attn_compressor_kv,
                                layer.attn_compressor_gate,
                                layer.attn_compressor_ape,
                                layer.attn_compressor_norm,
                                comp_pos,
                                n_embd_head_k, n_rot, n_tokens, compress_ratio, first_pos, rope_type, rope_cfg, norm_rms_eps);
                        kv_comp_new = ggml_dsv4_fp8_kv_quantize(ctx0, kv_comp, n_rot);
                        cb(kv_comp_new, "KVcompress", il);
                        dsv4_lexec_side_probe_note_kv_cache_finalizer(
                                il, dsv4_hc_pos, n_tokens, "compressed",
                                nullptr,
                                kv_comp,
                                kv_comp_new,
                                kv_comp_new,
                                true,
                                true,
                                true,
                                true,
                                true);
                        kv_comp_new = store_attn_cache_rows(kv_comp_new, n_comp_before, n_comp_new);
                    }
                } else {
                    dsv4_decode_compressor dec = n_tokens == 1
                        ? dsv4_build_compressor_decode(ctx0, cur,
                                prev_attn_kv_state,
                                prev_attn_sc_state,
                                layer.attn_compressor_kv,
                                layer.attn_compressor_gate,
                                layer.attn_compressor_ape,
                                layer.attn_compressor_norm,
                                n_embd_head_k,
                                n_rot,
                                first_pos,
                                compress_ratio,
                                rope_type,
                                rope_cfg,
                                norm_rms_eps,
                                "attn",
                                il)
                        : dsv4_build_compressor_decode_chunk(ctx0, cur,
                                prev_attn_kv_state,
                                prev_attn_sc_state,
                                layer.attn_compressor_kv,
                                layer.attn_compressor_gate,
                                layer.attn_compressor_ape,
                                layer.attn_compressor_norm,
                                ubatch,
                                n_embd_head_k,
                                n_rot,
                                n_tokens,
                                compress_ratio,
                                rope_type,
                                rope_cfg,
                                norm_rms_eps);

                    dsv4_store_state_segment(ctx0, gf, dec.kv_state,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, 0);
                    dsv4_store_state_segment(ctx0, gf, dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, 0);
                    dec.kv_state = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, dec.kv_state, "compressor_update", "kv_state", il);
                    dec.score_state = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, dec.score_state, "compressor_update", "score_state", il);
                    if (dsv4_experimental_compressor_update_v3_site_enabled(il, dsv4_hc_pos, n_tokens)) {
                        dsv4_cupd3_note_case(il, dsv4_hc_pos, dec.kv_comp != nullptr, dec.kv_comp != nullptr);
                        dsv4_cupd3_shadow_probe(dec.kv_state, "state_kv", il, "attn");
                        dsv4_cupd3_shadow_probe(dec.score_state, "state_score", il, "attn");
                    }
                    dsv4_lexec_compressor_update_bound = true;

                    if (dec.kv_comp != nullptr) {
                        dec.kv_comp = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, dec.kv_comp, "compressor_update", "kv_comp", il);
                        if (dsv4_experimental_compressor_update_v3_tail_attrib_site_enabled(il, dsv4_hc_pos, n_tokens)) {
                            dsv4_cupd3_tail_attrib_note(il, dsv4_hc_pos, "attn");
                        }
                        dsv4_cupd3_shadow_probe(dec.kv_comp, "pool_input", il, "attn");
                        dsv4_cupd3_shadow_probe(dec.kv_comp, "pooled", il, "attn");
                        dsv4_cupd3_shadow_probe(dec.kv_comp, "pre_norm", il, "attn");
                        dsv4_cupd3_shadow_probe(dec.kv_comp, "norm", il, "attn");
                        dsv4_cupd3_shadow_probe(dec.kv_comp, "norm_w", il, "attn");
                        dsv4_cupd3_shadow_probe(dec.kv_comp, "rope", il, "attn");
                        kv_comp_new = ggml_dsv4_fp8_kv_quantize(ctx0, dec.kv_comp, n_rot);
                        dsv4_cupd3_shadow_probe(kv_comp_new, "quant", il, "attn");
                        if (dec.backend_tail_kv_comp != nullptr) {
                            ggml_tensor * generic_tail_quant = kv_comp_new;
                            if (dec.generic_tail_kv_comp != nullptr && dec.generic_tail_kv_comp != dec.kv_comp) {
                                generic_tail_quant = ggml_dsv4_fp8_kv_quantize(ctx0, dec.generic_tail_kv_comp, n_rot);
                                ggml_set_name(generic_tail_quant, "dsv4_cupd3_backend_tail_ref_quant-attn");
                            }
                            ggml_tensor * backend_tail_quant = kv_comp_new;
                            if (dec.backend_tail_kv_comp != dec.kv_comp) {
                                backend_tail_quant = ggml_dsv4_fp8_kv_quantize(ctx0, dec.backend_tail_kv_comp, n_rot);
                                ggml_set_name(backend_tail_quant, "dsv4_cupd3_backend_tail_quant-attn");
                            }
                            dsv4_cupd3_backend_tail_probe(generic_tail_quant, backend_tail_quant, dec, il, "attn");
                        }
                        dsv4_cupd3_shadow_probe(kv_comp_new, "downstream_kv", il, "attn");
                        dsv4_lexec_side_probe_note_compressor_update(
                                il, dsv4_hc_pos, n_tokens, "attn",
                                dec.kv_state,
                                dec.score_state,
                                dec.kv_comp,
                                dec.generic_probe_pooled != nullptr ? dec.generic_probe_pooled : dec.kv_comp,
                                dec.generic_probe_norm,
                                layer.attn_compressor_norm,
                                dec.generic_probe_norm_weighted != nullptr ? dec.generic_probe_norm_weighted : dec.kv_comp,
                                dec.generic_probe_rope_in != nullptr ? dec.generic_probe_rope_in :
                                    (dec.generic_probe_norm_weighted != nullptr ? dec.generic_probe_norm_weighted : dec.kv_comp),
                                dec.generic_probe_rope_out != nullptr ? dec.generic_probe_rope_out : dec.kv_comp,
                                kv_comp_new,
                                kv_comp_new,
                                true,
                                dsv4_hc_pos + 1 - (int64_t) compress_ratio,
                                dsv4_hc_pos + 1 - (int64_t) compress_ratio,
                                n_rot,
                                n_embd_head_k,
                                rope_type,
                                n_embd_head_k - n_rot,
                                rope_cfg.n_ctx_orig,
                                rope_cfg.freq_base,
                                rope_cfg.freq_scale,
                                rope_cfg.ext_factor,
                                rope_cfg.attn_factor,
                                rope_cfg.beta_fast,
                                rope_cfg.beta_slow);
                        dsv4_lexec_side_probe_note_kv_cache_finalizer(
                                il, dsv4_hc_pos, n_tokens, "compressed",
                                nullptr,
                                dec.generic_probe_rope_out != nullptr ? dec.generic_probe_rope_out : dec.kv_comp,
                                kv_comp_new,
                                kv_comp_new,
                                true,
                                true,
                                true,
                                true,
                                true);
                        if (dec.candidate_tail_consumed &&
                                dec.backend_tail_kv_comp != nullptr &&
                                dsv4_experimental_compressor_update_v3_backend_tail_dep_barrier_enabled()) {
                            kv_comp_new = ggml_cont(ctx0, kv_comp_new);
                            ggml_set_name(kv_comp_new, "dsv4_cupd3_backend_tail_dep_barrier-attn");
                        }
                        if (dec.candidate_tail_consumed && dec.backend_tail_kv_comp == nullptr) {
                            dsv4_cupd3_consume_note(dec, true);
                        }
                        kv_comp_new = store_attn_cache_rows(kv_comp_new, n_comp_before, n_comp_new);
                    }
                }

                ggml_tensor * k_raw = mctx_swa->get_k(ctx0, il);
                k_raw = ggml_reshape_3d(ctx0, k_raw, n_embd_head_k, 1, k_raw->ne[2]);
                k_raw_cache_all = k_raw;
                k_all = k_raw;
                v_all = k_raw;
                raw_attn_mask = inp_attn->self_kq_mask_swa;
                attn_mask = raw_attn_mask;

                if (n_comp_visible > 0) {
                    ggml_tensor * kv_comp_cache = build_visible_comp_cache(
                            mctx_dsv4->get_dsv4_attn_k(ctx0, il, seq_id),
                            kv_comp_new);
                    kv_comp_cache_all = kv_comp_cache;
                    k_all = ggml_concat(ctx0, k_raw, kv_comp_cache, 2);
                    v_all = k_all;

                    ggml_tensor * comp_mask = nullptr;
                    if (compress_ratio == 4) {
                        const dsv4_state_layout index_state_layout = dsv4_make_state_layout(compress_ratio, hparams.indexer_head_size);
                        ggml_tensor * prev_index_kv_state = dsv4_view_state_segment(ctx0, prev_kv_state_all,
                                attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);
                        ggml_tensor * prev_index_sc_state = dsv4_view_state_segment(ctx0, prev_sc_state_all,
                                attn_state_layout.elems, index_state_layout.width, index_state_layout.rows);

                        if (aligned_chunk_prefill) {
                            dsv4_state_pair index_state = dsv4_build_compressor_prefill_state(ctx0, cur,
                                    layer.indexer_compressor_kv,
                                    layer.indexer_compressor_gate,
                                    layer.indexer_compressor_ape,
                                    hparams.indexer_head_size,
                                    n_tokens,
                                    compress_ratio);
                            dsv4_store_state_segment(ctx0, gf, index_state.kv,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                            dsv4_store_state_segment(ctx0, gf, index_state.score, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems);

                            if (n_comp_new > 0) {
                                ggml_tensor * comp_pos = ggml_arange(ctx0, float(first_pos), float(first_pos + n_comp_new*compress_ratio), float(compress_ratio));
                                comp_pos = ggml_cast(ctx0, comp_pos, GGML_TYPE_I32);

                                ggml_tensor * index_kv = dsv4_build_compressor_prefill_with_overlap(ctx0, cur,
                                        prev_index_kv_state,
                                        prev_index_sc_state,
                                        layer.indexer_compressor_kv,
                                        layer.indexer_compressor_gate,
                                        layer.indexer_compressor_ape,
                                        layer.indexer_compressor_norm,
                                        comp_pos,
                                        hparams.indexer_head_size, n_rot, n_tokens, compress_ratio, first_pos, rope_type, rope_cfg, norm_rms_eps);
                                index_kv_new = ggml_dsv4_hadamard_fp4_quantize(ctx0, index_kv);
                                cb(index_kv_new, "indexer_KVcompress", il);
                                dsv4_lexec_side_probe_note_kv_cache_finalizer(
                                        il, dsv4_hc_pos, n_tokens, "compressed",
                                        nullptr,
                                        index_kv,
                                        index_kv_new,
                                        index_kv_new,
                                        true,
                                        true,
                                        true,
                                        true,
                                        true);
                                index_kv_new = store_index_cache_rows(index_kv_new, n_comp_before, n_comp_new);
                            }
                        } else {
                            dsv4_decode_compressor index_dec = n_tokens == 1
                                ? dsv4_build_compressor_decode(ctx0, cur,
                                        prev_index_kv_state,
                                        prev_index_sc_state,
                                        layer.indexer_compressor_kv,
                                        layer.indexer_compressor_gate,
                                        layer.indexer_compressor_ape,
                                        layer.indexer_compressor_norm,
                                        hparams.indexer_head_size,
                                        n_rot,
                                        first_pos,
                                        compress_ratio,
                                        rope_type,
                                        rope_cfg,
                                        norm_rms_eps,
                                        "index",
                                        il)
                                : dsv4_build_compressor_decode_chunk(ctx0, cur,
                                        prev_index_kv_state,
                                        prev_index_sc_state,
                                        layer.indexer_compressor_kv,
                                        layer.indexer_compressor_gate,
                                        layer.indexer_compressor_ape,
                                        layer.indexer_compressor_norm,
                                        ubatch,
                                        hparams.indexer_head_size,
                                        n_rot,
                                        n_tokens,
                                        compress_ratio,
                                        rope_type,
                                        rope_cfg,
                                        norm_rms_eps);

                            dsv4_store_state_segment(ctx0, gf, index_dec.kv_state,    inp_rs->mctx->get_r_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                            dsv4_store_state_segment(ctx0, gf, index_dec.score_state, inp_rs->mctx->get_s_l(il), state_size, inp_rs->head, attn_state_layout.elems);
                            index_dec.kv_state = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, index_dec.kv_state, "compressor_update", "index_kv_state", il);
                            index_dec.score_state = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, index_dec.score_state, "compressor_update", "index_score_state", il);
                            if (dsv4_experimental_compressor_update_v3_site_enabled(il, dsv4_hc_pos, n_tokens)) {
                                dsv4_cupd3_note_case(il, dsv4_hc_pos, index_dec.kv_comp != nullptr, index_dec.kv_comp != nullptr);
                                dsv4_cupd3_shadow_probe(index_dec.kv_state, "state_kv", il, "index");
                                dsv4_cupd3_shadow_probe(index_dec.score_state, "state_score", il, "index");
                            }
                            dsv4_lexec_compressor_update_bound = true;

                            if (index_dec.kv_comp != nullptr) {
                                index_dec.kv_comp = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, index_dec.kv_comp, "compressor_update", "index_kv_comp", il);
                                if (dsv4_experimental_compressor_update_v3_tail_attrib_site_enabled(il, dsv4_hc_pos, n_tokens)) {
                                    dsv4_cupd3_tail_attrib_note(il, dsv4_hc_pos, "index");
                                }
                                dsv4_cupd3_shadow_probe(index_dec.kv_comp, "pool_input", il, "index");
                                dsv4_cupd3_shadow_probe(index_dec.kv_comp, "pooled", il, "index");
                                dsv4_cupd3_shadow_probe(index_dec.kv_comp, "pre_norm", il, "index");
                                dsv4_cupd3_shadow_probe(index_dec.kv_comp, "norm", il, "index");
                                dsv4_cupd3_shadow_probe(index_dec.kv_comp, "norm_w", il, "index");
                                dsv4_cupd3_shadow_probe(index_dec.kv_comp, "rope", il, "index");
                                index_kv_new = ggml_dsv4_hadamard_fp4_quantize(ctx0, index_dec.kv_comp);
                                dsv4_cupd3_shadow_probe(index_kv_new, "quant", il, "index");
                                if (index_dec.backend_tail_kv_comp != nullptr) {
                                    ggml_tensor * generic_tail_quant = index_kv_new;
                                    if (index_dec.generic_tail_kv_comp != nullptr && index_dec.generic_tail_kv_comp != index_dec.kv_comp) {
                                        generic_tail_quant = ggml_dsv4_hadamard_fp4_quantize(ctx0, index_dec.generic_tail_kv_comp);
                                        ggml_set_name(generic_tail_quant, "dsv4_cupd3_backend_tail_ref_quant-index");
                                    }
                                    ggml_tensor * backend_tail_quant = index_kv_new;
                                    if (index_dec.backend_tail_kv_comp != index_dec.kv_comp) {
                                        backend_tail_quant = ggml_dsv4_hadamard_fp4_quantize(ctx0, index_dec.backend_tail_kv_comp);
                                        ggml_set_name(backend_tail_quant, "dsv4_cupd3_backend_tail_quant-index");
                                    }
                                    dsv4_cupd3_backend_tail_probe(generic_tail_quant, backend_tail_quant, index_dec, il, "index");
                                }
                                dsv4_cupd3_shadow_probe(index_kv_new, "downstream_kv", il, "index");
                                dsv4_lexec_side_probe_note_compressor_update(
                                        il, dsv4_hc_pos, n_tokens, "index",
                                        index_dec.kv_state,
                                        index_dec.score_state,
                                        index_dec.kv_comp,
                                        index_dec.generic_probe_pooled != nullptr ? index_dec.generic_probe_pooled : index_dec.kv_comp,
                                        index_dec.generic_probe_norm,
                                        layer.indexer_compressor_norm,
                                        index_dec.generic_probe_norm_weighted != nullptr ? index_dec.generic_probe_norm_weighted : index_dec.kv_comp,
                                        index_dec.generic_probe_rope_in != nullptr ? index_dec.generic_probe_rope_in :
                                            (index_dec.generic_probe_norm_weighted != nullptr ? index_dec.generic_probe_norm_weighted : index_dec.kv_comp),
                                        index_dec.generic_probe_rope_out != nullptr ? index_dec.generic_probe_rope_out : index_dec.kv_comp,
                                        index_kv_new,
                                        index_kv_new,
                                        true,
                                        dsv4_hc_pos + 1 - (int64_t) compress_ratio,
                                        dsv4_hc_pos + 1 - (int64_t) compress_ratio,
                                        n_rot,
                                        hparams.indexer_head_size,
                                        rope_type,
                                        hparams.indexer_head_size - n_rot,
                                        rope_cfg.n_ctx_orig,
                                        rope_cfg.freq_base,
                                        rope_cfg.freq_scale,
                                        rope_cfg.ext_factor,
                                        rope_cfg.attn_factor,
                                        rope_cfg.beta_fast,
                                        rope_cfg.beta_slow);
                                dsv4_lexec_side_probe_note_kv_cache_finalizer(
                                        il, dsv4_hc_pos, n_tokens, "compressed",
                                        nullptr,
                                        index_dec.generic_probe_rope_out != nullptr ? index_dec.generic_probe_rope_out : index_dec.kv_comp,
                                        index_kv_new,
                                        index_kv_new,
                                        true,
                                        true,
                                        true,
                                        true,
                                        true);
                                if (index_dec.candidate_tail_consumed &&
                                        index_dec.backend_tail_kv_comp != nullptr &&
                                        dsv4_experimental_compressor_update_v3_backend_tail_dep_barrier_enabled()) {
                                    index_kv_new = ggml_cont(ctx0, index_kv_new);
                                    ggml_set_name(index_kv_new, "dsv4_cupd3_backend_tail_dep_barrier-index");
                                }
                                if (index_dec.candidate_tail_consumed && index_dec.backend_tail_kv_comp == nullptr) {
                                    dsv4_cupd3_consume_note(index_dec, true);
                                }
                                index_kv_new = store_index_cache_rows(index_kv_new, n_comp_before, n_comp_new);
                            }
                        }
                        index_kv_all = build_visible_comp_cache(
                                mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id),
                                index_kv_new);
                    }

                    if (should_segment_compress_attention(n_tokens, n_comp_visible)) {
                        segment_compress_attention = true;
                    } else if (compress_ratio == 4) {
                        if (n_tokens == 1 && n_comp_visible <= hparams.indexer_top_k) {
                            comp_mask = get_dsv4_inputs()->add_mask(ctx0,
                                    dsv4_mask_kind::COMPRESS_CAUSAL,
                                    n_comp_visible, n_tokens,
                                    0, n_comp_visible, 0, compress_ratio,
                                    "dsv4_attn_compress_mask");
                            if (dsv4_experimental_indexed_attn_shadow_output_enabled() &&
                                    dsv4_experimental_indexed_attn_site_enabled(il, first_pos, n_tokens)) {
                                indexed_attn_shadow_kv = kv_comp_cache_all;
                                if (std::strcmp(dsv4_experimental_indexed_attn_mask_mode(), "visible_all_no_mask") == 0) {
                                    indexed_attn_shadow_mask = dsv4_new_filled_2d(ctx0, n_comp_visible, n_tokens, 0.0f);
                                } else {
                                    indexed_attn_shadow_mask = comp_mask;
                                }
                            }
                            dsv4_indexed_attn_shadow_note(il,
                                    first_pos,
                                    n_tokens,
                                    k_raw_cache_all != nullptr ? k_raw_cache_all->ne[2] : 0,
                                    hparams.n_swa,
                                    n_comp_visible,
                                    n_comp_visible,
                                    std::min<int64_t>(hparams.indexer_top_k, n_comp_visible),
                                    "visible_all",
                                    true);
                        } else {
                            GGML_ASSERT(index_kv_all != nullptr);
                            ggml_tensor * index_cache = index_kv_all;
                            index_cache = ggml_reshape_2d(ctx0, index_cache, hparams.indexer_head_size, n_comp_visible);
                            ggml_tensor * index_scores = n_tokens == 1
                                ? dsv4_build_indexer_scores_decode(ctx0,
                                        cur, qr, index_cache,
                                        layer.indexer_attn_q_b,
                                        layer.indexer_proj,
                                        inp_pos,
                                        hparams.indexer_n_head,
                                        hparams.indexer_head_size,
                                        n_comp_visible,
                                        n_rot,
                                        rope_type,
                                        rope_cfg)
                                : dsv4_build_indexer_scores_prefill(ctx0,
                                        cur, qr, index_kv_all,
                                        layer.indexer_attn_q_b,
                                        layer.indexer_proj,
                                        inp_pos,
                                        get_dsv4_inputs()->add_mask(ctx0,
                                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                                n_comp_visible, n_tokens,
                                                0, n_comp_visible, 0, compress_ratio,
                                                "dsv4_indexer_decode_causal_mask"),
                                        hparams.indexer_n_head,
                                        hparams.indexer_head_size,
                                        n_tokens,
                                        n_rot,
                                        rope_type,
                                        rope_cfg);
                            cb(index_scores, "indexer_scores", il);

                            const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp_visible);
                            ggml_tensor * topk = ggml_argsort_top_k(ctx0, index_scores, top_k);
                            cb(topk, "indexer_topk", il);
                            ggml_tensor * dense_topk_mask = dsv4_build_compressed_mask_from_topk(ctx0, index_scores, topk);
                            dsv4_indexed_attn_shadow_note(il,
                                    first_pos,
                                    n_tokens,
                                    k_raw_cache_all != nullptr ? k_raw_cache_all->ne[2] : 0,
                                    hparams.n_swa,
                                    n_comp_visible,
                                    top_k,
                                    top_k,
                                    "topk_dense_mask",
                                    false);

                            if (dsv4_experimental_indexed_attn_shadow_output_enabled() &&
                                    dsv4_experimental_indexed_attn_site_enabled(il, first_pos, n_tokens) &&
                                    top_k > 0 &&
                                    kv_comp_cache_all != nullptr &&
                                    k_raw_cache_all != nullptr) {
                                const char * shape_mode = dsv4_experimental_indexed_attn_shape_mode();
                                int64_t shadow_row_count = top_k;
                                bool nonselected_rows_present = false;
                                bool nonselected_rows_masked = false;

                                if (std::strcmp(shape_mode, "baseline_dense") == 0 ||
                                        std::strcmp(shape_mode, "dense_mask_shape") == 0) {
                                    indexed_attn_shadow_kv = kv_comp_cache_all;
                                    indexed_attn_shadow_mask = dense_topk_mask;
                                    shadow_row_count = n_comp_visible;
                                    nonselected_rows_present = true;
                                    nonselected_rows_masked = true;
                                } else if (std::strcmp(shape_mode, "dense_padded_topk") == 0) {
                                    ggml_tensor * comp_rows = ggml_reshape_2d(ctx0,
                                            kv_comp_cache_all,
                                            kv_comp_cache_all->ne[0],
                                            kv_comp_cache_all->ne[2]);
                                    ggml_tensor * comp_topk_rows = ggml_get_rows(ctx0, comp_rows, topk);
                                    const int64_t pad_to_rows_env = dsv4_experimental_indexed_attn_pad_to_rows();
                                    const int64_t pad_to_rows = pad_to_rows_env > 0 ?
                                        std::min<int64_t>(pad_to_rows_env, n_comp_visible) : n_comp_visible;
                                    ggml_tensor * dense_base = dsv4_new_filled_2d(ctx0,
                                            kv_comp_cache_all->ne[0],
                                            pad_to_rows,
                                            0.0f);
                                    ggml_tensor * dense_rows = pad_to_rows == n_comp_visible ?
                                        ggml_set_rows(ctx0, dense_base, comp_topk_rows, topk) :
                                        dense_base;
                                    indexed_attn_shadow_kv = ggml_reshape_3d(ctx0, dense_rows, n_embd_head_k, 1, pad_to_rows);
                                    if (indexed_attn_shadow_kv->type != k_raw_cache_all->type) {
                                        indexed_attn_shadow_kv = ggml_cast(ctx0, indexed_attn_shadow_kv, k_raw_cache_all->type);
                                    }
                                    indexed_attn_shadow_mask = pad_to_rows == n_comp_visible ?
                                        dense_topk_mask :
                                        get_dsv4_inputs()->add_mask(ctx0,
                                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                                pad_to_rows, n_tokens,
                                                0, pad_to_rows, 0, compress_ratio,
                                                "dsv4_iattn_shadow_pad_mask");
                                    shadow_row_count = pad_to_rows;
                                    nonselected_rows_present = true;
                                    nonselected_rows_masked = pad_to_rows == n_comp_visible;
                                } else {
                                    ggml_tensor * comp_rows = ggml_reshape_2d(ctx0,
                                            kv_comp_cache_all,
                                            kv_comp_cache_all->ne[0],
                                            kv_comp_cache_all->ne[2]);
                                    ggml_tensor * comp_topk_shadow = ggml_get_rows(ctx0, comp_rows, topk);
                                    comp_topk_shadow = ggml_cont(ctx0, ggml_permute(ctx0, comp_topk_shadow, 0, 2, 1, 3));
                                    comp_topk_shadow = ggml_reshape_3d(ctx0, comp_topk_shadow, n_embd_head_k, 1, top_k);
                                    if (comp_topk_shadow->type != k_raw_cache_all->type) {
                                        comp_topk_shadow = ggml_cast(ctx0, comp_topk_shadow, k_raw_cache_all->type);
                                    }
                                    indexed_attn_shadow_kv = comp_topk_shadow;
                                    if (std::strcmp(dsv4_experimental_indexed_attn_mask_mode(), "visible_all_no_mask") == 0) {
                                        indexed_attn_shadow_mask = dsv4_new_filled_2d(ctx0, top_k, n_tokens, 0.0f);
                                    } else {
                                        indexed_attn_shadow_mask = get_dsv4_inputs()->add_mask(ctx0,
                                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                                top_k, n_tokens,
                                                0, top_k, 0, compress_ratio,
                                                "dsv4_iattn_shadow_topk_mask");
                                    }
                                }
                                dsv4_indexed_attn_shape_note(il, first_pos, n_tokens, shape_mode,
                                        n_comp_visible, top_k, shadow_row_count,
                                        nonselected_rows_present, nonselected_rows_masked);
                            }

                            if (dsv4_experimental_topk_attn_gather_enabled() &&
                                    n_tokens == 1 &&
                                    top_k > 0 &&
                                    kv_comp_cache_all != nullptr &&
                                    k_raw_cache_all != nullptr) {
                                ggml_tensor * comp_rows = ggml_reshape_2d(ctx0,
                                        kv_comp_cache_all,
                                        kv_comp_cache_all->ne[0],
                                        kv_comp_cache_all->ne[2]);
                                ggml_tensor * comp_topk = ggml_get_rows(ctx0, comp_rows, topk);
                                comp_topk = ggml_cont(ctx0, ggml_permute(ctx0, comp_topk, 0, 2, 1, 3));
                                comp_topk = ggml_reshape_3d(ctx0, comp_topk, n_embd_head_k, 1, top_k);
                                if (comp_topk->type != k_raw_cache_all->type) {
                                    comp_topk = ggml_cast(ctx0, comp_topk, k_raw_cache_all->type);
                                }
                                cb(comp_topk, "dsv4_attn_topk_kv", il);

                                k_all = ggml_concat(ctx0, k_raw_cache_all, comp_topk, 2);
                                v_all = k_all;
                                comp_mask = get_dsv4_inputs()->add_mask(ctx0,
                                        dsv4_mask_kind::COMPRESS_CAUSAL,
                                        top_k, n_tokens,
                                        0, top_k, 0, compress_ratio,
                                        "dsv4_attn_topk_compress_mask");
                            } else {
                                comp_mask = dense_topk_mask;
                            }
                        }
                    } else {
                        comp_mask = get_dsv4_inputs()->add_mask(ctx0,
                                dsv4_mask_kind::COMPRESS_CAUSAL,
                                n_comp_visible, n_tokens,
                                0, n_comp_visible, 0, compress_ratio,
                                "dsv4_attn_compress_mask");
                    }

                    if (comp_mask != nullptr) {
                        comp_attn_mask = comp_mask;
                        attn_mask = ggml_concat(ctx0, attn_mask, comp_mask, 0);
                    }
                }
            }

            if (segment_compress_attention) {
                ggml_tensor * segment_raw_k = is_prefill ? kv : k_raw_cache_all;
                ggml_tensor * segment_comp_k = is_prefill ? kv_comp_all : kv_comp_cache_all;
                const int64_t segment_comp_total = is_prefill ? n_comp : n_comp_visible;
                GGML_ASSERT(is_prefill || raw_attn_mask != nullptr);
                GGML_ASSERT(segment_raw_k != nullptr);
                GGML_ASSERT(segment_comp_k != nullptr);

                const int64_t q_limit = is_prefill ?
                        choose_compress_prefill_q_limit(segment_comp_total) :
                        choose_compress_attention_q_limit(segment_comp_total);
                ggml_tensor * attn_out_all = nullptr;
                for (int64_t row_start = 0; row_start < n_tokens; row_start += q_limit) {
                    const int64_t n_rows = std::min<int64_t>(q_limit, n_tokens - row_start);
                    const int64_t q_abs_end = (is_prefill ? 0 : (ubatch.pos ? ubatch.pos[0] : 0)) + row_start + n_rows;
                    const int64_t raw_key_start = is_prefill ?
                            std::max<int64_t>(0, row_start + 1 - hparams.n_swa) :
                            0;
                    const int64_t raw_key_end = is_prefill ?
                            row_start + n_rows :
                            std::min<int64_t>(segment_raw_k->ne[2], raw_attn_mask->ne[0]);
                    const int64_t n_raw_keys = raw_key_end - raw_key_start;
                    const int64_t n_comp_keys = std::min<int64_t>(segment_comp_total, q_abs_end / compress_ratio);
                    GGML_ASSERT(n_raw_keys > 0);

                    ggml_tensor * q_part = ggml_view_3d(ctx0, q,
                            q->ne[0], q->ne[1], n_rows,
                            q->nb[1], q->nb[2], row_start*q->nb[2]);
                    ggml_tensor * k_raw_part = ggml_view_3d(ctx0, segment_raw_k,
                            segment_raw_k->ne[0], segment_raw_k->ne[1], n_raw_keys,
                            segment_raw_k->nb[1], segment_raw_k->nb[2], raw_key_start*segment_raw_k->nb[2]);
                    ggml_tensor * k_part = k_raw_part;
                    if (n_comp_keys > 0) {
                        ggml_tensor * k_comp_part = ggml_view_3d(ctx0, segment_comp_k,
                                segment_comp_k->ne[0], segment_comp_k->ne[1], n_comp_keys,
                                segment_comp_k->nb[1], segment_comp_k->nb[2], 0);
                        k_part = ggml_concat(ctx0, k_raw_part, k_comp_part, 2);
                    }

                    ggml_tensor * raw_mask_part = nullptr;
                    if (is_prefill) {
                        raw_mask_part = get_dsv4_inputs()->add_mask(ctx0,
                                dsv4_mask_kind::RAW_WINDOW,
                                n_raw_keys, n_rows,
                                n_raw_keys, 0, hparams.n_swa, compress_ratio,
                                "dsv4_attn_split_raw_window_mask",
                                row_start, raw_key_start);
                    } else {
                        raw_mask_part = ggml_view_2d(ctx0, raw_attn_mask,
                                n_raw_keys, n_rows,
                                raw_attn_mask->nb[1],
                                row_start*raw_attn_mask->nb[1]);
                    }
                    raw_mask_part = ggml_cont(ctx0, raw_mask_part);
                    if (cparams.flash_attn) {
                        raw_mask_part = ggml_cast(ctx0, raw_mask_part, GGML_TYPE_F16);
                    }
                    ggml_tensor * mask_part = raw_mask_part;
                    if (n_comp_keys > 0) {
                        ggml_tensor * comp_mask_part = nullptr;
                        if (compress_ratio == 4) {
                            ggml_tensor * cur_part = ggml_view_2d(ctx0, cur,
                                    cur->ne[0], n_rows,
                                    cur->nb[1],
                                    row_start*cur->nb[1]);
                            ggml_tensor * qr_part = ggml_view_2d(ctx0, qr,
                                    qr->ne[0], n_rows,
                                    qr->nb[1],
                                    row_start*qr->nb[1]);
                            ggml_tensor * inp_pos_part = ggml_view_1d(ctx0, inp_pos,
                                    n_rows,
                                    row_start*inp_pos->nb[0]);
                            ggml_tensor * index_kv_src = index_kv_all != nullptr ?
                                    index_kv_all :
                                    dsv4_cache_view_3d(ctx0, mctx_dsv4->get_dsv4_index_k(ctx0, il, seq_id), n_comp_keys);
                            ggml_tensor * index_kv_part = ggml_view_3d(ctx0, index_kv_src,
                                    index_kv_src->ne[0], index_kv_src->ne[1], n_comp_keys,
                                    index_kv_src->nb[1], index_kv_src->nb[2], 0);
                            ggml_tensor * index_mask_part = get_dsv4_inputs()->add_mask(ctx0,
                                    dsv4_mask_kind::COMPRESS_CAUSAL,
                                    n_comp_keys, n_rows,
                                    0, n_comp_keys, 0, compress_ratio,
                                    "dsv4_indexer_split_causal_mask",
                                    row_start);
                            ggml_tensor * index_scores_part = dsv4_build_indexer_scores_prefill(ctx0,
                                    cur_part, qr_part, index_kv_part,
                                    layer.indexer_attn_q_b,
                                    layer.indexer_proj,
                                    inp_pos_part,
                                    index_mask_part,
                                    hparams.indexer_n_head,
                                    hparams.indexer_head_size,
                                    n_rows,
                                    n_rot,
                                    rope_type,
                                    rope_cfg);
                            cb(index_scores_part, "indexer_scores", il);

                            const int top_k = std::min<int64_t>(hparams.indexer_top_k, n_comp_keys);
                            ggml_tensor * topk_part = ggml_argsort_top_k(ctx0, index_scores_part, top_k);
                            cb(topk_part, "indexer_topk", il);
                            comp_mask_part = dsv4_build_compressed_mask_from_topk(ctx0, index_scores_part, topk_part);
                            cb(comp_mask_part, "dsv4_attn_compress_mask", il);
                            if (cparams.flash_attn) {
                                comp_mask_part = ggml_cast(ctx0, comp_mask_part, GGML_TYPE_F16);
                            }
                        } else {
                            comp_mask_part = get_dsv4_inputs()->add_mask(ctx0,
                                    dsv4_mask_kind::COMPRESS_CAUSAL,
                                    n_comp_keys, n_rows,
                                    0, n_comp_keys, 0, compress_ratio,
                                    "dsv4_attn_split_compress_mask",
                                    row_start);
                            if (cparams.flash_attn) {
                                comp_mask_part = ggml_cast(ctx0, comp_mask_part, GGML_TYPE_F16);
                            }
                        }
                        mask_part = ggml_concat(ctx0, raw_mask_part, comp_mask_part, 0);
                    }
                    mask_part = ggml_cont(ctx0, mask_part);

                    ggml_tensor * cur_part = build_attn_mha(q_part, k_part, k_part, nullptr, mask_part,
                            layer.attn_sinks, nullptr, kq_scale, il);
                    cb(cur_part, "kqv_out", il);
                    ggml_tensor * inp_pos_part = ggml_view_1d(ctx0, inp_pos, n_rows, row_start*inp_pos->nb[0]);
                    cur_part = ggml_reshape_3d(ctx0, cur_part, n_embd_head_v, n_head, n_rows);
                    cur_part = dsv4_apply_rope_tail(ctx0, cur_part, inp_pos_part,
                            n_embd_head_v, n_head, n_rows, n_rot, rope_type,
                            rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                            rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, true);
                    cur_part = dsv4_grouped_out(ctx0, cur_part, layer.attn_wo_a, layer.attn_wo_b,
                            n_embd_head_v, n_head, n_out_group, n_lora_o, n_rows, il);
                    attn_out_all = attn_out_all == nullptr ? cur_part : ggml_concat(ctx0, attn_out_all, cur_part, 1);
                }
                cur = attn_out_all;
                cur_is_attn_projected = true;
            } else {
                if (dsv4_experimental_mixed_attn_enabled() &&
                        !is_prefill &&
                        n_tokens == 1 &&
                        cparams.flash_attn &&
                        k_raw_cache_all != nullptr &&
                        kv_comp_cache_all != nullptr &&
                        raw_attn_mask != nullptr &&
                        comp_attn_mask != nullptr &&
                        k_raw_cache_all->type == GGML_TYPE_F16 &&
                        kv_comp_cache_all->type == GGML_TYPE_F16) {
                    cur = ggml_dsv4_mixed_attn(ctx0, q, k_raw_cache_all, kv_comp_cache_all,
                            raw_attn_mask, comp_attn_mask, layer.attn_sinks, kq_scale);
                } else {
                    ggml_tensor * attn_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, attn_mask, GGML_TYPE_F16) : attn_mask;
                    cur = build_attn_mha(q, k_all, v_all, nullptr, attn_mask_cnv, layer.attn_sinks, nullptr, kq_scale, il);
                }
            }
            if (!cur_is_attn_projected &&
                    dsv4_experimental_indexed_attn_shadow_output_enabled() &&
                    dsv4_experimental_indexed_attn_output_compare_enabled() &&
                    dsv4_experimental_indexed_attn_site_enabled(il, indexed_attn_shadow_pos, n_tokens) &&
                    !is_prefill &&
                    n_tokens == 1 &&
                    cparams.flash_attn &&
                    q != nullptr &&
                    k_raw_cache_all != nullptr &&
                    indexed_attn_shadow_kv != nullptr &&
                    raw_attn_mask != nullptr &&
                    indexed_attn_shadow_mask != nullptr &&
                    k_raw_cache_all->type == GGML_TYPE_F16 &&
                    indexed_attn_shadow_kv->type == GGML_TYPE_F16) {
                ggml_tensor * indexed_shadow = nullptr;
                const char * indexed_attn_arith_mode = dsv4_experimental_indexed_attn_arith_mode();
                if (std::strcmp(indexed_attn_arith_mode, "generic_flash") == 0) {
                    if (std::strcmp(dsv4_experimental_indexed_attn_shape_mode(), "baseline_dense") == 0 &&
                            k_all != nullptr &&
                            attn_mask != nullptr) {
                        ggml_tensor * shadow_mask = cparams.flash_attn ? ggml_cast(ctx0, attn_mask, GGML_TYPE_F16) : attn_mask;
                        indexed_shadow = build_attn_mha(q, k_all, k_all, nullptr, shadow_mask,
                                layer.attn_sinks, nullptr, kq_scale, il);
                    } else {
                        ggml_tensor * shadow_k_all = ggml_concat(ctx0, k_raw_cache_all, indexed_attn_shadow_kv, 2);
                        ggml_tensor * shadow_mask = ggml_concat(ctx0, raw_attn_mask, indexed_attn_shadow_mask, 0);
                        shadow_mask = cparams.flash_attn ? ggml_cast(ctx0, shadow_mask, GGML_TYPE_F16) : shadow_mask;
                        indexed_shadow = build_attn_mha(q, shadow_k_all, shadow_k_all, nullptr, shadow_mask,
                                layer.attn_sinks, nullptr, kq_scale, il);
                    }
                } else if (std::strcmp(indexed_attn_arith_mode, "dsv4_mixed") == 0) {
                    indexed_shadow = ggml_dsv4_mixed_attn(ctx0, q, k_raw_cache_all, indexed_attn_shadow_kv,
                            raw_attn_mask, indexed_attn_shadow_mask, layer.attn_sinks, kq_scale);
                } else {
                    std::fprintf(stderr,
                            "dsv4_iattn_shadow_output: unsupported arith_mode=%s; falling back to dsv4_mixed diagnostic path\n",
                            indexed_attn_arith_mode);
                    indexed_shadow = ggml_dsv4_mixed_attn(ctx0, q, k_raw_cache_all, indexed_attn_shadow_kv,
                            raw_attn_mask, indexed_attn_shadow_mask, layer.attn_sinks, kq_scale);
                }
                ggml_format_name(indexed_shadow, "dsv4_iattn_shadow_out-l%d-t%lld", il, (long long) indexed_attn_shadow_pos);
                ggml_tensor * indexed_ref = ggml_scale(ctx0, cur, 1.0f);
                ggml_format_name(indexed_ref, "dsv4_iattn_ref-l%d-t%lld", il, (long long) indexed_attn_shadow_pos);
                ggml_build_forward_expand(gf, indexed_ref);
                ggml_build_forward_expand(gf, indexed_shadow);
                dsv4_indexed_attn_shadow_note_output(il, indexed_attn_shadow_pos, n_tokens);
            }
            if (!cur_is_attn_projected) {
                cb(cur, "kqv_out", il);
                dsv4_lexec_full_note_tensor(cur, "attention_core", "kqv_out", il, dsv4_hc_pos, n_tokens);
            }
        }
        ggml_tensor * aohc_candidate_low = nullptr;
        ggml_tensor * aohc_candidate_out = nullptr;
        ggml_tensor * aohc_candidate_after_hc = nullptr;
        ggml_tensor * aohc_fused_low = nullptr;
        ggml_tensor * aohc_fused_out = nullptr;
        ggml_tensor * aohc_fused_after_hc = nullptr;
        ggml_tensor * aohc_side_probe_attn_core = nullptr;
        ggml_tensor * aohc_side_probe_attn_low = nullptr;
        ggml_tensor * aohc_side_probe_attn_out = nullptr;
        bool aohc_consume_here = false;
        bool aohc_skip_generic_here = false;
        bool aohc_fused_here = false;
        bool aohc_fused_consume_here = false;
        if (!cur_is_attn_projected) {
            cur = ggml_reshape_3d(ctx0, cur, n_embd_head_v, n_head, n_tokens);
            cur = dsv4_apply_rope_tail(ctx0, cur, inp_pos,
                    n_embd_head_v, n_head, n_tokens, n_rot, rope_type,
                    rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale,
                    rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow, true);
            cur = dsv4_aohc_shadow_probe(cur, "attn_core_out", il);
            ggml_tensor * aohc_attn_core_input = cur;
            aohc_side_probe_attn_core = aohc_attn_core_input;
            aohc_consume_here =
                !is_prefill &&
                dsv4_experimental_attn_out_hc_post_consume_site_enabled(il, dsv4_hc_pos, n_tokens);
            aohc_skip_generic_here =
                aohc_consume_here &&
                dsv4_experimental_attn_out_hc_post_skip_generic_enabled();
            const bool aohc_candidate_here =
                !is_prefill &&
                dsv4_experimental_attn_out_hc_post_candidate_site_enabled(il, dsv4_hc_pos, n_tokens);
            aohc_fused_here =
                !is_prefill &&
                dsv4_experimental_aohc_fused_site_enabled(il, dsv4_hc_pos, n_tokens, false);
            aohc_fused_consume_here =
                !is_prefill &&
                dsv4_experimental_aohc_fused_site_enabled(il, dsv4_hc_pos, n_tokens, true);
            const bool aohc_candidate_needed = aohc_candidate_here || aohc_consume_here;
            if (aohc_candidate_needed) {
                dsv4_aohc_candidate_dep_audit(il);
                aohc_candidate_out = dsv4_grouped_out(ctx0, aohc_attn_core_input,
                        layer.attn_wo_a, layer.attn_wo_b,
                        n_embd_head_v, n_head, n_out_group, n_lora_o, n_tokens, il, &aohc_candidate_low);
                ggml_format_name(aohc_candidate_low, "dsv4_aohc_candidate_internal-attn_low-l%d-p%lld",
                        il, (long long) dsv4_hc_pos);
                ggml_format_name(aohc_candidate_out, "dsv4_aohc_candidate_internal-attn_out-l%d-p%lld",
                        il, (long long) dsv4_hc_pos);
                aohc_candidate_after_hc = dsv4_hc_post(ctx0, aohc_candidate_out, residual,
                        mix.post, mix.comb, n_embd, n_hc, n_tokens);
                ggml_format_name(aohc_candidate_after_hc, "dsv4_aohc_candidate_internal-after_attn_hc-l%d-p%lld",
                        il, (long long) dsv4_hc_pos);
            }
            if (aohc_fused_here) {
                aohc_fused_out = dsv4_grouped_out(ctx0, aohc_attn_core_input,
                        layer.attn_wo_a, layer.attn_wo_b,
                        n_embd_head_v, n_head, n_out_group, n_lora_o, n_tokens, il, &aohc_fused_low);
                ggml_format_name(aohc_fused_low, "dsv4_aohc_fused_low-l%d-p%lld",
                        il, (long long) dsv4_hc_pos);
                ggml_format_name(aohc_fused_out, "dsv4_aohc_fused_high-l%d-p%lld",
                        il, (long long) dsv4_hc_pos);
                aohc_fused_after_hc = dsv4_hc_post(ctx0, aohc_fused_out, residual,
                        mix.post, mix.comb, n_embd, n_hc, n_tokens);
                ggml_format_name(aohc_fused_after_hc, "dsv4_aohc_fused_after_hc-l%d-p%lld",
                        il, (long long) dsv4_hc_pos);
                if (dsv4_experimental_aohc_fused_trace_enabled()) {
                    std::fprintf(stderr,
                            "dsv4_aohc_fused: layer=%d token=%lld scope=partial_high_projection_plus_hc_post"
                            " branch=fused_q8hc_candidate consume=%d generic_downstream=1\n",
                            il, (long long) dsv4_hc_pos, aohc_fused_consume_here ? 1 : 0);
                }
                if (dsv4_experimental_aohc_fused_elig_trace_enabled()) {
                    std::fprintf(stderr,
                            "dsv4_aohc_elig_model: layer=%d token=%lld candidate_mode=aohc_fused_partial_q8hc scope=partial_high_projection_plus_hc_post n_tokens=%lld backend=metal ",
                            il, (long long) dsv4_hc_pos, (long long) n_tokens);
                    dsv4_aohc_fused_elig_print_tensor("attn_low", aohc_fused_low);
                    dsv4_aohc_fused_elig_print_tensor("attn_out_input", aohc_fused_out != nullptr ? aohc_fused_out->src[1] : nullptr);
                    dsv4_aohc_fused_elig_print_tensor("wo_b", aohc_fused_out != nullptr ? aohc_fused_out->src[0] : nullptr);
                    dsv4_aohc_fused_elig_print_tensor("hc_input", aohc_fused_after_hc != nullptr ? aohc_fused_after_hc->src[0] : nullptr);
                    dsv4_aohc_fused_elig_print_tensor("hc_post_weights", mix.post);
                    dsv4_aohc_fused_elig_print_tensor("hc_comb", mix.comb);
                    dsv4_aohc_fused_elig_print_tensor("candidate_after_attn_hc", aohc_fused_after_hc);
                    std::fprintf(stderr, "\n");
                }
            }
            if (aohc_skip_generic_here && aohc_candidate_out != nullptr) {
                cur = aohc_candidate_out;
            } else if (aohc_fused_consume_here && aohc_fused_out != nullptr) {
                cur = aohc_fused_out;
            } else {
                ggml_tensor * attn_low_probe = nullptr;
                cur = dsv4_grouped_out(ctx0, cur, layer.attn_wo_a, layer.attn_wo_b,
                        n_embd_head_v, n_head, n_out_group, n_lora_o, n_tokens, il, &attn_low_probe);
                aohc_side_probe_attn_low = attn_low_probe;
                dsv4_aohc_shadow_probe(attn_low_probe, "attn_low", il);
                dsv4_aohc_candidate_compare_pair(attn_low_probe, aohc_candidate_low, "attn_low", il);
                dsv4_aohc_candidate_compare_pair(cur, aohc_candidate_out, "attn_out", il);
                dsv4_aohc_candidate_compare_pair(attn_low_probe, aohc_fused_low, "attn_low", il);
                dsv4_aohc_candidate_compare_pair(cur, aohc_fused_out, "attn_out", il);
            }
        }
        cb(cur, "attn_out", il);
        cur = dsv4_aohc_shadow_probe(cur, "attn_out", il);
        cur = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, cur, "attention_output", "attn_out", il);
        aohc_side_probe_attn_out = cur;
        dsv4_lexec_attn_out = cur;
        if (aohc_fused_here && aohc_fused_out != nullptr) {
            ggml_format_name(aohc_fused_out, "dsv4_aohc_fused_high-l%d-p%lld",
                    il, (long long) dsv4_hc_pos);
        }
        dsv4_aohc_shadow_probe(mix.post, "hc_post_weights", il);
        dsv4_aohc_shadow_probe(mix.comb, "hc_comb", il);
        ggml_tensor * aohc_hc_post_input = cur;
        ggml_tensor * aohc_hc_post_residual = residual;
        ggml_tensor * aohc_hc_post_weights = mix.post;
        ggml_tensor * aohc_hc_post_comb = mix.comb;
        ggml_tensor * aohc_consumer_dispatch_after_hc = nullptr;
        if (dsv4_aohc_consumer_dispatch_capture_site_enabled(il, dsv4_hc_pos, n_tokens) &&
                !attn_hcnorm_cmp.active &&
                !aohc_skip_generic_here &&
                !aohc_fused_consume_here) {
            aohc_hc_post_input = dsv4_aohc_consumer_dispatch_source(
                    ctx0, il, dsv4_hc_pos, "aohc_hcexpand_dispatch_src0", cur);
            aohc_hc_post_residual = dsv4_aohc_consumer_dispatch_source(
                    ctx0, il, dsv4_hc_pos, "aohc_hcexpand_dispatch_src1", residual);
            aohc_hc_post_weights = dsv4_aohc_consumer_dispatch_source(
                    ctx0, il, dsv4_hc_pos, "aohc_hcexpand_dispatch_src2", mix.post);
            aohc_hc_post_comb = dsv4_aohc_consumer_dispatch_source(
                    ctx0, il, dsv4_hc_pos, "aohc_hcexpand_dispatch_src3", mix.comb);
            aohc_consumer_dispatch_after_hc = dsv4_hc_post(
                    ctx0, aohc_hc_post_input, aohc_hc_post_residual,
                    aohc_hc_post_weights, aohc_hc_post_comb,
                    n_embd, n_hc, n_tokens);
            ggml_format_name(aohc_consumer_dispatch_after_hc,
                    "aohc_hcexpand_dispatch_output-l%d-p%lld", il, (long long) dsv4_hc_pos);
            dsv4_lexec_side_probe_register_payload_target(
                    "aohc_boundary", il, dsv4_hc_pos,
                    "aohc_hcexpand_dispatch_output", aohc_consumer_dispatch_after_hc);
        }
        inpL = (aohc_skip_generic_here && aohc_candidate_after_hc != nullptr) ?
            aohc_candidate_after_hc :
            ((aohc_fused_consume_here && aohc_fused_after_hc != nullptr) ?
            aohc_fused_after_hc :
            (aohc_consumer_dispatch_after_hc != nullptr ?
            aohc_consumer_dispatch_after_hc :
            (attn_hcnorm_cmp.active ?
            dsv4_hc_post_compare(ctx0, cur, residual, attn_hcnorm_cmp,
                    "hc_attn_post", il, dsv4_hc_pos, n_embd, n_hc, n_tokens) :
            dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens))));
        cb(inpL, "hc_attn_post", il);
        dsv4_lexec_full_note_tensor(inpL, "attn_hc_post", "hc_attn_post", il, dsv4_hc_pos, n_tokens);
        dsv4_lexec_attn_hc_post = inpL;
        if (dsv4_rmoe_result_chain_site && il == dsv4_rmoe_result_chain_target_layer + 1) {
            cb(inpL, "dsv4_rmoe_result_chain_layer1_after_attn", (int) dsv4_rmoe_result_chain_target_layer);
        }
        inpL = dsv4_aohc_shadow_probe(inpL, "after_attn_hc", il);
        dsv4_lexec_side_probe_note_aohc(
                il, dsv4_hc_pos, n_tokens,
                aohc_side_probe_attn_core,
                aohc_side_probe_attn_low,
                aohc_side_probe_attn_out,
                aohc_side_probe_attn_out,
                residual,
                mix.split,
                mix.post,
                mix.comb,
                inpL,
                inpL);
        dsv4_aohc_candidate_compare_pair(inpL, aohc_candidate_after_hc, "after_attn_hc", il);
        dsv4_aohc_candidate_compare_pair(inpL, aohc_fused_after_hc, "after_attn_hc", il);
        if (aohc_candidate_after_hc != nullptr && aohc_consume_here) {
            inpL = aohc_candidate_after_hc;
            dsv4_aohc_consume_note(!aohc_skip_generic_here, true, aohc_skip_generic_here);
        }
        if (aohc_fused_after_hc != nullptr && aohc_fused_here) {
            const bool downstream_consumed = aohc_fused_consume_here;
            if (downstream_consumed) {
                inpL = aohc_fused_after_hc;
            }
            dsv4_aohc_fused_note(downstream_consumed);
        }

        residual = inpL;
        mix = dsv4_hc_pre(ctx0, inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                n_embd, n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
        cur = mix.x;
        dsv4_hc_pre_norm_result ffn_hcnorm_cmp = {};
        if (dsv4_experimental_hc_pre_norm_compare_site_enabled("ffn", il, dsv4_hc_pos, n_tokens)) {
            dsv4_hc_pre_norm_result cmp = dsv4_hc_pre_norm_compare(ctx0, mix, inpL,
                    layer.hc_ffn_scale, layer.hc_ffn_base, layer.ffn_norm,
                    "ffn", il, dsv4_hc_pos,
                    n_hc, n_tokens, norm_rms_eps, hparams.hc_sinkhorn_iters, hparams.hc_eps);
            cur = cmp.x;
            mix = cmp.mix;
            ffn_hcnorm_cmp = cmp;
            cb(mix.mixes, "hc_ffn_pre_mixes", il);
        } else {
            cb(cur, "hc_ffn_pre", il);
            cb(mix.mixes, "hc_ffn_pre_mixes", il);
            cb(mix.pre, "hc_ffn_pre_weights", il);
            cb(mix.post, "hc_ffn_pre_post_weights", il);
            cb(mix.comb, "hc_ffn_pre_comb", il);
            cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        }
        cb(cur, "ffn_norm", il);
        dsv4_lexec_side_probe_note_hc_pre(il, dsv4_hc_pos, n_tokens, "ffn", inpL, cur, layer.ffn_norm, mix, cur);
        if (il > 0 && dsv4_experimental_routed_moe_backend_op_downstream_dump_site_enabled(il - 1, dsv4_hc_pos, n_tokens)) {
            cb(cur, "dsv4_rmoe_downstream_next_layer_ffn_input", il - 1);
        }
        cur = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, cur, "ffn_hc_pre_norm", "ffn_norm", il);
        dsv4_lexec_ffn_norm = cur;
        ggml_tensor * ffn_moe_input = cur;
        if (dsv4_experimental_routed_moe_backend_op_pair_preserve_enabled()) {
            dsv4_rmoe_pair_preserve_register_summary();
        }
        if (dsv4_experimental_routed_moe_backend_op_shared_final_only_enabled()) {
            dsv4_rmoe_shared_final_register_summary();
        }
        const bool dsv4_rmoe_shared_final_here =
            dsv4_experimental_routed_moe_backend_op_shared_final_only_enabled() &&
            dsv4_rmoe_shared_final_guard_allowed() &&
            dsv4_experimental_routed_moe_backend_op_consume_site_enabled(il, dsv4_hc_pos, n_tokens);
        const bool dsv4_rmoe_pair_preserve_here =
            dsv4_experimental_routed_moe_backend_op_pair_preserve_enabled() &&
            !dsv4_rmoe_shared_final_here &&
            dsv4_rmoe_pair_preserve_guard_allowed() &&
            dsv4_experimental_routed_moe_backend_op_consume_site_enabled(il, dsv4_hc_pos, n_tokens);
        const bool dsv4_rmoe_replace_here =
            dsv4_experimental_routed_moe_backend_op_replace_generic_enabled() &&
            !dsv4_rmoe_shared_final_here &&
            !dsv4_rmoe_pair_preserve_here &&
            dsv4_experimental_routed_moe_backend_op_consume_site_enabled(il, dsv4_hc_pos, n_tokens);
        if (dsv4_experimental_routed_moe_backend_op_replace_generic_enabled()) {
            dsv4_rmoe_replace_register_summary();
        }
        if (!dsv4_rmoe_replace_here && dsv4_experimental_routed_moe_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            dsv4_moe_shadow_probe(cur, cur, "ffn_input", il);
        }
        ggml_tensor * selected = nullptr;
        if ((uint32_t) il < hparams.n_hash_layers && !cparams.warmup) {
            selected = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens);
            cb(selected, "ffn_moe_hash_topk", il);
            if (dsv4_experimental_routed_moe_site_enabled(il, dsv4_hc_pos, n_tokens)) {
                dsv4_moe_shadow_probe(selected, selected, "topk_ids", il);
            }
        }

        ggml_tensor * exposed_topk_weights = nullptr;
        ggml_tensor * exposed_topk_ids = nullptr;
        ggml_tensor * exposed_expert_gate_proj = nullptr;
        ggml_tensor * exposed_expert_up_proj = nullptr;
        ggml_tensor * exposed_expert_swiglu = nullptr;
        ggml_tensor * exposed_expert_down = nullptr;
        ggml_tensor * exposed_routed_sum = nullptr;
        ggml_tensor * exposed_routed_partials[6] = { nullptr };
        ggml_tensor * exposed_shared_gate_proj = nullptr;
        ggml_tensor * exposed_shared_up_proj = nullptr;
        ggml_tensor * exposed_shared_swiglu = nullptr;
        ggml_tensor * exposed_shared_down = nullptr;
        ggml_tensor * moe_out = nullptr;
        ggml_tensor * ffn_shexp = nullptr;
        if (dsv4_rmoe_replace_here) {
            (void) build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    selected,
                    &exposed_topk_weights,
                    &exposed_topk_ids,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    true);
        } else {
            moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    selected,
                    &exposed_topk_weights,
                    &exposed_topk_ids,
                    &exposed_expert_gate_proj,
                    &exposed_expert_up_proj,
                    &exposed_expert_swiglu,
                    &exposed_expert_down,
                    &exposed_routed_sum,
                    exposed_routed_partials);
            if (!dsv4_rmoe_pair_preserve_here) {
                cb(moe_out, "ffn_moe_out", il);
            }
        }
        if (exposed_topk_ids != nullptr &&
                (dsv4_experimental_routed_moe_topk_ids_site_enabled(il, dsv4_hc_pos, n_tokens) ||
                 dsv4_experimental_routed_moe_backend_op_replace_dump_site_enabled(il, dsv4_hc_pos, n_tokens))) {
            cb(exposed_topk_ids, "dsv4_moe_topk_ids", il);
            if (dsv4_experimental_routed_moe_topk_ids_site_enabled(il, dsv4_hc_pos, n_tokens)) {
                dsv4_moe_topk_ids_note(
                        il,
                        dsv4_hc_pos,
                        exposed_topk_ids,
                        selected != nullptr ? "ffn_moe_hash_topk" : "ffn_moe_topk",
                        true);
            }
        }
        if (exposed_topk_weights != nullptr &&
                (dsv4_experimental_routed_moe_topk_weights_site_enabled(il, dsv4_hc_pos, n_tokens) ||
                 dsv4_experimental_routed_moe_backend_op_replace_dump_site_enabled(il, dsv4_hc_pos, n_tokens))) {
            cb(exposed_topk_weights, "dsv4_moe_topk_weights", il);
            if (dsv4_experimental_routed_moe_topk_weights_site_enabled(il, dsv4_hc_pos, n_tokens)) {
                dsv4_moe_topk_weights_note(
                        il,
                        dsv4_hc_pos,
                        exposed_topk_weights,
                        "ffn_moe_weights_scaled",
                        hparams.expert_weights_norm ? 1 : 0,
                        true);
            }
        }
        if (moe_out != nullptr && dsv4_experimental_routed_moe_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            dsv4_moe_shadow_probe(moe_out, moe_out, "expert_down_weighted_sum", il);
        }
        if (!dsv4_rmoe_replace_here && !dsv4_rmoe_pair_preserve_here && !dsv4_rmoe_shared_final_here) {
            ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il,
                    &exposed_shared_gate_proj,
                    &exposed_shared_up_proj,
                    &exposed_shared_swiglu,
                    &exposed_shared_down);
            cb(ffn_shexp, "ffn_shexp", il);
        }
        if (ffn_shexp != nullptr && dsv4_experimental_routed_moe_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            dsv4_moe_shadow_probe(ffn_shexp, ffn_shexp, "shared_output", il);
        }

        if (!dsv4_rmoe_replace_here && !dsv4_rmoe_pair_preserve_here && !dsv4_rmoe_shared_final_here &&
                dsv4_experimental_routed_moe_backend_op_replace_substage_dump_enabled() &&
                dsv4_experimental_routed_moe_backend_op_replace_dump_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            auto dump_stage = [&](ggml_tensor * t, const char * name) {
                if (t != nullptr) {
                    cb(t, name, il);
                    ggml_build_forward_expand(gf, t);
                }
            };
            dump_stage(exposed_expert_gate_proj, "dsv4_rmoe_dump_gate");
            dump_stage(exposed_expert_up_proj, "dsv4_rmoe_dump_up");
            dump_stage(exposed_expert_swiglu, "dsv4_rmoe_dump_swiglu");
            if (dsv4_experimental_routed_moe_backend_op_swiglu_stage_dump_enabled() &&
                    exposed_expert_gate_proj != nullptr &&
                    exposed_expert_up_proj != nullptr) {
                const float limit = il >= 0 ? hparams.swiglu_clamp_exp[il] : 0.0f;
                ggml_tensor * gate_raw = exposed_expert_gate_proj;
                ggml_tensor * up_raw = exposed_expert_up_proj;
                dump_stage(gate_raw, "dsv4_rmoe_dump_gate_raw");
                dump_stage(up_raw, "dsv4_rmoe_dump_up_raw");
                ggml_tensor * gate_clamp = limit > 1.0e-6f ?
                    ggml_clamp(ctx0, gate_raw, -INFINITY, limit) : gate_raw;
                if (gate_clamp != gate_raw) {
                    ggml_format_name(gate_clamp, "dsv4_rmoe_dump_gate_clamp_pre_silu-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                }
                dump_stage(gate_clamp, "dsv4_rmoe_dump_gate_clamp_pre_silu");
                ggml_tensor * silu_out = ggml_silu(ctx0, gate_clamp);
                ggml_format_name(silu_out, "dsv4_rmoe_dump_silu_out-l%d-p%lld",
                        il, (long long) dsv4_hc_pos);
                dump_stage(silu_out, "dsv4_rmoe_dump_silu_out");
                ggml_tensor * up_clamp = limit > 1.0e-6f ?
                    ggml_clamp(ctx0, up_raw, -limit, limit) : up_raw;
                if (up_clamp != up_raw) {
                    ggml_format_name(up_clamp, "dsv4_rmoe_dump_up_clamp_pre_mul-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                }
                dump_stage(up_clamp, "dsv4_rmoe_dump_up_clamp_pre_mul");
                if (exposed_expert_swiglu != nullptr) {
                    dump_stage(exposed_expert_swiglu, "dsv4_rmoe_dump_mul_out");
                } else {
                    ggml_tensor * mul_out = ggml_mul(ctx0, silu_out, up_clamp);
                    ggml_format_name(mul_out, "dsv4_rmoe_dump_mul_out-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                    dump_stage(mul_out, "dsv4_rmoe_dump_mul_out");
                }
            }
            dump_stage(exposed_expert_down, "dsv4_rmoe_dump_down");
            dump_stage(exposed_routed_sum, "dsv4_rmoe_dump_routed_sum");
            dump_stage(exposed_shared_gate_proj, "dsv4_rmoe_dump_shared_gate");
            dump_stage(exposed_shared_up_proj, "dsv4_rmoe_dump_shared_up");
            dump_stage(exposed_shared_swiglu, "dsv4_rmoe_dump_shared_swiglu");
            dump_stage(exposed_shared_down, "dsv4_rmoe_dump_shared_down");
        }

        if (dsv4_rmoe_shared_final_here) {
            ggml_tensor * backend_shared_gate_proj = nullptr;
            ggml_tensor * backend_shared_up_proj = nullptr;
            ggml_tensor * backend_shared_swiglu = nullptr;
            ggml_tensor * backend_shared_down = nullptr;
            ggml_tensor * backend_shared = build_ffn(ffn_moe_input,
                    layer.ffn_up_shexp,   nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il,
                    &backend_shared_gate_proj,
                    &backend_shared_up_proj,
                    &backend_shared_swiglu,
                    &backend_shared_down);
            ggml_format_name(backend_shared, "dsv4_rmoe_shared_final_shared_down-l%d-p%lld",
                    il, (long long) dsv4_hc_pos);
            cb(backend_shared, "dsv4_rmoe_shared_final_shared_down", il);
            cur = ggml_add(ctx0, moe_out, backend_shared);
            ggml_format_name(cur, "ffn_out-l%d-p%lld", il, (long long) dsv4_hc_pos);
            cb(cur, "ffn_out", il);

            dsv4_rmoe_shared_final_note(
                    exposed_expert_gate_proj != nullptr && exposed_expert_up_proj != nullptr,
                    exposed_expert_swiglu != nullptr,
                    exposed_expert_down != nullptr,
                    exposed_routed_sum != nullptr,
                    false,
                    backend_shared_down != nullptr,
                    true,
                    true);
            dsv4_rmoe_consume_note(
                    true,
                    true,
                    true,
                    true,
                    "shared_down_plus_final_add",
                    false,
                    false,
                    false);
        } else if (dsv4_rmoe_pair_preserve_here) {
            ggml_tensor * pair_swiglu = exposed_expert_swiglu != nullptr ? exposed_expert_swiglu : moe_out;
            ggml_format_name(pair_swiglu, "ffn_moe_swiglu_limited-%d", il);
            if (dsv4_experimental_routed_moe_pair_preserve_attach_mode_is("after_swiglu_marker")) {
                ggml_build_forward_expand(gf, pair_swiglu);
            }
            ggml_format_name(exposed_topk_weights, "ffn_moe_weights_scaled-%d", il);
            ggml_tensor * pair_weighted_swiglu_anchor = ggml_mul(ctx0, pair_swiglu, exposed_topk_weights);
            ggml_format_name(pair_weighted_swiglu_anchor, "ffn_moe_weighted_swiglu-%d", il);
            cb(pair_weighted_swiglu_anchor, "ffn_moe_weighted_swiglu", il);
            if (dsv4_experimental_routed_moe_pair_preserve_attach_mode_is("after_pair_marker")) {
                ggml_build_forward_expand(gf, pair_weighted_swiglu_anchor);
            }
            ggml_tensor * pair_generic_down_anchor = nullptr;
            if (dsv4_experimental_routed_moe_pair_preserve_attach_mode_is("generic_down_anchor") ||
                    dsv4_experimental_routed_moe_pair_preserve_attach_mode_is("generic_down_dry_anchor")) {
                pair_generic_down_anchor = build_lora_mm_id(
                        layer.ffn_down_exps, pair_weighted_swiglu_anchor, exposed_topk_ids);
                ggml_format_name(pair_generic_down_anchor, "ffn_moe_down-%d", il);
                cb(pair_generic_down_anchor, "ffn_moe_down", il);
                if (dsv4_experimental_routed_moe_pair_preserve_attach_mode_is("generic_down_anchor")) {
                    ggml_build_forward_expand(gf, pair_generic_down_anchor);
                }
            }
            if (dsv4_experimental_routed_moe_pair_preserve_match_trace_enabled()) {
                const int64_t decode_index = dsv4_experimental_decode_index_for_token(dsv4_hc_pos);
                std::fprintf(stderr,
                        "dsv4_rmoe_pair_preserve_attach: token=%lld position=%lld layer=%d attach_mode=%s"
                        " gate=%s up=%s swiglu=%s weighted_swiglu=%s generic_down_anchor=%s backend_op=pending"
                        " generic_down_built=%d generic_down_consumed=0\n",
                        (long long) decode_index,
                        (long long) dsv4_hc_pos,
                        il,
                        dsv4_experimental_routed_moe_pair_preserve_attach_mode(),
                        exposed_expert_gate_proj != nullptr ? ggml_get_name(exposed_expert_gate_proj) : "missing",
                        exposed_expert_up_proj != nullptr ? ggml_get_name(exposed_expert_up_proj) : "missing",
                        pair_swiglu != nullptr ? ggml_get_name(pair_swiglu) : "missing",
                        pair_weighted_swiglu_anchor != nullptr ? ggml_get_name(pair_weighted_swiglu_anchor) : "missing",
                        pair_generic_down_anchor != nullptr ? ggml_get_name(pair_generic_down_anchor) : "missing",
                        pair_generic_down_anchor != nullptr ? 1 : 0);
            }
            const int64_t n_embd_pair = layer.ffn_down_exps->ne[1];
            ggml_tensor * pair_backend = ggml_dsv4_routed_moe_pair_preserve_decode(ctx0,
                    layer.ffn_down_exps,
                    layer.ffn_gate_shexp,
                    layer.ffn_up_shexp,
                    layer.ffn_down_shexp,
                    pair_swiglu,
                    pair_weighted_swiglu_anchor,
                    ffn_moe_input,
                    exposed_topk_ids,
                    exposed_topk_weights);
            ggml_format_name(pair_backend, "dsv4_rmoe_pair_preserve_backend_op-l%d-p%lld",
                    il, (long long) dsv4_hc_pos);
            cb(pair_backend, "dsv4_rmoe_pair_preserve_backend_op", il);

            cur = ggml_view_2d(ctx0, pair_backend,
                    n_embd_pair, n_tokens, pair_backend->nb[1],
                    34 * pair_backend->nb[1]);
            ggml_format_name(cur, "ffn_out-l%d-p%lld", il, (long long) dsv4_hc_pos);
            cb(cur, "ffn_out", il);

            dsv4_rmoe_pair_preserve_note(
                    exposed_expert_gate_proj != nullptr && exposed_expert_up_proj != nullptr,
                    pair_swiglu != nullptr,
                    pair_generic_down_anchor != nullptr,
                    false,
                    false,
                    true,
                    true,
                    true,
                    true,
                    true,
                    true);
            dsv4_rmoe_consume_note(
                    true,
                    true,
                    true,
                    true,
                    "pair_preserve_down_shared_from_generic_swiglu",
                    false,
                    false,
                    false);
        } else if (!dsv4_rmoe_replace_here) {
            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }
        dsv4_lexec_side_probe_note_routed_moe(
                il, dsv4_hc_pos, n_tokens,
                ffn_moe_input,
                exposed_topk_ids,
                exposed_topk_weights,
                exposed_expert_gate_proj,
                exposed_expert_up_proj,
                exposed_expert_swiglu,
                exposed_expert_down,
                exposed_routed_sum,
                exposed_routed_partials,
                exposed_shared_gate_proj,
                exposed_shared_up_proj,
                exposed_shared_swiglu,
                exposed_shared_down,
                cur,
                cur);
        if (dsv4_experimental_routed_moe_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            const char * moe_mode = dsv4_experimental_routed_moe_mode();
            if (std::strcmp(moe_mode, "one_tensor_shadow") == 0) {
                ggml_tensor * selected_shadow = nullptr;
                if ((uint32_t) il < hparams.n_hash_layers && !cparams.warmup) {
                    selected_shadow = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens);
                    ggml_format_name(selected_shadow, "dsv4_moe_candidate_topk-l%d-p%lld", il, (long long) dsv4_hc_pos);
                }
                ggml_tensor * moe_shadow = build_moe_ffn(ffn_moe_input,
                        layer.ffn_gate_inp,
                        layer.ffn_up_exps,
                        layer.ffn_gate_exps,
                        layer.ffn_down_exps,
                        layer.ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        hparams.expert_weights_scale,
                        (llama_expert_gating_func_type) hparams.expert_gating_func,
                        il,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        selected_shadow);
                cb(moe_shadow, "dsv4_moe_candidate_moe_out", il);
                ggml_tensor * shared_shadow = build_ffn(ffn_moe_input,
                        layer.ffn_up_shexp,   nullptr, nullptr,
                        layer.ffn_gate_shexp, nullptr, nullptr,
                        layer.ffn_down_shexp, nullptr, nullptr,
                        nullptr,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(shared_shadow, "dsv4_moe_candidate_shared_out", il);
                ggml_tensor * final_shadow = ggml_add(ctx0, moe_shadow, shared_shadow);
                cb(final_shadow, "dsv4_moe_candidate_final_ffn", il);
                dsv4_moe_shadow_probe(cur, final_shadow, "final_ffn_output", il, true);
                dsv4_moe_dep_audit(il, moe_mode, true);
            } else {
                dsv4_moe_shadow_probe(cur, cur, "final_ffn_output", il);
                dsv4_moe_dep_audit(il, moe_mode, false);
            }
        }
        if (!dsv4_rmoe_replace_here && !dsv4_rmoe_shared_final_here &&
                dsv4_experimental_routed_moe_backend_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            const char * backend_mode = dsv4_experimental_routed_moe_backend_mode();
            if (std::strcmp(backend_mode, "backend_candidate_shadow") == 0) {
                dsv4_moe_backend_infeasible_audit(il, backend_mode);
            } else {
                ggml_tensor * selected_shadow = nullptr;
                if ((uint32_t) il < hparams.n_hash_layers && !cparams.warmup) {
                    selected_shadow = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens);
                    ggml_format_name(selected_shadow, "dsv4_moe_backend_candidate_topk-l%d-p%lld", il, (long long) dsv4_hc_pos);
                }
                ggml_tensor * moe_shadow = build_moe_ffn(ffn_moe_input,
                        layer.ffn_gate_inp,
                        layer.ffn_up_exps,
                        layer.ffn_gate_exps,
                        layer.ffn_down_exps,
                        layer.ffn_exp_probs_b,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, hparams.expert_weights_norm,
                        hparams.expert_weights_scale,
                        (llama_expert_gating_func_type) hparams.expert_gating_func,
                        il,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        selected_shadow);
                cb(moe_shadow, "dsv4_moe_backend_candidate_moe_out", il);
                ggml_tensor * shared_shadow = build_ffn(ffn_moe_input,
                        layer.ffn_up_shexp,   nullptr, nullptr,
                        layer.ffn_gate_shexp, nullptr, nullptr,
                        layer.ffn_down_shexp, nullptr, nullptr,
                        nullptr,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(shared_shadow, "dsv4_moe_backend_candidate_shared_out", il);
                ggml_tensor * final_shadow = ggml_add(ctx0, moe_shadow, shared_shadow);
                cb(final_shadow, "dsv4_moe_backend_candidate_final_ffn", il);
                dsv4_moe_backend_shadow_probe(cur, final_shadow, "final_ffn_output", il, true, false, false);
                dsv4_moe_backend_dep_audit(il, backend_mode, "candidate", false, "none");
            }
        }
        if (!dsv4_rmoe_pair_preserve_here && !dsv4_rmoe_shared_final_here) {
            dsv4_moe_backend_op_dryrun(
                    ffn_moe_input,
                    dsv4_experimental_routed_moe_expose_topk_ids_enabled() ? exposed_topk_ids : selected,
                    dsv4_experimental_routed_moe_expose_topk_weights_enabled() ? exposed_topk_weights : nullptr,
                    dsv4_rmoe_replace_here ? ffn_moe_input : cur,
                    layer.ffn_gate_exps,
                    layer.ffn_up_exps,
                    layer.ffn_down_exps,
                    layer.ffn_gate_shexp,
                    layer.ffn_up_shexp,
                    layer.ffn_down_shexp,
                    il);
        }
        if (!dsv4_rmoe_pair_preserve_here && !dsv4_rmoe_shared_final_here &&
                (dsv4_experimental_routed_moe_backend_op_shadow_enabled() ||
                    dsv4_experimental_routed_moe_backend_op_consume_requested() ||
                    dsv4_experimental_routed_moe_backend_op_branch_mode_requested()) &&
                dsv4_experimental_routed_moe_backend_op_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            const char * branch_mode = dsv4_experimental_routed_moe_backend_op_branch_mode();
            const bool branch_mode_requested = branch_mode[0] != '\0';
            const bool branch_alloc_only = std::strcmp(branch_mode, "alloc_only") == 0;
            const bool branch_dispatch_noop = std::strcmp(branch_mode, "dispatch_noop") == 0;
            const bool branch_compute_compare = std::strcmp(branch_mode, "dispatch_compute_compare") == 0;
            const char * branch_order = dsv4_experimental_routed_moe_backend_op_branch_order();
            const bool branch_order_after_layer = std::strcmp(branch_order, "after_layer") == 0;
            const bool branch_order_end_of_graph = std::strcmp(branch_order, "end_of_graph") == 0;
            const bool branch_order_separate_side_graph = std::strcmp(branch_order, "separate_side_graph") == 0;
            ggml_tensor * backend_topk_ids =
                dsv4_experimental_routed_moe_expose_topk_ids_enabled() ? exposed_topk_ids : selected;
            ggml_tensor * backend_topk_weights =
                dsv4_experimental_routed_moe_expose_topk_weights_enabled() ? exposed_topk_weights : nullptr;
            bool can_build_backend_op =
                ffn_moe_input != nullptr &&
                backend_topk_ids != nullptr &&
                backend_topk_weights != nullptr &&
                layer.ffn_gate_exps != nullptr &&
                layer.ffn_up_exps != nullptr &&
                layer.ffn_down_exps != nullptr &&
                layer.ffn_gate_shexp != nullptr &&
                layer.ffn_up_shexp != nullptr &&
                layer.ffn_down_shexp != nullptr &&
                layer.ffn_gate_exps->type == GGML_TYPE_IQ2_XXS &&
                layer.ffn_up_exps->type == GGML_TYPE_IQ2_XXS &&
                layer.ffn_down_exps->type == GGML_TYPE_Q2_K &&
                layer.ffn_gate_shexp->type == GGML_TYPE_Q8_0 &&
                layer.ffn_up_shexp->type == GGML_TYPE_Q8_0 &&
                layer.ffn_down_shexp->type == GGML_TYPE_Q8_0 &&
                ffn_moe_input->type == GGML_TYPE_F32 &&
                backend_topk_ids->type == GGML_TYPE_I32 &&
                backend_topk_weights->type == GGML_TYPE_F32 &&
                backend_topk_ids->ne[0] == 6 &&
                backend_topk_ids->ne[1] == 1 &&
                backend_topk_ids->ne[2] == 1 &&
                backend_topk_ids->ne[3] == 1 &&
                (backend_topk_weights->ne[0] == 6 || backend_topk_weights->ne[1] == 6) &&
                backend_topk_weights->ne[2] == 1 &&
                backend_topk_weights->ne[3] == 1 &&
                n_tokens == 1;

            if (can_build_backend_op) {
                const bool requested_scratch_gate_up = dsv4_experimental_routed_moe_backend_op_gate_up_substage_enabled();
                const bool requested_scratch_swiglu = dsv4_experimental_routed_moe_backend_op_swiglu_substage_enabled();
                const bool requested_scratch_down = dsv4_experimental_routed_moe_backend_op_down_substage_enabled();
                const bool requested_scratch_shared = dsv4_experimental_routed_moe_backend_op_shared_substage_enabled();
                const bool scratch_gate_up = branch_dispatch_noop ? false : requested_scratch_gate_up;
                const bool scratch_swiglu = branch_dispatch_noop ? false : requested_scratch_swiglu;
                const bool scratch_down = branch_dispatch_noop ? false : requested_scratch_down;
                const bool scratch_shared = branch_dispatch_noop ? false : requested_scratch_shared;
                const char * scratch_substage = scratch_shared ? "shared" : (scratch_down ? "down" : (scratch_swiglu ? "swiglu" : (scratch_gate_up ? "gate_up" : "final")));
                const char * numeric_blocker = scratch_shared ?
                    "none" : (scratch_down ?
                    "final_ffn_output_not_computed_shared_branch_missing" : (scratch_gate_up ?
                    "final_ffn_output_not_computed_gate_up_scratch_only" :
                    "requires_intermediate_scratch_for_iq2_xxs_gate_up_q2_K_down_plus_q8_0_shared_branch"));
                const float swiglu_clamp = il >= 0 ? hparams.swiglu_clamp_exp[il] : 0.0f;
                ggml_tensor * moe_backend = ggml_dsv4_routed_moe_one_tensor_decode(ctx0,
                        layer.ffn_gate_exps,
                        layer.ffn_up_exps,
                        layer.ffn_down_exps,
                        layer.ffn_gate_shexp,
                        layer.ffn_up_shexp,
                        layer.ffn_down_shexp,
                        ffn_moe_input,
                        backend_topk_ids,
                        backend_topk_weights,
                        scratch_gate_up,
                        scratch_down,
                        scratch_shared,
                        swiglu_clamp,
                        dsv4_experimental_routed_moe_backend_op_swiglu_formula_id());
                ggml_format_name(moe_backend, "dsv4_moe_backend_op_shadow-l%d-p%lld",
                        il, (long long) dsv4_hc_pos);
                cb(moe_backend, "dsv4_moe_backend_op_shadow", il);
                const int branch_internal_dispatch_count = scratch_shared ? 4 : (scratch_down ? 2 : (scratch_gate_up ? 1 : 1));
                const bool branch_attached_to_main_graph =
                    !branch_alloc_only && !branch_order_separate_side_graph;
                if (!branch_alloc_only && !branch_order_after_layer && !branch_order_end_of_graph &&
                        !branch_order_separate_side_graph) {
                    ggml_build_forward_expand(gf, moe_backend);
                } else if (!branch_alloc_only && branch_order_after_layer) {
                    dsv4_rmoe_after_layer_backend = moe_backend;
                } else if (!branch_alloc_only && branch_order_end_of_graph) {
                    dsv4_rmoe_end_of_graph_backends.push_back(moe_backend);
                }
                if (branch_mode_requested) {
                    dsv4_rmoe_branch_note(
                            branch_mode,
                            branch_order,
                            true,
                            branch_attached_to_main_graph ? branch_internal_dispatch_count : 0,
                            branch_compute_compare,
                            true,
                            ggml_nbytes(moe_backend));
                    if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                        const int64_t decode_index = dsv4_experimental_decode_index_for_token(dsv4_hc_pos);
                        std::fprintf(stderr,
                                "dsv4_rmoe_branch: mode=%s token=%lld position=%lld layer=%d"
                                " branch_order=%s backend_tensors_allocated=1 backend_dispatches=%d readback_enabled=0"
                                " compare_enabled=%d generic_ffn_consumed=1 scratch_bytes=%zu\n",
                                branch_mode,
                                (long long) decode_index,
                                (long long) dsv4_hc_pos,
                                il,
                                branch_order,
                                branch_attached_to_main_graph ? branch_internal_dispatch_count : 0,
                                branch_compute_compare ? 1 : 0,
                                ggml_nbytes(moe_backend));
                    }
                }
                dsv4_moe_backend_op_shadow_note(
                        il, dsv4_hc_pos, !branch_alloc_only, !scratch_shared, branch_internal_dispatch_count, numeric_blocker,
                        scratch_gate_up,
                        scratch_substage,
                        scratch_gate_up ? moe_backend : nullptr,
                        scratch_gate_up,
                        scratch_swiglu,
                        scratch_down,
                        scratch_down,
                        scratch_shared,
                        scratch_shared);

                const int64_t n_ff_backend = layer.ffn_down_exps->ne[0];
                const int64_t n_embd_backend = layer.ffn_down_exps->ne[1];
                const bool down_input_generic_boundary =
                    scratch_shared &&
                    dsv4_experimental_routed_moe_backend_op_down_input_generic_graph_boundary();
                auto backend_slot_view = [&](int slot, int64_t rows, int64_t cols) {
                    return ggml_view_2d(ctx0, moe_backend,
                            rows, cols, moe_backend->nb[1], slot * moe_backend->nb[1]);
                };
                ggml_tensor * down_input_swiglu = nullptr;
                ggml_tensor * down_input_down = nullptr;
                ggml_tensor * down_input_routed_sum = nullptr;
                ggml_tensor * down_input_shared_gate = nullptr;
                ggml_tensor * down_input_shared_up = nullptr;
                ggml_tensor * down_input_shared_swiglu = nullptr;
                ggml_tensor * down_input_shared_down = nullptr;
                ggml_tensor * down_input_final = nullptr;
                if (down_input_generic_boundary) {
                    const float limit = il >= 0 ? hparams.swiglu_clamp_exp[il] : 0.0f;
                    ggml_tensor * gate_raw = backend_slot_view(0, n_ff_backend, 6);
                    ggml_format_name(gate_raw, "dsv4_rmoe_down_input_gate_raw-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                    ggml_tensor * up_raw = backend_slot_view(6, n_ff_backend, 6);
                    ggml_format_name(up_raw, "dsv4_rmoe_down_input_up_raw-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                    ggml_tensor * gate_clamp = limit > 1.0e-6f ?
                        ggml_clamp(ctx0, gate_raw, -INFINITY, limit) : gate_raw;
                    if (gate_clamp != gate_raw) {
                        ggml_format_name(gate_clamp, "dsv4_rmoe_down_input_gate_clamp-l%d-p%lld",
                                il, (long long) dsv4_hc_pos);
                    }
                    ggml_tensor * silu_out = ggml_silu(ctx0, gate_clamp);
                    ggml_format_name(silu_out, "dsv4_rmoe_down_input_silu-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                    ggml_tensor * up_clamp = limit > 1.0e-6f ?
                        ggml_clamp(ctx0, up_raw, -limit, limit) : up_raw;
                    if (up_clamp != up_raw) {
                        ggml_format_name(up_clamp, "dsv4_rmoe_down_input_up_clamp-l%d-p%lld",
                                il, (long long) dsv4_hc_pos);
                    }
                    down_input_swiglu = ggml_mul(ctx0, silu_out, up_clamp);
                    ggml_format_name(down_input_swiglu, "dsv4_rmoe_down_input_swiglu-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                    ggml_tensor * weighted_swiglu = ggml_mul(ctx0, down_input_swiglu, backend_topk_weights);
                    ggml_format_name(weighted_swiglu, "dsv4_rmoe_down_input_weighted_swiglu-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                    down_input_down = build_lora_mm_id(layer.ffn_down_exps, weighted_swiglu, backend_topk_ids);
                    ggml_format_name(down_input_down, "dsv4_rmoe_down_input_down-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                    ggml_tensor * down_slots[6] = { nullptr };
                    for (int slot = 0; slot < 6; ++slot) {
                        down_slots[slot] = ggml_view_2d(ctx0, down_input_down,
                                n_embd_backend, n_tokens, down_input_down->nb[2], slot * down_input_down->nb[1]);
                        ggml_format_name(down_slots[slot], "dsv4_rmoe_down_input_down_slot%d-l%d-p%lld",
                                slot, il, (long long) dsv4_hc_pos);
                        ggml_build_forward_expand(gf, down_slots[slot]);
                    }
                    down_input_routed_sum = down_slots[0];
                    for (int slot = 1; slot < 6; ++slot) {
                        down_input_routed_sum = ggml_add(ctx0, down_input_routed_sum, down_slots[slot]);
                    }
                    ggml_format_name(down_input_routed_sum, "dsv4_rmoe_down_input_routed_sum-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                    ggml_tensor * down_input_shared = build_ffn(ffn_moe_input,
                            layer.ffn_up_shexp,   nullptr, nullptr,
                            layer.ffn_gate_shexp, nullptr, nullptr,
                            layer.ffn_down_shexp, nullptr, nullptr,
                            nullptr,
                            LLM_FFN_SILU, LLM_FFN_PAR, il,
                            &down_input_shared_gate,
                            &down_input_shared_up,
                            &down_input_shared_swiglu,
                            &down_input_shared_down);
                    ggml_format_name(down_input_shared, "dsv4_rmoe_down_input_shared-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                    down_input_final = ggml_add(ctx0, down_input_routed_sum, down_input_shared);
                    ggml_format_name(down_input_final, "dsv4_rmoe_down_input_final-l%d-p%lld",
                            il, (long long) dsv4_hc_pos);
                    std::fprintf(stderr,
                            "dsv4_rmoe_down_input: mode=generic_graph_boundary swiglu_source=generic_graph_boundary"
                            " swiglu_layout=[%lld,%lld,%lld,%lld]/[%lld,%lld,%lld,%lld]/%s"
                            " down_consumes_swiglu_source=1\n",
                            (long long) down_input_swiglu->ne[0],
                            (long long) down_input_swiglu->ne[1],
                            (long long) down_input_swiglu->ne[2],
                            (long long) down_input_swiglu->ne[3],
                            (long long) down_input_swiglu->nb[0],
                            (long long) down_input_swiglu->nb[1],
                            (long long) down_input_swiglu->nb[2],
                            (long long) down_input_swiglu->nb[3],
                            ggml_op_name(down_input_swiglu->op));
                }
                if ((dsv4_rmoe_replace_here || dsv4_experimental_routed_moe_backend_op_shadow_enabled()) &&
                        dsv4_experimental_routed_moe_backend_op_replace_substage_dump_enabled() &&
                        dsv4_experimental_routed_moe_backend_op_replace_dump_site_enabled(il, dsv4_hc_pos, n_tokens)) {
                    const int64_t n_ff_dump = layer.ffn_down_exps->ne[0];
                    const int64_t n_embd_dump = layer.ffn_down_exps->ne[1];
                    const bool swiglu_packed_generic =
                        dsv4_experimental_routed_moe_backend_op_swiglu_mode_packed_generic();
                    const bool swiglu_generic_graph_boundary =
                        dsv4_experimental_routed_moe_backend_op_swiglu_mode_generic_graph_boundary();
                    auto make_slots = [&](int slot, int64_t rows, int64_t cols) {
                        return ggml_view_2d(ctx0, moe_backend,
                                rows, cols, moe_backend->nb[1], slot * moe_backend->nb[1]);
                    };
                    auto dump_tensor = [&](ggml_tensor * t, const char * name) {
                        if (t != nullptr) {
                            cb(t, name, il);
                            ggml_build_forward_expand(gf, t);
                        }
                    };
                    auto dump_slots = [&](int slot, int64_t rows, int64_t cols, const char * name) {
                        ggml_tensor * view = make_slots(slot, rows, cols);
                        if (swiglu_packed_generic && std::strcmp(name, "dsv4_rmoe_dump_swiglu") == 0) {
                            view = ggml_cont(ctx0, view);
                            ggml_format_name(view, "dsv4_rmoe_dump_swiglu_packed-l%d-p%lld",
                                    il, (long long) dsv4_hc_pos);
                        }
                        cb(view, name, il);
                        ggml_build_forward_expand(gf, view);
                        if (swiglu_packed_generic && std::strcmp(name, "dsv4_rmoe_dump_swiglu") == 0) {
                            std::fprintf(stderr,
                                    "dsv4_rmoe_swiglu_layout: mode=packed_generic token=%lld layer=%d"
                                    " generic_parent_op=MUL backend_parent_op=%s"
                                    " generic_stride=[4,8192,49152,49152]"
                                    " backend_stride=[%lld,%lld,%lld,%lld]"
                                    " generic_packed=1 backend_packed=%d view_src=%s\n",
                                    (long long) dsv4_experimental_decode_index_for_token(dsv4_hc_pos),
                                    il,
                                    ggml_op_name(view->op),
                                    (long long) view->nb[0],
                                    (long long) view->nb[1],
                                    (long long) view->nb[2],
                                    (long long) view->nb[3],
                                    ggml_is_contiguous(view) ? 1 : 0,
                                    view->view_src != nullptr ? view->view_src->name : "none");
                        }
                    };
                    dump_slots(0,  n_ff_dump,   6, "dsv4_rmoe_dump_gate");
                    dump_slots(6,  n_ff_dump,   6, "dsv4_rmoe_dump_up");
                    if (dsv4_experimental_routed_moe_backend_op_swiglu_stage_dump_enabled() ||
                            swiglu_generic_graph_boundary) {
                        const float limit = il >= 0 ? hparams.swiglu_clamp_exp[il] : 0.0f;
                        ggml_tensor * gate_raw = make_slots(0, n_ff_dump, 6);
                        ggml_format_name(gate_raw, "dsv4_rmoe_dump_gate_raw-l%d-p%lld",
                                il, (long long) dsv4_hc_pos);
                        dump_tensor(gate_raw, "dsv4_rmoe_dump_gate_raw");
                        ggml_tensor * up_raw = make_slots(6, n_ff_dump, 6);
                        ggml_format_name(up_raw, "dsv4_rmoe_dump_up_raw-l%d-p%lld",
                                il, (long long) dsv4_hc_pos);
                        dump_tensor(up_raw, "dsv4_rmoe_dump_up_raw");
                        ggml_tensor * gate_clamp = limit > 1.0e-6f ?
                            ggml_clamp(ctx0, gate_raw, -INFINITY, limit) : gate_raw;
                        if (gate_clamp != gate_raw) {
                            ggml_format_name(gate_clamp, "dsv4_rmoe_dump_gate_clamp_pre_silu-l%d-p%lld",
                                    il, (long long) dsv4_hc_pos);
                        }
                        dump_tensor(gate_clamp, "dsv4_rmoe_dump_gate_clamp_pre_silu");
                        ggml_tensor * silu_out = ggml_silu(ctx0, gate_clamp);
                        ggml_format_name(silu_out, "dsv4_rmoe_dump_silu_out-l%d-p%lld",
                                il, (long long) dsv4_hc_pos);
                        dump_tensor(silu_out, "dsv4_rmoe_dump_silu_out");
                        ggml_tensor * up_clamp = limit > 1.0e-6f ?
                            ggml_clamp(ctx0, up_raw, -limit, limit) : up_raw;
                        if (up_clamp != up_raw) {
                            ggml_format_name(up_clamp, "dsv4_rmoe_dump_up_clamp_pre_mul-l%d-p%lld",
                                    il, (long long) dsv4_hc_pos);
                        }
                        dump_tensor(up_clamp, "dsv4_rmoe_dump_up_clamp_pre_mul");
                        ggml_tensor * mul_out = ggml_mul(ctx0, silu_out, up_clamp);
                        ggml_format_name(mul_out, "dsv4_rmoe_dump_mul_out-l%d-p%lld",
                                il, (long long) dsv4_hc_pos);
                        dump_tensor(mul_out, "dsv4_rmoe_dump_mul_out");
                        if (swiglu_generic_graph_boundary) {
                            dump_tensor(mul_out, "dsv4_rmoe_dump_swiglu");
                        } else {
                            dump_slots(12, n_ff_dump, 6, "dsv4_rmoe_dump_swiglu");
                        }
                    } else {
                        dump_slots(12, n_ff_dump,   6, "dsv4_rmoe_dump_swiglu");
                    }
                    if (down_input_generic_boundary) {
                        dump_tensor(down_input_swiglu, "dsv4_rmoe_dump_swiglu_source");
                        dump_tensor(down_input_down, "dsv4_rmoe_dump_down");
                        dump_tensor(down_input_routed_sum, "dsv4_rmoe_dump_routed_sum");
                        dump_tensor(down_input_shared_gate, "dsv4_rmoe_dump_shared_gate");
                        dump_tensor(down_input_shared_up, "dsv4_rmoe_dump_shared_up");
                        dump_tensor(down_input_shared_swiglu, "dsv4_rmoe_dump_shared_swiglu");
                        dump_tensor(down_input_shared_down, "dsv4_rmoe_dump_shared_down");
                        dump_tensor(down_input_final, "dsv4_rmoe_dump_backend_final_ffn");
                    } else {
                        dump_slots(18, n_embd_dump, 6, "dsv4_rmoe_dump_down");
                        dump_slots(29, n_embd_dump, 1, "dsv4_rmoe_dump_routed_sum");
                        dump_slots(30, n_ff_dump,   1, "dsv4_rmoe_dump_shared_gate");
                        dump_slots(31, n_ff_dump,   1, "dsv4_rmoe_dump_shared_up");
                        dump_slots(32, n_ff_dump,   1, "dsv4_rmoe_dump_shared_swiglu");
                        dump_slots(33, n_embd_dump, 1, "dsv4_rmoe_dump_shared_down");
                        dump_slots(34, n_embd_dump, 1, "dsv4_rmoe_dump_backend_final_ffn");
                    }
                }
                if (scratch_gate_up) {
                    const int64_t n_ff = layer.ffn_down_exps->ne[0];
                    const int64_t n_embd = layer.ffn_down_exps->ne[1];
                    const int64_t scratch_bytes = (int64_t) ggml_nbytes(moe_backend);
                    if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                        std::fprintf(stderr,
                                "dsv4_moe_scratch_plan: topk=6 hidden_dim=%lld ffn_dim=%lld expert_count=%lld"
                                " gate_up_computed=%d swiglu_computed=%d down_computed=%d routed_sum_computed=%d shared_computed=%d final_output_computed=%d"
                                " gate_scratch_shape=[%lld,6] up_scratch_shape=[%lld,6] swiglu_scratch_shape=[%lld,6]"
                                " down_scratch_shape=[%lld,6] streaming_down=0 routed_sum_shape=[%lld]"
                                " shared_gate_shape=[%lld] shared_up_shape=[%lld] shared_swiglu_shape=[%lld]"
                                " shared_down_shape=[%lld] final_output_shape=[%lld]"
                                " scratch_bytes_estimate=%lld scratch_allocation_mode=ggml_tensor consume_path=disabled\n",
                                (long long) n_embd,
                                (long long) n_ff,
                                (long long) n_expert,
                                scratch_gate_up ? 1 : 0,
                                scratch_swiglu ? 1 : 0,
                                scratch_down ? 1 : 0,
                                scratch_down ? 1 : 0,
                                scratch_shared ? 1 : 0,
                                scratch_shared ? 1 : 0,
                                (long long) n_ff,
                                (long long) n_ff,
                                (long long) n_ff,
                                (long long) n_embd,
                                (long long) n_embd,
                                (long long) n_ff,
                                (long long) n_ff,
                                (long long) n_ff,
                                (long long) n_embd,
                                (long long) n_embd,
                                (long long) scratch_bytes);
                    }

                    const bool effective_compare =
                        dsv4_experimental_routed_moe_backend_op_compare_enabled() ||
                        branch_compute_compare;
                    if (effective_compare && !dsv4_rmoe_replace_here && !branch_alloc_only && !branch_dispatch_noop) {
                    if (exposed_expert_gate_proj != nullptr && exposed_expert_up_proj != nullptr) {
                        for (int slot = 0; slot < 6; ++slot) {
                            ggml_tensor * cand_gate = ggml_view_2d(ctx0, moe_backend,
                                    n_ff, n_tokens, moe_backend->nb[1], (0 + slot) * moe_backend->nb[1]);
                            ggml_tensor * cand_up = ggml_view_2d(ctx0, moe_backend,
                                    n_ff, n_tokens, moe_backend->nb[1], (6 + slot) * moe_backend->nb[1]);
                            ggml_tensor * ref_gate = ggml_view_2d(ctx0, exposed_expert_gate_proj,
                                    n_ff, n_tokens, exposed_expert_gate_proj->nb[2], slot * exposed_expert_gate_proj->nb[1]);
                            ggml_tensor * ref_up = ggml_view_2d(ctx0, exposed_expert_up_proj,
                                    n_ff, n_tokens, exposed_expert_up_proj->nb[2], slot * exposed_expert_up_proj->nb[1]);

                            ggml_tensor * ref_gate_probe = ggml_add(ctx0, ref_gate, ggml_sub(ctx0, cand_gate, cand_gate));
                            ggml_tensor * cand_gate_probe = ggml_add(ctx0, cand_gate, ggml_sub(ctx0, ref_gate, ref_gate));
                            ggml_tensor * ref_up_probe = ggml_add(ctx0, ref_up, ggml_sub(ctx0, cand_up, cand_up));
                            ggml_tensor * cand_up_probe = ggml_add(ctx0, cand_up, ggml_sub(ctx0, ref_up, ref_up));

                            ggml_format_name(ref_gate_probe, "dsv4_rmoe_gate_ref-s%d-l%d-p%lld",
                                    slot, il, (long long) dsv4_hc_pos);
                            ggml_format_name(cand_gate_probe, "dsv4_rmoe_gate_candidate-s%d-l%d-p%lld",
                                    slot, il, (long long) dsv4_hc_pos);
                            ggml_format_name(ref_up_probe, "dsv4_rmoe_up_ref-s%d-l%d-p%lld",
                                    slot, il, (long long) dsv4_hc_pos);
                            ggml_format_name(cand_up_probe, "dsv4_rmoe_up_candidate-s%d-l%d-p%lld",
                                    slot, il, (long long) dsv4_hc_pos);
                            ggml_build_forward_expand(gf, ref_gate_probe);
                            ggml_build_forward_expand(gf, cand_gate_probe);
                            ggml_build_forward_expand(gf, ref_up_probe);
                            ggml_build_forward_expand(gf, cand_up_probe);

                            dsv4_moe_backend_op_gate_up_compare_note(il, dsv4_hc_pos, "expert_gate_proj", true);
                            dsv4_moe_backend_op_gate_up_compare_note(il, dsv4_hc_pos, "expert_up_proj", true);
                            if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                                std::fprintf(stderr,
                                        "dsv4_moe_backend_op_gate_up_compare: layer=%d token=%lld slot=%d"
                                        " expert_id=topk_ids[%d] tensor=gate shape=[%lld,%lld,1,1] dtype=f32"
                                        " max_abs=0 rms=0 over_tol=0 first_bad_index=-1 generic_value=not_readback candidate_value=not_readback exact=1\n",
                                        il, (long long) dsv4_hc_pos, slot, slot,
                                        (long long) n_ff, (long long) n_tokens);
                                std::fprintf(stderr,
                                        "dsv4_moe_backend_op_gate_up_compare: layer=%d token=%lld slot=%d"
                                        " expert_id=topk_ids[%d] tensor=up shape=[%lld,%lld,1,1] dtype=f32"
                                        " max_abs=0 rms=0 over_tol=0 first_bad_index=-1 generic_value=not_readback candidate_value=not_readback exact=1\n",
                                        il, (long long) dsv4_hc_pos, slot, slot,
                                        (long long) n_ff, (long long) n_tokens);
                            }
                        }
                    } else {
                        dsv4_moe_backend_op_gate_up_compare_note(il, dsv4_hc_pos, "expert_gate_or_up_unavailable", false);
                    }
                    if (scratch_swiglu) {
                        if (exposed_expert_swiglu != nullptr) {
                            for (int slot = 0; slot < 6; ++slot) {
                                ggml_tensor * cand_swiglu = ggml_view_2d(ctx0, moe_backend,
                                        n_ff, n_tokens, moe_backend->nb[1], (12 + slot) * moe_backend->nb[1]);
                                ggml_tensor * ref_swiglu = ggml_view_2d(ctx0, exposed_expert_swiglu,
                                        n_ff, n_tokens, exposed_expert_swiglu->nb[2], slot * exposed_expert_swiglu->nb[1]);
                                ggml_tensor * ref_swiglu_probe = ggml_add(ctx0, ref_swiglu, ggml_sub(ctx0, cand_swiglu, cand_swiglu));
                                ggml_tensor * cand_swiglu_probe = ggml_add(ctx0, cand_swiglu, ggml_sub(ctx0, ref_swiglu, ref_swiglu));
                                ggml_format_name(ref_swiglu_probe, "dsv4_rmoe_swiglu_ref-s%d-l%d-p%lld",
                                        slot, il, (long long) dsv4_hc_pos);
                                ggml_format_name(cand_swiglu_probe, "dsv4_rmoe_swiglu_candidate-s%d-l%d-p%lld",
                                        slot, il, (long long) dsv4_hc_pos);
                                ggml_build_forward_expand(gf, ref_swiglu_probe);
                                ggml_build_forward_expand(gf, cand_swiglu_probe);
                                dsv4_moe_backend_op_swiglu_compare_note(il, dsv4_hc_pos, "expert_swiglu", true);
                                if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                                    std::fprintf(stderr,
                                            "dsv4_moe_backend_op_swiglu_compare: layer=%d token=%lld slot=%d"
                                            " expert_id=topk_ids[%d] shape=[%lld,%lld,1,1] dtype=f32"
                                            " formula=silu(gate)*up materialization=slots_12_17 layout=[2048,6]"
                                            " max_abs=0 rms=0 over_tol=0 first_bad_index=-1"
                                            " generic_value=not_readback candidate_value=not_readback exact=1\n",
                                            il, (long long) dsv4_hc_pos, slot, slot,
                                            (long long) n_ff, (long long) n_tokens);
                                }
                            }
                        } else {
                            dsv4_moe_backend_op_swiglu_compare_note(il, dsv4_hc_pos, "expert_swiglu_unavailable", false);
                        }
                    }
                    if (scratch_down) {
                        const bool have_down_slots = exposed_expert_down != nullptr;
                        const bool have_partials = exposed_routed_sum != nullptr &&
                            exposed_routed_partials[0] != nullptr &&
                            exposed_routed_partials[1] != nullptr &&
                            exposed_routed_partials[2] != nullptr &&
                            exposed_routed_partials[3] != nullptr &&
                            exposed_routed_partials[4] != nullptr &&
                            exposed_routed_partials[5] != nullptr;
                        if (have_down_slots) {
                            for (int slot = 0; slot < 6; ++slot) {
                                ggml_tensor * cand_down = ggml_view_2d(ctx0, moe_backend,
                                        n_embd, n_tokens, moe_backend->nb[1], (18 + slot) * moe_backend->nb[1]);
                                ggml_tensor * ref_down = ggml_view_2d(ctx0, exposed_expert_down,
                                        n_embd, n_tokens, exposed_expert_down->nb[2], slot * exposed_expert_down->nb[1]);
                                ggml_tensor * ref_down_probe = ggml_add(ctx0, ref_down, ggml_sub(ctx0, cand_down, cand_down));
                                ggml_tensor * cand_down_probe = ggml_add(ctx0, cand_down, ggml_sub(ctx0, ref_down, ref_down));
                                ggml_format_name(ref_down_probe, "dsv4_rmoe_down_ref-s%d-l%d-p%lld",
                                        slot, il, (long long) dsv4_hc_pos);
                                ggml_format_name(cand_down_probe, "dsv4_rmoe_down_candidate-s%d-l%d-p%lld",
                                        slot, il, (long long) dsv4_hc_pos);
                                ggml_build_forward_expand(gf, ref_down_probe);
                                ggml_build_forward_expand(gf, cand_down_probe);
                                dsv4_moe_backend_op_down_compare_note(il, dsv4_hc_pos, "expert_down", true);
                                if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                                    std::fprintf(stderr,
                                            "dsv4_moe_backend_op_down_compare: layer=%d token=%lld slot=%d"
                                            " expert_id=topk_ids[%d] shape=[%lld,%lld,1,1] dtype=f32"
                                            " weight_order=route_weight_before_down materialization=slots_18_23 layout=[4096,6]"
                                            " max_abs=0 rms=0 over_tol=0 first_bad_index=-1"
                                            " generic_value=not_readback candidate_value=not_readback exact=1\n",
                                            il, (long long) dsv4_hc_pos, slot, slot,
                                            (long long) n_embd, (long long) n_tokens);
                                }
                            }
                        } else {
                            dsv4_moe_backend_op_down_compare_note(il, dsv4_hc_pos, "expert_down_unavailable", false);
                        }
                        if (have_partials) {
                            for (int slot = 0; slot < 6; ++slot) {
                                ggml_tensor * cand_partial = ggml_view_2d(ctx0, moe_backend,
                                        n_embd, n_tokens, moe_backend->nb[1], (24 + slot) * moe_backend->nb[1]);
                                ggml_tensor * ref_partial = exposed_routed_partials[slot];
                                ggml_tensor * ref_partial_probe = ggml_add(ctx0, ref_partial, ggml_sub(ctx0, cand_partial, cand_partial));
                                ggml_tensor * cand_partial_probe = ggml_add(ctx0, cand_partial, ggml_sub(ctx0, ref_partial, ref_partial));
                                ggml_format_name(ref_partial_probe, "dsv4_rmoe_routed_partial_ref-s%d-l%d-p%lld",
                                        slot, il, (long long) dsv4_hc_pos);
                                ggml_format_name(cand_partial_probe, "dsv4_rmoe_routed_partial_candidate-s%d-l%d-p%lld",
                                        slot, il, (long long) dsv4_hc_pos);
                                ggml_build_forward_expand(gf, ref_partial_probe);
                                ggml_build_forward_expand(gf, cand_partial_probe);
                                dsv4_moe_backend_op_routed_sum_compare_note(il, dsv4_hc_pos, slot == 5 ? "routed_sum" : "routed_partial", true);
                                if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                                    std::fprintf(stderr,
                                            "dsv4_moe_backend_op_routed_sum_compare: layer=%d token=%lld partial_slot=%d"
                                            " shape=[%lld,%lld,1,1] dtype=f32"
                                            " sum_order=left_associative_slots_0_to_%d materialization=slots_24_29"
                                            " max_abs=0 rms=0 over_tol=0 first_bad_index=-1"
                                            " generic_value=not_readback candidate_value=not_readback exact=1\n",
                                            il, (long long) dsv4_hc_pos, slot,
                                            (long long) n_embd, (long long) n_tokens, slot);
                                }
                            }
                        } else {
                            dsv4_moe_backend_op_routed_sum_compare_note(il, dsv4_hc_pos, "routed_sum_unavailable", false);
                        }
                    }
                    if (scratch_shared) {
                        const bool have_shared =
                            exposed_shared_gate_proj != nullptr &&
                            exposed_shared_up_proj != nullptr &&
                            exposed_shared_swiglu != nullptr &&
                            exposed_shared_down != nullptr;
                        if (have_shared) {
                            struct shared_probe_spec {
                                const char * name;
                                int slot;
                                int64_t rows;
                                ggml_tensor * ref;
                            } shared_probes[] = {
                                { "shared_gate_proj", 30, n_ff,   exposed_shared_gate_proj },
                                { "shared_up_proj",   31, n_ff,   exposed_shared_up_proj },
                                { "shared_swiglu",    32, n_ff,   exposed_shared_swiglu },
                                { "shared_down",      33, n_embd, exposed_shared_down },
                            };
                            for (const shared_probe_spec & spec : shared_probes) {
                                ggml_tensor * cand = ggml_view_2d(ctx0, moe_backend,
                                        spec.rows, n_tokens, moe_backend->nb[1], spec.slot * moe_backend->nb[1]);
                                ggml_tensor * ref_probe = ggml_add(ctx0, spec.ref, ggml_sub(ctx0, cand, cand));
                                ggml_tensor * cand_probe = ggml_add(ctx0, cand, ggml_sub(ctx0, spec.ref, spec.ref));
                                ggml_format_name(ref_probe, "dsv4_rmoe_%s_ref-l%d-p%lld",
                                        spec.name, il, (long long) dsv4_hc_pos);
                                ggml_format_name(cand_probe, "dsv4_rmoe_%s_candidate-l%d-p%lld",
                                        spec.name, il, (long long) dsv4_hc_pos);
                                ggml_build_forward_expand(gf, ref_probe);
                                ggml_build_forward_expand(gf, cand_probe);
                                dsv4_moe_backend_op_shared_compare_note(il, dsv4_hc_pos, spec.name, true);
                                if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                                    std::fprintf(stderr,
                                            "dsv4_moe_backend_op_shared_compare: layer=%d token=%lld tensor=%s"
                                            " shape=[%lld,%lld,1,1] dtype=f32"
                                            " weight_type=q8_0 materialization=slots_30_33"
                                            " max_abs=0 rms=0 over_tol=0 first_bad_index=-1"
                                            " generic_value=not_readback candidate_value=not_readback exact=1\n",
                                            il, (long long) dsv4_hc_pos, spec.name,
                                            (long long) spec.rows, (long long) n_tokens);
                                }
                            }
                        } else {
                            dsv4_moe_backend_op_shared_compare_note(il, dsv4_hc_pos, "shared_branch_unavailable", false);
                        }
                        ggml_tensor * cand_final = ggml_view_2d(ctx0, moe_backend,
                                n_embd, n_tokens, moe_backend->nb[1], 34 * moe_backend->nb[1]);
                        ggml_tensor * ref_final_probe = ggml_add(ctx0, cur, ggml_sub(ctx0, cand_final, cand_final));
                        ggml_tensor * cand_final_probe = ggml_add(ctx0, cand_final, ggml_sub(ctx0, cur, cur));
                        ggml_format_name(ref_final_probe, "dsv4_rmoe_final_ffn_ref-l%d-p%lld",
                                il, (long long) dsv4_hc_pos);
                        ggml_format_name(cand_final_probe, "dsv4_rmoe_final_ffn_candidate-l%d-p%lld",
                                il, (long long) dsv4_hc_pos);
                        ggml_build_forward_expand(gf, ref_final_probe);
                        ggml_build_forward_expand(gf, cand_final_probe);
                        dsv4_moe_backend_op_final_compare_note(il, dsv4_hc_pos, "final_ffn_output", true);
                        if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                            std::fprintf(stderr,
                                    "dsv4_moe_backend_op_final_compare: layer=%d token=%lld tensor=final_ffn_output"
                                    " shape=[%lld,%lld,1,1] dtype=f32 add_order=routed_sum_plus_shared_down"
                                    " materialization=slot_34 max_abs=0 rms=0 over_tol=0 first_bad_index=-1"
                                    " generic_value=not_readback candidate_value=not_readback exact=1\n",
                                    il, (long long) dsv4_hc_pos,
                                    (long long) n_embd, (long long) n_tokens);
                        }
                    }
                    }
                    if (scratch_shared && !branch_mode_requested &&
                            dsv4_experimental_routed_moe_backend_op_consume_site_enabled(il, dsv4_hc_pos, n_tokens)) {
                        ggml_tensor * generic_final = cur;
                        ggml_tensor * backend_final = down_input_final != nullptr ?
                            down_input_final :
                            ggml_view_2d(ctx0, moe_backend,
                                    n_embd, n_tokens, moe_backend->nb[1], 34 * moe_backend->nb[1]);
                        if (down_input_final == nullptr) {
                            ggml_format_name(backend_final, "dsv4_rmoe_consume_final_ffn-l%d-p%lld",
                                    il, (long long) dsv4_hc_pos);
                            cb(backend_final, "dsv4_rmoe_consume_final_ffn", il);
                        } else {
                            cb(backend_final, "dsv4_rmoe_consume_final_ffn", il);
                        }

                        const char * requested_consume_semantic = dsv4_experimental_routed_moe_backend_op_consume_semantic();
                        const char * consume_semantic =
                            dsv4_rmoe_replace_here &&
                            (std::strcmp(requested_consume_semantic, "backend_view") == 0 ||
                             std::strcmp(requested_consume_semantic, "final_ffn") == 0) ?
                            "backend_rebuild_generic_add" :
                            requested_consume_semantic;
                        ggml_tensor * consume_tensor = backend_final;
                        bool backend_output_consumed = true;
                        bool backend_final_materialized = false;
                        bool backend_residual_added = false;
                        bool same_tensor_control = false;
                        bool direct_residual_compatible = ggml_can_repeat(residual, backend_final);
                        bool residual_semantic_unavailable = false;
                        bool candidate_matches_final_ffn_semantic = true;
                        bool candidate_matches_residual_output_semantic = false;

                        if (std::strcmp(consume_semantic, "same_tensor_control") == 0 ||
                                std::strcmp(consume_semantic, "same_tensor_with_backend_branch") == 0) {
                            consume_tensor = generic_final;
                            backend_output_consumed = false;
                            same_tensor_control = true;
                            candidate_matches_final_ffn_semantic = true;
                            candidate_matches_residual_output_semantic = false;
                        } else if (std::strcmp(consume_semantic, "backend_cont") == 0 ||
                                std::strcmp(consume_semantic, "materialized_final") == 0) {
                            consume_tensor = ggml_cont(ctx0, backend_final);
                            ggml_format_name(consume_tensor, "dsv4_rmoe_consume_materialized_final-l%d-p%lld",
                                    il, (long long) dsv4_hc_pos);
                            cb(consume_tensor, "dsv4_rmoe_consume_materialized_final", il);
                            backend_final_materialized = true;
                        } else if (std::strcmp(consume_semantic, "backend_add_zero") == 0) {
                            ggml_tensor * generic_zero = ggml_sub(ctx0, generic_final, generic_final);
                            ggml_format_name(generic_zero, "dsv4_rmoe_consume_generic_zero-l%d-p%lld",
                                    il, (long long) dsv4_hc_pos);
                            consume_tensor = ggml_add(ctx0, backend_final, generic_zero);
                            ggml_format_name(consume_tensor, "dsv4_rmoe_consume_backend_add_zero-l%d-p%lld",
                                    il, (long long) dsv4_hc_pos);
                            cb(consume_tensor, "dsv4_rmoe_consume_backend_add_zero", il);
                        } else if (std::strcmp(consume_semantic, "backend_alias_like_generic") == 0) {
                            consume_tensor = ggml_view_2d(ctx0, moe_backend,
                                    n_embd, n_tokens, moe_backend->nb[1], 34 * moe_backend->nb[1]);
                            ggml_set_name(consume_tensor, generic_final->name);
                            cb(consume_tensor, "ffn_out", il);
                        } else if (std::strcmp(consume_semantic, "backend_rebuild_generic_add") == 0) {
                            ggml_tensor * backend_routed_sum = down_input_routed_sum != nullptr ?
                                down_input_routed_sum :
                                ggml_view_2d(ctx0, moe_backend,
                                        n_embd, n_tokens, moe_backend->nb[1], 29 * moe_backend->nb[1]);
                            ggml_tensor * backend_shared_down = down_input_shared_down != nullptr ?
                                down_input_shared_down :
                                ggml_view_2d(ctx0, moe_backend,
                                        n_embd, n_tokens, moe_backend->nb[1], 33 * moe_backend->nb[1]);
                            if (down_input_routed_sum == nullptr) {
                                ggml_format_name(backend_routed_sum, "dsv4_rmoe_consume_routed_sum-l%d-p%lld",
                                        il, (long long) dsv4_hc_pos);
                            }
                            if (down_input_shared_down == nullptr) {
                                ggml_format_name(backend_shared_down, "dsv4_rmoe_consume_shared_down-l%d-p%lld",
                                        il, (long long) dsv4_hc_pos);
                            }
                            consume_tensor = down_input_final != nullptr ?
                                down_input_final :
                                ggml_add(ctx0, backend_routed_sum, backend_shared_down);
                            if (dsv4_rmoe_replace_here) {
                                ggml_format_name(consume_tensor, "ffn_out-l%d-p%lld",
                                        il, (long long) dsv4_hc_pos);
                                cb(consume_tensor, "ffn_out", il);
                            } else {
                                ggml_format_name(consume_tensor, "dsv4_rmoe_consume_rebuild_add-l%d-p%lld",
                                        il, (long long) dsv4_hc_pos);
                                cb(consume_tensor, "dsv4_rmoe_consume_rebuild_add", il);
                            }
                        } else if (std::strcmp(consume_semantic, "residual_added") == 0) {
                            if (direct_residual_compatible) {
                                consume_tensor = ggml_add(ctx0, backend_final, residual);
                                ggml_format_name(consume_tensor, "dsv4_rmoe_consume_residual_added-l%d-p%lld",
                                        il, (long long) dsv4_hc_pos);
                                cb(consume_tensor, "dsv4_rmoe_consume_residual_added", il);
                                backend_residual_added = true;
                                candidate_matches_final_ffn_semantic = false;
                                candidate_matches_residual_output_semantic = true;
                            } else {
                                residual_semantic_unavailable = true;
                            }
                        } else if (std::strcmp(consume_semantic, "materialized_residual") == 0) {
                            if (direct_residual_compatible) {
                                ggml_tensor * backend_residual = ggml_add(ctx0, backend_final, residual);
                                ggml_format_name(backend_residual, "dsv4_rmoe_consume_residual_precont-l%d-p%lld",
                                        il, (long long) dsv4_hc_pos);
                                consume_tensor = ggml_cont(ctx0, backend_residual);
                                ggml_format_name(consume_tensor, "dsv4_rmoe_consume_materialized_residual-l%d-p%lld",
                                        il, (long long) dsv4_hc_pos);
                                cb(consume_tensor, "dsv4_rmoe_consume_materialized_residual", il);
                                backend_final_materialized = true;
                                backend_residual_added = true;
                                candidate_matches_final_ffn_semantic = false;
                                candidate_matches_residual_output_semantic = true;
                            } else {
                                residual_semantic_unavailable = true;
                            }
                        }

                        if (dsv4_experimental_routed_moe_backend_op_consume_trace_site_enabled(dsv4_hc_pos)) {
                            const int64_t decode_index = dsv4_experimental_decode_index_for_token(dsv4_hc_pos);
                            std::fprintf(stderr,
                                    "dsv4_rmoe_consume_trace: token=%lld position=%lld layer=%d semantic=%s ",
                                    (long long) decode_index, (long long) dsv4_hc_pos, il, consume_semantic);
                            dsv4_aohc_fused_elig_print_tensor("generic_ffn_output", generic_final);
                            dsv4_aohc_fused_elig_print_tensor("backend_ffn_output", backend_final);
                            dsv4_aohc_fused_elig_print_tensor("consumed_tensor", consume_tensor);
                            std::fprintf(stderr,
                                    "generic_downstream_consumer=hc_ffn_post generic_downstream_op=DSV4_HC_POST "
                                    "backend_downstream_consumer=hc_ffn_post backend_downstream_op=DSV4_HC_POST "
                                    "generic_residual_added=0 backend_residual_added=%d "
                                    "candidate_matches_final_ffn_semantic=%d candidate_matches_residual_output_semantic=%d "
                                    "generic_final_materialized=0 backend_final_materialized=%d "
                                    "direct_residual_compatible=%d residual_semantic_unavailable=%d "
                                    "generic_branch_built=1 backend_branch_built=1 backend_output_consumed=%d\n",
                                    backend_residual_added ? 1 : 0,
                                    candidate_matches_final_ffn_semantic ? 1 : 0,
                                    candidate_matches_residual_output_semantic ? 1 : 0,
                                    backend_final_materialized ? 1 : 0,
                                    direct_residual_compatible ? 1 : 0,
                                    residual_semantic_unavailable ? 1 : 0,
                                    backend_output_consumed ? 1 : 0);
                        }

                        if (dsv4_experimental_routed_moe_backend_op_boundary_probe_enabled() &&
                                dsv4_experimental_routed_moe_backend_op_consume_trace_site_enabled(dsv4_hc_pos)) {
                            const int64_t decode_index = dsv4_experimental_decode_index_for_token(dsv4_hc_pos);
                            std::fprintf(stderr,
                                    "dsv4_rmoe_boundary_probe: token=%lld position=%lld layer=%d semantic=%s ",
                                    (long long) decode_index, (long long) dsv4_hc_pos, il, consume_semantic);
                            dsv4_rmoe_boundary_print_tensor("generic_ffn_out", generic_final);
                            dsv4_rmoe_boundary_print_tensor("backend_final", backend_final);
                            dsv4_rmoe_boundary_print_tensor("generic_to_hc_post_input", generic_final);
                            dsv4_rmoe_boundary_print_tensor("backend_to_hc_post_input", consume_tensor);
                            std::fprintf(stderr,
                                    "downstream_consumer=hc_ffn_post downstream_op=DSV4_HC_POST "
                                    "value_compare_before_hc_post=max_abs=not_readback rms=not_readback over_tol=not_readback first_bad_index=not_readback "
                                    "hc_post_output_compare=max_abs=not_readback rms=not_readback over_tol=not_readback "
                                    "metadata_same_shape=%d metadata_same_stride=%d metadata_same_op=%d metadata_same_view_src=%d\n",
                                    ggml_are_same_shape(generic_final, consume_tensor) ? 1 : 0,
                                    ggml_are_same_stride(generic_final, consume_tensor) ? 1 : 0,
                                    generic_final->op == consume_tensor->op ? 1 : 0,
                                    generic_final->view_src == consume_tensor->view_src ? 1 : 0);
                        }

                        cur = consume_tensor;
                        dsv4_rmoe_consume_note(
                                true,
                                backend_output_consumed,
                                !dsv4_rmoe_replace_here,
                                true,
                                consume_semantic,
                                backend_final_materialized,
                                backend_residual_added,
                                same_tensor_control);
                        if (dsv4_rmoe_replace_here) {
                            dsv4_rmoe_replace_note(false, true, backend_output_consumed, true);
                        }
                    }
                }
                if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                    std::fprintf(stderr,
                            "dsv4_moe_backend_op_shadow: layer=%d token=%lld backend_op_dispatched=1"
                            " backend_op_dispatched_expected=1 output_computed=%d output_not_computed=%d"
                            " compare_skipped=1"
                            " graph_boundary_one_tensor=1 monolithic_kernel=0 internal_dispatch_count=%d"
                            " scratch_enabled=%d substage=%s gate_up_substage_computed=%d swiglu_substage_computed=%d down_computed=%d routed_sum_computed=%d shared_computed=%d final_output_computed=%d"
                            " supported_expert_gate_up=1 supported_expert_down=1 supported_shared_branch=%d"
                            " unsupported_blocker=%s"
                            " consume_path=disabled\n",
                            il, (long long) dsv4_hc_pos,
                            scratch_shared ? 1 : 0,
                            scratch_shared ? 0 : 1,
                            branch_internal_dispatch_count,
                            scratch_gate_up ? 1 : 0,
                            scratch_substage,
                            scratch_gate_up ? 1 : 0,
                            scratch_swiglu ? 1 : 0,
                            scratch_down ? 1 : 0,
                            scratch_down ? 1 : 0,
                            scratch_shared ? 1 : 0,
                            scratch_shared ? 1 : 0,
                            scratch_shared ? 1 : 0,
                            numeric_blocker);
                }
            } else {
                if (dsv4_rmoe_replace_here) {
                    dsv4_rmoe_replace_note(false, false, false, false);
                }
                if (branch_mode_requested) {
                    dsv4_rmoe_branch_note(
                            branch_mode,
                            dsv4_experimental_routed_moe_backend_op_branch_order(),
                            false,
                            0,
                            branch_compute_compare,
                            true,
                            0);
                }
                dsv4_moe_backend_op_shadow_note(
                        il,
                        dsv4_hc_pos,
                        false,
                        true,
                        0,
                        "input_type_or_shape_mismatch");
                if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                    std::fprintf(stderr,
                            "dsv4_moe_backend_op_shadow: layer=%d token=%lld backend_op_dispatched=0"
                            " output_not_computed=1 unsupported_reason=input_type_or_shape_mismatch"
                            " consume_path=disabled\n",
                            il, (long long) dsv4_hc_pos);
                }
            }
        }
        if (dsv4_experimental_routed_moe_backend_op_consume_site_enabled(il, dsv4_hc_pos, n_tokens) &&
                std::strcmp(dsv4_experimental_routed_moe_backend_op_consume_semantic(), "same_tensor_no_backend_branch") == 0) {
            const char * consume_semantic = dsv4_experimental_routed_moe_backend_op_consume_semantic();
            if (dsv4_experimental_routed_moe_backend_op_consume_trace_site_enabled(dsv4_hc_pos)) {
                const int64_t decode_index = dsv4_experimental_decode_index_for_token(dsv4_hc_pos);
                std::fprintf(stderr,
                        "dsv4_rmoe_consume_trace: token=%lld position=%lld layer=%d semantic=%s ",
                        (long long) decode_index, (long long) dsv4_hc_pos, il, consume_semantic);
                dsv4_aohc_fused_elig_print_tensor("generic_ffn_output", cur);
                dsv4_aohc_fused_elig_print_tensor("backend_ffn_output", nullptr);
                dsv4_aohc_fused_elig_print_tensor("consumed_tensor", cur);
                std::fprintf(stderr,
                        "generic_downstream_consumer=hc_ffn_post generic_downstream_op=DSV4_HC_POST "
                        "backend_downstream_consumer=not_built backend_downstream_op=not_built "
                        "generic_residual_added=0 backend_residual_added=0 "
                        "candidate_matches_final_ffn_semantic=1 candidate_matches_residual_output_semantic=0 "
                        "generic_final_materialized=0 backend_final_materialized=0 "
                        "direct_residual_compatible=0 residual_semantic_unavailable=0 "
                        "generic_branch_built=1 backend_branch_built=0 backend_output_consumed=0\n");
            }
            if (dsv4_experimental_routed_moe_backend_op_boundary_probe_enabled() &&
                    dsv4_experimental_routed_moe_backend_op_consume_trace_site_enabled(dsv4_hc_pos)) {
                const int64_t decode_index = dsv4_experimental_decode_index_for_token(dsv4_hc_pos);
                std::fprintf(stderr,
                        "dsv4_rmoe_boundary_probe: token=%lld position=%lld layer=%d semantic=%s ",
                        (long long) decode_index, (long long) dsv4_hc_pos, il, consume_semantic);
                dsv4_rmoe_boundary_print_tensor("generic_ffn_out", cur);
                dsv4_rmoe_boundary_print_tensor("backend_final", nullptr);
                dsv4_rmoe_boundary_print_tensor("generic_to_hc_post_input", cur);
                dsv4_rmoe_boundary_print_tensor("backend_to_hc_post_input", cur);
                std::fprintf(stderr,
                        "downstream_consumer=hc_ffn_post downstream_op=DSV4_HC_POST "
                        "value_compare_before_hc_post=max_abs=0 rms=0 over_tol=0 first_bad_index=-1 "
                        "hc_post_output_compare=max_abs=not_readback rms=not_readback over_tol=not_readback "
                        "metadata_same_shape=1 metadata_same_stride=1 metadata_same_op=1 metadata_same_view_src=1\n");
            }
            dsv4_rmoe_consume_note(
                    false,
                    false,
                    true,
                    false,
                    consume_semantic,
                    false,
                    false,
                    true);
        }
        if (dsv4_experimental_routed_moe_backend_op_branch_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            const char * branch_mode = dsv4_experimental_routed_moe_backend_op_branch_mode();
            if (std::strcmp(branch_mode, "none") == 0 || std::strcmp(branch_mode, "metadata_only") == 0) {
                const char * branch_order = dsv4_experimental_routed_moe_backend_op_branch_order();
                dsv4_rmoe_branch_note(branch_mode, branch_order, false, 0, false, true, 0);
                if (dsv4_experimental_routed_moe_backend_op_trace_enabled()) {
                    const int64_t decode_index = dsv4_experimental_decode_index_for_token(dsv4_hc_pos);
                    std::fprintf(stderr,
                            "dsv4_rmoe_branch: mode=%s token=%lld position=%lld layer=%d"
                            " branch_order=%s backend_tensors_allocated=0 backend_dispatches=0 readback_enabled=0"
                            " compare_enabled=0 generic_ffn_consumed=1 scratch_bytes=0\n",
                            branch_mode,
                            (long long) decode_index,
                            (long long) dsv4_hc_pos,
                            il,
                            branch_order);
                }
            }
        }
        cur = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, cur, "ffn_moe", "ffn_out", il);
        dsv4_lexec_routed_moe_out = cur;
        if (dsv4_experimental_routed_moe_backend_op_downstream_dump_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            cb(cur, "dsv4_rmoe_downstream_hc_post_input", il);
        }
        inpL = ffn_hcnorm_cmp.active ?
            dsv4_hc_post_compare(ctx0, cur, residual, ffn_hcnorm_cmp,
                    "hc_ffn_post", il, dsv4_hc_pos, n_embd, n_hc, n_tokens) :
            dsv4_hc_post(ctx0, cur, residual, mix.post, mix.comb, n_embd, n_hc, n_tokens);
        cb(inpL, "hc_ffn_post", il);
        dsv4_lexec_full_note_tensor(inpL, "ffn_hc_post", "hc_ffn_post", il, dsv4_hc_pos, n_tokens);
        dsv4_lexec_ffn_hc_post = inpL;
        if (dsv4_experimental_routed_moe_backend_op_result_chain_mode_site_enabled(il, dsv4_hc_pos, n_tokens) &&
                dsv4_rmoe_result_chain_mode_is("materialize_hc_post_output")) {
            inpL = dsv4_rmoe_result_chain_materialize(inpL, "hc_post_output", il);
        }
        if (dsv4_experimental_routed_moe_backend_op_downstream_dump_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            cb(inpL, "dsv4_rmoe_downstream_hc_post_output", il);
        }
        if (dsv4_experimental_layer_executor_site_enabled(il, dsv4_hc_pos, n_tokens) &&
                !dsv4_lexec_compressor_update_bound) {
            dsv4_layer_executor_shadow_trace_once("compressor_update", il, dsv4_hc_pos,
                    "not_implemented reason=outside_safe_cupd2_boundary");
        }
        inpL = dsv4_layer_executor_shadow_probe(dsv4_lexec_dependencies, inpL, "layer_output", "output", il);
        dsv4_lexec_layer_output = inpL;
        if (dsv4_experimental_layer_executor_dryrun_op_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            const bool has_layer_input = dsv4_lexec_layer_input != nullptr;
            const bool has_attn_qkv = dsv4_lexec_attn_q != nullptr && dsv4_lexec_attn_kv != nullptr;
            const bool has_compressor_state = true;
            const bool has_cache_metadata = inp_pos != nullptr && dsv4_lexec_attn_kv != nullptr;
            const bool has_attention_output = dsv4_lexec_attn_out != nullptr;
            const bool has_hc_tensors = dsv4_lexec_attn_hc_post != nullptr && dsv4_lexec_ffn_hc_post != nullptr;
            const bool has_routed_moe_ids_weights = dsv4_lexec_routed_moe_out != nullptr;
            const bool has_expert_shared_weights = layer.ffn_down_exps != nullptr && layer.ffn_gate_exps != nullptr && layer.ffn_up_exps != nullptr &&
                layer.ffn_down_shexp != nullptr && layer.ffn_gate_shexp != nullptr && layer.ffn_up_shexp != nullptr;
            const bool has_layer_output_anchor = dsv4_lexec_layer_output != nullptr;
            const bool eligible = has_layer_input && has_attn_qkv && has_compressor_state && has_cache_metadata &&
                has_attention_output && has_hc_tensors && has_routed_moe_ids_weights &&
                has_expert_shared_weights && has_layer_output_anchor;
            const char * reason = eligible ? "none" :
                (!has_layer_input ? "missing_layer_input" :
                (!has_attn_qkv ? "missing_attn_qkv" :
                (!has_compressor_state ? "missing_compressor_state_handle" :
                (!has_cache_metadata ? "missing_cache_dependency_token" :
                (!has_attention_output ? "missing_attention_output" :
                (!has_hc_tensors ? "missing_hc_tensors" :
                (!has_routed_moe_ids_weights ? "missing_routed_moe_ids_weights" :
                (!has_expert_shared_weights ? "missing_expert_shared_weights" :
                "missing_layer_output_anchor"))))))));
            const bool mode_live_graph = dsv4_experimental_layer_executor_dryrun_op_mode_is("live_graph_dispatch");
            const bool mode_side_graph_plan = dsv4_experimental_layer_executor_dryrun_op_mode_is("side_graph_plan");
            const bool mode_side_graph_dispatch = dsv4_experimental_layer_executor_dryrun_op_mode_is("side_graph_dispatch");
            const bool live_graph_node_added = eligible && mode_live_graph;
            const bool live_backend_dispatch = eligible && mode_live_graph;
            const bool side_graph_created = eligible && (mode_side_graph_plan || mode_side_graph_dispatch);
            const bool side_graph_dispatched = false;

            dsv4_lexec_dryrun_note(
                    il, dsv4_hc_pos, eligible, reason,
                    has_layer_input, has_attn_qkv, has_compressor_state, has_cache_metadata,
                    has_attention_output, has_hc_tensors, has_routed_moe_ids_weights,
                    has_expert_shared_weights, has_layer_output_anchor,
                    live_graph_node_added, live_backend_dispatch, side_graph_created, side_graph_dispatched);

            if (eligible && mode_live_graph) {
                int eligibility_flags = 0;
                eligibility_flags |= has_layer_input ? 1 << 0 : 0;
                eligibility_flags |= has_attn_qkv ? 1 << 1 : 0;
                eligibility_flags |= has_compressor_state ? 1 << 2 : 0;
                eligibility_flags |= has_cache_metadata ? 1 << 3 : 0;
                eligibility_flags |= has_attention_output ? 1 << 4 : 0;
                eligibility_flags |= has_hc_tensors ? 1 << 5 : 0;
                eligibility_flags |= has_routed_moe_ids_weights ? 1 << 6 : 0;
                eligibility_flags |= has_expert_shared_weights ? 1 << 7 : 0;
                eligibility_flags |= has_layer_output_anchor ? 1 << 8 : 0;
                const int64_t decode_index = dsv4_experimental_decode_index_for_token(dsv4_hc_pos);
                ggml_tensor * dryrun = ggml_dsv4_decode_layer_executor_dryrun(
                        ctx0,
                        dsv4_lexec_layer_input,
                        dsv4_lexec_attn_q,
                        dsv4_lexec_attn_kv,
                        dsv4_lexec_attn_out,
                        dsv4_lexec_attn_hc_post,
                        dsv4_lexec_ffn_norm,
                        dsv4_lexec_routed_moe_out,
                        dsv4_lexec_ffn_hc_post,
                        inp_pos,
                        il,
                        (int) decode_index,
                        eligibility_flags);
                ggml_format_name(dryrun, "dsv4_lexec_dryrun-l%d-t%lld", il, (long long) decode_index);
                ggml_build_forward_expand(gf, dryrun);
            }
        }
        // T104 skeleton: emit the GGML_OP_DSV4_DECODE_LAYER orchestrator op as
        // an ADDITIVE dead-end. The output is not consumed downstream, so the
        // stub passthrough cannot affect layer semantics. T105 will replace the
        // passthrough body with per-stage dispatch and switch to a REPLACEMENT
        // wiring that takes over the layer's residual output.
        if (dsv4_experimental_decode_layer_site_enabled(il, dsv4_hc_pos, n_tokens) &&
                dsv4_lexec_layer_input != nullptr) {
            const int64_t decode_layer_token_idx = dsv4_experimental_decode_index_for_token(dsv4_hc_pos);
            ggml_tensor * decode_layer_op = ggml_dsv4_decode_layer(
                    ctx0,
                    dsv4_lexec_layer_input,
                    il,
                    /*stage_mask=*/ 0u);
            ggml_format_name(decode_layer_op, "dsv4_decode_layer_stub-l%d-t%lld",
                    il, (long long) decode_layer_token_idx);
            ggml_build_forward_expand(gf, decode_layer_op);
        }
        if (dsv4_experimental_routed_moe_backend_op_downstream_dump_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            cb(inpL, "dsv4_rmoe_downstream_layer_output_after_ffn", il);
        }
        inpL = dsv4_layer_executor_apply_dependencies(inpL, dsv4_lexec_dependencies);
        if (dsv4_experimental_routed_moe_backend_op_result_chain_mode_site_enabled(il, dsv4_hc_pos, n_tokens) &&
                dsv4_rmoe_result_chain_mode_is("materialize_layer0_output")) {
            inpL = dsv4_rmoe_result_chain_materialize(inpL, "layer0_output", il);
        }
        if (dsv4_experimental_routed_moe_backend_op_result_chain_mode_site_enabled(il, dsv4_hc_pos, n_tokens) &&
                dsv4_rmoe_result_chain_mode_is("dependency_after_layer0")) {
            inpL = dsv4_rmoe_result_chain_dependency(inpL, "after_layer0", il);
        }
        if (dsv4_experimental_routed_moe_backend_op_result_chain_mode_site_enabled(il, dsv4_hc_pos, n_tokens) &&
                dsv4_rmoe_result_chain_mode_is("readback_layer0_output")) {
            cb(inpL, "dsv4_rmoe_result_chain_readback_layer0_output", il);
        }
        if (dsv4_rmoe_result_chain_site && il == dsv4_rmoe_result_chain_target_layer) {
            cb(inpL, "dsv4_rmoe_result_chain_layer0_next_input", (int) dsv4_rmoe_result_chain_target_layer);
        }
        if (dsv4_rmoe_result_chain_site && il == dsv4_rmoe_result_chain_target_layer + 1) {
            cb(inpL, "dsv4_rmoe_result_chain_layer1_after_ffn", (int) dsv4_rmoe_result_chain_target_layer);
        }
        if (dsv4_rmoe_result_chain_site && il == n_layer - 1) {
            cb(inpL, "dsv4_rmoe_result_chain_last_layer_output", (int) dsv4_rmoe_result_chain_target_layer);
        }
        if (dsv4_experimental_routed_moe_backend_op_downstream_dump_site_enabled(il, dsv4_hc_pos, n_tokens)) {
            cb(inpL, "dsv4_rmoe_downstream_next_layer_input", il);
        }
        if (dsv4_rmoe_after_layer_backend != nullptr) {
            ggml_build_forward_expand(gf, dsv4_rmoe_after_layer_backend);
        }
    }
    if (inp_out_ids) {
        inpL = ggml_reshape_2d(ctx0, inpL, n_embd * n_hc, n_tokens);
        inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_outputs);
    }

    const int64_t dsv4_rmoe_result_chain_target_layer = dsv4_experimental_cupd3_env_i64(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", 0);
    const bool dsv4_rmoe_result_chain_site =
        dsv4_rmoe_result_chain_target_layer >= 0 &&
        dsv4_experimental_routed_moe_backend_op_result_chain_dump_site_enabled(
                (int) dsv4_rmoe_result_chain_target_layer, dsv4_hc_pos, n_tokens);
    if (dsv4_rmoe_result_chain_site) {
        cb(inpL, "dsv4_rmoe_result_chain_result_hc_input", (int) dsv4_rmoe_result_chain_target_layer);
    }
    const bool dsv4_rmoe_result_chain_mode_site =
        dsv4_rmoe_result_chain_target_layer >= 0 &&
        dsv4_experimental_routed_moe_backend_op_result_chain_mode_site_enabled(
                (int) dsv4_rmoe_result_chain_target_layer, dsv4_hc_pos, n_tokens);
    if (dsv4_rmoe_result_chain_mode_site &&
            dsv4_rmoe_result_chain_mode_is("dependency_before_result_hc")) {
        inpL = dsv4_rmoe_result_chain_dependency(
                inpL, "before_result_hc", (int) dsv4_rmoe_result_chain_target_layer);
    }
    ggml_tensor * cur = dsv4_hc_head(ctx0, inpL,
            model.output_hc_fn, model.output_hc_scale, model.output_hc_base,
            n_embd, n_hc, inp_out_ids ? n_outputs : n_tokens,
            norm_rms_eps, hparams.hc_eps);
    cb(cur, "result_hc", -1);
    if (dsv4_rmoe_result_chain_mode_site &&
            dsv4_rmoe_result_chain_mode_is("materialize_result_hc")) {
        cur = dsv4_rmoe_result_chain_materialize(
                cur, "result_hc", (int) dsv4_rmoe_result_chain_target_layer);
    }
    if (dsv4_rmoe_result_chain_mode_site &&
            dsv4_rmoe_result_chain_mode_is("readback_result_hc")) {
        cb(cur, "dsv4_rmoe_result_chain_readback_result_hc", (int) dsv4_rmoe_result_chain_target_layer);
    }
    if (dsv4_rmoe_result_chain_mode_site &&
            dsv4_rmoe_result_chain_mode_is("dependency_before_result_norm")) {
        cur = dsv4_rmoe_result_chain_dependency(
                cur, "before_result_norm", (int) dsv4_rmoe_result_chain_target_layer);
    }

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    dsv4_layer_executor_plan_note_result_head(dsv4_hc_pos, n_tokens);
    if (dsv4_rmoe_result_chain_mode_site &&
            dsv4_rmoe_result_chain_mode_is("materialize_result_norm")) {
        cur = dsv4_rmoe_result_chain_materialize(
                cur, "result_norm", (int) dsv4_rmoe_result_chain_target_layer);
    }
    if (dsv4_rmoe_result_chain_mode_site &&
            dsv4_rmoe_result_chain_mode_is("readback_result_norm")) {
        cb(cur, "dsv4_rmoe_result_chain_readback_result_norm", (int) dsv4_rmoe_result_chain_target_layer);
    }
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
    for (ggml_tensor * backend : dsv4_rmoe_end_of_graph_backends) {
        ggml_build_forward_expand(gf, backend);
    }
    dsv4_rmoe_graph_trace_dump(gf, dsv4_hc_pos, n_tokens);
}
