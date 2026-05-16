#include "ggml-metal-ops.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include <cassert>
#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <limits>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

constexpr uint32_t GGML_METAL_RESOURCE_USAGE_READ  = 1u;
constexpr uint32_t GGML_METAL_RESOURCE_USAGE_WRITE = 2u;

static ggml_metal_buffer_id ggml_metal_get_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

static bool ggml_metal_mul_mat_id_experimental_split_glu_enabled(void);
static bool ggml_metal_dsv4_experimental_attn_out_decode_enabled(void);

static bool ggml_metal_dispatch_profile_enabled(void) {
    static int enabled = -1;
    if (enabled != -1) {
        return enabled == 1;
    }

    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISPATCH_PROFILE");
    if (value == nullptr || value[0] == '\0') {
        value = std::getenv("GGML_METAL_DISPATCH_PROFILE");
    }
    enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    return enabled == 1;
}

struct ggml_metal_dispatch_profile_entry {
    uint64_t calls = 0;
    uint64_t dispatches = 0;
};

static std::mutex g_ggml_metal_dispatch_profile_mutex;
static std::unordered_map<std::string, ggml_metal_dispatch_profile_entry> g_ggml_metal_dispatch_profile;

static std::string ggml_metal_dispatch_profile_normalize_name(const char * raw_name) {
    if (raw_name == nullptr || raw_name[0] == '\0') {
        return "(unnamed)";
    }

    std::string out;
    out.reserve(std::min<size_t>(std::strlen(raw_name), 96));
    bool in_digits = false;
    for (const char * p = raw_name; *p != '\0' && out.size() < 96; ++p) {
        const bool digit = *p >= '0' && *p <= '9';
        if (digit) {
            if (!in_digits) {
                out.push_back('#');
                in_digits = true;
            }
            continue;
        }
        in_digits = false;
        out.push_back(*p);
    }
    return out;
}

static void ggml_metal_dispatch_profile_record(const ggml_tensor * node, uint64_t dispatches) {
    if (!ggml_metal_dispatch_profile_enabled() || node == nullptr || dispatches == 0) {
        return;
    }

    std::string key = ggml_op_name(node->op);
    key.push_back(':');
    key += ggml_metal_dispatch_profile_normalize_name(ggml_get_name(node));

    std::lock_guard<std::mutex> lock(g_ggml_metal_dispatch_profile_mutex);
    ggml_metal_dispatch_profile_entry & entry = g_ggml_metal_dispatch_profile[key];
    entry.calls += 1;
    entry.dispatches += dispatches;
}

static void ggml_metal_dispatch_profile_log(void) {
    if (!ggml_metal_dispatch_profile_enabled()) {
        return;
    }

    std::vector<std::pair<std::string, ggml_metal_dispatch_profile_entry>> entries;
    {
        std::lock_guard<std::mutex> lock(g_ggml_metal_dispatch_profile_mutex);
        entries.reserve(g_ggml_metal_dispatch_profile.size());
        for (const auto & it : g_ggml_metal_dispatch_profile) {
            entries.push_back(it);
        }
    }

    if (entries.empty()) {
        return;
    }

    std::sort(entries.begin(), entries.end(), [](const auto & a, const auto & b) {
        if (a.second.dispatches != b.second.dispatches) {
            return a.second.dispatches > b.second.dispatches;
        }
        return a.second.calls > b.second.calls;
    });

    const size_t n = std::min<size_t>(entries.size(), 32);
    GGML_LOG_INFO("%s: top_dispatch_buckets=%zu\n", __func__, n);
    for (size_t i = 0; i < n; ++i) {
        GGML_LOG_INFO("%s: %2zu dispatch=%" PRIu64 " calls=%" PRIu64 " bucket=%s\n",
                __func__, i + 1, entries[i].second.dispatches, entries[i].second.calls, entries[i].first.c_str());
    }
}

static void ggml_metal_dispatch_profile_reset(void) {
    std::lock_guard<std::mutex> lock(g_ggml_metal_dispatch_profile_mutex);
    g_ggml_metal_dispatch_profile.clear();
}

static bool ggml_metal_dsv4_trace_enabled(void) {
    static int enabled = -1;
    if (enabled != -1) {
        return enabled == 1;
    }
    const char * value = std::getenv("LLAMA_FLASH_MOE_DSV4_TRACE");
    enabled = (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) ? 1 : 0;
    return enabled == 1;
}

static int ggml_metal_dsv4_trace_env_i(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

static const char * ggml_metal_dsv4_trace_path(void) {
    const char * value = std::getenv("LLAMA_FLASH_MOE_DSV4_TRACE_JSONL");
    return (value != nullptr && value[0] != '\0') ? value : "/tmp/dsv4_ours_trace.jsonl";
}

static std::mutex g_ggml_metal_dsv4_trace_mutex;
static FILE * g_ggml_metal_dsv4_trace_file = nullptr;

static bool ggml_metal_dsv4_trace_token_enabled(int token) {
    if (!ggml_metal_dsv4_trace_enabled()) {
        return false;
    }
    const int token_min = ggml_metal_dsv4_trace_env_i("LLAMA_FLASH_MOE_DSV4_TRACE_TOKEN_MIN", 32);
    const int token_max = ggml_metal_dsv4_trace_env_i("LLAMA_FLASH_MOE_DSV4_TRACE_TOKEN_MAX", 64);
    return token >= token_min && token <= token_max;
}

static FILE * ggml_metal_dsv4_trace_open_locked(void) {
    if (g_ggml_metal_dsv4_trace_file != nullptr) {
        return g_ggml_metal_dsv4_trace_file;
    }
    g_ggml_metal_dsv4_trace_file = std::fopen(ggml_metal_dsv4_trace_path(), "w");
    if (g_ggml_metal_dsv4_trace_file == nullptr) {
        std::fprintf(stderr, "ggml_metal_dsv4_trace: failed to open %s\n", ggml_metal_dsv4_trace_path());
    }
    return g_ggml_metal_dsv4_trace_file;
}

static void ggml_metal_dsv4_trace_json(FILE * f, const char * s) {
    std::fputc('"', f);
    if (s != nullptr) {
        for (const char * p = s; *p != '\0'; ++p) {
            const unsigned char c = (unsigned char) *p;
            if (c == '"' || c == '\\') {
                std::fputc('\\', f);
                std::fputc(c, f);
            } else if (c >= 0x20) {
                std::fputc(c, f);
            }
        }
    }
    std::fputc('"', f);
}

static int ggml_metal_dsv4_trace_layer_from_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }
    for (const char * p = name; *p != '\0'; ++p) {
        if ((p[0] == 'l' || p[0] == 'L') && std::isdigit((unsigned char) p[1])) {
            return std::atoi(p + 1);
        }
        if (p[0] == '_' && (p[1] == 'l' || p[1] == 'L') && std::isdigit((unsigned char) p[2])) {
            return std::atoi(p + 2);
        }
    }
    return -1;
}

static bool ggml_metal_dsv4_name_has(const char * haystack, const char * needle) {
    return haystack != nullptr && needle != nullptr && std::strstr(haystack, needle) != nullptr;
}

static const char * ggml_metal_dsv4_trace_stage_bucket(const ggml_tensor * node) {
    const char * name = node ? ggml_get_name(node) : "";
    const char * op = node ? ggml_op_name(node->op) : "";

    if (ggml_metal_dsv4_name_has(name, "ffn") || ggml_metal_dsv4_name_has(name, "moe")) {
        return "ffn";
    }
    if (ggml_metal_dsv4_name_has(name, "q_proj") || ggml_metal_dsv4_name_has(name, "k_proj") ||
        ggml_metal_dsv4_name_has(name, "v_proj") || ggml_metal_dsv4_name_has(name, "attn_q") ||
        ggml_metal_dsv4_name_has(name, "attn_k") || ggml_metal_dsv4_name_has(name, "attn_v")) {
        return "attn_qkv";
    }
    if (ggml_metal_dsv4_name_has(name, "compress") || ggml_metal_dsv4_name_has(name, "dcomp") ||
        ggml_metal_dsv4_name_has(name, "rope_fp8") || ggml_metal_dsv4_name_has(name, "rope_hfp4") ||
        node->op == GGML_OP_DSV4_DECODE_COMPRESS || node->op == GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE ||
        node->op == GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE_V2) {
        return "attn_compress";
    }
    if (ggml_metal_dsv4_name_has(name, "cache") || ggml_metal_dsv4_name_has(name, "kv") ||
        ggml_metal_dsv4_name_has(name, "set_rows") || node->op == GGML_OP_DSV4_KV_FINALIZE_DECODE) {
        return "kv_cache";
    }
    if (ggml_metal_dsv4_name_has(name, "flash_attn") || node->op == GGML_OP_FLASH_ATTN_EXT ||
        node->op == GGML_OP_DSV4_MIXED_ATTN) {
        return "attn_core";
    }
    if (ggml_metal_dsv4_name_has(name, "attn_out") || ggml_metal_dsv4_name_has(name, "aolow") ||
        ggml_metal_dsv4_name_has(name, "aodec") || ggml_metal_dsv4_name_has(name, "wo_") ||
        node->op == GGML_OP_DSV4_ATTN_OUT_DECODE) {
        return "attn_out";
    }
    if (ggml_metal_dsv4_name_has(name, "hc_attn_pre") || ggml_metal_dsv4_name_has(name, "hcnorm") ||
        ggml_metal_dsv4_name_has(name, "hcws") || node->op == GGML_OP_DSV4_HC_SPLIT_SINKHORN ||
        node->op == GGML_OP_DSV4_HC_WEIGHTED_SUM) {
        return "attn_hc_pre";
    }
    if (ggml_metal_dsv4_name_has(name, "hc_attn_post") || ggml_metal_dsv4_name_has(name, "hce") ||
        node->op == GGML_OP_DSV4_HC_EXPAND) {
        return "attn_hc_post";
    }
    if (ggml_metal_dsv4_name_has(name, "result_norm") || ggml_metal_dsv4_name_has(name, "output") || ggml_metal_dsv4_name_has(op, "SOFT_MAX")) {
        return "head";
    }
    if (ggml_metal_dsv4_name_has(name, "rope") || node->op == GGML_OP_ROPE || node->op == GGML_OP_DSV4_ROPE_TAIL ||
        node->op == GGML_OP_DSV4_FP8_KV_QUANTIZE || node->op == GGML_OP_DSV4_HADAMARD_FP4_QUANTIZE) {
        return "attn_kv";
    }
    return "other";
}

static const char * ggml_metal_dsv4_trace_detail_stage(const ggml_tensor * node, const char * stage_bucket) {
    const char * name = node ? ggml_get_name(node) : "";
    const char * op   = node ? ggml_op_name(node->op) : "";

    if (std::strcmp(stage_bucket ? stage_bucket : "", "ffn") != 0) {
        return stage_bucket ? stage_bucket : "other";
    }

    if (ggml_metal_dsv4_name_has(name, "ffn_norm")) {
        return "ffn_norm";
    }
    if (ggml_metal_dsv4_name_has(name, "moe_logits") ||
            ggml_metal_dsv4_name_has(name, "moe_probs") ||
            ggml_metal_dsv4_name_has(name, "moe_probs_biased")) {
        return "router";
    }
    if (ggml_metal_dsv4_name_has(name, "argsort") ||
            ggml_metal_dsv4_name_has(name, "hash_topk") ||
            ggml_metal_dsv4_name_has(name, "topk")) {
        return "topk";
    }
    if (ggml_metal_dsv4_name_has(name, "shexp")) {
        if (ggml_metal_dsv4_name_has(name, "swiglu") || ggml_metal_dsv4_name_has(op, "GLU")) {
            return "shared_swiglu";
        }
        if (ggml_metal_dsv4_name_has(op, "MUL_MAT")) {
            return "shared_down";
        }
        return "shared_gate_up";
    }
    if (ggml_metal_dsv4_name_has(name, "ffn_gate") || ggml_metal_dsv4_name_has(name, "ffn_up") ||
            ggml_metal_dsv4_name_has(name, "moe_gate") || ggml_metal_dsv4_name_has(name, "moe_up") ||
            ggml_metal_dsv4_name_has(name, "gate_clamped") || ggml_metal_dsv4_name_has(name, "up_clamped")) {
        return "expert_gate_up";
    }
    if (ggml_metal_dsv4_name_has(name, "swiglu") || ggml_metal_dsv4_name_has(name, "silu") ||
            ggml_metal_dsv4_name_has(op, "GLU") || ggml_metal_dsv4_name_has(name, "weighted_swiglu")) {
        return "expert_swiglu";
    }
    if (ggml_metal_dsv4_name_has(name, "moe_down")) {
        return "expert_down";
    }
    if (ggml_metal_dsv4_name_has(name, "moe_weights") ||
            ggml_metal_dsv4_name_has(name, "weights_sum") ||
            ggml_metal_dsv4_name_has(name, "weights_norm") ||
            ggml_metal_dsv4_name_has(name, "weights_scaled")) {
        return "expert_weighted_sum";
    }
    if (ggml_metal_dsv4_name_has(name, "ffn_out") || ggml_metal_dsv4_name_has(name, "hc_ffn_post")) {
        return "ffn_residual";
    }
    return "ffn_other";
}

static int ggml_metal_dsv4_trace_is_shared_stage(const char * detail_stage) {
    return detail_stage != nullptr && std::strstr(detail_stage, "shared_") == detail_stage ? 1 : 0;
}

static int ggml_metal_dsv4_trace_route_weight_applied(const char * name, const char * detail_stage) {
    if (detail_stage != nullptr && std::strcmp(detail_stage, "expert_weighted_sum") == 0) {
        return 1;
    }
    return ggml_metal_dsv4_name_has(name, "weighted") || ggml_metal_dsv4_name_has(name, "weights_scaled") ? 1 : 0;
}

static const char * ggml_metal_dsv4_trace_kernel_name(const ggml_tensor * node) {
    if (node == nullptr) {
        return "unknown";
    }
    switch (node->op) {
        case GGML_OP_DSV4_DECODE_COMPRESS: return "kernel_dsv4_decode_compress";
        case GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE: return "kernel_dsv4_compressor_update_decode";
        case GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE_V2: return "kernel_dsv4_compressor_update_decode_v2";
        case GGML_OP_DSV4_MIXED_ATTN: return "kernel_dsv4_mixed_attn_f16";
        case GGML_OP_DSV4_ATTN_OUT_DECODE: return "kernel_dsv4_attn_out_decode";
        case GGML_OP_DSV4_ROPE_TAIL: return "kernel_dsv4_rope_tail";
        case GGML_OP_DSV4_FP8_KV_QUANTIZE: return "kernel_dsv4_fp8_kv_quantize";
        case GGML_OP_DSV4_HADAMARD_FP4_QUANTIZE: return "kernel_dsv4_hadamard_fp4_quantize";
        case GGML_OP_DSV4_KV_FINALIZE_DECODE: return "kernel_dsv4_kv_finalize_decode";
        case GGML_OP_DSV4_HC_WEIGHTED_SUM: return "kernel_dsv4_hc_weighted_sum";
        case GGML_OP_DSV4_HC_EXPAND: return "kernel_dsv4_hc_expand";
        case GGML_OP_DSV4_FFN_MOE_DECODE_STAGE: return "kernel_dsv4_ffn_moe_decode_stage";
        case GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE: return "kernel_dsv4_routed_moe_one_tensor_decode_dryrun";
        case GGML_OP_DSV4_DECODE_LAYER_EXECUTOR_DRYRUN: return "kernel_dsv4_decode_layer_executor_dryrun";
        case GGML_OP_DSV4_DECODE_LAYER: return "kernel_dsv4_decode_layer_stub";
        default: return ggml_op_name(node->op);
    }
}

static int16_t ggml_metal_mul_mm_env_walk_mode() {
    static int16_t walk = -1;
    if (walk != -1) {
        return walk;
    }

    walk = GGML_METAL_MUL_MM_WALK_LEGACY;
    const char * value = std::getenv("GGML_METAL_MUL_MM_WALK");
    if (value == nullptr || value[0] == '\0') {
        return walk;
    }

    if (std::strcmp(value, "legacy") == 0) {
        walk = GGML_METAL_MUL_MM_WALK_LEGACY;
    } else if (std::strcmp(value, "regular") == 0) {
        walk = GGML_METAL_MUL_MM_WALK_REGULAR;
    } else if (std::strcmp(value, "morton") == 0) {
        walk = GGML_METAL_MUL_MM_WALK_MORTON;
    }

    return walk;
}

static int16_t ggml_metal_flash_attn_nonvec_env_walk_mode() {
    static int16_t walk = -1;
    if (walk != -1) {
        return walk;
    }

    walk = GGML_METAL_MUL_MM_WALK_LEGACY;
    const char * value = std::getenv("GGML_METAL_FLASH_ATTN_NONVEC_WALK");
    if (value == nullptr || value[0] == '\0') {
        return walk;
    }

    if (std::strcmp(value, "legacy") == 0) {
        walk = GGML_METAL_MUL_MM_WALK_LEGACY;
    } else if (std::strcmp(value, "regular") == 0) {
        walk = GGML_METAL_MUL_MM_WALK_REGULAR;
    } else if (std::strcmp(value, "morton") == 0) {
        walk = GGML_METAL_MUL_MM_WALK_MORTON;
    } else {
        std::fprintf(stderr,
                "%s: ignoring unsupported GGML_METAL_FLASH_ATTN_NONVEC_WALK=%s (supported: legacy, regular, morton)\n",
                __func__,
                value);
        std::fflush(stderr);
    }

    return walk;
}

static bool ggml_metal_flash_attn_mem_debug_enabled() {
    static int enabled = -1;
    if (enabled != -1) {
        return enabled == 1;
    }

    enabled = 0;
    const char * value = std::getenv("GGML_METAL_FLASH_ATTN_MEM_DEBUG");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    enabled = std::strcmp(value, "0") != 0 ? 1 : 0;
    return enabled == 1;
}

static bool ggml_metal_flash_attn_vec_metal4_env_enabled() {
    static int enabled = -1;
    if (enabled != -1) {
        return enabled == 1;
    }

    enabled = 0;
    const char * value = std::getenv("GGML_METAL_FLASH_ATTN_VEC_M4_ENABLE");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    enabled = std::strcmp(value, "0") != 0 ? 1 : 0;
    return enabled == 1;
}

static bool ggml_metal_flash_attn_nonvec_metal4_env_enabled() {
    static int enabled = -1;
    if (enabled != -1) {
        return enabled == 1;
    }

    enabled = 0;
    const char * value = std::getenv("GGML_METAL_FLASH_ATTN_NONVEC_M4_ENABLE");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    enabled = std::strcmp(value, "0") != 0 ? 1 : 0;
    return enabled == 1;
}

static bool ggml_metal_flash_attn_nonvec_2pass_env_enabled() {
    static int enabled = -1;
    if (enabled != -1) {
        return enabled == 1;
    }

    enabled = 1;
    const char * value = std::getenv("GGML_METAL_FLASH_ATTN_NONVEC_2PASS_ENABLE");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }

    enabled = std::strcmp(value, "0") != 0 ? 1 : 0;
    return enabled == 1;
}

static bool ggml_metal_flash_attn_vec_decode_single_wg_env_enabled() {
    static int enabled = -1;
    if (enabled != -1) {
        return enabled == 1;
    }

    enabled = 0;
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_SINGLE_WG");
    if (value == nullptr || value[0] == '\0') {
        value = std::getenv("GGML_METAL_FLASH_ATTN_VEC_SINGLE_WG");
    }
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    enabled = std::strcmp(value, "0") != 0 ? 1 : 0;
    return enabled == 1;
}

static int ggml_metal_flash_attn_vec_decode_nwg_env_override() {
    static int override_nwg = -2;
    if (override_nwg != -2) {
        return override_nwg;
    }

    override_nwg = -1;
    const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_NWG");
    if (value == nullptr || value[0] == '\0') {
        value = std::getenv("GGML_METAL_FLASH_ATTN_VEC_NWG");
    }
    if (value == nullptr || value[0] == '\0') {
        if (ggml_metal_mul_mat_id_experimental_split_glu_enabled()) {
            override_nwg = 16;
            return override_nwg;
        }
        return override_nwg;
    }

    const int parsed = std::atoi(value);
    if (parsed == 1 || parsed == 2 || parsed == 4 || parsed == 8 || parsed == 16 || parsed == 32) {
        override_nwg = parsed;
    } else {
        static bool warned = false;
        if (!warned) {
            GGML_LOG_INFO(
                    "%s: ignoring unsupported GGML_METAL_FLASH_ATTN_VEC_NWG=%s (supported: 1,2,4,8,16,32)\n",
                    __func__, value);
            warned = true;
        }
    }

    return override_nwg;
}

static bool ggml_metal_flash_attn_debug_enabled() {
    static int enabled = -1;
    if (enabled != -1) {
        return enabled == 1;
    }

    enabled = 0;
    const char * value = std::getenv("GGML_METAL_FLASH_ATTN_DEBUG");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    enabled = std::strcmp(value, "0") != 0 ? 1 : 0;
    return enabled == 1;
}

static int ggml_metal_flash_attn_nonvec_nsg_env_override() {
    static int override_nsg = -2;
    if (override_nsg != -2) {
        return override_nsg;
    }

    override_nsg = -1;

    const char * value = std::getenv("GGML_METAL_FLASH_ATTN_NONVEC_NSG");
    if (value == nullptr || value[0] == '\0') {
        return override_nsg;
    }

    const int parsed = std::atoi(value);
    switch (parsed) {
        case 1:
        case 2:
        case 4:
        case 8:
            override_nsg = parsed;
            break;
        default:
            std::fprintf(stderr,
                    "%s: ignoring unsupported GGML_METAL_FLASH_ATTN_NONVEC_NSG=%s (supported: 1,2,4,8)\n",
                    __func__,
                    value);
            std::fflush(stderr);
            break;
    }

    return override_nsg;
}

static int ggml_metal_flash_attn_nonvec_nwg_env_override() {
    static int override_nwg = -2;
    if (override_nwg != -2) {
        return override_nwg;
    }

    override_nwg = -1;

    const char * value = std::getenv("GGML_METAL_FLASH_ATTN_NONVEC_2PASS_NWG");
    if (value == nullptr || value[0] == '\0') {
        return override_nwg;
    }

    const int parsed = std::atoi(value);
    switch (parsed) {
        case 1:
        case 2:
        case 4:
        case 8:
        case 16:
            override_nwg = parsed;
            break;
        default:
            std::fprintf(stderr,
                    "%s: ignoring unsupported GGML_METAL_FLASH_ATTN_NONVEC_2PASS_NWG=%s (supported: 1,2,4,8,16)\n",
                    __func__,
                    value);
            std::fflush(stderr);
            break;
    }

    return override_nwg;
}

static bool ggml_metal_flash_attn_vec_metal4_supported_type(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16;
}

static bool ggml_metal_flash_attn_nonvec_metal4_supported_type(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

static bool ggml_metal_flash_attn_nonvec_2pass_supported_type(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32 || type == GGML_TYPE_BF16;
}

static int ggml_metal_flash_attn_nonvec_chunk_mode_env_override() {
    static int override_mode = -2;
    if (override_mode != -2) {
        return override_mode;
    }

    override_mode = -1;

    const char * value = std::getenv("GGML_METAL_FLASH_ATTN_NONVEC_2PASS_CHUNK_MODE");
    if (value == nullptr || value[0] == '\0') {
        return override_mode;
    }

    if (std::strcmp(value, "strided") == 0 || std::strcmp(value, "legacy") == 0 || std::strcmp(value, "0") == 0) {
        override_mode = GGML_METAL_FLASH_ATTN_CHUNK_STRIDED;
        return override_mode;
    }

    if (std::strcmp(value, "contiguous") == 0 || std::strcmp(value, "chunked") == 0 || std::strcmp(value, "1") == 0) {
        override_mode = GGML_METAL_FLASH_ATTN_CHUNK_CONTIGUOUS;
        return override_mode;
    }

    std::fprintf(stderr,
            "%s: ignoring unsupported GGML_METAL_FLASH_ATTN_NONVEC_2PASS_CHUNK_MODE=%s (supported: strided, contiguous)\n",
            __func__,
            value);
    std::fflush(stderr);

    return override_mode;
}

static int ggml_metal_flash_attn_nonvec_chunk_target_tokens_env_override() {
    static int override_tokens = -2;
    if (override_tokens != -2) {
        return override_tokens;
    }

    override_tokens = -1;

    const char * value = std::getenv("GGML_METAL_FLASH_ATTN_NONVEC_2PASS_CHUNK_TOKENS");
    if (value == nullptr || value[0] == '\0') {
        return override_tokens;
    }

    const int parsed = std::atoi(value);
    if (parsed > 0) {
        override_tokens = parsed;
        return override_tokens;
    }

    std::fprintf(stderr,
            "%s: ignoring unsupported GGML_METAL_FLASH_ATTN_NONVEC_2PASS_CHUNK_TOKENS=%s (must be > 0)\n",
            __func__,
            value);
    std::fflush(stderr);

    return override_tokens;
}

static int32_t ggml_metal_flash_attn_nonvec_chunk_mode_hint(const ggml_tensor * op) {
    if (op == nullptr) {
        return GGML_METAL_FLASH_ATTN_CHUNK_STRIDED;
    }

    const int override_mode = ggml_metal_flash_attn_nonvec_chunk_mode_env_override();
    if (override_mode >= 0) {
        return override_mode;
    }

    // Keep contiguous chunk ownership opt-in only for now. It can help some
    // 32K-48K prefill cases, but it is still numerically different from the
    // strided reference path and has regressed at 64K in current testing.
    return GGML_METAL_FLASH_ATTN_CHUNK_STRIDED;
}

static int32_t ggml_metal_flash_attn_nonvec_chunk_mode(const ggml_tensor * op, int32_t nwg) {
    if (op == nullptr || nwg <= 1) {
        return GGML_METAL_FLASH_ATTN_CHUNK_STRIDED;
    }

    return ggml_metal_flash_attn_nonvec_chunk_mode_hint(op);
}

static int32_t ggml_metal_flash_attn_nonvec_round_up_pow2(int32_t value) {
    int32_t result = 1;
    while (result < value && result < 16) {
        result <<= 1;
    }

    return result;
}

static int32_t ggml_metal_flash_attn_nonvec_nwg_for_contiguous_chunks(int64_t kv) {
    const int override_target = ggml_metal_flash_attn_nonvec_chunk_target_tokens_env_override();
    const int32_t target_tokens = override_target > 0 ? override_target : 16384;

    if (kv < 2LL*target_tokens) {
        return 1;
    }

    const int32_t desired_chunks = (int32_t) ((kv + target_tokens - 1)/target_tokens);
    return ggml_metal_flash_attn_nonvec_round_up_pow2(desired_chunks);
}

static bool ggml_metal_is_token_char(char c) {
    return (c >= '0' && c <= '9') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z');
}

static bool ggml_metal_text_contains_token(const char * text, const char * token) {
    if (text == nullptr || text[0] == '\0' || token == nullptr || token[0] == '\0') {
        return false;
    }

    const size_t token_len = std::strlen(token);
    const char * pos = text;

    while ((pos = std::strstr(pos, token)) != nullptr) {
        const char prev = pos == text ? '\0' : pos[-1];
        const char next = pos[token_len];
        if (!ggml_metal_is_token_char(prev) && !ggml_metal_is_token_char(next)) {
            return true;
        }
        pos += token_len;
    }

    return false;
}

static bool ggml_metal_device_name_contains_token(
        const ggml_metal_device_props * props_dev,
        const char * token) {
    if (props_dev == nullptr) {
        return false;
    }

    return ggml_metal_text_contains_token(props_dev->name, token) ||
           ggml_metal_text_contains_token(props_dev->desc, token);
}

static int32_t ggml_metal_flash_attn_nonvec_nwg(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr) {
        return 1;
    }

    if (!ggml_metal_flash_attn_nonvec_2pass_env_enabled()) {
        return 1;
    }

    if ((op->src[0]->ne[1] < 20) && (op->src[0]->ne[0] % 32 == 0)) {
        return 1;
    }

    if (op->src[0]->ne[1] <= 1) {
        return 1;
    }

    if (op->src[1]->ne[1] < 16384) {
        return 1;
    }

    if (op->src[1]->type != op->src[2]->type) {
        return 1;
    }

    if (!ggml_metal_flash_attn_nonvec_2pass_supported_type(op->src[1]->type)) {
        return 1;
    }

    const int override_nwg = ggml_metal_flash_attn_nonvec_nwg_env_override();
    if (override_nwg > 0) {
        return override_nwg;
    }

    const int64_t kv = op->src[1]->ne[1];
    const int32_t chunk_mode_hint = ggml_metal_flash_attn_nonvec_chunk_mode_hint(op);

    if (chunk_mode_hint == GGML_METAL_FLASH_ATTN_CHUNK_CONTIGUOUS) {
        return ggml_metal_flash_attn_nonvec_nwg_for_contiguous_chunks(kv);
    }

    if (kv > 32768) {
        return 8;
    }

    return 4;
}

static bool ggml_metal_flash_attn_vec_use_metal4(
        const ggml_metal_device_props * props_dev,
        const ggml_tensor * op) {
    if (props_dev == nullptr || op == nullptr || !props_dev->has_tensor) {
        return false;
    }

    if (!ggml_metal_flash_attn_vec_metal4_env_enabled()) {
        return false;
    }

    if (!ggml_metal_device_name_contains_token(props_dev, "M5")) {
        return false;
    }

    if (op->src[1] == nullptr || op->src[2] == nullptr) {
        return false;
    }

    if (op->src[1]->type != op->src[2]->type) {
        return false;
    }

    if (!ggml_metal_flash_attn_vec_metal4_supported_type(op->src[1]->type)) {
        return false;
    }

    return op->src[1]->ne[0] == 128 && op->src[2]->ne[0] == 128;
}

static bool ggml_metal_flash_attn_nonvec_use_metal4(
        const ggml_metal_device_props * props_dev,
        const ggml_tensor * op) {
    if (props_dev == nullptr || op == nullptr || !props_dev->has_tensor) {
        return false;
    }

    if (!ggml_metal_flash_attn_nonvec_metal4_env_enabled()) {
        return false;
    }

    if (!ggml_metal_device_name_contains_token(props_dev, "M5")) {
        return false;
    }

    if (op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr) {
        return false;
    }

    if (op->src[1]->type != op->src[2]->type) {
        return false;
    }

    if (!ggml_metal_flash_attn_nonvec_metal4_supported_type(op->src[1]->type)) {
        return false;
    }

    return op->src[1]->ne[0] == 128 && op->src[2]->ne[0] == 128;
}

static const char * ggml_metal_walk_mode_name(int32_t walk_mode) {
    switch (walk_mode) {
        case GGML_METAL_MUL_MM_WALK_REGULAR: return "regular";
        case GGML_METAL_MUL_MM_WALK_MORTON:  return "morton";
        case GGML_METAL_MUL_MM_WALK_LEGACY:
        default: return "legacy";
    }
}

static const char * ggml_metal_flash_attn_chunk_mode_name(int32_t chunk_mode) {
    switch (chunk_mode) {
        case GGML_METAL_FLASH_ATTN_CHUNK_CONTIGUOUS: return "contiguous";
        case GGML_METAL_FLASH_ATTN_CHUNK_STRIDED:
        default: return "strided";
    }
}

static void ggml_metal_flash_attn_log_path_once(
        const ggml_tensor * op,
        bool is_vec,
        bool use_metal4_sdpa,
        bool has_mask,
        bool has_bias,
        bool has_scap,
        bool has_kvpad,
        int32_t nsg,
        int32_t walk_mode,
        int32_t chunk_mode,
        int32_t nwg) {
    if (!ggml_metal_flash_attn_debug_enabled() || op == nullptr || op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr) {
        return;
    }

    const int32_t q  = int32_t(op->src[0]->ne[1]);
    const int32_t kv = int32_t(op->src[1]->ne[1]);
    const int32_t dk = int32_t(op->src[1]->ne[0]);
    const int32_t dv = int32_t(op->src[2]->ne[0]);
    const char * phase = q <= 1 ? "decode-like" : "prefill-like";
    const char * qt = ggml_type_name(op->src[0]->type);
    const char * kt = ggml_type_name(op->src[1]->type);
    const char * vt = ggml_type_name(op->src[2]->type);

    char key[256];
    std::snprintf(
            key, sizeof(key),
            "%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d|%d",
            is_vec ? 1 : 0,
            use_metal4_sdpa ? 1 : 0,
            q, kv, dk, dv,
            has_mask ? 1 : 0,
            has_bias ? 1 : 0,
            has_scap ? 1 : 0,
            nsg,
            walk_mode,
            chunk_mode,
            nwg);

    static std::mutex mu;
    static std::unordered_map<std::string, bool> seen;

    {
        std::lock_guard<std::mutex> lock(mu);
        if (!seen.emplace(key, true).second) {
            return;
        }
    }

    std::fprintf(
            stderr,
            "%s: path=%s%s%s phase=%s q=%d kv=%d dk=%d dv=%d nsg=%d nwg=%d walk=%s chunk=%s mask=%s bias=%s scap=%s kvpad=%s\n",
            __func__,
            is_vec ? "vec" : "non-vec",
            nwg > 1 ? "+2pass" : "",
            use_metal4_sdpa ? "+metal4" : "",
            phase,
            q,
            kv,
            dk,
            dv,
            nsg,
            nwg,
            ggml_metal_walk_mode_name(walk_mode),
            ggml_metal_flash_attn_chunk_mode_name(chunk_mode),
            has_mask ? "yes" : "no",
            has_bias ? "yes" : "no",
            has_scap ? "yes" : "no",
            has_kvpad ? "yes" : "no");
    std::fprintf(stderr, "%s: tensor types q=%s k=%s v=%s\n", __func__, qt, kt, vt);
    std::fflush(stderr);
}

static void ggml_metal_flash_attn_log_mem_once(
        const ggml_tensor * op,
        bool is_vec,
        bool use_metal4_sdpa,
        bool has_mask,
        bool has_kvpad,
        int32_t nsg,
        int32_t chunk_mode,
        int32_t nwg) {
    if (!ggml_metal_flash_attn_mem_debug_enabled() || op == nullptr || op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr) {
        return;
    }

    const int32_t q  = int32_t(op->src[0]->ne[1]);
    const int32_t kv = int32_t(op->src[1]->ne[1]);
    const int32_t dk = int32_t(op->src[1]->ne[0]);
    const int32_t dv = int32_t(op->src[2]->ne[0]);
    const char * phase = q <= 1 ? "decode-like" : "prefill-like";

    char key[256];
    std::snprintf(
            key, sizeof(key),
            "%d|%d|%d|%d|%d|%d|%d|%d|%d|%d",
            is_vec ? 1 : 0,
            use_metal4_sdpa ? 1 : 0,
            q, kv, dk, dv,
            has_mask ? 1 : 0,
            nsg,
            chunk_mode,
            nwg);

    static std::mutex mu;
    static std::unordered_map<std::string, bool> seen;

    {
        std::lock_guard<std::mutex> lock(mu);
        if (!seen.emplace(key, true).second) {
            return;
        }
    }

    const size_t dst_bytes   = ggml_nbytes(op);
    const size_t extra_pad   = ggml_metal_op_flash_attn_ext_extra_pad(op);
    const size_t extra_blk   = ggml_metal_op_flash_attn_ext_extra_blk(op);
    const size_t extra_tmp   = ggml_metal_op_flash_attn_ext_extra_tmp(op);
    const size_t extra_total = extra_pad + extra_blk + extra_tmp;
    const size_t op_bound    = dst_bytes + extra_total;

    const bool pad_reserved       = extra_pad != 0;
    const bool pad_forced_reserve = pad_reserved && !has_kvpad;
    const bool tmp_reserved       = extra_tmp != 0;
    const bool tmp_used_runtime   = is_vec ? (nwg > 1) : (nwg > 1);
    const bool tmp_forced_reserve = tmp_reserved && !tmp_used_runtime;

    std::fprintf(
            stderr,
            "%s: path=%s%s%s phase=%s q=%d kv=%d dk=%d dv=%d nsg=%d nwg=%d chunk=%s dst=%zu extra_pad=%zu extra_blk=%zu extra_tmp=%zu extra_total=%zu op_bound=%zu kvpad_runtime=%s pad_reserved=%s pad_forced=%s tmp_reserved=%s tmp_runtime=%s tmp_forced=%s\n",
            __func__,
            is_vec ? "vec" : "non-vec",
            nwg > 1 ? "+2pass" : "",
            use_metal4_sdpa ? "+metal4" : "",
            phase,
            q,
            kv,
            dk,
            dv,
            nsg,
            nwg,
            ggml_metal_flash_attn_chunk_mode_name(chunk_mode),
            dst_bytes,
            extra_pad,
            extra_blk,
            extra_tmp,
            extra_total,
            op_bound,
            has_kvpad ? "yes" : "no",
            pad_reserved ? "yes" : "no",
            pad_forced_reserve ? "yes" : "no",
            tmp_reserved ? "yes" : "no",
            tmp_used_runtime ? "yes" : "no",
            tmp_forced_reserve ? "yes" : "no");
    std::fflush(stderr);
}

static int32_t ggml_metal_mul_mm_dispatch_extent_morton(int32_t tiles_x, int32_t tiles_y) {
    int32_t side = 1;
    while (side < tiles_x || side < tiles_y) {
        side <<= 1;
    }
    return side * side;
}

static bool ggml_metal_mul_mat_id_get_decode_expert_ids(
        const ggml_tensor * ids,
        int32_t * expert_ids,
        int64_t n_ids,
        int64_t token_idx) {
    if (ids == nullptr || expert_ids == nullptr || ids->type != GGML_TYPE_I32 || ids->data == nullptr) {
        return false;
    }

    if (token_idx < 0 || token_idx >= ids->ne[1] || n_ids < 0 || n_ids > ids->ne[0]) {
        return false;
    }

    const char * ids_data = static_cast<const char *>(ids->data) + token_idx*ids->nb[1];
    for (int64_t i = 0; i < n_ids; ++i) {
        memcpy(&expert_ids[i], ids_data + i*ids->nb[0], sizeof(expert_ids[i]));
    }

    return true;
}

static bool ggml_metal_mul_mat_id_ids_are_decode_ready(const ggml_tensor * ids) {
    if (ids == nullptr || ids->type != GGML_TYPE_I32) {
        return false;
    }

    return ids->op == GGML_OP_NONE || ids->op == GGML_OP_MAP_CUSTOM1;
}

static bool ggml_metal_mul_mat_id_materialize_ids_if_needed(
        const ggml_tensor * op,
        ggml_metal_buffer_id bid_dst,
        ggml_metal_buffer_id & bid_ids,
        uint64_t & nb21_out) {
    const ggml_tensor * ids = op != nullptr ? op->src[2] : nullptr;
    if (ids == nullptr || ids->type != GGML_TYPE_I32) {
        return false;
    }

    if (bid_ids.metal != nullptr) {
        nb21_out = ids->nb[1];
        return true;
    }

    if (!ggml_metal_mul_mat_id_ids_are_decode_ready(ids)) {
        return false;
    }

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_expert_used <= 0 || n_tokens <= 0) {
        return false;
    }

    std::vector<int32_t> translated_ids(size_t(n_expert_used * n_tokens));
    for (int64_t token_idx = 0; token_idx < n_tokens; ++token_idx) {
        if (!ggml_metal_mul_mat_id_get_decode_expert_ids(
                ids,
                translated_ids.data() + size_t(token_idx * n_expert_used),
                n_expert_used,
                token_idx)) {
            return false;
        }
    }

    ggml_backend_buffer_t dst_buf = op->view_src ? op->view_src->buffer : op->buffer;
    if (dst_buf == nullptr) {
        return false;
    }

    ggml_metal_buffer_t metal_buf = (ggml_metal_buffer_t) dst_buf->context;
    if (metal_buf == nullptr) {
        return false;
    }

    const size_t ids_offset = ggml_nbytes(op) + ggml_metal_op_mul_mat_id_extra_tpe(op);
    ggml_metal_buffer_set_tensor(
            metal_buf,
            const_cast<ggml_tensor *>(op),
            translated_ids.data(),
            ids_offset,
            translated_ids.size() * sizeof(int32_t));

    bid_ids = bid_dst;
    bid_ids.offs += ids_offset;
    nb21_out = uint64_t(n_expert_used) * sizeof(int32_t);
    return true;
}

static bool ggml_metal_mul_mat_id_disable_decode_fast_path_for_op(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr) {
        return false;
    }

    const ggml_type type = op->src[0]->type;
    if (type != GGML_TYPE_IQ2_XXS && type != GGML_TYPE_IQ4_NL) {
        return false;
    }

    const char * name = ggml_get_name(op->src[0]);
    if (name == nullptr || name[0] == '\0') {
        return false;
    }

    return strstr(name, ".ffn_gate_up_exps.") != nullptr ||
           strstr(name, ".ffn_down_exps.") != nullptr;
}

static bool ggml_metal_mul_mat_id_experimental_split_glu_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU");
        enabled = (value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_split_glu_encode_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU_NO_FUSE");
        enabled = (value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0) ? 0 : 1;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_dsv4_limited_swiglu_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LIMITED_SWIGLU");
        if (value == nullptr || value[0] == '\0') {
            enabled = ggml_metal_mul_mat_id_experimental_split_glu_encode_enabled() ? 1 : 0;
        } else {
            enabled = strcmp(value, "0") != 0 ? 1 : 0;
        }
    }
    return enabled == 1;
}

static bool ggml_metal_env_flag_enabled_with_default(const char * name, bool default_enabled) {
    const char * value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_enabled;
    }
    return strcmp(value, "0") != 0;
}

static bool ggml_metal_mul_mat_m5_sgmatrix_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_M5_SGMATRIX",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_dsv4_pair_swiglu_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_PAIR_SWIGLU",
                ggml_metal_mul_mat_id_experimental_dsv4_limited_swiglu_enabled()) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_dsv4_weighted_swiglu_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_SWIGLU",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_rmoe_pair_preserve_match_trace_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_PAIR_PRESERVE_MATCH_TRACE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_dsv4_down_sum6_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_dsv4_shared_swiglu_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_SHARED_SWIGLU",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_dsv4_weighted_down_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_DOWN");
        enabled = (value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_pair_gate_up_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_PAIR_GATE_UP");
        enabled = (value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_decode_replay_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_decode_icb_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_ICB",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_rope_hadamard_fp4_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROPE_HFP4",
                ggml_metal_mul_mat_id_experimental_split_glu_enabled()) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_rope_fp8_kv_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROPE_FP8",
                ggml_metal_mul_mat_id_experimental_split_glu_enabled()) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_rope_fp8_kv_set_rows_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_SET_ROWS",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_hc_split_weighted_sum_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_SPLIT_WS",
                ggml_metal_mul_mat_id_experimental_split_glu_enabled()) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_decode_compress_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = (ggml_metal_env_flag_enabled_with_default(
                    "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DECODE_COMPRESS",
                    false) ||
                ggml_metal_env_flag_enabled_with_default(
                    "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL",
                    false)) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_attn_out_low_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_LOW",
                ggml_metal_mul_mat_id_experimental_split_glu_enabled()) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_attn_out_decode_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_DECODE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_trace_attn_out_decode_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_DECODE_TRACE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_compressor_update_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_compressor_update_v2_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_kv_finalize_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_ffn_moe_stage_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_ffn_moe_stage_compare_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_COMPARE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_trace_compressor_update_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_TRACE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_trace_compressor_update_v2_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_TRACE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_trace_kv_finalize_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_TRACE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_trace_ffn_moe_stage_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_TRACE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_compressor_update_fused_comp_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_FUSED_COMP",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_compressor_update_v2_fused_comp_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_q8_hc_expand_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_Q8_HC_EXPAND",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_aohc_fused_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_trace_aohc_fused_elig_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_ELIG_TRACE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_trace_q8hc_elig_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_Q8HC_ELIG_TRACE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_is_aohc_fused_high(const ggml_tensor * op) {
    return op != nullptr && std::strstr(op->name, "dsv4_aohc_fused_high") != nullptr;
}

static bool ggml_metal_dsv4_experimental_hc_pre_norm_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM",
                ggml_metal_mul_mat_id_experimental_split_glu_enabled()) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_experimental_hc_expand4_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_EXPAND4",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_trace_hc_pre_norm_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_TRACE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_compare_hc_pre_norm_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_COMPARE",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static const char * ggml_metal_dsv4_hc_pre_norm_scope(void) {
    static const char * scope = nullptr;
    static bool initialized = false;
    if (!initialized) {
        scope = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_SCOPE");
        initialized = true;
    }
    return scope != nullptr ? scope : "";
}

static bool ggml_metal_dsv4_trace_q8_hc_expand_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_Q8_HC_EXPAND_TRACE",
                false) ? 1 : 0;
    }
    return enabled == 1 || ggml_metal_dsv4_trace_q8hc_elig_enabled();
}

static bool ggml_metal_dsv4_experimental_compressor_pair_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_PAIR",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_dsv4_trace_compressor_pair_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        enabled = ggml_metal_env_flag_enabled_with_default(
                "LLAMA_FLASH_MOE_TRACE_DSV4_COMPRESSOR_PAIR",
                false) ? 1 : 0;
    }
    return enabled == 1;
}

static size_t ggml_metal_mul_mat_id_experimental_decode_replay_cache_limit(void) {
    static size_t limit = 0;
    if (limit == 0) {
        limit = 8192;

        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY_CACHE_LIMIT");
        if (value != nullptr && value[0] != '\0') {
            char * end = nullptr;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (end != value && parsed > 0) {
                limit = size_t(parsed);
            }
        }
    }
    return limit;
}

static bool ggml_metal_experimental_disable_generic_mm_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_GENERIC_MM");
        enabled = (value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_experimental_disable_mul_mm_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_MUL_MM");
        enabled = (value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1 || ggml_metal_experimental_disable_generic_mm_enabled();
}

static bool ggml_metal_experimental_disable_mul_mm_id_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_MUL_MM_ID");
        enabled = (value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1 || ggml_metal_experimental_disable_generic_mm_enabled();
}

static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_decode_mv_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_generic_mv_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_generic_mm_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_fused_glu_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_pair_gate_up_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_pair_swiglu_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_weighted_swiglu_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_down_sum6_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_down_sum6_weighted_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_split_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_decode_replay_hit_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_decode_replay_miss_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_decode_replay_insert_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_decode_replay_clear_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_decode_icb_exec_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_id_decode_icb_build_fail_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_mv_ext_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_mv_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_mm_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_mm_m5_expert_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_mm_m5_sgmatrix_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_mul_mat_shared_swiglu_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_compressor_pair_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_compressor_pair_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_rope_hadamard_fp4_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_rope_fp8_kv_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_rope_fp8_kv_set_rows_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_hc_split_weighted_sum_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_indexer_weighted_score_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_decode_compress_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_mixed_attn_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_attn_out_low_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_attn_out_decode_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_compressor_update_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_compressor_update_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_compressor_update_v2_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_compressor_update_v2_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_kv_finalize_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_kv_finalize_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_ffn_moe_stage_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_ffn_moe_stage_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_routed_moe_one_tensor_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_routed_moe_one_tensor_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_decode_layer_executor_dryrun_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_decode_layer_executor_dryrun_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_decode_layer_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_decode_layer_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_q8_hc_expand_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_q8_hc_expand_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_aohc_elig_candidate_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_aohc_elig_q8hc_eligible_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_aohc_elig_q8hc_rejected_count { 0 };
static std::mutex g_ggml_metal_dsv4_aohc_elig_mutex;
static std::string g_ggml_metal_dsv4_aohc_elig_first_reject_reason;
static std::string g_ggml_metal_dsv4_aohc_elig_first_reject_name;
static int g_ggml_metal_dsv4_aohc_elig_first_reject_idx = -1;
static std::atomic<uint64_t> g_ggml_metal_dsv4_hc_pre_norm_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_hc_pre_norm_trace_count { 0 };
static std::atomic<uint64_t> g_ggml_metal_dsv4_hc_expand4_count { 0 };
static thread_local int g_ggml_metal_mul_mat_m5_expert_scope_depth = 0;

constexpr int64_t GGML_METAL_MUL_MAT_ID_DECODE_REPLAY_MAX_EXPERTS = 32;

struct ggml_metal_mul_mat_id_decode_replay_key {
    ggml_type src0_type = GGML_TYPE_F32;
    ggml_type src1_type = GGML_TYPE_F32;
    int32_t ne00 = 0;
    int32_t ne01 = 0;
    int32_t ne02 = 0;
    int32_t ne10 = 0;
    int32_t ne11 = 0;
    int32_t ne0 = 0;
    uint64_t nb00 = 0;
    uint64_t nb01 = 0;
    uint64_t nb02 = 0;
    uint64_t nb10 = 0;
    uint64_t nb11 = 0;
    uint64_t nb_dst1 = 0;
    int32_t n_experts = 0;
    bool use_direct_dispatch = false;
    bool bind_runtime_buffers = false;
    uintptr_t src0_metal = 0;
    uintptr_t src1_metal = 0;
    uintptr_t dst_metal  = 0;
    uint64_t src0_base_offset = 0;
    uint64_t src1_base_offset = 0;
    uint64_t dst_base_offset  = 0;
    std::array<int32_t, GGML_METAL_MUL_MAT_ID_DECODE_REPLAY_MAX_EXPERTS> expert_ids = {};

    bool operator==(const ggml_metal_mul_mat_id_decode_replay_key & other) const {
        return src0_type == other.src0_type &&
                src1_type == other.src1_type &&
                ne00 == other.ne00 &&
                ne01 == other.ne01 &&
                ne02 == other.ne02 &&
                ne10 == other.ne10 &&
                ne11 == other.ne11 &&
                ne0 == other.ne0 &&
                nb00 == other.nb00 &&
                nb01 == other.nb01 &&
                nb02 == other.nb02 &&
                nb10 == other.nb10 &&
                nb11 == other.nb11 &&
                nb_dst1 == other.nb_dst1 &&
                n_experts == other.n_experts &&
                use_direct_dispatch == other.use_direct_dispatch &&
                bind_runtime_buffers == other.bind_runtime_buffers &&
                src0_metal == other.src0_metal &&
                src1_metal == other.src1_metal &&
                dst_metal == other.dst_metal &&
                src0_base_offset == other.src0_base_offset &&
                src1_base_offset == other.src1_base_offset &&
                dst_base_offset == other.dst_base_offset &&
                expert_ids == other.expert_ids;
    }
};

struct ggml_metal_mul_mat_id_decode_replay_key_hash {
    size_t operator()(const ggml_metal_mul_mat_id_decode_replay_key & key) const {
        size_t seed = 0;
        auto hash_combine = [&](size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };

        hash_combine(std::hash<int>{}(int(key.src0_type)));
        hash_combine(std::hash<int>{}(int(key.src1_type)));
        hash_combine(std::hash<int32_t>{}(key.ne00));
        hash_combine(std::hash<int32_t>{}(key.ne01));
        hash_combine(std::hash<int32_t>{}(key.ne02));
        hash_combine(std::hash<int32_t>{}(key.ne10));
        hash_combine(std::hash<int32_t>{}(key.ne11));
        hash_combine(std::hash<int32_t>{}(key.ne0));
        hash_combine(std::hash<uint64_t>{}(key.nb00));
        hash_combine(std::hash<uint64_t>{}(key.nb01));
        hash_combine(std::hash<uint64_t>{}(key.nb02));
        hash_combine(std::hash<uint64_t>{}(key.nb10));
        hash_combine(std::hash<uint64_t>{}(key.nb11));
        hash_combine(std::hash<uint64_t>{}(key.nb_dst1));
        hash_combine(std::hash<int32_t>{}(key.n_experts));
        hash_combine(std::hash<bool>{}(key.use_direct_dispatch));
        hash_combine(std::hash<bool>{}(key.bind_runtime_buffers));
        hash_combine(std::hash<uintptr_t>{}(key.src0_metal));
        hash_combine(std::hash<uintptr_t>{}(key.src1_metal));
        hash_combine(std::hash<uintptr_t>{}(key.dst_metal));
        hash_combine(std::hash<uint64_t>{}(key.src0_base_offset));
        hash_combine(std::hash<uint64_t>{}(key.src1_base_offset));
        hash_combine(std::hash<uint64_t>{}(key.dst_base_offset));
        for (int32_t i = 0; i < key.n_experts; ++i) {
            hash_combine(std::hash<int32_t>{}(key.expert_ids[i]));
        }

        return seed;
    }
};

struct ggml_metal_mul_mat_id_decode_replay_entry {
    ggml_metal_kargs_mul_mv args = {};
    size_t smem = 0;
    int tg0 = 0;
    int tptg0 = 32;
    int tptg1 = 1;
    int n_experts = 0;
    std::array<uint64_t, GGML_METAL_MUL_MAT_ID_DECODE_REPLAY_MAX_EXPERTS> src0_offsets = {};
    std::array<uint64_t, GGML_METAL_MUL_MAT_ID_DECODE_REPLAY_MAX_EXPERTS> src1_offsets = {};
    std::array<uint64_t, GGML_METAL_MUL_MAT_ID_DECODE_REPLAY_MAX_EXPERTS> dst_offsets = {};

    ggml_metal_owned_buffer_t args_buffer = nullptr;
    ggml_metal_icb_t icb = nullptr;

    ~ggml_metal_mul_mat_id_decode_replay_entry() {
        ggml_metal_icb_free(icb);
        ggml_metal_owned_buffer_free(args_buffer);
    }
};

static std::mutex g_ggml_metal_mul_mat_id_decode_replay_mutex;
static std::unordered_map<
        ggml_metal_mul_mat_id_decode_replay_key,
        std::shared_ptr<ggml_metal_mul_mat_id_decode_replay_entry>,
        ggml_metal_mul_mat_id_decode_replay_key_hash> g_ggml_metal_mul_mat_id_decode_replay_cache;

static bool ggml_metal_mul_mat_id_decode_mv_uses_direct_dispatch(const ggml_tensor * op) {
    if (op == nullptr || op->src[0] == nullptr) {
        return false;
    }

    return op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0;
}

static ggml_metal_mul_mat_id_decode_replay_key ggml_metal_mul_mat_id_decode_replay_make_key(
        const ggml_tensor * op,
        const int32_t * expert_ids,
        int64_t n_experts,
        bool bind_runtime_buffers) {
    ggml_metal_mul_mat_id_decode_replay_key key = {};

    key.src0_type = op->src[0]->type;
    key.src1_type = op->src[1]->type;
    key.ne00 = op->src[0]->ne[0];
    key.ne01 = op->src[0]->ne[1];
    key.ne02 = op->src[0]->ne[2];
    key.ne10 = op->src[1]->ne[0];
    key.ne11 = op->src[1]->ne[1];
    key.ne0 = op->ne[0];
    key.nb00 = op->src[0]->nb[0];
    key.nb01 = op->src[0]->nb[1];
    key.nb02 = op->src[0]->nb[2];
    key.nb10 = op->src[1]->nb[0];
    key.nb11 = op->src[1]->nb[1];
    key.nb_dst1 = op->nb[1];
    key.n_experts = int32_t(n_experts);
    key.use_direct_dispatch = ggml_metal_mul_mat_id_decode_mv_uses_direct_dispatch(op);
    key.bind_runtime_buffers = bind_runtime_buffers;

    if (bind_runtime_buffers) {
        const ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
        const ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
        const ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

        key.src0_metal = uintptr_t(bid_src0.metal);
        key.src1_metal = uintptr_t(bid_src1.metal);
        key.dst_metal  = uintptr_t(bid_dst.metal);
        key.src0_base_offset = bid_src0.offs;
        key.src1_base_offset = bid_src1.offs;
        key.dst_base_offset  = bid_dst.offs;
    }

    for (int32_t i = 0; i < key.n_experts; ++i) {
        key.expert_ids[i] = expert_ids[i];
    }

    return key;
}

static std::shared_ptr<ggml_metal_mul_mat_id_decode_replay_entry> ggml_metal_mul_mat_id_decode_replay_make_entry(
        ggml_metal_device_t dev,
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        const int32_t * expert_ids,
        int64_t n_experts,
        bool use_icb) {
    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

    auto entry = std::make_shared<ggml_metal_mul_mat_id_decode_replay_entry>();
    entry->smem = pipeline.smem;
    entry->tptg1 = pipeline.nsg;
    entry->n_experts = int(n_experts);

    const int nr0 = pipeline.nr0;

    entry->args = {
        /*.ne00 =*/ static_cast<int32_t>(op->src[0]->ne[0]),
        /*.ne01 =*/ static_cast<int32_t>(op->src[0]->ne[1]),
        /*.ne02 =*/ 1,
        /*.nb00 =*/ op->src[0]->nb[0],
        /*.nb01 =*/ op->src[0]->nb[1],
        /*.nb02 =*/ op->src[0]->nb[2],
        /*.nb03 =*/ op->src[0]->nb[2],
        /*.ne10 =*/ static_cast<int32_t>(op->src[1]->ne[0]),
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ op->src[1]->nb[0],
        /*.nb11 =*/ op->src[1]->nb[1],
        /*.nb12 =*/ op->src[1]->nb[2],
        /*.nb13 =*/ op->src[1]->nb[2],
        /*.ne0  =*/ static_cast<int32_t>(op->ne[0]),
        /*.ne1  =*/ 1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
        /*.src0_byte_off =*/ 0,
        /*.src1_byte_off =*/ 0,
        /*.dst_byte_off  =*/ 0,
    };

    if (ggml_metal_mul_mat_id_decode_mv_uses_direct_dispatch(op)) {
        entry->tg0 = (entry->args.ne01 + nr0 - 1) / nr0;
    } else {
        entry->tg0 = (entry->args.ne01 + nr0 * pipeline.nsg - 1) / (nr0 * pipeline.nsg);
    }

    for (int64_t idx_exp = 0; idx_exp < n_experts; ++idx_exp) {
        const int32_t expert_id = expert_ids[idx_exp];
        GGML_ASSERT(expert_id >= 0 && expert_id < op->src[0]->ne[2]);

        entry->src0_offsets[idx_exp] = uint64_t(expert_id) * op->src[0]->nb[2];
        entry->src1_offsets[idx_exp] = uint64_t(idx_exp % op->src[1]->ne[1]) * op->src[1]->nb[1];
        entry->dst_offsets[idx_exp] = uint64_t(idx_exp) * op->nb[1];
    }

    if (use_icb && ggml_metal_device_supports_compute_icb(dev)) {
        std::array<ggml_metal_kargs_mul_mv, GGML_METAL_MUL_MAT_ID_DECODE_REPLAY_MAX_EXPERTS> icb_args = {};
        const ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
        const ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
        const ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

        for (int idx_exp = 0; idx_exp < entry->n_experts; ++idx_exp) {
            icb_args[idx_exp] = entry->args;
            icb_args[idx_exp].src0_byte_off = bid_src0.offs + entry->src0_offsets[idx_exp];
            icb_args[idx_exp].src1_byte_off = bid_src1.offs + entry->src1_offsets[idx_exp];
            icb_args[idx_exp].dst_byte_off  = bid_dst.offs  + entry->dst_offsets[idx_exp];
        }

        entry->args_buffer = ggml_metal_owned_buffer_init(
                dev,
                icb_args.data(),
                size_t(entry->n_experts) * sizeof(icb_args[0]));
        entry->icb = ggml_metal_icb_compute_init(dev, size_t(n_experts), 4);

        if (entry->args_buffer != nullptr && entry->icb != nullptr) {
            const ggml_metal_buffer_id bid_args = ggml_metal_owned_buffer_get_id(entry->args_buffer);
            ggml_metal_buffer_id bid_src0_base = bid_src0;
            ggml_metal_buffer_id bid_src1_base = bid_src1;
            ggml_metal_buffer_id bid_dst_base  = bid_dst;

            bid_src0_base.offs = 0;
            bid_src1_base.offs = 0;
            bid_dst_base.offs  = 0;

            bool icb_ok = true;
            for (int idx_exp = 0; idx_exp < entry->n_experts; ++idx_exp) {
                ggml_metal_buffer_id bid_args_cur = bid_args;
                bid_args_cur.offs += size_t(idx_exp) * sizeof(icb_args[0]);

                icb_ok = ggml_metal_icb_encode_compute_dispatch(
                        entry->icb,
                        size_t(idx_exp),
                        pipeline,
                        bid_args_cur,
                        bid_src0_base,
                        bid_src1_base,
                        bid_dst_base,
                        entry->smem,
                        entry->tg0,
                        1,
                        1,
                        entry->tptg0,
                        entry->tptg1,
                        1);
                if (!icb_ok) {
                    break;
                }
            }

            if (!icb_ok) {
                ggml_metal_icb_free(entry->icb);
                ggml_metal_owned_buffer_free(entry->args_buffer);
                entry->icb = nullptr;
                entry->args_buffer = nullptr;
                g_ggml_metal_mul_mat_id_decode_icb_build_fail_count.fetch_add(1);
            }
        } else {
            ggml_metal_icb_free(entry->icb);
            ggml_metal_owned_buffer_free(entry->args_buffer);
            entry->icb = nullptr;
            entry->args_buffer = nullptr;
            g_ggml_metal_mul_mat_id_decode_icb_build_fail_count.fetch_add(1);
        }
    }

    return entry;
}

static bool ggml_metal_mul_mat_id_decode_replay_lookup(
        ggml_metal_device_t dev,
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        const int32_t * expert_ids,
        int64_t n_experts,
        bool use_icb,
        std::shared_ptr<const ggml_metal_mul_mat_id_decode_replay_entry> & entry) {
    const auto key = ggml_metal_mul_mat_id_decode_replay_make_key(op, expert_ids, n_experts, use_icb);

    std::lock_guard<std::mutex> lock(g_ggml_metal_mul_mat_id_decode_replay_mutex);

    const auto it = g_ggml_metal_mul_mat_id_decode_replay_cache.find(key);
    if (it != g_ggml_metal_mul_mat_id_decode_replay_cache.end()) {
        entry = it->second;
        g_ggml_metal_mul_mat_id_decode_replay_hit_count.fetch_add(1);
        return true;
    }

    g_ggml_metal_mul_mat_id_decode_replay_miss_count.fetch_add(1);
    auto new_entry = ggml_metal_mul_mat_id_decode_replay_make_entry(dev, lib, op, expert_ids, n_experts, use_icb);

    if (g_ggml_metal_mul_mat_id_decode_replay_cache.size() >= ggml_metal_mul_mat_id_experimental_decode_replay_cache_limit()) {
        g_ggml_metal_mul_mat_id_decode_replay_cache.clear();
        g_ggml_metal_mul_mat_id_decode_replay_clear_count.fetch_add(1);
    }

    g_ggml_metal_mul_mat_id_decode_replay_cache.emplace(key, new_entry);
    g_ggml_metal_mul_mat_id_decode_replay_insert_count.fetch_add(1);
    entry = new_entry;

    return false;
}

void ggml_metal_op_mul_mat_id_log_stats(void) {
    const uint64_t decode_mv = g_ggml_metal_mul_mat_id_decode_mv_count.load();
    const uint64_t generic_mv = g_ggml_metal_mul_mat_id_generic_mv_count.load();
    const uint64_t generic_mm = g_ggml_metal_mul_mat_id_generic_mm_count.load();
    const uint64_t fused_glu = g_ggml_metal_mul_mat_id_fused_glu_count.load();
    const uint64_t pair_gate_up = g_ggml_metal_mul_mat_id_pair_gate_up_count.load();
    const uint64_t pair_swiglu = g_ggml_metal_mul_mat_id_pair_swiglu_count.load();
    const uint64_t weighted_swiglu = g_ggml_metal_mul_mat_id_weighted_swiglu_count.load();
    const uint64_t down_sum6 = g_ggml_metal_mul_mat_id_down_sum6_count.load();
    const uint64_t down_sum6_weighted = g_ggml_metal_mul_mat_id_down_sum6_weighted_count.load();
    const uint64_t shared_swiglu = g_ggml_metal_mul_mat_shared_swiglu_count.load();
    const uint64_t replay_hit = g_ggml_metal_mul_mat_id_decode_replay_hit_count.load();
    const uint64_t replay_miss = g_ggml_metal_mul_mat_id_decode_replay_miss_count.load();
    const uint64_t replay_insert = g_ggml_metal_mul_mat_id_decode_replay_insert_count.load();
    const uint64_t replay_clear = g_ggml_metal_mul_mat_id_decode_replay_clear_count.load();
    const uint64_t icb_exec = g_ggml_metal_mul_mat_id_decode_icb_exec_count.load();
    const uint64_t icb_build_fail = g_ggml_metal_mul_mat_id_decode_icb_build_fail_count.load();
    const uint64_t dsv4_rope_hfp4 = g_ggml_metal_dsv4_rope_hadamard_fp4_count.load();
    const uint64_t dsv4_rope_fp8 = g_ggml_metal_dsv4_rope_fp8_kv_count.load();
    const uint64_t dsv4_kvset = g_ggml_metal_dsv4_rope_fp8_kv_set_rows_count.load();
    const uint64_t dsv4_hcws = g_ggml_metal_dsv4_hc_split_weighted_sum_count.load();
    const uint64_t dsv4_hcnorm = g_ggml_metal_dsv4_hc_pre_norm_count.load();
    const uint64_t dsv4_hce4 = g_ggml_metal_dsv4_hc_expand4_count.load();
    const uint64_t dsv4_iscore = g_ggml_metal_dsv4_indexer_weighted_score_count.load();
    const uint64_t dsv4_dcomp = g_ggml_metal_dsv4_decode_compress_count.load();
    const uint64_t dsv4_iattn = g_ggml_metal_dsv4_mixed_attn_count.load();
    const uint64_t dsv4_aolow = g_ggml_metal_dsv4_attn_out_low_count.load();
    const uint64_t dsv4_aodec = g_ggml_metal_dsv4_attn_out_decode_count.load();
    const uint64_t dsv4_cupd = g_ggml_metal_dsv4_compressor_update_count.load();
    const uint64_t dsv4_cupd2 = g_ggml_metal_dsv4_compressor_update_v2_count.load();
    const uint64_t dsv4_kvfin = g_ggml_metal_dsv4_kv_finalize_count.load();
    const uint64_t dsv4_ffnmoe = g_ggml_metal_dsv4_ffn_moe_stage_count.load();
    const uint64_t dsv4_rmoe = g_ggml_metal_dsv4_routed_moe_one_tensor_count.load();
    const uint64_t dsv4_lexec_dryrun = g_ggml_metal_dsv4_decode_layer_executor_dryrun_count.load();
    const uint64_t dsv4_lexec = g_ggml_metal_dsv4_decode_layer_count.load();
    const uint64_t metal_dispatch = ggml_metal_encoder_get_dispatch_count();
    size_t replay_cache_size = 0;

    {
        std::lock_guard<std::mutex> lock(g_ggml_metal_mul_mat_id_decode_replay_mutex);
        replay_cache_size = g_ggml_metal_mul_mat_id_decode_replay_cache.size();
    }

    if (decode_mv == 0 && generic_mv == 0 && generic_mm == 0 && fused_glu == 0 && pair_gate_up == 0 && pair_swiglu == 0 && weighted_swiglu == 0 && down_sum6 == 0 && down_sum6_weighted == 0 && shared_swiglu == 0 &&
        replay_hit == 0 && replay_miss == 0 && icb_exec == 0 && icb_build_fail == 0 && dsv4_rope_hfp4 == 0 && dsv4_rope_fp8 == 0 && dsv4_kvset == 0 && dsv4_hcws == 0 && dsv4_hcnorm == 0 && dsv4_hce4 == 0 && dsv4_iscore == 0 && dsv4_dcomp == 0 && dsv4_iattn == 0 && dsv4_aolow == 0 && dsv4_aodec == 0 && dsv4_cupd == 0 && dsv4_cupd2 == 0 && dsv4_kvfin == 0 && dsv4_ffnmoe == 0 && dsv4_rmoe == 0 && dsv4_lexec_dryrun == 0 && dsv4_lexec == 0 && metal_dispatch == 0) {
        return;
    }

    GGML_LOG_INFO("%s: mul_mat_id dec_mv=%" PRIu64 " pair=%" PRIu64 " pswiglu=%" PRIu64 " wpswiglu=%" PRIu64 " dsum6=%" PRIu64 " wdsum6=%" PRIu64 " shswiglu=%" PRIu64 " gen_mv=%" PRIu64 " gen_mm=%" PRIu64
            " metal_dispatch=%" PRIu64
            " fglu=%" PRIu64 " replay_hit=%" PRIu64 " replay_miss=%" PRIu64
            " replay_ins=%" PRIu64 " replay_clr=%" PRIu64
            " icb_exec=%" PRIu64 " icb_fail=%" PRIu64
            " dsv4_rope_hfp4=%" PRIu64 " dsv4_rope_fp8=%" PRIu64
            " dsv4_kvset=%" PRIu64
            " dsv4_hcws=%" PRIu64
            " dsv4_hcnorm=%" PRIu64
            " dsv4_hce4=%" PRIu64
            " dsv4_iscore=%" PRIu64
            " dsv4_dcomp=%" PRIu64
            " dsv4_iattn=%" PRIu64
            " dsv4_aolow=%" PRIu64
            " dsv4_aodec=%" PRIu64
            " dsv4_cupd=%" PRIu64
            " dsv4_cupd2=%" PRIu64
            " dsv4_kvfin=%" PRIu64
            " dsv4_ffnmoe=%" PRIu64
            " dsv4_rmoe=%" PRIu64
            " dsv4_lexec_dryrun=%" PRIu64
            " dsv4_lexec=%" PRIu64
            " replay_cache=%zu\n",
            __func__, decode_mv, pair_gate_up, pair_swiglu, weighted_swiglu, down_sum6, down_sum6_weighted, shared_swiglu, generic_mv, generic_mm, metal_dispatch,
            fused_glu, replay_hit, replay_miss, replay_insert, replay_clear, icb_exec, icb_build_fail, dsv4_rope_hfp4, dsv4_rope_fp8, dsv4_kvset, dsv4_hcws, dsv4_hcnorm, dsv4_hce4, dsv4_iscore, dsv4_dcomp, dsv4_iattn, dsv4_aolow, dsv4_aodec, dsv4_cupd, dsv4_cupd2, dsv4_kvfin, dsv4_ffnmoe, dsv4_rmoe, dsv4_lexec_dryrun, dsv4_lexec, replay_cache_size);
    ggml_metal_dispatch_profile_log();
}

void ggml_metal_op_mul_mat_log_stats(void) {
    const uint64_t mv_ext = g_ggml_metal_mul_mat_mv_ext_count.load();
    const uint64_t mv = g_ggml_metal_mul_mat_mv_count.load();
    const uint64_t mm = g_ggml_metal_mul_mat_mm_count.load();
    const uint64_t mm_m5_expert = g_ggml_metal_mul_mat_mm_m5_expert_count.load();
    const uint64_t mm_m5sg = g_ggml_metal_mul_mat_mm_m5_sgmatrix_count.load();
    const uint64_t shared_swiglu = g_ggml_metal_mul_mat_shared_swiglu_count.load();
    const uint64_t dsv4_cpair = g_ggml_metal_dsv4_compressor_pair_count.load();
    const uint64_t dsv4_q8hc = g_ggml_metal_dsv4_q8_hc_expand_count.load();

    if (mv_ext == 0 && mv == 0 && mm == 0 && mm_m5_expert == 0 && mm_m5sg == 0 && shared_swiglu == 0 && dsv4_cpair == 0 && dsv4_q8hc == 0) {
        return;
    }

    GGML_LOG_INFO("%s: mul_mat mv_ext=%" PRIu64 " mv=%" PRIu64 " mm=%" PRIu64 " mm_m5_expert=%" PRIu64 " mm_m5sg=%" PRIu64 " shswiglu=%" PRIu64 " dsv4_cpair=%" PRIu64 " dsv4_q8hc=%" PRIu64 "\n",
            __func__, mv_ext, mv, mm, mm_m5_expert, mm_m5sg, shared_swiglu, dsv4_cpair, dsv4_q8hc);

    const uint64_t aohc_candidate = g_ggml_metal_dsv4_aohc_elig_candidate_count.load();
    const uint64_t aohc_q8hc_eligible = g_ggml_metal_dsv4_aohc_elig_q8hc_eligible_count.load();
    const uint64_t aohc_q8hc_rejected = g_ggml_metal_dsv4_aohc_elig_q8hc_rejected_count.load();
    if (aohc_candidate != 0 || ggml_metal_dsv4_trace_aohc_fused_elig_enabled() || ggml_metal_dsv4_trace_q8hc_elig_enabled()) {
        std::string first_reason;
        std::string first_name;
        int first_idx = -1;
        {
            std::lock_guard<std::mutex> lock(g_ggml_metal_dsv4_aohc_elig_mutex);
            first_reason = g_ggml_metal_dsv4_aohc_elig_first_reject_reason.empty() ?
                "none" : g_ggml_metal_dsv4_aohc_elig_first_reject_reason;
            first_name = g_ggml_metal_dsv4_aohc_elig_first_reject_name.empty() ?
                "none" : g_ggml_metal_dsv4_aohc_elig_first_reject_name;
            first_idx = g_ggml_metal_dsv4_aohc_elig_first_reject_idx;
        }
        GGML_LOG_INFO("dsv4_aohc_elig_summary: candidate_cases=%" PRIu64
                " q8hc_eligible=%" PRIu64
                " q8hc_rejected=%" PRIu64
                " q8hc_dispatched=%" PRIu64
                " aohc_backend_fused_eligible=%" PRIu64
                " aohc_backend_fused_dispatched=%" PRIu64
                " dsv4_q8hc=%" PRIu64
                " first_reject_idx=%d first_reject_name=%s first_reject_reason=%s\n",
                aohc_candidate,
                aohc_q8hc_eligible,
                aohc_q8hc_rejected,
                dsv4_q8hc,
                aohc_q8hc_eligible,
                dsv4_q8hc,
                dsv4_q8hc,
                first_idx,
                first_name.c_str(),
                first_reason.c_str());
    }
}

void ggml_metal_op_mul_mat_set_experimental_m5_expert_active(bool active) {
    if (active) {
        ++g_ggml_metal_mul_mat_m5_expert_scope_depth;
    } else if (g_ggml_metal_mul_mat_m5_expert_scope_depth > 0) {
        --g_ggml_metal_mul_mat_m5_expert_scope_depth;
    }
}

void ggml_metal_op_mul_mat_id_get_stats(struct ggml_metal_mul_mat_id_stats * stats) {
    if (stats == nullptr) {
        return;
    }

    stats->replay_hit = g_ggml_metal_mul_mat_id_decode_replay_hit_count.load();
    stats->replay_miss = g_ggml_metal_mul_mat_id_decode_replay_miss_count.load();
    stats->replay_insert = g_ggml_metal_mul_mat_id_decode_replay_insert_count.load();
    stats->replay_clear = g_ggml_metal_mul_mat_id_decode_replay_clear_count.load();
    stats->icb_exec = g_ggml_metal_mul_mat_id_decode_icb_exec_count.load();
    stats->icb_build_fail = g_ggml_metal_mul_mat_id_decode_icb_build_fail_count.load();

    {
        std::lock_guard<std::mutex> lock(g_ggml_metal_mul_mat_id_decode_replay_mutex);
        stats->replay_cache_size = g_ggml_metal_mul_mat_id_decode_replay_cache.size();
    }
}

void ggml_metal_op_mul_mat_id_reset_stats(void) {
    g_ggml_metal_mul_mat_id_decode_mv_count.store(0);
    g_ggml_metal_mul_mat_id_generic_mv_count.store(0);
    g_ggml_metal_mul_mat_id_generic_mm_count.store(0);
    g_ggml_metal_mul_mat_id_fused_glu_count.store(0);
    g_ggml_metal_mul_mat_id_pair_gate_up_count.store(0);
    g_ggml_metal_mul_mat_id_pair_swiglu_count.store(0);
    g_ggml_metal_mul_mat_id_weighted_swiglu_count.store(0);
    g_ggml_metal_mul_mat_id_down_sum6_count.store(0);
    g_ggml_metal_mul_mat_id_down_sum6_weighted_count.store(0);
    g_ggml_metal_mul_mat_shared_swiglu_count.store(0);
    g_ggml_metal_mul_mat_id_split_trace_count.store(0);
    g_ggml_metal_mul_mat_id_decode_replay_hit_count.store(0);
    g_ggml_metal_mul_mat_id_decode_replay_miss_count.store(0);
    g_ggml_metal_mul_mat_id_decode_replay_insert_count.store(0);
    g_ggml_metal_mul_mat_id_decode_replay_clear_count.store(0);
    g_ggml_metal_mul_mat_id_decode_icb_exec_count.store(0);
    g_ggml_metal_mul_mat_id_decode_icb_build_fail_count.store(0);
    g_ggml_metal_dsv4_rope_hadamard_fp4_count.store(0);
    g_ggml_metal_dsv4_rope_fp8_kv_count.store(0);
    g_ggml_metal_dsv4_rope_fp8_kv_set_rows_count.store(0);
    g_ggml_metal_dsv4_hc_split_weighted_sum_count.store(0);
    g_ggml_metal_dsv4_hc_pre_norm_count.store(0);
    g_ggml_metal_dsv4_hc_pre_norm_trace_count.store(0);
    g_ggml_metal_dsv4_hc_expand4_count.store(0);
    g_ggml_metal_dsv4_indexer_weighted_score_count.store(0);
    g_ggml_metal_dsv4_decode_compress_count.store(0);
    g_ggml_metal_dsv4_mixed_attn_count.store(0);
    g_ggml_metal_dsv4_attn_out_low_count.store(0);
    g_ggml_metal_dsv4_attn_out_decode_count.store(0);
    g_ggml_metal_dsv4_compressor_update_count.store(0);
    g_ggml_metal_dsv4_compressor_update_trace_count.store(0);
    g_ggml_metal_dsv4_compressor_update_v2_count.store(0);
    g_ggml_metal_dsv4_compressor_update_v2_trace_count.store(0);
    g_ggml_metal_dsv4_kv_finalize_count.store(0);
    g_ggml_metal_dsv4_kv_finalize_trace_count.store(0);
    g_ggml_metal_dsv4_ffn_moe_stage_count.store(0);
    g_ggml_metal_dsv4_ffn_moe_stage_trace_count.store(0);
    g_ggml_metal_dsv4_routed_moe_one_tensor_count.store(0);
    g_ggml_metal_dsv4_routed_moe_one_tensor_trace_count.store(0);
    g_ggml_metal_dsv4_decode_layer_executor_dryrun_count.store(0);
    g_ggml_metal_dsv4_decode_layer_executor_dryrun_trace_count.store(0);
    g_ggml_metal_dsv4_decode_layer_count.store(0);
    g_ggml_metal_dsv4_decode_layer_trace_count.store(0);
    ggml_metal_encoder_reset_dispatch_count();
    ggml_metal_dispatch_profile_reset();

    std::lock_guard<std::mutex> lock(g_ggml_metal_mul_mat_id_decode_replay_mutex);
    g_ggml_metal_mul_mat_id_decode_replay_cache.clear();
}

void ggml_metal_op_mul_mat_reset_stats(void) {
    g_ggml_metal_mul_mat_mv_ext_count.store(0);
    g_ggml_metal_mul_mat_mv_count.store(0);
    g_ggml_metal_mul_mat_mm_count.store(0);
    g_ggml_metal_mul_mat_mm_m5_expert_count.store(0);
    g_ggml_metal_mul_mat_mm_m5_sgmatrix_count.store(0);
    g_ggml_metal_mul_mat_shared_swiglu_count.store(0);
    g_ggml_metal_dsv4_compressor_pair_count.store(0);
    g_ggml_metal_dsv4_q8_hc_expand_count.store(0);
    g_ggml_metal_dsv4_q8_hc_expand_trace_count.store(0);
    g_ggml_metal_dsv4_aohc_elig_candidate_count.store(0);
    g_ggml_metal_dsv4_aohc_elig_q8hc_eligible_count.store(0);
    g_ggml_metal_dsv4_aohc_elig_q8hc_rejected_count.store(0);
    {
        std::lock_guard<std::mutex> lock(g_ggml_metal_dsv4_aohc_elig_mutex);
        g_ggml_metal_dsv4_aohc_elig_first_reject_reason.clear();
        g_ggml_metal_dsv4_aohc_elig_first_reject_name.clear();
        g_ggml_metal_dsv4_aohc_elig_first_reject_idx = -1;
    }
}

struct ggml_metal_op {
    ggml_metal_op(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        int  idx_start,
        int  idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph,
        int  debug_fusion,
        int  trace_token,
        int  trace_command_buffer) {
        this->dev             = dev;
        this->lib             = ggml_metal_device_get_library(dev);
        this->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);
        this->mem_ranges      = ggml_mem_ranges_init(debug_graph);
        this->idx_start       = idx_start;
        this->idx_end         = idx_end;
        this->use_fusion      = use_fusion;
        this->use_concurrency = use_concurrency;
        this->use_capture     = use_capture;
        this->debug_graph     = debug_graph;
        this->debug_fusion    = debug_fusion;
        this->trace_token     = trace_token;
        this->trace_command_buffer = trace_command_buffer;
        this->gf              = gf;

        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        for (int i = idx_start; i < idx_end; i++) {
            if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }

        skip_local_idxs.resize(idxs.size(), 0);
        deferred_shared_swiglu_local_idxs.resize(idxs.size(), 0);
        deferred_weighted_swiglu_local_idxs.resize(idxs.size(), 0);
    }

    ~ggml_metal_op() {
        ggml_metal_encoder_end_encoding(this->enc);
        ggml_metal_encoder_free(this->enc);
        ggml_mem_ranges_free(this->mem_ranges);
    }

    int n_nodes() const {
        return idxs.size();
    }

    ggml_tensor * node(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return ggml_graph_node(gf, idxs[i]);
    }

    bool can_fuse(int i0, const ggml_op * ops, int n_ops) const {
        assert(use_fusion);
        assert(i0 >= 0 && i0 < n_nodes());

        if (i0 + n_ops > n_nodes()) {
            return false;
        }

        return ggml_can_fuse_ext(gf, idxs.data() + i0, ops, n_ops);
    }

    bool can_fuse_subgraph(int i0, const ggml_op * ops, int n_ops, const int * outputs, int n_outputs) const {
        assert(use_fusion);
        assert(i0 >= 0 && i0 < n_nodes());

        if (i0 + n_ops > n_nodes()) {
            return false;
        }

        int node_idxs[32];
        int output_idxs[32];

        assert(n_ops <= 32);
        assert(n_outputs <= 32);

        for (int i = 0; i < n_ops; ++i) {
            node_idxs[i] = idxs[i0 + i];
        }

        for (int i = 0; i < n_outputs; ++i) {
            const int rel_idx = outputs[i];
            if (rel_idx < 0 || rel_idx >= n_ops) {
                return false;
            }
            output_idxs[i] = node_idxs[rel_idx];
        }

        return ggml_can_fuse_subgraph_ext(gf, node_idxs, n_ops, ops, output_idxs, n_outputs);
    }

    bool can_fuse_subgraph_sparse(const int * local_idxs, int n_ops, const ggml_op * ops, const int * local_outputs, int n_outputs) const {
        assert(use_fusion);
        assert(n_ops <= 32);
        assert(n_outputs <= 32);

        int node_idxs[32];
        int output_idxs[32];

        for (int i = 0; i < n_ops; ++i) {
            if (local_idxs[i] < 0 || local_idxs[i] >= n_nodes()) {
                return false;
            }
            node_idxs[i] = idxs[local_idxs[i]];
        }

        for (int i = 0; i < n_outputs; ++i) {
            if (local_outputs[i] < 0 || local_outputs[i] >= n_nodes()) {
                return false;
            }
            output_idxs[i] = idxs[local_outputs[i]];
        }

        return ggml_can_fuse_subgraph_ext(gf, node_idxs, n_ops, ops, output_idxs, n_outputs);
    }

    bool is_skipped(int i) const {
        assert(i >= 0 && i < n_nodes());
        return skip_local_idxs[i] != 0;
    }

    void skip_node(int i) {
        assert(i >= 0 && i < n_nodes());
        skip_local_idxs[i] = 1;
    }

    bool is_deferred_shared_swiglu(int i) const {
        assert(i >= 0 && i < n_nodes());
        return deferred_shared_swiglu_local_idxs[i] != 0;
    }

    void defer_shared_swiglu(int i) {
        assert(i >= 0 && i < n_nodes());
        deferred_shared_swiglu_local_idxs[i] = 1;
    }

    bool is_deferred_weighted_swiglu(int i) const {
        assert(i >= 0 && i < n_nodes());
        return deferred_weighted_swiglu_local_idxs[i] != 0;
    }

    void defer_weighted_swiglu(int i) {
        assert(i >= 0 && i < n_nodes());
        deferred_weighted_swiglu_local_idxs[i] = 1;
    }

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    bool use_fusion;
    bool use_concurrency;
    bool use_capture;

    int debug_graph;
    int debug_fusion;
    int trace_token;
    int trace_command_buffer;

private:
    ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    // non-empty node indices
    std::vector<int> idxs;
    std::vector<uint8_t> skip_local_idxs;
    std::vector<uint8_t> deferred_shared_swiglu_local_idxs;
    std::vector<uint8_t> deferred_weighted_swiglu_local_idxs;
};

static bool ggml_metal_mul_mat_id_should_trace_split_glu(ggml_metal_op_t ctx, const ggml_tensor * op) {
    if (ctx->debug_fusion <= 0 || op == nullptr) {
        return false;
    }

    const uint64_t slot = g_ggml_metal_mul_mat_id_split_trace_count.fetch_add(1);
    return slot < 12;
}

static void ggml_metal_mul_mat_id_trace_split_glu_window(
        ggml_metal_op_t ctx,
        int idx,
        const ggml_tensor * op,
        const char * why) {
    if (!ggml_metal_mul_mat_id_should_trace_split_glu(ctx, op)) {
        return;
    }

    fprintf(stderr, "%s: split-glu probe reason=%s idx=%d op=%s/%s\n",
            __func__, why ? why : "?", idx, ggml_op_name(op->op), op->name);

    const int i1 = std::min(ctx->n_nodes(), idx + 18);
    for (int i = idx; i < i1; ++i) {
        ggml_tensor * node = ctx->node(i);
        fprintf(stderr, "%s:   node[%d]=%s/%s src0=%s src1=%s src2=%s\n",
                __func__,
                i,
                ggml_op_name(node->op),
                node->name,
                node->src[0] ? node->src[0]->name : "-",
                node->src[1] ? node->src[1]->name : "-",
                node->src[2] ? node->src[2]->name : "-");
    }
}

static bool ggml_metal_dsv4_rmoe_pair_preserve_match_trace_op(const ggml_tensor * op) {
    return ggml_metal_dsv4_rmoe_pair_preserve_match_trace_enabled() &&
           op != nullptr &&
           std::strstr(op->name, "ffn_moe_gate-0") != nullptr;
}

static void ggml_metal_dsv4_rmoe_pair_preserve_match_trace(
        ggml_metal_op_t ctx,
        int idx,
        const ggml_tensor * op,
        const char * decision,
        const char * reason,
        const ggml_tensor * peer,
        const ggml_tensor * swiglu,
        const ggml_tensor * weighted,
        const ggml_tensor * consumer) {
    if (!ggml_metal_dsv4_rmoe_pair_preserve_match_trace_op(op)) {
        return;
    }
    fprintf(stderr,
            "dsv4_rmoe_pair_preserve_match: layer=0 idx=%d gate=%s up=%s swiglu=%s weighted_swiglu=%s"
            " down_consumer=%s backend_pair_preserve_op=%s pair_decision=%s pswiglu_decision=%s fglu_decision=%s reject_reason=%s\n",
            idx,
            op ? op->name : "missing",
            peer ? peer->name : "missing",
            swiglu ? swiglu->name : "missing",
            weighted ? weighted->name : "missing",
            consumer ? consumer->name : "missing",
            consumer && consumer->op == GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE ? consumer->name : "missing",
            decision ? decision : "unknown",
            decision ? decision : "unknown",
            decision ? decision : "unknown",
            reason ? reason : "none");
    if (ctx != nullptr && ctx->debug_fusion > 0) {
        ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, reason ? reason : "pair-preserve-match");
    }
}

static bool ggml_metal_buffer_ranges_overlap(
        ggml_metal_buffer_id a,
        size_t a_size,
        ggml_metal_buffer_id b,
        size_t b_size) {
    if (a.metal != b.metal) {
        return false;
    }

    const size_t a0 = a.offs;
    const size_t a1 = a0 + a_size;
    const size_t b0 = b.offs;
    const size_t b1 = b0 + b_size;

    return a0 < b1 && b0 < a1;
}

static bool ggml_metal_tensor_overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    if (a == nullptr || b == nullptr) {
        return false;
    }

    return ggml_metal_buffer_ranges_overlap(
            ggml_metal_get_buffer_id(a), ggml_nbytes(a),
            ggml_metal_get_buffer_id(b), ggml_nbytes(b));
}

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        int idx_start,
        int idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int debug_graph,
        int debug_fusion,
        int trace_token,
        int trace_command_buffer) {
    ggml_metal_op_t res = new ggml_metal_op(
        dev,
        cmd_buf,
        gf,
        idx_start,
        idx_end,
        use_fusion,
        use_concurrency,
        use_capture,
        debug_graph,
        debug_fusion,
        trace_token,
        trace_command_buffer);

    return res;
}

void ggml_metal_op_free(ggml_metal_op_t ctx) {
    delete ctx;
}

int ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    return ctx->n_nodes();
}

static bool ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx) {
    if (!ctx->mem_ranges) {
        return true;
    }

    ggml_metal_encoder_memory_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static bool ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return false;
    }

    return ggml_mem_ranges_check(ctx->mem_ranges, node);
}

static bool ggml_metal_op_concurrency_add(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return true;
    }

    return ggml_mem_ranges_add(ctx->mem_ranges, node);
}

struct ggml_metal_mul_mat_id_glu_fuse_plan {
    ggml_tensor * glu   = nullptr;
    ggml_tensor * scale = nullptr;
    int n_fuse          = 0;
};

struct ggml_metal_mul_mat_id_split_glu_fuse_plan {
    ggml_tensor * peer      = nullptr;
    ggml_tensor * glu       = nullptr;
    int peer_local_idx      = -1;
    int glu_local_idx       = -1;
    int n_fuse_contiguous   = 0;
    bool limited_swiglu     = false;
    ggml_tensor * gate_clamp = nullptr;
    ggml_tensor * up_clamp   = nullptr;
    ggml_tensor * silu       = nullptr;
    ggml_tensor * limited    = nullptr;
    ggml_tensor * weighted   = nullptr;
    ggml_tensor * weights    = nullptr;
    int gate_clamp_local_idx = -1;
    int up_clamp_local_idx   = -1;
    int silu_local_idx       = -1;
    int limited_local_idx    = -1;
    int weighted_local_idx   = -1;
    bool weighted_swiglu     = false;
};

struct ggml_metal_mul_mat_id_pair_gate_up_plan {
    ggml_tensor * peer    = nullptr;
    int peer_local_idx    = -1;
};

struct ggml_metal_mul_mat_id_down_sum6_plan {
    ggml_tensor * dst = nullptr;
    ggml_tensor * weighted_src = nullptr;
    ggml_tensor * weighted_act = nullptr;
    ggml_tensor * weights = nullptr;
    int dst_local_idx = -1;
    int n_fuse_contiguous = 0;
    int add_local_idxs[5] = { -1, -1, -1, -1, -1 };
    bool weighted = false;
};

struct ggml_metal_mul_mat_shared_swiglu_plan {
    ggml_tensor * peer = nullptr;
    ggml_tensor * glu  = nullptr;
    int peer_local_idx = -1;
    int glu_local_idx  = -1;
    int n_fuse_contiguous = 0;
    bool defer_to_glu = false;
};

static bool ggml_metal_tensor_name_has_token(const ggml_tensor * t, const char * token) {
    return t != nullptr && token != nullptr && t->name[0] != '\0' && strstr(t->name, token) != nullptr;
}

static bool ggml_metal_tensor_is_view_of(const ggml_tensor * view, const ggml_tensor * src) {
    return view != nullptr && src != nullptr &&
           view->op == GGML_OP_VIEW &&
           view->src[0] == src &&
           view->ne[0] == src->ne[0] &&
           view->ne[1] == 1 &&
           view->ne[2] == 1 &&
           view->ne[3] == 1;
}

static bool ggml_metal_tensor_is_split_gate(const ggml_tensor * t) {
    return ggml_metal_tensor_name_has_token(t, "ffn_moe_gate") && !ggml_metal_tensor_name_has_token(t, "gate_up");
}

static bool ggml_metal_tensor_is_split_up(const ggml_tensor * t) {
    return ggml_metal_tensor_name_has_token(t, "ffn_moe_up") && !ggml_metal_tensor_name_has_token(t, "gate_up");
}

static bool ggml_metal_tensor_is_shared_gate(const ggml_tensor * t) {
    return ggml_metal_tensor_name_has_token(t, "ffn_gate") && !ggml_metal_tensor_name_has_token(t, "ffn_moe_gate");
}

static bool ggml_metal_tensor_is_shared_up(const ggml_tensor * t) {
    return ggml_metal_tensor_name_has_token(t, "ffn_up") && !ggml_metal_tensor_name_has_token(t, "ffn_moe_up");
}

static bool ggml_metal_mul_mat_is_flashmoe_prefill_expert(const ggml_tensor * t) {
    return ggml_metal_tensor_name_has_token(t, "flashmoe_prefill_expert_mm");
}

static bool ggml_metal_mul_mat_use_m5_expert_pipeline(
        const ggml_metal_device_props * props_dev,
        const ggml_tensor * op) {
    if (props_dev == nullptr || op == nullptr || !props_dev->has_tensor) {
        return false;
    }

    if (!ggml_metal_device_name_contains_token(props_dev, "M5")) {
        return false;
    }

    return g_ggml_metal_mul_mat_m5_expert_scope_depth > 0 ||
           ggml_metal_mul_mat_is_flashmoe_prefill_expert(op) ||
           ggml_metal_mul_mat_is_flashmoe_prefill_expert(op->src[0]) ||
           ggml_metal_mul_mat_is_flashmoe_prefill_expert(op->src[1]);
}

static bool ggml_metal_mul_mat_use_m5_sgmatrix_pipeline(
        const ggml_metal_device_props * props_dev) {
    if (props_dev == nullptr || !props_dev->has_simdgroup_mm || props_dev->has_tensor) {
        return false;
    }

    if (!ggml_metal_device_name_contains_token(props_dev, "M5")) {
        return false;
    }

    return ggml_metal_mul_mat_m5_sgmatrix_enabled();
}

static bool ggml_metal_mul_mat_id_get_pair_gate_up_plan(
        ggml_metal_op_t ctx,
        int idx,
        ggml_tensor * op,
        ggml_metal_mul_mat_id_pair_gate_up_plan * plan) {
    if (plan == nullptr) {
        return false;
    }

    *plan = {};

    if (!ggml_metal_mul_mat_id_experimental_pair_gate_up_enabled() &&
            !ggml_metal_mul_mat_id_experimental_dsv4_limited_swiglu_enabled()) {
        return false;
    }

    if (idx + 1 >= ctx->n_nodes()) {
        return false;
    }

    ggml_tensor * peer = ctx->node(idx + 1);
    if (peer->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    const bool is_gate_up_pair =
            (ggml_metal_tensor_is_split_gate(op) && ggml_metal_tensor_is_split_up(peer)) ||
            (ggml_metal_tensor_is_split_up(op) && ggml_metal_tensor_is_split_gate(peer));
    if (!is_gate_up_pair) {
        return false;
    }

    if (op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr ||
        peer->src[0] == nullptr || peer->src[1] == nullptr || peer->src[2] == nullptr) {
        return false;
    }

    if (op->src[1] != peer->src[1] || op->src[2] != peer->src[2]) {
        return false;
    }

    if (op->src[0]->type != peer->src[0]->type ||
        !ggml_are_same_shape(op->src[0], peer->src[0]) ||
        !ggml_are_same_stride(op->src[0], peer->src[0]) ||
        !ggml_are_same_shape(op, peer) ||
        !ggml_are_same_stride(op, peer)) {
        return false;
    }

    plan->peer = peer;
    plan->peer_local_idx = idx + 1;

    return true;
}

static bool ggml_metal_mul_mat_get_shared_swiglu_plan(
        ggml_metal_op_t ctx,
        int idx,
        ggml_tensor * op,
        ggml_metal_mul_mat_shared_swiglu_plan * plan) {
    if (ctx == nullptr || op == nullptr || plan == nullptr) {
        return false;
    }

    *plan = {};

    if (!ctx->use_fusion || !ggml_metal_mul_mat_id_experimental_dsv4_shared_swiglu_enabled()) {
        return false;
    }

    if (op->op != GGML_OP_MUL_MAT || !ggml_metal_tensor_is_shared_gate(op) ||
            idx + 2 >= ctx->n_nodes()) {
        return false;
    }

    const int peer_local_idx = idx + 1;
    ggml_tensor * peer = ctx->node(peer_local_idx);
    if (peer == nullptr || peer->op != GGML_OP_MUL_MAT || !ggml_metal_tensor_is_shared_up(peer)) {
        if (ctx->debug_fusion > 0) {
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "shared-peer-not-up");
        }
        return false;
    }

    if (op->src[0] == nullptr || op->src[1] == nullptr ||
            peer->src[0] == nullptr || peer->src[1] == nullptr ||
            op->src[1] != peer->src[1] ||
            op->src[0]->type != GGML_TYPE_Q8_0 || peer->src[0]->type != GGML_TYPE_Q8_0 ||
            op->src[1]->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(op->src[0], peer->src[0]) ||
            !ggml_are_same_stride(op->src[0], peer->src[0]) ||
            !ggml_are_same_shape(op, peer) ||
            !ggml_are_same_stride(op, peer)) {
        if (ctx->debug_fusion > 0) {
            GGML_LOG_INFO("%s: shared-shape op=%s src0=%s peer_src0=%s src1=%s op_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] peer_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] src0_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] peer_src0_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                    __func__, op->name,
                    op->src[0] ? ggml_type_name(op->src[0]->type) : "-",
                    peer->src[0] ? ggml_type_name(peer->src[0]->type) : "-",
                    op->src[1] ? ggml_type_name(op->src[1]->type) : "-",
                    op->ne[0], op->ne[1], op->ne[2], op->ne[3],
                    peer->ne[0], peer->ne[1], peer->ne[2], peer->ne[3],
                    op->src[0] ? op->src[0]->ne[0] : 0, op->src[0] ? op->src[0]->ne[1] : 0,
                    op->src[0] ? op->src[0]->ne[2] : 0, op->src[0] ? op->src[0]->ne[3] : 0,
                    peer->src[0] ? peer->src[0]->ne[0] : 0, peer->src[0] ? peer->src[0]->ne[1] : 0,
                    peer->src[0] ? peer->src[0]->ne[2] : 0, peer->src[0] ? peer->src[0]->ne[3] : 0);
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "shared-shape");
        }
        return false;
    }

    int glu_local_idx = -1;
    ggml_tensor * glu = nullptr;
    for (int cand = idx + 2; cand < std::min(ctx->n_nodes(), idx + 16); ++cand) {
        ggml_tensor * cand_node = ctx->node(cand);
        if (cand_node == nullptr || cand_node->op != GGML_OP_GLU ||
                ggml_get_glu_op(cand_node) != GGML_GLU_OP_SWIGLU) {
            continue;
        }
        if (cand_node->src[0] == op && cand_node->src[1] == peer &&
                cand_node->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(cand_node, op) &&
                ggml_are_same_stride(cand_node, op)) {
            glu_local_idx = cand;
            glu = cand_node;
            break;
        }
    }

    if (glu_local_idx < 0 || glu == nullptr) {
        if (ctx->debug_fusion > 0) {
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "shared-glu-not-found");
        }
        return false;
    }

    static const ggml_op ops_shared[] = {
        GGML_OP_MUL_MAT,
        GGML_OP_MUL_MAT,
        GGML_OP_GLU,
    };
    const int local_idxs[] = { idx, peer_local_idx, glu_local_idx };
    const int output_idxs[] = { glu_local_idx };
    if (!ctx->can_fuse_subgraph_sparse(local_idxs, 3, ops_shared, output_idxs, 1)) {
        if (ctx->debug_fusion > 0) {
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "shared-subgraph");
        }
        return false;
    }

    for (int mid = peer_local_idx + 1; mid < glu_local_idx; ++mid) {
        ggml_tensor * mid_node = ctx->node(mid);
        if (ggml_is_empty(mid_node) || (mid_node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (ggml_metal_tensor_overlaps(glu, mid_node) ||
            ggml_metal_tensor_overlaps(glu, mid_node->src[0]) ||
            ggml_metal_tensor_overlaps(glu, mid_node->src[1]) ||
            ggml_metal_tensor_overlaps(glu, mid_node->src[2]) ||
            ggml_metal_tensor_overlaps(glu, mid_node->src[3])) {
            if (ctx->debug_fusion > 0) {
                GGML_LOG_INFO("%s: shared-swiglu overlap idx=%d op=%s mid=%s/%s glu=%s\n",
                        __func__, idx, op->name, ggml_op_name(mid_node->op), mid_node->name, glu->name);
                ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "shared-overlap");
            }
            plan->defer_to_glu = true;
            break;
        }
    }

    plan->peer = peer;
    plan->glu = glu;
    plan->peer_local_idx = peer_local_idx;
    plan->glu_local_idx = glu_local_idx;
    plan->n_fuse_contiguous = 2;

    return true;
}

static bool ggml_metal_mul_mat_id_get_weighted_swiglu_sources(
        ggml_tensor * weighted,
        ggml_tensor ** act,
        ggml_tensor ** weights) {
    if (weighted == nullptr || act == nullptr || weights == nullptr ||
            weighted->op != GGML_OP_MUL ||
            !ggml_metal_tensor_name_has_token(weighted, "ffn_moe_weighted_swiglu") ||
            weighted->src[0] == nullptr || weighted->src[1] == nullptr) {
        return false;
    }

    ggml_tensor * a = weighted->src[0];
    ggml_tensor * b = weighted->src[1];
    if (ggml_metal_tensor_name_has_token(a, "ffn_moe_weights_scaled")) {
        std::swap(a, b);
    }

    if (!ggml_metal_tensor_name_has_token(a, "ffn_moe_swiglu_limited") ||
            !ggml_metal_tensor_name_has_token(b, "ffn_moe_weights_scaled") ||
            a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
            weighted->type != GGML_TYPE_F32) {
        return false;
    }

    *act = a;
    *weights = b;
    return true;
}

static bool ggml_metal_mul_mat_id_get_limited_swiglu_sources(
        ggml_tensor * limited,
        ggml_tensor ** gate,
        ggml_tensor ** up,
        ggml_tensor ** gate_clamp,
        ggml_tensor ** up_clamp,
        ggml_tensor ** silu,
        float * limit) {
    if (limited == nullptr || gate == nullptr || up == nullptr ||
            gate_clamp == nullptr || up_clamp == nullptr || silu == nullptr || limit == nullptr ||
            limited->op != GGML_OP_MUL ||
            !ggml_metal_tensor_name_has_token(limited, "ffn_moe_swiglu_limited") ||
            limited->src[0] == nullptr || limited->src[1] == nullptr) {
        return false;
    }

    ggml_tensor * maybe_silu = limited->src[0];
    ggml_tensor * maybe_up_clamp = limited->src[1];
    if (maybe_silu->op != GGML_OP_UNARY) {
        std::swap(maybe_silu, maybe_up_clamp);
    }

    if (maybe_silu == nullptr || maybe_up_clamp == nullptr ||
            maybe_silu->op != GGML_OP_UNARY ||
            ggml_get_unary_op(maybe_silu) != GGML_UNARY_OP_SILU ||
            maybe_up_clamp->op != GGML_OP_CLAMP ||
            maybe_silu->src[0] == nullptr ||
            maybe_silu->src[0]->op != GGML_OP_CLAMP ||
            maybe_silu->src[0]->src[0] == nullptr ||
            maybe_up_clamp->src[0] == nullptr ||
            maybe_silu->src[0]->src[0]->op != GGML_OP_MUL_MAT_ID ||
            maybe_up_clamp->src[0]->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    *silu = maybe_silu;
    *gate_clamp = maybe_silu->src[0];
    *up_clamp = maybe_up_clamp;
    *gate = (*gate_clamp)->src[0];
    *up = (*up_clamp)->src[0];
    *limit = ggml_get_op_params_f32(*gate_clamp, 1);

    return true;
}

static bool ggml_metal_mul_mat_id_get_down_sum6_plan(
        ggml_metal_op_t ctx,
        int idx,
        ggml_tensor * op,
        ggml_metal_mul_mat_id_down_sum6_plan * plan) {
    if (ctx == nullptr || op == nullptr || plan == nullptr) {
        return false;
    }

    *plan = {};

    if (!ggml_metal_mul_mat_id_experimental_dsv4_down_sum6_enabled() ||
            ggml_metal_dsv4_experimental_ffn_moe_stage_compare_enabled()) {
        return false;
    }

    if (!ggml_metal_tensor_name_has_token(op, "ffn_moe_down") ||
            op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr ||
            op->src[0]->type != GGML_TYPE_Q2_K || op->src[1]->type != GGML_TYPE_F32 ||
            op->src[2]->type != GGML_TYPE_I32 ||
            op->ne[1] != 6 || op->ne[2] != 1 || op->ne[3] != 1) {
        if (ctx->debug_fusion > 0 && ggml_metal_tensor_name_has_token(op, "ffn_moe_down")) {
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "down-sum6-shape");
        }
        return false;
    }

    if (idx + 5 >= ctx->n_nodes()) {
        if (ctx->debug_fusion > 0) {
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "down-sum6-short-graph");
        }
        return false;
    }

    ggml_tensor * first = ctx->node(idx + 1);
    if (first == nullptr || first->op != GGML_OP_ADD ||
            !ggml_metal_tensor_is_view_of(first->src[0], op) ||
            !ggml_metal_tensor_is_view_of(first->src[1], op)) {
        if (ctx->debug_fusion > 0) {
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "down-sum6-first-add");
        }
        return false;
    }
    plan->add_local_idxs[0] = idx + 1;

    ggml_tensor * acc = first;
    for (int i = 2; i < 6; ++i) {
        const int local_idx = idx + i;
        ggml_tensor * add = ctx->node(local_idx);
        if (add == nullptr || add->op != GGML_OP_ADD) {
            if (ctx->debug_fusion > 0) {
                ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "down-sum6-add-op");
            }
            return false;
        }

        const bool expected_order = add->src[0] == acc && ggml_metal_tensor_is_view_of(add->src[1], op);
        const bool swapped_order = add->src[1] == acc && ggml_metal_tensor_is_view_of(add->src[0], op);
        if (!expected_order && !swapped_order) {
            if (ctx->debug_fusion > 0) {
                ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "down-sum6-add-src");
            }
            return false;
        }

        acc = add;
        plan->add_local_idxs[i - 1] = local_idx;
    }

    plan->dst = acc;
    plan->dst_local_idx = idx + 5;
    plan->n_fuse_contiguous = 6;
    plan->weighted_src = op->src[1];
    if (ggml_metal_mul_mat_id_experimental_dsv4_weighted_down_enabled() &&
            ggml_metal_mul_mat_id_get_weighted_swiglu_sources(op->src[1], &plan->weighted_act, &plan->weights)) {
        const bool safe_weighted =
                !ggml_metal_tensor_overlaps(plan->dst, plan->weighted_act) &&
                !ggml_metal_tensor_overlaps(plan->dst, plan->weights);
        if (safe_weighted) {
            plan->weighted = true;
        } else if (ctx->debug_fusion > 0) {
            GGML_LOG_INFO("%s: weighted-down rejected alias dst=%s act=%s weights=%s act_overlap=%d weights_overlap=%d\n",
                    __func__,
                    plan->dst ? plan->dst->name : "-",
                    plan->weighted_act ? plan->weighted_act->name : "-",
                    plan->weights ? plan->weights->name : "-",
                    plan->dst && plan->weighted_act ? ggml_metal_tensor_overlaps(plan->dst, plan->weighted_act) : 0,
                    plan->dst && plan->weights ? ggml_metal_tensor_overlaps(plan->dst, plan->weights) : 0);
        }
    }
    return true;
}

static bool ggml_metal_mul_mat_id_get_glu_fuse_plan(
        ggml_metal_op_t ctx,
        int idx,
        ggml_tensor * op,
        ggml_metal_mul_mat_id_glu_fuse_plan * plan) {
    if (plan == nullptr) {
        return false;
    }

    *plan = {};

    if (!ctx->use_fusion) {
        return false;
    }

    if (idx + 1 >= ctx->n_nodes()) {
        return false;
    }

    if (ctx->debug_fusion > 0 && op->name[0] != '\0' && strstr(op->name, "ffn_moe_gate_up") != nullptr) {
        ggml_tensor * next0 = ctx->node(idx + 1);
        GGML_LOG_INFO("%s: probe op=%s next0=%s/%s\n",
                __func__, op->name, ggml_op_name(next0->op), next0->name);
        if (idx + 2 < ctx->n_nodes()) {
            ggml_tensor * next1 = ctx->node(idx + 2);
            GGML_LOG_INFO("%s: probe op=%s next1=%s/%s\n",
                    __func__, op->name, ggml_op_name(next1->op), next1->name);
        }
    }

    static const ggml_op ops_direct[] = {
        GGML_OP_MUL_MAT_ID,
        GGML_OP_GLU,
    };
    static const int outputs_direct[] = { 1 };

    ggml_tensor * glu = ctx->node(idx + 1);
    if (ctx->can_fuse_subgraph(idx, ops_direct, 2, outputs_direct, 1) &&
        glu->src[0] == op &&
        glu->src[1] == nullptr &&
        ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU) {
        plan->glu = glu;
        plan->n_fuse = 2;
        return true;
    }

    if (idx + 2 >= ctx->n_nodes()) {
        return false;
    }

    static const ggml_op ops_scaled_get_rows[] = {
        GGML_OP_MUL_MAT_ID,
        GGML_OP_GET_ROWS,
        GGML_OP_MUL,
        GGML_OP_GLU,
    };
    static const int outputs_scaled_get_rows[] = { 3 };

    ggml_tensor * get_rows = ctx->node(idx + 1);
    ggml_tensor * mul = ctx->node(idx + 2);
    glu = ctx->node(idx + 3 < ctx->n_nodes() ? idx + 3 : idx + 2);

    if (idx + 3 < ctx->n_nodes() &&
        ctx->can_fuse_subgraph(idx, ops_scaled_get_rows, 4, outputs_scaled_get_rows, 1) &&
        get_rows->type == GGML_TYPE_F32 &&
        get_rows->ne[0] == 1 &&
        glu->src[0] == mul &&
        glu->src[1] == nullptr &&
        ggml_get_glu_op(glu) == GGML_GLU_OP_SWIGLU &&
        ((mul->src[0] == op && mul->src[1] == get_rows) ||
         (mul->src[1] == op && mul->src[0] == get_rows))) {
        plan->glu = glu;
        plan->scale = get_rows;
        plan->n_fuse = 4;
        return true;
    }

    static const ggml_op ops_scaled[] = {
        GGML_OP_MUL_MAT_ID,
        GGML_OP_MUL,
        GGML_OP_GLU,
    };
    static const int outputs_scaled[] = { 2 };

    mul = ctx->node(idx + 1);
    glu = ctx->node(idx + 2);

    if (!ctx->can_fuse_subgraph(idx, ops_scaled, 3, outputs_scaled, 1) ||
        glu->src[0] != mul ||
        glu->src[1] != nullptr ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return false;
    }

    ggml_tensor * scale = nullptr;
    if (mul->src[0] == op) {
        scale = mul->src[1];
    } else if (mul->src[1] == op) {
        scale = mul->src[0];
    } else {
        return false;
    }

    if (scale == nullptr ||
        scale->type != GGML_TYPE_F32 ||
        scale->ne[0] != 1 ||
        !ggml_is_contiguous_rows(scale)) {
        return false;
    }

    plan->glu = glu;
    plan->scale = scale;
    plan->n_fuse = 3;

    return true;
}

static bool ggml_metal_mul_mat_id_get_split_glu_fuse_plan(
        ggml_metal_op_t ctx,
        int idx,
        ggml_tensor * op,
        ggml_metal_mul_mat_id_split_glu_fuse_plan * plan) {
    if (plan == nullptr) {
        return false;
    }

    *plan = {};

    if (!ctx->use_fusion || !ggml_metal_mul_mat_id_experimental_split_glu_enabled()) {
        return false;
    }

    if (ggml_metal_dsv4_experimental_attn_out_decode_enabled() &&
            ggml_metal_tensor_name_has_token(op, "ffn_moe_gate-1")) {
        ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "attn-out-preserve-overlap");
        return false;
    }

    if (idx + 2 >= ctx->n_nodes()) {
        return false;
    }

    const int peer_local_idx = idx + 1;
    ggml_tensor * peer = ctx->node(peer_local_idx);
    if (peer->op != GGML_OP_MUL_MAT_ID) {
        ggml_metal_dsv4_rmoe_pair_preserve_match_trace(
                ctx, idx, op, "rejected", "backend_op_breaks_pair_window",
                peer, nullptr, nullptr, peer);
        ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "peer-not-mul-mat-id");
        return false;
    }

    int glu_local_idx = -1;
    ggml_tensor * glu = nullptr;
    for (int cand = idx + 2; cand < std::min(ctx->n_nodes(), idx + 8); ++cand) {
        ggml_tensor * cand_node = ctx->node(cand);
        if (cand_node->op != GGML_OP_GLU) {
            continue;
        }
        const bool glu_uses_pair =
                (cand_node->src[0] == op && cand_node->src[1] == peer) ||
                (cand_node->src[0] == peer && cand_node->src[1] == op);
        if (glu_uses_pair) {
            glu_local_idx = cand;
            glu = cand_node;
            break;
        }
    }

    if (glu_local_idx < 0 || glu == nullptr) {
        if (!ggml_metal_mul_mat_id_experimental_dsv4_limited_swiglu_enabled()) {
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "glu-not-found");
            return false;
        }

        int gate_clamp_local_idx = -1;
        int up_clamp_local_idx = -1;
        int silu_local_idx = -1;
        int limited_local_idx = -1;
        ggml_tensor * gate_clamp = nullptr;
        ggml_tensor * up_clamp = nullptr;
        ggml_tensor * silu = nullptr;
        ggml_tensor * limited = nullptr;

        for (int cand = idx + 2; cand < std::min(ctx->n_nodes(), idx + 18); ++cand) {
            ggml_tensor * cand_node = ctx->node(cand);
            if (cand_node->op == GGML_OP_CLAMP && cand_node->src[0] == op) {
                gate_clamp_local_idx = cand;
                gate_clamp = cand_node;
            } else if (cand_node->op == GGML_OP_CLAMP && cand_node->src[0] == peer) {
                up_clamp_local_idx = cand;
                up_clamp = cand_node;
            } else if (cand_node->op == GGML_OP_UNARY &&
                    ggml_get_unary_op(cand_node) == GGML_UNARY_OP_SILU &&
                    cand_node->src[0] == gate_clamp) {
                silu_local_idx = cand;
                silu = cand_node;
            } else if (cand_node->op == GGML_OP_MUL &&
                    ((cand_node->src[0] == silu && cand_node->src[1] == up_clamp) ||
                     (cand_node->src[1] == silu && cand_node->src[0] == up_clamp))) {
                limited_local_idx = cand;
                limited = cand_node;
                break;
            }
        }

        if (limited_local_idx < 0 || limited == nullptr || gate_clamp == nullptr || up_clamp == nullptr || silu == nullptr) {
            ggml_metal_dsv4_rmoe_pair_preserve_match_trace(
                    ctx, idx, op, "rejected", "unexpected_op_after_swiglu",
                    peer, nullptr, nullptr, nullptr);
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "glu-not-found");
            return false;
        }

        glu_local_idx = limited_local_idx;
        glu = limited;
        plan->limited_swiglu = true;
        plan->gate_clamp = gate_clamp;
        plan->up_clamp = up_clamp;
        plan->silu = silu;
        plan->limited = limited;
        plan->gate_clamp_local_idx = gate_clamp_local_idx;
        plan->up_clamp_local_idx = up_clamp_local_idx;
        plan->silu_local_idx = silu_local_idx;
        plan->limited_local_idx = limited_local_idx;
    }

    if (peer->src[0] == nullptr || peer->src[1] == nullptr ||
        op->src[0] == nullptr || op->src[1] == nullptr ||
        glu->src[0] == nullptr || (!plan->limited_swiglu && glu->src[1] == nullptr)) {
        ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "missing-src");
        return false;
    }

    if (!plan->limited_swiglu && ggml_get_op_params_i32(glu, 1)) {
        ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "swapped-glu");
        return false;
    }

    if (op->src[1] != peer->src[1] || op->src[2] != peer->src[2]) {
        ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "mismatched-rhs-ids");
        return false;
    }

    if (op->src[0]->type != peer->src[0]->type ||
        !ggml_are_same_shape(op->src[0], peer->src[0]) ||
        !ggml_are_same_stride(op->src[0], peer->src[0])) {
        ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "mismatched-lhs-layout");
        return false;
    }

    if (plan->limited_swiglu) {
        static const ggml_op ops_limited_unweighted[] = {
            GGML_OP_MUL_MAT_ID,
            GGML_OP_MUL_MAT_ID,
            GGML_OP_CLAMP,
            GGML_OP_CLAMP,
            GGML_OP_UNARY,
            GGML_OP_MUL,
        };
        const int local_idxs_unweighted[] = {
            idx,
            peer_local_idx,
            plan->gate_clamp_local_idx,
            plan->up_clamp_local_idx,
            plan->silu_local_idx,
            glu_local_idx,
        };
        const int output_idxs[] = { glu_local_idx };
        if (!ctx->can_fuse_subgraph_sparse(local_idxs_unweighted, 6, ops_limited_unweighted, output_idxs, 1)) {
            ggml_metal_dsv4_rmoe_pair_preserve_match_trace(
                    ctx, idx, op, "rejected", "backend_op_breaks_pair_window",
                    peer, glu, nullptr, nullptr);
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "subgraph-mismatch");
            return false;
        }

        if (ggml_metal_mul_mat_id_experimental_dsv4_weighted_swiglu_enabled()) {
            bool saw_weighted_candidate = false;
            for (int cand = glu_local_idx + 1; cand < std::min(ctx->n_nodes(), idx + 24); ++cand) {
                ggml_tensor * cand_node = ctx->node(cand);
                if (cand_node == nullptr) {
                    continue;
                }
                if (cand_node->op == GGML_OP_MUL_MAT_ID &&
                        ggml_metal_tensor_name_has_token(cand_node, "ffn_moe_down")) {
                    ggml_metal_dsv4_rmoe_pair_preserve_match_trace(
                            ctx, idx, op, "accepted", "generic_down_consumer",
                            peer, glu, plan->weighted, cand_node);
                    break;
                }
                ggml_tensor * weighted_act = nullptr;
                ggml_tensor * weights = nullptr;
                if (!ggml_metal_mul_mat_id_get_weighted_swiglu_sources(cand_node, &weighted_act, &weights) ||
                        weighted_act != glu) {
                    if (cand_node->op == GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE && !saw_weighted_candidate) {
                        ggml_metal_dsv4_rmoe_pair_preserve_match_trace(
                                ctx, idx, op, "rejected", "consumer_changed_to_backend_op",
                                peer, glu, nullptr, cand_node);
                    }
                    continue;
                }
                saw_weighted_candidate = true;

                static const ggml_op ops_limited_weighted[] = {
                    GGML_OP_MUL_MAT_ID,
                    GGML_OP_MUL_MAT_ID,
                    GGML_OP_CLAMP,
                    GGML_OP_CLAMP,
                    GGML_OP_UNARY,
                    GGML_OP_MUL,
                    GGML_OP_MUL,
                };
                const int local_idxs_weighted[] = {
                    idx,
                    peer_local_idx,
                    plan->gate_clamp_local_idx,
                    plan->up_clamp_local_idx,
                    plan->silu_local_idx,
                    glu_local_idx,
                    cand,
                };
                const int weighted_output_idxs[] = { cand };
                if (ctx->can_fuse_subgraph_sparse(local_idxs_weighted, 7, ops_limited_weighted, weighted_output_idxs, 1)) {
                    plan->weighted_swiglu = true;
                    plan->weighted = cand_node;
                    plan->weights = weights;
                    plan->weighted_local_idx = cand;
                    ggml_metal_dsv4_rmoe_pair_preserve_match_trace(
                            ctx, idx, op, "accepted", "weighted_swiglu",
                            peer, glu, cand_node,
                            cand + 1 < ctx->n_nodes() ? ctx->node(cand + 1) : nullptr);
                } else if (ctx->debug_fusion > 0) {
                    ggml_metal_dsv4_rmoe_pair_preserve_match_trace(
                            ctx, idx, op, "rejected", "backend_op_breaks_pair_window",
                            peer, glu, cand_node,
                            cand + 1 < ctx->n_nodes() ? ctx->node(cand + 1) : nullptr);
                    ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "weighted-subgraph-mismatch");
                }
                break;
            }
            if (!plan->weighted_swiglu && !saw_weighted_candidate) {
                ggml_metal_dsv4_rmoe_pair_preserve_match_trace(
                        ctx, idx, op, "rejected", "missing_generic_down_consumer",
                        peer, glu, nullptr,
                        glu_local_idx + 1 < ctx->n_nodes() ? ctx->node(glu_local_idx + 1) : nullptr);
            }
        }
    } else {
        static const ggml_op ops_split[] = {
            GGML_OP_MUL_MAT_ID,
            GGML_OP_MUL_MAT_ID,
            GGML_OP_GLU,
        };
        const int local_idxs[] = { idx, peer_local_idx, glu_local_idx };
        const int output_idxs[] = { glu_local_idx };
        if (!ctx->can_fuse_subgraph_sparse(local_idxs, 3, ops_split, output_idxs, 1)) {
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "subgraph-mismatch");
            return false;
        }
    }

    for (int mid = peer_local_idx + 1; mid < glu_local_idx; ++mid) {
        if (plan->limited_swiglu &&
            (mid == plan->gate_clamp_local_idx ||
             mid == plan->up_clamp_local_idx ||
             mid == plan->silu_local_idx ||
             mid == plan->limited_local_idx)) {
            continue;
        }

        ggml_tensor * mid_node = ctx->node(mid);
        if (ggml_is_empty(mid_node) || (mid_node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (ggml_metal_tensor_overlaps(glu, mid_node) ||
            ggml_metal_tensor_overlaps(glu, mid_node->src[0]) ||
            ggml_metal_tensor_overlaps(glu, mid_node->src[1]) ||
            ggml_metal_tensor_overlaps(glu, mid_node->src[2]) ||
            ggml_metal_tensor_overlaps(glu, mid_node->src[3])) {
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "intervening-overlap");
            return false;
        }
    }

    plan->peer = peer;
    plan->glu = glu;
    plan->peer_local_idx = peer_local_idx;
    plan->glu_local_idx = glu_local_idx;
    plan->n_fuse_contiguous = peer_local_idx == idx + 1 ? 2 : 1;

    if (ggml_metal_dsv4_rmoe_pair_preserve_match_trace_op(op)) {
        ggml_tensor * consumer = glu_local_idx + 1 < ctx->n_nodes() ? ctx->node(glu_local_idx + 1) : nullptr;
        const char * reason = "ok";
        if (plan->limited_swiglu && !plan->weighted_swiglu) {
            if (consumer != nullptr && consumer->op == GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE) {
                reason = "consumer_changed_to_backend_op";
            } else if (consumer == nullptr) {
                reason = "missing_generic_down_consumer";
            } else {
                reason = "missing_generic_down_consumer";
            }
        } else if (plan->weighted_swiglu) {
            consumer = plan->weighted_local_idx + 1 < ctx->n_nodes() ? ctx->node(plan->weighted_local_idx + 1) : nullptr;
            reason = "weighted_swiglu";
        }
        ggml_metal_dsv4_rmoe_pair_preserve_match_trace(
                ctx,
                idx,
                op,
                plan->limited_swiglu && !plan->weighted_swiglu ? "accepted_unweighted_swiglu" : "accepted",
                reason,
                peer,
                glu,
                plan->weighted,
                consumer);
    }

    return true;
}

static int ggml_metal_encode_glu_from_sources(
        ggml_metal_op_t ctx,
        const ggml_tensor * dst_op,
        const ggml_tensor * src0,
        const ggml_tensor * src1) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0s, src0, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0s, src0, nb);
    GGML_TENSOR_LOCALS( int32_t, ned,  dst_op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nbd,  dst_op, nb);

    int32_t ne10 = ne0s0;
    uint64_t nb11 = nb0s1;

    if (src1) {
        ne10 = src1->ne[0];
        nb11 = src1->nb[1];
    }

    auto pipeline = ggml_metal_library_get_pipeline_glu(lib, dst_op);

    const int32_t swp = ggml_get_op_params_i32(dst_op, 1);
    const float alpha = ggml_get_op_params_f32(dst_op, 2);
    const float limit = ggml_get_op_params_f32(dst_op, 3);

    const int32_t i00 = swp ? ned0 : 0;
    const int32_t i10 = swp ? 0 : ned0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne0s0,
        /*.nb01 =*/ nb0s1,
        /*.ne10 =*/ src1 ? ne10 : ne0s0,
        /*.nb11 =*/ src1 ? nb11 : nb0s1,
        /*.ne0  =*/ ned0,
        /*.nb1  =*/ nbd1,
        /*.i00  =*/ src1 ? 0 : i00,
        /*.i10  =*/ src1 ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit,
    };

    const int64_t nrows = ggml_nrows(src0);
    const int32_t nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0s0/2);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src0), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src1 ? src1 : src0), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst_op), 3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

static int ggml_metal_encode_limited_swiglu_from_sources(
        ggml_metal_op_t ctx,
        const ggml_tensor * dst_op,
        const ggml_tensor * gate,
        const ggml_tensor * up,
        float limit) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(gate != nullptr);
    GGML_ASSERT(up != nullptr);
    GGML_ASSERT(dst_op != nullptr);
    GGML_ASSERT(gate->type == up->type);
    GGML_ASSERT(gate->type == GGML_TYPE_F32 || gate->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_are_same_shape(gate, up));
    GGML_ASSERT(ggml_are_same_shape(gate, dst_op));

    GGML_TENSOR_LOCALS( int32_t, ne0s, gate,   ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0s, gate,   nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1s, up,     nb);
    GGML_TENSOR_LOCALS( int32_t, ned,  dst_op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nbd,  dst_op, nb);

    const char * base = gate->type == GGML_TYPE_F32 ? "kernel_swiglu_limited_f32" : "kernel_swiglu_limited_f16";
    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, base);
    if (!pipeline.pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
    }

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne0s0,
        /*.nb01 =*/ nb0s1,
        /*.ne10 =*/ static_cast<int32_t>(up->ne[0]),
        /*.nb11 =*/ nb1s1,
        /*.ne0  =*/ ned0,
        /*.nb1  =*/ nbd1,
        /*.i00  =*/ 0,
        /*.i10  =*/ 0,
        /*.alpha=*/ 1.0f,
        /*.limit=*/ limit,
    };

    const int64_t nrows = ggml_nrows(gate);
    const int32_t nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0s0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(gate),   1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(up),     2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst_op), 3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

static ggml_tensor ggml_metal_make_buffer_tensor_2d(
        ggml_backend_buffer_t buffer,
        enum ggml_type type,
        int64_t ne0,
        int64_t ne1,
        void * data,
        const char * name = nullptr) {
    ggml_tensor tensor = {};

    tensor.type   = type;
    tensor.buffer = buffer;
    tensor.ne[0]  = ne0;
    tensor.ne[1]  = ne1;
    tensor.ne[2]  = 1;
    tensor.ne[3]  = 1;
    tensor.nb[0]  = ggml_type_size(type);
    tensor.nb[1]  = ggml_row_size(type, ne0);
    tensor.nb[2]  = tensor.nb[1] * tensor.ne[1];
    tensor.nb[3]  = tensor.nb[2];
    tensor.data   = data;

    if (name != nullptr && name[0] != '\0') {
        snprintf(tensor.name, sizeof(tensor.name), "%s", name);
    }

    return tensor;
}

static bool ggml_metal_encode_mul_mat_shared_swiglu(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_tensor * peer,
        const ggml_tensor * dst_op) {
    if (ctx == nullptr || op == nullptr || peer == nullptr || dst_op == nullptr ||
            op->src[0] == nullptr || op->src[1] == nullptr || peer->src[0] == nullptr) {
        return false;
    }

    if (op->op != GGML_OP_MUL_MAT || peer->op != GGML_OP_MUL_MAT ||
            op->src[0]->type != GGML_TYPE_Q8_0 || peer->src[0]->type != GGML_TYPE_Q8_0 ||
            op->src[1]->type != GGML_TYPE_F32 || dst_op->type != GGML_TYPE_F32 ||
            op->src[1] != peer->src[1] ||
            !ggml_are_same_shape(op, peer) ||
            !ggml_are_same_shape(op, dst_op)) {
        return false;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    if (ne00 != ne10 || ne12 % ne02 != 0 || ne13 % ne03 != 0) {
        return false;
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_shared_swiglu(lib, op);
    if (!pipeline.pipeline) {
        return false;
    }

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;
    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.src0_byte_off =*/ 0,
        /*.src1_byte_off =*/ 0,
        /*.dst_byte_off  =*/ 0,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]),   1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(peer->src[0]), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]),   3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(dst_op),       4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/nr0), ne11, ne12*ne13, 32, nsg, 1);

    return true;
}

static int ggml_metal_encode_mul_mat_from_tensors(
        ggml_metal_op_t ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst,
        bool allow_m5_expert) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    ggml_tensor op = *dst;
    op.op = dst->type == GGML_TYPE_F16 ? GGML_OP_MUL_MAT_F16 : GGML_OP_MUL_MAT;
    op.src[0] = const_cast<ggml_tensor *>(src0);
    op.src[1] = const_cast<ggml_tensor *>(src1);
    op.src[2] = nullptr;
    op.src[3] = nullptr;

    GGML_TENSOR_LOCALS( int32_t, ne0, src0, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, src0, nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, src1, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, src1, nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  dst,  ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  dst,  nb);

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;
    const bool output_f16 = dst->type == GGML_TYPE_F16;

    const int ne11_mm_min = 8;

    if (!output_f16 &&
        src1->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           src0->type == GGML_TYPE_F32  ||
           src0->type == GGML_TYPE_F16  ||
           src0->type == GGML_TYPE_BF16 ||
           src0->type == GGML_TYPE_Q4_0 ||
           src0->type == GGML_TYPE_Q4_1 ||
           src0->type == GGML_TYPE_Q5_0 ||
           src0->type == GGML_TYPE_Q5_1 ||
           src0->type == GGML_TYPE_Q8_0 ||
           src0->type == GGML_TYPE_MXFP4 ||
           src0->type == GGML_TYPE_F8_E4M3_B128 ||
           src0->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           src0->type == GGML_TYPE_Q4_K ||
           src0->type == GGML_TYPE_Q5_K ||
           src0->type == GGML_TYPE_Q6_K ||
           src0->type == GGML_TYPE_Q2_K ||
           src0->type == GGML_TYPE_Q3_K ||
           false) && (ne11 >= 4 && ne11 <= 8)
         )
       )
       ) {
        g_ggml_metal_mul_mat_mv_ext_count.fetch_add(1);

        const int nsg = 2;

        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const int16_t nypsg = 32/nxpsg;
        const int16_t r0ptg = nypsg*nsg;
              int16_t r1ptg = 4;

        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                GGML_ABORT("unsupported ne11");
        };

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, src0->type, src1->type, nsg, nxpsg, r1ptg);

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src0), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src1), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(&op),  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } else if (
        !ggml_is_transposed(src0) &&
        !ggml_is_transposed(src1) &&
        props_dev->has_simdgroup_mm && ne00 >= 64 && ne11 > ne11_mm_min &&
        !ggml_metal_experimental_disable_mul_mm_enabled()) {
        g_ggml_metal_mul_mat_mm_count.fetch_add(1);
        const bool use_m5_expert_pipeline = allow_m5_expert && ggml_metal_mul_mat_use_m5_expert_pipeline(props_dev, &op);
        if (use_m5_expert_pipeline) {
            g_ggml_metal_mul_mat_mm_m5_expert_count.fetch_add(1);
        }
        const bool use_m5_sgmatrix_pipeline = ggml_metal_mul_mat_use_m5_sgmatrix_pipeline(props_dev);
        if (use_m5_sgmatrix_pipeline) {
            g_ggml_metal_mul_mat_mm_m5_sgmatrix_count.fetch_add(1);
        }

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, &op, use_m5_expert_pipeline, use_m5_sgmatrix_pipeline);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src0), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src1), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(&op),  3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + 31)/32), ((ne01 + 63)/64), ne12*ne13, 128, 1, 1);
    } else {
        GGML_ASSERT(!output_f16 && "GGML_OP_MUL_MAT_F16 requires the matrix-matrix Metal path");

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, &op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        g_ggml_metal_mul_mat_mv_count.fetch_add(1);

        ggml_metal_kargs_mul_mv args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nr0  =*/ nr0,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
            /*.src0_byte_off =*/ 0,
            /*.src1_byte_off =*/ 0,
            /*.dst_byte_off  =*/ 0,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src0), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src1), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(&op),  3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/nr0), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
    }

    return 1;
}

static int ggml_metal_encode_swiglu_scaled_from_merged(
        ggml_metal_op_t ctx,
        const ggml_tensor * dst_op,
        const ggml_tensor * merged,
        const ggml_tensor * scale) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(scale != nullptr);
    GGML_ASSERT(ggml_get_glu_op(dst_op) == GGML_GLU_OP_SWIGLU);

    GGML_TENSOR_LOCALS( int32_t, ne0s, merged, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0s, merged, nb);
    GGML_TENSOR_LOCALS( int32_t, ned,  dst_op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nbd,  dst_op, nb);

    auto pipeline = ggml_metal_library_get_pipeline_glu_scaled(lib, dst_op);

    const int32_t swp = ggml_get_op_params_i32(dst_op, 1);
    const int32_t i00 = swp ? ned0 : 0;
    const int32_t i10 = swp ? 0 : ned0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne0s0,
        /*.nb01 =*/ nb0s1,
        /*.ne10 =*/ 1,
        /*.nb11 =*/ scale->nb[1],
        /*.ne0  =*/ ned0,
        /*.nb1  =*/ nbd1,
        /*.i00  =*/ i00,
        /*.i10  =*/ i10,
        /*.alpha=*/ ggml_get_op_params_f32(dst_op, 2),
        /*.limit=*/ ggml_get_op_params_f32(dst_op, 3),
    };

    const int64_t nrows = ggml_nrows(merged);
    const int32_t nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0s0/2);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(merged), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(scale),  2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst_op), 3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

static void ggml_metal_encode_mul_mat_id_decode_mv(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const int32_t * expert_ids,
        int64_t n_experts) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb12,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.src0_byte_off =*/ 0,
        /*.src1_byte_off =*/ 0,
        /*.dst_byte_off  =*/ 0,
    };

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    for (int64_t idx_exp = 0; idx_exp < n_experts; ++idx_exp) {
        const int32_t expert_id = expert_ids[idx_exp];
        GGML_ASSERT(expert_id >= 0 && expert_id < ne02);

        ggml_metal_buffer_id bid_src0_cur = bid_src0;
        ggml_metal_buffer_id bid_src1_cur = bid_src1;
        ggml_metal_buffer_id bid_dst_cur  = bid_dst;

        bid_src0_cur.offs += (uint64_t) expert_id * nb02;
        bid_src1_cur.offs += (uint64_t) (idx_exp % ne11) * nb11;
        bid_dst_cur.offs  += (uint64_t) idx_exp * op->nb[1];

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0_cur, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1_cur, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst_cur,  3);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), 1, 1, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), 1, 1, 32, nsg, 1);
        }
    }
}

static void ggml_metal_encode_mul_mat_id_decode_mv_replay(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_metal_mul_mat_id_decode_replay_entry & entry) {
    ggml_metal_encoder_t enc = ctx->enc;

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(ctx->lib, op);
    ggml_metal_kargs_mul_mv args = entry.args;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, entry.smem, 0);
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);

    for (int idx_exp = 0; idx_exp < entry.n_experts; ++idx_exp) {
        ggml_metal_buffer_id bid_src0_cur = bid_src0;
        ggml_metal_buffer_id bid_src1_cur = bid_src1;
        ggml_metal_buffer_id bid_dst_cur  = bid_dst;

        bid_src0_cur.offs += entry.src0_offsets[idx_exp];
        bid_src1_cur.offs += entry.src1_offsets[idx_exp];
        bid_dst_cur.offs  += entry.dst_offsets[idx_exp];

        ggml_metal_encoder_set_buffer(enc, bid_src0_cur, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1_cur, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst_cur,  3);
        ggml_metal_encoder_dispatch_threadgroups(enc, entry.tg0, 1, 1, entry.tptg0, entry.tptg1, 1);
    }
}

static bool ggml_metal_encode_mul_mat_id_decode_mv_icb(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_metal_mul_mat_id_decode_replay_entry & entry) {
    if (entry.icb == nullptr) {
        return false;
    }

    const ggml_metal_buffer_id bid_args = ggml_metal_owned_buffer_get_id(entry.args_buffer);
    const ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    const ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    const ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    std::array<std::pair<ggml_metal_buffer_id, uint32_t>, 4> resources = {{
        { bid_args, GGML_METAL_RESOURCE_USAGE_READ },
        { bid_src0, GGML_METAL_RESOURCE_USAGE_READ },
        { bid_src1, GGML_METAL_RESOURCE_USAGE_READ },
        { bid_dst,  GGML_METAL_RESOURCE_USAGE_WRITE },
    }};

    for (size_t i = 0; i < resources.size(); ++i) {
        if (resources[i].first.metal == nullptr || resources[i].second == 0) {
            continue;
        }

        for (size_t j = i + 1; j < resources.size(); ++j) {
            if (resources[i].first.metal == resources[j].first.metal) {
                resources[i].second |= resources[j].second;
                resources[j].second = 0;
            }
        }

        ggml_metal_encoder_use_resource(ctx->enc, resources[i].first, resources[i].second);
    }

    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, entry.smem, 0);
    ggml_metal_encoder_set_pipeline(ctx->enc, ggml_metal_library_get_pipeline_mul_mv(ctx->lib, op));
    return ggml_metal_encoder_execute_icb(ctx->enc, entry.icb, size_t(entry.n_experts));
}

static bool ggml_metal_encode_mul_mat_id_decode_mv_pair(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_tensor * peer,
        const int32_t * expert_ids,
        int64_t n_experts) {
    if (op == nullptr || peer == nullptr || expert_ids == nullptr) {
        return false;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_pair(lib, op);
    if (!pipeline.pipeline) {
        return false;
    }

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_peer_src0 = ggml_metal_get_buffer_id(peer->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);
    ggml_metal_buffer_id bid_peer_dst = ggml_metal_get_buffer_id(peer);

    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb12,
        /*.ne0  =*/ static_cast<int32_t>(op->ne[0]),
        /*.ne1  =*/ 1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.src0_byte_off =*/ 0,
        /*.src1_byte_off =*/ 0,
        /*.dst_byte_off  =*/ 0,
    };

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    for (int64_t idx_exp = 0; idx_exp < n_experts; ++idx_exp) {
        const int32_t expert_id = expert_ids[idx_exp];
        GGML_ASSERT(expert_id >= 0 && expert_id < ne02);

        ggml_metal_buffer_id bid_src0_cur = bid_src0;
        ggml_metal_buffer_id bid_peer_src0_cur = bid_peer_src0;
        ggml_metal_buffer_id bid_src1_cur = bid_src1;
        ggml_metal_buffer_id bid_dst_cur = bid_dst;
        ggml_metal_buffer_id bid_peer_dst_cur = bid_peer_dst;

        bid_src0_cur.offs += (uint64_t) expert_id * nb02;
        bid_peer_src0_cur.offs += (uint64_t) expert_id * peer->src[0]->nb[2];
        bid_src1_cur.offs += (uint64_t) (idx_exp % ne11) * nb11;
        bid_dst_cur.offs += (uint64_t) idx_exp * op->nb[1];
        bid_peer_dst_cur.offs += (uint64_t) idx_exp * peer->nb[1];

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0_cur, 1);
        ggml_metal_encoder_set_buffer(enc, bid_peer_src0_cur, 2);
        ggml_metal_encoder_set_buffer(enc, bid_src1_cur, 3);
        ggml_metal_encoder_set_buffer(enc, bid_dst_cur, 4);
        ggml_metal_encoder_set_buffer(enc, bid_peer_dst_cur, 5);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), 1, 1, 32, nsg, 1);
    }

    return true;
}

static bool ggml_metal_encode_mul_mat_id_decode_mv_pair_limited_swiglu(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_tensor * peer,
        const ggml_tensor * dst_op,
        const int32_t * expert_ids,
        int64_t n_experts,
        float limit) {
    if (op == nullptr || peer == nullptr || dst_op == nullptr || expert_ids == nullptr) {
        return false;
    }

    if (dst_op->type != GGML_TYPE_F32 || op->src[0]->type != peer->src[0]->type ||
            op->src[1]->type != GGML_TYPE_F32 || op->ne[0] != dst_op->ne[0]) {
        return false;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_pair_swiglu(lib, op);
    if (!pipeline.pipeline) {
        return false;
    }

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_peer_src0 = ggml_metal_get_buffer_id(peer->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst_op);

    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb12,
        /*.ne0  =*/ static_cast<int32_t>(op->ne[0]),
        /*.ne1  =*/ 1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.src0_byte_off =*/ 0,
        /*.src1_byte_off =*/ 0,
        /*.dst_byte_off  =*/ 0,
    };

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    for (int64_t idx_exp = 0; idx_exp < n_experts; ++idx_exp) {
        const int32_t expert_id = expert_ids[idx_exp];
        GGML_ASSERT(expert_id >= 0 && expert_id < ne02);

        ggml_metal_buffer_id bid_src0_cur = bid_src0;
        ggml_metal_buffer_id bid_peer_src0_cur = bid_peer_src0;
        ggml_metal_buffer_id bid_src1_cur = bid_src1;
        ggml_metal_buffer_id bid_dst_cur = bid_dst;

        bid_src0_cur.offs += (uint64_t) expert_id * nb02;
        bid_peer_src0_cur.offs += (uint64_t) expert_id * peer->src[0]->nb[2];
        bid_src1_cur.offs += (uint64_t) (idx_exp % ne11) * nb11;
        bid_dst_cur.offs += (uint64_t) idx_exp * dst_op->nb[1];

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_bytes(enc, &limit, sizeof(limit), 1);
        ggml_metal_encoder_set_buffer(enc, bid_src0_cur, 2);
        ggml_metal_encoder_set_buffer(enc, bid_peer_src0_cur, 3);
        ggml_metal_encoder_set_buffer(enc, bid_src1_cur, 4);
        ggml_metal_encoder_set_buffer(enc, bid_dst_cur, 5);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), 1, 1, 32, nsg, 1);
    }

    return true;
}

static bool ggml_metal_encode_mul_mat_id_decode_mv_pair_limited_swiglu_weighted(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_tensor * peer,
        const ggml_tensor * dst_op,
        const ggml_tensor * weights,
        const int32_t * expert_ids,
        int64_t n_experts,
        float limit) {
    if (ctx == nullptr || op == nullptr || peer == nullptr || dst_op == nullptr ||
            weights == nullptr || expert_ids == nullptr ||
            op->src[0] == nullptr || op->src[1] == nullptr || peer->src[0] == nullptr) {
        return false;
    }

    if (dst_op->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
            op->src[0]->type != peer->src[0]->type ||
            op->src[1]->type != GGML_TYPE_F32 || op->ne[0] != dst_op->ne[0]) {
        return false;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_pair_swiglu_weighted(lib, op);
    if (!pipeline.pipeline) {
        return false;
    }

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_peer_src0 = ggml_metal_get_buffer_id(peer->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst_op);
    ggml_metal_buffer_id bid_weights = ggml_metal_get_buffer_id(weights);

    if (bid_weights.metal == nullptr) {
        return false;
    }

    const uint64_t weight_slot_stride = weights->ne[1] == n_experts ? weights->nb[1] : weights->nb[0];
    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb12,
        /*.ne0  =*/ static_cast<int32_t>(op->ne[0]),
        /*.ne1  =*/ 1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.src0_byte_off =*/ 0,
        /*.src1_byte_off =*/ 0,
        /*.dst_byte_off  =*/ 0,
    };

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    for (int64_t idx_exp = 0; idx_exp < n_experts; ++idx_exp) {
        const int32_t expert_id = expert_ids[idx_exp];
        GGML_ASSERT(expert_id >= 0 && expert_id < ne02);

        ggml_metal_buffer_id bid_src0_cur = bid_src0;
        ggml_metal_buffer_id bid_peer_src0_cur = bid_peer_src0;
        ggml_metal_buffer_id bid_src1_cur = bid_src1;
        ggml_metal_buffer_id bid_dst_cur = bid_dst;
        ggml_metal_buffer_id bid_weights_cur = bid_weights;

        bid_src0_cur.offs += (uint64_t) expert_id * nb02;
        bid_peer_src0_cur.offs += (uint64_t) expert_id * peer->src[0]->nb[2];
        bid_src1_cur.offs += (uint64_t) (idx_exp % ne11) * nb11;
        bid_dst_cur.offs += (uint64_t) idx_exp * dst_op->nb[1];
        bid_weights_cur.offs += (uint64_t) idx_exp * weight_slot_stride;

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_bytes(enc, &limit, sizeof(limit), 1);
        ggml_metal_encoder_set_buffer(enc, bid_src0_cur, 2);
        ggml_metal_encoder_set_buffer(enc, bid_peer_src0_cur, 3);
        ggml_metal_encoder_set_buffer(enc, bid_src1_cur, 4);
        ggml_metal_encoder_set_buffer(enc, bid_dst_cur, 5);
        ggml_metal_encoder_set_buffer(enc, bid_weights_cur, 6);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), 1, 1, 32, nsg, 1);
    }

    return true;
}

static bool ggml_metal_encode_mul_mat_id_decode_mv_sum6(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_tensor * dst_op) {
    if (ctx == nullptr || op == nullptr || dst_op == nullptr ||
            op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr) {
        return false;
    }

    if (op->src[0]->type != GGML_TYPE_Q2_K || op->src[1]->type != GGML_TYPE_F32 ||
            op->src[2]->type != GGML_TYPE_I32 || dst_op->type != GGML_TYPE_F32 ||
            op->ne[1] != 6 || op->ne[2] != 1 || op->ne[3] != 1 ||
            dst_op->ne[0] != op->ne[0] || dst_op->ne[1] != 1 || dst_op->ne[2] != 1 || dst_op->ne[3] != 1) {
        return false;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id_sum6(lib, op);
    if (!pipeline.pipeline) {
        return false;
    }

    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv_id args = {
        /*.nei0 =*/ ne20,
        /*.nei1 =*/ ne21,
        /*.nbi1 =*/ nb21,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.ne0  =*/ static_cast<int32_t>(dst_op->ne[0]),
        /*.ne1  =*/ static_cast<int32_t>(dst_op->ne[1]),
        /*.nb1  =*/ static_cast<uint64_t>(dst_op->nb[1]),
        /*.nr0  =*/ nr0,
    };

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(dst_op), 3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 4);

    ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ne21, 1, 32, nsg, 1);

    return true;
}

static bool ggml_metal_encode_mul_mat_id_decode_mv_sum6_weighted(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_tensor * act,
        const ggml_tensor * weights,
        const ggml_tensor * dst_op) {
    if (ctx == nullptr || op == nullptr || act == nullptr || weights == nullptr || dst_op == nullptr ||
            op->src[0] == nullptr || op->src[2] == nullptr) {
        return false;
    }

    if (op->src[0]->type != GGML_TYPE_Q2_K || act->type != GGML_TYPE_F32 ||
            weights->type != GGML_TYPE_F32 || op->src[2]->type != GGML_TYPE_I32 ||
            dst_op->type != GGML_TYPE_F32 ||
            op->ne[1] != 6 || op->ne[2] != 1 || op->ne[3] != 1 ||
            act->ne[1] != 6 || act->ne[2] != 1 || act->ne[3] != 1 ||
            weights->ne[1] != 6 ||
            dst_op->ne[0] != op->ne[0] || dst_op->ne[1] != 1 || dst_op->ne[2] != 1 || dst_op->ne[3] != 1) {
        return false;
    }

    if (ggml_metal_tensor_overlaps(dst_op, act) || ggml_metal_tensor_overlaps(dst_op, weights)) {
        return false;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, act,        ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, act,        nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id_sum6_weighted(lib, op);
    if (!pipeline.pipeline) {
        return false;
    }

    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv_id args = {
        /*.nei0 =*/ ne20,
        /*.nei1 =*/ ne21,
        /*.nbi1 =*/ nb21,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.ne0  =*/ static_cast<int32_t>(dst_op->ne[0]),
        /*.ne1  =*/ static_cast<int32_t>(dst_op->ne[1]),
        /*.nb1  =*/ static_cast<uint64_t>(dst_op->nb[1]),
        /*.nr0  =*/ nr0,
    };

    const uint64_t weight_slot_stride = weights->ne[1] == 6 ? weights->nb[1] : weights->nb[0];
    const uint64_t weight_token_stride = weights->ne[2] > 1 ? weights->nb[2] : 0;
    uint64_t weight_strides[2] = { weight_slot_stride, weight_token_stride };

    static std::atomic<int> log_count { 0 };
    if ((ctx->debug_fusion > 0 || getenv("GGML_METAL_FUSION_DEBUG") != nullptr) && log_count.fetch_add(1) < 8) {
        fprintf(stderr, "%s: weighted-down op=%s act=%s weights=%s dst=%s overlaps act=%d weights=%d weighted_src=%d op=%d act_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] act_nb=[%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "] weights_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] weights_nb=[%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "] slot_stride=%" PRIu64 " token_stride=%" PRIu64 " ids_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] ids_nb=[%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]\n",
                __func__, op->name, act->name, weights->name, dst_op->name,
                ggml_metal_tensor_overlaps(dst_op, act),
                ggml_metal_tensor_overlaps(dst_op, weights),
                ggml_metal_tensor_overlaps(dst_op, op->src[1]),
                ggml_metal_tensor_overlaps(dst_op, op),
                act->ne[0], act->ne[1], act->ne[2], act->ne[3],
                (uint64_t) act->nb[0], (uint64_t) act->nb[1], (uint64_t) act->nb[2], (uint64_t) act->nb[3],
                weights->ne[0], weights->ne[1], weights->ne[2], weights->ne[3],
                (uint64_t) weights->nb[0], (uint64_t) weights->nb[1], (uint64_t) weights->nb[2], (uint64_t) weights->nb[3],
                weight_slot_stride, weight_token_stride,
                op->src[2]->ne[0], op->src[2]->ne[1], op->src[2]->ne[2], op->src[2]->ne[3],
                (uint64_t) op->src[2]->nb[0], (uint64_t) op->src[2]->nb[1], (uint64_t) op->src[2]->nb[2], (uint64_t) op->src[2]->nb[3]);
    }

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_bytes(enc, weight_strides, sizeof(weight_strides), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(act),        3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(dst_op),     4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 5);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights),    6);

    ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ne21, 1, 32, nsg, 1);

    return true;
}

static void ggml_metal_prepare_fused_concurrency(
        ggml_metal_op_t ctx,
        std::initializer_list<const ggml_tensor *> fused_nodes) {
    for (const ggml_tensor * node : fused_nodes) {
        if (node == nullptr || ggml_is_empty(node) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        if (!ggml_metal_op_concurrency_check(ctx, node)) {
            ggml_metal_op_concurrency_reset(ctx);
            break;
        }
    }
}

static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {
    struct ggml_tensor * node = ctx->node(idx);

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    if (ctx->is_skipped(idx)) {
        return 1;
    }

    if (ggml_is_empty(node)) {
        return 1;
    }

    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            {
                // noop -> next node
                if (ctx->debug_graph > 0) {
                    GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(node->op), "(noop)");
                }
            } return 1;
        default:
            {
            } break;
    }

    if (!ggml_metal_device_supports_op(ctx->dev, node)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(node));
        GGML_ABORT("unsupported op");
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 1;
    }

    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        const bool is_concurrent = ggml_metal_op_concurrency_check(ctx, node);

        if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        if (ctx->debug_graph > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %-12s %s\n", __func__, idx, ggml_op_name(node->op), ggml_get_name(node), is_concurrent ? "(concurrent)" : "");
        }
        if (ctx->debug_graph > 1) {
            GGML_TENSOR_LOCALS( int64_t, ne0, node->src[0], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb0, node->src[0], nb);
            GGML_TENSOR_LOCALS( int64_t, ne1, node->src[1], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb1, node->src[1], nb);
            GGML_TENSOR_LOCALS( int64_t, ne2, node->src[2], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb2, node->src[2], nb);
            GGML_TENSOR_LOCALS( int64_t, ne3, node->src[3], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb3, node->src[3], nb);
            GGML_TENSOR_LOCALS( int64_t, ne,  node,         ne);
            GGML_TENSOR_LOCALS(uint64_t, nb,  node,         nb);

            if (node->src[0]) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(node->src[0]), node->src[0]->name);
            }
            if (node->src[1]) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(node->src[1]), node->src[1]->name);
            }
            if (node->src[2]) {
                GGML_LOG_DEBUG("%s: src2 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[2]->type), ne20, ne21, ne22, ne23, nb20, nb21, nb22, nb23,
                        ggml_is_contiguous(node->src[2]), node->src[2]->name);
            }
            if (node->src[3]) {
                GGML_LOG_DEBUG("%s: src3 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[3]->type), ne30, ne31, ne32, ne33, nb30, nb31, nb32, nb33,
                        ggml_is_contiguous(node->src[3]), node->src[3]->name);
            }
            if (node) {
                GGML_LOG_DEBUG("%s: node  - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(node->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        node->name);
            }
        }
    }

    switch (node->op) {
        case GGML_OP_CONCAT:
            {
                n_fuse = ggml_metal_op_concat(ctx, idx);
            } break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } break;
        case GGML_OP_ADD_ID:
            {
                n_fuse = ggml_metal_op_add_id(ctx, idx);
            } break;
        case GGML_OP_REPEAT:
            {
                n_fuse = ggml_metal_op_repeat(ctx, idx);
            } break;
        case GGML_OP_ACC:
            {
                n_fuse = ggml_metal_op_acc(ctx, idx);
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
            {
                n_fuse = ggml_metal_op_unary(ctx, idx);
            } break;
        case GGML_OP_GLU:
            {
                n_fuse = ggml_metal_op_glu(ctx, idx);
            } break;
        case GGML_OP_SUM:
            {
                n_fuse = ggml_metal_op_sum(ctx, idx);
            } break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            {
                n_fuse = ggml_metal_op_sum_rows(ctx, idx);
            } break;
        case GGML_OP_CUMSUM:
            {
                n_fuse = ggml_metal_op_cumsum(ctx, idx);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_fuse = ggml_metal_op_soft_max(ctx, idx);
            } break;
        case GGML_OP_SSM_CONV:
            {
                n_fuse = ggml_metal_op_ssm_conv(ctx, idx);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                n_fuse = ggml_metal_op_ssm_scan(ctx, idx);
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            {
                n_fuse = ggml_metal_op_rwkv(ctx, idx);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                n_fuse = ggml_metal_op_gated_delta_net(ctx, idx);
            } break;
        case GGML_OP_DSV4_HC_SPLIT_SINKHORN:
            {
                n_fuse = ggml_metal_op_dsv4_hc_split_sinkhorn(ctx, idx);
            } break;
        case GGML_OP_DSV4_HC_WEIGHTED_SUM:
            {
                n_fuse = ggml_metal_op_dsv4_hc_weighted_sum(ctx, idx);
            } break;
        case GGML_OP_DSV4_HC_EXPAND:
            {
                n_fuse = ggml_metal_op_dsv4_hc_expand(ctx, idx);
            } break;
        case GGML_OP_DSV4_FP8_KV_QUANTIZE:
            {
                n_fuse = ggml_metal_op_dsv4_fp8_kv_quantize(ctx, idx);
            } break;
        case GGML_OP_DSV4_HADAMARD_FP4_QUANTIZE:
            {
                n_fuse = ggml_metal_op_dsv4_hadamard_fp4_quantize(ctx, idx);
            } break;
        case GGML_OP_DSV4_INDEXER_WEIGHTED_SCORE:
            {
                n_fuse = ggml_metal_op_dsv4_indexer_weighted_score(ctx, idx);
            } break;
        case GGML_OP_DSV4_COMPRESSOR_PAIR_PROJ:
            {
                n_fuse = ggml_metal_op_dsv4_compressor_pair_proj(ctx, idx);
            } break;
        case GGML_OP_DSV4_MIXED_ATTN:
            {
                n_fuse = ggml_metal_op_dsv4_mixed_attn(ctx, idx);
            } break;
        case GGML_OP_DSV4_DECODE_COMPRESS:
            {
                n_fuse = ggml_metal_op_dsv4_decode_compress(ctx, idx);
            } break;
        case GGML_OP_DSV4_ROPE_TAIL:
            {
                n_fuse = ggml_metal_op_dsv4_rope_tail(ctx, idx);
            } break;
        case GGML_OP_DSV4_ATTN_OUT_DECODE:
            {
                n_fuse = ggml_metal_op_dsv4_attn_out_decode(ctx, idx);
            } break;
        case GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE:
            {
                n_fuse = ggml_metal_op_dsv4_compressor_update_decode(ctx, idx);
            } break;
        case GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE_V2:
            {
                n_fuse = ggml_metal_op_dsv4_compressor_update_decode_v2(ctx, idx);
            } break;
        case GGML_OP_DSV4_KV_FINALIZE_DECODE:
            {
                n_fuse = ggml_metal_op_dsv4_kv_finalize_decode(ctx, idx);
            } break;
        case GGML_OP_DSV4_FFN_MOE_DECODE_STAGE:
            {
                n_fuse = ggml_metal_op_dsv4_ffn_moe_decode_stage(ctx, idx);
            } break;
        case GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE:
            {
                n_fuse = ggml_metal_op_dsv4_routed_moe_one_tensor_decode(ctx, idx);
            } break;
        case GGML_OP_DSV4_DECODE_LAYER_EXECUTOR_DRYRUN:
            {
                n_fuse = ggml_metal_op_dsv4_decode_layer_executor_dryrun(ctx, idx);
            } break;
        case GGML_OP_DSV4_DECODE_LAYER:
            {
                n_fuse = ggml_metal_op_dsv4_decode_layer(ctx, idx);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                n_fuse = ggml_metal_op_solve_tri(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT:
            {
                n_fuse = ggml_metal_op_mul_mat(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT_F16:
            {
                n_fuse = ggml_metal_op_mul_mat_f16(ctx, idx);
            } break;
        case GGML_OP_FLASHMOE_SPLIT_GLU:
            {
                n_fuse = ggml_metal_op_flashmoe_split_glu(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
            } break;
        case GGML_OP_GET_ROWS:
            {
                n_fuse = ggml_metal_op_get_rows(ctx, idx);
            } break;
        case GGML_OP_SET_ROWS:
            {
                n_fuse = ggml_metal_op_set_rows(ctx, idx);
            } break;
        case GGML_OP_DIAG:
            {
                n_fuse = ggml_metal_op_diag(ctx, idx);
            } break;
        case GGML_OP_L2_NORM:
            {
                n_fuse = ggml_metal_op_l2_norm(ctx, idx);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                n_fuse = ggml_metal_op_group_norm(ctx, idx);
            } break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            {
                n_fuse = ggml_metal_op_norm(ctx, idx);
            } break;
        case GGML_OP_ROPE:
            {
                n_fuse = ggml_metal_op_rope(ctx, idx);
            } break;
        case GGML_OP_IM2COL:
            {
                n_fuse = ggml_metal_op_im2col(ctx, idx);
            } break;
        case GGML_OP_CONV_2D:
            {
                n_fuse = ggml_metal_op_conv_2d(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = ggml_metal_op_conv_transpose_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_fuse = ggml_metal_op_conv_transpose_2d(ctx, idx);
            } break;
        case GGML_OP_UPSCALE:
            {
                n_fuse = ggml_metal_op_upscale(ctx, idx);
            } break;
        case GGML_OP_PAD:
            {
                n_fuse = ggml_metal_op_pad(ctx, idx);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = ggml_metal_op_pad_reflect_1d(ctx, idx);
            } break;
        case GGML_OP_ARANGE:
            {
                n_fuse = ggml_metal_op_arange(ctx, idx);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = ggml_metal_op_timestep_embedding(ctx, idx);
            } break;
        case GGML_OP_ARGSORT:
            {
                n_fuse = ggml_metal_op_argsort(ctx, idx);
            } break;
        case GGML_OP_TOP_K:
            {
                n_fuse = ggml_metal_op_top_k(ctx, idx);
            } break;
        case GGML_OP_TRI:
            {
                n_fuse = ggml_metal_op_tri(ctx, idx);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = ggml_metal_op_flash_attn_ext(ctx, idx);
            } break;
        case GGML_OP_SET:
            {
                n_fuse = ggml_metal_op_set(ctx, idx);
            } break;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            {
                n_fuse = ggml_metal_op_cpy(ctx, idx);
            } break;
        case GGML_OP_POOL_1D:
            {
                n_fuse = ggml_metal_op_pool_1d(ctx, idx);
            } break;
        case GGML_OP_POOL_2D:
            {
                n_fuse = ggml_metal_op_pool_2d(ctx, idx);
            } break;
        case GGML_OP_ARGMAX:
            {
                n_fuse = ggml_metal_op_argmax(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                n_fuse = ggml_metal_op_opt_step_adamw(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_SGD:
            {
                n_fuse = ggml_metal_op_opt_step_sgd(ctx, idx);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                n_fuse = ggml_metal_op_count_equal(ctx, idx);
            } break;
        default:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(node->op));
                GGML_ABORT("fatal error");
            }
    }

    if (ctx->debug_graph > 0) {
        if (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    return n_fuse;
}

static void ggml_metal_dsv4_trace_record_node(
        const ggml_metal_op_t ctx,
        const ggml_tensor * node,
        int local_idx,
        uint64_t dispatch_before,
        uint64_t dispatch_after) {
    const uint64_t dispatch_count = dispatch_after > dispatch_before ? dispatch_after - dispatch_before : 0;
    if (dispatch_count == 0 || node == nullptr || !ggml_metal_dsv4_trace_token_enabled(ctx->trace_token)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_ggml_metal_dsv4_trace_mutex);
    FILE * f = ggml_metal_dsv4_trace_open_locked();
    if (f == nullptr) {
        return;
    }

    const char * name = ggml_get_name(node);
    const char * op_name = ggml_op_name(node->op);
    const char * stage_bucket = ggml_metal_dsv4_trace_stage_bucket(node);
    const char * stage = ggml_metal_dsv4_trace_detail_stage(node, stage_bucket);
    const char * kernel = ggml_metal_dsv4_trace_kernel_name(node);
    const int layer = ggml_metal_dsv4_trace_layer_from_name(name);

    std::fprintf(f,
            "{\"source\":\"ours\",\"impl\":\"ours\",\"token\":%d,\"layer\":%d,\"stage_bucket\":",
            ctx->trace_token,
            layer);
    ggml_metal_dsv4_trace_json(f, stage_bucket);
    std::fprintf(f, ",\"stage\":");
    ggml_metal_dsv4_trace_json(f, stage);
    std::fprintf(f, ",\"kernel\":");
    ggml_metal_dsv4_trace_json(f, kernel);
    std::fprintf(f, ",\"ggml_op\":");
    ggml_metal_dsv4_trace_json(f, op_name);
    std::fprintf(f, ",\"tensor_name\":");
    ggml_metal_dsv4_trace_json(f, name);
    std::fprintf(f, ",\"tensor\":");
    ggml_metal_dsv4_trace_json(f, name);
    std::fprintf(f, ",\"dtype\":");
    ggml_metal_dsv4_trace_json(f, ggml_type_name(node->type));
    char shape[96];
    std::snprintf(shape, sizeof(shape), "[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]",
            node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
    std::fprintf(f, ",\"shape\":");
    ggml_metal_dsv4_trace_json(f, shape);
    std::fprintf(f,
            ",\"dispatch_index\":%" PRIu64
            ",\"dispatch_index_end\":%" PRIu64
            ",\"dispatch_count\":%" PRIu64
            ",\"command_buffer_index\":%d"
            ",\"encoder_index\":%d"
            ",\"local_node_index\":%d"
            ",\"expert_count\":%d"
            ",\"topk\":%d"
            ",\"expert_id\":%d"
            ",\"slot\":%d"
            ",\"is_shared\":%d"
            ",\"route_weight_applied\":%d"
            ",\"ne0\":%" PRId64
            ",\"ne1\":%" PRId64
            ",\"ne2\":%" PRId64
            ",\"ne3\":%" PRId64
            "}\n",
            dispatch_before + 1,
            dispatch_after,
            dispatch_count,
            ctx->trace_command_buffer,
            ctx->trace_command_buffer,
            local_idx,
            std::strcmp(stage_bucket, "ffn") == 0 ? 256 : 0,
            std::strcmp(stage_bucket, "ffn") == 0 ? 6 : 0,
            -1,
            -1,
            ggml_metal_dsv4_trace_is_shared_stage(stage),
            ggml_metal_dsv4_trace_route_weight_applied(name, stage),
            node->ne[0],
            node->ne[1],
            node->ne[2],
            node->ne[3]);
    std::fflush(f);
}

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx) {
    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_push(ctx->enc, ggml_op_desc(ctx->node(idx)));
    }

    const bool profile_dispatches = ggml_metal_dispatch_profile_enabled();
    const bool trace_dispatches = ggml_metal_dsv4_trace_token_enabled(ctx->trace_token);
    const uint64_t dispatch_before = (profile_dispatches || trace_dispatches) ? ggml_metal_encoder_get_dispatch_count() : 0;
    int res = ggml_metal_op_encode_impl(ctx, idx);
    if (profile_dispatches || trace_dispatches) {
        const uint64_t dispatch_after = ggml_metal_encoder_get_dispatch_count();
        if (profile_dispatches) {
            ggml_metal_dispatch_profile_record(ctx->node(idx), dispatch_after - dispatch_before);
        }
        if (trace_dispatches) {
            ggml_metal_dsv4_trace_record_node(ctx, ctx->node(idx), idx, dispatch_before, dispatch_after);
        }
    }
    if (idx + res > ctx->n_nodes()) {
        GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_pop(ctx->enc);
    }

    return res;
}

int ggml_metal_op_concat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t dim = ((const int32_t *) op->op_params)[0];

    ggml_metal_kargs_concat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_CONCAT);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_repeat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_repeat(lib, op->type);

    ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_acc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    auto pipeline = ggml_metal_library_get_pipeline_bin_one(lib, GGML_OP_ADD);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    int nth = 1;

    while (2*nth < args.ne0 && nth < nth_max) {
        nth *= 2;
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    return 1;
}

int ggml_metal_op_unary(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_unary args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.slope =*/ 0.0,
        /*.scale =*/ 0.0,
        /*.bias  =*/ 0.0,
        /*.val   =*/ 0.0,
        /*.min   =*/ 0.0,
        /*.max   =*/ 0.0,
    };

    if (op->op == GGML_OP_LEAKY_RELU) {
        args.slope = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_SCALE) {
        args.scale = ggml_get_op_params_f32(op, 0);
        args.bias  = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_FILL) {
        args.val = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_CLAMP) {
        args.min = ggml_get_op_params_f32(op, 0);
        args.max = ggml_get_op_params_f32(op, 1);
    }

    auto pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    if (pipeline.cnt) {
        const int n = pipeline.c4 ? ggml_nelements(op)/4 : ggml_nelements(op);

        ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        const int nth = MIN(args.ne00, nth_max);

        const int nk0 = (args.ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}

int ggml_metal_op_glu(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (ctx->is_deferred_shared_swiglu(idx) &&
            op->src[0] != nullptr && op->src[1] != nullptr &&
            op->src[0]->op == GGML_OP_MUL_MAT &&
            op->src[1]->op == GGML_OP_MUL_MAT &&
            ggml_metal_encode_mul_mat_shared_swiglu(ctx, op->src[0], op->src[1], op)) {
        g_ggml_metal_mul_mat_shared_swiglu_count.fetch_add(1);
        return 1;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    if (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    auto pipeline = ggml_metal_library_get_pipeline_glu(lib, op);

    const int32_t swp = ggml_get_op_params_i32(op, 1);
    const float alpha = ggml_get_op_params_f32(op, 2);
    const float limit = ggml_get_op_params_f32(op, 3);

    const int32_t i00 = swp ? ne0 : 0;
    const int32_t i10 = swp ? 0 : ne0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit
    };

    const int64_t nrows = ggml_nrows(op->src[0]);

    const int32_t nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00/2);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op  = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const uint64_t n = (uint64_t) ggml_nelements(op->src[0]);

    ggml_metal_kargs_sum args = {
        /*.np =*/ n,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum(lib, op);

    int nth = 32; // SIMD width

    while (nth < (int) n && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) n);

    const int nsg = (nth + 31) / 32;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, nsg * sizeof(float), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_sum_rows args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum_rows(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < args.ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) args.ne00);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_cumsum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline_blk = ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_blk)) {
        nth *= 2;
    }

    GGML_ASSERT(ne00 <= nth*nth);

    const int64_t net0 = (ne00 + nth - 1) / nth;
    const int64_t net1 = ne01;
    const int64_t net2 = ne02;
    const int64_t net3 = ne03;

    const uint64_t nbt0 = sizeof(float);
    const uint64_t nbt1 = net0*nbt0;
    const uint64_t nbt2 = net1*nbt1;
    const uint64_t nbt3 = net2*nbt2;

    const size_t smem = GGML_PAD(32*sizeof(float), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    {
        ggml_metal_kargs_cumsum_blk args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.net0 =*/ net0,
            /*.net1 =*/ net1,
            /*.net2 =*/ net2,
            /*.net3 =*/ net3,
            /*.nbt0 =*/ nbt0,
            /*.nbt1 =*/ nbt1,
            /*.nbt2 =*/ nbt2,
            /*.nbt3 =*/ nbt3,
            /*.outb =*/ ne00 > nth,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
    }

    if (ne00 > nth) {
        ggml_metal_op_concurrency_reset(ctx);

        {
            ggml_metal_kargs_cumsum_blk args = {
                /*.ne00 =*/ net0,
                /*.ne01 =*/ net1,
                /*.ne02 =*/ net2,
                /*.ne03 =*/ net3,
                /*.nb00 =*/ nbt0,
                /*.nb01 =*/ nbt1,
                /*.nb02 =*/ nbt2,
                /*.nb03 =*/ nbt3,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
                /*.outb =*/ false,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, net1, net2, net3, nth, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline_add = ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            ggml_metal_kargs_cumsum_add args = {
                /*.ne00 =*/ ne00,
                /*.ne01 =*/ ne01,
                /*.ne02 =*/ ne02,
                /*.ne03 =*/ ne03,
                /*.nb00 =*/ nb00,
                /*.nb01 =*/ nb01,
                /*.nb02 =*/ nb02,
                /*.nb03 =*/ nb03,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
        }
    }

    return 1;
}

int ggml_metal_op_get_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_get_rows(lib, op->src[0]->type);

    ggml_metal_kargs_get_rows args = {
        /*.ne00t =*/ ggml_is_quantized(op->src[0]->type) ? ne00/16 : ne00,
        /*.ne00  =*/ ne00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    const int nth = std::min(args.ne00t, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const int nw0 = (args.ne00t + nth - 1)/nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne10, ne11, ne12, nth, 1, 1);

    return 1;
}

int ggml_metal_op_set_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_set_rows(lib, op->src[1]->type, op->type);

    const int32_t nk0 = ne0/ggml_blck_size(op->type);

    int nth = 32; // SIMD width

    while (nth < nk0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_kargs_set_rows args = {
        /*.nk0  =*/ nk0,
        /*.ne01 =*/ ne01,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_diag(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t,  ne, op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

    ggml_metal_kargs_diag args = {
        /*.ne00 =*/ne00,
        /*.ne01 =*/ne01,
        /*.ne02 =*/ne02,
        /*.ne03 =*/ne03,
        /*.nb00 =*/nb00,
        /*.nb01 =*/nb01,
        /*.nb02 =*/nb02,
        /*.nb03 =*/nb03,
        /*.ne0  =*/ne0,
        /*.ne1  =*/ne1,
        /*.ne2  =*/ne2,
        /*.ne3  =*/ne3,
        /*.nb0  =*/nb0,
        /*.nb1  =*/nb1,
        /*.nb2  =*/nb2,
        /*.nb3  =*/nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_diag(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, 32, 1, 1);

    return 1;
}

int ggml_metal_op_soft_max(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale;
    float max_bias;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    ggml_metal_kargs_soft_max args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_head_log2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_soft_max(lib, op);

    int nth = 32; // SIMD width

    if (ne00%4 == 0) {
        while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    const bool use_batched = (ne1 > 1);

    if (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        int BATCH_SIZE;
        if      (ne1 > 128) BATCH_SIZE = 256;
        else if (ne1 > 64 ) BATCH_SIZE = 128;
        else if (ne1 > 32 ) BATCH_SIZE = 64;
        else if (ne1 > 16 ) BATCH_SIZE = 32;
        else if (ne1 > 8  ) BATCH_SIZE = 16;
        else if (ne1 > 4  ) BATCH_SIZE = 8;
        else                BATCH_SIZE = 2;

        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        const int n_token_batches = (ne1 + BATCH_SIZE - 1) / BATCH_SIZE;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne1, ne02, 1, 1, 1);
    }

    return 1;
}

int ggml_metal_op_ssm_scan(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const ggml_tensor * src3 = op->src[3];
    const ggml_tensor * src4 = op->src[4];
    const ggml_tensor * src5 = op->src[5];
    const ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ n_seqs,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb00         =*/ nb00,
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb10         =*/ nb10,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.ns12         =*/ nb12/nb10,
        /*.nb13         =*/ nb13,
        /*.nb20         =*/ nb20,
        /*.nb21         =*/ nb21,
        /*.ns21         =*/ nb21/nb20,
        /*.nb22         =*/ nb22,
        /*.ne30         =*/ ne30,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.ns42         =*/ nb42/nb40,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.ns52         =*/ nb52/nb50,
        /*.nb53         =*/ nb53,
        /*.nb0          =*/ nb0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_ssm_scan(lib, op);

    GGML_ASSERT(d_state <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 7);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         8);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, d_inner, n_head, n_seqs, d_state, 1, 1);

    return 1;
}

int ggml_metal_op_rwkv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    auto pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &B, sizeof(B), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &T, sizeof(T), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &C, sizeof(C), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &H, sizeof(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

int ggml_metal_op_gated_delta_net(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;


    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net(lib, op);

    int ida = 0;

    ggml_metal_kargs_gated_delta_net args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20,
        /*.ne21 =*/ ne21,
        /*.ne22 =*/ ne22,
        /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20,
        /*.nb21 =*/ nb21,
        /*.nb22 =*/ nb22,
        /*.nb23 =*/ nb23,
        /*.ns02 =*/ (int32_t) (nb02/sizeof(float)),
        /*.ns12 =*/ (int32_t) (nb12/sizeof(float)),
        /*.ns22 =*/ (int32_t) (nb22/sizeof(float)),
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++); // state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/nsg, op->src[2]->ne[1], op->src[2]->ne[3], 32, nsg, 1);

    return 1;
}

int ggml_metal_op_solve_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_solve_tri args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + nsg - 1)/nsg, ne02, ne03, 32, nsg, 1);

    return 1;
}

int ggml_metal_op_set(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[1]->type, op->type);

    GGML_ASSERT(ne10 % ggml_blck_size(op->src[1]->type) == 0);

    int64_t nk0 = ne10;
    if (ggml_is_quantized(op->src[1]->type)) {
        nk0 = ne10/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne10/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[1]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb10,
        /*.nb01 =*/ nb11,
        /*.nb02 =*/ nb12,
        /*.nb03 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ ggml_element_size(op),
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    bid_dst.offs += offs;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne11 + nrptg - 1)/nrptg, ne12, ne13, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_cpy(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

    GGML_ASSERT(ne00 % ggml_blck_size(op->src[0]->type) == 0);

    int64_t nk0 = ne00;
    if (ggml_is_quantized(op->src[0]->type)) {
        nk0 = ne00/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne00/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_pool_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t s0 = opts[2];
    const int32_t p0 = opts[3];

    const int64_t IW = op->src[0]->ne[0];
    const int64_t OW = op->ne[0];

    const int64_t np = ggml_nelements(op);

    ggml_metal_kargs_pool_1d args_pool_1d = {
        /* .k0 = */  k0,
        /* .s0 = */  s0,
        /* .p0 = */  p0,
        /* .IW = */  IW,
        /* .OW = */  OW,
        /* .np = */  np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_1d, sizeof(args_pool_1d),  0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}


int ggml_metal_op_pool_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t k1 = opts[2];
    const int32_t s0 = opts[3];
    const int32_t s1 = opts[4];
    const int32_t p0 = opts[5];
    const int32_t p1 = opts[6];

    const int64_t IH = op->src[0]->ne[1];
    const int64_t IW = op->src[0]->ne[0];

    const int64_t N  = op->ne[3];
    const int64_t OC = op->ne[2];
    const int64_t OH = op->ne[1];
    const int64_t OW = op->ne[0];

    const int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d args_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_2d, sizeof(args_pool_2d), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

static bool ggml_metal_mul_mat_is_dsv4_compressor_kv_proj(const ggml_tensor * op) {
    return op != nullptr &&
            (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_F16) &&
            strstr(op->name, "dsv4_comp_kv_proj") != nullptr &&
            op->type == GGML_TYPE_F32 &&
            op->src[0] != nullptr &&
            op->src[1] != nullptr &&
            op->src[0]->type == GGML_TYPE_F16 &&
            op->src[1]->type == GGML_TYPE_F32;
}

static bool ggml_metal_mul_mat_is_dsv4_compressor_score_proj(const ggml_tensor * op) {
    return op != nullptr &&
            op->op == GGML_OP_MUL_MAT &&
            strstr(op->name, "dsv4_comp_score_proj") != nullptr &&
            op->type == GGML_TYPE_F32 &&
            op->src[0] != nullptr &&
            op->src[1] != nullptr &&
            op->src[0]->type == GGML_TYPE_F16 &&
            op->src[1]->type == GGML_TYPE_F32;
}

static bool ggml_metal_mul_mat_get_dsv4_compressor_pair_peer(
        ggml_metal_op_t ctx,
        int idx,
        const ggml_tensor * op,
        ggml_tensor ** peer) {
    if (ggml_metal_dsv4_trace_compressor_pair_enabled() &&
            op != nullptr &&
            op->op == GGML_OP_MUL_MAT &&
            op->src[0] != nullptr &&
            op->src[1] != nullptr &&
            op->src[1]->ne[1] == 1 &&
            g_ggml_metal_dsv4_compressor_pair_trace_count.fetch_add(1) < 64) {
        const ggml_tensor * cand = idx + 1 < ctx->n_nodes() ? ctx->node(idx + 1) : nullptr;
        GGML_LOG_INFO("%s: idx=%d op=%s name=%s type=%s src0=%s src1=%s ne=[%lld,%lld,%lld,%lld] src0_ne=[%lld,%lld,%lld,%lld] src1_ne=[%lld,%lld,%lld,%lld] next=%s next_name=%s\n",
                __func__, idx, ggml_op_name(op->op), op->name, ggml_type_name(op->type),
                op->src[0] ? ggml_type_name(op->src[0]->type) : "null",
                op->src[1] ? ggml_type_name(op->src[1]->type) : "null",
                (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3],
                op->src[0] ? (long long) op->src[0]->ne[0] : -1, op->src[0] ? (long long) op->src[0]->ne[1] : -1,
                op->src[0] ? (long long) op->src[0]->ne[2] : -1, op->src[0] ? (long long) op->src[0]->ne[3] : -1,
                op->src[1] ? (long long) op->src[1]->ne[0] : -1, op->src[1] ? (long long) op->src[1]->ne[1] : -1,
                op->src[1] ? (long long) op->src[1]->ne[2] : -1, op->src[1] ? (long long) op->src[1]->ne[3] : -1,
                cand ? ggml_op_name(cand->op) : "null",
                cand ? cand->name : "null");
    }

    if (!ggml_metal_dsv4_experimental_compressor_pair_enabled() ||
            !ggml_metal_mul_mat_is_dsv4_compressor_kv_proj(op) ||
            idx + 1 >= ctx->n_nodes()) {
        return false;
    }

    ggml_tensor * cand = ctx->node(idx + 1);
    if (!ggml_metal_mul_mat_is_dsv4_compressor_score_proj(cand)) {
        return false;
    }
    if (op->src[1] != cand->src[1]) {
        return false;
    }
    if (ggml_is_transposed(op->src[0]) || ggml_is_transposed(op->src[1]) ||
            ggml_is_transposed(cand->src[0]) || ggml_is_transposed(cand->src[1])) {
        return false;
    }
    if (op->src[0]->ne[0] != cand->src[0]->ne[0] ||
            op->src[0]->ne[1] != cand->src[0]->ne[1] ||
            op->src[0]->ne[2] != cand->src[0]->ne[2] ||
            op->src[0]->ne[3] != cand->src[0]->ne[3]) {
        return false;
    }
    if (op->src[1]->ne[0] != cand->src[1]->ne[0] ||
            op->src[1]->ne[1] != cand->src[1]->ne[1] ||
            op->src[1]->ne[2] != cand->src[1]->ne[2] ||
            op->src[1]->ne[3] != cand->src[1]->ne[3]) {
        return false;
    }
    if (op->ne[0] != cand->ne[0] ||
            op->ne[1] != cand->ne[1] ||
            op->ne[2] != cand->ne[2] ||
            op->ne[3] != cand->ne[3]) {
        return false;
    }
    if (op->src[1]->ne[1] != 1 || op->src[0]->ne[0] % 4 != 0) {
        return false;
    }

    *peer = cand;
    return true;
}

static bool ggml_metal_encode_mul_mat_dsv4_compressor_pair(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_tensor * peer) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_compressor_pair(lib, op);
    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.src0_byte_off =*/ 0,
        /*.src1_byte_off =*/ 0,
        /*.dst_byte_off  =*/ 0,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]),   1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(peer->src[0]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]),   3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),           4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(peer),         5);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/nr0, 1, ne12*ne13, 32, nsg, 1);

    g_ggml_metal_dsv4_compressor_pair_count.fetch_add(1);
    return true;
}

static bool ggml_metal_mul_mat_get_dsv4_q8_hc_expand(
        ggml_metal_op_t ctx,
        int idx,
        const ggml_tensor * op,
        ggml_tensor ** next_out) {
    const bool aohc_fused_candidate =
        ggml_metal_dsv4_experimental_aohc_fused_enabled() &&
        ggml_metal_dsv4_is_aohc_fused_high(op);
    auto reject = [&](const char * reason) -> bool {
        if (aohc_fused_candidate) {
            g_ggml_metal_dsv4_aohc_elig_q8hc_rejected_count.fetch_add(1);
            std::lock_guard<std::mutex> lock(g_ggml_metal_dsv4_aohc_elig_mutex);
            if (g_ggml_metal_dsv4_aohc_elig_first_reject_reason.empty()) {
                g_ggml_metal_dsv4_aohc_elig_first_reject_reason = reason;
                g_ggml_metal_dsv4_aohc_elig_first_reject_name = op != nullptr ? op->name : "null";
                g_ggml_metal_dsv4_aohc_elig_first_reject_idx = idx;
            }
        }
        if ((ggml_metal_dsv4_trace_q8_hc_expand_enabled() || ggml_metal_dsv4_trace_aohc_fused_elig_enabled()) &&
                op != nullptr &&
                op->op == GGML_OP_MUL_MAT &&
                op->src[0] != nullptr &&
                op->src[1] != nullptr &&
                (aohc_fused_candidate || (op->src[0]->type == GGML_TYPE_Q8_0 &&
                 op->src[1]->type == GGML_TYPE_F32 &&
                 op->src[1]->ne[1] == 1)) &&
                g_ggml_metal_dsv4_q8_hc_expand_trace_count.fetch_add(1) < 64) {
            const ggml_tensor * cand = idx + 1 < ctx->n_nodes() ? ctx->node(idx + 1) : nullptr;
            GGML_LOG_INFO("%s: reject=%s aohc_candidate=%d idx=%d name=%s op=%s type=%s ne=[%lld,%lld,%lld,%lld] src0_name=%s src0_type=%s src0_ne=[%lld,%lld,%lld,%lld] src0_nb=[%llu,%llu,%llu,%llu] src1_name=%s src1_type=%s src1_ne=[%lld,%lld,%lld,%lld] src1_nb=[%llu,%llu,%llu,%llu] next=%s next_name=%s next_src0=%d\n",
                    __func__, reason, aohc_fused_candidate ? 1 : 0, idx, op->name,
                    ggml_op_name(op->op), ggml_type_name(op->type),
                    (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3],
                    op->src[0]->name, ggml_type_name(op->src[0]->type),
                    (long long) op->src[0]->ne[0], (long long) op->src[0]->ne[1], (long long) op->src[0]->ne[2], (long long) op->src[0]->ne[3],
                    (unsigned long long) op->src[0]->nb[0], (unsigned long long) op->src[0]->nb[1], (unsigned long long) op->src[0]->nb[2], (unsigned long long) op->src[0]->nb[3],
                    op->src[1]->name, ggml_type_name(op->src[1]->type),
                    (long long) op->src[1]->ne[0], (long long) op->src[1]->ne[1], (long long) op->src[1]->ne[2], (long long) op->src[1]->ne[3],
                    (unsigned long long) op->src[1]->nb[0], (unsigned long long) op->src[1]->nb[1], (unsigned long long) op->src[1]->nb[2], (unsigned long long) op->src[1]->nb[3],
                    cand != nullptr ? ggml_op_name(cand->op) : "null",
                    cand != nullptr ? cand->name : "null",
                    cand != nullptr && cand->src[0] == op);
        }
        return false;
    };

    if (next_out == nullptr || !ctx->use_fusion ||
            (!ggml_metal_dsv4_experimental_q8_hc_expand_enabled() && !aohc_fused_candidate)) {
        return reject("disabled");
    }
    if (op == nullptr || op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr) {
        return reject("op");
    }
    if (idx + 1 >= ctx->n_nodes()) {
        return reject("no-next");
    }

    ggml_tensor * next = ctx->node(idx + 1);
    if (next == nullptr || next->op != GGML_OP_DSV4_HC_EXPAND || next->src[0] != op) {
        return reject("next");
    }
    if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_Q8_0 || op->src[1]->type != GGML_TYPE_F32 ||
            next->type != GGML_TYPE_F32 ||
            next->src[1] == nullptr || next->src[2] == nullptr || next->src[3] == nullptr ||
            next->src[1]->type != GGML_TYPE_F32 || next->src[2]->type != GGML_TYPE_F32 || next->src[3]->type != GGML_TYPE_F32) {
        return reject("type");
    }
    if (ggml_is_transposed(op->src[0]) || ggml_is_transposed(op->src[1])) {
        return reject("transposed");
    }
    if (op->src[0]->ne[0] != op->src[1]->ne[0] ||
            op->src[0]->ne[1] != op->ne[0] ||
            op->src[0]->ne[2] != 1 ||
            op->src[0]->ne[3] != 1 ||
            op->src[1]->ne[1] != 1 ||
            op->src[1]->ne[2] != 1 ||
            op->src[1]->ne[3] != 1 ||
            op->ne[1] != 1 ||
            op->ne[2] != 1 ||
            op->ne[3] != 1 ||
            (op->src[0]->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0)) != 0) {
        return reject("mul-shape");
    }
    if (next->ne[0] != op->ne[0] ||
            next->ne[1] != 4 ||
            next->ne[2] != 1 ||
            next->ne[3] != 1 ||
            next->src[1]->ne[0] != op->ne[0] ||
            next->src[1]->ne[1] != 4 ||
            next->src[1]->ne[2] != 1 ||
            next->src[2]->ne[0] != 4 ||
            next->src[2]->ne[1] != 1 ||
            next->src[3]->ne[0] != 4 ||
            next->src[3]->ne[1] != 4 ||
            next->src[3]->ne[2] != 1) {
        return reject("hc-shape");
    }

    static const ggml_op ops[] = {
        GGML_OP_MUL_MAT,
        GGML_OP_DSV4_HC_EXPAND,
    };
    static const int outputs[] = { 0, 1 };
    if (!ctx->can_fuse_subgraph(idx, ops, 2, outputs, 2)) {
        return reject("subgraph");
    }

    if (aohc_fused_candidate) {
        g_ggml_metal_dsv4_aohc_elig_q8hc_eligible_count.fetch_add(1);
    }
    *next_out = next;
    return true;
}

static bool ggml_metal_encode_mul_mat_dsv4_q8_hc_expand(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_tensor * next) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_q8_hc_expand(lib, op);
    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv mv_args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.src0_byte_off =*/ 0,
        /*.src1_byte_off =*/ 0,
        /*.dst_byte_off  =*/ 0,
    };

    ggml_metal_kargs_dsv4_hc_expand hc_args = {
        /*.n_embd    =*/ next->ne[0],
        /*.n_hc      =*/ next->ne[1],
        /*.n_tokens  =*/ next->ne[2],
        /*.n_elem    =*/ next->ne[0] * next->ne[1] * next->ne[2],
        /*.block_nb0 =*/ op->nb[0],
        /*.block_nb1 =*/ op->nb[1],
        /*.res_nb0   =*/ next->src[1]->nb[0],
        /*.res_nb1   =*/ next->src[1]->nb[1],
        /*.res_nb2   =*/ next->src[1]->nb[2],
        /*.post_nb0  =*/ next->src[2]->nb[0],
        /*.post_nb1  =*/ next->src[2]->nb[1],
        /*.comb_nb0  =*/ next->src[3]->nb[0],
        /*.comb_nb1  =*/ next->src[3]->nb[1],
        /*.comb_nb2  =*/ next->src[3]->nb[2],
        /*.dst_nb0   =*/ next->nb[0],
        /*.dst_nb1   =*/ next->nb[1],
        /*.dst_nb2   =*/ next->nb[2],
    };

    auto bid_block = ggml_metal_get_buffer_id(op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &mv_args, sizeof(mv_args), 0);
    ggml_metal_encoder_set_bytes   (enc, &hc_args, sizeof(hc_args), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]),   2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]),   3);
    ggml_metal_encoder_set_buffer  (enc, bid_block,                              4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(next->src[1]), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(next->src[2]), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(next->src[3]), 7);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(next),         8);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/nr0, 1, 1, 32, nsg, 1);

    if (ggml_metal_dsv4_trace_q8_hc_expand_enabled() &&
            g_ggml_metal_dsv4_q8_hc_expand_trace_count.fetch_add(1) < 64) {
        GGML_LOG_INFO("%s: fused name=%s next=%s rows=%d cols=%d hc=%lld\n",
                __func__, op->name, next->name, ne01, ne00, (long long) next->ne[1]);
    }

    g_ggml_metal_dsv4_q8_hc_expand_count.fetch_add(1);
    return true;
}

int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_F16);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;
    const bool output_f16 = op->op == GGML_OP_MUL_MAT_F16;

    ggml_tensor * dsv4_compressor_pair_peer = nullptr;
    const bool can_dsv4_compressor_pair =
            ggml_metal_mul_mat_get_dsv4_compressor_pair_peer(ctx, idx, op, &dsv4_compressor_pair_peer);
    if (!output_f16 &&
            can_dsv4_compressor_pair &&
            ggml_metal_encode_mul_mat_dsv4_compressor_pair(ctx, op, dsv4_compressor_pair_peer)) {
        return 2;
    }

    ggml_tensor * dsv4_q8_hc_expand_next = nullptr;
    if (ggml_metal_dsv4_is_aohc_fused_high(op)) {
        g_ggml_metal_dsv4_aohc_elig_candidate_count.fetch_add(1);
        if (ggml_metal_dsv4_trace_aohc_fused_elig_enabled() &&
                g_ggml_metal_dsv4_q8_hc_expand_trace_count.load() < 64) {
            const ggml_tensor * next = idx + 1 < ctx->n_nodes() ? ctx->node(idx + 1) : nullptr;
            GGML_LOG_INFO("dsv4_aohc_elig: candidate_mode=aohc_fused_partial_q8hc backend=metal idx=%d name=%s op=%s type=%s n_tokens=%lld attn_out_input=%s attn_out_input_op=%s attn_out_input_type=%s wo_b=%s wo_b_type=%s high_shape=[%lld,%lld,%lld,%lld] input_shape=[%lld,%lld,%lld,%lld] wo_b_shape=[%lld,%lld,%lld,%lld] next_op=%s next_name=%s next_src0_is_high=%d\n",
                    idx, op->name, ggml_op_name(op->op), ggml_type_name(op->type),
                    (long long) op->ne[1],
                    op->src[1] != nullptr ? op->src[1]->name : "null",
                    op->src[1] != nullptr ? ggml_op_name(op->src[1]->op) : "null",
                    op->src[1] != nullptr ? ggml_type_name(op->src[1]->type) : "null",
                    op->src[0] != nullptr ? op->src[0]->name : "null",
                    op->src[0] != nullptr ? ggml_type_name(op->src[0]->type) : "null",
                    (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3],
                    op->src[1] != nullptr ? (long long) op->src[1]->ne[0] : -1LL,
                    op->src[1] != nullptr ? (long long) op->src[1]->ne[1] : -1LL,
                    op->src[1] != nullptr ? (long long) op->src[1]->ne[2] : -1LL,
                    op->src[1] != nullptr ? (long long) op->src[1]->ne[3] : -1LL,
                    op->src[0] != nullptr ? (long long) op->src[0]->ne[0] : -1LL,
                    op->src[0] != nullptr ? (long long) op->src[0]->ne[1] : -1LL,
                    op->src[0] != nullptr ? (long long) op->src[0]->ne[2] : -1LL,
                    op->src[0] != nullptr ? (long long) op->src[0]->ne[3] : -1LL,
                    next != nullptr ? ggml_op_name(next->op) : "null",
                    next != nullptr ? next->name : "null",
                    next != nullptr && next->src[0] == op ? 1 : 0);
        }
    }
    if (!output_f16 &&
            ggml_metal_mul_mat_get_dsv4_q8_hc_expand(ctx, idx, op, &dsv4_q8_hc_expand_next) &&
            ggml_metal_encode_mul_mat_dsv4_q8_hc_expand(ctx, op, dsv4_q8_hc_expand_next)) {
        return 2;
    }

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const int ne11_mm_min = 8;

    ggml_metal_mul_mat_shared_swiglu_plan shared_swiglu_plan = {};
    if (!output_f16 &&
            ggml_metal_mul_mat_get_shared_swiglu_plan(ctx, idx, op, &shared_swiglu_plan)) {
        if (shared_swiglu_plan.defer_to_glu) {
            ctx->skip_node(shared_swiglu_plan.peer_local_idx);
            ctx->defer_shared_swiglu(shared_swiglu_plan.glu_local_idx);
            return shared_swiglu_plan.n_fuse_contiguous;
        }

        ggml_metal_prepare_fused_concurrency(ctx, {
            shared_swiglu_plan.peer,
            shared_swiglu_plan.glu,
        });

        if (ggml_metal_encode_mul_mat_shared_swiglu(ctx, op, shared_swiglu_plan.peer, shared_swiglu_plan.glu)) {
            g_ggml_metal_mul_mat_shared_swiglu_count.fetch_add(1);
            ctx->skip_node(shared_swiglu_plan.peer_local_idx);
            ctx->skip_node(shared_swiglu_plan.glu_local_idx);
            if (!ggml_metal_op_concurrency_add(ctx, shared_swiglu_plan.glu)) {
                ggml_metal_op_concurrency_reset(ctx);
            }
            return shared_swiglu_plan.n_fuse_contiguous;
        }
    }

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    if (!output_f16 &&
        op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_BF16 ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_F8_E4M3_B128 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           op->src[0]->type == GGML_TYPE_Q2_K ||
           op->src[0]->type == GGML_TYPE_Q3_K ||
           false) && (ne11 >= 4 && ne11 <= 8)
         )
       )
       ) {
        g_ggml_metal_mul_mat_mv_ext_count.fetch_add(1);

        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const int nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        const int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be something cleverer
        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                GGML_ABORT("unsupported ne11");
        };

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op->src[0]->type, op->src[1]->type, nsg, nxpsg, r1ptg);

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } else if (
        !ggml_is_transposed(op->src[0]) &&
        !ggml_is_transposed(op->src[1]) &&
        // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
        // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
        props_dev->has_simdgroup_mm && ne00 >= 64 && ne11 > ne11_mm_min &&
        !ggml_metal_experimental_disable_mul_mm_enabled()) {
        g_ggml_metal_mul_mat_mm_count.fetch_add(1);
        const bool use_m5_expert_pipeline = ggml_metal_mul_mat_use_m5_expert_pipeline(props_dev, op);
        if (use_m5_expert_pipeline) {
            g_ggml_metal_mul_mat_mm_m5_expert_count.fetch_add(1);
        }
        const bool use_m5_sgmatrix_pipeline = ggml_metal_mul_mat_use_m5_sgmatrix_pipeline(props_dev);
        if (use_m5_sgmatrix_pipeline) {
            g_ggml_metal_mul_mat_mm_m5_sgmatrix_count.fetch_add(1);
        }

        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op, use_m5_expert_pipeline, use_m5_sgmatrix_pipeline);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        const size_t smem = pipeline.smem;
        const int32_t tiles_x = (ne11 + 31)/32;
        const int32_t tiles_y = (ne01 + 63)/64;
        const int16_t walk_mode = ggml_metal_mul_mm_env_walk_mode();

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        if (walk_mode == GGML_METAL_MUL_MM_WALK_LEGACY) {
            ggml_metal_encoder_dispatch_threadgroups(enc, tiles_x, tiles_y, ne12*ne13, 128, 1, 1);
        } else if (walk_mode == GGML_METAL_MUL_MM_WALK_REGULAR) {
            ggml_metal_encoder_dispatch_threadgroups(enc, tiles_x * tiles_y, 1, ne12*ne13, 128, 1, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, ggml_metal_mul_mm_dispatch_extent_morton(tiles_x, tiles_y), 1, ne12*ne13, 128, 1, 1);
        }
    } else {
        GGML_ASSERT(!output_f16 && "GGML_OP_MUL_MAT_F16 requires the matrix-matrix Metal path");

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;
        g_ggml_metal_mul_mat_mv_count.fetch_add(1);

        ggml_metal_kargs_mul_mv args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nr0  =*/ nr0,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
            /*.src0_byte_off =*/ 0,
            /*.src1_byte_off =*/ 0,
            /*.dst_byte_off  =*/ 0,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        }
    }

    return 1;
}

int ggml_metal_op_mul_mat_f16(ggml_metal_op_t ctx, int idx) {
    GGML_ASSERT(ctx->node(idx)->op == GGML_OP_MUL_MAT_F16);
    return ggml_metal_op_mul_mat(ctx, idx);
}

size_t ggml_metal_op_flashmoe_split_glu_extra_tmp(const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_FLASHMOE_SPLIT_GLU);
    GGML_ASSERT(op->src[2] != nullptr);
    GGML_ASSERT(op->src[3] != nullptr);

    const int64_t n_ff = op->src[2]->ne[0];
    const int64_t n_tokens = op->src[3]->ne[1];

    const size_t matrix_bytes = GGML_PAD(size_t(n_ff) * size_t(n_tokens) * sizeof(float), TENSOR_ALIGNMENT);
    return 3 * matrix_bytes;
}

int ggml_metal_op_flashmoe_split_glu(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    GGML_ASSERT(op->op == GGML_OP_FLASHMOE_SPLIT_GLU);
    GGML_ASSERT(op->src[0] != nullptr);
    GGML_ASSERT(op->src[1] != nullptr);
    GGML_ASSERT(op->src[2] != nullptr);
    GGML_ASSERT(op->src[3] != nullptr);

    GGML_TENSOR_LOCALS( int32_t, gate_ne, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, gate_nb, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, up_ne,   op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, up_nb,   op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, down_ne, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, down_nb, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, in_ne,   op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, in_nb,   op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, out_ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, out_nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    GGML_ASSERT(ggml_are_same_stride(op->src[0], op->src[1]));
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[3]->type == GGML_TYPE_F32);
    GGML_ASSERT(gate_ne0 == in_ne0);
    GGML_ASSERT(up_ne0   == in_ne0);
    GGML_ASSERT(gate_ne1 == up_ne1);
    GGML_ASSERT(down_ne0 == gate_ne1);
    GGML_ASSERT(out_ne0  == down_ne1);
    GGML_ASSERT(out_ne1  == in_ne1);
    GGML_ASSERT(in_ne2 == 1 && in_ne3 == 1);
    GGML_ASSERT(out_ne2 == 1 && out_ne3 == 1);

    const size_t matrix_bytes = GGML_PAD(size_t(gate_ne1) * size_t(in_ne1) * sizeof(float), TENSOR_ALIGNMENT);
    char * scratch_base = static_cast<char *>(op->data) + ggml_nbytes(op);

    ggml_tensor gate_tmp = ggml_metal_make_buffer_tensor_2d(op->buffer, GGML_TYPE_F32, gate_ne1, in_ne1, scratch_base, "split_glu_gate_ref");
    ggml_tensor up_tmp   = ggml_metal_make_buffer_tensor_2d(op->buffer, GGML_TYPE_F32, gate_ne1, in_ne1, scratch_base + matrix_bytes, "split_glu_up_ref");
    ggml_tensor act_tmp  = ggml_metal_make_buffer_tensor_2d(op->buffer, GGML_TYPE_F32, gate_ne1, in_ne1, scratch_base + 2*matrix_bytes, "split_glu_act_ref");

    ggml_tensor act_op = act_tmp;
    act_op.op = GGML_OP_GLU;
    act_op.src[0] = &gate_tmp;
    act_op.src[1] = &up_tmp;
    ggml_set_op_params_i32(&act_op, 0, ggml_get_op_params_i32(op, 0));
    ggml_set_op_params_i32(&act_op, 1, 0);

    ggml_metal_encode_mul_mat_from_tensors(ctx, op->src[0], op->src[3], &gate_tmp, false);
    ggml_metal_encode_mul_mat_from_tensors(ctx, op->src[1], op->src[3], &up_tmp,   false);
    ggml_metal_op_concurrency_reset(ctx);

    ggml_metal_encode_glu_from_sources(ctx, &act_op, &gate_tmp, &up_tmp);
    ggml_metal_op_concurrency_reset(ctx);

    ggml_metal_encode_mul_mat_from_tensors(ctx, op->src[2], &act_tmp, op, false);

    return 1;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert

    return ggml_type_size(GGML_TYPE_I32)*ne02;
}

size_t ggml_metal_op_mul_mat_id_extra_ids(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    return ggml_type_size(GGML_TYPE_I32)*ne02*ne21;
}

static bool ggml_metal_mul_mat_id_is_dsv4_attn_out_low(const ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT_ID ||
            op->src[0] == nullptr || op->src[1] == nullptr || op->src[2] == nullptr) {
        return false;
    }
    if (strstr(op->name, "dsv4_attn_out_low") == nullptr) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 ||
            op->src[0]->type != GGML_TYPE_Q8_0 ||
            op->src[1]->type != GGML_TYPE_F32 ||
            op->src[2]->type != GGML_TYPE_I32) {
        return false;
    }

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(int64_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(int64_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(int64_t, ne,  op,         ne);

    return ne03 == 1 &&
            ne13 == 1 &&
            ne20 == ne02 &&
            ne11 == ne02 &&
            ne12 == ne21 &&
            ne00 == ne10 &&
            ne0  == ne01 &&
            ne1  == ne20;
}

int ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    if (ggml_metal_dsv4_experimental_attn_out_low_enabled() &&
            ggml_metal_mul_mat_id_is_dsv4_attn_out_low(op)) {
        auto pipeline = ggml_metal_library_get_pipeline_dsv4_attn_out_low(lib, op);

        const int nr0 = pipeline.nr0;
        const int nsg = pipeline.nsg;
        const size_t smem = pipeline.smem;

        GGML_ASSERT(ne00 >= nsg*nr0);
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/nr0, 1, ne20*ne21, 32, nsg, 1);

        g_ggml_metal_dsv4_attn_out_low_count.fetch_add(1);
        return 1;
    }

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    // ne20 = n_used_experts
    // ne21 = n_rows (batch size)
    const int ne21_mm_id_min = 32;

    // Slot-bank decode fast path: for a single token, the routed op is just K plain mat-vec
    // dispatches against selected expert slices. This keeps the hot decode case off the generic
    // mul_mv_id path while still reading dynamic expert ids from the ids tensor.
    constexpr int64_t ne20_decode_mv_max = 32;
    if (!ggml_metal_mul_mat_id_disable_decode_fast_path_for_op(op) &&
        ggml_metal_mul_mat_id_ids_are_decode_ready(op->src[2]) &&
        ne21 == 1 && ne20 > 0 && ne20 <= ne20_decode_mv_max) {
        std::array<int32_t, ne20_decode_mv_max> expert_ids = {};
        if (ggml_metal_mul_mat_id_get_decode_expert_ids(op->src[2], expert_ids.data(), ne20, 0)) {
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "decode-fast-path");
            ggml_metal_mul_mat_id_down_sum6_plan down_sum6_plan = {};
            if (ggml_metal_mul_mat_id_get_down_sum6_plan(ctx, idx, op, &down_sum6_plan)) {
                if (down_sum6_plan.weighted) {
                    ggml_metal_op_concurrency_reset(ctx);
                }
                const bool encoded_down_sum6 =
                        down_sum6_plan.weighted ?
                            ggml_metal_encode_mul_mat_id_decode_mv_sum6_weighted(
                                    ctx, op, down_sum6_plan.weighted_act, down_sum6_plan.weights, down_sum6_plan.dst) :
                            ggml_metal_encode_mul_mat_id_decode_mv_sum6(ctx, op, down_sum6_plan.dst);
                if (encoded_down_sum6) {
                g_ggml_metal_mul_mat_id_down_sum6_count.fetch_add(1);
                if (down_sum6_plan.weighted) {
                    g_ggml_metal_mul_mat_id_down_sum6_weighted_count.fetch_add(1);
                }
                for (int i = 0; i < 5; ++i) {
                    ctx->skip_node(down_sum6_plan.add_local_idxs[i]);
                }
                if (!ggml_metal_op_concurrency_add(ctx, down_sum6_plan.dst)) {
                    ggml_metal_op_concurrency_reset(ctx);
                }
                return down_sum6_plan.n_fuse_contiguous;
                }
            }

            ggml_metal_mul_mat_id_pair_gate_up_plan pair_gate_up_plan = {};
            ggml_metal_mul_mat_id_split_glu_fuse_plan split_glu_fuse_plan = {};
            ggml_metal_mul_mat_id_glu_fuse_plan glu_fuse_plan = {};
            const bool can_pair_gate_up = ggml_metal_mul_mat_id_get_pair_gate_up_plan(ctx, idx, op, &pair_gate_up_plan);
            const bool allow_encode_fusion = ggml_metal_mul_mat_id_experimental_split_glu_encode_enabled();
            const bool can_fuse_split_glu = allow_encode_fusion && ggml_metal_mul_mat_id_get_split_glu_fuse_plan(ctx, idx, op, &split_glu_fuse_plan);
            const bool can_fuse_glu = allow_encode_fusion && ggml_metal_mul_mat_id_get_glu_fuse_plan(ctx, idx, op, &glu_fuse_plan);

            if (can_pair_gate_up && can_fuse_split_glu && pair_gate_up_plan.peer == split_glu_fuse_plan.peer) {
                ggml_metal_prepare_fused_concurrency(ctx, {
                    pair_gate_up_plan.peer,
                    split_glu_fuse_plan.glu,
                });
            } else if (can_pair_gate_up) {
                ggml_metal_prepare_fused_concurrency(ctx, {
                    pair_gate_up_plan.peer,
                });
            } else if (can_fuse_split_glu) {
                ggml_metal_prepare_fused_concurrency(ctx, {
                    split_glu_fuse_plan.peer,
                    split_glu_fuse_plan.glu,
                });
            } else if (can_fuse_glu) {
                ggml_metal_prepare_fused_concurrency(ctx, {
                    glu_fuse_plan.scale,
                    glu_fuse_plan.glu,
                });
            }

            if (can_fuse_split_glu && split_glu_fuse_plan.limited_swiglu &&
                    split_glu_fuse_plan.weighted_swiglu) {
                ctx->skip_node(split_glu_fuse_plan.peer_local_idx);
                ctx->skip_node(split_glu_fuse_plan.gate_clamp_local_idx);
                ctx->skip_node(split_glu_fuse_plan.up_clamp_local_idx);
                ctx->skip_node(split_glu_fuse_plan.silu_local_idx);
                ctx->skip_node(split_glu_fuse_plan.glu_local_idx);
                ctx->defer_weighted_swiglu(split_glu_fuse_plan.weighted_local_idx);
                return split_glu_fuse_plan.n_fuse_contiguous;
            }

            bool encoded_pair_limited_swiglu = false;
            if (can_fuse_split_glu && split_glu_fuse_plan.limited_swiglu &&
                    ggml_metal_mul_mat_id_experimental_dsv4_pair_swiglu_enabled()) {
                const float limit = ggml_get_op_params_f32(split_glu_fuse_plan.gate_clamp, 1);
                encoded_pair_limited_swiglu =
                        ggml_metal_encode_mul_mat_id_decode_mv_pair_limited_swiglu(
                                ctx,
                                op,
                                split_glu_fuse_plan.peer,
                                split_glu_fuse_plan.glu,
                                expert_ids.data(),
                                ne20,
                                limit);
            }

            const bool encoded_pair_gate_up =
                    !encoded_pair_limited_swiglu &&
                    can_pair_gate_up &&
                    ggml_metal_encode_mul_mat_id_decode_mv_pair(ctx, op, pair_gate_up_plan.peer, expert_ids.data(), ne20);

            if (encoded_pair_limited_swiglu) {
                g_ggml_metal_mul_mat_id_pair_swiglu_count.fetch_add(1);
                g_ggml_metal_mul_mat_id_fused_glu_count.fetch_add(1);
                ctx->skip_node(split_glu_fuse_plan.peer_local_idx);
                ctx->skip_node(split_glu_fuse_plan.gate_clamp_local_idx);
                ctx->skip_node(split_glu_fuse_plan.up_clamp_local_idx);
                ctx->skip_node(split_glu_fuse_plan.silu_local_idx);
                ctx->skip_node(split_glu_fuse_plan.glu_local_idx);
                if (!ggml_metal_op_concurrency_add(ctx, split_glu_fuse_plan.glu)) {
                    ggml_metal_op_concurrency_reset(ctx);
                }
                return split_glu_fuse_plan.n_fuse_contiguous;
            }

            if (encoded_pair_gate_up) {
                g_ggml_metal_mul_mat_id_pair_gate_up_count.fetch_add(1);
                ctx->skip_node(pair_gate_up_plan.peer_local_idx);
            } else {
                g_ggml_metal_mul_mat_id_decode_mv_count.fetch_add(1);
                const bool allow_replay =
                        ggml_metal_mul_mat_id_experimental_decode_replay_enabled() &&
                        !can_fuse_split_glu &&
                        !can_fuse_glu;
                if (allow_replay) {
                    const bool allow_icb =
                            ggml_metal_mul_mat_id_experimental_decode_icb_enabled() &&
                            ctx->use_concurrency;
                    std::shared_ptr<const ggml_metal_mul_mat_id_decode_replay_entry> replay_entry;
                    const bool replay_hit =
                            ggml_metal_mul_mat_id_decode_replay_lookup(ctx->dev, ctx->lib, op, expert_ids.data(), ne20, allow_icb, replay_entry);
                    if (replay_hit) {
                        if (allow_icb && ggml_metal_encode_mul_mat_id_decode_mv_icb(ctx, op, *replay_entry)) {
                            g_ggml_metal_mul_mat_id_decode_icb_exec_count.fetch_add(1);
                        } else {
                            ggml_metal_encode_mul_mat_id_decode_mv_replay(ctx, op, *replay_entry);
                        }
                    } else {
                        ggml_metal_encode_mul_mat_id_decode_mv(ctx, op, expert_ids.data(), ne20);
                    }
                } else {
                    ggml_metal_encode_mul_mat_id_decode_mv(ctx, op, expert_ids.data(), ne20);
                }
            }

            if (can_fuse_split_glu) {
                if (!encoded_pair_gate_up) {
                    ggml_metal_encode_mul_mat_id_decode_mv(ctx, split_glu_fuse_plan.peer, expert_ids.data(), ne20);
                }
                ggml_metal_op_concurrency_reset(ctx);
                if (split_glu_fuse_plan.limited_swiglu) {
                    const float limit = ggml_get_op_params_f32(split_glu_fuse_plan.gate_clamp, 1);
                    ggml_metal_encode_limited_swiglu_from_sources(
                            ctx,
                            split_glu_fuse_plan.glu,
                            op,
                            split_glu_fuse_plan.peer,
                            limit);
                    ctx->skip_node(split_glu_fuse_plan.gate_clamp_local_idx);
                    ctx->skip_node(split_glu_fuse_plan.up_clamp_local_idx);
                    ctx->skip_node(split_glu_fuse_plan.silu_local_idx);
                    if (split_glu_fuse_plan.limited_local_idx >= 0 &&
                        split_glu_fuse_plan.limited_local_idx != split_glu_fuse_plan.glu_local_idx) {
                        ctx->skip_node(split_glu_fuse_plan.limited_local_idx);
                    }
                } else {
                    ggml_metal_op_glu(ctx, split_glu_fuse_plan.glu_local_idx);
                }
                if (!encoded_pair_gate_up) {
                    ctx->skip_node(split_glu_fuse_plan.peer_local_idx);
                }
                ctx->skip_node(split_glu_fuse_plan.glu_local_idx);
                if (!ggml_metal_op_concurrency_add(ctx, split_glu_fuse_plan.glu)) {
                    ggml_metal_op_concurrency_reset(ctx);
                }
                g_ggml_metal_mul_mat_id_fused_glu_count.fetch_add(1);
                return split_glu_fuse_plan.n_fuse_contiguous;
            }

            if (encoded_pair_gate_up) {
                return 2;
            }

            if (can_fuse_glu) {
                if (glu_fuse_plan.scale) {
                    if (glu_fuse_plan.n_fuse == 4) {
                        ggml_metal_op_get_rows(ctx, idx + 1);
                    }
                    ggml_metal_encode_swiglu_scaled_from_merged(ctx, glu_fuse_plan.glu, op, glu_fuse_plan.scale);
                } else {
                    ggml_metal_encode_glu_from_sources(ctx, glu_fuse_plan.glu, op, nullptr);
                }

                g_ggml_metal_mul_mat_id_fused_glu_count.fetch_add(1);
                return glu_fuse_plan.n_fuse;
            }

            return 1;
        }
    }

    if (props_dev->has_simdgroup_mm && ne00 >= 64 && (ne21 >= ne21_mm_id_min) &&
        !ggml_metal_experimental_disable_mul_mm_id_enabled()) {
        g_ggml_metal_mul_mat_id_generic_mm_count.fetch_add(1);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne02,
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(ne02 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        // this barrier is always needed because the next kernel has to wait for the id maps to be computed
        ggml_metal_op_concurrency_reset(ctx);

        {
            const bool use_m5_sgmatrix_pipeline = ggml_metal_mul_mat_use_m5_sgmatrix_pipeline(props_dev);
            if (use_m5_sgmatrix_pipeline) {
                g_ggml_metal_mul_mat_mm_m5_sgmatrix_count.fetch_add(1);
            }
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op, use_m5_sgmatrix_pipeline);

            ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne21  =*/ ne21, // n_tokens
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);

            const size_t smem = pipeline.smem;

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne21 + 31)/32, (ne01 + 63)/64, ne02, 128, 1, 1);
        }
    } else {
        g_ggml_metal_mul_mat_id_generic_mv_count.fetch_add(1);

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_buffer_id bid_src2_mv = bid_src2;
        uint64_t nb21_mv = nb21;
        ggml_metal_mul_mat_id_materialize_ids_if_needed(op, bid_dst, bid_src2_mv, nb21_mv);

        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21_mv,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
        };

        if (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer(enc, bid_src2_mv, 4);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
    }

    return 1;
}

int ggml_metal_op_add_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_ADD_ID);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    return 1;
}

bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) && (ne00 % 32 == 0);
}

size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);

    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (false) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11*ne12*ne13 +
                nb21*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } else {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11*ne12*ne13 +
                nb21*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_blk(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    if (!has_mask) {
        return res;
    }

    const bool is_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    const int nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPSG : OP_FLASH_ATTN_EXT_NQPSG;
    const int ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    const int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    const int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += GGML_PAD(ggml_type_size(GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (true) {
        const int64_t nwg = 32;
        const int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += ggml_type_size(GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));
    }

    const int32_t nonvec_nwg = ggml_metal_flash_attn_nonvec_nwg(op);
    if (nonvec_nwg > 1) {
        const int64_t nrows = (int64_t) op->ne[1]*op->ne[2]*op->ne[3];
        const size_t res_nonvec = ggml_type_size(GGML_TYPE_F32)*(nrows*nonvec_nwg*(ne20 + 2));
        res = std::max(res, res_nonvec);
    }

    return res;
}

int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16);
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= op->src[0]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask to be at least n_queries big");
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[0] >= op->src[1]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask KV width to be at least the KV cache width");

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale,         ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;
    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(ne01 < 65536);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_src3 = has_mask  ? ggml_metal_get_buffer_id(op->src[3]) : bid_src0;
    ggml_metal_buffer_id bid_src4 = has_sinks ? ggml_metal_get_buffer_id(op->src[4]) : bid_src0;

    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    if (!ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;
        const bool use_metal4_qk = ggml_metal_flash_attn_nonvec_use_metal4(props_dev, op);

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11,
                /*.nb12    =*/nb12,
                /*.nb13    =*/nb13,
                /*.nb21    =*/nb21,
                /*.nb22    =*/nb22,
                /*.nb23    =*/nb23,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (has_mask) {
            assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        const int is_q = ggml_is_quantized(op->src[1]->type) ? 1 : 0;
        const int needs_k_scratch = (is_q || use_metal4_qk) ? 1 : 0;
        const int32_t walk_mode = ggml_metal_flash_attn_nonvec_env_walk_mode();

        // 2*(2*ncpsg)
        // ncpsg soft_max values + ncpsg mask values
        //
        // 16*32*(nsg)
        // the shared memory needed for the simdgroups to load the KV cache
        // each thread loads (dequantizes) 16 head elements, there are 32 threads in th SG
        //
#define FATTN_SMEM(nsg) (GGML_PAD((nqptg*(ne00 + 2*GGML_PAD(ne20, 64) + 2*(2*ncpsg)) + needs_k_scratch*(16*32*(nsg)))*(sizeof(float)/2), 16))

        //int64_t nsgmax = 4;
        //
        //if (is_q) {
        //    nsgmax = 2;
        //    while (true) {
        //        const size_t smem = FATTN_SMEM(nsgmax);
        //        if (smem > props_dev->max_theadgroup_memory_size) {
        //            break;
        //        }
        //        nsgmax *= 2;
        //    }
        //    nsgmax /= 2;
        //}

        // simdgroups per threadgroup (a.k.a. warps)
        //nsg = ne01 <= nqptg ? MAX(4, MIN(nsgmax, MIN(ne11/ncpsg, (int64_t) pipeline.maxTotalThreadsPerThreadgroup/32))) : 4;
        int32_t nsg = ne00 >= 512 ? 8 : 4;
        const int env_nsg = ggml_metal_flash_attn_nonvec_nsg_env_override();
        if (env_nsg > 0) {
            nsg = env_nsg;
        }

        const size_t smem = FATTN_SMEM(nsg);
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        const int32_t nwg = ggml_metal_flash_attn_nonvec_nwg(op);
        const int32_t chunk_mode = ggml_metal_flash_attn_nonvec_chunk_mode(op, nwg);

        ggml_metal_flash_attn_log_path_once(op, false, use_metal4_qk, has_mask, has_bias, has_scap, has_kvpad, nsg, walk_mode, chunk_mode, nwg);
        ggml_metal_flash_attn_log_mem_once(op, false, use_metal4_qk, has_mask, has_kvpad, nsg, chunk_mode, nwg);

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, use_metal4_qk, nsg, walk_mode, chunk_mode, nwg);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
        ggml_metal_encoder_set_buffer  (enc, nwg > 1 ? bid_tmp : bid_dst,  8);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        const int32_t q_blocks = (ne01 + nqptg - 1)/nqptg;
        if (walk_mode == GGML_METAL_MUL_MM_WALK_REGULAR) {
            ggml_metal_encoder_dispatch_threadgroups(enc, q_blocks*ne02, 1, ne03*nwg, 32, nsg, 1);
        } else if (walk_mode == GGML_METAL_MUL_MM_WALK_MORTON) {
            ggml_metal_encoder_dispatch_threadgroups(enc, ggml_metal_mul_mm_dispatch_extent_morton(q_blocks, ne02), 1, ne03*nwg, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, q_blocks, ne02, ne03*nwg, 32, nsg, 1);
        }

        if (nwg > 1) {
            assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

            ggml_metal_op_concurrency_reset(ctx);

            const int32_t nrows = ne1*ne2*ne3;

            ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                nrows,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
        }
#undef FATTN_SMEM
    } else {
        // half4x4 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_VEC_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        const int nhptg = 1;                           // heads per threadgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 1  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11 % ncpsg != 0;

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11,
                /*.nb12    =*/nb12,
                /*.nb13    =*/nb13,
                /*.nb21    =*/nb21,
                /*.nb22    =*/nb22,
                /*.nb23    =*/nb23,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        GGML_ASSERT(ne10 >= ne20);

        // ne00 + 2*ncpsg*(nsg)
        // for each query, we load it as f16 in shared memory (ne00)
        // and store the soft_max values and the mask
        //
        // ne20*(nsg)
        // each simdgroup has a full f32 head vector in shared mem to accumulate results
        //
        const bool use_metal4_sdpa = ggml_metal_flash_attn_vec_use_metal4(props_dev, op);
#define FATTN_SMEM(nsg)    (GGML_PAD(((GGML_PAD(ne00, 128) + 4*ncpsg + 2*GGML_PAD(ne20, 128))*(nsg))*(sizeof(float)/2), 16))
#define FATTN_SMEM_M4(nsg) (GGML_PAD(( GGML_PAD(ne00, 128) + (nsg)*(ncpsg*ne10 + 4*ncpsg + 2*GGML_PAD(ne20, 128)) )*(sizeof(float)/2), 16))

        int64_t nsg = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        if (use_metal4_sdpa) {
            // The current Metal4 vec kernel has a simdgroup-local masked-block
            // early-continue before later threadgroup barriers. Keeping NSG=1
            // avoids divergent simdgroups in the same threadgroup around those
            // barriers. Do not tune this upward without refactoring the kernel
            // control flow to make the skip decision threadgroup-uniform.
            nwg = 32;
            nsg = 1;
        } else if (ne01 == 1 && ggml_metal_flash_attn_vec_decode_single_wg_env_enabled()) {
            // DSV4 decode has a small masked KV range; avoid the 32-way split
            // and reduction kernel while this experimental path is enabled.
            nwg = 1;
            nsg = 4;
        } else {
            nwg = 32;
            nsg = 1;
            const int override_nwg = ne01 == 1 ? ggml_metal_flash_attn_vec_decode_nwg_env_override() : -1;
            if (override_nwg > 0) {
                nwg = override_nwg;
            }
            while (2*nwg*nsg*ncpsg < ne11 && nsg < 4) {
                nsg *= 2;
            }
        }

        ggml_metal_flash_attn_log_path_once(op, true, use_metal4_sdpa, has_mask, has_bias, has_scap, has_kvpad, (int32_t) nsg, GGML_METAL_MUL_MM_WALK_LEGACY, GGML_METAL_FLASH_ATTN_CHUNK_STRIDED, nwg);
        ggml_metal_flash_attn_log_mem_once(op, true, use_metal4_sdpa, has_mask, has_kvpad, (int32_t) nsg, GGML_METAL_FLASH_ATTN_CHUNK_STRIDED, nwg);

        ggml_metal_kargs_flash_attn_ext_vec args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        auto pipeline = use_metal4_sdpa
                ? ggml_metal_library_get_pipeline_flash_attn_ext_vec_metal4(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, nwg)
                : ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, nwg);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);

        const size_t smem = use_metal4_sdpa ? FATTN_SMEM_M4(nsg) : FATTN_SMEM(nsg);

        //printf("smem: %zu, max: %zu, nsg = %d, nsgmax = %d\n", smem, props_dev->max_theadgroup_memory_size, (int) nsg, (int) nsgmax);
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        if (nwg == 1) {
            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);
        } else {
            // sanity checks
            assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

            GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            // write the results from each workgroup into a temp buffer
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);

            // sync the 2 kernels
            ggml_metal_op_concurrency_reset(ctx);

            // reduce the results from the workgroups
            {
                const int32_t nrows = ne1*ne2*ne3;

                ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                ggml_metal_encoder_set_pipeline(enc, pipeline0);
                ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

                ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
#undef FATTN_SMEM
#undef FATTN_SMEM_M4
    }

    return 1;
}

int ggml_metal_op_bin(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion;

    const int debug_fusion = ctx->debug_fusion;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    if (ctx->is_deferred_weighted_swiglu(idx) &&
            ggml_metal_mul_mat_id_experimental_dsv4_weighted_swiglu_enabled() &&
            ggml_metal_tensor_name_has_token(op, "ffn_moe_weighted_swiglu")) {
        ggml_tensor * limited = nullptr;
        ggml_tensor * weights = nullptr;
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * gate_clamp = nullptr;
        ggml_tensor * up_clamp = nullptr;
        ggml_tensor * silu = nullptr;
        float limit = 0.0f;

        const bool valid_sources =
                ggml_metal_mul_mat_id_get_weighted_swiglu_sources(op, &limited, &weights) &&
                ggml_metal_mul_mat_id_get_limited_swiglu_sources(limited, &gate, &up, &gate_clamp, &up_clamp, &silu, &limit) &&
                gate != nullptr && up != nullptr &&
                gate->op == GGML_OP_MUL_MAT_ID && up->op == GGML_OP_MUL_MAT_ID &&
                ggml_metal_tensor_is_split_gate(gate) && ggml_metal_tensor_is_split_up(up) &&
                gate->src[0] != nullptr && gate->src[1] != nullptr && gate->src[2] != nullptr &&
                up->src[0] != nullptr && up->src[1] != nullptr && up->src[2] != nullptr &&
                gate->src[1] == up->src[1] && gate->src[2] == up->src[2] &&
                gate->src[0]->type == up->src[0]->type &&
                !ggml_metal_tensor_overlaps(op, weights) &&
                ggml_are_same_shape(gate->src[0], up->src[0]) &&
                ggml_are_same_stride(gate->src[0], up->src[0]) &&
                ggml_are_same_shape(limited, op) &&
                ggml_are_same_stride(limited, op);

        constexpr int64_t ne20_decode_mv_max = 32;
        if (valid_sources &&
                ggml_metal_mul_mat_id_ids_are_decode_ready(gate->src[2]) &&
                gate->ne[1] > 0 && gate->ne[1] <= ne20_decode_mv_max &&
                gate->ne[2] == 1 && gate->ne[3] == 1) {
            std::array<int32_t, ne20_decode_mv_max> expert_ids = {};
            if (ggml_metal_mul_mat_id_get_decode_expert_ids(gate->src[2], expert_ids.data(), gate->ne[1], 0) &&
                    ggml_metal_encode_mul_mat_id_decode_mv_pair_limited_swiglu_weighted(
                            ctx, gate, up, op, weights, expert_ids.data(), gate->ne[1], limit)) {
                g_ggml_metal_mul_mat_id_pair_swiglu_count.fetch_add(1);
                g_ggml_metal_mul_mat_id_weighted_swiglu_count.fetch_add(1);
                g_ggml_metal_mul_mat_id_fused_glu_count.fetch_add(1);
                return 1;
            }
        }

        GGML_ABORT("deferred weighted SwiGLU fusion failed");
    }

    if (ggml_metal_mul_mat_id_experimental_dsv4_limited_swiglu_enabled() &&
            ggml_metal_tensor_name_has_token(op, "ffn_moe_weighted_swiglu") &&
            idx + 1 < ctx->n_nodes()) {
        ggml_tensor * next = ctx->node(idx + 1);
        ggml_metal_mul_mat_id_down_sum6_plan plan = {};
        if (next != nullptr && next->op == GGML_OP_MUL_MAT_ID &&
                ggml_metal_mul_mat_id_get_down_sum6_plan(ctx, idx + 1, next, &plan) &&
                plan.weighted) {
            return 1;
        }
    }

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    ggml_op fops[8];

    int n_fuse = 1;

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion) {
        fops[0] = GGML_OP_ADD;
        fops[1] = GGML_OP_ADD;
        fops[2] = GGML_OP_ADD;
        fops[3] = GGML_OP_ADD;
        fops[4] = GGML_OP_ADD;
        fops[5] = GGML_OP_ADD;
        fops[6] = GGML_OP_ADD;
        fops[7] = GGML_OP_ADD;

        // note: in metal, we sometimes encode the graph in parallel so we have to avoid fusing ops
        //       across splits. idx_end indicates the last node in the current split
        for (n_fuse = 0; n_fuse <= 6; ++n_fuse) {
            if (!ctx->can_fuse(idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            ggml_tensor * f0 = ctx->node(idx + n_fuse);
            ggml_tensor * f1 = ctx->node(idx + n_fuse + 1);

            if (f0 != f1->src[0]) {
                break;
            }

            // b[0] === b[1] === ...
            if (!ggml_are_same_layout(f0->src[1], f1->src[1])) {
                break;
            }

            // only fuse ops if src1 is in the same Metal buffer
            ggml_metal_buffer_id bid_fuse = ggml_metal_get_buffer_id(f1->src[1]);
            if (bid_fuse.metal != bid_src1.metal) {
                break;
            }

            //ctx->fuse_cnt[ops[n_fuse + 1]->op]++;

            args.o1[n_fuse + 1] = bid_fuse.offs;
        }

        ++n_fuse;

        if (debug_fusion > 1 && n_fuse > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    struct ggml_metal_pipeline_with_params pipeline;

    pipeline = ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne10 = ne10/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

    if (pipeline.cnt) {
        ggml_metal_encoder_dispatch_threadgroups(enc, args.ne0, ggml_nrows(op), 1, 1, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        int nth = 1;

        while (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return n_fuse;
}

int ggml_metal_op_l2_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_l2_norm args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.eps   =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_l2_norm(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_group_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];

    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));

    ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);

    int nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion;

    const int debug_fusion = ctx->debug_fusion;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_t =*/ ne00 % 4 == 0 ? ne00/4 : ne00,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
    };

    ggml_op fops[8];

    int n_fuse = 1;

    ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b)
    // d[2] = add(d[1], c)
    if (use_fusion) {
        fops[0] = op->op;
        fops[1] = GGML_OP_MUL;
        fops[2] = GGML_OP_ADD;

        for (n_fuse = 0; n_fuse <= 1; ++n_fuse) {
            if (!ctx->can_fuse(idx + n_fuse, fops + n_fuse, 2)) {
                break;
            }

            ggml_tensor * f0 = ctx->node(idx + n_fuse);
            ggml_tensor * f1 = ctx->node(idx + n_fuse + 1);

            if (f0 != f1->src[0]) {
                break;
            }

            if (f1->src[1]->ne[0] != op->ne[0]) {
                break;
            }

            if (!ggml_is_contiguous_rows(f1->src[1])) {
                break;
            }

            if (f1->type != GGML_TYPE_F32) {
                break;
            }

            //ctx->fuse_cnt[f1->op]++;

            bid_fuse[n_fuse] = ggml_metal_get_buffer_id(f1->src[1]);

            args.nef1[n_fuse + 1] = f1->src[1]->ne[1];
            args.nef2[n_fuse + 1] = f1->src[1]->ne[2];
            args.nef3[n_fuse + 1] = f1->src[1]->ne[3];

            args.nbf1[n_fuse + 1] = f1->src[1]->nb[1];
            args.nbf2[n_fuse + 1] = f1->src[1]->nb[2];
            args.nbf3[n_fuse + 1] = f1->src[1]->nb[3];
        }

        ++n_fuse;

        if (debug_fusion > 1 && n_fuse > 1) {
            if (n_fuse == 2) {
                GGML_LOG_DEBUG("%s: fuse: %s + MUL\n", __func__, ggml_op_name(op->op));
            }
            if (n_fuse == 3) {
                GGML_LOG_DEBUG("%s: fuse: %s + MUL + ADD\n", __func__, ggml_op_name(op->op));
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    int nth = 32; // SIMD width

    while (nth < args.ne00_t && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, args.ne00_t);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0,    1);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[0], 2);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[1], 3);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,     4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return n_fuse;
}

int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    const int nth = std::min(1024, ne00);

    const int n_past     = ((const int32_t *) op->op_params)[0];
    const int n_dims     = ((const int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const int n_ctx_orig = ((const int32_t *) op->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 10, sizeof(float));

    // mrope
    const int sect_0 = ((const int32_t *) op->op_params)[11];
    const int sect_1 = ((const int32_t *) op->op_params)[12];
    const int sect_2 = ((const int32_t *) op->op_params)[13];
    const int sect_3 = ((const int32_t *) op->op_params)[14];

    ggml_metal_kargs_rope args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
        /* src2        =*/ op->src[2] != nullptr,
    };

    auto pipeline = ggml_metal_library_get_pipeline_rope(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

static bool ggml_metal_dsv4_hc_split_can_fuse_weighted_sum(ggml_metal_op_t ctx, int idx, const ggml_tensor * op, const ggml_tensor * next);

struct ggml_metal_dsv4_hc_pre_norm_plan {
    ggml_tensor * sum  = nullptr;
    ggml_tensor * norm = nullptr;
    ggml_tensor * mul  = nullptr;
    int           n_fuse = 4;
};

static bool ggml_metal_dsv4_hc_split_can_fuse_weighted_sum_norm(
        ggml_metal_op_t ctx,
        int idx,
        const ggml_tensor * op,
        ggml_metal_dsv4_hc_pre_norm_plan * plan);

static bool ggml_metal_encode_dsv4_hc_split_weighted_sum_norm(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_metal_dsv4_hc_pre_norm_plan & plan);

int ggml_metal_op_dsv4_hc_split_sinkhorn(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_HC_SPLIT_SINKHORN);

    const int64_t n_rows = ggml_nrows(op->src[0]);

    ggml_metal_kargs_dsv4_hc_split_sinkhorn args = {
        /*.n_hc           =*/ ggml_get_op_params_i32(op, 0),
        /*.sinkhorn_iters =*/ ggml_get_op_params_i32(op, 1),
        /*.eps            =*/ ggml_get_op_params_f32(op, 2),
        /*.n_rows         =*/ n_rows,
        /*.mixes_nb1      =*/ op->src[0]->nb[1],
        /*.dst_nb1        =*/ op->nb[1],
    };

    ggml_metal_dsv4_hc_pre_norm_plan norm_plan = {};
    if (ggml_metal_dsv4_hc_split_can_fuse_weighted_sum_norm(ctx, idx, op, &norm_plan) &&
            ggml_metal_encode_dsv4_hc_split_weighted_sum_norm(ctx, op, norm_plan)) {
        return norm_plan.n_fuse;
    }

    if (idx + 1 < ctx->n_nodes()) {
        ggml_tensor * next = ctx->node(idx + 1);
        if (ggml_metal_dsv4_hc_split_can_fuse_weighted_sum(ctx, idx, op, next)) {
            ggml_metal_kargs_dsv4_hc_weighted_sum sum_args = {
                /*.n_embd   =*/ next->ne[0],
                /*.n_hc     =*/ next->src[0]->ne[1],
                /*.n_tokens =*/ next->ne[1],
                /*.n_elem   =*/ next->ne[0] * next->ne[1],
                /*.x_nb0    =*/ next->src[0]->nb[0],
                /*.x_nb1    =*/ next->src[0]->nb[1],
                /*.x_nb2    =*/ next->src[0]->nb[2],
                /*.w_nb0    =*/ next->src[1]->nb[0],
                /*.w_nb1    =*/ next->src[1]->nb[1],
                /*.dst_nb0  =*/ next->nb[0],
                /*.dst_nb1  =*/ next->nb[1],
            };

            auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc_split_sinkhorn_weighted_sum(lib, op);
            const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), std::min(256, (int) next->ne[0]));
            const int64_t n_chunks = (next->ne[0] + nth - 1) / nth;

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args,     sizeof(args),     0);
            ggml_metal_encoder_set_bytes   (enc, &sum_args, sizeof(sum_args), 1);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]),   2);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]),   3);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]),   4);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(next->src[0]), 5);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),           6);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(next),         7);
            ggml_metal_encoder_dispatch_threadgroups(enc, n_chunks, n_rows, 1, nth, 1, 1);

            g_ggml_metal_dsv4_hc_split_weighted_sum_count.fetch_add(1);
            return 2;
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc_split_sinkhorn(lib, op);
    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) n_rows);
    const int ntg = (n_rows + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_dsv4_hc_weighted_sum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_HC_WEIGHTED_SUM);

    const int64_t n_elem = op->ne[0] * op->ne[1];

    ggml_metal_kargs_dsv4_hc_weighted_sum args = {
        /*.n_embd   =*/ op->ne[0],
        /*.n_hc     =*/ op->src[0]->ne[1],
        /*.n_tokens =*/ op->ne[1],
        /*.n_elem   =*/ n_elem,
        /*.x_nb0    =*/ op->src[0]->nb[0],
        /*.x_nb1    =*/ op->src[0]->nb[1],
        /*.x_nb2    =*/ op->src[0]->nb[2],
        /*.w_nb0    =*/ op->src[1]->nb[0],
        /*.w_nb1    =*/ op->src[1]->nb[1],
        /*.dst_nb0  =*/ op->nb[0],
        /*.dst_nb1  =*/ op->nb[1],
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc_weighted_sum(lib, op);
    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) n_elem);
    const int ntg = (n_elem + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_dsv4_hc_expand(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_HC_EXPAND);

    const int64_t n_elem = op->ne[0] * op->ne[1] * op->ne[2];

    ggml_metal_kargs_dsv4_hc_expand args = {
        /*.n_embd    =*/ op->ne[0],
        /*.n_hc      =*/ op->ne[1],
        /*.n_tokens  =*/ op->ne[2],
        /*.n_elem    =*/ n_elem,
        /*.block_nb0 =*/ op->src[0]->nb[0],
        /*.block_nb1 =*/ op->src[0]->nb[1],
        /*.res_nb0   =*/ op->src[1]->nb[0],
        /*.res_nb1   =*/ op->src[1]->nb[1],
        /*.res_nb2   =*/ op->src[1]->nb[2],
        /*.post_nb0  =*/ op->src[2]->nb[0],
        /*.post_nb1  =*/ op->src[2]->nb[1],
        /*.comb_nb0  =*/ op->src[3]->nb[0],
        /*.comb_nb1  =*/ op->src[3]->nb[1],
        /*.comb_nb2  =*/ op->src[3]->nb[2],
        /*.dst_nb0   =*/ op->nb[0],
        /*.dst_nb1   =*/ op->nb[1],
        /*.dst_nb2   =*/ op->nb[2],
    };

    const bool use_expand4 =
            ggml_metal_dsv4_experimental_hc_expand4_enabled() &&
            op->type == GGML_TYPE_F32 &&
            op->src[0]->type == GGML_TYPE_F32 &&
            op->src[1]->type == GGML_TYPE_F32 &&
            op->src[2]->type == GGML_TYPE_F32 &&
            op->src[3]->type == GGML_TYPE_F32 &&
            op->ne[1] == 4;

    auto pipeline = use_expand4 ?
        ggml_metal_library_get_pipeline_dsv4_hc_expand4(lib, op) :
        ggml_metal_library_get_pipeline_dsv4_hc_expand(lib, op);
    const int64_t n_work = use_expand4 ? op->ne[0] * op->ne[2] : n_elem;
    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) n_work);
    const int ntg = (n_work + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         5);
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    if (use_expand4) {
        g_ggml_metal_dsv4_hc_expand4_count.fetch_add(1);
    }

    return 1;
}

static bool ggml_metal_dsv4_hc_split_can_fuse_weighted_sum(ggml_metal_op_t ctx, int idx, const ggml_tensor * op, const ggml_tensor * next) {
    if (!ggml_metal_dsv4_experimental_hc_split_weighted_sum_enabled()) {
        return false;
    }
    if (idx + 1 >= ctx->n_nodes() || next == nullptr || next->op != GGML_OP_DSV4_HC_WEIGHTED_SUM) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        next->type != GGML_TYPE_F32 || next->src[0]->type != GGML_TYPE_F32 || next->src[1]->type != GGML_TYPE_F32) {
        return false;
    }

    const int n_hc = ggml_get_op_params_i32(op, 0);
    const int64_t n_rows = ggml_nrows(op->src[0]);
    const ggml_tensor * weights = next->src[1];
    const ggml_tensor * weight_src = weights;
    if (weight_src != nullptr && weight_src->op == GGML_OP_CONT && weight_src->src[0] != nullptr) {
        weight_src = weight_src->src[0];
    }
    const bool weights_are_pre_view =
            weights == op ||
            weight_src == op ||
            (weight_src->view_src == op && weight_src->view_offs == 0);

    if (!weights_are_pre_view || n_hc <= 0 || n_hc > 16 || n_rows <= 0) {
        return false;
    }
    if (weights->ne[0] != n_hc || weights->ne[1] != n_rows || weights->ne[2] != 1 || weights->ne[3] != 1) {
        return false;
    }
    if (next->src[0]->ne[1] != n_hc || next->src[0]->ne[2] != n_rows || next->src[0]->ne[3] != 1) {
        return false;
    }
    if (next->ne[0] != next->src[0]->ne[0] || next->ne[1] != n_rows) {
        return false;
    }
    return true;
}

static bool ggml_metal_dsv4_hc_split_can_fuse_weighted_sum_norm(
        ggml_metal_op_t ctx,
        int idx,
        const ggml_tensor * op,
        ggml_metal_dsv4_hc_pre_norm_plan * plan) {
    auto reject = [&](const char * reason) -> bool {
        if (ggml_metal_dsv4_trace_hc_pre_norm_enabled() &&
                op != nullptr &&
                op->op == GGML_OP_DSV4_HC_SPLIT_SINKHORN &&
                g_ggml_metal_dsv4_hc_pre_norm_trace_count.fetch_add(1) < 512) {
            const ggml_tensor * n1 = idx + 1 < ctx->n_nodes() ? ctx->node(idx + 1) : nullptr;
            const ggml_tensor * n2 = idx + 2 < ctx->n_nodes() ? ctx->node(idx + 2) : nullptr;
            const ggml_tensor * n3 = idx + 3 < ctx->n_nodes() ? ctx->node(idx + 3) : nullptr;
            fprintf(stderr, "%s: reject=%s idx=%d op=%s n1=%s/%s n2=%s/%s n3=%s/%s\n",
                    __func__, reason, idx, op->name,
                    n1 ? ggml_op_name(n1->op) : "null", n1 ? n1->name : "null",
                    n2 ? ggml_op_name(n2->op) : "null", n2 ? n2->name : "null",
                    n3 ? ggml_op_name(n3->op) : "null", n3 ? n3->name : "null");
        }
        return false;
    };

    if (plan == nullptr || !ctx->use_fusion || !ggml_metal_dsv4_experimental_hc_pre_norm_enabled()) {
        return reject("disabled");
    }
    if (idx + 3 >= ctx->n_nodes()) {
        return reject("window");
    }

    int sum_local = idx + 1;
    ggml_op pre_op = GGML_OP_NONE;
    ggml_tensor * maybe_pre = ctx->node(sum_local);
    if (maybe_pre != nullptr &&
            (maybe_pre->op == GGML_OP_VIEW || maybe_pre->op == GGML_OP_CONT) &&
            (maybe_pre->src[0] == op || maybe_pre->view_src == op ||
             (maybe_pre->src[0] != nullptr && maybe_pre->src[0]->view_src == op))) {
        pre_op = maybe_pre->op;
        sum_local++;
    }
    if (sum_local + 2 >= ctx->n_nodes()) {
        return reject("window-view");
    }

    ggml_tensor * sum  = ctx->node(sum_local);
    ggml_tensor * norm = ctx->node(sum_local + 1);
    ggml_tensor * mul  = ctx->node(sum_local + 2);

    if (!ggml_metal_dsv4_hc_split_can_fuse_weighted_sum(ctx, idx, op, sum)) {
        return reject("base");
    }
    if (norm == nullptr || norm->op != GGML_OP_RMS_NORM || norm->src[0] != sum) {
        return reject("norm");
    }
    if (mul == nullptr || mul->op != GGML_OP_MUL || mul->src[0] != norm || mul->src[1] == nullptr) {
        return reject("mul");
    }
    if (op->src[1] == nullptr || op->src[2] == nullptr ||
            op->src[1]->type != GGML_TYPE_F32 || op->src[2]->type != GGML_TYPE_F32 ||
            sum->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32) {
        return reject("type");
    }

    const int n_hc = ggml_get_op_params_i32(op, 0);
    const int64_t n_rows = ggml_nrows(op->src[0]);
    if (n_hc != 4 || n_rows != 1 || sum->ne[1] != 1 || (sum->ne[0] % 4) != 0) {
        return reject("decode-shape");
    }
    if (ggml_metal_dsv4_compare_hc_pre_norm_enabled() &&
            std::strstr(sum->name, "dsv4_hcnorm_sum_fused-") == nullptr) {
        return reject("compare-ref");
    }
    if (sum->ne[0] != norm->ne[0] || sum->ne[1] != norm->ne[1] ||
            norm->ne[0] != mul->ne[0] || norm->ne[1] != mul->ne[1] ||
            mul->src[1]->ne[0] != sum->ne[0]) {
        return reject("shape");
    }
    const char * scope = ggml_metal_dsv4_hc_pre_norm_scope();
    if (scope[0] != '\0' && std::strcmp(scope, "0") != 0 && std::strcmp(scope, "all") != 0) {
        const bool is_attn = std::strstr(sum->name, "hc_attn_pre") != nullptr;
        const bool is_ffn  = std::strstr(sum->name, "hc_ffn_pre")  != nullptr;
        if ((std::strcmp(scope, "attn") == 0 && !is_attn) ||
                (std::strcmp(scope, "ffn") == 0 && !is_ffn)) {
            return reject("scope");
        }
    }
    if (sum->src[0]->nb[0] != sizeof(float) ||
            sum->nb[0] != sizeof(float) ||
            norm->nb[0] != sizeof(float) ||
            mul->nb[0] != sizeof(float) ||
            mul->src[1]->nb[0] != sizeof(float)) {
        return reject("stride");
    }
    if (!ggml_is_contiguous_rows(sum) || !ggml_is_contiguous_rows(norm) ||
            !ggml_is_contiguous_rows(mul) || !ggml_is_contiguous_rows(mul->src[1])) {
        return reject("layout");
    }

    if (pre_op != GGML_OP_NONE) {
        ggml_op ops[] = {
            GGML_OP_DSV4_HC_SPLIT_SINKHORN,
            pre_op,
            GGML_OP_DSV4_HC_WEIGHTED_SUM,
            GGML_OP_RMS_NORM,
            GGML_OP_MUL,
        };
        static const int outputs[] = { 0, 1, 2, 4 };
        if (!ctx->can_fuse_subgraph(idx, ops, 5, outputs, 4)) {
            return reject("subgraph-view");
        }
        plan->n_fuse = 5;
    } else {
        static const ggml_op ops[] = {
            GGML_OP_DSV4_HC_SPLIT_SINKHORN,
            GGML_OP_DSV4_HC_WEIGHTED_SUM,
            GGML_OP_RMS_NORM,
            GGML_OP_MUL,
        };
        static const int outputs[] = { 0, 1, 3 };
        if (!ctx->can_fuse_subgraph(idx, ops, 4, outputs, 3)) {
            return reject("subgraph");
        }
        plan->n_fuse = 4;
    }

    plan->sum = sum;
    plan->norm = norm;
    plan->mul = mul;
    return true;
}

static bool ggml_metal_encode_dsv4_hc_split_weighted_sum_norm(
        ggml_metal_op_t ctx,
        const ggml_tensor * op,
        const ggml_metal_dsv4_hc_pre_norm_plan & plan) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * sum  = plan.sum;
    const ggml_tensor * norm = plan.norm;
    const ggml_tensor * mul  = plan.mul;

    float norm_eps = 0.0f;
    memcpy(&norm_eps, norm->op_params, sizeof(float));

    ggml_metal_kargs_dsv4_hc_split_sinkhorn split_args = {
        /*.n_hc           =*/ ggml_get_op_params_i32(op, 0),
        /*.sinkhorn_iters =*/ ggml_get_op_params_i32(op, 1),
        /*.eps            =*/ ggml_get_op_params_f32(op, 2),
        /*.n_rows         =*/ ggml_nrows(op->src[0]),
        /*.mixes_nb1      =*/ op->src[0]->nb[1],
        /*.dst_nb1        =*/ op->nb[1],
    };

    ggml_metal_kargs_dsv4_hc_weighted_sum sum_args = {
        /*.n_embd   =*/ sum->ne[0],
        /*.n_hc     =*/ sum->src[0]->ne[1],
        /*.n_tokens =*/ sum->ne[1],
        /*.n_elem   =*/ sum->ne[0] * sum->ne[1],
        /*.x_nb0    =*/ sum->src[0]->nb[0],
        /*.x_nb1    =*/ sum->src[0]->nb[1],
        /*.x_nb2    =*/ sum->src[0]->nb[2],
        /*.w_nb0    =*/ sum->src[1]->nb[0],
        /*.w_nb1    =*/ sum->src[1]->nb[1],
        /*.dst_nb0  =*/ sum->nb[0],
        /*.dst_nb1  =*/ sum->nb[1],
    };

    ggml_metal_kargs_dsv4_hc_weighted_sum_norm norm_args = {
        /*.n_embd    =*/ sum->ne[0],
        /*.n_tokens  =*/ sum->ne[1],
        /*.sum_nb0   =*/ sum->nb[0],
        /*.sum_nb1   =*/ sum->nb[1],
        /*.norm_nb0  =*/ norm->nb[0],
        /*.norm_nb1  =*/ norm->nb[1],
        /*.mul_nb0   =*/ mul->nb[0],
        /*.mul_nb1   =*/ mul->nb[1],
        /*.weight_nb0=*/ mul->src[1]->nb[0],
        /*.norm_eps  =*/ norm_eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc_split_sinkhorn_weighted_sum_norm(lib, op);
    int nth = 32;
    const int64_t ne00_t = sum->ne[0] / 4;
    while (nth < ne00_t && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }
    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min<int>(nth, ne00_t);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &split_args, sizeof(split_args), 0);
    ggml_metal_encoder_set_bytes   (enc, &sum_args,   sizeof(sum_args),   1);
    ggml_metal_encoder_set_bytes   (enc, &norm_args,  sizeof(norm_args),  2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]),  3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]),  4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]),  5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(sum->src[0]), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mul->src[1]), 7);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),          8);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(sum),         9);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(norm),       10);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mul),        11);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, sum->ne[1], 1, 1, nth, 1, 1);

    if (ggml_metal_dsv4_trace_hc_pre_norm_enabled() &&
            g_ggml_metal_dsv4_hc_pre_norm_trace_count.fetch_add(1) < 512) {
        fprintf(stderr, "%s: fused split=%s sum=%s norm=%s mul=%s n_embd=%lld\n",
                __func__, op->name, sum->name, norm->name, mul->name, (long long) sum->ne[0]);
    }

    g_ggml_metal_dsv4_hc_split_weighted_sum_count.fetch_add(1);
    g_ggml_metal_dsv4_hc_pre_norm_count.fetch_add(1);
    return true;
}

static int ggml_metal_dsv4_count_future_consumers(ggml_metal_op_t ctx, int idx, const ggml_tensor * producer) {
    int consumers = 0;
    for (int i = idx + 1; i < ctx->n_nodes(); ++i) {
        const ggml_tensor * node = ctx->node(i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (node->src[s] == producer) {
                ++consumers;
            }
        }
    }
    return consumers;
}

static bool ggml_metal_dsv4_rope_tail_can_fuse_base(ggml_metal_op_t ctx, int idx, const ggml_tensor * op, const ggml_tensor * next) {
    if (idx + 1 >= ctx->n_nodes() || next == nullptr || next->src[0] != op) {
        return false;
    }
    if (ggml_metal_dsv4_count_future_consumers(ctx, idx, op) != 1) {
        return false;
    }
    if (op->src[0] == nullptr || op->src[1] == nullptr) {
        return false;
    }
    if (op->src[0]->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || next->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[0]->nb[0] != ggml_type_size(op->src[0]->type) || op->nb[0] != ggml_type_size(op->type) || next->nb[0] != ggml_type_size(next->type)) {
        return false;
    }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (next->src[0]->ne[d] != op->ne[d]) {
            return false;
        }
    }
    const int n_dims = ((const int32_t *) op->op_params)[0];
    if (n_dims <= 0 || n_dims > op->ne[0]) {
        return false;
    }
    return true;
}

static bool ggml_metal_dsv4_rope_tail_can_fuse_hadamard_fp4(ggml_metal_op_t ctx, int idx, const ggml_tensor * op, const ggml_tensor * next) {
    if (!ggml_metal_dsv4_experimental_rope_hadamard_fp4_enabled()) {
        return false;
    }
    if (next == nullptr || next->op != GGML_OP_DSV4_HADAMARD_FP4_QUANTIZE) {
        return false;
    }
    if (!ggml_metal_dsv4_rope_tail_can_fuse_base(ctx, idx, op, next)) {
        return false;
    }
    const int64_t head_dim = op->ne[0];
    return head_dim > 0 && head_dim <= 256 && (head_dim % 32) == 0 && (head_dim & (head_dim - 1)) == 0;
}

static bool ggml_metal_dsv4_rope_tail_can_fuse_fp8_kv(ggml_metal_op_t ctx, int idx, const ggml_tensor * op, const ggml_tensor * next) {
    if (!ggml_metal_dsv4_experimental_rope_fp8_kv_enabled()) {
        return false;
    }
    if (next == nullptr || next->op != GGML_OP_DSV4_FP8_KV_QUANTIZE) {
        return false;
    }
    if (!ggml_metal_dsv4_rope_tail_can_fuse_base(ctx, idx, op, next)) {
        return false;
    }
    const int n_dims = ((const int32_t *) op->op_params)[0];
    const int n_rot  = ggml_get_op_params_i32(next, 0);
    const int64_t n_nope = op->ne[0] - n_rot;
    const int64_t head_dim = op->ne[0];
    return n_rot == n_dims && n_nope >= 0 && (n_nope % 64) == 0 && head_dim <= 1024;
}

struct ggml_metal_dsv4_kv_set_rows_fuse_plan {
    ggml_tensor * set_rows = nullptr;
    int           n_fuse   = 0;
};

static bool ggml_metal_dsv4_rope_tail_can_fuse_fp8_kv_set_rows(
        ggml_metal_op_t ctx,
        int idx,
        const ggml_tensor * op,
        const ggml_tensor * next,
        ggml_metal_dsv4_kv_set_rows_fuse_plan * plan) {
    auto reject = [&](const char * reason) -> bool {
        static std::atomic<uint64_t> trace_count { 0 };
        const char * trace = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_SET_ROWS_TRACE");
        if ((ctx->debug_fusion > 0 || (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0)) &&
                trace_count.fetch_add(1) < 32) {
            const ggml_tensor * n2 = idx + 2 < ctx->n_nodes() ? ctx->node(idx + 2) : nullptr;
            const ggml_tensor * n3 = idx + 3 < ctx->n_nodes() ? ctx->node(idx + 3) : nullptr;
            fprintf(stderr, "%s: reject %s op=%s next=%s n2=%s/%s n3=%s/%s\n",
                    "ggml_metal_dsv4_rope_tail_can_fuse_fp8_kv_set_rows",
                    reason,
                    op != nullptr ? op->name : "(null)",
                    next != nullptr ? ggml_op_name(next->op) : "(null)",
                    n2 != nullptr ? ggml_op_name(n2->op) : "(null)",
                    n2 != nullptr ? n2->name : "(null)",
                    n3 != nullptr ? ggml_op_name(n3->op) : "(null)",
                    n3 != nullptr ? n3->name : "(null)");
        }
        return false;
    };

    if (plan == nullptr || !ggml_metal_dsv4_experimental_rope_fp8_kv_set_rows_enabled()) {
        return reject("disabled");
    }
    if (!ggml_metal_dsv4_rope_tail_can_fuse_fp8_kv(ctx, idx, op, next)) {
        return reject("base");
    }
    if (op->src[0]->type != GGML_TYPE_F32 || next->type != GGML_TYPE_F32) {
        return reject("type");
    }
    if (op->ne[1] != 1 || op->ne[3] != 1) {
        return reject("rope-shape");
    }

    const ggml_tensor * set_src = nullptr;
    ggml_tensor * set_rows = nullptr;
    bool has_view = false;

    if (idx + 2 < ctx->n_nodes()) {
        ggml_tensor * cand = ctx->node(idx + 2);
        if (cand->op == GGML_OP_SET_ROWS) {
            set_src = next;
            set_rows = cand;
        } else if (cand->op == GGML_OP_VIEW && cand->src[0] == next && idx + 3 < ctx->n_nodes()) {
            ggml_tensor * cand2 = ctx->node(idx + 3);
            if (cand2->op == GGML_OP_SET_ROWS) {
                set_src = cand;
                set_rows = cand2;
                has_view = true;
            }
        }
    }

    if (set_rows == nullptr || set_rows->src[0] != set_src || set_rows->src[1] == nullptr) {
        return reject("set-not-found");
    }
    if (set_rows->type != GGML_TYPE_F16 && set_rows->type != GGML_TYPE_F32) {
        return reject("set-type");
    }
    if (set_rows->src[1]->type != GGML_TYPE_I32 && set_rows->src[1]->type != GGML_TYPE_I64) {
        return reject("idx-type");
    }
    if (set_src->type != GGML_TYPE_F32 || set_src->ne[0] != op->ne[0] || set_src->ne[1] != op->ne[2]) {
        return reject("set-src-shape");
    }
    if (set_rows->ne[0] != set_src->ne[0] || set_rows->src[1]->ne[0] != set_src->ne[1]) {
        return reject("set-shape");
    }
    if (set_rows->src[1]->ne[1] != 1 || set_rows->src[1]->ne[2] != 1 || set_rows->src[1]->ne[3] != 1) {
        return reject("idx-shape");
    }

    if (has_view) {
        static const ggml_op ops[] = {
            GGML_OP_DSV4_ROPE_TAIL,
            GGML_OP_DSV4_FP8_KV_QUANTIZE,
            GGML_OP_VIEW,
            GGML_OP_SET_ROWS,
        };
        static const int outputs[] = { 1, 3 };
        if (!ctx->can_fuse_subgraph(idx, ops, 4, outputs, 2)) {
            return reject("subgraph-view");
        }
        plan->n_fuse = 4;
    } else {
        static const ggml_op ops[] = {
            GGML_OP_DSV4_ROPE_TAIL,
            GGML_OP_DSV4_FP8_KV_QUANTIZE,
            GGML_OP_SET_ROWS,
        };
        static const int outputs[] = { 1, 2 };
        if (!ctx->can_fuse_subgraph(idx, ops, 3, outputs, 2)) {
            return reject("subgraph-direct");
        }
        plan->n_fuse = 3;
    }

    plan->set_rows = set_rows;
    return true;
}

int ggml_metal_op_dsv4_fp8_kv_quantize(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_FP8_KV_QUANTIZE);

    const int64_t n_rot   = ggml_get_op_params_i32(op, 0);
    const int64_t n_nope  = op->src[0]->ne[0] - n_rot;
    const int64_t n_rows  = op->src[0]->ne[1] * op->src[0]->ne[2] * op->src[0]->ne[3];
    const int64_t n_blks  = n_nope / 64;
    const int64_t n_work  = n_rows * (n_blks + 1);

    ggml_metal_kargs_dsv4_fp8_kv_quantize args = {
        /*.head_dim =*/ op->src[0]->ne[0],
        /*.n_nope   =*/ n_nope,
        /*.n_rows   =*/ n_rows,
        /*.n_blocks =*/ n_blks,
        /*.ne1      =*/ op->src[0]->ne[1],
        /*.ne2      =*/ op->src[0]->ne[2],
        /*.src_nb0  =*/ op->src[0]->nb[0],
        /*.src_nb1  =*/ op->src[0]->nb[1],
        /*.src_nb2  =*/ op->src[0]->nb[2],
        /*.src_nb3  =*/ op->src[0]->nb[3],
        /*.dst_nb0  =*/ op->nb[0],
        /*.dst_nb1  =*/ op->nb[1],
        /*.dst_nb2  =*/ op->nb[2],
        /*.dst_nb3  =*/ op->nb[3],
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_fp8_kv_quantize(lib, op);
    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) n_work);
    const int ntg = (n_work + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_dsv4_hadamard_fp4_quantize(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_HADAMARD_FP4_QUANTIZE);

    const int64_t n_rows = op->src[0]->ne[1] * op->src[0]->ne[2] * op->src[0]->ne[3];

    ggml_metal_kargs_dsv4_hadamard_fp4_quantize args = {
        /*.head_dim =*/ op->src[0]->ne[0],
        /*.n_rows   =*/ n_rows,
        /*.ne1      =*/ op->src[0]->ne[1],
        /*.ne2      =*/ op->src[0]->ne[2],
        /*.src_nb0  =*/ op->src[0]->nb[0],
        /*.src_nb1  =*/ op->src[0]->nb[1],
        /*.src_nb2  =*/ op->src[0]->nb[2],
        /*.src_nb3  =*/ op->src[0]->nb[3],
        /*.dst_nb0  =*/ op->nb[0],
        /*.dst_nb1  =*/ op->nb[1],
        /*.dst_nb2  =*/ op->nb[2],
        /*.dst_nb3  =*/ op->nb[3],
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hadamard_fp4_quantize(lib, op);
    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) n_rows);
    const int ntg = (n_rows + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_dsv4_indexer_weighted_score(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_INDEXER_WEIGHTED_SCORE);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    const int64_t n_elem = op->ne[0] * op->ne[1];

    ggml_metal_kargs_dsv4_indexer_weighted_score args = {
        /*.n_comp      =*/ op->ne[0],
        /*.n_tokens    =*/ op->ne[1],
        /*.n_heads     =*/ op->src[0]->ne[2],
        /*.n_elem      =*/ n_elem,
        /*.score_nb0   =*/ op->src[0]->nb[0],
        /*.score_nb1   =*/ op->src[0]->nb[1],
        /*.score_nb2   =*/ op->src[0]->nb[2],
        /*.weights_nb0 =*/ op->src[1]->nb[0],
        /*.weights_nb1 =*/ op->src[1]->nb[1],
        /*.dst_nb0     =*/ op->nb[0],
        /*.dst_nb1     =*/ op->nb[1],
        /*.scale       =*/ ggml_get_op_params_f32(op, 0),
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_indexer_weighted_score(lib, op);
    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) n_elem);
    const int ntg = (n_elem + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    g_ggml_metal_dsv4_indexer_weighted_score_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_compressor_pair_proj(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_COMPRESSOR_PAIR_PROJ);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F16);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->ne[0] == op->src[1]->ne[0]);
    GGML_ASSERT(op->src[0]->ne[1] == op->src[1]->ne[1]);
    GGML_ASSERT(op->src[2]->ne[0] == op->src[0]->ne[0]);
    GGML_ASSERT(op->src[2]->ne[1] == 1);
    GGML_ASSERT(op->ne[0] == 2*op->src[0]->ne[1]);
    GGML_ASSERT(op->ne[1] == 1);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[2], nb);

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_compressor_pair(lib, op);
    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    const size_t smem = pipeline.smem;
    const int32_t width = op->src[0]->ne[1];

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ width,
        /*.ne1  =*/ ne11,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.src0_byte_off =*/ 0,
        /*.src1_byte_off =*/ 0,
        /*.dst_byte_off  =*/ 0,
    };

    ggml_metal_buffer_id bid_dst_b = ggml_metal_get_buffer_id(op);
    bid_dst_b.offs += size_t(width) * ggml_type_size(op->type);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);
    ggml_metal_encoder_set_buffer  (enc, bid_dst_b,                            5);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, (width + nr0 - 1)/nr0, 1, ne12*ne13, 32, nsg, 1);

    g_ggml_metal_dsv4_compressor_pair_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_mixed_attn(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_MIXED_ATTN);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F16);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_F16);
    GGML_ASSERT(op->src[3]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    ggml_metal_kargs_dsv4_mixed_attn args = {
        /*.n_heads       =*/ int32_t(op->src[0]->ne[1]),
        /*.n_raw         =*/ int32_t(op->src[1]->ne[2]),
        /*.n_comp        =*/ int32_t(op->src[2]->ne[2]),
        /*.q_nb1         =*/ op->src[0]->nb[1],
        /*.q_nb2         =*/ op->src[0]->nb[2],
        /*.raw_nb2       =*/ op->src[1]->nb[2],
        /*.comp_nb2      =*/ op->src[2]->nb[2],
        /*.raw_mask_nb0  =*/ op->src[3]->nb[0],
        /*.raw_mask_nb1  =*/ op->src[3]->nb[1],
        /*.comp_mask_nb0 =*/ op->src[4]->nb[0],
        /*.comp_mask_nb1 =*/ op->src[4]->nb[1],
        /*.dst_nb1       =*/ op->nb[1],
        /*.dst_nb2       =*/ op->nb[2],
        /*.scale         =*/ ggml_get_op_params_f32(op, 0),
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_mixed_attn(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         7);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, 4*128*4*sizeof(ggml_fp16_t), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, op->src[0]->ne[2], (op->src[0]->ne[1] + 7)/8, 1, 32, 8, 1);

    g_ggml_metal_dsv4_mixed_attn_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_attn_out_decode(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_ATTN_OUT_DECODE);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    if (!ggml_metal_dsv4_experimental_attn_out_decode_enabled()) {
        return 0;
    }

    GGML_TENSOR_LOCALS( int32_t, nea, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nba, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, neh, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nbh, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, neb, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nbb, op->src[2], nb);

    const int32_t group_dim = nea0;
    const int32_t rank      = nea1;
    const int32_t n_groups  = nea2;
    const int32_t low_dim   = neb0;
    const int32_t out_dim   = neb1;

    GGML_ASSERT(nea3 == 1);
    GGML_ASSERT(neh0 == group_dim);
    GGML_ASSERT(neh1 == n_groups);
    GGML_ASSERT(neh2 == 1);
    GGML_ASSERT(neh3 == 1);
    GGML_ASSERT(neb0 == rank*n_groups);
    GGML_ASSERT(neb2 == 1);
    GGML_ASSERT(neb3 == 1);
    GGML_ASSERT(op->ne[0] == out_dim + low_dim);
    GGML_ASSERT(op->ne[1] == 1);

    if (group_dim <= 0 || rank <= 0 || n_groups <= 0 || low_dim <= 0 || out_dim <= 0) {
        return 0;
    }

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    auto pipeline_low = ggml_metal_library_get_pipeline_dsv4_attn_out_low(lib, op);
    GGML_ASSERT(pipeline_low.smem <= props_dev->max_theadgroup_memory_size);

    ggml_metal_kargs_mul_mv_id low_args = {
        /*.nei0 =*/ n_groups,
        /*.nei1 =*/ 1,
        /*.nbi1 =*/ 0,
        /*.ne00 =*/ group_dim,
        /*.ne01 =*/ rank,
        /*.ne02 =*/ n_groups,
        /*.nb00 =*/ nba0,
        /*.nb01 =*/ nba1,
        /*.nb02 =*/ nba2,
        /*.ne10 =*/ neh0,
        /*.ne11 =*/ neh1,
        /*.ne12 =*/ neh2,
        /*.ne13 =*/ neh3,
        /*.nb10 =*/ nbh0,
        /*.nb11 =*/ nbh1,
        /*.nb12 =*/ nbh2,
        /*.ne0  =*/ rank,
        /*.ne1  =*/ n_groups,
        /*.nb1  =*/ uint64_t(rank) * sizeof(float),
        /*.nr0  =*/ pipeline_low.nr0,
    };

    ggml_metal_buffer_id bid_dst_low = ggml_metal_get_buffer_id(op);
    bid_dst_low.offs += uint64_t(out_dim) * sizeof(float);

    ggml_metal_encoder_set_pipeline(enc, pipeline_low);
    ggml_metal_encoder_set_bytes(enc, &low_args, sizeof(low_args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer(enc, bid_dst_low, 3);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline_low.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(
            enc,
            (rank + pipeline_low.nr0 - 1)/pipeline_low.nr0,
            1,
            n_groups,
            32,
            pipeline_low.nsg,
            1);

    ggml_metal_op_concurrency_reset(ctx);

    if (ggml_metal_dsv4_trace_attn_out_decode_enabled()) {
        static std::atomic<uint64_t> trace_count { 0 };
        const uint64_t n = trace_count.fetch_add(1);
        if (n < 8) {
            GGML_LOG_INFO("%s: dsv4_attn_out_decode group_dim=%d rank=%d groups=%d low=%d out=%d\n",
                    __func__, group_dim, rank, n_groups, low_dim, out_dim);
        }
    }

    ggml_tensor low_tmp = ggml_metal_make_buffer_tensor_2d(
            op->buffer,
            GGML_TYPE_F32,
            low_dim,
            1,
            static_cast<char *>(op->data) + uint64_t(out_dim) * sizeof(float),
            "dsv4_attn_out_decode_low_tmp");
    ggml_tensor out_tmp = ggml_metal_make_buffer_tensor_2d(
            op->buffer,
            GGML_TYPE_F32,
            out_dim,
            1,
            op->data,
            "dsv4_attn_out_decode_out_tmp");

    ggml_metal_encode_mul_mat_from_tensors(ctx, op->src[2], &low_tmp, &out_tmp, false);

    g_ggml_metal_dsv4_attn_out_decode_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_decode_compress(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_DECODE_COMPRESS);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[3]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    if (!ggml_metal_dsv4_experimental_decode_compress_enabled()) {
        return 0;
    }

    const int64_t head_dim = op->ne[0];
    const int32_t n_dims   = ggml_get_op_params_i32(op, 0);

    ggml_metal_kargs_dsv4_decode_compress args = {
        /*.head_dim  =*/ head_dim,
        /*.n_pool    =*/ op->src[0]->ne[1],
        /*.n_dims    =*/ n_dims,
        /*.n_nope    =*/ (int32_t) (head_dim - n_dims),
        /*.n_ctx_orig=*/ ggml_get_op_params_i32(op, 2),
        /*.mode      =*/ ggml_get_op_params_i32(op, 1),
        /*.kv_nb0    =*/ op->src[0]->nb[0],
        /*.kv_nb1    =*/ op->src[0]->nb[1],
        /*.score_nb0 =*/ op->src[1]->nb[0],
        /*.score_nb1 =*/ op->src[1]->nb[1],
        /*.norm_nb0  =*/ op->src[2]->nb[0],
        /*.dst_nb0   =*/ op->nb[0],
        /*.freq_base =*/ ggml_get_op_params_f32(op, 4),
        /*.freq_scale=*/ ggml_get_op_params_f32(op, 5),
        /*.ext_factor=*/ ggml_get_op_params_f32(op, 6),
        /*.attn_factor=*/ ggml_get_op_params_f32(op, 7),
        /*.beta_fast =*/ ggml_get_op_params_f32(op, 8),
        /*.beta_slow =*/ ggml_get_op_params_f32(op, 9),
        /*.norm_eps  =*/ ggml_get_op_params_f32(op, 10),
        /*.output_stage=*/ ggml_get_op_params_i32(op, 11),
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_decode_compress(lib, op);
    const int nth_max = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), std::min(256, (int) head_dim));
    int nth = 1;
    while (nth*2 <= nth_max) {
        nth *= 2;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         5);
    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    g_ggml_metal_dsv4_decode_compress_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_compressor_update_decode(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F16);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[3]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    if (!ggml_metal_dsv4_experimental_compressor_update_enabled()) {
        return 0;
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[2], nb);

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    const int32_t width = op->src[0]->ne[1];
    const int64_t rows = op->src[3]->ne[1];
    const int32_t n_dims = ggml_get_op_params_i32(op, 0);
    const int32_t pos = ggml_get_op_params_i32(op, 3);
    const int32_t compress_ratio = ggml_get_op_params_i32(op, 4);
    const int32_t pos_mod = compress_ratio > 0 ? pos % compress_ratio : 0;
    const int32_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = compress_ratio > 0 && (pos + 1) % compress_ratio == 0;
    const bool fused_comp = should_compress && ggml_metal_dsv4_experimental_compressor_update_fused_comp_enabled();
    const int64_t head_dim = compress_ratio == 4 ? width/2 : width;
    const int64_t state_elems = width * rows;
    const size_t elem_size = ggml_type_size(op->type);
    const size_t pair_off_bytes = size_t(2*state_elems + head_dim) * elem_size;

    auto pair_pipeline = ggml_metal_library_get_pipeline_dsv4_compressor_pair(lib, op);
    const int nr0 = pair_pipeline.nr0;
    const int nsg = pair_pipeline.nsg;
    const size_t smem = pair_pipeline.smem;

    ggml_metal_kargs_mul_mv pair_args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ width,
        /*.ne1  =*/ ne11,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.src0_byte_off =*/ 0,
        /*.src1_byte_off =*/ 0,
        /*.dst_byte_off  =*/ 0,
    };

    ggml_metal_buffer_id bid_pair_kv = ggml_metal_get_buffer_id(op);
    bid_pair_kv.offs += pair_off_bytes;
    ggml_metal_buffer_id bid_pair_score = bid_pair_kv;
    bid_pair_score.offs += size_t(width) * elem_size;

    ggml_metal_encoder_set_pipeline(enc, pair_pipeline);
    ggml_metal_encoder_set_bytes   (enc, &pair_args, sizeof(pair_args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, bid_pair_kv,                         4);
    ggml_metal_encoder_set_buffer  (enc, bid_pair_score,                      5);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, (width + nr0 - 1)/nr0, 1, ne12*ne13, 32, nsg, 1);
    ggml_metal_encoder_memory_barrier(enc);

    ggml_metal_kargs_dsv4_compressor_update_decode args = {
        /*.head_dim  =*/ head_dim,
        /*.width     =*/ width,
        /*.rows      =*/ rows,
        /*.state_elems=*/ state_elems,
        /*.pos       =*/ pos,
        /*.pos_mod   =*/ pos_mod,
        /*.row       =*/ row,
        /*.compress_ratio=*/ compress_ratio,
        /*.should_compress=*/ should_compress ? 1 : 0,
        /*.fused_comp=*/ fused_comp ? 1 : 0,
        /*.n_pool    =*/ compress_ratio == 4 ? 2*compress_ratio : compress_ratio,
        /*.n_dims    =*/ n_dims,
        /*.n_nope    =*/ (int32_t) (head_dim - n_dims),
        /*.n_ctx_orig=*/ ggml_get_op_params_i32(op, 2),
        /*.mode      =*/ ggml_get_op_params_i32(op, 1),
        /*.prev_kv_nb0=*/ op->src[3]->nb[0],
        /*.prev_kv_nb1=*/ op->src[3]->nb[1],
        /*.prev_score_nb0=*/ op->src[4]->nb[0],
        /*.prev_score_nb1=*/ op->src[4]->nb[1],
        /*.ape_nb0   =*/ op->src[5]->nb[0],
        /*.ape_nb1   =*/ op->src[5]->nb[1],
        /*.norm_nb0  =*/ op->src[6]->nb[0],
        /*.dst_nb0   =*/ op->nb[0],
        /*.freq_base =*/ ggml_get_op_params_f32(op, 5),
        /*.freq_scale=*/ ggml_get_op_params_f32(op, 6),
        /*.ext_factor=*/ ggml_get_op_params_f32(op, 7),
        /*.attn_factor=*/ ggml_get_op_params_f32(op, 8),
        /*.beta_fast =*/ ggml_get_op_params_f32(op, 9),
        /*.beta_slow =*/ ggml_get_op_params_f32(op, 10),
        /*.norm_eps  =*/ ggml_get_op_params_f32(op, 11),
    };

    auto update_pipeline = ggml_metal_library_get_pipeline_dsv4_compressor_update_decode(lib, op);
    const int nth_max = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(update_pipeline), std::min(256, (int) std::max<int64_t>(1, head_dim)));
    int nth = 1;
    while (nth*2 <= nth_max) {
        nth *= 2;
    }
    const int64_t n_state_tg = std::max<int64_t>(1, (state_elems + nth - 1)/nth);

    if (ggml_metal_dsv4_trace_compressor_update_enabled() &&
            g_ggml_metal_dsv4_compressor_update_trace_count.fetch_add(1) < 64) {
        GGML_LOG_INFO("%s: dsv4_compressor_update_decode width=%d rows=%lld head=%lld ratio=%d pos=%d row=%d compress=%d fused_comp=%d state_tg=%lld\n",
                __func__, width, (long long) rows, (long long) head_dim, compress_ratio, pos, row, should_compress ? 1 : 0, fused_comp ? 1 : 0, (long long) n_state_tg);
    }

    ggml_metal_encoder_set_pipeline(enc, update_pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_pair_kv,                         1);
    ggml_metal_encoder_set_buffer  (enc, bid_pair_score,                      2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         7);
    ggml_metal_encoder_dispatch_threadgroups(enc, n_state_tg, 1, 1, nth, 1, 1);

    g_ggml_metal_dsv4_compressor_update_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_compressor_update_decode_v2(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_COMPRESSOR_UPDATE_DECODE_V2);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[3]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    if (!ggml_metal_dsv4_experimental_compressor_update_v2_enabled()) {
        return 0;
    }

    const int32_t width = op->src[2]->ne[0];
    const int64_t rows = op->src[2]->ne[1];
    const int32_t n_dims = ggml_get_op_params_i32(op, 0);
    const int32_t pos = ggml_get_op_params_i32(op, 3);
    const int32_t compress_ratio = ggml_get_op_params_i32(op, 4);
    const int32_t pos_mod = compress_ratio > 0 ? pos % compress_ratio : 0;
    const int32_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = compress_ratio > 0 && (pos + 1) % compress_ratio == 0;
    const bool fused_comp = should_compress && ggml_metal_dsv4_experimental_compressor_update_v2_fused_comp_enabled();
    const int64_t head_dim = compress_ratio == 4 ? width/2 : width;
    const int64_t state_elems = int64_t(width) * rows;
    const int64_t pool_rows = compress_ratio == 4 ? 2*compress_ratio : compress_ratio;
    const int64_t pool_off_elems = 2*state_elems + head_dim;

    ggml_metal_kargs_dsv4_compressor_update_decode_v2 args = {
        /*.head_dim  =*/ head_dim,
        /*.width     =*/ width,
        /*.rows      =*/ rows,
        /*.state_elems=*/ state_elems,
        /*.pool_off_elems=*/ pool_off_elems,
        /*.pos       =*/ pos,
        /*.pos_mod   =*/ pos_mod,
        /*.row       =*/ row,
        /*.compress_ratio=*/ compress_ratio,
        /*.should_compress=*/ should_compress ? 1 : 0,
        /*.fused_comp=*/ fused_comp ? 1 : 0,
        /*.n_pool    =*/ (int32_t) pool_rows,
        /*.n_dims    =*/ n_dims,
        /*.n_nope    =*/ (int32_t) (head_dim - n_dims),
        /*.n_ctx_orig=*/ ggml_get_op_params_i32(op, 2),
        /*.mode      =*/ ggml_get_op_params_i32(op, 1),
        /*.kv_nb0    =*/ op->src[0]->nb[0],
        /*.score_nb0 =*/ op->src[1]->nb[0],
        /*.prev_kv_nb0=*/ op->src[2]->nb[0],
        /*.prev_kv_nb1=*/ op->src[2]->nb[1],
        /*.prev_score_nb0=*/ op->src[3]->nb[0],
        /*.prev_score_nb1=*/ op->src[3]->nb[1],
        /*.ape_nb0   =*/ op->src[4]->nb[0],
        /*.ape_nb1   =*/ op->src[4]->nb[1],
        /*.norm_nb0  =*/ op->src[5]->nb[0],
        /*.dst_nb0   =*/ op->nb[0],
        /*.freq_base =*/ ggml_get_op_params_f32(op, 5),
        /*.freq_scale=*/ ggml_get_op_params_f32(op, 6),
        /*.ext_factor=*/ ggml_get_op_params_f32(op, 7),
        /*.attn_factor=*/ ggml_get_op_params_f32(op, 8),
        /*.beta_fast =*/ ggml_get_op_params_f32(op, 9),
        /*.beta_slow =*/ ggml_get_op_params_f32(op, 10),
        /*.norm_eps  =*/ ggml_get_op_params_f32(op, 11),
    };

    auto update_pipeline = ggml_metal_library_get_pipeline_dsv4_compressor_update_decode_v2(lib, op);
    const int nth_max = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(update_pipeline), std::min(256, (int) std::max<int64_t>(1, head_dim)));
    int nth = 1;
    while (nth*2 <= nth_max) {
        nth *= 2;
    }
    const int64_t n_state_tg = std::max<int64_t>(1, (state_elems + nth - 1)/nth);

    if (ggml_metal_dsv4_trace_compressor_update_v2_enabled() &&
            g_ggml_metal_dsv4_compressor_update_v2_trace_count.fetch_add(1) < 64) {
        GGML_LOG_INFO("%s: dsv4_compressor_update_decode_v2 width=%d rows=%lld head=%lld ratio=%d pos=%d row=%d compress=%d fused_comp=%d state_tg=%lld\n",
                __func__, width, (long long) rows, (long long) head_dim, compress_ratio, pos, row, should_compress ? 1 : 0, fused_comp ? 1 : 0, (long long) n_state_tg);
    }

    ggml_metal_encoder_set_pipeline(enc, update_pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         7);
    ggml_metal_encoder_dispatch_threadgroups(enc, n_state_tg, 1, 1, nth, 1, 1);

    g_ggml_metal_dsv4_compressor_update_v2_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_kv_finalize_decode(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_KV_FINALIZE_DECODE);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32 || op->src[2]->type == GGML_TYPE_I64);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    if (!ggml_metal_dsv4_experimental_kv_finalize_enabled()) {
        return 0;
    }

    const int32_t width  = (int32_t) op->src[0]->ne[0];
    const int32_t n_rows = (int32_t) op->src[0]->ne[2];
    const int32_t dry_run = ggml_get_op_params_i32(op, 0);

    ggml_metal_kargs_dsv4_kv_finalize_decode args = {
        /*.width     =*/ width,
        /*.n_rows    =*/ n_rows,
        /*.dry_run   =*/ dry_run,
        /*.src_nb0   =*/ op->src[0]->nb[0],
        /*.src_nb1   =*/ op->src[0]->nb[1],
        /*.src_nb2   =*/ op->src[0]->nb[2],
        /*.src_nb3   =*/ op->src[0]->nb[3],
        /*.cache_nb0 =*/ op->src[1]->nb[0],
        /*.cache_nb1 =*/ op->src[1]->nb[1],
        /*.cache_nb2 =*/ op->src[1]->nb[2],
        /*.cache_nb3 =*/ op->src[1]->nb[3],
        /*.rows_nb0  =*/ op->src[2]->nb[0],
        /*.rows_nb1  =*/ op->src[2]->nb[1],
        /*.rows_nb2  =*/ op->src[2]->nb[2],
        /*.dst_nb0   =*/ op->nb[0],
        /*.dst_nb1   =*/ op->nb[1],
        /*.dst_nb2   =*/ op->nb[2],
        /*.dst_nb3   =*/ op->nb[3],
    };

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_kv_finalize_decode(lib, op);
    const int nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int64_t n = int64_t(width) * int64_t(n_rows);
    const int64_t n_tg = std::max<int64_t>(1, (n + nth - 1)/nth);

    if (ggml_metal_dsv4_trace_kv_finalize_enabled() &&
            g_ggml_metal_dsv4_kv_finalize_trace_count.fetch_add(1) < 256) {
        GGML_LOG_INFO("%s: dsv4_kv_finalize width=%d rows=%d dry_run=%d src=%s/%s cache=%s/%s rows=%s/%s\n",
                __func__,
                width,
                n_rows,
                dry_run,
                ggml_get_name(op->src[0]), ggml_type_name(op->src[0]->type),
                ggml_get_name(op->src[1]), ggml_type_name(op->src[1]->type),
                ggml_get_name(op->src[2]), ggml_type_name(op->src[2]->type));
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);
    ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, nth, 1, 1);

    g_ggml_metal_dsv4_kv_finalize_count.fetch_add(1);
    return 1;
}

struct ggml_metal_dsv4_ffn_moe_stage_v2_gate_args {
    uint64_t slot_stride;
    uint64_t weight_slot_stride;
    uint64_t weight_token_stride;
    int32_t swiglu_formula_mode;
};

struct ggml_metal_dsv4_ffn_moe_stage_v2_weight_args_global {
    uint32_t n_ff;
    uint64_t slot_stride;
    uint64_t weight_slot_stride;
    uint64_t weight_token_stride;
};

static int ggml_metal_op_dsv4_ffn_moe_decode_stage_v2(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_FFN_MOE_DECODE_STAGE);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_IQ2_XXS);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_IQ2_XXS);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(op->src[3]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[4]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    if (!ggml_metal_dsv4_experimental_ffn_moe_stage_enabled()) {
        return 0;
    }

    auto gate_pipeline = ggml_metal_library_get_pipeline_dsv4_ffn_moe_stage_v2_gate(lib);
    auto weight_pipeline = ggml_metal_library_get_pipeline_dsv4_ffn_moe_stage_v2_weight(lib);
    auto down_pipeline = ggml_metal_library_get_pipeline_dsv4_ffn_moe_stage_v2_down(lib);
    if (!gate_pipeline.pipeline || !weight_pipeline.pipeline || !down_pipeline.pipeline) {
        return 0;
    }

    const ggml_tensor * gate    = op->src[0];
    const ggml_tensor * up      = op->src[1];
    const ggml_tensor * down    = op->src[2];
    const ggml_tensor * x       = op->src[3];
    const ggml_tensor * ids     = op->src[4];
    const ggml_tensor * weights = op->src[5];

    GGML_TENSOR_LOCALS( int32_t, neg, gate, ne);
    GGML_TENSOR_LOCALS(uint64_t, nbg, gate, nb);
    GGML_TENSOR_LOCALS( int32_t, ned, down, ne);
    GGML_TENSOR_LOCALS(uint64_t, nbd, down, nb);
    GGML_TENSOR_LOCALS( int32_t, nex, x, ne);
    GGML_TENSOR_LOCALS(uint64_t, nbx, x, nb);
    GGML_TENSOR_LOCALS( int32_t, nei, ids, ne);
    GGML_TENSOR_LOCALS(uint64_t, nbi, ids, nb);

    float clamp = ggml_get_op_params_f32(op, 0);
    const uint64_t weight_slot_stride = weights->ne[1] == 6 ? weights->nb[1] : weights->nb[0];
    ggml_metal_dsv4_ffn_moe_stage_v2_gate_args gate_extra = {
        /*.slot_stride         =*/ static_cast<uint64_t>(op->nb[1]),
        /*.weight_slot_stride  =*/ weight_slot_stride,
        /*.weight_token_stride =*/ static_cast<uint64_t>(weights->nb[2]),
        /*.swiglu_formula_mode =*/ 0,
    };

    ggml_metal_kargs_mul_mv_id gate_args = {
        /*.nei0 =*/ nei0,
        /*.nei1 =*/ nei1,
        /*.nbi1 =*/ nbi1,
        /*.ne00 =*/ neg0,
        /*.ne01 =*/ neg1,
        /*.ne02 =*/ neg2,
        /*.nb00 =*/ nbg0,
        /*.nb01 =*/ nbg1,
        /*.nb02 =*/ nbg2,
        /*.ne10 =*/ nex0,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.ne13 =*/ 1,
        /*.nb10 =*/ nbx0,
        /*.nb11 =*/ nbx1,
        /*.nb12 =*/ nbx2,
        /*.ne0  =*/ neg1,
        /*.ne1  =*/ 6,
        /*.nb1  =*/ static_cast<uint64_t>(op->nb[1]),
        /*.nr0  =*/ gate_pipeline.nr0,
    };

    ggml_metal_encoder_set_pipeline(enc, gate_pipeline);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, gate_pipeline.smem, 0);
    ggml_metal_encoder_set_bytes (enc, &gate_args, sizeof(gate_args), 0);
    ggml_metal_encoder_set_bytes (enc, &clamp, sizeof(clamp), 1);
    ggml_metal_encoder_set_bytes (enc, &gate_extra, sizeof(gate_extra), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(gate),    3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(up),      4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),       5);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),      6);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(ids),     7);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights), 8);
    ggml_metal_encoder_dispatch_threadgroups(
            enc,
            (neg1 + gate_pipeline.nr0*gate_pipeline.nsg - 1)/(gate_pipeline.nr0*gate_pipeline.nsg),
            1,
            6,
            32,
            gate_pipeline.nsg,
            1);

    ggml_metal_encoder_memory_barrier(enc);

    struct ggml_metal_dsv4_ffn_moe_stage_v2_weight_args {
        uint32_t n_ff;
        uint64_t slot_stride;
        uint64_t weight_slot_stride;
        uint64_t weight_token_stride;
    };

    ggml_metal_dsv4_ffn_moe_stage_v2_weight_args weight_extra = {
        /*.n_ff                =*/ static_cast<uint32_t>(neg1),
        /*.slot_stride         =*/ static_cast<uint64_t>(op->nb[1]),
        /*.weight_slot_stride  =*/ weight_slot_stride,
        /*.weight_token_stride =*/ static_cast<uint64_t>(weights->nb[2]),
    };
    const uint32_t n_weight = static_cast<uint32_t>(neg1 * 6);
    ggml_metal_encoder_set_pipeline(enc, weight_pipeline);
    ggml_metal_encoder_set_bytes (enc, &weight_extra, sizeof(weight_extra), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),      1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_dispatch_threadgroups(
            enc,
            (n_weight + 255u)/256u,
            1,
            1,
            256,
            1,
            1);

    ggml_metal_encoder_memory_barrier(enc);

    ggml_metal_kargs_mul_mv_id down_args = {
        /*.nei0 =*/ nei0,
        /*.nei1 =*/ nei1,
        /*.nbi1 =*/ nbi1,
        /*.ne00 =*/ ned0,
        /*.ne01 =*/ ned1,
        /*.ne02 =*/ ned2,
        /*.nb00 =*/ nbd0,
        /*.nb01 =*/ nbd1,
        /*.nb02 =*/ nbd2,
        /*.ne10 =*/ neg1,
        /*.ne11 =*/ 6,
        /*.ne12 =*/ 1,
        /*.ne13 =*/ 1,
        /*.nb10 =*/ sizeof(float),
        /*.nb11 =*/ static_cast<uint64_t>(op->nb[1]),
        /*.nb12 =*/ static_cast<uint64_t>(6 * op->nb[1]),
        /*.ne0  =*/ ned1,
        /*.ne1  =*/ 6,
        /*.nb1  =*/ static_cast<uint64_t>(op->nb[1]),
        /*.nr0  =*/ down_pipeline.nr0,
    };

    ggml_metal_buffer_id mid_buf = ggml_metal_get_buffer_id(op);
    ggml_metal_buffer_id out_buf = ggml_metal_get_buffer_id(op);
    mid_buf.offs += static_cast<uint64_t>(12 * op->nb[1]);
    out_buf.offs += static_cast<uint64_t>(18 * op->nb[1]);

    ggml_metal_encoder_set_pipeline(enc, down_pipeline);
    ggml_metal_encoder_set_bytes (enc, &down_args, sizeof(down_args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down), 1);
    ggml_metal_encoder_set_buffer(enc, mid_buf, 2);
    ggml_metal_encoder_set_buffer(enc, out_buf, 3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(ids), 4);
    ggml_metal_encoder_dispatch_threadgroups(
            enc,
            (ned1 + down_pipeline.nr0*down_pipeline.nsg - 1)/(down_pipeline.nr0*down_pipeline.nsg),
            1,
            1,
            32,
            down_pipeline.nsg,
            1);

    if (ggml_metal_dsv4_trace_ffn_moe_stage_enabled() &&
            g_ggml_metal_dsv4_ffn_moe_stage_trace_count.fetch_add(1) < 128) {
        GGML_LOG_INFO("%s: dsv4_ffn_moe_stage_v2 full_stage=true owns_router=false owns_gate_up=true owns_swiglu=true owns_down=true owns_weighted_sum=true owns_shared=false nff=%d nembd=%d gate=%s up=%s down=%s x=%s ids=%s weights=%s clamp=%g\n",
                __func__,
                neg1, ned1,
                ggml_get_name(gate),
                ggml_get_name(up),
                ggml_get_name(down),
                ggml_get_name(x),
                ggml_get_name(ids),
                ggml_get_name(weights),
                double(clamp));
    }

    g_ggml_metal_dsv4_ffn_moe_stage_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_ffn_moe_decode_stage(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_FFN_MOE_DECODE_STAGE);

    if (op->src[3] != nullptr) {
        return ggml_metal_op_dsv4_ffn_moe_decode_stage_v2(ctx, idx);
    }

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    if (!ggml_metal_dsv4_experimental_ffn_moe_stage_enabled()) {
        return 0;
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_ffn_moe_decode_stage(lib, op);
    if (!pipeline.pipeline) {
        return 0;
    }

    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;

    ggml_metal_kargs_mul_mv_id args = {
        /*.nei0 =*/ ne20,
        /*.nei1 =*/ ne21,
        /*.nbi1 =*/ nb21,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.ne0  =*/ static_cast<int32_t>(op->ne[0]),
        /*.ne1  =*/ static_cast<int32_t>(op->ne[1]),
        /*.nb1  =*/ static_cast<uint64_t>(op->nb[1]),
        /*.nr0  =*/ nr0,
    };

    if (ggml_metal_dsv4_trace_ffn_moe_stage_enabled() &&
            g_ggml_metal_dsv4_ffn_moe_stage_trace_count.fetch_add(1) < 128) {
        GGML_LOG_INFO("%s: dsv4_ffn_moe_stage out=%lld slots=%lld nff=%d experts=%d tokens=%d down=%s act=%s ids=%s\n",
                __func__,
                (long long) op->ne[0], (long long) op->ne[1],
                ne00, ne02, ne12,
                ggml_get_name(op->src[0]),
                ggml_get_name(op->src[1]),
                ggml_get_name(op->src[2]));
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 4);
    ggml_metal_encoder_dispatch_threadgroups(
            enc,
            (op->ne[0] + nr0*nsg - 1)/(nr0*nsg),
            ne12,
            1,
            32,
            nsg,
            1);

    g_ggml_metal_dsv4_ffn_moe_stage_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_routed_moe_one_tensor_decode(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_ROUTED_MOE_ONE_TENSOR_DECODE);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    const bool pair_preserve_mode = ggml_get_op_params_i32(op, 5) != 0;
    if (pair_preserve_mode) {
        GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
        GGML_ASSERT(op->src[9] != nullptr);
        GGML_ASSERT(op->src[9]->type == GGML_TYPE_F32);
    } else {
        GGML_ASSERT(op->src[0]->type == GGML_TYPE_IQ2_XXS);
        GGML_ASSERT(op->src[1]->type == GGML_TYPE_IQ2_XXS);
    }
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(op->src[3]->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(op->src[4]->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(op->src[5]->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(op->src[pair_preserve_mode ? 1 : 6]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[7]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->src[8]->type == GGML_TYPE_F32);

    const int scratch_mode = ggml_get_op_params_i32(op, 0);
    if (scratch_mode == 1) {
        auto gate_pipeline = pair_preserve_mode ?
            ggml_metal_pipeline_with_params {} :
            ggml_metal_library_get_pipeline_dsv4_ffn_moe_stage_v2_gate(lib);
        const int swiglu_formula_mode = ggml_get_op_params_i32(op, 4);
        const bool swiglu_separate_kernel = swiglu_formula_mode == 4;
        auto swiglu_pipeline = swiglu_separate_kernel ?
            ggml_metal_library_get_pipeline_dsv4_routed_moe_swiglu_slots(lib) :
            ggml_metal_pipeline_with_params {};
        const bool scratch_down = ggml_get_op_params_i32(op, 2) != 0;
        const bool scratch_shared = ggml_get_op_params_i32(op, 3) != 0;
        auto down_pipeline = scratch_down ? ggml_metal_library_get_pipeline_dsv4_routed_moe_down_slots(lib) : ggml_metal_pipeline_with_params {};
        auto shared_gate_pipeline = scratch_shared ? ggml_metal_library_get_pipeline_dsv4_routed_moe_shared_gate_up_swiglu(lib) : ggml_metal_pipeline_with_params {};
        auto shared_down_pipeline = scratch_shared ? ggml_metal_library_get_pipeline_dsv4_routed_moe_shared_down_final(lib) : ggml_metal_pipeline_with_params {};
        if ((!pair_preserve_mode && !gate_pipeline.pipeline) ||
                (swiglu_separate_kernel && !swiglu_pipeline.pipeline) ||
                (scratch_down && !down_pipeline.pipeline) ||
                (scratch_shared && (!shared_gate_pipeline.pipeline || !shared_down_pipeline.pipeline))) {
            return 0;
        }

        const ggml_tensor * pair_swiglu = pair_preserve_mode ? op->src[9] : nullptr;
        const ggml_tensor * gate    = pair_preserve_mode ? pair_swiglu : op->src[0];
        const ggml_tensor * up      = pair_preserve_mode ? pair_swiglu : op->src[1];
        const ggml_tensor * down    = op->src[2];
        const ggml_tensor * shared_gate = op->src[3];
        const ggml_tensor * shared_up   = op->src[4];
        const ggml_tensor * shared_down = op->src[5];
        const ggml_tensor * x       = pair_preserve_mode ? op->src[1] : op->src[6];
        const ggml_tensor * ids     = op->src[7];
        const ggml_tensor * weights = op->src[8];

        GGML_TENSOR_LOCALS( int32_t, neg, gate, ne);
        GGML_TENSOR_LOCALS(uint64_t, nbg, gate, nb);
        GGML_TENSOR_LOCALS( int32_t, ned, down, ne);
        GGML_TENSOR_LOCALS(uint64_t, nbd, down, nb);
        GGML_TENSOR_LOCALS( int32_t, nex, x, ne);
        GGML_TENSOR_LOCALS(uint64_t, nbx, x, nb);
        GGML_TENSOR_LOCALS( int32_t, nei, ids, ne);
        GGML_TENSOR_LOCALS(uint64_t, nbi, ids, nb);
        const int32_t n_ff_slots = pair_preserve_mode ? neg0 : neg1;

        GGML_ASSERT(op->ne[0] >= n_ff_slots);
        GGML_ASSERT(op->ne[1] == (scratch_shared ? 35 : (scratch_down ? 30 : 24)));
        GGML_ASSERT(n_ff_slots == ned0);
        GGML_ASSERT(ned1 == op->ne[0]);

        float clamp = ggml_get_op_params_f32(op, 1);
        const uint64_t weight_slot_stride = weights->ne[1] == 6 ? weights->nb[1] : weights->nb[0];
        ggml_metal_dsv4_ffn_moe_stage_v2_gate_args gate_extra = {
            /*.slot_stride         =*/ static_cast<uint64_t>(op->nb[1]),
            /*.weight_slot_stride  =*/ weight_slot_stride,
            /*.weight_token_stride =*/ static_cast<uint64_t>(weights->nb[2]),
            /*.swiglu_formula_mode =*/ ggml_get_op_params_i32(op, 4),
        };

        ggml_metal_kargs_mul_mv_id gate_args = {
            /*.nei0 =*/ nei0,
            /*.nei1 =*/ nei1,
            /*.nbi1 =*/ nbi1,
            /*.ne00 =*/ neg0,
            /*.ne01 =*/ neg1,
            /*.ne02 =*/ neg2,
            /*.nb00 =*/ nbg0,
            /*.nb01 =*/ nbg1,
            /*.nb02 =*/ nbg2,
            /*.ne10 =*/ nex0,
            /*.ne11 =*/ 1,
            /*.ne12 =*/ 1,
            /*.ne13 =*/ 1,
            /*.nb10 =*/ nbx0,
            /*.nb11 =*/ nbx1,
            /*.nb12 =*/ nbx2,
            /*.ne0  =*/ n_ff_slots,
            /*.ne1  =*/ 6,
            /*.nb1  =*/ static_cast<uint64_t>(op->nb[1]),
            /*.nr0  =*/ pair_preserve_mode ? 1 : gate_pipeline.nr0,
        };

        if (g_ggml_metal_dsv4_routed_moe_one_tensor_trace_count.fetch_add(1) < 128) {
            GGML_LOG_INFO("%s: dsv4_routed_moe_one_tensor_decode scratch_gate_up=%d pair_preserve_mode=%d"
                    " output_computed=%d final_ffn_output_not_computed=%d"
                    " gate_up_substage_computed=%d swiglu_substage_computed=%d down_computed=%d routed_sum_computed=%d shared_computed=%d final_output_computed=%d"
                    " graph_boundary_one_tensor=1 monolithic_kernel=0 internal_dispatch_count=%d"
                    " scratch_allocation_mode=ggml_tensor scratch_shape=[%lld,%lld,%lld,%lld]"
                    " gate_scratch_shape=[%d,6] up_scratch_shape=[%d,6] swiglu_scratch_shape=[%d,6]"
                    " down_scratch_shape=[%d,6] routed_sum_shape=[%d]"
                    " out=%s x=%s ids=%s weights=%s gate=%s up=%s swiglu_source=%s clamp=%g consume_path=%s\n",
                    __func__,
                    pair_preserve_mode ? 0 : 1,
                    pair_preserve_mode ? 1 : 0,
                    scratch_shared ? 1 : 0,
                    scratch_shared ? 0 : 1,
                    pair_preserve_mode ? 0 : 1,
                    1,
                    scratch_down ? 1 : 0,
                    scratch_down ? 1 : 0,
                    scratch_shared ? 1 : 0,
                    scratch_shared ? 1 : 0,
                    pair_preserve_mode ? (scratch_shared ? 3 : (scratch_down ? 1 : 0)) : (scratch_shared ? 4 : (scratch_down ? 2 : 1)),
                    (long long) op->ne[0], (long long) op->ne[1],
                    (long long) op->ne[2], (long long) op->ne[3],
                    pair_preserve_mode ? 0 : n_ff_slots,
                    pair_preserve_mode ? 0 : n_ff_slots,
                    n_ff_slots,
                    scratch_down ? ned1 : 0,
                    scratch_down ? ned1 : 0,
                    ggml_get_name(op),
                    ggml_get_name(x),
                    ggml_get_name(ids),
                    ggml_get_name(weights),
                    ggml_get_name(gate),
                    ggml_get_name(up),
                    pair_preserve_mode ? ggml_get_name(pair_swiglu) : "internal_scratch",
                    double(clamp),
                    pair_preserve_mode ? "single_layer_pair_preserve" : "disabled");
        }

        if (!pair_preserve_mode) {
            ggml_metal_encoder_set_pipeline(enc, gate_pipeline);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, gate_pipeline.smem, 0);
            ggml_metal_encoder_set_bytes (enc, &gate_args, sizeof(gate_args), 0);
            ggml_metal_encoder_set_bytes (enc, &clamp, sizeof(clamp), 1);
            ggml_metal_encoder_set_bytes (enc, &gate_extra, sizeof(gate_extra), 2);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(gate),    3);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(up),      4);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),       5);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),      6);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(ids),     7);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights), 8);
            ggml_metal_encoder_dispatch_threadgroups(
                    enc,
                    (n_ff_slots + gate_pipeline.nr0*gate_pipeline.nsg - 1)/(gate_pipeline.nr0*gate_pipeline.nsg),
                    1,
                    6,
                    32,
                    gate_pipeline.nsg,
                    1);
        }

        if (!pair_preserve_mode && swiglu_separate_kernel) {
            ggml_metal_encoder_memory_barrier(enc);

            ggml_metal_dsv4_ffn_moe_stage_v2_weight_args_global swiglu_args = {
                /*.n_ff                =*/ static_cast<uint32_t>(n_ff_slots),
                /*.slot_stride         =*/ static_cast<uint64_t>(op->nb[1]),
                /*.weight_slot_stride  =*/ 0,
                /*.weight_token_stride =*/ 0,
            };

            ggml_metal_encoder_set_pipeline(enc, swiglu_pipeline);
            ggml_metal_encoder_set_bytes (enc, &swiglu_args, sizeof(swiglu_args), 0);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);
            ggml_metal_encoder_dispatch_threadgroups(
                    enc,
                    (neg1 * 6 + 255) / 256,
                    1,
                    1,
                    256,
                    1,
                    1);
        }

        if (scratch_down) {
            if (!pair_preserve_mode) {
                ggml_metal_encoder_memory_barrier(enc);
            }

            ggml_metal_kargs_mul_mv_id down_args = {
                /*.nei0 =*/ nei0,
                /*.nei1 =*/ nei1,
                /*.nbi1 =*/ nbi1,
                /*.ne00 =*/ ned0,
                /*.ne01 =*/ ned1,
                /*.ne02 =*/ ned2,
                /*.nb00 =*/ nbd0,
                /*.nb01 =*/ nbd1,
                /*.nb02 =*/ nbd2,
                /*.ne10 =*/ n_ff_slots,
                /*.ne11 =*/ 6,
                /*.ne12 =*/ 1,
                /*.ne13 =*/ 1,
                /*.nb10 =*/ pair_preserve_mode ? nbg0 : sizeof(float),
                /*.nb11 =*/ pair_preserve_mode ? nbg1 : static_cast<uint64_t>(op->nb[1]),
                /*.nb12 =*/ pair_preserve_mode ? nbg2 : static_cast<uint64_t>(6 * op->nb[1]),
                /*.ne0  =*/ ned1,
                /*.ne1  =*/ static_cast<int32_t>(op->ne[1]),
                /*.nb1  =*/ static_cast<uint64_t>(op->nb[1]),
                /*.nr0  =*/ down_pipeline.nr0,
            };
            const uint64_t weight_slot_stride = weights->ne[1] == 6 ? weights->nb[1] : weights->nb[0];
            const uint64_t weight_token_stride = weights->ne[2] > 1 ? weights->nb[2] : 0;
            uint64_t weight_strides[2] = { weight_slot_stride, weight_token_stride };

            ggml_metal_buffer_id mid_buf = ggml_metal_get_buffer_id(op);
            if (pair_preserve_mode) {
                mid_buf = ggml_metal_get_buffer_id(pair_swiglu);
            } else {
                mid_buf.offs += static_cast<uint64_t>(12 * op->nb[1]);
            }

            ggml_metal_encoder_set_pipeline(enc, down_pipeline);
            ggml_metal_encoder_set_bytes (enc, &down_args, sizeof(down_args), 0);
            ggml_metal_encoder_set_bytes (enc, weight_strides, sizeof(weight_strides), 1);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(down),    2);
            ggml_metal_encoder_set_buffer(enc, mid_buf,                           3);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),      4);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(ids),     5);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights), 6);
            ggml_metal_encoder_dispatch_threadgroups(
                    enc,
                    (ned1 + down_pipeline.nr0*down_pipeline.nsg - 1)/(down_pipeline.nr0*down_pipeline.nsg),
                    1,
                    1,
                    32,
                    down_pipeline.nsg,
                    1);
        }

        if (scratch_shared) {
            ggml_metal_encoder_memory_barrier(enc);

            GGML_TENSOR_LOCALS( int32_t, nesg, shared_gate, ne);
            GGML_TENSOR_LOCALS(uint64_t, nbsg, shared_gate, nb);
            GGML_TENSOR_LOCALS( int32_t, nesd, shared_down, ne);
            GGML_TENSOR_LOCALS(uint64_t, nbsd, shared_down, nb);

            ggml_metal_kargs_mul_mv shared_gate_args = {
                /*.ne00 =*/ nesg0,
                /*.ne01 =*/ nesg1,
                /*.ne02 =*/ nesg2,
                /*.nb00 =*/ nbsg0,
                /*.nb01 =*/ nbsg1,
                /*.nb02 =*/ nbsg2,
                /*.nb03 =*/ nbsg3,
                /*.ne10 =*/ nex0,
                /*.ne11 =*/ 1,
                /*.ne12 =*/ 1,
                /*.nb10 =*/ nbx0,
                /*.nb11 =*/ nbx1,
                /*.nb12 =*/ nbx2,
                /*.nb13 =*/ nbx3,
                /*.ne0  =*/ static_cast<int32_t>(op->ne[0]),
                /*.ne1  =*/ static_cast<int32_t>(op->ne[1]),
                /*.nr0  =*/ shared_gate_pipeline.nr0,
                /*.r2   =*/ 1,
                /*.r3   =*/ 1,
                /*.src0_byte_off =*/ 0,
                /*.src1_byte_off =*/ 0,
                /*.dst_byte_off  =*/ 0,
            };

            ggml_metal_encoder_set_pipeline(enc, shared_gate_pipeline);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, shared_gate_pipeline.smem, 0);
            ggml_metal_encoder_set_bytes (enc, &shared_gate_args, sizeof(shared_gate_args), 0);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(shared_gate), 1);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(shared_up),   2);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),           3);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),          4);
            ggml_metal_encoder_dispatch_threadgroups(
                    enc,
                    (nesg1 + shared_gate_pipeline.nr0*shared_gate_pipeline.nsg - 1)/(shared_gate_pipeline.nr0*shared_gate_pipeline.nsg),
                    1,
                    1,
                    32,
                    shared_gate_pipeline.nsg,
                    1);

            ggml_metal_encoder_memory_barrier(enc);

            ggml_metal_kargs_mul_mv shared_down_args = {
                /*.ne00 =*/ nesd0,
                /*.ne01 =*/ nesd1,
                /*.ne02 =*/ nesd2,
                /*.nb00 =*/ nbsd0,
                /*.nb01 =*/ nbsd1,
                /*.nb02 =*/ nbsd2,
                /*.nb03 =*/ nbsd3,
                /*.ne10 =*/ nesg1,
                /*.ne11 =*/ 1,
                /*.ne12 =*/ 1,
                /*.nb10 =*/ sizeof(float),
                /*.nb11 =*/ static_cast<uint64_t>(op->nb[1]),
                /*.nb12 =*/ static_cast<uint64_t>(op->nb[1]),
                /*.nb13 =*/ static_cast<uint64_t>(op->nb[1]),
                /*.ne0  =*/ static_cast<int32_t>(op->ne[0]),
                /*.ne1  =*/ static_cast<int32_t>(op->ne[1]),
                /*.nr0  =*/ shared_down_pipeline.nr0,
                /*.r2   =*/ 1,
                /*.r3   =*/ 1,
                /*.src0_byte_off =*/ 0,
                /*.src1_byte_off =*/ 0,
                /*.dst_byte_off  =*/ 0,
            };

            ggml_metal_buffer_id shared_swiglu_buf = ggml_metal_get_buffer_id(op);
            shared_swiglu_buf.offs += static_cast<uint64_t>(32 * op->nb[1]);

            ggml_metal_encoder_set_pipeline(enc, shared_down_pipeline);
            ggml_metal_encoder_set_threadgroup_memory_size(enc, shared_down_pipeline.smem, 0);
            ggml_metal_encoder_set_bytes (enc, &shared_down_args, sizeof(shared_down_args), 0);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(shared_down), 1);
            ggml_metal_encoder_set_buffer(enc, shared_swiglu_buf,                    2);
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         3);
            ggml_metal_encoder_dispatch_threadgroups(
                    enc,
                    (nesd1 + shared_down_pipeline.nr0*shared_down_pipeline.nsg - 1)/(shared_down_pipeline.nr0*shared_down_pipeline.nsg),
                    1,
                    1,
                    32,
                    shared_down_pipeline.nsg,
                    1);
        }

        g_ggml_metal_dsv4_routed_moe_one_tensor_count.fetch_add(1);
        return 1;
    }

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_routed_moe_one_tensor_decode(lib, op);
    if (!pipeline.pipeline) {
        return 0;
    }

    if (g_ggml_metal_dsv4_routed_moe_one_tensor_trace_count.fetch_add(1) < 128) {
        GGML_LOG_INFO("%s: dsv4_routed_moe_one_tensor_decode dry_run=true output_not_computed=1"
                " graph_boundary_one_tensor=1 monolithic_kernel=0 internal_dispatch_count=1"
                " owns_router=false consumes_topk=true owns_gate_up=true owns_swiglu=true owns_down=true"
                " owns_weighted_sum=true owns_shared=true output_computed=false"
                " unsupported_blocker=requires_intermediate_scratch_for_iq2_xxs_gate_up_q2_K_down_plus_q8_0_shared_branch"
                " out=%s x=%s ids=%s weights=%s gate=%s up=%s down=%s shared_gate=%s shared_up=%s shared_down=%s\n",
                __func__,
                ggml_get_name(op),
                ggml_get_name(op->src[6]),
                ggml_get_name(op->src[7]),
                ggml_get_name(op->src[8]),
                ggml_get_name(op->src[0]),
                ggml_get_name(op->src[1]),
                ggml_get_name(op->src[2]),
                ggml_get_name(op->src[3]),
                ggml_get_name(op->src[4]),
                ggml_get_name(op->src[5]));
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);

    g_ggml_metal_dsv4_routed_moe_one_tensor_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_decode_layer_executor_dryrun(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_DECODE_LAYER_EXECUTOR_DRYRUN);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->ne[0] >= 4);
    for (int i = 0; i < 9; ++i) {
        GGML_ASSERT(op->src[i] != nullptr);
    }

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_decode_layer_executor_dryrun(lib, op);
    if (!pipeline.pipeline) {
        return 0;
    }

    int32_t layer = ggml_get_op_params_i32(op, 0);
    int32_t token = ggml_get_op_params_i32(op, 1);
    int32_t flags = ggml_get_op_params_i32(op, 2);
    if (g_ggml_metal_dsv4_decode_layer_executor_dryrun_trace_count.fetch_add(1) < 128) {
        GGML_LOG_INFO("%s: dsv4_decode_layer_executor_dryrun layer=%d token=%d eligibility_flags=0x%x"
                " output_consumed=0 cache_mutation=disabled side_effects=disabled"
                " graph_boundary=full_decode_layer dry_run=true"
                " layer_input=%s attn_q=%s attn_kv=%s attn_out=%s attn_hc_post=%s"
                " ffn_norm=%s routed_moe_out=%s ffn_hc_post=%s pos=%s\n",
                __func__,
                layer,
                token,
                flags,
                ggml_get_name(op->src[0]),
                ggml_get_name(op->src[1]),
                ggml_get_name(op->src[2]),
                ggml_get_name(op->src[3]),
                ggml_get_name(op->src[4]),
                ggml_get_name(op->src[5]),
                ggml_get_name(op->src[6]),
                ggml_get_name(op->src[7]),
                ggml_get_name(op->src[8]));
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &layer, sizeof(layer), 0);
    ggml_metal_encoder_set_bytes(enc, &token, sizeof(token), 1);
    ggml_metal_encoder_set_bytes(enc, &flags, sizeof(flags), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);
    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);

    g_ggml_metal_dsv4_decode_layer_executor_dryrun_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_decode_layer(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_DECODE_LAYER);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0] != nullptr);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(op->src[0], op));
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op));

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_decode_layer(lib, op);
    if (!pipeline.pipeline) {
        return 0;
    }

    const int32_t  layer_index = ggml_get_op_params_i32(op, 0);
    const uint32_t stage_mask  = (uint32_t) ggml_get_op_params_i32(op, 1);
    const uint64_t total_elems = (uint64_t) ggml_nelements(op);

    if (g_ggml_metal_dsv4_decode_layer_trace_count.fetch_add(1) < 128) {
        GGML_LOG_INFO("%s: dsv4_decode_layer layer=%d stage_mask=0x%x"
                " total_elems=%llu output_consumed=0 cache_mutation=disabled"
                " side_effects=disabled graph_boundary=full_decode_layer"
                " stub_passthrough=true src0=%s\n",
                __func__,
                layer_index,
                stage_mask,
                (unsigned long long) total_elems,
                ggml_get_name(op->src[0]));
    }

    ggml_metal_kargs_dsv4_decode_layer args = {
        /*.layer_index     =*/ layer_index,
        /*.stage_mask      =*/ stage_mask,
        /*.total_elems_f32 =*/ total_elems,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         2);

    // One thread per element, in 256-thread groups.
    const int threads_per_group = 256;
    const int n_groups = (int) ((total_elems + threads_per_group - 1) / threads_per_group);
    ggml_metal_encoder_dispatch_threadgroups(enc, n_groups, 1, 1, threads_per_group, 1, 1);

    g_ggml_metal_dsv4_decode_layer_count.fetch_add(1);
    return 1;
}

int ggml_metal_op_dsv4_rope_tail(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_DSV4_ROPE_TAIL);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(op->type == op->src[0]->type);
    GGML_ASSERT(nb00 == ggml_type_size(op->src[0]->type));
    GGML_ASSERT(nb0  == ggml_type_size(op->type));
    GGML_ASSERT(ne10 >= ne02);

    const int n_dims     = ((const int32_t *) op->op_params)[0];
    const int n_ctx_orig = ((const int32_t *) op->op_params)[2];
    const int inverse    = ((const int32_t *) op->op_params)[3];
    const int n_nope     = ne00 - n_dims;

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 9, sizeof(float));

    ggml_metal_kargs_dsv4_rope_tail args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_dims      =*/ n_dims,
        /*.n_nope      =*/ n_nope,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.inverse     =*/ inverse,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /*.src2        =*/ op->src[2] != nullptr,
    };

    if (idx + 1 < ctx->n_nodes()) {
        ggml_tensor * next = ctx->node(idx + 1);

        if (ggml_metal_dsv4_rope_tail_can_fuse_hadamard_fp4(ctx, idx, op, next)) {
            const int64_t n_rows = op->ne[1] * op->ne[2] * op->ne[3];

            ggml_metal_kargs_dsv4_hadamard_fp4_quantize quant_args = {
                /*.head_dim =*/ op->ne[0],
                /*.n_rows   =*/ n_rows,
                /*.ne1      =*/ op->ne[1],
                /*.ne2      =*/ op->ne[2],
                /*.src_nb0  =*/ op->nb[0],
                /*.src_nb1  =*/ op->nb[1],
                /*.src_nb2  =*/ op->nb[2],
                /*.src_nb3  =*/ op->nb[3],
                /*.dst_nb0  =*/ next->nb[0],
                /*.dst_nb1  =*/ next->nb[1],
                /*.dst_nb2  =*/ next->nb[2],
                /*.dst_nb3  =*/ next->nb[3],
            };

            auto pipeline = ggml_metal_library_get_pipeline_dsv4_rope_hadamard_fp4_quantize(lib, op);
            const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) n_rows);
            const int ntg = (n_rows + nth - 1) / nth;

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args,       sizeof(args),       0);
            ggml_metal_encoder_set_bytes   (enc, &quant_args, sizeof(quant_args), 1);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 2);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 3);
            if (op->src[2]) {
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 4);
            } else {
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 4);
            }
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(next),       5);
            ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

            g_ggml_metal_dsv4_rope_hadamard_fp4_count.fetch_add(1);
            return 2;
        }

        ggml_metal_dsv4_kv_set_rows_fuse_plan kv_set_rows_plan;
        if (next->op == GGML_OP_DSV4_FP8_KV_QUANTIZE) {
            static std::atomic<uint64_t> kv_set_rows_probe_count { 0 };
            const char * trace = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_SET_ROWS_TRACE");
            if (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0 &&
                    kv_set_rows_probe_count.fetch_add(1) < 16) {
                const ggml_tensor * n2 = idx + 2 < ctx->n_nodes() ? ctx->node(idx + 2) : nullptr;
                const ggml_tensor * n3 = idx + 3 < ctx->n_nodes() ? ctx->node(idx + 3) : nullptr;
                fprintf(stderr, "%s: kvset probe op=%s next=%s n2=%s/%s n3=%s/%s\n",
                        __func__,
                        op->name,
                        ggml_op_name(next->op),
                        n2 != nullptr ? ggml_op_name(n2->op) : "(null)",
                        n2 != nullptr ? n2->name : "(null)",
                        n3 != nullptr ? ggml_op_name(n3->op) : "(null)",
                        n3 != nullptr ? n3->name : "(null)");
            }
        }
        if (ggml_metal_dsv4_rope_tail_can_fuse_fp8_kv_set_rows(ctx, idx, op, next, &kv_set_rows_plan)) {
            ggml_tensor * set_rows = kv_set_rows_plan.set_rows;
            ggml_tensor * set_src  = set_rows->src[0];

            const int64_t n_rot   = ggml_get_op_params_i32(next, 0);
            const int64_t n_nope  = op->ne[0] - n_rot;
            const int64_t n_rows  = op->ne[1] * op->ne[2] * op->ne[3];
            const int64_t n_blks  = n_nope / 64;

            ggml_metal_kargs_dsv4_fp8_kv_quantize quant_args = {
                /*.head_dim =*/ op->ne[0],
                /*.n_nope   =*/ n_nope,
                /*.n_rows   =*/ n_rows,
                /*.n_blocks =*/ n_blks,
                /*.ne1      =*/ op->ne[1],
                /*.ne2      =*/ op->ne[2],
                /*.src_nb0  =*/ op->nb[0],
                /*.src_nb1  =*/ op->nb[1],
                /*.src_nb2  =*/ op->nb[2],
                /*.src_nb3  =*/ op->nb[3],
                /*.dst_nb0  =*/ next->nb[0],
                /*.dst_nb1  =*/ next->nb[1],
                /*.dst_nb2  =*/ next->nb[2],
                /*.dst_nb3  =*/ next->nb[3],
            };

            ggml_metal_kargs_set_rows set_args = {
                /*.nk0  =*/ (int32_t) (set_rows->ne[0]/ggml_blck_size(set_rows->type)),
                /*.ne01 =*/ (int32_t) set_src->ne[1],
                /*.nb01 =*/ set_src->nb[1],
                /*.nb02 =*/ set_src->nb[2],
                /*.nb03 =*/ set_src->nb[3],
                /*.ne11 =*/ (int32_t) set_rows->src[1]->ne[1],
                /*.ne12 =*/ (int32_t) set_rows->src[1]->ne[2],
                /*.nb10 =*/ set_rows->src[1]->nb[0],
                /*.nb11 =*/ set_rows->src[1]->nb[1],
                /*.nb12 =*/ set_rows->src[1]->nb[2],
                /*.nb1  =*/ set_rows->nb[1],
                /*.nb2  =*/ set_rows->nb[2],
                /*.nb3  =*/ set_rows->nb[3],
            };

            auto pipeline = ggml_metal_library_get_pipeline_dsv4_rope_fp8_kv_set_rows(lib, op, set_rows);
            const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), std::min(256, (int) op->ne[0]));

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args,       sizeof(args),       0);
            ggml_metal_encoder_set_bytes   (enc, &quant_args, sizeof(quant_args), 1);
            ggml_metal_encoder_set_bytes   (enc, &set_args,   sizeof(set_args),   2);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]),       3);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]),       4);
            if (op->src[2]) {
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]),     5);
            } else {
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]),     5);
            }
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(next),             6);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(set_rows->src[1]), 7);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(set_rows),         8);
            ggml_metal_encoder_dispatch_threadgroups(enc, n_rows, 1, 1, nth, 1, 1);

            g_ggml_metal_dsv4_rope_fp8_kv_count.fetch_add(1);
            g_ggml_metal_dsv4_rope_fp8_kv_set_rows_count.fetch_add(1);
            return kv_set_rows_plan.n_fuse;
        }

        if (ggml_metal_dsv4_rope_tail_can_fuse_fp8_kv(ctx, idx, op, next)) {
            const int64_t n_rot   = ggml_get_op_params_i32(next, 0);
            const int64_t n_nope  = op->ne[0] - n_rot;
            const int64_t n_rows  = op->ne[1] * op->ne[2] * op->ne[3];
            const int64_t n_blks  = n_nope / 64;

            ggml_metal_kargs_dsv4_fp8_kv_quantize quant_args = {
                /*.head_dim =*/ op->ne[0],
                /*.n_nope   =*/ n_nope,
                /*.n_rows   =*/ n_rows,
                /*.n_blocks =*/ n_blks,
                /*.ne1      =*/ op->ne[1],
                /*.ne2      =*/ op->ne[2],
                /*.src_nb0  =*/ op->nb[0],
                /*.src_nb1  =*/ op->nb[1],
                /*.src_nb2  =*/ op->nb[2],
                /*.src_nb3  =*/ op->nb[3],
                /*.dst_nb0  =*/ next->nb[0],
                /*.dst_nb1  =*/ next->nb[1],
                /*.dst_nb2  =*/ next->nb[2],
                /*.dst_nb3  =*/ next->nb[3],
            };

            auto pipeline = ggml_metal_library_get_pipeline_dsv4_rope_fp8_kv_quantize(lib, op);
            const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), std::min(256, (int) op->ne[0]));

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args,       sizeof(args),       0);
            ggml_metal_encoder_set_bytes   (enc, &quant_args, sizeof(quant_args), 1);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 2);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 3);
            if (op->src[2]) {
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 4);
            } else {
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 4);
            }
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(next),       5);
            ggml_metal_encoder_dispatch_threadgroups(enc, n_rows, 1, 1, nth, 1, 1);

            g_ggml_metal_dsv4_rope_fp8_kv_count.fetch_add(1);
            return 2;
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_rope_tail(lib, op);
    const int nth = std::min(1024, ne00);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);
    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_im2col(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];
    const int32_t p1 = ((const int32_t *)(op->op_params))[3];
    const int32_t d0 = ((const int32_t *)(op->op_params))[4];
    const int32_t d1 = ((const int32_t *)(op->op_params))[5];

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;

    const int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    const int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    const int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    const int32_t IW =         op->src[1]->ne[0];

    const int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    const int32_t KW =         op->src[0]->ne[0];

    const int32_t OH = is_2D ? op->ne[2] : 1;
    const int32_t OW =         op->ne[1];

    const int32_t CHW = IC * KH * KW;

    const uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    const uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;

    ggml_metal_kargs_im2col args = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    auto pipeline = ggml_metal_library_get_pipeline_im2col(lib, op);

    GGML_ASSERT(KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const uint64_t ntptg0 = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)/(KH*KW), N);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);

    return 1;
}

int ggml_metal_op_conv_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.IC   =*/ ne02,
        /*.OC   =*/ ne03,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const uint64_t n_out = ggml_nelements(op);

    uint64_t tg = (n_out + nth - 1)/nth;
    tg = std::max<uint64_t>(tg, 1);
    tg = std::min<uint64_t>(tg, (uint64_t) std::numeric_limits<int>::max());

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_transpose_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[1];
    const int32_t IL = op->src[1]->ne[0];

    const int32_t K  = op->src[0]->ne[0];

    const int32_t OL = op->ne[0];
    const int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_conv_transpose_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[2];
    const int32_t IH = op->src[1]->ne[1];
    const int32_t IW = op->src[1]->ne[0];

    const int32_t KH = op->src[0]->ne[1];
    const int32_t KW = op->src[0]->ne[0];

    const int32_t OW = op->ne[0];
    const int32_t OH = op->ne[1];
    const int32_t OC = op->ne[2];

    ggml_metal_kargs_conv_transpose_2d args = {
        /*.IC  =*/ IC,
        /*.IH  =*/ IH,
        /*.IW  =*/ IW,
        /*.KH  =*/ KH,
        /*.KW  =*/ KW,
        /*.OC  =*/ OC,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // Metal requires buffer size to be multiple of 16 bytes
    const size_t smem = GGML_PAD(KW * KH * sizeof(float), 16);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC, KW, KH, 1);

    return 1;
}

int ggml_metal_op_upscale(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float sf0 = (float)ne0/op->src[0]->ne[0];
    float sf1 = (float)ne1/op->src[0]->ne[1];
    float sf2 = (float)ne2/op->src[0]->ne[2];
    float sf3 = (float)ne3/op->src[0]->ne[3];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);

    float poffs = 0.5f;

    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        poffs = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ggml_metal_kargs_upscale args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.sf0   =*/ sf0,
        /*.sf1   =*/ sf1,
        /*.sf2   =*/ sf2,
        /*.sf3   =*/ sf3,
        /*.poffs =*/ poffs,
    };

    auto pipeline = ggml_metal_library_get_pipeline_upscale(lib, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad_reflect_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((const int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((const int32_t *)(op->op_params))[1]
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_arange(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float start;
    float step;

    memcpy(&start, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&step,  ((const int32_t *) op->op_params) + 2, sizeof(float));

    ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    const int nth = std::min(1024, ne0);

    auto pipeline = ggml_metal_library_get_pipeline_arange(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int dim        = op->op_params[0];
    const int max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    auto pipeline = ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const int nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argmax(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_argmax args = {
        /*.ne00 = */ ne00,
        /*.nb01 = */ nb01,
    };

    auto pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    int nth = 32; // SIMD width
    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argsort(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    const int npr = (ne00 + nth - 1)/nth;

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ nth,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    int len = nth;

    while (len < ne00) {
        ggml_metal_op_concurrency_reset(ctx);

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ ne00,
            /*.len   =*/ len,
        };

        // merges per row
        const int nm = (ne00 + 2*len - 1) / (2*len);

        const int nth = std::min(512, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge));

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

int ggml_metal_op_top_k(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    const int npr = (ne00 + nth - 1)/nth;

    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += sizeof(int32_t)*ggml_nelements(op->src[0]);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    const int top_k = ne0;

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ std::min(nth, top_k), // for each block, keep just the top_k indices
    };

    if (npr > 1) {
        args.ne0 = (npr - 1)*args.top_k + std::min(ne00 - (npr - 1)*nth, args.top_k);
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_top_k_merge(lib, op);

    int len = args.top_k;

    while (len < args.ne0) {
        ggml_metal_op_concurrency_reset(ctx);

        // merges per row
        const int nm = (args.ne0 + 2*len - 1) / (2*len);

        const int nth = std::min(512, std::min(len, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge)));

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ args.ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ nm == 1 ? top_k : args.ne0, // the final merge outputs top_k elements
            /*.len   =*/ len,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

int ggml_metal_op_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_tri args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_tri(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_adamw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_adamw args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_sgd(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_sgd args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_count_equal(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    {
        ggml_metal_kargs_memset args = { /*.val =*/ 0 };

        auto pipeline = ggml_metal_library_get_pipeline_memset(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    {
        ggml_metal_kargs_count_equal args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
        };

        auto pipeline = ggml_metal_library_get_pipeline_count_equal(lib, op);

        const size_t smem = pipeline.smem;

        const int nth = 32*pipeline.nsg;

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}
