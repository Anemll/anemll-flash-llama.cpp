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
#include <limits>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

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
        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_REPLAY");
        enabled = (value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0) ? 1 : 0;
    }
    return enabled == 1;
}

static bool ggml_metal_mul_mat_id_experimental_decode_icb_enabled(void) {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DECODE_ICB");
        enabled = (value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0) ? 1 : 0;
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
    const uint64_t replay_hit = g_ggml_metal_mul_mat_id_decode_replay_hit_count.load();
    const uint64_t replay_miss = g_ggml_metal_mul_mat_id_decode_replay_miss_count.load();
    const uint64_t replay_insert = g_ggml_metal_mul_mat_id_decode_replay_insert_count.load();
    const uint64_t replay_clear = g_ggml_metal_mul_mat_id_decode_replay_clear_count.load();
    const uint64_t icb_exec = g_ggml_metal_mul_mat_id_decode_icb_exec_count.load();
    const uint64_t icb_build_fail = g_ggml_metal_mul_mat_id_decode_icb_build_fail_count.load();
    size_t replay_cache_size = 0;

    {
        std::lock_guard<std::mutex> lock(g_ggml_metal_mul_mat_id_decode_replay_mutex);
        replay_cache_size = g_ggml_metal_mul_mat_id_decode_replay_cache.size();
    }

    if (decode_mv == 0 && generic_mv == 0 && generic_mm == 0 && fused_glu == 0 && pair_gate_up == 0 &&
        replay_hit == 0 && replay_miss == 0 && icb_exec == 0 && icb_build_fail == 0) {
        return;
    }

    GGML_LOG_INFO("%s: mul_mat_id dec_mv=%" PRIu64 " pair=%" PRIu64 " gen_mv=%" PRIu64 " gen_mm=%" PRIu64
            " fglu=%" PRIu64 " replay_hit=%" PRIu64 " replay_miss=%" PRIu64
            " replay_ins=%" PRIu64 " replay_clr=%" PRIu64
            " icb_exec=%" PRIu64 " icb_fail=%" PRIu64 " replay_cache=%zu\n",
            __func__, decode_mv, pair_gate_up, generic_mv, generic_mm, fused_glu,
            replay_hit, replay_miss, replay_insert, replay_clear, icb_exec, icb_build_fail, replay_cache_size);
}

void ggml_metal_op_mul_mat_log_stats(void) {
    const uint64_t mv_ext = g_ggml_metal_mul_mat_mv_ext_count.load();
    const uint64_t mv = g_ggml_metal_mul_mat_mv_count.load();
    const uint64_t mm = g_ggml_metal_mul_mat_mm_count.load();
    const uint64_t mm_m5_expert = g_ggml_metal_mul_mat_mm_m5_expert_count.load();

    if (mv_ext == 0 && mv == 0 && mm == 0 && mm_m5_expert == 0) {
        return;
    }

    GGML_LOG_INFO("%s: mul_mat mv_ext=%" PRIu64 " mv=%" PRIu64 " mm=%" PRIu64 " mm_m5_expert=%" PRIu64 "\n",
            __func__, mv_ext, mv, mm, mm_m5_expert);
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
    g_ggml_metal_mul_mat_id_split_trace_count.store(0);
    g_ggml_metal_mul_mat_id_decode_replay_hit_count.store(0);
    g_ggml_metal_mul_mat_id_decode_replay_miss_count.store(0);
    g_ggml_metal_mul_mat_id_decode_replay_insert_count.store(0);
    g_ggml_metal_mul_mat_id_decode_replay_clear_count.store(0);
    g_ggml_metal_mul_mat_id_decode_icb_exec_count.store(0);
    g_ggml_metal_mul_mat_id_decode_icb_build_fail_count.store(0);

    std::lock_guard<std::mutex> lock(g_ggml_metal_mul_mat_id_decode_replay_mutex);
    g_ggml_metal_mul_mat_id_decode_replay_cache.clear();
}

void ggml_metal_op_mul_mat_reset_stats(void) {
    g_ggml_metal_mul_mat_mv_ext_count.store(0);
    g_ggml_metal_mul_mat_mv_count.store(0);
    g_ggml_metal_mul_mat_mm_count.store(0);
    g_ggml_metal_mul_mat_mm_m5_expert_count.store(0);
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
        int  debug_fusion) {
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

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    bool use_fusion;
    bool use_concurrency;
    bool use_capture;

    int debug_graph;
    int debug_fusion;

private:
    ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    // non-empty node indices
    std::vector<int> idxs;
    std::vector<uint8_t> skip_local_idxs;
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

    const int i1 = std::min(ctx->n_nodes(), idx + 6);
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
        int debug_fusion) {
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
        debug_fusion);

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
};

struct ggml_metal_mul_mat_id_pair_gate_up_plan {
    ggml_tensor * peer    = nullptr;
    int peer_local_idx    = -1;
};

static bool ggml_metal_tensor_name_has_token(const ggml_tensor * t, const char * token) {
    return t != nullptr && token != nullptr && t->name[0] != '\0' && strstr(t->name, token) != nullptr;
}

static bool ggml_metal_tensor_is_split_gate(const ggml_tensor * t) {
    return ggml_metal_tensor_name_has_token(t, "ffn_moe_gate") && !ggml_metal_tensor_name_has_token(t, "gate_up");
}

static bool ggml_metal_tensor_is_split_up(const ggml_tensor * t) {
    return ggml_metal_tensor_name_has_token(t, "ffn_moe_up") && !ggml_metal_tensor_name_has_token(t, "gate_up");
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

static bool ggml_metal_mul_mat_id_get_pair_gate_up_plan(
        ggml_metal_op_t ctx,
        int idx,
        ggml_tensor * op,
        ggml_metal_mul_mat_id_pair_gate_up_plan * plan) {
    if (plan == nullptr) {
        return false;
    }

    *plan = {};

    if (!ggml_metal_mul_mat_id_experimental_pair_gate_up_enabled()) {
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

    if (idx + 2 >= ctx->n_nodes()) {
        return false;
    }

    const int peer_local_idx = idx + 1;
    ggml_tensor * peer = ctx->node(peer_local_idx);
    if (peer->op != GGML_OP_MUL_MAT_ID) {
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
        ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "glu-not-found");
        return false;
    }

    if (peer->src[0] == nullptr || peer->src[1] == nullptr ||
        op->src[0] == nullptr || op->src[1] == nullptr ||
        glu->src[0] == nullptr || glu->src[1] == nullptr) {
        ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "missing-src");
        return false;
    }

    if (ggml_get_op_params_i32(glu, 1)) {
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
            ggml_metal_mul_mat_id_trace_split_glu_window(ctx, idx, op, "intervening-overlap");
            return false;
        }
    }

    plan->peer = peer;
    plan->glu = glu;
    plan->peer_local_idx = peer_local_idx;
    plan->glu_local_idx = glu_local_idx;
    plan->n_fuse_contiguous = peer_local_idx == idx + 1 ? 2 : 1;

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

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, &op, use_m5_expert_pipeline);

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
        case GGML_OP_DSV4_ROPE_TAIL:
            {
                n_fuse = ggml_metal_op_dsv4_rope_tail(ctx, idx);
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

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx) {
    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_push(ctx->enc, ggml_op_desc(ctx->node(idx)));
    }

    int res = ggml_metal_op_encode_impl(ctx, idx);
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

    // find the break-even point where the matrix-matrix kernel becomes more efficient compared
    // to the matrix-vector kernel
    const int ne11_mm_min = 8;

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

        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op, use_m5_expert_pipeline);

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

            const bool encoded_pair_gate_up =
                    can_pair_gate_up &&
                    ggml_metal_encode_mul_mat_id_decode_mv_pair(ctx, op, pair_gate_up_plan.peer, expert_ids.data(), ne20);

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
                ggml_metal_op_glu(ctx, split_glu_fuse_plan.glu_local_idx);
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
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

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
        } else if (false) {
            // for small KV caches, we could launch a single workgroup and write the results directly to dst/
            // however, this does not lead to significant improvement, so disabled
            nwg = 1;
            nsg = 4;
        } else {
            nwg = 32;
            nsg = 1;
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
            assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) == 0);

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

    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc_expand(lib, op);
    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) n_elem);
    const int ntg = (n_elem + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         5);
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
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
