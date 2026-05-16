#include "chat.h"
#include "common.h"
#include "arg.h"
#include "console.h"
// #include "log.h"

#include "server-context.h"
#include "server-task.h"
#include "ggml-cpu.h"

#include <array>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>
#include <signal.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

static std::atomic<bool> g_is_interrupted = false;
static bool should_stop() {
    return g_is_interrupted.load();
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (g_is_interrupted.load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    g_is_interrupted.store(true);
}
#endif

namespace {

bool cli_env_enabled(const char * name) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

std::string cli_env_string(const char * name, const char * fallback) {
    const char * v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? std::string(v) : std::string(fallback);
}

int cli_env_int(const char * name, int fallback) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    char * end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

template <size_t N>
std::vector<std::string> cli_enabled_envs(const std::array<const char *, N> & names) {
    std::vector<std::string> out;
    for (const char * name : names) {
        if (cli_env_enabled(name)) {
            out.emplace_back(name);
        }
    }
    return out;
}

json cli_strings_to_json_array(const std::vector<std::string> & values) {
    json out = json::array();
    for (const std::string & value : values) {
        out.push_back(value);
    }
    return out;
}

std::string cli_join_strings(const std::vector<std::string> & values) {
    std::string out;
    for (const std::string & value : values) {
        if (!out.empty()) {
            out += ",";
        }
        out += value;
    }
    return out;
}

bool cli_string_vectors_equal_as_sets(std::vector<std::string> lhs, std::vector<std::string> rhs) {
    std::sort(lhs.begin(), lhs.end());
    std::sort(rhs.begin(), rhs.end());
    return lhs == rhs;
}

std::vector<std::string> cli_dsv4_hotpath_intrusive_flags() {
    static constexpr std::array<const char *, 28> names = {{
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SHADOW",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DUMP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_TRACE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_KERNEL_TRACE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SHADOW",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_TRACE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CANDIDATE_TRACE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_TRACE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SHADOW",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TRACE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_TRACE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_DECODE_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_COMPARE",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_TRACE_STAGES",
        "LLAMA_FLASH_MOE_METAL_STAGE_PROFILE",
        "LLAMA_FLASH_MOE_METAL_STAGE_PROFILE_DETAIL",
    }};
    return cli_enabled_envs(names);
}

std::vector<std::string> cli_dsv4_hotpath_rejected_flags() {
    static constexpr std::array<const char *, 20> names = {{
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_FUSED_COMP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_SWIGLU",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_SHARED_SWIGLU",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_WEIGHTED_DOWN",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_MIXED_ATTN",
        "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE",
    }};
    return cli_enabled_envs(names);
}

std::string cli_dsv4_hotpath_under_test_name() {
    return cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_UNDER_TEST", "");
}

bool cli_dsv4_hotpath_under_test_enabled() {
    return !cli_dsv4_hotpath_under_test_name().empty();
}

bool cli_dsv4_hotpath_under_test_name_allowed(const std::string & name) {
    return name == "ffnmoe_v2_full_consume" ||
        name == "cupd2_fused_comp" ||
        name == "down_sum6" ||
        name == "aohc_single_layer_consume" ||
        name == "aohc_single_layer_skip_generic" ||
        name == "aohc_fused_single_layer_consume" ||
        name == "aohc_backend_fused_single_layer_consume" ||
        name == "aohc_backend_fused_layer_set_consume" ||
        name == "cupd3_tail_single_layer_consume" ||
        name == "cupd3_backend_tail_single_layer_consume" ||
        name == "rmoe_backend_single_layer_consume" ||
        name == "rmoe_backend_single_layer_replace_generic" ||
        name == "rmoe_backend_pair_preserve_single_layer" ||
        name == "rmoe_backend_shared_final_single_layer" ||
        name == "rmoe_generic_lowering_parity" ||
        name == "dsv4_layer_executor_metadata_only" ||
        name == "dsv4_layer_executor_side_probe_hcnorm" ||
        name == "dsv4_layer_executor_side_probe_rmoe" ||
        name == "dsv4_layer_executor_side_probe_aohc" ||
        name == "dsv4_layer_executor_side_probe_compressor_update" ||
        name == "dsv4_layer_executor_side_probe_kv_cache_finalizer";
}

std::vector<std::string> cli_dsv4_hotpath_under_test_rejected_flags(const std::string & name) {
    if (name == "ffnmoe_v2_full_consume") {
        return {"LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME"};
    }
    if (name == "cupd2_fused_comp") {
        return {"LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP"};
    }
    if (name == "down_sum6") {
        return {"LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6"};
    }
    if (name == "aohc_single_layer_consume") {
        return {"LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME"};
    }
    if (name == "aohc_single_layer_skip_generic") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC",
        };
    }
    if (name == "aohc_fused_single_layer_consume" ||
            name == "aohc_backend_fused_single_layer_consume" ||
            name == "aohc_backend_fused_layer_set_consume") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME",
        };
    }
    if (name == "cupd3_tail_single_layer_consume") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL",
        };
    }
    if (name == "cupd3_backend_tail_single_layer_consume") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME",
        };
    }
    if (name == "rmoe_backend_single_layer_consume") {
        if (!cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BRANCH_MODE", "").empty()) {
            return {};
        }
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
        };
    }
    if (name == "rmoe_backend_single_layer_replace_generic") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC",
        };
    }
    if (name == "rmoe_backend_pair_preserve_single_layer") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
        };
    }
    if (name == "rmoe_backend_shared_final_single_layer") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
        };
    }
    if (name == "rmoe_generic_lowering_parity") {
        return {};
    }
    if (name == "dsv4_layer_executor_metadata_only") {
        return {};
    }
    if (name == "dsv4_layer_executor_side_probe_hcnorm") {
        return {};
    }
    if (name == "dsv4_layer_executor_side_probe_rmoe") {
        return {};
    }
    if (name == "dsv4_layer_executor_side_probe_aohc") {
        return {};
    }
    if (name == "dsv4_layer_executor_side_probe_compressor_update") {
        return {};
    }
    if (name == "dsv4_layer_executor_side_probe_kv_cache_finalizer") {
        return {};
    }
    return {};
}

std::vector<std::string> cli_dsv4_hotpath_under_test_required_flags(const std::string & name) {
    if (name == "ffnmoe_v2_full_consume") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME",
        };
    }
    if (name == "cupd2_fused_comp") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP",
        };
    }
    if (name == "down_sum6") {
        return {"LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6"};
    }
    if (name == "aohc_single_layer_consume") {
        return {"LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME"};
    }
    if (name == "aohc_single_layer_skip_generic") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC",
        };
    }
    if (name == "aohc_fused_single_layer_consume" ||
            name == "aohc_backend_fused_single_layer_consume" ||
            name == "aohc_backend_fused_layer_set_consume") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME",
        };
    }
    if (name == "cupd3_tail_single_layer_consume") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL",
        };
    }
    if (name == "cupd3_backend_tail_single_layer_consume") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME",
        };
    }
    if (name == "rmoe_backend_single_layer_consume") {
        if (!cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BRANCH_MODE", "").empty()) {
            return {
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP",
            };
        }
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
        };
    }
    if (name == "rmoe_backend_single_layer_replace_generic") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC",
        };
    }
    if (name == "rmoe_backend_pair_preserve_single_layer") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_PAIR_PRESERVE",
        };
    }
    if (name == "rmoe_backend_shared_final_single_layer") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SHARED_FINAL_ONLY",
        };
    }
    if (name == "rmoe_generic_lowering_parity") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY",
        };
    }
    if (name == "dsv4_layer_executor_metadata_only") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
        };
    }
    if (name == "dsv4_layer_executor_side_probe_hcnorm") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE",
        };
    }
    if (name == "dsv4_layer_executor_side_probe_rmoe") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE",
        };
    }
    if (name == "dsv4_layer_executor_side_probe_aohc") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE",
        };
    }
    if (name == "dsv4_layer_executor_side_probe_compressor_update") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE",
        };
    }
    if (name == "dsv4_layer_executor_side_probe_kv_cache_finalizer") {
        return {
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE",
        };
    }
    return {};
}

std::vector<std::string> cli_dsv4_hotpath_under_test_active_flags(const std::string & name) {
    const std::vector<std::string> required = cli_dsv4_hotpath_under_test_required_flags(name);
    std::vector<std::string> out;
    for (const std::string & flag : required) {
        if (cli_env_enabled(flag.c_str())) {
            out.push_back(flag);
        }
    }
    return out;
}

std::string cli_dsv4_hotpath_neutral_guard_error() {
    const std::vector<std::string> intrusive = cli_dsv4_hotpath_intrusive_flags();
    if (!intrusive.empty()) {
        return "intrusive_flags=" + cli_join_strings(intrusive);
    }

    const std::string under_test = cli_dsv4_hotpath_under_test_name();
    const bool has_under_test = !under_test.empty();
    if (cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_ALLOW_REJECTED_UNDER_TEST") &&
            !has_under_test) {
        return "allow_rejected_under_test_without_name";
    }

    const std::vector<std::string> rejected = cli_dsv4_hotpath_rejected_flags();
    if (!has_under_test) {
        if (!rejected.empty()) {
            return "rejected_paths=" + cli_join_strings(rejected);
        }
        return "";
    }

    if (!cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_ALLOW_REJECTED_UNDER_TEST")) {
        return "under_test_requires_allow_rejected_under_test";
    }
    if (cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_REJECTED_UNDER_TEST_ACK", "") != "not_accepted") {
        return "under_test_requires_ack_not_accepted";
    }
    if (!cli_dsv4_hotpath_under_test_name_allowed(under_test)) {
        return "unsupported_under_test=" + under_test;
    }
    const std::vector<std::string> expected_rejected = cli_dsv4_hotpath_under_test_rejected_flags(under_test);
    if (!cli_string_vectors_equal_as_sets(rejected, expected_rejected)) {
        return "under_test_rejected_flags_mismatch expected=" +
            cli_join_strings(expected_rejected) + " actual=" + cli_join_strings(rejected);
    }
    const std::vector<std::string> required = cli_dsv4_hotpath_under_test_required_flags(under_test);
    const std::vector<std::string> active = cli_dsv4_hotpath_under_test_active_flags(under_test);
    if (!cli_string_vectors_equal_as_sets(active, required)) {
        return "under_test_required_flags_mismatch expected=" +
            cli_join_strings(required) + " actual=" + cli_join_strings(active);
    }
    return "";
}

std::string cli_json_safe_text(std::string text) {
    text.resize(validate_utf8(text));
    return text;
}

std::string oracle_sanitize_name(const std::string & name) {
    std::string out;
    out.reserve(name.size());
    for (const char ch : name) {
        if ((ch >= 'a' && ch <= 'z') ||
            (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') ||
            ch == '_' || ch == '-' || ch == '.') {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
    }
    return out;
}

std::pair<std::string, int> oracle_parse_name_layer(const std::string & raw_name) {
    const size_t dash = raw_name.rfind('-');
    if (dash != std::string::npos && dash + 1 < raw_name.size()) {
        bool all_digits = true;
        for (size_t i = dash + 1; i < raw_name.size(); ++i) {
            if (raw_name[i] < '0' || raw_name[i] > '9') {
                all_digits = false;
                break;
            }
        }
        if (all_digits) {
            return { raw_name.substr(0, dash), std::atoi(raw_name.c_str() + dash + 1) };
        }
    }
    return { raw_name, -1 };
}

bool oracle_wants_base_name(const std::string_view base_name) {
    if (const char * filter = std::getenv("LLAMA_ORACLE_TENSORS")) {
        if (filter[0] != '\0') {
            std::string_view list(filter);
            size_t pos = 0;
            while (pos <= list.size()) {
                const size_t comma = list.find(',', pos);
                const size_t end = comma == std::string_view::npos ? list.size() : comma;
                std::string_view item = list.substr(pos, end - pos);
                while (!item.empty() && item.front() == ' ') {
                    item.remove_prefix(1);
                }
                while (!item.empty() && item.back() == ' ') {
                    item.remove_suffix(1);
                }
                if (item == base_name) {
                    return true;
                }
                if (comma == std::string_view::npos) {
                    break;
                }
                pos = comma + 1;
            }
            return false;
        }
    }

    return base_name == "embd" ||
           base_name == "hc_attn_pre" ||
           base_name == "attn_norm" ||
           base_name == "q_pe" ||
           base_name == "k_pe" ||
           base_name == "kv_cmpr" ||
           base_name == "q_nope_absorbed_perm" ||
           base_name == "attn_out" ||
           base_name == "kqv_out" ||
           base_name == "ffn_inp" ||
           base_name == "ffn_norm" ||
           base_name == "ffn_moe_topk" ||
           base_name == "ffn_moe_topk_reduced" ||
           base_name == "ffn_moe_hash_topk" ||
           base_name == "ffn_shexp" ||
           base_name == "ffn_out" ||
           base_name == "l_out" ||
           base_name == "result_norm" ||
           base_name == "result_output";
}

struct oracle_tensor_record {
    int eval_index = 0;
    std::string phase;
    std::string name;
    int layer = -1;
    std::array<int64_t, 4> ne = {0, 0, 0, 0};
    std::string dtype;
    std::string file;
};

struct oracle_logits_record {
    int eval_index = 0;
    std::string phase;
    std::vector<int32_t> token_ids;
    std::vector<float> logits;
};

class cli_oracle_dump {
public:
    cli_oracle_dump(const std::filesystem::path & out_dir, int topk)
        : out_dir_(out_dir), tensor_dir_(out_dir / "tensors"), logits_topk_(std::max(1, topk)) {
        std::filesystem::create_directories(tensor_dir_);
    }

    bool enabled() const {
        return !out_dir_.empty();
    }

    void set_prompt(const std::string & prompt, std::vector<llama_token> prompt_ids) {
        prompt_ = prompt;
        prompt_ids_ = std::move(prompt_ids);
    }

    void set_model_path(const std::string & model_path) {
        model_path_ = model_path;
    }

    bool wants_tensor(const ggml_tensor * t) const {
        if (t == nullptr || t->name[0] == '\0') {
            return false;
        }
        const auto [base_name, _layer] = oracle_parse_name_layer(std::string(t->name));
        return oracle_wants_base_name(base_name);
    }

    bool handle_tensor(const ggml_tensor * t) {
        if (t == nullptr || !wants_tensor(t)) {
            return true;
        }

        const std::string raw_name(t->name);
        const auto [base_name, layer_index] = oracle_parse_name_layer(raw_name);
        if (base_name == "result_output") {
            capture_topk_logits(t);
            ++eval_index_;
            return true;
        }

        const auto values = flatten_tensor_f32(t);
        const std::string file_name = tensor_file_name(raw_name, tensor_records_.size());
        const std::filesystem::path file_path = tensor_dir_ / file_name;
        std::ofstream fout(file_path, std::ios::binary);
        if (!fout) {
            throw std::runtime_error("failed to open oracle tensor output file: " + file_path.string());
        }
        fout.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
        fout.close();

        oracle_tensor_record record;
        record.eval_index = eval_index_;
        record.phase = phase_for_eval(eval_index_);
        record.name = base_name;
        record.layer = layer_index;
        for (int i = 0; i < 4; ++i) {
            record.ne[i] = t->ne[i];
        }
        record.dtype = "f32";
        record.file = std::string("tensors/") + file_name;
        tensor_records_.push_back(std::move(record));
        return true;
    }

    void finish() const {
        if (!enabled()) {
            return;
        }

        json manifest;
        manifest["format"] = "llama-cli-oracle-v1";
        manifest["prompt"] = prompt_;
        manifest["prompt_ids"] = prompt_ids_;
        manifest["prompt_token_count"] = prompt_ids_.size();
        manifest["model_path"] = model_path_;
        manifest["logits_topk"] = logits_topk_;
        manifest["tensor_names"] = json::array({
            "embd",
            "hc_attn_pre",
            "attn_norm",
            "q_pe",
            "k_pe",
            "kv_cmpr",
            "q_nope_absorbed_perm",
            "attn_out",
            "kqv_out",
            "ffn_inp",
            "ffn_norm",
            "ffn_moe_topk",
            "ffn_moe_topk_reduced",
            "ffn_moe_hash_topk",
            "ffn_shexp",
            "ffn_out",
            "l_out",
            "result_norm",
        });
        manifest["records"] = json::array();
        for (const auto & record : tensor_records_) {
            manifest["records"].push_back({
                {"record_index", manifest["records"].size()},
                {"eval_index", record.eval_index},
                {"phase", record.phase},
                {"name", record.name},
                {"layer", record.layer},
                {"ne", {record.ne[0], record.ne[1], record.ne[2], record.ne[3]}},
                {"dtype", record.dtype},
                {"file", record.file},
            });
        }
        manifest["logits"] = json::array();
        for (const auto & record : logits_records_) {
            manifest["logits"].push_back({
                {"eval_index", record.eval_index},
                {"phase", record.phase},
                {"token_ids", record.token_ids},
                {"logits", record.logits},
            });
        }

        std::ofstream fout(out_dir_ / "manifest.json", std::ios::binary);
        if (!fout) {
            throw std::runtime_error("failed to open oracle manifest output file");
        }
        fout << manifest.dump(2) << "\n";
    }

private:
    std::filesystem::path out_dir_;
    std::filesystem::path tensor_dir_;
    int logits_topk_ = 32;
    int eval_index_ = 0;
    std::string prompt_;
    std::string model_path_;
    std::vector<llama_token> prompt_ids_;
    std::vector<oracle_tensor_record> tensor_records_;
    std::vector<oracle_logits_record> logits_records_;

    std::string phase_for_eval(int eval_index) const {
        return eval_index < static_cast<int>(prompt_ids_.size()) ? "prefill" : "decode";
    }

    std::string tensor_file_name(const std::string & name, size_t record_index) const {
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "%06zu_%06d_", record_index, eval_index_);
        return std::string(prefix) + oracle_sanitize_name(name) + ".bin";
    }

    static float read_tensor_value_f32(
        const uint8_t * data,
        ggml_type type,
        const size_t * nb,
        size_t i0,
        size_t i1,
        size_t i2,
        size_t i3) {
        const size_t offset = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
        switch (type) {
            case GGML_TYPE_F32:
                return *(const float *) &data[offset];
            case GGML_TYPE_F16:
                return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[offset]);
            case GGML_TYPE_BF16:
                return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[offset]);
            case GGML_TYPE_I32:
                return (float) *(const int32_t *) &data[offset];
            case GGML_TYPE_I16:
                return (float) *(const int16_t *) &data[offset];
            case GGML_TYPE_I8:
                return (float) *(const int8_t *) &data[offset];
            default:
                throw std::runtime_error("unsupported oracle tensor dtype: " + std::string(ggml_type_name(type)));
        }
    }

    static std::vector<float> flatten_tensor_f32(const ggml_tensor * t) {
        const int64_t n0 = std::max<int64_t>(1, t->ne[0]);
        const int64_t n1 = std::max<int64_t>(1, t->ne[1]);
        const int64_t n2 = std::max<int64_t>(1, t->ne[2]);
        const int64_t n3 = std::max<int64_t>(1, t->ne[3]);
        const bool is_host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);
        std::vector<uint8_t> host_copy;
        if (!is_host) {
            host_copy.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, host_copy.data(), 0, host_copy.size());
        }
        const uint8_t * data = is_host ? (const uint8_t *) t->data : host_copy.data();
        if (data == nullptr) {
            const char * tensor_name = t->name[0] != '\0' ? t->name : "<unnamed>";
            throw std::runtime_error("oracle tensor has no readable data for " + std::string(tensor_name));
        }
        std::vector<float> values;
        values.reserve(static_cast<size_t>(n0 * n1 * n2 * n3));
        for (int64_t i3 = 0; i3 < n3; ++i3) {
            for (int64_t i2 = 0; i2 < n2; ++i2) {
                for (int64_t i1 = 0; i1 < n1; ++i1) {
                    for (int64_t i0 = 0; i0 < n0; ++i0) {
                        values.push_back(read_tensor_value_f32(
                            data,
                            t->type,
                            t->nb,
                            static_cast<size_t>(i0),
                            static_cast<size_t>(i1),
                            static_cast<size_t>(i2),
                            static_cast<size_t>(i3)));
                    }
                }
            }
        }
        return values;
    }

    void capture_topk_logits(const ggml_tensor * t) {
        const auto values = flatten_tensor_f32(t);
        std::vector<int32_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        const size_t keep = std::min<size_t>(static_cast<size_t>(logits_topk_), indices.size());
        if (keep == 0) {
            return;
        }
        std::partial_sort(
            indices.begin(),
            indices.begin() + static_cast<ptrdiff_t>(keep),
            indices.end(),
            [&](const int32_t lhs, const int32_t rhs) {
                return values[static_cast<size_t>(lhs)] > values[static_cast<size_t>(rhs)];
            }
        );

        oracle_logits_record record;
        record.eval_index = eval_index_;
        record.phase = phase_for_eval(eval_index_);
        record.token_ids.reserve(keep);
        record.logits.reserve(keep);
        for (size_t i = 0; i < keep; ++i) {
            const int32_t token_id = indices[i];
            record.token_ids.push_back(token_id);
            record.logits.push_back(values[static_cast<size_t>(token_id)]);
        }
        logits_records_.push_back(std::move(record));
    }
};

class cli_dsv4_value_probe {
public:
    bool enabled() const {
        return cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_VALUE_PROBE") ||
               cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_DECODE_COMPRESS_INTERNAL_PROBE");
    }

    bool wants_tensor(const ggml_tensor * t) const {
        if (!enabled() || t == nullptr || t->name[0] == '\0') {
            return false;
        }
        const std::string_view name(t->name);
        return name.rfind("dsv4_cupd3_value_probe_", 0) == 0 ||
               name.rfind("dsv4_dcomp_internal_probe_", 0) == 0 ||
               name.rfind("dsv4_dci_", 0) == 0;
    }

    bool handle_tensor(const ggml_tensor * t) {
        if (t == nullptr || !wants_tensor(t)) {
            return true;
        }
        const parsed_name parsed = parse_name(t->name);
        if (!parsed.valid) {
            return true;
        }

        tensor_snapshot snapshot;
        snapshot.name = t->name;
        snapshot.stage = parsed.stage;
        snapshot.stream = parsed.stream;
        snapshot.label = parsed.label;
        snapshot.layer = parsed.layer;
        snapshot.token = parsed.token;
        snapshot.dtype = ggml_type_name(t->type);
        for (int i = 0; i < 4; ++i) {
            snapshot.ne[i] = t->ne[i];
            snapshot.nb[i] = t->nb[i];
        }
        snapshot.bytes = read_tensor_bytes(t);
        if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16) {
            snapshot.values = flatten_tensor_f32(t, snapshot.bytes);
        }

        auto & pair = pending_[parsed.key()];
        if (parsed.role == "ref") {
            pair.ref = std::move(snapshot);
            pair.has_ref = true;
        } else if (parsed.role == "cand") {
            pair.cand = std::move(snapshot);
            pair.has_cand = true;
        }
        if (pair.has_ref && pair.has_cand && !pair.reported) {
            report_pair(pair.ref, pair.cand);
            pair.reported = true;
        }
        return true;
    }

private:
    struct parsed_name {
        bool valid = false;
        std::string role;
        std::string stage;
        std::string stream;
        std::string label;
        int layer = -1;
        int64_t token = -1;

        std::string key() const {
            return label + "|" + stage + "|" + stream + "|l" + std::to_string(layer) + "|p" + std::to_string(token);
        }
    };

    struct tensor_snapshot {
        std::string name;
        std::string stage;
        std::string stream;
        std::string label;
        int layer = -1;
        int64_t token = -1;
        std::string dtype;
        std::array<int64_t, 4> ne = {0, 0, 0, 0};
        std::array<size_t, 4> nb = {0, 0, 0, 0};
        std::vector<uint8_t> bytes;
        std::vector<float> values;
    };

    struct tensor_pair {
        tensor_snapshot ref;
        tensor_snapshot cand;
        bool has_ref = false;
        bool has_cand = false;
        bool reported = false;
    };

    std::unordered_map<std::string, tensor_pair> pending_;

    static parsed_name parse_name(const std::string & raw_name) {
        parsed_name out;
        std::string_view prefix;
        std::string_view label;
        if (std::string_view(raw_name).rfind("dsv4_cupd3_value_probe_", 0) == 0) {
            prefix = "dsv4_cupd3_value_probe_";
            label = "dsv4_cupd3_value_probe";
        } else if (std::string_view(raw_name).rfind("dsv4_dcomp_internal_probe_", 0) == 0) {
            prefix = "dsv4_dcomp_internal_probe_";
            label = "dsv4_dcomp_internal_probe";
        } else if (std::string_view(raw_name).rfind("dsv4_dci_", 0) == 0) {
            prefix = "dsv4_dci_";
            label = "dsv4_dcomp_internal_probe";
        } else {
            return out;
        }
        const std::string rest = raw_name.substr(prefix.size());
        std::vector<std::string> parts;
        size_t pos = 0;
        while (pos <= rest.size()) {
            const size_t dash = rest.find('-', pos);
            const size_t end = dash == std::string::npos ? rest.size() : dash;
            parts.push_back(rest.substr(pos, end - pos));
            if (dash == std::string::npos) {
                break;
            }
            pos = dash + 1;
        }
        if (parts.size() != 5 || (parts[0] != "ref" && parts[0] != "cand") ||
                parts[3].size() < 2 || parts[3][0] != 'l' ||
                parts[4].size() < 2 || parts[4][0] != 'p') {
            return out;
        }
        out.valid = true;
        out.role = parts[0];
        out.stage = parts[1] == "normw" ? "norm_weighted" : parts[1];
        out.stream = parts[2];
        out.label = std::string(label);
        out.layer = std::atoi(parts[3].c_str() + 1);
        out.token = std::atoll(parts[4].c_str() + 1);
        return out;
    }

    static std::vector<uint8_t> read_tensor_bytes(const ggml_tensor * t) {
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        if (bytes.empty()) {
            return bytes;
        }
        const bool is_host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);
        if (is_host) {
            if (t->data != nullptr) {
                std::memcpy(bytes.data(), t->data, bytes.size());
            }
        } else {
            ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        }
        return bytes;
    }

    static float read_tensor_value_f32(
            const uint8_t * data,
            ggml_type type,
            const size_t * nb,
            size_t i0,
            size_t i1,
            size_t i2,
            size_t i3) {
        const size_t offset = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
        switch (type) {
            case GGML_TYPE_F32:
                return *(const float *) &data[offset];
            case GGML_TYPE_F16:
                return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[offset]);
            case GGML_TYPE_BF16:
                return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[offset]);
            default:
                return NAN;
        }
    }

    static std::vector<float> flatten_tensor_f32(const ggml_tensor * t, const std::vector<uint8_t> & bytes) {
        std::vector<float> values;
        if (bytes.empty()) {
            return values;
        }
        const int64_t n0 = std::max<int64_t>(1, t->ne[0]);
        const int64_t n1 = std::max<int64_t>(1, t->ne[1]);
        const int64_t n2 = std::max<int64_t>(1, t->ne[2]);
        const int64_t n3 = std::max<int64_t>(1, t->ne[3]);
        values.reserve(static_cast<size_t>(n0 * n1 * n2 * n3));
        for (int64_t i3 = 0; i3 < n3; ++i3) {
            for (int64_t i2 = 0; i2 < n2; ++i2) {
                for (int64_t i1 = 0; i1 < n1; ++i1) {
                    for (int64_t i0 = 0; i0 < n0; ++i0) {
                        values.push_back(read_tensor_value_f32(
                            bytes.data(),
                            t->type,
                            t->nb,
                            static_cast<size_t>(i0),
                            static_cast<size_t>(i1),
                            static_cast<size_t>(i2),
                            static_cast<size_t>(i3)));
                    }
                }
            }
        }
        return values;
    }

    static std::string shape_csv(const tensor_snapshot & s) {
        std::ostringstream out;
        out << "[" << s.ne[0] << "," << s.ne[1] << "," << s.ne[2] << "," << s.ne[3] << "]";
        return out.str();
    }

    static std::string stride_csv(const tensor_snapshot & s) {
        std::ostringstream out;
        out << "[" << s.nb[0] << "," << s.nb[1] << "," << s.nb[2] << "," << s.nb[3] << "]";
        return out.str();
    }

    static std::string first_bytes_hex(const std::vector<uint8_t> & bytes) {
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        const size_t n = std::min<size_t>(32, bytes.size());
        for (size_t i = 0; i < n; ++i) {
            if (i != 0) {
                out << "";
            }
            out << std::setw(2) << static_cast<unsigned>(bytes[i]);
        }
        return out.str();
    }

    static void report_pair(const tensor_snapshot & ref, const tensor_snapshot & cand) {
        const size_t n = std::min(ref.values.size(), cand.values.size());
        double sum_sq = 0.0;
        float max_abs = 0.0f;
        size_t first_bad = SIZE_MAX;
        for (size_t i = 0; i < n; ++i) {
            const float diff = std::fabs(ref.values[i] - cand.values[i]);
            if (diff > max_abs) {
                max_abs = diff;
            }
            if (diff != 0.0f && first_bad == SIZE_MAX) {
                first_bad = i;
            }
            sum_sq += double(diff) * double(diff);
        }
        const double rms = n > 0 ? std::sqrt(sum_sq / double(n)) : 0.0;
        const bool value_exact = n == ref.values.size() && n == cand.values.size() && max_abs == 0.0f;
        const bool byte_exact = ref.bytes == cand.bytes;
        const float generic_value = first_bad != SIZE_MAX ? ref.values[first_bad] : (n > 0 ? ref.values[0] : 0.0f);
        const float backend_value = first_bad != SIZE_MAX ? cand.values[first_bad] : (n > 0 ? cand.values[0] : 0.0f);
        std::fprintf(stderr,
                "%s: token=%lld layer=%d stream=%s stage=%s"
                " shape=%s dtype=%s stride=%s byte_count=%zu"
                " value_exact=%d byte_exact=%d max_abs=%.9g rms=%.9g over_tol=%d"
                " first_bad_index=%lld generic_value=%.9g backend_value=%.9g"
                " generic_bytes_first_32=%s backend_bytes_first_32=%s\n",
                ref.label.empty() ? "dsv4_cupd3_value_probe" : ref.label.c_str(),
                (long long) ref.token,
                ref.layer,
                ref.stream.c_str(),
                ref.stage.c_str(),
                shape_csv(ref).c_str(),
                ref.dtype.c_str(),
                stride_csv(ref).c_str(),
                ref.bytes.size(),
                value_exact ? 1 : 0,
                byte_exact ? 1 : 0,
                max_abs,
                rms,
                max_abs == 0.0f ? 0 : 1,
                first_bad != SIZE_MAX ? (long long) first_bad : -1LL,
                generic_value,
                backend_value,
                first_bytes_hex(ref.bytes).c_str(),
                first_bytes_hex(cand.bytes).c_str());
    }
};

class cli_dsv4_rmoe_replace_dump {
public:
    cli_dsv4_rmoe_replace_dump(const std::filesystem::path & out_path)
        : out_path_(out_path),
          mode_(cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC") ? "replace" :
                (cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SHADOW") ? "shadow" : "baseline")),
          target_layer_(cli_env_int("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", 0)),
          token_min_(cli_env_int("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN", 1)),
          token_max_(cli_env_int("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX", 80)),
          gate_up_byte_dump_(cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_GATE_UP_BYTE_DUMP")),
          swiglu_byte_dump_(cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_BYTE_DUMP")),
          swiglu_stage_dump_(cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_STAGE_DUMP")),
          downstream_dump_(cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_DOWNSTREAM_DUMP")),
          result_chain_dump_(cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_RESULT_CHAIN_DUMP")) {
        if (out_path_.has_parent_path()) {
            std::filesystem::create_directories(out_path_.parent_path());
        }
        fout_.open(out_path_, std::ios::binary);
        if (!fout_) {
            throw std::runtime_error("failed to open routed-MoE replace dump file: " + out_path_.string());
        }
    }

    bool enabled() const {
        return fout_.is_open();
    }

    bool wants_tensor(const ggml_tensor * t) const {
        if (!enabled() || t == nullptr || t->name[0] == '\0') {
            return false;
        }
        const auto [base_name, layer] = oracle_parse_name_layer(std::string(t->name));
        if ((downstream_dump_ || result_chain_dump_) && (base_name == "result_hc" || base_name == "result_norm")) {
            return is_decode_tensor(base_name, t);
        }
        if (layer != target_layer_) {
            return false;
        }
        if (base_name == "ffn_norm" || base_name == "ffn_out" || base_name == "hc_ffn_post" ||
                base_name.rfind("dsv4_rmoe_downstream_", 0) == 0 ||
                base_name.rfind("dsv4_rmoe_result_chain_", 0) == 0 ||
                base_name.rfind("dsv4_rmoe_dump_", 0) == 0 ||
                base_name == "dsv4_moe_topk_ids" || base_name == "dsv4_moe_topk_weights" ||
                base_name == "ffn_moe_hash_topk" || base_name == "ffn_moe_topk") {
            return is_decode_tensor(base_name, t);
        }
        return false;
    }

    bool handle_tensor(const ggml_tensor * t) {
        if (t == nullptr || !wants_tensor(t)) {
            return true;
        }
        const auto [base_name, layer] = oracle_parse_name_layer(std::string(t->name));
        const int record_layer = ((downstream_dump_ || result_chain_dump_) &&
                (base_name == "result_hc" || base_name == "result_norm")) ? target_layer_ : layer;
        ensure_record(record_layer);

        if (base_name == "ffn_norm") {
            current_["ffn_input_hash"] = tensor_hash_hex(t);
            current_["ffn_input"] = tensor_stats_json(t);
        } else if (base_name == "dsv4_moe_topk_ids" || base_name == "ffn_moe_hash_topk" || base_name == "ffn_moe_topk") {
            current_["topk_ids"] = tensor_i32_values_json(t);
            current_["topk_ids_tensor"] = std::string(t->name);
        } else if (base_name == "dsv4_moe_topk_weights") {
            current_["topk_weights"] = tensor_f32_values_json(t, 6);
            current_["topk_weights_tensor"] = std::string(t->name);
        } else if (base_name.rfind("dsv4_rmoe_dump_", 0) == 0) {
            std::string stage = base_name.substr(std::strlen("dsv4_rmoe_dump_"));
            const bool backend_final_stage = stage == "backend_final_ffn";
            if (backend_final_stage) {
                stage = "final_ffn";
            }
            current_["stages"][stage] = tensor_stats_json(t);
            if (gate_up_byte_dump_ && (stage == "gate" || stage == "up")) {
                current_["gate_up_byte_dump"][stage] = tensor_slot_stats_json(t, stage);
            }
            if (swiglu_byte_dump_ && (stage == "gate" || stage == "up" || stage == "swiglu")) {
                current_["swiglu_byte_dump"][stage] = tensor_slot_stats_json(t, stage);
                if (stage == "swiglu") {
                    current_["swiglu_parent"] = tensor_parent_json(t);
                }
            }
            if (swiglu_stage_dump_ && is_swiglu_stage_dump_stage(stage)) {
                current_["swiglu_stage_dump"][stage] = tensor_slot_stats_json(t, stage);
                current_["swiglu_stage_parent"][stage] = tensor_parent_json(t);
            }
            if (swiglu_stage_dump_ && stage == "swiglu") {
                json & stage_dump = current_["swiglu_stage_dump"];
                if (!stage_dump.contains("mul_out")) {
                    stage_dump["mul_out"] = tensor_slot_stats_json(t, "mul_out");
                    current_["swiglu_stage_parent"]["mul_out"] = tensor_parent_json(t);
                }
            }
            if (mode_ == "shadow" && backend_final_stage) {
                flush_record();
            }
        } else if (base_name.rfind("dsv4_rmoe_downstream_", 0) == 0) {
            const std::string stage = base_name.substr(std::strlen("dsv4_rmoe_downstream_"));
            const json stats = tensor_stats_json(t);
            current_["downstream"][stage] = stats;
            if (stage == "hc_post_input") {
                current_["hc_ffn_post_input"] = stats;
                current_["hc_ffn_post_input_hash"] = stats["hash"];
            } else if (stage == "hc_post_output") {
                current_["hc_ffn_post_output"] = stats;
                current_["hc_ffn_post_output_hash"] = stats["hash"];
            }
        } else if (base_name.rfind("dsv4_rmoe_result_chain_", 0) == 0) {
            const std::string stage = base_name.substr(std::strlen("dsv4_rmoe_result_chain_"));
            current_["result_chain"][stage] = tensor_stats_json(t);
        } else if (base_name == "ffn_out") {
            const json stats = tensor_stats_json(t);
            current_["backend_or_generic_final_hash"] = stats["hash"];
            current_["hc_ffn_post_input_hash"] = stats["hash"];
            current_["final"] = stats;
            current_["hc_ffn_post_input"] = stats;
            current_["first_8_final_values"] = stats["first_8_values"];
            current_["hc_ffn_post_output_unavailable"] = true;
            if (mode_ != "shadow" && !downstream_dump_ && !result_chain_dump_) {
                flush_record();
            }
        } else if (base_name == "hc_ffn_post") {
            if (!current_active_) {
                ensure_record(layer);
            }
            const json stats = tensor_stats_json(t);
            current_["hc_ffn_post_output_hash"] = stats["hash"];
            current_["hc_ffn_post_output"] = stats;
            current_["first_8_hc_post_values"] = stats["first_8_values"];
            if (!downstream_dump_ && !result_chain_dump_) {
                flush_record();
            }
        } else if ((downstream_dump_ || result_chain_dump_) && (base_name == "result_hc" || base_name == "result_norm")) {
            if (!current_active_) {
                ensure_record(target_layer_);
            }
            const json stats = tensor_stats_json(t);
            if (base_name == "result_hc") {
                current_["downstream"]["result_hc"] = stats;
                if (result_chain_dump_) {
                    current_["result_chain"]["result_hc"] = stats;
                }
            } else {
                current_["downstream"]["logits_input"] = stats;
                if (result_chain_dump_) {
                    current_["result_chain"]["result_norm"] = stats;
                    current_["result_chain"]["logits_input"] = stats;
                }
                flush_record();
            }
        }
        return true;
    }

    void finish() {
        if (current_active_) {
            flush_record();
        }
        if (fout_.is_open()) {
            fout_.flush();
        }
    }

private:
    std::filesystem::path out_path_;
    std::ofstream fout_;
    std::string mode_;
    int target_layer_ = 0;
    int token_min_ = 1;
    int token_max_ = 80;
    int token_ = 1;
    bool gate_up_byte_dump_ = false;
    bool swiglu_byte_dump_ = false;
    bool swiglu_stage_dump_ = false;
    bool downstream_dump_ = false;
    bool result_chain_dump_ = false;
    bool current_active_ = false;
    json current_;

    void ensure_record(int layer) {
        if (current_active_) {
            return;
        }
        current_ = json::object();
        current_["format"] = "dsv4-rmoe-replace-dump-v1";
        current_["token"] = token_;
        current_["layer"] = layer;
        current_["mode"] = mode_;
        current_["swiglu_mode"] = cli_env_string(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_MODE",
                "backend_scratch");
        current_["swiglu_formula"] = cli_env_string(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_FORMULA",
                "backend_current");
        current_active_ = true;
    }

    void flush_record() {
        if (!current_active_) {
            return;
        }
        const int token = current_.value("token", token_);
        if (token >= token_min_ && token <= token_max_) {
            fout_ << current_.dump() << "\n";
        }
        current_ = json::object();
        current_active_ = false;
        ++token_;
    }

    static bool is_decode_tensor(const std::string & base_name, const ggml_tensor * t) {
        if (base_name == "dsv4_moe_topk_weights") {
            return (t->ne[0] == 1 && t->ne[1] == 6) || (t->ne[0] == 6 && t->ne[1] == 1);
        }
        if (base_name.rfind("dsv4_rmoe_dump_", 0) == 0) {
            return t->ne[1] == 1 || t->ne[1] == 6 || t->ne[2] == 6;
        }
        if (base_name == "dsv4_moe_topk_ids" || base_name == "ffn_moe_hash_topk" || base_name == "ffn_moe_topk") {
            return t->ne[0] == 6 && t->ne[1] == 1;
        }
        return t->ne[1] == 1;
    }

    static bool is_swiglu_stage_dump_stage(const std::string & stage) {
        return stage == "gate_raw" ||
               stage == "gate_clamp_pre_silu" ||
               stage == "silu_out" ||
               stage == "silu_clamp_post" ||
               stage == "up_raw" ||
               stage == "up_clamp_pre_mul" ||
               stage == "mul_out";
    }

    static std::vector<uint8_t> read_tensor_bytes(const ggml_tensor * t) {
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        if (bytes.empty()) {
            return bytes;
        }
        const bool is_host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);
        if (is_host) {
            if (t->data != nullptr) {
                std::memcpy(bytes.data(), t->data, bytes.size());
            }
        } else {
            ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        }
        return bytes;
    }

    static std::string tensor_hash_hex(const ggml_tensor * t) {
        const std::vector<uint8_t> bytes = read_tensor_bytes(t);
        uint64_t hash = 1469598103934665603ull;
        for (uint8_t b : bytes) {
            hash ^= uint64_t(b);
            hash *= 1099511628211ull;
        }
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16) << hash;
        return out.str();
    }

    static float read_tensor_value_f32(
            const uint8_t * data,
            ggml_type type,
            const size_t * nb,
            size_t i0,
            size_t i1,
            size_t i2,
            size_t i3) {
        const size_t offset = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
        switch (type) {
            case GGML_TYPE_F32:
                return *(const float *) &data[offset];
            case GGML_TYPE_F16:
                return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[offset]);
            case GGML_TYPE_BF16:
                return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[offset]);
            case GGML_TYPE_I32:
                return (float) *(const int32_t *) &data[offset];
            default:
                return NAN;
        }
    }

    static std::vector<float> flatten_tensor_f32(const ggml_tensor * t, const std::vector<uint8_t> & bytes) {
        std::vector<float> values;
        if (bytes.empty()) {
            return values;
        }
        const int64_t n0 = std::max<int64_t>(1, t->ne[0]);
        const int64_t n1 = std::max<int64_t>(1, t->ne[1]);
        const int64_t n2 = std::max<int64_t>(1, t->ne[2]);
        const int64_t n3 = std::max<int64_t>(1, t->ne[3]);
        values.reserve(static_cast<size_t>(n0 * n1 * n2 * n3));
        for (int64_t i3 = 0; i3 < n3; ++i3) {
            for (int64_t i2 = 0; i2 < n2; ++i2) {
                for (int64_t i1 = 0; i1 < n1; ++i1) {
                    for (int64_t i0 = 0; i0 < n0; ++i0) {
                        values.push_back(read_tensor_value_f32(
                            bytes.data(), t->type, t->nb,
                            static_cast<size_t>(i0), static_cast<size_t>(i1),
                            static_cast<size_t>(i2), static_cast<size_t>(i3)));
                    }
                }
            }
        }
        return values;
    }

    static json tensor_shape_json(const ggml_tensor * t) {
        return json::array({t->ne[0], t->ne[1], t->ne[2], t->ne[3]});
    }

    static json tensor_stride_json(const ggml_tensor * t) {
        return json::array({t->nb[0], t->nb[1], t->nb[2], t->nb[3]});
    }

    static json tail_values_json(const std::vector<float> & values, size_t count) {
        json out = json::array();
        const size_t n = std::min(count, values.size());
        const size_t start = values.size() > n ? values.size() - n : 0;
        for (size_t i = start; i < values.size(); ++i) {
            out.push_back(values[i]);
        }
        return out;
    }

    static json first_values_json(const std::vector<float> & values, size_t count) {
        json out = json::array();
        const size_t n = std::min(count, values.size());
        for (size_t i = 0; i < n; ++i) {
            out.push_back(values[i]);
        }
        return out;
    }

    static json tensor_stats_json(const ggml_tensor * t) {
        const std::vector<uint8_t> bytes = read_tensor_bytes(t);
        const std::vector<float> values = flatten_tensor_f32(t, bytes);
        double sum = 0.0;
        double sumsq = 0.0;
        float min_v = values.empty() ? 0.0f : values[0];
        float max_v = values.empty() ? 0.0f : values[0];
        for (float v : values) {
            sum += double(v);
            sumsq += double(v) * double(v);
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
        json out;
        out["tensor_name"] = t->name;
        out["op"] = ggml_op_name(t->op);
        out["shape"] = tensor_shape_json(t);
        out["stride"] = tensor_stride_json(t);
        out["nbytes"] = ggml_nbytes(t);
        out["dtype"] = ggml_type_name(t->type);
        out["view_src"] = t->view_src != nullptr ? std::string(t->view_src->name) : "";
        out["storage_offset"] = t->view_offs;
        out["contiguous"] = ggml_is_contiguous(t);
        out["contiguously_allocated"] = ggml_is_contiguously_allocated(t);
        out["src0"] = t->src[0] != nullptr ? std::string(t->src[0]->name) : "";
        out["src1"] = t->src[1] != nullptr ? std::string(t->src[1]->name) : "";
        out["hash"] = tensor_hash_hex(t);
        out["sum"] = sum;
        out["sumsq"] = sumsq;
        out["min"] = min_v;
        out["max"] = max_v;
        out["first_8_values"] = first_values_json(values, 8);
        out["last_8_values"] = tail_values_json(values, 8);
        return out;
    }

    static json tensor_parent_json(const ggml_tensor * t) {
        json out;
        out["tensor_name"] = t->name;
        out["op"] = ggml_op_name(t->op);
        out["shape"] = tensor_shape_json(t);
        out["stride"] = tensor_stride_json(t);
        out["dtype"] = ggml_type_name(t->type);
        out["view_src"] = t->view_src != nullptr ? std::string(t->view_src->name) : "";
        out["storage_offset"] = t->view_offs;
        out["contiguous"] = ggml_is_contiguous(t);
        out["contiguously_allocated"] = ggml_is_contiguously_allocated(t);
        return out;
    }

    static uint64_t fnv1a_update(uint64_t hash, uint8_t b) {
        hash ^= uint64_t(b);
        hash *= 1099511628211ull;
        return hash;
    }

    static std::string hash_bytes_hex(const std::vector<uint8_t> & bytes) {
        uint64_t hash = 1469598103934665603ull;
        for (uint8_t b : bytes) {
            hash = fnv1a_update(hash, b);
        }
        std::ostringstream out;
        out << std::hex << std::setfill('0') << std::setw(16) << hash;
        return out.str();
    }

    static std::string bytes_hex_prefix(const std::vector<uint8_t> & bytes, size_t count) {
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        const size_t n = std::min(count, bytes.size());
        for (size_t i = 0; i < n; ++i) {
            out << std::setw(2) << unsigned(bytes[i]);
        }
        return out.str();
    }

    json tensor_slot_stats_json(const ggml_tensor * t, const std::string & stage) const {
        json out = json::array();
        if (t == nullptr || t->type != GGML_TYPE_F32) {
            return out;
        }
        const std::vector<uint8_t> bytes = read_tensor_bytes(t);
        if (bytes.empty()) {
            return out;
        }
        const int64_t rows = std::max<int64_t>(1, t->ne[0]);
        const int64_t slots = std::max<int64_t>(1, std::min<int64_t>(6, t->ne[1]));
        const json topk_ids = current_.contains("topk_ids") ? current_["topk_ids"] : json::array();
        const json topk_weights = current_.contains("topk_weights") ? current_["topk_weights"] : json::array();
        for (int64_t slot = 0; slot < slots; ++slot) {
            std::vector<float> values;
            values.reserve((size_t) rows);
            std::vector<uint8_t> slot_bytes;
            slot_bytes.reserve((size_t) rows * sizeof(float));
            for (int64_t row = 0; row < rows; ++row) {
                const size_t offset = (size_t) slot * t->nb[1] + (size_t) row * t->nb[0];
                if (offset + sizeof(float) <= bytes.size()) {
                    const float v = *(const float *) &bytes[offset];
                    values.push_back(v);
                    const uint8_t * p = &bytes[offset];
                    slot_bytes.insert(slot_bytes.end(), p, p + sizeof(float));
                }
            }
            double sum = 0.0;
            double sumsq = 0.0;
            float min_v = values.empty() ? 0.0f : values[0];
            float max_v = values.empty() ? 0.0f : values[0];
            for (float v : values) {
                sum += double(v);
                sumsq += double(v) * double(v);
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }
            json item;
            item["stage"] = stage;
            item["slot"] = slot;
            item["expert_id"] = (topk_ids.is_array() && (size_t) slot < topk_ids.size()) ? topk_ids[(size_t) slot] : json(nullptr);
            item["topk_weight"] = (topk_weights.is_array() && (size_t) slot < topk_weights.size()) ? topk_weights[(size_t) slot] : json(nullptr);
            item["tensor_name"] = t->name;
            item["op"] = ggml_op_name(t->op);
            item["shape"] = json::array({rows, 1, 1, 1});
            item["parent_shape"] = tensor_shape_json(t);
            item["dtype"] = ggml_type_name(t->type);
            item["stride"] = tensor_stride_json(t);
            item["parent_stride"] = tensor_stride_json(t);
            item["view_src"] = t->view_src != nullptr ? std::string(t->view_src->name) : "";
            item["storage_offset"] = t->view_offs;
            item["contiguous"] = ggml_is_contiguous(t);
            item["contiguously_allocated"] = ggml_is_contiguously_allocated(t);
            item["slot_offset_bytes"] = (int64_t) slot * (int64_t) t->nb[1];
            item["checksum"] = hash_bytes_hex(slot_bytes);
            item["first_64_bytes"] = bytes_hex_prefix(slot_bytes, 64);
            item["sum"] = sum;
            item["sumsq"] = sumsq;
            item["min"] = min_v;
            item["max"] = max_v;
            item["first_16_values"] = first_values_json(values, 16);
            item["last_16_values"] = tail_values_json(values, 16);
            out.push_back(std::move(item));
        }
        return out;
    }

    static json tensor_f32_values_json(const ggml_tensor * t, size_t count) {
        const std::vector<uint8_t> bytes = read_tensor_bytes(t);
        const std::vector<float> values = flatten_tensor_f32(t, bytes);
        return first_values_json(values, count);
    }

    static json tensor_i32_values_json(const ggml_tensor * t) {
        const std::vector<uint8_t> bytes = read_tensor_bytes(t);
        json out = json::array();
        if (bytes.empty()) {
            return out;
        }
        const int64_t n0 = std::max<int64_t>(1, t->ne[0]);
        const int64_t n1 = std::max<int64_t>(1, t->ne[1]);
        for (int64_t i1 = 0; i1 < n1; ++i1) {
            for (int64_t i0 = 0; i0 < n0; ++i0) {
                const size_t offset = static_cast<size_t>(i1) * t->nb[1] + static_cast<size_t>(i0) * t->nb[0];
                if (t->type == GGML_TYPE_I32) {
                    out.push_back(*(const int32_t *) &bytes[offset]);
                } else {
                    out.push_back((int) read_tensor_value_f32(bytes.data(), t->type, t->nb, (size_t) i0, (size_t) i1, 0, 0));
                }
            }
        }
        return out;
    }
};

class cli_dsv4_rmoe_result_chain_readback {
public:
    cli_dsv4_rmoe_result_chain_readback()
        : mode_(cli_env_string(
                "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_RESULT_CHAIN_MODE", "none")),
          target_layer_(cli_env_int("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", 0)) {
    }

    bool enabled() const {
        return mode_.rfind("readback_", 0) == 0;
    }

    bool wants_tensor(const ggml_tensor * t) const {
        if (!enabled() || t == nullptr || t->name[0] == '\0') {
            return false;
        }
        const auto [base_name, layer] = oracle_parse_name_layer(std::string(t->name));
        return layer == target_layer_ && base_name.rfind("dsv4_rmoe_result_chain_readback_", 0) == 0;
    }

    bool handle_tensor(const ggml_tensor * t) {
        if (!wants_tensor(t)) {
            return true;
        }
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        if (bytes.empty()) {
            return true;
        }
        const bool is_host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);
        if (is_host) {
            if (t->data != nullptr) {
                std::memcpy(bytes.data(), t->data, bytes.size());
            }
        } else {
            ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
        }
        ++readbacks_;
        return true;
    }

    int readbacks() const {
        return readbacks_;
    }

private:
    std::string mode_;
    int target_layer_ = 0;
    int readbacks_ = 0;
};

class cli_dsv4_logit_dump {
public:
    cli_dsv4_logit_dump(
            const std::filesystem::path & out_path,
            int                           topk,
            bool                          trace_stages,
            bool                          hot_path_neutral)
        : out_path_(out_path),
          logits_topk_(std::max(2, topk)),
          trace_stages_(trace_stages),
          hot_path_neutral_(hot_path_neutral) {
        if (out_path_.has_parent_path()) {
            std::filesystem::create_directories(out_path_.parent_path());
        }
        fout_.open(out_path_, std::ios::binary);
        if (!fout_) {
            throw std::runtime_error("failed to open DSV4 logit dump file: " + out_path_.string());
        }
    }

    bool enabled() const {
        return fout_.is_open();
    }

    void set_vocab(const llama_vocab * vocab) {
        vocab_ = vocab;
    }

    void set_model_path(const std::string & model_path) {
        model_path_ = model_path;
    }

    void start_request(const std::string & prompt, std::vector<llama_token> prompt_ids) {
        prompt_ = prompt;
        prompt_ids_ = std::move(prompt_ids);
        eval_index_ = 0;
        generated_index_ = 0;
        request_index_++;
        wrote_metadata_for_request_ = false;
    }

    int topk() const {
        return logits_topk_;
    }

    bool wants_tensor(const ggml_tensor * t) const {
        if (!enabled() || t == nullptr || t->name[0] == '\0') {
            return false;
        }
        const auto [base_name, _layer] = oracle_parse_name_layer(std::string(t->name));
        return base_name == "result_output";
    }

    bool handle_tensor(const ggml_tensor * t) {
        if (t == nullptr || !wants_tensor(t)) {
            return true;
        }
        write_metadata_once();

        const auto values = flatten_tensor_f32(t);
        std::vector<int32_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        const size_t keep = std::min<size_t>(static_cast<size_t>(logits_topk_), indices.size());
        if (keep == 0) {
            ++eval_index_;
            return true;
        }
        std::partial_sort(
            indices.begin(),
            indices.begin() + static_cast<ptrdiff_t>(keep),
            indices.end(),
            [&](const int32_t lhs, const int32_t rhs) {
                return values[static_cast<size_t>(lhs)] > values[static_cast<size_t>(rhs)];
            }
        );

        const int prediction_index = eval_index_ - static_cast<int>(prompt_ids_.size()) + 1;
        json topk = json::array();
        for (size_t i = 0; i < keep; ++i) {
            const int32_t id = indices[i];
            topk.push_back({
                {"rank", static_cast<int>(i + 1)},
                {"id", id},
                {"logit", values[static_cast<size_t>(id)]},
                {"text", token_piece(id)},
            });
        }

        const int32_t top1_id = indices[0];
        const int32_t top2_id = keep > 1 ? indices[1] : indices[0];
        const float top1_logit = values[static_cast<size_t>(top1_id)];
        const float top2_logit = values[static_cast<size_t>(top2_id)];

        json record;
        record["kind"] = "logits";
        record["format"] = "dsv4-logit-compare-v1";
        record["request_index"] = request_index_;
        record["eval_index"] = eval_index_;
        record["position"] = eval_index_;
        record["phase"] = prediction_index >= 0 ? "decode" : "prefill";
        record["token_index"] = prediction_index;
        record["prompt_token_count"] = prompt_ids_.size();
        record["tensor_name"] = t->name;
        record["shape"] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
        record["topk_requested"] = logits_topk_;
        record["topk_recorded"] = static_cast<int>(keep);
        record["top1_id"] = top1_id;
        record["top1_text"] = token_piece(top1_id);
        record["top1_logit"] = top1_logit;
        record["top2_id"] = top2_id;
        record["top2_text"] = token_piece(top2_id);
        record["top2_logit"] = top2_logit;
        record["top1_top2_margin"] = top1_logit - top2_logit;
        record["predicted_token_id"] = top1_id;
        record["predicted_token_text"] = token_piece(top1_id);
        record["topk"] = std::move(topk);
        if (hot_path_neutral_) {
            append_hot_path_neutral_metadata(record);
        }
        if (trace_stages_) {
            record["active_stage_flags"] = active_stage_flags();
        }
        fout_ << record.dump() << "\n";
        ++eval_index_;
        return true;
    }

    void record_partial(const completion_token_output & output) {
        if (!enabled() || output.probs.empty()) {
            return;
        }
        write_metadata_once();

        const size_t keep = std::min<size_t>(static_cast<size_t>(logits_topk_), output.probs.size());
        json topk = json::array();
        for (size_t i = 0; i < keep; ++i) {
            const auto & p = output.probs[i];
            topk.push_back({
                {"rank", static_cast<int>(i + 1)},
                {"id", p.tok},
                {"logit", p.logit},
                {"prob", p.prob},
                {"text", cli_json_safe_text(p.txt)},
            });
        }

        const auto & top1 = output.probs[0];
        const auto & top2 = keep > 1 ? output.probs[1] : output.probs[0];

        json record;
        record["kind"] = "logits";
        record["format"] = "dsv4-logit-compare-v1";
        record["source"] = "server_prob_output";
        record["request_index"] = request_index_;
        record["eval_index"] = generated_index_;
        record["position"] = static_cast<int>(prompt_ids_.size()) + generated_index_;
        record["phase"] = "decode";
        record["token_index"] = generated_index_;
        record["prompt_token_count"] = prompt_ids_.size();
        record["topk_requested"] = logits_topk_;
        record["topk_recorded"] = static_cast<int>(keep);
        record["sampled_token_id"] = output.tok;
        record["sampled_token_text"] = cli_json_safe_text(output.text_to_send);
        record["top1_id"] = top1.tok;
        record["top1_text"] = cli_json_safe_text(top1.txt);
        record["top1_logit"] = top1.logit;
        record["top1_prob"] = top1.prob;
        record["top2_id"] = top2.tok;
        record["top2_text"] = cli_json_safe_text(top2.txt);
        record["top2_logit"] = top2.logit;
        record["top2_prob"] = top2.prob;
        record["top1_top2_margin"] = top1.logit - top2.logit;
        record["predicted_token_id"] = top1.tok;
        record["predicted_token_text"] = cli_json_safe_text(top1.txt);
        record["topk"] = std::move(topk);
        if (hot_path_neutral_) {
            append_hot_path_neutral_metadata(record);
        }
        if (trace_stages_) {
            record["active_stage_flags"] = active_stage_flags();
        }
        fout_ << record.dump() << "\n";
        ++generated_index_;
    }

    void finish() {
        if (fout_.is_open()) {
            fout_.flush();
        }
    }

    const std::filesystem::path & path() const {
        return out_path_;
    }

private:
    std::filesystem::path out_path_;
    std::ofstream fout_;
    int logits_topk_ = 20;
    bool trace_stages_ = false;
    bool hot_path_neutral_ = false;
    bool wrote_metadata_for_request_ = false;
    int request_index_ = -1;
    int eval_index_ = 0;
    int generated_index_ = 0;
    const llama_vocab * vocab_ = nullptr;
    std::string prompt_;
    std::string model_path_;
    std::vector<llama_token> prompt_ids_;

    void write_metadata_once() {
        if (wrote_metadata_for_request_) {
            return;
        }
        json meta;
        meta["kind"] = "metadata";
        meta["format"] = "dsv4-logit-compare-v1";
        meta["request_index"] = request_index_;
        meta["model_path"] = model_path_;
        meta["prompt"] = prompt_;
        meta["prompt_ids"] = prompt_ids_;
        meta["prompt_token_count"] = prompt_ids_.size();
        meta["topk_requested"] = logits_topk_;
        if (hot_path_neutral_) {
            append_hot_path_neutral_metadata(meta);
        }
        if (trace_stages_) {
            meta["active_stage_flags"] = active_stage_flags();
        }
        fout_ << meta.dump() << "\n";
        wrote_metadata_for_request_ = true;
    }

    static void append_hot_path_neutral_metadata(json & record) {
        const std::vector<std::string> intrusive = cli_dsv4_hotpath_intrusive_flags();
        const std::vector<std::string> rejected = cli_dsv4_hotpath_rejected_flags();
        const std::string under_test = cli_dsv4_hotpath_under_test_name();
        const bool experimental_under_test = !under_test.empty();
        const std::vector<std::string> under_test_flags =
            experimental_under_test ? cli_dsv4_hotpath_under_test_active_flags(under_test) : std::vector<std::string>{};
        const std::string guard_error = cli_dsv4_hotpath_neutral_guard_error();
        record["hot_path_neutral_validation"] = true;
        record["validation_hot_path_neutral"] = true;
        record["validation_source"] = "server_prob_output";
        record["wall_clock_neutral"] = false;
        record["wall_clock_note"] = "forces sampler top-k probability payloads and writes JSONL; use only for validation, not speed";
        record["sampling_n_probs_forced_for_validation"] = true;
        record["graph_eval_callback_registered_for_logit_dump"] = false;
        record["ggml_graph_nodes_added_for_validation"] = false;
        record["tensor_readbacks_added_for_validation"] = false;
        record["consume_path_enabled_for_validation"] = false;
        record["cache_mutation_enabled_for_validation"] = false;
        record["hot_path_neutral_intrusive_flags_enabled"] = cli_strings_to_json_array(intrusive);
        record["hot_path_neutral_rejected_paths_enabled"] = cli_strings_to_json_array(rejected);
        record["hot_path_neutral_guard_ok"] = guard_error.empty();
        record["hot_path_neutral_guard_error"] = guard_error;
        record["experimental_under_test"] = experimental_under_test;
        record["under_test_name"] = experimental_under_test ? under_test : "";
        record["under_test_flags"] = cli_strings_to_json_array(under_test_flags);
        record["under_test_changes_graph"] =
            experimental_under_test &&
            under_test != "dsv4_layer_executor_metadata_only" &&
            under_test != "dsv4_layer_executor_side_probe_hcnorm" &&
            under_test != "dsv4_layer_executor_side_probe_rmoe" &&
            under_test != "dsv4_layer_executor_side_probe_aohc" &&
            under_test != "dsv4_layer_executor_side_probe_compressor_update" &&
            under_test != "dsv4_layer_executor_side_probe_kv_cache_finalizer";
        record["path_accepted"] = false;
        record["acceptance_policy"] = "transcript_exact";
        record["under_test_allow_rejected"] =
            cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_ALLOW_REJECTED_UNDER_TEST");
        record["under_test_ack"] =
            cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_REJECTED_UNDER_TEST_ACK", "");
        if (under_test == "aohc_single_layer_consume" || under_test == "aohc_single_layer_skip_generic") {
            record["consume_layer"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_LAYER", "");
            record["consume_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_CONSUME_MODE", "");
            record["skip_generic"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC");
            record["skip_generic_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_HC_POST_SKIP_GENERIC_MODE", "");
        }
        if (under_test == "aohc_fused_single_layer_consume" ||
                under_test == "aohc_backend_fused_single_layer_consume" ||
                under_test == "aohc_backend_fused_layer_set_consume") {
            record["consume_layer"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYER", "");
            record["layers"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYERS", "");
            const std::string layers = cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_LAYERS", "");
            int layer_count = 0;
            if (!layers.empty()) {
                layer_count = 1;
                for (char c : layers) {
                    if (c == ',') {
                        layer_count++;
                    }
                }
            }
            record["layer_count"] = layer_count;
            record["fused"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED");
            record["fused_consume"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_AOHC_FUSED_CONSUME");
            record["backend_fused"] =
                under_test == "aohc_backend_fused_single_layer_consume" ||
                under_test == "aohc_backend_fused_layer_set_consume";
        }
        if (under_test == "cupd3_tail_single_layer_consume") {
            record["consume_layer"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_LAYER", "");
            record["consume_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME_MODE", "");
            record["projection_source"] = "generic";
            record["cache_mutation_mode"] = "generic_existing_write";
            record["candidate_cache_side_effect"] = false;
            record["skip_generic_tail"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL");
        }
        if (under_test == "cupd3_backend_tail_single_layer_consume") {
            record["consume_layer"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_LAYER", "");
            record["consume_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME_MODE", "");
            record["projection_source"] = "generic";
            record["cache_mutation_mode"] = "generic_existing_write";
            record["candidate_cache_side_effect"] = false;
            record["backend_tail"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL");
            record["backend_tail_consume"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME");
            record["backend_tail_dep_barrier"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_DEP_BARRIER");
            record["backend_tail_emit_only"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME_EMIT_ONLY");
            record["backend_tail_stream"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_STREAM", "all");
            record["backend_tail_attn_layout_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_ATTN_LAYOUT_MODE", "backend_layout");
            record["backend_tail_attn_row_probe"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_ATTN_ROW_PROBE");
            record["backend_tail_value_probe"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_VALUE_PROBE");
            record["decode_compress_internal_probe"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_DECODE_COMPRESS_INTERNAL_PROBE");
        }
        if (under_test == "rmoe_backend_single_layer_consume" ||
                under_test == "rmoe_backend_single_layer_replace_generic" ||
                under_test == "rmoe_backend_pair_preserve_single_layer" ||
                under_test == "rmoe_backend_shared_final_single_layer") {
            record["consume_layer"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_LAYER", "");
            record["consume_token_min"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MIN", "");
            record["consume_token_max"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TOKEN_MAX", "");
            record["rmoe_backend_op"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP");
            record["rmoe_backend_consume"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME");
            record["rmoe_backend_replace_generic"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC");
            record["replace_generic"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC");
            record["pair_preserve"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_PAIR_PRESERVE");
            record["pair_preserve_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_PAIR_PRESERVE_MODE", "");
            record["shared_final_only"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SHARED_FINAL_ONLY");
            record["shared_final_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SHARED_FINAL_MODE", "");
            record["rmoe_backend_substage"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SUBSTAGE", "");
            record["swiglu_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_SWIGLU_MODE", "");
            record["down_input"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_DOWN_INPUT", "internal_scratch");
            record["rmoe_backend_branch_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BRANCH_MODE", "");
            record["rmoe_backend_branch_order"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BRANCH_ORDER", "");
            record["rmoe_backend_consume_semantic"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_SEMANTIC", "backend_view");
            record["rmoe_backend_consume_trace"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME_TRACE");
            record["rmoe_backend_boundary_probe"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_BOUNDARY_PROBE");
            record["rmoe_backend_result_chain_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_RESULT_CHAIN_MODE", "none");
            record["rmoe_backend_graph_trace"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_GRAPH_TRACE");
        }
        if (under_test == "rmoe_generic_lowering_parity") {
            record["lowering_parity"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY");
            record["lowering_parity_layer"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY_LAYER", "");
            record["lowering_parity_token_min"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY_TOKEN_MIN", "");
            record["lowering_parity_token_max"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY_TOKEN_MAX", "");
            record["lowering_parity_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_GENERIC_LOWERING_PARITY_MODE", "normal");
            record["generic_ffn_built"] = true;
            record["backend_ffn_built"] = false;
        }
        if (under_test == "dsv4_layer_executor_metadata_only" ||
                under_test == "dsv4_layer_executor_side_probe_hcnorm" ||
                under_test == "dsv4_layer_executor_side_probe_rmoe" ||
                under_test == "dsv4_layer_executor_side_probe_aohc" ||
                under_test == "dsv4_layer_executor_side_probe_compressor_update" ||
                under_test == "dsv4_layer_executor_side_probe_kv_cache_finalizer") {
            record["dsv4_layer_executor_dryrun_op"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP");
            record["dsv4_layer_executor_dryrun_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_DRYRUN_OP_MODE", "live_graph_dispatch");
            record["dsv4_layer_executor_side_probe"] =
                cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE");
            record["dsv4_layer_executor_side_probe_stage"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_STAGE", "");
            record["dsv4_layer_executor_side_probe_mode"] =
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SIDE_PROBE_MODE", "");
            record["dsv4_layer_executor_live_graph_nodes_added"] = false;
            record["dsv4_layer_executor_live_backend_dispatches"] = false;
            record["dsv4_layer_executor_output_consumed"] = false;
            record["dsv4_layer_executor_cache_mutation"] = "disabled";
        }
    }

    std::string token_piece(int32_t token_id) const {
        if (vocab_ == nullptr) {
            return "";
        }
        return cli_json_safe_text(common_token_to_piece(vocab_, static_cast<llama_token>(token_id), true));
    }

    static json active_stage_flags() {
        const std::array<const char *, 40> names = {{
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_PRE_NORM_COMPARE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_COMPARE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V2_FUSED_COMP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SHADOW",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_COMPARE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_SKIP_GENERIC_TAIL",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_TAIL_ATTRIB",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_DRIFT_TRACE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_ATTN_ROW_PROBE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_VALUE_PROBE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_DECODE_COMPRESS_INTERNAL_PROBE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_DEP_BARRIER",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_CONSUME_EMIT_ONLY",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_ABORT_ON_MISMATCH",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_GENERIC",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_KV_FINALIZE_COMPARE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_COMPARE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_FFN_MOE_STAGE_FULL_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_DOWN_SUM6",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_DECODE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ATTN_OUT_DECODE_COMPARE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_MIXED_ATTN",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_Q8_HC_EXPAND",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HC_EXPAND4",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_SHADOW",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_COMPARE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LAYER_EXECUTOR_CONSUME",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_VALIDATE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_TRACE_STAGES",
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU",
        }};
        json flags = json::object();
        for (const char * name : names) {
            flags[name] = cli_env_enabled(name);
        }
        return flags;
    }

    static float read_tensor_value_f32(
        const uint8_t * data,
        ggml_type type,
        const size_t * nb,
        size_t i0,
        size_t i1,
        size_t i2,
        size_t i3) {
        const size_t offset = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
        switch (type) {
            case GGML_TYPE_F32:
                return *(const float *) &data[offset];
            case GGML_TYPE_F16:
                return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[offset]);
            case GGML_TYPE_BF16:
                return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[offset]);
            default:
                throw std::runtime_error("unsupported DSV4 logit tensor dtype: " + std::string(ggml_type_name(type)));
        }
    }

    static std::vector<float> flatten_tensor_f32(const ggml_tensor * t) {
        const int64_t n0 = std::max<int64_t>(1, t->ne[0]);
        const int64_t n1 = std::max<int64_t>(1, t->ne[1]);
        const int64_t n2 = std::max<int64_t>(1, t->ne[2]);
        const int64_t n3 = std::max<int64_t>(1, t->ne[3]);
        const bool is_host = t->buffer == nullptr || ggml_backend_buffer_is_host(t->buffer);
        std::vector<uint8_t> host_copy;
        if (!is_host) {
            host_copy.resize(ggml_nbytes(t));
            ggml_backend_tensor_get(t, host_copy.data(), 0, host_copy.size());
        }
        const uint8_t * data = is_host ? (const uint8_t *) t->data : host_copy.data();
        if (data == nullptr) {
            const char * tensor_name = t->name[0] != '\0' ? t->name : "<unnamed>";
            throw std::runtime_error("DSV4 logit tensor has no readable data for " + std::string(tensor_name));
        }
        std::vector<float> values;
        values.reserve(static_cast<size_t>(n0 * n1 * n2 * n3));
        for (int64_t i3 = 0; i3 < n3; ++i3) {
            for (int64_t i2 = 0; i2 < n2; ++i2) {
                for (int64_t i1 = 0; i1 < n1; ++i1) {
                    for (int64_t i0 = 0; i0 < n0; ++i0) {
                        values.push_back(read_tensor_value_f32(
                            data,
                            t->type,
                            t->nb,
                            static_cast<size_t>(i0),
                            static_cast<size_t>(i1),
                            static_cast<size_t>(i2),
                            static_cast<size_t>(i3)));
                    }
                }
            }
        }
        return values;
    }
};

static cli_dsv4_logit_dump * g_dsv4_logit_dump = nullptr;

struct cli_eval_capture_state {
    cli_oracle_dump * oracle = nullptr;
    cli_dsv4_value_probe * value_probe = nullptr;
    cli_dsv4_rmoe_replace_dump * rmoe_replace_dump = nullptr;
    cli_dsv4_rmoe_result_chain_readback * rmoe_result_chain_readback = nullptr;
    cli_dsv4_logit_dump * logit_dump = nullptr;
};

static bool cli_eval_capture_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<cli_eval_capture_state *>(user_data);
    if (state == nullptr) {
        return true;
    }
    if (ask) {
        return (state->oracle != nullptr && state->oracle->enabled() && state->oracle->wants_tensor(t)) ||
               (state->value_probe != nullptr && state->value_probe->enabled() && state->value_probe->wants_tensor(t)) ||
               (state->rmoe_replace_dump != nullptr && state->rmoe_replace_dump->enabled() && state->rmoe_replace_dump->wants_tensor(t)) ||
               (state->rmoe_result_chain_readback != nullptr && state->rmoe_result_chain_readback->enabled() && state->rmoe_result_chain_readback->wants_tensor(t)) ||
               (state->logit_dump != nullptr && state->logit_dump->enabled() && state->logit_dump->wants_tensor(t));
    }
    if (state->oracle != nullptr && state->oracle->enabled() && state->oracle->wants_tensor(t)) {
        if (!state->oracle->handle_tensor(t)) {
            return false;
        }
    }
    if (state->logit_dump != nullptr && state->logit_dump->enabled() && state->logit_dump->wants_tensor(t)) {
        if (!state->logit_dump->handle_tensor(t)) {
            return false;
        }
    }
    if (state->rmoe_replace_dump != nullptr && state->rmoe_replace_dump->enabled() && state->rmoe_replace_dump->wants_tensor(t)) {
        if (!state->rmoe_replace_dump->handle_tensor(t)) {
            return false;
        }
    }
    if (state->rmoe_result_chain_readback != nullptr && state->rmoe_result_chain_readback->enabled() && state->rmoe_result_chain_readback->wants_tensor(t)) {
        if (!state->rmoe_result_chain_readback->handle_tensor(t)) {
            return false;
        }
    }
    if (state->value_probe != nullptr && state->value_probe->enabled() && state->value_probe->wants_tensor(t)) {
        if (!state->value_probe->handle_tensor(t)) {
            return false;
        }
    }
    return true;
}

} // namespace

struct cli_context {
    server_context ctx_server;
    json messages = json::array();
    std::vector<raw_buffer> input_files;
    task_params defaults;
    bool verbose_prompt;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    int enable_reasoning = -1;
    int reasoning_budget = -1;
    std::string reasoning_budget_message;

    // thread for showing "loading" animation
    std::atomic<bool> loading_show;

    cli_context(const common_params & params) {
        defaults.sampling    = params.sampling;
        defaults.speculative = params.speculative;
        defaults.n_keep      = params.n_keep;
        defaults.n_predict   = params.n_predict;
        defaults.antiprompt  = params.antiprompt;

        defaults.stream = true; // make sure we always use streaming mode
        defaults.timings_per_token = true; // in order to get timings even when we cancel mid-way
        // defaults.return_progress = true; // TODO: show progress

        verbose_prompt = params.verbose_prompt;
        reasoning_format = params.reasoning_format;
        enable_reasoning = params.enable_reasoning;
        reasoning_budget = params.reasoning_budget;
        reasoning_budget_message = params.reasoning_budget_message;
    }

    std::string generate_completion(result_timings & out_timings) {
        return generate_completion_impl(out_timings, std::nullopt);
    }

    std::string generate_raw_completion(const std::string & prompt, result_timings & out_timings) {
        return generate_completion_impl(out_timings, prompt);
    }

private:
    std::string generate_completion_impl(result_timings & out_timings, const std::optional<std::string> & raw_prompt) {
        server_response_reader rd = ctx_server.get_response_reader();
        common_chat_params chat_params;
        if (!raw_prompt.has_value()) {
            chat_params = format_chat();
        }
        {
            // TODO: reduce some copies here in the future
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id         = rd.get_new_id();
            task.index      = 0;
            task.params     = defaults;           // copy
            task.cli_prompt = raw_prompt.has_value() ? *raw_prompt : chat_params.prompt; // copy
            task.cli_files  = input_files;        // copy
            task.cli        = true;

            if (!raw_prompt.has_value()) {
                // chat template settings
                task.params.chat_parser_params = common_chat_parser_params(chat_params);
                task.params.chat_parser_params.reasoning_format = reasoning_format;
                if (!chat_params.parser.empty()) {
                    task.params.chat_parser_params.parser.load(chat_params.parser);
                }

                // reasoning budget sampler
                if (reasoning_budget >= 0 &&
                    reasoning_format != COMMON_REASONING_FORMAT_NONE &&
                    !chat_params.thinking_end_tag.empty()) {
                    const llama_vocab * vocab = llama_model_get_vocab(
                        llama_get_model(ctx_server.get_llama_context()));

                    task.params.sampling.reasoning_budget_tokens = reasoning_budget;
                    task.params.sampling.generation_prompt = chat_params.generation_prompt;

                    if (!chat_params.thinking_start_tag.empty()) {
                        task.params.sampling.reasoning_budget_start =
                            common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
                    }
                    task.params.sampling.reasoning_budget_end =
                        common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
                    task.params.sampling.reasoning_budget_forced =
                        common_tokenize(vocab, reasoning_budget_message + chat_params.thinking_end_tag, false, true);
                }
            } else {
                // Raw completions should stream plain content deltas without invoking the
                // chat parser or tool-call recovery machinery.
                task.params.chat_parser_params = common_chat_parser_params();
                task.params.chat_parser_params.format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
                task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_NONE;
                task.params.chat_parser_params.reasoning_in_content = false;
                task.params.chat_parser_params.generation_prompt.clear();
                task.params.chat_parser_params.parse_tool_calls = false;
            }

            if (g_dsv4_logit_dump != nullptr && g_dsv4_logit_dump->enabled()) {
                const llama_context * lctx = ctx_server.get_llama_context();
                const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(lctx));
                g_dsv4_logit_dump->start_request(
                    task.cli_prompt,
                    common_tokenize(vocab, task.cli_prompt, false, true));
                task.params.sampling.n_probs = std::max(task.params.sampling.n_probs, g_dsv4_logit_dump->topk());
                task.params.post_sampling_probs = false;
            }

            rd.post_task({std::move(task)});
        }

        if (verbose_prompt && !raw_prompt.has_value()) {
            console::set_display(DISPLAY_TYPE_PROMPT);
            console::log("%s\n\n", chat_params.prompt.c_str());
            console::set_display(DISPLAY_TYPE_RESET);
        }

        // wait for first result
        console::spinner::start();
        server_task_result_ptr result = rd.next(should_stop);

        console::spinner::stop();
        std::string curr_content;
        bool is_thinking = false;
        bool suppress_raw_thinking = enable_reasoning == 0;
        bool defer_initial_content = suppress_raw_thinking && reasoning_format != COMMON_REASONING_FORMAT_NONE;
        bool saw_reasoning_delta = false;
        bool inside_raw_think = false;
        std::string raw_think_scan_buf;
        std::string deferred_content_buf;

        auto emit_content = [&](const std::string & text) {
            if (text.empty()) {
                return;
            }
            if (is_thinking) {
                console::log("\n[End thinking]\n\n");
                console::set_display(DISPLAY_TYPE_RESET);
                is_thinking = false;
            }
            curr_content += text;
            console::log("%s", text.c_str());
            console::flush();
        };

        auto handle_content_delta = [&](const std::string & delta) {
            if (delta.empty()) {
                return;
            }

            if (!suppress_raw_thinking) {
                emit_content(delta);
                return;
            }

            static const std::string think_open = "<think>";
            static const std::string think_close = "</think>";

            raw_think_scan_buf += delta;

            while (!raw_think_scan_buf.empty()) {
                if (inside_raw_think) {
                    const size_t close_pos = raw_think_scan_buf.find(think_close);
                    if (close_pos == std::string::npos) {
                        const size_t keep = think_close.size() > 1 ? think_close.size() - 1 : 0;
                        if (raw_think_scan_buf.size() > keep) {
                            raw_think_scan_buf.erase(0, raw_think_scan_buf.size() - keep);
                        }
                        break;
                    }

                    raw_think_scan_buf.erase(0, close_pos + think_close.size());
                    inside_raw_think = false;
                    continue;
                }

                const size_t open_pos = raw_think_scan_buf.find(think_open);
                const size_t close_pos = raw_think_scan_buf.find(think_close);

                if (close_pos != std::string::npos &&
                    (open_pos == std::string::npos || close_pos < open_pos) &&
                    curr_content.empty()) {
                    raw_think_scan_buf.erase(0, close_pos + think_close.size());
                    continue;
                }

                if (open_pos == std::string::npos) {
                    const size_t keep = think_open.size() > 1 ? think_open.size() - 1 : 0;
                    if (raw_think_scan_buf.size() <= keep) {
                        break;
                    }
                    const size_t emit_len = raw_think_scan_buf.size() - keep;
                    emit_content(raw_think_scan_buf.substr(0, emit_len));
                    raw_think_scan_buf.erase(0, emit_len);
                    break;
                }

                emit_content(raw_think_scan_buf.substr(0, open_pos));
                raw_think_scan_buf.erase(0, open_pos + think_open.size());
                inside_raw_think = true;
            }
        };

        while (result) {
            if (should_stop()) {
                break;
            }
            if (result->is_error()) {
                json err_data = result->to_json();
                if (err_data.contains("message")) {
                    console::error("Error: %s\n", err_data["message"].get<std::string>().c_str());
                } else {
                    console::error("Error: %s\n", err_data.dump().c_str());
                }
                return curr_content;
            }
            auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                out_timings = std::move(res_partial->timings);
                if (g_dsv4_logit_dump != nullptr && g_dsv4_logit_dump->enabled()) {
                    g_dsv4_logit_dump->record_partial(res_partial->prob_output);
                }
                if (raw_prompt.has_value()) {
                    emit_content(res_partial->content);
                    result = rd.next(should_stop);
                    continue;
                }
                for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                    if (!diff.content_delta.empty()) {
                        if (defer_initial_content && !saw_reasoning_delta && curr_content.empty()) {
                            deferred_content_buf += diff.content_delta;
                        } else {
                            handle_content_delta(diff.content_delta);
                        }
                    }
                    if (!diff.reasoning_content_delta.empty()) {
                        saw_reasoning_delta = true;
                        if (defer_initial_content) {
                            deferred_content_buf.clear();
                        }
                        if (enable_reasoning == 0) {
                            continue;
                        }
                        console::set_display(DISPLAY_TYPE_REASONING);
                        if (!is_thinking) {
                            console::log("[Start thinking]\n");
                        }
                        is_thinking = true;
                        console::log("%s", diff.reasoning_content_delta.c_str());
                        console::flush();
                    }
                }
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                out_timings = std::move(res_final->timings);
                break;
            }
            result = rd.next(should_stop);
        }

        if (defer_initial_content && !saw_reasoning_delta && !deferred_content_buf.empty()) {
            handle_content_delta(deferred_content_buf);
            deferred_content_buf.clear();
        }

        if (suppress_raw_thinking && !inside_raw_think && !raw_think_scan_buf.empty()) {
            emit_content(raw_think_scan_buf);
            raw_think_scan_buf.clear();
        }

        g_is_interrupted.store(false);
        // server_response_reader automatically cancels pending tasks upon destruction
        return curr_content;
    }

public:

    // TODO: support remote files in the future (http, https, etc)
    std::string load_input_file(const std::string & fname, bool is_media) {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            return "";
        }
        if (is_media) {
            raw_buffer buf;
            buf.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            input_files.push_back(std::move(buf));
            return mtmd_default_marker();
        } else {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            return content;
        }
    }

    common_chat_params format_chat() {
        auto meta = ctx_server.get_meta();
        auto & chat_params = meta.chat_params;

        common_chat_templates_inputs inputs;
        inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
        inputs.tools                 = {}; // TODO
        inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
        inputs.json_schema           = ""; // TODO
        inputs.grammar               = ""; // TODO
        inputs.use_jinja             = chat_params.use_jinja;
        inputs.parallel_tool_calls   = false;
        inputs.add_generation_prompt = true;
        inputs.reasoning_format      = reasoning_format;
        inputs.force_pure_content    = chat_params.force_pure_content;
        const bool template_supports_thinking = common_chat_templates_support_enable_thinking(chat_params.tmpls.get());
        if (enable_reasoning == 0 || reasoning_format == COMMON_REASONING_FORMAT_NONE) {
            inputs.enable_thinking = false;
        } else if (enable_reasoning == 1) {
            inputs.enable_thinking = template_supports_thinking;
        } else {
            inputs.enable_thinking = chat_params.enable_thinking ? template_supports_thinking : false;
        }

        // Apply chat template to the list of messages
        return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
    }
};

// TODO?: Make this reusable, enums, docs
static const std::array<const std::string, 6> cmds = {
    "/audio ",
    "/clear",
    "/exit",
    "/image ",
    "/read ",
    "/regen",
};

static std::vector<std::pair<std::string, size_t>> auto_completion_callback(std::string_view line, size_t cursor_byte_pos) {
    std::vector<std::pair<std::string, size_t>> matches;
    std::string cmd;

    if (line.length() > 1 && line[0] == '/' && !std::any_of(cmds.begin(), cmds.end(), [line](const std::string & prefix) {
        return string_starts_with(line, prefix);
    })) {
        auto it = cmds.begin();

        while ((it = std::find_if(it, cmds.end(), [line](const std::string & cmd_line) {
            return string_starts_with(cmd_line, line);
        })) != cmds.end()) {
            matches.emplace_back(*it, (*it).length());
            ++it;
        }
    } else {
        auto it = std::find_if(cmds.begin(), cmds.end(), [line](const std::string & prefix) {
            return prefix.back() == ' ' && string_starts_with(line, prefix);
        });

        if (it != cmds.end()) {
            cmd = *it;
        }
    }

    if (!cmd.empty() && line.length() >= cmd.length() && cursor_byte_pos >= cmd.length()) {
        const std::string path_prefix  = std::string(line.substr(cmd.length(), cursor_byte_pos - cmd.length()));
        const std::string path_postfix = std::string(line.substr(cursor_byte_pos));
        auto cur_dir = std::filesystem::current_path();
        std::string cur_dir_str = cur_dir.string();
        std::string expanded_prefix = path_prefix;

#if !defined(_WIN32)
        if (string_starts_with(path_prefix, "~")) {
            const char * home = std::getenv("HOME");
            if (home && home[0]) {
                expanded_prefix = std::string(home) + path_prefix.substr(1);
            }
        }
        if (string_starts_with(expanded_prefix, "/")) {
#else
        if (std::isalpha(expanded_prefix[0]) && expanded_prefix.find(':') == 1) {
#endif
            cur_dir = std::filesystem::path(expanded_prefix).parent_path();
            cur_dir_str = "";
        } else if (!path_prefix.empty()) {
            cur_dir /= std::filesystem::path(path_prefix).parent_path();
        }

        std::error_code ec;
        for (const auto & entry : std::filesystem::directory_iterator(cur_dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.exists(ec)) {
                ec.clear();
                continue;
            }

            const std::string path_full = entry.path().string();
            std::string path_entry = !cur_dir_str.empty() && string_starts_with(path_full, cur_dir_str) ? path_full.substr(cur_dir_str.length() + 1) : path_full;

            if (entry.is_directory(ec)) {
                path_entry.push_back(std::filesystem::path::preferred_separator);
            }

            if (expanded_prefix.empty() || string_starts_with(path_entry, expanded_prefix)) {
                std::string updated_line = cmd + path_entry;
                matches.emplace_back(updated_line + path_postfix, updated_line.length());
            }

            if (ec) {
                ec.clear();
            }
        }

        if (matches.empty()) {
            std::string updated_line = cmd + path_prefix;
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        // Add the longest common prefix
        if (!expanded_prefix.empty() && matches.size() > 1) {
            const std::string_view match0(matches[0].first);
            const std::string_view match1(matches[1].first);
            auto it = std::mismatch(match0.begin(), match0.end(), match1.begin(), match1.end());
            size_t len = it.first - match0.begin();

            for (size_t i = 2; i < matches.size(); ++i) {
                const std::string_view matchi(matches[i].first);
                auto cmp = std::mismatch(match0.begin(), match0.end(), matchi.begin(), matchi.end());
                len = std::min(len, static_cast<size_t>(cmp.first - match0.begin()));
            }

            std::string updated_line = std::string(match0.substr(0, len));
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        std::sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) {
            return a.first.compare(0, a.second, b.first, 0, b.second) < 0;
        });
    }

    return matches;
}

int main(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    // TODO: maybe support it later?
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED && !params.moe_trace_harness) {
        console::error("--no-conversation is not supported by llama-cli\n");
        console::error("please use llama-completion instead\n");
    }

    std::optional<cli_oracle_dump> oracle_dump;
    if (!params.oracle_dump.empty()) {
        if (!params.moe_trace_harness) {
            console::error("--oracle-dump currently requires --moe-trace-harness so the recorded prompt ids match a single raw completion request\n");
            return 1;
        }
        if (params.prompt.empty()) {
            console::error("--oracle-dump requires --prompt so the capture can be tied to a single known request\n");
            return 1;
        }
        oracle_dump.emplace(std::filesystem::path(params.oracle_dump), params.oracle_topk);
    }

    const bool dsv4_hot_path_neutral_validate =
        cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_HOTPATH_NEUTRAL_VALIDATE");
    if (dsv4_hot_path_neutral_validate) {
        const std::string guard_error = cli_dsv4_hotpath_neutral_guard_error();
        if (!guard_error.empty()) {
            console::error(
                "DSV4 hot-path-neutral validation refused this configuration: %s\n",
                guard_error.c_str());
            return 1;
        }
    }

    std::optional<cli_dsv4_logit_dump> dsv4_logit_dump;
    if (dsv4_hot_path_neutral_validate ||
            cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE")) {
        const std::filesystem::path dump_path(cli_env_string(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_DUMP",
            dsv4_hot_path_neutral_validate ?
                "/tmp/dsv4_hotpath_neutral_logits.jsonl" :
                "/tmp/dsv4_logit_compare.jsonl").c_str());
        const int topk = std::max(2, cli_env_int("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_TOPK", 20));
        dsv4_logit_dump.emplace(
            dump_path,
            topk,
            cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_LOGIT_COMPARE_TRACE_STAGES"),
            dsv4_hot_path_neutral_validate);
        g_dsv4_logit_dump = &*dsv4_logit_dump;
        if (dsv4_hot_path_neutral_validate) {
            console::log(
                "DSV4 hot-path-neutral validation: writing %s; source=server_prob_output; graph callbacks/readbacks/consume paths disabled\n",
                dump_path.string().c_str());
        }
    }

    std::optional<cli_dsv4_value_probe> dsv4_value_probe;
    if (cli_env_enabled("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_COMPRESSOR_UPDATE_V3_BACKEND_TAIL_VALUE_PROBE")) {
        dsv4_value_probe.emplace();
    }

    std::optional<cli_dsv4_rmoe_replace_dump> dsv4_rmoe_replace_dump;
    if (!cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_DUMP", "").empty()) {
        dsv4_rmoe_replace_dump.emplace(std::filesystem::path(cli_env_string(
            "LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_DUMP", "")));
        console::log("DSV4 routed-MoE replace dump: %s\n",
                cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_REPLACE_DUMP", "").c_str());
    }

    std::optional<cli_dsv4_rmoe_result_chain_readback> dsv4_rmoe_result_chain_readback;
    if (cli_env_string("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DSV4_ROUTED_MOE_BACKEND_OP_RESULT_CHAIN_MODE", "none").rfind("readback_", 0) == 0) {
        dsv4_rmoe_result_chain_readback.emplace();
    }

    cli_eval_capture_state eval_capture_state;
    eval_capture_state.oracle = oracle_dump.has_value() ? &*oracle_dump : nullptr;
    eval_capture_state.value_probe = dsv4_value_probe.has_value() ? &*dsv4_value_probe : nullptr;
    eval_capture_state.rmoe_replace_dump = dsv4_rmoe_replace_dump.has_value() ? &*dsv4_rmoe_replace_dump : nullptr;
    eval_capture_state.rmoe_result_chain_readback = dsv4_rmoe_result_chain_readback.has_value() ? &*dsv4_rmoe_result_chain_readback : nullptr;
    // The DSV4 logit harness records server-side sampled-token logits. Avoid
    // registering it as a graph eval callback because result_output may not
    // have a readable tensor backing in normal CLI scheduling.
    eval_capture_state.logit_dump = nullptr;
    if (eval_capture_state.oracle != nullptr ||
            eval_capture_state.value_probe != nullptr ||
            eval_capture_state.rmoe_replace_dump != nullptr ||
            eval_capture_state.rmoe_result_chain_readback != nullptr ||
            eval_capture_state.logit_dump != nullptr) {
        params.cb_eval = cli_eval_capture_cb_eval;
        params.cb_eval_user_data = &eval_capture_state;
    }

    common_init();

    // struct that contains llama context and inference
    cli_context ctx_cli(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    // TODO: avoid using atexit() here by making `console` a singleton
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    console::set_display(DISPLAY_TYPE_RESET);
    console::set_completion_callback(auto_completion_callback);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    // Print Flash-MoE settings before loading
    if (params.moe_mode != "stock") {
        auto env_enabled = [](const char * name) -> bool {
            const char * v = std::getenv(name);
            return v && v[0] != '\0' && v[0] != '0';
        };
        auto env_flag = [&](const char * name) -> const char * {
            return env_enabled(name) ? "on" : "off";
        };
        auto env_value = [](const char * name) -> const char * {
            const char * v = std::getenv(name);
            return (v && v[0] != '\0') ? v : "-";
        };
        auto deepseek2_gpu_bank_mode = []() -> const char * {
            const char * disable = std::getenv("LLAMA_FLASH_MOE_DISABLE_UNSAFE_DEEPSEEK2_GPU_BANK");
            if (disable && disable[0] != '\0' && disable[0] != '0') {
                return "off";
            }
            const char * allow = std::getenv("LLAMA_FLASH_MOE_ALLOW_UNSAFE_DEEPSEEK2_GPU_BANK");
            if (allow && allow[0] != '\0' && allow[0] != '0') {
                return "on (forced)";
            }
            return "on (default)";
        };
        fprintf(stderr, "\nFlash-MoE settings:\n");
        fprintf(stderr, "  mode             = %s\n", params.moe_mode.c_str());
        if (!params.moe_sidecar.empty()) {
            fprintf(stderr, "  sidecar          = %s\n", params.moe_sidecar.c_str());
        }
        if (!params.moe_prefetch_sidecar.empty()) {
            fprintf(stderr, "  prefetch-sidecar = %s\n", params.moe_prefetch_sidecar.c_str());
        }
        if (!params.moe_secondary_sidecar.empty()) {
            fprintf(stderr, "  secondary-sidecar = %s\n", params.moe_secondary_sidecar.c_str());
        }
        if (!params.moe_tertiary_sidecar.empty()) {
            fprintf(stderr, "  tertiary-sidecar = %s\n", params.moe_tertiary_sidecar.c_str());
        }
        fprintf(stderr, "  slot-bank        = %d\n", params.moe_slot_bank);
        fprintf(stderr, "  prefill-banks    = %d\n", params.moe_prefill_banks);
        fprintf(stderr, "  prefill-batch    = %d%s\n",
                params.moe_prefill_batch > 0 ? params.moe_prefill_batch : 8192,
                params.moe_prefill_batch > 0 ? "" : " (default)");
        fprintf(stderr, "  force-prefill-batch = %s\n", params.moe_force_prefill_batch ? "on" : "off");
        if (params.moe_prefill_micro_batch == COMMON_MOE_PREFILL_MICRO_BATCH_AUTO) {
            fprintf(stderr, "  prefill-micro-batch = auto (adaptive by prompt length)\n");
        } else {
            fprintf(stderr, "  prefill-micro-batch = %d%s\n",
                    params.moe_prefill_micro_batch > 0 ? params.moe_prefill_micro_batch :
                            (params.moe_prefill_batch > 0 ? params.moe_prefill_batch : 8192),
                    params.moe_prefill_micro_batch > 0 ? "" : " (follows prefill-batch)");
        }
        fprintf(stderr, "  prefill-next-hot = %d\n", params.moe_prefill_next_hot_experts);
        fprintf(stderr, "  prefill-next-hot-exclusive-drives = %s\n",
                params.moe_prefill_next_hot_exclusive_drives ? "on" : "off");
        fprintf(stderr, "  topk-override    = %d\n", params.moe_topk_override);
        fprintf(stderr, "  cache-io-split   = %d\n", params.moe_cache_io_split);
        fprintf(stderr, "  prefetch-cache-io-split = %d%s\n",
                params.moe_prefetch_cache_io_split > 0 ? params.moe_prefetch_cache_io_split : params.moe_cache_io_split,
                params.moe_prefetch_cache_io_split > 0 ? "" : " (follows cache-io-split)");
        fprintf(stderr, "  demand-stripe    = %s\n", params.moe_demand_stripe.empty() ? "off" : params.moe_demand_stripe.c_str());
        fprintf(stderr, "  demand-distribute = %s\n", params.moe_demand_distribute.empty() ? "off" : params.moe_demand_distribute.c_str());
        fprintf(stderr, "  demand-concurrent = %s\n", params.moe_demand_concurrent ? "on" : "off");
        fprintf(stderr, "  prefill-stripe   = %s\n", params.moe_prefill_stripe.empty() ? "follow demand" : params.moe_prefill_stripe.c_str());
        fprintf(stderr, "  prefill-distribute = %s\n", params.moe_prefill_distribute.empty() ? "follow demand" : params.moe_prefill_distribute.c_str());
        fprintf(stderr, "  prefetch-stripe  = %s\n", params.moe_prefetch_stripe.empty() ? "off" : params.moe_prefetch_stripe.c_str());
        fprintf(stderr, "  prefetch-distribute = %s\n", params.moe_prefetch_distribute.empty() ? "off" : params.moe_prefetch_distribute.c_str());
        fprintf(stderr, "  prefill-layer-major = %s\n", params.moe_prefill_layer_major ? "on" : "off");
        fprintf(stderr, "  prefetch-temporal = %s\n", (params.moe_prefetch_temporal || params.moe_prefetch_temporal_sparse) ? "on" : "off");
        fprintf(stderr, "  prefetch-temporal-sparse = %s\n", params.moe_prefetch_temporal_sparse ? "on" : "off");
        fprintf(stderr, "  predict-prev-token = %s\n", params.moe_predict_prev_token ? "on" : "off");
        fprintf(stderr, "  predict-top1-prev = %s\n", params.moe_predict_top1_prev ? "on" : "off");
        fprintf(stderr, "  predictor        = %s\n", params.moe_predictor.empty() ? "off" : params.moe_predictor.c_str());
        if (!params.moe_predictor.empty()) {
            fprintf(stderr, "  predictor-prefetch-topk = %d%s\n",
                    params.moe_predictor_prefetch_topk,
                    params.moe_predictor_prefetch_topk > 0 ? "" : " (follows predictor)");
        }
        fprintf(stderr, "  shared-only      = %s\n", params.moe_shared_only ? "on" : "off");
        fprintf(stderr, "  sort-decode-ids  = %s\n", params.moe_sort_decode_expert_ids ? "on" : "off");
        fprintf(stderr, "  trace-harness    = %s\n", params.moe_trace_harness ? "on" : "off");
        fprintf(stderr, "  n_gpu_layers     = %d\n", params.n_gpu_layers);
        {
#ifdef LLAMA_FLASH_MOE_GPU_BANK
            bool gpu_bank_compiled = true;
#else
            bool gpu_bank_compiled = false;
#endif
            bool gpu_bank_disabled = env_enabled("LLAMA_FLASH_MOE_DISABLE_GPU_BANK");
            bool gpu_bank_effective = gpu_bank_compiled && !gpu_bank_disabled;
            fprintf(stderr, "  gpu-bank         = %s (compiled=%s, env-disable=%s)\n",
                    gpu_bank_effective ? "on" : "off",
                    gpu_bank_compiled ? "on" : "off",
                    gpu_bank_disabled ? "on" : "off");
        }
        fprintf(stderr, "  ds2/kimi-gpu-bank = %s\n", deepseek2_gpu_bank_mode());
        fprintf(stderr, "  parallel-reads   = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_PARALLEL_SLOT_READS"));
        fprintf(stderr, "  async-upload     = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_ASYNC_SLOT_UPLOAD"));
        fprintf(stderr, "  mixed-slot-buffer= %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_MIXED_SLOT_BUFFER"));
        fprintf(stderr, "  metal-slot-decode= %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SLOT_DECODE"));
        fprintf(stderr, "  metal-split-glu  = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_SPLIT_GLU"));
        fprintf(stderr, "  metal-disable-mm = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_MUL_MM"));
        fprintf(stderr, "  metal-disable-mm-id = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_MUL_MM_ID"));
        fprintf(stderr, "  metal-disable-generic-mm = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_GENERIC_MM"));
        fprintf(stderr, "  metal-disable-op-mul-mat = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_OP_MUL_MAT"));
        fprintf(stderr, "  metal-disable-op-mul-mat-id = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_OP_MUL_MAT_ID"));
        fprintf(stderr, "  metal-disable-op-get-rows = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_OP_GET_ROWS"));
        fprintf(stderr, "  metal-disable-routed = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_ROUTED"));
        fprintf(stderr, "  metal-routed-only = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_ROUTED_ONLY"));
        fprintf(stderr, "  metal-disable-routed-post = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_ROUTED_POST"));
        fprintf(stderr, "  metal-disable-routed-types = %s\n", env_value("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_ROUTED_TYPES"));
        fprintf(stderr, "  metal-disable-shared-types = %s\n", env_value("LLAMA_FLASH_MOE_EXPERIMENTAL_METAL_DISABLE_SHARED_TYPES"));
        fprintf(stderr, "  cpu-vis-writes   = %s\n", env_flag("LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES"));
        fprintf(stderr, "\n");
    }

    console::log("\nLoading model... "); // followed by loading animation
    console::spinner::start();
    if (!ctx_cli.ctx_server.load_model(params)) {
        console::spinner::stop();
        console::error("\nFailed to load the model\n");
        return 1;
    }

    console::spinner::stop();
    console::log("\n");

    std::thread inference_thread([&ctx_cli]() {
        ctx_cli.ctx_server.start_loop();
    });

    auto inf = ctx_cli.ctx_server.get_meta();
    if (oracle_dump.has_value()) {
        const llama_context * lctx = ctx_cli.ctx_server.get_llama_context();
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(lctx));
        oracle_dump->set_model_path(inf.model_path);
        oracle_dump->set_prompt(params.prompt, common_tokenize(vocab, params.prompt, false, true));
    }
    if (dsv4_logit_dump.has_value()) {
        const llama_context * lctx = ctx_cli.ctx_server.get_llama_context();
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(lctx));
        dsv4_logit_dump->set_vocab(vocab);
        dsv4_logit_dump->set_model_path(inf.model_path);
    }
    std::string modalities = "text";
    if (inf.has_inp_image) {
        modalities += ", vision";
    }
    if (inf.has_inp_audio) {
        modalities += ", audio";
    }

    auto add_system_prompt = [&]() {
        if (!params.system_prompt.empty()) {
            ctx_cli.messages.push_back({
                {"role",    "system"},
                {"content", params.system_prompt}
            });
        }
    };
    add_system_prompt();

    console::log("\n");
    console::log("%s\n", LLAMA_ASCII_LOGO);
    console::log("build      : %s\n", inf.build_info.c_str());
    console::log("model      : %s\n", inf.model_name.c_str());
    console::log("modalities : %s\n", modalities.c_str());
    if (!params.system_prompt.empty()) {
        console::log("using custom system prompt\n");
    }
    if (params.moe_trace_harness) {
        console::log("mode       : Flash-MoE trace harness (raw completion)\n");
    }
    console::log("\n");
    if (!params.moe_trace_harness) {
        console::log("available commands:\n");
        console::log("  /exit or Ctrl+C     stop or exit\n");
        console::log("  /regen              regenerate the last response\n");
        console::log("  /clear              clear the chat history\n");
        console::log("  /read               add a text file\n");
        if (inf.has_inp_image) {
            console::log("  /image <file>       add an image file\n");
        }
        if (inf.has_inp_audio) {
            console::log("  /audio <file>       add an audio file\n");
        }
        console::log("\n");
    }

    if (params.moe_trace_harness) {
        if (params.prompt.empty()) {
            console::error("Flash-MoE trace harness requires --prompt\n");
            ctx_cli.ctx_server.terminate();
            inference_thread.join();
            return 1;
        }

        result_timings timings;
        if (params.display_prompt) {
            console::log("\n> %s\n\n", params.prompt.c_str());
        } else {
            console::log("\n");
        }
        std::string assistant_content = ctx_cli.generate_raw_completion(params.prompt, timings);
        (void) assistant_content;
        console::log("\n");

        if (params.show_timings) {
            console::set_display(DISPLAY_TYPE_INFO);
            console::log("\n");
            console::log("[ Prompt: %.1f t/s | Generation: %.1f t/s ] [ tokens - prefill: %d, decode: %d ]\n",
                    timings.prompt_per_second,
                    timings.predicted_per_second,
                    timings.prompt_n,
                    timings.predicted_n);
            console::set_display(DISPLAY_TYPE_RESET);
        }

        console::set_display(DISPLAY_TYPE_RESET);
        console::log("\nExiting...\n");
        ctx_cli.ctx_server.terminate();
        inference_thread.join();
        if (oracle_dump.has_value()) {
            oracle_dump->finish();
            console::log("oracle dump: %s\n", params.oracle_dump.c_str());
        }
        if (dsv4_logit_dump.has_value()) {
            dsv4_logit_dump->finish();
            console::log("DSV4 logit dump: %s\n", dsv4_logit_dump->path().string().c_str());
        }
        if (dsv4_rmoe_replace_dump.has_value()) {
            dsv4_rmoe_replace_dump->finish();
        }
        common_log_set_verbosity_thold(LOG_LEVEL_INFO);
        llama_memory_breakdown_print(ctx_cli.ctx_server.get_llama_context());
        return 0;
    }

    // interactive loop
    std::string cur_msg;
    while (true) {
        std::string buffer;
        console::set_display(DISPLAY_TYPE_USER_INPUT);
        if (params.prompt.empty()) {
            console::log("\n> ");
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
        } else {
            // process input prompt from args
            for (auto & fname : params.image) {
                std::string marker = ctx_cli.load_input_file(fname, true);
                if (marker.empty()) {
                    console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                    break;
                }
                console::log("Loaded media from '%s'\n", fname.c_str());
                cur_msg += marker;
            }
            buffer = params.prompt;
            if (buffer.size() > 500) {
                console::log("\n> %s ... (truncated)\n", buffer.substr(0, 500).c_str());
            } else {
                console::log("\n> %s\n", buffer.c_str());
            }
            params.prompt.clear(); // only use it once
        }
        console::set_display(DISPLAY_TYPE_RESET);
        console::log("\n");

        if (should_stop()) {
            g_is_interrupted.store(false);
            break;
        }

        // remove trailing newline
        if (!buffer.empty() &&buffer.back() == '\n') {
            buffer.pop_back();
        }

        // skip empty messages
        if (buffer.empty()) {
            continue;
        }

        bool add_user_msg = true;

        // process commands
        if (string_starts_with(buffer, "/exit")) {
            break;
        } else if (string_starts_with(buffer, "/regen")) {
            if (ctx_cli.messages.size() >= 2) {
                size_t last_idx = ctx_cli.messages.size() - 1;
                ctx_cli.messages.erase(last_idx);
                add_user_msg = false;
            } else {
                console::error("No message to regenerate.\n");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            ctx_cli.messages.clear();
            add_system_prompt();

            ctx_cli.input_files.clear();
            console::log("Chat history cleared.\n");
            continue;
        } else if (
                (string_starts_with(buffer, "/image ") && inf.has_inp_image) ||
                (string_starts_with(buffer, "/audio ") && inf.has_inp_audio)) {
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname = string_strip(buffer.substr(7));
            std::string marker = ctx_cli.load_input_file(fname, true);
            if (marker.empty()) {
                console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            cur_msg += marker;
            console::log("Loaded media from '%s'\n", fname.c_str());
            continue;
        } else if (string_starts_with(buffer, "/read ")) {
            std::string fname = string_strip(buffer.substr(6));
            std::string marker = ctx_cli.load_input_file(fname, false);
            if (marker.empty()) {
                console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            if (inf.fim_sep_token != LLAMA_TOKEN_NULL) {
                cur_msg += common_token_to_piece(ctx_cli.ctx_server.get_llama_context(), inf.fim_sep_token, true);
                cur_msg += fname;
                cur_msg.push_back('\n');
            } else {
                cur_msg += "--- File: ";
                cur_msg += fname;
                cur_msg += " ---\n";
            }
            cur_msg += marker;
            console::log("Loaded text from '%s'\n", fname.c_str());
            continue;
        } else {
            // not a command
            cur_msg += buffer;
        }

        // generate response
        if (add_user_msg) {
            ctx_cli.messages.push_back({
                {"role",    "user"},
                {"content", cur_msg}
            });
            cur_msg.clear();
        }
        result_timings timings;
        std::string assistant_content = ctx_cli.generate_completion(timings);
        ctx_cli.messages.push_back({
            {"role",    "assistant"},
            {"content", assistant_content}
        });
        console::log("\n");

        if (params.show_timings) {
            console::set_display(DISPLAY_TYPE_INFO);
            console::log("\n");
            console::log("[ Prompt: %.1f t/s | Generation: %.1f t/s ] [ tokens - prefill: %d, decode: %d ]\n",
                    timings.prompt_per_second,
                    timings.predicted_per_second,
                    timings.prompt_n,
                    timings.predicted_n);
            console::set_display(DISPLAY_TYPE_RESET);
        }

        if (params.single_turn) {
            break;
        }
    }

    console::set_display(DISPLAY_TYPE_RESET);

    console::log("\nExiting...\n");
    ctx_cli.ctx_server.terminate();
    inference_thread.join();
    if (oracle_dump.has_value()) {
        oracle_dump->finish();
        console::log("oracle dump: %s\n", params.oracle_dump.c_str());
    }
    if (dsv4_logit_dump.has_value()) {
        dsv4_logit_dump->finish();
        console::log("DSV4 logit dump: %s\n", dsv4_logit_dump->path().string().c_str());
    }
    if (dsv4_rmoe_replace_dump.has_value()) {
        dsv4_rmoe_replace_dump->finish();
    }

    // bump the log level to display timings
    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    llama_memory_breakdown_print(ctx_cli.ctx_server.get_llama_context());

    return 0;
}
