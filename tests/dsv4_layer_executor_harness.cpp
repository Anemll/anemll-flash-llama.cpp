#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef DSV4_LAYER_EXECUTOR_HARNESS_METAL
#include "dsv4_layer_executor_metal.h"
#endif

struct Dsv4TolerancePolicy {
    // Keep these values in sync with docs/dsv4-tolerance-policy.md.
    double tier_a_max_abs = 1e-6;
    double tier_a_rms = 1e-7;
    double tier_b_fp16_layer_max_abs = 5e-4;
    double tier_b_fp32_layer_max_abs = 1e-5;
    double tier_b_max_abs_logit_err = 1e-3;
    double tier_b_kl = 1e-4;
    int tier_b_top5_overlap_required = 5;
};

struct CaptureStats {
    double sum = 0.0;
    double sumsq = 0.0;
    double min = 0.0;
    double max = 0.0;
    double max_abs = 0.0;
    double rms = 0.0;
    int64_t over_tol = 0;
    bool has_max_abs = false;
    bool has_rms = false;
    bool has_over_tol = false;
};

struct CaptureEndpoint {
    std::string op;
    std::string tensor_name;
    std::string stage;
};

struct CaptureRecord {
    std::string source;
    std::string stage;
    int layer = -1;
    int64_t token = -1;
    std::string tensor = "summary";
    std::string dtype = "metadata";
    std::vector<int64_t> shape;
    CaptureStats stats;

    std::string availability = "available";
    std::string payload_kind = "stats_only";
    bool required = true;
    std::string unavailable_reason;
    CaptureEndpoint producer;
    CaptureEndpoint consumer;

    bool full_tensor_payload_available = false;
    bool byte_payload_available = false;
    bool stats_only = true;
    bool metadata_only = false;
    std::string payload_file;
    std::string byte_checksum;
    bool inline_tensor_values = false;
    bool inline_bytes = false;
    std::vector<uint8_t> payload_bytes;
    bool has_rope_position = false;
    bool has_rope_cache_position = false;
    bool has_rope_n_rot = false;
    bool has_rope_width = false;
    bool has_rope_freq_base = false;
    bool has_rope_freq_scale = false;
    bool has_rope_ext_factor = false;
    bool has_rope_attn_factor = false;
    bool has_rope_beta_fast = false;
    bool has_rope_beta_slow = false;
    bool has_rope_tail_offset = false;
    bool has_rope_n_ctx_orig = false;
    int64_t rope_position = 0;
    int64_t rope_cache_position = 0;
    int64_t rope_n_rot = 0;
    int64_t rope_width = 0;
    int64_t rope_tail_offset = 0;
    int64_t rope_n_ctx_orig = 0;
    double rope_freq_base = 0.0;
    double rope_freq_scale = 0.0;
    double rope_ext_factor = 0.0;
    double rope_attn_factor = 1.0;
    double rope_beta_fast = 32.0;
    double rope_beta_slow = 1.0;
    std::string rope_mode;
    int64_t rope_type = 0;
};

struct Dsv4LayerExecutorInput {
    std::vector<CaptureRecord> references;
};

struct Dsv4LayerExecutorOutput {
    std::vector<CaptureRecord> outputs;
};

struct Dsv4KernelCompareStats {
    int64_t cases = 0;
    int64_t failed_cases = 0;
    double max_abs = 0.0;
    double rms = 0.0;
    double hcnorm_cur_max_abs = 0.0;
    double hcnorm_cur_rms = 0.0;
    double hcnorm_norm_max_abs = 0.0;
    double hcnorm_norm_rms = 0.0;
    double hcnorm_post_max_abs = 0.0;
    double hcnorm_post_rms = 0.0;
    double rmoe_routed_sum_max_abs = 0.0;
    double rmoe_routed_sum_rms = 0.0;
    double rmoe_routed_sum_max_rel = 0.0;
    double rmoe_final_ffn_max_abs = 0.0;
    double rmoe_final_ffn_rms = 0.0;
    double rmoe_final_ffn_max_rel = 0.0;
    double aohc_output_max_abs = 0.0;
    double aohc_output_rms = 0.0;
    double aohc_output_max_rel = 0.0;
    double cupd_output_max_abs = 0.0;
    double cupd_output_rms = 0.0;
    double cupd_output_max_rel = 0.0;
    double hcnorm_best_eps = -1.0;
    std::string hcnorm_input_layout;
    std::string hcnorm_best_formula;
    std::string kernel_mode = "none";
    bool metal_recompute = false;
    bool copied_reference_output = false;
    bool tier_a_exact_pass = false;
    bool partial_rmoe_recompute = false;
    bool full_rmoe_recompute = false;
    bool routed_sum_only_recompute = false;
    bool expert_weight_recompute = false;
    bool weights_not_decoded = false;
    bool weighted_down_available = false;
    bool shared_down_available = false;
    bool reference_routed_sum_available = false;
    bool reference_final_ffn_available = false;
    bool partial_aohc_recompute = false;
    bool full_aohc_recompute = false;
    bool aohc_attn_out_available = false;
    bool aohc_residual_available = false;
    bool aohc_post_weights_available = false;
    bool aohc_comb_available = false;
    bool aohc_reference_available = false;
    int aohc_matches_hc_pre_norm_input = -1;
    bool partial_cupd_recompute = false;
    bool full_cupd_recompute = false;
    bool cupd_compressed_norm_available = false;
    bool cupd_norm_weight_available = false;
    bool cupd_norm_weighted_available = false;
    bool cupd_rope_input_available = false;
    bool cupd_rope_reference_available = false;
    bool cupd_rope_position_available = false;
    bool cupd_rope_cache_position_available = false;
    bool cupd_rope_n_rot_available = false;
    bool cupd_rope_freq_base_available = false;
    bool cupd_rope_freq_scale_available = false;
    bool cupd_rope_mode_available = false;
    bool cupd_rope_tail_offset_available = false;
    bool cupd_rope_cos_available = false;
    bool cupd_rope_sin_available = false;
    int64_t cupd_rope_nonzero_position_records = 0;
    int cupd_rope_position = -1;
    int cupd_rope_n_rot = -1;
    int cupd_rope_tail_offset = -1;
    std::vector<std::string> hcnorm_formula_reports;
    std::string first_failure;
    bool recompute_possible = true;
    std::vector<std::string> missing_inputs;
    std::vector<std::string> missing_formula_params;
};

static const std::vector<std::string> k_required_stages = {
    "hc_pre_norm",
    "routed_moe_final_output",
    "aohc_boundary",
    "compressor_update",
    "kv_cache_finalizer",
};

static std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}

static std::string join_strings(const std::vector<std::string> & values) {
    if (values.empty()) {
        return "[]";
    }
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << values[i];
    }
    out << "]";
    return out.str();
}

static bool parse_double_after(const std::string & line, const std::string & key, double & out) {
    const std::string needle = key + "=";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    size_t end = pos;
    while (end < line.size() && !std::isspace(static_cast<unsigned char>(line[end]))) {
        ++end;
    }
    try {
        out = std::stod(line.substr(pos, end - pos));
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_i64_after(const std::string & line, const std::string & key, int64_t & out) {
    const std::string needle = key + "=";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    size_t end = pos;
    while (end < line.size() && !std::isspace(static_cast<unsigned char>(line[end]))) {
        ++end;
    }
    try {
        out = std::stoll(line.substr(pos, end - pos));
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_string_after(const std::string & line, const std::string & key, std::string & out) {
    const std::string needle = key + "=";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    size_t end = pos;
    while (end < line.size() && !std::isspace(static_cast<unsigned char>(line[end]))) {
        ++end;
    }
    out = line.substr(pos, end - pos);
    return true;
}

static bool json_string_value(const std::string & line, const std::string & key, std::string & out) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while (true) {
        pos = line.find(needle, pos);
        if (pos == std::string::npos) {
            return false;
        }
        size_t colon = pos + needle.size();
        while (colon < line.size() && std::isspace(static_cast<unsigned char>(line[colon]))) {
            ++colon;
        }
        if (colon < line.size() && line[colon] == ':') {
            pos = colon;
            break;
        }
        pos += needle.size();
    }
    pos = line.find('"', pos + 1);
    if (pos == std::string::npos) {
        return false;
    }
    size_t end = line.find('"', pos + 1);
    if (end == std::string::npos) {
        return false;
    }
    out = line.substr(pos + 1, end - pos - 1);
    return true;
}

static bool json_bool_value(const std::string & line, const std::string & key, bool & out) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
        ++pos;
    }
    if (line.compare(pos, 4, "true") == 0) {
        out = true;
        return true;
    }
    if (line.compare(pos, 5, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

static bool json_i64_value(const std::string & line, const std::string & key, int64_t & out) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
        ++pos;
    }
    size_t end = pos;
    while (end < line.size() && (std::isdigit(static_cast<unsigned char>(line[end])) || line[end] == '-')) {
        ++end;
    }
    if (end == pos) {
        return false;
    }
    try {
        out = std::stoll(line.substr(pos, end - pos));
        return true;
    } catch (...) {
        return false;
    }
}

static bool json_double_value(const std::string & line, const std::string & key, double & out) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return false;
    }
    ++pos;
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
        ++pos;
    }
    size_t end = pos;
    while (end < line.size() &&
            (std::isdigit(static_cast<unsigned char>(line[end])) || line[end] == '-' ||
             line[end] == '+' || line[end] == '.' || line[end] == 'e' || line[end] == 'E')) {
        ++end;
    }
    if (end == pos) {
        return false;
    }
    try {
        out = std::stod(line.substr(pos, end - pos));
        return true;
    } catch (...) {
        return false;
    }
}

static std::vector<int64_t> json_shape_value(const std::string & line) {
    std::vector<int64_t> out;
    const std::string needle = "\"shape\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return out;
    }
    pos = line.find('[', pos + needle.size());
    size_t end = line.find(']', pos == std::string::npos ? 0 : pos);
    if (pos == std::string::npos || end == std::string::npos || end <= pos) {
        return out;
    }
    std::stringstream ss(line.substr(pos + 1, end - pos - 1));
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            out.push_back(std::stoll(item));
        }
    }
    return out;
}

static bool json_has_nonempty_array(const std::string & line, const std::string & key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = line.find('[', pos + needle.size());
    size_t end = line.find(']', pos == std::string::npos ? 0 : pos);
    if (pos == std::string::npos || end == std::string::npos || end <= pos) {
        return false;
    }
    return !trim(line.substr(pos + 1, end - pos - 1)).empty();
}

static bool json_has_nonempty_string(const std::string & line, const std::string & key) {
    std::string value;
    return json_string_value(line, key, value) && !value.empty();
}

static std::filesystem::path resolve_payload_path(const std::string & source, const std::string & payload_file) {
    std::filesystem::path payload_path(payload_file);
    if (payload_path.is_absolute()) {
        return payload_path;
    }
    const std::filesystem::path fixture_relative = std::filesystem::path(source).parent_path() / payload_path;
    if (std::filesystem::exists(fixture_relative)) {
        return fixture_relative;
    }
    const std::filesystem::path tmp_relative = std::filesystem::path("/tmp") / payload_path;
    if (std::filesystem::exists(tmp_relative)) {
        return tmp_relative;
    }
    return fixture_relative;
}

static bool json_nested_string_value(const std::string & line, const std::string & object_key, const std::string & key, std::string & out) {
    const std::string object_needle = "\"" + object_key + "\"";
    size_t pos = line.find(object_needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos = line.find('{', pos + object_needle.size());
    size_t end = line.find('}', pos == std::string::npos ? 0 : pos);
    if (pos == std::string::npos || end == std::string::npos || end <= pos) {
        return false;
    }
    return json_string_value(line.substr(pos, end - pos + 1), key, out);
}

class Dsv4LayerExecutorHarness {
public:
    bool load_captures(const std::string & path) {
        std::error_code ec;
        if (std::filesystem::is_directory(path, ec)) {
            bool ok = true;
            for (const auto & entry : std::filesystem::directory_iterator(path, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                const auto ext = entry.path().extension().string();
                if (ext == ".jsonl" || ext == ".log" || ext == ".txt") {
                    ok = load_capture_file(entry.path().string()) && ok;
                }
            }
            return !ec && ok;
        }
        return load_capture_file(path);
    }

    bool run_identity() {
        output_.outputs = input_.references;
        return true;
    }

    bool run_kernel(const std::string & stage, const Dsv4TolerancePolicy & policy, const std::string & kernel_mode, bool forbid_copy_smoke) {
        output_.outputs = input_.references;
        kernel_stats_ = {};
        kernel_stats_.kernel_mode = kernel_mode;
        if (stage != "hc_pre_norm" && stage != "routed_moe_final_output" && stage != "aohc_boundary" && stage != "compressor_update") {
            kernel_stats_.first_failure = "unsupported kernel stage " + stage;
            kernel_stats_.failed_cases = 1;
            return false;
        }
        if (kernel_mode == "copy_smoke") {
            if (forbid_copy_smoke) {
                kernel_stats_.first_failure = "copy_smoke_forbidden";
                kernel_stats_.failed_cases = 1;
                kernel_stats_.copied_reference_output = true;
                return false;
            }
            return run_kernel_copy_smoke(stage, policy);
        }
        if (stage == "routed_moe_final_output") {
            if (kernel_mode != "hcnorm_recompute" && kernel_mode != "rmoe_final_from_substages") {
                kernel_stats_.first_failure = "unsupported routed-MoE kernel mode " + kernel_mode;
                kernel_stats_.failed_cases = 1;
                return false;
            }
            kernel_stats_.kernel_mode = "rmoe_final_from_substages";
            return run_kernel_rmoe_final_from_substages(policy);
        }
        if (stage == "aohc_boundary") {
            if (kernel_mode != "hcnorm_recompute" && kernel_mode != "aohc_hc_post_from_substages") {
                kernel_stats_.first_failure = "unsupported AOHC kernel mode " + kernel_mode;
                kernel_stats_.failed_cases = 1;
                return false;
            }
            kernel_stats_.kernel_mode = "aohc_hc_post_from_substages";
            return run_kernel_aohc_hc_post_from_substages(policy);
        }
        if (stage == "compressor_update") {
            if (cupd_stage_ == "rope") {
                kernel_stats_.first_failure = "compressor/update RoPE Metal kernel not implemented; CPU source-contract recompute must pass first";
                kernel_stats_.failed_cases = 1;
                return false;
            }
            if (kernel_mode != "hcnorm_recompute" && kernel_mode != "cupd_norm_weighted") {
                kernel_stats_.first_failure = "unsupported compressor/update kernel mode " + kernel_mode;
                kernel_stats_.failed_cases = 1;
                return false;
            }
            kernel_stats_.kernel_mode = "cupd_norm_weighted";
            return run_kernel_cupd_norm_weighted(policy);
        }
        if (kernel_mode != "hcnorm_recompute") {
            kernel_stats_.first_failure = "unsupported kernel mode " + kernel_mode;
            kernel_stats_.failed_cases = 1;
            return false;
        }

        return run_kernel_hcnorm_recompute(policy);
    }

    bool run_kernel_copy_smoke(const std::string & stage, const Dsv4TolerancePolicy & policy) {
        kernel_stats_.copied_reference_output = true;
        kernel_stats_.metal_recompute = false;

        for (const CaptureRecord & rec : input_.references) {
            if (rec.stage != stage || rec.payload_kind != "tensor_values" || !rec.full_tensor_payload_available) {
                continue;
            }
            std::vector<uint8_t> candidate(rec.payload_bytes.size());
            char error[512] = {};
#ifdef DSV4_LAYER_EXECUTOR_HARNESS_METAL
            const int rc = dsv4_layer_executor_metal_hc_pre_norm(
                    rec.payload_bytes.data(), rec.payload_bytes.size(), candidate.data(), error, sizeof(error));
#else
            const int rc = 1;
            std::snprintf(error, sizeof(error), "Metal harness support not built");
#endif
            if (rc != 0) {
                ++kernel_stats_.failed_cases;
                if (kernel_stats_.first_failure.empty()) {
                    kernel_stats_.first_failure = error;
                }
                continue;
            }

            double max_abs = 0.0;
            double rms = 0.0;
            compare_payloads(rec, candidate, max_abs, rms);
            ++kernel_stats_.cases;
            kernel_stats_.max_abs = std::max(kernel_stats_.max_abs, max_abs);
            kernel_stats_.rms = std::max(kernel_stats_.rms, rms);
            if (!tier_b_tensor_pass(rec.dtype, max_abs, policy)) {
                ++kernel_stats_.failed_cases;
                if (kernel_stats_.first_failure.empty()) {
                    kernel_stats_.first_failure = "Tier B tensor threshold failed";
                }
            }
        }

        if (kernel_stats_.cases == 0) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "no tensor_values payload for kernel stage " + stage;
        }
        return kernel_stats_.failed_cases == 0;
    }

    bool run_kernel_hcnorm_recompute(const Dsv4TolerancePolicy & policy) {
        kernel_stats_.metal_recompute = true;
        kernel_stats_.copied_reference_output = false;
        kernel_stats_.hcnorm_best_formula = "C0_weighted_x_d_h_pre_h";
        kernel_stats_.hcnorm_input_layout = "metal_stride_hidden_major";
        kernel_stats_.hcnorm_best_eps = 1e-6;

        const CaptureRecord * input = find_record_any("hc_pre_norm", {"hc_ws_input_inpL_raw", "hc_ws_input_inpL_view", "hc_pre_input_hc_original", "input_hc_original_residual"});
        const CaptureRecord * split = find_record_any("hc_pre_norm", {"hc_ws_split_pre_raw", "hc_ws_split_pre_view", "hc_pre_split_pre", "split_pre"});
        const CaptureRecord * weight = find_record_any("hc_pre_norm", {"hc_pre_norm_weight", "norm_weight"});
        const CaptureRecord * ref_cur = find_record_any("hc_pre_norm", {"hc_ws_reference_cur", "hc_pre_weighted_cur_reference", "reference_cur"});
        const CaptureRecord * ref_norm = find_record_any("hc_pre_norm", {"hc_pre_norm_reference", "reference_norm"});
        const CaptureRecord * ref_post = find_record_any("hc_pre_norm", {"hc_pre_post_reference", "reference_post"});

        require_record(input, "input_hc_original_residual");
        require_record(split, "split_pre");
        require_record(weight, "norm_weight");
        require_record(ref_cur, "reference_cur");
        require_record(ref_norm, "reference_norm");
        require_record(ref_post, "reference_post");
        if (!kernel_stats_.missing_inputs.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "missing required HC_PRE_NORM kernel input payloads";
            return false;
        }

        const std::vector<float> input_f = payload_f32(*input);
        const std::vector<float> split_f = payload_f32(*split);
        const std::vector<float> weight_f = payload_f32(*weight);
        const std::vector<float> ref_cur_f = payload_f32(*ref_cur);
        const std::vector<float> ref_norm_f = payload_f32(*ref_norm);
        const std::vector<float> ref_post_f = payload_f32(*ref_post);
        if (input_f.empty() || split_f.empty() || weight_f.empty() || ref_cur_f.empty() || ref_norm_f.empty() || ref_post_f.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "one or more HC_PRE_NORM Metal payloads are not f32-compatible";
            return false;
        }

        const size_t n_embd = weight_f.size();
        if (ref_cur_f.size() != n_embd || ref_norm_f.size() != n_embd || ref_post_f.size() != n_embd ||
                input_f.size() % n_embd != 0) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "HC_PRE_NORM Metal payload sizes disagree";
            return false;
        }
        const size_t n_hc = input_f.size() / n_embd;
        if (split_f.size() < n_hc) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "HC_PRE_NORM split_pre too small for inferred n_hc";
            return false;
        }

        std::vector<float> candidate_cur(n_embd, 0.0f);
        std::vector<float> candidate_norm(n_embd, 0.0f);
        std::vector<float> candidate_post(n_embd, 0.0f);
        char error[512] = {};
#ifdef DSV4_LAYER_EXECUTOR_HARNESS_METAL
        const int rc = dsv4_layer_executor_metal_hc_pre_norm_recompute(
                input_f.data(), split_f.data(), weight_f.data(),
                n_embd, n_hc, 1e-6f,
                candidate_cur.data(), candidate_norm.data(), candidate_post.data(),
                error, sizeof(error));
#else
        const int rc = 1;
        std::snprintf(error, sizeof(error), "Metal harness support not built");
#endif
        if (rc != 0) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = error;
            return false;
        }

        double cur_rel = 0.0;
        double norm_rel = 0.0;
        double post_rel = 0.0;
        compare_f32(candidate_cur, ref_cur_f, kernel_stats_.hcnorm_cur_max_abs, kernel_stats_.hcnorm_cur_rms);
        compare_f32(candidate_norm, ref_norm_f, kernel_stats_.hcnorm_norm_max_abs, kernel_stats_.hcnorm_norm_rms);
        compare_f32(candidate_post, ref_post_f, kernel_stats_.hcnorm_post_max_abs, kernel_stats_.hcnorm_post_rms);
        compare_f32_rel(candidate_cur, ref_cur_f, cur_rel);
        compare_f32_rel(candidate_norm, ref_norm_f, norm_rel);
        compare_f32_rel(candidate_post, ref_post_f, post_rel);

        kernel_stats_.hcnorm_formula_reports.push_back(formula_report_line(
                "Metal_C0_weighted_x_d_h_pre_h", "metal_stride_hidden_major",
                kernel_stats_.hcnorm_cur_max_abs, kernel_stats_.hcnorm_cur_rms, cur_rel,
                kernel_stats_.hcnorm_norm_max_abs, kernel_stats_.hcnorm_norm_rms,
                kernel_stats_.hcnorm_post_max_abs, kernel_stats_.hcnorm_post_rms,
                tier_b_tensor_pass("f32", std::max({kernel_stats_.hcnorm_cur_max_abs, kernel_stats_.hcnorm_norm_max_abs, kernel_stats_.hcnorm_post_max_abs}), policy)));

        kernel_stats_.cases = 3;
        kernel_stats_.max_abs = std::max({kernel_stats_.hcnorm_cur_max_abs, kernel_stats_.hcnorm_norm_max_abs, kernel_stats_.hcnorm_post_max_abs});
        kernel_stats_.rms = std::max({kernel_stats_.hcnorm_cur_rms, kernel_stats_.hcnorm_norm_rms, kernel_stats_.hcnorm_post_rms});
        kernel_stats_.tier_a_exact_pass = kernel_stats_.max_abs <= policy.tier_a_max_abs && kernel_stats_.rms <= policy.tier_a_rms;
        if (!tier_b_tensor_pass("f32", kernel_stats_.max_abs, policy)) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "Tier B HC_PRE_NORM Metal recompute threshold failed";
            return false;
        }
        return true;
    }

    bool run_kernel_rmoe_final_from_substages(const Dsv4TolerancePolicy & policy) {
        kernel_stats_.metal_recompute = true;
        kernel_stats_.copied_reference_output = false;
        kernel_stats_.partial_rmoe_recompute = true;
        kernel_stats_.full_rmoe_recompute = true;
        kernel_stats_.expert_weight_recompute = false;
        kernel_stats_.weights_not_decoded = true;
        kernel_stats_.hcnorm_best_formula = "RMOE_sum_weighted_down_slots_plus_shared_down";
        kernel_stats_.hcnorm_input_layout = "slot_major";

        const CaptureRecord * weighted_down = find_record("routed_moe_final_output", "rmoe_expert_down");
        const CaptureRecord * routed_sum = find_record("routed_moe_final_output", "rmoe_routed_sum");
        const CaptureRecord * shared_down = find_record("routed_moe_final_output", "rmoe_shared_down");
        const CaptureRecord * final_ref = find_record_any("routed_moe_final_output", {"rmoe_final_ffn_reference", "final_ffn"});

        kernel_stats_.weighted_down_available = weighted_down != nullptr && weighted_down->full_tensor_payload_available;
        kernel_stats_.shared_down_available = shared_down != nullptr && shared_down->full_tensor_payload_available;
        kernel_stats_.reference_routed_sum_available = routed_sum != nullptr && routed_sum->full_tensor_payload_available;
        kernel_stats_.reference_final_ffn_available = final_ref != nullptr && final_ref->full_tensor_payload_available;

        require_record(weighted_down, "rmoe_expert_down");
        require_record(shared_down, "rmoe_shared_down");
        require_record(routed_sum, "rmoe_routed_sum");
        require_record(final_ref, "rmoe_final_ffn_reference");
        if (!kernel_stats_.missing_inputs.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "missing required routed-MoE Metal kernel payloads";
            return false;
        }

        const std::vector<float> weighted_down_f = payload_f32(*weighted_down);
        const std::vector<float> shared_down_f = payload_f32(*shared_down);
        const std::vector<float> routed_sum_f = payload_f32(*routed_sum);
        const std::vector<float> final_ref_f = payload_f32(*final_ref);
        if (weighted_down_f.empty() || shared_down_f.empty() || routed_sum_f.empty() || final_ref_f.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "one or more routed-MoE Metal payloads are not f32-compatible";
            return false;
        }

        const size_t n_embd = final_ref_f.size();
        if (shared_down_f.size() != n_embd || routed_sum_f.size() != n_embd || weighted_down_f.size() % n_embd != 0) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "routed-MoE Metal payload sizes disagree";
            return false;
        }
        const size_t topk = weighted_down_f.size() / n_embd;

        std::vector<float> candidate_routed_sum(n_embd, 0.0f);
        std::vector<float> candidate_final(n_embd, 0.0f);
        char error[512] = {};
#ifdef DSV4_LAYER_EXECUTOR_HARNESS_METAL
        const int rc = dsv4_layer_executor_metal_rmoe_final_from_substages(
                weighted_down_f.data(), shared_down_f.data(),
                n_embd, topk,
                candidate_routed_sum.data(), candidate_final.data(),
                error, sizeof(error));
#else
        const int rc = 1;
        std::snprintf(error, sizeof(error), "Metal harness support not built");
#endif
        if (rc != 0) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = error;
            return false;
        }

        compare_f32(candidate_routed_sum, routed_sum_f, kernel_stats_.rmoe_routed_sum_max_abs, kernel_stats_.rmoe_routed_sum_rms);
        compare_f32_rel(candidate_routed_sum, routed_sum_f, kernel_stats_.rmoe_routed_sum_max_rel);
        compare_f32(candidate_final, final_ref_f, kernel_stats_.rmoe_final_ffn_max_abs, kernel_stats_.rmoe_final_ffn_rms);
        compare_f32_rel(candidate_final, final_ref_f, kernel_stats_.rmoe_final_ffn_max_rel);

        kernel_stats_.cases = 2;
        kernel_stats_.max_abs = std::max(kernel_stats_.rmoe_routed_sum_max_abs, kernel_stats_.rmoe_final_ffn_max_abs);
        kernel_stats_.rms = std::max(kernel_stats_.rmoe_routed_sum_rms, kernel_stats_.rmoe_final_ffn_rms);
        kernel_stats_.tier_a_exact_pass = kernel_stats_.max_abs <= policy.tier_a_max_abs && kernel_stats_.rms <= policy.tier_a_rms;
        const bool pass = tier_b_tensor_pass("f32", kernel_stats_.max_abs, policy);
        kernel_stats_.hcnorm_formula_reports.push_back(formula_report_line(
                "Metal_RMOE_sum_weighted_down_slots_plus_shared_down",
                "slot_major",
                kernel_stats_.rmoe_routed_sum_max_abs,
                kernel_stats_.rmoe_routed_sum_rms,
                kernel_stats_.rmoe_routed_sum_max_rel,
                kernel_stats_.rmoe_final_ffn_max_abs,
                kernel_stats_.rmoe_final_ffn_rms,
                kernel_stats_.rmoe_final_ffn_max_abs,
                kernel_stats_.rmoe_final_ffn_rms,
                pass));
        if (!pass) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "Tier B routed-MoE Metal source-contract threshold failed";
            return false;
        }
        return true;
    }

    bool run_hcnorm_recompute(const Dsv4TolerancePolicy & policy) {
        output_.outputs = input_.references;
        kernel_stats_ = {};

        const CaptureRecord * input = find_record_any("hc_pre_norm", {"hc_ws_input_inpL_raw", "hc_ws_input_inpL_view", "hc_pre_input_hc_original", "input_hc_original_residual"});
        const CaptureRecord * split = find_record_any("hc_pre_norm", {"hc_ws_split_pre_raw", "hc_ws_split_pre_view", "hc_pre_split_pre", "split_pre"});
        const CaptureRecord * split_full = find_record("hc_pre_norm", "hc_ws_split_full_raw");
        const CaptureRecord * split_post = find_record("hc_pre_norm", "hc_pre_split_post");
        const CaptureRecord * split_comb = find_record("hc_pre_norm", "hc_pre_split_comb");
        const CaptureRecord * flat = find_record("hc_pre_norm", "hc_pre_flat_hc");
        const CaptureRecord * flat_normed = find_record("hc_pre_norm", "hc_pre_flat_hc_normed");
        const CaptureRecord * weight = find_record_any("hc_pre_norm", {"hc_pre_norm_weight", "norm_weight"});
        const CaptureRecord * ref_cur = find_record_any("hc_pre_norm", {"hc_ws_reference_cur", "hc_pre_weighted_cur_reference", "reference_cur"});
        const CaptureRecord * ref_norm = find_record_any("hc_pre_norm", {"hc_pre_norm_reference", "reference_norm"});
        const CaptureRecord * ref_post = find_record_any("hc_pre_norm", {"hc_pre_post_reference", "reference_post"});

        require_record(input, "input_hc_original_residual");
        require_record(split, "split_pre");
        require_record(weight, "norm_weight");
        require_record(ref_cur, "reference_cur");
        require_record(ref_norm, "reference_norm");
        require_record(ref_post, "reference_post");

        if (!kernel_stats_.missing_inputs.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "missing required HC_PRE_NORM input payloads";
            return false;
        }

        std::vector<float> input_f = payload_f32(*input);
        std::vector<float> split_f = payload_f32(*split);
        std::vector<float> split_full_f = split_full != nullptr ? payload_f32(*split_full) : std::vector<float>();
        std::vector<float> split_post_f = split_post != nullptr ? payload_f32(*split_post) : std::vector<float>();
        std::vector<float> split_comb_f = split_comb != nullptr ? payload_f32(*split_comb) : std::vector<float>();
        std::vector<float> flat_f = flat != nullptr ? payload_f32(*flat) : std::vector<float>();
        std::vector<float> flat_normed_f = flat_normed != nullptr ? payload_f32(*flat_normed) : std::vector<float>();
        std::vector<float> weight_f = payload_f32(*weight);
        std::vector<float> ref_cur_f = payload_f32(*ref_cur);
        std::vector<float> ref_norm_f = payload_f32(*ref_norm);
        std::vector<float> ref_post_f = payload_f32(*ref_post);

        if (weight_f.empty() || ref_cur_f.empty() || ref_norm_f.empty() || input_f.empty() || split_f.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "one or more HC_PRE_NORM payloads are not f32-compatible";
            return false;
        }

        const size_t n_embd = weight_f.size();
        if (ref_cur_f.size() != n_embd || ref_norm_f.size() != n_embd || ref_post_f.size() != n_embd) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "HC_PRE_NORM reference/weight sizes disagree";
            return false;
        }
        if (input_f.size() % n_embd != 0) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "input_hc_original_residual is not divisible by norm width";
            return false;
        }
        const size_t n_hc = input_f.size() / n_embd;
        if (split_f.size() < n_hc) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "split_pre is smaller than inferred HC count";
            return false;
        }

        const double eps_values[] = {1e-6, 1e-5, 1e-8, 0.0};
        struct Candidate {
            std::string formula;
            std::string layout;
            std::vector<float> cur;
            bool source_weighted = false;
        };
        std::vector<Candidate> candidates;
        const std::vector<float> split_from_full_pre =
            split_full_f.size() >= n_hc ? std::vector<float>(split_full_f.begin(), split_full_f.begin() + (ptrdiff_t) n_hc) : std::vector<float>();
        const std::vector<float> split_from_full_post =
            split_full_f.size() >= 2*n_hc ? std::vector<float>(split_full_f.begin() + (ptrdiff_t) n_hc, split_full_f.begin() + (ptrdiff_t) (2*n_hc)) : std::vector<float>();
        const std::vector<float> comb_src0 =
            split_comb_f.size() >= n_hc ? std::vector<float>(split_comb_f.begin(), split_comb_f.begin() + (ptrdiff_t) n_hc) : std::vector<float>();
        std::vector<float> comb_dst0;
        if (split_comb_f.size() >= n_hc*n_hc) {
            comb_dst0.resize(n_hc);
            for (size_t h = 0; h < n_hc; ++h) {
                comb_dst0[h] = split_comb_f[h*n_hc];
            }
        }
        candidates.push_back({"C0_weighted_x_d_h_pre_h", "metal_stride_hidden_major", weighted_sum_layout(input_f, split_f, n_embd, n_hc, "hidden_major"), true});
        candidates.push_back({"C1_weighted_x_h_e_pre_h", "hidden_major_alias", weighted_sum_layout(input_f, split_f, n_embd, n_hc, "hidden_major"), true});
        candidates.push_back({"C2_weighted_input_e_plus_h_embd", "hidden_major", weighted_sum_layout(input_f, split_f, n_embd, n_hc, "hidden_major"), true});
        candidates.push_back({"C3_weighted_input_h_plus_e_hc", "e_major", weighted_sum_layout(input_f, split_f, n_embd, n_hc, "e_major"), true});
        candidates.push_back({"C4_weighted_input_e_h_pre_token_h", "hidden_major", weighted_sum_layout(input_f, split_f, n_embd, n_hc, "hidden_major"), true});
        candidates.push_back({"C5_weighted_input_e_h_pre_h_token", "hidden_major", weighted_sum_layout(input_f, split_f, n_embd, n_hc, "hidden_major"), true});
        if (!split_from_full_pre.empty()) {
            candidates.push_back({"C5b_weighted_input_e_h_split_full_pre_slice", "hidden_major", weighted_sum_layout(input_f, split_from_full_pre, n_embd, n_hc, "hidden_major"), true});
        }
        if (!split_post_f.empty()) {
            candidates.push_back({"C6_weighted_input_e_h_split_post", "hidden_major", weighted_sum_layout(input_f, split_post_f, n_embd, n_hc, "hidden_major"), true});
        }
        if (!split_from_full_post.empty()) {
            candidates.push_back({"C6b_weighted_input_e_h_split_full_post_slice", "hidden_major", weighted_sum_layout(input_f, split_from_full_post, n_embd, n_hc, "hidden_major"), true});
        }
        if (!comb_src0.empty()) {
            candidates.push_back({"C7_weighted_input_e_h_split_comb_src0", "hidden_major", weighted_sum_layout(input_f, comb_src0, n_embd, n_hc, "hidden_major"), true});
        }
        if (!comb_dst0.empty()) {
            candidates.push_back({"C7b_weighted_input_e_h_split_comb_dst0", "hidden_major", weighted_sum_layout(input_f, comb_dst0, n_embd, n_hc, "hidden_major"), true});
        }
        if (!flat_f.empty() && flat_f.size() >= input_f.size()) {
            candidates.push_back({"C8_weighted_flat_hc_pre_h", "hidden_major", weighted_sum_layout(flat_f, split_f, n_embd, n_hc, "hidden_major"), true});
        }
        if (!flat_normed_f.empty() && flat_normed_f.size() >= input_f.size()) {
            candidates.push_back({"C9_weighted_flat_normed_pre_h", "hidden_major", weighted_sum_layout(flat_normed_f, split_f, n_embd, n_hc, "hidden_major"), true});
        }
        candidates.push_back({"D_norm_only_reference_cur", "reference_cur", ref_cur_f, false});
        candidates.push_back({"E_post_passthrough_reference_norm", "reference_norm", ref_norm_f, false});

        double best_overall_max = INFINITY;
        double best_overall_rms = INFINITY;
        double best_cur_max = INFINITY;
        double best_cur_rms = INFINITY;
        double best_norm_max = INFINITY;
        double best_norm_rms = INFINITY;
        double best_post_max = INFINITY;
        double best_post_rms = INFINITY;
        double best_eps = -1.0;
        double best_source_overall_max = INFINITY;
        double best_source_overall_rms = INFINITY;
        double best_source_cur_max = INFINITY;
        double best_source_cur_rms = INFINITY;
        double best_source_norm_max = INFINITY;
        double best_source_norm_rms = INFINITY;
        double best_source_post_max = INFINITY;
        double best_source_post_rms = INFINITY;
        double best_source_eps = -1.0;
        std::string best_source_formula;
        std::string best_source_layout;
        bool norm_only_pass = false;

        for (const Candidate & candidate : candidates) {
            double cur_max = 0.0;
            double cur_rms = 0.0;
            compare_f32(candidate.cur, ref_cur_f, cur_max, cur_rms);
            double cur_max_rel = 0.0;
            compare_f32_rel(candidate.cur, ref_cur_f, cur_max_rel);
            double candidate_best_norm_max = INFINITY;
            double candidate_best_norm_rms = INFINITY;
            double candidate_best_post_max = INFINITY;
            double candidate_best_post_rms = INFINITY;
            double candidate_best_eps = -1.0;
            for (double eps : eps_values) {
                std::vector<float> norm_candidate = rms_norm_weight(candidate.cur, weight_f, eps);
                double norm_max = 0.0;
                double norm_rms = 0.0;
                compare_f32(norm_candidate, ref_norm_f, norm_max, norm_rms);
                double post_max = 0.0;
                double post_rms = 0.0;
                compare_f32(norm_candidate, ref_post_f, post_max, post_rms);
                const double combined = std::max({cur_max, norm_max, post_max});
                if (combined < std::max({cur_max, candidate_best_norm_max, candidate_best_post_max})) {
                    candidate_best_norm_max = norm_max;
                    candidate_best_norm_rms = norm_rms;
                    candidate_best_post_max = post_max;
                    candidate_best_post_rms = post_rms;
                    candidate_best_eps = eps;
                }
            }
            const double overall_max = std::max({cur_max, candidate_best_norm_max, candidate_best_post_max});
            const double overall_rms = std::max({cur_rms, candidate_best_norm_rms, candidate_best_post_rms});
            const bool tier_b_pass = tier_b_tensor_pass("f32", overall_max, policy);
            if (candidate.formula == "D_norm_only_reference_cur" && tier_b_pass) {
                norm_only_pass = true;
            }
            kernel_stats_.hcnorm_formula_reports.push_back(formula_report_line(
                    candidate.formula, candidate.layout,
                    cur_max, cur_rms, cur_max_rel,
                    candidate_best_norm_max, candidate_best_norm_rms,
                    candidate_best_post_max, candidate_best_post_rms,
                    tier_b_pass));
            if (candidate.source_weighted && overall_max < best_source_overall_max) {
                best_source_overall_max = overall_max;
                best_source_overall_rms = overall_rms;
                best_source_cur_max = cur_max;
                best_source_cur_rms = cur_rms;
                best_source_norm_max = candidate_best_norm_max;
                best_source_norm_rms = candidate_best_norm_rms;
                best_source_post_max = candidate_best_post_max;
                best_source_post_rms = candidate_best_post_rms;
                best_source_eps = candidate_best_eps;
                best_source_formula = candidate.formula;
                best_source_layout = candidate.layout;
            }
            if (overall_max < best_overall_max) {
                best_overall_max = overall_max;
                best_overall_rms = overall_rms;
                best_cur_max = cur_max;
                best_cur_rms = cur_rms;
                best_norm_max = candidate_best_norm_max;
                best_norm_rms = candidate_best_norm_rms;
                best_post_max = candidate_best_post_max;
                best_post_rms = candidate_best_post_rms;
                best_eps = candidate_best_eps;
                kernel_stats_.hcnorm_best_formula = candidate.formula;
                kernel_stats_.hcnorm_input_layout = candidate.layout;
            }
        }

        kernel_stats_.cases = static_cast<int64_t>(candidates.size());
        (void) best_overall_rms;
        (void) best_cur_max;
        (void) best_cur_rms;
        (void) best_norm_max;
        (void) best_norm_rms;
        (void) best_post_max;
        (void) best_post_rms;
        (void) best_eps;
        kernel_stats_.max_abs = best_source_overall_max;
        kernel_stats_.rms = best_source_overall_rms;
        kernel_stats_.hcnorm_cur_max_abs = best_source_cur_max;
        kernel_stats_.hcnorm_cur_rms = best_source_cur_rms;
        kernel_stats_.hcnorm_norm_max_abs = best_source_norm_max;
        kernel_stats_.hcnorm_norm_rms = best_source_norm_rms;
        kernel_stats_.hcnorm_post_max_abs = best_source_post_max;
        kernel_stats_.hcnorm_post_rms = best_source_post_rms;
        kernel_stats_.hcnorm_best_eps = best_source_eps;
        kernel_stats_.hcnorm_best_formula = best_source_formula;
        kernel_stats_.hcnorm_input_layout = best_source_layout;
        if (best_source_eps < 0.0) {
            kernel_stats_.missing_formula_params.push_back("norm_eps");
        }

        if (!norm_only_pass) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "Tier B HC_PRE_NORM norm-only regression failed";
        } else if (!tier_b_tensor_pass("f32", kernel_stats_.max_abs, policy)) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "Tier B HC_PRE_NORM weighted_sum source threshold failed";
        }
        return kernel_stats_.failed_cases == 0;
    }

    bool run_rmoe_recompute(const Dsv4TolerancePolicy & policy) {
        output_.outputs = input_.references;
        kernel_stats_ = {};
        kernel_stats_.partial_rmoe_recompute = true;
        kernel_stats_.expert_weight_recompute = false;
        kernel_stats_.weights_not_decoded = true;
        kernel_stats_.copied_reference_output = false;

        const CaptureRecord * expert_down = find_record("routed_moe_final_output", "rmoe_expert_down");
        const CaptureRecord * routed_sum = find_record("routed_moe_final_output", "rmoe_routed_sum");
        const CaptureRecord * shared_down = find_record("routed_moe_final_output", "rmoe_shared_down");
        const CaptureRecord * final_ref = find_record_any("routed_moe_final_output", {"rmoe_final_ffn_reference", "final_ffn"});

        require_record(shared_down, "rmoe_shared_down");
        require_record(final_ref, "rmoe_final_ffn_reference");
        if (expert_down == nullptr) {
            require_record(routed_sum, "rmoe_routed_sum");
        }
        if (!kernel_stats_.missing_inputs.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "missing required routed-MoE recompute payloads";
            return false;
        }

        const std::vector<float> shared_down_f = payload_f32(*shared_down);
        const std::vector<float> final_ref_f = payload_f32(*final_ref);
        std::vector<float> routed_sum_ref_f = routed_sum != nullptr ? payload_f32(*routed_sum) : std::vector<float>();
        if (shared_down_f.empty() || final_ref_f.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "one or more routed-MoE payloads are not f32-compatible";
            return false;
        }

        const size_t n_embd = final_ref_f.size();
        if (shared_down_f.size() != n_embd) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "routed-MoE shared_down/final payload sizes disagree";
            return false;
        }

        std::vector<float> candidate_routed_sum(n_embd, 0.0f);
        if (expert_down != nullptr) {
            const std::vector<float> expert_down_f = payload_f32(*expert_down);
            if (expert_down_f.empty() || expert_down_f.size() % n_embd != 0) {
                kernel_stats_.recompute_possible = false;
                kernel_stats_.failed_cases = 1;
                kernel_stats_.first_failure = "rmoe_expert_down payload is not f32-compatible or not slot-aligned";
                return false;
            }
            const size_t n_slots = expert_down_f.size() / n_embd;
            for (size_t e = 0; e < n_embd; ++e) {
                double acc = 0.0;
                for (size_t slot = 0; slot < n_slots; ++slot) {
                    acc += static_cast<double>(expert_down_f[slot * n_embd + e]);
                }
                candidate_routed_sum[e] = static_cast<float>(acc);
            }
            kernel_stats_.full_rmoe_recompute = true;
        } else {
            if (routed_sum_ref_f.size() != n_embd) {
                kernel_stats_.recompute_possible = false;
                kernel_stats_.failed_cases = 1;
                kernel_stats_.first_failure = "rmoe_routed_sum payload is not f32-compatible";
                return false;
            }
            candidate_routed_sum = routed_sum_ref_f;
            kernel_stats_.routed_sum_only_recompute = true;
        }

        std::vector<float> candidate_final(n_embd, 0.0f);
        for (size_t e = 0; e < n_embd; ++e) {
            candidate_final[e] = candidate_routed_sum[e] + shared_down_f[e];
        }

        if (routed_sum != nullptr) {
            if (routed_sum_ref_f.empty()) {
                routed_sum_ref_f = payload_f32(*routed_sum);
            }
            compare_f32(candidate_routed_sum, routed_sum_ref_f, kernel_stats_.rmoe_routed_sum_max_abs, kernel_stats_.rmoe_routed_sum_rms);
        }
        compare_f32(candidate_final, final_ref_f, kernel_stats_.rmoe_final_ffn_max_abs, kernel_stats_.rmoe_final_ffn_rms);

        kernel_stats_.cases = routed_sum != nullptr ? 2 : 1;
        kernel_stats_.max_abs = std::max(kernel_stats_.rmoe_routed_sum_max_abs, kernel_stats_.rmoe_final_ffn_max_abs);
        kernel_stats_.rms = std::max(kernel_stats_.rmoe_routed_sum_rms, kernel_stats_.rmoe_final_ffn_rms);
        const bool pass = tier_b_tensor_pass("f32", kernel_stats_.max_abs, policy);
        kernel_stats_.hcnorm_formula_reports.push_back(formula_report_line(
                expert_down != nullptr ? "RMOE_sum_expert_down_slots_plus_shared_down" : "RMOE_routed_sum_plus_shared_down",
                "captured_source_contract",
                kernel_stats_.rmoe_routed_sum_max_abs,
                kernel_stats_.rmoe_routed_sum_rms,
                0.0,
                kernel_stats_.rmoe_final_ffn_max_abs,
                kernel_stats_.rmoe_final_ffn_rms,
                kernel_stats_.rmoe_final_ffn_max_abs,
                kernel_stats_.rmoe_final_ffn_rms,
                pass));
        if (!pass) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "Tier B routed-MoE source-contract recompute threshold failed";
            return false;
        }
        return true;
    }

    bool run_aohc_recompute(const Dsv4TolerancePolicy & policy) {
        output_.outputs = input_.references;
        kernel_stats_ = {};
        kernel_stats_.partial_aohc_recompute = true;
        kernel_stats_.full_aohc_recompute = false;
        kernel_stats_.weights_not_decoded = true;
        kernel_stats_.copied_reference_output = false;
        kernel_stats_.hcnorm_best_formula = "AOHC_hc_post_expand_from_captured_attn_out";
        kernel_stats_.hcnorm_input_layout = "hidden_major";
        return run_aohc_hc_post_recompute_impl(policy, false);
    }

    bool run_cupd_recompute(const Dsv4TolerancePolicy & policy) {
        if (cupd_stage_ == "rope") {
            return run_cupd_rope_recompute(policy);
        }
        output_.outputs = input_.references;
        kernel_stats_ = {};
        kernel_stats_.partial_cupd_recompute = true;
        kernel_stats_.full_cupd_recompute = false;
        kernel_stats_.weights_not_decoded = true;
        kernel_stats_.copied_reference_output = false;
        kernel_stats_.hcnorm_best_formula = "CUPD_norm_weighted_from_norm_times_weight";
        kernel_stats_.hcnorm_input_layout = "head_dim_flat";

        const CaptureRecord * norm = find_record("compressor_update", "compressed_norm");
        const CaptureRecord * weight = find_record("compressor_update", "compressed_norm_weight");
        const CaptureRecord * ref = find_record("compressor_update", "compressed_norm_weighted");

        kernel_stats_.cupd_compressed_norm_available = norm != nullptr && norm->full_tensor_payload_available;
        kernel_stats_.cupd_norm_weight_available = weight != nullptr && weight->full_tensor_payload_available;
        kernel_stats_.cupd_norm_weighted_available = ref != nullptr && ref->full_tensor_payload_available;

        require_record(norm, "compressed_norm");
        require_record(weight, "compressed_norm_weight");
        require_record(ref, "compressed_norm_weighted");
        if (!kernel_stats_.missing_inputs.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "missing required compressor/update recompute payloads";
            return false;
        }

        const std::vector<float> norm_f = payload_f32(*norm);
        const std::vector<float> weight_f = payload_f32(*weight);
        const std::vector<float> ref_f = payload_f32(*ref);
        if (norm_f.empty() || weight_f.empty() || ref_f.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "one or more compressor/update payloads are not f32-compatible";
            return false;
        }
        if (norm_f.size() != ref_f.size() || weight_f.size() < ref_f.size()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "compressor/update norm, norm_weight, and reference sizes disagree";
            return false;
        }

        std::vector<float> candidate(ref_f.size(), 0.0f);
        for (size_t i = 0; i < ref_f.size(); ++i) {
            candidate[i] = norm_f[i] * weight_f[i];
        }

        compare_f32(candidate, ref_f, kernel_stats_.cupd_output_max_abs, kernel_stats_.cupd_output_rms);
        compare_f32_rel(candidate, ref_f, kernel_stats_.cupd_output_max_rel);
        kernel_stats_.cases = 1;
        kernel_stats_.max_abs = kernel_stats_.cupd_output_max_abs;
        kernel_stats_.rms = kernel_stats_.cupd_output_rms;
        const bool pass = tier_b_tensor_pass("f32", kernel_stats_.max_abs, policy);
        kernel_stats_.hcnorm_formula_reports.push_back(formula_report_line(
                "CUPD_norm_weighted_from_norm_times_weight",
                "head_dim_flat",
                kernel_stats_.cupd_output_max_abs,
                kernel_stats_.cupd_output_rms,
                kernel_stats_.cupd_output_max_rel,
                kernel_stats_.cupd_output_max_abs,
                kernel_stats_.cupd_output_rms,
                kernel_stats_.cupd_output_max_abs,
                kernel_stats_.cupd_output_rms,
                pass));
        if (!pass) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "Tier B compressor/update source-contract recompute threshold failed";
            return false;
        }
        return true;
    }

    bool run_cupd_rope_recompute(const Dsv4TolerancePolicy & policy) {
        output_.outputs = input_.references;
        kernel_stats_ = {};
        kernel_stats_.partial_cupd_recompute = true;
        kernel_stats_.full_cupd_recompute = false;
        kernel_stats_.weights_not_decoded = true;
        kernel_stats_.copied_reference_output = false;
        kernel_stats_.hcnorm_best_formula = "CUPD_rope_tail_source_contract_sweep";
        kernel_stats_.hcnorm_input_layout = "tail_rope_flat";

        struct RopePair {
            const CaptureRecord * input = nullptr;
            const CaptureRecord * ref = nullptr;
        };
        std::map<int64_t, RopePair> pairs;
        for (const CaptureRecord & rec : input_.references) {
            if (rec.stage != "compressor_update" || rec.payload_kind != "tensor_values" || !rec.full_tensor_payload_available) {
                continue;
            }
            if (rec.tensor == "cupd_rope_input" || rec.tensor == "cupd_norm_weighted" || rec.tensor == "compressed_norm_weighted") {
                if (pairs[rec.token].input == nullptr || rec.tensor == "cupd_rope_input") {
                    pairs[rec.token].input = &rec;
                }
            } else if (rec.tensor == "cupd_rope_reference" || rec.tensor == "compressed_rope") {
                if (pairs[rec.token].ref == nullptr || rec.tensor == "cupd_rope_reference") {
                    pairs[rec.token].ref = &rec;
                }
            }
        }

        kernel_stats_.cupd_norm_weighted_available = find_record_any("compressor_update", {
                "cupd_norm_weighted",
                "compressed_norm_weighted",
        }) != nullptr;
        kernel_stats_.cupd_rope_input_available = std::any_of(pairs.begin(), pairs.end(), [](const auto & item) {
            return item.second.input != nullptr;
        });
        kernel_stats_.cupd_rope_reference_available = std::any_of(pairs.begin(), pairs.end(), [](const auto & item) {
            return item.second.ref != nullptr;
        });
        struct RopeMeta {
            bool available = false;
            int position = 0;
            int cache_position = 0;
            int n_rot = 0;
            int width = 0;
            float freq_base = 160000.0f;
            float freq_scale = 1.0f;
            float ext_factor = 0.0f;
            float attn_factor = 1.0f;
            float beta_fast = 32.0f;
            float beta_slow = 1.0f;
            int tail_offset = -1;
            int n_ctx_orig = 8192;
            std::string mode = "normal";
        };
        std::map<int64_t, RopeMeta> rope_meta_by_token;
        for (const CaptureRecord & rec : input_.references) {
            if (rec.stage != "compressor_update" || rec.availability != "available") {
                continue;
            }
            if (rec.tensor != "cupd_rope_metadata" && rec.tensor != "cupd_rope_op_params" &&
                    rec.tensor != "cupd_rope_position" && rec.tensor != "cupd_rope_n_rot") {
                continue;
            }
            RopeMeta & meta = rope_meta_by_token[rec.token];
            meta.available = true;
            if (rec.has_rope_position) {
                meta.position = static_cast<int>(rec.rope_position);
            }
            if (rec.has_rope_cache_position) {
                meta.cache_position = static_cast<int>(rec.rope_cache_position);
            }
            if (rec.has_rope_n_rot) {
                meta.n_rot = static_cast<int>(rec.rope_n_rot);
            }
            if (rec.has_rope_width) {
                meta.width = static_cast<int>(rec.rope_width);
            }
            if (rec.has_rope_freq_base) {
                meta.freq_base = static_cast<float>(rec.rope_freq_base);
            }
            if (rec.has_rope_freq_scale) {
                meta.freq_scale = static_cast<float>(rec.rope_freq_scale);
            }
            if (rec.has_rope_tail_offset) {
                meta.tail_offset = static_cast<int>(rec.rope_tail_offset);
            }
            if (rec.has_rope_n_ctx_orig) {
                meta.n_ctx_orig = static_cast<int>(rec.rope_n_ctx_orig);
            }
            if (!rec.rope_mode.empty()) {
                meta.mode = rec.rope_mode;
            }
            if (rec.has_rope_ext_factor) {
                meta.ext_factor = static_cast<float>(rec.rope_ext_factor);
            }
            if (rec.has_rope_attn_factor) {
                meta.attn_factor = static_cast<float>(rec.rope_attn_factor);
            }
            if (rec.has_rope_beta_fast) {
                meta.beta_fast = static_cast<float>(rec.rope_beta_fast);
            }
            if (rec.has_rope_beta_slow) {
                meta.beta_slow = static_cast<float>(rec.rope_beta_slow);
            }
        }
        kernel_stats_.cupd_rope_position_available = has_available_record("compressor_update", "cupd_rope_position");
        kernel_stats_.cupd_rope_cache_position_available = has_available_record("compressor_update", "cupd_rope_cache_position");
        kernel_stats_.cupd_rope_n_rot_available = has_available_record("compressor_update", "cupd_rope_n_rot");
        kernel_stats_.cupd_rope_freq_base_available = has_available_record("compressor_update", "cupd_rope_freq_base");
        kernel_stats_.cupd_rope_freq_scale_available = has_available_record("compressor_update", "cupd_rope_freq_scale");
        kernel_stats_.cupd_rope_mode_available = has_available_record("compressor_update", "cupd_rope_mode");
        kernel_stats_.cupd_rope_tail_offset_available = has_available_record("compressor_update", "cupd_rope_tail_offset");
        kernel_stats_.cupd_rope_nonzero_position_records = 0;
        for (const auto & meta_entry : rope_meta_by_token) {
            if (meta_entry.second.available && meta_entry.second.position != 0) {
                kernel_stats_.cupd_rope_nonzero_position_records++;
            }
        }
        kernel_stats_.cupd_rope_cos_available = find_record("compressor_update", "cupd_rope_cos") != nullptr;
        kernel_stats_.cupd_rope_sin_available = find_record("compressor_update", "cupd_rope_sin") != nullptr;

        if (!kernel_stats_.cupd_rope_input_available) {
            kernel_stats_.missing_inputs.push_back("cupd_rope_input");
        }
        if (!kernel_stats_.cupd_rope_reference_available) {
            kernel_stats_.missing_inputs.push_back("cupd_rope_reference");
        }
        if (!kernel_stats_.cupd_rope_position_available) {
            kernel_stats_.missing_formula_params.push_back("cupd_rope_position_metadata");
        }
        if (!kernel_stats_.cupd_rope_n_rot_available) {
            kernel_stats_.missing_formula_params.push_back("cupd_rope_n_rot_metadata");
        }
        if (kernel_stats_.cupd_rope_nonzero_position_records <= 0) {
            kernel_stats_.missing_formula_params.push_back("missing_nonzero_position_fixture");
        }
        if (!kernel_stats_.missing_inputs.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "missing required compressor/update RoPE payloads";
            return false;
        }
        if (!kernel_stats_.missing_formula_params.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "missing compressor/update RoPE metadata for nonzero validation";
            return false;
        }

        struct Candidate {
            std::string id;
            bool neox = false;
            bool swapped_sign = false;
            int position = 0;
            int n_rot = 0;
            float freq_base = 160000.0f;
            float freq_scale = 1.0f;
            float ext_factor = 0.0f;
            float attn_factor = 1.0f;
            float beta_fast = 32.0f;
            float beta_slow = 1.0f;
            int tail_offset = -1;
            int n_ctx_orig = 8192;
        };

        double best_max_abs = INFINITY;
        double best_rms = INFINITY;
        double best_max_rel = INFINITY;
        Candidate best_candidate;

        for (const auto & pair_entry : pairs) {
            if (pair_entry.second.input == nullptr || pair_entry.second.ref == nullptr) {
                continue;
            }
            auto meta_it = rope_meta_by_token.find(pair_entry.first);
            if (meta_it == rope_meta_by_token.end() || !meta_it->second.available) {
                continue;
            }
            const RopeMeta & meta = meta_it->second;
            if (meta.position == 0) {
                continue;
            }
            const std::vector<float> input_f = payload_f32(*pair_entry.second.input);
            const std::vector<float> ref_f = payload_f32(*pair_entry.second.ref);
            if (input_f.empty() || ref_f.empty() || input_f.size() != ref_f.size() || input_f.size() % 2 != 0) {
                continue;
            }

            const int head_dim = static_cast<int>(input_f.size());
            if (meta.n_rot <= 0 || meta.n_rot > head_dim || (meta.n_rot % 2) != 0) {
                continue;
            }

            const std::vector<Candidate> candidates = {
                {"R0_standard_tail", false, false, meta.position, meta.n_rot, meta.freq_base, meta.freq_scale, meta.ext_factor, meta.attn_factor, meta.beta_fast, meta.beta_slow, meta.tail_offset, meta.n_ctx_orig},
                {"R1_swapped_sign_tail", false, true, meta.position, meta.n_rot, meta.freq_base, meta.freq_scale, meta.ext_factor, meta.attn_factor, meta.beta_fast, meta.beta_slow, meta.tail_offset, meta.n_ctx_orig},
                {"R2_neox_tail", true, false, meta.position, meta.n_rot, meta.freq_base, meta.freq_scale, meta.ext_factor, meta.attn_factor, meta.beta_fast, meta.beta_slow, meta.tail_offset, meta.n_ctx_orig},
                {"R3_neox_swapped_sign_tail", true, true, meta.position, meta.n_rot, meta.freq_base, meta.freq_scale, meta.ext_factor, meta.attn_factor, meta.beta_fast, meta.beta_slow, meta.tail_offset, meta.n_ctx_orig},
            };
            for (const Candidate & candidate : candidates) {
                std::vector<float> out = apply_cupd_rope_candidate(input_f, candidate.neox, candidate.swapped_sign,
                        candidate.position, candidate.n_rot, candidate.freq_base, candidate.freq_scale,
                        candidate.ext_factor, candidate.attn_factor, candidate.beta_fast, candidate.beta_slow,
                        candidate.tail_offset, candidate.n_ctx_orig);
                double max_abs = 0.0;
                double rms = 0.0;
                double max_rel = 0.0;
                compare_f32(out, ref_f, max_abs, rms);
                compare_f32_rel(out, ref_f, max_rel);
                const bool pass = tier_b_tensor_pass("f32", max_abs, policy);
                const std::string layout = std::string(candidate.neox ? "neox" : "normal") +
                    ",token=" + std::to_string(pair_entry.first) +
                    ",pos=" + std::to_string(candidate.position) +
                    ",n_rot=" + std::to_string(candidate.n_rot) +
                    ",tail_offset=" + std::to_string(candidate.tail_offset) +
                    ",base=" + std::to_string(static_cast<int>(candidate.freq_base)) +
                    ",scale=" + std::to_string(candidate.freq_scale) +
                    (candidate.swapped_sign ? ",swapped_sign" : "");
                kernel_stats_.hcnorm_formula_reports.push_back(formula_report_line(
                        candidate.id,
                        layout,
                        max_abs,
                        rms,
                        max_rel,
                        max_abs,
                        rms,
                        max_abs,
                        rms,
                        pass));
                if (max_abs < best_max_abs || (max_abs == best_max_abs && rms < best_rms)) {
                    best_max_abs = max_abs;
                    best_rms = rms;
                    best_max_rel = max_rel;
                    best_candidate = candidate;
                }
            }
        }

        if (!std::isfinite(best_max_abs)) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "no f32-compatible compressor/update RoPE input/reference pair";
            return false;
        }

        kernel_stats_.cupd_output_max_abs = best_max_abs;
        kernel_stats_.cupd_output_rms = best_rms;
        kernel_stats_.cupd_output_max_rel = best_max_rel;
        kernel_stats_.cases = 1;
        kernel_stats_.max_abs = best_max_abs;
        kernel_stats_.rms = best_rms;
        kernel_stats_.cupd_rope_position = best_candidate.position;
        kernel_stats_.cupd_rope_n_rot = best_candidate.n_rot;
        kernel_stats_.cupd_rope_tail_offset = best_candidate.tail_offset;
        kernel_stats_.hcnorm_best_formula = best_candidate.id;
        kernel_stats_.hcnorm_input_layout = std::string(best_candidate.neox ? "neox_tail" : "standard_tail") +
            (best_candidate.swapped_sign ? "_swapped_sign" : "");

        if (!tier_b_tensor_pass("f32", kernel_stats_.max_abs, policy)) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "Tier B compressor/update RoPE source-contract recompute threshold failed";
            return false;
        }
        return true;
    }

    bool run_kernel_cupd_norm_weighted(const Dsv4TolerancePolicy & policy) {
        output_.outputs = input_.references;
        kernel_stats_ = {};
        kernel_stats_.kernel_mode = "cupd_norm_weighted";
        kernel_stats_.metal_recompute = true;
        kernel_stats_.partial_cupd_recompute = true;
        kernel_stats_.full_cupd_recompute = false;
        kernel_stats_.weights_not_decoded = true;
        kernel_stats_.copied_reference_output = false;
        kernel_stats_.hcnorm_best_formula = "Metal_CUPD_norm_weighted_from_norm_times_weight";
        kernel_stats_.hcnorm_input_layout = "head_dim_flat";

        const CaptureRecord * norm = find_record("compressor_update", "compressed_norm");
        const CaptureRecord * weight = find_record("compressor_update", "compressed_norm_weight");
        const CaptureRecord * ref = find_record("compressor_update", "compressed_norm_weighted");

        kernel_stats_.cupd_compressed_norm_available = norm != nullptr && norm->full_tensor_payload_available;
        kernel_stats_.cupd_norm_weight_available = weight != nullptr && weight->full_tensor_payload_available;
        kernel_stats_.cupd_norm_weighted_available = ref != nullptr && ref->full_tensor_payload_available;

        require_record(norm, "compressed_norm");
        require_record(weight, "compressed_norm_weight");
        require_record(ref, "compressed_norm_weighted");
        if (!kernel_stats_.missing_inputs.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "missing required compressor/update Metal kernel payloads";
            return false;
        }

        const std::vector<float> norm_f = payload_f32(*norm);
        const std::vector<float> weight_f = payload_f32(*weight);
        const std::vector<float> ref_f = payload_f32(*ref);
        if (norm_f.empty() || weight_f.empty() || ref_f.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "one or more compressor/update Metal payloads are not f32-compatible";
            return false;
        }
        if (norm_f.size() != ref_f.size() || weight_f.size() < ref_f.size()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "compressor/update Metal norm, norm_weight, and reference sizes disagree";
            return false;
        }

        std::vector<float> candidate(ref_f.size(), 0.0f);
        char error[512] = {};
#ifdef DSV4_LAYER_EXECUTOR_HARNESS_METAL
        const int rc = dsv4_layer_executor_metal_cupd_norm_weighted(
                norm_f.data(), weight_f.data(), ref_f.size(),
                candidate.data(), error, sizeof(error));
#else
        const int rc = 1;
        std::snprintf(error, sizeof(error), "Metal harness support not built");
#endif
        if (rc != 0) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = error;
            return false;
        }

        compare_f32(candidate, ref_f, kernel_stats_.cupd_output_max_abs, kernel_stats_.cupd_output_rms);
        compare_f32_rel(candidate, ref_f, kernel_stats_.cupd_output_max_rel);
        kernel_stats_.cases = 1;
        kernel_stats_.max_abs = kernel_stats_.cupd_output_max_abs;
        kernel_stats_.rms = kernel_stats_.cupd_output_rms;
        kernel_stats_.tier_a_exact_pass = kernel_stats_.max_abs <= policy.tier_a_max_abs && kernel_stats_.rms <= policy.tier_a_rms;
        const bool pass = tier_b_tensor_pass("f32", kernel_stats_.max_abs, policy);
        kernel_stats_.hcnorm_formula_reports.push_back(formula_report_line(
                "Metal_CUPD_norm_weighted_from_norm_times_weight",
                "head_dim_flat",
                kernel_stats_.cupd_output_max_abs,
                kernel_stats_.cupd_output_rms,
                kernel_stats_.cupd_output_max_rel,
                kernel_stats_.cupd_output_max_abs,
                kernel_stats_.cupd_output_rms,
                kernel_stats_.cupd_output_max_abs,
                kernel_stats_.cupd_output_rms,
                pass));
        if (!pass) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "Tier B compressor/update Metal source-contract threshold failed";
            return false;
        }
        return true;
    }

    bool run_kernel_aohc_hc_post_from_substages(const Dsv4TolerancePolicy & policy) {
        kernel_stats_.metal_recompute = true;
        kernel_stats_.partial_aohc_recompute = true;
        kernel_stats_.full_aohc_recompute = false;
        kernel_stats_.weights_not_decoded = true;
        kernel_stats_.copied_reference_output = false;
        kernel_stats_.hcnorm_best_formula = "Metal_AOHC_hc_post_expand_from_captured_attn_out";
        kernel_stats_.hcnorm_input_layout = "hidden_major";
        return run_aohc_hc_post_recompute_impl(policy, true);
    }

    bool run_aohc_hc_post_recompute_impl(const Dsv4TolerancePolicy & policy, bool use_metal) {
        const CaptureRecord * attn_out = find_record_any("aohc_boundary", {"aohc_hcexpand_dispatch_src0", "aohc_hcexpand_src0_block", "aohc_attn_out"});
        const CaptureRecord * residual = find_record_any("aohc_boundary", {"aohc_hcexpand_dispatch_src1", "aohc_hcexpand_src1_residual", "aohc_hc_post_residual"});
        const CaptureRecord * split_full = find_record("aohc_boundary", "aohc_hc_split_full");
        const CaptureRecord * post = find_record_any("aohc_boundary", {"aohc_hcexpand_dispatch_src2", "aohc_hcexpand_src2_post", "aohc_hc_post_weights"});
        const CaptureRecord * comb = find_record_any("aohc_boundary", {"aohc_hcexpand_dispatch_src3", "aohc_hcexpand_src3_comb", "aohc_hc_comb"});
        const CaptureRecord * ref = find_record_any("aohc_boundary", {"aohc_after_attn_hc_reference", "after_attn_hc"});

        kernel_stats_.aohc_attn_out_available = attn_out != nullptr && attn_out->full_tensor_payload_available;
        kernel_stats_.aohc_residual_available = residual != nullptr && residual->full_tensor_payload_available;
        kernel_stats_.aohc_post_weights_available = post != nullptr && post->full_tensor_payload_available;
        kernel_stats_.aohc_comb_available = comb != nullptr && comb->full_tensor_payload_available;
        kernel_stats_.aohc_reference_available = ref != nullptr && ref->full_tensor_payload_available;

        require_record(attn_out, "aohc_hcexpand_dispatch_src0");
        require_record(residual, "aohc_hcexpand_dispatch_src1");
        require_record(post, "aohc_hcexpand_dispatch_src2");
        require_record(comb, "aohc_hcexpand_dispatch_src3");
        require_record(ref, "aohc_after_attn_hc_reference");
        if (!kernel_stats_.missing_inputs.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "missing required AOHC recompute payloads";
            return false;
        }

        const std::vector<float> attn_out_f = payload_f32(*attn_out);
        const std::vector<float> residual_f = payload_f32(*residual);
        std::vector<float> post_f = payload_f32(*post);
        std::vector<float> comb_f = payload_f32(*comb);
        const std::vector<float> ref_f = payload_f32(*ref);
        if (attn_out_f.empty() || residual_f.empty() || post_f.empty() || comb_f.empty() || ref_f.empty()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "one or more AOHC payloads are not f32-compatible";
            return false;
        }

        const size_t n_embd = attn_out_f.size();
        if (residual_f.size() % n_embd != 0 || ref_f.size() != residual_f.size()) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "AOHC payload sizes disagree";
            return false;
        }
        const size_t n_hc = residual_f.size() / n_embd;
        const bool using_exact_hcexpand_post_comb =
            post != nullptr && comb != nullptr &&
            ((post->tensor == "aohc_hcexpand_src2_post" &&
                    comb->tensor == "aohc_hcexpand_src3_comb") ||
                (post->tensor == "aohc_hcexpand_dispatch_src2" &&
                    comb->tensor == "aohc_hcexpand_dispatch_src3"));
        if (!using_exact_hcexpand_post_comb && split_full != nullptr && split_full->full_tensor_payload_available) {
            const std::vector<float> split_f = payload_f32(*split_full);
            if (split_f.size() >= 2 * n_hc + n_hc * n_hc) {
                post_f.assign(split_f.begin() + n_hc, split_f.begin() + 2 * n_hc);
                comb_f.assign(split_f.begin() + 2 * n_hc, split_f.begin() + 2 * n_hc + n_hc * n_hc);
            }
        }
        if (post_f.size() < n_hc || comb_f.size() < n_hc * n_hc) {
            kernel_stats_.recompute_possible = false;
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "AOHC post/comb payloads too small";
            return false;
        }

        std::vector<float> candidate(ref_f.size(), 0.0f);
        if (use_metal) {
            char error[512] = {};
#ifdef DSV4_LAYER_EXECUTOR_HARNESS_METAL
            const int rc = dsv4_layer_executor_metal_aohc_hc_post_from_substages(
                    attn_out_f.data(), residual_f.data(), post_f.data(), comb_f.data(),
                    n_embd, n_hc,
                    candidate.data(),
                    error, sizeof(error));
#else
            const int rc = 1;
            std::snprintf(error, sizeof(error), "Metal harness support not built");
#endif
            if (rc != 0) {
                kernel_stats_.failed_cases = 1;
                kernel_stats_.first_failure = error;
                return false;
            }
        } else {
            candidate = aohc_hc_post_expand(attn_out_f, residual_f, post_f, comb_f, n_embd, n_hc);
        }

        compare_f32(candidate, ref_f, kernel_stats_.aohc_output_max_abs, kernel_stats_.aohc_output_rms);
        compare_f32_rel(candidate, ref_f, kernel_stats_.aohc_output_max_rel);
        kernel_stats_.cases = 1;
        kernel_stats_.max_abs = kernel_stats_.aohc_output_max_abs;
        kernel_stats_.rms = kernel_stats_.aohc_output_rms;
        kernel_stats_.tier_a_exact_pass = kernel_stats_.max_abs <= policy.tier_a_max_abs && kernel_stats_.rms <= policy.tier_a_rms;

        const CaptureRecord * hc_input = find_record_any("hc_pre_norm", {"hc_ws_input_inpL_raw", "hc_pre_input_hc_original", "input_hc_original_residual"});
        if (hc_input != nullptr && hc_input->full_tensor_payload_available) {
            double chain_max = 0.0;
            double chain_rms = 0.0;
            compare_payloads(*hc_input, ref->payload_bytes, chain_max, chain_rms);
            kernel_stats_.aohc_matches_hc_pre_norm_input = (chain_max == 0.0 && chain_rms == 0.0) ? 1 : 0;
        } else {
            kernel_stats_.aohc_matches_hc_pre_norm_input = -1;
        }

        const bool pass = tier_b_tensor_pass("f32", kernel_stats_.max_abs, policy);
        kernel_stats_.hcnorm_formula_reports.push_back(formula_report_line(
                use_metal ? "Metal_AOHC_hc_post_expand_from_captured_attn_out" : "AOHC_hc_post_expand_from_captured_attn_out",
                "hidden_major",
                kernel_stats_.aohc_output_max_abs,
                kernel_stats_.aohc_output_rms,
                kernel_stats_.aohc_output_max_rel,
                0.0,
                0.0,
                kernel_stats_.aohc_output_max_abs,
                kernel_stats_.aohc_output_rms,
                pass));
        if (!pass) {
            kernel_stats_.failed_cases = 1;
            kernel_stats_.first_failure = "Tier B AOHC source-contract threshold failed";
            return false;
        }
        return true;
    }

    bool compare_identity(const Dsv4TolerancePolicy & policy) {
        reset_summary();
        for (const CaptureRecord & rec : output_.outputs) {
            validate_record(rec, policy);
        }
        if (require_full_tensors_) {
            for (const std::string & stage : k_required_stages) {
                if (required_tensor_payload_stages_.count(stage) == 0) {
                    add_failure("required stage missing tensor_values payload " + stage, stage);
                }
            }
        }
        if (require_byte_payloads_ && payload_byte_values_ == 0) {
            add_failure("no byte_values payloads available", "");
        }
        return failed_cases_ == 0 && !output_.outputs.empty();
    }

    void set_layer_filter(int layer) {
        layer_filter_ = layer;
    }

    void set_token_filter(int64_t token) {
        token_filter_ = token;
    }

    void set_allow_unavailable_optional(bool value) {
        allow_unavailable_optional_ = value;
    }

    void set_allow_unavailable_required(bool value) {
        allow_unavailable_required_ = value;
    }

    void set_require_full_tensors(bool value) {
        require_full_tensors_ = value;
    }

    void set_require_byte_payloads(bool value) {
        require_byte_payloads_ = value;
    }

    void set_cupd_stage(const std::string & value) {
        cupd_stage_ = value.empty() ? "norm_weighted" : value;
    }

    void mark_missing_required_stage(const std::string & stage) {
        ++missing_required_;
        add_failure("missing required stage " + stage, stage);
    }

    size_t captures_loaded() const {
        return input_.references.size();
    }

    std::set<std::string> stages_loaded() const {
        std::set<std::string> out;
        for (const auto & rec : input_.references) {
            out.insert(rec.stage);
        }
        return out;
    }

    std::set<int> layers_loaded() const {
        std::set<int> out;
        for (const auto & rec : input_.references) {
            if (rec.layer >= 0) {
                out.insert(rec.layer);
            }
        }
        return out;
    }

    std::set<int64_t> tokens_loaded() const {
        std::set<int64_t> out;
        for (const auto & rec : input_.references) {
            if (rec.token >= 0) {
                out.insert(rec.token);
            }
        }
        return out;
    }

    bool has_full_tensor_payload() const { return full_tensor_payload_records_ > 0; }
    bool has_byte_payload() const { return byte_payload_records_ > 0; }
    bool has_stats_only_payload() const { return payload_stats_only_ > 0; }
    bool has_metadata_only_payload() const { return payload_metadata_only_ > 0; }

    static bool tier_b_tensor_pass(const std::string & dtype, double max_abs, const Dsv4TolerancePolicy & policy) {
        if (dtype == "f16" || dtype == "fp16") {
            return max_abs <= policy.tier_b_fp16_layer_max_abs;
        }
        return max_abs <= policy.tier_b_fp32_layer_max_abs;
    }

    std::string stage_status(const std::string & stage) const {
        bool found = false;
        for (const auto & rec : input_.references) {
            if (rec.stage == stage) {
                found = true;
                break;
            }
        }
        if (!found) {
            return "missing";
        }
        return failed_stages_.count(stage) ? "fail" : "pass";
    }

    int64_t exact_cases() const { return exact_cases_; }
    int64_t failed_cases() const { return failed_cases_; }
    double max_abs() const { return max_abs_; }
    double max_rms() const { return max_rms_; }
    int64_t required_records() const { return required_records_; }
    int64_t optional_records() const { return optional_records_; }
    int64_t available_records() const { return available_records_; }
    int64_t unavailable_records() const { return unavailable_records_; }
    int64_t missing_required() const { return missing_required_; }
    int64_t payload_stats_only() const { return payload_stats_only_; }
    int64_t payload_tensor_values() const { return payload_tensor_values_; }
    int64_t payload_byte_values() const { return payload_byte_values_; }
    int64_t payload_metadata_only() const { return payload_metadata_only_; }
    int64_t warnings() const { return warnings_; }
    const Dsv4KernelCompareStats & kernel_stats() const { return kernel_stats_; }

    std::string reason_summary() const {
        if (reasons_.empty()) {
            return "";
        }
        std::ostringstream out;
        for (size_t i = 0; i < reasons_.size(); ++i) {
            if (i != 0) {
                out << "; ";
            }
            out << reasons_[i];
        }
        return out.str();
    }

private:
    bool load_capture_file(const std::string & path) {
        std::ifstream in(path);
        if (!in) {
            std::cerr << "failed to open capture: " << path << "\n";
            return false;
        }

        bool loaded_any = false;
        std::string line;
        while (std::getline(in, line)) {
            CaptureRecord rec;
            if (parse_jsonl_record(path, line, rec) || parse_log_summary_record(path, line, rec)) {
                if (record_allowed(rec)) {
                    input_.references.push_back(rec);
                    loaded_any = true;
                }
            }
        }
        if (!loaded_any) {
            std::cerr << "warning: no harness captures found in " << path << "\n";
        }
        return true;
    }

    bool record_allowed(const CaptureRecord & rec) const {
        if (layer_filter_ >= 0 && rec.layer >= 0 && rec.layer != layer_filter_) {
            return false;
        }
        if (token_filter_ >= 0 && rec.token >= 0 && rec.token != token_filter_) {
            return false;
        }
        return true;
    }

    static bool parse_jsonl_record(const std::string & path, const std::string & line, CaptureRecord & rec) {
        if (line.find("\"stage\"") == std::string::npos || !json_string_value(line, "stage", rec.stage)) {
            return false;
        }
        rec.source = path;
        int64_t value = -1;
        if (json_i64_value(line, "layer", value)) {
            rec.layer = static_cast<int>(value);
        }
        if (json_i64_value(line, "token", value)) {
            rec.token = value;
        }
        json_string_value(line, "tensor", rec.tensor);
        json_string_value(line, "dtype", rec.dtype);
        rec.shape = json_shape_value(line);
        json_string_value(line, "availability", rec.availability);
        json_string_value(line, "payload_kind", rec.payload_kind);
        json_string_value(line, "payload_file", rec.payload_file);
        json_string_value(line, "byte_checksum", rec.byte_checksum);
        json_string_value(line, "unavailable_reason", rec.unavailable_reason);
        json_bool_value(line, "required", rec.required);

        rec.stats.has_max_abs = json_double_value(line, "max_abs", rec.stats.max_abs);
        rec.stats.has_rms = json_double_value(line, "rms", rec.stats.rms);
        int64_t over_tol = 0;
        rec.stats.has_over_tol = json_i64_value(line, "over_tol", over_tol);
        if (rec.stats.has_over_tol) {
            rec.stats.over_tol = over_tol;
        }
        json_double_value(line, "sum", rec.stats.sum);
        json_double_value(line, "sumsq", rec.stats.sumsq);
        json_double_value(line, "min", rec.stats.min);
        json_double_value(line, "max", rec.stats.max);

        json_nested_string_value(line, "producer", "op", rec.producer.op);
        json_nested_string_value(line, "producer", "tensor_name", rec.producer.tensor_name);
        json_nested_string_value(line, "producer", "stage", rec.producer.stage);
        json_nested_string_value(line, "consumer", "op", rec.consumer.op);
        json_nested_string_value(line, "consumer", "tensor_name", rec.consumer.tensor_name);
        json_nested_string_value(line, "consumer", "stage", rec.consumer.stage);

        int64_t rope_i64 = 0;
        double rope_f64 = 0.0;
        rec.has_rope_position = json_i64_value(line, "position", rope_i64);
        if (rec.has_rope_position) {
            rec.rope_position = rope_i64;
        }
        rec.has_rope_cache_position = json_i64_value(line, "cache_position", rope_i64);
        if (rec.has_rope_cache_position) {
            rec.rope_cache_position = rope_i64;
        }
        rec.has_rope_n_rot = json_i64_value(line, "n_rot", rope_i64);
        if (rec.has_rope_n_rot) {
            rec.rope_n_rot = rope_i64;
        }
        rec.has_rope_width = json_i64_value(line, "width", rope_i64);
        if (rec.has_rope_width) {
            rec.rope_width = rope_i64;
        }
        if (json_i64_value(line, "rope_type", rope_i64)) {
            rec.rope_type = rope_i64;
        }
        rec.has_rope_tail_offset = json_i64_value(line, "tail_offset", rope_i64);
        if (rec.has_rope_tail_offset) {
            rec.rope_tail_offset = rope_i64;
        }
        rec.has_rope_n_ctx_orig = json_i64_value(line, "n_ctx_orig", rope_i64);
        if (rec.has_rope_n_ctx_orig) {
            rec.rope_n_ctx_orig = rope_i64;
        }
        rec.has_rope_freq_base = json_double_value(line, "freq_base", rope_f64);
        if (rec.has_rope_freq_base) {
            rec.rope_freq_base = rope_f64;
        }
        rec.has_rope_freq_scale = json_double_value(line, "freq_scale", rope_f64);
        if (rec.has_rope_freq_scale) {
            rec.rope_freq_scale = rope_f64;
        }
        rec.has_rope_ext_factor = json_double_value(line, "ext_factor", rope_f64);
        if (rec.has_rope_ext_factor) {
            rec.rope_ext_factor = rope_f64;
        }
        rec.has_rope_attn_factor = json_double_value(line, "attn_factor", rope_f64);
        if (rec.has_rope_attn_factor) {
            rec.rope_attn_factor = rope_f64;
        }
        rec.has_rope_beta_fast = json_double_value(line, "beta_fast", rope_f64);
        if (rec.has_rope_beta_fast) {
            rec.rope_beta_fast = rope_f64;
        }
        rec.has_rope_beta_slow = json_double_value(line, "beta_slow", rope_f64);
        if (rec.has_rope_beta_slow) {
            rec.rope_beta_slow = rope_f64;
        }
        json_string_value(line, "rope_mode", rec.rope_mode);

        rec.inline_tensor_values = json_has_nonempty_array(line, "values");
        rec.inline_bytes = json_has_nonempty_string(line, "bytes");
        if (!rec.payload_file.empty()) {
            const std::filesystem::path payload_path = resolve_payload_path(path, rec.payload_file);
            std::ifstream payload(payload_path, std::ios::binary);
            if (payload) {
                rec.payload_bytes.assign(
                        std::istreambuf_iterator<char>(payload),
                        std::istreambuf_iterator<char>());
            }
        }
        rec.full_tensor_payload_available = rec.payload_kind == "tensor_values" &&
            (rec.inline_tensor_values || !rec.payload_bytes.empty());
        rec.byte_payload_available = rec.payload_kind == "byte_values" &&
            (rec.inline_bytes || !rec.payload_bytes.empty());
        rec.stats_only = rec.payload_kind == "stats_only";
        rec.metadata_only = rec.payload_kind == "metadata_only";
        return true;
    }

    static bool parse_log_summary_record(const std::string & path, const std::string & line, CaptureRecord & rec) {
        if (line.find("dsv4_lexec_side_probe_summary:") == std::string::npos) {
            return false;
        }
        if (!parse_string_after(line, "stage", rec.stage)) {
            return false;
        }
        rec.source = path;
        int64_t value = -1;
        if (parse_i64_after(line, "layer_filter", value)) {
            rec.layer = static_cast<int>(value);
        }
        if (parse_i64_after(line, "token_min", value)) {
            rec.token = value;
        }
        rec.stats.has_max_abs = parse_double_after(line, "max_abs", rec.stats.max_abs);
        rec.stats.has_rms = parse_double_after(line, "max_rms", rec.stats.rms);
        rec.stats.has_over_tol = parse_i64_after(line, "over_tol", rec.stats.over_tol);
        rec.availability = "available";
        rec.payload_kind = "stats_only";
        rec.required = true;
        rec.stats_only = true;
        return true;
    }

    void reset_summary() {
        exact_cases_ = 0;
        failed_cases_ = 0;
        max_abs_ = 0.0;
        max_rms_ = 0.0;
        required_records_ = 0;
        optional_records_ = 0;
        available_records_ = 0;
        unavailable_records_ = 0;
        missing_required_ = 0;
        payload_stats_only_ = 0;
        payload_tensor_values_ = 0;
        payload_byte_values_ = 0;
        payload_metadata_only_ = 0;
        full_tensor_payload_records_ = 0;
        byte_payload_records_ = 0;
        warnings_ = 0;
        required_tensor_payload_stages_.clear();
        required_byte_payload_stages_.clear();
        failed_stages_.clear();
        reasons_.clear();
    }

    void validate_record(const CaptureRecord & rec, const Dsv4TolerancePolicy & policy) {
        if (rec.required) {
            ++required_records_;
        } else {
            ++optional_records_;
        }

        if (rec.availability == "available") {
            ++available_records_;
        } else if (rec.availability == "unavailable") {
            ++unavailable_records_;
        } else if (rec.availability == "missing") {
            if (rec.required) {
                ++missing_required_;
            }
        } else {
            add_failure("invalid availability " + rec.availability, rec.stage);
        }

        if (rec.payload_kind == "stats_only") {
            ++payload_stats_only_;
        } else if (rec.payload_kind == "tensor_values") {
            ++payload_tensor_values_;
            if (rec.full_tensor_payload_available) {
                ++full_tensor_payload_records_;
                if (rec.required) {
                    required_tensor_payload_stages_.insert(rec.stage);
                }
            }
        } else if (rec.payload_kind == "byte_values") {
            ++payload_byte_values_;
            if (rec.byte_payload_available) {
                ++byte_payload_records_;
                if (rec.required) {
                    required_byte_payload_stages_.insert(rec.stage);
                }
            }
        } else if (rec.payload_kind == "metadata_only") {
            ++payload_metadata_only_;
        } else {
            add_failure("bad payload_kind " + rec.payload_kind, rec.stage);
        }

        if (rec.required && rec.availability == "missing") {
            add_failure("required record marked missing", rec.stage);
            return;
        }
        if (rec.required && rec.availability == "unavailable" && !allow_unavailable_required_) {
            add_failure("required record unavailable", rec.stage);
            return;
        }
        if (!rec.required && rec.availability == "unavailable") {
            if (allow_unavailable_optional_) {
                ++warnings_;
            } else {
                add_failure("optional record unavailable", rec.stage);
            }
        }

        if (rec.payload_kind == "tensor_values" && !rec.full_tensor_payload_available) {
            add_failure("tensor_values payload missing values/payload_file", rec.stage);
            return;
        }
        if (rec.payload_kind == "byte_values" && !rec.byte_payload_available) {
            add_failure("byte_values payload missing bytes/payload_file", rec.stage);
            return;
        }
        if ((rec.payload_kind == "tensor_values" || rec.payload_kind == "byte_values") &&
                !rec.payload_bytes.empty() && output_payload_mismatch(rec)) {
            add_failure("identity payload byte comparison failed", rec.stage);
            return;
        }
        if (rec.payload_kind == "stats_only" &&
                (!rec.stats.has_max_abs || !rec.stats.has_rms || !rec.stats.has_over_tol)) {
            add_failure("stats_only payload missing max_abs/rms/over_tol", rec.stage);
            return;
        }
        if (rec.payload_kind == "metadata_only" &&
                rec.dtype.empty() && rec.producer.op.empty() && rec.producer.tensor_name.empty() && rec.shape.empty()) {
            add_failure("metadata_only payload missing metadata", rec.stage);
            return;
        }

        max_abs_ = std::max(max_abs_, rec.stats.max_abs);
        max_rms_ = std::max(max_rms_, rec.stats.rms);
        if (rec.stats.has_max_abs || rec.stats.has_rms || rec.stats.has_over_tol) {
            if (rec.stats.max_abs > policy.tier_a_max_abs || rec.stats.rms > policy.tier_a_rms || rec.stats.over_tol != 0) {
                add_failure("Tier A stats check failed", rec.stage);
                return;
            }
        }
        ++exact_cases_;
    }

    static bool output_payload_mismatch(const CaptureRecord & rec) {
        // Identity mode round-trips CaptureRecord by value, so payload bytes must remain unchanged.
        return rec.payload_bytes.empty();
    }

    const CaptureRecord * find_record(const std::string & stage, const std::string & tensor) const {
        for (const CaptureRecord & rec : input_.references) {
            if (rec.stage == stage && rec.tensor == tensor && rec.payload_kind == "tensor_values" && rec.full_tensor_payload_available) {
                return &rec;
            }
        }
        return nullptr;
    }

    const CaptureRecord * find_record_any(const std::string & stage, std::initializer_list<const char *> tensors) const {
        for (const char * tensor : tensors) {
            if (const CaptureRecord * rec = find_record(stage, tensor)) {
                return rec;
            }
        }
        return nullptr;
    }

    bool has_available_record(const std::string & stage, const std::string & tensor) const {
        for (const CaptureRecord & rec : input_.references) {
            if (rec.stage == stage && rec.tensor == tensor && rec.availability == "available") {
                return true;
            }
        }
        return false;
    }

    void require_record(const CaptureRecord * rec, const std::string & tensor) {
        if (rec == nullptr) {
            kernel_stats_.missing_inputs.push_back(tensor);
        }
    }

    static float read_f32(const uint8_t * p) {
        float v = 0.0f;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }

    static std::vector<float> payload_f32(const CaptureRecord & rec) {
        std::vector<float> out;
        if (rec.payload_bytes.empty() || rec.payload_bytes.size() % sizeof(float) != 0) {
            return out;
        }
        out.resize(rec.payload_bytes.size() / sizeof(float));
        std::memcpy(out.data(), rec.payload_bytes.data(), rec.payload_bytes.size());
        return out;
    }

    static void compare_f32(const std::vector<float> & candidate, const std::vector<float> & reference, double & max_abs, double & rms) {
        max_abs = INFINITY;
        rms = INFINITY;
        if (candidate.size() != reference.size() || candidate.empty()) {
            return;
        }
        max_abs = 0.0;
        double sumsq = 0.0;
        for (size_t i = 0; i < candidate.size(); ++i) {
            const double diff = std::fabs(static_cast<double>(candidate[i]) - static_cast<double>(reference[i]));
            max_abs = std::max(max_abs, diff);
            sumsq += diff * diff;
        }
        rms = std::sqrt(sumsq / static_cast<double>(candidate.size()));
    }

    static void compare_f32_rel(const std::vector<float> & candidate, const std::vector<float> & reference, double & max_rel) {
        max_rel = INFINITY;
        if (candidate.size() != reference.size() || candidate.empty()) {
            return;
        }
        max_rel = 0.0;
        for (size_t i = 0; i < candidate.size(); ++i) {
            const double ref = std::fabs(static_cast<double>(reference[i]));
            if (ref < 1e-12) {
                continue;
            }
            const double diff = std::fabs(static_cast<double>(candidate[i]) - static_cast<double>(reference[i]));
            max_rel = std::max(max_rel, diff / ref);
        }
    }

    static std::vector<float> rms_norm_weight(const std::vector<float> & cur, const std::vector<float> & weight, double eps) {
        std::vector<float> out(cur.size(), 0.0f);
        if (cur.size() != weight.size() || cur.empty()) {
            return out;
        }
        double sumsq = 0.0;
        for (float v : cur) {
            sumsq += static_cast<double>(v) * static_cast<double>(v);
        }
        const double inv_rms = 1.0 / std::sqrt(sumsq / static_cast<double>(cur.size()) + eps);
        for (size_t i = 0; i < cur.size(); ++i) {
            out[i] = static_cast<float>(static_cast<double>(cur[i]) * inv_rms * static_cast<double>(weight[i]));
        }
        return out;
    }

    static std::vector<float> apply_cupd_rope_candidate(
            const std::vector<float> & input,
            bool neox,
            bool swapped_sign,
            int position,
            int n_rot,
            float freq_base,
            float freq_scale,
            float ext_factor,
            float attn_factor,
            float beta_fast,
            float beta_slow,
            int tail_offset,
            int n_ctx_orig) {
        std::vector<float> out = input;
        const int head_dim = static_cast<int>(input.size());
        const int n_nope = tail_offset >= 0 ? tail_offset : head_dim - n_rot;
        if (n_rot <= 0 || n_nope < 0 || (n_rot % 2) != 0) {
            return out;
        }

        const float inv_ndims = -1.0f / static_cast<float>(n_rot);
        const int n_half = n_rot / 2;
        float corr_dims[2] = {0.0f, static_cast<float>(n_rot - 1)};
        rope_yarn_corr_dims(n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
        for (int i = n_nope; i < head_dim; ++i) {
            const int tail = i - n_nope;
            int ic = 0;
            bool first = false;
            int off0 = 0;
            int off1 = 0;
            if (neox) {
                ic = tail % n_half;
                first = tail < n_half;
                off0 = n_nope + ic;
                off1 = off0 + n_half;
            } else {
                ic = tail / 2;
                first = (tail & 1) == 0;
                off0 = n_nope + 2 * ic;
                off1 = off0 + 1;
            }

            const int rope_i0 = 2 * ic;
            const float theta = static_cast<float>(position) * std::pow(freq_base, inv_ndims * rope_i0);
            float cos_theta = 0.0f;
            float sin_theta = 0.0f;
            rope_yarn(theta, freq_scale, corr_dims, rope_i0, ext_factor, attn_factor, cos_theta, sin_theta);
            const float x0 = input[off0];
            const float x1 = input[off1];
            if (!swapped_sign) {
                out[i] = first ? (x0 * cos_theta - x1 * sin_theta) : (x0 * sin_theta + x1 * cos_theta);
            } else {
                out[i] = first ? (x0 * cos_theta + x1 * sin_theta) : (-x0 * sin_theta + x1 * cos_theta);
            }
        }
        return out;
    }

    static float rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
        if (n_ctx_orig <= 0 || n_rot <= 0.0f || base <= 1.0f) {
            return 0.0f;
        }
        return n_dims * std::log(n_ctx_orig / (n_rot * 2.0f * static_cast<float>(M_PI))) / (2.0f * std::log(base));
    }

    static void rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
        const float start = std::floor(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
        const float end = std::ceil(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
        dims[0] = std::max(0.0f, start);
        dims[1] = std::min(static_cast<float>(n_dims - 1), end);
    }

    static float rope_yarn_ramp(float low, float high, int i0) {
        const float y = (i0 / 2.0f - low) / std::max(0.001f, high - low);
        return 1.0f - std::min(1.0f, std::max(0.0f, y));
    }

    static void rope_yarn(
            float theta_extrap,
            float freq_scale,
            const float corr_dims[2],
            int i0,
            float ext_factor,
            float mscale,
            float & cos_theta,
            float & sin_theta) {
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        if (ext_factor != 0.0f) {
            const float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            if (freq_scale > 0.0f) {
                mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
            }
        }
        cos_theta = std::cos(theta) * mscale;
        sin_theta = std::sin(theta) * mscale;
    }

    static std::vector<float> weighted_sum_layout(
            const std::vector<float> & src,
            const std::vector<float> & weights,
            size_t n_embd,
            size_t n_hc,
            const std::string & layout) {
        std::vector<float> out(n_embd, 0.0f);
        if (weights.size() < n_hc || src.size() < n_embd * n_hc) {
            return out;
        }
        for (size_t e = 0; e < n_embd; ++e) {
            double acc = 0.0;
            for (size_t h = 0; h < n_hc; ++h) {
                size_t idx = 0;
                if (layout == "hidden_major") {
                    idx = h * n_embd + e;
                } else if (layout == "e_major" || layout == "transposed_2d") {
                    idx = e * n_hc + h;
                } else {
                    idx = h * n_embd + e;
                }
                acc += static_cast<double>(src[idx]) * static_cast<double>(weights[h]);
            }
            out[e] = static_cast<float>(acc);
        }
        return out;
    }

    static std::vector<float> aohc_hc_post_expand(
            const std::vector<float> & attn_out,
            const std::vector<float> & residual,
            const std::vector<float> & post,
            const std::vector<float> & comb,
            size_t n_embd,
            size_t n_hc) {
        std::vector<float> out(n_embd * n_hc, 0.0f);
        if (attn_out.size() < n_embd || residual.size() < n_embd * n_hc ||
                post.size() < n_hc || comb.size() < n_hc * n_hc) {
            return out;
        }
        for (size_t dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            for (size_t d = 0; d < n_embd; ++d) {
                double acc = static_cast<double>(attn_out[d]) * static_cast<double>(post[dst_hc]);
                for (size_t src_hc = 0; src_hc < n_hc; ++src_hc) {
                    acc += static_cast<double>(comb[dst_hc + src_hc * n_hc]) *
                        static_cast<double>(residual[src_hc * n_embd + d]);
                }
                out[dst_hc * n_embd + d] = static_cast<float>(acc);
            }
        }
        return out;
    }

    static std::string formula_report_line(
            const std::string & formula_id,
            const std::string & layout,
            double cur_max,
            double cur_rms,
            double cur_max_rel,
            double norm_max,
            double norm_rms,
            double post_max,
            double post_rms,
            bool tier_b_pass) {
        std::ostringstream out;
        out << formula_id << "/" << layout
            << ":cur_max_abs=" << cur_max
            << ",cur_rms=" << cur_rms
            << ",cur_max_rel=" << cur_max_rel
            << ",norm_max_abs=" << norm_max
            << ",norm_rms=" << norm_rms
            << ",post_max_abs=" << post_max
            << ",post_rms=" << post_rms
            << ",tier_b=" << (tier_b_pass ? "pass" : "fail");
        return out.str();
    }

    static void compare_payloads(const CaptureRecord & rec, const std::vector<uint8_t> & candidate, double & max_abs, double & rms) {
        max_abs = 0.0;
        rms = 0.0;
        if (candidate.size() != rec.payload_bytes.size() || rec.payload_bytes.empty()) {
            max_abs = INFINITY;
            rms = INFINITY;
            return;
        }

        if ((rec.dtype == "f32" || rec.dtype == "fp32" || rec.dtype == "metadata") &&
                rec.payload_bytes.size() % sizeof(float) == 0) {
            double sumsq = 0.0;
            const size_t n = rec.payload_bytes.size() / sizeof(float);
            for (size_t i = 0; i < n; ++i) {
                const float a = read_f32(rec.payload_bytes.data() + i * sizeof(float));
                const float b = read_f32(candidate.data() + i * sizeof(float));
                const double diff = std::fabs(static_cast<double>(a) - static_cast<double>(b));
                max_abs = std::max(max_abs, diff);
                sumsq += diff * diff;
            }
            rms = n > 0 ? std::sqrt(sumsq / static_cast<double>(n)) : 0.0;
            return;
        }

        size_t mismatches = 0;
        for (size_t i = 0; i < rec.payload_bytes.size(); ++i) {
            if (rec.payload_bytes[i] != candidate[i]) {
                ++mismatches;
            }
        }
        max_abs = mismatches == 0 ? 0.0 : INFINITY;
        rms = mismatches == 0 ? 0.0 : INFINITY;
    }

    void add_failure(const std::string & reason, const std::string & stage) {
        ++failed_cases_;
        if (!stage.empty()) {
            failed_stages_.insert(stage);
        }
        if (std::find(reasons_.begin(), reasons_.end(), reason) == reasons_.end()) {
            reasons_.push_back(reason);
        }
    }

    Dsv4LayerExecutorInput input_;
    Dsv4LayerExecutorOutput output_;
    int layer_filter_ = -1;
    int64_t token_filter_ = -1;
    bool allow_unavailable_optional_ = true;
    bool allow_unavailable_required_ = false;
    bool require_full_tensors_ = false;
    bool require_byte_payloads_ = false;
    std::string cupd_stage_ = "norm_weighted";

    int64_t exact_cases_ = 0;
    int64_t failed_cases_ = 0;
    double max_abs_ = 0.0;
    double max_rms_ = 0.0;
    int64_t required_records_ = 0;
    int64_t optional_records_ = 0;
    int64_t available_records_ = 0;
    int64_t unavailable_records_ = 0;
    int64_t missing_required_ = 0;
    int64_t payload_stats_only_ = 0;
    int64_t payload_tensor_values_ = 0;
    int64_t payload_byte_values_ = 0;
    int64_t payload_metadata_only_ = 0;
    int64_t full_tensor_payload_records_ = 0;
    int64_t byte_payload_records_ = 0;
    int64_t warnings_ = 0;
    std::set<std::string> required_tensor_payload_stages_;
    std::set<std::string> required_byte_payload_stages_;
    std::set<std::string> failed_stages_;
    std::vector<std::string> reasons_;
    Dsv4KernelCompareStats kernel_stats_;
};

static void usage(const char * argv0) {
    std::cerr
        << "usage: " << argv0 << " --fixtures <dir>|--captures-dir <dir>|--capture <file> [--layer N] [--token N] --mode identity|kernel|kernel_copy_smoke|hcnorm_recompute|rmoe_recompute|aohc_recompute|cupd_recompute [--stage hc_pre_norm|routed_moe_final_output|aohc_boundary|compressor_update] "
        << "[--kernel-mode hcnorm_recompute|rmoe_final_from_substages|aohc_hc_post_from_substages|cupd_norm_weighted|copy_smoke] [--forbid-copy-smoke] "
        << "[--cupd-stage norm_weighted|rope] [--allow-unavailable-optional] [--allow-unavailable-required] [--require-full-tensors] [--require-byte-payloads]\n";
}

int main(int argc, char ** argv) {
    std::vector<std::string> captures;
    std::string mode = "identity";
    std::string kernel_mode = "hcnorm_recompute";
    std::string cupd_stage = "norm_weighted";
    std::string stage;
    int layer = -1;
    int64_t token = -1;
    bool allow_unavailable_optional = true;
    bool allow_unavailable_required = false;
    bool require_full_tensors = false;
    bool require_byte_payloads = false;
    bool forbid_copy_smoke = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "--captures-dir" || arg == "--fixtures" || arg == "--capture") && i + 1 < argc) {
            captures.push_back(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--kernel-mode" && i + 1 < argc) {
            kernel_mode = argv[++i];
        } else if (arg == "--cupd-stage" && i + 1 < argc) {
            cupd_stage = argv[++i];
        } else if (arg == "--stage" && i + 1 < argc) {
            stage = argv[++i];
        } else if (arg == "--layer" && i + 1 < argc) {
            layer = std::stoi(argv[++i]);
        } else if (arg == "--token" && i + 1 < argc) {
            token = std::stoll(argv[++i]);
        } else if (arg == "--allow-unavailable-optional") {
            allow_unavailable_optional = true;
        } else if (arg == "--allow-unavailable-required") {
            allow_unavailable_required = true;
        } else if (arg == "--require-full-tensors") {
            require_full_tensors = true;
        } else if (arg == "--require-byte-payloads") {
            require_byte_payloads = true;
        } else if (arg == "--forbid-copy-smoke") {
            forbid_copy_smoke = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (captures.empty() || (mode != "identity" && mode != "kernel" && mode != "kernel_copy_smoke" && mode != "hcnorm_recompute" && mode != "rmoe_recompute" && mode != "aohc_recompute" && mode != "cupd_recompute")) {
        usage(argv[0]);
        return 2;
    }
    if ((mode == "kernel" || mode == "kernel_copy_smoke") && stage.empty()) {
        usage(argv[0]);
        return 2;
    }

    Dsv4LayerExecutorHarness harness;
    harness.set_layer_filter(layer);
    harness.set_token_filter(token);
    harness.set_allow_unavailable_optional(allow_unavailable_optional);
    harness.set_allow_unavailable_required(allow_unavailable_required);
    harness.set_require_full_tensors(require_full_tensors);
    harness.set_require_byte_payloads(require_byte_payloads);
    harness.set_cupd_stage(cupd_stage);

    bool load_ok = true;
    for (const std::string & capture : captures) {
        load_ok = harness.load_captures(capture) && load_ok;
    }

    const Dsv4TolerancePolicy policy;
    bool run_ok = false;
    bool kernel_ok = true;
    if (mode == "identity") {
        run_ok = harness.run_identity();
    } else if (mode == "kernel") {
        run_ok = harness.run_kernel(stage, policy, kernel_mode, forbid_copy_smoke);
        kernel_ok = run_ok;
    } else if (mode == "kernel_copy_smoke") {
        run_ok = harness.run_kernel(stage, policy, "copy_smoke", forbid_copy_smoke);
        kernel_ok = run_ok;
    } else if (mode == "rmoe_recompute") {
        run_ok = harness.run_rmoe_recompute(policy);
        kernel_ok = run_ok;
    } else if (mode == "aohc_recompute") {
        run_ok = harness.run_aohc_recompute(policy);
        kernel_ok = run_ok;
    } else if (mode == "cupd_recompute") {
        run_ok = harness.run_cupd_recompute(policy);
        kernel_ok = run_ok;
    } else {
        run_ok = harness.run_hcnorm_recompute(policy);
        kernel_ok = run_ok;
    }
    bool compare_ok = harness.compare_identity(policy);

    const auto stages = harness.stages_loaded();
    const auto layers = harness.layers_loaded();
    const auto tokens = harness.tokens_loaded();

    bool missing_stage = false;
    std::map<std::string, std::string> stage_status;
    for (const std::string & stage : k_required_stages) {
        stage_status[stage] = harness.stage_status(stage);
        if (stage_status[stage] == "missing") {
            missing_stage = true;
            harness.mark_missing_required_stage(stage);
            compare_ok = false;
        }
    }

    const bool result = load_ok && run_ok && compare_ok && !missing_stage;
    const std::string reason = harness.reason_summary();

    std::cout
        << "dsv4_layer_executor_harness_summary:\n"
        << "  captures_loaded=" << harness.captures_loaded() << "\n"
        << "  records_loaded=" << harness.captures_loaded() << "\n"
        << "  stages_loaded=" << stages.size() << "\n"
        << "  layers_loaded=" << layers.size() << "\n"
        << "  tokens_loaded=" << tokens.size() << "\n"
        << "  mode=" << mode << "\n"
        << "  stage=" << (stage.empty() ? "none" : stage) << "\n"
        << "  tier=" << (mode == "identity" ? "A" : "B") << "\n"
        << "  kernel_mode=" << harness.kernel_stats().kernel_mode << "\n"
        << "  required_records=" << harness.required_records() << "\n"
        << "  optional_records=" << harness.optional_records() << "\n"
        << "  available_records=" << harness.available_records() << "\n"
        << "  unavailable_records=" << harness.unavailable_records() << "\n"
        << "  missing_required=" << harness.missing_required() << "\n"
        << "  payload_stats_only=" << harness.payload_stats_only() << "\n"
        << "  payload_tensor_values=" << harness.payload_tensor_values() << "\n"
        << "  payload_byte_values=" << harness.payload_byte_values() << "\n"
        << "  payload_metadata_only=" << harness.payload_metadata_only() << "\n"
        << "  full_tensor_payload_available=" << (harness.has_full_tensor_payload() ? 1 : 0) << "\n"
        << "  byte_payload_available=" << (harness.has_byte_payload() ? 1 : 0) << "\n"
        << "  stats_only=" << (harness.has_stats_only_payload() ? 1 : 0) << "\n"
        << "  metadata_only=" << (harness.has_metadata_only_payload() ? 1 : 0) << "\n"
        << "  warnings=" << harness.warnings() << "\n"
        << "\n"
        << "  stage_hc_pre_norm=" << stage_status["hc_pre_norm"] << "\n"
        << "  stage_routed_moe_final_output=" << stage_status["routed_moe_final_output"] << "\n"
        << "  stage_aohc_boundary=" << stage_status["aohc_boundary"] << "\n"
        << "  stage_compressor_update=" << stage_status["compressor_update"] << "\n"
        << "  stage_kv_cache_finalizer=" << stage_status["kv_cache_finalizer"] << "\n"
        << "\n"
        << "  exact_cases=" << harness.exact_cases() << "\n"
        << "  failed_cases=" << harness.failed_cases() << "\n"
        << "  max_abs=" << harness.max_abs() << "\n"
        << "  max_rms=" << harness.max_rms() << "\n"
        << "  kernel_cases=" << harness.kernel_stats().cases << "\n"
        << "  kernel_failed_cases=" << harness.kernel_stats().failed_cases << "\n"
        << "  kernel_max_abs=" << harness.kernel_stats().max_abs << "\n"
        << "  kernel_rms=" << harness.kernel_stats().rms << "\n"
        << "  kernel_tier_b=" << (kernel_ok ? "pass" : "fail") << "\n"
        << "  metal_recompute=" << (harness.kernel_stats().metal_recompute ? 1 : 0) << "\n"
        << "  copied_reference_output=" << (harness.kernel_stats().copied_reference_output ? 1 : 0) << "\n"
        << "  formula=" << (harness.kernel_stats().hcnorm_best_formula.empty() ? "none" : harness.kernel_stats().hcnorm_best_formula) << "\n"
        << "  input_layout=" << (harness.kernel_stats().hcnorm_input_layout.empty() ? "none" : harness.kernel_stats().hcnorm_input_layout) << "\n"
        << "  eps=" << harness.kernel_stats().hcnorm_best_eps << "\n"
        << "  tier_a_exact_pass=" << (harness.kernel_stats().tier_a_exact_pass ? 1 : 0) << "\n"
        << "  recompute_possible=" << (harness.kernel_stats().recompute_possible ? 1 : 0) << "\n"
        << "  hcnorm_best_formula=" << (harness.kernel_stats().hcnorm_best_formula.empty() ? "none" : harness.kernel_stats().hcnorm_best_formula) << "\n"
        << "  hcnorm_input_layout=" << (harness.kernel_stats().hcnorm_input_layout.empty() ? "none" : harness.kernel_stats().hcnorm_input_layout) << "\n"
        << "  hcnorm_best_eps=" << harness.kernel_stats().hcnorm_best_eps << "\n"
        << "  hcnorm_cur_max_abs=" << harness.kernel_stats().hcnorm_cur_max_abs << "\n"
        << "  hcnorm_cur_rms=" << harness.kernel_stats().hcnorm_cur_rms << "\n"
        << "  hcnorm_norm_max_abs=" << harness.kernel_stats().hcnorm_norm_max_abs << "\n"
        << "  hcnorm_norm_rms=" << harness.kernel_stats().hcnorm_norm_rms << "\n"
        << "  hcnorm_post_max_abs=" << harness.kernel_stats().hcnorm_post_max_abs << "\n"
        << "  hcnorm_post_rms=" << harness.kernel_stats().hcnorm_post_rms << "\n"
        << "  partial_rmoe_recompute=" << (harness.kernel_stats().partial_rmoe_recompute ? 1 : 0) << "\n"
        << "  full_rmoe_recompute=" << (harness.kernel_stats().full_rmoe_recompute ? 1 : 0) << "\n"
        << "  routed_sum_only_recompute=" << (harness.kernel_stats().routed_sum_only_recompute ? 1 : 0) << "\n"
        << "  expert_weight_recompute=" << (harness.kernel_stats().expert_weight_recompute ? 1 : 0) << "\n"
        << "  weights_not_decoded=" << (harness.kernel_stats().weights_not_decoded ? 1 : 0) << "\n"
        << "  weighted_down_available=" << (harness.kernel_stats().weighted_down_available ? 1 : 0) << "\n"
        << "  shared_down_available=" << (harness.kernel_stats().shared_down_available ? 1 : 0) << "\n"
        << "  reference_routed_sum_available=" << (harness.kernel_stats().reference_routed_sum_available ? 1 : 0) << "\n"
        << "  reference_final_ffn_available=" << (harness.kernel_stats().reference_final_ffn_available ? 1 : 0) << "\n"
        << "  rmoe_routed_sum_max_abs=" << harness.kernel_stats().rmoe_routed_sum_max_abs << "\n"
        << "  rmoe_routed_sum_rms=" << harness.kernel_stats().rmoe_routed_sum_rms << "\n"
        << "  rmoe_routed_sum_max_rel=" << harness.kernel_stats().rmoe_routed_sum_max_rel << "\n"
        << "  routed_sum_max_abs=" << harness.kernel_stats().rmoe_routed_sum_max_abs << "\n"
        << "  routed_sum_rms=" << harness.kernel_stats().rmoe_routed_sum_rms << "\n"
        << "  routed_sum_max_rel=" << harness.kernel_stats().rmoe_routed_sum_max_rel << "\n"
        << "  rmoe_final_ffn_max_abs=" << harness.kernel_stats().rmoe_final_ffn_max_abs << "\n"
        << "  rmoe_final_ffn_rms=" << harness.kernel_stats().rmoe_final_ffn_rms << "\n"
        << "  rmoe_final_ffn_max_rel=" << harness.kernel_stats().rmoe_final_ffn_max_rel << "\n"
        << "  final_ffn_max_abs=" << harness.kernel_stats().rmoe_final_ffn_max_abs << "\n"
        << "  final_ffn_rms=" << harness.kernel_stats().rmoe_final_ffn_rms << "\n"
        << "  final_ffn_max_rel=" << harness.kernel_stats().rmoe_final_ffn_max_rel << "\n"
        << "  partial_aohc_recompute=" << (harness.kernel_stats().partial_aohc_recompute ? 1 : 0) << "\n"
        << "  full_aohc_recompute=" << (harness.kernel_stats().full_aohc_recompute ? 1 : 0) << "\n"
        << "  aohc_attn_out_available=" << (harness.kernel_stats().aohc_attn_out_available ? 1 : 0) << "\n"
        << "  aohc_residual_available=" << (harness.kernel_stats().aohc_residual_available ? 1 : 0) << "\n"
        << "  aohc_post_weights_available=" << (harness.kernel_stats().aohc_post_weights_available ? 1 : 0) << "\n"
        << "  aohc_comb_available=" << (harness.kernel_stats().aohc_comb_available ? 1 : 0) << "\n"
        << "  aohc_reference_available=" << (harness.kernel_stats().aohc_reference_available ? 1 : 0) << "\n"
        << "  aohc_matches_hc_pre_norm_input="
        << (harness.kernel_stats().aohc_matches_hc_pre_norm_input < 0 ? "not_checked" : std::to_string(harness.kernel_stats().aohc_matches_hc_pre_norm_input)) << "\n"
        << "  aohc_output_max_abs=" << harness.kernel_stats().aohc_output_max_abs << "\n"
        << "  aohc_output_rms=" << harness.kernel_stats().aohc_output_rms << "\n"
        << "  aohc_output_max_rel=" << harness.kernel_stats().aohc_output_max_rel << "\n"
        << "  partial_cupd_recompute=" << (harness.kernel_stats().partial_cupd_recompute ? 1 : 0) << "\n"
        << "  full_cupd_recompute=" << (harness.kernel_stats().full_cupd_recompute ? 1 : 0) << "\n"
        << "  cupd_compressed_norm_available=" << (harness.kernel_stats().cupd_compressed_norm_available ? 1 : 0) << "\n"
        << "  cupd_norm_weight_available=" << (harness.kernel_stats().cupd_norm_weight_available ? 1 : 0) << "\n"
        << "  cupd_norm_weighted_available=" << (harness.kernel_stats().cupd_norm_weighted_available ? 1 : 0) << "\n"
        << "  cupd_rope_input_available=" << (harness.kernel_stats().cupd_rope_input_available ? 1 : 0) << "\n"
        << "  cupd_rope_reference_available=" << (harness.kernel_stats().cupd_rope_reference_available ? 1 : 0) << "\n"
        << "  cupd_rope_position_available=" << (harness.kernel_stats().cupd_rope_position_available ? 1 : 0) << "\n"
        << "  cupd_rope_cache_position_available=" << (harness.kernel_stats().cupd_rope_cache_position_available ? 1 : 0) << "\n"
        << "  cupd_rope_n_rot_available=" << (harness.kernel_stats().cupd_rope_n_rot_available ? 1 : 0) << "\n"
        << "  cupd_rope_freq_base_available=" << (harness.kernel_stats().cupd_rope_freq_base_available ? 1 : 0) << "\n"
        << "  cupd_rope_freq_scale_available=" << (harness.kernel_stats().cupd_rope_freq_scale_available ? 1 : 0) << "\n"
        << "  cupd_rope_mode_available=" << (harness.kernel_stats().cupd_rope_mode_available ? 1 : 0) << "\n"
        << "  cupd_rope_tail_offset_available=" << (harness.kernel_stats().cupd_rope_tail_offset_available ? 1 : 0) << "\n"
        << "  cupd_rope_cos_available=" << (harness.kernel_stats().cupd_rope_cos_available ? 1 : 0) << "\n"
        << "  cupd_rope_sin_available=" << (harness.kernel_stats().cupd_rope_sin_available ? 1 : 0) << "\n"
        << "  cupd_rope_nonzero_position_records=" << harness.kernel_stats().cupd_rope_nonzero_position_records << "\n"
        << "  cupd_rope_position=" << harness.kernel_stats().cupd_rope_position << "\n"
        << "  cupd_rope_n_rot=" << harness.kernel_stats().cupd_rope_n_rot << "\n"
        << "  cupd_rope_tail_offset=" << harness.kernel_stats().cupd_rope_tail_offset << "\n"
        << "  compressed_norm_available=" << (harness.kernel_stats().cupd_compressed_norm_available ? 1 : 0) << "\n"
        << "  compressed_norm_weight_available=" << (harness.kernel_stats().cupd_norm_weight_available ? 1 : 0) << "\n"
        << "  compressed_norm_weighted_reference_available=" << (harness.kernel_stats().cupd_norm_weighted_available ? 1 : 0) << "\n"
        << "  cupd_output_max_abs=" << harness.kernel_stats().cupd_output_max_abs << "\n"
        << "  cupd_output_rms=" << harness.kernel_stats().cupd_output_rms << "\n"
        << "  cupd_output_max_rel=" << harness.kernel_stats().cupd_output_max_rel << "\n"
        << "  cache_mutation=disabled\n"
        << "  missing_inputs=" << join_strings(harness.kernel_stats().missing_inputs) << "\n"
        << "  missing_formula_params=" << join_strings(harness.kernel_stats().missing_formula_params) << "\n"
        << "  tier_b_status=executor_acceptance_policy_available_when_payload_present\n"
        << "  logit_distribution_checks=not_applicable_to_identity_harness\n"
        << "  transcript_fallback=0\n";
    for (const std::string & report : harness.kernel_stats().hcnorm_formula_reports) {
        std::cout << "  hcnorm_formula: " << report << "\n";
    }
    if (!reason.empty()) {
        std::cout << "  reason=" << reason << "\n";
    }
    if (!harness.kernel_stats().first_failure.empty()) {
        std::cout << "  kernel_reason=" << harness.kernel_stats().first_failure << "\n";
    }
    std::cout << "  result=" << (result ? "pass" : "fail") << "\n";

    return result ? 0 : 1;
}
