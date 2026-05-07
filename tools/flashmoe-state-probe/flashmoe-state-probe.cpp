#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct probe_config {
    std::vector<int32_t> prefixes;
    std::string csv_path;
    std::string save_dir;
    std::string only;
    bool save_mismatches = true;
    bool stop_on_mismatch = false;
    int topk = 8;
    int decode = 0;
};

struct probe_result {
    bool ok = false;
    int ret = 0;
    int32_t prefix = 0;
    bool forced = false;
    size_t state_size = 0;
    uint64_t state_hash = 0;
    uint64_t logits_hash = 0;
    int64_t logits_finite = 0;
    int64_t logits_nan = 0;
    int64_t logits_inf = 0;
    std::vector<int32_t> top_tokens;
    std::vector<float> top_logits;
    std::vector<float> logits;
    std::vector<llama_token> decode_tokens;
    uint64_t decode_hash = 0;
    std::string decode_text;
    std::string error;
    std::vector<uint8_t> state;
};

struct tensor_fingerprint_ctx {
    bool forced = false;
    int32_t prefix = 0;
    std::unordered_map<std::string, int> counts;
};

struct logits_diff {
    bool ok = false;
    bool exact = false;
    int32_t max_abs_id = -1;
    double max_abs = 0.0;
    double rms = 0.0;
    double base_top1_delta = 0.0;
    double base_top1_margin = 0.0;
    double forced_top1_margin = 0.0;
};

static void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s [common llama args] --state-probe-range A:B[:STEP] [--state-probe-csv FILE]\n"
            "\n"
            "Flash-MoE diagnostic: for each prefix length, run a baseline context and a forced-prefill\n"
            "context, then compare serialized sequence-state hashes after prefill.\n"
            "\n"
            "custom options:\n"
            "  --state-probe-prefix N          add one prefix length\n"
            "  --state-probe-range A:B[:STEP]  add inclusive prefix range\n"
            "  --state-probe-csv FILE          write CSV summary\n"
            "  --state-probe-topk N            compare top-N next-token logits (default: 8)\n"
            "  --state-probe-decode N          greedily decode N tokens after each prefix and compare tokens\n"
            "  --state-probe-save-dir DIR      also save raw state bytes for mismatched prefixes\n"
            "  --state-probe-save-all          save raw state bytes for all prefixes\n"
            "  --state-probe-only MODE         run only 'baseline' or 'forced' for tensor diagnostics\n"
            "  --state-probe-stop-on-mismatch  stop after first mismatch\n"
            "\n",
            argv0);
}

static bool parse_i32(const std::string & s, int32_t & out) {
    char * end = nullptr;
    errno = 0;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0' || v < 0 || v > INT32_MAX) {
        return false;
    }
    out = (int32_t) v;
    return true;
}

static bool add_range(const std::string & spec, std::vector<int32_t> & out) {
    const size_t p0 = spec.find(':');
    if (p0 == std::string::npos) {
        return false;
    }
    const size_t p1 = spec.find(':', p0 + 1);
    int32_t first = 0;
    int32_t last = 0;
    int32_t step = 1;
    if (!parse_i32(spec.substr(0, p0), first)) {
        return false;
    }
    if (!parse_i32(spec.substr(p0 + 1, p1 == std::string::npos ? std::string::npos : p1 - p0 - 1), last)) {
        return false;
    }
    if (p1 != std::string::npos && !parse_i32(spec.substr(p1 + 1), step)) {
        return false;
    }
    if (step <= 0 || first > last) {
        return false;
    }
    for (int32_t n = first; n <= last; n += step) {
        out.push_back(n);
        if (INT32_MAX - n < step) {
            break;
        }
    }
    return true;
}

static uint64_t fnv1a64(const uint8_t * data, size_t size) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static bool env_flag(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static int tensor_fingerprint_layer() {
    static int layer = INT32_MIN;
    if (layer == INT32_MIN) {
        layer = 0;
        const char * value = std::getenv("LLAMA_STATE_PROBE_TENSOR_LAYER");
        if (value != nullptr && std::strcmp(value, "all") == 0) {
            layer = -1;
        } else {
            int32_t parsed = 0;
            if (value != nullptr && parse_i32(value, parsed)) {
            layer = parsed;
            }
        }
    }

    return layer;
}

static const char * tensor_fingerprint_base_filter() {
    const char * value = std::getenv("LLAMA_STATE_PROBE_TENSOR_BASE");
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

static int64_t tensor_fingerprint_rows_limit() {
    static int64_t rows = -1;
    if (rows < 0) {
        rows = 8192;
        if (const char * value = std::getenv("LLAMA_STATE_PROBE_TENSOR_ROWS")) {
            int32_t parsed = 0;
            if (parse_i32(value, parsed) && parsed > 0) {
                rows = parsed;
            }
        }
    }

    return rows;
}

struct tensor_scan_result {
    uint64_t hash = 1469598103934665603ULL;
    int64_t finite = 0;
    int64_t nan = 0;
    int64_t inf = 0;
    bool has_float = false;
};

static void tensor_note_float(tensor_scan_result & out, float value) {
    if (std::isnan(value)) {
        out.nan++;
    } else if (std::isinf(value)) {
        out.inf++;
    } else {
        out.finite++;
    }
}

static tensor_scan_result tensor_scan_rows(ggml_tensor * t, int64_t token_dim, int64_t row_start, int64_t rows) {
    const size_t elem = ggml_row_size(t->type, 1);
    size_t row_bytes = 0;
    size_t row_stride = 0;
    if (token_dim == 2) {
        row_bytes = size_t(t->ne[0]) * size_t(t->ne[1]) * elem;
        row_stride = size_t(t->nb[2]);
    } else {
        row_bytes = size_t(t->ne[0]) * elem;
        row_stride = size_t(t->nb[1]);
    }

    std::vector<uint8_t> row(row_bytes);
    tensor_scan_result out;
    out.has_float = t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16;
    for (int64_t i = row_start; i < row_start + rows; ++i) {
        const size_t offset = size_t(i) * row_stride;
        ggml_backend_tensor_get(t, row.data(), offset, row.size());
        for (uint8_t b : row) {
            out.hash ^= b;
            out.hash *= 1099511628211ULL;
        }

        if (!out.has_float) {
            continue;
        }

        const int64_t n_values = int64_t(row_bytes / elem);
        if (t->type == GGML_TYPE_F32) {
            const auto * vals = reinterpret_cast<const float *>(row.data());
            for (int64_t j = 0; j < n_values; ++j) {
                tensor_note_float(out, vals[j]);
            }
        } else if (t->type == GGML_TYPE_F16) {
            const auto * vals = reinterpret_cast<const ggml_fp16_t *>(row.data());
            for (int64_t j = 0; j < n_values; ++j) {
                tensor_note_float(out, ggml_fp16_to_fp32(vals[j]));
            }
        } else if (t->type == GGML_TYPE_BF16) {
            const auto * vals = reinterpret_cast<const ggml_bf16_t *>(row.data());
            for (int64_t j = 0; j < n_values; ++j) {
                tensor_note_float(out, ggml_bf16_to_fp32(vals[j]));
            }
        }
    }

    return out;
}

static bool tensor_base_wanted(const std::string & base) {
    if (const char * filter = tensor_fingerprint_base_filter()) {
        return base == filter;
    }

    static const char * bases[] = {
        "hc_attn_pre",
        "hc_attn_pre_mixes",
        "hc_attn_pre_weights",
        "hc_attn_pre_post_weights",
        "hc_attn_pre_comb",
        "attn_norm",
        "q_lora",
        "q_lora_norm",
        "Qnorm",
        "Qcur",
        "KVnorm",
        "KVrope",
        "KVcur",
        "KVcompress",
        "indexer_KVcompress",
        "indexer_scores",
        "indexer_topk",
        "dsv4_attn_static_mask",
        "dsv4_attn_raw_window_mask",
        "dsv4_attn_compress_mask",
        "kqv_out",
        "attn_out",
        "hc_attn_post",
        "hc_ffn_pre",
        "ffn_norm",
        "ffn_moe_topk",
        "ffn_moe_topk_reduced",
        "ffn_moe_weights",
        "ffn_moe_weights_norm",
        "ffn_moe_weights_scaled",
        "ffn_moe_prefill_tail",
        "ffn_moe_out",
        "ffn_shexp",
        "ffn_out",
    };

    for (const char * item : bases) {
        if (base == item) {
            return true;
        }
    }

    return false;
}

static bool tensor_name_wanted(const char * name) {
    if (name == nullptr) {
        return false;
    }

    if (std::strcmp(name, "self_kq_mask_swa_cnv") == 0) {
        return tensor_fingerprint_layer() == 0;
    }

    const char * dash = std::strrchr(name, '-');
    if (dash == nullptr || dash == name || dash[1] == '\0') {
        return false;
    }

    int32_t layer = 0;
    const int wanted_layer = tensor_fingerprint_layer();
    if (!parse_i32(dash + 1, layer) || (wanted_layer >= 0 && layer != wanted_layer)) {
        return false;
    }

    return tensor_base_wanted(std::string(name, dash - name));
}

static size_t tensor_type_size(ggml_type type) {
    return ggml_row_size(type, 1);
}

static bool tensor_fingerprint_cb(ggml_tensor * t, bool ask, void * user_data) {
    const char * name = ggml_get_name(t);
    const bool wanted = tensor_name_wanted(name);
    if (ask) {
        return wanted;
    }
    if (!wanted) {
        return true;
    }

    auto * ctx = static_cast<tensor_fingerprint_ctx *>(user_data);
    const int call = ctx != nullptr ? ctx->counts[name]++ : 0;

    const int64_t token_dim = t->ne[2] > 1 ? 2 : 1;

    const int64_t rows_total = t->ne[token_dim];
    const int64_t rows = std::min<int64_t>(rows_total, tensor_fingerprint_rows_limit());
    const size_t elem = tensor_type_size(t->type);
    size_t row_bytes = 0;
    if (token_dim == 2) {
        row_bytes = size_t(t->ne[0]) * size_t(t->ne[1]) * elem;
    } else {
        row_bytes = size_t(t->ne[0]) * elem;
    }

    bool ok = true;
    const tensor_scan_result scan = tensor_scan_rows(t, token_dim, 0, rows);

    tensor_scan_result tail_scan;
    bool has_tail = rows_total > rows;
    if (has_tail) {
        tail_scan = tensor_scan_rows(t, token_dim, rows_total - rows, rows);
    }

    tensor_scan_result half0_scan;
    tensor_scan_result half1_scan;
    bool has_half = false;
    if (rows_total > 0 && rows_total % 2 == 0 && rows_total <= 4096) {
        const int64_t half_rows = rows_total / 2;
        if (half_rows > 0) {
            has_half = true;
            half0_scan = tensor_scan_rows(t, token_dim, 0, half_rows);
            half1_scan = tensor_scan_rows(t, token_dim, half_rows, half_rows);
        }
    }

    fprintf(stderr,
            "state_probe_tensor forced=%d prefix=%d name=%s call=%d type=%s ne=[%lld,%lld,%lld,%lld] token_dim=%lld rows=%lld/%lld row_bytes=%zu hash=%016" PRIx64 " tail_hash=%016" PRIx64 " half_hash=%016" PRIx64 ":%016" PRIx64 " finite=%lld nan=%lld inf=%lld tail_finite=%lld tail_nan=%lld tail_inf=%lld half_nan=%lld:%lld ok=%d\n",
            ctx != nullptr && ctx->forced ? 1 : 0,
            ctx != nullptr ? ctx->prefix : -1,
            name,
            call,
            ggml_type_name(t->type),
            (long long) t->ne[0],
            (long long) t->ne[1],
            (long long) t->ne[2],
            (long long) t->ne[3],
            (long long) token_dim,
            (long long) rows,
            (long long) rows_total,
            row_bytes,
            scan.hash,
            has_tail ? tail_scan.hash : 0,
            has_half ? half0_scan.hash : 0,
            has_half ? half1_scan.hash : 0,
            (long long) scan.finite,
            (long long) scan.nan,
            (long long) scan.inf,
            (long long) (has_tail ? tail_scan.finite : 0),
            (long long) (has_tail ? tail_scan.nan : 0),
            (long long) (has_tail ? tail_scan.inf : 0),
            (long long) (has_half ? half0_scan.nan : 0),
            (long long) (has_half ? half1_scan.nan : 0),
            ok ? 1 : 0);
    fflush(stderr);

    return true;
}

static bool write_file(const std::filesystem::path & path, const std::vector<uint8_t> & data) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        return false;
    }
    fout.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) data.size());
    return (bool) fout;
}

static uint32_t dsv4_safe_prefill_batch_for_ctx(uint32_t n_ctx) {
    return n_ctx >= 65536 ? 2048 : 8192;
}

static bool decode_range(llama_context * ctx, const std::vector<llama_token> & tokens, int32_t start, int32_t count, int32_t total_prefix) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (int32_t i = 0; i < count; ++i) {
        const bool want_logits = start + i + 1 == total_prefix;
        common_batch_add(batch, tokens[start + i], start + i, { 0 }, want_logits);
    }
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret == 0;
}

static int32_t greedy_token_from_logits(const std::vector<float> & logits) {
    if (logits.empty()) {
        return -1;
    }

    int32_t best = 0;
    float best_logit = logits[0];
    for (int32_t i = 1; i < (int32_t) logits.size(); ++i) {
        if (logits[i] > best_logit) {
            best = i;
            best_logit = logits[i];
        }
    }
    return best;
}

static void save_tokens(const std::filesystem::path & path, const std::vector<llama_token> & tokens, int32_t prefix) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream fout(path);
    for (int32_t i = 0; i < prefix && i < (int32_t) tokens.size(); ++i) {
        fout << tokens[i] << '\n';
    }
}

static probe_result run_one(
        llama_model * model,
        const common_params & params,
        const std::vector<llama_token> & tokens,
        int32_t prefix,
        bool forced,
        int topk,
        int decode_n,
        bool collect_state) {
    probe_result res;
    res.prefix = prefix;
    res.forced = forced;

    common_params run_params = params;
    run_params.moe_force_prefill_batch = forced;
    tensor_fingerprint_ctx fingerprint_ctx;
    if (env_flag("LLAMA_STATE_PROBE_TENSOR_FINGERPRINT")) {
        fingerprint_ctx.forced = forced;
        fingerprint_ctx.prefix = prefix;
        run_params.cb_eval = tensor_fingerprint_cb;
        run_params.cb_eval_user_data = &fingerprint_ctx;
    }

    llama_context_params cparams = common_context_params_to_llama(run_params);
    llama_context * raw_ctx = llama_init_from_model(model, cparams);
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(raw_ctx, llama_free);
    if (!ctx) {
        res.error = "failed to create context";
        return res;
    }

    uint32_t chunk_limit = forced ?
            (params.moe_prefill_batch > 0 ? (uint32_t) params.moe_prefill_batch : (uint32_t) prefix) :
            (params.moe_prefill_batch > 0 ? (uint32_t) params.moe_prefill_batch : 8192);
    if (!forced) {
        chunk_limit = std::min<uint32_t>(chunk_limit, dsv4_safe_prefill_batch_for_ctx(params.n_ctx));
    }
    chunk_limit = std::max<uint32_t>(1, chunk_limit);

    for (int32_t start = 0; start < prefix; ) {
        const int32_t n_cur = std::min<int32_t>(prefix - start, (int32_t) chunk_limit);
        if (!decode_range(ctx.get(), tokens, start, n_cur, prefix)) {
            res.ret = -1;
            res.error = "llama_decode failed";
            return res;
        }
        start += n_cur;
    }

    const float * logits = llama_get_logits_ith(ctx.get(), -1);
    if (logits == nullptr) {
        res.error = "missing logits";
        return res;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    res.logits.assign(logits, logits + n_vocab);
    res.logits_hash = fnv1a64(reinterpret_cast<const uint8_t *>(res.logits.data()), sizeof(float) * (size_t) n_vocab);
    for (float value : res.logits) {
        if (std::isnan(value)) {
            res.logits_nan++;
        } else if (std::isinf(value)) {
            res.logits_inf++;
        } else {
            res.logits_finite++;
        }
    }

    std::vector<int32_t> order(n_vocab);
    std::iota(order.begin(), order.end(), 0);
    const int keep = std::min<int>(std::max(1, topk), n_vocab);
    std::partial_sort(order.begin(), order.begin() + keep, order.end(),
            [&](int32_t a, int32_t b) {
                return res.logits[a] > res.logits[b];
            });
    res.top_tokens.assign(order.begin(), order.begin() + keep);
    res.top_logits.reserve(keep);
    for (int32_t id : res.top_tokens) {
        res.top_logits.push_back(res.logits[id]);
    }

    if (decode_n > 0) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        std::vector<float> cur_logits = res.logits;
        res.decode_tokens.reserve(decode_n);

        for (int i = 0; i < decode_n; ++i) {
            const int32_t id = greedy_token_from_logits(cur_logits);
            if (id < 0) {
                res.error = "missing decode logits";
                return res;
            }

            res.decode_tokens.push_back((llama_token) id);
            if (llama_vocab_is_eog(vocab, (llama_token) id)) {
                break;
            }

            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, (llama_token) id, prefix + i, { 0 }, true);
            const int ret = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            if (ret != 0) {
                res.ret = ret;
                res.error = "llama_decode failed during greedy decode";
                return res;
            }

            const float * next_logits = llama_get_logits_ith(ctx.get(), -1);
            if (next_logits == nullptr) {
                res.error = "missing logits during greedy decode";
                return res;
            }
            cur_logits.assign(next_logits, next_logits + n_vocab);
        }

        if (!res.decode_tokens.empty()) {
            res.decode_hash = fnv1a64(
                    reinterpret_cast<const uint8_t *>(res.decode_tokens.data()),
                    sizeof(llama_token) * res.decode_tokens.size());
            res.decode_text = common_detokenize(vocab, res.decode_tokens, false);
        }
    }

    if (collect_state) {
        res.state_size = llama_state_seq_get_size_ext(ctx.get(), 0, 0);
        if (res.state_size == 0) {
            res.error = "state size is zero";
            return res;
        }

        res.state.resize(res.state_size);
        const size_t got = llama_state_seq_get_data_ext(ctx.get(), res.state.data(), res.state.size(), 0, 0);
        if (got != res.state_size) {
            res.error = "state serialization size mismatch";
            res.state_size = got;
            return res;
        }

        res.state_hash = fnv1a64(res.state.data(), res.state.size());
    }

    res.ok = true;
    return res;
}

static std::string join_tokens(const std::vector<int32_t> & toks) {
    std::string out;
    for (size_t i = 0; i < toks.size(); ++i) {
        if (i > 0) {
            out += ':';
        }
        out += std::to_string(toks[i]);
    }
    return out;
}

static std::string join_llama_tokens(const std::vector<llama_token> & toks) {
    std::string out;
    for (size_t i = 0; i < toks.size(); ++i) {
        if (i > 0) {
            out += ':';
        }
        out += std::to_string(toks[i]);
    }
    return out;
}

static logits_diff compare_logits(const probe_result & base, const probe_result & forced) {
    logits_diff out;
    if (!base.ok || !forced.ok || base.logits.empty() || base.logits.size() != forced.logits.size()) {
        return out;
    }
    out.ok = true;
    out.exact = base.logits_hash == forced.logits_hash;

    long double sum_sq = 0.0;
    for (size_t i = 0; i < base.logits.size(); ++i) {
        if (!std::isfinite(base.logits[i]) || !std::isfinite(forced.logits[i])) {
            sum_sq = std::numeric_limits<long double>::quiet_NaN();
            continue;
        }
        const double d = (double) forced.logits[i] - (double) base.logits[i];
        const double ad = std::fabs(d);
        if (ad > out.max_abs) {
            out.max_abs = ad;
            out.max_abs_id = (int32_t) i;
        }
        sum_sq += (long double) d * (long double) d;
    }
    out.rms = std::sqrt((double) (sum_sq / (long double) base.logits.size()));

    if (!base.top_tokens.empty()) {
        const int32_t id = base.top_tokens.front();
        out.base_top1_delta = (double) forced.logits[id] - (double) base.logits[id];
    }
    if (base.top_logits.size() >= 2) {
        out.base_top1_margin = (double) base.top_logits[0] - (double) base.top_logits[1];
    }
    if (forced.top_logits.size() >= 2) {
        out.forced_top1_margin = (double) forced.top_logits[0] - (double) forced.top_logits[1];
    }
    return out;
}

static std::vector<char *> build_argv(std::vector<std::string> & args) {
    std::vector<char *> out;
    out.reserve(args.size());
    for (std::string & s : args) {
        out.push_back(s.data());
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_ALL, "C");

    probe_config cfg;
    std::vector<std::string> common_args;
    common_args.emplace_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", name);
                usage(argv[0]);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--state-probe-prefix") {
            int32_t n = 0;
            const std::string v = need_value("--state-probe-prefix");
            if (!parse_i32(v, n)) {
                fprintf(stderr, "invalid --state-probe-prefix: %s\n", v.c_str());
                return 1;
            }
            cfg.prefixes.push_back(n);
        } else if (arg == "--state-probe-range") {
            const std::string v = need_value("--state-probe-range");
            if (!add_range(v, cfg.prefixes)) {
                fprintf(stderr, "invalid --state-probe-range: %s\n", v.c_str());
                return 1;
            }
        } else if (arg == "--state-probe-csv") {
            cfg.csv_path = need_value("--state-probe-csv");
        } else if (arg == "--state-probe-topk") {
            const std::string v = need_value("--state-probe-topk");
            int32_t n = 0;
            if (!parse_i32(v, n) || n <= 0) {
                fprintf(stderr, "invalid --state-probe-topk: %s\n", v.c_str());
                return 1;
            }
            cfg.topk = n;
        } else if (arg == "--state-probe-decode") {
            const std::string v = need_value("--state-probe-decode");
            int32_t n = 0;
            if (!parse_i32(v, n)) {
                fprintf(stderr, "invalid --state-probe-decode: %s\n", v.c_str());
                return 1;
            }
            cfg.decode = n;
        } else if (arg == "--state-probe-save-dir") {
            cfg.save_dir = need_value("--state-probe-save-dir");
        } else if (arg == "--state-probe-save-all") {
            cfg.save_mismatches = false;
        } else if (arg == "--state-probe-only") {
            cfg.only = need_value("--state-probe-only");
            if (cfg.only != "baseline" && cfg.only != "forced") {
                fprintf(stderr, "invalid --state-probe-only: %s\n", cfg.only.c_str());
                return 1;
            }
        } else if (arg == "--state-probe-stop-on-mismatch") {
            cfg.stop_on_mismatch = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            common_args.push_back(arg);
        }
    }

    if (cfg.prefixes.empty()) {
        fprintf(stderr, "no probe prefixes supplied\n");
        usage(argv[0]);
        return 1;
    }
    std::sort(cfg.prefixes.begin(), cfg.prefixes.end());
    cfg.prefixes.erase(std::unique(cfg.prefixes.begin(), cfg.prefixes.end()), cfg.prefixes.end());

    common_params params;
    auto common_argv = build_argv(common_args);
    if (!common_params_parse((int) common_argv.size(), common_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.prompt.empty()) {
        fprintf(stderr, "state probe requires --prompt or --file\n");
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "state probe requires --model\n");
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * raw_model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(raw_model, llama_model_free);
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", params.model.path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true, true);
    fprintf(stderr, "state_probe: prompt_tokens=%zu prefill_batch=%d topk=%d slot_bank=%d ctx=%d\n",
            tokens.size(), params.moe_prefill_batch, params.moe_topk_override,
            params.moe_slot_bank, params.n_ctx);

    std::ofstream csv;
    if (!cfg.csv_path.empty()) {
        std::filesystem::create_directories(std::filesystem::path(cfg.csv_path).parent_path());
        csv.open(cfg.csv_path);
        csv << "prefix,baseline_status,baseline_top1,baseline_logits_hash,baseline_logits_finite,baseline_logits_nan,baseline_logits_inf,baseline_state_size,baseline_state_hash,"
               "forced_status,forced_top1,forced_logits_hash,forced_logits_finite,forced_logits_nan,forced_logits_inf,forced_state_size,forced_state_hash,"
               "top1_match,topk_match,logits_exact,max_abs,max_abs_id,rms,base_top1_margin,forced_top1_margin,base_top1_delta\n";
    }

    int first_bad = -1;
    for (const int32_t prefix : cfg.prefixes) {
        if (prefix <= 0 || prefix > (int32_t) tokens.size()) {
            fprintf(stderr, "prefix %d out of range for prompt_tokens=%zu\n", prefix, tokens.size());
            return 1;
        }

        const bool run_baseline = cfg.only != "forced";
        const bool run_forced = cfg.only != "baseline";

        probe_result base;
        probe_result forced;
        if (run_baseline) {
            fprintf(stderr, "state_probe: prefix=%d baseline\n", prefix);
            base = run_one(model.get(), params, tokens, prefix, false, cfg.topk, cfg.decode, !cfg.save_dir.empty());
        }
        if (run_forced) {
            fprintf(stderr, "state_probe: prefix=%d forced\n", prefix);
            forced = run_one(model.get(), params, tokens, prefix, true, cfg.topk, cfg.decode, !cfg.save_dir.empty());
        }

        if (!run_baseline || !run_forced) {
            const probe_result & one = run_forced ? forced : base;
            printf("prefix=%d %s=%s top1=%d logits=%016" PRIx64 " finite=%lld nan=%lld inf=%lld top=%s state_size=%zu state_hash=%016" PRIx64
                   " decode_hash=%016" PRIx64 " decode=%s\n",
                   prefix,
                   run_forced ? "forced" : "baseline",
                   one.ok ? "ok" : one.error.c_str(),
                   one.top_tokens.empty() ? -1 : one.top_tokens.front(),
                   one.logits_hash,
                   (long long) one.logits_finite,
                   (long long) one.logits_nan,
                   (long long) one.logits_inf,
                   join_tokens(one.top_tokens).c_str(),
                   one.state_size,
                   one.state_hash,
                   one.decode_hash,
                   join_llama_tokens(one.decode_tokens).c_str());
            fflush(stdout);
            continue;
        }

        const bool top1_match = base.ok && forced.ok &&
            !base.top_tokens.empty() && !forced.top_tokens.empty() &&
            base.top_tokens.front() == forced.top_tokens.front();
        const bool topk_match = base.ok && forced.ok &&
            base.top_tokens == forced.top_tokens;
        const bool decode_match = base.ok && forced.ok &&
            base.decode_tokens == forced.decode_tokens;
        const logits_diff diff = compare_logits(base, forced);
        const bool state_match = base.ok && forced.ok &&
            base.state_size == forced.state_size &&
            base.state_hash == forced.state_hash;
        const bool any_mismatch = !top1_match || !topk_match || !decode_match ||
            !state_match || !(diff.ok && diff.exact);

        printf("prefix=%d baseline=%s top1=%d logits=%016" PRIx64 " finite=%lld nan=%lld inf=%lld top=%s state_size=%zu state_hash=%016" PRIx64
               " decode_hash=%016" PRIx64 " decode=%s"
               " forced=%s top1=%d logits=%016" PRIx64 " finite=%lld nan=%lld inf=%lld top=%s state_size=%zu state_hash=%016" PRIx64
               " decode_hash=%016" PRIx64 " decode=%s"
               " top1_match=%s topk_match=%s decode_match=%s logits_exact=%s max_abs=%.9g max_abs_id=%d rms=%.9g"
               " base_top1_margin=%.9g forced_top1_margin=%.9g base_top1_delta=%.9g\n",
               prefix,
               base.ok ? "ok" : base.error.c_str(),
               base.top_tokens.empty() ? -1 : base.top_tokens.front(),
               base.logits_hash,
               (long long) base.logits_finite,
               (long long) base.logits_nan,
               (long long) base.logits_inf,
               join_tokens(base.top_tokens).c_str(),
               base.state_size, base.state_hash,
               base.decode_hash,
               join_llama_tokens(base.decode_tokens).c_str(),
               forced.ok ? "ok" : forced.error.c_str(),
               forced.top_tokens.empty() ? -1 : forced.top_tokens.front(),
               forced.logits_hash,
               (long long) forced.logits_finite,
               (long long) forced.logits_nan,
               (long long) forced.logits_inf,
               join_tokens(forced.top_tokens).c_str(),
               forced.state_size, forced.state_hash,
               forced.decode_hash,
               join_llama_tokens(forced.decode_tokens).c_str(),
               top1_match ? "yes" : "no",
               topk_match ? "yes" : "no",
               decode_match ? "yes" : "no",
               diff.ok && diff.exact ? "yes" : "no",
               diff.ok ? diff.max_abs : std::numeric_limits<double>::quiet_NaN(),
               diff.max_abs_id,
               diff.ok ? diff.rms : std::numeric_limits<double>::quiet_NaN(),
               diff.ok ? diff.base_top1_margin : std::numeric_limits<double>::quiet_NaN(),
               diff.ok ? diff.forced_top1_margin : std::numeric_limits<double>::quiet_NaN(),
               diff.ok ? diff.base_top1_delta : std::numeric_limits<double>::quiet_NaN());
        fflush(stdout);

        if (cfg.decode > 0) {
            fprintf(stderr, "state_probe_decode baseline=%s\n", base.decode_text.c_str());
            fprintf(stderr, "state_probe_decode forced=%s\n", forced.decode_text.c_str());
        }

        if (csv) {
            csv << prefix << ','
                << (base.ok ? "ok" : base.error) << ','
                << (base.top_tokens.empty() ? -1 : base.top_tokens.front()) << ','
                << std::hex << base.logits_hash << std::dec << ','
                << base.logits_finite << ','
                << base.logits_nan << ','
                << base.logits_inf << ','
                << base.state_size << ','
                << std::hex << base.state_hash << std::dec << ','
                << (forced.ok ? "ok" : forced.error) << ','
                << (forced.top_tokens.empty() ? -1 : forced.top_tokens.front()) << ','
                << std::hex << forced.logits_hash << std::dec << ','
                << forced.logits_finite << ','
                << forced.logits_nan << ','
                << forced.logits_inf << ','
                << forced.state_size << ','
                << std::hex << forced.state_hash << std::dec << ','
                << (top1_match ? "yes" : "no") << ','
                << (topk_match ? "yes" : "no") << ','
                << (diff.ok && diff.exact ? "yes" : "no") << ','
                << (diff.ok ? diff.max_abs : std::numeric_limits<double>::quiet_NaN()) << ','
                << diff.max_abs_id << ','
                << (diff.ok ? diff.rms : std::numeric_limits<double>::quiet_NaN()) << ','
                << (diff.ok ? diff.base_top1_margin : std::numeric_limits<double>::quiet_NaN()) << ','
                << (diff.ok ? diff.forced_top1_margin : std::numeric_limits<double>::quiet_NaN()) << ','
                << (diff.ok ? diff.base_top1_delta : std::numeric_limits<double>::quiet_NaN()) << '\n';
            csv.flush();
        }

        const bool should_save = !cfg.save_dir.empty() && (any_mismatch || !cfg.save_mismatches);
        if (should_save) {
            const std::filesystem::path dir(cfg.save_dir);
            const std::string pfx = "prefix_" + std::to_string(prefix);
            if (base.ok) {
                write_file(dir / (pfx + "_baseline.state.bin"), base.state);
            }
            if (forced.ok) {
                write_file(dir / (pfx + "_forced.state.bin"), forced.state);
            }
            save_tokens(dir / (pfx + ".tokens.txt"), tokens, prefix);
        }

        if (any_mismatch && first_bad < 0) {
            first_bad = prefix;
            if (cfg.stop_on_mismatch) {
                break;
            }
        }
    }

    fprintf(stderr, "state_probe: first_mismatch=%d\n", first_bad);
    return first_bad < 0 ? 0 : 2;
}
