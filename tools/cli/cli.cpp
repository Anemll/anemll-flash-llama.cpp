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
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
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
    return base_name == "embd" ||
           base_name == "attn_norm" ||
           base_name == "q_pe" ||
           base_name == "k_pe" ||
           base_name == "kv_cmpr" ||
           base_name == "q_nope_absorbed_perm" ||
           base_name == "attn_out" ||
           base_name == "kqv_out" ||
           base_name == "ffn_inp" ||
           base_name == "ffn_norm" ||
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
        const std::string file_name = tensor_file_name(raw_name);
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
            "attn_norm",
            "q_pe",
            "k_pe",
            "kv_cmpr",
            "q_nope_absorbed_perm",
            "attn_out",
            "kqv_out",
            "ffn_inp",
            "ffn_norm",
            "ffn_shexp",
            "ffn_out",
            "l_out",
            "result_norm",
        });
        manifest["records"] = json::array();
        for (const auto & record : tensor_records_) {
            manifest["records"].push_back({
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

    std::string tensor_file_name(const std::string & name) const {
        char prefix[32];
        std::snprintf(prefix, sizeof(prefix), "%06d_", eval_index_);
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

static bool cli_oracle_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * oracle = static_cast<cli_oracle_dump *>(user_data);
    if (oracle == nullptr || !oracle->enabled()) {
        return true;
    }
    if (ask) {
        return oracle->wants_tensor(t);
    }
    return oracle->handle_tensor(t);
}

} // namespace

struct cli_context {
    server_context ctx_server;
    json messages = json::array();
    std::vector<raw_buffer> input_files;
    task_params defaults;
    bool verbose_prompt;
    bool show_prefill_timing;
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
        show_prefill_timing = params.show_timings && params.moe_prefill_layer_major;
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
        bool printed_prefill_timing = false;
        std::string raw_think_scan_buf;
        std::string deferred_content_buf;

        auto maybe_print_prefill_timing = [&](const result_timings & timings) {
            if (!show_prefill_timing || printed_prefill_timing || timings.prompt_n <= 0 || timings.prompt_per_second <= 0) {
                return;
            }

            console::set_display(DISPLAY_TYPE_INFO);
            console::log("[ Flash-MoE prefill: %.1f t/s | tokens: %d | %.1f ms ]\n",
                    timings.prompt_per_second,
                    timings.prompt_n,
                    timings.prompt_ms);
            console::set_display(DISPLAY_TYPE_RESET);
            printed_prefill_timing = true;
        };

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
                maybe_print_prefill_timing(out_timings);
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
                maybe_print_prefill_timing(out_timings);
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
        params.cb_eval = cli_oracle_cb_eval;
        params.cb_eval_user_data = &*oracle_dump;
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
        fprintf(stderr, "  prefill-stripe   = %s\n", params.moe_prefill_stripe.empty() ? "follow demand" : params.moe_prefill_stripe.c_str());
        fprintf(stderr, "  prefill-distribute = %s\n", params.moe_prefill_distribute.empty() ? "follow demand" : params.moe_prefill_distribute.c_str());
        fprintf(stderr, "  prefetch-stripe  = %s\n", params.moe_prefetch_stripe.empty() ? "off" : params.moe_prefetch_stripe.c_str());
        fprintf(stderr, "  prefetch-distribute = %s\n", params.moe_prefetch_distribute.empty() ? "off" : params.moe_prefetch_distribute.c_str());
        fprintf(stderr, "  prefill-layer-major = %s\n", params.moe_prefill_layer_major ? "on" : "off");
        fprintf(stderr, "  prefetch-temporal = %s\n", (params.moe_prefetch_temporal || params.moe_prefetch_temporal_sparse) ? "on" : "off");
        fprintf(stderr, "  prefetch-temporal-sparse = %s\n", params.moe_prefetch_temporal_sparse ? "on" : "off");
        fprintf(stderr, "  predict-prev-token = %s\n", params.moe_predict_prev_token ? "on" : "off");
        fprintf(stderr, "  predict-top1-prev = %s\n", params.moe_predict_top1_prev ? "on" : "off");
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

    // bump the log level to display timings
    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    llama_memory_breakdown_print(ctx_cli.ctx_server.get_llama_context());

    return 0;
}
