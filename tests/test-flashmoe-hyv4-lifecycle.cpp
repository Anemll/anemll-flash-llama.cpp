#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// Opt-in, real-model integration test. Never download a model or run a large
// allocation from the default CTest suite; pass the dense/sidecar arguments.
enum class stop_kind { none, exception, callback, backend };
struct stop_state {
    stop_kind kind = stop_kind::none;
    bool fired = false;
    int completed_joins = 0;
    std::atomic<bool> abort{false};
    std::atomic<int> abort_checks{0};
};

static bool eval(ggml_tensor * tensor, bool ask, void * data) {
    auto & state = *static_cast<stop_state *>(data);
    const bool first_router = std::strcmp(ggml_get_name(tensor), "ffn_moe_topk-1") == 0;
    if (ask) {
        return first_router;
    }
    if (std::strncmp(ggml_get_name(tensor), "ffn_moe_io_join-", 16) == 0) {
        ++state.completed_joins;
    }
    if (!first_router || state.fired || state.kind == stop_kind::none) {
        return true;
    }
    state.fired = true;
    if (state.kind == stop_kind::exception) {
        throw std::runtime_error("intentional HY4 lifecycle test interruption after top-k");
    }
    if (state.kind == stop_kind::backend) {
        state.abort.store(true);
    }
    return state.kind != stop_kind::callback;
}

static void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    if (argc == 1) {
        std::fprintf(stderr, "SKIP: supply a local HY4 dense GGUF and --moe-sidecar for the native lifecycle test\n");
        return 77;
    }
    try {
        // Exercise HY4's default overlap instead of forcing it on and hiding a
        // default-selection regression. Explicit environment overrides are honored.
        common_params params;
        params.moe_mode = "slot-bank";
        params.moe_slot_bank = 8;
        params.moe_topk_override = 8;
        params.moe_cache_io_split = 4;
        params.slot8 = true;
        params.fp16_head = true;
        params.n_ctx = 128;
        params.n_batch = params.n_ubatch = 1;
        params.warmup = false;
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
            return 1;
        }
        if (params.n_gpu_layers < 0) {
            params.n_gpu_layers = 999;
        }
        stop_state state;
        params.cb_eval = eval;
        params.cb_eval_user_data = &state;
        common_init();
        auto initialized = common_init_from_params(params);
        auto * ctx = initialized->context();
        auto * model = initialized->model();
        check(ctx != nullptr && model != nullptr, "model/context initialization failed");
        char architecture[64] = {};
        llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture));
        check(std::strcmp(architecture, "hyv4") == 0, "this integration test requires HY4");
        llama_set_abort_callback(ctx, [](void * data) {
            auto & stop = *static_cast<stop_state *>(data);
            ++stop.abort_checks;
            return stop.abort.load();
        }, &state);
        const auto prompt = common_tokenize(ctx, "Hello", false, true);
        const auto alternate = common_tokenize(ctx, "World", false, true);
        check(!prompt.empty() && !alternate.empty(), "test tokenization failed");
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const auto decode_one = [&](llama_token token) {
            llama_memory_clear(llama_get_memory(ctx), true);
            return llama_decode(ctx, llama_batch_get_one(&token, 1));
        };
        // Cold bank guarantees the first interrupted router has pending reads.
        state.kind = stop_kind::exception;
        check(decode_one(alternate.front()) != 0 && state.fired, "exception did not interrupt the cold-bank graph");
        state.kind = stop_kind::none;
        check(decode_one(prompt.front()) == 0, "context reuse after cold-bank interruption failed");
        check(state.completed_joins > 0, "HY4 shared-I/O overlap did not actually execute; check GPU bank and direct-read settings");
        std::vector<float> expected(llama_get_logits(ctx), llama_get_logits(ctx) + n_vocab);
        for (const auto kind : {stop_kind::callback, stop_kind::backend, stop_kind::exception}) {
            state.kind = kind;
            state.fired = false;
            state.abort.store(false);
            state.abort_checks.store(0);
            const int status = decode_one(alternate.front());
            check(state.fired, "test did not reach the selected router");
            if (kind == stop_kind::backend && state.abort_checks.load() == 0) {
                // Metal's tiny callback-split graphs may never reach its
                // command-buffer abort polling point. Do not claim coverage.
                std::fprintf(stderr, "NOT EXERCISED: backend did not poll abort callback for these small graph segments\n");
            } else if (kind != stop_kind::callback) {
                check(status != 0, "backend/exception interruption unexpectedly succeeded");
            }
            state.kind = stop_kind::none;
            state.abort.store(false);
            check(decode_one(prompt.front()) == 0, "context reuse after interrupted graph failed");
            check(std::memcmp(expected.data(), llama_get_logits(ctx), expected.size() * sizeof(float)) == 0,
                    "full-vocabulary logits changed after interruption/context reuse");
        }
        std::fprintf(stderr, "PASS: cold-bank exception, eval-callback stop, repeat exception; reused-context full-vocabulary logits bit-identical (backend-abort coverage reported separately)\n");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    return 0;
}
