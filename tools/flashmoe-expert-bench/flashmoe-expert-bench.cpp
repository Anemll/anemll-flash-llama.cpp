#include "llama.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

enum class bench_component {
    up,
    down,
    full,
};

struct bench_config {
    std::string model_path;
    std::string sidecar_path;
    int32_t layer = -1;
    int32_t expert = 0;
    int32_t tokens = 128;
    int32_t warmup = 2;
    int32_t iters = 10;
    int32_t seed = 123;
    int32_t n_gpu_layers = -1;
    int32_t nk = 32;
    std::string walk = "legacy";
    bench_component component = bench_component::full;
    std::string dump_output_path;
    std::string compare_output_path;
};

struct bench_stats {
    int32_t layer = -1;
    int64_t rows_in = 0;
    int64_t rows_out = 0;
    ggml_type weight_type = GGML_TYPE_COUNT;
    double total_ms = 0.0;
    double ms_per_iter = 0.0;
    double tokens_per_second = 0.0;
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
};

struct resolved_model_input {
    std::string model_path;
    std::string sidecar_path;
};

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
            "Usage: %s -m MODEL [--component up|down|full] [--layer N] [--expert N]\n"
            "          [--tokens N] [--warmup N] [--iters N] [--seed N]\n"
            "          [--nk N] [--walk legacy|regular|morton]\n"
            "          [--dump-output FILE] [--compare-output FILE] [--n-gpu-layers N]\n"
            "          [--sidecar PATH]\n"
            "\n"
            "MODEL may be a full GGUF, a Flash-MoE package directory containing model-dense.gguf + sidecar/, or flashmoe-package.json.\n",
            argv0);
}

static bool parse_int_arg(const char * arg, int32_t & out) {
    char * end = nullptr;
    const long value = std::strtol(arg, &end, 10);
    if (end == arg || *end != '\0' || value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    out = int32_t(value);
    return true;
}

static bench_component parse_component(const std::string & value) {
    if (value == "up") {
        return bench_component::up;
    }
    if (value == "down") {
        return bench_component::down;
    }
    if (value == "full") {
        return bench_component::full;
    }
    throw std::runtime_error("unsupported component: " + value);
}

static bench_config parse_args(int argc, char ** argv) {
    bench_config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model") {
            cfg.model_path = require_value(arg.c_str());
        } else if (arg == "--component") {
            cfg.component = parse_component(require_value(arg.c_str()));
        } else if (arg == "--layer") {
            if (!parse_int_arg(require_value(arg.c_str()), cfg.layer)) {
                throw std::runtime_error("invalid --layer value");
            }
        } else if (arg == "--expert") {
            if (!parse_int_arg(require_value(arg.c_str()), cfg.expert)) {
                throw std::runtime_error("invalid --expert value");
            }
        } else if (arg == "--tokens") {
            if (!parse_int_arg(require_value(arg.c_str()), cfg.tokens) || cfg.tokens <= 0) {
                throw std::runtime_error("invalid --tokens value");
            }
        } else if (arg == "--warmup") {
            if (!parse_int_arg(require_value(arg.c_str()), cfg.warmup) || cfg.warmup < 0) {
                throw std::runtime_error("invalid --warmup value");
            }
        } else if (arg == "--iters") {
            if (!parse_int_arg(require_value(arg.c_str()), cfg.iters) || cfg.iters <= 0) {
                throw std::runtime_error("invalid --iters value");
            }
        } else if (arg == "--seed") {
            if (!parse_int_arg(require_value(arg.c_str()), cfg.seed)) {
                throw std::runtime_error("invalid --seed value");
            }
        } else if (arg == "--nk") {
            if (!parse_int_arg(require_value(arg.c_str()), cfg.nk) || cfg.nk < 32 || cfg.nk > 128 || (cfg.nk % 16) != 0) {
                throw std::runtime_error("invalid --nk value");
            }
        } else if (arg == "--walk") {
            cfg.walk = require_value(arg.c_str());
            if (cfg.walk != "legacy" && cfg.walk != "regular" && cfg.walk != "morton") {
                throw std::runtime_error("invalid --walk value");
            }
        } else if (arg == "--dump-output") {
            cfg.dump_output_path = require_value(arg.c_str());
        } else if (arg == "--compare-output") {
            cfg.compare_output_path = require_value(arg.c_str());
        } else if (arg == "--sidecar") {
            cfg.sidecar_path = require_value(arg.c_str());
        } else if (arg == "--n-gpu-layers" || arg == "-ngl") {
            if (!parse_int_arg(require_value(arg.c_str()), cfg.n_gpu_layers)) {
                throw std::runtime_error("invalid --n-gpu-layers value");
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (cfg.model_path.empty()) {
        throw std::runtime_error("missing required --model path");
    }

    return cfg;
}

static bool is_split_moe_layer(const llama_layer & layer) {
    return layer.ffn_gate_exps != nullptr &&
           layer.ffn_up_exps   != nullptr &&
           layer.ffn_down_exps != nullptr;
}

static resolved_model_input resolve_model_input(const bench_config & cfg) {
    namespace fs = std::filesystem;

    const fs::path raw_input = fs::absolute(fs::path(cfg.model_path));
    resolved_model_input resolved = {
        /*.model_path   =*/ raw_input.string(),
        /*.sidecar_path =*/ cfg.sidecar_path,
    };

    auto maybe_set_package_sidecar = [&](const fs::path & package_dir) {
        const fs::path dense_model = package_dir / "model-dense.gguf";
        const fs::path sidecar_dir = package_dir / "sidecar";
        if (fs::exists(dense_model) && fs::exists(sidecar_dir)) {
            resolved.model_path = dense_model.string();
            if (resolved.sidecar_path.empty()) {
                resolved.sidecar_path = sidecar_dir.string();
            }
            return true;
        }
        return false;
    };

    if (fs::is_directory(raw_input)) {
        if (!maybe_set_package_sidecar(raw_input)) {
            throw std::runtime_error("directory model input must contain model-dense.gguf and sidecar/: " + raw_input.string());
        }
    } else if (raw_input.filename() == "flashmoe-package.json") {
        const fs::path package_dir = raw_input.parent_path();
        if (!maybe_set_package_sidecar(package_dir)) {
            throw std::runtime_error("flashmoe-package.json parent must contain model-dense.gguf and sidecar/: " + package_dir.string());
        }
    } else if (resolved.sidecar_path.empty()) {
        const fs::path sibling_sidecar = raw_input.parent_path() / "sidecar";
        if (raw_input.filename() == "model-dense.gguf" && fs::exists(sibling_sidecar)) {
            resolved.sidecar_path = sibling_sidecar.string();
        }
    }

    if (!resolved.sidecar_path.empty()) {
        resolved.sidecar_path = fs::absolute(fs::path(resolved.sidecar_path)).string();
    }

    return resolved;
}

static int32_t find_default_split_layer(const llama_model & model) {
    for (int32_t il = 0; il < int32_t(model.layers.size()); ++il) {
        if (is_split_moe_layer(model.layers[il])) {
            return il;
        }
    }
    return -1;
}

static size_t expert_slice_bytes(const ggml_tensor * tensor) {
    if (tensor->ne[2] > 1) {
        return tensor->nb[2];
    }
    if (tensor->ne[3] > 1) {
        return tensor->nb[3];
    }
    return ggml_nbytes(tensor);
}

static int64_t expert_count(const ggml_tensor * tensor) {
    if (tensor->ne[2] > 1) {
        return tensor->ne[2];
    }
    if (tensor->ne[3] > 1) {
        return tensor->ne[3];
    }
    return 1;
}

static void read_expert_slice(const ggml_tensor * tensor, int32_t expert, std::vector<uint8_t> & out) {
    const size_t bytes = expert_slice_bytes(tensor);
    const int64_t n_expert = expert_count(tensor);
    if (expert < 0 || expert >= n_expert) {
        throw std::runtime_error("expert index out of range for tensor " + std::string(tensor->name));
    }

    out.resize(bytes);
    ggml_backend_tensor_get(const_cast<ggml_tensor *>(tensor), out.data(), size_t(expert) * bytes, bytes);
}

static void read_expert_slice_from_sidecar(
        const llama_model & model,
        const ggml_tensor * tensor,
        int32_t expert,
        std::vector<uint8_t> & out) {
    const auto * entry = model.flash_moe_sidecar_entry_for(tensor->name);
    if (entry == nullptr || entry->repacked_path.empty()) {
        read_expert_slice(tensor, expert, out);
        return;
    }

    const int64_t n_expert = expert_count(tensor);
    if (expert < 0 || expert >= n_expert) {
        throw std::runtime_error("expert index out of range for sidecar tensor " + std::string(tensor->name));
    }

    const size_t bytes = entry->bytes_per_expert;
    if (bytes == 0) {
        throw std::runtime_error("invalid sidecar bytes_per_expert for tensor " + std::string(tensor->name));
    }

    out.resize(bytes);

    std::ifstream in(entry->repacked_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open sidecar file: " + entry->repacked_path);
    }

    const size_t offset = entry->repacked_offset + size_t(expert) * bytes;
    in.seekg(std::streamoff(offset), std::ios::beg);
    if (!in) {
        throw std::runtime_error("failed to seek sidecar file: " + entry->repacked_path);
    }

    in.read(reinterpret_cast<char *>(out.data()), std::streamsize(bytes));
    if (!in) {
        throw std::runtime_error("failed to read expert bytes from sidecar file: " + entry->repacked_path);
    }
}

static ggml_backend_t init_exec_backend(const ggml_tensor * reference) {
    ggml_backend_dev_t dev = nullptr;
    if (reference != nullptr && reference->buffer != nullptr) {
        dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(reference->buffer));
    }
    if (dev == nullptr) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    }
    if (dev != nullptr) {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend != nullptr) {
            return backend;
        }
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        throw std::runtime_error("failed to initialize execution backend");
    }
    ggml_backend_cpu_set_n_threads(backend, std::max(1u, std::thread::hardware_concurrency()));
    return backend;
}

static void fill_input(std::vector<float> & data, int32_t seed) {
    for (size_t i = 0; i < data.size(); ++i) {
        const double x = double(i + 1 + seed);
        data[i] = float(0.5 * std::sin(x * 0.013) + 0.35 * std::cos(x * 0.007));
    }
}

static std::vector<float> read_reference_file(const std::string & path, size_t expected_count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open compare file: " + path);
    }
    in.seekg(0, std::ios::end);
    const size_t bytes = size_t(in.tellg());
    in.seekg(0, std::ios::beg);

    if (bytes != expected_count * sizeof(float)) {
        throw std::runtime_error("compare file size mismatch");
    }

    std::vector<float> ref(expected_count);
    in.read(reinterpret_cast<char *>(ref.data()), std::streamsize(bytes));
    if (!in) {
        throw std::runtime_error("failed to read compare file");
    }
    return ref;
}

static void write_output_file(const std::string & path, const std::vector<float> & data) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    out.write(reinterpret_cast<const char *>(data.data()), std::streamsize(data.size() * sizeof(float)));
    if (!out) {
        throw std::runtime_error("failed to write output file");
    }
}

static bench_stats run_bench(const bench_config & cfg, const llama_model & model) {
    const int32_t layer_index = cfg.layer >= 0 ? cfg.layer : find_default_split_layer(model);
    if (layer_index < 0 || layer_index >= int32_t(model.layers.size())) {
        throw std::runtime_error("unable to find a valid split-MoE layer");
    }

    const llama_layer & layer = model.layers[layer_index];
    if (!is_split_moe_layer(layer)) {
        throw std::runtime_error("selected layer is not a split-MoE layer");
    }

    const ggml_tensor * gate_src = layer.ffn_gate_exps;
    const ggml_tensor * up_src   = layer.ffn_up_exps;
    const ggml_tensor * down_src = layer.ffn_down_exps;

    const int64_t n_expert = expert_count(up_src);
    if (cfg.expert < 0 || cfg.expert >= n_expert) {
        throw std::runtime_error("expert index out of range");
    }

    std::vector<uint8_t> gate_bytes;
    std::vector<uint8_t> up_bytes;
    std::vector<uint8_t> down_bytes;
    if (cfg.component == bench_component::full) {
        read_expert_slice_from_sidecar(model, gate_src, cfg.expert, gate_bytes);
    }
    if (cfg.component == bench_component::up || cfg.component == bench_component::full) {
        read_expert_slice_from_sidecar(model, up_src, cfg.expert, up_bytes);
    }
    if (cfg.component == bench_component::down || cfg.component == bench_component::full) {
        read_expert_slice_from_sidecar(model, down_src, cfg.expert, down_bytes);
    }

    ggml_backend_t backend = init_exec_backend(up_src);

    ggml_init_params init_params = {
        /*.mem_size   =*/ 1024 * 1024 + 64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(init_params);
    if (ctx == nullptr) {
        ggml_backend_free(backend);
        throw std::runtime_error("failed to allocate ggml context");
    }

    ggml_tensor * input = nullptr;
    ggml_tensor * gate_w = nullptr;
    ggml_tensor * up_w = nullptr;
    ggml_tensor * down_w = nullptr;
    ggml_tensor * out = nullptr;
    ggml_type weight_type = GGML_TYPE_COUNT;

    switch (cfg.component) {
        case bench_component::up:
            input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, up_src->ne[0], cfg.tokens);
            up_w  = ggml_new_tensor_2d(ctx, up_src->type, up_src->ne[0], up_src->ne[1]);
            out   = ggml_cont(ctx, ggml_mul_mat(ctx, up_w, input));
            weight_type = up_src->type;
            break;
        case bench_component::down:
            input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, down_src->ne[0], cfg.tokens);
            down_w = ggml_new_tensor_2d(ctx, down_src->type, down_src->ne[0], down_src->ne[1]);
            out = ggml_cont(ctx, ggml_mul_mat(ctx, down_w, input));
            weight_type = down_src->type;
            break;
        case bench_component::full:
            input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, gate_src->ne[0], cfg.tokens);
            gate_w = ggml_new_tensor_2d(ctx, gate_src->type, gate_src->ne[0], gate_src->ne[1]);
            up_w   = ggml_new_tensor_2d(ctx, up_src->type, up_src->ne[0], up_src->ne[1]);
            down_w = ggml_new_tensor_2d(ctx, down_src->type, down_src->ne[0], down_src->ne[1]);
            out = ggml_cont(ctx, ggml_mul_mat(ctx, down_w, ggml_swiglu_split(ctx, ggml_mul_mat(ctx, gate_w, input), ggml_mul_mat(ctx, up_w, input))));
            weight_type = up_src->type;
            break;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        throw std::runtime_error("failed to allocate backend tensors");
    }

    std::vector<float> input_data(size_t(input->ne[0]) * size_t(input->ne[1]));
    fill_input(input_data, cfg.seed);
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    if (gate_w != nullptr) {
        ggml_backend_tensor_set(gate_w, gate_bytes.data(), 0, gate_bytes.size());
    }
    if (up_w != nullptr) {
        ggml_backend_tensor_set(up_w, up_bytes.data(), 0, up_bytes.size());
    }
    if (down_w != nullptr) {
        ggml_backend_tensor_set(down_w, down_bytes.data(), 0, down_bytes.size());
    }

    for (int i = 0; i < cfg.warmup; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < cfg.iters; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const int64_t t_end = ggml_time_us();

    std::vector<float> output(size_t(out->ne[0]) * size_t(out->ne[1]));
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    bench_stats stats;
    stats.layer = layer_index;
    stats.rows_in = input->ne[0];
    stats.rows_out = out->ne[0];
    stats.weight_type = weight_type;
    stats.total_ms = double(t_end - t_start) / 1000.0;
    stats.ms_per_iter = stats.total_ms / double(cfg.iters);
    stats.tokens_per_second = (double(cfg.tokens) * double(cfg.iters)) / (stats.total_ms / 1000.0);

    if (!cfg.compare_output_path.empty()) {
        const std::vector<float> ref = read_reference_file(cfg.compare_output_path, output.size());
        double sum_abs = 0.0;
        for (size_t i = 0; i < output.size(); ++i) {
            const float diff = std::fabs(output[i] - ref[i]);
            stats.max_abs = std::max(stats.max_abs, diff);
            sum_abs += diff;
        }
        stats.mean_abs = float(sum_abs / double(output.size()));
    }

    if (!cfg.dump_output_path.empty()) {
        write_output_file(cfg.dump_output_path, output);
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return stats;
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    try {
        const bench_config cfg = parse_args(argc, argv);
        const resolved_model_input model_input = resolve_model_input(cfg);

        std::string nk = std::to_string(cfg.nk);
        setenv("GGML_METAL_MUL_MM_NK", nk.c_str(), 1);
        setenv("GGML_METAL_MUL_MM_WALK", cfg.walk.c_str(), 1);

        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = cfg.n_gpu_layers;
        if (!model_input.sidecar_path.empty()) {
            model_params.moe_sidecar_path = model_input.sidecar_path.c_str();
            model_params.moe_mode = "slot-bank";
            model_params.moe_slot_bank = 1;
        }

        llama_model * model = llama_model_load_from_file(model_input.model_path.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("failed to load model");
        }

        const bench_stats stats = run_bench(cfg, *model);

        const char * component = cfg.component == bench_component::up ? "up" :
                                 cfg.component == bench_component::down ? "down" : "full";
        std::printf("component=%s layer=%d expert=%d tokens=%d nk=%d walk=%s rows_in=%lld rows_out=%lld weight_type=%s total_ms=%.3f ms_per_iter=%.3f tok_s=%.2f",
                component,
                stats.layer,
                cfg.expert,
                cfg.tokens,
                cfg.nk,
                cfg.walk.c_str(),
                (long long) stats.rows_in,
                (long long) stats.rows_out,
                ggml_type_name(stats.weight_type),
                stats.total_ms,
                stats.ms_per_iter,
                stats.tokens_per_second);
        if (!cfg.compare_output_path.empty()) {
            std::printf(" max_abs=%.9g mean_abs=%.9g", stats.max_abs, stats.mean_abs);
        }
        std::printf("\n");

        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & err) {
        std::fprintf(stderr, "error: %s\n", err.what());
        print_usage(argv[0]);
        return 1;
    }
}
