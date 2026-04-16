#include "llama-context.h"

#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"

#ifdef GGML_USE_METAL
#include "../ggml/src/ggml-metal/ggml-metal-ops.h"
#endif

#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unistd.h>

//
// llama_context
//

static bool flash_moe_backend_trace_enabled();
static void flash_moe_log_routed_backends(struct ggml_cgraph * gf, ggml_backend_sched_t sched);

static bool flash_moe_log_colors_enabled() {
    const char * value = std::getenv("LLAMA_LOG_COLORS");
    if (value != nullptr && value[0] != '\0') {
        if (std::strcmp(value, "off") == 0 || std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0) {
            return false;
        }
        if (std::strcmp(value, "on") == 0 || std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0) {
            return true;
        }
    }

    return isatty(fileno(stderr)) != 0;
}

class llama_flash_moe_slot_runtime : public llm_flash_moe_slot_runtime_i {
public:
    explicit llama_flash_moe_slot_runtime(const llama_model & model, bool perf_profile)
            : model(model),
              slot_count(model.flash_moe_slot_bank_size()),
              expert_count(model.hparams.n_expert),
              resident_bank_source(model.flash_moe_resident_source_enabled()),
              oracle_all_hit(model.flash_moe_oracle_all_hit_enabled()),
              oracle_prefetch(model.flash_moe_oracle_prefetch_enabled()),
              temporal_prefetch(model.flash_moe_temporal_prefetch_enabled()),
              temporal_prefetch_sparse(model.flash_moe_temporal_prefetch_sparse_enabled()),
              predict_prev_token(model.flash_moe_predict_prev_token_enabled()),
              predict_top1_prev(model.flash_moe_predict_top1_prev_enabled()),
              secondary_sidecar_enabled(model.flash_moe_secondary_sidecar_enabled()),
              demand_stripe_requested(model.flash_moe_demand_stripe_requested()),
              demand_stripe_enabled(model.flash_moe_demand_stripe_enabled()),
              prefetch_stripe_enabled(model.flash_moe_prefetch_stripe_enabled()),
              perf_profile(perf_profile),
              async_slot_upload(async_slot_upload_enabled()),
              parallel_slot_reads(parallel_slot_reads_enabled()),
              mixed_slot_buffer(mixed_slot_buffer_enabled()),
              cache_io_split(std::max<int32_t>(model.flash_moe_cache_io_split(), cache_io_split_from_env())),
              prefetch_cache_io_split(std::max<int32_t>(model.flash_moe_prefetch_cache_io_split(), cache_io_split_from_env())),
              demand_stripe_weights(model.flash_moe_demand_stripe_weights()),
              prefetch_stripe_weights(model.flash_moe_prefetch_stripe_weights()) {
        if (const char * path = model.flash_moe_trace_file(); oracle_all_hit || oracle_prefetch) {
            if (path == nullptr || path[0] == '\0') {
                throw std::runtime_error("Flash-MoE oracle replay mode requires a replay trace file");
            }
        } else if (const char * path = model.flash_moe_trace_file(); path != nullptr && path[0] != '\0') {
            trace_fp = std::fopen(path, "w");
            if (trace_fp == nullptr) {
                throw std::runtime_error(format("failed to open Flash-MoE trace file '%s'", path));
            }
            std::setvbuf(trace_fp, nullptr, _IOLBF, 0);
        }

        layers.resize(model.layers.size());
        native_slot_map_ud.resize(model.layers.size());

        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto & state = layers[il];
            state.n_slots = slot_count;
            state.slot_to_expert.assign(slot_count, -1);
            state.expert_to_slot.assign(expert_count, -1);
            state.slot_age.assign(slot_count, 0);
            state.slot_reserved_epoch.assign(slot_count, 0);
            state.request_seen_epoch.assign(expert_count, 0);
            state.request_slot.assign(expert_count, -1);
            state.temporal_prefetch_experts.reserve(std::max<int32_t>(1, model.moe_n_expert_used()));
            state.predicted_experts.reserve(std::max<int32_t>(1, model.moe_n_expert_used()));
            state.current_token_experts.reserve(std::max<int32_t>(1, model.moe_n_expert_used()));

            const auto & layer = model.layers[il];
            bind_tensor(layer.ffn_gate_up_exps, state.gate_up_tensor, state.gate_up_entry, state.prefetch_gate_up_entry, state.secondary_gate_up_entry, state.tertiary_gate_up_entry, state.enabled);
            bind_tensor(layer.ffn_gate_exps,    state.gate_tensor,    state.gate_entry,    state.prefetch_gate_entry,    state.secondary_gate_entry,    state.tertiary_gate_entry,    state.enabled);
            bind_tensor(layer.ffn_up_exps,      state.up_tensor,      state.up_entry,      state.prefetch_up_entry,      state.secondary_up_entry,      state.tertiary_up_entry,      state.enabled);
            bind_tensor(layer.ffn_down_exps,    state.down_tensor,    state.down_entry,    state.prefetch_down_entry,    state.secondary_down_entry,    state.tertiary_down_entry,    state.enabled);

            auto bind_mixed_field = [&](ggml_tensor * tensor, const llama_flash_moe_sidecar_entry * entry, routed_family family) {
                if (tensor == nullptr || entry == nullptr) {
                    return;
                }

                state.mixed_slot_fields.push_back({ tensor, entry, family, state.mixed_slot_bytes });
                state.mixed_slot_bytes += entry->bytes_per_expert;
            };

            bind_mixed_field(state.gate_up_tensor, state.gate_up_entry, routed_family::gate_up);
            bind_mixed_field(state.gate_tensor,    state.gate_entry,    routed_family::gate);
            bind_mixed_field(state.up_tensor,      state.up_entry,      routed_family::up);
            bind_mixed_field(state.down_tensor,    state.down_entry,    routed_family::down);

            auto bind_prefetch_mixed_field = [&](ggml_tensor * tensor, const llama_flash_moe_sidecar_entry * entry, routed_family family) {
                if (tensor == nullptr || entry == nullptr) {
                    return;
                }

                state.mixed_prefetch_slot_fields.push_back({ tensor, entry, family, state.mixed_prefetch_slot_bytes });
                state.mixed_prefetch_slot_bytes += entry->bytes_per_expert;
            };

            bind_prefetch_mixed_field(state.gate_up_tensor, state.prefetch_gate_up_entry, routed_family::gate_up);
            bind_prefetch_mixed_field(state.gate_tensor,    state.prefetch_gate_entry,    routed_family::gate);
            bind_prefetch_mixed_field(state.up_tensor,      state.prefetch_up_entry,      routed_family::up);
            bind_prefetch_mixed_field(state.down_tensor,    state.prefetch_down_entry,    routed_family::down);

            auto bind_secondary_mixed_field = [&](ggml_tensor * tensor, const llama_flash_moe_sidecar_entry * entry, routed_family family) {
                if (tensor == nullptr || entry == nullptr) {
                    return;
                }

                state.mixed_secondary_slot_fields.push_back({ tensor, entry, family, state.mixed_secondary_slot_bytes });
                state.mixed_secondary_slot_bytes += entry->bytes_per_expert;
            };

            bind_secondary_mixed_field(state.gate_up_tensor, state.secondary_gate_up_entry, routed_family::gate_up);
            bind_secondary_mixed_field(state.gate_tensor,    state.secondary_gate_entry,    routed_family::gate);
            bind_secondary_mixed_field(state.up_tensor,      state.secondary_up_entry,      routed_family::up);
            bind_secondary_mixed_field(state.down_tensor,    state.secondary_down_entry,    routed_family::down);
        }

        touched_slots.reserve(slot_count);

        if (oracle_all_hit || oracle_prefetch) {
            load_oracle_trace(model.flash_moe_trace_file());
        }

        if (resident_bank_source) {
            preload_resident_banks();
            resident_full_bank_pending = slot_count == expert_count;
        }

        if (parallel_slot_reads || cache_io_split > 1 || demand_stripe_enabled || prefetch_stripe_enabled) {
            start_read_pool();
        }

        if (cache_io_split > 1) {
            LLAMA_LOG_INFO("%s: Flash-MoE routed installs will split expert preads into %d page-aligned chunks\n",
                    __func__, cache_io_split);
        }

        if (mixed_slot_buffer) {
            LLAMA_LOG_INFO("%s: Flash-MoE mixed slot staging is enabled; miss reads will batch into slot-shaped buffers before tensor uploads\n",
                    __func__);
        }

        if (oracle_all_hit && !model.hparams.no_alloc) {
            // Prime the replayed resident set during runtime construction so oracle-all-hit
            // acts as a true hit-only decode path instead of charging installs to the first
            // measured routed calls.
            prime_oracle_trace();
        }
    }

    ~llama_flash_moe_slot_runtime() override {
        stop_read_pool();
        for (auto & [_, uploader] : async_uploaders) {
            for (auto & buffer : uploader.buffers) {
                if (buffer.pending) {
                    ggml_backend_event_synchronize(buffer.event);
                    buffer.pending = false;
                }
                if (buffer.event != nullptr) {
                    ggml_backend_event_free(buffer.event);
                }
                if (buffer.host_buffer != nullptr) {
                    ggml_backend_buffer_free(buffer.host_buffer);
                }
            }
            if (uploader.backend != nullptr) {
                ggml_backend_free(uploader.backend);
            }
        }
        if (trace_fp != nullptr) {
            std::fclose(trace_fp);
        }
        log_runtime_summary();
        if (oracle_prefetch && oracle_prefetch_repairs > 0) {
            LLAMA_LOG_WARN("%s: Flash-MoE oracle-prefetch repaired %zu replay records on demand; replay result is conservative until the handoff path is tightened\n",
                    __func__, oracle_prefetch_repairs);
        }
        for (auto & [_, fd] : fds) {
            if (fd >= 0) {
                close(fd);
            }
        }
    }

    bool uses_layer(int layer) const override {
        return layer >= 0 && layer < (int) layers.size() && layers[layer].enabled;
    }

    bool uses_native_slot_map(int layer) const override {
        return uses_layer(layer) &&
                (model.arch == LLM_ARCH_QWEN35MOE || model.arch == LLM_ARCH_DEEPSEEK2) &&
                !native_slot_map_disabled();
    }

    void bind_slot_ids_input(int layer, ggml_tensor * slot_ids) override {
        GGML_ASSERT(uses_layer(layer));
        layers[layer].slot_ids_input = slot_ids;
    }

    ggml_tensor * build_slot_ids_tensor(ggml_context * ctx0, ggml_tensor * selected_experts, int layer) override {
        GGML_ASSERT(uses_native_slot_map(layer));

        auto & userdata = native_slot_map_ud[layer];
        userdata.runtime = this;
        userdata.layer = layer;

        return ggml_map_custom1(ctx0, selected_experts, native_slot_map_custom_op, 1, &userdata);
    }

    bool wants_tensor(const ggml_tensor * tensor) const {
        int layer = -1;
        return parse_topk_layer(ggml_get_name(tensor), layer) &&
                uses_layer(layer) &&
                !uses_native_slot_map(layer);
    }

    void temporal_prefetch_after_decode() {
        if (!temporal_prefetch && !prediction_enabled()) {
            return;
        }

        const bool use_temporal_sparse = temporal_prefetch && temporal_prefetch_sparse && !prediction_enabled();
        const size_t sparse_layer_parity = temporal_prefetch_sparse_even_layers ? 0 : 1;

        for (size_t layer = 0; layer < layers.size(); ++layer) {
            auto & state = layers[layer];
            if (!state.enabled) {
                continue;
            }

            if (prediction_enabled()) {
                if (!state.predicted_valid || state.predicted_experts.empty()) {
                    continue;
                }

                prefetch_experts(state, (int) layer, state.predicted_experts, prediction_mode_name(), predictor_prefetch_stats);
            } else if (!state.temporal_prefetch_experts.empty()) {
                if (use_temporal_sparse && ((layer & 1U) != sparse_layer_parity)) {
                    continue;
                }
                prefetch_experts(state, (int) layer, state.temporal_prefetch_experts, "temporal-prefetch", prefetch_stats);
            }
        }

        if (use_temporal_sparse) {
            temporal_prefetch_sparse_even_layers = !temporal_prefetch_sparse_even_layers;
        }
    }

    void prime_oracle_trace() {
        if (!oracle_all_hit || oracle_primed) {
            return;
        }

        for (size_t layer = 0; layer < layers.size(); ++layer) {
            auto & state = layers[layer];
            if (!state.enabled) {
                continue;
            }

            for (int32_t slot = 0; slot < state.n_slots; ++slot) {
                const int32_t expert = state.slot_to_expert[slot];
                if (expert < 0) {
                    continue;
                }

                loads.clear();
                loads.emplace_back(expert, slot);
                const auto install = install_loads(state, loads);
                oracle_prime_stats.calls++;
                oracle_prime_stats.unique_experts += install.experts;
                oracle_prime_stats.miss_experts += install.experts;
                oracle_prime_stats.bytes_loaded += install.bytes;
                oracle_prime_stats.pread_ops += install.pread_ops;
                oracle_prime_stats.resident_copy_ops += install.resident_copy_ops;
                oracle_prime_stats.install_us += install.install_us;
                oracle_prime_stats.source_us += install.source_us;
                oracle_prime_stats.source_wall_us += install.source_wall_us;
                oracle_prime_stats.upload_us += install.upload_us;
                accumulate_install_breakdown(oracle_prime_stats, install);
                oracle_prime_stats.total_us += install.install_us;
                state.slot_age[slot] = ++age;
            }
        }

        oracle_primed = true;
    }

    bool handle_tensor(ggml_tensor * tensor) {
        int layer = -1;
        if (!parse_topk_layer(ggml_get_name(tensor), layer) || !uses_layer(layer)) {
            return true;
        }

        auto & state = layers[layer];
        if (state.slot_ids_input == nullptr) {
            throw std::runtime_error(format("Flash-MoE slot ids input not bound for layer %d", layer));
        }

        process_topk_tensor(layer, tensor, state.slot_ids_input);

        return true;
    }

private:
    struct native_slot_map_userdata {
        llama_flash_moe_slot_runtime * runtime = nullptr;
        int layer = -1;
    };

    static void native_slot_map_custom_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
        GGML_UNUSED(nth);

        if (ith != 0) {
            return;
        }

        auto * slot_userdata = static_cast<native_slot_map_userdata *>(userdata);
        GGML_ASSERT(slot_userdata != nullptr);
        GGML_ASSERT(slot_userdata->runtime != nullptr);
        GGML_ASSERT(slot_userdata->layer >= 0);

        slot_userdata->runtime->process_topk_tensor(slot_userdata->layer, a, dst);
    }

    void process_topk_tensor(int layer, const ggml_tensor * tensor, ggml_tensor * slot_ids_tensor) {
        GGML_ASSERT(uses_layer(layer));
        GGML_ASSERT(slot_ids_tensor != nullptr);

        auto & state = layers[layer];

        if (resident_full_bank_pending) {
            eager_materialize_full_bank_if_possible();
        }

        if (tensor->type != GGML_TYPE_I32) {
            throw std::runtime_error(format("Flash-MoE expected I32 top-k tensor for layer %d", layer));
        }

        const int64_t n_expert_used = tensor->ne[0];
        const int64_t n_tokens = tensor->ne[1];
        const size_t n_ids = size_t(n_expert_used * n_tokens);
        state.stats.calls++;
        state.stats.token_refs += n_ids;
        if (oracle_all_hit) {
            if (!oracle_primed) {
                prime_oracle_trace();
            }
            const int64_t t_handle_start_us = ggml_time_us();
            const auto & record = next_oracle_record(layer, n_expert_used, n_tokens);
            touched_slots.clear();
            touched_slots.reserve(record.slot_ids.size());
            const uint32_t epoch = next_request_epoch();
            for (const int32_t slot : record.slot_ids) {
                if (slot >= 0 && slot < state.n_slots && state.slot_reserved_epoch[slot] != epoch) {
                    state.slot_reserved_epoch[slot] = epoch;
                    touched_slots.push_back(slot);
                }
            }
            state.stats.unique_experts += touched_slots.size();
            state.stats.hit_experts += touched_slots.size();
            const int64_t t_write_start_us = ggml_time_us();
            write_slot_ids_tensor(slot_ids_tensor, record.slot_ids);
            state.stats.slot_write_us += ggml_time_us() - t_write_start_us;
            for (const int32_t slot : touched_slots) {
                state.slot_age[slot] = ++age;
            }
            state.stats.total_us += ggml_time_us() - t_handle_start_us;
            return;
        }
        if (oracle_prefetch) {
            if (!oracle_prefetch_primed) {
                prime_oracle_prefetch_record(0, nullptr);
                oracle_prefetch_primed = true;
            }
            const int64_t t_handle_start_us = ggml_time_us();
            const size_t current_record_index = oracle_cursor;
            const auto & record = next_oracle_record(layer, n_expert_used, n_tokens);
            if (!oracle_record_is_resident(record)) {
                prime_oracle_prefetch_record(current_record_index, nullptr);
                oracle_prefetch_repairs++;
            }
            materialize_oracle_slot_ids(record, slot_ids);

            touched_slots.clear();
            touched_slots.reserve(slot_ids.size());
            const uint32_t epoch = next_request_epoch();
            for (const int32_t slot : slot_ids) {
                if (slot >= 0 && slot < state.n_slots && state.slot_reserved_epoch[slot] != epoch) {
                    state.slot_reserved_epoch[slot] = epoch;
                    touched_slots.push_back(slot);
                }
            }
            state.stats.unique_experts += touched_slots.size();
            state.stats.hit_experts += touched_slots.size();
            const int64_t t_write_start_us = ggml_time_us();
            write_slot_ids_tensor(slot_ids_tensor, slot_ids);
            state.stats.slot_write_us += ggml_time_us() - t_write_start_us;
            for (const int32_t slot : touched_slots) {
                state.slot_age[slot] = ++age;
            }

            if (oracle_cursor < oracle_records.size()) {
                const auto & next_record = oracle_records[oracle_cursor];
                prime_oracle_prefetch_record(oracle_cursor, next_record.layer == layer ? &slot_ids : nullptr);
            }
            state.stats.total_us += ggml_time_us() - t_handle_start_us;
            return;
        }

        const int64_t t_handle_start_us = ggml_time_us();
        const int64_t t_topk_start_us = ggml_time_us();
        read_topk_ids_tensor(tensor, n_expert_used, n_tokens);
        state.stats.topk_read_us += ggml_time_us() - t_topk_start_us;

        slot_ids.resize(n_ids);
        touched_slots.clear();
        loads.clear();
        loads.reserve(n_ids);
        if (temporal_prefetch) {
            state.temporal_prefetch_experts.clear();
            state.temporal_prefetch_experts.reserve(n_ids);
        }
        if (prediction_enabled()) {
            state.current_token_experts.clear();
            state.current_token_experts.reserve(n_ids);
        }

        const uint32_t epoch = next_request_epoch();

        const int64_t t_resolve_start_us = ggml_time_us();
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t expert = topk_ids[i];
            if (expert < 0 || expert >= expert_count) {
                throw std::runtime_error(format(
                    "Flash-MoE expert id %d is out of range for slot-bank with %d experts",
                    expert, expert_count));
            }

            if (state.request_seen_epoch[expert] == epoch) {
                slot_ids[i] = state.request_slot[expert];
                continue;
            }

            if (temporal_prefetch && !prediction_enabled()) {
                state.temporal_prefetch_experts.push_back(expert);
            }
            if (prediction_enabled()) {
                state.current_token_experts.push_back(expert);
            }

            int32_t slot = state.expert_to_slot[expert];
            if (slot >= 0) {
                state.slot_reserved_epoch[slot] = epoch;
            } else {
                slot = select_slot(state, epoch);
                if (slot < 0) {
                    throw std::runtime_error(format(
                        "Flash-MoE slot-bank overflow in layer %d: need more than %d resident experts for this batch",
                        layer, state.n_slots));
                }

                state.slot_reserved_epoch[slot] = epoch;
                loads.emplace_back(expert, slot);
            }

            state.request_seen_epoch[expert] = epoch;
            state.request_slot[expert] = slot;
            slot_ids[i] = slot;
            touched_slots.push_back(slot);
        }
        state.stats.slot_resolve_us += ggml_time_us() - t_resolve_start_us;
        state.stats.unique_experts += touched_slots.size();
        state.stats.miss_experts += loads.size();
        state.stats.hit_experts += touched_slots.size() - loads.size();

        const install_metrics install = install_loads(state, loads);
        state.stats.bytes_loaded += install.bytes;
        state.stats.pread_ops += install.pread_ops;
        state.stats.resident_copy_ops += install.resident_copy_ops;
        state.stats.cold_loads += install.cold_loads;
        state.stats.evict_loads += install.evict_loads;
        state.stats.install_us += install.install_us;
        state.stats.source_us += install.source_us;
        state.stats.source_wall_us += install.source_wall_us;
        state.stats.upload_us += install.upload_us;
        accumulate_install_breakdown(state.stats, install);

        for (const int32_t slot : touched_slots) {
            state.slot_age[slot] = ++age;
        }

        const int64_t t_trace_start_us = ggml_time_us();
        write_trace(layer, n_expert_used, n_tokens);
        state.stats.trace_write_us += ggml_time_us() - t_trace_start_us;

        if (prediction_enabled()) {
            if (state.predicted_valid) {
                predictor_stats.layer_calls++;
                predictor_stats.predicted_experts += state.predicted_experts.size();
                predictor_stats.actual_experts += state.current_token_experts.size();
                predictor_stats.overlap_experts += count_overlap_experts(state.predicted_experts, state.current_token_experts);
            }

            state.predicted_experts = state.current_token_experts;
            if (predict_top1_prev && state.predicted_experts.size() > 1) {
                state.predicted_experts.resize(1);
            }
            state.predicted_valid = !state.predicted_experts.empty();
        }

        const int64_t t_write_start_us = ggml_time_us();
        write_slot_ids_tensor(slot_ids_tensor, slot_ids);
        state.stats.slot_write_us += ggml_time_us() - t_write_start_us;
        state.stats.total_us += ggml_time_us() - t_handle_start_us;
    }
    struct oracle_record {
        int32_t layer = -1;
        int32_t n_expert_used = 0;
        int32_t n_tokens = 0;
        std::vector<int32_t> experts;
        std::vector<int32_t> slot_ids;
    };

    struct install_metrics {
        size_t  experts = 0;
        size_t  bytes = 0;
        std::array<uint64_t, 3> source_lane_bytes = { 0, 0, 0 };
        size_t  pread_ops = 0;
        size_t  resident_copy_ops = 0;
        size_t  cold_loads = 0;
        size_t  evict_loads = 0;
        int64_t source_us = 0;
        int64_t source_wall_us = 0;
        int64_t upload_us = 0;
        int64_t install_us = 0;
        int64_t gate_up_install_us = 0;
        int64_t gate_install_us = 0;
        int64_t up_install_us = 0;
        int64_t down_install_us = 0;
        uint64_t gate_up_bytes = 0;
        uint64_t gate_bytes = 0;
        uint64_t up_bytes = 0;
        uint64_t down_bytes = 0;
    };

    struct routed_metrics {
        uint64_t calls = 0;
        uint64_t token_refs = 0;
        uint64_t unique_experts = 0;
        uint64_t hit_experts = 0;
        uint64_t miss_experts = 0;
        uint64_t bytes_loaded = 0;
        std::array<uint64_t, 3> source_lane_bytes = { 0, 0, 0 };
        uint64_t pread_ops = 0;
        uint64_t resident_copy_ops = 0;
        uint64_t cold_loads = 0;
        uint64_t evict_loads = 0;
        int64_t total_us = 0;
        int64_t topk_read_us = 0;
        int64_t slot_resolve_us = 0;
        int64_t install_us = 0;
        int64_t source_us = 0;
        int64_t source_wall_us = 0;
        int64_t upload_us = 0;
        int64_t slot_write_us = 0;
        int64_t trace_write_us = 0;
        int64_t gate_up_install_us = 0;
        int64_t gate_install_us = 0;
        int64_t up_install_us = 0;
        int64_t down_install_us = 0;
        uint64_t gate_up_bytes = 0;
        uint64_t gate_bytes = 0;
        uint64_t up_bytes = 0;
        uint64_t down_bytes = 0;
    };

    struct predictor_metrics {
        uint64_t layer_calls = 0;
        uint64_t predicted_experts = 0;
        uint64_t actual_experts = 0;
        uint64_t overlap_experts = 0;
    };

    struct resident_bank_file {
        std::vector<uint8_t> data;
    };

    struct async_upload_buffer_state {
        ggml_backend_buffer_t host_buffer = nullptr;
        ggml_backend_event_t event = nullptr;
        void * host_ptr = nullptr;
        bool pending = false;
    };

    struct async_slot_uploader {
        ggml_backend_t backend = nullptr;
        ggml_backend_dev_t dev = nullptr;
        size_t buffer_size = 0;
        size_t next_buffer = 0;
        std::vector<async_upload_buffer_state> buffers;
    };

    enum class routed_family : uint8_t {
        gate_up,
        gate,
        up,
        down,
    };

    struct mixed_slot_field {
        ggml_tensor * tensor = nullptr;
        const llama_flash_moe_sidecar_entry * entry = nullptr;
        routed_family family = routed_family::gate;
        size_t slot_offset = 0;
    };

    struct install_chunk {
        ggml_tensor * tensor = nullptr;
        const llama_flash_moe_sidecar_entry * entry = nullptr;
        const llama_flash_moe_sidecar_entry * secondary_entry = nullptr;
        const llama_flash_moe_sidecar_entry * tertiary_entry = nullptr;
        int source_lane = 0;
        routed_family family = routed_family::gate;
        int32_t expert = -1;
        int32_t slot = -1;
        size_t task_begin = 0;
        int32_t task_count = 0;
        uint8_t * direct_dst = nullptr;
        std::vector<uint8_t> bytes;
        bool use_striped_read = false;
    };

    struct install_slot_field {
        const mixed_slot_field * field = nullptr;
        size_t task_begin = 0;
        int32_t task_count = 0;
    };

    struct install_slot_buffer {
        int32_t expert = -1;
        int32_t slot = -1;
        int source_lane = 0;
        std::vector<uint8_t> bytes;
        std::vector<install_slot_field> fields;
    };

    struct pending_slot_load {
        int32_t expert = -1;
        int32_t slot = -1;
        bool use_secondary = false;
    };

    struct pread_result {
        ssize_t result = 0;
        int64_t elapsed_us = 0;
    };

    struct pread_task {
        int fd = -1;
        void * dst = nullptr;
        off_t offset = 0;
        size_t size = 0;
        ssize_t result = 0;
        int64_t elapsed_us = 0;
    };

    struct read_thread_pool {
        std::mutex mutex;
        std::condition_variable work_ready;
        std::condition_variable work_done;
        std::vector<std::thread> workers;
        pread_task * tasks = nullptr;
        int num_tasks = 0;
        int tasks_completed = 0;
        int generation = 0;
        int completed_generation = 0;
        bool shutdown = false;
    };

    struct layer_state {
        bool enabled = false;
        int32_t n_slots = 0;
        ggml_tensor * slot_ids_input = nullptr;

        ggml_tensor * gate_up_tensor = nullptr;
        ggml_tensor * gate_tensor    = nullptr;
        ggml_tensor * up_tensor      = nullptr;
        ggml_tensor * down_tensor    = nullptr;

        const llama_flash_moe_sidecar_entry * gate_up_entry = nullptr;
        const llama_flash_moe_sidecar_entry * gate_entry    = nullptr;
        const llama_flash_moe_sidecar_entry * up_entry      = nullptr;
        const llama_flash_moe_sidecar_entry * down_entry    = nullptr;
        const llama_flash_moe_sidecar_entry * prefetch_gate_up_entry = nullptr;
        const llama_flash_moe_sidecar_entry * prefetch_gate_entry    = nullptr;
        const llama_flash_moe_sidecar_entry * prefetch_up_entry      = nullptr;
        const llama_flash_moe_sidecar_entry * prefetch_down_entry    = nullptr;
        const llama_flash_moe_sidecar_entry * secondary_gate_up_entry = nullptr;
        const llama_flash_moe_sidecar_entry * secondary_gate_entry    = nullptr;
        const llama_flash_moe_sidecar_entry * secondary_up_entry      = nullptr;
        const llama_flash_moe_sidecar_entry * secondary_down_entry    = nullptr;
        const llama_flash_moe_sidecar_entry * tertiary_gate_up_entry = nullptr;
        const llama_flash_moe_sidecar_entry * tertiary_gate_entry    = nullptr;
        const llama_flash_moe_sidecar_entry * tertiary_up_entry      = nullptr;
        const llama_flash_moe_sidecar_entry * tertiary_down_entry    = nullptr;

        std::vector<int32_t>  slot_to_expert;
        std::vector<int32_t>  expert_to_slot;
        std::vector<uint64_t> slot_age;
        std::vector<uint32_t> slot_reserved_epoch;
        std::vector<uint32_t> request_seen_epoch;
        std::vector<int32_t>  request_slot;
        std::vector<int32_t>  temporal_prefetch_experts;
        std::vector<int32_t>  predicted_experts;
        std::vector<int32_t>  current_token_experts;
        bool predicted_valid = false;
        std::vector<mixed_slot_field> mixed_slot_fields;
        std::vector<mixed_slot_field> mixed_prefetch_slot_fields;
        std::vector<mixed_slot_field> mixed_secondary_slot_fields;
        size_t mixed_slot_bytes = 0;
        size_t mixed_prefetch_slot_bytes = 0;
        size_t mixed_secondary_slot_bytes = 0;
        int32_t resident_count = 0;
        int32_t peak_resident_count = 0;
        routed_metrics stats;
    };

    const llama_model & model;
    int32_t slot_count = 0;
    int32_t expert_count = 0;
    uint64_t age = 0;
    uint32_t request_epoch = 1;
    bool resident_bank_source = false;
    bool resident_full_bank_pending = false;
    bool oracle_all_hit = false;
    bool oracle_prefetch = false;
    bool temporal_prefetch = false;
    bool temporal_prefetch_sparse = false;
    bool temporal_prefetch_sparse_even_layers = true;
    bool predict_prev_token = false;
    bool predict_top1_prev = false;
    bool secondary_sidecar_enabled = false;
    bool demand_stripe_requested = false;
    bool demand_stripe_enabled = false;
    bool prefetch_stripe_enabled = false;
    bool perf_profile = false;
    bool async_slot_upload = false;
    bool parallel_slot_reads = false;
    bool mixed_slot_buffer = false;
    int32_t cache_io_split = 1;
    int32_t prefetch_cache_io_split = 1;
    std::array<int32_t, 3> demand_stripe_weights = { 1, 0, 0 };
    std::array<int32_t, 3> prefetch_stripe_weights = { 1, 0, 0 };
    bool oracle_primed = false;
    bool oracle_prefetch_primed = false;
    size_t oracle_prefetch_repairs = 0;
    size_t async_slot_upload_buffer_size = 0;
    routed_metrics resident_prime_stats;
    routed_metrics prefetch_stats;
    routed_metrics predictor_prefetch_stats;
    routed_metrics oracle_prime_stats;
    predictor_metrics predictor_stats;
    std::vector<native_slot_map_userdata> native_slot_map_ud;
    std::vector<layer_state> layers;
    std::mutex fds_mutex;
    std::unordered_map<std::string, int> fds;
    std::unordered_map<std::string, resident_bank_file> resident_banks;
    std::unordered_map<ggml_backend_dev_t, async_slot_uploader> async_uploaders;
    read_thread_pool read_pool;
    std::vector<int32_t> topk_ids;
    std::vector<int32_t> slot_ids;
    std::vector<int32_t> touched_slots;
    std::vector<std::pair<int32_t, int32_t>> loads;
    std::vector<uint8_t> staging;
    std::vector<oracle_record> oracle_records;
    size_t oracle_cursor = 0;
    FILE * trace_fp = nullptr;
    uint64_t trace_seq = 0;

    static bool parse_topk_layer(const char * name, int & layer) {
        return name != nullptr && sscanf(name, "ffn_moe_topk-%d", &layer) == 1;
    }

    static bool native_slot_map_disabled() {
        const char * value = std::getenv("LLAMA_FLASH_MOE_DISABLE_NATIVE_SLOT_MAP");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }

    static bool async_slot_upload_enabled() {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_ASYNC_SLOT_UPLOAD");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }

    static bool parallel_slot_reads_enabled() {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_PARALLEL_SLOT_READS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }

    static bool batched_install_reads_enabled() {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_BATCHED_INSTALL_READS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }

    bool prediction_enabled() const {
        return predict_prev_token || predict_top1_prev;
    }

    const char * prediction_mode_name() const {
        if (predict_top1_prev) {
            return "predict-top1-prev";
        }
        if (predict_prev_token) {
            return "predict-prev-token";
        }
        return nullptr;
    }

    static uint64_t count_overlap_experts(const std::vector<int32_t> & lhs, const std::vector<int32_t> & rhs) {
        uint64_t overlap = 0;
        for (const int32_t a : lhs) {
            for (const int32_t b : rhs) {
                if (a == b) {
                    overlap++;
                    break;
                }
            }
        }
        return overlap;
    }

    void log_perf_profile_table(const routed_metrics & total, size_t routed_layers) const {
        if (!perf_profile || total.calls == 0 || routed_layers == 0) {
            return;
        }

        const double token_estimate = double(total.calls) / double(routed_layers);
        const double hit_pct = total.unique_experts > 0 ? 100.0 * double(total.hit_experts) / double(total.unique_experts) : 0.0;
        if (token_estimate <= 0.0) {
            return;
        }

        const auto us_to_ms_per_layer = [routed_layers](int64_t us) {
            return double(std::max<int64_t>(0, us)) / 1000.0 / double(routed_layers);
        };
        const auto us_to_ms_per_token = [token_estimate](int64_t us) {
            return double(std::max<int64_t>(0, us)) / 1000.0 / token_estimate;
        };
        const auto bytes_to_mb_per_layer = [routed_layers](uint64_t bytes) {
            return double(bytes) / (1024.0 * 1024.0) / double(routed_layers);
        };

        const int64_t routing_us = total.topk_read_us + total.slot_resolve_us;
        const int64_t install_overhead_us = total.install_us - total.source_wall_us - total.upload_us;
        const int64_t slot_meta_us = total.slot_write_us + total.trace_write_us;
        const int64_t other_us = total.total_us - total.topk_read_us - total.slot_resolve_us - total.install_us - total.slot_write_us - total.trace_write_us;

        const int64_t breakdown_total_us =
                std::max<int64_t>(0, routing_us) +
                std::max<int64_t>(0, total.source_wall_us) +
                std::max<int64_t>(0, total.upload_us) +
                std::max<int64_t>(0, install_overhead_us) +
                std::max<int64_t>(0, slot_meta_us) +
                std::max<int64_t>(0, other_us);

        const auto log_row = [&](const char * name, int64_t us, uint64_t bytes) {
            const double pct = breakdown_total_us > 0 ? 100.0 * double(std::max<int64_t>(0, us)) / double(breakdown_total_us) : 0.0;
            LLAMA_LOG_INFO("%s: %-24s %8.2f %10.1f %8zu %10.2f %6.1f%%\n",
                    __func__,
                    name,
                    us_to_ms_per_layer(us),
                    bytes_to_mb_per_layer(bytes),
                    routed_layers,
                    us_to_ms_per_token(us),
                    pct);
        };

        LLAMA_LOG_INFO("%s: Flash-MoE profile (accumulated routed metrics, --perf)\n", __func__);
        LLAMA_LOG_INFO("%s: Component                 ms/layer   MB/layer   Layers   ms/token      %%\n", __func__);
        LLAMA_LOG_INFO("%s: -----------------------------------------------------------------------------\n", __func__);
        log_row("Routing + slot resolve", routing_us, 0);
        log_row("Expert I/O source", total.source_wall_us, total.bytes_loaded);
        log_row("Expert upload", total.upload_us, total.bytes_loaded);
        log_row("Install overhead", install_overhead_us, 0);
        log_row("Slot/meta writes", slot_meta_us, 0);
        log_row("Other routed", other_us, 0);
        LLAMA_LOG_INFO("%s: -----------------------------------------------------------------------------\n", __func__);

        const double total_ms_per_token = us_to_ms_per_token(breakdown_total_us);
        const double total_bytes_per_token_gib = double(total.bytes_loaded) / (1024.0 * 1024.0 * 1024.0) / token_estimate;
        LLAMA_LOG_INFO("%s: Routed total%49.2f  100.0%%\n", __func__, total_ms_per_token);
        LLAMA_LOG_INFO("%s: Accumulated per token: %.2f ms, %.2f GiB routed expert bytes\n",
                __func__, total_ms_per_token, total_bytes_per_token_gib);

        const bool use_colors = flash_moe_log_colors_enabled();
        const char * ansi_reset = use_colors ? "\033[0m" : "";
        const char * ansi_label = use_colors ? "\033[1m\x1b[38;5;39m" : "";
        const char * ansi_slot_hit = use_colors ? "\033[1m\x1b[38;5;82m" : "";
        const char * ansi_replay_hit = use_colors ? "\033[1m\x1b[38;5;214m" : "";

        LLAMA_LOG_INFO("%s: %sLegend%s\n", __func__, ansi_label, ansi_reset);
        LLAMA_LOG_INFO("%s:   %sslot-bank cached expert hit rate:%s %s%.1f%%%s\n",
                __func__, ansi_label, ansi_reset, ansi_slot_hit, hit_pct, ansi_reset);
        if (total.source_us != total.source_wall_us) {
            LLAMA_LOG_INFO("%s:   %scumulative source task time:%s %.2f ms/token (wall I/O source above, cumulative can exceed wall under split/stripe overlap)\n",
                    __func__, ansi_label, ansi_reset, us_to_ms_per_token(total.source_us));
        }

#ifdef GGML_USE_METAL
        struct ggml_metal_mul_mat_id_stats metal_stats = {};
        ggml_metal_op_mul_mat_id_get_stats(&metal_stats);
        const uint64_t replay_total = metal_stats.replay_hit + metal_stats.replay_miss;
        if (replay_total > 0) {
            const double replay_hit_pct = 100.0 * double(metal_stats.replay_hit) / double(replay_total);
            LLAMA_LOG_INFO("%s:   %sMetal replay cache hit rate:%s %s%.1f%%%s\n",
                    __func__, ansi_label, ansi_reset, ansi_replay_hit, replay_hit_pct, ansi_reset);
        }
#endif
    }

    static bool mixed_slot_buffer_enabled() {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_MIXED_SLOT_BUFFER");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }

    bool effective_batched_install_reads() const {
        return !demand_stripe_enabled && parallel_slot_reads && (batched_install_reads_enabled() || cache_io_split > 1);
    }

    static int32_t cache_io_split_from_env() {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_CACHE_IO_SPLIT");
        if (value == nullptr || value[0] == '\0') {
            return 1;
        }
        return std::max(1, std::atoi(value));
    }

    static bool cpu_visible_slot_writes_enabled() {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_CPU_VISIBLE_SLOT_WRITES");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }

    static bool force_backend_tensor_writes_enabled() {
        const char * value = std::getenv("LLAMA_FLASH_MOE_EXPERIMENTAL_FORCE_BACKEND_TENSOR_WRITES");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }

    static constexpr int32_t flash_moe_page_bytes = 16 * 1024;
    static constexpr int32_t flash_moe_max_cache_io_split = 8;

    static int32_t active_cache_io_split(size_t bytes_per_expert, int32_t requested_split) {
        int32_t chunks = std::max<int32_t>(1, requested_split);
        chunks = std::min<int32_t>(chunks, flash_moe_max_cache_io_split);
        if (bytes_per_expert == 0 || (bytes_per_expert % flash_moe_page_bytes) != 0) {
            return 1;
        }

        const size_t pages = bytes_per_expert / flash_moe_page_bytes;
        if ((size_t) chunks > pages) {
            chunks = (int32_t) pages;
        }
        return std::max<int32_t>(1, chunks);
    }

    static std::array<size_t, 3> active_stripe_bytes(
            size_t bytes_per_expert,
            bool stripe_enabled,
            const std::array<int32_t, 3> & stripe_weights) {
        if (!stripe_enabled || bytes_per_expert == 0) {
            return { bytes_per_expert, 0, 0 };
        }

        const int64_t total_weight =
                int64_t(stripe_weights[0]) +
                int64_t(stripe_weights[1]) +
                int64_t(stripe_weights[2]);
        if (total_weight <= 0) {
            return { bytes_per_expert, 0, 0 };
        }

        std::array<size_t, 3> bytes = { 0, 0, 0 };
        std::array<int64_t, 3> remainders = { 0, 0, 0 };
        size_t assigned_bytes = 0;

        for (size_t lane = 0; lane < bytes.size(); ++lane) {
            const int64_t weight = std::max<int32_t>(0, stripe_weights[lane]);
            if (weight == 0) {
                continue;
            }

            const uint64_t scaled = uint64_t(bytes_per_expert) * uint64_t(weight);
            bytes[lane] = size_t(scaled / uint64_t(total_weight));
            remainders[lane] = scaled % total_weight;
            assigned_bytes += bytes[lane];
        }

        size_t remaining_bytes = bytes_per_expert - assigned_bytes;
        while (remaining_bytes > 0) {
            size_t best_lane = 0;
            bool found_lane = false;
            for (size_t lane = 0; lane < bytes.size(); ++lane) {
                if (stripe_weights[lane] <= 0) {
                    continue;
                }
                if (!found_lane || remainders[lane] > remainders[best_lane]) {
                    best_lane = lane;
                    found_lane = true;
                }
            }
            if (!found_lane) {
                bytes[0] += remaining_bytes;
                break;
            }
            bytes[best_lane]++;
            remainders[best_lane] = -1;
            remaining_bytes--;
        }

        return bytes;
    }

    void start_read_pool() {
        if (!read_pool.workers.empty()) {
            return;
        }

        constexpr size_t n_workers = 8;
        read_pool.shutdown = false;
        read_pool.tasks = nullptr;
        read_pool.num_tasks = 0;
        read_pool.tasks_completed = 0;
        read_pool.generation = 0;
        read_pool.completed_generation = 0;
        read_pool.workers.reserve(n_workers);
        for (size_t idx = 0; idx < n_workers; ++idx) {
            read_pool.workers.emplace_back([this, idx]() {
                int my_generation = 0;
                while (true) {
                    pread_task * tasks = nullptr;
                    int num_tasks = 0;
                    {
                        std::unique_lock<std::mutex> lock(read_pool.mutex);
                        read_pool.work_ready.wait(lock, [this, my_generation]() {
                            return read_pool.shutdown || read_pool.generation != my_generation;
                        });
                        if (read_pool.shutdown) {
                            return;
                        }
                        my_generation = read_pool.generation;
                        tasks = read_pool.tasks;
                        num_tasks = read_pool.num_tasks;
                    }

                    const int worker_count = int(read_pool.workers.size());
                    for (int task_idx = int(idx); task_idx < num_tasks; task_idx += worker_count) {
                        pread_task & task = tasks[task_idx];
                        const int64_t t_start_us = ggml_time_us();
                        task.result = pread(task.fd, task.dst, task.size, task.offset);
                        task.elapsed_us = ggml_time_us() - t_start_us;
                    }

                    {
                        std::lock_guard<std::mutex> lock(read_pool.mutex);
                        read_pool.tasks_completed++;
                        if (read_pool.tasks_completed == int(n_workers)) {
                            read_pool.completed_generation = my_generation;
                            read_pool.work_done.notify_one();
                        }
                    }
                }
            });
        }
    }

    void stop_read_pool() {
        {
            std::lock_guard<std::mutex> lock(read_pool.mutex);
            read_pool.shutdown = true;
        }
        read_pool.work_ready.notify_all();
        for (auto & worker : read_pool.workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        read_pool.workers.clear();
        read_pool.tasks = nullptr;
        read_pool.num_tasks = 0;
        read_pool.tasks_completed = 0;
    }

    void execute_pread_tasks(std::vector<pread_task> & tasks) {
        if (tasks.empty()) {
            return;
        }

        if (read_pool.workers.empty()) {
            for (auto & task : tasks) {
                const int64_t t_start_us = ggml_time_us();
                task.result = pread(task.fd, task.dst, task.size, task.offset);
                task.elapsed_us = ggml_time_us() - t_start_us;
            }
            return;
        }

        int generation = 0;
        {
            std::lock_guard<std::mutex> lock(read_pool.mutex);
            read_pool.tasks = tasks.data();
            read_pool.num_tasks = int(tasks.size());
            read_pool.tasks_completed = 0;
            read_pool.generation++;
            generation = read_pool.generation;
        }

        read_pool.work_ready.notify_all();

        std::unique_lock<std::mutex> lock(read_pool.mutex);
        read_pool.work_done.wait(lock, [this, generation]() {
            return read_pool.completed_generation >= generation || read_pool.shutdown;
        });
    }

    const oracle_record & next_oracle_record(int layer, int64_t n_expert_used, int64_t n_tokens) {
        if (oracle_cursor >= oracle_records.size()) {
            throw std::runtime_error(format(
                "Flash-MoE oracle trace exhausted at layer %d after %zu routed calls",
                layer, oracle_cursor));
        }

        const auto & record = oracle_records[oracle_cursor++];
        if (record.layer != layer || record.n_expert_used != n_expert_used || record.n_tokens != n_tokens) {
            throw std::runtime_error(format(
                "Flash-MoE oracle trace mismatch at routed call %zu: expected layer=%d k=%d tokens=%d, got layer=%d k=%d tokens=%d",
                oracle_cursor - 1,
                record.layer, record.n_expert_used, record.n_tokens,
                layer, (int) n_expert_used, (int) n_tokens));
        }

        return record;
    }

    void load_oracle_trace(const char * path) {
        using json = nlohmann::json;

        std::ifstream fin(path);
        if (!fin.is_open()) {
            throw std::runtime_error(format("failed to open Flash-MoE oracle trace '%s'", path));
        }

        std::vector<int32_t> next_slot_for_layer(layers.size(), 0);
        std::string line;
        size_t line_no = 0;
        size_t total_unique_experts = 0;

        while (std::getline(fin, line)) {
            ++line_no;
            if (line.empty()) {
                continue;
            }

            const json record_json = json::parse(line, nullptr, true, true);
            oracle_record record;
            record.layer = record_json.at("layer").get<int32_t>();
            record.n_expert_used = record_json.at("n_expert_used").get<int32_t>();
            record.n_tokens = record_json.at("n_tokens").get<int32_t>();
            record.experts = record_json.at("experts").get<std::vector<int32_t>>();

            const size_t expected_ids = size_t(record.n_expert_used) * size_t(record.n_tokens);
            if (record.layer < 0 || record.layer >= (int32_t) layers.size()) {
                throw std::runtime_error(format(
                    "Flash-MoE oracle trace '%s' line %zu has invalid layer %d",
                    path, line_no, record.layer));
            }
            if (!uses_layer(record.layer)) {
                continue;
            }
            if (record.experts.size() != expected_ids) {
                throw std::runtime_error(format(
                    "Flash-MoE oracle trace '%s' line %zu has %zu experts, expected %zu",
                    path, line_no, record.experts.size(), expected_ids));
            }

            auto & state = layers[record.layer];
            if (oracle_all_hit) {
                record.slot_ids.resize(record.experts.size());
            }
            for (size_t i = 0; i < record.experts.size(); ++i) {
                const int32_t expert = record.experts[i];
                if (expert < 0 || expert >= expert_count) {
                    throw std::runtime_error(format(
                        "Flash-MoE oracle trace '%s' line %zu has out-of-range expert %d",
                        path, line_no, expert));
                }

                if (!oracle_all_hit) {
                    continue;
                }

                int32_t slot = state.expert_to_slot[expert];
                if (slot < 0) {
                    slot = next_slot_for_layer[record.layer]++;
                    if (slot >= slot_count) {
                        throw std::runtime_error(format(
                            "Flash-MoE oracle-all-hit needs %d slots in layer %d, but only %d are configured",
                            slot + 1, record.layer, slot_count));
                    }
                    state.expert_to_slot[expert] = slot;
                    state.slot_to_expert[slot] = expert;
                    total_unique_experts++;
                }
                record.slot_ids[i] = slot;
            }

            oracle_records.emplace_back(std::move(record));
        }

        if (oracle_records.empty()) {
            throw std::runtime_error(format("Flash-MoE oracle trace '%s' produced no routed replay records", path));
        }

        LLAMA_LOG_INFO("%s: loaded Flash-MoE oracle trace %s with %zu routed calls%s\n",
                __func__, path, oracle_records.size(),
                oracle_all_hit ? format(" and %zu unique experts across replayed layers", total_unique_experts).c_str() : "");
    }

    void materialize_oracle_slot_ids(const oracle_record & record, std::vector<int32_t> & out_slot_ids) {
        auto & state = layers[record.layer];
        out_slot_ids.resize(record.experts.size());
        for (size_t i = 0; i < record.experts.size(); ++i) {
            const int32_t expert = record.experts[i];
            const int32_t slot = state.expert_to_slot[expert];
            if (slot < 0 || slot >= state.n_slots) {
                throw std::runtime_error(format(
                    "Flash-MoE oracle-prefetch expected expert %d to be resident in layer %d",
                    expert, record.layer));
            }
            out_slot_ids[i] = slot;
        }
    }

    bool oracle_record_is_resident(const oracle_record & record) const {
        const auto & state = layers[record.layer];
        for (const int32_t expert : record.experts) {
            const int32_t slot = state.expert_to_slot[expert];
            if (slot < 0 || slot >= state.n_slots) {
                return false;
            }
        }
        return true;
    }

    void prefetch_experts(
            layer_state & state,
            int layer,
            const std::vector<int32_t> & experts,
            const char * mode_name,
            routed_metrics & stats_out) {
        const int64_t t_prefetch_start_us = ggml_time_us();
        const uint32_t epoch = next_request_epoch();

        touched_slots.clear();
        loads.clear();
        touched_slots.reserve(experts.size());
        loads.reserve(experts.size());

        for (const int32_t expert : experts) {
            if (expert < 0 || expert >= expert_count) {
                throw std::runtime_error(format(
                    "Flash-MoE %s encountered out-of-range expert %d in layer %d",
                    mode_name, expert, layer));
            }
            if (state.request_seen_epoch[expert] == epoch) {
                continue;
            }

            int32_t slot = state.expert_to_slot[expert];
            if (slot >= 0) {
                state.slot_reserved_epoch[slot] = epoch;
            } else {
                slot = select_slot(state, epoch);
                if (slot < 0) {
                    throw std::runtime_error(format(
                        "Flash-MoE %s needs more than %d slots in layer %d",
                        mode_name, state.n_slots, layer));
                }
                state.slot_reserved_epoch[slot] = epoch;
                loads.emplace_back(expert, slot);
            }

            state.request_seen_epoch[expert] = epoch;
            state.request_slot[expert] = slot;
            touched_slots.push_back(slot);
        }

        const install_metrics install = install_loads(state, loads, true);

        for (const int32_t slot : touched_slots) {
            state.slot_age[slot] = ++age;
        }

        stats_out.calls++;
        stats_out.unique_experts += touched_slots.size();
        stats_out.miss_experts += loads.size();
        stats_out.hit_experts += touched_slots.size() - loads.size();
        stats_out.bytes_loaded += install.bytes;
        stats_out.pread_ops += install.pread_ops;
        stats_out.resident_copy_ops += install.resident_copy_ops;
        stats_out.cold_loads += install.cold_loads;
        stats_out.evict_loads += install.evict_loads;
        stats_out.total_us += ggml_time_us() - t_prefetch_start_us;
        stats_out.install_us += install.install_us;
        stats_out.source_us += install.source_us;
        stats_out.source_wall_us += install.source_wall_us;
        stats_out.upload_us += install.upload_us;
        accumulate_install_breakdown(stats_out, install);
    }

    void prime_oracle_prefetch_record(size_t record_index, const std::vector<int32_t> * protected_slots) {
        if (record_index >= oracle_records.size()) {
            return;
        }

        const int64_t t_prefetch_start_us = ggml_time_us();
        const auto & record = oracle_records[record_index];
        auto & state = layers[record.layer];
        if (!state.enabled) {
            return;
        }

        const uint32_t epoch = next_request_epoch();
        if (protected_slots != nullptr) {
            for (const int32_t slot : *protected_slots) {
                if (slot >= 0 && slot < state.n_slots) {
                    state.slot_reserved_epoch[slot] = epoch;
                }
            }
        }

        touched_slots.clear();
        loads.clear();
        touched_slots.reserve(record.experts.size());
        loads.reserve(record.experts.size());

        for (const int32_t expert : record.experts) {
            if (expert < 0 || expert >= expert_count) {
                throw std::runtime_error(format(
                    "Flash-MoE oracle-prefetch trace contains out-of-range expert %d in layer %d",
                    expert, record.layer));
            }
            if (state.request_seen_epoch[expert] == epoch) {
                continue;
            }

            int32_t slot = state.expert_to_slot[expert];
            if (slot >= 0) {
                state.slot_reserved_epoch[slot] = epoch;
            } else {
                slot = select_slot(state, epoch);
                if (slot < 0) {
                    throw std::runtime_error(format(
                        "Flash-MoE oracle-prefetch needs more than %d slots in layer %d for the current+next replay window",
                        state.n_slots, record.layer));
                }
                state.slot_reserved_epoch[slot] = epoch;
                loads.emplace_back(expert, slot);
            }

            state.request_seen_epoch[expert] = epoch;
            state.request_slot[expert] = slot;
            touched_slots.push_back(slot);
        }

        const install_metrics install = install_loads(state, loads, true);

        for (const int32_t slot : touched_slots) {
            state.slot_age[slot] = ++age;
        }

        prefetch_stats.calls++;
        prefetch_stats.unique_experts += touched_slots.size();
        prefetch_stats.miss_experts += loads.size();
        prefetch_stats.hit_experts += touched_slots.size() - loads.size();
        prefetch_stats.bytes_loaded += install.bytes;
        prefetch_stats.pread_ops += install.pread_ops;
        prefetch_stats.resident_copy_ops += install.resident_copy_ops;
        prefetch_stats.cold_loads += install.cold_loads;
        prefetch_stats.evict_loads += install.evict_loads;
        prefetch_stats.install_us += install.install_us;
        prefetch_stats.source_us += install.source_us;
        prefetch_stats.source_wall_us += install.source_wall_us;
        prefetch_stats.upload_us += install.upload_us;
        accumulate_install_breakdown(prefetch_stats, install);
        prefetch_stats.total_us += ggml_time_us() - t_prefetch_start_us;
    }

    void bind_tensor(
            ggml_tensor * tensor,
            ggml_tensor *& tensor_out,
            const llama_flash_moe_sidecar_entry *& entry_out,
            const llama_flash_moe_sidecar_entry *& prefetch_entry_out,
            const llama_flash_moe_sidecar_entry *& secondary_entry_out,
            const llama_flash_moe_sidecar_entry *& tertiary_entry_out,
            bool & enabled_out) {
        if (tensor == nullptr) {
            return;
        }

        const auto * entry = model.flash_moe_sidecar_entry_for(ggml_get_name(tensor));
        if (entry == nullptr) {
            throw std::runtime_error(format("missing Flash-MoE sidecar entry for '%s'", ggml_get_name(tensor)));
        }

        const size_t expected_slot_bytes = entry->bytes_per_expert * slot_count;
        if (ggml_nbytes(tensor) != expected_slot_bytes) {
            throw std::runtime_error(format(
                "Flash-MoE slot tensor '%s' has %zu bytes, expected %zu for %d slots",
                ggml_get_name(tensor), ggml_nbytes(tensor), expected_slot_bytes, slot_count));
        }

        const auto * prefetch_entry = model.flash_moe_prefetch_sidecar_entry_for(ggml_get_name(tensor));
        if (prefetch_entry == nullptr) {
            throw std::runtime_error(format("missing Flash-MoE prefetch sidecar entry for '%s'", ggml_get_name(tensor)));
        }

        if (prefetch_entry->bytes_per_expert != entry->bytes_per_expert ||
            prefetch_entry->exact_byte_length != entry->exact_byte_length ||
            prefetch_entry->source_format != entry->source_format ||
            prefetch_entry->quant_type != entry->quant_type) {
            throw std::runtime_error(format(
                "Flash-MoE prefetch sidecar entry '%s' is incompatible with the primary sidecar",
                ggml_get_name(tensor)));
        }

        const auto * secondary_entry = model.flash_moe_secondary_sidecar_entry_for(ggml_get_name(tensor));
        if (secondary_entry == nullptr) {
            throw std::runtime_error(format("missing Flash-MoE secondary sidecar entry for '%s'", ggml_get_name(tensor)));
        }

        if (secondary_entry->bytes_per_expert != entry->bytes_per_expert ||
            secondary_entry->exact_byte_length != entry->exact_byte_length ||
            secondary_entry->source_format != entry->source_format ||
            secondary_entry->quant_type != entry->quant_type) {
            throw std::runtime_error(format(
                "Flash-MoE secondary sidecar entry '%s' is incompatible with the primary sidecar",
                ggml_get_name(tensor)));
        }

        const auto * tertiary_entry = model.flash_moe_tertiary_sidecar_entry_for(ggml_get_name(tensor));
        if (tertiary_entry == nullptr) {
            throw std::runtime_error(format("missing Flash-MoE tertiary sidecar entry for '%s'", ggml_get_name(tensor)));
        }

        if (tertiary_entry->bytes_per_expert != entry->bytes_per_expert ||
            tertiary_entry->exact_byte_length != entry->exact_byte_length ||
            tertiary_entry->source_format != entry->source_format ||
            tertiary_entry->quant_type != entry->quant_type) {
            throw std::runtime_error(format(
                "Flash-MoE tertiary sidecar entry '%s' is incompatible with the primary sidecar",
                ggml_get_name(tensor)));
        }

        tensor_out = tensor;
        entry_out = entry;
        prefetch_entry_out = prefetch_entry;
        secondary_entry_out = secondary_entry;
        tertiary_entry_out = tertiary_entry;
        enabled_out = true;
        async_slot_upload_buffer_size = std::max(async_slot_upload_buffer_size, entry->bytes_per_expert);
    }

    void destroy_async_uploader(async_slot_uploader & uploader) {
        for (auto & buffer : uploader.buffers) {
            if (buffer.pending) {
                ggml_backend_event_synchronize(buffer.event);
                buffer.pending = false;
            }
            if (buffer.event != nullptr) {
                ggml_backend_event_free(buffer.event);
                buffer.event = nullptr;
            }
            if (buffer.host_buffer != nullptr) {
                ggml_backend_buffer_free(buffer.host_buffer);
                buffer.host_buffer = nullptr;
            }
            buffer.host_ptr = nullptr;
        }

        uploader.buffers.clear();

        if (uploader.backend != nullptr) {
            ggml_backend_free(uploader.backend);
            uploader.backend = nullptr;
        }

        uploader.dev = nullptr;
        uploader.buffer_size = 0;
        uploader.next_buffer = 0;
    }

    async_slot_uploader * get_async_uploader(ggml_tensor * tensor, size_t required_size) {
        if (!async_slot_upload || required_size == 0) {
            return nullptr;
        }

        ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (buf == nullptr || tensor->data == nullptr || tensor_cpu_visible_data(tensor) != nullptr) {
            return nullptr;
        }

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev == nullptr) {
            return nullptr;
        }

        if (buft != ggml_backend_dev_buffer_type(dev)) {
            return nullptr;
        }

        auto it = async_uploaders.find(dev);
        if (it != async_uploaders.end()) {
            return it->second.buffer_size >= required_size ? &it->second : nullptr;
        }

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
            return nullptr;
        }

        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
        if (host_buft == nullptr) {
            return nullptr;
        }

        async_slot_uploader uploader;
        uploader.dev = dev;
        uploader.buffer_size = std::max(async_slot_upload_buffer_size, required_size);
        uploader.backend = ggml_backend_dev_init(dev, nullptr);
        if (uploader.backend == nullptr) {
            return nullptr;
        }

        constexpr size_t n_buffers = 8;
        uploader.buffers.reserve(n_buffers);
        for (size_t idx = 0; idx < n_buffers; ++idx) {
            async_upload_buffer_state buffer;
            buffer.host_buffer = ggml_backend_buft_alloc_buffer(host_buft, uploader.buffer_size);
            if (buffer.host_buffer == nullptr) {
                destroy_async_uploader(uploader);
                return nullptr;
            }
            buffer.host_ptr = ggml_backend_buffer_get_base(buffer.host_buffer);
            if (buffer.host_ptr == nullptr) {
                destroy_async_uploader(uploader);
                return nullptr;
            }
            buffer.event = ggml_backend_event_new(dev);
            if (buffer.event == nullptr) {
                destroy_async_uploader(uploader);
                return nullptr;
            }
            uploader.buffers.emplace_back(buffer);
        }

        LLAMA_LOG_INFO("%s: Flash-MoE async slot uploads enabled for device %s with %zu pinned buffers of %.2f MiB each\n",
                __func__, ggml_backend_dev_name(dev), uploader.buffers.size(), uploader.buffer_size / 1024.0 / 1024.0);

        auto [inserted_it, inserted] = async_uploaders.emplace(dev, std::move(uploader));
        GGML_ASSERT(inserted);
        return &inserted_it->second;
    }

    int64_t flush_async_uploads() {
        int64_t wait_us = 0;

        for (auto & [_, uploader] : async_uploaders) {
            for (auto & buffer : uploader.buffers) {
                if (!buffer.pending) {
                    continue;
                }

                const int64_t t_wait_start_us = ggml_time_us();
                ggml_backend_event_synchronize(buffer.event);
                wait_us += ggml_time_us() - t_wait_start_us;
                buffer.pending = false;
            }
        }

        return wait_us;
    }

    void accumulate_install_breakdown(routed_metrics & dst, const install_metrics & src) const {
        dst.gate_up_install_us += src.gate_up_install_us;
        dst.gate_install_us += src.gate_install_us;
        dst.up_install_us += src.up_install_us;
        dst.down_install_us += src.down_install_us;
        dst.gate_up_bytes += src.gate_up_bytes;
        dst.gate_bytes += src.gate_bytes;
        dst.up_bytes += src.up_bytes;
        dst.down_bytes += src.down_bytes;
        for (size_t lane = 0; lane < dst.source_lane_bytes.size(); ++lane) {
            dst.source_lane_bytes[lane] += src.source_lane_bytes[lane];
        }
    }

    void accumulate_metrics(routed_metrics & dst, const routed_metrics & src) const {
        dst.calls += src.calls;
        dst.token_refs += src.token_refs;
        dst.unique_experts += src.unique_experts;
        dst.hit_experts += src.hit_experts;
        dst.miss_experts += src.miss_experts;
        dst.bytes_loaded += src.bytes_loaded;
        for (size_t lane = 0; lane < dst.source_lane_bytes.size(); ++lane) {
            dst.source_lane_bytes[lane] += src.source_lane_bytes[lane];
        }
        dst.pread_ops += src.pread_ops;
        dst.resident_copy_ops += src.resident_copy_ops;
        dst.cold_loads += src.cold_loads;
        dst.evict_loads += src.evict_loads;
        dst.total_us += src.total_us;
        dst.topk_read_us += src.topk_read_us;
        dst.slot_resolve_us += src.slot_resolve_us;
        dst.install_us += src.install_us;
        dst.source_us += src.source_us;
        dst.source_wall_us += src.source_wall_us;
        dst.upload_us += src.upload_us;
        dst.slot_write_us += src.slot_write_us;
        dst.trace_write_us += src.trace_write_us;
        dst.gate_up_install_us += src.gate_up_install_us;
        dst.gate_install_us += src.gate_install_us;
        dst.up_install_us += src.up_install_us;
        dst.down_install_us += src.down_install_us;
        dst.gate_up_bytes += src.gate_up_bytes;
        dst.gate_bytes += src.gate_bytes;
        dst.up_bytes += src.up_bytes;
        dst.down_bytes += src.down_bytes;
    }

    void preload_resident_banks() {
        std::vector<std::string> unique_paths;
        unique_paths.reserve(layers.size() * 4);
        for (const auto & state : layers) {
            if (!state.enabled) {
                continue;
            }
            for (const auto * entry : { state.gate_up_entry, state.gate_entry, state.up_entry, state.down_entry }) {
                if (entry == nullptr) {
                    continue;
                }
                if (resident_banks.find(entry->repacked_path) == resident_banks.end()) {
                    unique_paths.push_back(entry->repacked_path);
                    resident_banks.emplace(entry->repacked_path, resident_bank_file{});
                }
            }
        }

        size_t total_bytes = 0;
        for (const auto & path : unique_paths) {
            auto & bank = resident_banks[path];
            std::ifstream fin(path, std::ios::binary | std::ios::ate);
            if (!fin.is_open()) {
                throw std::runtime_error(format("failed to open Flash-MoE resident packed bank '%s'", path.c_str()));
            }

            const std::streamoff size_off = fin.tellg();
            if (size_off < 0) {
                throw std::runtime_error(format("failed to size Flash-MoE resident packed bank '%s'", path.c_str()));
            }

            const size_t size = static_cast<size_t>(size_off);
            bank.data.resize(size);
            fin.seekg(0, std::ios::beg);
            if (!fin.read(reinterpret_cast<char *>(bank.data.data()), static_cast<std::streamsize>(size))) {
                throw std::runtime_error(format("failed to preload Flash-MoE resident packed bank '%s'", path.c_str()));
            }
            total_bytes += size;
        }

        LLAMA_LOG_INFO("%s: preloaded %zu Flash-MoE resident packed-bank files (%.2f GiB total)\n",
                __func__, unique_paths.size(), total_bytes / 1024.0 / 1024.0 / 1024.0);
    }

    void eager_materialize_full_bank_if_possible() {
        if (!resident_full_bank_pending || !resident_bank_source || slot_count != expert_count) {
            return;
        }

        LLAMA_LOG_INFO("%s: eager materializing full resident slot bank for %d experts across routed layers\n",
                __func__, expert_count);

        for (size_t layer = 0; layer < layers.size(); ++layer) {
            auto & state = layers[layer];
            if (!state.enabled) {
                continue;
            }

            loads.clear();
            loads.reserve(expert_count);
            for (int32_t expert = 0; expert < expert_count; ++expert) {
                loads.emplace_back(expert, expert);
            }

            const auto install = install_loads(state, loads);
            resident_prime_stats.calls++;
            resident_prime_stats.unique_experts += install.experts;
            resident_prime_stats.miss_experts += install.experts;
            resident_prime_stats.bytes_loaded += install.bytes;
            resident_prime_stats.pread_ops += install.pread_ops;
            resident_prime_stats.resident_copy_ops += install.resident_copy_ops;
            resident_prime_stats.cold_loads += install.cold_loads;
            resident_prime_stats.evict_loads += install.evict_loads;
            resident_prime_stats.install_us += install.install_us;
            resident_prime_stats.source_us += install.source_us;
            resident_prime_stats.source_wall_us += install.source_wall_us;
            resident_prime_stats.upload_us += install.upload_us;
            accumulate_install_breakdown(resident_prime_stats, install);
            resident_prime_stats.total_us += install.install_us;

            for (int32_t slot = 0; slot < state.n_slots; ++slot) {
                state.slot_age[slot] = ++age;
            }
        }

        resident_full_bank_pending = false;
    }

    void log_runtime_summary() const {
        routed_metrics total;
        std::vector<std::pair<int32_t, routed_metrics>> layer_metrics;
        layer_metrics.reserve(layers.size());

        for (size_t il = 0; il < layers.size(); ++il) {
            const auto & stats = layers[il].stats;
            if (stats.calls == 0) {
                continue;
            }
            accumulate_metrics(total, stats);
            layer_metrics.emplace_back((int32_t) il, stats);
        }

        if (total.calls == 0 && prefetch_stats.calls == 0 && predictor_prefetch_stats.calls == 0 && oracle_prime_stats.miss_experts == 0) {
            return;
        }

        const double hit_pct = total.unique_experts > 0 ? 100.0 * double(total.hit_experts) / double(total.unique_experts) : 0.0;
        const double miss_per_call = total.calls > 0 ? double(total.miss_experts) / double(total.calls) : 0.0;
        const double miss_bytes_gib = total.bytes_loaded / 1024.0 / 1024.0 / 1024.0;
        const int64_t other_us = total.total_us - total.topk_read_us - total.slot_resolve_us - total.install_us - total.slot_write_us - total.trace_write_us;
        const auto log_lane_breakdown = [](const char * label, const routed_metrics & stats) {
            const auto lane_gib = [&stats](size_t lane) {
                return stats.source_lane_bytes[lane] / 1024.0 / 1024.0 / 1024.0;
            };
            const auto lane_pct = [&stats](size_t lane) {
                return stats.bytes_loaded > 0 ? 100.0 * double(stats.source_lane_bytes[lane]) / double(stats.bytes_loaded) : 0.0;
            };
            LLAMA_LOG_INFO("%s: Flash-MoE %s lanes primary=%.2f GiB (%.1f%%) secondary=%.2f GiB (%.1f%%) tertiary=%.2f GiB (%.1f%%)\n",
                    __func__,
                    label,
                    lane_gib(0), lane_pct(0),
                    lane_gib(1), lane_pct(1),
                    lane_gib(2), lane_pct(2));
        };

        LLAMA_LOG_INFO("%s: Flash-MoE routed src=%s calls=%" PRIu64 " refs=%" PRIu64 " uniq=%" PRIu64 " hit=%.1f%% miss/call=%.2f bytes=%.2f GiB topk=%.3f ms resolve=%.3f ms install=%.3f ms source=%.3f ms source_sum=%.3f ms upload=%.3f ms slotwr=%.3f ms trace=%.3f ms other=%.3f ms pread=%" PRIu64 " rcopy=%" PRIu64 " iosplit=%d async=%s preads=%s batchrd=%s mixbuf=%s cpuvis=%s\n",
                __func__,
                resident_bank_source ? "resident-packed" :
                oracle_all_hit ? "oracle-all-hit" :
                oracle_prefetch ? "oracle-prefetch" : "pread-slot-bank",
                total.calls, total.token_refs, total.unique_experts, hit_pct, miss_per_call, miss_bytes_gib,
                total.topk_read_us / 1000.0, total.slot_resolve_us / 1000.0, total.install_us / 1000.0,
                total.source_wall_us / 1000.0, total.source_us / 1000.0, total.upload_us / 1000.0, total.slot_write_us / 1000.0,
                total.trace_write_us / 1000.0, std::max<int64_t>(0, other_us) / 1000.0,
                total.pread_ops, total.resident_copy_ops,
                cache_io_split,
                async_slot_upload ? "on" : "off",
                parallel_slot_reads ? "on" : "off",
                effective_batched_install_reads() ? "on" : "off",
                mixed_slot_buffer ? "on" : "off",
                cpu_visible_slot_writes_enabled() ? "on" : "off");
        if (total.bytes_loaded > 0 || total.gate_up_install_us > 0 || total.gate_install_us > 0 || total.up_install_us > 0 || total.down_install_us > 0) {
            LLAMA_LOG_INFO("%s: Flash-MoE install gate_up=%.3f ms / %.2f GiB gate=%.3f ms / %.2f GiB up=%.3f ms / %.2f GiB down=%.3f ms / %.2f GiB\n",
                    __func__,
                    total.gate_up_install_us / 1000.0, total.gate_up_bytes / 1024.0 / 1024.0 / 1024.0,
                    total.gate_install_us / 1000.0, total.gate_bytes / 1024.0 / 1024.0 / 1024.0,
                    total.up_install_us / 1000.0, total.up_bytes / 1024.0 / 1024.0 / 1024.0,
                    total.down_install_us / 1000.0, total.down_bytes / 1024.0 / 1024.0 / 1024.0);
            log_lane_breakdown("on-demand", total);
        }
        LLAMA_LOG_INFO("%s: Flash-MoE residency cold=%" PRIu64 " evict=%" PRIu64 "\n",
                __func__, total.cold_loads, total.evict_loads);
        log_perf_profile_table(total, layer_metrics.size());

        if (prefetch_stats.calls > 0) {
            const double prefetch_hit_pct = prefetch_stats.unique_experts > 0 ?
                    100.0 * double(prefetch_stats.hit_experts) / double(prefetch_stats.unique_experts) : 0.0;
            LLAMA_LOG_INFO("%s: Flash-MoE prefetch calls=%" PRIu64 " uniq=%" PRIu64 " hit=%.1f%% miss=%" PRIu64 " bytes=%.2f GiB total=%.3f ms install=%.3f ms source=%.3f ms source_sum=%.3f ms upload=%.3f ms pread=%" PRIu64 " rcopy=%" PRIu64 "\n",
                    __func__,
                    prefetch_stats.calls, prefetch_stats.unique_experts, prefetch_hit_pct, prefetch_stats.miss_experts,
                    prefetch_stats.bytes_loaded / 1024.0 / 1024.0 / 1024.0,
                    prefetch_stats.total_us / 1000.0, prefetch_stats.install_us / 1000.0,
                    prefetch_stats.source_wall_us / 1000.0, prefetch_stats.source_us / 1000.0, prefetch_stats.upload_us / 1000.0,
                    prefetch_stats.pread_ops, prefetch_stats.resident_copy_ops);
            log_lane_breakdown("prefetch", prefetch_stats);
        }
        if (predictor_prefetch_stats.calls > 0) {
            const double predict_prefetch_hit_pct = predictor_prefetch_stats.unique_experts > 0 ?
                    100.0 * double(predictor_prefetch_stats.hit_experts) / double(predictor_prefetch_stats.unique_experts) : 0.0;
            LLAMA_LOG_INFO("%s: Flash-MoE %s calls=%" PRIu64 " uniq=%" PRIu64 " hit=%.1f%% miss=%" PRIu64 " bytes=%.2f GiB total=%.3f ms install=%.3f ms source=%.3f ms source_sum=%.3f ms upload=%.3f ms pread=%" PRIu64 " rcopy=%" PRIu64 "\n",
                    __func__,
                    prediction_mode_name(),
                    predictor_prefetch_stats.calls, predictor_prefetch_stats.unique_experts, predict_prefetch_hit_pct, predictor_prefetch_stats.miss_experts,
                    predictor_prefetch_stats.bytes_loaded / 1024.0 / 1024.0 / 1024.0,
                    predictor_prefetch_stats.total_us / 1000.0, predictor_prefetch_stats.install_us / 1000.0,
                    predictor_prefetch_stats.source_wall_us / 1000.0, predictor_prefetch_stats.source_us / 1000.0, predictor_prefetch_stats.upload_us / 1000.0,
                    predictor_prefetch_stats.pread_ops, predictor_prefetch_stats.resident_copy_ops);
            log_lane_breakdown(prediction_mode_name(), predictor_prefetch_stats);
        }
        if (predictor_stats.layer_calls > 0) {
            const double predictor_precision = predictor_stats.predicted_experts > 0 ?
                    100.0 * double(predictor_stats.overlap_experts) / double(predictor_stats.predicted_experts) : 0.0;
            const double predictor_coverage = predictor_stats.actual_experts > 0 ?
                    100.0 * double(predictor_stats.overlap_experts) / double(predictor_stats.actual_experts) : 0.0;
            LLAMA_LOG_INFO("%s: Flash-MoE %s overlap=%" PRIu64 "/%" PRIu64 " precision=%.1f%% coverage=%.1f%% layers=%" PRIu64 "\n",
                    __func__,
                    prediction_mode_name(),
                    predictor_stats.overlap_experts, predictor_stats.predicted_experts,
                    predictor_precision, predictor_coverage, predictor_stats.layer_calls);
        }

        if (resident_prime_stats.miss_experts > 0) {
            LLAMA_LOG_INFO("%s: Flash-MoE resident-slot-bank prime installs=%" PRIu64 " bytes=%.2f GiB total=%.3f ms install=%.3f ms source=%.3f ms source_sum=%.3f ms upload=%.3f ms resident_copy_ops=%" PRIu64 "\n",
                    __func__,
                    resident_prime_stats.miss_experts,
                    resident_prime_stats.bytes_loaded / 1024.0 / 1024.0 / 1024.0,
                    resident_prime_stats.total_us / 1000.0, resident_prime_stats.install_us / 1000.0,
                    resident_prime_stats.source_wall_us / 1000.0, resident_prime_stats.source_us / 1000.0, resident_prime_stats.upload_us / 1000.0,
                    resident_prime_stats.resident_copy_ops);
        }

        if (oracle_prime_stats.miss_experts > 0) {
            LLAMA_LOG_INFO("%s: Flash-MoE oracle-all-hit prime installs=%" PRIu64 " bytes=%.2f GiB total=%.3f ms install=%.3f ms source=%.3f ms source_sum=%.3f ms upload=%.3f ms\n",
                    __func__,
                    oracle_prime_stats.miss_experts,
                    oracle_prime_stats.bytes_loaded / 1024.0 / 1024.0 / 1024.0,
                    oracle_prime_stats.total_us / 1000.0, oracle_prime_stats.install_us / 1000.0,
                    oracle_prime_stats.source_wall_us / 1000.0, oracle_prime_stats.source_us / 1000.0, oracle_prime_stats.upload_us / 1000.0);
        }

        std::stable_sort(layer_metrics.begin(), layer_metrics.end(),
                [](const auto & lhs, const auto & rhs) {
                    if (lhs.second.install_us != rhs.second.install_us) {
                        return lhs.second.install_us > rhs.second.install_us;
                    }
                    return lhs.first < rhs.first;
                });

        const size_t top_layers = std::min<size_t>(8, layer_metrics.size());
        for (size_t i = 0; i < top_layers; ++i) {
            const auto & [layer, stats] = layer_metrics[i];
            const double layer_hit_pct = stats.unique_experts > 0 ? 100.0 * double(stats.hit_experts) / double(stats.unique_experts) : 0.0;
            LLAMA_LOG_INFO("%s: Flash-MoE layer=%d calls=%" PRIu64 " uniq=%" PRIu64 " hit=%.1f%% miss=%" PRIu64 " cold=%" PRIu64 " evict=%" PRIu64 " peak=%d bytes=%.2f MiB install=%.3f ms\n",
                    __func__,
                    layer, stats.calls, stats.unique_experts, layer_hit_pct, stats.miss_experts,
                    stats.cold_loads, stats.evict_loads, layers[layer].peak_resident_count,
                    stats.bytes_loaded / 1024.0 / 1024.0,
                    stats.install_us / 1000.0);
            LLAMA_LOG_DEBUG("%s: Flash-MoE layer %d calls=%" PRIu64 " unique=%" PRIu64 " hit=%.1f%% misses=%" PRIu64 " cold=%" PRIu64 " evict=%" PRIu64 " resident-peak=%d bytes=%.2f MiB topk=%.3f ms resolve=%.3f ms install=%.3f ms slot-write=%.3f ms trace=%.3f ms\n",
                    __func__,
                    layer, stats.calls, stats.unique_experts, layer_hit_pct, stats.miss_experts,
                    stats.cold_loads, stats.evict_loads, layers[layer].peak_resident_count,
                    stats.bytes_loaded / 1024.0 / 1024.0,
                    stats.topk_read_us / 1000.0, stats.slot_resolve_us / 1000.0,
                    stats.install_us / 1000.0, stats.slot_write_us / 1000.0, stats.trace_write_us / 1000.0);
        }
    }

    int fd_for(const std::string & path) {
        std::lock_guard<std::mutex> lock(fds_mutex);
        auto it = fds.find(path);
        if (it != fds.end()) {
            return it->second;
        }

        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error(format("failed to open Flash-MoE bank '%s'", path.c_str()));
        }

        fds.emplace(path, fd);
        return fd;
    }

    uint32_t next_request_epoch() {
        ++request_epoch;
        if (request_epoch != 0) {
            return request_epoch;
        }

        request_epoch = 1;
        for (auto & state : layers) {
            std::fill(state.slot_reserved_epoch.begin(), state.slot_reserved_epoch.end(), 0);
            std::fill(state.request_seen_epoch.begin(), state.request_seen_epoch.end(), 0);
        }

        return request_epoch;
    }

    static int32_t select_slot(const layer_state & state, uint32_t epoch) {
        for (int32_t slot = 0; slot < state.n_slots; ++slot) {
            if (state.slot_reserved_epoch[slot] != epoch && state.slot_to_expert[slot] < 0) {
                return slot;
            }
        }

        int32_t victim = -1;
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (int32_t slot = 0; slot < state.n_slots; ++slot) {
            if (state.slot_reserved_epoch[slot] == epoch) {
                continue;
            }
            if (state.slot_age[slot] < oldest) {
                oldest = state.slot_age[slot];
                victim = slot;
            }
        }

        return victim;
    }

    static uint8_t * tensor_host_data(ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return nullptr;
        }

        ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (buf == nullptr || !ggml_backend_buffer_is_host(buf) || tensor->data == nullptr) {
            return nullptr;
        }

        return static_cast<uint8_t *>(tensor->data);
    }

    static uint8_t * tensor_cpu_visible_data(ggml_tensor * tensor) {
        if (tensor == nullptr || tensor->data == nullptr) {
            return nullptr;
        }

        ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (buf == nullptr) {
            return nullptr;
        }

        if (ggml_backend_buffer_is_host(buf)) {
            return static_cast<uint8_t *>(tensor->data);
        }

        if (cpu_visible_slot_writes_enabled() && ggml_backend_buffer_get_base(buf) != nullptr) {
            return static_cast<uint8_t *>(tensor->data);
        }

        return nullptr;
    }

    static uint8_t * tensor_slot_cpu_visible_data(
            ggml_tensor * tensor,
            const llama_flash_moe_sidecar_entry * entry,
            int32_t slot) {
        if (entry == nullptr || slot < 0) {
            return nullptr;
        }

        if (uint8_t * base = tensor_cpu_visible_data(tensor)) {
            return base + size_t(slot) * entry->bytes_per_expert;
        }

        return nullptr;
    }

    static const uint8_t * tensor_host_data(const ggml_tensor * tensor) {
        return tensor_host_data(const_cast<ggml_tensor *>(tensor));
    }

    static void add_family_install(install_metrics & metrics, routed_family family, int64_t install_us, uint64_t bytes) {
        switch (family) {
            case routed_family::gate_up:
                metrics.gate_up_install_us += install_us;
                metrics.gate_up_bytes += bytes;
                break;
            case routed_family::gate:
                metrics.gate_install_us += install_us;
                metrics.gate_bytes += bytes;
                break;
            case routed_family::up:
                metrics.up_install_us += install_us;
                metrics.up_bytes += bytes;
                break;
            case routed_family::down:
                metrics.down_install_us += install_us;
                metrics.down_bytes += bytes;
                break;
        }
    }

    void read_topk_ids_tensor(const ggml_tensor * tensor, int64_t n_expert_used, int64_t n_tokens) {
        const size_t row_bytes = size_t(n_expert_used) * sizeof(int32_t);
        const size_t n_ids = size_t(n_expert_used * n_tokens);
        topk_ids.resize(n_ids);

        if (const uint8_t * src = tensor_host_data(tensor)) {
            for (int64_t token = 0; token < n_tokens; ++token) {
                std::memcpy(
                        topk_ids.data() + token * n_expert_used,
                        src + size_t(token) * tensor->nb[1],
                        row_bytes);
            }
            return;
        }

        for (int64_t token = 0; token < n_tokens; ++token) {
            ggml_backend_tensor_get(
                    tensor,
                    topk_ids.data() + token * n_expert_used,
                    size_t(token) * tensor->nb[1],
                    row_bytes);
        }
    }

    void write_slot_ids_tensor(ggml_tensor * tensor, const std::vector<int32_t> & values) {
        const size_t bytes = values.size() * sizeof(int32_t);
        if (!force_backend_tensor_writes_enabled()) {
            if (uint8_t * dst = tensor_host_data(tensor)) {
                std::memcpy(dst, values.data(), bytes);
                return;
            }
        }

        ggml_backend_tensor_set(tensor, values.data(), 0, bytes);
    }

    bool entry_requires_runtime_transcode(const llama_flash_moe_sidecar_entry * entry) const {
        return entry != nullptr && entry->source_format != llama_flash_moe_sidecar_format::gguf_bytes;
    }

    bool layer_has_runtime_transcode(const layer_state & state) const {
        return entry_requires_runtime_transcode(state.gate_up_entry) ||
               entry_requires_runtime_transcode(state.gate_entry) ||
               entry_requires_runtime_transcode(state.up_entry) ||
               entry_requires_runtime_transcode(state.down_entry);
    }

    install_metrics transcode_affine_2bit_qwen397b(
            ggml_tensor * tensor,
            const llama_flash_moe_sidecar_entry * entry,
            int32_t expert,
            uint8_t * out) {
        static constexpr int32_t affine_group_size = 64;
        static constexpr size_t affine_expert_size = 3932160;

        struct affine_projection_desc {
            size_t weight_offset;
            size_t scale_offset;
            size_t bias_offset;
            int32_t rows;
            int32_t cols;
            int32_t groups;
            int32_t packed_cols;
        };

        const auto projection_desc = [&]() -> affine_projection_desc {
            if (entry->tensor_family == "ffn_gate_exps") {
                return { 0, 1048576, 1179648, 1024, 4096, 64, 256 };
            }
            if (entry->tensor_family == "ffn_up_exps") {
                return { 1310720, 2359296, 2490368, 1024, 4096, 64, 256 };
            }
            if (entry->tensor_family == "ffn_down_exps") {
                return { 2621440, 3670016, 3801088, 4096, 1024, 16, 64 };
            }

            throw std::runtime_error(format(
                "Flash-MoE affine 2-bit source does not support tensor family '%s' for tensor '%s'",
                entry->tensor_family.c_str(), entry->tensor_name.c_str()));
        }();

        if (entry->quant_type == GGML_TYPE_COUNT || !ggml_is_quantized(entry->quant_type)) {
            throw std::runtime_error(format(
                "Flash-MoE affine 2-bit source requires a quantized target type for tensor '%s'",
                entry->tensor_name.c_str()));
        }

        const int64_t n_per_row = tensor->ne[0];
        const int64_t nrows = tensor->ne[1];
        if (n_per_row <= 0 || nrows <= 0) {
            throw std::runtime_error(format(
                "Flash-MoE affine 2-bit source received invalid tensor shape for '%s'",
                entry->tensor_name.c_str()));
        }

        struct affine_scratch {
            std::vector<uint8_t> expert_blob;
            std::vector<float> projection;
            std::vector<float> aligned;
        };
        thread_local affine_scratch scratch;

        scratch.expert_blob.resize(affine_expert_size);

        install_metrics metrics;
        metrics.bytes = entry->bytes_per_expert;

        const int fd = fd_for(entry->repacked_path);
        const off_t offset = static_cast<off_t>(entry->repacked_offset + size_t(expert) * affine_expert_size);
        const int64_t t_read_start_us = ggml_time_us();
        const ssize_t n_read = pread(fd, scratch.expert_blob.data(), scratch.expert_blob.size(), offset);
        metrics.pread_ops++;
        if (n_read != (ssize_t) scratch.expert_blob.size()) {
            throw std::runtime_error(format(
                "failed to read affine 2-bit expert %d for tensor '%s' from '%s'",
                expert, entry->tensor_name.c_str(), entry->repacked_path.c_str()));
        }

        auto bf16_to_f32 = [](uint16_t bits) -> float {
            uint32_t word = uint32_t(bits) << 16;
            float value;
            std::memcpy(&value, &word, sizeof(value));
            return value;
        };

        const uint8_t * blob = scratch.expert_blob.data();
        const auto * packed = reinterpret_cast<const uint32_t *>(blob + projection_desc.weight_offset);
        const auto * scales = reinterpret_cast<const uint16_t *>(blob + projection_desc.scale_offset);
        const auto * biases = reinterpret_cast<const uint16_t *>(blob + projection_desc.bias_offset);

        scratch.projection.resize(size_t(projection_desc.rows) * size_t(projection_desc.cols));
        for (int32_t row = 0; row < projection_desc.rows; ++row) {
            float * row_dst = scratch.projection.data() + size_t(row) * size_t(projection_desc.cols);
            const uint32_t * packed_row = packed + size_t(row) * size_t(projection_desc.packed_cols);
            const uint16_t * scale_row = scales + size_t(row) * size_t(projection_desc.groups);
            const uint16_t * bias_row = biases + size_t(row) * size_t(projection_desc.groups);

            for (int32_t group = 0; group < projection_desc.groups; ++group) {
                const float scale = bf16_to_f32(scale_row[group]);
                const float bias = bf16_to_f32(bias_row[group]);
                float * group_dst = row_dst + size_t(group) * affine_group_size;
                const uint32_t * packed_group = packed_row + size_t(group) * 4;

                for (int32_t word_idx = 0; word_idx < 4; ++word_idx) {
                    const uint32_t word = packed_group[word_idx];
                    for (int32_t lane = 0; lane < 16; ++lane) {
                        const float q = float((word >> (lane * 2)) & 0x3u);
                        group_dst[word_idx * 16 + lane] = q * scale + bias;
                    }
                }
            }
        }

        const float * quant_src = nullptr;
        if (nrows == projection_desc.rows && n_per_row == projection_desc.cols) {
            quant_src = scratch.projection.data();
        } else if (nrows == projection_desc.cols && n_per_row == projection_desc.rows) {
            scratch.aligned.resize(size_t(nrows) * size_t(n_per_row));
            for (int32_t row = 0; row < projection_desc.rows; ++row) {
                const float * src_row = scratch.projection.data() + size_t(row) * size_t(projection_desc.cols);
                for (int32_t col = 0; col < projection_desc.cols; ++col) {
                    scratch.aligned[size_t(col) * size_t(n_per_row) + size_t(row)] = src_row[col];
                }
            }
            quant_src = scratch.aligned.data();
        } else {
            throw std::runtime_error(format(
                "Flash-MoE affine 2-bit source shape %dx%d does not match tensor '%s' shape %" PRId64 "x%" PRId64,
                projection_desc.rows, projection_desc.cols, entry->tensor_name.c_str(), nrows, n_per_row));
        }

        const size_t expected_nbytes = ggml_row_size(entry->quant_type, n_per_row) * size_t(nrows);
        if (expected_nbytes != entry->bytes_per_expert) {
            throw std::runtime_error(format(
                "Flash-MoE affine 2-bit tensor '%s' expects %zu bytes for target type %s but manifest recorded %zu",
                entry->tensor_name.c_str(), expected_nbytes, ggml_type_name(entry->quant_type), entry->bytes_per_expert));
        }

        const size_t written = ggml_quantize_chunk(entry->quant_type, quant_src, out, 0, nrows, n_per_row, nullptr);
        if (written != entry->bytes_per_expert) {
            throw std::runtime_error(format(
                "Flash-MoE affine 2-bit tensor '%s' quantized to %zu bytes, expected %zu",
                entry->tensor_name.c_str(), written, entry->bytes_per_expert));
        }

        metrics.source_us += ggml_time_us() - t_read_start_us;
        metrics.source_wall_us += metrics.source_us;
        metrics.install_us += metrics.source_us;
        return metrics;
    }

    install_metrics read_expert_bytes(
            ggml_tensor * tensor,
            const llama_flash_moe_sidecar_entry * entry,
            const llama_flash_moe_sidecar_entry * secondary_entry,
            const llama_flash_moe_sidecar_entry * tertiary_entry,
            int source_lane,
            int32_t expert,
            uint8_t * out,
            int32_t requested_cache_io_split,
            bool use_striped_read,
            bool use_prefetch_stripe) {
        install_metrics metrics;
        if (entry == nullptr || out == nullptr) {
            return metrics;
        }

        if (entry->source_format == llama_flash_moe_sidecar_format::affine_2bit_qwen397b) {
            metrics = transcode_affine_2bit_qwen397b(tensor, entry, expert, out);
            if (source_lane >= 0 && source_lane < (int) metrics.source_lane_bytes.size()) {
                metrics.source_lane_bytes[(size_t) source_lane] += metrics.bytes;
            }
            return metrics;
        }

        const off_t offset = static_cast<off_t>(entry->repacked_offset + size_t(expert) * entry->bytes_per_expert);
        metrics.bytes = entry->bytes_per_expert;
        const int64_t t_read_start_us = ggml_time_us();

        if (resident_bank_source) {
            auto it = resident_banks.find(entry->repacked_path);
            if (it == resident_banks.end()) {
                throw std::runtime_error(format(
                    "missing Flash-MoE resident packed bank '%s'",
                    entry->repacked_path.c_str()));
            }

            const auto & bank = it->second.data;
            const size_t copy_offset = static_cast<size_t>(offset);
            if (copy_offset + entry->bytes_per_expert > bank.size()) {
                throw std::runtime_error(format(
                    "Flash-MoE resident packed bank '%s' is too small for tensor '%s' expert %d",
                    entry->repacked_path.c_str(), entry->tensor_name.c_str(), expert));
            }

            std::memcpy(out, bank.data() + copy_offset, entry->bytes_per_expert);
            metrics.source_us += ggml_time_us() - t_read_start_us;
            metrics.source_wall_us += ggml_time_us() - t_read_start_us;
            metrics.resident_copy_ops++;
            metrics.install_us += metrics.source_us;
            metrics.source_lane_bytes[0] += metrics.bytes;
            return metrics;
        }

        const int fd = fd_for(entry->repacked_path);
        const bool use_striped_demand_read =
                use_striped_read &&
                entry->source_format == llama_flash_moe_sidecar_format::gguf_bytes &&
                secondary_entry != nullptr &&
                tertiary_entry != nullptr;

        if (use_striped_demand_read) {
            const auto stripe_bytes = active_stripe_bytes(
                    entry->bytes_per_expert,
                    true,
                    use_prefetch_stripe ? prefetch_stripe_weights : demand_stripe_weights);
            const llama_flash_moe_sidecar_entry * stripe_entries[3] = { entry, secondary_entry, tertiary_entry };
            size_t byte_cursor = 0;
            ssize_t total_read = 0;
            std::vector<pread_task> tasks;
            tasks.reserve(3);

            for (size_t lane = 0; lane < 3; ++lane) {
                if (stripe_bytes[lane] == 0) {
                    continue;
                }

                const auto * lane_entry = stripe_entries[lane];
                const size_t chunk_offset = byte_cursor;
                const size_t chunk_size = stripe_bytes[lane];
                const off_t lane_offset = static_cast<off_t>(
                        lane_entry->repacked_offset + size_t(expert) * lane_entry->bytes_per_expert + chunk_offset);
                byte_cursor += stripe_bytes[lane];
                metrics.source_lane_bytes[lane] += chunk_size;

                tasks.push_back({
                        fd_for(lane_entry->repacked_path),
                        out + chunk_offset,
                        lane_offset,
                        chunk_size,
                        0,
                        0,
                });
            }

            execute_pread_tasks(tasks);

            for (const auto & task : tasks) {
                metrics.pread_ops++;
                metrics.source_us += task.elapsed_us;
                if (task.result > 0) {
                    total_read += task.result;
                }
            }

            if (total_read != (ssize_t) entry->bytes_per_expert) {
                throw std::runtime_error(format(
                    "failed to striped-read expert %d for tensor '%s' across '%s', '%s', '%s'",
                    expert,
                    entry->tensor_name.c_str(),
                    entry->repacked_path.c_str(),
                    secondary_entry->repacked_path.c_str(),
                    tertiary_entry->repacked_path.c_str()));
            }

            metrics.source_wall_us += ggml_time_us() - t_read_start_us;
            metrics.install_us += metrics.source_us;
            return metrics;
        }

        const int32_t chunks = active_cache_io_split(entry->bytes_per_expert, requested_cache_io_split);

        if (chunks <= 1) {
            const ssize_t n_read = pread(fd, out, entry->bytes_per_expert, offset);
            metrics.source_us += ggml_time_us() - t_read_start_us;
            metrics.pread_ops++;
            if (n_read != (ssize_t) entry->bytes_per_expert) {
                throw std::runtime_error(format(
                    "failed to read expert %d for tensor '%s' from '%s'",
                    expert, entry->tensor_name.c_str(), entry->repacked_path.c_str()));
            }
        } else {
            const size_t total_pages = entry->bytes_per_expert / flash_moe_page_bytes;
            size_t page_cursor = 0;
            ssize_t total_read = 0;
            std::vector<pread_task> tasks;
            tasks.reserve(chunks);

            for (int32_t chunk = 0; chunk < chunks; ++chunk) {
                size_t pages_this_chunk = total_pages / (size_t) chunks;
                if ((size_t) chunk < (total_pages % (size_t) chunks)) {
                    pages_this_chunk++;
                }
                const size_t chunk_offset = page_cursor * flash_moe_page_bytes;
                const size_t chunk_size = pages_this_chunk * flash_moe_page_bytes;
                page_cursor += pages_this_chunk;

                tasks.push_back({
                        fd,
                        out + chunk_offset,
                        offset + (off_t) chunk_offset,
                        chunk_size,
                        0,
                        0,
                });
            }

            execute_pread_tasks(tasks);

            for (const auto & task : tasks) {
                metrics.pread_ops++;
                metrics.source_us += task.elapsed_us;
                if (task.result > 0) {
                    total_read += task.result;
                }
            }

            if (total_read != (ssize_t) entry->bytes_per_expert) {
                throw std::runtime_error(format(
                    "failed to split-read expert %d for tensor '%s' from '%s'",
                    expert, entry->tensor_name.c_str(), entry->repacked_path.c_str()));
            }
        }

        metrics.source_wall_us += ggml_time_us() - t_read_start_us;
        metrics.install_us += metrics.source_us;
        if (source_lane >= 0 && source_lane < (int) metrics.source_lane_bytes.size()) {
            metrics.source_lane_bytes[(size_t) source_lane] += metrics.bytes;
        }
        return metrics;
    }

    install_metrics upload_expert_bytes(
            ggml_tensor * tensor,
            const llama_flash_moe_sidecar_entry * entry,
            int32_t slot,
            const uint8_t * src) {
        install_metrics metrics;
        if (tensor == nullptr || entry == nullptr || src == nullptr) {
            return metrics;
        }

        const int64_t t_upload_start_us = ggml_time_us();
        if (!force_backend_tensor_writes_enabled()) {
            if (uint8_t * base = tensor_cpu_visible_data(tensor)) {
                uint8_t * dst = base + size_t(slot) * entry->bytes_per_expert;
                std::memcpy(dst, src, entry->bytes_per_expert);
                metrics.install_us += ggml_time_us() - t_upload_start_us;
                return metrics;
            }
        }

        if (async_slot_uploader * uploader = get_async_uploader(tensor, entry->bytes_per_expert)) {
            auto & buffer = uploader->buffers[uploader->next_buffer];

            if (buffer.pending) {
                const int64_t t_wait_start_us = ggml_time_us();
                ggml_backend_event_synchronize(buffer.event);
                metrics.upload_us += ggml_time_us() - t_wait_start_us;
                buffer.pending = false;
            }

            std::memcpy(buffer.host_ptr, src, entry->bytes_per_expert);
            ggml_backend_tensor_set_async(
                    uploader->backend,
                    tensor,
                    buffer.host_ptr,
                    size_t(slot) * entry->bytes_per_expert,
                    entry->bytes_per_expert);
            ggml_backend_event_record(buffer.event, uploader->backend);
            buffer.pending = true;
            uploader->next_buffer = (uploader->next_buffer + 1) % uploader->buffers.size();
            metrics.upload_us += ggml_time_us() - t_upload_start_us;
            metrics.install_us += ggml_time_us() - t_upload_start_us;
            return metrics;
        }

        staging.resize(entry->bytes_per_expert);
        std::memcpy(staging.data(), src, entry->bytes_per_expert);
        ggml_backend_tensor_set(tensor, staging.data(), size_t(slot) * entry->bytes_per_expert, entry->bytes_per_expert);
        metrics.upload_us += ggml_time_us() - t_upload_start_us;
        metrics.install_us += ggml_time_us() - t_upload_start_us;
        return metrics;
    }

    install_metrics load_into_slot(
            ggml_tensor * tensor,
            const llama_flash_moe_sidecar_entry * entry,
            const llama_flash_moe_sidecar_entry * secondary_entry,
            const llama_flash_moe_sidecar_entry * tertiary_entry,
            int source_lane,
            int32_t expert,
            int32_t slot,
            int32_t requested_cache_io_split,
            bool use_striped_read,
            bool use_prefetch_stripe) {
        install_metrics metrics;
        const int64_t t_install_start_us = ggml_time_us();
        if (tensor == nullptr || entry == nullptr) {
            return metrics;
        }

        if (!force_backend_tensor_writes_enabled()) {
            if (uint8_t * dst = tensor_slot_cpu_visible_data(tensor, entry, slot)) {
                metrics = read_expert_bytes(tensor, entry, secondary_entry, tertiary_entry, source_lane, expert, dst, requested_cache_io_split, use_striped_read, use_prefetch_stripe);
                metrics.install_us = ggml_time_us() - t_install_start_us;
                return metrics;
            }
        }

        staging.resize(entry->bytes_per_expert);
        metrics = read_expert_bytes(tensor, entry, secondary_entry, tertiary_entry, source_lane, expert, staging.data(), requested_cache_io_split, use_striped_read, use_prefetch_stripe);
        const auto upload = upload_expert_bytes(tensor, entry, slot, staging.data());
        metrics.upload_us += upload.upload_us;
        metrics.install_us = ggml_time_us() - t_install_start_us;
        return metrics;
    }

    install_metrics install_loads(
            layer_state & state,
            const std::vector<std::pair<int32_t, int32_t>> & pending_loads,
            bool use_prefetch_entries = false) {
        struct sidecar_entry_set {
            const llama_flash_moe_sidecar_entry * gate_up = nullptr;
            const llama_flash_moe_sidecar_entry * gate = nullptr;
            const llama_flash_moe_sidecar_entry * up = nullptr;
            const llama_flash_moe_sidecar_entry * down = nullptr;
            const std::vector<mixed_slot_field> * mixed_fields = nullptr;
            size_t mixed_bytes = 0;
        };

        install_metrics totals;
        totals.experts = pending_loads.size();

        const int64_t t_install_start_us = ggml_time_us();
        const bool has_runtime_transcode = layer_has_runtime_transcode(state);
        const sidecar_entry_set primary_entries = {
            state.gate_up_entry,
            state.gate_entry,
            state.up_entry,
            state.down_entry,
            &state.mixed_slot_fields,
            state.mixed_slot_bytes,
        };
        const sidecar_entry_set prefetch_entries = {
            state.prefetch_gate_up_entry,
            state.prefetch_gate_entry,
            state.prefetch_up_entry,
            state.prefetch_down_entry,
            &state.mixed_prefetch_slot_fields,
            state.mixed_prefetch_slot_bytes,
        };
        const sidecar_entry_set secondary_entries = {
            state.secondary_gate_up_entry,
            state.secondary_gate_entry,
            state.secondary_up_entry,
            state.secondary_down_entry,
            &state.mixed_secondary_slot_fields,
            state.mixed_secondary_slot_bytes,
        };
        const sidecar_entry_set & default_entries = use_prefetch_entries ? prefetch_entries : primary_entries;
        const bool use_secondary_last_miss =
                !use_prefetch_entries &&
                !demand_stripe_requested &&
                !demand_stripe_enabled &&
                secondary_sidecar_enabled &&
                pending_loads.size() == 4;
        std::vector<pending_slot_load> scheduled_loads;
        scheduled_loads.reserve(pending_loads.size());

        for (size_t load_idx = 0; load_idx < pending_loads.size(); ++load_idx) {
            const auto & [expert, slot] = pending_loads[load_idx];
            const int32_t evicted = state.slot_to_expert[slot];
            if (evicted >= 0 && evicted < expert_count) {
                state.expert_to_slot[evicted] = -1;
                totals.evict_loads++;
            } else {
                totals.cold_loads++;
                state.resident_count++;
                state.peak_resident_count = std::max(state.peak_resident_count, state.resident_count);
            }
            scheduled_loads.push_back({ expert, slot, use_secondary_last_miss && load_idx + 1 == pending_loads.size() });
        }

        auto entries_for_load = [&](const pending_slot_load & load) -> const sidecar_entry_set & {
            if (load.use_secondary) {
                return secondary_entries;
            }
            return default_entries;
        };

        auto cache_io_split_for_load = [&](const pending_slot_load & load) -> int32_t {
            if (load.use_secondary) {
                return cache_io_split;
            }
            return use_prefetch_entries ? prefetch_cache_io_split : cache_io_split;
        };

        auto source_lane_for_load = [&](const pending_slot_load & load) -> int {
            return load.use_secondary ? 1 : 0;
        };

        auto use_stripe_for_load = [&](const pending_slot_load & load) -> bool {
            return (use_prefetch_entries ? prefetch_stripe_enabled : demand_stripe_enabled) && !load.use_secondary;
        };

        auto use_prefetch_stripe_for_load = [&](const pending_slot_load & load) -> bool {
            return use_prefetch_entries && prefetch_stripe_enabled && !load.use_secondary;
        };

        auto secondary_entry_for_family = [&](routed_family family) -> const llama_flash_moe_sidecar_entry * {
            switch (family) {
                case routed_family::gate_up: return state.secondary_gate_up_entry;
                case routed_family::gate:    return state.secondary_gate_entry;
                case routed_family::up:      return state.secondary_up_entry;
                case routed_family::down:    return state.secondary_down_entry;
            }
            return nullptr;
        };

        auto tertiary_entry_for_family = [&](routed_family family) -> const llama_flash_moe_sidecar_entry * {
            switch (family) {
                case routed_family::gate_up: return state.tertiary_gate_up_entry;
                case routed_family::gate:    return state.tertiary_gate_entry;
                case routed_family::up:      return state.tertiary_up_entry;
                case routed_family::down:    return state.tertiary_down_entry;
            }
            return nullptr;
        };

        auto accumulate_install = [&](const install_metrics & metrics, int64_t & dst_us, uint64_t & dst_bytes) {
            totals.bytes += metrics.bytes;
            for (size_t lane = 0; lane < totals.source_lane_bytes.size(); ++lane) {
                totals.source_lane_bytes[lane] += metrics.source_lane_bytes[lane];
            }
            totals.pread_ops += metrics.pread_ops;
            totals.resident_copy_ops += metrics.resident_copy_ops;
            totals.source_us += metrics.source_us;
            totals.source_wall_us += metrics.source_wall_us;
            totals.upload_us += metrics.upload_us;
            dst_us += metrics.install_us;
            dst_bytes += metrics.bytes;
        };

        if (parallel_slot_reads && !resident_bank_source) {
            if (!has_runtime_transcode && !(use_prefetch_entries ? prefetch_stripe_enabled : demand_stripe_enabled) && effective_batched_install_reads()) {
                if (mixed_slot_buffer && default_entries.mixed_bytes > 0 && default_entries.mixed_fields != nullptr && !default_entries.mixed_fields->empty()) {
                    std::vector<install_slot_buffer> slot_buffers;
                    std::vector<pread_task> tasks;
                    const int32_t reserve_split = use_prefetch_entries ? prefetch_cache_io_split : cache_io_split;
                    slot_buffers.reserve(scheduled_loads.size());
                    tasks.reserve(scheduled_loads.size() * default_entries.mixed_fields->size() * std::max<int32_t>(1, reserve_split));

                    for (const auto & load : scheduled_loads) {
                        const auto & entries = entries_for_load(load);
                        auto & slot_buffer = slot_buffers.emplace_back();
                        slot_buffer.expert = load.expert;
                        slot_buffer.slot = load.slot;
                        slot_buffer.source_lane = source_lane_for_load(load);
                        slot_buffer.bytes.resize(entries.mixed_bytes);
                        slot_buffer.fields.reserve(entries.mixed_fields != nullptr ? entries.mixed_fields->size() : 0);

                        for (const auto & field : *entries.mixed_fields) {
                            auto & install_field = slot_buffer.fields.emplace_back();
                            install_field.field = &field;
                            install_field.task_begin = tasks.size();

                            const int fd = fd_for(field.entry->repacked_path);
                            const off_t expert_offset = static_cast<off_t>(field.entry->repacked_offset + size_t(load.expert) * field.entry->bytes_per_expert);
                            const int32_t split = active_cache_io_split(field.entry->bytes_per_expert, cache_io_split_for_load(load));
                            install_field.task_count = split;

                            if (split <= 1) {
                                tasks.push_back({
                                        fd,
                                        slot_buffer.bytes.data() + field.slot_offset,
                                        expert_offset,
                                        field.entry->bytes_per_expert,
                                        0,
                                        0,
                                });
                                continue;
                            }

                            const size_t total_pages = field.entry->bytes_per_expert / flash_moe_page_bytes;
                            size_t page_cursor = 0;
                            for (int32_t chunk_idx = 0; chunk_idx < split; ++chunk_idx) {
                                size_t pages_this_chunk = total_pages / (size_t) split;
                                if ((size_t) chunk_idx < (total_pages % (size_t) split)) {
                                    pages_this_chunk++;
                                }
                                const size_t chunk_offset = page_cursor * flash_moe_page_bytes;
                                const size_t chunk_size = pages_this_chunk * flash_moe_page_bytes;
                                page_cursor += pages_this_chunk;

                                tasks.push_back({
                                        fd,
                                        slot_buffer.bytes.data() + field.slot_offset + chunk_offset,
                                        expert_offset + (off_t) chunk_offset,
                                        chunk_size,
                                        0,
                                        0,
                                });
                            }
                        }
                    }

                    const int64_t t_source_start_us = ggml_time_us();
                    execute_pread_tasks(tasks);
                    const int64_t source_wall_us = ggml_time_us() - t_source_start_us;

                    totals.source_wall_us += source_wall_us;

                    for (auto & slot_buffer : slot_buffers) {
                        for (const auto & install_field : slot_buffer.fields) {
                            const auto & field = *install_field.field;
                            install_metrics metrics;
                            metrics.bytes = field.entry->bytes_per_expert;
                            metrics.source_lane_bytes[(size_t) slot_buffer.source_lane] += metrics.bytes;

                            ssize_t total_read = 0;
                            for (int32_t task_idx = 0; task_idx < install_field.task_count; ++task_idx) {
                                const auto & task = tasks[install_field.task_begin + size_t(task_idx)];
                                metrics.pread_ops++;
                                metrics.source_us += task.elapsed_us;
                                if (task.result > 0) {
                                    total_read += task.result;
                                }
                            }

                            if (total_read != (ssize_t) field.entry->bytes_per_expert) {
                                throw std::runtime_error(format(
                                    "failed to split-read expert %d for tensor '%s' from '%s'",
                                    slot_buffer.expert, field.entry->tensor_name.c_str(), field.entry->repacked_path.c_str()));
                            }

                            metrics.install_us += metrics.source_us;
                            const auto upload = upload_expert_bytes(
                                    field.tensor,
                                    field.entry,
                                    slot_buffer.slot,
                                    slot_buffer.bytes.data() + field.slot_offset);
                            metrics.upload_us += upload.upload_us;
                            metrics.install_us += upload.install_us;
                            add_family_install(metrics, field.family, metrics.install_us, metrics.bytes);

                            switch (field.family) {
                                case routed_family::gate_up:
                                    accumulate_install(metrics, totals.gate_up_install_us, totals.gate_up_bytes);
                                    break;
                                case routed_family::gate:
                                    accumulate_install(metrics, totals.gate_install_us, totals.gate_bytes);
                                    break;
                                case routed_family::up:
                                    accumulate_install(metrics, totals.up_install_us, totals.up_bytes);
                                    break;
                                case routed_family::down:
                                    accumulate_install(metrics, totals.down_install_us, totals.down_bytes);
                                    break;
                            }
                        }
                    }
                } else {
                    std::vector<install_chunk> chunks;
                    std::vector<pread_task> tasks;
                    const int32_t reserve_split = use_prefetch_entries ? prefetch_cache_io_split : cache_io_split;
                    chunks.reserve(scheduled_loads.size() * 4);
                    tasks.reserve(scheduled_loads.size() * 4 * std::max<int32_t>(1, reserve_split));

                    auto queue_chunk = [&](const pending_slot_load & load, ggml_tensor * tensor, const llama_flash_moe_sidecar_entry * entry, routed_family family) {
                        if (tensor == nullptr || entry == nullptr) {
                            return;
                        }

                            auto & chunk = chunks.emplace_back();
                            chunk.tensor = tensor;
                            chunk.entry = entry;
                            chunk.source_lane = source_lane_for_load(load);
                            chunk.family = family;
                        chunk.expert = load.expert;
                        chunk.slot = load.slot;
                        chunk.direct_dst = force_backend_tensor_writes_enabled() ? nullptr : tensor_slot_cpu_visible_data(tensor, entry, load.slot);
                        if (chunk.direct_dst == nullptr) {
                            chunk.bytes.resize(entry->bytes_per_expert);
                        }
                        chunk.task_begin = tasks.size();

                        const int fd = fd_for(entry->repacked_path);
                        const off_t expert_offset = static_cast<off_t>(entry->repacked_offset + size_t(load.expert) * entry->bytes_per_expert);
                        const int32_t split = active_cache_io_split(entry->bytes_per_expert, cache_io_split_for_load(load));
                        chunk.task_count = split;

                        if (split <= 1) {
                            tasks.push_back({
                                    fd,
                                    chunk.direct_dst != nullptr ? chunk.direct_dst : chunk.bytes.data(),
                                    expert_offset,
                                    entry->bytes_per_expert,
                                    0,
                                    0,
                            });
                            return;
                        }

                        const size_t total_pages = entry->bytes_per_expert / flash_moe_page_bytes;
                        size_t page_cursor = 0;
                        for (int32_t chunk_idx = 0; chunk_idx < split; ++chunk_idx) {
                            size_t pages_this_chunk = total_pages / (size_t) split;
                            if ((size_t) chunk_idx < (total_pages % (size_t) split)) {
                                pages_this_chunk++;
                            }
                            const size_t chunk_offset = page_cursor * flash_moe_page_bytes;
                            const size_t chunk_size = pages_this_chunk * flash_moe_page_bytes;
                            page_cursor += pages_this_chunk;

                            tasks.push_back({
                                    fd,
                                    (chunk.direct_dst != nullptr ? chunk.direct_dst : chunk.bytes.data()) + chunk_offset,
                                    expert_offset + (off_t) chunk_offset,
                                    chunk_size,
                                    0,
                                    0,
                            });
                        }
                    };

                    for (const auto & load : scheduled_loads) {
                        const auto & entries = entries_for_load(load);
                        queue_chunk(load, state.gate_up_tensor, entries.gate_up, routed_family::gate_up);
                        queue_chunk(load, state.gate_tensor,    entries.gate,    routed_family::gate);
                        queue_chunk(load, state.up_tensor,      entries.up,      routed_family::up);
                        queue_chunk(load, state.down_tensor,    entries.down,    routed_family::down);
                    }

                    const int64_t t_source_start_us = ggml_time_us();
                    execute_pread_tasks(tasks);
                    const int64_t source_wall_us = ggml_time_us() - t_source_start_us;

                    totals.source_wall_us += source_wall_us;

                    for (auto & chunk : chunks) {
                        install_metrics metrics;
                        metrics.bytes = chunk.entry->bytes_per_expert;
                        metrics.source_lane_bytes[(size_t) chunk.source_lane] += metrics.bytes;

                        ssize_t total_read = 0;
                        for (int32_t task_idx = 0; task_idx < chunk.task_count; ++task_idx) {
                            const auto & task = tasks[chunk.task_begin + size_t(task_idx)];
                            metrics.pread_ops++;
                            metrics.source_us += task.elapsed_us;
                            if (task.result > 0) {
                                total_read += task.result;
                            }
                        }

                        if (total_read != (ssize_t) chunk.entry->bytes_per_expert) {
                            throw std::runtime_error(format(
                                "failed to split-read expert %d for tensor '%s' from '%s'",
                                chunk.expert, chunk.entry->tensor_name.c_str(), chunk.entry->repacked_path.c_str()));
                        }

                        metrics.install_us += metrics.source_us;
                        if (chunk.direct_dst == nullptr) {
                            const auto upload = upload_expert_bytes(chunk.tensor, chunk.entry, chunk.slot, chunk.bytes.data());
                            metrics.upload_us += upload.upload_us;
                            metrics.install_us += upload.install_us;
                        }
                        add_family_install(metrics, chunk.family, metrics.install_us, metrics.bytes);

                        switch (chunk.family) {
                            case routed_family::gate_up:
                                accumulate_install(metrics, totals.gate_up_install_us, totals.gate_up_bytes);
                                break;
                            case routed_family::gate:
                                accumulate_install(metrics, totals.gate_install_us, totals.gate_bytes);
                                break;
                            case routed_family::up:
                                accumulate_install(metrics, totals.up_install_us, totals.up_bytes);
                                break;
                            case routed_family::down:
                                accumulate_install(metrics, totals.down_install_us, totals.down_bytes);
                                break;
                        }
                    }
                }
            } else {
                std::vector<install_chunk> chunks;
                std::vector<std::future<install_metrics>> futures;
                chunks.reserve(scheduled_loads.size() * 4);
                futures.reserve(scheduled_loads.size() * 4);

                    auto queue_chunk = [&](const pending_slot_load & load, ggml_tensor * tensor, const llama_flash_moe_sidecar_entry * entry, routed_family family) {
                        if (tensor == nullptr || entry == nullptr) {
                            return;
                        }

                    auto & chunk = chunks.emplace_back();
                        chunk.tensor = tensor;
                        chunk.entry = entry;
                        chunk.secondary_entry = use_stripe_for_load(load) ? secondary_entry_for_family(family) : nullptr;
                        chunk.tertiary_entry = use_stripe_for_load(load) ? tertiary_entry_for_family(family) : nullptr;
                        chunk.family = family;
                        chunk.expert = load.expert;
                        chunk.slot = load.slot;
                        chunk.use_striped_read = use_stripe_for_load(load);
                        chunk.direct_dst = force_backend_tensor_writes_enabled() ? nullptr : tensor_slot_cpu_visible_data(tensor, entry, load.slot);
                        if (chunk.direct_dst == nullptr) {
                            chunk.bytes.resize(entry->bytes_per_expert);
                        }

                        const int32_t requested_split = cache_io_split_for_load(load);
                        const bool use_prefetch_stripe = use_prefetch_stripe_for_load(load);
                        futures.emplace_back(std::async(std::launch::async, [this, &chunk, requested_split, use_prefetch_stripe]() {
                        return read_expert_bytes(
                                chunk.tensor,
                                chunk.entry,
                                chunk.secondary_entry,
                                chunk.tertiary_entry,
                                chunk.source_lane,
                                chunk.expert,
                                chunk.direct_dst != nullptr ? chunk.direct_dst : chunk.bytes.data(),
                                requested_split,
                                chunk.use_striped_read,
                                use_prefetch_stripe);
                    }));
                };

                for (const auto & load : scheduled_loads) {
                    const auto & entries = entries_for_load(load);
                    queue_chunk(load, state.gate_up_tensor, entries.gate_up, routed_family::gate_up);
                    queue_chunk(load, state.gate_tensor,    entries.gate,    routed_family::gate);
                    queue_chunk(load, state.up_tensor,      entries.up,      routed_family::up);
                    queue_chunk(load, state.down_tensor,    entries.down,    routed_family::down);
                }

                std::vector<bool> chunk_done(chunks.size(), false);
                size_t remaining = chunks.size();

                auto consume_chunk = [&](size_t idx) {
                    auto metrics = futures[idx].get();
                    if (chunks[idx].direct_dst == nullptr) {
                        const auto upload = upload_expert_bytes(chunks[idx].tensor, chunks[idx].entry, chunks[idx].slot, chunks[idx].bytes.data());
                        metrics.upload_us += upload.upload_us;
                        metrics.install_us += upload.install_us;
                    }
                    add_family_install(metrics, chunks[idx].family, metrics.install_us, metrics.bytes);

                    switch (chunks[idx].family) {
                        case routed_family::gate_up:
                            accumulate_install(metrics, totals.gate_up_install_us, totals.gate_up_bytes);
                            break;
                        case routed_family::gate:
                            accumulate_install(metrics, totals.gate_install_us, totals.gate_bytes);
                            break;
                        case routed_family::up:
                            accumulate_install(metrics, totals.up_install_us, totals.up_bytes);
                            break;
                        case routed_family::down:
                            accumulate_install(metrics, totals.down_install_us, totals.down_bytes);
                            break;
                    }
                    chunk_done[idx] = true;
                    --remaining;
                };

                while (remaining > 0) {
                    bool progressed = false;
                    for (size_t idx = 0; idx < chunks.size(); ++idx) {
                        if (chunk_done[idx]) {
                            continue;
                        }
                        if (futures[idx].wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                            continue;
                        }
                        consume_chunk(idx);
                        progressed = true;
                    }

                    if (progressed) {
                        continue;
                    }

                    for (size_t idx = 0; idx < chunks.size(); ++idx) {
                        if (!chunk_done[idx]) {
                            consume_chunk(idx);
                            break;
                        }
                    }
                }
            }
        } else {
            for (const auto & load : scheduled_loads) {
                const auto & entries = entries_for_load(load);
                const int32_t requested_split = cache_io_split_for_load(load);
                const bool use_striped_read = use_stripe_for_load(load);
                const bool use_prefetch_stripe = use_prefetch_stripe_for_load(load);
                {
                    const auto metrics = load_into_slot(
                            state.gate_up_tensor,
                            entries.gate_up,
                            use_striped_read ? state.secondary_gate_up_entry : nullptr,
                            use_striped_read ? state.tertiary_gate_up_entry : nullptr,
                            source_lane_for_load(load),
                            load.expert,
                            load.slot,
                            requested_split,
                            use_striped_read,
                            use_prefetch_stripe);
                    accumulate_install(metrics, totals.gate_up_install_us, totals.gate_up_bytes);
                }
                {
                    const auto metrics = load_into_slot(
                            state.gate_tensor,
                            entries.gate,
                            use_striped_read ? state.secondary_gate_entry : nullptr,
                            use_striped_read ? state.tertiary_gate_entry : nullptr,
                            source_lane_for_load(load),
                            load.expert,
                            load.slot,
                            requested_split,
                            use_striped_read,
                            use_prefetch_stripe);
                    accumulate_install(metrics, totals.gate_install_us, totals.gate_bytes);
                }
                {
                    const auto metrics = load_into_slot(
                            state.up_tensor,
                            entries.up,
                            use_striped_read ? state.secondary_up_entry : nullptr,
                            use_striped_read ? state.tertiary_up_entry : nullptr,
                            source_lane_for_load(load),
                            load.expert,
                            load.slot,
                            requested_split,
                            use_striped_read,
                            use_prefetch_stripe);
                    accumulate_install(metrics, totals.up_install_us, totals.up_bytes);
                }
                {
                    const auto metrics = load_into_slot(
                            state.down_tensor,
                            entries.down,
                            use_striped_read ? state.secondary_down_entry : nullptr,
                            use_striped_read ? state.tertiary_down_entry : nullptr,
                            source_lane_for_load(load),
                            load.expert,
                            load.slot,
                            requested_split,
                            use_striped_read,
                            use_prefetch_stripe);
                    accumulate_install(metrics, totals.down_install_us, totals.down_bytes);
                }
            }
        }

        for (const auto & load : scheduled_loads) {
            state.slot_to_expert[load.slot] = load.expert;
            state.expert_to_slot[load.expert] = load.slot;
        }

        totals.upload_us += flush_async_uploads();
        totals.install_us = ggml_time_us() - t_install_start_us;
        return totals;
    }

    void write_trace(int layer, int64_t n_expert_used, int64_t n_tokens) {
        if (trace_fp == nullptr) {
            return;
        }

        std::fprintf(trace_fp,
                "{\"seq\":%" PRIu64 ",\"layer\":%d,\"n_expert_used\":%" PRId64 ",\"n_tokens\":%" PRId64 ",\"experts\":[",
                trace_seq++, layer, n_expert_used, n_tokens);

        for (size_t i = 0; i < topk_ids.size(); ++i) {
            std::fprintf(trace_fp, "%s%d", i == 0 ? "" : ",", topk_ids[i]);
        }

        std::fprintf(trace_fp, "],\"slots\":[");
        for (size_t i = 0; i < slot_ids.size(); ++i) {
            std::fprintf(trace_fp, "%s%d", i == 0 ? "" : ",", slot_ids[i]);
        }

        std::fprintf(trace_fp, "]}\n");
        std::fflush(trace_fp);
    }
};

static bool llama_context_flash_moe_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ctx = static_cast<llama_context *>(user_data);
    return ctx->flash_moe_eval_cb(t, ask);
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings       = params.embeddings;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;
    cparams.warmup           = false;
    cparams.n_expert_used    = params.moe_force_expert >= 0 ? 1 : model.moe_n_expert_used();
    cparams.moe_force_expert = params.moe_force_expert;
    cparams.moe_shared_only  = params.moe_shared_only;
    cparams.moe_router_only  = params.moe_router_only;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    if (model.flash_moe_slot_bank_enabled()) {
        flash_moe_slot_runtime = std::make_unique<llama_flash_moe_slot_runtime>(model, !cparams.no_perf);
        flash_moe_cb_eval_downstream = cparams.cb_eval;
        flash_moe_cb_eval_downstream_user_data = cparams.cb_eval_user_data;
        cparams.cb_eval = llama_context_flash_moe_eval_cb;
        cparams.cb_eval_user_data = this;
    }

    if (cparams.moe_shared_only) {
        if (hparams.n_expert_shared == 0 && hparams.n_ff_shexp == 0) {
            LLAMA_LOG_WARN("%s: --moe-shared-only is enabled, but model arch %d does not advertise shared experts; routed MoE will be bypassed anyway, which may not be meaningful\n",
                    __func__, (int) model.arch);
        } else {
            LLAMA_LOG_INFO("%s: --moe-shared-only is enabled; routed experts will be bypassed at graph build time while shared experts remain active\n",
                    __func__);
        }
    }

    if (cparams.moe_router_only) {
        LLAMA_LOG_INFO("%s: --moe-router-only is enabled; routed gating/top-k stays active but routed expert matmuls are bypassed and return zero contribution\n",
                __func__);
    }

    if (cparams.moe_force_expert >= 0) {
        if ((uint32_t) cparams.moe_force_expert >= hparams.n_expert) {
            throw std::invalid_argument(format(
                "--moe-force-expert=%d is out of range for this model (valid routed expert ids: 0..%u)",
                cparams.moe_force_expert, hparams.n_expert > 0 ? hparams.n_expert - 1 : 0));
        }
        LLAMA_LOG_INFO("%s: --moe-force-expert=%d is enabled; routed selection is clamped to expert %d with K=1 for every token\n",
                __func__, cparams.moe_force_expert, cparams.moe_force_expert);
    }

    if (model.flash_moe_temporal_prefetch_enabled()) {
        LLAMA_LOG_WARN("%s: --moe-prefetch-temporal currently refreshes the current token's routed experts after decode to bias slot residency; it is not a future expert predictor in this build\n",
                __func__);
    }
    if (model.flash_moe_temporal_prefetch_sparse_enabled()) {
        LLAMA_LOG_INFO("%s: --moe-prefetch-temporal-sparse is enabled; temporal prefetch alternates even and odd routed layers on successive decode steps\n",
                __func__);
    }
    if (model.flash_moe_predict_prev_token_enabled()) {
        LLAMA_LOG_INFO("%s: --moe-predict-prev-token is enabled; each layer reuses the previous token's full routed expert set as the next-token slot-bank prefetch target\n",
                __func__);
    }
    if (model.flash_moe_predict_top1_prev_enabled()) {
        LLAMA_LOG_INFO("%s: --moe-predict-top1-prev is enabled; each layer reuses only the first previous-token routed expert as the next-token slot-bank prefetch target\n",
                __func__);
    }

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (auto * dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                auto * dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);

            if (!graph_reuse_disable) {
                // TODO: figure out a way to make graph reuse work with pipeline parallelism
                // ref: https://github.com/ggml-org/llama.cpp/pull/20463
                LLAMA_LOG_WARN("%s: graph reuse is currently not compatible with pipeline parallelism - disabling\n", __func__);

                graph_reuse_disable = true;
            }
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    // resolve automatic Flash Attention use
    if (cparams.auto_fa) {
        auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
        if (!gf) {
            throw std::runtime_error("failed to reserve graph for Flash Attention check");
        }

        const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FATTN) + 1;
        bool fa_device_mismatch = false;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }
            ggml_backend_dev_t device_fa = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

            // TODO: instead of the tensor names, use a map to keep track of which (FA) tensors belong to which layer
            GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FATTN "-", prefix_len) == 0);
            const int il = std::stoi(n->name + prefix_len);
            ggml_backend_dev_t device_kv = model.dev_layer(il);
            if (device_fa != device_kv) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
        }

        if (fa_device_mismatch) {
            cparams.flash_attn = false;
            LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }

        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", __func__);

        if (cparams.fused_gdn_ar) {
            auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (autoregressive)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_AR) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_AR "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ar = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (autoregressive) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (autoregressive) enabled\n", __func__);
            }
        }

        if (cparams.fused_gdn_ch) {
            // more than one token in the batch per sequence in order to take the chunked path
            // note: n_outputs must match n_tokens for embedding models with mean/rank pooling,
            // because build_pooling creates inp_mean with shape [n_tokens, n_seqs] and multiplies
            // it with t_embd which is reduced to [n_outputs, ...] via out_ids. if n_outputs != n_tokens,
            // the ggml_mul_mat assertion fails. this matches the pp reservation below (line ~553).
            const uint32_t n_tokens_ch = 16*n_seqs;
            auto * gf = graph_reserve(n_tokens_ch, n_seqs, n_tokens_ch, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (chunked)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_CH) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_CH "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ch = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (chunked) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (chunked) enabled\n", __func__);
            }
        }

        cparams.auto_fgdn = false;
    }

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worster case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (value && model.flash_moe_slot_bank_enabled()) {
        LLAMA_LOG_INFO("%s: Flash-MoE slot-bank mode keeps warmup on the routed top-k path\n", __func__);
        return;
    }

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        n_reused++;
    } else {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }

        flash_moe_log_routed_backends(gf, sched.get());
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    const int64_t n_embd  = hparams.n_embd_inp();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits = res->get_logits();
    auto * t_embd = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_embd);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_idx)*sizeof(float), n_embd*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<float> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

static bool flash_moe_backend_trace_enabled() {
    static const bool enabled = []() {
        const char * val = getenv("LLAMA_FLASH_MOE_BACKEND_TRACE");
        return val != nullptr && val[0] != '\0' && strcmp(val, "0") != 0;
    }();
    return enabled;
}

static void flash_moe_log_routed_backends(struct ggml_cgraph * gf, ggml_backend_sched_t sched) {
    if (!flash_moe_backend_trace_enabled() || gf == nullptr || sched == nullptr) {
        return;
    }

    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (node == nullptr || node->op != GGML_OP_MUL_MAT_ID) {
            continue;
        }

        ggml_backend_t node_backend = ggml_backend_sched_get_tensor_backend(sched, node);
        const char * node_backend_name = node_backend ? ggml_backend_name(node_backend) : "NULL";

        const ggml_tensor * src0 = node->src[0];
        const ggml_tensor * src1 = node->src[1];
        const ggml_tensor * src2 = node->src[2];

        ggml_backend_t src0_backend = src0 ? ggml_backend_sched_get_tensor_backend(sched, const_cast<ggml_tensor *>(src0)) : nullptr;
        ggml_backend_t src1_backend = src1 ? ggml_backend_sched_get_tensor_backend(sched, const_cast<ggml_tensor *>(src1)) : nullptr;
        ggml_backend_t src2_backend = src2 ? ggml_backend_sched_get_tensor_backend(sched, const_cast<ggml_tensor *>(src2)) : nullptr;

        LLAMA_LOG_INFO("%s: node=%s backend=%s src0=%s src1=%s src2=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                __func__,
                node->name,
                node_backend_name,
                src0_backend ? ggml_backend_name(src0_backend) : "NULL",
                src1_backend ? ggml_backend_name(src1_backend) : "NULL",
                src2_backend ? ggml_backend_name(src2_backend) : "NULL",
                node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
    }
}

int llama_context::decode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;
        const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        if (flash_moe_slot_runtime) {
            flash_moe_slot_runtime->temporal_prefetch_after_decode();
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits = res->get_logits();
        auto * t_embd   = cparams.embeddings ? res->get_embd() : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd*seq_idx)*sizeof(float), n_embd*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers && (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched.get());

            copy_tensor_async_floats    (res->t_sampled_logits, sampling.logits,     stride, sampling.logits_count,     seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_probs,  sampling.probs,      stride, sampling.probs_count,      seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates,     sampling.candidates, stride, sampling.candidates_count, seq_to_output_row, sched.get());
        }

        n_outputs_prev += n_outputs;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits = true;
    bool has_embd   = cparams.embeddings;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }


    size_t backend_float_count = 0;
    size_t backend_token_count = 0;

    logits.size = has_logits ? n_vocab*n_outputs_max : 0;
    embd.size   = has_embd ? n_embd_out*n_outputs_max : 0;

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + backend_float_count) * sizeof(float) +
        (                          backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    return n_outputs_max;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd.data[i0*n_embd + k], embd.data[i1*n_embd + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        n_outputs = std::max(n_outputs, n_tokens);

        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, LLM_GRAPH_TYPE_DEFAULT);

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    flash_moe_log_routed_backends(gf, sched.get());

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.samplers    =*/ sampling.samplers,
        /*.flash_moe_slot_runtime =*/ flash_moe_slot_runtime.get(),
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

bool llama_context::flash_moe_eval_cb(ggml_tensor * t, bool ask) {
    const bool internal_need = flash_moe_slot_runtime && flash_moe_slot_runtime->wants_tensor(t);
    const bool downstream_need = flash_moe_cb_eval_downstream ?
        flash_moe_cb_eval_downstream(t, ask, flash_moe_cb_eval_downstream_user_data) : false;

    if (ask) {
        return internal_need || downstream_need;
    }

    bool ok = true;
    if (internal_need) {
        ok = flash_moe_slot_runtime->handle_tensor(t);
    }
    if (flash_moe_cb_eval_downstream) {
        ok = ok && flash_moe_cb_eval_downstream(t, ask, flash_moe_cb_eval_downstream_user_data);
    }

    return ok;
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    ggml_status status = GGML_STATUS_SUCCESS;
    try {
        status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
        if (status != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: Flash-MoE graph compute failed: %s\n", __func__, err.what());
        status = GGML_STATUS_FAILED;
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }

        if (il != -1 && model.flash_moe_slot_bank_enabled() &&
                (strcmp(name, "ffn_moe_slot_ids") == 0 || strcmp(name, "ffn_moe_slot_ids_native") == 0)) {
            const auto & dev_layer = model.dev_layer(il);
            for (const auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == dev_layer) {
                    ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy() = default;

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(const ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    size_t size_written = 0;
};

class llama_io_write_buffer : public llama_io_write_i {
public:
    llama_io_write_buffer(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;
};

class llama_io_read_buffer : public llama_io_read_i {
public:
    llama_io_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(const ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io;
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_buffer io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io;
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    llama_io_write_buffer io(dst, size);
    try {
        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    llama_io_read_buffer io(src, size);
    try {
        return state_seq_read_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DEFAULT);

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.moe_force_expert            =*/ -1,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.moe_shared_only             =*/ false,
        /*.moe_router_only             =*/ false,
        /*.samplers                    =*/ nullptr,
        /*.n_samplers                  =*/ 0,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto * memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    return ctx->get_memory();
}

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}

size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

void llama_memory_breakdown_print(const struct llama_context * ctx) {
    const std::vector<ggml_backend_dev_t> & devices = ctx->get_model().devices;

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown = ctx->memory_breakdown();

    std::vector<std::array<std::string, 9>> table_data;
    table_data.reserve(devices.size());
    const std::string template_header = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
    const std::string template_gpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
    const std::string template_other  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

    table_data.push_back({template_header, "memory breakdown [MiB]", "total", "free", "self", "model", "context", "compute", "unaccounted"});

    constexpr size_t MiB = 1024 * 1024;
    const std::vector<std::string> desc_prefixes_strip = {"NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "};

    // track seen buffer types to avoid double counting:
    std::set<ggml_backend_buffer_type_t> seen_buffer_types;

    // accumulative memory breakdown for each device and for host:
    std::vector<llama_memory_breakdown_data> mb_dev(devices.size());
    llama_memory_breakdown_data              mb_host;

    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (ggml_backend_buft_is_host(buft)) {
            mb_host.model   += mb.model;
            mb_host.context += mb.context;
            mb_host.compute += mb.compute;
            seen_buffer_types.insert(buft);
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (dev) {
            int i_dev = -1;
            for (size_t i = 0; i < devices.size(); i++) {
                if (devices[i] == dev) {
                    i_dev = i;
                    break;
                }
            }
            if (i_dev != -1) {
                mb_dev[i_dev].model   += mb.model;
                mb_dev[i_dev].context += mb.context;
                mb_dev[i_dev].compute += mb.compute;
                seen_buffer_types.insert(buft);
                continue;
            }
        }
    }

    // print memory breakdown for each device:
    for (size_t i = 0; i < devices.size(); i++) {
        ggml_backend_dev_t          dev = devices[i];
        llama_memory_breakdown_data mb  = mb_dev[i];

        const std::string name = ggml_backend_dev_name(dev);
        std::string desc = ggml_backend_dev_description(dev);
        for (const std::string & prefix : desc_prefixes_strip) {
            if (desc.length() >= prefix.length() && desc.substr(0, prefix.length()) == prefix) {
                desc = desc.substr(prefix.length());
            }
        }

        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);

        const size_t self = mb.model + mb.context + mb.compute;
        const size_t unaccounted = total - self - free;

        table_data.push_back({
            template_gpu,
            "  - " + name + " (" + desc + ")",
            std::to_string(total / MiB),
            std::to_string(free / MiB),
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            std::to_string(unaccounted / MiB)});
    }

    // print memory breakdown for host:
    {
        const size_t self = mb_host.model + mb_host.context + mb_host.compute;
        table_data.push_back({
            template_other,
            "  - Host",
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb_host.model / MiB),
            std::to_string(mb_host.context / MiB),
            std::to_string(mb_host.compute / MiB),
            ""}); // unaccounted
    }

    // print memory breakdown for all remaining buffer types:
    for (const auto & buft_mb : memory_breakdown) {
        ggml_backend_buffer_type_t          buft = buft_mb.first;
        const llama_memory_breakdown_data & mb   = buft_mb.second;
        if (seen_buffer_types.count(buft) == 1) {
            continue;
        }
        const std::string name = ggml_backend_buft_name(buft);
        const size_t self = mb.model + mb.context + mb.compute;
        table_data.push_back({
            template_other,
            "  - " + name,
            "", // total
            "", // free
            std::to_string(self / MiB),
            std::to_string(mb.model / MiB),
            std::to_string(mb.context / MiB),
            std::to_string(mb.compute / MiB),
            ""}); // unaccounted
        seen_buffer_types.insert(buft);
    }

    for (size_t j = 1; j < table_data[0].size(); j++) {
        size_t max_len = 0;
        for (const auto & td : table_data) {
            max_len = std::max(max_len, td[j].length());
        }
        for (auto & td : table_data) {
            td[j].insert(j == 1 ? td[j].length() : 0, max_len - td[j].length(), ' ');
        }
    }
    for (const auto & td : table_data) {
        LLAMA_LOG_INFO(td[0].c_str(),
            __func__, td[1].c_str(), td[2].c_str(), td[3].c_str(), td[4].c_str(), td[5].c_str(),
            td[6].c_str(), td[7].c_str(), td[8].c_str());
    }
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}
