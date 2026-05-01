#include "llama-memory-hybrid-iswa.h"

#include "ggml-backend.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint32_t DSV4_COMPRESSED_KV_STATE_MAGIC   = 0x44535634; // "DSV4"
constexpr uint32_t DSV4_COMPRESSED_KV_STATE_VERSION = 1;
constexpr uint32_t DSV4_COMPRESSED_DECODE_UBATCH_MAX = 64;
constexpr uint32_t DSV4_COMPRESSED_CONT_PREFILL_UBATCH_MAX = 8192;
constexpr uint64_t DSV4_COMPRESSED_CONT_PREFILL_WORK_MAX = 64ull*1024ull*1024ull;

static bool dsv4_split_trace_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_FLASH_MOE_DEBUG_DSV4_SPLIT");
        enabled = value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 ? 1 : 0;
    }

    return enabled == 1;
}

static uint32_t dsv4_cont_prefill_ubatch_max() {
    static uint32_t value = 0;
    static bool initialized = false;
    if (!initialized) {
        value = DSV4_COMPRESSED_CONT_PREFILL_UBATCH_MAX;

        const char * env = std::getenv("LLAMA_DSV4_CONT_PREFILL_UBATCH_MAX");
        if (env != nullptr && env[0] != '\0') {
            char * end = nullptr;
            const unsigned long parsed = std::strtoul(env, &end, 10);
            if (end != env && parsed > 0 && parsed <= std::numeric_limits<uint32_t>::max()) {
                value = (uint32_t) parsed;
            }
        }

        initialized = true;
    }

    return value;
}

static uint64_t dsv4_cont_prefill_work_max() {
    static uint64_t value = 0;
    static bool initialized = false;
    if (!initialized) {
        value = DSV4_COMPRESSED_CONT_PREFILL_WORK_MAX;

        const char * env = std::getenv("LLAMA_DSV4_CONT_PREFILL_WORK_MAX");
        if (env != nullptr && env[0] != '\0') {
            char * end = nullptr;
            const unsigned long long parsed = std::strtoull(env, &end, 10);
            if (end != env && parsed > 0) {
                value = parsed;
            }
        }

        initialized = true;
    }

    return value;
}

static bool dsv4_align_long_prefill_enabled() {
    static int enabled = -1;
    if (enabled == -1) {
        const char * value = std::getenv("LLAMA_DSV4_ALIGN_LONG_PREFILL");
        enabled = value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 ? 1 : 0;
    }

    return enabled == 1;
}

static uint32_t dsv4_cont_prefill_adaptive_ubatch_max(
        llama_pos split_pos,
        uint32_t  n_ubatch,
        uint32_t  min_compress_ratio,
        uint32_t  alignment) {
    uint32_t cap = std::min<uint32_t>(n_ubatch, dsv4_cont_prefill_ubatch_max());
    if (split_pos <= 0 || min_compress_ratio == 0 || cap <= DSV4_COMPRESSED_DECODE_UBATCH_MAX) {
        return cap;
    }

    const uint64_t ratio = std::max<uint32_t>(1, min_compress_ratio);
    const uint64_t visible_rows = (uint64_t(split_pos) + cap + ratio - 1)/ratio;
    const uint64_t work_max = dsv4_cont_prefill_work_max();

    if (visible_rows > 0 && uint64_t(cap)*visible_rows > work_max) {
        uint64_t adaptive = std::max<uint64_t>(
                DSV4_COMPRESSED_DECODE_UBATCH_MAX,
                work_max/visible_rows);
        if (alignment > 1 && adaptive >= alignment) {
            adaptive = std::max<uint64_t>(alignment, (adaptive/alignment)*alignment);
        }
        cap = std::min<uint32_t>(cap, uint32_t(std::min<uint64_t>(adaptive, std::numeric_limits<uint32_t>::max())));
    }

    return std::max<uint32_t>(1, cap);
}

static bool dsv4_batch_is_long_single_seq_prefill(const llama_batch & batch) {
    if (batch.n_tokens <= (int32_t) DSV4_COMPRESSED_DECODE_UBATCH_MAX ||
        batch.pos == nullptr ||
        batch.n_seq_id == nullptr ||
        batch.seq_id == nullptr ||
        batch.seq_id[0] == nullptr ||
        batch.n_seq_id[0] != 1) {
        return false;
    }

    const llama_seq_id seq_id = batch.seq_id[0][0];
    const llama_pos pos0 = batch.pos[0];

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i] == nullptr || batch.seq_id[i][0] != seq_id) {
            return false;
        }
        if (batch.pos[i] != pos0 + i) {
            return false;
        }
    }

    return true;
}

struct dsv4_row_range {
    uint32_t begin = 0;
    uint32_t end   = 0;

    uint32_t size() const {
        GGML_ASSERT(end >= begin);
        return end - begin;
    }
};

static dsv4_row_range dsv4_make_row_range(uint32_t n_comp, uint32_t ratio, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(ratio > 0);

    if (n_comp == 0) {
        return {};
    }

    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    if (p0 >= p1) {
        return {};
    }

    const uint64_t row_begin = (uint64_t) p0 / ratio;
    uint64_t row_end;
    if (p1 == std::numeric_limits<llama_pos>::max()) {
        row_end = n_comp;
    } else {
        row_end = ((uint64_t) p1 + ratio - 1) / ratio;
    }

    return {
        (uint32_t) std::min<uint64_t>(row_begin, n_comp),
        (uint32_t) std::min<uint64_t>(row_end,   n_comp),
    };
}

static size_t dsv4_cache_row_size(const ggml_tensor * t) {
    GGML_ASSERT(t != nullptr);

    const size_t row_size = ggml_row_size(t->type, t->ne[0]);
    GGML_ASSERT((size_t) t->nb[1] == row_size);
    GGML_ASSERT((size_t) t->nb[2] == row_size*(size_t) t->ne[1]);

    return row_size;
}

static size_t dsv4_cache_offset(const ggml_tensor * t, llama_seq_id seq_id, uint32_t row) {
    GGML_ASSERT(seq_id >= 0);
    GGML_ASSERT(row <= (uint32_t) t->ne[1]);

    return (size_t) seq_id*(size_t) t->nb[2] + (size_t) row*(size_t) t->nb[1];
}

static void dsv4_zero_cache_rows(ggml_tensor * t, llama_seq_id seq_id, uint32_t row_start, uint32_t n_rows) {
    if (t == nullptr || n_rows == 0) {
        return;
    }

    const size_t row_size = dsv4_cache_row_size(t);
    const size_t n_bytes  = (size_t) n_rows*row_size;
    const size_t offset   = dsv4_cache_offset(t, seq_id, row_start);

    thread_local std::vector<uint8_t> zeros;
    if (zeros.size() < n_bytes) {
        zeros.resize(n_bytes);
    }
    std::fill(zeros.begin(), zeros.begin() + n_bytes, 0);
    ggml_backend_tensor_set(t, zeros.data(), offset, n_bytes);
}

static void dsv4_copy_cache_rows(ggml_tensor * t, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, uint32_t row_start, uint32_t n_rows) {
    if (t == nullptr || n_rows == 0 || seq_id_src == seq_id_dst) {
        return;
    }

    const size_t row_size   = dsv4_cache_row_size(t);
    const size_t n_bytes    = (size_t) n_rows*row_size;
    const size_t src_offset = dsv4_cache_offset(t, seq_id_src, row_start);
    const size_t dst_offset = dsv4_cache_offset(t, seq_id_dst, row_start);

    thread_local std::vector<uint8_t> tmp;
    if (tmp.size() < n_bytes) {
        tmp.resize(n_bytes);
    }
    ggml_backend_tensor_get(t, tmp.data(), src_offset, n_bytes);
    ggml_backend_tensor_set(t, tmp.data(), dst_offset, n_bytes);
}

} // namespace

//
// llama_memory_hybrid_iswa
//

llama_memory_hybrid_iswa::llama_memory_hybrid_iswa(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr) :
    hparams(model.hparams),
    mem_attn(new llama_kv_cache_iswa(
        model,
        type_k,
        type_v,
        v_trans,
        offload,
        swa_full,
        unified,
        kv_size,
        n_seq_max,
        n_ubatch,
        n_pad,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recurrent(il); }
            : filter_attn,
        nullptr
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recurrent(il); }
            : filter_recr
    )) {
    if (model.arch != LLM_ARCH_DEEPSEEK4) {
        return;
    }

    dsv4_n_seq_max = n_seq_max;
    dsv4_cache_layers.resize(hparams.n_layer);

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*hparams.n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);
            return ctx;
        }

        return it->second.get();
    };

    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        const uint32_t ratio = hparams.attn_compress_ratio[il];
        if (ratio == 0) {
            continue;
        }

        const uint32_t n_comp = std::max<uint32_t>(1, (kv_size + ratio - 1) / ratio);

        const char * dev_name = "CPU";
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);
            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: DeepSeek4 compressed KV layer %3d: dev = %s, ratio = %u, rows = %u\n",
                __func__, il, dev_name, ratio, n_comp);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for DeepSeek4 compressed KV cache");
        }

        auto & cache = dsv4_cache_layers[il];
        cache.n_comp = n_comp;
        cache.attn_k = ggml_new_tensor_3d(ctx, type_k, hparams.n_embd_head_k(il), n_comp, dsv4_n_seq_max);
        ggml_format_name(cache.attn_k, "cache_dsv4_attn_k_l%d", il);

        if (ratio == 4) {
            cache.index_k = ggml_new_tensor_3d(ctx, type_k, hparams.indexer_head_size, n_comp, dsv4_n_seq_max);
            ggml_format_name(cache.index_k, "cache_dsv4_index_k_l%d", il);
        }
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (model.hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, 0);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate DeepSeek4 compressed KV cache buffer");
        }

        LLAMA_LOG_INFO("%s: %10s DeepSeek4 compressed KV buffer size = %8.2f MiB\n", __func__,
                ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        dsv4_ctxs_bufs.emplace_back(std::move(ctx), buf);
    }
}

llama_memory_context_ptr llama_memory_hybrid_iswa::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    const bool dsv4_compressed = has_dsv4_compressed_kv();
    uint32_t dsv4_prefill_alignment = 1;
    uint32_t dsv4_prefill_min_compress_ratio = 0;
    if (dsv4_compressed) {
        for (uint32_t il = 0; il < hparams.n_layer; ++il) {
            const uint32_t ratio = hparams.attn_compress_ratio[il];
            if (ratio == 0) {
                continue;
            }
            dsv4_prefill_alignment = std::max<uint32_t>(dsv4_prefill_alignment, ratio);
            dsv4_prefill_min_compress_ratio = dsv4_prefill_min_compress_ratio == 0 ?
                    ratio : std::min<uint32_t>(dsv4_prefill_min_compress_ratio, ratio);
        }
    }

    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (dsv4_compressed) {
                // DeepSeek V4 compressed attention keeps sequence-local compressor
                // state and compressed cache rows. Process one sequence set per
                // ubatch while still allowing multi-sequence batches at the API
                // level.
                uint32_t n_ubatch_dsv4 = n_ubatch;
                const auto & batch = balloc.get_batch();
                const bool first_split = balloc.get_n_used() == 0;
                const bool starts_at_zero = batch.pos == nullptr || batch.pos[0] == 0;
                const bool long_single_seq_prefill = dsv4_batch_is_long_single_seq_prefill(batch);
                llama_pos split_pos = batch.pos != nullptr && balloc.get_n_used() < balloc.get_n_tokens() ?
                        batch.pos[balloc.get_n_used()] :
                        (llama_pos) balloc.get_n_used();
                if (long_single_seq_prefill && dsv4_prefill_alignment > 1 && split_pos % dsv4_prefill_alignment != 0) {
                    const uint32_t align_tokens = dsv4_prefill_alignment - uint32_t(split_pos % dsv4_prefill_alignment);
                    n_ubatch_dsv4 = std::min<uint32_t>(n_ubatch_dsv4, std::min<uint32_t>(align_tokens, DSV4_COMPRESSED_DECODE_UBATCH_MAX));
                } else if (long_single_seq_prefill && split_pos > 0) {
                    // Continuation prefill attends over already-visible compressed
                    // cache rows, so cap the slice by both token count and the
                    // token*visible-compressed-row work area used by DSV4 masks.
                    n_ubatch_dsv4 = dsv4_cont_prefill_adaptive_ubatch_max(
                            split_pos,
                            n_ubatch_dsv4,
                            dsv4_prefill_min_compress_ratio,
                            dsv4_prefill_alignment);
                } else if (!long_single_seq_prefill && (!first_split || !starts_at_zero)) {
                    // Decode-like compressed-attention chunks build one compressor
                    // update per token and can otherwise exhaust the graph metadata
                    // arena on long contexts. Long contiguous prompt continuations
                    // are still prefill, even when their first position is > 0; if
                    // they start mid-compression-window, split a small prefix first.
                    n_ubatch_dsv4 = std::min<uint32_t>(n_ubatch_dsv4, DSV4_COMPRESSED_DECODE_UBATCH_MAX);
                }
                if (long_single_seq_prefill &&
                    dsv4_align_long_prefill_enabled() &&
                    dsv4_prefill_alignment > 1 &&
                    split_pos % dsv4_prefill_alignment == 0) {
                    const uint32_t remaining = balloc.get_n_tokens() - balloc.get_n_used();
                    const uint32_t desired = std::min<uint32_t>(n_ubatch_dsv4, remaining);
                    if (desired > dsv4_prefill_alignment) {
                        uint32_t aligned = desired - desired % dsv4_prefill_alignment;
                        if (aligned == 0) {
                            aligned = dsv4_prefill_alignment;
                        }
                        if (aligned < desired || (desired == remaining && remaining % dsv4_prefill_alignment != 0)) {
                            n_ubatch_dsv4 = aligned;
                        }
                    }
                }
                if (dsv4_split_trace_enabled()) {
                    LLAMA_LOG_INFO("%s: dsv4 split n_batch=%d n_used=%u n_ubatch=%u effective=%u pos0=%d split_pos=%d align=%u first=%d start0=%d long_prefill=%d\n",
                            __func__,
                            batch.n_tokens,
                            balloc.get_n_used(),
                            n_ubatch,
                            n_ubatch_dsv4,
                            batch.pos != nullptr ? batch.pos[0] : -1,
                            split_pos,
                            dsv4_prefill_alignment,
                            first_split ? 1 : 0,
                            starts_at_zero ? 1 : 0,
                            long_single_seq_prefill ? 1 : 0);
                }
                ubatch = balloc.split_seq(n_ubatch_dsv4);
            } else if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_base()->get_n_stream() == 1);
                ubatch = balloc.split_equal(n_ubatch, !unified);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache (iswa version returns both base and swa slot infos)
        auto sinfos_base = mem_attn->get_base()->prepare(ubatches);
        if (sinfos_base.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention base ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        auto sinfos_swa = mem_attn->get_swa()->prepare(ubatches);
        if (sinfos_swa.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention swa ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_iswa::init_full() {
    return std::make_unique<llama_memory_hybrid_iswa_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_iswa::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_iswa_context>(this, lctx, optimize);
}

bool llama_memory_hybrid_iswa::get_can_shift() const {
    if (has_dsv4_compressed_kv()) {
        return false;
    }

    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid_iswa::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);

    if (data) {
        for (auto & [_, buf] : dsv4_ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_memory_hybrid_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    if (!mem_attn->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    dsv4_seq_rm(seq_id, p0, p1);
    return true;
}

void llama_memory_hybrid_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    dsv4_seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid_iswa::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
    dsv4_seq_keep(seq_id);
}

void llama_memory_hybrid_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (has_dsv4_compressed_kv() && shift != 0) {
        GGML_ABORT("DeepSeek V4 compressed KV cache does not support K-shift");
    }

    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (has_dsv4_compressed_kv() && d != 1) {
        GGML_ABORT("DeepSeek V4 compressed KV cache does not support position division");
    }

    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid_iswa::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & [_, buf] : dsv4_ctxs_bufs) {
        mb[ggml_backend_buffer_get_type(buf.get())] += ggml_backend_buffer_get_size(buf.get());
    }
    return mb;
}

void llama_memory_hybrid_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    mem_attn->state_write(io, seq_id, flags);
    mem_recr->state_write(io, seq_id, flags);
    dsv4_state_write(io, seq_id);
}

void llama_memory_hybrid_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    mem_attn->state_read(io, seq_id, flags);
    mem_recr->state_read(io, seq_id, flags);
    dsv4_state_read(io, seq_id);
}

void llama_memory_hybrid_iswa::dsv4_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!has_dsv4_compressed_kv()) {
        return;
    }

    if (seq_id >= 0) {
        GGML_ASSERT((uint32_t) seq_id < dsv4_n_seq_max);
        for (int32_t il = 0; il < (int32_t) dsv4_cache_layers.size(); ++il) {
            dsv4_clear_rows(seq_id, il, p0, p1);
        }
        return;
    }

    for (uint32_t seq = 0; seq < dsv4_n_seq_max; ++seq) {
        for (int32_t il = 0; il < (int32_t) dsv4_cache_layers.size(); ++il) {
            dsv4_clear_rows(seq, il, p0, p1);
        }
    }
}

void llama_memory_hybrid_iswa::dsv4_seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (!has_dsv4_compressed_kv() || seq_id_src == seq_id_dst) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (uint32_t) seq_id_src < dsv4_n_seq_max);
    GGML_ASSERT(seq_id_dst >= 0 && (uint32_t) seq_id_dst < dsv4_n_seq_max);

    for (int32_t il = 0; il < (int32_t) dsv4_cache_layers.size(); ++il) {
        dsv4_copy_rows(seq_id_src, seq_id_dst, il, p0, p1);
    }
}

void llama_memory_hybrid_iswa::dsv4_seq_keep(llama_seq_id seq_id) {
    if (!has_dsv4_compressed_kv()) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);

    for (uint32_t seq = 0; seq < dsv4_n_seq_max; ++seq) {
        if ((llama_seq_id) seq == seq_id) {
            continue;
        }

        dsv4_clear_seq(seq);
    }
}

void llama_memory_hybrid_iswa::dsv4_clear_seq(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);

    for (const auto & layer : dsv4_cache_layers) {
        dsv4_zero_cache_rows(layer.attn_k,  seq_id, 0, layer.n_comp);
        dsv4_zero_cache_rows(layer.index_k, seq_id, 0, layer.n_comp);
    }
}

void llama_memory_hybrid_iswa::dsv4_clear_rows(llama_seq_id seq_id, int32_t il, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());

    const uint32_t ratio = hparams.attn_compress_ratio[il];
    if (ratio == 0) {
        return;
    }

    const auto & layer = dsv4_cache_layers[il];
    const auto range = dsv4_make_row_range(layer.n_comp, ratio, p0, p1);

    dsv4_zero_cache_rows(layer.attn_k,  seq_id, range.begin, range.size());
    dsv4_zero_cache_rows(layer.index_k, seq_id, range.begin, range.size());
}

void llama_memory_hybrid_iswa::dsv4_copy_rows(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, int32_t il, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id_src >= 0 && (uint32_t) seq_id_src < dsv4_n_seq_max);
    GGML_ASSERT(seq_id_dst >= 0 && (uint32_t) seq_id_dst < dsv4_n_seq_max);
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());

    const uint32_t ratio = hparams.attn_compress_ratio[il];
    if (ratio == 0) {
        return;
    }

    const auto & layer = dsv4_cache_layers[il];
    const auto range = dsv4_make_row_range(layer.n_comp, ratio, p0, p1);

    dsv4_copy_cache_rows(layer.attn_k,  seq_id_src, seq_id_dst, range.begin, range.size());
    dsv4_copy_cache_rows(layer.index_k, seq_id_src, seq_id_dst, range.begin, range.size());
}

uint32_t llama_memory_hybrid_iswa::dsv4_n_state_rows(int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());

    const uint32_t ratio = hparams.attn_compress_ratio[il];
    if (ratio == 0) {
        return 0;
    }

    const llama_pos pos_max = mem_attn->seq_pos_max(seq_id);
    if (pos_max < 0) {
        return 0;
    }

    const uint64_t n_rows = ((uint64_t) pos_max + 1) / ratio;
    return (uint32_t) std::min<uint64_t>(n_rows, dsv4_cache_layers[il].n_comp);
}

void llama_memory_hybrid_iswa::dsv4_state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    if (!has_dsv4_compressed_kv()) {
        return;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max));

    std::vector<llama_seq_id> seq_ids;
    auto seq_has_rows = [&](llama_seq_id seq) {
        for (int32_t il = 0; il < (int32_t) dsv4_cache_layers.size(); ++il) {
            if (dsv4_n_state_rows(il, seq) > 0) {
                return true;
            }
        }
        return false;
    };

    if (seq_id >= 0) {
        if (seq_has_rows(seq_id)) {
            seq_ids.push_back(seq_id);
        }
    } else {
        for (uint32_t seq = 0; seq < dsv4_n_seq_max; ++seq) {
            if (seq_has_rows(seq)) {
                seq_ids.push_back(seq);
            }
        }
    }

    const uint32_t magic   = DSV4_COMPRESSED_KV_STATE_MAGIC;
    const uint32_t version = DSV4_COMPRESSED_KV_STATE_VERSION;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_seq   = seq_ids.size();

    io.write(&magic,   sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&n_layer, sizeof(n_layer));
    io.write(&n_seq,   sizeof(n_seq));

    for (uint32_t il = 0; il < n_layer; ++il) {
        const auto & layer = dsv4_cache_layers[il];

        const uint32_t n_comp = layer.n_comp;
        io.write(&n_comp, sizeof(n_comp));

        const uint32_t has_attn = layer.attn_k != nullptr;
        io.write(&has_attn, sizeof(has_attn));
        if (has_attn) {
            const int32_t  type_i   = (int32_t) layer.attn_k->type;
            const uint64_t row_size = dsv4_cache_row_size(layer.attn_k);
            io.write(&type_i,   sizeof(type_i));
            io.write(&row_size, sizeof(row_size));
        }

        const uint32_t has_index = layer.index_k != nullptr;
        io.write(&has_index, sizeof(has_index));
        if (has_index) {
            const int32_t  type_i   = (int32_t) layer.index_k->type;
            const uint64_t row_size = dsv4_cache_row_size(layer.index_k);
            io.write(&type_i,   sizeof(type_i));
            io.write(&row_size, sizeof(row_size));
        }
    }

    for (llama_seq_id seq : seq_ids) {
        io.write(&seq, sizeof(seq));

        for (uint32_t il = 0; il < n_layer; ++il) {
            const auto & layer = dsv4_cache_layers[il];
            const uint32_t n_rows = dsv4_n_state_rows(il, seq);

            if (layer.attn_k != nullptr) {
                const uint64_t row_size = dsv4_cache_row_size(layer.attn_k);
                io.write(&n_rows, sizeof(n_rows));
                if (n_rows > 0) {
                    io.write_tensor(layer.attn_k, dsv4_cache_offset(layer.attn_k, seq, 0), (size_t) n_rows*row_size);
                }
            }

            if (layer.index_k != nullptr) {
                const uint64_t row_size = dsv4_cache_row_size(layer.index_k);
                io.write(&n_rows, sizeof(n_rows));
                if (n_rows > 0) {
                    io.write_tensor(layer.index_k, dsv4_cache_offset(layer.index_k, seq, 0), (size_t) n_rows*row_size);
                }
            }
        }
    }
}

void llama_memory_hybrid_iswa::dsv4_state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    if (!has_dsv4_compressed_kv()) {
        return;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max));

    uint32_t magic;
    uint32_t version;
    uint32_t n_layer;
    uint32_t n_seq;

    io.read_to(&magic,   sizeof(magic));
    io.read_to(&version, sizeof(version));
    io.read_to(&n_layer, sizeof(n_layer));
    io.read_to(&n_seq,   sizeof(n_seq));

    if (magic != DSV4_COMPRESSED_KV_STATE_MAGIC) {
        throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: bad magic");
    }
    if (version != DSV4_COMPRESSED_KV_STATE_VERSION) {
        throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: bad version");
    }
    if (n_layer != hparams.n_layer || n_layer != dsv4_cache_layers.size()) {
        throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: mismatched layer count");
    }

    struct layer_meta {
        uint32_t n_comp = 0;
        bool has_attn = false;
        int32_t attn_type = -1;
        uint64_t attn_row_size = 0;
        bool has_index = false;
        int32_t index_type = -1;
        uint64_t index_row_size = 0;
    };

    std::vector<layer_meta> meta(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        auto & m = meta[il];
        const auto & layer = dsv4_cache_layers[il];

        io.read_to(&m.n_comp, sizeof(m.n_comp));

        uint32_t has_attn;
        io.read_to(&has_attn, sizeof(has_attn));
        m.has_attn = has_attn != 0;
        if (m.has_attn) {
            io.read_to(&m.attn_type,     sizeof(m.attn_type));
            io.read_to(&m.attn_row_size, sizeof(m.attn_row_size));
        }

        uint32_t has_index;
        io.read_to(&has_index, sizeof(has_index));
        m.has_index = has_index != 0;
        if (m.has_index) {
            io.read_to(&m.index_type,     sizeof(m.index_type));
            io.read_to(&m.index_row_size, sizeof(m.index_row_size));
        }

        const bool local_has_attn  = layer.attn_k  != nullptr;
        const bool local_has_index = layer.index_k != nullptr;

        if (m.n_comp != layer.n_comp || m.has_attn != local_has_attn || m.has_index != local_has_index) {
            throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: mismatched layer layout");
        }
        if (local_has_attn) {
            const int32_t  type_i   = (int32_t) layer.attn_k->type;
            const uint64_t row_size = dsv4_cache_row_size(layer.attn_k);
            if (m.attn_type != type_i || m.attn_row_size != row_size) {
                throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: mismatched attention cache type");
            }
        }
        if (local_has_index) {
            const int32_t  type_i   = (int32_t) layer.index_k->type;
            const uint64_t row_size = dsv4_cache_row_size(layer.index_k);
            if (m.index_type != type_i || m.index_row_size != row_size) {
                throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: mismatched index cache type");
            }
        }
    }

    if (seq_id == -1) {
        for (auto & [_, buf] : dsv4_ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    } else {
        dsv4_clear_seq(seq_id);
    }

    for (uint32_t is = 0; is < n_seq; ++is) {
        llama_seq_id src_seq_id;
        io.read_to(&src_seq_id, sizeof(src_seq_id));

        const llama_seq_id dst_seq_id = seq_id == -1 ? src_seq_id : seq_id;
        if (dst_seq_id < 0 || (uint32_t) dst_seq_id >= dsv4_n_seq_max) {
            throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: invalid sequence id");
        }

        for (uint32_t il = 0; il < n_layer; ++il) {
            const auto & layer = dsv4_cache_layers[il];

            if (layer.attn_k != nullptr) {
                uint32_t n_rows;
                io.read_to(&n_rows, sizeof(n_rows));
                if (n_rows > layer.n_comp) {
                    throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: too many attention rows");
                }
                if (n_rows > 0) {
                    const size_t row_size = dsv4_cache_row_size(layer.attn_k);
                    ggml_backend_tensor_set(layer.attn_k, io.read((size_t) n_rows*row_size),
                            dsv4_cache_offset(layer.attn_k, dst_seq_id, 0), (size_t) n_rows*row_size);
                }
            }

            if (layer.index_k != nullptr) {
                uint32_t n_rows;
                io.read_to(&n_rows, sizeof(n_rows));
                if (n_rows > layer.n_comp) {
                    throw std::runtime_error("failed to restore DeepSeek V4 compressed KV cache: too many index rows");
                }
                if (n_rows > 0) {
                    const size_t row_size = dsv4_cache_row_size(layer.index_k);
                    ggml_backend_tensor_set(layer.index_k, io.read((size_t) n_rows*row_size),
                            dsv4_cache_offset(layer.index_k, dst_seq_id, 0), (size_t) n_rows*row_size);
                }
            }
        }
    }
}

llama_kv_cache_iswa * llama_memory_hybrid_iswa::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid_iswa::get_mem_recr() const {
    return mem_recr.get();
}

bool llama_memory_hybrid_iswa::has_dsv4_compressed_kv() const {
    for (const auto & layer : dsv4_cache_layers) {
        if (layer.n_comp != 0) {
            return true;
        }
    }

    return false;
}

uint32_t llama_memory_hybrid_iswa::get_dsv4_n_comp(int32_t il) const {
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());
    return dsv4_cache_layers[il].n_comp;
}

ggml_tensor * llama_memory_hybrid_iswa::get_dsv4_attn_k(ggml_context * ctx, int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);

    ggml_tensor * t = dsv4_cache_layers[il].attn_k;
    GGML_ASSERT(t != nullptr);

    return ggml_view_2d(ctx, t, t->ne[0], t->ne[1], t->nb[1], seq_id*t->nb[2]);
}

ggml_tensor * llama_memory_hybrid_iswa::get_dsv4_index_k(ggml_context * ctx, int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(il >= 0 && il < (int32_t) dsv4_cache_layers.size());
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < dsv4_n_seq_max);

    ggml_tensor * t = dsv4_cache_layers[il].index_k;
    GGML_ASSERT(t != nullptr);

    return ggml_view_2d(ctx, t, t->ne[0], t->ne[1], t->nb[1], seq_id*t->nb[2]);
}

//
// llama_memory_hybrid_iswa_context
//

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(llama_memory_hybrid_iswa * mem) :
    mem(mem),
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(
        llama_memory_hybrid_iswa * mem,
                   llama_context * lctx,
                            bool   optimize) :
    mem(mem),
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_iswa_context::llama_memory_hybrid_iswa_context(
           llama_memory_hybrid_iswa * mem,
                    slot_info_vec_t   sinfos_base,
                    slot_info_vec_t   sinfos_swa,
          std::vector<llama_ubatch>   ubatches) :
    mem(mem),
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_iswa_context(mem->get_mem_attn(), std::move(sinfos_base), std::move(sinfos_swa), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_iswa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_iswa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

uint32_t llama_memory_hybrid_iswa_context::get_ubatch_index() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return i_next;
}

uint32_t llama_memory_hybrid_iswa_context::get_ubatch_count() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches.size();
}

const llama_kv_cache_iswa_context * llama_memory_hybrid_iswa_context::get_attn() const {
    return static_cast<const llama_kv_cache_iswa_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_iswa_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}

bool llama_memory_hybrid_iswa_context::has_dsv4_compressed_kv() const {
    return mem != nullptr && mem->has_dsv4_compressed_kv();
}

uint32_t llama_memory_hybrid_iswa_context::get_dsv4_n_comp(int32_t il) const {
    GGML_ASSERT(mem != nullptr);
    return mem->get_dsv4_n_comp(il);
}

ggml_tensor * llama_memory_hybrid_iswa_context::get_dsv4_attn_k(ggml_context * ctx, int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(mem != nullptr);
    return mem->get_dsv4_attn_k(ctx, il, seq_id);
}

ggml_tensor * llama_memory_hybrid_iswa_context::get_dsv4_index_k(ggml_context * ctx, int32_t il, llama_seq_id seq_id) const {
    GGML_ASSERT(mem != nullptr);
    return mem->get_dsv4_index_k(ctx, il, seq_id);
}
