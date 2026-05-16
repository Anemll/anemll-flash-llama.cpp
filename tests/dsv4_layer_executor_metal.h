#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int dsv4_layer_executor_metal_hc_pre_norm(
        const uint8_t * reference,
        size_t          nbytes,
        uint8_t       * output,
        char          * error,
        size_t          error_size);

int dsv4_layer_executor_metal_hc_pre_norm_recompute(
        const float * input_hc,
        const float * split_pre,
        const float * norm_weight,
        size_t        n_embd,
        size_t        n_hc,
        float         eps,
        float       * candidate_cur,
        float       * candidate_norm,
        float       * candidate_post,
        char        * error,
        size_t        error_size);

int dsv4_layer_executor_metal_rmoe_final_from_substages(
        const float * weighted_down,
        const float * shared_down,
        size_t        n_embd,
        size_t        topk,
        float       * candidate_routed_sum,
        float       * candidate_final_ffn,
        char        * error,
        size_t        error_size);

int dsv4_layer_executor_metal_aohc_hc_post_from_substages(
        const float * attn_out,
        const float * residual,
        const float * post,
        const float * comb,
        size_t        n_embd,
        size_t        n_hc,
        float       * candidate_after_attn_hc,
        char        * error,
        size_t        error_size);

int dsv4_layer_executor_metal_cupd_norm_weighted(
        const float * compressed_norm,
        const float * compressed_norm_weight,
        size_t        n,
        float       * candidate_norm_weighted,
        char        * error,
        size_t        error_size);

#ifdef __cplusplus
}
#endif
