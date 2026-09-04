#pragma once

#include "ggml.h"

// The post gate has no reduction: each output is the same rounded F32
// multiply followed by the same rounded F32 addition as the stream loop.
// Casts preserve support for strided inputs and the original output dtype.
static inline ggml_tensor * hyv4_hc_post_broadcast(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        int64_t n_hc,
        int64_t n_embd) {
    const int64_t n_tokens = x->ne[1];
    GGML_ASSERT(x->ne[0] == n_embd && x->ne[2] == 1 && x->ne[3] == 1);
    GGML_ASSERT(residual->ne[0] == n_embd && residual->ne[1] == n_hc &&
            residual->ne[2] == n_tokens && residual->ne[3] == 1);
    GGML_ASSERT(post->ne[0] == n_hc && post->ne[1] == n_tokens &&
            post->ne[2] == 1 && post->ne[3] == 1);

    ggml_tensor * x_f32 = ggml_cast(ctx, x, GGML_TYPE_F32);
    ggml_tensor * post_f32 = ggml_cast(ctx, post, GGML_TYPE_F32);
    ggml_tensor * residual_f32 = ggml_cast(ctx, residual, GGML_TYPE_F32);
    x_f32 = ggml_reshape_3d(ctx, x_f32, n_embd, 1, n_tokens);
    post_f32 = ggml_reshape_3d(ctx, post_f32, 1, n_hc, n_tokens);
    ggml_tensor * expanded = ggml_repeat(ctx, x_f32, residual_f32);
    ggml_tensor * gated = ggml_mul(ctx, expanded, post_f32);
    ggml_tensor * out = ggml_add(ctx, residual_f32, gated);
    return ggml_cast(ctx, out, residual->type);
}
