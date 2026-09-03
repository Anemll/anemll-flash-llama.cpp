#include "models.h"

#include <cmath>

// HY4 uses independent Hyper-Connections (iHC).  Unlike DeepSeek-V4 HC,
// iHC has only a pre-reduction and per-stream post gate: it does not have the
// Sinkhorn/comb matrix.  Keep these helpers local rather than reusing the
// DeepSeek-V4 ones so the two layouts cannot be confused.

static ggml_tensor * hyv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, i0 * t->nb[0]);
}

static ggml_tensor * hyv4_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], i0 * t->nb[0]);
}

// Reduce hc streams x[:, h, :] with weights w[h, :] in F32, matching the
// reference iHC reduction path before converting back to the model dtype.
static ggml_tensor * hyv4_hc_reduce(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * w,
        int64_t n_hc,
        int64_t n_embd,
        int64_t n_tokens,
        ggml_type out_type) {
    ggml_tensor * x_f32 = ggml_cast(ctx, x, GGML_TYPE_F32);
    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < n_hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx, x_f32, n_embd, n_tokens, x_f32->nb[2], ih * x_f32->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx, w, 1, n_tokens, w->nb[1], ih * w->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx, xh, wh);
        result = result == nullptr ? cur : ggml_add(ctx, result, cur);
    }
    return ggml_cast(ctx, result, out_type);
}

static ggml_tensor * hyv4_hc_pre(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        int64_t n_hc,
        int64_t n_embd,
        float norm_rms_eps,
        float hc_eps,
        float hc_magnitude,
        ggml_tensor ** post) {
    const int64_t n_tokens = x->ne[2];
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == n_hc);

    ggml_tensor * flat = ggml_reshape_2d(ctx, x, n_hc * n_embd, n_tokens);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx, hc_fn, flat_norm); // [2 * n_hc, n_tokens]

    ggml_tensor * scale_pre  = hyv4_view_1d(ctx, hc_scale, 1, 0);
    ggml_tensor * scale_post = hyv4_view_1d(ctx, hc_scale, 1, 1);
    ggml_tensor * base_pre   = hyv4_view_1d(ctx, hc_base, n_hc, 0);
    ggml_tensor * base_post  = hyv4_view_1d(ctx, hc_base, n_hc, n_hc);

    ggml_tensor * pre = hyv4_view_2d(ctx, mixes, n_hc, n_tokens, 0);
    pre = ggml_mul(ctx, pre, scale_pre);
    pre = ggml_add(ctx, pre, base_pre);
    pre = ggml_sigmoid(ctx, pre);
    pre = ggml_scale_bias(ctx, pre, 1.0f, hc_eps);

    ggml_tensor * po = hyv4_view_2d(ctx, mixes, n_hc, n_tokens, n_hc);
    po = ggml_mul(ctx, po, scale_post);
    po = ggml_add(ctx, po, base_post);
    po = ggml_sigmoid(ctx, po);
    po = ggml_scale(ctx, po, hc_magnitude);
    *post = ggml_scale_bias(ctx, po, 1.0f, hc_eps);

    return hyv4_hc_reduce(ctx, x, pre, n_hc, n_embd, n_tokens, x->type);
}

static ggml_tensor * hyv4_hc_post(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        int64_t n_hc,
        int64_t n_embd) {
    const int64_t n_tokens = x->ne[1];
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == n_hc);

    ggml_tensor * x_f32 = ggml_cast(ctx, x, GGML_TYPE_F32);
    ggml_tensor * post_f32 = ggml_cast(ctx, post, GGML_TYPE_F32);
    ggml_tensor * residual_f32 = ggml_cast(ctx, residual, GGML_TYPE_F32);

    ggml_tensor * out = nullptr;
    for (int64_t ih = 0; ih < n_hc; ++ih) {
        ggml_tensor * residual_h = ggml_view_2d(ctx, residual_f32, n_embd, n_tokens,
                residual_f32->nb[2], ih * residual_f32->nb[1]);
        ggml_tensor * post_h = ggml_view_2d(ctx, post_f32, 1, n_tokens,
                post_f32->nb[1], ih * post_f32->nb[0]);
        ggml_tensor * cur = ggml_add(ctx, residual_h, ggml_mul(ctx, x_f32, post_h));
        cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
        out = out == nullptr ? cur : ggml_concat(ctx, out, cur, 1);
    }

    return ggml_cast(ctx, out, residual->type);
}

static ggml_tensor * hyv4_hc_head(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        int64_t n_hc,
        int64_t n_embd,
        float norm_rms_eps,
        float hc_eps) {
    const int64_t n_tokens = x->ne[2];
    ggml_tensor * flat = ggml_reshape_2d(ctx, x, n_hc * n_embd, n_tokens);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx, flat, norm_rms_eps);
    ggml_tensor * pre = ggml_mul_mat(ctx, hc_fn, flat_norm); // [n_hc, n_tokens]
    pre = ggml_mul(ctx, pre, hc_scale);
    pre = ggml_add(ctx, pre, hc_base);
    pre = ggml_sigmoid(ctx, pre);
    pre = ggml_scale_bias(ctx, pre, 1.0f, hc_eps);
    return hyv4_hc_reduce(ctx, x, pre, n_hc, n_embd, n_tokens, x->type);
}

llm_build_hyv4::llm_build_hyv4(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_hc = hparams.n_hc;
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const int64_t q_lora_rank = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    GGML_ASSERT(n_hc > 0);
    GGML_ASSERT(n_embd_head_qk_nope > 0);
    GGML_ASSERT(q_lora_rank > 0);
    GGML_ASSERT(kv_lora_rank > 0);

    // The fork has no native DSA cache yet.  Retain the exact MLA/iHC/MoE
    // graph and use the normal KV cache.  It is mathematically identical while
    // the causal history fits indexer_top_k (2048 for Hy4-preview), but becomes
    // a documented full-attention fallback beyond that point.
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_k();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, n_hc, n_tokens, 1);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, n_hc, n_tokens);

    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head_k_mla));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;

        ggml_tensor * cur = hyv4_hc_pre(ctx0, inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                n_hc, n_embd, norm_rms_eps, hparams.hc_eps, hparams.hc_magnitude, &post);
        cb(cur, "hc_attn_pre", il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
        q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
        q = ggml_mul_mat(ctx0, layer.wq_b, q);

        ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head, 0);
        ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
                ggml_row_size(q->type, n_embd_head_qk_nope));

        ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
        ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
        ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

        q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

        q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
        ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
        q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

        ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
        kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
        ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
        ggml_tensor * Vcur = kv_cmpr;

        // Apply wv_b inside build_attn, then the Hy4-specific gated MLA and wo.
        ggml_tensor * attn = build_attn(inp_attn, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, layer.wv_b, kq_scale, il);
        ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, layer.wqkv_gate, cur));
        attn = ggml_mul(ctx0, attn, gate);
        cur = build_lora_mm(layer.wo, attn);
        cb(cur, "attn_out", il);

        inpL = hyv4_hc_post(ctx0, cur, residual, post, n_hc, n_embd);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = hyv4_hc_pre(ctx0, inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                n_hc, n_embd, norm_rms_eps, hparams.hc_eps, hparams.hc_magnitude, &post);
        cb(cur, "hc_ffn_pre", il);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                    layer.ffn_up, nullptr, nullptr,
                    layer.ffn_gate, nullptr, nullptr,
                    layer.ffn_down, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_dense_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU,
                    hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * shared_out = build_ffn(cur,
                    layer.ffn_up_shexp, nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cur = ggml_add(ctx0, moe_out, shared_out);
            cb(cur, "ffn_out", il);
        }

        inpL = hyv4_hc_post(ctx0, cur, residual, post, n_hc, n_embd);
        cb(inpL, "l_out", il);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd * n_hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, n_hc, n_outputs);
    }

    ggml_tensor * cur = hyv4_hc_head(ctx0, inpL,
            model.output_hc_fn, model.output_hc_scale, model.output_hc_base,
            n_hc, n_embd, norm_rms_eps, hparams.hc_eps);
    cb(cur, "result_hc", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
