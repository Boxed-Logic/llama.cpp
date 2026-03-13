#include "models.h"

#include "llama-memory-recurrent.h"

llm_build_qwen35::llm_build_qwen35(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // MTP head blocks (last nextn_predict_layers) are not part of the main model
    // forward pass — they're handled separately for MTP prediction. Only iterate
    // the n_main main model layers here.
    const int n_main = n_layer - (int)hparams.nextn_predict_layers;

    // ─── MTP HEAD MODE ───────────────────────────────────────────────────────
    // Skip main layers 0..n_main-1; run only the MTP head layer (blk.n_main).
    //
    // Inputs (from batch):
    //   batch.token = prev token ID  → used for tok_embd lookup
    //   batch.embd  = h_last [n_embd] → pre-output-norm hidden state from ctx_tgt
    //
    // Graph: enorm(tok_emb) + hnorm(h_last) → eh_proj → attn+FFN (il=n_main)
    //        → t_embd (pre-norm, for chaining) → shared_head_norm → lm_head → logits
    if (cparams.mtp_head_mode) {
        auto * inp_hybrid     = build_inp_mem_hybrid();
        ggml_tensor * inp_pos     = build_inp_pos();
        ggml_tensor * inp_out_ids = build_inp_out_ids();

        // Build input object for BOTH tokens (emb lookup) and embd (h_last).
        // llm_graph_input_embd::set_input() fills both when present in the batch.
        auto inp_embd_obj = std::make_unique<llm_graph_input_embd>(n_embd);

        inp_embd_obj->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_tokens);
        ggml_set_input(inp_embd_obj->tokens);
        cb(inp_embd_obj->tokens, "inp_tokens", -1);
        res->t_inp_tokens = inp_embd_obj->tokens;

        inp_embd_obj->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, ubatch.n_tokens);
        ggml_set_input(inp_embd_obj->embd);
        cb(inp_embd_obj->embd, "inp_embd", -1);

        // Token embedding lookup (batch.token)
        ggml_tensor * tok_emb = ggml_get_rows(ctx0, model.tok_embd, inp_embd_obj->tokens);
        // h_last: pre-output-norm hidden state from ctx_tgt (batch.embd)
        ggml_tensor * h_last = inp_embd_obj->embd;

        // Register with framework so set_inputs() fills both tensors from the batch
        res->add_input(std::move(inp_embd_obj));

        // nextn projection: enorm(emb) concat hnorm(h) → eh_proj → [n_embd, n_tokens]
        const int il = n_main;
        const auto & layer = model.layers[il];

        ggml_tensor * normed_emb = build_norm(tok_emb, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
        cb(normed_emb, "mtp_normed_emb", il);

        ggml_tensor * normed_h = build_norm(h_last, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
        cb(normed_h, "mtp_normed_h", il);

        // Concatenate along dim 0: [2*n_embd, n_tokens]
        ggml_tensor * combined = ggml_concat(ctx0, normed_emb, normed_h, 0);
        cb(combined, "mtp_combined", il);

        // eh_proj shape: {2*n_embd, n_embd} → ggml_mul_mat maps [2*n_embd, n_tokens] → [n_embd, n_tokens]
        ggml_tensor * projected = ggml_mul_mat(ctx0, layer.nextn.eh_proj, combined);
        cb(projected, "mtp_projected", il);

        ggml_build_forward_expand(gf, projected);

        // Run MTP attention layer + post-attn-norm + FFN
        ggml_tensor * inpL  = projected;
        ggml_tensor * cur;
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        cur = build_layer_attn(inp_hybrid->get_attn(), cur, inp_pos, sections, il);

        if (inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        // Capture pre-output-norm h_mtp for chaining (next draft step's h_last input)
        res->t_embd = cur;

        // Output norm: use MTP head's shared_head_norm if present, else fall back
        ggml_tensor * out_norm_w = (layer.nextn.shared_head_norm != nullptr)
                                        ? layer.nextn.shared_head_norm
                                        : model.output_norm;
        cur = build_norm(cur, out_norm_w, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        // Shared LM head (same weights as main model)
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
        return; // skip main model loop
    }

    // ─── NORMAL / DRAFT PATH ─────────────────────────────────────────────────

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    // In MTP draft mode, use pure recurrent memory (no KV cache for attention layers).
    // In normal mode, use the full hybrid (attention + recurrent) memory.
    llm_graph_input_mem_hybrid * inp_hybrid = nullptr;
    llm_graph_input_rs         * inp_recr   = nullptr;
    if (cparams.mtp_draft_mode) {
        // Draft mode: recurrent-only memory context
        inp_recr = build_rs_inp();
    } else {
        inp_hybrid = build_inp_mem_hybrid();
    }

    // inp_pos is only needed for full-attention layers (RoPE); skip in draft mode
    ggml_tensor * inp_pos     = cparams.mtp_draft_mode ? nullptr : build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_main; ++il) {
        ggml_tensor * inpSA = inpL;

        // MTP draft mode: for full-attention layers, skip both attention and FFN.
        // Only the recurrent (linear attention) layers are evaluated, making the
        // draft forward pass much faster. This enables Qwen3.5 self-speculative decoding.
        const bool skip_full_attn = (!hparams.is_recurrent(il)) && cparams.mtp_draft_mode;

        if (skip_full_attn) {
            // Full attention layer bypassed in draft mode - pure pass-through.
            // Handle inp_out_ids at the final layer so output slicing is correct.
            if (il == n_main - 1 && inp_out_ids) {
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }
            // inpL is unchanged - this layer is an identity in draft mode
            continue;
        }

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            auto * recr = cparams.mtp_draft_mode ? inp_recr : inp_hybrid->get_recr();
            cur = build_layer_attn_linear(recr, cur, il);
        } else {
            // Full attention layer (only reached in non-draft mode)
            cur = build_layer_attn(inp_hybrid->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_main - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Capture pre-output-norm hidden state for MTP chaining.
    // llama_get_embeddings_ith() on ctx_tgt returns this value, which is then
    // passed as h_last to the MTP head via batch.embd.
    res->t_embd = cur;

    // Final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_qwen35::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llm_build_qwen35::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llm_build_qwen35::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llm_build_qwen35::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur);
    alpha = ggml_cont_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    // Get convolution states from cache
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // Build the convolution states tensor
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    // Calculate convolution kernel size
    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    // Update convolution state cache
    // Extract the last (conv_kernel_size - 1) states from conv_input
    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                     conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target =
        ggml_view_1d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels * n_seqs,
                     kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        // TODO: try to avoid these explicit repeats by utilizing op broadcast
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    // Choose between build_delta_net_chunking, build_delta_net_recurrent, and build_delta_net_autoregressive based on n_tokens
    std::pair<ggml_tensor *, ggml_tensor *> attn_out; // pair of (output, new_state)
    if (n_seq_tokens == 1) {
        attn_out = build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il);
    } else {
        attn_out = build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, il);
    }
    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);

    // Update the recurrent states
    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, new_state,
                ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llm_build_qwen35::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, NULL,
        model.layers[il].ffn_gate, NULL, NULL,
        model.layers[il].ffn_down, NULL, NULL,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}
