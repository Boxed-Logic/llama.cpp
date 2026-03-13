/**
 * qwen35-mtp.cpp — Qwen3.5 Multi-Token Prediction via MTP Head Speculative Decoding
 *
 * Correct MTP implementation matching vLLM's approach:
 *   1. Run full target model → capture final hidden state h (pre-output-norm)
 *   2. For each draft step, run the MTP head layer (blk.n_main) with:
 *        projected = eh_proj @ [enorm(tok_emb(prev_tok)); hnorm(h)] → [n_embd]
 *        h_new = mtp_attn_layer(projected, pos)   (full attention + FFN, real KV cache)
 *        draft_tok = lm_head(shared_head_norm(h_new))
 *   3. Verify batch with full target model; greedy accept/reject prefix
 *   4. Snapshot/restore to keep recurrent (DeltaNet) state consistent
 *
 * IMPORTANT: CPU/Metal limitation
 *   Speculative decoding gains its speedup from batch verification being nearly
 *   free on GPU (memory-bandwidth bound).  On CPU/Metal, batch processing scales
 *   linearly with token count, so the verify batch overhead exceeds any savings
 *   from draft acceptance.  This example is primarily useful for:
 *     - Validating MTP head correctness (accept rates, output quality)
 *     - GPU benchmarking (where batch verification IS cheap)
 *   For CPU-only inference, use --no-mtp for best throughput.
 *
 * Usage:
 *   llama-qwen35-mtp -m model.gguf -p "Your prompt here" -n 200 --draft 4
 */

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ─── helpers ────────────────────────────────────────────────────────────────

static void print_usage(const char * prog) {
    LOG_INF("usage: %s [options] -m <model.gguf>\n\n", prog);
    LOG_INF("  -m, --model          path to Qwen3.5 GGUF model\n");
    LOG_INF("  -p, --prompt         input prompt\n");
    LOG_INF("  -n, --n-predict      tokens to generate (default: 200)\n");
    LOG_INF("  -c, --ctx-size       context size (default: 2048)\n");
    LOG_INF("  -d, --draft          draft tokens per step (default: 4)\n");
    LOG_INF("  -t, --threads        CPU threads (default: 4)\n");
    LOG_INF("  --ngl, -ngl          GPU layers to offload (default: 0 = CPU only)\n");
    LOG_INF("  --temp               sampling temperature (default: 0.0 = greedy)\n");
    LOG_INF("  --no-mtp             disable MTP, use standard autoregressive decode\n");
    LOG_INF("\n");
}

// Initialise a llama_batch that can carry both tokens and embeddings.
struct MtpBatch {
    llama_batch    batch;
    std::vector<llama_token>    token_buf;
    std::vector<float>          embd_buf;   // [n_embd * n_tokens_max]
    std::vector<llama_pos>      pos_buf;
    std::vector<int32_t>        n_seq_id_buf;
    std::vector<llama_seq_id>   seq_id_buf;
    std::vector<llama_seq_id *> seq_ids_buf;
    std::vector<int8_t>         logits_buf;
    int32_t n_embd;

    MtpBatch(int32_t n_tokens_max, int32_t embd_dim) : n_embd(embd_dim) {
        token_buf.resize(n_tokens_max);
        embd_buf.resize((size_t)n_tokens_max * embd_dim, 0.0f);
        pos_buf.resize(n_tokens_max);
        n_seq_id_buf.resize(n_tokens_max, 1);
        seq_id_buf.resize(n_tokens_max, 0);
        seq_ids_buf.resize(n_tokens_max);
        logits_buf.resize(n_tokens_max, 0);

        for (int i = 0; i < n_tokens_max; ++i) {
            seq_ids_buf[i] = &seq_id_buf[i];
        }

        batch.n_tokens     = 0;
        batch.token        = token_buf.data();
        batch.embd         = embd_buf.data();
        batch.pos          = pos_buf.data();
        batch.n_seq_id     = n_seq_id_buf.data();
        batch.seq_id       = seq_ids_buf.data();
        batch.logits       = logits_buf.data();
    }

    void set_single(llama_token tok, const float * h, llama_pos pos) {
        batch.n_tokens      = 1;
        token_buf[0]        = tok;
        pos_buf[0]          = pos;
        seq_id_buf[0]       = 0;
        logits_buf[0]       = 1;
        memcpy(embd_buf.data(), h, (size_t)n_embd * sizeof(float));
    }

    void set_single_no_logits(llama_token tok, const float * h, llama_pos pos) {
        set_single(tok, h, pos);
        logits_buf[0] = 0;
    }
};

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    // ── parse args ──────────────────────────────────────────────────────────
    std::string model_path;
    std::string prompt   = "The theory of everything unifies";
    int32_t  n_predict   = 200;
    int32_t  n_ctx       = 2048;
    int32_t  n_draft     = 4;
    int32_t  n_threads   = 4;
    int32_t  n_gpu_layers = 0;
    float    temperature = 0.0f;
    bool     disable_mtp = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      ((a == "-m" || a == "--model")      && i+1<argc) { model_path  = argv[++i]; }
        else if ((a == "-p" || a == "--prompt")     && i+1<argc) { prompt      = argv[++i]; }
        else if ((a == "-n" || a == "--n-predict")  && i+1<argc) { n_predict   = atoi(argv[++i]); }
        else if ((a == "-c" || a == "--ctx-size")   && i+1<argc) { n_ctx       = atoi(argv[++i]); }
        else if ((a == "-d" || a == "--draft")      && i+1<argc) { n_draft     = atoi(argv[++i]); }
        else if ((a == "-t" || a == "--threads")    && i+1<argc) { n_threads   = atoi(argv[++i]); }
        else if ((a == "--ngl" || a == "-ngl")  && i+1<argc) { n_gpu_layers = atoi(argv[++i]); }
        else if (a == "--temp"   && i+1<argc) { temperature = atof(argv[++i]); }
        else if (a == "--no-mtp")               { disable_mtp = true; }
        else if (a == "-h" || a == "--help")    { print_usage(argv[0]); return 0; }
        else {
            LOG_ERR("Unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        LOG_ERR("Error: -m <model> is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    common_init();
    llama_backend_init();

    // ── load model ──────────────────────────────────────────────────────────
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        LOG_ERR("Failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const int32_t       n_embd = llama_model_n_embd(model);

    // ── target context (full model, captures hidden states) ─────────────────
    llama_context_params cparams_tgt = llama_context_default_params();
    cparams_tgt.n_ctx            = (uint32_t)n_ctx;
    cparams_tgt.n_batch          = (uint32_t)n_ctx;
    cparams_tgt.n_threads        = n_threads;
    cparams_tgt.n_threads_batch  = n_threads;
    cparams_tgt.embeddings       = true;   // expose hidden states via llama_get_embeddings_ith
    cparams_tgt.no_output_all    = true;   // only compute outputs for tokens with logits=1
    cparams_tgt.n_seq_max        = 2;      // seq 0 = active; seq 1 = snapshot

    llama_context * ctx_tgt = llama_init_from_model(model, cparams_tgt);
    if (!ctx_tgt) {
        LOG_ERR("Failed to create target context\n");
        llama_model_free(model);
        return 1;
    }

    // ── MTP head context (runs only layer n_main) ────────────────────────────
    llama_context * ctx_mtp = nullptr;
    if (!disable_mtp) {
        llama_context_params cparams_mtp = llama_context_default_params();
        cparams_mtp.n_ctx            = (uint32_t)n_ctx;
        cparams_mtp.n_batch          = (uint32_t)(n_draft + 2);
        cparams_mtp.n_threads        = n_threads;
        cparams_mtp.n_threads_batch  = n_threads;
        cparams_mtp.mtp_head_mode    = true;
        cparams_mtp.embeddings       = true;  // capture h_mtp for chaining
        cparams_mtp.no_output_all    = true;  // only compute outputs for logits=1 tokens
        cparams_mtp.n_seq_max        = 2;     // seq 0 = active; seq 1 = snapshot

        ctx_mtp = llama_init_from_model(model, cparams_mtp);
        if (!ctx_mtp) {
            LOG_ERR("Failed to create MTP head context — falling back to standard decode\n");
            disable_mtp = true;
        }
    }

    const bool use_mtp = !disable_mtp && ctx_mtp;

    // ── tokenize prompt ─────────────────────────────────────────────────────
    std::vector<llama_token> tokens_inp =
        common_tokenize(ctx_tgt, prompt, /*add_special=*/true, /*parse_special=*/true);

    if ((int)tokens_inp.size() >= n_ctx) {
        LOG_ERR("Prompt too long (%d tokens) for context size %d\n",
                (int)tokens_inp.size(), n_ctx);
        return 1;
    }

    // ── samplers ────────────────────────────────────────────────────────────
    llama_sampler * smpl_tgt;
    {
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        smpl_tgt = llama_sampler_chain_init(sparams);
        if (temperature <= 0.0f) {
            llama_sampler_chain_add(smpl_tgt, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(smpl_tgt, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(smpl_tgt, llama_sampler_init_dist(42));
        }
    }

    llama_sampler * smpl_dft = nullptr;
    if (use_mtp) {
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        smpl_dft = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl_dft, llama_sampler_init_greedy());
    }

    // ── prefill prompt ──────────────────────────────────────────────────────
    LOG_INF("\n== Qwen3.5 MTP%s ==\n", use_mtp ? " (MTP head speculative)" : " (disabled)");
    LOG_INF("Prompt: \"%s\"\n", prompt.c_str());
    LOG_INF("n_predict=%d  n_draft=%d  temperature=%.2f  n_embd=%d\n\n",
            n_predict, use_mtp ? n_draft : 0, temperature, n_embd);

    llama_batch batch_tgt = llama_batch_init(n_draft + 2, 0, 1);
    MtpBatch mtp_batch(n_draft + 2, n_embd);

    std::vector<float> h_last(n_embd, 0.0f);

    llama_token id_last = 0;
    int n_past     = 0;
    int n_total    = 0;
    int n_drafted  = 0;
    int n_accepted = 0;

    {
        const int n_inp = (int)tokens_inp.size();

        llama_batch batch_prefill = llama_batch_init(n_inp, 0, 1);
        for (int i = 0; i < n_inp; ++i) {
            common_batch_add(batch_prefill, tokens_inp[i], i, {0}, use_mtp ? true : (i == n_inp - 1));
        }

        llama_decode(ctx_tgt, batch_prefill);
        llama_batch_free(batch_prefill);
        n_past = n_inp;

        id_last = llama_sampler_sample(smpl_tgt, ctx_tgt, -1);
        ++n_total;

        LOG_INF("Output: ");
        if (!llama_vocab_is_eog(vocab, id_last)) {
            LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
            fflush(stdout);
        }

        if (use_mtp) {
            const float * h_ptr = llama_get_embeddings_ith(ctx_tgt, n_inp - 1);
            if (h_ptr) {
                memcpy(h_last.data(), h_ptr, (size_t)n_embd * sizeof(float));
            }

            // Prefill ctx_mtp serially using h from ctx_tgt
            for (int i = 0; i < n_inp; ++i) {
                const float * h_i = llama_get_embeddings_ith(ctx_tgt, i);
                if (!h_i) break;
                llama_token tok_next = (i + 1 < n_inp) ? tokens_inp[i + 1] : id_last;
                mtp_batch.set_single_no_logits(tok_next, h_i, i);
                if (llama_decode(ctx_mtp, mtp_batch.batch) != 0) {
                    LOG_ERR("ctx_mtp prefill failed at position %d\n", i);
                    break;
                }
            }
        }
    }

    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);

    // ── timing ──────────────────────────────────────────────────────────────
    const auto t_start = ggml_time_us();

    double t_draft_ms = 0, t_verify_ms = 0;
    double t_restore_tgt_ms = 0, t_restore_mtp_ms = 0;

    // ── decode loop ─────────────────────────────────────────────────────────
    while (n_total < n_predict) {
        if (use_mtp) {
            const llama_token id_last_iter = id_last;
            const int         old_n_past   = n_past;

            // ── SNAPSHOT ────────────────────────────────────────────────────
            // Save seq 0 → seq 1 for both target and MTP contexts.
            // Recurrent state (DeltaNet) is sequential and cannot be partially
            // rolled back, so we must restore from snapshot after verification.
            llama_memory_seq_cp(mem_tgt, 0, 1, -1, -1);
            llama_memory_t mem_mtp = llama_get_memory(ctx_mtp);
            llama_memory_seq_cp(mem_mtp, 0, 1, -1, -1);

            std::vector<float> h_snap = h_last;

            // ── DRAFT PHASE ─────────────────────────────────────────────────
            std::vector<llama_token> draft;
            draft.reserve(n_draft);
            std::vector<std::vector<float>> h_mtp(n_draft, std::vector<float>(n_embd));

            const float * h_input = h_last.data();
            llama_token   tok_input = id_last;

            double t0 = now_ms();
            for (int d = 0; d < n_draft; ++d) {
                mtp_batch.set_single(tok_input, h_input, old_n_past + d);

                if (llama_decode(ctx_mtp, mtp_batch.batch) != 0) {
                    break;
                }

                llama_synchronize(ctx_mtp);
                const float * emb_out = llama_get_embeddings_ith(ctx_mtp, -1);
                if (!emb_out) break;
                memcpy(h_mtp[d].data(), emb_out, (size_t)n_embd * sizeof(float));

                llama_token tok = llama_sampler_sample(smpl_dft, ctx_mtp, -1);

                if (llama_vocab_is_eog(vocab, tok)) break;
                draft.push_back(tok);

                tok_input = tok;
                h_input   = h_mtp[d].data();
            }
            t_draft_ms += now_ms() - t0;

            // ── VERIFY PHASE ────────────────────────────────────────────────
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, id_last, n_past, {0}, true);
            for (size_t d = 0; d < draft.size(); ++d) {
                common_batch_add(batch_tgt, draft[d], n_past + 1 + (int)d, {0}, true);
            }

            t0 = now_ms();
            if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                LOG_ERR("\nllama_decode (target) failed\n");
                break;
            }
            t_verify_ms += now_ms() - t0;

            // Collect h_verify for h_last update and MTP replay
            int n_verify = (int)draft.size() + 1;
            std::vector<std::vector<float>> h_verify(n_verify, std::vector<float>(n_embd));
            for (int j = 0; j < n_verify; ++j) {
                const float * emb = llama_get_embeddings_ith(ctx_tgt, j);
                if (emb) {
                    memcpy(h_verify[j].data(), emb, (size_t)n_embd * sizeof(float));
                }
            }

            // Greedy acceptance
            n_drafted += (int)draft.size();

            std::vector<llama_token> accepted;
            for (int pos = 0; pos <= (int)draft.size(); ++pos) {
                llama_token sampled = llama_sampler_sample(smpl_tgt, ctx_tgt, pos);
                accepted.push_back(sampled);
                if (pos < (int)draft.size() && sampled != draft[pos]) {
                    break;
                }
            }

            n_accepted += (int)accepted.size() - 1;
            n_past     += (int)accepted.size();
            n_total    += (int)accepted.size();

            bool stop = false;
            for (llama_token tok : accepted) {
                if (llama_vocab_is_eog(vocab, tok)) { stop = true; break; }
                LOG("%s", common_token_to_piece(ctx_tgt, tok).c_str());
                fflush(stdout);
            }

            // ── RESTORE ctx_tgt ─────────────────────────────────────────────
            // Restore from snapshot, then replay accepted tokens to advance
            // the recurrent state correctly.
            t0 = now_ms();
            {
                llama_memory_seq_rm(mem_tgt, 0, -1, -1);
                llama_memory_seq_cp(mem_tgt, 1, 0, -1, -1);
                llama_memory_seq_rm(mem_tgt, 1, -1, -1);

                // Replay accepted tokens one by one through target model.
                // Sequential replay is faster than batched on CPU (less
                // graph overhead per token).
                for (int j = 0; j < (int)accepted.size(); ++j) {
                    llama_token tok = (j == 0) ? id_last_iter : accepted[j - 1];
                    common_batch_clear(batch_tgt);
                    common_batch_add(batch_tgt, tok, old_n_past + j, {0}, false);
                    if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                        LOG_ERR("target replay failed at step %d\n", j);
                        break;
                    }
                }
            }
            t_restore_tgt_ms += now_ms() - t0;

            // ── RESTORE ctx_mtp ─────────────────────────────────────────────
            // Restore MTP KV from snapshot, replay accepted positions with
            // correct h_verify (from target model).
            t0 = now_ms();
            {
                llama_memory_seq_rm(mem_mtp, 0, -1, -1);
                llama_memory_seq_cp(mem_mtp, 1, 0, -1, -1);
                llama_memory_seq_rm(mem_mtp, 1, -1, -1);

                // Replay accepted MTP inputs using h_verify from target
                for (int j = 0; j < (int)accepted.size(); ++j) {
                    llama_token tok_next = (j == 0) ? id_last_iter : accepted[j - 1];
                    const float * h_j = (j == 0) ? h_snap.data() : h_verify[j - 1].data();
                    mtp_batch.set_single_no_logits(tok_next, h_j, old_n_past + j);
                    if (llama_decode(ctx_mtp, mtp_batch.batch) != 0) {
                        LOG_ERR("MTP replay failed at step %d\n", j);
                        break;
                    }
                }

                h_last = h_verify[(int)accepted.size() - 1];
            }
            t_restore_mtp_ms += now_ms() - t0;

            id_last = accepted.back();
            if (stop || n_total >= n_predict) break;

        } else {
            // ── STANDARD AUTOREGRESSIVE (no MTP) ────────────────────────────
            llama_batch b = llama_batch_get_one(&id_last, 1);
            if (llama_decode(ctx_tgt, b) != 0) {
                LOG_ERR("\nllama_decode failed\n");
                break;
            }
            id_last = llama_sampler_sample(smpl_tgt, ctx_tgt, -1);
            ++n_past;
            ++n_total;

            if (llama_vocab_is_eog(vocab, id_last)) break;
            LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
            fflush(stdout);
        }
    }

    const auto t_end      = ggml_time_us();
    const double elapsed_s = (t_end - t_start) * 1e-6;

    // ── statistics ──────────────────────────────────────────────────────────
    LOG_INF("\n\n====== Performance ======\n");
    LOG_INF("mode          : %s\n",
            use_mtp ? "Qwen3.5 MTP (MTP-head speculative)" : "standard autoregressive");
    LOG_INF("tokens        : %d\n", n_total);
    LOG_INF("elapsed       : %.3f s\n", elapsed_s);
    LOG_INF("throughput    : %.2f tok/s\n", n_total / elapsed_s);
    if (use_mtp && n_drafted > 0) {
        LOG_INF("drafted       : %d\n", n_drafted);
        LOG_INF("accepted      : %d\n", n_accepted);
        LOG_INF("accept rate   : %.1f%%\n", 100.0 * n_accepted / n_drafted);
        LOG_INF("speedup steps : %.2fx  (fewer decode() calls)\n",
                (double)n_total / (double)(n_total - n_accepted));
        LOG_INF("--- phase timing (ms) ---\n");
        LOG_INF("  draft       : %.1f ms\n", t_draft_ms);
        LOG_INF("  verify      : %.1f ms\n", t_verify_ms);
        LOG_INF("  restore_tgt : %.1f ms\n", t_restore_tgt_ms);
        LOG_INF("  restore_mtp : %.1f ms\n", t_restore_mtp_ms);
    }
    LOG_INF("=========================\n");

    // ── cleanup ──────────────────────────────────────────────────────────────
    llama_batch_free(batch_tgt);
    llama_sampler_free(smpl_tgt);
    if (smpl_dft)  llama_sampler_free(smpl_dft);
    if (ctx_mtp)   llama_free(ctx_mtp);
    llama_free(ctx_tgt);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
