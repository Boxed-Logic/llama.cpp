/**
 * qwen35-mtp.cpp — Qwen3.5 Multi-Token Prediction via Hybrid Self-Speculative Decoding
 *
 * Implements MTP for Qwen3.5's hybrid architecture:
 *   - Draft context: only recurrent (Delta Net / linear attention) layers active
 *     Full-attention layers are bypassed — enabling a fast O(n_recurrent_layers) forward pass
 *   - Target context: full model verification of the drafted tokens
 *
 * Speedup comes from the draft phase being ~(interval/(interval-1)) times cheaper than
 * the full model (for full_attn_interval=4, roughly 4/3 ≈ 1.33× cheaper per forward pass),
 * and each draft pass may produce multiple accepted tokens.
 *
 * Usage:
 *   llama-qwen35-mtp -m model.gguf -p "Your prompt here" -n 200 --draft 8
 */

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ─── helpers ────────────────────────────────────────────────────────────────

static void print_usage(const char * prog) {
    LOG_INF("usage: %s [options] -m <model.gguf>\n\n", prog);
    LOG_INF("  -m, --model          path to Qwen3.5 GGUF model\n");
    LOG_INF("  -p, --prompt         input prompt\n");
    LOG_INF("  -n, --n-predict      tokens to generate (default: 200)\n");
    LOG_INF("  -c, --ctx-size       context size (default: 2048)\n");
    LOG_INF("  -d, --draft          draft tokens per step (default: 8)\n");
    LOG_INF("  -t, --threads        CPU threads (default: 4)\n");
    LOG_INF("  --temp               sampling temperature (default: 0.0 = greedy)\n");
    LOG_INF("  --no-mtp             disable MTP, use standard autoregressive decode\n");
    LOG_INF("\n");
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    // ── parse args ──────────────────────────────────────────────────────────
    std::string model_path;
    std::string prompt      = "The theory of everything unifies";
    int32_t  n_predict      = 200;
    int32_t  n_ctx          = 2048;
    int32_t  n_draft        = 8;
    int32_t  n_threads      = 4;
    float    temperature    = 0.0f;
    bool     disable_mtp    = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "-m" || a == "--model")     && i + 1 < argc) { model_path = argv[++i]; }
        else if ((a == "-p" || a == "--prompt") && i + 1 < argc) { prompt    = argv[++i]; }
        else if ((a == "-n" || a == "--n-predict") && i + 1 < argc) { n_predict = atoi(argv[++i]); }
        else if ((a == "-c" || a == "--ctx-size")  && i + 1 < argc) { n_ctx    = atoi(argv[++i]); }
        else if ((a == "-d" || a == "--draft")     && i + 1 < argc) { n_draft  = atoi(argv[++i]); }
        else if ((a == "-t" || a == "--threads")   && i + 1 < argc) { n_threads = atoi(argv[++i]); }
        else if (a == "--temp"   && i + 1 < argc) { temperature = atof(argv[++i]); }
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
    mparams.n_gpu_layers       = 0; // CPU-only for portability; override with env if desired

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        LOG_ERR("Failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ── target context (full model) ─────────────────────────────────────────
    llama_context_params cparams_tgt = llama_context_default_params();
    cparams_tgt.n_ctx            = (uint32_t)n_ctx;
    // n_batch must cover: (a) initial prompt prefill, (b) verify batch of n_draft+1,
    // (c) replay batch of at most n_draft+1. Use n_ctx to handle arbitrary prompts.
    cparams_tgt.n_batch          = (uint32_t)n_ctx;
    cparams_tgt.n_threads        = n_threads;
    cparams_tgt.n_threads_batch  = n_threads;
    cparams_tgt.mtp_draft_mode   = false;
    // n_seq_max = 2: seq 0 = active target state; seq 1 = pre-verify snapshot.
    // The verify batch contaminates ctx_tgt's recurrent state with rejected draft tokens;
    // restoring from the snapshot and replaying only accepted tokens fixes this.
    cparams_tgt.n_seq_max        = 2;

    llama_context * ctx_tgt = llama_init_from_model(model, cparams_tgt);
    if (!ctx_tgt) {
        LOG_ERR("Failed to create target context\n");
        llama_model_free(model);
        return 1;
    }

    // ── draft context (recurrent-only) ──────────────────────────────────────
    // mtp_draft_mode = true causes full-attention layers to be bypassed in Qwen3.5,
    // creating a cheaper forward pass using only the Delta Net recurrent layers.
    llama_context * ctx_dft = nullptr;
    if (!disable_mtp) {
        llama_context_params cparams_dft = llama_context_default_params();
        cparams_dft.n_ctx           = (uint32_t)n_ctx;
        cparams_dft.n_batch         = 1;
        cparams_dft.n_threads       = n_threads;
        cparams_dft.n_threads_batch = n_threads;
        cparams_dft.mtp_draft_mode  = true;   // ← key: skip full-attention layers
        // n_seq_max = 2: slot 0 = active draft state; slot 1 = snapshot taken before
        // each draft phase and used to restore to the exact accepted-token state.
        // Without this, rejected draft tokens contaminate the recurrent state.
        cparams_dft.n_seq_max       = 2;

        ctx_dft = llama_init_from_model(model, cparams_dft);
        if (!ctx_dft) {
            LOG_ERR("Failed to create draft context — falling back to standard decode\n");
            disable_mtp = true;
        }
    }

    const bool use_mtp = !disable_mtp && ctx_dft;

    // ── tokenize prompt ─────────────────────────────────────────────────────
    std::vector<llama_token> tokens_inp;
    {
        tokens_inp = common_tokenize(ctx_tgt, prompt, /*add_special=*/true, /*parse_special=*/true);
    }

    if ((int)tokens_inp.size() >= n_ctx) {
        LOG_ERR("Prompt too long (%d tokens) for context size %d\n", (int)tokens_inp.size(), n_ctx);
        return 1;
    }

    // ── set up sampler ──────────────────────────────────────────────────────
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
        // Always greedy for draft to maximise acceptance
        llama_sampler_chain_add(smpl_dft, llama_sampler_init_greedy());
    }

    // ── encode prompt (both contexts) ───────────────────────────────────────
    LOG_INF("\n== Qwen3.5 MTP%s ==\n", use_mtp ? " (draft+verify)" : " (disabled)");
    LOG_INF("Prompt: \"%s\"\n", prompt.c_str());
    LOG_INF("n_predict=%d  n_draft=%d  temperature=%.2f\n\n", n_predict, use_mtp ? n_draft : 0, temperature);

    // Encode all but last prompt token (last is handled as first decode step)
    {
        auto inp_minus_last = std::vector<llama_token>(tokens_inp.begin(), tokens_inp.end() - 1);
        if (!inp_minus_last.empty()) {
            llama_decode(ctx_tgt, llama_batch_get_one(inp_minus_last.data(), (int)inp_minus_last.size()));
            if (use_mtp) {
                // Draft context has n_batch=1 (single-token), so prefill token-by-token
                for (llama_token & tok : inp_minus_last) {
                    llama_decode(ctx_dft, llama_batch_get_one(&tok, 1));
                }
            }
        }
    }

    llama_token id_last = tokens_inp.back();
    int n_past     = (int)tokens_inp.size() - 1;
    int n_total    = 0;
    int n_drafted  = 0;
    int n_accepted = 0;

    // ── timing ──────────────────────────────────────────────────────────────
    const auto t_start = ggml_time_us();

    // ── decode loop ─────────────────────────────────────────────────────────
    LOG_INF("Output: ");
    fflush(stdout);

    llama_batch batch_tgt = llama_batch_init(n_draft + 2, 0, 1);

    llama_memory_t mem_dft = use_mtp ? llama_get_memory(ctx_dft) : nullptr;
    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);

    while (n_total < n_predict) {
        if (use_mtp) {
            // ── SNAPSHOT draft context before drafting ────────────────────────
            // We copy seq 0 → seq 1 so we can restore exactly to this state after
            // partial acceptance (rejected draft tokens would otherwise contaminate
            // the Delta Net recurrent state, reducing future acceptance rates).
            llama_memory_seq_cp(mem_dft, 0, 1, 0, -1);
            llama_memory_seq_cp(mem_tgt, 0, 1, 0, -1);
            llama_token id_last_iter = id_last; // saved for both replays below
            const int old_n_past     = n_past;

            // ── DRAFT PHASE ──────────────────────────────────────────────────
            // Generate n_draft candidate tokens using the cheap recurrent-only path.
            std::vector<llama_token> draft;
            draft.reserve(n_draft);

            // Feed id_last into the draft context
            {
                llama_batch b = llama_batch_get_one(&id_last, 1);
                llama_decode(ctx_dft, b);
            }

            for (int d = 0; d < n_draft; ++d) {
                llama_token tok = llama_sampler_sample(smpl_dft, ctx_dft, -1);
                if (llama_vocab_is_eog(vocab, tok)) break;
                draft.push_back(tok);

                llama_batch b = llama_batch_get_one(&tok, 1);
                if (llama_decode(ctx_dft, b) != 0) break;
            }

            // ── VERIFY PHASE ─────────────────────────────────────────────────
            // Submit [id_last, draft[0], draft[1], ...] to the target model in one batch.
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, id_last, n_past, {0}, true);
            for (size_t d = 0; d < draft.size(); ++d) {
                common_batch_add(batch_tgt, draft[d], n_past + 1 + (int)d, {0}, true);
            }

            if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                LOG_ERR("\nllama_decode (target) failed\n");
                break;
            }

            // Greedy acceptance: walk target logits, accept tokens matching draft
            n_drafted += (int)draft.size();

            std::vector<llama_token> accepted;
            for (int pos = 0; pos <= (int)draft.size(); ++pos) {
                llama_token sampled = llama_sampler_sample(smpl_tgt, ctx_tgt, pos);
                accepted.push_back(sampled);
                if (pos < (int)draft.size() && sampled != draft[pos]) {
                    break; // mismatch — stop accepting
                }
            }

            // The last element of accepted is always a fresh target sample (bonus token).
            n_accepted  += (int)accepted.size() - 1;
            n_past      += (int)accepted.size();
            n_total     += (int)accepted.size();

            bool stop = false;
            for (llama_token tok : accepted) {
                if (llama_vocab_is_eog(vocab, tok)) { stop = true; break; }
                LOG("%s", common_token_to_piece(ctx_tgt, tok).c_str());
                fflush(stdout);
            }

            // ── RESTORE target context ────────────────────────────────────────
            // The verify batch advanced ctx_tgt's recurrent state to old_n_past+n_draft.
            // seq_rm cannot roll it back. Restore from snapshot and replay accepted
            // tokens in one batch so the recurrent state is exactly at n_past-1.
            llama_memory_seq_rm(mem_tgt, 0, 0, -1);     // clear seq 0 completely
            llama_memory_seq_cp(mem_tgt, 1, 0, 0, -1);  // restore from snapshot
            llama_memory_seq_rm(mem_tgt, 1, 0, -1);     // free snapshot slot

            if (!stop) {
                common_batch_clear(batch_tgt);
                common_batch_add(batch_tgt, id_last_iter, old_n_past, {0}, false);
                for (int i = 0; i < (int)accepted.size() - 1; ++i) {
                    common_batch_add(batch_tgt, accepted[i], old_n_past + 1 + i, {0}, false);
                }
                llama_decode(ctx_tgt, batch_tgt);
            }

            // ── RESTORE draft context ─────────────────────────────────────────
            // Reset seq 0 to the snapshot, replay accepted tokens one-by-one so
            // ctx_dft's recurrent state is at n_past-1 for the next draft phase.
            llama_memory_seq_rm(mem_dft, 0, 0, -1);     // clear seq 0 completely
            llama_memory_seq_cp(mem_dft, 1, 0, 0, -1);  // restore from snapshot
            llama_memory_seq_rm(mem_dft, 1, 0, -1);     // free snapshot slot

            {
                llama_decode(ctx_dft, llama_batch_get_one(&id_last_iter, 1));
                for (int i = 0; i < (int)accepted.size() - 1; ++i) {
                    llama_decode(ctx_dft, llama_batch_get_one(&accepted[i], 1));
                }
            }

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

    const auto t_end = ggml_time_us();
    const double elapsed_s = (t_end - t_start) * 1e-6;

    // ── statistics ──────────────────────────────────────────────────────────
    LOG_INF("\n\n====== Performance ======\n");
    LOG_INF("mode          : %s\n", use_mtp ? "Qwen3.5 MTP (self-speculative)" : "standard autoregressive");
    LOG_INF("tokens        : %d\n", n_total);
    LOG_INF("elapsed       : %.3f s\n", elapsed_s);
    LOG_INF("throughput    : %.2f tok/s\n", n_total / elapsed_s);
    if (use_mtp && n_drafted > 0) {
        LOG_INF("drafted       : %d\n", n_drafted);
        LOG_INF("accepted      : %d\n", n_accepted);
        LOG_INF("accept rate   : %.1f%%\n", 100.0 * n_accepted / n_drafted);
        LOG_INF("speedup steps : %.2fx  (fewer decode() calls)\n",
                (double)n_total / (double)(n_total - n_accepted));
    }
    LOG_INF("=========================\n");

    // ── cleanup ──────────────────────────────────────────────────────────────
    llama_batch_free(batch_tgt);
    llama_sampler_free(smpl_tgt);
    if (smpl_dft) llama_sampler_free(smpl_dft);
    if (ctx_dft)  llama_free(ctx_dft);
    llama_free(ctx_tgt);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
