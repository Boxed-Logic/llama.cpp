#!/usr/bin/env python3
"""
Create a minimal synthetic Qwen3.5-0.8B GGUF model for testing MTP implementation.

This creates a structurally correct but randomly-initialized model with the same
tensor layout and hyperparameters as Qwen3.5-0.8B. The model produces random outputs
but is sufficient to test the MTP infrastructure and measure throughput.

Architecture:
  - 16 transformer layers (hybrid: linear + full attention)
  - Full attention every 4th layer (layers 3, 7, 11, 15)
  - Recurrent (Delta Net) layers: 0,1,2,4,5,6,8,9,10,12,13,14
  - embedding size: 1024
  - attention heads: 16, kv heads: 8
  - FFN size: 4096
  - Vocab size: 4096 (small for test)
"""

import struct
import numpy as np
import os
import sys

# Try to use the bundled gguf package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gguf-py'))

try:
    from gguf import GGUFWriter, GGMLQuantizationType, GGUFValueType
except ImportError:
    print("ERROR: gguf package not found. Install with: pip install gguf")
    sys.exit(1)

# ─── Architecture hyperparameters ───────────────────────────────────────────

ARCH = "qwen35"
N_LAYER = 16
N_EMBD = 1024
N_HEAD = 16
N_HEAD_KV = 8
N_FF = 4096
N_VOCAB = 4096          # small for testing
N_CTX_TRAIN = 4096
N_ROPE_DIM = 64         # n_embd / n_head = 64
FULL_ATTN_INTERVAL = 4  # every 4th layer is full attention

# Qwen3.5 linear attention (Delta Net) parameters
SSM_D_CONV = 4          # conv kernel size
SSM_D_INNER = N_EMBD    # inner size (d_inner = n_embd for 0.8B)
SSM_D_STATE = 64        # key/value head dimension
SSM_DT_RANK = 16        # number of value heads
SSM_N_GROUP = 8         # number of key heads (must divide SSM_DT_RANK evenly)
RMS_NORM_EPS = 1e-6

RNG = np.random.default_rng(42)

def rand_f32(shape):
    """Gaussian random float32 tensor, normalized to unit RMS."""
    arr = RNG.standard_normal(shape).astype(np.float32)
    rms = np.sqrt(np.mean(arr ** 2)) + 1e-8
    return (arr / rms).astype(np.float32)

def ones_f32(shape):
    return np.ones(shape, dtype=np.float32)

def is_recurrent(layer_idx):
    """Returns True if layer uses linear (Delta Net) attention, False for full attention."""
    return (layer_idx + 1) % FULL_ATTN_INTERVAL != 0

def create_qwen35_gguf(output_path: str):
    writer = GGUFWriter(output_path, ARCH)

    # ── Global metadata ──────────────────────────────────────────────────────
    writer.add_name("Qwen3.5-0.8B-synthetic-test")
    writer.add_description("Synthetic Qwen3.5-0.8B for MTP testing (random weights)")
    writer.add_file_type(GGMLQuantizationType.F32)

    # ── Core architecture ────────────────────────────────────────────────────
    writer.add_block_count(N_LAYER)
    writer.add_context_length(N_CTX_TRAIN)
    writer.add_embedding_length(N_EMBD)
    writer.add_feed_forward_length(N_FF)
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    writer.add_layer_norm_rms_eps(RMS_NORM_EPS)
    writer.add_rope_dimension_count(N_ROPE_DIM)

    # Qwen3.5 RoPE sections (required for MRoPE)
    writer.add_rope_dimension_sections([N_ROPE_DIM // 2, 0, N_ROPE_DIM // 2, 0])

    # Linear attention (Delta Net) parameters
    writer.add_ssm_conv_kernel(SSM_D_CONV)
    writer.add_ssm_inner_size(SSM_D_INNER)
    writer.add_ssm_state_size(SSM_D_STATE)
    writer.add_ssm_time_step_rank(SSM_DT_RANK)
    writer.add_ssm_group_count(SSM_N_GROUP)

    # Hybrid attention pattern
    writer.add_uint32("qwen35.full_attention_interval", FULL_ATTN_INTERVAL)

    # Vocabulary (minimal BPE-style)
    # Use simple byte-level vocabulary for testing
    tokens = []
    scores = []
    token_types = []
    for i in range(N_VOCAB):
        if i == 0:
            tokens.append("<unk>")
            token_types.append(2)   # UNKNOWN
        elif i == 1:
            tokens.append("<s>")
            token_types.append(3)   # BOS
        elif i == 2:
            tokens.append("</s>")
            token_types.append(3)   # EOS
        else:
            tokens.append(f"tok_{i}")
            token_types.append(1)   # NORMAL
        scores.append(-float(i))

    writer.add_tokenizer_model("gpt2")
    # Add minimal BPE merges for a functional (but toy) vocabulary
    # Format: "token_a token_b" indicating token_a + token_b merge
    merges = [f"tok_{i} tok_{i+1}" for i in range(3, N_VOCAB - 1)]
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)

    # ── Tensors ──────────────────────────────────────────────────────────────
    print(f"Creating tensors for {N_LAYER}-layer Qwen3.5 model...")

    # Token embeddings
    writer.add_tensor("token_embd.weight", rand_f32([N_VOCAB, N_EMBD]))

    # Output norm + head
    writer.add_tensor("output_norm.weight", ones_f32([N_EMBD]))
    writer.add_tensor("output.weight", rand_f32([N_VOCAB, N_EMBD]))

    head_dim = N_EMBD // N_HEAD       # = 64
    n_embd_k_gqa = N_HEAD_KV * head_dim  # = 512

    # Linear attention dimensions
    head_k_dim = SSM_D_STATE          # = 64
    head_v_dim = SSM_D_STATE          # = 64
    n_k_heads  = SSM_N_GROUP          # = 8
    n_v_heads  = SSM_DT_RANK          # = 16
    key_dim    = head_k_dim * n_k_heads   # = 512
    value_dim  = head_v_dim * n_v_heads   # = 1024
    conv_dim   = key_dim * 2 + value_dim  # = 2048

    for il in range(N_LAYER):
        prefix = f"blk.{il}"
        recurrent = is_recurrent(il)
        layer_type = "recurrent" if recurrent else "full-attn"
        print(f"  layer {il:2d} [{layer_type}]")

        # Always present
        writer.add_tensor(f"{prefix}.attn_norm.weight",             ones_f32([N_EMBD]))
        writer.add_tensor(f"{prefix}.post_attention_norm.weight",   ones_f32([N_EMBD]))  # attn_post_norm

        if not recurrent:
            # Full attention tensors
            # wq: [n_embd, n_head * head_dim * 2]  (Q + Q-gate in Qwen3.5)
            writer.add_tensor(f"{prefix}.attn_q.weight",     rand_f32([N_HEAD * head_dim * 2, N_EMBD]))
            writer.add_tensor(f"{prefix}.attn_k.weight",     rand_f32([n_embd_k_gqa, N_EMBD]))
            writer.add_tensor(f"{prefix}.attn_v.weight",     rand_f32([n_embd_k_gqa, N_EMBD]))
            writer.add_tensor(f"{prefix}.attn_output.weight",rand_f32([N_EMBD, N_HEAD * head_dim]))
            writer.add_tensor(f"{prefix}.attn_q_norm.weight",ones_f32([head_dim]))
            writer.add_tensor(f"{prefix}.attn_k_norm.weight",ones_f32([head_dim]))
        else:
            # Linear attention (Delta Net) tensors
            # in_proj_qkv: combined Q, K, V projection
            writer.add_tensor(f"{prefix}.attn_qkv.weight",  rand_f32([key_dim * 2 + value_dim, N_EMBD]))
            # in_proj_z: gate
            writer.add_tensor(f"{prefix}.attn_gate.weight", rand_f32([value_dim, N_EMBD]))
            # conv1d kernel
            writer.add_tensor(f"{prefix}.ssm_conv1d.weight",rand_f32([conv_dim, SSM_D_CONV]))
            # dt bias
            writer.add_tensor(f"{prefix}.ssm_dt.bias",      np.zeros([n_v_heads], dtype=np.float32))
            # A_noscan (non-scan A parameter)
            writer.add_tensor(f"{prefix}.ssm_a",            -np.ones([n_v_heads], dtype=np.float32))
            # beta and alpha projections
            writer.add_tensor(f"{prefix}.ssm_beta.weight",  rand_f32([n_v_heads, N_EMBD]))
            writer.add_tensor(f"{prefix}.ssm_alpha.weight", rand_f32([n_v_heads, N_EMBD]))
            # output norm (for gated norm)
            writer.add_tensor(f"{prefix}.ssm_norm.weight",  ones_f32([head_v_dim]))
            # output projection
            writer.add_tensor(f"{prefix}.ssm_out.weight",   rand_f32([N_EMBD, value_dim]))

        # FFN (all layers)
        writer.add_tensor(f"{prefix}.ffn_gate.weight", rand_f32([N_FF, N_EMBD]))
        writer.add_tensor(f"{prefix}.ffn_down.weight", rand_f32([N_EMBD, N_FF]))
        writer.add_tensor(f"{prefix}.ffn_up.weight",   rand_f32([N_FF, N_EMBD]))

    print("Writing GGUF file...")
    writer.write_header_to_file(output_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"✓ Model written to: {output_path}")
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "/home/user/llama.cpp/models/qwen35-0.8b-test.gguf"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    create_qwen35_gguf(output)
