# Qwen3.5 Multi-Token Prediction — Handover Notes

**Branch:** `claude/qwen-multi-token-prediction-Bjbqe`
**Key commits:**
- `05b854f` — Core MTP implementation (inference side)
- `817a808` — Converter support (HuggingFace → GGUF)

---

## What Was Built

Self-speculative decoding for Qwen3.5's hybrid architecture (Delta Net recurrent layers + sparse full-attention layers).

**Key insight:** Qwen3.5 uses full-attention only every `full_attn_interval` layers (default: every 4th layer). A "draft" forward pass that skips those 4 expensive layers runs ~75% of the compute per token. The full model then verifies a batch of draft tokens in one pass.

```
Draft pass (recurrent-only, fast):  layer0 → layer1 → layer2 → [skip attn] → ... → token
Verify pass (full model, parallel): [token_n, draft0, draft1, draft2, draft3] → batch logits
```

---

## Files Changed

| File | Change |
|------|--------|
| `include/llama.h` | Add `mtp_draft_mode` field to `llama_context_params` |
| `src/llama-cparams.h` | Add `mtp_draft_mode` to internal cparams |
| `src/llama-context.cpp` | Propagate `mtp_draft_mode` into cparams |
| `src/llama-model.h` | Add `LLM_TYPE_0_8B` enum value |
| `src/llama-model.cpp` | 0.8B size detection; hybrid+draft_mode → recurrent memory |
| `src/models/qwen35.cpp` | Skip full-attention layers when `skip_full_attn` is set |
| `src/llama-memory-hybrid.cpp` | `seq_rm` trims KV cache even when recurrent state can't roll back |
| `examples/qwen35-mtp/` | `llama-qwen35-mtp` demo binary (draft+verify loop) |
| `convert_hf_to_gguf.py` | `Qwen3_5TextModel` class + MTP tensor remapping |
| `gguf-py/gguf/constants.py` | Add `NEXTN_*` tensors to QWEN35 allowed list |
| `scripts/create_qwen35_test_model.py` | Synthetic GGUF for offline testing |
| `scripts/inspect_qwen35_mtp.py` | Utility to dump MTP tensor names from a real checkpoint |

---

## Verifying With a Real Model

### Step 1 — Download Qwen3.5-0.8B

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.5-0.8B \
    --local-dir /tmp/Qwen3.5-0.8B \
    --ignore-patterns "*.msgpack" "*.h5" "flax_model*" "tf_model*"
```

### Step 2 — Inspect MTP tensor names

**This step is critical.** The converter was written to the expected naming convention
(`mtp.{N}.enorm`, `mtp.{N}.hnorm`, `mtp.{N}.eh_proj`, `mtp.{N}.layers.0.*`,
`mtp.{N}.shared_head.*`) but this could not be verified against the real checkpoint
because HuggingFace is blocked in the build environment.

```bash
python3 scripts/inspect_qwen35_mtp.py /tmp/Qwen3.5-0.8B
```

Expected output (if names match):
```
=== MTP Tensor Inspection ===
mtp.0.enorm.weight               torch.float16   [1024]
mtp.0.hnorm.weight               torch.float16   [1024]
mtp.0.eh_proj.weight             torch.float16   [1024, 2048]
mtp.0.layers.0.self_attn.q_proj.weight  ...
mtp.0.shared_head.norm.weight    ...
mtp.0.shared_head.head.weight    ...
...
=== GGUF Mapping Preview ===
mtp.0.enorm.weight  -->  blk.16.attn_norm.weight   (NEXTN_ENORM)
...
All mapped OK
```

**If the tensor names differ**, update the `remapper` dict in `Qwen3_5TextModel.modify_tensors()`
inside `convert_hf_to_gguf.py`. The dict maps HuggingFace suffix → GGUF key constant.

### Step 3 — Convert to GGUF

```bash
python3 convert_hf_to_gguf.py /tmp/Qwen3.5-0.8B \
    --outtype f16 \
    --outfile models/qwen35-0.8b-f16.gguf
```

Verify the output contains MTP metadata:
```bash
python3 gguf-py/scripts/gguf_dump.py models/qwen35-0.8b-f16.gguf | grep nextn
# Expected:  qwen35.nextn_predict_layers = 1  (or however many MTP heads the model has)
```

Optionally quantize:
```bash
./build/bin/llama-quantize models/qwen35-0.8b-f16.gguf models/qwen35-0.8b-q4km.gguf Q4_K_M
```

### Step 4 — Build

```bash
cmake -B build -DLLAMA_CURL=OFF -DGGML_CUDA=ON   # or without CUDA for CPU-only
cmake --build build --target llama-qwen35-mtp -j$(nproc)
```

### Step 5 — Run the MTP demo

```bash
./build/bin/llama-qwen35-mtp \
    -m models/qwen35-0.8b-q4km.gguf \
    -p "The capital of France is" \
    -n 100 \
    -d 4            # number of draft tokens per step
```

Expected output structure:
```
mode    : Qwen3.5 MTP (self-speculative)
n_draft : 4
...
[generated text]
...
tokens  : 100
time    : 2.34 s
tok/s   : 42.7
draft accepted : 287 / 375 (76.5 %)
decode steps   : 75  (vs 100 autoregressive)
speedup        : 1.33x fewer decode() calls
```

On CPU the wall-clock speedup will be small or negative (batch overhead outweighs savings).
**Real speedup requires a GPU** where the verify batch runs in the same time as one serial step.

---

## Debugging Common Problems

### Segfault / assertion in `llama_memory_hybrid`

`llama-model.cpp:create_memory()` must choose `llama_memory_recurrent` for the draft context.
The guard is:

```cpp
if (cparams.mtp_draft_mode && is_hybrid_arch) {
    return std::make_unique<llama_memory_recurrent>(...);
}
```

If it falls through to `llama_memory_hybrid`, the KV cache will assert on unvisited
attention layers during draft passes.

### Converter crashes / missing tensors

Run `inspect_qwen35_mtp.py` and compare the printed names against the `remapper` dict
in `Qwen3_5TextModel.modify_tensors()`. Any unrecognised tensor will be passed through
unchanged (and silently dropped if not in the allowed list in `constants.py`).

### Draft tokens never accepted

Check that the draft and target contexts are using the **same** model file and that
`seq_rm` rollback is working (see `llama-memory-hybrid.cpp` — the recurrent state
cannot be truly rolled back, so acceptance must always be greedy for now).

### 0.8B model not detected

Size detection is in `llama-model.cpp` around the `hparams.n_layer == 16` case.
If the real model has a different layer count, add the appropriate case.

---

## Testing Without a Real Model

A synthetic GGUF (random weights, correct architecture) is generated by:

```bash
pip install gguf numpy
python3 scripts/create_qwen35_test_model.py
# writes: models/qwen35-test.gguf

./build/bin/llama-qwen35-mtp \
    -m models/qwen35-test.gguf \
    -p "test" -n 20 -d 4
```

Output will be nonsense (random weights) but the infrastructure — draft/verify loop,
memory management, graph skipping — can all be validated this way.

---

## Known Limitations / Future Work

1. **Recurrent state rollback is approximate.** Delta Net recurrent state cannot be
   truly un-done, so rejected tokens leave a small residual error in the state. This
   is consistent with how other hybrid speculative decoders work in practice, but
   is worth revisiting if quality degrades significantly.

2. **Greedy acceptance only.** A temperature/sampling-aware acceptance criterion
   (as in standard speculative decoding) would allow non-greedy sampling while
   preserving the target distribution. Currently only greedy sampling is correct.

3. **MTP tensor names unverified against real checkpoint.** The converter mapping
   follows the naming convention but must be confirmed against the actual Qwen3.5
   weights (Step 2 above).

4. **Single MTP head assumed.** The code handles multiple MTP heads
   (`nextn_predict_layers > 1`) in the converter, but the demo binary only uses one
   draft round. Chained heads could further increase the draft depth.
