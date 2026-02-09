# Optimization Proposal: Granite 4 H Tiny Q4_K_M — Prefill & Decode on AMD64 AVX2

## Executive Summary

This document is a complete implementation proposal for optimizing the **Granite Hybrid
(Mamba2 + Attention + MoE)** model with **Q4_K_M quantization** on AMD64 laptop CPUs with
AVX2. It covers both **prefill** (multi-token prompt processing) and **decode** (single-token
generation).

Five changes are proposed across four files. Combined estimate: **+15-25% prefill, +15-30%
decode**.

---

## Architecture Context

The Granite Hybrid model (`src/models/granite-hybrid.cpp:24-49`) alternates per-layer between:

- **Mamba2 (SSM) layers** — `hparams.is_recurrent(il) == true`
- **Attention layers** — standard GQA with optional RoPE
- **FFN layers** — dense SwiGLU or MoE with softmax gating (+ optional shared expert)

### Critical Path Difference: Prefill vs Decode

| Aspect | Prefill | Decode |
|--------|---------|--------|
| Tokens per call | 100s-1000s | 1 |
| Matmul dispatch | Repacked GEMM (`ggml_gemm_q4_K_8x8_q8_K`) | Standard vec_dot (`ggml_vec_dot_q4_K_q8_K`) |
| Matmul % of time | ~65-70% | ~40-50% |
| SSM state I/O | Amortized across tokens | **Full state read+write per layer per token** |
| SSM scan % of time | ~10-15% | **~25-35%** |
| Bottleneck | Compute + weight bandwidth | **State memory bandwidth** |

The key finding is that **decode uses a completely different matmul path** (standard `vec_dot`
via `ggml-cpu.c:1365-1421`, NOT the repacked GEMV) and is dominated by **SSM state memory
bandwidth** (~19 MB read + ~19 MB write per Mamba2 layer).

---

## Change 1: Prefetch in Q4_K vec_dot (Decode + Prefill Fallback)

**File:** `ggml/src/ggml-cpu/arch/x86/quants.c`
**Function:** `ggml_vec_dot_q4_K_q8_K` (line 1742)
**Targets:** Decode matmul, prefill when repack is disabled

### What

Add `_mm_prefetch` for the next 2 blocks of both weight (`x`) and activation (`y`) data inside
the AVX2 main loop. This mirrors the existing pattern in `ggml_vec_dot_q4_0_q8_0` (line
624-643) which already prefetches.

### Exact Change

At `quants.c:1768`, inside the `for (int i = 0; i < nb; ++i)` loop, insert before the existing
body:

```c
for (int i = 0; i < nb; ++i) {

+       // Prefetch weight and activation blocks 2 iterations ahead
+       if (i + 2 < nb) {
+           _mm_prefetch((const char *)&x[i + 2],       _MM_HINT_T0);
+           _mm_prefetch((const char *)&x[i + 2].qs[64], _MM_HINT_T0); // 2nd cache line of qs[]
+           _mm_prefetch((const char *)&y[i + 2],       _MM_HINT_T0);
+           _mm_prefetch((const char *)&y[i + 2].qs[128], _MM_HINT_T0); // 2nd half of Q8_K qs[]
+       }

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        // ... rest unchanged ...
```

**Rationale for distance=2:** Each Q4_K block is 144 bytes (~3 cache lines). The inner loop
has 4 iterations per block doing 2 loads each = ~20 instructions of compute. At ~5 ns per
instruction, that's ~100 ns — enough time for an L2 prefetch (~12 ns) but not L3 (~35 ns).
Prefetching 2 blocks ahead gives ~200 ns of lead time, covering even L3 access.

### Impact

| Metric | Estimate | Rationale |
|--------|----------|-----------|
| **Decode** | **+5-8%** | This IS the decode matmul path; weight streaming is the bottleneck |
| Prefill | +2-4% | Fallback path when repacking not active; modest since GEMM path dominates |

---

## Change 2: Prefetch in Repacked GEMV and GEMM (Prefill + Batched Decode)

**File:** `ggml/src/ggml-cpu/arch/x86/repack.cpp`
**Functions:** `ggml_gemv_q4_K_8x8_q8_K` (line 1392), `ggml_gemm_q4_K_8x8_q8_K` (line 1957)
**Targets:** Prefill matmul (dominant path)

### What

Add prefetch to the block-level loop in both the GEMV (single-row) and GEMM (multi-row)
repacked kernels. Each `block_q4_Kx8` is 1168 bytes (~18 cache lines). The kernel's inner loop
issues 8 × 256-bit loads from `b_ptr[b].qs` per sub-block, plus scale loads — all hitting cold
cache.

### Exact Change — GEMV

At `repack.cpp:1448`, inside `for (int64_t b = 0; b < nb; b++)`:

```c
for (int64_t b = 0; b < nb; b++) {

+           // Prefetch next Q4_Kx8 block header + first 4 cache lines of qs
+           if (b + 1 < nb) {
+               _mm_prefetch((const char *)&b_ptr[b + 1],         _MM_HINT_T0); // d, dmin, scales
+               _mm_prefetch((const char *)b_ptr[b + 1].qs,       _MM_HINT_T0); // qs[0..63]
+               _mm_prefetch((const char *)b_ptr[b + 1].qs + 64,  _MM_HINT_T0); // qs[64..127]
+               _mm_prefetch((const char *)b_ptr[b + 1].qs + 128, _MM_HINT_T0); // qs[128..191]
+               _mm_prefetch((const char *)b_ptr[b + 1].qs + 192, _MM_HINT_T0); // qs[192..255]
+               _mm_prefetch((const char *)&a_ptr[b + 1],         _MM_HINT_T0); // Q8_K activation
+           }

            const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));
            // ... rest unchanged ...
```

### Exact Change — GEMM

At `repack.cpp:1957`, same pattern in the analogous block loop inside `ggml_gemm_q4_K_8x8_q8_K`.
Add identical prefetch for `b_ptr[b+1]` and `a_ptr[b+1]` at the top of the block loop. The GEMM
processes 4 activation rows (`block_q8_Kx4`), so also prefetch:

```c
+               _mm_prefetch((const char *)&a_ptr[b + 1],          _MM_HINT_T0);
+               _mm_prefetch((const char *)a_ptr[b + 1].qs + 128,  _MM_HINT_T0);
```

### Impact

| Metric | Estimate | Rationale |
|--------|----------|-----------|
| **Prefill** | **+5-10%** | Repacked GEMM is THE prefill matmul path; hides L2/L3 latency on weight streaming |
| Decode | +0% | Decode does NOT use the repacked path (uses vec_dot instead) |

---

## Change 3: Prefetch SSM State in Scan Kernel (Decode Dominant)

**File:** `ggml/src/ggml-cpu/ops.cpp`
**Function:** `ggml_compute_forward_ssm_scan_f32` (line 9185)
**Targets:** Decode (primary), Prefill (secondary)

### What

The Mamba2 SSM scan loop at lines 9244-9336 iterates over `n_head` heads, then `nr` (dim)
rows per head, accessing the state array `s0[i0 + ii*nc]` where `ii = i1 + h*nr`. For a
typical Granite Hybrid with d_state=16, dim~=128 (head_dim), n_head~=24:

- State per head: `dim * d_state * 4 bytes = 128 * 16 * 4 = 8 KB`
- Total state per layer: `n_head * 8 KB = 192 KB` (fits in L2 but not L1)
- State is read AND written (s0 read, s written) = 384 KB of traffic per layer

The inner `d_state` loop (nc=16) is already SIMD vectorized via `GGML_F32_VEC` macros, but
there is **zero prefetching** of the next dim-row's state data. Since the stride between
consecutive `i1` iterations is `nc * sizeof(float) = 64 bytes` = exactly 1 cache line, and
the loop body takes only ~10-15 cycles (2 loads + 2 muls + 1 add + 1 FMA + 1 store for 16
floats), the pipeline stalls waiting for the next cache line on every iteration.

### Exact Change

At `ops.cpp:9251`, inside the `for (int i1 = 0; i1 < nr; ++i1)` loop, before the SIMD section:

```c
                    for (int i1 = 0; i1 < nr; ++i1) {
                        const int ii = i1 + h*nr;
                        const float x_dt = x[ii] * dt_soft_plus;
                        float sumf = 0.0f;

+                       // Prefetch state for 4 rows ahead (256 bytes = 4 cache lines of lead)
+                       if (i1 + 4 < nr) {
+                           _mm_prefetch((const char *)(s0 + (i1 + 4 + h*nr)*nc), _MM_HINT_T0);
+                           _mm_prefetch((const char *)(s  + (i1 + 4 + h*nr)*nc), _MM_HINT_T1);
+                       }

#if defined(GGML_SIMD)
```

Note: `s0` (input state) is prefetched to L1 (`_MM_HINT_T0`) since it's read immediately.
`s` (output state) is prefetched to L2 (`_MM_HINT_T1`) since it's written — this primes the
cache line for the write-allocate without polluting L1.

Also add prefetch at the head-level boundary to pre-warm the first rows of the next head:

```c
                for (int h = ih0; h < ih1; ++h) {
                    const float dt_soft_plus = ggml_compute_softplus_f32(dt[h]);
                    const float dA = expf(dt_soft_plus * A[h]);
                    const int g = h / (nh / ng);

+                   // Prefetch B and C vectors for this head's group
+                   _mm_prefetch((const char *)(B + g*nc), _MM_HINT_T0);
+                   _mm_prefetch((const char *)(C + g*nc), _MM_HINT_T0);
+                   // Prefetch first state rows for this head
+                   _mm_prefetch((const char *)(s0 + h*nr*nc), _MM_HINT_T0);

                    for (int i1 = 0; i1 < nr; ++i1) {
```

### Impact

| Metric | Estimate | Rationale |
|--------|----------|-----------|
| **Decode** | **+8-15%** | SSM scan is ~25-35% of decode; state streaming is the dominant cost |
| Prefill | +2-4% | SSM scan is ~10-15% of prefill; state still accessed but amortized |

This is the **single highest-impact decode optimization** because it directly targets the
~384 KB per-layer state streaming that dominates decode time.

---

## Change 4: SIMD Vectorization of SSM Convolution (Prefill + Decode)

**File:** `ggml/src/ggml-cpu/ops.cpp`
**Function:** `ggml_compute_forward_ssm_conv_f32` (line 9115)
**Targets:** Both prefill and decode

### What

The SSM convolution kernel is **entirely scalar**. The inner loop iterates over `d_conv`
(typically 4) and the outer loop over `d_inner` rows. The outer loop is trivially vectorizable
across rows since each row's dot product is independent.

The key insight: `d_conv` is small enough to fully unroll, and the `c` (weight) array has
stride `nc` (=`d_conv`=4) between rows — meaning `c[i0 + i1*nc]` for 8 consecutive `i1`
values loads from addresses spaced 16 bytes apart. This is exactly what `_mm256_i32gather_ps`
does, but since gather is slow on AMD Zen, we use explicit `_mm256_set_ps` construction
instead.

### Exact Change

Replace the loop body at `ops.cpp:9143-9165` with:

```c
    for (int i3 = 0; i3 < n_s; ++i3) {
        for (int i2 = 0; i2 < n_t; ++i2) {
            const float * s = (const float *) ((const char *) src0->data + ir0*(src0->nb[1]) + i2*(src0->nb[0]) + i3*(src0->nb[2]));
            const float * c = (const float *) ((const char *) src1->data + ir0*(src1->nb[1]));
            float * x = (float *) ((char *) dst->data + ir0*(dst->nb[0]) + i2*(dst->nb[1]) + i3*(dst->nb[2]));

#if defined(__AVX2__) && defined(__FMA__)
            // Vectorize across d_inner rows: process 8 rows at a time
            const int ir8 = ir & ~7;
            for (int i1 = 0; i1 < ir8; i1 += 8) {
                __m256 sum = _mm256_setzero_ps();
                for (int i0 = 0; i0 < nc; ++i0) {
                    // Gather 8 values from s[i0 + (i1+k)*ncs] for k=0..7
                    __m256 sv = _mm256_set_ps(
                        s[i0 + (i1+7)*ncs], s[i0 + (i1+6)*ncs],
                        s[i0 + (i1+5)*ncs], s[i0 + (i1+4)*ncs],
                        s[i0 + (i1+3)*ncs], s[i0 + (i1+2)*ncs],
                        s[i0 + (i1+1)*ncs], s[i0 + (i1+0)*ncs]);
                    // Gather 8 values from c[i0 + (i1+k)*nc] for k=0..7
                    __m256 cv = _mm256_set_ps(
                        c[i0 + (i1+7)*nc], c[i0 + (i1+6)*nc],
                        c[i0 + (i1+5)*nc], c[i0 + (i1+4)*nc],
                        c[i0 + (i1+3)*nc], c[i0 + (i1+2)*nc],
                        c[i0 + (i1+1)*nc], c[i0 + (i1+0)*nc]);
                    sum = _mm256_fmadd_ps(sv, cv, sum);
                }
                _mm256_storeu_ps(x + i1, sum);
            }
            // Scalar remainder
            for (int i1 = ir8; i1 < ir; ++i1) {
                float sumf = 0.0f;
                for (int i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
#else
            // Original scalar path
            for (int i1 = 0; i1 < ir; ++i1) {
                float sumf = 0.0f;
                for (int i0 = 0; i0 < nc; ++i0) {
                    sumf += s[i0 + i1*ncs] * c[i0 + i1*nc];
                }
                x[i1] = sumf;
            }
#endif
        }
    }
```

### Impact

| Metric | Estimate | Rationale |
|--------|----------|-----------|
| Prefill | **+3-5%** | SSM conv is ~5-8% of prefill; 2-3x kernel speedup |
| **Decode** | **+3-5%** | Same kernel, same proportion of decode time |
| SSM conv kernel itself | **2-3x** | 8 rows per iteration vs 1; offset by gather construction cost |

---

## Change 5: Increase TENSOR_ALIGNMENT to 64 Bytes (Global)

**File:** `ggml/src/ggml-impl.h`
**Line:** 42
**Targets:** All operations

### What

Change `#define TENSOR_ALIGNMENT 32` to `#define TENSOR_ALIGNMENT 64`.

Currently tensor data within allocation buffers is aligned to 32 bytes (`ggml-alloc.c:81`),
but cache lines are 64 bytes. This means ~50% of tensors start at a 32-byte offset within a
cache line, causing every `_mm256_loadu` at the tensor start to split across two cache lines.

### Exact Change

```c
- #define TENSOR_ALIGNMENT 32
+ #define TENSOR_ALIGNMENT 64
```

### Impact

| Metric | Estimate | Rationale |
|--------|----------|-----------|
| Prefill | **+1-3%** | Eliminates cache-line splits on tensor-start loads across all kernels |
| Decode | **+1-2%** | Same benefit, proportionally smaller since fewer matmuls |
| Memory overhead | **+0.01%** | At most 32 extra bytes of padding per tensor |

This is a low-impact but zero-risk change that benefits every operation in the system.

---

## Combined Impact Summary

| # | Change | File | Prefill | Decode | Risk | Effort |
|---|--------|------|:-------:|:------:|:----:|:------:|
| 1 | Q4_K vec_dot prefetch | `arch/x86/quants.c` | +2-4% | **+5-8%** | None | 15 min |
| 2 | Repacked GEMV/GEMM prefetch | `arch/x86/repack.cpp` | **+5-10%** | +0% | None | 30 min |
| 3 | SSM scan state prefetch | `ops.cpp` | +2-4% | **+8-15%** | None | 30 min |
| 4 | SSM conv AVX2 vectorization | `ops.cpp` | +3-5% | +3-5% | Low | 2 hrs |
| 5 | TENSOR_ALIGNMENT 32→64 | `ggml-impl.h` | +1-3% | +1-2% | None | 5 min |
| | **Combined (non-additive)** | | **+12-22%** | **+15-25%** | | |

### Implementation Order

1. **Changes 1, 2, 3, 5** — All prefetch + alignment changes. Implement together, benchmark
   as one batch. Zero correctness risk (prefetch is non-functional; alignment is transparent).
   **~1 hour total.**

2. **Change 4** — SSM conv SIMD. Implement separately since it changes computation.
   Requires careful validation. **~2 hours.**

---

## Testing Plan

### Phase 1: Build Verification

```bash
# Clean build with AVX2
cmake -B build -DGGML_AVX2=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Verify binary runs
./build/bin/llama-cli --version
```

### Phase 2: Correctness — Prefetch-Only Changes (1, 2, 3, 5)

Since prefetch instructions are non-functional hints and alignment is transparent, these
changes should produce **bit-identical output**. Verification:

```bash
# 1. Backend ops — full test suite
./build/bin/test-backend-ops -b CPU

# 2. Quantization accuracy
./build/bin/test-quantize-perf

# 3. Matmul correctness (includes repacked paths)
./build/bin/test-backend-ops -o MUL_MAT -b CPU

# 4. SSM ops correctness
./build/bin/test-backend-ops -o SSM_CONV -b CPU
./build/bin/test-backend-ops -o SSM_SCAN -b CPU

# 5. Bit-exact output comparison
./build/bin/llama-cli -m granite-hybrid-Q4_K_M.gguf \
    -p "The capital of France is" -n 50 --seed 42 --temp 0 2>/dev/null > out_optimized.txt
# Compare with baseline build output
diff out_baseline.txt out_optimized.txt  # Must be identical
```

### Phase 3: Correctness — SSM Conv SIMD (Change 4)

This changes the computation path, so float rounding may differ slightly:

```bash
# 1. SSM conv backend test (checks against reference implementation)
./build/bin/test-backend-ops -o SSM_CONV -b CPU

# 2. Perplexity regression (tolerance: <0.01 PPL difference)
./build/bin/llama-perplexity -m granite-hybrid-Q4_K_M.gguf \
    -f wikitext-2-raw/wiki.test.raw --chunks 50

# 3. End-to-end text comparison (allow minor float differences)
./build/bin/llama-cli -m granite-hybrid-Q4_K_M.gguf \
    -p "Explain quantum computing in simple terms" -n 100 --seed 42 --temp 0

# 4. Edge case: d_conv != 4 (if any models use different values)
# Run with a Mamba-1 model that may have different d_conv
./build/bin/test-backend-ops -o SSM_CONV -b CPU

# 5. Thread safety: run with different thread counts
./build/bin/llama-bench -m granite-hybrid-Q4_K_M.gguf -p 128 -n 32 -t 1 -r 3
./build/bin/llama-bench -m granite-hybrid-Q4_K_M.gguf -p 128 -n 32 -t 4 -r 3
./build/bin/llama-bench -m granite-hybrid-Q4_K_M.gguf -p 128 -n 32 -t 8 -r 3
```

### Phase 4: Performance Benchmarks

```bash
# A/B: Build baseline (pre-patch) and optimized versions
# Baseline:
git stash && cmake --build build -j$(nproc) && cp build/bin/llama-bench llama-bench-base
git stash pop && cmake --build build -j$(nproc) && cp build/bin/llama-bench llama-bench-opt

# Prefill benchmark (multiple prompt lengths)
for pp in 128 256 512 1024; do
    echo "=== pp=$pp ==="
    ./llama-bench-base -m granite-hybrid-Q4_K_M.gguf -p $pp -n 0 -r 5
    ./llama-bench-opt  -m granite-hybrid-Q4_K_M.gguf -p $pp -n 0 -r 5
done

# Decode benchmark
./llama-bench-base -m granite-hybrid-Q4_K_M.gguf -p 0 -n 128 -r 5
./llama-bench-opt  -m granite-hybrid-Q4_K_M.gguf -p 0 -n 128 -r 5

# Combined (realistic workload)
./llama-bench-base -m granite-hybrid-Q4_K_M.gguf -p 512 -n 128 -r 5
./llama-bench-opt  -m granite-hybrid-Q4_K_M.gguf -p 512 -n 128 -r 5

# Thread scaling
for t in 1 2 4 8; do
    echo "=== threads=$t ==="
    ./llama-bench-opt -m granite-hybrid-Q4_K_M.gguf -p 256 -n 64 -t $t -r 3
done
```

### Phase 5: Hardware Performance Counters

```bash
# Cache miss rates (before vs after)
perf stat -e cache-misses,cache-references,L1-dcache-load-misses,\
L1-dcache-loads,LLC-load-misses,LLC-loads \
    ./build/bin/llama-bench -m granite-hybrid-Q4_K_M.gguf -p 512 -n 0 -r 1

# Instruction throughput
perf stat -e instructions,cycles,branches,branch-misses \
    ./build/bin/llama-bench -m granite-hybrid-Q4_K_M.gguf -p 0 -n 128 -r 1
```

### Phase 6: Regression Guard

```bash
# Non-hybrid model (pure transformer) should not regress
./llama-bench-base -m llama-7b-Q4_K_M.gguf -p 512 -n 128 -r 3
./llama-bench-opt  -m llama-7b-Q4_K_M.gguf -p 512 -n 128 -r 3

# Pure Mamba model (if available) should see maximum SSM benefit
./llama-bench-base -m mamba-model.gguf -p 512 -n 128 -r 3
./llama-bench-opt  -m mamba-model.gguf -p 512 -n 128 -r 3
```

---

## Risk Assessment

| Change | Correctness Risk | Performance Risk | Regression Risk |
|--------|:----------------:|:----------------:|:---------------:|
| 1. vec_dot prefetch | **Zero** — prefetch is a hint, does not change values | None — prefetch can only help or be ignored by CPU | None — no code path change |
| 2. Repack prefetch | **Zero** — same reasoning | None | None |
| 3. SSM scan prefetch | **Zero** — same reasoning | None | None |
| 4. SSM conv SIMD | **Low** — FMA may produce slightly different float rounding vs scalar; validated by test suite | Very low — `_mm256_set_ps` construction has overhead | None for non-SSM models |
| 5. TENSOR_ALIGNMENT | **Zero** — only changes padding between tensors | Very low — slightly more memory (~32B/tensor) | None |

---

## Appendix: Why These Changes Specifically Help Granite Hybrid

### The Mamba2 State Bandwidth Problem

During decode, each Mamba2 layer reads the full state matrix `s0` and writes the updated
state `s`. For Granite Hybrid with typical dimensions:

```
State per layer = d_state × head_dim × n_head × sizeof(float)
                = 16 × 128 × 24 × 4 = 196,608 bytes ≈ 192 KB

Read + Write per layer = 384 KB
With N recurrent layers = N × 384 KB per token
```

On a laptop with ~30 GB/s memory bandwidth, 12 recurrent layers = 4.5 MB/token, consuming
~150 μs of pure bandwidth time. Prefetching (Change 3) reduces this by pipelining reads
ahead of computation, effectively hiding 30-50% of the latency.

### The Decode Matmul Path Mismatch

The repacked GEMV/GEMM kernels (Change 2) are only invoked when the activation matrix has
multiple rows (prefill). For single-token decode, the dispatch logic at `ggml-cpu.c:1365` sets
`chunk_size=64` and falls through to `ggml_vec_dot_q4_K_q8_K` — the **non-repacked** path.
This is why Change 1 (vec_dot prefetch) is critical for decode performance, even though the
repacked path appears to be "the optimized one."

### MoE Expert Weight Access

In MoE layers, expert weights are selected dynamically based on gating scores. Only
`n_expert_used` (typically 2) of `n_expert` (typically 8) experts are activated per token.
The prefetch changes in the matmul kernels (Changes 1, 2) help because the selected expert
weights are streamed sequentially once chosen — the access pattern within each expert is
identical to a dense matmul.
