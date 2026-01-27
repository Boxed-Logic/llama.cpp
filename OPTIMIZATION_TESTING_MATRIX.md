# CPU Optimization Testing Matrix for GitHub Actions

## Summary

This document outlines which optimizations from the CPU kernel analysis can be tested on GitHub Actions, and provides strategies for testing those that cannot.

---

## ✅ Fully Testable on Standard GitHub Actions

These optimizations can be compiled, run, and performance-tested on standard `ubuntu-latest` or `windows-latest` runners:

| # | Optimization | Required ISA | Test Type | Notes |
|---|--------------|--------------|-----------|-------|
| **#2** | Software Prefetching | None (compiler intrinsics) | ✅ Functional + Perf | Works on all CPUs |
| **#3** | Horizontal Sum Optimization | AVX2 | ✅ Functional + Perf | AVX2 guaranteed on runners |
| **#4** | F16C Conversion | F16C | ✅ Functional + Perf | F16C guaranteed on runners |
| **#5** | Loop Unrolling | AVX2 | ✅ Functional + Perf | AVX2 guaranteed |
| **#6** | Decode Fast Path | None | ✅ Functional + Perf | Pure algorithmic change |
| **#8** | IMROPE Vectorization | AVX2/FMA | ✅ Functional + Perf | Can test with AVX2+FMA |
| **#9** | Fused Flash Attention | AVX2 | ✅ Functional + Perf | Base version testable |
| **#10** | Aligned Stores | None | ✅ Functional + Perf | Works everywhere |
| **#11** | Format Specialization | None | ✅ Functional + Perf | Template-based |
| **#12** | FMA Consistency | FMA3 | ✅ Functional + Perf | FMA3 guaranteed |

---

## ⚠️ Conditionally Testable (AVX-512)

These can be tested on GitHub Actions **most of the time**, but hardware is not guaranteed:

| # | Optimization | Required ISA | Strategy |
|---|--------------|--------------|----------|
| **#1a** | VNNI Expansion (AVX512_VNNI) | AVX-512 VNNI | ⚠️ **Use runtime detection**<br>• Compile with `-mavx512f -mavx512vnni`<br>• Test will PASS on ~80% of runners (Intel)<br>• Test will SKIP on ~20% of runners (AMD EPYC)<br>• Use `CPUID` checks to skip if unavailable |

**Testing Approach:**
```yaml
- name: Test AVX512_VNNI optimizations
  run: |
    # Check if AVX512_VNNI is available
    if grep -q avx512vnni /proc/cpuinfo; then
      echo "✅ AVX512_VNNI available - running tests"
      ./test-vnni-optimizations
    else
      echo "⚠️  AVX512_VNNI not available - skipping"
      exit 0
    fi
```

---

## ❌ NOT Testable on Standard GitHub Actions

These require hardware not available in GitHub's runner pool:

| # | Optimization | Required ISA | Why Not Available | Alternative Strategy |
|---|--------------|--------------|-------------------|----------------------|
| **#1b** | VNNI Expansion (AVX_VNNI) | AVX_VNNI | Requires Intel 12th gen (Alder Lake) or AMD Zen 4<br>GitHub uses Cascade Lake (2019) | **Use Intel SDE emulator** (see below) |
| **#7** | MoE Sparse GEMV | None (algorithmic) | Can test **functionally** but perf won't reflect target CPUs | **Functional tests only** on GHA<br>Perf tests on self-hosted |

---

## Testing Strategy Recommendations

### 1. Multi-Tier Testing Approach

```yaml
name: CPU Optimization Tests

jobs:
  # Tier 1: Baseline tests (guaranteed to work)
  test-baseline:
    runs-on: ubuntu-latest
    steps:
      - name: Build with AVX2/FMA/F16C
        run: |
          cmake -B build \
            -DCMAKE_C_FLAGS="-mavx2 -mfma -mf16c" \
            -DCMAKE_CXX_FLAGS="-mavx2 -mfma -mf16c"
          cmake --build build

      - name: Test Optimizations #2-6, #8-12
        run: |
          cd build
          # Test prefetching
          ./test-prefetch
          # Test F16C conversions
          ./test-f16c
          # Test loop unrolling
          ./test-unroll
          # ... etc

  # Tier 2: Conditional AVX-512 tests
  test-avx512:
    runs-on: ubuntu-latest
    steps:
      - name: Build with AVX-512
        run: |
          cmake -B build \
            -DCMAKE_C_FLAGS="-mavx512f -mavx512bw -mavx512vnni" \
            -DCMAKE_CXX_FLAGS="-mavx512f -mavx512bw -mavx512vnni"
          cmake --build build

      - name: Test AVX512_VNNI (conditional)
        run: |
          if grep -q avx512vnni /proc/cpuinfo; then
            ./build/test-vnni-avx512
          else
            echo "⚠️  Skipping AVX512_VNNI tests (not available on this runner)"
          fi

  # Tier 3: Intel SDE emulation for AVX_VNNI
  test-avx-vnni-emulated:
    runs-on: ubuntu-latest
    steps:
      - name: Download Intel SDE
        run: |
          wget https://downloadmirror.intel.com/823664/sde-external-9.33.0-2024-01-07-lin.tar.xz
          tar xf sde-external-9.33.0-2024-01-07-lin.tar.xz

      - name: Build with AVX_VNNI
        run: |
          cmake -B build \
            -DCMAKE_C_FLAGS="-mavxvnni" \
            -DCMAKE_CXX_FLAGS="-mavxvnni"
          cmake --build build

      - name: Test under Intel SDE (emulated)
        run: |
          # Run tests under Alder Lake emulation
          ./sde-external-9.33.0-2024-01-07-lin/sde64 \
            -adl -- ./build/test-vnni-avx
```

### 2. Intel SDE (Software Development Emulator)

For testing **AVX_VNNI** (optimization #1b), use Intel SDE:

**What it does:**
- Emulates newer CPU instruction sets on older hardware
- Can emulate Alder Lake (12th gen) features on Cascade Lake runners
- **Downside:** Very slow (~100x slower), only for functional testing

**Example usage:**
```bash
# Download and extract Intel SDE
wget https://downloadmirror.intel.com/823664/sde-external-9.33.0-2024-01-07-lin.tar.xz
tar xf sde-external-9.33.0-2024-01-07-lin.tar.xz

# Run your test binary under Alder Lake emulation
./sde64 -adl -- ./test-avx-vnni

# This confirms your code WORKS, but don't benchmark it
```

### 3. Performance Testing Strategy

**On GitHub Actions (free):**
- ✅ Functional correctness for all optimizations
- ✅ Relative performance comparison (before/after) for guaranteed ISAs
- ⚠️ AVX-512 performance will be inconsistent
- ❌ Cannot test AVX_VNNI performance

**For accurate performance testing:**

**Option A: Self-Hosted Runners** (Recommended)
```yaml
jobs:
  perf-test-modern-intel:
    runs-on: [self-hosted, linux, avx-vnni]  # Your own hardware
    steps:
      - name: Benchmark on real 12th/13th/14th gen Intel
        run: ./benchmark --model qwen3.gguf
```

**Option B: Third-Party CI with Modern CPUs**
- [Cirrus CI](https://cirrus-ci.org/) - Offers AMD Zen 4 runners
- [Namespace](https://namespace.so/) - Custom hardware
- [Blacksmith](https://useblacksmith.io/) - Performance-focused runners

**Option C: Manual Testing**
- Test locally on your development machines
- Document results in PR comments

### 4. Recommended GitHub Actions Workflow

Create `.github/workflows/cpu-optimizations.yml`:

```yaml
name: CPU Kernel Optimizations

on: [push, pull_request]

jobs:
  # Job 1: Build matrix for different ISA levels
  build-matrix:
    strategy:
      matrix:
        isa:
          - name: AVX2-baseline
            flags: "-mavx2 -mfma -mf16c"
            test_opts: "2,3,4,5,6,8,9,10,11,12"
          - name: AVX512-optional
            flags: "-mavx2 -mavx512f -mavx512bw -mavx512vnni"
            test_opts: "1a"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build with ${{ matrix.isa.name }}
        run: |
          cmake -B build \
            -DCMAKE_C_FLAGS="${{ matrix.isa.flags }}" \
            -DCMAKE_CXX_FLAGS="${{ matrix.isa.flags }}" \
            -DGGML_NATIVE=OFF
          cmake --build build -j$(nproc)

      - name: Run functional tests
        run: |
          cd build
          ctest --output-on-failure

      - name: Benchmark (relative comparison)
        run: |
          # Test decode speed on Qwen3
          ./build/llama-bench \
            --model models/qwen3-8b-q4_k.gguf \
            --n-prompt 512 --n-gen 128
          # Test on Qwen3-VL
          ./build/llama-bench \
            --model models/qwen3-vl-q4_k.gguf \
            --n-prompt 512 --n-gen 128

  # Job 2: Check CPU features available
  check-cpu:
    runs-on: ubuntu-latest
    steps:
      - name: Report CPU features
        run: |
          echo "=== CPU Info ==="
          cat /proc/cpuinfo | grep "model name" | head -1
          echo ""
          echo "=== Instruction Sets ==="
          grep flags /proc/cpuinfo | head -1 | tr ' ' '\n' | grep -E "avx|fma|vnni|f16c"

      - name: Check specific features
        run: |
          echo "AVX2:          $(grep -q avx2 /proc/cpuinfo && echo ✅ || echo ❌)"
          echo "FMA:           $(grep -q fma /proc/cpuinfo && echo ✅ || echo ❌)"
          echo "F16C:          $(grep -q f16c /proc/cpuinfo && echo ✅ || echo ❌)"
          echo "AVX-512F:      $(grep -q avx512f /proc/cpuinfo && echo ✅ || echo ❌)"
          echo "AVX512_VNNI:   $(grep -q avx512vnni /proc/cpuinfo && echo ✅ || echo ❌)"
          echo "AVX_VNNI:      $(grep -q avx_vnni /proc/cpuinfo && echo ✅ || echo ❌)"

  # Job 3: Compare performance (before/after optimization)
  compare-performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build baseline (main branch)
        run: |
          git checkout main
          cmake -B build-baseline -DCMAKE_C_FLAGS="-mavx2 -mfma -mf16c"
          cmake --build build-baseline

      - name: Build optimized (current branch)
        run: |
          git checkout -
          cmake -B build-optimized -DCMAKE_C_FLAGS="-mavx2 -mfma -mf16c"
          cmake --build build-optimized

      - name: Benchmark comparison
        run: |
          echo "=== Baseline Performance ==="
          ./build-baseline/llama-bench --model qwen3.gguf -n 100

          echo "=== Optimized Performance ==="
          ./build-optimized/llama-bench --model qwen3.gguf -n 100

          # Parse and compare results
          # (Add script to calculate speedup percentage)
```

---

## Final Recommendations

### ✅ **CAN Auto-Test on GitHub Actions:**
- Optimizations #2, #3, #4, #5, #6, #8, #9, #10, #11, #12 (100% coverage)
- Optimization #1a (AVX512_VNNI) with conditional skipping (~80% success rate)

### ⚠️ **Requires Special Setup:**
- Optimization #1b (AVX_VNNI): Use Intel SDE for functional tests only
- Performance testing: Use self-hosted runners or manual testing

### 📊 **Testing Priorities:**

1. **High Priority** (test on every PR):
   - Functional correctness for all optimizations
   - Baseline performance regression tests (AVX2)
   - Build with multiple ISA levels

2. **Medium Priority** (test on release):
   - AVX-512 performance (on Intel runners when available)
   - Cross-platform builds (Windows, Linux, macOS)

3. **Low Priority** (manual testing):
   - AVX_VNNI performance on 12th gen+ Intel
   - AMD Zen 4 specific optimizations
   - Absolute performance numbers on target hardware

---

## Example: Detecting Features at Runtime

Include this in your test suite to auto-skip unavailable features:

```cpp
// test-cpu-features.cpp
#include <cpuid.h>
#include <stdio.h>

bool has_avx512_vnni() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ecx & (1 << 11)) != 0;  // AVX512_VNNI is bit 11 of ECX
}

bool has_avx_vnni() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 1, eax, ebx, ecx, edx);
    return (eax & (1 << 4)) != 0;  // AVX_VNNI is bit 4 of EAX
}

int main() {
    printf("AVX512_VNNI: %s\n", has_avx512_vnni() ? "✅" : "❌");
    printf("AVX_VNNI:    %s\n", has_avx_vnni() ? "✅" : "❌");
    return 0;
}
```

Use this to conditionally run tests based on actual hardware capabilities.

---

## Summary Table

| Optimization | GitHub Actions | Strategy |
|--------------|----------------|----------|
| #1a (AVX512_VNNI) | ⚠️ Conditional | Runtime check + skip if unavailable |
| #1b (AVX_VNNI) | ❌ Not available | Intel SDE emulation (functional only) |
| #2-6, #8-12 | ✅ Fully supported | Standard testing + benchmarking |
| #7 (MoE) | ✅ Functional only | Correctness tests, not performance |

**Bottom line:** You can automate **~90% of optimization testing** on GitHub Actions, with ~10% requiring self-hosted runners or manual testing for accurate performance validation.
