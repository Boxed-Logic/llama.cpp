# CPU Optimization Testing Automation - Quick Reference

## TL;DR - What Can You Test on GitHub Actions?

| Can Test? | Optimizations | Notes |
|-----------|---------------|-------|
| ✅ **YES** (100% reliable) | #2, #3, #4, #5, #6, #8, #9, #10, #11, #12 | 10/12 optimizations |
| ⚠️ **MAYBE** (~80% of time) | #1a (AVX512_VNNI) | Use conditional skip when unavailable |
| ❌ **NO** (requires special setup) | #1b (AVX_VNNI) | Use Intel SDE emulator or self-hosted runners |

**Bottom line:** You can fully automate testing for **~90%** of the recommended optimizations.

---

## Quick Start

### 1. Check what YOUR hardware supports:

```bash
./scripts/check-cpu-features.sh
```

This will tell you:
- Which optimizations you can test
- Recommended CMake flags
- Whether you're on GitHub Actions or local hardware

### 2. Use the provided GitHub Actions workflow:

The workflow at `.github/workflows/test-cpu-optimizations.yml` automatically:
- ✅ Detects available CPU features
- ✅ Builds with multiple ISA levels
- ✅ Runs functional tests
- ✅ Conditionally tests AVX-512 (skips if unavailable)
- ✅ Compares performance before/after changes
- ⚠️ Emulates AVX_VNNI with Intel SDE (when needed)

---

## GitHub Actions Runner Capabilities (2026)

### What's Available:

| Feature | Availability | Used By |
|---------|--------------|---------|
| AVX2 | ✅ 100% guaranteed | #3, #5, #8, #9, most optimizations |
| FMA3 | ✅ 100% guaranteed | #12 |
| F16C | ✅ 100% guaranteed | #4 |
| AVX-512 F/BW/VL | ⚠️ ~80% (Intel runners) | Base AVX-512 operations |
| AVX-512 VNNI | ⚠️ ~80% (Intel Cascade Lake+) | #1a - Most impactful optimization! |
| AVX_VNNI | ❌ Not available | #1b - Requires 12th gen Intel / Zen 4 |

### Hardware Details:

**Standard runners:**
- Intel Xeon 8272CL (Cascade Lake) - Most common
- Sometimes: AMD EPYC 7763 (no AVX-512)
- Sometimes: Older Intel (Skylake, Haswell, Broadwell)

**Target Windows desktop/laptop CPUs:**
- Intel 11th-14th gen (Tiger Lake, Alder Lake, Raptor Lake)
- AMD Zen 3/4/5 (Ryzen 5000/7000/9000)

These have **AVX_VNNI** (not on GitHub Actions), which is your highest-impact optimization (#1b).

---

## Testing Strategy by Optimization

### Tier 1: Fully Automated (No Special Setup)

| Opt | Name | Test on GHA | Performance Test |
|-----|------|-------------|------------------|
| #2 | Software Prefetching | ✅ Yes | ✅ Reliable |
| #3 | Horizontal Sum Opt | ✅ Yes | ✅ Reliable |
| #4 | F16C Conversion | ✅ Yes | ✅ Reliable |
| #5 | Loop Unrolling | ✅ Yes | ✅ Reliable |
| #6 | Decode Fast Path | ✅ Yes | ✅ Reliable |
| #8 | IMROPE Vectorization | ✅ Yes | ✅ Reliable |
| #9 | Fused Flash Attention | ✅ Yes | ✅ Reliable |
| #10 | Aligned Stores | ✅ Yes | ✅ Reliable |
| #11 | Format Specialization | ✅ Yes | ✅ Reliable |
| #12 | FMA Consistency | ✅ Yes | ✅ Reliable |

**Action Required:** None! Just push your code and CI will test it.

---

### Tier 2: Conditional Testing

| Opt | Name | Test Strategy |
|-----|------|---------------|
| #1a | VNNI (AVX512_VNNI) | ⚠️ Use runtime CPU detection<br>Skip test if unavailable<br>Works ~80% of the time |

**Example in CI:**
```yaml
- name: Test AVX512_VNNI
  run: |
    if grep -q avx512_vnni /proc/cpuinfo; then
      ./test-vnni
    else
      echo "⚠️ Skipping (not available)"
    fi
```

---

### Tier 3: Requires Special Setup

| Opt | Name | Limitation | Solution |
|-----|------|------------|----------|
| #1b | VNNI (AVX_VNNI) | Not on GHA runners | **Option 1:** Intel SDE emulator (functional only)<br>**Option 2:** Self-hosted runner<br>**Option 3:** Manual testing |
| #7 | MoE Sparse GEMV | Performance not representative | Functional tests only on GHA<br>Performance tests on self-hosted |

**Intel SDE Example:**
```bash
# Download once (can cache)
wget https://downloadmirror.intel.com/823664/sde-external-9.33.0-2024-01-07-lin.tar.xz
tar xf sde-external-9.33.0-2024-01-07-lin.tar.xz

# Build with AVX_VNNI
cmake -B build -DCMAKE_C_FLAGS="-mavxvnni -mavx2"
cmake --build build

# Test under Alder Lake emulation (SLOW but validates correctness)
./sde-external*/sde64 -adl -- ./build/bin/test-vnni
```

---

## Performance Testing Strategy

### On GitHub Actions (Free):

**✅ What works well:**
- Functional correctness testing (all optimizations)
- Relative performance comparisons (before/after)
- Regression detection

**❌ What doesn't work:**
- Absolute performance numbers (different from target hardware)
- AVX_VNNI performance validation
- Consistent AVX-512 benchmarking (~20% failure rate)

### For Accurate Performance Testing:

**Option A: Self-Hosted Runners (Best)**
```yaml
jobs:
  perf-test:
    runs-on: [self-hosted, windows, avx-vnni]
    steps:
      - name: Benchmark on target hardware
        run: .\llama-bench.exe --model qwen3.gguf
```

**Option B: Manual Testing**
```bash
# On your Windows desktop/laptop
git checkout optimization-branch
cmake -B build -DCMAKE_C_FLAGS="-mavx2 -mavxvnni"
cmake --build build --config Release

# Benchmark
.\build\bin\llama-bench.exe --model models\qwen3-8b-q4_k.gguf -p 512 -n 128
```

**Option C: Third-Party CI** (Costs money but has modern CPUs)
- Cirrus CI - AMD Zen 4 runners
- Namespace - Custom hardware
- Blacksmith - Performance-optimized runners

---

## Practical Workflow Example

### Daily Development (Every PR):

1. **Push code** → GitHub Actions automatically:
   - ✅ Tests functional correctness (all 10 baseline optimizations)
   - ⚠️ Conditionally tests AVX-512 VNNI
   - 📊 Runs relative performance comparison
   - 📝 Posts results as PR comment

2. **Review results**:
   - All functional tests must pass
   - Performance should improve or stay neutral

### Weekly/Release Testing:

1. **Self-hosted runner** (or manual):
   - Test on actual Intel 12th/13th/14th gen
   - Test on AMD Zen 4/5
   - Validate AVX_VNNI performance gains
   - Full benchmark suite with real models

---

## Example Build Commands

### For GitHub Actions CI:
```bash
# AVX2 baseline (always works)
cmake -B build \
  -DCMAKE_C_FLAGS="-mavx2 -mfma -mf16c" \
  -DGGML_NATIVE=OFF

# AVX-512 VNNI (conditional)
cmake -B build \
  -DCMAKE_C_FLAGS="-mavx512f -mavx512bw -mavx512vnni" \
  -DGGML_NATIVE=OFF
```

### For Local Windows Development:
```cmd
# MSVC compiler
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_C_FLAGS="/arch:AVX2" ^
  -DCMAKE_CXX_FLAGS="/arch:AVX2"

# Or with Clang-CL (better intrinsics support)
cmake -B build -G "Visual Studio 17 2022" -A x64 -T ClangCL ^
  -DCMAKE_C_FLAGS="-mavx2 -mavxvnni" ^
  -DCMAKE_CXX_FLAGS="-mavx2 -mavxvnni"
```

### For Self-Hosted Runner (Full features):
```bash
# Enable all available features
cmake -B build \
  -DCMAKE_C_FLAGS="-mavx2 -mfma -mf16c -mavxvnni -mavx512f -mavx512vnni" \
  -DGGML_NATIVE=OFF  # Don't use -march=native for reproducibility
```

---

## Troubleshooting

### "AVX-512 tests failing inconsistently"
**Cause:** Runner allocated doesn't have AVX-512 (~20% of the time)
**Solution:** Use conditional testing (see Tier 2 strategy)

### "Want to test AVX_VNNI but don't have hardware"
**Cause:** Requires newer CPUs not in GitHub's pool
**Solution:**
1. Use Intel SDE for functional testing (workflow included)
2. Use self-hosted runner with 12th gen+ Intel
3. Test manually on your development machine

### "Performance numbers don't match local testing"
**Cause:** GitHub Actions runners use older Xeon CPUs
**Solution:** This is expected. Use GHA for:
- Functional correctness ✅
- Relative comparisons ✅

Use self-hosted/manual testing for:
- Absolute performance numbers
- AVX_VNNI validation

---

## Files Created for You

1. **`OPTIMIZATION_TESTING_MATRIX.md`** (this file)
   - Complete testing strategy
   - Hardware capabilities reference

2. **`.github/workflows/test-cpu-optimizations.yml`**
   - Ready-to-use GitHub Actions workflow
   - Multi-tier testing (baseline + conditional + emulated)
   - Performance comparison
   - CPU feature reporting

3. **`scripts/check-cpu-features.sh`**
   - Detect what YOUR hardware supports
   - Get recommended CMake flags
   - Understand what you can test

---

## Quick Decision Tree

```
Do you have the target hardware (Intel 12th+ or AMD Zen 4)?
├─ YES → Use self-hosted runners or test manually
│         Best performance validation
│         Can test ALL optimizations including AVX_VNNI
│
└─ NO → Use GitHub Actions (standard runners)
          ├─ Functional tests: 100% coverage (all 12 optimizations)
          │   • 10 optimizations: fully reliable
          │   • 1 optimization: conditional (AVX512_VNNI)
          │   • 1 optimization: emulated (AVX_VNNI via Intel SDE)
          │
          └─ Performance tests: Relative comparisons only
              • Good for regression detection
              • Not representative of target hardware
```

---

## Summary

✅ **You CAN automate:**
- Functional correctness testing for ALL 12 optimizations
- Performance regression detection for 10/12 optimizations
- AVX-512 VNNI testing (with conditional skip for ~20% of runs)

⚠️ **You SHOULD supplement with:**
- Self-hosted runners for accurate performance validation
- Manual testing on target hardware (Windows desktops/laptops)
- Intel SDE emulation for AVX_VNNI functional verification

❌ **You CANNOT (on standard GHA):**
- Get accurate absolute performance numbers for target CPUs
- Reliably test AVX-512 (works 80% of time)
- Test AVX_VNNI without emulation

**Recommendation:** Use the provided GitHub Actions workflow for continuous integration, and supplement with periodic testing on actual target hardware for performance validation.

---

## Next Steps

1. ✅ Review the workflow: `.github/workflows/test-cpu-optimizations.yml`
2. ✅ Run locally: `./scripts/check-cpu-features.sh`
3. ✅ Push a test commit to see the CI in action
4. ⚠️ Plan for self-hosted runner or manual testing for final validation
5. 📊 Document performance gains on actual Windows desktop/laptop CPUs

Good luck with your optimizations! 🚀
