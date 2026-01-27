#!/bin/bash
# CPU Feature Detection Script for llama.cpp Optimizations
# Usage: ./scripts/check-cpu-features.sh

set -e

echo "========================================"
echo "CPU Feature Detection for llama.cpp"
echo "========================================"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
else
    OS="Unknown"
fi

echo "Operating System: $OS"
echo ""

# Function to check a CPU flag
check_flag() {
    local flag=$1
    local name=$2
    local opt_number=$3

    if [[ "$OS" == "Linux" ]]; then
        if grep -q "$flag" /proc/cpuinfo 2>/dev/null; then
            echo "✅ $name" $(if [ -n "$opt_number" ]; then echo "- Enables optimization $opt_number"; fi)
            return 0
        else
            echo "❌ $name" $(if [ -n "$opt_number" ]; then echo "- Cannot test optimization $opt_number"; fi)
            return 1
        fi
    elif [[ "$OS" == "macOS" ]]; then
        if sysctl -a 2>/dev/null | grep -q "hw.optional.${flag}.*: 1"; then
            echo "✅ $name" $(if [ -n "$opt_number" ]; then echo "- Enables optimization $opt_number"; fi)
            return 0
        else
            echo "❌ $name" $(if [ -n "$opt_number" ]; then echo "- Cannot test optimization $opt_number"; fi)
            return 1
        fi
    else
        echo "⚠️  $name - Detection not supported on $OS"
        return 2
    fi
}

# Display CPU model
echo "=== CPU Information ==="
if [[ "$OS" == "Linux" ]]; then
    grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs echo "Model:"
    grep "cpu MHz" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs echo "Speed:"
    echo "Cores: $(nproc)"
elif [[ "$OS" == "macOS" ]]; then
    sysctl -n machdep.cpu.brand_string | xargs echo "Model:"
    echo "Cores: $(sysctl -n hw.ncpu)"
fi
echo ""

# Check instruction sets
echo "=== Instruction Set Support ==="
echo ""

echo "Baseline Features (Required for all optimizations):"
check_flag "sse" "SSE"
check_flag "sse2" "SSE2"
check_flag "avx" "AVX"
check_flag "avx2" "AVX2"
echo ""

echo "Guaranteed on GitHub Actions (Testable: #2-12):"
check_flag "fma" "FMA3" "#12"
check_flag "f16c" "F16C" "#4"
echo ""

echo "Conditionally Available on GitHub Actions (~80%):"
check_flag "avx512f" "AVX-512 Foundation"
check_flag "avx512bw" "AVX-512 Byte/Word"
check_flag "avx512vl" "AVX-512 Vector Length"
has_avx512_vnni=false
if check_flag "avx512_vnni" "AVX-512 VNNI" "#1a"; then
    has_avx512_vnni=true
fi
echo ""

echo "NOT Available on GitHub Actions (Requires self-hosted):"
has_avx_vnni=false
if check_flag "avx_vnni" "AVX_VNNI (Alder Lake+)" "#1b"; then
    has_avx_vnni=true
fi
check_flag "amx_tile" "AMX Tiles"
check_flag "amx_int8" "AMX INT8"
echo ""

# Summary
echo "========================================"
echo "OPTIMIZATION TESTING SUMMARY"
echo "========================================"
echo ""

# Count testable optimizations
testable_count=0
conditional_count=0
not_testable_count=0

# Always testable: #2, #3, #4, #5, #6, #8, #9, #10, #11, #12 = 10 optimizations
testable_count=10

echo "✅ Fully Testable on This Hardware:"
echo "   - #2: Software Prefetching"
echo "   - #3: Horizontal Sum Optimization"
echo "   - #4: F16C Conversion"
echo "   - #5: Loop Unrolling"
echo "   - #6: Decode Fast Path"
echo "   - #8: IMROPE Vectorization"
echo "   - #9: Fused Flash Attention"
echo "   - #10: Aligned Stores"
echo "   - #11: Format Specialization"
echo "   - #12: FMA Consistency"
echo ""

if [ "$has_avx512_vnni" = true ]; then
    echo "⚠️  Conditionally Testable:"
    echo "   - #1a: VNNI Expansion (AVX512_VNNI) ✅ Available"
    conditional_count=1
else
    echo "⚠️  Not Testable - Missing AVX512_VNNI:"
    echo "   - #1a: VNNI Expansion (AVX512_VNNI) ❌"
    not_testable_count=$((not_testable_count + 1))
fi
echo ""

if [ "$has_avx_vnni" = true ]; then
    echo "✅ Advanced Features Available:"
    echo "   - #1b: VNNI Expansion (AVX_VNNI) ✅ Available"
    testable_count=$((testable_count + 1))
else
    echo "❌ Not Testable - Requires Newer Hardware:"
    echo "   - #1b: VNNI Expansion (AVX_VNNI) - Requires Intel 12th gen+ or AMD Zen 4+"
    echo "   - Use Intel SDE emulator for functional testing"
    not_testable_count=$((not_testable_count + 1))
fi
echo ""

echo "📊 Coverage Summary:"
echo "   - Fully testable: $testable_count/12 optimizations"
echo "   - Conditionally testable: $conditional_count/12 optimizations"
echo "   - Requires emulation/self-hosted: $not_testable_count/12 optimizations"
echo ""

# GitHub Actions specific advice
if [[ -n "$GITHUB_ACTIONS" ]]; then
    echo "🤖 Running on GitHub Actions"
    echo ""
    if [ "$has_avx512_vnni" = true ]; then
        echo "✅ Great! You got an Intel runner with AVX512_VNNI support"
        echo "   This happens ~80% of the time on standard GitHub Actions runners"
    else
        echo "⚠️  You got a runner without AVX512_VNNI (AMD EPYC or older Intel)"
        echo "   This happens ~20% of the time - tests should gracefully skip"
    fi
    echo ""
fi

# Recommendations
echo "========================================"
echo "RECOMMENDATIONS"
echo "========================================"
echo ""

if [[ "$OS" == "Linux" ]] && [[ -n "$GITHUB_ACTIONS" ]]; then
    echo "For GitHub Actions CI:"
    echo "  1. Test optimizations #2-12 on every PR (guaranteed to work)"
    echo "  2. Make AVX512_VNNI tests conditional (check /proc/cpuinfo first)"
    echo "  3. Use Intel SDE for AVX_VNNI functional testing"
    echo "  4. Use self-hosted runners for accurate performance testing"
elif [[ "$OS" == "Linux" ]] || [[ "$OS" == "macOS" ]]; then
    echo "For local testing:"
    echo "  1. Build with appropriate flags for your CPU"
    echo "  2. Run benchmarks to validate optimizations"
    echo "  3. Compare before/after performance with llama-bench"
elif [[ "$OS" == "Windows" ]]; then
    echo "For Windows development:"
    echo "  1. Focus on Intel 11th-14th gen and AMD Zen 3/4/5 (most common)"
    echo "  2. Test with MSVC or Clang-CL compiler"
    echo "  3. Use /arch:AVX2 (MSVC) or -mavx2 (Clang) for baseline"
fi
echo ""

# CMake flags suggestion
echo "========================================"
echo "SUGGESTED CMAKE FLAGS"
echo "========================================"
echo ""

if [ "$has_avx_vnni" = true ]; then
    echo "For your hardware (AVX_VNNI available):"
    echo "  cmake -B build \\"
    echo "    -DCMAKE_C_FLAGS=\"-mavx2 -mfma -mf16c -mavxvnni\" \\"
    echo "    -DCMAKE_CXX_FLAGS=\"-mavx2 -mfma -mf16c -mavxvnni\" \\"
    echo "    -DGGML_NATIVE=OFF"
elif [ "$has_avx512_vnni" = true ]; then
    echo "For your hardware (AVX512_VNNI available):"
    echo "  cmake -B build \\"
    echo "    -DCMAKE_C_FLAGS=\"-mavx2 -mavx512f -mavx512bw -mavx512vnni\" \\"
    echo "    -DCMAKE_CXX_FLAGS=\"-mavx2 -mavx512f -mavx512bw -mavx512vnni\" \\"
    echo "    -DGGML_NATIVE=OFF"
else
    echo "For your hardware (AVX2 baseline):"
    echo "  cmake -B build \\"
    echo "    -DCMAKE_C_FLAGS=\"-mavx2 -mfma -mf16c\" \\"
    echo "    -DCMAKE_CXX_FLAGS=\"-mavx2 -mfma -mf16c\" \\"
    echo "    -DGGML_NATIVE=OFF"
fi
echo ""

echo "✅ CPU feature detection complete!"
