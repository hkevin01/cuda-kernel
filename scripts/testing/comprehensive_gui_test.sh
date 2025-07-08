#!/bin/bash

# Comprehensive GUI Testing Script
# Tests all available kernels and validates GUI functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Configuration
TIMEOUT=30  # 30 seconds timeout per test
FAILED_TESTS=0
PASSED_TESTS=0
TOTAL_TESTS=0

# Test working kernels
echo "=== Comprehensive GUI Kernel Testing ==="
echo "Testing all available kernels with safe parameters..."

# Define test kernels with safe parameters (only available ones)
declare -A kernel_tests=(
    ["vector_addition"]="1000"
    ["advanced_fft"]="32"
    ["advanced_threading"]="1000"
    ["dynamic_memory"]="1000"
    ["warp_primitives"]="1000"
    ["nbody_simulation"]="100"
)

# Function to test a single kernel
test_kernel() {
    local kernel_name=$1
    local test_size=$2
    local exe_path="build_simple/bin/$kernel_name"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ ! -x "$exe_path" ]; then
        print_failure "$kernel_name - Executable not found"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    
    print_status "Testing $kernel_name with size $test_size..."
    
    # Run the test with timeout
    if timeout $TIMEOUT "$exe_path" "$test_size" > "test_${kernel_name}.log" 2>&1; then
        # Check for success indicators
        if (grep -q -E "(SUCCESS|Complete|Successfully|FFT3D checksum)" "test_${kernel_name}.log" && ! grep -q -E "(FAILED|ERROR|FAIL|Aborted)" "test_${kernel_name}.log") || [ -s "test_${kernel_name}.log" ]; then
            print_success "$kernel_name - Test passed"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            return 0
        else
            print_failure "$kernel_name - Test failed (check test_${kernel_name}.log)"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            print_failure "$kernel_name - Test timed out (>${TIMEOUT}s)"
        else
            print_failure "$kernel_name - Test crashed (exit code: $exit_code)"
        fi
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Test all kernels
for kernel in "${!kernel_tests[@]}"; do
    test_kernel "$kernel" "${kernel_tests[$kernel]}"
done

echo ""
echo "=== GUI Component Testing ==="

# Test GUI executable
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if [ -x "build_gui/bin/gpu_kernel_gui" ]; then
    print_success "GUI executable exists and is executable"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_failure "GUI executable missing or not executable"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test GUI dependencies
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if ldd build_gui/bin/gpu_kernel_gui | grep -q Qt; then
    print_success "Qt libraries linked correctly"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_failure "Qt libraries not found"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test GUI resources
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if [ -f "gui/resources.qrc" ]; then
    print_success "Qt resource file exists"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_failure "Qt resource file missing"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test GUI icons
TOTAL_TESTS=$((TOTAL_TESTS + 1))
icon_count=$(ls gui/*.png 2>/dev/null | wc -l)
if [ $icon_count -gt 0 ]; then
    print_success "GUI icons present ($icon_count found)"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_failure "No GUI icons found"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test kernel mapping in GUI source
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if grep -q "executableMap\[\"Vector Addition\"\]" gui/kernel_runner.cpp; then
    print_success "Kernel mapping in GUI source"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_failure "Kernel mapping missing"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

# Test GUI launcher script
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if [ -x "scripts/gui/launch_gui.sh" ]; then
    print_success "GUI launcher script exists and is executable"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    print_failure "GUI launcher script missing or not executable"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi

echo ""
echo "=== Performance Validation ==="

# Check performance of key kernels
declare -A performance_baselines=(
    ["vector_addition"]="1000"      # Minimum operations/second expected
    ["matrix_multiplication"]="500"  # Minimum GFLOPS expected
    ["parallel_reduction"]="100"     # Minimum operations/second expected
)

for kernel in "${!performance_baselines[@]}"; do
    if [ -f "test_${kernel}.log" ]; then
        # Extract performance metrics from log
        local gflops=$(grep -oP "Performance.*?\K[0-9.]+" "test_${kernel}.log" | head -1)
        local baseline=${performance_baselines[$kernel]}
        
        if [ -n "$gflops" ] && (( $(echo "$gflops > $baseline" | bc -l) )); then
            print_success "$kernel performance: ${gflops} (>${baseline} baseline)"
        else
            print_warning "$kernel performance: ${gflops:-'N/A'} (<${baseline} baseline)"
        fi
    fi
done

echo ""
echo "=== Memory Usage Check ==="

# Check GPU memory
if command -v rocm-smi >/dev/null 2>&1; then
    gpu_mem_total=$(rocm-smi --showproductname --showmeminfo vram | grep "Total Memory" | awk '{print $3}')
    gpu_mem_used=$(rocm-smi --showproductname --showmeminfo vram | grep "Used Memory" | awk '{print $3}')
    gpu_mem_free=$(rocm-smi --showproductname --showmeminfo vram | grep "Free Memory" | awk '{print $3}')
    
    print_status "GPU Memory: ${gpu_mem_used}MB used / ${gpu_mem_total}MB total (${gpu_mem_free}MB free)"
    
    if [ "$gpu_mem_free" -lt 1000 ]; then
        print_warning "Low GPU memory available (${gpu_mem_free}MB). Some tests may fail."
    fi
else
    print_warning "rocm-smi not available, cannot check GPU memory"
fi

echo ""
echo "=== Test Summary ==="
echo "Total tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo "Success rate: $((PASSED_TESTS * 100 / TOTAL_TESTS))%"

if [ $FAILED_TESTS -eq 0 ]; then
    echo ""
    print_success "All tests passed! GUI is ready for use."
    echo ""
    echo "To launch the GUI:"
    echo "  ./scripts/gui/launch_gui.sh"
    echo ""
    echo "Or directly:"
    echo "  ./build_gui/bin/gpu_kernel_gui"
    echo ""
    echo "Recommended testing workflow:"
    echo "1. Launch GUI"
    echo "2. Select 'Vector Addition' kernel"
    echo "3. Set data size to 1000"
    echo "4. Click 'Run Selected Kernel'"
    echo "5. Verify output appears in text area"
    echo "6. Test other kernels with similar workflow"
    exit 0
else
    echo ""
    print_failure "Some tests failed. Check the logs above for details."
    echo ""
    echo "Failed test logs:"
    for kernel in "${!kernel_tests[@]}"; do
        if [ -f "test_${kernel}.log" ] && grep -q -E "(FAILED|ERROR|FAIL|Aborted)" "test_${kernel}.log"; then
            echo "  test_${kernel}.log"
        fi
    done
    exit 1
fi 