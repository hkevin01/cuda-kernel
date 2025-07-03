#!/bin/bash

# CUDA Test Script
# This script runs all examples and validates their output

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

# Check if we're in the build directory
if [ ! -d "bin" ]; then
    print_status "Not in build directory. Attempting to build..."
    if [ -f "../scripts/build.sh" ]; then
        ../scripts/build.sh
        cd build
    else
        print_failure "Could not find build script or build directory"
        exit 1
    fi
fi

# Test configuration
TIMEOUT=300  # 5 minutes timeout per test
FAILED_TESTS=0
PASSED_TESTS=0
TOTAL_TESTS=0

# Function to run a single test
run_test() {
    local exe_name=$1
    local exe_path="bin/$exe_name"
    local test_log="test_${exe_name}.log"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ ! -x "$exe_path" ]; then
        print_failure "$exe_name - Executable not found"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
    
    print_status "Running $exe_name..."
    
    # Run the test with timeout
    if timeout $TIMEOUT ./$exe_path > "$test_log" 2>&1; then
        # Check for common success indicators
        if grep -q -E "(PASSED|SUCCESS|Complete)" "$test_log" && ! grep -q -E "(FAILED|ERROR|FAIL)" "$test_log"; then
            print_success "$exe_name - Test passed"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            return 0
        else
            print_failure "$exe_name - Test failed (check $test_log)"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            print_failure "$exe_name - Test timed out (>${TIMEOUT}s)"
        else
            print_failure "$exe_name - Test crashed (exit code: $exit_code)"
        fi
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Function to check CUDA environment
check_cuda_env() {
    print_status "Checking CUDA environment..."
    
    # Check CUDA driver
    if ! nvidia-smi > /dev/null 2>&1; then
        print_failure "NVIDIA driver not available"
        return 1
    fi
    
    # Check CUDA runtime
    if ! nvcc --version > /dev/null 2>&1; then
        print_failure "CUDA compiler not available"
        return 1
    fi
    
    # Check for CUDA-capable devices
    local gpu_count=$(nvidia-smi -L | wc -l)
    if [ $gpu_count -eq 0 ]; then
        print_failure "No CUDA-capable devices found"
        return 1
    fi
    
    print_success "CUDA environment OK ($gpu_count GPU(s) detected)"
    return 0
}

# Function to run performance regression tests
run_performance_tests() {
    print_status "Running performance regression tests..."
    
    # These are baseline performance expectations (adjust based on your hardware)
    declare -A performance_baselines=(
        ["vector_addition"]="1000"      # Minimum GFLOPS expected
        ["matrix_multiplication"]="500"  # Minimum GFLOPS expected
        ["parallel_reduction"]="100"     # Minimum GFLOPS expected
    )
    
    for exe in "${!performance_baselines[@]}"; do
        if [ -x "bin/$exe" ]; then
            local log_file="test_${exe}.log"
            if [ -f "$log_file" ]; then
                # Extract performance metrics from log
                local gflops=$(grep -oP "Performance.*?\K[0-9.]+" "$log_file" | head -1)
                local baseline=${performance_baselines[$exe]}
                
                if [ -n "$gflops" ] && (( $(echo "$gflops > $baseline" | bc -l) )); then
                    print_success "$exe performance: ${gflops} GFLOPS (>${baseline} baseline)"
                else
                    print_warning "$exe performance: ${gflops:-'N/A'} GFLOPS (<${baseline} baseline)"
                fi
            fi
        fi
    done
}

# Function to check memory usage
check_memory_usage() {
    print_status "Checking memory usage..."
    
    # Get GPU memory info
    local gpu_mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    local gpu_mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local gpu_mem_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    
    print_status "GPU Memory: ${gpu_mem_used}MB used / ${gpu_mem_total}MB total (${gpu_mem_free}MB free)"
    
    # Check if we have enough memory for tests
    if [ $gpu_mem_free -lt 1000 ]; then
        print_warning "Low GPU memory available (${gpu_mem_free}MB). Some tests may fail."
    fi
}

# Parse command line arguments
QUICK_MODE=false
PERFORMANCE_TEST=false
EXECUTABLES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -p|--performance)
            PERFORMANCE_TEST=true
            shift
            ;;
        -a|--all)
            EXECUTABLES=("vector_addition" "matrix_multiplication" "parallel_reduction" "convolution_2d" "monte_carlo")
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [EXECUTABLES...]"
            echo "Options:"
            echo "  -q, --quick        Quick test mode (shorter runs)"
            echo "  -p, --performance  Include performance regression tests"
            echo "  -a, --all          Test all executables"
            echo "  -h, --help         Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -a                    # Test all executables"
            echo "  $0 vector_addition       # Test specific executable"
            echo "  $0 -p -a                 # Test all with performance checks"
            exit 0
            ;;
        *)
            EXECUTABLES+=("$1")
            shift
            ;;
    esac
done

# If no executables specified, test all available ones
if [ ${#EXECUTABLES[@]} -eq 0 ]; then
    EXECUTABLES=()
    for exe in bin/*; do
        if [ -x "$exe" ]; then
            EXECUTABLES+=($(basename "$exe"))
        fi
    done
fi

print_status "Starting test suite..."
print_status "Test timeout: ${TIMEOUT}s per test"
print_status "Executables to test: ${EXECUTABLES[*]}"

# Environment checks
if ! check_cuda_env; then
    print_failure "CUDA environment check failed"
    exit 1
fi

check_memory_usage

# Run tests
print_status "Running functional tests..."
for exe in "${EXECUTABLES[@]}"; do
    run_test "$exe"
done

# Performance tests
if [ "$PERFORMANCE_TEST" = true ]; then
    run_performance_tests
fi

# Summary
echo ""
print_status "=== Test Summary ==="
print_status "Total tests: $TOTAL_TESTS"
print_success "Passed: $PASSED_TESTS"
print_failure "Failed: $FAILED_TESTS"

if [ $FAILED_TESTS -eq 0 ]; then
    print_success "All tests passed!"
    exit 0
else
    print_failure "$FAILED_TESTS test(s) failed"
    print_status "Check individual log files for details:"
    ls -1 test_*.log 2>/dev/null || true
    exit 1
fi
