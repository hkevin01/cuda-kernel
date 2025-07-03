#!/bin/bash

# CUDA Profiling Script
# This script runs performance profiling on CUDA kernels

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[PROFILE]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the build directory
if [ ! -d "bin" ]; then
    print_error "bin directory not found. Please run from build directory or build the project first."
    exit 1
fi

# Create profile output directory
mkdir -p profiles

# Check for profiling tools
PROFILER=""
if command -v nsys &> /dev/null; then
    PROFILER="nsys"
    print_status "Using Nsight Systems for profiling"
elif command -v nvprof &> /dev/null; then
    PROFILER="nvprof"
    print_status "Using nvprof for profiling (legacy)"
else
    print_error "No CUDA profiler found. Please install Nsight Systems or nvprof."
    exit 1
fi

# Function to profile an executable
profile_executable() {
    local exe_name=$1
    local exe_path="bin/$exe_name"
    
    if [ ! -x "$exe_path" ]; then
        print_error "Executable $exe_path not found or not executable"
        return 1
    fi
    
    print_status "Profiling $exe_name..."
    
    if [ "$PROFILER" = "nsys" ]; then
        # Nsight Systems profiling
        nsys profile \
            --output=profiles/${exe_name}_profile \
            --force-overwrite=true \
            --trace=cuda,nvtx \
            --stats=true \
            ./$exe_path
    else
        # Legacy nvprof profiling
        nvprof \
            --output-profile profiles/${exe_name}_profile.nvvp \
            --log-file profiles/${exe_name}_profile.log \
            --print-gpu-trace \
            ./$exe_path
    fi
    
    if [ $? -eq 0 ]; then
        print_success "$exe_name profiling completed"
    else
        print_error "$exe_name profiling failed"
        return 1
    fi
}

# Function to run metrics analysis
analyze_metrics() {
    local exe_name=$1
    local exe_path="bin/$exe_name"
    
    if [ ! -x "$exe_path" ]; then
        return 1
    fi
    
    print_status "Analyzing metrics for $exe_name..."
    
    # Key metrics to collect
    local metrics=(
        "achieved_occupancy"
        "gld_efficiency"
        "gst_efficiency"
        "shared_load_transactions"
        "shared_store_transactions"
        "dram_read_throughput"
        "dram_write_throughput"
        "flop_count_sp"
        "flop_count_sp_add"
        "flop_count_sp_mul"
        "flop_count_sp_fma"
    )
    
    if [ "$PROFILER" = "nvprof" ]; then
        for metric in "${metrics[@]}"; do
            echo "=== $metric ===" >> profiles/${exe_name}_metrics.txt
            nvprof --metrics $metric ./$exe_path 2>> profiles/${exe_name}_metrics.txt
            echo "" >> profiles/${exe_name}_metrics.txt
        done
    fi
}

# Parse command line arguments
EXECUTABLES=()
RUN_METRICS=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--metrics)
            RUN_METRICS=true
            shift
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -a|--all)
            EXECUTABLES=("vector_addition" "matrix_multiplication" "parallel_reduction" "convolution_2d" "monte_carlo")
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [EXECUTABLES...]"
            echo "Options:"
            echo "  -m, --metrics    Run detailed metrics analysis (nvprof only)"
            echo "  -q, --quick      Quick profiling mode"
            echo "  -a, --all        Profile all executables"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -a                              # Profile all executables"
            echo "  $0 vector_addition                 # Profile specific executable"
            echo "  $0 -m matrix_multiplication        # Profile with metrics"
            exit 0
            ;;
        *)
            EXECUTABLES+=("$1")
            shift
            ;;
    esac
done

# If no executables specified, show available ones
if [ ${#EXECUTABLES[@]} -eq 0 ]; then
    print_status "Available executables:"
    for exe in bin/*; do
        if [ -x "$exe" ]; then
            echo "  - $(basename "$exe")"
        fi
    done
    echo ""
    echo "Use -a to profile all, or specify executable names"
    echo "Use -h for help"
    exit 0
fi

print_status "Starting profiling session..."
print_status "Profiler: $PROFILER"
print_status "Executables: ${EXECUTABLES[*]}"

# Profile each executable
for exe in "${EXECUTABLES[@]}"; do
    profile_executable "$exe"
    
    if [ "$RUN_METRICS" = true ] && [ "$PROFILER" = "nvprof" ]; then
        analyze_metrics "$exe"
    fi
    
    echo ""
done

print_success "Profiling session completed!"
print_status "Profile data saved in: profiles/"

if [ "$PROFILER" = "nsys" ]; then
    print_status "View profiles with:"
    echo "  nsys-ui profiles/*.nsys-rep"
else
    print_status "View profiles with:"
    echo "  nvvp profiles/*.nvvp"
    if [ "$RUN_METRICS" = true ]; then
        echo "  cat profiles/*_metrics.txt"
    fi
fi
