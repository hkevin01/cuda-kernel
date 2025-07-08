#!/bin/bash

# Hybrid Build Script for GPU Kernel Project
# Uses existing working build systems but provides unified interface

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[BUILD]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_failure() {
    echo -e "${RED}[FAILURE]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo "=== Hybrid GPU Kernel Project Build ==="

# Configuration
BUILD_TYPE="Release"
PARALLEL_JOBS=$(nproc)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --debug     Build in debug mode"
            echo "  -j, --jobs N    Use N parallel jobs (default: auto-detect)"
            echo "  -c, --clean     Clean build directories before building"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            print_failure "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check prerequisites
print_status "Checking prerequisites..."

# Check for HIP
if ! command -v hipcc >/dev/null 2>&1; then
    print_failure "HIP compiler (hipcc) not found. Please install ROCm."
    exit 1
fi

# Check for Qt6
if ! pkg-config --exists Qt6Core Qt6Widgets 2>/dev/null; then
    print_failure "Qt6 not found. Please install Qt6 development packages."
    exit 1
fi

# Check for CMake
if ! command -v cmake >/dev/null 2>&1; then
    print_failure "CMake not found. Please install CMake."
    exit 1
fi

print_success "All prerequisites found"

# Clean build directories if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_status "Cleaning build directories..."
    rm -rf build_gui build_simple
fi

# Step 1: Build GUI using existing system
print_status "Step 1: Building GUI..."
mkdir -p build_gui
cd build_gui

cmake ../gui \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CXX_FLAGS="-O3 -std=c++14"

if [ $? -ne 0 ]; then
    print_failure "GUI CMake configuration failed"
    exit 1
fi

make -j"$PARALLEL_JOBS"

if [ $? -ne 0 ]; then
    print_failure "GUI build failed"
    exit 1
fi

print_success "GUI built successfully"
cd ..

# Step 2: Build kernels using existing system
print_status "Step 2: Building kernels..."

# Create build directory for kernels
mkdir -p build_simple/bin

# Build vector addition
print_status "Building vector_addition..."
cd src/01_vector_addition
hipcc -O3 -std=c++14 -I../common -o ../../build_simple/bin/vector_addition main_hip.cpp vector_addition_hip.hip ../common/hip_utils.cpp ../common/timer.cpp ../common/helper_functions.cpp
cd ../..

# Build advanced threading (safe version)
print_status "Building advanced_threading..."
cd src/07_advanced_threading
hipcc -O3 -std=c++14 -I../common -o ../../build_simple/bin/advanced_threading main_hip_safe.cpp advanced_threading_hip_safe.hip ../common/hip_utils.cpp ../common/timer.cpp ../common/helper_functions.cpp
cd ../..

# Build matrix multiplication
print_status "Building matrix_multiplication..."
cd src/02_matrix_multiplication
hipcc -O3 -std=c++14 -I../common -o ../../build_simple/bin/matrix_multiplication main_hip.cpp matrix_mul_hip.hip ../common/hip_utils.cpp ../common/timer.cpp ../common/helper_functions.cpp
cd ../..

# Build parallel reduction
print_status "Building parallel_reduction..."
cd src/03_parallel_reduction
hipcc -O3 -std=c++14 -I../common -o ../../build_simple/bin/parallel_reduction main_hip.cpp reduction_hip.hip ../common/hip_utils.cpp ../common/timer.cpp ../common/helper_functions.cpp
cd ../..

# Build convolution 2D
print_status "Building convolution_2d..."
cd src/04_convolution_2d
hipcc -O3 -std=c++14 -I../common -o ../../build_simple/bin/convolution_2d main_hip.cpp convolution_hip.hip ../common/hip_utils.cpp ../common/timer.cpp ../common/helper_functions.cpp
cd ../..

# Build monte carlo
print_status "Building monte_carlo..."
cd src/05_monte_carlo
hipcc -O3 -std=c++14 -I../common -o ../../build_simple/bin/monte_carlo main_hip.cpp monte_carlo_hip.hip ../common/hip_utils.cpp ../common/timer.cpp ../common/helper_functions.cpp
cd ../..

# Build dynamic memory
print_status "Building dynamic_memory..."
cd src/08_dynamic_memory
hipcc -O3 -std=c++14 -I../common -o ../../build_simple/bin/dynamic_memory main_hip.cpp dynamic_memory_hip.hip ../common/hip_utils.cpp ../common/timer.cpp ../common/helper_functions.cpp
cd ../..

# Build warp primitives (simplified version)
print_status "Building warp_primitives..."
cd src/09_warp_primitives
hipcc -O3 -std=c++14 -I../common -o ../../build_simple/bin/warp_primitives main_hip.cpp warp_primitives_hip.hip ../common/hip_utils.cpp ../common/timer.cpp ../common/helper_functions.cpp
cd ../..

print_success "All kernels built successfully"

# Step 3: Verify build results
print_status "Step 3: Verifying build results..."

# Check for kernel executables
KERNEL_COUNT=0
EXPECTED_KERNELS=8

for kernel in vector_addition matrix_multiplication parallel_reduction convolution_2d monte_carlo advanced_threading dynamic_memory warp_primitives; do
    if [ -x "build_simple/bin/$kernel" ]; then
        print_success "✓ $kernel"
        KERNEL_COUNT=$((KERNEL_COUNT + 1))
    else
        print_failure "✗ $kernel (missing)"
    fi
done

# Check for GUI executable
if [ -x "build_gui/bin/gpu_kernel_gui" ]; then
    print_success "✓ gpu_kernel_gui"
    GUI_BUILT=true
else
    print_failure "✗ gpu_kernel_gui (missing)"
    GUI_BUILT=false
fi

echo ""
echo "=== Build Summary ==="
echo "Kernels built: $KERNEL_COUNT/$EXPECTED_KERNELS"
echo "GUI built: $GUI_BUILT"
echo "Build type: $BUILD_TYPE"
echo ""
echo "Build directories:"
echo "  GUI: build_gui/bin/"
echo "  Kernels: build_simple/bin/"

if [ $KERNEL_COUNT -eq $EXPECTED_KERNELS ] && [ "$GUI_BUILT" = true ]; then
    print_success "All components built successfully!"
    echo ""
    echo "To run the GUI:"
    echo "  ./build_gui/bin/gpu_kernel_gui"
    echo ""
    echo "To test individual kernels:"
    echo "  ./build_simple/bin/vector_addition 1000"
    echo "  ./build_simple/bin/matrix_multiplication 256"
    echo "  # ... etc"
    echo ""
    echo "To run comprehensive tests:"
    echo "  ./scripts/testing/comprehensive_gui_test.sh"
    exit 0
else
    print_failure "Some components failed to build"
    exit 1
fi 