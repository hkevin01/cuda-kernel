#!/bin/bash

# Unified Build Script for GPU Kernel Project
# Builds all kernels and GUI in a single, consistent build system

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

echo "=== Unified GPU Kernel Project Build ==="

# Configuration
BUILD_DIR="build_unified"
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
            echo "  -c, --clean     Clean build directory before building"
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

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_status "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
print_status "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Configure build
print_status "Configuring build with CMake..."
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CXX_FLAGS="-O3 -std=c++14" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ $? -ne 0 ]; then
    print_failure "CMake configuration failed"
    exit 1
fi

print_success "CMake configuration completed"

# Build project
print_status "Building project with $PARALLEL_JOBS parallel jobs..."
make -j"$PARALLEL_JOBS"

if [ $? -ne 0 ]; then
    print_failure "Build failed"
    exit 1
fi

print_success "Build completed successfully"

# Verify build results
print_status "Verifying build results..."

cd ..

# Check for kernel executables
KERNEL_COUNT=0
EXPECTED_KERNELS=10

for kernel in vector_addition matrix_multiplication parallel_reduction convolution_2d monte_carlo advanced_fft advanced_threading dynamic_memory warp_primitives nbody_simulation; do
    if [ -x "$BUILD_DIR/bin/$kernel" ]; then
        print_success "✓ $kernel"
        KERNEL_COUNT=$((KERNEL_COUNT + 1))
    else
        print_failure "✗ $kernel (missing)"
    fi
done

# Check for GUI executable
if [ -x "$BUILD_DIR/bin/gpu_kernel_gui" ]; then
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
echo "Build directory: $BUILD_DIR"
echo "Build type: $BUILD_TYPE"

if [ $KERNEL_COUNT -eq $EXPECTED_KERNELS ] && [ "$GUI_BUILT" = true ]; then
    print_success "All components built successfully!"
    echo ""
    echo "To run the GUI:"
    echo "  ./$BUILD_DIR/bin/gpu_kernel_gui"
    echo ""
    echo "To test individual kernels:"
    echo "  ./$BUILD_DIR/bin/vector_addition 1000"
    echo "  ./$BUILD_DIR/bin/matrix_multiplication 256"
    echo "  # ... etc"
    echo ""
    echo "To run comprehensive tests:"
    echo "  ./scripts/testing/comprehensive_gui_test.sh"
    exit 0
else
    print_failure "Some components failed to build"
    exit 1
fi 