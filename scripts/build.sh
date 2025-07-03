#!/bin/bash

# CUDA Kernel Project Build Script
# This script builds all examples with proper error handling

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "CMakeLists.txt not found. Please run this script from the project root directory."
    exit 1
fi

# Parse command line arguments
BUILD_TYPE="Release"
CLEAN_BUILD=false
ENABLE_PROFILING=false
VERBOSE=false
JOBS=$(nproc)

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -p|--profile)
            ENABLE_PROFILING=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -d, --debug      Build in Debug mode (default: Release)"
            echo "  -c, --clean      Clean build directory first"
            echo "  -p, --profile    Enable profiling support"
            echo "  -v, --verbose    Verbose build output"
            echo "  -j, --jobs N     Use N parallel jobs (default: $(nproc))"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Building CUDA Kernel Examples"
print_status "Build type: $BUILD_TYPE"
print_status "Parallel jobs: $JOBS"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    print_error "CUDA compiler (nvcc) not found. Please install CUDA Toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
print_status "CUDA version: $CUDA_VERSION"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    print_error "CMake not found. Please install CMake 3.20 or later."
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
print_status "CMake version: $CMAKE_VERSION"

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    print_status "Cleaning build directory..."
    rm -rf build/
fi

# Create build directory
mkdir -p build
cd build

# Configure CMake
print_status "Configuring build..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"

if [ "$ENABLE_PROFILING" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_PROFILING=ON"
    print_status "Profiling enabled"
fi

if [ "$VERBOSE" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_VERBOSE_MAKEFILE=ON"
fi

# Run CMake configuration
if ! cmake $CMAKE_ARGS ..; then
    print_error "CMake configuration failed"
    exit 1
fi

# Build the project
print_status "Building project with $JOBS parallel jobs..."
if [ "$VERBOSE" = true ]; then
    make -j$JOBS
else
    make -j$JOBS > /dev/null
fi

if [ $? -eq 0 ]; then
    print_success "Build completed successfully!"
else
    print_error "Build failed"
    exit 1
fi

# List built executables
print_status "Built executables:"
for exe in bin/*; do
    if [ -x "$exe" ]; then
        echo "  - $(basename "$exe")"
    fi
done

print_success "All examples built successfully!"
print_status "To run examples:"
echo "  cd build"
echo "  ./bin/vector_addition"
echo "  ./bin/matrix_multiplication"
echo "  ./bin/parallel_reduction"
echo "  ./bin/convolution_2d"
echo "  ./bin/monte_carlo"

print_status "Build script completed"
