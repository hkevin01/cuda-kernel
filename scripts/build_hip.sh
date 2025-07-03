#!/bin/bash

# HIP Build Script for AMD GPUs
# Builds all examples using hipcc directly

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building GPU Kernel Examples with HIP/ROCm${NC}"
echo "=================================================="

# Check if hipcc is available
if ! command -v hipcc &> /dev/null; then
    echo -e "${RED}Error: hipcc not found. Please install ROCm.${NC}"
    echo "See docs/installation/rocm-setup.md for installation instructions."
    exit 1
fi

echo -e "${GREEN}✓ HIP compiler found${NC}"
hipcc --version | head -1

# Get project root directory
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="$PROJECT_ROOT/build_hip"

# Create build directory
mkdir -p "$BUILD_DIR"

# Common compiler flags
HIP_FLAGS="-O3 -std=c++14 -I$PROJECT_ROOT/src/common"

echo -e "\n${YELLOW}Building examples...${NC}"

# 1. Vector Addition
echo -e "${GREEN}Building Vector Addition...${NC}"
cd "$PROJECT_ROOT/src/01_vector_addition"

if [ -f "main_hip.cpp" ]; then
    # Build with hipcc - compile all sources together to avoid linking issues
    hipcc $HIP_FLAGS \
          -o "$BUILD_DIR/vector_addition_hip" \
          main_hip.cpp \
          "$PROJECT_ROOT/src/common/timer.cpp" \
          "$PROJECT_ROOT/src/common/helper_functions.cpp" \
          "$PROJECT_ROOT/src/common/hip_utils.cpp"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Vector Addition built successfully${NC}"
    else
        echo -e "${RED}✗ Vector Addition build failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ main_hip.cpp not found, skipping Vector Addition${NC}"
fi

cd "$PROJECT_ROOT"

# Test if the build worked
if [ -f "$BUILD_DIR/vector_addition_hip" ]; then
    echo -e "\n${GREEN}Build completed successfully!${NC}"
    echo ""
    echo "Available executables:"
    ls -la "$BUILD_DIR"/*_hip 2>/dev/null || echo "No executables found"
    echo ""
    echo "To run the vector addition example:"
    echo "  ./$BUILD_DIR/vector_addition_hip"
    echo ""
    echo "To run with custom vector size (e.g., 1M elements):"
    echo "  ./$BUILD_DIR/vector_addition_hip 1048576"
else
    echo -e "\n${RED}Build failed!${NC}"
    exit 1
fi
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Host wrapper function
void matrix_multiply_hip(const float* A, const float* B, float* C, int width) {
    float *d_A, *d_B, *d_C;
    size_t size = width * width * sizeof(float);
    
    // Allocate device memory
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);
    
    // Copy data to device
    hipMemcpy(d_A, A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, size, hipMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (width + block.y - 1) / block.y);
    
    matrixMulHIP<<<grid, block>>>(d_A, d_B, d_C, width);
    
    // Copy result back
    hipMemcpy(C, d_C, size, hipMemcpyDeviceToHost);
    
    // Cleanup
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}
EOF

# Create a simple main file for matrix multiplication
cat > src/02_matrix_multiplication/main_hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

extern void matrix_multiply_hip(const float* A, const float* B, float* C, int width);

int main() {
    const int N = 512;
    std::cout << "HIP Matrix Multiplication: " << N << "x" << N << std::endl;
    
    // Allocate host memory
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);
    
    // Time the multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiply_hip(A.data(), B.data(), C.data(), N);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    
    // Verify result (should be N * 1.0 * 2.0 = 2*N)
    std::cout << "Sample result: " << C[0] << " (expected: " << 2.0f * N << ")" << std::endl;
    
    return 0;
}
EOF

cd src/02_matrix_multiplication
hipcc $HIP_FLAGS matrix_mul_hip.hip main_hip.cpp -o ../../$BUILD_DIR/matrix_multiplication_hip
cd ../..

echo ""
echo -e "${GREEN}Build completed!${NC}"
echo ""
echo "Built executables in $BUILD_DIR/:"
ls -la $BUILD_DIR/

echo ""
echo "To test the examples:"
echo "  cd $BUILD_DIR"
echo "  ./vector_addition_hip"
echo "  ./matrix_multiplication_hip"

echo ""
echo -e "${GREEN}✓ All examples built successfully with HIP/ROCm!${NC}"
