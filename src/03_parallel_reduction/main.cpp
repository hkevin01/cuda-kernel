#include "cuda_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>

// Simple reduction kernel - will be implemented later
__global__ void reductionKernel(const float* input, float* output, size_t n) {
    // TODO: Implement reduction kernel
    extern __shared__ float sdata[];
    
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Simple reduction in shared memory
    for (size_t s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

float reductionCPU(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

int main() {
    std::cout << "=== CUDA Parallel Reduction Example ===" << std::endl;
    std::cout << "Note: This is a basic implementation. Full version coming soon!" << std::endl;
    
    setOptimalDevice();
    printDeviceInfo();
    
    const size_t N = 1024 * 1024;
    
    // Allocate and initialize data
    float *h_input, *h_output;
    allocateHostMemory(&h_input, N);
    allocateHostMemory(&h_output, 1);
    
    generateRandomFloats(h_input, N, 1.0f, 2.0f);
    
    // CPU reduction
    CPUTimer cpu_timer;
    cpu_timer.start();
    float cpu_result = reductionCPU(h_input, N);
    double cpu_time = cpu_timer.stop();
    
    // GPU reduction (basic implementation)
    float *d_input, *d_output;
    allocateDeviceMemory(&d_input, N);
    allocateDeviceMemory(&d_output, 1);
    
    copyHostToDevice(d_input, h_input, N);
    
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    CudaTimer gpu_timer;
    gpu_timer.start();
    
    // Note: This is a simplified version - needs multiple kernel launches for large data
    reductionKernel<<<1, block_size, block_size * sizeof(float)>>>(d_input, d_output, std::min(N, (size_t)block_size));
    CUDA_CHECK_KERNEL();
    
    float gpu_time = gpu_timer.stop();
    
    copyDeviceToHost(h_output, d_output, 1);
    
    std::cout << "CPU Result: " << cpu_result << " (time: " << cpu_time << " ms)" << std::endl;
    std::cout << "GPU Result: " << h_output[0] << " (time: " << gpu_time << " ms)" << std::endl;
    std::cout << "Note: GPU result is partial - full implementation needed" << std::endl;
    
    // Cleanup
    freeHostMemory(h_input);
    freeHostMemory(h_output);
    freeDeviceMemory(d_input);
    freeDeviceMemory(d_output);
    
    std::cout << "=== Parallel Reduction Complete ===" << std::endl;
    return 0;
}
