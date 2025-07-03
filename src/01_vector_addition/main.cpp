#include "cuda_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>

// CPU implementation for comparison
void vectorAddCPU(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Basic CUDA kernel for vector addition
__global__ void vectorAddKernel(const float* a, const float* b, float* c, size_t n) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Optimized CUDA kernel with grid-stride loop
__global__ void vectorAddKernelOptimized(const float* a, const float* b, float* c, size_t n) {
    // Calculate global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better performance with large arrays
    for (size_t i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    std::cout << "=== CUDA Vector Addition Example ===" << std::endl;
    
    // Initialize CUDA
    setOptimalDevice();
    printDeviceInfo();
    
    // Problem size
    const size_t N = 32 * 1024 * 1024; // 32M elements
    const size_t bytes = N * sizeof(float);
    
    std::cout << "Vector size: " << formatNumber(N) << " elements" << std::endl;
    std::cout << "Memory per vector: " << formatBytes(bytes) << std::endl;
    std::cout << "Total memory required: " << formatBytes(3 * bytes) << std::endl;
    
    // Allocate host memory
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    allocateHostMemory(&h_a, N);
    allocateHostMemory(&h_b, N);
    allocateHostMemory(&h_c_cpu, N);
    allocateHostMemory(&h_c_gpu, N);
    
    // Initialize input data
    std::cout << "Initializing input data..." << std::endl;
    generateRandomFloats(h_a, N, 1.0f, 10.0f);
    generateRandomFloats(h_b, N, 1.0f, 10.0f);
    
    // Print sample data
    printArray(h_a, N, "Vector A (sample)", 5);
    printArray(h_b, N, "Vector B (sample)", 5);
    
    // CPU implementation
    std::cout << "\nRunning CPU implementation..." << std::endl;
    CPUTimer cpu_timer;
    cpu_timer.start();
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    double cpu_time = cpu_timer.stop();
    
    printArray(h_c_cpu, N, "CPU Result (sample)", 5);
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    allocateDeviceMemory(&d_a, N);
    allocateDeviceMemory(&d_b, N);
    allocateDeviceMemory(&d_c, N);
    
    // Copy data to device
    copyHostToDevice(d_a, h_a, N);
    copyHostToDevice(d_b, h_b, N);
    
    // GPU implementation - Basic kernel
    std::cout << "\nRunning GPU implementation (basic kernel)..." << std::endl;
    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;
    
    CudaTimer gpu_timer;
    gpu_timer.start();
    
    vectorAddKernel<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_KERNEL();
    
    float gpu_time_basic = gpu_timer.stop();
    
    // Copy result back to host
    copyDeviceToHost(h_c_gpu, d_c, N);
    printArray(h_c_gpu, N, "GPU Result Basic (sample)", 5);
    
    // Verify results
    bool basic_correct = verifyResults(h_c_cpu, h_c_gpu, N, 1e-5f);
    
    // GPU implementation - Optimized kernel
    std::cout << "\nRunning GPU implementation (optimized kernel)..." << std::endl;
    DeviceInfo device_info = getDeviceInfo();
    const int optimal_grid_size = std::min((int)((N + block_size - 1) / block_size), 
                                          device_info.multiprocessor_count * 2);
    
    gpu_timer.start();
    
    vectorAddKernelOptimized<<<optimal_grid_size, block_size>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_KERNEL();
    
    float gpu_time_optimized = gpu_timer.stop();
    
    // Copy result back to host
    copyDeviceToHost(h_c_gpu, d_c, N);
    printArray(h_c_gpu, N, "GPU Result Optimized (sample)", 5);
    
    // Verify results
    bool optimized_correct = verifyResults(h_c_cpu, h_c_gpu, N, 1e-5f);
    
    // Performance analysis
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    
    // Calculate metrics
    const size_t total_bytes = 3 * bytes; // Read A, Read B, Write C
    const size_t operations = N; // N additions
    
    double cpu_bandwidth = calculateBandwidth(total_bytes, cpu_time);
    double gpu_bandwidth_basic = calculateBandwidth(total_bytes, gpu_time_basic);
    double gpu_bandwidth_optimized = calculateBandwidth(total_bytes, gpu_time_optimized);
    
    double cpu_gflops = calculateGFlops(operations, cpu_time);
    double gpu_gflops_basic = calculateGFlops(operations, gpu_time_basic);
    double gpu_gflops_optimized = calculateGFlops(operations, gpu_time_optimized);
    
    // Print results
    PerformanceMetrics::Results results_basic = {
        cpu_time, gpu_time_basic, cpu_time / gpu_time_basic,
        gpu_bandwidth_basic, gpu_gflops_basic, basic_correct
    };
    
    PerformanceMetrics::Results results_optimized = {
        cpu_time, gpu_time_optimized, cpu_time / gpu_time_optimized,
        gpu_bandwidth_optimized, gpu_gflops_optimized, optimized_correct
    };
    
    PerformanceMetrics::printResults("Vector Addition (Basic)", results_basic);
    PerformanceMetrics::printResults("Vector Addition (Optimized)", results_optimized);
    
    std::cout << "\nOptimization improvement: " 
              << (gpu_time_basic / gpu_time_optimized) << "x faster" << std::endl;
    
    // Memory bandwidth comparison
    size_t memory_available = getAvailableMemory();
    std::cout << "\nMemory Usage:" << std::endl;
    std::cout << "Available GPU memory: " << formatBytes(memory_available) << std::endl;
    std::cout << "Used memory: " << formatBytes(3 * bytes) << std::endl;
    std::cout << "Memory utilization: " 
              << (100.0 * 3 * bytes / memory_available) << "%" << std::endl;
    
    // Theoretical peak performance
    std::cout << "\nTheoretical Analysis:" << std::endl;
    std::cout << "Memory-bound operation (3 memory ops per FLOP)" << std::endl;
    std::cout << "Achieved bandwidth: " << gpu_bandwidth_optimized << " GB/s" << std::endl;
    
    // Cleanup
    freeHostMemory(h_a);
    freeHostMemory(h_b);
    freeHostMemory(h_c_cpu);
    freeHostMemory(h_c_gpu);
    freeDeviceMemory(d_a);
    freeDeviceMemory(d_b);
    freeDeviceMemory(d_c);
    
    std::cout << "\n=== Vector Addition Complete ===" << std::endl;
    
    return 0;
}
