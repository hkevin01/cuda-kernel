#include "matrix_mul.h"
#ifdef USE_CUDA
#include "cuda_utils.h"
#include <cublas_v2.h>
#endif
#include "timer.h"
#include "helper_functions.h"
#include <iostream>

int main()
{
    std::cout << "=== CUDA Matrix Multiplication Example ===" << std::endl;

    // Initialize CUDA
    setOptimalDevice();
    printDeviceInfo();

    // Matrix dimensions: C = A * B
    // A: M x K, B: K x N, C: M x N
    const int M = 2048; // Rows of A and C
    const int K = 2048; // Cols of A, Rows of B
    const int N = 2048; // Cols of B and C

    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);
    const size_t total_memory = size_A + size_B + size_C;

    std::cout << "Matrix dimensions:" << std::endl;
    std::cout << "A: " << M << " x " << K << " (" << formatBytes(size_A) << ")" << std::endl;
    std::cout << "B: " << K << " x " << N << " (" << formatBytes(size_B) << ")" << std::endl;
    std::cout << "C: " << M << " x " << N << " (" << formatBytes(size_C) << ")" << std::endl;
    std::cout << "Total GPU memory required: " << formatBytes(total_memory) << std::endl;

    // Check available memory
    size_t available_memory = getAvailableMemory();
    if (total_memory > available_memory)
    {
        std::cerr << "Insufficient GPU memory! Required: " << formatBytes(total_memory)
                  << ", Available: " << formatBytes(available_memory) << std::endl;
        return -1;
    }

    // Allocate host memory
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    allocateHostMemory(&h_A, M * K);
    allocateHostMemory(&h_B, K * N);
    allocateHostMemory(&h_C_cpu, M * N);
    allocateHostMemory(&h_C_gpu, M * N);

    // Initialize matrices
    std::cout << "\nInitializing matrices..." << std::endl;
    initializeMatrix(h_A, M, K, 0.0f, 1.0f);
    initializeMatrix(h_B, K, N, 0.0f, 1.0f);

    // Print sample data
    printMatrix(h_A, M, K, "Matrix A (sample)");
    printMatrix(h_B, K, N, "Matrix B (sample)");

    // CPU implementation
    std::cout << "Running CPU implementation..." << std::endl;
    CPUTimer cpu_timer;
    cpu_timer.start();
    matrixMulCPU(h_A, h_B, h_C_cpu, M, N, K);
    double cpu_time = cpu_timer.stop();

    printMatrix(h_C_cpu, M, N, "CPU Result (sample)");

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    allocateDeviceMemory(&d_A, M * K);
    allocateDeviceMemory(&d_B, K * N);
    allocateDeviceMemory(&d_C, M * N);

    // Copy data to device
    copyHostToDevice(d_A, h_A, M * K);
    copyHostToDevice(d_B, h_B, K * N);

    // Test 1: Naive implementation
    std::cout << "\n--- Testing Naive Implementation ---" << std::endl;
    dim3 block_size_naive(16, 16);
    dim3 grid_size_naive((N + block_size_naive.x - 1) / block_size_naive.x,
                         (M + block_size_naive.y - 1) / block_size_naive.y);

    CudaTimer gpu_timer;
    gpu_timer.start();

    matrixMulNaive<<<grid_size_naive, block_size_naive>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();

    float gpu_time_naive = gpu_timer.stop();

    copyDeviceToHost(h_C_gpu, d_C, M * N);
    bool naive_correct = verifyMatrixResult(h_C_cpu, h_C_gpu, M, N);

    // Test 2: Shared memory implementation
    std::cout << "\n--- Testing Shared Memory Implementation ---" << std::endl;
    dim3 block_size_shared(16, 16);
    dim3 grid_size_shared((N + block_size_shared.x - 1) / block_size_shared.x,
                          (M + block_size_shared.y - 1) / block_size_shared.y);

    gpu_timer.start();

    matrixMulShared<<<grid_size_shared, block_size_shared>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();

    float gpu_time_shared = gpu_timer.stop();

    copyDeviceToHost(h_C_gpu, d_C, M * N);
    bool shared_correct = verifyMatrixResult(h_C_cpu, h_C_gpu, M, N);

    // Test 3: Optimized implementation
    std::cout << "\n--- Testing Optimized Implementation ---" << std::endl;
    dim3 block_size_opt(16, 16);
    dim3 grid_size_opt((N + block_size_opt.x - 1) / block_size_opt.x,
                       (M + block_size_opt.y - 1) / block_size_opt.y);

    gpu_timer.start();

    matrixMulOptimized<<<grid_size_opt, block_size_opt>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK_KERNEL();

    float gpu_time_optimized = gpu_timer.stop();

    copyDeviceToHost(h_C_gpu, d_C, M * N);
    bool optimized_correct = verifyMatrixResult(h_C_cpu, h_C_gpu, M, N);

    // Test 4: cuBLAS implementation for comparison
    std::cout << "\n--- Testing cuBLAS Implementation ---" << std::endl;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    const float alpha = 1.0f, beta = 0.0f;

    gpu_timer.start();

    // cuBLAS uses column-major format, so we compute C^T = B^T * A^T
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N, // B^T
                d_A, K, // A^T
                &beta,
                d_C, N); // C^T

    CUDA_CHECK(cudaDeviceSynchronize());
    float gpu_time_cublas = gpu_timer.stop();

    copyDeviceToHost(h_C_gpu, d_C, M * N);
    bool cublas_correct = verifyMatrixResult(h_C_cpu, h_C_gpu, M, N);

    cublasDestroy(cublas_handle);

    // Performance analysis
    std::cout << "\n=== Performance Analysis ===" << std::endl;

    const size_t total_ops = 2ULL * M * N * K;            // Multiply-add operations
    const size_t memory_ops = (size_A + size_B + size_C); // Memory transferred

    // Calculate performance metrics
    double cpu_gflops = calculateGFlops(total_ops, cpu_time);
    double naive_gflops = calculateGFlops(total_ops, gpu_time_naive);
    double shared_gflops = calculateGFlops(total_ops, gpu_time_shared);
    double optimized_gflops = calculateGFlops(total_ops, gpu_time_optimized);
    double cublas_gflops = calculateGFlops(total_ops, gpu_time_cublas);

    double naive_bandwidth = calculateBandwidth(memory_ops, gpu_time_naive);
    double shared_bandwidth = calculateBandwidth(memory_ops, gpu_time_shared);
    double optimized_bandwidth = calculateBandwidth(memory_ops, gpu_time_optimized);
    double cublas_bandwidth = calculateBandwidth(memory_ops, gpu_time_cublas);

    // Print detailed results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nTiming Results:" << std::endl;
    std::cout << "CPU Time:        " << cpu_time << " ms" << std::endl;
    std::cout << "GPU Naive:       " << gpu_time_naive << " ms" << std::endl;
    std::cout << "GPU Shared:      " << gpu_time_shared << " ms" << std::endl;
    std::cout << "GPU Optimized:   " << gpu_time_optimized << " ms" << std::endl;
    std::cout << "cuBLAS:          " << gpu_time_cublas << " ms" << std::endl;

    std::cout << "\nPerformance (GFLOPS):" << std::endl;
    std::cout << "CPU:             " << cpu_gflops << " GFLOPS" << std::endl;
    std::cout << "GPU Naive:       " << naive_gflops << " GFLOPS" << std::endl;
    std::cout << "GPU Shared:      " << shared_gflops << " GFLOPS" << std::endl;
    std::cout << "GPU Optimized:   " << optimized_gflops << " GFLOPS" << std::endl;
    std::cout << "cuBLAS:          " << cublas_gflops << " GFLOPS" << std::endl;

    std::cout << "\nSpeedup vs CPU:" << std::endl;
    std::cout << "GPU Naive:       " << (cpu_time / gpu_time_naive) << "x" << std::endl;
    std::cout << "GPU Shared:      " << (cpu_time / gpu_time_shared) << "x" << std::endl;
    std::cout << "GPU Optimized:   " << (cpu_time / gpu_time_optimized) << "x" << std::endl;
    std::cout << "cuBLAS:          " << (cpu_time / gpu_time_cublas) << "x" << std::endl;

    std::cout << "\nVerification:" << std::endl;
    std::cout << "Naive:           " << (naive_correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Shared:          " << (shared_correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Optimized:       " << (optimized_correct ? "PASSED" : "FAILED") << std::endl;
    std::cout << "cuBLAS:          " << (cublas_correct ? "PASSED" : "FAILED") << std::endl;

    // Analysis
    std::cout << "\n=== Optimization Analysis ===" << std::endl;
    std::cout << "Shared vs Naive:     " << (gpu_time_naive / gpu_time_shared) << "x improvement" << std::endl;
    std::cout << "Optimized vs Shared: " << (gpu_time_shared / gpu_time_optimized) << "x improvement" << std::endl;
    std::cout << "cuBLAS efficiency:   " << (optimized_gflops / cublas_gflops * 100) << "% of cuBLAS performance" << std::endl;

    // Memory efficiency
    DeviceInfo device_info = getDeviceInfo();
    std::cout << "\nMemory Analysis:" << std::endl;
    std::cout << "Arithmetic intensity: " << (double)total_ops / memory_ops << " FLOPS/byte" << std::endl;
    std::cout << "Memory utilization:   " << (total_memory * 100.0 / device_info.global_memory) << "%" << std::endl;

    // Cleanup
    freeHostMemory(h_A);
    freeHostMemory(h_B);
    freeHostMemory(h_C_cpu);
    freeHostMemory(h_C_gpu);
    freeDeviceMemory(d_A);
    freeDeviceMemory(d_B);
    freeDeviceMemory(d_C);

    std::cout << "\n=== Matrix Multiplication Complete ===" << std::endl;

    return 0;
}
