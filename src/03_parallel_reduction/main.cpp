#include "reduction.h"
#include "../common/cuda_utils.h"
#include "../common/timer.h"
#include "../common/helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>

void benchmark_reduction_kernel(const std::string &kernel_name, int kernel_type,
                                const float *data, int n, int iterations = 10)
{
    Timer timer;
    float total_time = 0.0f;
    float result = 0.0f;

    // Warmup
    result = reduce_gpu(data, n, kernel_type);

    // Benchmark
    timer.start();
    for (int i = 0; i < iterations; i++)
    {
        result = reduce_gpu(data, n, kernel_type);
    }
    timer.stop();
    total_time = timer.getElapsedTime();

    float avg_time = total_time / iterations;
    float bandwidth = (n * sizeof(float)) / (avg_time * 1e-3) / 1e9; // GB/s

    std::cout << std::setw(20) << kernel_name
              << std::setw(12) << std::fixed << std::setprecision(3) << avg_time << " ms"
              << std::setw(12) << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
              << std::setw(15) << std::fixed << std::setprecision(6) << result
              << std::endl;
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    int n = 1 << 24; // Default: 16M elements
    if (argc > 1)
    {
        n = std::stoi(argv[1]);
    }

    std::cout << "=== CUDA Parallel Reduction Benchmark ===" << std::endl;
    std::cout << "Array size: " << n << " elements ("
              << (n * sizeof(float)) / (1024 * 1024) << " MB)" << std::endl;

    // Initialize CUDA device
    initializeCUDA();
    printDeviceInfo();

    // Allocate and initialize host data
    std::vector<float> h_data(n);
    generate_random_data(h_data.data(), n);

    // Compute CPU reference
    Timer timer;
    timer.start();
    float cpu_result = reduce_cpu(h_data.data(), n);
    timer.stop();
    float cpu_time = timer.getElapsedTime();

    std::cout << "\n--- CPU Reference ---" << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(3) << cpu_time << " ms" << std::endl;
    std::cout << "Result: " << std::fixed << std::setprecision(6) << cpu_result << std::endl;

    // GPU Benchmarks
    std::cout << "\n--- GPU Kernel Benchmarks ---" << std::endl;
    std::cout << std::setw(20) << "Kernel"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "Bandwidth"
              << std::setw(15) << "Result"
              << std::endl;
    std::cout << std::string(59, '-') << std::endl;

    // Test different kernel implementations
    benchmark_reduction_kernel("Naive", 0, h_data.data(), n);
    benchmark_reduction_kernel("Optimized", 1, h_data.data(), n);
    benchmark_reduction_kernel("Warp Optimized", 2, h_data.data(), n);
    benchmark_reduction_kernel("Coop Groups", 3, h_data.data(), n);

    // Verify results
    std::cout << "\n--- Verification ---" << std::endl;
    std::vector<std::string> kernel_names = {"Naive", "Optimized", "Warp Optimized", "Coop Groups"};
    for (int i = 0; i < 4; i++)
    {
        float gpu_result = reduce_gpu(h_data.data(), n, i);
        bool is_correct = verify_reduction(cpu_result, gpu_result);
        std::cout << std::setw(15) << kernel_names[i] << ": "
                  << (is_correct ? "PASS" : "FAIL")
                  << " (Error: " << std::scientific << std::setprecision(2)
                  << fabs(cpu_result - gpu_result) << ")" << std::endl;
    }

    // Performance Analysis
    std::cout << "\n--- Performance Analysis ---" << std::endl;
    float naive_time = 0.0f, optimized_time = 0.0f;

    Timer bench_timer;

    // Measure naive kernel
    bench_timer.start();
    for (int i = 0; i < 10; i++)
    {
        reduce_gpu(h_data.data(), n, 0);
    }
    bench_timer.stop();
    naive_time = bench_timer.getElapsedTime() / 10.0f;

    // Measure optimized kernel
    bench_timer.start();
    for (int i = 0; i < 10; i++)
    {
        reduce_gpu(h_data.data(), n, 2);
    }
    bench_timer.stop();
    optimized_time = bench_timer.getElapsedTime() / 10.0f;

    float speedup = naive_time / optimized_time;
    std::cout << "Speedup (Warp Optimized vs Naive): " << std::fixed << std::setprecision(2)
              << speedup << "x" << std::endl;

    // Memory bandwidth analysis
    float theoretical_bandwidth = getTheoreticalBandwidth();
    float achieved_bandwidth = (n * sizeof(float)) / (optimized_time * 1e-3) / 1e9;
    float efficiency = (achieved_bandwidth / theoretical_bandwidth) * 100.0f;

    std::cout << "Theoretical Bandwidth: " << std::fixed << std::setprecision(1)
              << theoretical_bandwidth << " GB/s" << std::endl;
    std::cout << "Achieved Bandwidth: " << std::fixed << std::setprecision(1)
              << achieved_bandwidth << " GB/s" << std::endl;
    std::cout << "Memory Efficiency: " << std::fixed << std::setprecision(1)
              << efficiency << "%" << std::endl;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "✓ Demonstrated multiple reduction optimization techniques" << std::endl;
    std::cout << "✓ Eliminated warp divergence through sequential addressing" << std::endl;
    std::cout << "✓ Utilized warp shuffle instructions for better performance" << std::endl;
    std::cout << "✓ Achieved " << std::fixed << std::setprecision(1) << efficiency
              << "% memory bandwidth efficiency" << std::endl;

    return 0;
}

float reductionCPU(const float *data, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        sum += data[i];
    }
    return sum;
}

int main()
{
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
