#include <hip/hip_runtime.h>
#include "hip_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>

// HIP kernel for vector addition
__global__ void vectorAddHIP(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// CPU reference implementation
void vectorAddCPU(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// Verify results
bool verifyResults(const float *gpu_result, const float *cpu_result, int n, float tolerance = 1e-5)
{
    for (int i = 0; i < n; i++)
    {
        if (fabs(gpu_result[i] - cpu_result[i]) > tolerance)
        {
            std::cout << "Mismatch at index " << i << ": GPU=" << gpu_result[i]
                      << ", CPU=" << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Initialize data
void initializeData(float *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    int n = 1 << 24; // Default: 16M elements
    if (argc > 1)
    {
        n = std::stoi(argv[1]);
    }

    std::cout << "=== HIP Vector Addition Benchmark ===" << std::endl;
    std::cout << "Vector size: " << n << " elements ("
              << (n * sizeof(float)) / (1024 * 1024) << " MB per vector)" << std::endl;

    // Initialize HIP
    initializeHIP();
    printHIPDeviceInfo();

    // Allocate host memory
    std::vector<float> h_a(n), h_b(n), h_c_gpu(n), h_c_cpu(n);

    // Initialize data
    srand(42); // Fixed seed for reproducibility
    initializeData(h_a.data(), n);
    initializeData(h_b.data(), n);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c, n * sizeof(float)));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), n * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), n * sizeof(float), hipMemcpyHostToDevice));

    // Setup execution configuration
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    std::cout << "\nExecution Configuration:" << std::endl;
    std::cout << "Block size: " << blockSize << std::endl;
    std::cout << "Grid size: " << gridSize << std::endl;
    std::cout << "Total threads: " << gridSize * blockSize << std::endl;

    // Warmup run
    hipLaunchKernelGGL(vectorAddHIP, dim3(gridSize), dim3(blockSize), 0, 0, d_a, d_b, d_c, n);
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark GPU kernel using HIP events for precise timing
    hipEvent_t start_event, stop_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));

    const int iterations = 1000; // More iterations for better timing accuracy

    HIP_CHECK(hipEventRecord(start_event, 0));
    for (int i = 0; i < iterations; i++)
    {
        hipLaunchKernelGGL(vectorAddHIP, dim3(gridSize), dim3(blockSize), 0, 0, d_a, d_b, d_c, n);
    }
    HIP_CHECK(hipEventRecord(stop_event, 0));
    HIP_CHECK(hipEventSynchronize(stop_event));

    float gpu_time_ms;
    HIP_CHECK(hipEventElapsedTime(&gpu_time_ms, start_event, stop_event));
    float gpu_time = gpu_time_ms / iterations; // Average time per iteration

    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));

    // Copy result back to host
    HIP_CHECK(hipMemcpy(h_c_gpu.data(), d_c, n * sizeof(float), hipMemcpyDeviceToHost));

    // CPU reference computation
    CPUTimer cpu_timer;
    cpu_timer.start();
    vectorAddCPU(h_a.data(), h_b.data(), h_c_cpu.data(), n);
    float cpu_time = cpu_timer.stop();

    // Performance analysis
    size_t bytes_transferred = 3 * n * sizeof(float); // Read A, B; Write C
    float bandwidth = calculateBandwidth(bytes_transferred, gpu_time);
    float theoretical_bandwidth = getHIPTheoreticalBandwidth();
    float efficiency = (bandwidth / theoretical_bandwidth) * 100.0f;

    // Results
    std::cout << "\n--- Performance Results ---" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    std::cout << std::setprecision(1);
    std::cout << "Achieved bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << "Theoretical bandwidth: " << theoretical_bandwidth << " GB/s" << std::endl;
    std::cout << "Memory efficiency: " << efficiency << "%" << std::endl;

    // Verification
    std::cout << "\n--- Verification ---" << std::endl;
    bool correct = verifyResults(h_c_gpu.data(), h_c_cpu.data(), n);
    std::cout << "Result: " << (correct ? "PASS" : "FAIL") << std::endl;

    if (correct)
    {
        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "✓ HIP vector addition completed successfully" << std::endl;
        std::cout << "✓ Achieved " << std::fixed << std::setprecision(1) << efficiency
                  << "% memory bandwidth efficiency" << std::endl;
        std::cout << "✓ " << std::fixed << std::setprecision(1) << cpu_time / gpu_time
                  << "x speedup over CPU" << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    return correct ? 0 : 1;
}
