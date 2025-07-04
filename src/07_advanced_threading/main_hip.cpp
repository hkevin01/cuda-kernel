#include <hip/hip_runtime.h>
#include "hip_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <random>

// Define structures first
struct ThreadData
{
    float4 position;
    float4 velocity;
    float energy;
    int state;
};

// Kernel declarations
extern "C"
{
    __global__ void advancedThreadSync(ThreadData *data, float *results, int *counters, int n, int iterations);
    __global__ void lockFreeOperations(int *data, int *results, int n, int operations_per_thread);
    __global__ void complexMemoryPatterns(float4 *input, float4 *output, int width, int height, int depth);
    __global__ void producerConsumerPattern(int *input_queue, int *output_queue, int *queue_heads,
                                            int *queue_tails, volatile int *flags, int queue_size, int num_items);
}

class AdvancedThreadingBenchmark
{
private:
    int n;
    ThreadData *h_thread_data;
    ThreadData *d_thread_data;
    float *h_results;
    float *d_results;
    int *h_counters;
    int *d_counters;

public:
    AdvancedThreadingBenchmark(int size) : n(size)
    {
        // Allocate host memory
        h_thread_data = new ThreadData[n];
        h_results = new float[n + 128]; // Extra space for block results
        h_counters = new int[128]();

        // Initialize data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        for (int i = 0; i < n; i++)
        {
            h_thread_data[i].position = {dis(gen), dis(gen), dis(gen), dis(gen)};
            h_thread_data[i].velocity = {dis(gen) * 0.1f, dis(gen) * 0.1f, dis(gen) * 0.1f, dis(gen) * 0.1f};
            h_thread_data[i].energy = dis(gen);
            h_thread_data[i].state = 0;
        }

        // Allocate device memory
        HIP_CHECK(hipMalloc(&d_thread_data, n * sizeof(ThreadData)));
        HIP_CHECK(hipMalloc(&d_results, (n + 128) * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_counters, 128 * sizeof(int)));

        // Copy to device
        HIP_CHECK(hipMemcpy(d_thread_data, h_thread_data, n * sizeof(ThreadData), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_counters, 0, 128 * sizeof(int)));
    }

    ~AdvancedThreadingBenchmark()
    {
        delete[] h_thread_data;
        delete[] h_results;
        delete[] h_counters;
        HIP_CHECK(hipFree(d_thread_data));
        HIP_CHECK(hipFree(d_results));
        HIP_CHECK(hipFree(d_counters));
    }

    void runAdvancedSync()
    {
        std::cout << "\n=== Advanced Thread Synchronization Test ===" << std::endl;

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        int iterations = 10;

        std::cout << "Configuration: " << gridSize << " blocks, " << blockSize << " threads/block" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;

        CPUTimer timer;
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup
        hipLaunchKernelGGL(advancedThreadSync, dim3(gridSize), dim3(blockSize), 0, 0,
                           d_thread_data, d_results, d_counters, n, iterations);
        HIP_CHECK(hipDeviceSynchronize());

        // Benchmark
        const int runs = 100;
        HIP_CHECK(hipEventRecord(start));

        for (int run = 0; run < runs; run++)
        {
            HIP_CHECK(hipMemset(d_counters, 0, 128 * sizeof(int)));
            hipLaunchKernelGGL(advancedThreadSync, dim3(gridSize), dim3(blockSize), 0, 0,
                               d_thread_data, d_results, d_counters, n, iterations);
        }

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= runs;

        // Copy results back
        HIP_CHECK(hipMemcpy(h_results, d_results, (n + 128) * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_counters, d_counters, 128 * sizeof(int), hipMemcpyDeviceToHost));

        // Analyze results
        float total_energy = 0.0f;
        for (int i = 0; i < n; i++)
        {
            total_energy += h_results[i];
        }

        int total_syncs = 0;
        for (int i = 0; i < gridSize; i++)
        {
            total_syncs += h_counters[i];
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Total energy computed: " << total_energy << std::endl;
        std::cout << "Total synchronizations: " << total_syncs << std::endl;
        std::cout << "Performance: " << (n * iterations * runs) / (gpu_time * 1000.0f)
                  << " operations/second" << std::endl;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
};

class LockFreeOperationsBenchmark
{
private:
    int n;
    int *h_data;
    int *h_results;
    int *d_data;
    int *d_results;

public:
    LockFreeOperationsBenchmark(int size) : n(size)
    {
        h_data = new int[n];
        h_results = new int[n + 1];

        // Initialize with random data
        for (int i = 0; i < n; i++)
        {
            h_data[i] = rand() % 1000;
        }

        HIP_CHECK(hipMalloc(&d_data, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_results, (n + 1) * sizeof(int)));

        HIP_CHECK(hipMemcpy(d_data, h_data, n * sizeof(int), hipMemcpyHostToDevice));
    }

    ~LockFreeOperationsBenchmark()
    {
        delete[] h_data;
        delete[] h_results;
        HIP_CHECK(hipFree(d_data));
        HIP_CHECK(hipFree(d_results));
    }

    void runLockFreeTest()
    {
        std::cout << "\n=== Lock-Free Operations Test ===" << std::endl;

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        int operations_per_thread = 100;

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Reset data
        HIP_CHECK(hipMemcpy(d_data, h_data, n * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_results, 0, (n + 1) * sizeof(int)));

        HIP_CHECK(hipEventRecord(start));

        hipLaunchKernelGGL(lockFreeOperations, dim3(gridSize), dim3(blockSize), 0, 0,
                           d_data, d_results, n, operations_per_thread);

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

        // Copy results
        HIP_CHECK(hipMemcpy(h_results, d_results, (n + 1) * sizeof(int), hipMemcpyDeviceToHost));

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Total atomic operations: " << n * operations_per_thread << std::endl;
        std::cout << "Final accumulator: " << h_results[n] << std::endl;
        std::cout << "Atomic ops/second: " << (n * operations_per_thread) / (gpu_time / 1000.0f) << std::endl;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
};

class ComplexMemoryPatternsBenchmark
{
private:
    int width, height, depth;
    float4 *h_input;
    float4 *h_output;
    float4 *d_input;
    float4 *d_output;

public:
    ComplexMemoryPatternsBenchmark(int w, int h, int d) : width(w), height(h), depth(d)
    {
        int size = width * height * depth;
        h_input = new float4[size];
        h_output = new float4[size];

        // Initialize with random data
        for (int i = 0; i < size; i++)
        {
            h_input[i] = {
                static_cast<float>(rand()) / RAND_MAX,
                static_cast<float>(rand()) / RAND_MAX,
                static_cast<float>(rand()) / RAND_MAX,
                static_cast<float>(rand()) / RAND_MAX};
        }

        HIP_CHECK(hipMalloc(&d_input, size * sizeof(float4)));
        HIP_CHECK(hipMalloc(&d_output, size * sizeof(float4)));

        HIP_CHECK(hipMemcpy(d_input, h_input, size * sizeof(float4), hipMemcpyHostToDevice));
    }

    ~ComplexMemoryPatternsBenchmark()
    {
        delete[] h_input;
        delete[] h_output;
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }

    void runMemoryPatternsTest()
    {
        std::cout << "\n=== Complex Memory Patterns Test ===" << std::endl;
        std::cout << "Volume: " << width << "x" << height << "x" << depth << std::endl;

        dim3 blockSize(8, 8, 8);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y,
                      (depth + blockSize.z - 1) / blockSize.z);

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start));

        hipLaunchKernelGGL(complexMemoryPatterns, gridSize, blockSize, 0, 0,
                           d_input, d_output, width, height, depth);

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

        // Copy results
        int size = width * height * depth;
        HIP_CHECK(hipMemcpy(h_output, d_output, size * sizeof(float4), hipMemcpyDeviceToHost));

        // Calculate memory bandwidth
        size_t bytes_transferred = size * sizeof(float4) * 2; // Read + write
        float bandwidth = bytes_transferred / (gpu_time / 1000.0f) / (1024 * 1024 * 1024);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Memory bandwidth: " << bandwidth << " GB/s" << std::endl;
        std::cout << "Elements processed: " << size << std::endl;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
};

int main(int argc, char **argv)
{
    std::cout << "=== Advanced GPU Threading and Synchronization Benchmarks ===" << std::endl;

    // Initialize HIP
    initializeHIP();
    printHIPDeviceInfo();

    // Parse command line arguments
    int n = (argc > 1) ? std::stoi(argv[1]) : 1024 * 1024;
    int volume_size = (argc > 2) ? std::stoi(argv[2]) : 64;

    std::cout << "\nBenchmark Configuration:" << std::endl;
    std::cout << "Thread data size: " << n << " elements" << std::endl;
    std::cout << "3D volume size: " << volume_size << "^3" << std::endl;

    try
    {
        // Run advanced threading synchronization benchmark
        AdvancedThreadingBenchmark threading_bench(n);
        threading_bench.runAdvancedSync();

        // Run lock-free operations benchmark
        LockFreeOperationsBenchmark lockfree_bench(n);
        lockfree_bench.runLockFreeTest();

        // Run complex memory patterns benchmark
        ComplexMemoryPatternsBenchmark memory_bench(volume_size, volume_size, volume_size);
        memory_bench.runMemoryPatternsTest();

        std::cout << "\n=== All Advanced Threading Tests Completed Successfully ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
