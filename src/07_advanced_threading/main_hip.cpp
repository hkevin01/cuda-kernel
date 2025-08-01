#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hip/hip_runtime.h>
#include "hip_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <random>

// Define structures
struct ThreadData
{
    float4 position;
    float4 velocity;
    float energy;
    int state;
};

// Safe kernel declarations (using the safe version)
extern "C"
{
    void launchSafeAdvancedThreading(ThreadData *data, float *results, int *counters, int n, int iterations, int blockSize, int gridSize);
    void launchSafeLockFreeOperations(int *data, int *results, int n, int operations_per_thread, int blockSize, int gridSize);
    void launchSafeMemoryPatterns(float4 *input, float4 *output, int width, int height, int blockSize, int gridSize);
}

class SafeAdvancedThreadingBenchmark
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
    SafeAdvancedThreadingBenchmark(int size) : n(size)
    {
        // Allocate host memory
        h_thread_data = new ThreadData[n];
        h_results = new float[n + 64]; // Extra space for block results
        h_counters = new int[64]();

        // Initialize data with safe bounds
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-5.0f, 5.0f);  // Smaller range

        for (int i = 0; i < n; i++)
        {
            h_thread_data[i].position = {dis(gen), dis(gen), dis(gen), dis(gen)};
            h_thread_data[i].velocity = {dis(gen) * 0.05f, dis(gen) * 0.05f, dis(gen) * 0.05f, dis(gen) * 0.05f};
            h_thread_data[i].energy = dis(gen);
            h_thread_data[i].state = 0;
        }

        // Allocate device memory
        HIP_CHECK(hipMalloc(&d_thread_data, n * sizeof(ThreadData)));
        HIP_CHECK(hipMalloc(&d_results, (n + 64) * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_counters, 64 * sizeof(int)));

        // Copy to device
        HIP_CHECK(hipMemcpy(d_thread_data, h_thread_data, n * sizeof(ThreadData), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_counters, 0, 64 * sizeof(int)));
    }

    ~SafeAdvancedThreadingBenchmark()
    {
        delete[] h_thread_data;
        delete[] h_results;
        delete[] h_counters;
        HIP_CHECK(hipFree(d_thread_data));
        HIP_CHECK(hipFree(d_results));
        HIP_CHECK(hipFree(d_counters));
    }

    void runSafeAdvancedSync()
    {
        std::cout << "\n=== Safe Advanced Thread Synchronization Test ===" << std::endl;

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        // Limit grid size for safety
        gridSize = std::min(gridSize, 32);
        int iterations = 10;

        std::cout << "Configuration: " << gridSize << " blocks, " << blockSize << " threads/block" << std::endl;
        std::cout << "Data size: " << n << " elements" << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Warmup run
        launchSafeAdvancedThreading(d_thread_data, d_results, d_counters, n, iterations, blockSize, gridSize);
        HIP_CHECK(hipDeviceSynchronize());

        // Check for errors after warmup
        hipError_t error = hipGetLastError();
        if (error != hipSuccess) {
            std::cerr << "HIP error after warmup: " << hipGetErrorString(error) << std::endl;
            return;
        }

        // Benchmark with fewer runs for safety
        const int runs = 10;
        HIP_CHECK(hipEventRecord(start));

        for (int run = 0; run < runs; run++)
        {
            HIP_CHECK(hipMemset(d_counters, 0, 64 * sizeof(int)));
            launchSafeAdvancedThreading(d_thread_data, d_results, d_counters, n, iterations, blockSize, gridSize);
            
            // Check for errors each run
            error = hipGetLastError();
            if (error != hipSuccess) {
                std::cerr << "HIP error in run " << run << ": " << hipGetErrorString(error) << std::endl;
                break;
            }
        }

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= runs;

        // Copy results back
        HIP_CHECK(hipMemcpy(h_results, d_results, (n + 64) * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_counters, d_counters, 64 * sizeof(int), hipMemcpyDeviceToHost));

        // Analyze results with safety checks
        float total_energy = 0.0f;
        int valid_results = 0;
        for (int i = 0; i < n; i++)
        {
            if (std::isfinite(h_results[i])) {
                total_energy += h_results[i];
                valid_results++;
            }
        }

        int total_syncs = 0;
        for (int i = 0; i < gridSize; i++)
        {
            total_syncs += h_counters[i];
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Valid results: " << valid_results << "/" << n << std::endl;
        std::cout << "Total energy computed: " << total_energy << std::endl;
        std::cout << "Total synchronizations: " << total_syncs << std::endl;
        std::cout << "Performance: " << (valid_results * iterations * runs) / (gpu_time * 1000.0f)
                  << " operations/second" << std::endl;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
};

class SafeLockFreeOperationsBenchmark
{
private:
    int n;
    int *h_data;
    int *h_results;
    int *d_data;
    int *d_results;

public:
    SafeLockFreeOperationsBenchmark(int size) : n(size)
    {
        h_data = new int[n];
        h_results = new int[n + 1];

        // Initialize with safe values
        for (int i = 0; i < n; i++)
        {
            h_data[i] = i % 1000;  // Bounded values
        }

        HIP_CHECK(hipMalloc(&d_data, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_results, (n + 1) * sizeof(int)));

        HIP_CHECK(hipMemcpy(d_data, h_data, n * sizeof(int), hipMemcpyHostToDevice));
    }

    ~SafeLockFreeOperationsBenchmark()
    {
        delete[] h_data;
        delete[] h_results;
        HIP_CHECK(hipFree(d_data));
        HIP_CHECK(hipFree(d_results));
    }

    void runSafeLockFreeTest()
    {
        std::cout << "\n=== Safe Lock-Free Operations Test ===" << std::endl;

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        int operations_per_thread = 10;

        std::cout << "Configuration: " << gridSize << " blocks, " << blockSize << " threads/block" << std::endl;
        std::cout << "Operations per thread: " << operations_per_thread << std::endl;

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start));
        launchSafeLockFreeOperations(d_data, d_results, n, operations_per_thread, blockSize, gridSize);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

        // Copy results back
        HIP_CHECK(hipMemcpy(h_results, d_results, (n + 1) * sizeof(int), hipMemcpyDeviceToHost));

        // Analyze results
        int total_operations = 0;
        for (int i = 0; i < n; i++)
        {
            total_operations += h_results[i];
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Total operations: " << total_operations << std::endl;
        std::cout << "Performance: " << (total_operations / gpu_time) << " operations/ms" << std::endl;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <data_size>" << std::endl;
    std::cout << "  data_size: Number of elements to process (default: 10000)" << std::endl;
    std::cout << std::endl;
    std::cout << "This program runs safe advanced threading benchmarks using HIP." << std::endl;
    std::cout << "The safe version avoids complex cooperative groups and uses" << std::endl;
    std::cout << "bounded operations to prevent system crashes." << std::endl;
}

int main(int argc, char* argv[])
{
    std::cout << "=== Safe Advanced GPU Threading and Synchronization Benchmarks ===" << std::endl;
    
    // Initialize HIP
    HIP_CHECK(hipInit(0));
    
    // Get device info
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    std::cout << "HIP initialized successfully with " << deviceCount << " device(s)" << std::endl;
    
    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        return -1;
    }
    
    // Set device
    HIP_CHECK(hipSetDevice(0));
    
    // Print device info
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "\n=== HIP Device Information ===" << std::endl;
    std::cout << "Device 0: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Number of multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Memory clock rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Theoretical bandwidth: " << (2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8.0) / 1000000.0 << " GB/s" << std::endl;
    std::cout << "================================" << std::endl;
    
    // Parse command line arguments
    int data_size = 10000;
    if (argc > 1) {
        data_size = std::atoi(argv[1]);
    }
    
    if (data_size <= 0) {
        std::cerr << "Invalid data size: " << argv[1] << std::endl;
        printUsage(argv[0]);
        return -1;
    }
    
    std::cout << "\nBenchmark Configuration:" << std::endl;
    std::cout << "Thread data size: " << data_size << " elements" << std::endl;
    
    try {
        // Run safe advanced threading benchmark
        SafeAdvancedThreadingBenchmark advancedBench(data_size);
        advancedBench.runSafeAdvancedSync();
        
        // Run safe lock-free operations benchmark
        SafeLockFreeOperationsBenchmark lockFreeBench(data_size);
        lockFreeBench.runSafeLockFreeTest();
        
        std::cout << "\n=== All Safe Advanced Threading Tests Completed Successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during benchmark execution: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
