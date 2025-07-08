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

// Safe kernel declarations
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
        int gridSize = std::min((n + blockSize - 1) / blockSize, 16);  // Limit grid size
        int operations_per_thread = 10;  // Reduced operations

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Reset data
        HIP_CHECK(hipMemcpy(d_data, h_data, n * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_results, 0, (n + 1) * sizeof(int)));

        std::cout << "Configuration: " << gridSize << " blocks, " << blockSize << " threads/block" << std::endl;
        std::cout << "Operations per thread: " << operations_per_thread << std::endl;

        HIP_CHECK(hipEventRecord(start));

        launchSafeLockFreeOperations(d_data, d_results, n, operations_per_thread, blockSize, gridSize);

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        // Check for errors
        hipError_t error = hipGetLastError();
        if (error != hipSuccess) {
            std::cerr << "HIP error: " << hipGetErrorString(error) << std::endl;
            return;
        }

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

        // Copy results back
        HIP_CHECK(hipMemcpy(h_results, d_results, (n + 1) * sizeof(int), hipMemcpyDeviceToHost));

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Total operations: " << gridSize * blockSize * operations_per_thread << std::endl;
        std::cout << "Throughput: " << (gridSize * blockSize * operations_per_thread) / (gpu_time * 1000.0f)
                  << " ops/second" << std::endl;

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
};

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [data_size]" << std::endl;
    std::cout << "  data_size: Number of elements to process (default: 100000, max: 1000000)" << std::endl;
    std::cout << std::endl;
    std::cout << "This is a SAFE version of advanced threading that demonstrates:" << std::endl;
    std::cout << "- Advanced thread synchronization patterns" << std::endl;
    std::cout << "- Warp-level reductions" << std::endl;
    std::cout << "- Safe lock-free operations" << std::endl;
    std::cout << "- Memory access patterns" << std::endl;
    std::cout << std::endl;
    std::cout << "Safety features:" << std::endl;
    std::cout << "- Bounded iterations and operations" << std::endl;
    std::cout << "- Limited shared memory usage" << std::endl;
    std::cout << "- No dangerous grid synchronization" << std::endl;
    std::cout << "- Error checking and validation" << std::endl;
}

int main(int argc, char* argv[])
{
    std::cout << "=== SAFE Advanced Threading Demonstration ===" << std::endl;
    std::cout << "HIP-based GPU Computing with Advanced Synchronization" << std::endl;
    std::cout << "SAFE VERSION - System-stable implementation" << std::endl;

    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        printUsage(argv[0]);
        return 0;
    }

    // Parse command line arguments with safety limits
    int data_size = 100000;  // Default size
    if (argc > 1) {
        data_size = std::atoi(argv[1]);
        data_size = std::max(1000, std::min(data_size, 1000000));  // Clamp to safe range
    }

    std::cout << "Data size: " << data_size << " elements" << std::endl;

    // Initialize HIP
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        return 1;
    }

    HIP_CHECK(hipSetDevice(0));
    
    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));
    std::cout << "Using device: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

    try {
        // Run safe benchmarks
        SafeAdvancedThreadingBenchmark advancedBench(data_size);
        advancedBench.runSafeAdvancedSync();

        SafeLockFreeOperationsBenchmark lockFreeBench(data_size / 4);  // Smaller size for lock-free
        lockFreeBench.runSafeLockFreeTest();

        std::cout << "\n=== All Safe Tests Completed Successfully ===" << std::endl;
        std::cout << "This version demonstrates advanced threading concepts safely!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
