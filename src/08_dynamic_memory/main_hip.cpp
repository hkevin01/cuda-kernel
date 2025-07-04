#include <hip/hip_runtime.h>
#include "hip_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <random>

// Include kernel declarations
struct GPUMemoryPool
{
    char *pool;
    size_t *allocation_sizes;
    int *free_list;
    int pool_size;
    int max_allocations;
    int next_free;
};

struct DynamicNode
{
    float4 data;
    int *children;
    int num_children;
    int depth;
};

struct ComplexQueue
{
    float *data;
    volatile int *head;
    volatile int *tail;
    int *priorities;
    int capacity;
    int num_priorities;
};

extern "C"
{
    __global__ void dynamicTreeBuild(GPUMemoryPool *memory_pool, DynamicNode *nodes,
                                     float4 *input_data, int *results, int n, int max_depth);
    __global__ void gpuQuickSort(float *data, int *indices, GPUMemoryPool *memory_pool, int n, int depth);
    __global__ void complexMemoryCoalescing(float *input, float *output, int *pattern,
                                            int width, int height, int stride_pattern);
    __global__ void complexProducerConsumer(ComplexQueue *queues, float *results,
                                            int num_queues, int items_per_thread, int num_consumers);
}

class DynamicMemoryBenchmark
{
private:
    GPUMemoryPool h_pool, *d_pool;
    char *d_memory_pool;
    size_t *d_allocation_sizes;
    int *d_free_list;

public:
    DynamicMemoryBenchmark()
    {
        // Initialize memory pool on host
        h_pool.pool_size = 64 * 1024 * 1024; // 64MB pool
        h_pool.max_allocations = 10000;
        h_pool.next_free = 0;

        // Allocate device memory for pool
        HIP_CHECK(hipMalloc(&d_memory_pool, h_pool.pool_size));
        HIP_CHECK(hipMalloc(&d_allocation_sizes, h_pool.max_allocations * sizeof(size_t)));
        HIP_CHECK(hipMalloc(&d_free_list, h_pool.max_allocations * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_pool, sizeof(GPUMemoryPool)));

        // Set up pool structure on device
        h_pool.pool = d_memory_pool;
        h_pool.allocation_sizes = d_allocation_sizes;
        h_pool.free_list = d_free_list;

        HIP_CHECK(hipMemcpy(d_pool, &h_pool, sizeof(GPUMemoryPool), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_allocation_sizes, 0, h_pool.max_allocations * sizeof(size_t)));
        HIP_CHECK(hipMemset(d_free_list, 0, h_pool.max_allocations * sizeof(int)));
    }

    ~DynamicMemoryBenchmark()
    {
        HIP_CHECK(hipFree(d_memory_pool));
        HIP_CHECK(hipFree(d_allocation_sizes));
        HIP_CHECK(hipFree(d_free_list));
        HIP_CHECK(hipFree(d_pool));
    }

    void runDynamicTreeTest(int n)
    {
        std::cout << "\n=== Dynamic Tree Building Test ===" << std::endl;
        std::cout << "Number of root nodes: " << n << std::endl;

        // Allocate memory for nodes and data
        DynamicNode *d_nodes;
        float4 *d_input_data;
        int *d_results;

        HIP_CHECK(hipMalloc(&d_nodes, n * 4 * sizeof(DynamicNode))); // Extra space for children
        HIP_CHECK(hipMalloc(&d_input_data, n * sizeof(float4)));
        HIP_CHECK(hipMalloc(&d_results, (n + 1) * sizeof(int)));

        // Initialize input data
        std::vector<float4> h_input_data(n);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

        for (int i = 0; i < n; i++)
        {
            h_input_data[i] = {dis(gen), dis(gen), dis(gen), dis(gen)};
        }

        HIP_CHECK(hipMemcpy(d_input_data, h_input_data.data(), n * sizeof(float4), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_results, 0, (n + 1) * sizeof(int)));

        // Reset memory pool
        HIP_CHECK(hipMemcpy(d_pool, &h_pool, sizeof(GPUMemoryPool), hipMemcpyHostToDevice));

        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        int max_depth = 3;

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start));

        hipLaunchKernelGGL(dynamicTreeBuild, dim3(gridSize), dim3(blockSize), 0, 0,
                           d_pool, d_nodes, d_input_data, d_results, n, max_depth);

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

        // Copy results back
        std::vector<int> h_results(n + 1);
        HIP_CHECK(hipMemcpy(h_results.data(), d_results, (n + 1) * sizeof(int), hipMemcpyDeviceToHost));

        int total_nodes = 0;
        int successful_trees = 0;
        for (int i = 0; i < n; i++)
        {
            if (h_results[i] > 0)
            {
                total_nodes += h_results[i];
                successful_trees++;
            }
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Successful trees: " << successful_trees << "/" << n << std::endl;
        std::cout << "Total nodes created: " << total_nodes << std::endl;
        std::cout << "Nodes allocated from pool: " << h_results[n] << std::endl;
        std::cout << "Average nodes per tree: " << (successful_trees > 0 ? (float)total_nodes / successful_trees : 0) << std::endl;

        HIP_CHECK(hipFree(d_nodes));
        HIP_CHECK(hipFree(d_input_data));
        HIP_CHECK(hipFree(d_results));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    GPUMemoryPool *getDevicePool() { return d_pool; }
};

class MemoryCoalescingBenchmark
{
private:
    float *d_input;
    float *d_output;
    int *d_pattern;
    int width, height;

public:
    MemoryCoalescingBenchmark(int w, int h) : width(w), height(h)
    {
        int size = width * height;

        HIP_CHECK(hipMalloc(&d_input, size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_output, size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_pattern, 32 * sizeof(int)));

        // Initialize input data
        std::vector<float> h_input(size);
        std::vector<int> h_pattern(32);

        for (int i = 0; i < size; i++)
        {
            h_input[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        for (int i = 0; i < 32; i++)
        {
            h_pattern[i] = rand() % 4;
        }

        HIP_CHECK(hipMemcpy(d_input, h_input.data(), size * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_pattern, h_pattern.data(), 32 * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_output, 0, size * sizeof(float)));
    }

    ~MemoryCoalescingBenchmark()
    {
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_pattern));
    }

    void runCoalescingTest()
    {
        std::cout << "\n=== Complex Memory Coalescing Test ===" << std::endl;
        std::cout << "Matrix size: " << width << "x" << height << std::endl;

        int blockSize = 256;
        int gridSize = (width * height + blockSize - 1) / blockSize;

        std::vector<int> stride_patterns = {1, 2, 4, 8, 16, 32, 64, 128};

        for (int stride : stride_patterns)
        {
            HIP_CHECK(hipMemset(d_output, 0, width * height * sizeof(float)));

            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipEventRecord(start));

            hipLaunchKernelGGL(complexMemoryCoalescing, dim3(gridSize), dim3(blockSize), 0, 0,
                               d_input, d_output, d_pattern, width, height, stride);

            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));

            float gpu_time;
            HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

            size_t bytes_transferred = width * height * sizeof(float) * 2;
            float bandwidth = bytes_transferred / (gpu_time / 1000.0f) / (1024 * 1024 * 1024);

            std::cout << "Stride " << stride << ": " << std::fixed << std::setprecision(1)
                      << bandwidth << " GB/s (" << std::setprecision(3) << gpu_time << " ms)" << std::endl;

            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));
        }
    }
};

int main(int argc, char **argv)
{
    std::cout << "=== Advanced GPU Dynamic Memory Management Benchmarks ===" << std::endl;

    // Initialize HIP
    initializeHIP();
    printHIPDeviceInfo();

    // Parse command line arguments
    int n = (argc > 1) ? std::stoi(argv[1]) : 1024;
    int matrix_size = (argc > 2) ? std::stoi(argv[2]) : 1024;

    std::cout << "\nBenchmark Configuration:" << std::endl;
    std::cout << "Tree nodes: " << n << std::endl;
    std::cout << "Matrix size: " << matrix_size << "x" << matrix_size << std::endl;

    try
    {
        // Run dynamic memory allocation benchmark
        DynamicMemoryBenchmark memory_bench;
        memory_bench.runDynamicTreeTest(n);

        // Run memory coalescing benchmark
        MemoryCoalescingBenchmark coalescing_bench(matrix_size, matrix_size);
        coalescing_bench.runCoalescingTest();

        std::cout << "\n=== All Dynamic Memory Tests Completed Successfully ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
