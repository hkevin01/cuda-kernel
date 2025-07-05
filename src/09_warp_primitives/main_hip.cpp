#include <hip/hip_runtime.h>
#include "hip_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <random>

// Kernel declarations - forward declarations for kernels defined in .hip files
__global__ void advancedTensorOperations(const float *A, const float *B, float *C,
                                         int M, int N, int K, float alpha, float beta);
__global__ void warpPrimitivesShowcase(const int *input, int *output, int *warp_results, int n);
__global__ void multiLevelReduction(const float *input, float *output, int n, int reduction_type);
__global__ void warpMatrixTranspose(const float *input, float *output, int rows, int cols);
__global__ void warpBitonicSort(int *data, int n);
__global__ void warpPrefixSum(const float *input_float, const int *input_int,
                              float *output_float, int *output_int, long long *output_combined, int n);
__global__ void warpOptimizedConvolution(const float *input, const float *filter, float *output,
                                        int width, int height, int filter_size);
__global__ void multiWarpMatrixMul(const float *A, const float *B, float *C, int M, int N, int K);
__global__ void warpStringProcessing(const char *text, const char *pattern, int *matches,
                                    int text_length, int pattern_length);
__global__ void warpGraphTraversal(const int *adjacency_matrix, int *visited, int *distance,
                                  int *queue, int *queue_size, int num_vertices, int current_level);

class WarpPrimitivesBenchmark
{
public:
    void runTensorOperations(int M, int N, int K)
    {
        std::cout << "\n=== Advanced Tensor Operations Test ===" << std::endl;
        std::cout << "Matrix dimensions: " << M << "x" << N << " * " << N << "x" << K << std::endl;

        // Allocate memory
        float *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));

        // Initialize data
        std::vector<float> h_A(M * K), h_B(K * N);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < M * K; i++)
            h_A[i] = dis(gen);
        for (int i = 0; i < K * N; i++)
            h_B[i] = dis(gen);

        HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));

        // Launch configuration
        dim3 blockSize(256);
        dim3 gridSize((M * N + blockSize.x - 1) / blockSize.x, (M + 3) / 4);

        float alpha = 1.0f, beta = 0.0f;

        // Benchmark
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        const int iterations = 100;
        HIP_CHECK(hipEventRecord(start));

        for (int i = 0; i < iterations; i++)
        {
            hipLaunchKernelGGL(advancedTensorOperations, gridSize, blockSize, 0, 0,
                               d_A, d_B, d_C, M, N, K, alpha, beta);
        }

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= iterations;

        // Calculate FLOPS
        long long ops = 2LL * M * N * K; // Multiply-add operations
        float gflops = ops / (gpu_time / 1000.0f) / 1e9f;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

        // Verify result (basic check)
        std::vector<float> h_C(M * N);
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

        float sum = 0.0f;
        for (int i = 0; i < std::min(100, M * N); i++)
        {
            sum += fabsf(h_C[i]);
        }
        std::cout << "Result checksum (first 100 elements): " << sum << std::endl;

        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void runWarpPrimitivesShowcase(int n)
    {
        std::cout << "\n=== Warp Primitives Showcase ===" << std::endl;
        std::cout << "Array size: " << n << " elements" << std::endl;

        // Allocate memory
        int *d_input, *d_output, *d_warp_results;
        HIP_CHECK(hipMalloc(&d_input, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_output, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_warp_results, (n / 32 + 1) * 8 * sizeof(int)));

        // Initialize data
        std::vector<int> h_input(n);
        for (int i = 0; i < n; i++)
        {
            h_input[i] = rand() % 1000;
        }

        HIP_CHECK(hipMemcpy(d_input, h_input.data(), n * sizeof(int), hipMemcpyHostToDevice));

        // Launch kernel
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start));
        hipLaunchKernelGGL(warpPrimitivesShowcase, dim3(gridSize), dim3(blockSize), 0, 0,
                           d_input, d_output, d_warp_results, n);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

        // Copy results
        std::vector<int> h_output(n);
        std::vector<int> h_warp_results((n / 32 + 1) * 8);
        HIP_CHECK(hipMemcpy(h_output.data(), d_output, n * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_warp_results.data(), d_warp_results, (n / 32 + 1) * 8 * sizeof(int), hipMemcpyDeviceToHost));

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;

        // Analyze warp results
        int num_warps = (n + 31) / 32;
        long long total_sum = 0;
        int total_max = INT_MIN, total_min = INT_MAX;

        for (int w = 0; w < num_warps; w++)
        {
            total_sum += h_warp_results[w * 8 + 0];
            total_max = std::max(total_max, h_warp_results[w * 8 + 1]);
            total_min = std::min(total_min, h_warp_results[w * 8 + 2]);
        }

        std::cout << "Warp reduction results:" << std::endl;
        std::cout << "  Total sum: " << total_sum << std::endl;
        std::cout << "  Global max: " << total_max << std::endl;
        std::cout << "  Global min: " << total_min << std::endl;
        std::cout << "  Warps processed: " << num_warps << std::endl;

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_warp_results));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void runMultiLevelReduction(int n)
    {
        std::cout << "\n=== Multi-Level Reduction Test ===" << std::endl;
        std::cout << "Array size: " << n << " elements" << std::endl;

        // Allocate memory
        float *d_input, *d_output;
        int num_blocks = (n + 1023) / 1024;

        HIP_CHECK(hipMalloc(&d_input, n * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_output, num_blocks * sizeof(float)));

        // Initialize data
        std::vector<float> h_input(n);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(1.0f, 10.0f);

        for (int i = 0; i < n; i++)
        {
            h_input[i] = dis(gen);
        }

        HIP_CHECK(hipMemcpy(d_input, h_input.data(), n * sizeof(float), hipMemcpyHostToDevice));

        // Test different reduction types
        std::vector<std::string> reduction_names = {"Sum", "Max", "Min", "Product", "RMS"};

        for (int type = 0; type < 5; type++)
        {
            HIP_CHECK(hipMemset(d_output, 0, num_blocks * sizeof(float)));

            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));

            HIP_CHECK(hipEventRecord(start));
            hipLaunchKernelGGL(multiLevelReduction, dim3(num_blocks), dim3(1024), 0, 0,
                               d_input, d_output, n, type);
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));

            float gpu_time;
            HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

            // Copy and reduce final result on CPU
            std::vector<float> h_output(num_blocks);
            HIP_CHECK(hipMemcpy(h_output.data(), d_output, num_blocks * sizeof(float), hipMemcpyDeviceToHost));

            float final_result = h_output[0];
            for (int i = 1; i < num_blocks; i++)
            {
                switch (type)
                {
                case 0:
                    final_result += h_output[i];
                    break;
                case 1:
                    final_result = std::max(final_result, h_output[i]);
                    break;
                case 2:
                    final_result = std::min(final_result, h_output[i]);
                    break;
                case 3:
                    final_result = std::min(final_result * h_output[i], 1e6f);
                    break;
                case 4:
                    final_result = sqrtf((final_result * final_result + h_output[i] * h_output[i]) / 2.0f);
                    break;
                }
            }

            std::cout << reduction_names[type] << ": " << std::fixed << std::setprecision(3)
                      << final_result << " (" << gpu_time << " ms)" << std::endl;

            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));
        }

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
    }

    void runMatrixTranspose(int rows, int cols)
    {
        std::cout << "\n=== Warp Matrix Transpose Test ===" << std::endl;
        std::cout << "Matrix size: " << rows << "x" << cols << std::endl;

        // Allocate memory
        float *d_input, *d_output;
        HIP_CHECK(hipMalloc(&d_input, rows * cols * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_output, cols * rows * sizeof(float)));

        // Initialize data
        std::vector<float> h_input(rows * cols);
        for (int i = 0; i < rows * cols; i++)
        {
            h_input[i] = static_cast<float>(i);
        }

        HIP_CHECK(hipMemcpy(d_input, h_input.data(), rows * cols * sizeof(float), hipMemcpyHostToDevice));

        // Launch kernel
        int total_warps = ((rows * cols) + 31) / 32;
        int blockSize = 256;
        int gridSize = (total_warps * 32 + blockSize - 1) / blockSize;

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start));
        hipLaunchKernelGGL(warpMatrixTranspose, dim3(gridSize), dim3(blockSize), 0, 0,
                           d_input, d_output, rows, cols);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

        // Verify transpose
        std::vector<float> h_output(cols * rows);
        HIP_CHECK(hipMemcpy(h_output.data(), d_output, cols * rows * sizeof(float), hipMemcpyDeviceToHost));

        bool correct = true;
        for (int i = 0; i < std::min(100, rows); i++)
        {
            for (int j = 0; j < std::min(100, cols); j++)
            {
                if (h_input[i * cols + j] != h_output[j * rows + i])
                {
                    correct = false;
                    break;
                }
            }
            if (!correct)
                break;
        }

        size_t bytes_transferred = rows * cols * sizeof(float) * 2;
        float bandwidth = bytes_transferred / (gpu_time / 1000.0f) / (1024 * 1024 * 1024);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;
        std::cout << "Transpose: " << (correct ? "CORRECT" : "INCORRECT") << std::endl;

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void runWarpSort(int n)
    {
        std::cout << "\n=== Warp Bitonic Sort Test ===" << std::endl;
        std::cout << "Array size: " << n << " elements" << std::endl;

        // Allocate memory
        int *d_data;
        HIP_CHECK(hipMalloc(&d_data, n * sizeof(int)));

        // Initialize data
        std::vector<int> h_data(n);
        for (int i = 0; i < n; i++)
        {
            h_data[i] = rand() % 10000;
        }

        HIP_CHECK(hipMemcpy(d_data, h_data.data(), n * sizeof(int), hipMemcpyHostToDevice));

        // Launch kernel
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start));
        hipLaunchKernelGGL(warpBitonicSort, dim3(gridSize), dim3(blockSize), 0, 0, d_data, n);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

        // Verify sorting
        std::vector<int> h_result(n);
        HIP_CHECK(hipMemcpy(h_result.data(), d_data, n * sizeof(int), hipMemcpyDeviceToHost));

        bool sorted = true;
        for (int i = 1; i < n; i++)
        {
            if (h_result[i] < h_result[i - 1])
            {
                sorted = false;
                break;
            }
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Elements/second: " << n / (gpu_time / 1000.0f) << std::endl;
        std::cout << "Sort result: " << (sorted ? "CORRECT" : "INCORRECT") << std::endl;

        HIP_CHECK(hipFree(d_data));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void runPrefixSum(int n)
    {
        std::cout << "\n=== Warp Prefix Sum Test ===" << std::endl;
        std::cout << "Array size: " << n << " elements" << std::endl;

        // Allocate memory
        float *d_input_float, *d_output_float;
        int *d_input_int, *d_output_int;
        long long *d_output_combined;

        HIP_CHECK(hipMalloc(&d_input_float, n * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_input_int, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_output_float, n * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_output_int, n * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_output_combined, n * sizeof(long long)));

        // Initialize data
        std::vector<float> h_input_float(n);
        std::vector<int> h_input_int(n);

        for (int i = 0; i < n; i++)
        {
            h_input_float[i] = static_cast<float>(i % 100) / 10.0f;
            h_input_int[i] = i % 50;
        }

        HIP_CHECK(hipMemcpy(d_input_float, h_input_float.data(), n * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_input_int, h_input_int.data(), n * sizeof(int), hipMemcpyHostToDevice));

        // Launch kernel
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start));
        hipLaunchKernelGGL(warpPrefixSum, dim3(gridSize), dim3(blockSize), 0, 0,
                           d_input_float, d_input_int, d_output_float, d_output_int, d_output_combined, n);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

        // Verify results
        std::vector<float> h_output_float(n);
        std::vector<int> h_output_int(n);
        std::vector<long long> h_output_combined(n);

        HIP_CHECK(hipMemcpy(h_output_float.data(), d_output_float, n * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_output_int.data(), d_output_int, n * sizeof(int), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_output_combined.data(), d_output_combined, n * sizeof(long long), hipMemcpyDeviceToHost));

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Sample results (first 10):" << std::endl;
        for (int i = 0; i < std::min(10, n); i++)
        {
            std::cout << "  [" << i << "] float: " << h_output_float[i]
                      << ", int: " << h_output_int[i]
                      << ", combined: " << h_output_combined[i] << std::endl;
        }

        HIP_CHECK(hipFree(d_input_float));
        HIP_CHECK(hipFree(d_input_int));
        HIP_CHECK(hipFree(d_output_float));
        HIP_CHECK(hipFree(d_output_int));
        HIP_CHECK(hipFree(d_output_combined));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void runOptimizedConvolution(int width, int height)
    {
        std::cout << "\n=== Warp Optimized Convolution Test ===" << std::endl;
        std::cout << "Image size: " << width << "x" << height << std::endl;

        const int filter_size = 3;

        // Allocate memory
        float *d_input, *d_output, *d_filter;
        HIP_CHECK(hipMalloc(&d_input, width * height * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_output, width * height * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_filter, filter_size * filter_size * sizeof(float)));

        // Initialize data
        std::vector<float> h_input(width * height);
        std::vector<float> h_filter = {-1, -1, -1, -1, 8, -1, -1, -1, -1}; // Edge detection

        for (int i = 0; i < width * height; i++)
        {
            h_input[i] = sin(i * 0.01f) + cos(i * 0.02f);
        }

        HIP_CHECK(hipMemcpy(d_input, h_input.data(), width * height * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_filter, h_filter.data(), filter_size * filter_size * sizeof(float), hipMemcpyHostToDevice));

        // Launch kernel
        int total_warps = ((width * height) + 31) / 32;
        int blockSize = 256;
        int gridSize = (total_warps * 32 + blockSize - 1) / blockSize;

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start));
        hipLaunchKernelGGL(warpOptimizedConvolution, dim3(gridSize), dim3(blockSize), 0, 0,
                           d_input, d_filter, d_output, width, height, filter_size);
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));

        // Calculate performance metrics
        long long operations = (long long)width * height * filter_size * filter_size;
        float gops = operations / (gpu_time / 1000.0f) / 1e9f;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Performance: " << gops << " GOPS" << std::endl;

        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_filter));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    void runMultiWarpMatrixMul(int M, int N, int K)
    {
        std::cout << "\n=== Multi-Warp Matrix Multiplication Test ===" << std::endl;
        std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << std::endl;

        // Allocate memory
        float *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));

        // Initialize data
        std::vector<float> h_A(M * K), h_B(K * N);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < M * K; i++)
            h_A[i] = dis(gen);
        for (int i = 0; i < K * N; i++)
            h_B[i] = dis(gen);

        HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));

        // Launch configuration for multi-warp approach
        dim3 blockSize(128); // 4 warps per block
        dim3 gridSize((N + 127) / 128, (M + 127) / 128);

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        const int iterations = 50;
        HIP_CHECK(hipEventRecord(start));

        for (int i = 0; i < iterations; i++)
        {
            hipLaunchKernelGGL(multiWarpMatrixMul, gridSize, blockSize, 0, 0,
                               d_A, d_B, d_C, M, N, K);
        }

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= iterations;

        // Calculate FLOPS
        long long ops = 2LL * M * N * K;
        float gflops = ops / (gpu_time / 1000.0f) / 1e9f;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
};

int main(int argc, char **argv)
{
    std::cout << "=== Advanced Warp Primitives and Tensor Operations Benchmarks ===" << std::endl;

    // Initialize HIP
    initializeHIP();
    printHIPDeviceInfo();

    // Parse command line arguments
    int matrix_size = (argc > 1) ? std::stoi(argv[1]) : 512;
    int array_size = (argc > 2) ? std::stoi(argv[2]) : 1024 * 1024;

    std::cout << "\nBenchmark Configuration:" << std::endl;
    std::cout << "Matrix size: " << matrix_size << "x" << matrix_size << std::endl;
    std::cout << "Array size: " << array_size << " elements" << std::endl;

    try
    {
        WarpPrimitivesBenchmark benchmark;

        // Run tensor operations
        benchmark.runTensorOperations(matrix_size, matrix_size, matrix_size);

        // Run warp primitives showcase
        benchmark.runWarpPrimitivesShowcase(array_size);

        // Run multi-level reduction
        benchmark.runMultiLevelReduction(array_size);

        // Run matrix transpose
        benchmark.runMatrixTranspose(matrix_size, matrix_size);

        // Run warp sort (limited to warp-sized chunks)
        benchmark.runWarpSort(std::min(array_size, 32 * 1024));

        // Run prefix sum operations
        benchmark.runPrefixSum(array_size);

        // Run optimized convolution
        int conv_size = std::min(matrix_size, 512);
        benchmark.runOptimizedConvolution(conv_size, conv_size);

        // Run multi-warp matrix multiplication
        int mul_size = std::min(matrix_size, 256);
        benchmark.runMultiWarpMatrixMul(mul_size, mul_size, mul_size);

        std::cout << "\n=== All Warp Primitives Tests Completed Successfully ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
