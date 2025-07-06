#include "hip_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>

// Function declarations from HIP kernel file
extern "C" {
    void launchMatrixMulNaive(const float *A, const float *B, float *C,
                              int M, int N, int K,
                              dim3 gridSize, dim3 blockSize);
    void launchMatrixMulShared(const float *A, const float *B, float *C,
                               int M, int N, int K,
                               dim3 gridSize, dim3 blockSize);
    void launchMatrixMulCoalesced(const float *A, const float *B, float *C,
                                  int M, int N, int K,
                                  dim3 gridSize, dim3 blockSize);
    
    void initializeMatrix(float *matrix, int rows, int cols, bool random);
    void printMatrix(const float *matrix, int rows, int cols, const char *name);
    void matrixMulCPU(const float *A, const float *B, float *C, int M, int N, int K);
    bool verifyResult(const float *gpuResult, const float *cpuResult, int size, float tolerance);
}

class MatrixMultiplicationBenchmark
{
private:
    int M, N, K;  // Matrix dimensions: A(M×K) * B(K×N) = C(M×N)
    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C;
    
public:
    MatrixMultiplicationBenchmark(int m, int n, int k) : M(m), N(n), K(k)
    {
        // Allocate host memory
        h_A = new float[M * K];
        h_B = new float[K * N];
        h_C = new float[M * N];
        h_C_ref = new float[M * N];
        
        // Initialize matrices
        initializeMatrix(h_A, M, K, true);
        initializeMatrix(h_B, K, N, true);
        initializeMatrix(h_C, M, N, false);
        initializeMatrix(h_C_ref, M, N, false);
        
        // Allocate device memory
        HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));
        
        // Copy data to device
        HIP_CHECK(hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice));
    }
    
    ~MatrixMultiplicationBenchmark()
    {
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_C_ref;
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }
    
    void runNaiveVersion()
    {
        std::cout << "\n=== Naive Matrix Multiplication ===" << std::endl;
        
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                      (M + blockSize.y - 1) / blockSize.y);
        
        // Reset result matrix
        HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
        
        // Create events for timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Warmup
        launchMatrixMulNaive(d_A, d_B, d_C, M, N, K, gridSize, blockSize);
        
        // Benchmark
        HIP_CHECK(hipEventRecord(start));
        const int iterations = 10;
        for (int i = 0; i < iterations; i++)
        {
            launchMatrixMulNaive(d_A, d_B, d_C, M, N, K, gridSize, blockSize);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= iterations;
        
        // Copy result back
        HIP_CHECK(hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
        
        // Calculate performance metrics
        double gflops = (2.0 * M * N * K) / (gpu_time / 1000.0) / 1e9;
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
        std::cout << "Block size: " << blockSize.x << "x" << blockSize.y << std::endl;
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    void runSharedMemoryVersion()
    {
        std::cout << "\n=== Shared Memory Tiled Matrix Multiplication ===" << std::endl;
        
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                      (M + blockSize.y - 1) / blockSize.y);
        
        // Reset result matrix
        HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
        
        // Create events for timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Warmup
        launchMatrixMulShared(d_A, d_B, d_C, M, N, K, gridSize, blockSize);
        
        // Benchmark
        HIP_CHECK(hipEventRecord(start));
        const int iterations = 10;
        for (int i = 0; i < iterations; i++)
        {
            launchMatrixMulShared(d_A, d_B, d_C, M, N, K, gridSize, blockSize);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= iterations;
        
        // Copy result back
        HIP_CHECK(hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
        
        // Calculate performance metrics
        double gflops = (2.0 * M * N * K) / (gpu_time / 1000.0) / 1e9;
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
        std::cout << "Speedup vs naive: " << "varies" << std::endl;
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    void runCoalescedVersion()
    {
        std::cout << "\n=== Memory Coalesced Matrix Multiplication ===" << std::endl;
        
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                      (M + blockSize.y - 1) / blockSize.y);
        
        // Reset result matrix
        HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));
        
        // Create events for timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Warmup
        launchMatrixMulCoalesced(d_A, d_B, d_C, M, N, K, gridSize, blockSize);
        
        // Benchmark
        HIP_CHECK(hipEventRecord(start));
        const int iterations = 10;
        for (int i = 0; i < iterations; i++)
        {
            launchMatrixMulCoalesced(d_A, d_B, d_C, M, N, K, gridSize, blockSize);
        }
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= iterations;
        
        // Copy result back
        HIP_CHECK(hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));
        
        // Calculate performance metrics
        double gflops = (2.0 * M * N * K) / (gpu_time / 1000.0) / 1e9;
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    void runCPUReference()
    {
        std::cout << "\n=== CPU Reference Implementation ===" << std::endl;
        
        CPUTimer timer;
        timer.start();
        
        matrixMulCPU(h_A, h_B, h_C_ref, M, N, K);
        
        timer.stop();
        
        double cpu_time = timer.elapsed();
        double gflops = (2.0 * M * N * K) / cpu_time / 1e9;
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "CPU time: " << cpu_time * 1000.0 << " ms" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    }
    
    void verifyResults()
    {
        std::cout << "\n=== Result Verification ===" << std::endl;
        
        // Use the last GPU result for verification
        bool correct = verifyResult(h_C, h_C_ref, M * N, 1e-3f);
        
        if (correct)
        {
            std::cout << "✓ Results match CPU reference implementation" << std::endl;
        }
        else
        {
            std::cout << "✗ Results do not match CPU reference implementation" << std::endl;
        }
        
        // Print small sample of results
        if (M <= 8 && N <= 8)
        {
            printMatrix(h_A, M, K, "Matrix A");
            printMatrix(h_B, K, N, "Matrix B");
            printMatrix(h_C, M, N, "GPU Result");
            printMatrix(h_C_ref, M, N, "CPU Reference");
        }
    }
};

int main(int argc, char **argv)
{
    std::cout << "=== Matrix Multiplication GPU Benchmark ===" << std::endl;
    
    // Initialize HIP
    initializeHIP();
    printHIPDeviceInfo();
    
    // Parse command line arguments
    int M = (argc > 1) ? std::stoi(argv[1]) : 512;
    int N = (argc > 2) ? std::stoi(argv[2]) : 512;
    int K = (argc > 3) ? std::stoi(argv[3]) : 512;
    
    std::cout << "\nMatrix dimensions:" << std::endl;
    std::cout << "A: " << M << "×" << K << std::endl;
    std::cout << "B: " << K << "×" << N << std::endl;
    std::cout << "C: " << M << "×" << N << std::endl;
    std::cout << "Total operations: " << (2.0 * M * N * K) / 1e9 << " billion" << std::endl;
    
    try
    {
        MatrixMultiplicationBenchmark benchmark(M, N, K);
        
        // Run all implementations
        benchmark.runNaiveVersion();
        benchmark.runSharedMemoryVersion();
        benchmark.runCoalescedVersion();
        
        // Run CPU reference for verification
        if (M * N * K <= 1e8) // Only for smaller matrices to avoid long CPU times
        {
            benchmark.runCPUReference();
            benchmark.verifyResults();
        }
        
        std::cout << "\n=== Matrix Multiplication Benchmark Completed ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
