#ifdef USE_CUDA
#include "cuda_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <curand_kernel.h>
#endif
#include <iostream>
#include <cmath>

// Use gpu_utils.h for platform-agnostic GPU code

// Monte Carlo Pi estimation kernel
__global__ void monteCarloPiKernel(float *results, int n_samples_per_thread, unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize random state
    curandState state;
    curand_init(seed, tid, 0, &state);

    int inside_circle = 0;

    for (int i = 0; i < n_samples_per_thread; ++i)
    {
        float x = curand_uniform(&state) * 2.0f - 1.0f; // Random between -1 and 1
        float y = curand_uniform(&state) * 2.0f - 1.0f;

        if (x * x + y * y <= 1.0f)
        {
            inside_circle++;
        }
    }

    results[tid] = (float)inside_circle;
}

// CPU Monte Carlo Pi estimation
float monteCarloPiCPU(long long total_samples)
{
    int inside_circle = 0;

    for (long long i = 0; i < total_samples; ++i)
    {
        float x = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float y = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

        if (x * x + y * y <= 1.0f)
        {
            inside_circle++;
        }
    }

    return 4.0f * inside_circle / total_samples;
}

int main()
{
    std::cout << "=== CUDA Monte Carlo Simulation Example ===" << std::endl;
    std::cout << "Estimating Pi using Monte Carlo method" << std::endl;

    setOptimalDevice();
    printDeviceInfo();

    const long long total_samples = 100000000LL; // 100M samples
    const int block_size = 256;
    const int grid_size = 1024;
    const int total_threads = block_size * grid_size;
    const int samples_per_thread = total_samples / total_threads;

    std::cout << "Total samples: " << formatNumber(total_samples) << std::endl;
    std::cout << "Threads: " << total_threads << std::endl;
    std::cout << "Samples per thread: " << samples_per_thread << std::endl;

    // CPU estimation
    std::cout << "\nRunning CPU Monte Carlo..." << std::endl;
    srand(42); // Fixed seed for reproducibility

    CPUTimer cpu_timer;
    cpu_timer.start();
    float pi_cpu = monteCarloPiCPU(1000000); // Smaller sample for CPU
    double cpu_time = cpu_timer.stop();

    std::cout << "CPU Pi estimate: " << pi_cpu << " (1M samples)" << std::endl;

    // GPU estimation
    std::cout << "Running GPU Monte Carlo..." << std::endl;

    float *d_results, *h_results;
    allocateHostMemory(&h_results, total_threads);
    allocateDeviceMemory(&d_results, total_threads);

    CudaTimer gpu_timer;
    gpu_timer.start();

    monteCarloPiKernel<<<grid_size, block_size>>>(d_results, samples_per_thread, 42ULL);
    CUDA_CHECK_KERNEL();

    float gpu_time = gpu_timer.stop();

    // Copy results and compute final estimate
    copyDeviceToHost(h_results, d_results, total_threads);

    float total_inside = sumArray(h_results, total_threads);
    float pi_gpu = 4.0f * total_inside / (total_threads * samples_per_thread);

    // Results
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "True value of Pi: " << M_PI << std::endl;
    std::cout << "CPU estimate: " << pi_cpu << " (error: " << std::abs(pi_cpu - M_PI) << ")" << std::endl;
    std::cout << "GPU estimate: " << pi_gpu << " (error: " << std::abs(pi_gpu - M_PI) << ")" << std::endl;

    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "CPU time (1M samples): " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time (" << formatNumber(total_samples) << " samples): " << gpu_time << " ms" << std::endl;

    // Calculate effective speedup (accounting for different sample counts)
    double normalized_cpu_time = cpu_time * (total_samples / 1000000.0);
    std::cout << "Effective speedup: " << (normalized_cpu_time / gpu_time) << "x" << std::endl;

    // Samples per second
    double samples_per_sec = total_samples / (gpu_time / 1000.0);
    std::cout << "GPU throughput: " << formatNumber(samples_per_sec) << " samples/sec" << std::endl;

    // Statistical analysis
    std::cout << "\n=== Statistical Analysis ===" << std::endl;
    double theoretical_std = sqrt(M_PI * (4.0 - M_PI) / total_samples);
    double actual_error = std::abs(pi_gpu - M_PI);
    std::cout << "Theoretical std dev: " << theoretical_std << std::endl;
    std::cout << "Actual error: " << actual_error << std::endl;
    std::cout << "Error within 3σ: " << (actual_error < 3 * theoretical_std ? "YES" : "NO") << std::endl;

    // Cleanup
    freeHostMemory(h_results);
    freeDeviceMemory(d_results);

    std::cout << "\n=== Monte Carlo Simulation Complete ===" << std::endl;
    return 0;
}
