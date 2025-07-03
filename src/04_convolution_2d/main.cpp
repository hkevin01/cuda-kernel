#include "convolution.h"
#include "../common/cuda_utils.h"
#include "../common/timer.h"
#include "../common/helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>

void benchmark_convolution_kernel(const std::string &kernel_name, int method,
                                  const float *image, const float *kernel,
                                  int width, int height, int kernel_size,
                                  int iterations = 10)
{
    Timer timer;
    float total_time = 0.0f;

    // Warmup
    float *result = convolution_gpu(image, kernel, width, height, kernel_size, method);
    delete[] result;

    // Benchmark
    timer.start();
    for (int i = 0; i < iterations; i++)
    {
        result = convolution_gpu(image, kernel, width, height, kernel_size, method);
        delete[] result;
    }
    timer.stop();
    total_time = timer.getElapsedTime();

    float avg_time = total_time / iterations;
    size_t bytes_accessed = 2 * width * height * sizeof(float); // Read input + write output
    float bandwidth = bytes_accessed / (avg_time * 1e-3) / 1e9; // GB/s

    // Calculate theoretical operations
    long long ops = (long long)width * height * kernel_size * kernel_size * 2; // multiply-add
    float gflops = ops / (avg_time * 1e-3) / 1e9;

    std::cout << std::setw(18) << kernel_name
              << std::setw(12) << std::fixed << std::setprecision(3) << avg_time << " ms"
              << std::setw(12) << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
              << std::setw(12) << std::fixed << std::setprecision(2) << gflops << " GFLOPS"
              << std::endl;
}

int main(int argc, char **argv)
{
    // Parse command line arguments
    int width = 1024;    // Default image width
    int height = 1024;   // Default image height
    int kernel_size = 5; // Default kernel size

    if (argc > 1)
        width = std::stoi(argv[1]);
    if (argc > 2)
        height = std::stoi(argv[2]);
    if (argc > 3)
        kernel_size = std::stoi(argv[3]);

    // Ensure kernel size is odd
    if (kernel_size % 2 == 0)
        kernel_size++;

    std::cout << "=== CUDA 2D Convolution Benchmark ===" << std::endl;
    std::cout << "Image size: " << width << "x" << height
              << " (" << (width * height * sizeof(float)) / (1024 * 1024) << " MB)" << std::endl;
    std::cout << "Kernel size: " << kernel_size << "x" << kernel_size << std::endl;

    // Initialize CUDA device
    initializeCUDA();
    printDeviceInfo();

    // Allocate and initialize host data
    std::vector<float> h_image(width * height);
    std::vector<float> h_kernel(kernel_size * kernel_size);
    std::vector<float> h_output_cpu(width * height);

    // Generate test data
    generate_test_image(h_image.data(), width, height);
    generate_gaussian_kernel(h_kernel.data(), kernel_size, 1.0f);

    std::cout << "\nGenerated test image with edge patterns and Gaussian blur kernel" << std::endl;

    // CPU reference computation
    Timer timer;
    timer.start();
    convolution_cpu(h_image.data(), h_kernel.data(), h_output_cpu.data(),
                    width, height, kernel_size);
    timer.stop();
    float cpu_time = timer.getElapsedTime();

    std::cout << "\n--- CPU Reference ---" << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(3) << cpu_time << " ms" << std::endl;

    // GPU Benchmarks
    std::cout << "\n--- GPU Kernel Benchmarks ---" << std::endl;
    std::cout << std::setw(18) << "Method"
              << std::setw(12) << "Time (ms)"
              << std::setw(12) << "Bandwidth"
              << std::setw(12) << "Performance"
              << std::endl;
    std::cout << std::string(54, '-') << std::endl;

    // Test different implementations
    benchmark_convolution_kernel("Naive", 0, h_image.data(), h_kernel.data(),
                                 width, height, kernel_size);
    benchmark_convolution_kernel("Shared Memory", 1, h_image.data(), h_kernel.data(),
                                 width, height, kernel_size);

    // Only test constant memory if kernel is small enough
    if (kernel_size * kernel_size <= 225)
    {
        benchmark_convolution_kernel("Constant Memory", 2, h_image.data(), h_kernel.data(),
                                     width, height, kernel_size);
    }

    benchmark_convolution_kernel("Texture Memory", 3, h_image.data(), h_kernel.data(),
                                 width, height, kernel_size);

    // Verify results
    std::cout << "\n--- Verification ---" << std::endl;
    std::vector<std::string> method_names = {"Naive", "Shared Memory", "Constant Memory", "Texture Memory"};

    for (int method = 0; method < 4; method++)
    {
        // Skip constant memory if kernel is too large
        if (method == 2 && kernel_size * kernel_size > 225)
        {
            std::cout << std::setw(15) << method_names[method] << ": SKIPPED (kernel too large)" << std::endl;
            continue;
        }

        float *gpu_result = convolution_gpu(h_image.data(), h_kernel.data(),
                                            width, height, kernel_size, method);
        bool is_correct = verify_convolution(h_output_cpu.data(), gpu_result,
                                             width, height);

        // Calculate max error for reporting
        float max_error = 0.0f;
        for (int i = 0; i < width * height; i++)
        {
            float error = fabsf(h_output_cpu[i] - gpu_result[i]);
            max_error = fmaxf(max_error, error);
        }

        std::cout << std::setw(15) << method_names[method] << ": "
                  << (is_correct ? "PASS" : "FAIL")
                  << " (Max Error: " << std::scientific << std::setprecision(2)
                  << max_error << ")" << std::endl;

        delete[] gpu_result;
    }

    // Performance Analysis
    std::cout << "\n--- Performance Analysis ---" << std::endl;

    Timer bench_timer;

    // Measure naive vs optimized performance
    bench_timer.start();
    for (int i = 0; i < 10; i++)
    {
        float *result = convolution_gpu(h_image.data(), h_kernel.data(),
                                        width, height, kernel_size, 0);
        delete[] result;
    }
    bench_timer.stop();
    float naive_time = bench_timer.getElapsedTime() / 10.0f;

    bench_timer.start();
    for (int i = 0; i < 10; i++)
    {
        float *result = convolution_gpu(h_image.data(), h_kernel.data(),
                                        width, height, kernel_size, 1);
        delete[] result;
    }
    bench_timer.stop();
    float optimized_time = bench_timer.getElapsedTime() / 10.0f;

    float speedup = naive_time / optimized_time;
    std::cout << "Speedup (Shared Memory vs Naive): " << std::fixed << std::setprecision(2)
              << speedup << "x" << std::endl;

    // Memory efficiency analysis
    size_t total_bytes = width * height * sizeof(float) * 2; // Input + output
    float achieved_bandwidth = total_bytes / (optimized_time * 1e-3) / 1e9;
    float theoretical_bandwidth = getTheoreticalBandwidth();
    float efficiency = (achieved_bandwidth / theoretical_bandwidth) * 100.0f;

    std::cout << "Memory Bandwidth Efficiency: " << std::fixed << std::setprecision(1)
              << efficiency << "%" << std::endl;

    // Computational intensity analysis
    long long total_ops = (long long)width * height * kernel_size * kernel_size * 2;
    float arithmetic_intensity = (float)total_ops / total_bytes;

    std::cout << "Arithmetic Intensity: " << std::fixed << std::setprecision(2)
              << arithmetic_intensity << " FLOPS/byte" << std::endl;

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "✓ Implemented multiple convolution optimization strategies" << std::endl;
    std::cout << "✓ Demonstrated shared memory tiling for data reuse" << std::endl;
    std::cout << "✓ Utilized constant memory for small kernels" << std::endl;
    std::cout << "✓ Leveraged texture memory for automatic caching" << std::endl;
    std::cout << "✓ Achieved " << std::fixed << std::setprecision(1) << speedup
              << "x speedup with optimizations" << std::endl;

    // Application insights
    std::cout << "\n--- Industry Applications ---" << std::endl;
    std::cout << "• Computer Vision: Edge detection, noise reduction, feature extraction" << std::endl;
    std::cout << "• Deep Learning: Convolutional neural network layers" << std::endl;
    std::cout << "• Image Processing: Gaussian blur, sharpening, emboss effects" << std::endl;
    std::cout << "• Signal Processing: Digital filters, pattern recognition" << std::endl;

    return 0;
}

void convolution2DCPU(const float *input, float *output, const float *kernel,
                      int width, int height, int kernel_size)
{
    int half_kernel = kernel_size / 2;

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float sum = 0.0f;

            for (int ky = 0; ky < kernel_size; ++ky)
            {
                for (int kx = 0; kx < kernel_size; ++kx)
                {
                    int ix = x - half_kernel + kx;
                    int iy = y - half_kernel + ky;

                    if (ix >= 0 && ix < width && iy >= 0 && iy < height)
                    {
                        sum += input[iy * width + ix] * kernel[ky * kernel_size + kx];
                    }
                }
            }

            output[y * width + x] = sum;
        }
    }
}

int main()
{
    std::cout << "=== CUDA 2D Convolution Example ===" << std::endl;
    std::cout << "Note: This is a basic implementation. Optimized version coming soon!" << std::endl;

    setOptimalDevice();
    printDeviceInfo();

    const int width = 1024;
    const int height = 1024;
    const int kernel_size = 5;
    const size_t image_size = width * height;
    const size_t kernel_elements = kernel_size * kernel_size;

    std::cout << "Image size: " << width << "x" << height << std::endl;
    std::cout << "Kernel size: " << kernel_size << "x" << kernel_size << std::endl;

    // Allocate host memory
    float *h_input, *h_output_cpu, *h_output_gpu, *h_kernel;
    allocateHostMemory(&h_input, image_size);
    allocateHostMemory(&h_output_cpu, image_size);
    allocateHostMemory(&h_output_gpu, image_size);
    allocateHostMemory(&h_kernel, kernel_elements);

    // Initialize data
    generateRandomFloats(h_input, image_size, 0.0f, 1.0f);
    generateRandomFloats(h_kernel, kernel_elements, -0.1f, 0.1f);

    // Normalize kernel
    float kernel_sum = sumArray(h_kernel, kernel_elements);
    for (size_t i = 0; i < kernel_elements; ++i)
    {
        h_kernel[i] /= kernel_sum;
    }

    // CPU convolution
    CPUTimer cpu_timer;
    cpu_timer.start();
    convolution2DCPU(h_input, h_output_cpu, h_kernel, width, height, kernel_size);
    double cpu_time = cpu_timer.stop();

    // GPU convolution
    float *d_input, *d_output, *d_kernel;
    allocateDeviceMemory(&d_input, image_size);
    allocateDeviceMemory(&d_output, image_size);
    allocateDeviceMemory(&d_kernel, kernel_elements);

    copyHostToDevice(d_input, h_input, image_size);
    copyHostToDevice(d_kernel, h_kernel, kernel_elements);

    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);

    CudaTimer gpu_timer;
    gpu_timer.start();

    convolution2DKernel<<<grid_size, block_size>>>(d_input, d_output, d_kernel, width, height, kernel_size);
    CUDA_CHECK_KERNEL();

    float gpu_time = gpu_timer.stop();

    copyDeviceToHost(h_output_gpu, d_output, image_size);

    // Verify results
    bool correct = verifyResults(h_output_cpu, h_output_gpu, image_size, 1e-4f);

    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time / gpu_time) << "x" << std::endl;
    std::cout << "Verification: " << (correct ? "PASSED" : "FAILED") << std::endl;

    // Performance analysis
    size_t operations = image_size * kernel_elements * 2; // multiply and add
    double gpu_gflops = calculateGFlops(operations, gpu_time);
    std::cout << "Performance: " << gpu_gflops << " GFLOPS" << std::endl;

    // Cleanup
    freeHostMemory(h_input);
    freeHostMemory(h_output_cpu);
    freeHostMemory(h_output_gpu);
    freeHostMemory(h_kernel);
    freeDeviceMemory(d_input);
    freeDeviceMemory(d_output);
    freeDeviceMemory(d_kernel);

    std::cout << "=== 2D Convolution Complete ===" << std::endl;
    return 0;
}
