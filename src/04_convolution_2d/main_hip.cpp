#include "hip_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>
#include <vector>
#include <iomanip>

// Function declarations from HIP kernel file
extern "C" {
    void launchConvolution2DNaive(float *input, float *filter, float *output,
                                 int width, int height, int filter_size,
                                 dim3 gridSize, dim3 blockSize);
    void launchConvolution2DShared(float *input, float *filter, float *output,
                                  int width, int height, int filter_size,
                                  dim3 gridSize, dim3 blockSize);
    void launchConvolution2DConstant(float *input, float *filter, float *output,
                                    int width, int height, int filter_size,
                                    dim3 gridSize, dim3 blockSize);
    void launchSeparableConvolution(float *input, float *temp, float *output, float *filter,
                                   int width, int height, int filter_size,
                                   dim3 gridSize, dim3 blockSize);
    
    void createGaussianFilter(float *filter, int size, float sigma);
    void createSobelFilter(float *filter_x, float *filter_y);
    void initializeImage(float *image, int width, int height, bool random);
    void printImage(const float *image, int width, int height, const char *name);
    void convolution2DCPU(const float *input, const float *filter, float *output,
                          int width, int height, int filter_size);
    bool verifyConvolution(const float *gpu_result, const float *cpu_result, int size, float tolerance);
}

class ConvolutionBenchmark
{
private:
    int width, height;
    int filter_size;
    float *h_input, *h_output, *h_output_ref, *h_filter, *h_temp;
    float *d_input, *d_output, *d_filter, *d_temp;
    
public:
    ConvolutionBenchmark(int w, int h, int fsize) : width(w), height(h), filter_size(fsize)
    {
        int image_size = width * height;
        int filter_elements = filter_size * filter_size;
        
        // Allocate host memory
        h_input = new float[image_size];
        h_output = new float[image_size];
        h_output_ref = new float[image_size];
        h_filter = new float[filter_elements];
        h_temp = new float[image_size];
        
        // Initialize data
        initializeImage(h_input, width, height, false);
        createGaussianFilter(h_filter, filter_size, 1.0f);
        
        // Allocate device memory
        HIP_CHECK(hipMalloc(&d_input, image_size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_output, image_size * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_filter, filter_elements * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_temp, image_size * sizeof(float)));
        
        // Copy data to device
        HIP_CHECK(hipMemcpy(d_input, h_input, image_size * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_filter, h_filter, filter_elements * sizeof(float), hipMemcpyHostToDevice));
    }
    
    ~ConvolutionBenchmark()
    {
        delete[] h_input;
        delete[] h_output;
        delete[] h_output_ref;
        delete[] h_filter;
        delete[] h_temp;
        HIP_CHECK(hipFree(d_input));
        HIP_CHECK(hipFree(d_output));
        HIP_CHECK(hipFree(d_filter));
        HIP_CHECK(hipFree(d_temp));
    }
    
    void runNaiveConvolution()
    {
        std::cout << "\n=== Naive 2D Convolution ===" << std::endl;
        
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        
        // Reset output
        HIP_CHECK(hipMemset(d_output, 0, width * height * sizeof(float)));
        
        // Create events for timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Warmup
        launchConvolution2DNaive(d_input, d_filter, d_output, width, height, filter_size,
                                gridSize, blockSize);
        
        // Benchmark
        const int iterations = 10;
        HIP_CHECK(hipEventRecord(start));
        
        for (int i = 0; i < iterations; i++)
        {
            launchConvolution2DNaive(d_input, d_filter, d_output, width, height, filter_size,
                                    gridSize, blockSize);
        }
        
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= iterations;
        
        // Copy result back
        HIP_CHECK(hipMemcpy(h_output, d_output, width * height * sizeof(float), hipMemcpyDeviceToHost));
        
        // Calculate performance metrics
        long long operations = (long long)width * height * filter_size * filter_size;
        double gops = operations / (gpu_time / 1000.0) / 1e9;
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
        std::cout << "Block size: " << blockSize.x << "x" << blockSize.y << std::endl;
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Performance: " << gops << " GOPS" << std::endl;
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    void runSharedMemoryConvolution()
    {
        std::cout << "\n=== Shared Memory 2D Convolution ===" << std::endl;
        
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        
        // Reset output
        HIP_CHECK(hipMemset(d_output, 0, width * height * sizeof(float)));
        
        // Create events for timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Warmup
        launchConvolution2DShared(d_input, d_filter, d_output, width, height, filter_size,
                                 gridSize, blockSize);
        
        // Benchmark
        const int iterations = 10;
        HIP_CHECK(hipEventRecord(start));
        
        for (int i = 0; i < iterations; i++)
        {
            launchConvolution2DShared(d_input, d_filter, d_output, width, height, filter_size,
                                     gridSize, blockSize);
        }
        
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= iterations;
        
        // Copy result back
        HIP_CHECK(hipMemcpy(h_output, d_output, width * height * sizeof(float), hipMemcpyDeviceToHost));
        
        // Calculate performance metrics
        long long operations = (long long)width * height * filter_size * filter_size;
        double gops = operations / (gpu_time / 1000.0) / 1e9;
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Performance: " << gops << " GOPS" << std::endl;
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    void runConstantMemoryConvolution()
    {
        std::cout << "\n=== Constant Memory 2D Convolution ===" << std::endl;
        
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        
        // Reset output
        HIP_CHECK(hipMemset(d_output, 0, width * height * sizeof(float)));
        
        // Create events for timing
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        
        // Warmup
        launchConvolution2DConstant(d_input, d_filter, d_output, width, height, filter_size,
                                   gridSize, blockSize);
        
        // Benchmark
        const int iterations = 10;
        HIP_CHECK(hipEventRecord(start));
        
        for (int i = 0; i < iterations; i++)
        {
            launchConvolution2DConstant(d_input, d_filter, d_output, width, height, filter_size,
                                       gridSize, blockSize);
        }
        
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        
        float gpu_time;
        HIP_CHECK(hipEventElapsedTime(&gpu_time, start, stop));
        gpu_time /= iterations;
        
        // Copy result back
        HIP_CHECK(hipMemcpy(h_output, d_output, width * height * sizeof(float), hipMemcpyDeviceToHost));
        
        // Calculate performance metrics
        long long operations = (long long)width * height * filter_size * filter_size;
        double gops = operations / (gpu_time / 1000.0) / 1e9;
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
        std::cout << "Performance: " << gops << " GOPS" << std::endl;
        
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    
    void runCPUReference()
    {
        std::cout << "\n=== CPU Reference Implementation ===" << std::endl;
        
        CPUTimer timer;
        timer.start();
        
        convolution2DCPU(h_input, h_filter, h_output_ref, width, height, filter_size);
        
        timer.stop();
        
        double cpu_time = timer.elapsed();
        long long operations = (long long)width * height * filter_size * filter_size;
        double gops = operations / cpu_time / 1e9;
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "CPU time: " << cpu_time * 1000.0 << " ms" << std::endl;
        std::cout << "Performance: " << gops << " GOPS" << std::endl;
    }
    
    void verifyResults()
    {
        std::cout << "\n=== Result Verification ===" << std::endl;
        
        bool correct = verifyConvolution(h_output, h_output_ref, width * height, 1e-3f);
        
        if (correct)
        {
            std::cout << "✓ Results match CPU reference implementation" << std::endl;
        }
        else
        {
            std::cout << "✗ Results do not match CPU reference implementation" << std::endl;
        }
        
        // Print small sample if images are small
        if (width <= 16 && height <= 16)
        {
            printImage(h_input, width, height, "Input Image");
            printImage(h_output, width, height, "GPU Result");
            printImage(h_output_ref, width, height, "CPU Reference");
        }
    }
};

int main(int argc, char **argv)
{
    std::cout << "=== 2D Convolution GPU Benchmark ===" << std::endl;
    
    // Initialize HIP
    initializeHIP();
    printHIPDeviceInfo();
    
    // Parse command line arguments
    int width = (argc > 1) ? std::stoi(argv[1]) : 1024;
    int height = (argc > 2) ? std::stoi(argv[2]) : 1024;
    int filter_size = (argc > 3) ? std::stoi(argv[3]) : 5;
    
    // Validate filter size
    if (filter_size % 2 == 0 || filter_size < 3 || filter_size > 15)
    {
        std::cerr << "Filter size must be an odd number between 3 and 15" << std::endl;
        return 1;
    }
    
    std::cout << "\nBenchmark Configuration:" << std::endl;
    std::cout << "Image size: " << width << "×" << height << std::endl;
    std::cout << "Filter size: " << filter_size << "×" << filter_size << std::endl;
    std::cout << "Total pixels: " << width * height << std::endl;
    
    try
    {
        ConvolutionBenchmark benchmark(width, height, filter_size);
        
        // Run all implementations
        benchmark.runNaiveConvolution();
        benchmark.runSharedMemoryConvolution();
        benchmark.runConstantMemoryConvolution();
        
        // Run CPU reference for verification (only for smaller images)
        if (width * height <= 1024 * 1024)
        {
            benchmark.runCPUReference();
            benchmark.verifyResults();
        }
        
        std::cout << "\n=== 2D Convolution Benchmark Completed ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
