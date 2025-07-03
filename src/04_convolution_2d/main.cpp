#include "cuda_utils.h"
#include "timer.h"
#include "helper_functions.h"
#include <iostream>

// Basic 2D convolution kernel
__global__ void convolution2DKernel(const float* input, float* output, 
                                    const float* kernel, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        float sum = 0.0f;
        int half_kernel = kernel_size / 2;
        
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int ix = x - half_kernel + kx;
                int iy = y - half_kernel + ky;
                
                // Handle boundaries (zero padding)
                if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    sum += input[iy * width + ix] * kernel[ky * kernel_size + kx];
                }
            }
        }
        
        output[y * width + x] = sum;
    }
}

void convolution2DCPU(const float* input, float* output, const float* kernel, 
                     int width, int height, int kernel_size) {
    int half_kernel = kernel_size / 2;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ix = x - half_kernel + kx;
                    int iy = y - half_kernel + ky;
                    
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        sum += input[iy * width + ix] * kernel[ky * kernel_size + kx];
                    }
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}

int main() {
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
    for (size_t i = 0; i < kernel_elements; ++i) {
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
