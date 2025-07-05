#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
// Use gpu_utils.h for platform-agnostic GPU code

// Different convolution kernel implementations
__global__ void convolution_naive(const float *input, const float *kernel,
                                  float *output, int width, int height,
                                  int kernel_size);

__global__ void convolution_shared_memory(const float *input, const float *kernel,
                                          float *output, int width, int height,
                                          int kernel_size);

__global__ void convolution_constant_memory(const float *input, float *output,
                                            int width, int height, int kernel_size);

__global__ void convolution_separable(const float *input, const float *kernel_x,
                                      const float *kernel_y, float *temp, float *output,
                                      int width, int height, int kernel_size);

__global__ void convolution_texture(cudaTextureObject_t texObj, const float *kernel,
                                    float *output, int width, int height,
                                    int kernel_size);

// Host functions
void convolution_cpu(const float *input, const float *kernel, float *output,
                     int width, int height, int kernel_size);

float *convolution_gpu(const float *input, const float *kernel, int width,
                       int height, int kernel_size, int method = 1);

// Utility functions
void generate_gaussian_kernel(float *kernel, int size, float sigma);
void generate_test_image(float *image, int width, int height);
bool verify_convolution(const float *cpu_result, const float *gpu_result,
                        int width, int height, float tolerance = 1e-3);

// Constant memory for small kernels (max 15x15)
__constant__ float d_kernel[225];

#endif // CONVOLUTION_H
