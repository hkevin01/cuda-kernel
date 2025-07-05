#ifndef REDUCTION_H
#define REDUCTION_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
// Use gpu_utils.h for platform-agnostic GPU code

// Different reduction kernel implementations for performance comparison
__global__ void reduce_naive(float *input, float *output, int n);
__global__ void reduce_optimized(float *input, float *output, int n);
__global__ void reduce_warp_optimized(float *input, float *output, int n);
__global__ void reduce_coop_groups(float *input, float *output, int n);

// Host functions for reduction operations
float reduce_cpu(const float *data, int n);
float reduce_gpu(const float *data, int n, int kernel_type = 1);

// Utility functions
void generate_random_data(float *data, int n);
bool verify_reduction(float cpu_result, float gpu_result, float tolerance = 1e-3);

#endif // REDUCTION_H
