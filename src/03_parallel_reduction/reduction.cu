#include "reduction.h"
#include "../common/cuda_utils.h"
#include <cub/cub.cuh>

// Naive reduction kernel - demonstrates basic concept but has divergent warps
__global__ void reduce_naive(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Copy global memory to shared memory
    extern __shared__ float sdata[];
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Reduction in shared memory - causes warp divergence
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Optimized reduction - eliminates warp divergence
__global__ void reduce_optimized(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Sequential addressing to avoid warp divergence
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Highly optimized reduction with unrolling and warp-level primitives
__global__ void reduce_warp_optimized(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    extern __shared__ float sdata[];
    
    // Load two elements per thread to reduce kernel launch overhead
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // Unroll the last warp for better performance
    if (blockDim.x >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    
    // Warp-level reduction using shuffle instructions
    if (tid < 32) {
        volatile float* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Modern reduction using warp shuffle primitives
__global__ void reduce_coop_groups(float* input, float* output, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    extern __shared__ float sdata[];
    
    // Load two elements per thread
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    
    // Reduce within warp using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    // Store warp results in shared memory
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = sum;
    }
    __syncthreads();
    
    // Reduce across warps
    if (tid < blockDim.x / warpSize) {
        sum = sdata[tid];
        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (tid == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// CPU reference implementation
float reduce_cpu(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// GPU reduction wrapper function
float reduce_gpu(const float* data, int n, int kernel_type) {
    float* d_input;
    float* d_output;
    float* h_output;
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    // For warp-optimized kernel, adjust grid size since it processes 2 elements per thread
    if (kernel_type == 2 || kernel_type == 3) {
        grid_size = (n + (block_size * 2) - 1) / (block_size * 2);
    }
    
    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, grid_size * sizeof(float)));
    h_output = new float[grid_size];
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, data, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch appropriate kernel
    int shared_mem_size = block_size * sizeof(float);
    
    switch (kernel_type) {
        case 0:
            reduce_naive<<<grid_size, block_size, shared_mem_size>>>(d_input, d_output, n);
            break;
        case 1:
            reduce_optimized<<<grid_size, block_size, shared_mem_size>>>(d_input, d_output, n);
            break;
        case 2:
            reduce_warp_optimized<<<grid_size, block_size, shared_mem_size>>>(d_input, d_output, n);
            break;
        case 3:
            reduce_coop_groups<<<grid_size, block_size, shared_mem_size>>>(d_input, d_output, n);
            break;
        default:
            reduce_optimized<<<grid_size, block_size, shared_mem_size>>>(d_input, d_output, n);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Final reduction on CPU if multiple blocks
    float final_result = 0.0f;
    for (int i = 0; i < grid_size; i++) {
        final_result += h_output[i];
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    delete[] h_output;
    
    return final_result;
}

// Generate random data for testing
void generate_random_data(float* data, int n) {
    srand(42); // Fixed seed for reproducible results
    for (int i = 0; i < n; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Verify reduction results
bool verify_reduction(float cpu_result, float gpu_result, float tolerance) {
    float diff = fabs(cpu_result - gpu_result);
    float relative_error = diff / fabs(cpu_result);
    return relative_error < tolerance;
}
