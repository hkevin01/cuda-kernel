#include "hip_utils.h"
#include <hip/hip_runtime.h>

// Naive reduction kernel - demonstrates basic concept but has divergent warps
__global__ void reduce_naive(float *input, float *output, int n)
{
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy global memory to shared memory
    extern __shared__ float sdata[];
    if (i < n)
    {
        sdata[tid] = input[i];
    }
    else
    {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Reduction in shared memory - causes warp divergence
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

// Optimized reduction - eliminates warp divergence
__global__ void reduce_optimized(float *input, float *output, int n)
{
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sdata[];
    if (i < n)
    {
        sdata[tid] = input[i];
    }
    else
    {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Optimized reduction - no warp divergence
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

// Unrolled reduction for better performance
__global__ void reduce_unrolled(float *input, float *output, int n)
{
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sdata[];
    if (i < n)
    {
        sdata[tid] = input[i];
    }
    else
    {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Unrolled reduction
    if (blockDim.x >= 512)
    {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockDim.x >= 256)
    {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockDim.x >= 128)
    {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    // Final reduction in warp
    if (tid < 32)
    {
        volatile float *vmem = sdata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

// Warp shuffle reduction (most efficient)
__global__ void reduce_warp_shuffle(float *input, float *output, int n)
{
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < n) ? input[i] : 0.0f;

    // Warp-level reduction using shuffle operations
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffffffffffffULL, val, offset);
    }

    __shared__ float warp_results[32];

    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Store warp results
    if (lane_id == 0)
    {
        warp_results[warp_id] = val;
    }
    __syncthreads();

    // Reduce warp results
    if (warp_id == 0)
    {
        val = (lane_id < (blockDim.x / 32)) ? warp_results[lane_id] : 0.0f;
        
        for (int offset = 16; offset > 0; offset /= 2)
        {
            val += __shfl_down_sync(0xffffffffffffffffULL, val, offset);
        }
        
        if (lane_id == 0)
        {
            output[blockIdx.x] = val;
        }
    }
}

// Multiple elements per thread for better memory bandwidth
__global__ void reduce_multiple_elements(float *input, float *output, int n)
{
    int tid = threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    
    // Grid-stride loop to process multiple elements per thread
    while (i < n)
    {
        sum += input[i];
        i += gridSize;
    }

    extern __shared__ float sdata[];
    sdata[tid] = sum;
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp reduction
    if (tid < 32)
    {
        volatile float *vmem = sdata;
        if (tid + 32 < blockDim.x) vmem[tid] += vmem[tid + 32];
        if (tid + 16 < blockDim.x) vmem[tid] += vmem[tid + 16];
        if (tid + 8 < blockDim.x) vmem[tid] += vmem[tid + 8];
        if (tid + 4 < blockDim.x) vmem[tid] += vmem[tid + 4];
        if (tid + 2 < blockDim.x) vmem[tid] += vmem[tid + 2];
        if (tid + 1 < blockDim.x) vmem[tid] += vmem[tid + 1];
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

// Host function wrappers
extern "C" {

void launchReduceNaive(float *input, float *output, int n, int blockSize, int gridSize)
{
    int sharedMemSize = blockSize * sizeof(float);
    hipLaunchKernelGGL(reduce_naive, dim3(gridSize), dim3(blockSize), sharedMemSize, 0,
                       input, output, n);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchReduceOptimized(float *input, float *output, int n, int blockSize, int gridSize)
{
    int sharedMemSize = blockSize * sizeof(float);
    hipLaunchKernelGGL(reduce_optimized, dim3(gridSize), dim3(blockSize), sharedMemSize, 0,
                       input, output, n);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchReduceUnrolled(float *input, float *output, int n, int blockSize, int gridSize)
{
    int sharedMemSize = blockSize * sizeof(float);
    hipLaunchKernelGGL(reduce_unrolled, dim3(gridSize), dim3(blockSize), sharedMemSize, 0,
                       input, output, n);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchReduceWarpShuffle(float *input, float *output, int n, int blockSize, int gridSize)
{
    hipLaunchKernelGGL(reduce_warp_shuffle, dim3(gridSize), dim3(blockSize), 0, 0,
                       input, output, n);
    HIP_CHECK(hipDeviceSynchronize());
}

void launchReduceMultipleElements(float *input, float *output, int n, int blockSize, int gridSize)
{
    int sharedMemSize = blockSize * sizeof(float);
    hipLaunchKernelGGL(reduce_multiple_elements, dim3(gridSize), dim3(blockSize), sharedMemSize, 0,
                       input, output, n);
    HIP_CHECK(hipDeviceSynchronize());
}

}

// Utility functions - wrapped for C++ linkage
extern "C" {

void initializeArray(float *array, int size, bool random = true)
{
    if (random)
    {
        for (int i = 0; i < size; i++)
        {
            array[i] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    else
    {
        for (int i = 0; i < size; i++)
        {
            array[i] = 1.0f; // For easy verification
        }
    }
}

float cpuReduce(const float *array, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum;
}

bool verifyReduction(float gpu_result, float cpu_result, float tolerance = 1e-3f)
{
    float error = fabsf(gpu_result - cpu_result);
    float relative_error = error / fabsf(cpu_result);
    
    return relative_error < tolerance;
}

} // extern "C"
