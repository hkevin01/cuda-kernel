#include <hip/hip_runtime.h>

// HIP kernel for vector addition
__global__ void vectorAddHIP(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// HIP kernel with improved memory access pattern
__global__ void vectorAddHIPOptimized(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process multiple elements per thread for better memory bandwidth
    int stride = blockDim.x * gridDim.x;
    
    while (idx < n) {
        c[idx] = a[idx] + b[idx];
        idx += stride;
    }
}

// HIP kernel using vector types for better memory throughput
__global__ void vectorAddHIPVectorized(const float4* a, const float4* b, float4* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float4 a_vec = a[idx];
        float4 b_vec = b[idx];
        float4 c_vec;
        
        c_vec.x = a_vec.x + b_vec.x;
        c_vec.y = a_vec.y + b_vec.y;
        c_vec.z = a_vec.z + b_vec.z;
        c_vec.w = a_vec.w + b_vec.w;
        
        c[idx] = c_vec;
    }
}
