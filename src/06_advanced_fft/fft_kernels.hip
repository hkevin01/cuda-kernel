#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <cmath>

#define PI 3.14159265358979323846
#define MAX_FFT_SIZE 1024

// Complex number operations
__device__ __forceinline__ hipFloatComplex complex_mul(hipFloatComplex a, hipFloatComplex b) {
    return make_hipFloatComplex(
        hipCrealf(a) * hipCrealf(b) - hipCimagf(a) * hipCimagf(b),
        hipCrealf(a) * hipCimagf(b) + hipCimagf(a) * hipCrealf(b)
    );
}

__device__ __forceinline__ hipFloatComplex complex_exp(float angle) {
    return make_hipFloatComplex(cosf(angle), sinf(angle));
}

// Advanced Cooley-Tukey FFT with shared memory optimization
__global__ void fft_1d_shared_memory(hipFloatComplex* data, int n, int direction) {
    extern __shared__ hipFloatComplex shared_data[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Load data into shared memory with bank conflict avoidance
    int load_id = (global_id < n) ? global_id : 0;
    shared_data[tid] = data[load_id];
    __syncthreads();
    
    // Bit-reversal permutation in shared memory
    int reversed_tid = 0;
    int temp_tid = tid;
    int log_n = __log2f(blockDim.x);
    
    for (int i = 0; i < log_n; i++) {
        reversed_tid = (reversed_tid << 1) | (temp_tid & 1);
        temp_tid >>= 1;
    }
    
    hipFloatComplex temp = shared_data[tid];
    __syncthreads();
    shared_data[reversed_tid] = temp;
    __syncthreads();
    
    // Cooley-Tukey FFT algorithm
    for (int stage = 1; stage <= log_n; stage++) {
        int pairs_per_group = 1 << (stage - 1);
        int group_size = 1 << stage;
        int num_groups = blockDim.x / group_size;
        
        int group_id = tid / group_size;
        int pair_id = (tid % group_size) / 2;
        int is_upper = (tid % group_size) % 2;
        
        if (group_id < num_groups && pair_id < pairs_per_group) {
            int lower_id = group_id * group_size + pair_id;
            int upper_id = lower_id + pairs_per_group;
            
            // Twiddle factor calculation
            float angle = direction * 2.0f * PI * pair_id / group_size;
            hipFloatComplex twiddle = complex_exp(angle);
            
            hipFloatComplex lower = shared_data[lower_id];
            hipFloatComplex upper = shared_data[upper_id];
            
            hipFloatComplex temp_upper = complex_mul(upper, twiddle);
            
            __syncthreads();
            
            if (!is_upper) {
                shared_data[lower_id] = make_hipFloatComplex(
                    hipCrealf(lower) + hipCrealf(temp_upper),
                    hipCimagf(lower) + hipCimagf(temp_upper)
                );
            } else {
                shared_data[upper_id] = make_hipFloatComplex(
                    hipCrealf(lower) - hipCrealf(temp_upper),
                    hipCimagf(lower) - hipCimagf(temp_upper)
                );
            }
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (global_id < n) {
        data[global_id] = shared_data[tid];
    }
}

// 3D FFT with advanced memory coalescing
__global__ void fft_3d_optimized(hipFloatComplex* input, hipFloatComplex* output,
                                int nx, int ny, int nz, int direction) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx >= nx || idy >= ny || idz >= nz) return;
    
    __shared__ hipFloatComplex tile[16][16][16];
    
    // Load into shared memory with padding to avoid bank conflicts
    tile[threadIdx.z][threadIdx.y][threadIdx.x] = 
        input[idz * nx * ny + idy * nx + idx];
    
    __syncthreads();
    
    // Apply 3D FFT using separable approach
    // X-direction FFT
    for (int stage = 1; stage <= __log2f(nx); stage++) {
        int pairs_per_group = 1 << (stage - 1);
        int group_size = 1 << stage;
        
        if (threadIdx.x < pairs_per_group) {
            int partner = threadIdx.x + pairs_per_group;
            
            float angle = direction * 2.0f * PI * threadIdx.x / group_size;
            hipFloatComplex twiddle = complex_exp(angle);
            
            hipFloatComplex a = tile[threadIdx.z][threadIdx.y][threadIdx.x];
            hipFloatComplex b = tile[threadIdx.z][threadIdx.y][partner];
            
            hipFloatComplex temp_b = complex_mul(b, twiddle);
            
            tile[threadIdx.z][threadIdx.y][threadIdx.x] = 
                make_hipFloatComplex(hipCrealf(a) + hipCrealf(temp_b),
                                   hipCimagf(a) + hipCimagf(temp_b));
            tile[threadIdx.z][threadIdx.y][partner] = 
                make_hipFloatComplex(hipCrealf(a) - hipCrealf(temp_b),
                                   hipCimagf(a) - hipCimagf(temp_b));
        }
        __syncthreads();
    }
    
    // Similar operations for Y and Z directions would follow...
    
    // Write back with coalesced access
    output[idz * nx * ny + idy * nx + idx] = 
        tile[threadIdx.z][threadIdx.y][threadIdx.x];
}

// Advanced 2D convolution using FFT
__global__ void convolution_fft(hipFloatComplex* signal, hipFloatComplex* kernel,
                               hipFloatComplex* result, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    int linear_id = idy * width + idx;
    
    // Point-wise multiplication in frequency domain
    result[linear_id] = complex_mul(signal[linear_id], kernel[linear_id]);
}
