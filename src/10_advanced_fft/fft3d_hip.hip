#include <hip/hip_runtime.h>
#include <cmath>

#define PI 3.14159265358979323846f
#define MAX_FFT_SIZE 1024

// Complex number structure optimized for GPU
struct Complex {
    float real, imag;
    
    __device__ __host__ Complex() : real(0.0f), imag(0.0f) {}
    __device__ __host__ Complex(float r, float i) : real(r), imag(i) {}
    
    __device__ __forceinline__ Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    __device__ __forceinline__ Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    __device__ __forceinline__ Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
                      real * other.imag + imag * other.real);
    }
    
    __device__ __forceinline__ float magnitude() const {
        return sqrtf(real * real + imag * imag);
    }
};

// Optimized twiddle factor computation with caching
__device__ Complex computeTwiddle(int k, int N, int direction) {
    float angle = direction * 2.0f * PI * k / N;
    return Complex(cosf(angle), sinf(angle));
}

// Advanced 1D FFT kernel with shared memory optimization
__global__ void fft1D_optimized(
    Complex* data,
    int n,
    int direction,
    int stride
) {
    extern __shared__ Complex shared_data[];
    
    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    
    // Load data into shared memory with stride pattern
    if (global_id < n) {
        shared_data[tid] = data[global_id * stride];
    } else {
        shared_data[tid] = Complex(0.0f, 0.0f);
    }
    __syncthreads();
    
    // Bit-reversal permutation
    int log_n = __log2f(blockDim.x);
    int reversed_tid = 0;
    int temp_tid = tid;
    
    for (int i = 0; i < log_n; i++) {
        reversed_tid = (reversed_tid << 1) | (temp_tid & 1);
        temp_tid >>= 1;
    }
    
    Complex temp = shared_data[tid];
    __syncthreads();
    if (reversed_tid < blockDim.x) {
        shared_data[reversed_tid] = temp;
    }
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
            
            // Optimized twiddle factor computation
            Complex twiddle = computeTwiddle(pair_id, group_size, direction);
            
            Complex lower = shared_data[lower_id];
            Complex upper = shared_data[upper_id];
            Complex temp_upper = upper * twiddle;
            
            __syncthreads();
            
            if (!is_upper) {
                shared_data[lower_id] = lower + temp_upper;
            } else {
                shared_data[upper_id] = lower - temp_upper;
            }
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (global_id < n) {
        data[global_id * stride] = shared_data[tid];
    }
}

// 3D FFT with advanced memory access optimization
__global__ void fft3D_separable(
    Complex* data,
    Complex* temp_data,
    int nx, int ny, int nz,
    int direction,
    int axis  // 0=x, 1=y, 2=z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx >= nx || idy >= ny || idz >= nz) return;
    
    __shared__ Complex tile[16][16][16];  // 3D shared memory tile
    
    // Load data with optimal access pattern
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int local_z = threadIdx.z;
    
    if (local_x < 16 && local_y < 16 && local_z < 16) {
        int global_idx = idz * nx * ny + idy * nx + idx;
        tile[local_z][local_y][local_x] = data[global_idx];
    }
    __syncthreads();
    
    // Perform 1D FFT along specified axis
    Complex line_data[16];
    int line_size;
    
    switch (axis) {
        case 0: // X-direction
            line_size = min(16, nx - (blockIdx.x * 16));
            for (int i = 0; i < line_size; i++) {
                line_data[i] = tile[local_z][local_y][i];
            }
            break;
            
        case 1: // Y-direction
            line_size = min(16, ny - (blockIdx.y * 16));
            for (int i = 0; i < line_size; i++) {
                line_data[i] = tile[local_z][i][local_x];
            }
            break;
            
        case 2: // Z-direction
            line_size = min(16, nz - (blockIdx.z * 16));
            for (int i = 0; i < line_size; i++) {
                line_data[i] = tile[i][local_y][local_x];
            }
            break;
    }
    
    // Simplified 1D FFT for demonstration
    // In practice, would use optimized radix-2/4/8 algorithms
    for (int stage = 1; stage < line_size; stage *= 2) {
        for (int i = 0; i < line_size; i += stage * 2) {
            if (local_x < stage && i + local_x + stage < line_size) {
                Complex twiddle = computeTwiddle(local_x, stage * 2, direction);
                
                Complex lower = line_data[i + local_x];
                Complex upper = line_data[i + local_x + stage];
                Complex temp_upper = upper * twiddle;
                
                line_data[i + local_x] = lower + temp_upper;
                line_data[i + local_x + stage] = lower - temp_upper;
            }
        }
        __syncthreads();
    }
    
    // Store back to shared memory
    switch (axis) {
        case 0:
            for (int i = 0; i < line_size; i++) {
                tile[local_z][local_y][i] = line_data[i];
            }
            break;
        case 1:
            for (int i = 0; i < line_size; i++) {
                tile[local_z][i][local_x] = line_data[i];
            }
            break;
        case 2:
            for (int i = 0; i < line_size; i++) {
                tile[i][local_y][local_x] = line_data[i];
            }
            break;
    }
    __syncthreads();
    
    // Write back to global memory with coalesced access
    if (local_x < 16 && local_y < 16 && local_z < 16) {
        int global_idx = idz * nx * ny + idy * nx + idx;
        temp_data[global_idx] = tile[local_z][local_y][local_x];
    }
}

// Advanced convolution using FFT
__global__ void convolution_fft(
    Complex* signal_fft,
    Complex* kernel_fft,
    Complex* result_fft,
    int nx, int ny, int nz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx >= nx || idy >= ny || idz >= nz) return;
    
    int linear_id = idz * nx * ny + idy * nx + idx;
    
    // Point-wise multiplication in frequency domain
    result_fft[linear_id] = signal_fft[linear_id] * kernel_fft[linear_id];
}

// Memory-intensive 3D filtering operation
__global__ void advanced_3d_filter(
    Complex* input,
    Complex* output,
    float* filter_coeffs,
    int nx, int ny, int nz,
    int filter_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx >= nx || idy >= ny || idz >= nz) return;
    
    __shared__ Complex shared_input[8][8][8];
    __shared__ float shared_filter[64];  // Assuming max filter size 4x4x4
    
    // Collaborative loading of filter coefficients
    int filter_elements = filter_size * filter_size * filter_size;
    for (int i = threadIdx.x; i < filter_elements; i += blockDim.x) {
        if (i < 64) {
            shared_filter[i] = filter_coeffs[i];
        }
    }
    
    // Load input data into shared memory
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int local_z = threadIdx.z;
    
    if (local_x < 8 && local_y < 8 && local_z < 8) {
        int global_idx = idz * nx * ny + idy * nx + idx;
        shared_input[local_z][local_y][local_x] = input[global_idx];
    }
    __syncthreads();
    
    // Apply 3D convolution filter
    Complex result(0.0f, 0.0f);
    int half_filter = filter_size / 2;
    
    for (int fz = -half_filter; fz <= half_filter; fz++) {
        for (int fy = -half_filter; fy <= half_filter; fy++) {
            for (int fx = -half_filter; fx <= half_filter; fx++) {
                int sample_x = local_x + fx;
                int sample_y = local_y + fy;
                int sample_z = local_z + fz;
                
                // Boundary checks
                if (sample_x >= 0 && sample_x < 8 && 
                    sample_y >= 0 && sample_y < 8 && 
                    sample_z >= 0 && sample_z < 8) {
                    
                    int filter_idx = (fz + half_filter) * filter_size * filter_size + 
                                   (fy + half_filter) * filter_size + 
                                   (fx + half_filter);
                    
                    Complex sample = shared_input[sample_z][sample_y][sample_x];
                    float coeff = shared_filter[filter_idx];
                    
                    result = result + Complex(sample.real * coeff, sample.imag * coeff);
                }
            }
        }
    }
    
    // Write result
    int global_idx = idz * nx * ny + idy * nx + idx;
    output[global_idx] = result;
}
