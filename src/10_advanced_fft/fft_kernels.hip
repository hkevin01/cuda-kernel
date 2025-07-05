#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hip/hip_runtime.h>
#include <cmath>

#define PI 3.14159265358979323846f
#define MAX_FFT_SIZE 1024

// Complex number operations optimized for GPU
struct Complex {
    float real, imag;
    
    __device__ __host__ Complex(float r = 0.0f, float i = 0.0f) : real(r), imag(i) {}
    
    __device__ __host__ Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }
    
    __device__ __host__ Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }
    
    __device__ __host__ Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
                      real * other.imag + imag * other.real);
    }
    
    __device__ __host__ float magnitude() const {
        return sqrtf(real * real + imag * imag);
    }
    
    __device__ __host__ float phase() const {
        return atan2f(imag, real);
    }
};

__device__ Complex exp_complex(float angle) {
    return Complex(cosf(angle), sinf(angle));
}

// Advanced 1D FFT with shared memory optimization and bank conflict avoidance
__global__ void fft_1d_radix2_shared(
    Complex* data,
    int n,
    int direction,
    int stride = 1
) {
    extern __shared__ Complex shared_data[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_id = bid * blockDim.x + tid;
    
    // Load data into shared memory with bank conflict avoidance
    int load_id = (global_id * stride < n) ? global_id * stride : 0;
    shared_data[tid] = data[load_id];
    __syncthreads();
    
    // Bit-reversal permutation
    int reversed_tid = 0;
    int temp_tid = tid;
    int log_n = 0;
    int temp_n = blockDim.x;
    while (temp_n > 1) {
        log_n++;
        temp_n >>= 1;
    }
    
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
    
    // Cooley-Tukey FFT algorithm with optimized twiddle factor computation
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
            
            // Optimized twiddle factor calculation
            float angle = direction * 2.0f * PI * pair_id / group_size;
            Complex twiddle = exp_complex(angle);
            
            Complex lower = shared_data[lower_id];
            Complex upper = shared_data[upper_id];
            
            Complex temp_upper = upper * twiddle;
            
            __syncthreads();
            
            if (!is_upper && lower_id < blockDim.x) {
                shared_data[lower_id] = lower + temp_upper;
            } else if (is_upper && upper_id < blockDim.x) {
                shared_data[upper_id] = lower - temp_upper;
            }
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (global_id * stride < n) {
        data[global_id * stride] = shared_data[tid];
    }
}

// Advanced 3D FFT with memory coalescing and cache optimization
__global__ void fft_3d_optimized(
    Complex* input,
    Complex* output,
    int nx, int ny, int nz,
    int direction,
    int phase // 0=x, 1=y, 2=z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (idx >= nx || idy >= ny || idz >= nz) return;
    
    __shared__ Complex tile[8][8][8];
    __shared__ Complex twiddles[64]; // Precomputed twiddle factors
    
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    int local_z = threadIdx.z;
    
    // Load data into shared memory with optimal memory access pattern
    if (local_x < 8 && local_y < 8 && local_z < 8) {
        int src_idx = idz * nx * ny + idy * nx + idx;
        tile[local_z][local_y][local_x] = input[src_idx];
    }
    
    // Precompute twiddle factors for current phase
    if (threadIdx.x < 64 && threadIdx.y == 0 && threadIdx.z == 0) {
        int size = (phase == 0) ? nx : (phase == 1) ? ny : nz;
        float angle = direction * 2.0f * PI * threadIdx.x / size;
        twiddles[threadIdx.x] = exp_complex(angle);
    }
    
    __syncthreads();
    
    // Perform 1D FFT along the specified dimension
    if (phase == 0) { // X-direction FFT
        for (int stage = 1; stage <= 3; stage++) { // Assuming 8x8x8 tile
            int pairs_per_group = 1 << (stage - 1);
            int group_size = 1 << stage;
            
            if (local_x < pairs_per_group) {
                int partner = local_x + pairs_per_group;
                
                Complex a = tile[local_z][local_y][local_x];
                Complex b = tile[local_z][local_y][partner];
                
                int twiddle_idx = local_x * 8 / group_size;
                Complex twiddle = twiddles[twiddle_idx];
                Complex temp_b = b * twiddle;
                
                tile[local_z][local_y][local_x] = a + temp_b;
                if (partner < 8) {
                    tile[local_z][local_y][partner] = a - temp_b;
                }
            }
            __syncthreads();
        }
    } else if (phase == 1) { // Y-direction FFT
        for (int stage = 1; stage <= 3; stage++) {
            int pairs_per_group = 1 << (stage - 1);
            int group_size = 1 << stage;
            
            if (local_y < pairs_per_group) {
                int partner = local_y + pairs_per_group;
                
                Complex a = tile[local_z][local_y][local_x];
                Complex b = tile[local_z][partner][local_x];
                
                int twiddle_idx = local_y * 8 / group_size;
                Complex twiddle = twiddles[twiddle_idx];
                Complex temp_b = b * twiddle;
                
                tile[local_z][local_y][local_x] = a + temp_b;
                if (partner < 8) {
                    tile[local_z][partner][local_x] = a - temp_b;
                }
            }
            __syncthreads();
        }
    } else { // Z-direction FFT
        for (int stage = 1; stage <= 3; stage++) {
            int pairs_per_group = 1 << (stage - 1);
            int group_size = 1 << stage;
            
            if (local_z < pairs_per_group) {
                int partner = local_z + pairs_per_group;
                
                Complex a = tile[local_z][local_y][local_x];
                Complex b = tile[partner][local_y][local_x];
                
                int twiddle_idx = local_z * 8 / group_size;
                Complex twiddle = twiddles[twiddle_idx];
                Complex temp_b = b * twiddle;
                
                tile[local_z][local_y][local_x] = a + temp_b;
                if (partner < 8) {
                    tile[partner][local_y][local_x] = a - temp_b;
                }
            }
            __syncthreads();
        }
    }
    
    // Write back with coalesced access pattern
    if (local_x < 8 && local_y < 8 && local_z < 8) {
        int dst_idx = idz * nx * ny + idy * nx + idx;
        output[dst_idx] = tile[local_z][local_y][local_x];
    }
}

// Advanced convolution using FFT (frequency domain multiplication)
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
    
    // Point-wise complex multiplication in frequency domain
    Complex signal = signal_fft[linear_id];
    Complex kernel = kernel_fft[linear_id];
    
    result_fft[linear_id] = signal * kernel;
}

// Memory-intensive batch FFT processing
__global__ void batch_fft_process(
    Complex* batch_data,
    Complex* batch_output,
    int batch_size,
    int fft_size,
    int direction
) {
    int batch_id = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_id >= batch_size || tid >= fft_size) return;
    
    extern __shared__ Complex shared_batch_data[];
    
    // Load batch data into shared memory
    int global_offset = batch_id * fft_size;
    shared_batch_data[threadIdx.x] = batch_data[global_offset + tid];
    __syncthreads();
    
    // Perform FFT on this batch
    int log_n = 0;
    int temp_n = blockDim.x;
    while (temp_n > 1) {
        log_n++;
        temp_n >>= 1;
    }
    
    // Bit-reversal
    int reversed_tid = 0;
    int temp_tid = threadIdx.x;
    for (int i = 0; i < log_n; i++) {
        reversed_tid = (reversed_tid << 1) | (temp_tid & 1);
        temp_tid >>= 1;
    }
    
    Complex temp = shared_batch_data[threadIdx.x];
    __syncthreads();
    if (reversed_tid < blockDim.x) {
        shared_batch_data[reversed_tid] = temp;
    }
    __syncthreads();
    
    // FFT computation with memory bandwidth optimization
    for (int stage = 1; stage <= log_n; stage++) {
        int half_size = 1 << (stage - 1);
        int full_size = 1 << stage;
        
        if (threadIdx.x < blockDim.x / 2) {
            int group = threadIdx.x / half_size;
            int pos = threadIdx.x % half_size;
            
            int lower_idx = group * full_size + pos;
            int upper_idx = lower_idx + half_size;
            
            if (upper_idx < blockDim.x) {
                float angle = direction * 2.0f * PI * pos / full_size;
                Complex twiddle = exp_complex(angle);
                
                Complex lower = shared_batch_data[lower_idx];
                Complex upper = shared_batch_data[upper_idx];
                Complex temp_upper = upper * twiddle;
                
                shared_batch_data[lower_idx] = lower + temp_upper;
                shared_batch_data[upper_idx] = lower - temp_upper;
            }
        }
        __syncthreads();
    }
    
    // Write back to global memory
    batch_output[global_offset + tid] = shared_batch_data[threadIdx.x];
}

// Advanced spectral analysis kernel
__global__ void spectral_analysis(
    Complex* fft_data,
    float* magnitude_spectrum,
    float* phase_spectrum,
    float* power_spectrum,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    Complex sample = fft_data[tid];
    
    float magnitude = sample.magnitude();
    float phase = sample.phase();
    float power = magnitude * magnitude;
    
    magnitude_spectrum[tid] = magnitude;
    phase_spectrum[tid] = phase;
    power_spectrum[tid] = power;
    
    // Advanced spectral features
    if (tid > 0 && tid < n - 1) {
        // Spectral centroid contribution
        atomicAdd(&magnitude_spectrum[n], tid * magnitude);
        
        // Spectral rolloff contribution
        atomicAdd(&power_spectrum[n], power);
    }
}

// Wrapper functions for launching kernels from C++ files
extern "C" {
    void launchFFT1DRadix2Shared(Complex* data, int n, int direction, int stride, int blockSize, int gridSize) {
        fft_1d_radix2_shared<<<gridSize, blockSize, blockSize * sizeof(Complex)>>>(data, n, direction, stride);
    }
    
    void launchFFT3DOptimized(Complex* input, Complex* output, int nx, int ny, int nz, int direction, int phase, int blockSize, int gridSize) {
        fft_3d_optimized<<<gridSize, blockSize>>>(input, output, nx, ny, nz, direction, phase);
    }
}
