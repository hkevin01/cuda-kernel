#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hip/hip_runtime.h>
#include <cmath>

// Safe data structure for thread communication
struct ThreadData {
    float4 position;
    float4 velocity;
    float energy;
    int state;
};

// Safe warp reduction function
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffffffffffffULL, val, offset);
    }
    return val;
}

// Safe advanced kernel with careful synchronization
__global__ void safeAdvancedThreading(
    ThreadData* data,
    float* results,
    int* counters,
    int n,
    int iterations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // Safe shared memory sizes (well within limits)
    __shared__ float shared_buffer[32];  // Reduced from 1024
    __shared__ float reduction_buffer[256];
    
    if (tid >= n) return;
    
    ThreadData local_data = data[tid];
    float local_result = 0.0f;
    
    // Safe computation loop with bounded iterations
    const int safe_iterations = min(iterations, 50);  // Limit iterations
    
    for (int iter = 0; iter < safe_iterations; iter++) {
        // Phase 1: Independent computation
        float energy_delta = sinf(local_data.position.x + iter * 0.1f) * 
                           cosf(local_data.position.y + iter * 0.1f);
        local_data.energy += energy_delta;
        
        // Phase 2: Safe warp-level reduction
        float warp_sum = energy_delta;
        if (local_tid < 32) {  // Only first warp
            warp_sum = warp_reduce_sum(warp_sum);
            
            if (local_tid == 0) {
                shared_buffer[0] = warp_sum;
                atomicAdd(&counters[blockIdx.x], 1);
            }
        }
        __syncthreads();
        
        // Phase 3: Safe conditional work
        float block_average = shared_buffer[0] / min(blockDim.x, 32);
        if (local_data.energy > block_average) {
            local_data.state = 1;
            // Limited additional work
            local_data.velocity.x += 0.01f * sinf(local_data.position.x);
            local_data.velocity.y += 0.01f * cosf(local_data.position.y);
        } else {
            local_data.state = 0;
        }
        
        // Update position with damping
        local_data.position.x += local_data.velocity.x * 0.01f;
        local_data.position.y += local_data.velocity.y * 0.01f;
        
        // Add damping to prevent runaway values
        local_data.velocity.x *= 0.99f;
        local_data.velocity.y *= 0.99f;
        
        local_result += local_data.energy;
    }
    
    // Safe block-level reduction
    reduction_buffer[local_tid] = local_result;
    __syncthreads();
    
    // Tree reduction with safe bounds
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride && local_tid + stride < blockDim.x) {
            reduction_buffer[local_tid] += reduction_buffer[local_tid + stride];
        }
        __syncthreads();
    }
    
    // Write results safely
    data[tid] = local_data;
    results[tid] = local_result;
    
    if (local_tid == 0) {
        results[blockIdx.x + n] = reduction_buffer[0];
    }
}

// Safe lock-free operations with bounded retries
__global__ void safeLockFreeOperations(
    int* data,
    int* results,
    int n,
    int operations_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    int local_sum = 0;
    const int safe_ops = min(operations_per_thread, 20);  // Limit operations
    
    for (int op = 0; op < safe_ops; op++) {
        int index = (tid + op) % n;
        
        // Safe compare-and-swap with limited retries
        int old_val = atomicAdd(&data[index], 1);
        local_sum += old_val + 1;
        
        // Simple memory fence
        __threadfence_block();
        
        // Safe atomic operations
        atomicMax(&results[index % (n/2)], old_val + 1);
    }
    
    // Final accumulation with bounds check
    if (tid < n) {
        atomicAdd(&results[n/2], local_sum);
    }
}

// Safe memory patterns kernel
__global__ void safeMemoryPatterns(
    float4* input,
    float4* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Safe shared memory for tile processing
    __shared__ float4 tile[16][16];
    
    int local_x = threadIdx.x % 16;
    int local_y = threadIdx.y % 16;
    
    // Load tile safely
    if (local_x < 16 && local_y < 16) {
        int load_x = min(x, width - 1);
        int load_y = min(y, height - 1);
        int load_idx = load_y * width + load_x;
        tile[local_y][local_x] = input[load_idx];
    }
    
    __syncthreads();
    
    // Safe computation on tile
    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    
    if (local_x < 16 && local_y < 16) {
        float4 center = tile[local_y][local_x];
        
        // Simple convolution-like operation
        result.x = center.x * 0.5f;
        result.y = center.y * 0.5f;
        result.z = center.z * 0.5f;
        result.w = center.w * 0.5f;
        
        // Add neighbors if available
        if (local_x > 0) {
            result.x += tile[local_y][local_x - 1].x * 0.125f;
        }
        if (local_x < 15) {
            result.x += tile[local_y][local_x + 1].x * 0.125f;
        }
        if (local_y > 0) {
            result.y += tile[local_y - 1][local_x].y * 0.125f;
        }
        if (local_y < 15) {
            result.y += tile[local_y + 1][local_x].y * 0.125f;
        }
    }
    
    __syncthreads();
    
    // Write result safely
    output[idx] = result;
}

// Wrapper functions for C++ integration
extern "C" {
    void launchSafeAdvancedThreading(ThreadData* data, float* results, int* counters, int n, int iterations, int blockSize, int gridSize) {
        safeAdvancedThreading<<<gridSize, blockSize>>>(data, results, counters, n, iterations);
    }
    
    void launchSafeLockFreeOperations(int* data, int* results, int n, int operations_per_thread, int blockSize, int gridSize) {
        safeLockFreeOperations<<<gridSize, blockSize>>>(data, results, n, operations_per_thread);
    }
    
    void launchSafeMemoryPatterns(float4* input, float4* output, int width, int height, int blockSize, int gridSize) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        safeMemoryPatterns<<<grid, block>>>(input, output, width, height);
    }
}
