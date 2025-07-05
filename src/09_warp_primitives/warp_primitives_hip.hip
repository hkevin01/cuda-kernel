#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

// Manual warp reduction functions with 64-bit masks
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffffffffffffULL, val, offset);
    }
    return val;
}

__device__ int warp_reduce_max(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        int temp = __shfl_down_sync(0xffffffffffffffffULL, val, offset);
        val = max(val, temp);
    }
    return val;
}

__device__ int warp_reduce_min(int val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        int temp = __shfl_down_sync(0xffffffffffffffffULL, val, offset);
        val = min(val, temp);
    }
    return val;
}

__device__ int warp_scan_inclusive(int val) {
    for (int offset = 1; offset < 32; offset *= 2) {
        int temp = __shfl_up_sync(0xffffffffffffffffULL, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += temp;
        }
    }
    return val;
}

// Advanced tensor computation using warp primitives
__global__ void advancedTensorOperations(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta
) {
    // Get warp and thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    
    // Create warp-level cooperative group
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Each warp processes a tile of the output matrix
    int warp_row = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
    int warp_col = warp_id * 32;
    
    if (warp_row >= M || warp_col >= N) return;
    
    // Register tiling for better performance
    float4 c_vals[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    
    // Advanced matrix multiplication with warp-level optimizations
    for (int k = 0; k < K; k += 32) {
        // Collaborative loading within warp
        float4 a_vals[4];
        float b_val;
        
        // Load A values (each thread loads 4 elements)
        for (int i = 0; i < 4; i++) {
            int row = warp_row + i;
            if (row < M && k + lane_id < K) {
                int idx = row * K + k + lane_id;
                float a_temp = A[idx];
                a_vals[i] = {a_temp, a_temp, a_temp, a_temp};
            } else {
                a_vals[i] = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
        
        // Process 32 k-iterations using warp shuffle
        for (int kk = 0; kk < min(32, K - k); kk++) {
            // Broadcast B values across warp
            int b_col = warp_col + lane_id;
            if (b_col < N && k + kk < K) {
                b_val = B[(k + kk) * N + b_col];
            } else {
                b_val = 0.0f;
            }
            
            // Get A value for this k iteration
            float a_broadcast = __shfl_sync(0xffffffffffffffffULL, a_vals[0].x, kk);
            
            // Complex tensor operation with multiple accumulations
            for (int i = 0; i < 4; i++) {
                float a_val = __shfl_sync(0xffffffffffffffffULL, a_vals[i].x, kk);
                
                // Warp-level reduction for partial products
                float partial_product = a_val * b_val;
                
                // Use warp primitives for advanced operations
                float warp_sum = __shfl_down_sync(0xffffffffffffffffULL, partial_product, 16);
                partial_product += warp_sum;
                warp_sum = __shfl_down_sync(0xffffffffffffffffULL, partial_product, 8);
                partial_product += warp_sum;
                warp_sum = __shfl_down_sync(0xffffffffffffffffULL, partial_product, 4);
                partial_product += warp_sum;
                warp_sum = __shfl_down_sync(0xffffffffffffffffULL, partial_product, 2);
                partial_product += warp_sum;
                warp_sum = __shfl_down_sync(0xffffffffffffffffULL, partial_product, 1);
                partial_product += warp_sum;
                
                // Complex accumulation pattern
                c_vals[i].x += partial_product * alpha;
                c_vals[i].y += partial_product * alpha * 0.9f;
                c_vals[i].z += partial_product * alpha * 0.8f;
                c_vals[i].w += partial_product * alpha * 0.7f;
            }
        }
    }
    
    // Write results with complex beta blending
    for (int i = 0; i < 4; i++) {
        int row = warp_row + i;
        int col = warp_col + lane_id;
        
        if (row < M && col < N) {
            int idx = row * N + col;
            float old_val = (beta != 0.0f) ? C[idx] : 0.0f;
            
            // Complex final computation
            float final_val = c_vals[i].x + c_vals[i].y * 0.1f + 
                             c_vals[i].z * 0.01f + c_vals[i].w * 0.001f;
            
            C[idx] = final_val + beta * old_val;
        }
    }
}

// Warp-level primitive showcase
__global__ void warpPrimitivesShowcase(
    const int* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ warp_results,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int lane_id = warp.thread_rank();
    int warp_id = tid / 32;
    
    int value = input[tid];
    
    // 1. Warp shuffle operations
    int shuffled_up = __shfl_up_sync(0xffffffffffffffffULL, value, 1);
    int shuffled_down = __shfl_down_sync(0xffffffffffffffffULL, value, 1);
    int shuffled_xor = __shfl_xor_sync(0xffffffffffffffffULL, value, 1);
    int shuffled_idx = __shfl_sync(0xffffffffffffffffULL, value, (lane_id + 16) % 32);
    
    // 2. Warp vote operations
    bool predicate = (value % 2 == 0);
    bool all_even = __all_sync(0xffffffffffffffffULL, predicate);
    bool any_even = __any_sync(0xffffffffffffffffULL, predicate);
    unsigned long long ballot_result = __ballot_sync(0xffffffffffffffffULL, predicate);
    
    // 3. Warp reduction operations using manual implementation
    int sum = warp_reduce_sum(value);
    int max_val = warp_reduce_max(value);
    int min_val = warp_reduce_min(value);
    
    // 4. Warp scan operations using manual implementation
    int inclusive_scan = warp_scan_inclusive(value);
    int exclusive_scan = __shfl_up_sync(0xffffffffffffffffULL, inclusive_scan, 1);
    if (lane_id == 0) exclusive_scan = 0;
    
    // 5. Complex warp-level computation
    int complex_result = value;
    
    // Multi-stage shuffle-based computation
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor = __shfl_xor_sync(0xffffffffffffffffULL, complex_result, offset);
        complex_result = (complex_result * neighbor) % 1000007; // Large prime
    }
    
    // Conditional warp operations
    auto active_mask = __ballot_sync(0xffffffffffffffffULL, value > 50);
    if (value > 50) {
        // Manual reduction for active threads
        int temp_result = complex_result;
        for (int offset = 16; offset > 0; offset /= 2) {
            int neighbor = __shfl_down_sync(active_mask, temp_result, offset);
            if ((active_mask >> (lane_id + offset)) & 1) {
                temp_result += neighbor;
            }
        }
        complex_result = temp_result;
    }
    
    // Store individual thread results
    output[tid] = shuffled_up + shuffled_down + shuffled_xor + shuffled_idx + 
                  (all_even ? 1000 : 0) + (any_even ? 100 : 0) + 
                  inclusive_scan + exclusive_scan + complex_result;
    
    // Store warp-level results (one per warp)
    if (lane_id == 0 && warp_id < n / 32) {
        warp_results[warp_id * 8 + 0] = sum;
        warp_results[warp_id * 8 + 1] = max_val;
        warp_results[warp_id * 8 + 2] = min_val;
        warp_results[warp_id * 8 + 3] = (int)ballot_result;
        warp_results[warp_id * 8 + 4] = all_even ? 1 : 0;
        warp_results[warp_id * 8 + 5] = any_even ? 1 : 0;
        warp_results[warp_id * 8 + 6] = inclusive_scan;
        warp_results[warp_id * 8 + 7] = exclusive_scan;
    }
}

// Advanced parallel reduction using multiple warp primitives
__global__ void multiLevelReduction(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    int reduction_type
) {
    __shared__ float shared_data[1024];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    // Load data
    float value = (tid < n) ? input[tid] : 0.0f;
    
    // Apply different reduction operations based on type
    switch (reduction_type) {
        case 0: // Sum - manual warp reduction
            for (int offset = 16; offset > 0; offset /= 2) {
                value += warp.shfl_down(value, offset);
            }
            break;
        case 1: // Max - manual warp reduction
            for (int offset = 16; offset > 0; offset /= 2) {
                float neighbor = warp.shfl_down(value, offset);
                value = fmaxf(value, neighbor);
            }
            break;
        case 2: // Min - manual warp reduction
            for (int offset = 16; offset > 0; offset /= 2) {
                float neighbor = warp.shfl_down(value, offset);
                value = fminf(value, neighbor);
            }
            break;
        case 3: // Product (with overflow protection)
            for (int offset = 1; offset < 32; offset *= 2) {
                float neighbor = warp.shfl_down(value, offset);
                if (warp.thread_rank() + offset < 32) {
                    value = fminf(value * neighbor, 1e6f); // Prevent overflow
                }
            }
            break;
        case 4: // Complex reduction (RMS)
            value = value * value; // Square first
            value = warp_reduce_sum(value);
            if (warp.thread_rank() == 0) {
                value = sqrtf(value / 32.0f);
            }
            break;
    }
    
    // Store warp results in shared memory
    if (warp.thread_rank() == 0) {
        shared_data[threadIdx.x / 32] = value;
    }
    
    block.sync();
    
    // Second level reduction using first warp
    if (threadIdx.x < 32) {
        float warp_val = (threadIdx.x < blockDim.x / 32) ? shared_data[threadIdx.x] : 0.0f;
        
        // Final warp reduction
        switch (reduction_type) {
            case 0: // Sum
                for (int offset = 16; offset > 0; offset /= 2) {
                    warp_val += warp.shfl_down(warp_val, offset);
                }
                break;
            case 1: // Max
                for (int offset = 16; offset > 0; offset /= 2) {
                    float neighbor = warp.shfl_down(warp_val, offset);
                    warp_val = fmaxf(warp_val, neighbor);
                }
                break;
            case 2: // Min
                for (int offset = 16; offset > 0; offset /= 2) {
                    float neighbor = warp.shfl_down(warp_val, offset);
                    warp_val = fminf(warp_val, neighbor);
                }
                break;
            case 3:
                for (int offset = 1; offset < 32; offset *= 2) {
                    float neighbor = warp.shfl_down(warp_val, offset);
                    if (warp.thread_rank() + offset < 32) {
                        warp_val = fminf(warp_val * neighbor, 1e6f);
                    }
                }
                break;
            case 4:
                warp_val = warp_reduce_sum(warp_val);
                if (warp.thread_rank() == 0) {
                    warp_val = sqrtf(warp_val / (blockDim.x / 32.0f));
                }
                break;
        }
        
        if (threadIdx.x == 0) {
            output[blockIdx.x] = warp_val;
        }
    }
}

// Warp-level matrix transpose with advanced patterns
__global__ void warpMatrixTranspose(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    __shared__ float tile[32][33]; // Avoid bank conflicts with padding
    
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    
    // Calculate tile coordinates
    int tile_row = (warp_id / ((cols + 31) / 32)) * 32;
    int tile_col = (warp_id % ((cols + 31) / 32)) * 32;
    
    if (tile_row >= rows || tile_col >= cols) return;
    
    // Collaborative loading using warp shuffle
    for (int i = 0; i < 32; i++) {
        int src_row = tile_row + i;
        int src_col = tile_col + lane_id;
        
        float value = 0.0f;
        if (src_row < rows && src_col < cols) {
            value = input[src_row * cols + src_col];
        }
        
        // Use warp shuffle to distribute data
        for (int j = 0; j < 32; j++) {
            float shuffled_val = warp.shfl(value, j);
            if (i < 32 && j < 32) {
                tile[j][i] = shuffled_val;
            }
        }
    }
    
    // Synchronize at warp level (using block barrier instead)
    __syncthreads();
    
    // Collaborative storing with transpose
    for (int i = 0; i < 32; i++) {
        int dst_row = tile_col + i;
        int dst_col = tile_row + lane_id;
        
        if (dst_row < cols && dst_col < rows && i < 32 && lane_id < 32) {
            output[dst_row * rows + dst_col] = tile[i][lane_id];
        }
    }
}

// Advanced warp-level sorting network
__global__ void warpBitonicSort(
    int* data,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int lane_id = warp.thread_rank();
    
    // Load data into registers
    int value = data[tid];
    
    // Bitonic sort using warp shuffle operations
    for (int size = 2; size <= 32; size *= 2) {
        for (int stride = size / 2; stride > 0; stride /= 2) {
            // Determine comparison direction
            bool ascending = ((lane_id & size) == 0);
            
            // Get partner value
            int partner = lane_id ^ stride;
            int partner_value = warp.shfl(value, partner);
            
            // Compare and swap
            bool should_swap = (value > partner_value) ^ ascending ^ (lane_id > partner);
            if (should_swap) {
                value = partner_value;
            }
            
            // Synchronize warp - use thread barrier
            __syncthreads();
        }
    }
    
    // Store sorted result
    data[tid] = value;
}

// Complex warp-level prefix sum with multiple data types
__global__ void warpPrefixSum(
    const float* input_float,
    const int* input_int,
    float* output_float,
    int* output_int,
    long long* output_combined,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    
    // Load values
    float f_val = input_float[tid];
    int i_val = input_int[tid];
    
    // Float prefix sum - manual implementation
    float f_prefix = f_val;
    for (int offset = 1; offset < 32; offset *= 2) {
        float neighbor = warp.shfl_up(f_prefix, offset);
        if (warp.thread_rank() >= offset) {
            f_prefix += neighbor;
        }
    }
    f_prefix = warp.shfl_up(f_prefix, 1); // Convert to exclusive
    if (warp.thread_rank() == 0) f_prefix = 0.0f;
    
    // Integer prefix sum - manual implementation  
    int i_prefix = i_val;
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor = warp.shfl_up(i_prefix, offset);
        if (warp.thread_rank() >= offset) {
            i_prefix += neighbor;
        }
    }
    i_prefix = warp.shfl_up(i_prefix, 1); // Convert to exclusive
    if (warp.thread_rank() == 0) i_prefix = 0;
    
    // Combined computation with warp-level communication
    long long combined = (long long)(f_val * 1000.0f) + i_val;
    
    // Custom prefix sum for combined values
    for (int offset = 1; offset < 32; offset *= 2) {
        long long neighbor = warp.shfl_up(combined, offset);
        if (warp.thread_rank() >= offset) {
            combined += neighbor;
        }
    }
    
    // Store results
    output_float[tid] = f_prefix;
    output_int[tid] = i_prefix;
    output_combined[tid] = combined;
}

// Advanced warp-level convolution with shared memory optimization
__global__ void warpOptimizedConvolution(
    const float* __restrict__ input,
    const float* __restrict__ filter,
    float* __restrict__ output,
    int width, int height, int filter_size
) {
    __shared__ float shared_input[34][34]; // 32x32 tile + 2-pixel halo for 3x3 filter
    
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = warp.thread_rank();
    
    // Calculate tile coordinates
    int tile_x = (warp_id % ((width + 31) / 32)) * 32;
    int tile_y = (warp_id / ((width + 31) / 32)) * 32;
    
    int radius = filter_size / 2;
    
    // Collaborative loading with halo
    for (int dy = 0; dy < 34; dy += 1) {
        for (int dx = lane_id; dx < 34; dx += 32) {
            int src_x = tile_x + dx - radius;
            int src_y = tile_y + dy - radius;
            
            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                shared_input[dy][dx] = input[src_y * width + src_x];
            } else {
                shared_input[dy][dx] = 0.0f;
            }
        }
    }
    
    __syncthreads();
    
    // Process convolution using warp cooperation
    for (int local_y = 0; local_y < 32; local_y++) {
        int global_x = tile_x + lane_id;
        int global_y = tile_y + local_y;
        
        if (global_x < width && global_y < height) {
            float result = 0.0f;
            
            // Convolution computation with warp-level optimization
            for (int fy = 0; fy < filter_size; fy++) {
                for (int fx = 0; fx < filter_size; fx++) {
                    int local_x = lane_id + radius;
                    int local_sy = local_y + radius;
                    
                    float input_val = shared_input[local_sy + fy - radius][local_x + fx - radius];
                    float filter_val = filter[fy * filter_size + fx];
                    
                    result += input_val * filter_val;
                }
            }
            
            // Warp-level result sharing and processing - manual reduction
            float result_max = fabsf(result);
            for (int offset = 16; offset > 0; offset /= 2) {
                float neighbor = warp.shfl_down(result_max, offset);
                result_max = fmaxf(result_max, neighbor);
            }
            if (result_max > 0.0f) {
                result = result / result_max; // Normalize by warp max
            }
            
            output[global_y * width + global_x] = result;
        }
    }
}

// Multi-warp cooperative matrix multiplication with advanced tiling
__global__ void multiWarpMatrixMul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    __shared__ float shared_A[128][32]; // 4 warps worth of A data
    __shared__ float shared_B[32][128]; // 4 warps worth of B data
    
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    
    int warp_id = threadIdx.x / 32;
    int lane_id = warp.thread_rank();
    
    // Multi-warp tile coordinates
    int block_row = blockIdx.y * 128;
    int block_col = blockIdx.x * 128;
    
    // Each warp handles a 32x32 subtile
    int warp_row = block_row + (warp_id / 4) * 32;
    int warp_col = block_col + (warp_id % 4) * 32;
    
    float c_reg[4] = {0.0f}; // Register tiling
    
    // Main computation loop
    for (int k_tile = 0; k_tile < K; k_tile += 32) {
        // Cooperative loading of A and B tiles
        // Each warp loads its portion
        for (int i = 0; i < 4; i++) {
            int row = warp_row + i * 8 + (lane_id / 4);
            int col = k_tile + (lane_id % 4) * 8;
            
            if (row < M && col < K) {
                shared_A[warp_id * 32 + i * 8 + (lane_id / 4)][lane_id % 4] = A[row * K + col];
            }
        }
        
        for (int i = 0; i < 4; i++) {
            int row = k_tile + i * 8 + (lane_id / 4);
            int col = warp_col + (lane_id % 4) * 8;
            
            if (row < K && col < N) {
                shared_B[i * 8 + (lane_id / 4)][warp_id * 32 + (lane_id % 4)] = B[row * N + col];
            }
        }
        
        block.sync();
        
        // Warp-level computation with advanced scheduling
        for (int k = 0; k < 32; k++) {
            // Load A and B values with warp shuffle optimization
            float a_val = shared_A[warp_id * 32 + (lane_id / 8)][k];
            float b_vals[4];
            
            for (int j = 0; j < 4; j++) {
                b_vals[j] = shared_B[k][warp_id * 32 + j * 8 + (lane_id % 8)];
            }
            
            // Compute partial products with warp-level communication
            for (int j = 0; j < 4; j++) {
                float prod = a_val * b_vals[j];
                
                // Advanced warp reduction pattern
                prod = warp.shfl_xor(prod, 16);
                prod = warp.shfl_xor(prod, 8);
                prod = warp.shfl_xor(prod, 4);
                
                c_reg[j] += prod;
            }
        }
        
        block.sync();
    }
    
    // Write results back with warp cooperation
    for (int i = 0; i < 4; i++) {
        int row = warp_row + (lane_id / 8);
        int col = warp_col + i * 8 + (lane_id % 8);
        
        if (row < M && col < N) {
            C[row * N + col] = c_reg[i];
        }
    }
}

// Advanced warp-level pattern matching and string processing
__global__ void warpStringProcessing(
    const char* __restrict__ text,
    const char* __restrict__ pattern,
    int* __restrict__ matches,
    int text_length,
    int pattern_length
) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = warp.thread_rank();
    
    // Each warp processes 32 positions in parallel
    int base_pos = (tid / 32) * 32;
    int pos = base_pos + lane_id;
    
    if (pos + pattern_length > text_length) return;
    
    // Collaborative pattern matching
    bool match = true;
    
    // Load pattern collaboratively
    char pattern_chars[32];
    for (int i = lane_id; i < pattern_length; i += 32) {
        pattern_chars[i] = pattern[i];
    }
    
    // Warp-level string comparison
    for (int i = 0; i < pattern_length; i++) {
        char text_char = text[pos + i];
        char pattern_char = warp.shfl(pattern_chars[i % 32], i / 32);
        
        if (text_char != pattern_char) {
            match = false;
            break;
        }
    }
    
    // Warp-level match aggregation
    unsigned match_mask = __ballot_sync(__activemask(), match);
    int match_count = __popc(match_mask);
    
    // Store results
    if (lane_id == 0) {
        for (int i = 0; i < 32; i++) {
            if ((match_mask >> i) & 1) {
                atomicAdd(&matches[0], 1);
            }
        }
    }
}

// Advanced warp-level graph traversal (BFS on adjacency matrix)
__global__ void warpGraphTraversal(
    const int* __restrict__ adjacency_matrix,
    int* __restrict__ visited,
    int* __restrict__ distance,
    int* __restrict__ queue,
    int* queue_size,
    int num_vertices,
    int current_level
) {
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = warp.thread_rank();
    
    // Each warp processes vertices at current level
    int warp_id = tid / 32;
    int vertices_per_warp = (num_vertices + 31) / 32;
    
    int start_vertex = warp_id * vertices_per_warp;
    int end_vertex = min(start_vertex + vertices_per_warp, num_vertices);
    
    // Process vertices in current level
    for (int v = start_vertex + lane_id; v < end_vertex; v += 32) {
        if (v < num_vertices && distance[v] == current_level) {
            // Explore neighbors
            for (int neighbor = 0; neighbor < num_vertices; neighbor++) {
                if (adjacency_matrix[v * num_vertices + neighbor] && 
                    atomicCAS(&visited[neighbor], 0, 1) == 0) {
                    
                    distance[neighbor] = current_level + 1;
                    
                    // Add to next level queue using warp cooperation
                    int queue_pos = atomicAdd(queue_size, 1);
                    if (queue_pos < num_vertices) {
                        queue[queue_pos] = neighbor;
                    }
                }
            }
        }
    }
    
    // Warp-level synchronization for queue management - use thread barrier
    __syncthreads();
}

// Wrapper functions for launching kernels from C++ files
extern "C" {
    void launchAdvancedTensorOperations(float* A, float* B, float* C, int M, int N, int K, float alpha, float beta, int blockSize, int gridSize) {
        advancedTensorOperations<<<gridSize, blockSize>>>(A, B, C, M, N, K, alpha, beta);
    }
    
    void launchWarpPrimitivesShowcase(int* input, int* output, int* warp_results, int n, int blockSize, int gridSize) {
        warpPrimitivesShowcase<<<gridSize, blockSize>>>(input, output, warp_results, n);
    }
    
    void launchMultiLevelReduction(float* input, float* output, int n, int type, int blockSize, int gridSize) {
        multiLevelReduction<<<gridSize, blockSize>>>(input, output, n, type);
    }
    
    void launchWarpMatrixTranspose(float* input, float* output, int rows, int cols, int blockSize, int gridSize) {
        warpMatrixTranspose<<<gridSize, blockSize>>>(input, output, rows, cols);
    }
    
    void launchWarpBitonicSort(int* data, int n, int blockSize, int gridSize) {
        warpBitonicSort<<<gridSize, blockSize>>>(data, n);
    }
    
    void launchWarpPrefixSum(float* input_float, int* input_int, float* output_float, int* output_int, long long* output_combined, int n, int blockSize, int gridSize) {
        warpPrefixSum<<<gridSize, blockSize>>>(input_float, input_int, output_float, output_int, output_combined, n);
    }
    
    void launchWarpOptimizedConvolution(float* input, float* filter, float* output, int width, int height, int filter_size, int blockSize, int gridSize) {
        warpOptimizedConvolution<<<gridSize, blockSize>>>(input, filter, output, width, height, filter_size);
    }
    
    void launchMultiWarpMatrixMul(float* A, float* B, float* C, int M, int N, int K, int blockSize, int gridSize) {
        multiWarpMatrixMul<<<gridSize, blockSize>>>(A, B, C, M, N, K);
    }
}
