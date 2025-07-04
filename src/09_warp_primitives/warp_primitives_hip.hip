#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

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
            float a_broadcast = warp.shfl(a_vals[0].x, kk);
            
            // Complex tensor operation with multiple accumulations
            for (int i = 0; i < 4; i++) {
                float a_val = warp.shfl(a_vals[i].x, kk);
                
                // Warp-level reduction for partial products
                float partial_product = a_val * b_val;
                
                // Use warp primitives for advanced operations
                float warp_sum = warp.shfl_down(partial_product, 16);
                partial_product += warp_sum;
                warp_sum = warp.shfl_down(partial_product, 8);
                partial_product += warp_sum;
                warp_sum = warp.shfl_down(partial_product, 4);
                partial_product += warp_sum;
                warp_sum = warp.shfl_down(partial_product, 2);
                partial_product += warp_sum;
                warp_sum = warp.shfl_down(partial_product, 1);
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
    int shuffled_up = warp.shfl_up(value, 1);
    int shuffled_down = warp.shfl_down(value, 1);
    int shuffled_xor = warp.shfl_xor(value, 1);
    int shuffled_idx = warp.shfl(value, (lane_id + 16) % 32);
    
    // 2. Warp vote operations
    bool predicate = (value % 2 == 0);
    bool all_even = warp.all(predicate);
    bool any_even = warp.any(predicate);
    unsigned ballot_result = warp.ballot(predicate);
    
    // 3. Warp reduction operations
    int sum = cg::reduce(warp, value, cg::plus<int>());
    int max_val = cg::reduce(warp, value, cg::greater<int>());
    int min_val = cg::reduce(warp, value, cg::less<int>());
    
    // 4. Warp scan operations
    int inclusive_scan = cg::inclusive_scan(warp, value, cg::plus<int>());
    int exclusive_scan = cg::exclusive_scan(warp, value, cg::plus<int>());
    
    // 5. Complex warp-level computation
    int complex_result = value;
    
    // Multi-stage shuffle-based computation
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor = warp.shfl_xor(complex_result, offset);
        complex_result = (complex_result * neighbor) % 1000007; // Large prime
    }
    
    // Conditional warp operations
    auto active_mask = warp.ballot(value > 50);
    if (value > 50) {
        auto active_threads = cg::coalesced_threads();
        complex_result = cg::reduce(active_threads, complex_result, cg::plus<int>());
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
        warp_results[warp_id * 8 + 3] = ballot_result;
        warp_results[warp_id * 8 + 4] = all_even ? 1 : 0;
        warp_results[warp_id * 8 + 5] = any_even ? 1 : 0;
        warp_results[warp_id * 8 + 6] = active_mask;
        warp_results[warp_id * 8 + 7] = complex_result;
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
        case 0: // Sum
            value = cg::reduce(warp, value, cg::plus<float>());
            break;
        case 1: // Max
            value = cg::reduce(warp, value, cg::greater<float>());
            break;
        case 2: // Min
            value = cg::reduce(warp, value, cg::less<float>());
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
            value = cg::reduce(warp, value, cg::plus<float>());
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
            case 0:
                warp_val = cg::reduce(warp, warp_val, cg::plus<float>());
                break;
            case 1:
                warp_val = cg::reduce(warp, warp_val, cg::greater<float>());
                break;
            case 2:
                warp_val = cg::reduce(warp, warp_val, cg::less<float>());
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
                warp_val = cg::reduce(warp, warp_val, cg::plus<float>());
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
    
    // Synchronize at warp level (using shared memory barrier)
    __syncwarp();
    
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
            
            // Synchronize warp
            __syncwarp();
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
    
    // Float prefix sum
    float f_prefix = cg::exclusive_scan(warp, f_val, cg::plus<float>());
    
    // Integer prefix sum
    int i_prefix = cg::exclusive_scan(warp, i_val, cg::plus<int>());
    
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
