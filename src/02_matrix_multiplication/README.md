# Matrix Multiplication Example

## Overview
This example demonstrates advanced CUDA optimization techniques through matrix multiplication, a fundamental operation in scientific computing and machine learning. It showcases multiple optimization strategies from naive implementation to highly optimized tiled approaches.

## Key Concepts Demonstrated

### Memory Hierarchy Optimization
- **Shared Memory Tiling**: Reducing global memory accesses through data reuse
- **Memory Coalescing**: Optimizing memory access patterns
- **Bank Conflict Avoidance**: Ensuring efficient shared memory usage
- **Register Utilization**: Maximizing compute throughput

### Algorithmic Optimizations
- **Blocking/Tiling**: Breaking computation into cache-friendly chunks
- **Loop Unrolling**: Reducing instruction overhead
- **Thread Divergence Minimization**: Ensuring warp efficiency

## Code Structure

### Files
- `main.cpp`: Host code with multiple kernel benchmarks
- `matrix_mul.cu`: Multiple CUDA kernel implementations
- `matrix_mul.h`: Function declarations and kernel interfaces

### Kernel Implementations
1. **Naive**: Direct implementation without optimization
2. **Shared Memory**: Basic tiled approach using shared memory
3. **Optimized Tiled**: Advanced tiling with multiple optimizations
4. **cuBLAS**: NVIDIA's highly optimized library implementation

## Algorithm Analysis

### Computational Complexity
- **Time Complexity**: O(n³) for n×n matrices
- **Space Complexity**: O(n²) for matrix storage
- **Arithmetic Intensity**: 2n FLOPs per element (compute-bound for large matrices)

### Memory Access Patterns
```
Naive: Each element requires n reads from A and n reads from B
Tiled: Each tile element is loaded once and reused for block_size computations
```

### Optimization Strategy
```
1. Naive:     No data reuse, high memory bandwidth requirement
2. Tiled:     Block data reuse, reduced global memory traffic
3. Optimized: Additional register blocking, prefetching, unrolling
```

## Performance Characteristics

### Expected Results (RTX 3080)
- **Naive**: ~500 GFLOPS, memory-bound
- **Shared Memory**: ~2000 GFLOPS, improved data reuse
- **Optimized**: ~4000+ GFLOPS, approaching compute limits
- **cuBLAS**: ~8000+ GFLOPS, highly optimized assembly

### Key Metrics
- **Memory Bandwidth**: Decreases with better algorithms
- **Compute Utilization**: Increases with optimization
- **Occupancy**: Should be >50% for good performance

## Tiling Algorithm Details

### Shared Memory Tiling
```cpp
// Load tile of A and B into shared memory
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];

// Each thread loads one element
As[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];

// Compute partial result using shared data
for (int k = 0; k < TILE_SIZE; k++) {
    sum += As[ty][k] * Bs[k][tx];
}
```

### Register Blocking
- Load multiple elements per thread
- Compute multiple output elements simultaneously
- Reduce shared memory pressure

## Industry Applications

### High-Performance Computing
- **Scientific Simulations**: Linear solvers, eigenvalue problems
- **Computational Fluid Dynamics**: Sparse matrix operations
- **Quantum Chemistry**: Molecular orbital calculations

### Machine Learning
- **Deep Neural Networks**: Forward/backward propagation
- **Training Acceleration**: Gradient computations, weight updates
- **Inference Optimization**: Real-time model execution

### Graphics and Visualization
- **3D Transformations**: Model, view, projection matrices
- **Ray Tracing**: Intersection calculations
- **Computer Vision**: Feature extraction, filtering

## Build and Run

```bash
# Build example
cd src/02_matrix_multiplication
mkdir -p build && cd build
cmake .. && make

# Run with default size (1024x1024)
./matrix_multiplication

# Run with custom matrix size
./matrix_multiplication 2048

# Enable cuBLAS comparison
./matrix_multiplication 1024 1
```

## Expected Output
```
=== CUDA Matrix Multiplication Benchmark ===
Matrix size: 1024x1024 (8.0 MB per matrix)
Device: NVIDIA GeForce RTX 3080

--- CPU Reference ---
Time: 2847.123 ms
Performance: 0.75 GFLOPS

--- GPU Kernel Benchmarks ---
Method              Time (ms)   Performance    Efficiency
--------------------------------------------------------
Naive                  4.256      504.2 GFLOPS      6.3%
Shared Memory          1.087     1974.3 GFLOPS     24.7%
Optimized Tiled        0.543     3951.8 GFLOPS     49.4%
cuBLAS                 0.267     8038.2 GFLOPS     100.0%

--- Verification ---
Naive: PASS (Max Error: 0.00e+00)
Shared Memory: PASS (Max Error: 1.19e-05)
Optimized: PASS (Max Error: 2.38e-05)

Speedup vs CPU: 10667x (cuBLAS)
```

## Optimization Techniques

### Memory Optimizations
1. **Coalesced Access**: Ensure 128-byte aligned reads
2. **Shared Memory Banking**: Avoid bank conflicts
3. **Prefetching**: Hide memory latency with computation
4. **Register Blocking**: Reduce memory traffic

### Compute Optimizations
1. **Thread Block Sizing**: Balance occupancy and shared memory
2. **Loop Unrolling**: Reduce branch overhead
3. **Instruction Scheduling**: Maximize pipeline utilization
4. **Tensor Cores**: Use specialized ML hardware (A100, H100)

## Performance Tuning

### Key Parameters
- **TILE_SIZE**: Usually 16x16 or 32x32 for optimal shared memory usage
- **BLOCK_SIZE**: 16x16 threads per block is common
- **REG_BLOCKING**: 4x4 register tiles for advanced optimization

### Architecture-Specific Tuning
- **Compute Capability**: Adjust for specific GPU features
- **Memory Hierarchy**: Optimize for L1/L2 cache sizes
- **Warp Scheduler**: Consider instruction mix and latency

## Common Issues

### Performance Problems
1. **Bank Conflicts**: Use padding or transpose techniques
2. **Low Occupancy**: Reduce shared memory usage or increase blocks
3. **Memory Bound**: Implement more aggressive blocking

### Numerical Issues
1. **Floating-Point Precision**: Use appropriate tolerances
2. **Overflow/Underflow**: Consider mixed-precision arithmetic
3. **Accumulation Order**: May affect final precision

## Further Reading
- [Matrix Multiplication Optimization](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [Shared Memory Best Practices](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [GEMM Optimization Guide](https://github.com/flame/how-to-optimize-gemm)
