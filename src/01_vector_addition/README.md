# Vector Addition Example

## Overview
This example demonstrates basic CUDA programming concepts through vector addition, the "Hello World" of parallel computing. It shows fundamental GPU memory management, kernel execution, and performance analysis.

## Key Concepts Demonstrated

### CUDA Programming Fundamentals
- **Device Memory Management**: `cudaMalloc`, `cudaMemcpy`, `cudaFree`
- **Kernel Launch Configuration**: Grid and block dimensions
- **Thread Indexing**: Using `blockIdx`, `blockDim`, and `threadIdx`
- **Error Handling**: CUDA error checking with `CUDA_CHECK` macro

### Performance Optimization
- **Memory Bandwidth Utilization**: Measuring achieved vs theoretical bandwidth
- **Occupancy Analysis**: Understanding thread block sizing
- **Coalesced Memory Access**: Ensuring efficient memory patterns

## Code Structure

### Files
- `main.cpp`: Host code with benchmarking and verification
- `vector_addition.cu`: CUDA kernel implementation
- `vector_addition.h`: Function declarations

### Key Functions
- `vectorAdd`: CUDA kernel for parallel vector addition
- `vectorAddCPU`: CPU reference implementation for verification
- `benchmarkVectorAdd`: Performance measurement wrapper

## Algorithm Analysis

### Computational Complexity
- **Time Complexity**: O(n) - linear in vector size
- **Space Complexity**: O(n) - requires memory for three vectors
- **Arithmetic Intensity**: 1 FLOP per 12 bytes (very memory-bound)

### Memory Access Pattern
```
Thread 0: A[0] + B[0] → C[0]
Thread 1: A[1] + B[1] → C[1]
Thread 2: A[2] + B[2] → C[2]
...
Thread n: A[n] + B[n] → C[n]
```

## Performance Characteristics

### Expected Results
- **Memory Bandwidth**: Should achieve 80-90% of theoretical peak
- **Scalability**: Linear scaling with vector size (up to memory limits)
- **GPU vs CPU Speedup**: 5-20x depending on hardware

### Optimization Opportunities
1. **Vector Types**: Using `float4` for improved memory throughput
2. **Memory Patterns**: Ensuring 128-byte aligned accesses
3. **Thread Block Size**: Tuning for specific GPU architecture

## Industry Applications

### Real-World Use Cases
- **Linear Algebra Libraries**: BLAS, cuBLAS operations
- **Scientific Computing**: Element-wise array operations
- **Machine Learning**: Gradient computations, activation functions
- **Signal Processing**: Sample-wise transformations

### Performance Requirements
- **HPC Applications**: Need >80% bandwidth efficiency
- **Real-time Systems**: Require predictable, low-latency execution
- **Batch Processing**: Optimize for maximum throughput

## Build and Run

```bash
# Build specific example
cd src/01_vector_addition
mkdir -p build && cd build
cmake .. && make

# Run with default parameters (16M elements)
./vector_addition

# Run with custom vector size
./vector_addition 1048576

# Run with different block sizes
./vector_addition 1048576 256
```

## Expected Output
```
=== CUDA Vector Addition Benchmark ===
Device: NVIDIA GeForce RTX 3080
Vector size: 16777216 elements (64.0 MB per vector)

--- CPU Reference ---
Time: 45.123 ms

--- GPU Kernel ---
Time: 2.456 ms
Achieved Bandwidth: 78.2 GB/s
Theoretical Bandwidth: 936.2 GB/s
Memory Efficiency: 83.5%

--- Verification ---
Result: PASS (Max Error: 0.00e+00)

Speedup: 18.4x
```

## Common Issues and Solutions

### Performance Problems
1. **Low Bandwidth**: Check memory alignment and coalescing
2. **Poor Occupancy**: Adjust block size (typically 128-512 threads)
3. **CPU Overhead**: Use streams for overlapping computation

### Debugging Tips
1. **Verify Results**: Always compare with CPU reference
2. **Check Errors**: Use `CUDA_CHECK` for all CUDA calls
3. **Profile**: Use `nvprof` or Nsight Systems for detailed analysis

## Further Reading
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Memory Coalescing Patterns](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
