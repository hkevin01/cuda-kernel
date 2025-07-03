# CUDA Kernel Development Instructions for GitHub Copilot

## Project Context
This is a CUDA kernel programming project focusing on industry-relevant parallel computing examples. The project demonstrates:

1. **Vector Addition** - Basic parallel programming patterns
2. **Matrix Multiplication** - Shared memory optimization and tiling
3. **Parallel Reduction** - Advanced synchronization and warp primitives
4. **2D Convolution** - Image processing and boundary handling
5. **Monte Carlo Simulation** - Random number generation and statistical computing

## Code Conventions

### CUDA Kernel Patterns
- Always include boundary checks in kernels
- Use grid-stride loops for variable-sized data
- Implement shared memory tiling for matrix operations
- Use cooperative groups for synchronization
- Include proper error checking with CUDA_CHECK macro

### Memory Management
- Use pinned memory for host allocations
- Always check for sufficient GPU memory before allocation
- Implement proper cleanup in destructors/error paths
- Use async memory transfers when possible

### Performance Considerations
- Optimize for memory coalescing
- Minimize divergent branching
- Use appropriate block sizes (typically 256 threads)
- Implement multi-level optimizations (naive → shared → optimized)

### Error Handling
- Use CUDA_CHECK macro for all CUDA API calls
- Include CUDA_CHECK_KERNEL after kernel launches
- Provide meaningful error messages
- Handle insufficient memory gracefully

### Documentation
- Include performance analysis in comments
- Document algorithm complexity
- Explain optimization techniques used
- Provide theoretical vs actual performance comparisons

## Suggested Code Patterns

When generating CUDA kernels, prefer these patterns:

```cuda
// Grid-stride loop pattern
__global__ void kernel(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        // Process data[i]
    }
}

// Shared memory tiling pattern
__global__ void tiledKernel(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Load tile, synchronize, compute, repeat
}
```

## Build and Test Integration
- All code must compile with CMake build system
- Include CPU reference implementations for verification
- Add performance timing and analysis
- Ensure compatibility with CUDA 11.8+ and compute capability 7.5+
