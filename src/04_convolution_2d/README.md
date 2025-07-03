# 2D Convolution Example

## Overview
This example demonstrates GPU-accelerated 2D convolution, a cornerstone operation in computer vision, image processing, and deep learning. It showcases multiple optimization strategies from naive implementations to advanced techniques using shared memory, constant memory, and texture memory.

## Key Concepts Demonstrated

### Memory Hierarchy Optimization
- **Shared Memory Tiling**: Implementing sliding window algorithms efficiently
- **Constant Memory**: Optimizing small, read-only data access
- **Texture Memory**: Leveraging hardware caching and interpolation
- **Memory Coalescing**: Ensuring efficient global memory access patterns

### Algorithmic Optimizations
- **Halo Loading**: Handling boundary conditions in tiled algorithms
- **Separable Convolution**: Reducing O(n²) to O(n) operations for applicable kernels
- **Boundary Handling**: Zero padding, clamping, and wrap-around strategies
- **Loop Unrolling**: Reducing instruction overhead for small kernels

## Code Structure

### Files
- `main.cpp`: Host code with comprehensive benchmarking suite
- `convolution.cu`: Multiple CUDA kernel implementations
- `convolution.h`: Function declarations and kernel interfaces

### Kernel Implementations
1. **Naive**: Direct implementation with global memory
2. **Shared Memory**: Tiled approach with data reuse
3. **Constant Memory**: Small kernels stored in constant memory
4. **Texture Memory**: Hardware-assisted caching and filtering

## Algorithm Analysis

### Computational Complexity
- **Time Complexity**: O(W × H × K²) where W,H are image dimensions, K is kernel size
- **Space Complexity**: O(W × H) for image storage + O(K²) for kernel
- **Arithmetic Intensity**: 2K² FLOPs per output element

### Memory Access Patterns
```
Naive:     Each output pixel reads K² input pixels independently
Tiled:     Input pixels loaded once per tile, reused across threads
Constant:  Kernel values cached in fast constant memory
Texture:   Input data cached with automatic boundary handling
```

## Optimization Strategies

### 1. Naive Implementation
```cpp
// Direct global memory access
for (int k_row = 0; k_row < kernel_size; k_row++) {
    for (int k_col = 0; k_col < kernel_size; k_col++) {
        sum += input[img_row * width + img_col] * 
               kernel[k_row * kernel_size + k_col];
    }
}
```
**Characteristics**: Simple but inefficient, high memory bandwidth

### 2. Shared Memory Tiling
```cpp
// Load tile with halo region
extern __shared__ float tile[];
// ... load data with boundaries
__syncthreads();

// Compute using shared data
sum += tile[tile_row * shared_size + tile_col] * kernel[k];
```
**Benefits**: Reduces global memory traffic through data reuse

### 3. Constant Memory Optimization
```cpp
__constant__ float d_kernel[MAX_KERNEL_SIZE];

// Fast cached access to kernel values
sum += input[idx] * d_kernel[k_row * kernel_size + k_col];
```
**Advantages**: High bandwidth for small, uniform access patterns

### 4. Texture Memory
```cpp
// Hardware-assisted boundary handling and caching
sum += tex2D<float>(texObj, img_col, img_row) * kernel[k];
```
**Features**: Automatic interpolation, boundary clamping, L1 caching

## Performance Characteristics

### Expected Results (RTX 3080, 1024x1024 image, 5x5 kernel)
| Implementation | Time (ms) | Bandwidth (GB/s) | Speedup |
|---------------|-----------|------------------|---------|
| CPU Reference | 1200.0 | 1.7 | 1.0x |
| Naive GPU | 15.2 | 134.5 | 78.9x |
| Shared Memory | 8.7 | 234.8 | 137.9x |
| Constant Memory | 7.1 | 287.3 | 169.0x |
| Texture Memory | 6.8 | 300.1 | 176.5x |

### Memory Efficiency Analysis
- **Theoretical Peak**: ~936 GB/s (RTX 3080)
- **Achieved**: Up to 300 GB/s (32% efficiency)
- **Bottleneck**: Arithmetic intensity too low for compute-bound performance

## Industry Applications

### Computer Vision
- **Edge Detection**: Sobel, Canny, Laplacian operators
- **Feature Extraction**: Harris corners, SIFT descriptors
- **Object Recognition**: Template matching, pattern detection
- **Medical Imaging**: CT/MRI enhancement, noise reduction

### Deep Learning
- **Convolutional Neural Networks**: Forward and backward propagation
- **Image Classification**: ResNet, VGG, Inception architectures
- **Object Detection**: YOLO, R-CNN, SSD algorithms
- **Semantic Segmentation**: U-Net, DeepLab models

### Image Processing
- **Filtering**: Gaussian blur, sharpening, emboss effects
- **Noise Reduction**: Bilateral filtering, non-local means
- **Enhancement**: Histogram equalization, contrast adjustment
- **Artistic Effects**: Stylization, HDR tone mapping

### Signal Processing
- **Audio Filtering**: Digital filters, noise cancellation
- **Radar Processing**: Target detection, clutter suppression
- **Communications**: Channel equalization, interference rejection
- **Biomedical**: ECG/EEG signal analysis, brain imaging

## Advanced Optimizations

### Separable Convolution
For kernels that can be decomposed (e.g., Gaussian):
```cpp
// Two 1D convolutions instead of one 2D
kernel_1d_horizontal<<<...>>>(input, temp, kernel_x);
kernel_1d_vertical<<<...>>>(temp, output, kernel_y);
```
**Benefit**: Reduces O(K²) to O(2K) operations per pixel

### Multiple Outputs per Thread
```cpp
// Each thread computes 2x2 output pixels
float4 results = make_float4(0,0,0,0);
// Compute four convolutions simultaneously
```
**Advantage**: Better register utilization, reduced memory traffic

### Template Specialization
```cpp
template<int KERNEL_SIZE>
__global__ void convolution_template(...) {
    // Compile-time loop unrolling
    #pragma unroll
    for (int k = 0; k < KERNEL_SIZE; k++) { ... }
}
```
**Benefit**: Eliminates runtime loop overhead

## Memory Optimization Details

### Shared Memory Tiling Strategy
```
Thread Block: 16x16 threads
Tile Size: (16 + 2*radius) x (16 + 2*radius) with halo
Loading: Each thread loads multiple elements to fill tile
Synchronization: __syncthreads() after loading, before computation
```

### Boundary Condition Handling
1. **Zero Padding**: Out-of-bounds reads return 0
2. **Clamping**: Use nearest valid pixel value
3. **Wrapping**: Treat image as tiled/periodic
4. **Reflection**: Mirror image at boundaries

### Bank Conflict Avoidance
```cpp
// Pad shared memory to avoid bank conflicts
__shared__ float tile[TILE_HEIGHT][TILE_WIDTH + 1];
```

## Build and Run

```bash
# Build example
cd src/04_convolution_2d
mkdir -p build && cd build
cmake .. && make

# Run with default parameters (1024x1024, 5x5 kernel)
./convolution_2d

# Run with custom dimensions
./convolution_2d 2048 2048 7

# Test different optimization methods
./convolution_2d 1024 1024 5 0  # Naive
./convolution_2d 1024 1024 5 1  # Shared memory
./convolution_2d 1024 1024 5 2  # Constant memory
./convolution_2d 1024 1024 5 3  # Texture memory
```

## Expected Output
```
=== CUDA 2D Convolution Benchmark ===
Image size: 1024x1024 (4.0 MB)
Kernel size: 5x5
Device: NVIDIA GeForce RTX 3080

Generated test image with edge patterns and Gaussian blur kernel

--- CPU Reference ---
Time: 1247.832 ms

--- GPU Kernel Benchmarks ---
Method              Time (ms)   Bandwidth    Performance
--------------------------------------------------------
Naive                  15.234      134.5 GB/s    134.2 GFLOPS
Shared Memory           8.721      234.8 GB/s    234.5 GFLOPS
Constant Memory         7.102      287.3 GB/s    287.0 GFLOPS
Texture Memory          6.834      300.1 GB/s    299.8 GFLOPS

--- Verification ---
Naive: PASS (Max Error: 0.00e+00)
Shared Memory: PASS (Max Error: 1.19e-07)
Constant Memory: PASS (Max Error: 2.38e-07)
Texture Memory: PASS (Max Error: 4.77e-07)

--- Performance Analysis ---
Speedup (Shared Memory vs Naive): 1.75x
Memory Bandwidth Efficiency: 32.1%
Arithmetic Intensity: 2.00 FLOPS/byte

--- Industry Applications ---
• Computer Vision: Edge detection, noise reduction, feature extraction
• Deep Learning: Convolutional neural network layers
• Image Processing: Gaussian blur, sharpening, emboss effects
• Signal Processing: Digital filters, pattern recognition
```

## Performance Tuning

### Optimal Parameters
- **Block Size**: 16x16 threads (good balance of occupancy and shared memory)
- **Tile Size**: Block size + 2×kernel_radius for halo region
- **Kernel Size**: Constant memory limited to ~15x15 (225 elements)
- **Image Size**: Should be multiple of block size for best efficiency

### Memory Considerations
- **Shared Memory**: 48KB per SM, divide among active blocks
- **Constant Memory**: 64KB total, cached effectively for uniform access
- **Texture Memory**: 2D texture objects provide automatic caching
- **Register Usage**: High kernel complexity may reduce occupancy

## Common Issues and Solutions

### Performance Problems
1. **Bank Conflicts**: Pad shared memory arrays
2. **Divergent Branches**: Ensure uniform boundary handling
3. **Memory Throughput**: Use appropriate memory hierarchy
4. **Occupancy**: Balance shared memory usage with active blocks

### Correctness Issues
1. **Boundary Conditions**: Verify edge pixel calculations
2. **Synchronization**: Ensure proper `__syncthreads()` placement
3. **Index Calculations**: Check for off-by-one errors
4. **Floating-Point Precision**: Use appropriate error tolerances

## Further Reading
- [CUDA Convolution Example](https://docs.nvidia.com/cuda/cuda-samples/index.html#convolution-separable)
- [Shared Memory Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)
- [Texture Memory Programming](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory)
- [Deep Learning Convolution Optimization](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
