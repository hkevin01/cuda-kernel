# CUDA Kernel Programming Examples

A comprehensive collection of CUDA kernel examples demonstrating essential parallel computing techniques for modern GPU programming. This project focuses on the most in-demand CUDA programming skills required in industry today.

## 🎯 Project Overview

This repository contains 5 carefully selected CUDA kernel examples that represent the core skills needed for CUDA development jobs in 2025:

1. **Vector Addition** - Foundation parallel programming
2. **Matrix Multiplication** - Shared memory optimization
3. **Parallel Reduction** - Advanced synchronization and optimization
4. **2D Convolution** - Image processing and compute kernels
5. **Monte Carlo Simulation** - Random number generation and statistical computing

## 📁 Project Structure

```
cuda-kernel/
├── src/                          # Source code
│   ├── 01_vector_addition/       # Basic parallel vector operations
│   ├── 02_matrix_multiplication/ # Optimized matrix operations
│   ├── 03_parallel_reduction/    # Reduction algorithms
│   ├── 04_convolution_2d/        # 2D convolution kernels
│   ├── 05_monte_carlo/           # Monte Carlo simulations
│   └── common/                   # Shared utilities and headers
├── docs/                         # Documentation
│   ├── theory/                   # Theoretical background
│   ├── performance/              # Performance analysis
│   └── api/                      # API documentation
├── scripts/                      # Build and utility scripts
│   ├── build.sh                  # Build automation
│   ├── profile.sh                # Performance profiling
│   └── test.sh                   # Testing automation
├── .github/                      # GitHub Actions CI/CD
│   └── workflows/
├── .copilot/                     # GitHub Copilot configurations
└── CMakeLists.txt               # Build configuration
```

## 🚀 Getting Started

### Prerequisites

- **CUDA Toolkit** 12.0 or later
- **CMake** 3.20 or later
- **C++ Compiler** with C++17 support
- **NVIDIA GPU** with compute capability 3.5+

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd cuda-kernel

# Build all examples
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run vector addition example
./bin/vector_addition

# Run with profiling
nvprof ./bin/matrix_multiplication
```

## 📋 Examples Overview

### 1. Vector Addition
**File**: `src/01_vector_addition/`
- **Concept**: Basic GPU parallelization
- **Industry Use**: Foundation for all CUDA programming
- **Skills**: Thread indexing, memory management, kernel launch

### 2. Matrix Multiplication
**File**: `src/02_matrix_multiplication/`
- **Concept**: Shared memory optimization
- **Industry Use**: Deep learning, scientific computing
- **Skills**: Tiling, shared memory, memory coalescing

### 3. Parallel Reduction
**File**: `src/03_parallel_reduction/`
- **Concept**: Efficient parallel algorithms
- **Industry Use**: Sum, max/min operations, statistics
- **Skills**: Warp primitives, cooperative groups, optimization

### 4. 2D Convolution
**File**: `src/04_convolution_2d/`
- **Concept**: Image processing and CNNs
- **Industry Use**: Computer vision, deep learning
- **Skills**: 2D thread blocks, boundary handling, performance tuning

### 5. Monte Carlo Simulation
**File**: `src/05_monte_carlo/`
- **Concept**: Random number generation and statistics
- **Industry Use**: Financial modeling, physics simulations
- **Skills**: cuRAND, statistical algorithms, large-scale parallelism

## 🎯 Industry Relevance

These examples cover the most sought-after CUDA skills in 2025:

- **AI/Machine Learning**: Matrix operations, convolutions, reductions
- **High-Performance Computing**: Memory optimization, algorithm design
- **Computer Vision**: Image processing, filter operations
- **Financial Technology**: Monte Carlo methods, parallel algorithms
- **Scientific Computing**: Numerical methods, simulation techniques

## 🛠️ Build System

The project uses CMake for cross-platform builds:

```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..

# Enable profiling
cmake -DENABLE_PROFILING=ON ..
```

## 📊 Performance Analysis

Each example includes:
- Theoretical performance analysis
- Memory bandwidth utilization
- Profiling instructions
- Optimization techniques
- Comparison with CPU implementations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## 📚 Learning Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA CUDA Samples repository for inspiration
- CUDA community for best practices
- Contributors and maintainers

---

**Built with ❤️ for the CUDA community**
