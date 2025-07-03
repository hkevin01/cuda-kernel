# GPU Kernel Programming Examples (CUDA + ROCm)

A comprehensive collection of GPU kernel examples demonstrating essential parallel computing techniques for modern GPU programming. This project supports both NVIDIA CUDA and AMD ROCm platforms, focusing on the most in-demand GPU programming skills required in industry today.

## 🎯 Project Overview

This repository contains 5 carefully selected GPU kernel examples that represent the core skills needed for GPU development jobs in 2025, supporting both NVIDIA and AMD hardware:

1. **Vector Addition** - Foundation parallel programming
2. **Matrix Multiplication** - Shared memory optimization  
3. **Parallel Reduction** - Advanced synchronization and optimization
4. **2D Convolution** - Image processing and compute kernels
5. **Monte Carlo Simulation** - Random number generation and statistical computing

## 🖥️ Platform Support

- **NVIDIA GPUs**: CUDA toolkit (11.0+)
- **AMD GPUs**: ROCm platform (5.0+)
- **Cross-platform**: HIP (Heterogeneous Interface for Portability)

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

#### Hardware Requirements
- **NVIDIA GPU**: GTX 10-series or newer (compute capability 6.0+)
- **AMD GPU**: RX 6000/7000 series, Vega, or MI-series
- **Memory**: 4GB+ GPU memory recommended

#### Software Requirements

**For NVIDIA GPUs (CUDA)**
- **CUDA Toolkit** 11.0 or later
- **CMake** 3.20 or later
- **C++ Compiler** with C++17 support

**For AMD GPUs (ROCm)**
- **ROCm** 5.0 or later
- **HIP** development tools
- **CMake** 3.20 or later
- **C++ Compiler** with C++17 support

### Quick Start

#### Automatic Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd cuda-kernel

# Run the automated setup script
./scripts/setup_gpu_platform.sh
```

The setup script will:
- Detect your GPU hardware (NVIDIA or AMD)
- Check for required software (CUDA or ROCm)
- Offer to install missing components
- Build the project with appropriate platform support

#### Manual Setup

**For NVIDIA GPUs (CUDA)**
```bash
# Install CUDA Toolkit 11.0+
# See docs/installation/cuda-setup.md for detailed instructions

# Build with CUDA
mkdir build && cd build
cmake -DUSE_CUDA=ON ..
make -j$(nproc)
```

**For AMD GPUs (ROCm)**
```bash
# Install ROCm 5.0+
# See docs/installation/rocm-setup.md for detailed instructions

# Build with ROCm/HIP
mkdir build && cd build
cmake -DUSE_HIP=ON ..
make -j$(nproc)
```

#### Run Examples
```bash
# From build directory
./01_vector_addition
./02_matrix_multiplication
./03_parallel_reduction
./04_convolution_2d
./05_monte_carlo
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
