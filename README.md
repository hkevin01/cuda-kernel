[![Build](https://img.shields.io/github/actions/workflow/status/hkevin01/cuda-kernel/ci.yml?branch=main&logo=github)](https://github.com/hkevin01/cuda-kernel/actions)
[![License](https://img.shields.io/github/license/hkevin01/cuda-kernel?color=blue)](LICENSE)
[![Platform](https://img.shields.io/badge/GPU-NVIDIA%20%7C%20AMD-green?logo=nvidia&logoColor=white)](#)
[![Educational](https://img.shields.io/badge/Focus-Education%20%26%20Demos-blueviolet?logo=academia)](#)
[![Language](https://img.shields.io/badge/Language-C%2B%2B%20%7C%20CUDA%20%7C%20HIP-orange?logo=cplusplus)](#)
[![GUI](https://img.shields.io/badge/GUI-Qt6-green?logo=qt)](#)

# GPU Kernel Project 🚀

**Learn GPU programming through hands-on examples!** 

This project makes GPU computing accessible with 9 interactive examples that demonstrate the incredible parallel processing power of modern graphics cards. From simple array operations to complex physics simulations, see how GPUs can accelerate computations by 10-100x compared to traditional CPUs.

**Perfect for**: Students learning parallel computing, developers exploring GPU acceleration, researchers needing performance, or anyone curious about how modern AI and graphics work under the hood.

**What makes this special**: Interactive GUI + educational descriptions + real code + performance metrics = the complete GPU learning experience!

## 🚀 Quick Start

### 🎯 **Complete Beginner? Start Here!**
```bash
# 1. Clone this repository
git clone <repository-url>
cd cuda-kernel

# 2. Run the interactive GUI (it will build everything automatically)
./run.sh

# 3. In the GUI: Select "Vector Addition" → Click "Run" → See GPU magic! ✨
```

That's it! The GUI will guide you through everything else.

### Prerequisites
- **HIP/ROCm** (for AMD GPUs) or **CUDA** (for NVIDIA GPUs)
- **Qt6** development packages
- **CMake** 3.16 or later
- **C++14** compatible compiler

### Advanced Usage
```bash
# Quick start - build and run GUI (auto-detects platform)
./run.sh

# Specific platform
./run.sh -p hip        # For AMD GPUs
./run.sh -p cuda       # For NVIDIA GPUs

# Force rebuild
./run.sh -b

# Build specific components
./scripts/build/build_gui_hip.sh    # Build GUI with HIP
./scripts/build/build_kernels_safely.sh  # Build kernels safely
```

## 📁 Project Structure

```
cuda-kernel/
├── src/                    # GPU Kernel Source Code
│   ├── 01_vector_addition/     # Parallel array addition (GPU basics)
│   ├── 02_matrix_multiplication/ # Linear algebra operations (ML/graphics)
│   ├── 03_parallel_reduction/   # Data aggregation algorithms (statistics)
│   ├── 04_convolution_2d/       # Image filtering (computer vision)
│   ├── 05_monte_carlo/          # Random sampling simulations (modeling)
│   ├── 07_advanced_threading/   # Thread synchronization patterns (cooperation)
│   ├── 08_dynamic_memory/       # GPU memory management (optimization)
│   ├── 11_nbody_simulation/     # Physics simulations (gravitational forces)
│   └── common/                  # Shared utilities and helper functions
├── gui/                    # Qt-based GUI application for interactive testing
├── tests/                  # Unit test source code and test framework
├── docs/                   # Documentation, guides, and project status
├── logs/                   # Runtime logs and test output files
├── scripts/               # Organized build and utility scripts
│   ├── build/                  # Build scripts for different platforms
│   ├── testing/               # Automated test scripts and validation
│   ├── gui/                   # GUI launch and setup scripts
│   └── verification/          # Project verification and status checks
├── build*/                # Build output directories (generated)
└── run.sh                 # Main launcher script (start here!)
```

## 🔧 Build Options

### Quick Build (Recommended)
```bash
# Auto-detect platform and build
./run.sh

# Force rebuild with specific platform
./run.sh -b -p hip     # Rebuild for AMD GPUs
./run.sh -b -p cuda    # Rebuild for NVIDIA GPUs
```

### Organized Build Scripts
```bash
# GPU-specific builds
./scripts/build/build_gui_hip.sh        # GUI with HIP/ROCm
./scripts/build/build_hip.sh            # HIP kernels
./scripts/build/build_kernels_safely.sh # Safe kernel build

# Legacy builds
./scripts/build_working.sh --clean      # Build all working components
./scripts/build_unified.sh              # Unified build system
```

### Manual Component Builds
```bash
# Build GUI only
mkdir build_gui && cd build_gui
cmake ../gui && make

# Build individual kernels
cd src/01_vector_addition
hipcc -O3 -std=c++14 -I../common -o ../../build/bin/vector_addition \
    main_hip.cpp vector_addition_hip.hip ../common/*.cpp
```

## 🌟 Why GPU Computing?

**GPU vs CPU**: While CPUs have 4-16 powerful cores optimized for complex tasks, GPUs have thousands of simpler cores designed for parallel work. Think of it as the difference between having a few brilliant professors versus an entire classroom of students working together.

**Real Performance**: A typical GPU operation can be **10-100x faster** than CPU for parallel tasks:
- **Vector Addition**: CPU processes 1 element at a time, GPU processes thousands simultaneously
- **Matrix Multiplication**: Critical for AI/ML - GPUs make training neural networks practical
- **Image Processing**: Apply the same filter to millions of pixels in parallel
- **Simulations**: Model complex systems with thousands of interacting components

**Why These Examples Matter**: Each kernel demonstrates a fundamental parallel computing pattern that appears in real applications - from Instagram filters to weather prediction to training ChatGPT.

## 🎯 Available Kernels

### ✅ Working Examples

#### **1. Vector Addition** 🧮
**What it does**: Adds two arrays element by element - the "Hello World" of GPU programming.
**Why it matters**: Shows the simplest form of parallel computing where thousands of GPU cores work simultaneously, like having an army of calculators adding corresponding numbers from two lists.
**Use cases**: Foundation for all GPU operations, basic mathematical computations, data processing pipelines.

#### **2. Matrix Multiplication** 🔢
**What it does**: Multiplies two matrices together using advanced memory optimization techniques.
**Why it matters**: The backbone of machine learning, computer graphics, and scientific computing. GPUs can perform thousands of multiply-add operations simultaneously.
**Use cases**: Neural networks, 3D graphics transformations, solving systems of equations, image processing.

#### **3. Parallel Reduction** ⬇️
**What it does**: Efficiently finds the sum, maximum, or minimum value from a large array.
**Why it matters**: Demonstrates how to combine results from thousands of parallel threads without conflicts. Like having a tournament where winners advance to the next round.
**Use cases**: Statistical analysis, finding peaks in data, aggregating sensor readings, calculating totals.

#### **4. 2D Convolution** 🖼️
**What it does**: Applies filters to images (blur, sharpen, edge detection, etc.).
**Why it matters**: The foundation of image processing and computer vision. Shows how GPUs excel at processing pixels in parallel.
**Use cases**: Photo editing, medical imaging, computer vision, video processing, Instagram filters.

#### **5. Monte Carlo Simulation** 🎯
**What it does**: Uses random sampling to solve complex mathematical problems.
**Why it matters**: Like throwing millions of darts at a dartboard to calculate π. Shows GPU's power for statistical simulations with massive parallelism.
**Use cases**: Financial modeling, weather prediction, risk analysis, game AI, scientific research.

#### **6. Advanced Threading** 🧵
**What it does**: Shows how thousands of GPU threads coordinate and work together like a synchronized orchestra.

**Key Concepts Explained Simply**:
- **Warp-level Programming**: A "warp" is like a team of 32 GPU threads that always work in lockstep - imagine 32 people doing synchronized swimming, they must all do the same move at the same time
- **Thread Cooperation**: Like workers on an assembly line - each thread does part of the work and passes results to others
- **Barrier Synchronization**: Like saying "everyone wait here until the whole team is ready" - ensures threads don't get ahead of each other
- **Shared Memory**: Like a shared workspace where threads can leave notes for each other

**Real Examples Demonstrated**:
- **Producer-Consumer**: Some threads create data while others process it (like a chef cooking while a waiter serves)
- **Multi-stage Pipeline**: Breaking complex work into stages where each thread specializes (like an assembly line)
- **Warp Reduction**: Those 32-thread teams working together to combine their results super efficiently
- **Safe Communication**: How to pass data between threads without chaos or conflicts

**Why it matters**: Shows the sophisticated coordination patterns that make GPUs incredibly powerful - it's like the difference between 1000 people working randomly vs 1000 people working as a perfectly coordinated team.

**Use cases**: Complex algorithms, image/video processing pipelines, scientific simulations, any situation where you need threads to cooperate rather than just work independently.

#### **7. Dynamic Memory Management** 💾
**What it does**: Shows how to allocate and manage GPU memory during program execution.
**Why it matters**: Essential for applications that don't know memory requirements beforehand. Demonstrates safe GPU memory practices.
**Use cases**: Adaptive algorithms, dynamic data structures, memory-intensive applications.

#### **8. Advanced FFT (Fast Fourier Transform)** 📊
**What it does**: Converts signals between time and frequency domains using optimized algorithms.
**Why it matters**: Critical for signal processing, showing how GPUs accelerate complex mathematical transformations.
**Use cases**: Audio processing, image compression, wireless communications, scientific analysis.

#### **9. N-Body Simulation** 🌌
**What it does**: Simulates gravitational forces between particles (planets, stars, molecules).
**Why it matters**: Shows GPU's incredible power for physics simulations, computing forces between thousands of objects simultaneously.
**Use cases**: Astronomy simulations, molecular dynamics, game physics, scientific modeling.

### ⚠️ Platform Notes
- **Advanced FFT**: Fully functional with optimized implementations
- **N-Body Simulation**: Includes collision detection and force calculations
- **All kernels**: Support both AMD HIP and NVIDIA CUDA platforms

## 🖥️ Interactive GUI Features

The Qt-based GUI transforms GPU learning from intimidating code into an interactive experience:

### 🎯 **What You'll See**
- **Kernel Browser**: Choose from 9 different GPU examples with clear descriptions
- **Real-time Configuration**: Adjust data sizes, iterations, and parameters with sliders
- **Live Performance Metrics**: Watch execution times, memory bandwidth, and throughput
- **Educational Content**: Learn what each kernel does and why it matters
- **Visual Feedback**: Color-coded output showing success, errors, and performance data

### 🚀 **Why It's Useful**
- **Learning**: Understand GPU concepts without diving into complex code first
- **Experimentation**: Try different parameters and see immediate results
- **Benchmarking**: Compare performance across different configurations
- **Debugging**: Clear error messages and status information
- **Teaching**: Perfect for classroom demonstrations or self-study

### 🎮 **Getting Started**
```bash
./run.sh                           # Launch the GUI
# 1. Select a kernel (start with "Vector Addition")
# 2. Read the description to understand what it does
# 3. Adjust parameters if desired
# 4. Click "Run" and watch the magic happen!
```

**Pro Tip**: Start with "Vector Addition" to see the basics, then try "2D Convolution" to see real image processing, and "N-Body Simulation" for impressive physics!

## 🧪 Testing & Examples

### 🎮 Interactive Testing (Recommended)
The easiest way to explore the kernels is through the interactive GUI:
```bash
./run.sh                    # Launch GUI with auto-detected GPU platform
./run.sh -p hip            # Force AMD HIP platform
./run.sh -p cuda           # Force NVIDIA CUDA platform
```

The GUI provides:
- **Real-time parameter adjustment**: Change data sizes, iterations, and configurations
- **Performance monitoring**: See execution times and memory bandwidth
- **Educational descriptions**: Learn what each kernel does and why it matters
- **Error handling**: Clear feedback if something goes wrong

### 🔬 Command Line Testing
Run individual kernels directly for scripting or detailed analysis:

```bash
# Vector Addition - Add two 1-million element arrays
./build/bin/vector_addition 1000000

# Matrix Multiplication - Multiply two 512x512 matrices  
./build/bin/matrix_multiplication 512

# Parallel Reduction - Find sum of 10-million numbers
./build/bin/parallel_reduction 10000000

# 2D Convolution - Apply 5x5 filter to 1024x1024 image
./build/bin/convolution_2d 1024 5

# Monte Carlo - Calculate π using 1-million random samples
./build/bin/monte_carlo 1000000

# N-Body Simulation - Simulate 2048 particles for 100 steps
./build/bin/nbody_simulation 2048 100
```

### 🚀 Automated Testing
Comprehensive test suites for validation and benchmarking:
```bash
# Run all kernel tests with performance measurements
./scripts/testing/comprehensive_gui_test.sh

# Quick functionality check (great for CI/CD)
./scripts/testing/quick_gui_test.sh

# Test specific kernel executable detection
./scripts/testing/test_gui_kernel_detection.sh
```

## 📊 Performance

Each kernel includes multiple optimization strategies:
- **Naive implementations** for educational purposes
- **Shared memory optimizations** for better memory access
- **Memory coalescing** for improved bandwidth
- **Warp-level primitives** for efficient synchronization

## 🔍 Debugging

### Build Issues
1. Check prerequisites: `./scripts/build_working.sh --help`
2. Clean build: `./scripts/build_working.sh --clean`
3. Debug build: `./scripts/build_working.sh --debug`

### Runtime Issues
1. Check GPU availability: `rocm-smi` or `nvidia-smi`
2. Verify HIP/CUDA installation: `hipcc --version`
3. Check Qt installation: `pkg-config --exists Qt6Core`

## 📚 Documentation

- [Unified Build Summary](UNIFIED_BUILD_SUMMARY.md) - Detailed build system analysis
- [GPU Safety Guide](docs/GPU_SAFETY_GUIDE.md) - Best practices for GPU programming
- [Performance Optimization](docs/performance/optimization-guide.md) - Performance tuning tips
- [CUDA Fundamentals](docs/theory/cuda-fundamentals.md) - Theoretical background

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes
4. **Test** thoroughly with the provided scripts
5. **Submit** a pull request

### Development Guidelines
- Follow existing code style
- Add comprehensive tests
- Update documentation
- Ensure HIP/CUDA compatibility

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **AMD ROCm** team for HIP framework
- **NVIDIA** for CUDA platform
- **Qt** team for the GUI framework
- **Open source community** for inspiration and tools

---

**Note**: This project demonstrates various GPU programming techniques and is intended for educational and research purposes. Production use may require additional optimization and testing.
