# GPU Kernel Programming Examples (CUDA + ROCm)

A comprehensive collection of GPU kernel examples demonstrating essential parallel computing techniques for modern GPU programming. This project supports both NVIDIA CUDA and AMD ROCm platforms, focusing on the most in-demand GPU programming skills required in industry today.

## 📸 Application Screenshots

### GPU Kernel GUI Application
This project includes a modern Qt-based GUI application for running and analyzing GPU kernels with real-time performance monitoring.

![GPU Kernel GUI - Main Interface](screenshots/gui_main.png)
*Main interface showing comprehensive kernel selection, configuration panel, and tabbed workspace for Examples, Results, Performance monitoring, and Tests*

### Key GUI Features Shown:
- **Kernel Selection Panel**: Complete list of available kernels from basic vector operations to advanced 3D FFT and N-body simulations
- **Dynamic Configuration**: Real-time parameter adjustment for iterations, data size, and platform selection (CUDA/HIP)
- **Tabbed Interface**: Organized workspace with dedicated tabs for kernel execution, examples, results analysis, performance monitoring, and testing
- **Cross-Platform Support**: Native HIP/ROCm integration with seamless platform switching
- **Professional UI**: Modern Qt interface with comprehensive menu bar and toolbar for all GPU development needs

> **Status**: ✅ GUI application is fully functional and operational. Library conflicts have been resolved!

## 🎯 Project Overview

This repository contains a comprehensive GPU kernel programming framework with both command-line examples and a modern Qt-based GUI application. The project includes carefully selected GPU kernel examples that represent the core skills needed for GPU development jobs in 2025, supporting both NVIDIA and AMD hardware:

### Core Examples
1. **Vector Addition** - Foundation parallel programming
2. **Advanced Threading** - Synchronization and lock-free algorithms
3. **Warp Primitives** - Cooperative groups and advanced warp operations
4. **Advanced 3D FFT** - Complex signal processing algorithms
5. **N-body Simulation** - Physics simulation with spatial optimizations

### GUI Application Features
- **Interactive Kernel Launcher**: Run and test kernels with customizable parameters
- **Comprehensive Kernel Library**: Includes both basic and advanced examples:
  - **Basic Kernels**: Vector Addition, Matrix Multiplication, Parallel Reduction, 2D Convolution, Monte Carlo
  - **Advanced Kernels**: Advanced FFT, Advanced Threading, Dynamic Memory, Warp Primitives, 3D FFT, N-Body Simulation
- **Real-time Performance Monitoring**: Live charts showing execution time, memory usage, and throughput
- **Result Visualization**: Advanced viewers for kernel outputs and comparisons
- **Cross-platform Support**: Built with Qt for Windows, Linux, and macOS with native HIP/ROCm integration

## 🖥️ Platform Support

- **NVIDIA GPUs**: CUDA toolkit (11.0+)
- **AMD GPUs**: ROCm platform (5.0+)
- **Cross-platform**: HIP (Heterogeneous Interface for Portability)

## 📁 Project Structure

```
cuda-kernel/
├── src/                              # Source code
│   ├── 01_vector_addition/           # Basic parallel vector operations
│   ├── 07_advanced_threading/        # Advanced synchronization techniques
│   ├── 08_dynamic_memory/            # Dynamic GPU memory management
│   ├── 09_warp_primitives/           # Warp-level operations and cooperative groups
│   ├── 10_advanced_fft/              # 3D Fast Fourier Transform implementation
│   ├── 11_nbody_simulation/          # N-body physics simulation
│   └── common/                       # Shared utilities and headers
├── gui/                              # Qt-based GUI application
│   ├── mainwindow.cpp/.h/.ui         # Main application window
│   ├── kernel_runner.cpp/.h          # Kernel execution management
│   ├── performance_widget.cpp/.h     # Performance monitoring
│   ├── result_viewer.cpp/.h          # Results visualization
│   └── resources.qrc                 # GUI resources and icons
├── docs/                             # Documentation
│   ├── theory/                       # Theoretical background
│   ├── performance/                  # Performance analysis
│   └── api/                          # API documentation
├── scripts/                          # Build and utility scripts
│   ├── build.sh                      # Build automation
│   ├── profile.sh                    # Performance profiling
│   └── test.sh                       # Testing automation
├── build_hip/                        # HIP/ROCm build artifacts
├── build_gui/                        # GUI application build
├── .github/                          # GitHub Actions CI/CD
│   └── workflows/
├── .copilot/                         # GitHub Copilot configurations
├── launch_gui.sh                     # GUI launcher script
└── CMakeLists.txt                   # Build configuration
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

**Command Line Examples**
```bash
# From build directory
./01_vector_addition_hip
./07_advanced_threading_hip
./09_warp_primitives_simplified_hip
```

**GUI Application**
```bash
# Build the GUI (requires Qt5)
cd gui
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run the GUI application (fully functional)
./bin/gpu_kernel_gui

# Or use the launcher script (handles environment setup)
cd /path/to/cuda-kernel
./launch_gui.sh
```

> **GUI Requirements**: Qt5 development libraries, OpenGL support
> **Status**: ✅ Fully operational with HIP/ROCm platform support

## 📋 Examples Overview

### 1. Vector Addition
**File**: `src/01_vector_addition/`
- **Concept**: Basic GPU parallelization
- **Industry Use**: Foundation for all GPU programming
- **Skills**: Thread indexing, memory management, kernel launch

### 2. Advanced Threading
**File**: `src/07_advanced_threading/`
- **Concept**: Synchronization and lock-free algorithms
- **Industry Use**: High-performance concurrent systems
- **Skills**: Atomics, memory fences, cooperative groups

### 3. Warp Primitives
**File**: `src/09_warp_primitives/`
- **Concept**: Warp-level operations and collective algorithms
- **Industry Use**: Deep learning kernels, high-performance computing
- **Skills**: Shuffle operations, vote functions, cooperative groups

### 4. Advanced 3D FFT
**File**: `src/10_advanced_fft/`
- **Concept**: Complex signal processing algorithms
- **Industry Use**: Scientific computing, signal processing, quantum computing
- **Skills**: Shared memory optimization, tiled algorithms, complex mathematics

### 5. N-body Simulation
**File**: `src/11_nbody_simulation/`
- **Concept**: Physics simulation with spatial optimizations
- **Industry Use**: Game development, scientific simulation, molecular dynamics
- **Skills**: Spatial data structures, memory coalescing, performance optimization

### 6. Dynamic Memory Management
**File**: `src/08_dynamic_memory/`
- **Concept**: GPU memory allocation and management
- **Industry Use**: Dynamic algorithms, graph processing, adaptive algorithms
- **Skills**: GPU malloc/free, memory pools, dynamic parallelism

## 🎯 Industry Relevance

These examples cover the most sought-after GPU programming skills in 2025:

- **AI/Machine Learning**: Advanced memory optimization, warp-level operations
- **High-Performance Computing**: 3D FFT, N-body simulations, advanced threading
- **Game Development**: Physics simulations, real-time rendering algorithms
- **Scientific Computing**: Signal processing, molecular dynamics, quantum computing
- **Financial Technology**: Monte Carlo methods, parallel algorithms, optimization
- **Computer Vision**: Advanced convolutions, image processing pipelines

## 🖥️ GUI Features

The included Qt-based GUI application provides:

### 🚀 Interactive Kernel Execution
- Select and run any available kernel example
- Customize kernel parameters and input sizes
- Real-time execution status and error reporting

### 📊 Performance Monitoring
- Live charts showing execution time and memory usage
- Throughput analysis and bandwidth utilization
- Comparison tools for different kernel configurations

### 📈 Result Visualization
- Output data visualization with customizable views
- Performance metrics and statistical analysis
- Export capabilities for reports and presentations

### 🔧 Development Tools
- Integrated profiling and debugging information
- Code viewing and editing capabilities
- Build system integration

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

## 🔧 Troubleshooting

### GUI Application
The GUI application has been successfully tested and is fully operational with HIP/ROCm on AMD hardware.

**To run the GUI:**
```bash
# Method 1: Direct execution
./build_gui/bin/gpu_kernel_gui

# Method 2: Using the launcher script (recommended)
./launch_gui.sh
```

**Previous Library Conflicts (RESOLVED):**
If you previously encountered symbol lookup errors, these have been resolved through:
- Proper Qt library configuration
- Correct LD_LIBRARY_PATH settings
- System-native library usage (avoiding snap conflicts)

### AMD ROCm Setup
For optimal ROCm performance:
```bash
# Add user to render group
sudo usermod -a -G render $USER

# Set ROCm paths
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
```

### CUDA Compute Capability
Ensure your GPU supports the required compute capability:
```bash
# Check GPU info
nvidia-smi
# or
rocm-smi
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

### Official Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [ROCm Documentation](https://rocmdocs.amd.com/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)

### Developer Resources
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/)
- [AMD GPU Development](https://developer.amd.com/resources/rocm-resources/)
- [Qt Documentation](https://doc.qt.io/) for GUI development

### Project Documentation
- `docs/theory/` - Mathematical background for each algorithm
- `docs/performance/` - Performance analysis and optimization guides  
- `docs/api/` - API documentation for the project components
- Each example directory contains detailed README files with implementation notes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA CUDA Samples repository for inspiration
- CUDA community for best practices
- Contributors and maintainers

---

**Built with ❤️ for the CUDA community**
