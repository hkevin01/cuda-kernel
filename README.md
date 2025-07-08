# GPU Kernel Project

A comprehensive collection of GPU kernels implemented in HIP/CUDA with a Qt-based GUI for interactive testing and performance analysis.

## 🚀 Quick Start

### Prerequisites
- **HIP/ROCm** (for AMD GPUs) or **CUDA** (for NVIDIA GPUs)
- **Qt6** development packages
- **CMake** 3.16 or later
- **C++14** compatible compiler

### Build and Run
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
├── src/                    # Kernel source code
│   ├── 01_vector_addition/     # Basic vector operations
│   ├── 02_matrix_multiplication/ # Linear algebra
│   ├── 03_parallel_reduction/   # Reduction algorithms
│   ├── 04_convolution_2d/       # Image processing
│   ├── 05_monte_carlo/          # Statistical computation
│   ├── 07_advanced_threading/   # Thread synchronization
│   ├── 08_dynamic_memory/       # Memory management
│   ├── 11_nbody_simulation/     # N-body physics
│   └── common/                  # Shared utilities
├── gui/                    # Qt-based GUI application
├── tests/                  # Unit test source code
├── docs/                   # Documentation and status files
├── logs/                   # Runtime and test logs
├── scripts/               # Organized build and utility scripts
│   ├── build/                  # Build scripts
│   ├── testing/               # Test scripts
│   ├── gui/                   # GUI launch scripts
│   └── verification/          # Verification scripts
├── build*/                # Build output directories
└── run.sh                 # Main launcher script
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

## 🎯 Available Kernels

### ✅ Working Kernels
- **Vector Addition**: Basic element-wise operations
- **Matrix Multiplication**: Linear algebra with multiple optimizations
- **Parallel Reduction**: Efficient reduction algorithms
- **2D Convolution**: Image processing with shared memory
- **Monte Carlo**: Statistical computation examples
- **Advanced Threading**: Safe thread synchronization patterns
- **Dynamic Memory**: Memory management and coalescing

### ⚠️ Known Issues
- **Warp Primitives**: HIP compatibility issues
- **Advanced FFT**: Missing dependencies
- **N-Body Simulation**: Requires additional libraries

## 🖥️ GUI Features

The Qt-based GUI provides:
- **Interactive kernel testing** with parameter adjustment
- **Performance benchmarking** with timing measurements
- **Result visualization** for kernel outputs
- **Batch testing** capabilities
- **Error reporting** and debugging information

### Running the GUI
```bash
./build_gui/bin/gpu_kernel_gui
```

## 🧪 Testing

### Individual Kernel Tests
```bash
# Test vector addition
./build_simple/bin/vector_addition 1000

# Test matrix multiplication
./build_simple/bin/matrix_multiplication 256

# Test parallel reduction
./build_simple/bin/parallel_reduction 10000
```

### Comprehensive Testing
```bash
# Run all tests
./scripts/testing/comprehensive_gui_test.sh

# Quick GUI test
./scripts/testing/quick_gui_test.sh
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
