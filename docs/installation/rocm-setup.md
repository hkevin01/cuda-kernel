# ROCm Installation Guide

This guide helps you set up AMD ROCm for running GPU kernels on AMD graphics cards.

## System Requirements

### Supported Operating Systems
- Ubuntu 20.04, 22.04, 24.04 LTS
- RHEL/CentOS 8, 9
- SLES 15

### Supported AMD GPUs
- **RDNA/RDNA2/RDNA3**: RX 6000, RX 7000 series
- **Vega**: RX Vega 56/64, Radeon VII
- **Polaris**: RX 580, RX 570 (limited support)
- **MI Series**: Professional compute cards

## Installation Methods

### Method 1: Package Manager (Recommended)

#### Ubuntu/Debian
```bash
# Add ROCm repository
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.2.0/ jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Update and install ROCm
sudo apt update
sudo apt install rocm-dev rocm-libs rocblas rocfft rocrand rocsparse

# Add user to render group
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# Reboot required
sudo reboot
```

#### RHEL/CentOS/Fedora
```bash
# Add ROCm repository
sudo tee /etc/yum.repos.d/rocm.repo <<EOF
[rocm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/rhel9/6.2.0/main
enabled=1
priority=50
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

# Install ROCm
sudo dnf install rocm-dev rocm-libs rocblas rocfft rocrand rocsparse

# Add user to render group
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER

# Reboot required
sudo reboot
```

### Method 2: AMD Installer Script
```bash
# Download and run the installer
curl -O https://repo.radeon.com/amdgpu-install/6.2.0/ubuntu/jammy/amdgpu-install_6.2.60200-1_all.deb
sudo dpkg -i amdgpu-install_6.2.60200-1_all.deb
sudo apt update

# Install with ROCm support
sudo amdgpu-install --usecase=rocm
```

## Verification

### Check ROCm Installation
```bash
# Check ROCm version
rocm-smi --version

# List available devices
rocm-smi

# Check HIP installation
hipconfig

# Test HIP compilation
echo '#include <hip/hip_runtime.h>
#include <iostream>
int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " HIP devices" << std::endl;
    return 0;
}' > test_hip.cpp

hipcc test_hip.cpp -o test_hip
./test_hip
```

### Expected Output
```
Found 1 HIP devices
```

## Environment Setup

### Add to ~/.bashrc or ~/.zshrc
```bash
# ROCm environment variables
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# HIP environment variables
export HIP_PLATFORM=amd
export HIP_COMPILER=clang
export HIP_RUNTIME=rocclr
```

### Reload environment
```bash
source ~/.bashrc
# OR
source ~/.zshrc
```

## Building the Project with ROCm

### Clone and build
```bash
git clone <repository-url>
cd cuda-kernel

# Build with HIP support
mkdir build && cd build
cmake -DUSE_HIP=ON ..
make -j$(nproc)
```

### Run examples
```bash
# Vector addition
./01_vector_addition

# Matrix multiplication  
./02_matrix_multiplication

# Parallel reduction
./03_parallel_reduction

# 2D Convolution
./04_convolution_2d

# Monte Carlo simulation
./05_monte_carlo
```

## Troubleshooting

### Common Issues

#### Permission Denied
```bash
# Add user to required groups
sudo usermod -a -G render $USER
sudo usermod -a -G video $USER
sudo reboot
```

#### GPU Not Detected
```bash
# Check if GPU is visible
lspci | grep -i amd
rocm-smi

# Check kernel module
lsmod | grep amdgpu
```

#### Compilation Errors
```bash
# Install development packages
sudo apt install build-essential cmake

# Check HIP installation
which hipcc
hipcc --version
```

#### Performance Issues
```bash
# Check GPU utilization
watch -n 1 rocm-smi

# Enable GPU performance mode
echo performance | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
```

### ROCm Documentation
- [ROCm Installation Guide](https://docs.amd.com/bundle/ROCm-Installation-Guide-v6.2/page/How_to_Install_ROCm.html)
- [HIP Programming Guide](https://docs.amd.com/bundle/HIP-Programming-Guide-v6.2/page/Introduction_to_HIP_Programming_Guide.html)
- [ROCm Developer Tools](https://docs.amd.com/bundle/ROCm-Developer-Tools/page/ROCm-Developer-Tools.html)

### Community Support
- [ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)
- [AMD Developer Community](https://community.amd.com/t5/rocm/ct-p/amd-rocm)
- [HIP Documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/)

## Performance Tuning for AMD GPUs

### Memory Optimization
```cpp
// Use appropriate memory patterns for AMD architecture
// AMD GPUs have different memory hierarchy than NVIDIA
```

### Wavefront Considerations
```cpp
// AMD uses 64-thread wavefronts (vs 32-thread NVIDIA warps)
// Adjust block sizes accordingly
int blockSize = 256; // Good for AMD (4 wavefronts)
```

### ROCm Profiling Tools
```bash
# Install profiling tools
sudo apt install rocprofiler-dev roctracer-dev

# Profile application
rocprof --stats ./your_application
```
