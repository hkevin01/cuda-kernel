# GPU Kernel Development Safety Guide

## ⚠️ CRITICAL SAFETY WARNING

**The Advanced Threading kernel (`advanced_threading`) causes SYSTEM-LEVEL GPU CRASHES that can:**
- **Crash your display driver** 
- **Lock up the entire system**
- **Cause corrupted display output** (red/pink/blue pixels, black screen)
- **Require hard reboot**

**This kernel has been DISABLED in the GUI for safety.**

## 🛡️ Safe Development Practices

### 1. **Hardware Protection**
- **Always start with small test data** (< 1000 elements)
- **Use timeouts** for all kernel tests: `timeout 5s ./kernel`
- **Monitor GPU temperature** during development
- **Have system monitoring tools ready** (htop, nvidia-smi, rocm-smi)

### 2. **Development Kit Recommendations**

#### **For AMD GPUs (Current Setup: RX 5600 XT):**
```bash
# Your current ROCm setup
ROCm 6.2.2 - ✅ Good for development
HIP runtime - ✅ Works well
hipcc compiler - ✅ Stable

# Recommended development approach:
export ROCM_PATH=/opt/rocm-6.2.2
export HIP_PLATFORM=amd
```

#### **For NVIDIA GPUs (Alternative):**
```bash
CUDA 12.x - Recommended for stability
CUDA 11.8 - More compatible with older hardware
Nsight tools - Excellent debugging
```

### 3. **Safe Testing Protocol**

#### **Step 1: Manual Testing**
```bash
# Test with minimal data first
timeout 3s ./build/bin/kernel_name 64

# Gradually increase if stable
timeout 5s ./build/bin/kernel_name 1024
timeout 10s ./build/bin/kernel_name 10000
```

#### **Step 2: Resource Monitoring**
```bash
# Monitor GPU during testing
watch -n 1 rocm-smi
# or for NVIDIA:
watch -n 1 nvidia-smi
```

#### **Step 3: GUI Testing**
```bash
# Use the safe launcher
./launch_working_gui.sh
# Only test kernels marked as "SAFE" in GUI
```

### 4. **Known Safe Kernels**
- ✅ **Vector Addition** - Safe, well tested
- ✅ **Dynamic Memory** - Safe for small inputs
- ✅ **Advanced FFT** - Generally stable
- ✅ **Warp Primitives** - Safe with proper bounds
- ✅ **N-Body Simulation** - Safe for reasonable particle counts

### 5. **Dangerous Patterns to Avoid**

#### **Memory Issues:**
- Infinite loops in kernels
- Unbounded memory allocation
- Race conditions without proper synchronization
- Grid-wide synchronization (`__syncgrid()`)

#### **GPU-Killing Code:**
```cpp
// DANGEROUS: Can crash GPU
while(1) { /* infinite loop in kernel */ }

// DANGEROUS: Massive memory allocation
malloc(SIZE_MAX);

// DANGEROUS: Grid synchronization without checks
cg::grid_group grid = cg::this_grid();
grid.sync(); // Can deadlock/crash
```

### 6. **Emergency Recovery**

#### **If System Freezes:**
1. **Magic SysRq keys:** `Alt + SysRq + F` (kill processes)
2. **SSH from another machine** if network still works
3. **Hard reboot** as last resort
4. **Check system logs** after reboot: `journalctl -b -1`

#### **GPU Recovery:**
```bash
# Reset GPU (AMD)
sudo systemctl restart gdm
# or
sudo modprobe -r amdgpu && sudo modprobe amdgpu

# Reset GPU (NVIDIA)  
sudo nvidia-smi --gpu-reset
```

### 7. **Debugging Crashed Kernels**

#### **Use Compute Sanitizer (NVIDIA):**
```bash
compute-sanitizer ./kernel_name
```

#### **Use ROCm Debugging (AMD):**
```bash
rocgdb ./kernel_name
HSA_ENABLE_DEBUG=1 ./kernel_name
```

#### **Kernel Profiling:**
```bash
# AMD
rocprof ./kernel_name

# NVIDIA
nsys profile ./kernel_name
```

### 8. **Recommended Development Environment**

#### **Hardware:**
- **Minimum 16GB RAM** for GPU development
- **Good cooling** for GPU during testing
- **Dual monitor setup** (in case primary display crashes)
- **UPS** to prevent data loss during crashes

#### **Software:**
- **Linux with latest drivers**
- **Version control** (git) - commit frequently
- **Remote access** (SSH) for emergency recovery
- **System monitoring tools**

### 9. **Build System Recommendations**

#### **Use Conservative Flags:**
```cmake
# Safe compilation flags
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O1 -g")  # Lower optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O1 -g -lineinfo")

# Enable bounds checking in debug builds
set(CMAKE_HIP_FLAGS_DEBUG "-O0 -g -DDEBUG")
```

#### **Staged Building:**
```bash
# Build one kernel at a time
make vector_addition
make dynamic_memory  
# Test before building more
```

### 10. **Current Project Status**

#### **✅ Safe to Use:**
- Vector Addition
- Dynamic Memory (small inputs)
- Advanced FFT
- Warp Primitives
- N-Body Simulation

#### **⚠️ Needs Work:**
- Matrix Multiplication (not built yet)
- Parallel Reduction (not built yet) 
- 2D Convolution (not built yet)
- Monte Carlo (not built yet)

#### **🚫 DANGEROUS - DO NOT RUN:**
- **Advanced Threading** - System crashes

## Summary

**Your AMD RX 5600 XT with ROCm 6.2.2 is a good setup for GPU development.** The main issue is the specific advanced threading kernel implementation. Focus on:

1. **Building the missing kernels safely**
2. **Testing with small inputs first**
3. **Using proper timeouts and monitoring**
4. **Avoiding complex synchronization patterns**

Keep this safety guide handy and always test incrementally!
