#ifndef GPU_UTILS_H
#define GPU_UTILS_H

// Platform detection and unified GPU programming interface
// Supports both CUDA (NVIDIA) and HIP (AMD) platforms

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
// AMD ROCm/HIP platform
#define GPU_PLATFORM_HIP
#include "hip_utils.h"
#include <hip/hip_runtime.h>

// Unified API macros for HIP
#define GPU_CHECK(call) HIP_CHECK(call)
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuGetLastError hipGetLastError

// HIP-specific functions
inline void initializeGPU() { initializeHIP(); }
inline void printGPUDeviceInfo() { printHIPDeviceInfo(); }
inline float getGPUTheoreticalBandwidth() { return getHIPTheoreticalBandwidth(); }

#define GPU_PLATFORM_NAME "AMD ROCm/HIP"

#elif defined(__CUDACC__) || defined(CUDA_VERSION)
// NVIDIA CUDA platform
#define GPU_PLATFORM_CUDA
#include "cuda_utils.h"
#include <cuda_runtime.h>

// Unified API macros for CUDA
#define GPU_CHECK(call) CUDA_CHECK(call)
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceProp_t cudaDeviceProp_t
#define gpuGetLastError cudaGetLastError

// CUDA-specific functions
inline void initializeGPU() { initializeCUDA(); }
inline void printGPUDeviceInfo() { printDeviceInfo(); }
inline float getGPUTheoreticalBandwidth() { return getTheoreticalBandwidth(); }

#define GPU_PLATFORM_NAME "NVIDIA CUDA"

#else
#error "No supported GPU platform detected. Please compile with CUDA or HIP."
#endif

// Common GPU programming utilities that work across platforms
#include <iostream>
#include <string>

// Platform-agnostic helper functions
inline std::string getGPUPlatformName()
{
    return GPU_PLATFORM_NAME;
}

inline void checkGPUPlatform()
{
    std::cout << "Using GPU platform: " << getGPUPlatformName() << std::endl;
}

// Template wrapper for unified memory management
template <typename T>
void allocateGPUMemory(T **ptr, size_t count)
{
    GPU_CHECK(gpuMalloc(ptr, count * sizeof(T)));
}

template <typename T>
void freeGPUMemory(T *ptr)
{
    GPU_CHECK(gpuFree(ptr));
}

template <typename T>
void copyToGPU(T *dst, const T *src, size_t count)
{
    GPU_CHECK(gpuMemcpy(dst, src, count * sizeof(T), gpuMemcpyHostToDevice));
}

template <typename T>
void copyFromGPU(T *dst, const T *src, size_t count)
{
    GPU_CHECK(gpuMemcpy(dst, src, count * sizeof(T), gpuMemcpyDeviceToHost));
}

inline void synchronizeGPU()
{
    GPU_CHECK(gpuDeviceSynchronize());
}

#endif // GPU_UTILS_H
