#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hip/hip_runtime.h>
#include "hip_utils.h"
#include <iostream>
#include <iomanip>

void initializeHIP()
{
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "No HIP-capable devices found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Set device 0 as default
    HIP_CHECK(hipSetDevice(0));

    std::cout << "HIP initialized successfully with " << deviceCount << " device(s)" << std::endl;
}

void printHIPDeviceInfo()
{
    int device;
    HIP_CHECK(hipGetDevice(&device));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    std::cout << "\n=== HIP Device Information ===" << std::endl;
    std::cout << "Device " << device << ": " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total global memory: " << std::fixed << std::setprecision(1)
              << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Number of multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;

    // Calculate theoretical bandwidth
    float bandwidth = getHIPTheoreticalBandwidth();
    std::cout << "Theoretical bandwidth: " << std::fixed << std::setprecision(1)
              << bandwidth << " GB/s" << std::endl;
    std::cout << "================================" << std::endl;
}

float getHIPTheoreticalBandwidth()
{
    int device;
    HIP_CHECK(hipGetDevice(&device));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    // Bandwidth = (Memory Clock Rate * Memory Bus Width * 2) / 8
    // Factor of 2 for DDR, divide by 8 to convert bits to bytes
    float bandwidth = (float)(prop.memoryClockRate * 2 * prop.memoryBusWidth) / (8 * 1e6);
    return bandwidth;
}

float calculateBandwidth(size_t bytes, float time_ms)
{
    if (time_ms <= 0.0f)
        return 0.0f;
    return (bytes / (time_ms * 1e-3)) / 1e9; // GB/s
}

int calculateOptimalBlockSize(const void *kernel, size_t dynamicSharedMem)
{
    int minGridSize, blockSize;

    // HIP doesn't have direct occupancy calculator, so we use a heuristic
    // Based on typical optimal block sizes for AMD GPUs
    hipDeviceProp_t prop;
    int device;
    HIP_CHECK(hipGetDevice(&device));
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    // For AMD GPUs, wavefront size is typically 64
    // Good block sizes are usually multiples of wavefront size
    int wavefrontSize = prop.warpSize; // On AMD, this is the wavefront size

    // Start with a reasonable default and ensure it's a multiple of wavefront size
    blockSize = 256;
    if (blockSize % wavefrontSize != 0)
    {
        blockSize = ((blockSize + wavefrontSize - 1) / wavefrontSize) * wavefrontSize;
    }

    // Ensure we don't exceed maximum threads per block
    if (blockSize > prop.maxThreadsPerBlock)
    {
        blockSize = (prop.maxThreadsPerBlock / wavefrontSize) * wavefrontSize;
    }

    return blockSize;
}
