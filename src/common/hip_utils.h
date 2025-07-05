#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#define HIP_ENABLE_WARP_SYNC_BUILTINS
#include <hip/hip_runtime.h>
#include <iostream>
#include <string>

// HIP error checking macro
#define HIP_CHECK(call)                                                 \
    do                                                                  \
    {                                                                   \
        hipError_t err = call;                                          \
        if (err != hipSuccess)                                          \
        {                                                               \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << hipGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Initialize HIP device
void initializeHIP();

// Print device information
void printHIPDeviceInfo();

// Get theoretical memory bandwidth
float getHIPTheoreticalBandwidth();

// Device memory management wrappers
template <typename T>
void hipMallocWrapper(T **ptr, size_t size)
{
    HIP_CHECK(hipMalloc(ptr, size));
}

template <typename T>
void hipMemcpyWrapper(T *dst, const T *src, size_t size, hipMemcpyKind kind)
{
    HIP_CHECK(hipMemcpy(dst, src, size, kind));
}

template <typename T>
void hipFreeWrapper(T *ptr)
{
    HIP_CHECK(hipFree(ptr));
}

// Synchronization wrappers
inline void hipSynchronizeWrapper()
{
    HIP_CHECK(hipDeviceSynchronize());
}

// Memory bandwidth calculation
float calculateBandwidth(size_t bytes, float time_ms);

// Occupancy calculator
int calculateOptimalBlockSize(const void *kernel, size_t dynamicSharedMem = 0);

#endif // HIP_UTILS_H
