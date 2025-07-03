#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                  \
    do                                                                    \
    {                                                                     \
        cudaError_t error = call;                                         \
        if (error != cudaSuccess)                                         \
        {                                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// CUDA kernel error checking
#define CUDA_CHECK_KERNEL()                                                               \
    do                                                                                    \
    {                                                                                     \
        cudaError_t error = cudaGetLastError();                                           \
        if (error != cudaSuccess)                                                         \
        {                                                                                 \
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                                      \
        }                                                                                 \
        CUDA_CHECK(cudaDeviceSynchronize());                                              \
    } while (0)

// Device information
struct DeviceInfo
{
    int device_id;
    std::string name;
    size_t global_memory;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int warp_size;
    int compute_capability_major;
    int compute_capability_minor;
};

// Utility functions
void printDeviceInfo(int device_id = 0);
DeviceInfo getDeviceInfo(int device_id = 0);
void setOptimalDevice();
size_t getAvailableMemory();

// Memory management helpers
template <typename T>
void allocateHostMemory(T **ptr, size_t size)
{
    CUDA_CHECK(cudaMallocHost(ptr, size * sizeof(T)));
}

template <typename T>
void allocateDeviceMemory(T **ptr, size_t size)
{
    CUDA_CHECK(cudaMalloc(ptr, size * sizeof(T)));
}

template <typename T>
void copyHostToDevice(T *d_ptr, const T *h_ptr, size_t size)
{
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void copyDeviceToHost(T *h_ptr, const T *d_ptr, size_t size)
{
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void freeHostMemory(T *ptr)
{
    if (ptr)
    {
        CUDA_CHECK(cudaFreeHost(ptr));
    }
}

template <typename T>
void freeDeviceMemory(T *ptr)
{
    if (ptr)
    {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// Grid and block dimension helpers
dim3 calculateGridSize(int total_threads, int block_size);
dim3 calculate2DGridSize(int width, int height, int block_x, int block_y);

// Performance helpers
double calculateBandwidth(size_t bytes, float time_ms);
double calculateGFlops(size_t operations, float time_ms);

// Random number generation
void generateRandomFloats(float *data, size_t size, float min_val = 0.0f, float max_val = 1.0f);
void generateRandomInts(int *data, size_t size, int min_val = 0, int max_val = 100);

// Verification helpers
template <typename T>
bool verifyResults(const T *expected, const T *actual, size_t size, T tolerance = static_cast<T>(1e-5))
{
    for (size_t i = 0; i < size; ++i)
    {
        if (std::abs(expected[i] - actual[i]) > tolerance)
        {
            std::cerr << "Verification failed at index " << i
                      << ": expected " << expected[i]
                      << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Timing utilities
class CudaTimer
{
public:
    CudaTimer();
    ~CudaTimer();

    void start();
    float stop(); // Returns elapsed time in milliseconds

private:
    cudaEvent_t start_event, stop_event;
};
