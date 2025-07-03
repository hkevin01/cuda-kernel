#include "cuda_utils.h"
#include <random>
#include <algorithm>
#include <cmath>

void printDeviceInfo(int device_id) {
    DeviceInfo info = getDeviceInfo(device_id);
    
    std::cout << "=== CUDA Device Information ===" << std::endl;
    std::cout << "Device ID: " << info.device_id << std::endl;
    std::cout << "Device Name: " << info.name << std::endl;
    std::cout << "Global Memory: " << info.global_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << info.multiprocessor_count << std::endl;
    std::cout << "Max Threads per Block: " << info.max_threads_per_block << std::endl;
    std::cout << "Max Threads per MP: " << info.max_threads_per_multiprocessor << std::endl;
    std::cout << "Warp Size: " << info.warp_size << std::endl;
    std::cout << "Compute Capability: " << info.compute_capability_major 
              << "." << info.compute_capability_minor << std::endl;
    std::cout << "===============================" << std::endl;
}

DeviceInfo getDeviceInfo(int device_id) {
    DeviceInfo info;
    cudaDeviceProp prop;
    
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    info.device_id = device_id;
    info.name = std::string(prop.name);
    info.global_memory = prop.totalGlobalMem;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    info.warp_size = prop.warpSize;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    
    return info;
}

void setOptimalDevice() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        exit(1);
    }
    
    // Find device with highest compute capability
    int best_device = 0;
    int best_major = 0, best_minor = 0;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        if (prop.major > best_major || (prop.major == best_major && prop.minor > best_minor)) {
            best_device = i;
            best_major = prop.major;
            best_minor = prop.minor;
        }
    }
    
    CUDA_CHECK(cudaSetDevice(best_device));
    std::cout << "Using CUDA device " << best_device << std::endl;
}

size_t getAvailableMemory() {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    return free_mem;
}

dim3 calculateGridSize(int total_threads, int block_size) {
    return dim3((total_threads + block_size - 1) / block_size);
}

dim3 calculate2DGridSize(int width, int height, int block_x, int block_y) {
    return dim3((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);
}

double calculateBandwidth(size_t bytes, float time_ms) {
    return (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0); // GB/s
}

double calculateGFlops(size_t operations, float time_ms) {
    return (operations / 1e9) / (time_ms / 1000.0); // GFLOPS
}

void generateRandomFloats(float* data, size_t size, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

void generateRandomInts(int* data, size_t size, int min_val, int max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

// CUDA Timer implementation
CudaTimer::CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
}

void CudaTimer::start() {
    CUDA_CHECK(cudaEventRecord(start_event));
}

float CudaTimer::stop() {
    CUDA_CHECK(cudaEventRecord(stop_event));
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    
    float elapsed_time;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
    
    return elapsed_time;
}
