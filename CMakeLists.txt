cmake_minimum_required(VERSION 3.20)
project(GPUKernelExamples)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform detection and configuration
option(USE_CUDA "Use NVIDIA CUDA" OFF)
option(USE_HIP "Use AMD ROCm/HIP" OFF)

# Auto-detect platform if not specified
if(NOT USE_CUDA AND NOT USE_HIP)
    # Try to find CUDA first
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(USE_CUDA ON)
        message(STATUS "Auto-detected NVIDIA CUDA platform")
    else()
        # Try to find HIP
        find_package(hip QUIET)
        if(hip_FOUND)
            set(USE_HIP ON)
            message(STATUS "Auto-detected AMD ROCm/HIP platform")
        else()
            message(FATAL_ERROR "No GPU platform found. Please install CUDA or ROCm.")
        endif()
    endif()
endif()

# Configure for CUDA platform
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Set CUDA architectures (adjust based on your target GPUs)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    
    # CUDA compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")
    
    set(GPU_PLATFORM "CUDA")
    set(GPU_RUNTIME_LIB CUDA::cudart)
    set(GPU_BLAS_LIB CUDA::cublas)
    
    message(STATUS "Configuring for NVIDIA CUDA platform")
endif()

# Configure for HIP platform  
if(USE_HIP)
    # Set up ROCm path
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm-6.2.2" CACHE PATH "Path to ROCm installation")
    endif()
    
    # Add ROCm to module and prefix paths
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
    list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip")
    
    # Find HIP package
    find_package(hip REQUIRED CONFIG PATHS ${ROCM_PATH}/lib/cmake/hip)
    find_package(rocblas QUIET CONFIG PATHS ${ROCM_PATH}/lib/cmake/rocblas)
    
    # Enable HIP language
    enable_language(HIP)
    
    # HIP compiler flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_AMD__")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fPIC")
    
    set(GPU_PLATFORM "HIP")
    set(GPU_RUNTIME_LIB hip::host)
    if(rocblas_FOUND)
        set(GPU_BLAS_LIB roc::rocblas)
    endif()
    
    message(STATUS "Configuring for AMD ROCm/HIP platform")
    message(STATUS "ROCm path: ${ROCM_PATH}")
endif()

# Common compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Option for profiling
option(ENABLE_PROFILING "Enable profiling support" OFF)
if(ENABLE_PROFILING)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --profile")
endif()

# Include directories
include_directories(src/common)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Common utilities library
add_library(cuda_common STATIC
    src/common/timer.cpp
    src/common/helper_functions.cpp
)

# Add platform-specific source files
if(USE_CUDA)
    target_sources(cuda_common PRIVATE src/common/cuda_utils.cu)
    target_link_libraries(cuda_common ${GPU_RUNTIME_LIB})
elseif(USE_HIP)
    target_sources(cuda_common PRIVATE src/common/hip_utils.cpp)
    target_link_libraries(cuda_common ${GPU_RUNTIME_LIB})
endif()

# Example 1: Vector Addition (has HIP version)
if(USE_CUDA)
    add_executable(vector_addition
        src/01_vector_addition/vector_addition.cu
        src/01_vector_addition/main.cpp
    )
elseif(USE_HIP)
    add_executable(vector_addition
        src/01_vector_addition/vector_addition_hip.hip
        src/01_vector_addition/main_hip.cpp
    )
endif()
target_link_libraries(vector_addition cuda_common ${GPU_RUNTIME_LIB})

# Note: Examples 2-5 don't have HIP versions yet, so they're skipped for HIP builds

# Example 7: Advanced Threading (has HIP version)
if(USE_CUDA)
    # Skip for now - no CUDA version
elseif(USE_HIP)
    add_executable(advanced_threading
        src/07_advanced_threading/advanced_threading_hip.hip
        src/07_advanced_threading/main_hip.cpp
    )
    target_link_libraries(advanced_threading cuda_common ${GPU_RUNTIME_LIB})
endif()

# Example 8: Dynamic Memory (has HIP version)
if(USE_CUDA)
    # Skip for now - no CUDA version
elseif(USE_HIP)
    add_executable(dynamic_memory
        src/08_dynamic_memory/dynamic_memory_hip.hip
        src/08_dynamic_memory/main_hip.cpp
    )
    target_link_libraries(dynamic_memory cuda_common ${GPU_RUNTIME_LIB})
endif()

# Example 9: Warp Primitives (has HIP version)
if(USE_CUDA)
    # Skip for now - no CUDA version
elseif(USE_HIP)
    add_executable(warp_primitives
        src/09_warp_primitives/warp_primitives_hip.hip
        src/09_warp_primitives/main_hip.cpp
    )
    target_link_libraries(warp_primitives cuda_common ${GPU_RUNTIME_LIB})
endif()

# Example 10: Advanced FFT (has HIP version)
if(USE_CUDA)
    # Skip for now - no CUDA version
elseif(USE_HIP)
    add_executable(advanced_fft
        src/10_advanced_fft/fft3d_hip.hip
    )
    target_link_libraries(advanced_fft cuda_common ${GPU_RUNTIME_LIB})
endif()

# Example 11: N-Body Simulation (has HIP version)
if(USE_CUDA)
    # Skip for now - no CUDA version
elseif(USE_HIP)
    add_executable(nbody_simulation
        src/11_nbody_simulation/nbody_hip.hip
    )
    target_link_libraries(nbody_simulation cuda_common ${GPU_RUNTIME_LIB})
endif()

# Install targets
if(USE_CUDA)
    install(TARGETS 
        vector_addition 
        matrix_multiplication 
        parallel_reduction 
        convolution_2d 
        monte_carlo
        DESTINATION bin
    )
elseif(USE_HIP)
    install(TARGETS 
        vector_addition 
        advanced_threading
        dynamic_memory
        warp_primitives
        advanced_fft
        nbody_simulation
        DESTINATION bin
    )
endif()

# Testing (optional)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# GUI (optional)
option(BUILD_GUI "Build GUI application" ON)
if(BUILD_GUI)
    add_subdirectory(gui)
endif()

# Print configuration info
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Profiling Enabled: ${ENABLE_PROFILING}")
message(STATUS "GUI Enabled: ${BUILD_GUI}")
