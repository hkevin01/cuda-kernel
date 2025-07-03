cmake_minimum_required(VERSION 3.20)
project(CudaKernelExamples CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust based on your target GPUs)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

# Release flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -DNDEBUG")

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
    src/common/cuda_utils.cu
    src/common/timer.cpp
    src/common/helper_functions.cpp
)

target_link_libraries(cuda_common CUDA::cudart)

# Example 1: Vector Addition
add_executable(vector_addition
    src/01_vector_addition/vector_addition.cu
    src/01_vector_addition/main.cpp
)
target_link_libraries(vector_addition cuda_common CUDA::cudart)

# Example 2: Matrix Multiplication
add_executable(matrix_multiplication
    src/02_matrix_multiplication/matrix_mul.cu
    src/02_matrix_multiplication/main.cpp
)
target_link_libraries(matrix_multiplication cuda_common CUDA::cudart CUDA::cublas)

# Example 3: Parallel Reduction
add_executable(parallel_reduction
    src/03_parallel_reduction/reduction.cu
    src/03_parallel_reduction/main.cpp
)
target_link_libraries(parallel_reduction cuda_common CUDA::cudart)

# Example 4: 2D Convolution
add_executable(convolution_2d
    src/04_convolution_2d/convolution.cu
    src/04_convolution_2d/main.cpp
)
target_link_libraries(convolution_2d cuda_common CUDA::cudart)

# Example 5: Monte Carlo Simulation
add_executable(monte_carlo
    src/05_monte_carlo/monte_carlo.cu
    src/05_monte_carlo/main.cpp
)
target_link_libraries(monte_carlo cuda_common CUDA::cudart CUDA::curand)

# Install targets
install(TARGETS 
    vector_addition 
    matrix_multiplication 
    parallel_reduction 
    convolution_2d 
    monte_carlo
    DESTINATION bin
)

# Testing (optional)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration info
message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Profiling Enabled: ${ENABLE_PROFILING}")
