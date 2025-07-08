cmake_minimum_required(VERSION 3.16)
project(GPUKernelProject VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Qt6 REQUIRED COMPONENTS Core Widgets)
find_package(HIP REQUIRED)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Common include directories
include_directories(${CMAKE_SOURCE_DIR}/src/common)

# Common source files
set(COMMON_SOURCES
    src/common/hip_utils.cpp
    src/common/timer.cpp
    src/common/helper_functions.cpp
)

# Build only the working kernels for now
# Vector Addition (simple, known to work)
add_executable(vector_addition 
    src/01_vector_addition/main_hip.cpp 
    src/01_vector_addition/vector_addition_hip.hip 
    ${COMMON_SOURCES}
)
set_target_properties(vector_addition PROPERTIES HIP_CXX_FLAGS "-O3 -std=c++14")
target_link_libraries(vector_addition hip::host)

# Advanced Threading (safe version)
add_executable(advanced_threading 
    src/07_advanced_threading/main_hip_safe.cpp 
    src/07_advanced_threading/advanced_threading_hip_safe.hip 
    ${COMMON_SOURCES}
)
set_target_properties(advanced_threading PROPERTIES HIP_CXX_FLAGS "-O3 -std=c++14")
target_link_libraries(advanced_threading hip::host)

# GUI Application
set(GUI_SOURCES
    gui/main.cpp
    gui/mainwindow.cpp
    gui/kernel_runner.cpp
    gui/performance_widget.cpp
    gui/result_viewer.cpp
    gui/test_runner.cpp
    gui/example_tabs.cpp
    gui/resources.qrc
)

# Create GUI executable
add_executable(gpu_kernel_gui ${GUI_SOURCES})

# Link Qt libraries
target_link_libraries(gpu_kernel_gui 
    Qt6::Core 
    Qt6::Widgets
)

# Set GUI properties
set_target_properties(gpu_kernel_gui PROPERTIES
    OUTPUT_NAME gpu_kernel_gui
)

# Enable automoc for Qt
set_target_properties(gpu_kernel_gui PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
)

# Install targets
install(TARGETS gpu_kernel_gui vector_addition advanced_threading
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Print build information
message(STATUS "Building ${PROJECT_NAME} version ${PROJECT_VERSION}")
message(STATUS "Kernels to build: vector_addition, advanced_threading")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
