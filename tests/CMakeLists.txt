# Test suite CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Find Google Test
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Try to find GTest via pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GTEST QUIET gtest gtest_main)
        if(GTEST_FOUND)
            set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIRS})
            set(GTEST_LIBRARIES ${GTEST_LIBRARIES})
        endif()
    endif()
endif()

# If GTest is not found, we'll use a simple test framework
if(NOT GTest_FOUND AND NOT GTEST_FOUND)
    message(STATUS "Google Test not found, using simple test framework")
    add_library(simple_test STATIC
        test_framework.cpp
    )
    target_include_directories(simple_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    set(TEST_FRAMEWORK simple_test)
else()
    set(TEST_FRAMEWORK ${GTEST_LIBRARIES})
    include_directories(${GTEST_INCLUDE_DIRS})
endif()

# Test executables
set(TEST_EXECUTABLES)

# Test 1: Vector Addition
add_executable(test_vector_addition test_vector_addition.cpp)
target_link_libraries(test_vector_addition cuda_common ${GPU_RUNTIME_LIB} ${TEST_FRAMEWORK})
list(APPEND TEST_EXECUTABLES test_vector_addition)

# Test 2: Advanced Threading (HIP only)
if(USE_HIP)
    add_executable(test_advanced_threading test_advanced_threading.cpp)
    target_link_libraries(test_advanced_threading cuda_common ${GPU_RUNTIME_LIB} ${TEST_FRAMEWORK})
    list(APPEND TEST_EXECUTABLES test_advanced_threading)
endif()

# Test 3: Dynamic Memory (HIP only)
if(USE_HIP)
    add_executable(test_dynamic_memory test_dynamic_memory.cpp)
    target_link_libraries(test_dynamic_memory cuda_common ${GPU_RUNTIME_LIB} ${TEST_FRAMEWORK})
    list(APPEND TEST_EXECUTABLES test_dynamic_memory)
endif()

# Test 4: Warp Primitives (HIP only)
if(USE_HIP)
    add_executable(test_warp_primitives test_warp_primitives.cpp)
    target_link_libraries(test_warp_primitives cuda_common ${GPU_RUNTIME_LIB} ${TEST_FRAMEWORK})
    list(APPEND TEST_EXECUTABLES test_warp_primitives)
endif()

# Test 5: Advanced FFT (HIP only)
if(USE_HIP)
    add_executable(test_advanced_fft test_advanced_fft.cpp)
    target_link_libraries(test_advanced_fft cuda_common ${GPU_RUNTIME_LIB} ${TEST_FRAMEWORK})
    list(APPEND TEST_EXECUTABLES test_advanced_fft)
endif()

# Test 6: N-Body Simulation (HIP only)
if(USE_HIP)
    add_executable(test_nbody_simulation test_nbody_simulation.cpp)
    target_link_libraries(test_nbody_simulation cuda_common ${GPU_RUNTIME_LIB} ${TEST_FRAMEWORK})
    list(APPEND TEST_EXECUTABLES test_nbody_simulation)
endif()

# Test 7: Integration Tests
add_executable(test_integration test_integration.cpp)
target_link_libraries(test_integration cuda_common ${GPU_RUNTIME_LIB} ${TEST_FRAMEWORK})
list(APPEND TEST_EXECUTABLES test_integration)

# Test 8: Performance Tests
add_executable(test_performance test_performance.cpp)
target_link_libraries(test_performance cuda_common ${GPU_RUNTIME_LIB} ${TEST_FRAMEWORK})
list(APPEND TEST_EXECUTABLES test_performance)

# Add tests to CTest
foreach(test_exec ${TEST_EXECUTABLES})
    add_test(NAME ${test_exec} COMMAND ${test_exec})
    set_tests_properties(${test_exec} PROPERTIES
        TIMEOUT 300
        ENVIRONMENT "HIP_VISIBLE_DEVICES=0"
    )
endforeach()

# Custom test target
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS ${TEST_EXECUTABLES}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests"
) 