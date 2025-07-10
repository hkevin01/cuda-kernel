# GUI CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Find Qt6 (or Qt5 as fallback)
find_package(Qt6 QUIET COMPONENTS Core Widgets Charts Xml)
if(Qt6_FOUND)
    set(QT_VERSION 6)
    message(STATUS "Found Qt6")
else()
    find_package(Qt5 QUIET COMPONENTS Core Widgets Charts Xml)
    if(Qt5_FOUND)
        set(QT_VERSION 5)
        message(STATUS "Found Qt5")
    else()
        message(FATAL_ERROR "Qt5 or Qt6 not found. Please install Qt development packages.")
    endif()
endif()

# Set Qt version
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# GUI executable
add_executable(gpu_kernel_gui
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    kernel_runner.cpp
    kernel_runner.h
    result_viewer.cpp
    result_viewer.h
    performance_widget.cpp
    performance_widget.h
    example_tabs.cpp
    example_tabs.h
    example_loader.cpp
    example_loader.h
    resources.qrc
)

# Link Qt libraries
if(QT_VERSION EQUAL 6)
    target_link_libraries(gpu_kernel_gui 
        Qt6::Core 
        Qt6::Widgets 
        Qt6::Charts
        Qt6::Xml
    )
else()
    target_link_libraries(gpu_kernel_gui 
        Qt5::Core 
        Qt5::Widgets 
        Qt5::Charts
        Qt5::Xml
    )
endif()

# Include directories
target_include_directories(gpu_kernel_gui PRIVATE
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src
)

# Platform-specific includes
if(USE_HIP)
    target_compile_definitions(gpu_kernel_gui PRIVATE USE_HIP)
elseif(USE_CUDA)
    target_compile_definitions(gpu_kernel_gui PRIVATE USE_CUDA)
endif()

# Set output directory
set_target_properties(gpu_kernel_gui PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install target
install(TARGETS gpu_kernel_gui DESTINATION bin) 