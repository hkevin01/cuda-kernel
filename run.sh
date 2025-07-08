#!/bin/bash

# GPU Kernel Examples GUI Launcher
# This script builds and runs the GUI application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect GPU platform
detect_platform() {
    if command_exists nvidia-smi; then
        echo "cuda"
    elif command_exists rocm-smi; then
        echo "hip"
    else
        echo "unknown"
    fi
}

# Function to check Qt installation
check_qt() {
    if command_exists qmake; then
        QT_VERSION=$(qmake -query QT_VERSION 2>/dev/null || echo "unknown")
        print_success "Found Qt version: $QT_VERSION"
        return 0
    elif command_exists qt6-qmake; then
        QT_VERSION=$(qt6-qmake -query QT_VERSION 2>/dev/null || echo "unknown")
        print_success "Found Qt6 version: $QT_VERSION"
        return 0
    else
        print_error "Qt not found. Please install Qt development packages."
        print_status "Run: ./scripts/setup_gui.sh to install dependencies"
        return 1
    fi
}

# Function to build the project
build_project() {
    local platform=$1
    local build_type=${2:-Release}
    
    print_status "Building project for $platform platform..."
    
    # Ensure we're in the project root directory
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$script_dir"
    
    # Create build directory if it doesn't exist
    if [ ! -d "build" ]; then
        mkdir build
    fi
    
    cd build
    
    # Configure with CMake
    print_status "Configuring with CMake..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=$build_type \
        -DBUILD_GUI=ON \
        -DUSE_${platform^^}=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    
    # Build (exclude problematic warp_primitives for now)
    print_status "Building project..."
    make gpu_kernel_gui vector_addition advanced_threading advanced_fft dynamic_memory nbody_simulation -j$(nproc)
    
    cd "$script_dir"
}

# Function to run the GUI
run_gui() {
    local platform=$1
    local gui_path
    
    # Use the appropriate GUI build based on platform
    if [ "$platform" = "hip" ]; then
        gui_path="build_gui_hip/bin/gpu_kernel_gui"
    else
        gui_path="build/bin/gpu_kernel_gui"
    fi
    
    if [ ! -f "$gui_path" ]; then
        print_error "GUI executable not found at $gui_path"
        print_status "Building project first..."
        if [ "$platform" = "hip" ]; then
            print_status "Building HIP GUI..."
            bash scripts/build/build_gui_hip.sh
        else
            build_project $platform
        fi
    fi
    
    # Create logs directory
    if [ ! -d "logs" ]; then
        mkdir -p logs
    fi
    
    # Clear previous log file
    > logs/gui.log
    
    print_success "Launching GPU Kernel Examples GUI..."
    print_status "Platform: $platform"
    print_status "Executable: $gui_path"
    print_status "Log file: logs/gui.log"
    
    # Set environment variable for platform instead of command line argument
    export GPU_PLATFORM=$platform
    
    # Fix library path issues by explicitly using system libraries
    # This prevents conflicts with snap-provided libraries
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH"
    
    # Unset potentially conflicting snap-related variables
    unset SNAP SNAP_COMMON SNAP_DATA SNAP_USER_COMMON SNAP_USER_DATA
    
    # Try to run with explicit library preloading to avoid pthread issues
    if [ -f "/lib/x86_64-linux-gnu/libpthread.so.0" ]; then
        export LD_PRELOAD="/lib/x86_64-linux-gnu/libpthread.so.0:/lib/x86_64-linux-gnu/libc.so.6"
    fi
    
    # Run the GUI with error handling
    if ! "$gui_path" "$@" 2>&1 | tee logs/gui.log; then
        print_error "GUI failed to start. Check logs/gui.log for details."
        print_warning "This might be due to library conflicts or missing dependencies."
        print_status "Try running manually: $gui_path"
        return 1
    fi
}

# Function to show help
show_help() {
    echo "GPU Kernel Examples GUI Launcher"
    echo ""
    echo "Usage: $0 [OPTIONS] [GUI_OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  -p, --platform PLATFORM    Specify GPU platform (cuda|hip|auto)"
    echo "  -b, --build                Force rebuild the project"
    echo "  -d, --debug                Build in debug mode"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "GUI_OPTIONS:"
    echo "  --test-mode                Run GUI in test mode"
    echo "  --help                     Show GUI help"
    echo "  --version                  Show GUI version"
    echo ""
    echo "Examples:"
    echo "  $0                         # Auto-detect platform and run"
    echo "  $0 -p hip                  # Run with HIP platform"
    echo "  $0 -p cuda --test-mode     # Run with CUDA in test mode"
    echo "  $0 -b -p hip               # Rebuild and run with HIP"
    echo ""
    echo "Platform Detection:"
    echo "  auto (default): Auto-detect based on available drivers"
    echo "  cuda: Use NVIDIA CUDA platform"
    echo "  hip: Use AMD ROCm/HIP platform"
}

# Main script logic
main() {
    local platform="auto"
    local force_build=false
    local build_type="Release"
    local gui_args=()
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--platform)
                platform="$2"
                shift 2
                ;;
            -b|--build)
                force_build=true
                shift
                ;;
            -d|--debug)
                build_type="Debug"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            --)
                shift
                gui_args+=("$@")
                break
                ;;
            *)
                gui_args+=("$1")
                shift
                ;;
        esac
    done
    
    print_status "GPU Kernel Examples GUI Launcher"
    print_status "Build type: $build_type"
    
    # Check Qt installation
    if ! check_qt; then
        exit 1
    fi
    
    # Auto-detect platform if needed
    if [ "$platform" = "auto" ]; then
        platform=$(detect_platform)
        if [ "$platform" = "unknown" ]; then
            print_error "Could not auto-detect GPU platform"
            print_status "Please specify platform manually: $0 -p cuda or $0 -p hip"
            exit 1
        fi
        print_status "Auto-detected platform: $platform"
    fi
    
    # Validate platform
    if [ "$platform" != "cuda" ] && [ "$platform" != "hip" ]; then
        print_error "Invalid platform: $platform"
        print_status "Supported platforms: cuda, hip"
        exit 1
    fi
    
    # Check if build is needed
    if [ "$force_build" = true ] || [ ! -f "build/bin/gpu_kernel_gui" ]; then
        build_project $platform $build_type
    else
        print_status "GUI executable found, skipping build"
    fi
    
    # Run the GUI
    run_gui $platform "${gui_args[@]}"
}

# Run main function with all arguments
main "$@" 