#!/bin/bash

# Proper GUI launcher that uses the correct build directory
# This script launches the GUI from build/bin/ with all working kernels

echo "=== GPU Kernel GUI Launcher (Fixed Build) ==="

# Navigate to the project root
cd "$(dirname "$0")"

# Check if we're in the right place
if [ ! -f "run.sh" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check if the GUI exists
GUI_PATH="build/bin/gpu_kernel_gui"
if [ ! -f "$GUI_PATH" ]; then
    echo "Error: GUI not found at $GUI_PATH"
    echo "Please build the project first with: ./run.sh"
    exit 1
fi

# Show available kernels
echo "=== Available GPU Kernels ==="
cd build/bin/
for exe in vector_addition advanced_threading advanced_fft dynamic_memory nbody_simulation warp_primitives; do
    if [ -x "$exe" ]; then
        echo "  ✓ $exe"
    else
        echo "  ✗ $exe (not found)"
    fi
done

echo ""
echo "=== Launching GUI ==="
echo "GUI Path: $(pwd)/gpu_kernel_gui"
echo "Working Directory: $(pwd)"

# Launch the GUI
./gpu_kernel_gui
