#!/bin/bash
# Launch the GPU Kernel GUI

# Navigate to the project root
cd $(dirname "$0")/..

# Run the GUI
./build_gui_hip/bin/gpu_kernel_gui
