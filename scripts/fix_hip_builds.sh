#!/bin/bash

# Script to fix HIP kernel launch issues by adding wrapper functions
# This script applies the same pattern used in vector_addition to other examples

echo "Fixing HIP kernel launch issues..."

# Function to add wrapper functions to a .hip file
add_wrapper_functions() {
    local hip_file=$1
    local kernel_name=$2
    local wrapper_name=$3
    
    echo "Adding wrapper functions to $hip_file"
    
    # Check if wrapper already exists
    if grep -q "$wrapper_name" "$hip_file"; then
        echo "  Wrapper already exists, skipping..."
        return
    fi
    
    # Add wrapper functions at the end of the file
    cat >> "$hip_file" << EOF

// Wrapper functions for launching kernels from C++ files
extern "C" {
    void ${wrapper_name}(void* args[], int blockSize, int gridSize) {
        ${kernel_name}<<<gridSize, blockSize>>>(*((float**)args[0]), *((float**)args[1]), *((float**)args[2]), *((int*)args[3]));
    }
}
EOF
}

# Function to update main file to use wrapper
update_main_file() {
    local main_file=$1
    local wrapper_name=$2
    
    echo "Updating $main_file to use wrapper functions"
    
    # Replace kernel launches with wrapper calls
    sed -i "s/${kernel_name}<<<.*>>>(.*);/${wrapper_name}(args, blockSize, gridSize);/g" "$main_file"
}

# Apply fixes to each example
echo "Fixing advanced threading..."
add_wrapper_functions "src/07_advanced_threading/advanced_threading_hip.hip" "advancedThreadSync" "launchAdvancedThreadSync"

echo "Fixing dynamic memory..."
add_wrapper_functions "src/08_dynamic_memory/dynamic_memory_hip.hip" "dynamicTreeBuild" "launchDynamicTreeBuild"

echo "Fixing warp primitives..."
add_wrapper_functions "src/09_warp_primitives/warp_primitives_hip.hip" "warpPrimitivesShowcase" "launchWarpPrimitivesShowcase"

echo "Fixing advanced FFT..."
add_wrapper_functions "src/10_advanced_fft/fft_kernels.hip" "fft_1d_radix2_shared" "launchFFT1D"

echo "Fixing nbody simulation..."
add_wrapper_functions "src/11_nbody_simulation/nbody_hip.hip" "calculateForces_optimized" "launchNBodyForces"

echo "HIP build fixes completed!"
echo "You may need to manually update the main files to use the wrapper functions." 