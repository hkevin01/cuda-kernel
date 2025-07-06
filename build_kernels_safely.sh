#!/bin/bash

# Safe kernel building script
# This builds new kernels one at a time with safety checks

set -e  # Exit on any error

echo "=== Safe Kernel Builder ==="
echo "Building missing kernels one by one..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to build and test a kernel
build_and_test_kernel() {
    local kernel_name=$1
    local test_args=${2:-"100"}
    
    echo -e "\n${YELLOW}Building $kernel_name...${NC}"
    
    # Build the specific kernel
    cd build
    if make $kernel_name 2>&1; then
        echo -e "${GREEN}✓ Build successful${NC}"
        
        # Test if executable exists
        if [ -f "bin/$kernel_name" ]; then
            echo -e "${YELLOW}Testing $kernel_name with small input...${NC}"
            
            # Test with timeout for safety
            if timeout 5s ./bin/$kernel_name $test_args >/dev/null 2>&1; then
                echo -e "${GREEN}✓ $kernel_name works safely${NC}"
                return 0
            else
                echo -e "${RED}✗ $kernel_name failed or timed out${NC}"
                return 1
            fi
        else
            echo -e "${RED}✗ Executable not found${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Build failed${NC}"
        return 1
    fi
}

# Change to project directory
cd /home/kevin/Projects/cuda-kernel

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
    cd build
    cmake .. -DUSE_HIP=ON -DCMAKE_BUILD_TYPE=Release
    cd ..
fi

# List of kernels to build (starting with safest ones)
kernels_to_build=(
    "matrix_multiplication:512"
    "parallel_reduction:10000" 
    "monte_carlo:1000:100"
    "convolution_2d:256:256:3"
)

successful_builds=()
failed_builds=()

echo -e "\n${YELLOW}Starting safe kernel builds...${NC}"

for kernel_spec in "${kernels_to_build[@]}"; do
    IFS=':' read -ra PARTS <<< "$kernel_spec"
    kernel_name="${PARTS[0]}"
    test_args="${PARTS[1]:-100}"
    
    echo -e "\n================================================"
    echo -e "Building: $kernel_name"
    echo -e "Test args: $test_args"
    echo -e "================================================"
    
    if build_and_test_kernel "$kernel_name" "$test_args"; then
        successful_builds+=("$kernel_name")
        echo -e "${GREEN}$kernel_name added to working kernels${NC}"
    else
        failed_builds+=("$kernel_name")
        echo -e "${RED}$kernel_name needs debugging${NC}"
    fi
    
    # Small delay to let system recover
    sleep 2
done

# Summary
echo -e "\n${YELLOW}=== BUILD SUMMARY ===${NC}"
echo -e "${GREEN}Successful builds (${#successful_builds[@]}):${NC}"
for kernel in "${successful_builds[@]}"; do
    echo -e "  ✓ $kernel"
done

echo -e "\n${RED}Failed builds (${#failed_builds[@]}):${NC}"
for kernel in "${failed_builds[@]}"; do
    echo -e "  ✗ $kernel"
done

# Update GUI if any builds were successful
if [ ${#successful_builds[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}Rebuilding GUI to include new kernels...${NC}"
    cd build
    if make gpu_kernel_gui; then
        echo -e "${GREEN}✓ GUI updated successfully${NC}"
        echo -e "\nYou can now run: ./launch_working_gui.sh"
    else
        echo -e "${RED}✗ GUI build failed${NC}"
    fi
fi

echo -e "\n${GREEN}Safe building complete!${NC}"

# List available kernels
echo -e "\n${YELLOW}Available kernels:${NC}"
ls -la build/bin/ | grep -v gpu_kernel_gui | grep -E "^-.*x.*" | awk '{print "  " $9}'
