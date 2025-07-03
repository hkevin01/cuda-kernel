#!/bin/bash

# GPU Platform Detection and Setup Script
# Detects NVIDIA CUDA or AMD ROCm and sets up the build environment

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}GPU Platform Detection and Setup${NC}"
echo "======================================"

# Function to detect NVIDIA GPU
detect_nvidia() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
        return 0
    fi
    return 1
}

# Function to detect AMD GPU
detect_amd() {
    if lspci | grep -qi "amd.*vga\|amd.*display\|radeon\|navi"; then
        echo -e "${GREEN}✓ AMD GPU detected${NC}"
        lspci | grep -i "amd.*vga\|amd.*display\|radeon\|navi" | head -1 | sed 's/.*: //'
        return 0
    fi
    return 1
}

# Function to check CUDA installation
check_cuda() {
    if command -v nvcc &> /dev/null; then
        echo -e "${GREEN}✓ CUDA toolkit found${NC}"
        nvcc --version | grep "release"
        return 0
    else
        echo -e "${YELLOW}⚠ CUDA toolkit not found${NC}"
        return 1
    fi
}

# Function to check ROCm installation
check_rocm() {
    if command -v hipcc &> /dev/null; then
        echo -e "${GREEN}✓ ROCm/HIP found${NC}"
        hipcc --version | head -1
        return 0
    else
        echo -e "${YELLOW}⚠ ROCm/HIP not found${NC}"
        return 1
    fi
}

# Function to install CUDA (Ubuntu/Debian)
install_cuda_ubuntu() {
    echo -e "${YELLOW}Installing CUDA for Ubuntu/Debian...${NC}"
    
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    
    # Install CUDA Toolkit
    sudo apt-get install -y cuda-toolkit-12-6
    
    # Install development tools
    sudo apt-get install -y build-essential cmake
    
    echo -e "${GREEN}CUDA installation completed. Please reboot and run this script again.${NC}"
}

# Function to install ROCm (Ubuntu/Debian)
install_rocm_ubuntu() {
    echo -e "${YELLOW}Installing ROCm for Ubuntu/Debian...${NC}"
    
    # Add ROCm repository
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.2.0/ jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list
    
    sudo apt update
    sudo apt install -y rocm-dev rocm-libs rocblas rocfft rocrand rocsparse build-essential cmake
    
    # Add user to required groups
    sudo usermod -a -G render $USER
    sudo usermod -a -G video $USER
    
    echo -e "${GREEN}ROCm installation completed. Please reboot and run this script again.${NC}"
}

# Function to build project
build_project() {
    local platform=$1
    
    echo -e "${GREEN}Building project for $platform...${NC}"
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure with appropriate platform
    if [[ "$platform" == "CUDA" ]]; then
        cmake -DUSE_CUDA=ON ..
    elif [[ "$platform" == "HIP" ]]; then
        cmake -DUSE_HIP=ON ..
    fi
    
    # Build
    make -j$(nproc)
    
    echo -e "${GREEN}Build completed successfully!${NC}"
    echo ""
    echo "Available executables:"
    find . -name "*vector_addition" -o -name "*matrix_multiplication" -o -name "*parallel_reduction" -o -name "*convolution_2d" -o -name "*monte_carlo" | head -5
}

# Main detection logic
echo "Detecting GPU hardware..."

NVIDIA_DETECTED=false
AMD_DETECTED=false

if detect_nvidia; then
    NVIDIA_DETECTED=true
fi

if detect_amd; then
    AMD_DETECTED=true
fi

if [[ "$NVIDIA_DETECTED" == false && "$AMD_DETECTED" == false ]]; then
    echo -e "${RED}✗ No compatible GPU detected${NC}"
    echo "This project requires either NVIDIA or AMD GPUs"
    exit 1
fi

echo ""
echo "Checking software installations..."

CUDA_AVAILABLE=false
ROCM_AVAILABLE=false

if [[ "$NVIDIA_DETECTED" == true ]]; then
    if check_cuda; then
        CUDA_AVAILABLE=true
    fi
fi

if [[ "$AMD_DETECTED" == true ]]; then
    if check_rocm; then
        ROCM_AVAILABLE=true
    fi
fi

# Installation suggestions
if [[ "$NVIDIA_DETECTED" == true && "$CUDA_AVAILABLE" == false ]]; then
    echo ""
    echo -e "${YELLOW}NVIDIA GPU detected but CUDA not installed.${NC}"
    echo "Options:"
    echo "1. Install CUDA automatically (Ubuntu/Debian)"
    echo "2. Manual installation (see docs/installation/cuda-setup.md)"
    echo "3. Exit"
    
    read -p "Choose option [1-3]: " choice
    case $choice in
        1)
            if [[ -f /etc/debian_version ]]; then
                install_cuda_ubuntu
                exit 0
            else
                echo -e "${RED}Automatic installation only supported on Ubuntu/Debian${NC}"
                echo "Please see docs/installation/cuda-setup.md for manual installation"
                exit 1
            fi
            ;;
        2)
            echo "Please see docs/installation/cuda-setup.md for installation instructions"
            exit 0
            ;;
        3)
            exit 0
            ;;
    esac
fi

if [[ "$AMD_DETECTED" == true && "$ROCM_AVAILABLE" == false ]]; then
    echo ""
    echo -e "${YELLOW}AMD GPU detected but ROCm not installed.${NC}"
    echo "Options:"
    echo "1. Install ROCm automatically (Ubuntu/Debian)"
    echo "2. Manual installation (see docs/installation/rocm-setup.md)"
    echo "3. Exit"
    
    read -p "Choose option [1-3]: " choice
    case $choice in
        1)
            if [[ -f /etc/debian_version ]]; then
                install_rocm_ubuntu
                exit 0
            else
                echo -e "${RED}Automatic installation only supported on Ubuntu/Debian${NC}"
                echo "Please see docs/installation/rocm-setup.md for manual installation"
                exit 1
            fi
            ;;
        2)
            echo "Please see docs/installation/rocm-setup.md for installation instructions"
            exit 0
            ;;
        3)
            exit 0
            ;;
    esac
fi

# Build project with available platform
echo ""
if [[ "$CUDA_AVAILABLE" == true ]]; then
    echo -e "${GREEN}Ready to build with CUDA support${NC}"
    build_project "CUDA"
elif [[ "$ROCM_AVAILABLE" == true ]]; then
    echo -e "${GREEN}Ready to build with ROCm/HIP support${NC}"
    build_project "HIP"
else
    echo -e "${RED}No compatible GPU platform software found${NC}"
    echo "Please install either CUDA or ROCm and run this script again"
    exit 1
fi

echo ""
echo -e "${GREEN}Setup completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. cd build"
echo "2. Run any of the example executables"
echo "3. See individual example README files for usage details"
