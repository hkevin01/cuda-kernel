#!/bin/bash

# Setup script for GUI dependencies and build

set -e

echo "=== GPU Kernel Examples GUI Setup ==="

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    echo "Could not detect OS"
    exit 1
fi

echo "Detected OS: $OS $VER"

# Install Qt dependencies based on distribution
if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    echo "Installing Qt dependencies for Ubuntu/Debian..."
    sudo apt update
    sudo apt install -y \
        qt6-base-dev \
        qt6-charts-dev \
        qt6-tools-dev \
        qt6-tools-dev-tools \
        libqt6charts6 \
        libqt6charts6-dev \
        qtcreator \
        cmake \
        build-essential
    
    # Fallback to Qt5 if Qt6 is not available
    if ! dpkg -l | grep -q qt6-base-dev; then
        echo "Qt6 not found, installing Qt5..."
        sudo apt install -y \
            qt5-default \
            qtbase5-dev \
            qtcharts5-dev \
            qttools5-dev \
            qttools5-dev-tools \
            libqt5charts5-dev \
            qtcreator
    fi

elif [[ "$OS" == *"Fedora"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"CentOS"* ]]; then
    echo "Installing Qt dependencies for Fedora/RHEL/CentOS..."
    sudo dnf install -y \
        qt6-qtbase-devel \
        qt6-qtcharts-devel \
        qt6-qttools-devel \
        qt6-qttools-static \
        qt6-qtcreator \
        cmake \
        gcc-c++ \
        make
    
    # Fallback to Qt5
    if ! rpm -qa | grep -q qt6-qtbase-devel; then
        echo "Qt6 not found, installing Qt5..."
        sudo dnf install -y \
            qt5-qtbase-devel \
            qt5-qtcharts-devel \
            qt5-qttools-devel \
            qt5-qtcreator
    fi

elif [[ "$OS" == *"Arch"* ]]; then
    echo "Installing Qt dependencies for Arch Linux..."
    sudo pacman -S --noconfirm \
        qt6-base \
        qt6-charts \
        qt6-tools \
        qt6-creator \
        cmake \
        gcc \
        make
    
    # Fallback to Qt5
    if ! pacman -Q qt6-base >/dev/null 2>&1; then
        echo "Qt6 not found, installing Qt5..."
        sudo pacman -S --noconfirm \
            qt5-base \
            qt5-charts \
            qt5-tools \
            qt5-creator
    fi

else
    echo "Unsupported OS: $OS"
    echo "Please install Qt development packages manually:"
    echo "- Qt6: qt6-base-dev, qt6-charts-dev, qt6-tools-dev"
    echo "- Qt5: qt5-default, qtbase5-dev, qtcharts5-dev"
    exit 1
fi

echo "Qt dependencies installed successfully!"

# Check if Qt is available
echo "Checking Qt installation..."
if command -v qmake >/dev/null 2>&1; then
    QT_VERSION=$(qmake -query QT_VERSION)
    echo "Found Qt version: $QT_VERSION"
elif command -v qt6-qmake >/dev/null 2>&1; then
    QT_VERSION=$(qt6-qmake -query QT_VERSION)
    echo "Found Qt6 version: $QT_VERSION"
else
    echo "Warning: Qt not found in PATH"
    echo "You may need to add Qt to your PATH or install Qt manually"
fi

# Create build directory and configure
echo "Configuring build..."
mkdir -p build
cd build

# Configure with GUI enabled
cmake .. -DBUILD_GUI=ON -DUSE_HIP=ON

echo ""
echo "=== Setup Complete ==="
echo "To build the project with GUI:"
echo "  cd build"
echo "  make -j$(nproc)"
echo ""
echo "To run the GUI:"
echo "  ./bin/gpu_kernel_gui"
echo ""
echo "To run without GUI:"
echo "  make -j$(nproc) -DBUILD_GUI=OFF" 