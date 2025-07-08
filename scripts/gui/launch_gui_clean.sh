#!/bin/bash

# Clean GUI Launcher - Avoids snap library conflicts
# This script sets up a clean environment for running the GUI

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}GPU Kernel GUI Launcher${NC}"
echo "=========================="

# Get project root directory  
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

# Check for GUI executable
GUI_PATHS=(
    "$PROJECT_ROOT/build_gui/bin/gpu_kernel_gui"
    "$PROJECT_ROOT/build_gui_hip/bin/gpu_kernel_gui"
    "$PROJECT_ROOT/build/bin/gpu_kernel_gui"
)

GUI_EXECUTABLE=""
for path in "${GUI_PATHS[@]}"; do
    if [ -x "$path" ]; then
        GUI_EXECUTABLE="$path"
        echo -e "${GREEN}✓ Found GUI executable: $path${NC}"
        break
    fi
done

if [ -z "$GUI_EXECUTABLE" ]; then
    echo -e "${RED}Error: GPU Kernel GUI executable not found!${NC}"
    echo "Please build the GUI first:"
    echo "  bash scripts/build/build_gui.sh  (for CUDA)"
    echo "  bash scripts/build/build_gui_hip.sh  (for HIP/ROCm)"
    exit 1
fi

# Set up clean environment to avoid snap conflicts
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/rocm/lib:$LD_LIBRARY_PATH"
export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt6/plugins"

# Remove snap directories from library path that might cause conflicts
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed 's|/snap/[^:]*:||g')

# Ensure display is set
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi

echo -e "${YELLOW}Starting GUI...${NC}"
echo "Working directory: $PROJECT_ROOT"
echo "Executable: $GUI_EXECUTABLE"
echo "Display: $DISPLAY"

# Change to project root and launch GUI
cd "$PROJECT_ROOT"
exec "$GUI_EXECUTABLE"
