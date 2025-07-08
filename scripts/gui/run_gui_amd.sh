#!/bin/bash

# Launch Script for GPU Kernel GUI (HIP/AMD Version)
# -----------------------------------------------------
# This script resolves common library path issues on systems
# that use Snap by explicitly setting the library path.

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}--- Launching GPU Kernel GUI for HIP/AMD ---${NC}"

# Get project root directory
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GUI_EXECUTABLE="$PROJECT_ROOT/build_gui_hip/bin/gpu_kernel_gui"

# Check if the executable exists
if [ ! -f "$GUI_EXECUTABLE" ]; then
    echo -e "${RED}Error: GUI executable not found at $GUI_EXECUTABLE${NC}"
    echo -e "${YELLOW}Please build the HIP GUI first using the script:${NC}"
    echo "    bash scripts/build/build_gui_hip.sh"
    exit 1
fi

# Set the library path to prioritize system libraries over Snap's
# This is the key to fixing the '__libc_pthread_init' symbol lookup error
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "[INFO] Using library path: $LD_LIBRARY_PATH"
echo "[INFO] Launching GUI..."

# Run the GUI
cd "$(dirname "$GUI_EXECUTABLE")"
./"$(basename "$GUI_EXECUTABLE")"

echo -e "${GREEN}GUI exited.${NC}"
