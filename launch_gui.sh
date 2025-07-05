#!/bin/bash

# GUI Launcher with Library Isolation
# This script fixes the snap library conflict issue

echo "=== GPU Kernel GUI Launcher (Fixed) ==="

# Completely reset the library path to use only system libraries
export LD_LIBRARY_PATH=""
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/lib"

# Remove all snap-related environment variables that could interfere
unset SNAP SNAP_COMMON SNAP_DATA SNAP_USER_COMMON SNAP_USER_DATA
unset SNAP_CONTEXT SNAP_INSTANCE_NAME SNAP_INSTANCE_KEY SNAP_REVISION

# Set up clean environment
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"

# Check if GUI executable exists
GUI_PATH="./build_gui/bin/gpu_kernel_gui"
if [ ! -f "$GUI_PATH" ]; then
    echo "Error: GUI executable not found at $GUI_PATH"
    echo "Please build the GUI first with:"
    echo "  cd gui && cmake . -B ../build_gui && cd ../build_gui && make"
    exit 1
fi

# Check if we have required executables
echo "=== Checking available examples ==="
EXECUTABLES_FOUND=0
for exe in build_hip/*_hip; do
    if [ -f "$exe" ] && [ -x "$exe" ]; then
        echo "✓ Found: $(basename "$exe")"
        EXECUTABLES_FOUND=$((EXECUTABLES_FOUND + 1))
    fi
done

if [ $EXECUTABLES_FOUND -eq 0 ]; then
    echo "Warning: No GPU kernel executables found in build_hip/"
    echo "Please build some examples first:"
    echo "  cd src/01_vector_addition && hipcc -O3 -std=c++14 -I../common main_hip.cpp ../common/*.cpp vector_addition_hip.hip -o ../../build_hip/01_vector_addition_hip"
else
    echo "Found $EXECUTABLES_FOUND example executables"
fi

echo ""
echo "=== Launching GUI ==="
echo "GUI Path: $GUI_PATH"
echo "Library Path: $LD_LIBRARY_PATH"

# Launch the GUI with clean environment
exec "$GUI_PATH" "$@"
