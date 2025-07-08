#!/bin/bash

# This script verifies that the GUI's kernel executable search logic can find all
# the expected kernel executables.

# Navigate to the project root directory
cd "$(dirname "$0")/../.." || exit

PROJECT_ROOT=$(pwd)

# List of kernel executables that should be built
# This list should match the 'executableMap' in kernel_runner.cpp, excluding duplicates.
EXPECTED_EXECUTABLES=(
    "vector_addition"
    "matrix_multiplication"
    "parallel_reduction"
    "convolution_2d"
    "monte_carlo"
    "advanced_fft"
    "advanced_threading"
    "dynamic_memory"
    "nbody_simulation"
)

# These are the search paths used in the C++ code in findKernelExecutable()
SEARCH_PATHS=(
    "$PROJECT_ROOT/build_gui_hip/bin/"
    "$PROJECT_ROOT/build_simple/bin/"
    "$PROJECT_ROOT/build/bin/"
    "$PROJECT_ROOT/build_hip/bin/"
)

echo "[INFO] Verifying kernel executable discovery..."
echo "[INFO] Project Root: $PROJECT_ROOT"

ALL_FOUND=true

for executable_name in "${EXPECTED_EXECUTABLES[@]}"; do
    FOUND=false
    for search_path in "${SEARCH_PATHS[@]}"; do
        full_path="${search_path}${executable_name}"
        if [[ -f "$full_path" && -x "$full_path" ]]; then
            echo "[SUCCESS] Found '$executable_name' at: $full_path"
            FOUND=true
            break
        fi
    done

    if [ "$FOUND" = false ]; then
        echo "[FAILURE] Could not find executable for '$executable_name'"
        ALL_FOUND=false
    fi
done

echo ""
if [ "$ALL_FOUND" = true ]; then
    echo "[SUCCESS] All expected kernel executables were found successfully!"
    echo "[INFO] The GUI should no longer show '(Not Built)' for these kernels."
else
    echo "[FAILURE] One or more kernel executables could not be found."
    echo "[INFO] The GUI will likely show '(Not Built)' for the failed kernels."
    exit 1
fi
