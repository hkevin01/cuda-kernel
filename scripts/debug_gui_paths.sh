#!/bin/bash

# Debug script to test GUI path resolution
echo "=== GUI Path Debug Test ==="

cd /home/kevin/Projects/cuda-kernel/build/bin

echo "Current directory: $(pwd)"
echo "GUI executable: $(pwd)/gpu_kernel_gui"
echo ""

echo "Testing search paths that GUI would use:"
echo "1. Same directory: $(pwd)/vector_addition"
test -x "$(pwd)/vector_addition" && echo "   ✓ Found" || echo "   ✗ Not found"

echo "2. ../bin/: $(pwd)/../bin/vector_addition"
test -x "$(pwd)/../bin/vector_addition" && echo "   ✓ Found" || echo "   ✗ Not found"

echo "3. ../build/bin/: $(pwd)/../build/bin/vector_addition"
test -x "$(pwd)/../build/bin/vector_addition" && echo "   ✓ Found" || echo "   ✗ Not found"

echo "4. ../../build/bin/: $(pwd)/../../build/bin/vector_addition"
test -x "$(pwd)/../../build/bin/vector_addition" && echo "   ✓ Found" || echo "   ✗ Not found"

echo ""
echo "Testing kernel execution with absolute path:"
echo "Command: $(pwd)/vector_addition 1000"
timeout 3s ./vector_addition 1000 2>&1 | head -3
